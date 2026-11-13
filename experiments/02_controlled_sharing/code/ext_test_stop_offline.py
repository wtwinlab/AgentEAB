#!/usr/bin/env python3
"""The stop / resume / recovery contract for the extension collector. Zero model calls.

    python3 ext_test_stop_offline.py
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]

FAIL: List[str] = []
N = {"n": 0}


def ck(cond: Any, label: str) -> bool:
    N["n"] += 1
    if not cond:
        FAIL.append(label)
    return bool(cond)


SANDBOX = ROOT / "workspace/benchmark_extension/_stop_test_sandbox"


def _env(extra: Dict[str, str] = None) -> Dict[str, str]:
    e = dict(os.environ)
    e["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), str(ROOT / "code"), str(ROOT / "experiments/06_cost_resources/code"),
         e.get("PYTHONPATH", "")])
    e["PAM_FAKE_RUN"] = "0.45"
    e["EXT_SANDBOX"] = str(SANDBOX)
    e["EXT_GATE_OVERRIDE"] = "open"
    if extra:
        e.update(extra)
    return e


RUNNER = r'''
import json, os, pathlib, sys
sand = pathlib.Path(os.environ["EXT_SANDBOX"])
sand.mkdir(parents=True, exist_ok=True)
import ext_stop_control as SC
import ext_design as D
# redirect every write to the sandbox; the real tree is never touched by this test
SC.OUT = sand; SC.CONTROL = sand / "CONTROL.json"; SC.STATUS = sand / "STATUS.json"
SC.GATE = sand / "GATE.json"; SC.MANIFEST = sand / "manifest.jsonl"
if os.environ.get("EXT_GATE_OVERRIDE") == "open":
    SC.gate_state = lambda: (True, [{"condition": "test override", "satisfied": True,
                                     "detail": "EXT_GATE_OVERRIDE=open"}])
    SC.require_gate = SC.gate_state
import ext_collect as C
C.OUT = sand; C.RAW = sand / "raw"; C.MANIFEST = sand / "manifest.jsonl"
D.OUT = sand
sys.exit(C.main())
'''


def _run(args: List[str], extra=None, timeout=240) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", RUNNER, *args], cwd=str(HERE), env=_env(extra),
                          capture_output=True, text=True, timeout=timeout)


def _shard_rows() -> List[Dict[str, Any]]:
    rows = []
    raw = SANDBOX / "raw"
    if not raw.exists():
        return rows
    for p in sorted(raw.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _dupes() -> Dict[str, int]:
    c = collections.Counter(r["key"] for r in _shard_rows() if r.get("status") == "ok")
    return {k: v for k, v in c.items() if v > 1}


# ---------------------------------------------------------------- 1. the gate
def test_gate() -> Dict[str, Any]:
    import ext_stop_control as SC
    open_, conds = SC.gate_state()
    ck(len(conds) == 3, f"1: the gate has {len(conds)} conditions, expected 3")
    ck(all("condition" in c and "satisfied" in c for c in conds),
       "1: a gate condition does not report whether it is satisfied")
    # on this host the real gate is closed, so the real --run must refuse
    env = dict(EXT_GATE_OVERRIDE="closed")
    r = _run(["--run", "--workers", "4", "--smoke", "1", "--benchmark", "gpqa"], extra=env)
    ck(r.returncode == 4,
       f"1: --run with the gate closed exited {r.returncode}, expected 4; stdout={r.stdout[-300:]}")
    ck("GATE" in r.stdout and "closed" in r.stdout.lower(),
       "1: --run did not say which gate conditions were unmet")
    # and a second call must not have cached the verdict
    r2 = _run(["--run", "--workers", "4", "--smoke", "1", "--benchmark", "gpqa"], extra=env)
    ck(r2.returncode == 4, "1: the gate verdict looks cached across processes")
    return {"conditions": conds, "real_gate_open": open_,
            "closed_run_exit": r.returncode, "second_closed_run_exit": r2.returncode}


# ---------------------------------------------------------------- 2-5. the stop contract
def test_stop() -> Dict[str, Any]:
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    SANDBOX.mkdir(parents=True)
    import ext_stop_control as SC

    # a full GPQA queue: 6,200 synthetic unit-runs at 0.45 s each, so a stop lands mid-run
    proc = subprocess.Popen([sys.executable, "-c", RUNNER, "--run", "--workers", "16",
                             "--benchmark", "gpqa"],
                            cwd=str(HERE), env=_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    # wait until it is demonstrably dispatching
    deadline = time.time() + 60
    st: Dict[str, Any] = {}
    while time.time() < deadline:
        try:
            st = json.loads((SANDBOX / "STATUS.json").read_text(encoding="utf-8"))
        except Exception:
            st = {}
        if st.get("completed_unit_runs", 0) >= 8 and st.get("inflight", 0) > 0:
            break
        time.sleep(0.5)
    ck(st.get("inflight", 0) > 0, f"2: the collector never reported in-flight work: {st}")
    remaining_at_stop = st.get("remaining_in_benchmark", 0)
    done_at_stop = st.get("completed_unit_runs", 0)
    ck(remaining_at_stop > 100,
       f"2: only {remaining_at_stop} items were queued when the stop was requested, so this does "
       f"not test a mid-queue stop")

    t0 = time.time()
    # write the flag into the sandbox exactly as the CLI's `request_stop` would
    tmp = SANDBOX / "CONTROL.json.tmp"
    tmp.write_text(json.dumps({"state": "stop", "by": "test",
                               "requested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                              time.gmtime())}) + "\n",
                   encoding="utf-8")
    os.replace(tmp, SANDBOX / "CONTROL.json")
    stop_return = time.time() - t0
    ck(stop_return < 1.0, f"3: requesting the stop took {stop_return:.2f}s, not immediate")

    out, _ = proc.communicate(timeout=SC.DRAIN_SECONDS + 120)
    ck(proc.returncode == 0, f"2: the collector exited {proc.returncode} after a stop")
    ck("[STOP] requested" in out, "2: the collector did not report observing the stop")
    rows = _shard_rows()
    ok = [r for r in rows if r.get("status") == "ok"]
    ck(len(ok) >= done_at_stop,
       f"4: {done_at_stop} unit-runs were reported complete before the stop but only {len(ok)} "
       f"are on disk -- a reported result was lost")
    import ext_collect as C
    C.RAW = SANDBOX / "raw"
    dk = C.done_keys()
    ck(len(dk) == len(ok),
       f"4: the done-set has {len(dk)} keys but {len(ok)} ok records are on disk")
    ck(not _dupes(), f"4: duplicate keys on disk after the stop: {list(_dupes())[:3]}")
    ck(len(ok) < 6200, f"2: the collector finished the whole benchmark ({len(ok)}) instead of "
                       f"stopping mid-queue")

    # 5. no self-restart while the flag is set
    r = _run(["--run", "--workers", "8", "--benchmark", "gpqa"])
    ck(r.returncode == 2, f"5: a second --run with the stop flag set exited {r.returncode}, "
                          f"expected 2")
    after = len([x for x in _shard_rows() if x.get("status") == "ok"])
    ck(after == len(ok), f"5: the refused run still wrote {after - len(ok)} records")
    return {"ok_before_stop_reported": done_at_stop, "queued_at_stop": remaining_at_stop,
            "ok_on_disk": len(ok), "stop_request_seconds": round(stop_return, 3),
            "second_run_exit": r.returncode, "duplicate_keys": len(_dupes())}


# ---------------------------------------------------------------- 6. resume skips finished work
def test_resume(prev_ok: int) -> Dict[str, Any]:
    (SANDBOX / "CONTROL.json").write_text(
        json.dumps({"state": "run", "by": "test"}) + "\n", encoding="utf-8")
    r = _run(["--plan", "--benchmark", "gpqa"])
    ck(r.returncode == 0, f"6: --plan after resume exited {r.returncode}")
    pending = None
    for line in r.stdout.splitlines():
        if line.strip().startswith("gpqa"):
            pending = int(line.split()[1].replace(",", ""))
    ck(pending is not None and pending == 6200 - prev_ok,
       f"6: --plan reports {pending} pending, expected {6200 - prev_ok} (6,200 minus the "
       f"{prev_ok} already on disk)")

    # finish a bounded slice and check nothing already done was re-run
    r2 = _run(["--run", "--workers", "16", "--benchmark", "gpqa", "--smoke", "4"])
    ck(r2.returncode == 0, f"6: the resumed run exited {r2.returncode}: {r2.stdout[-400:]}")
    ck(not _dupes(), f"6: the resumed run produced duplicate keys: {list(_dupes())[:3]}")
    now = len([x for x in _shard_rows() if x.get("status") == "ok"])
    ck(now > prev_ok, "6: the resumed run collected nothing")
    return {"pending_reported": pending, "ok_after_resume": now,
            "new_records": now - prev_ok, "duplicate_keys": len(_dupes())}


# ---------------------------------------------------------------- 7. a hard kill mid-flight
def test_hard_kill() -> Dict[str, Any]:
    """SIGKILL with requests in flight: no flag, no drain, no flush. Then restart and reconcile."""
    before = len([x for x in _shard_rows() if x.get("status") == "ok"])
    proc = subprocess.Popen([sys.executable, "-c", RUNNER, "--run", "--workers", "16",
                             "--benchmark", "gpqa"],
                            cwd=str(HERE), env=_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    deadline = time.time() + 60
    st: Dict[str, Any] = {}
    while time.time() < deadline:
        try:
            st = json.loads((SANDBOX / "STATUS.json").read_text(encoding="utf-8"))
        except Exception:
            st = {}
        if st.get("completed_unit_runs", 0) >= before + 20 and st.get("inflight", 0) > 0:
            break
        time.sleep(0.4)
    ck(st.get("inflight", 0) > 0, f"7: nothing was in flight when the kill was issued: {st}")
    inflight_at_kill = st.get("inflight", 0)
    reported = st.get("completed_unit_runs", 0)
    # kill the whole process group, so the pool's workers die with the parent -- a supervisor that
    # kills only the wrapper is the mistake this reproduces deliberately
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=30)
    ck(proc.returncode != 0, "7: the process survived SIGKILL")
    time.sleep(1.0)

    survived = [x for x in _shard_rows() if x.get("status") == "ok"]
    ck(len(survived) >= reported,
       f"7: the collector had reported {reported} complete but only {len(survived)} survived the "
       f"kill -- a result was buffered rather than fsynced")
    ck(not _dupes(), f"7: duplicate keys after the kill: {list(_dupes())[:3]}")
    killed_total = len(survived)

    # restart: it must skip every surviving record and finish the design
    r = _run(["--run", "--workers", "24", "--benchmark", "gpqa"], timeout=900)
    ck(r.returncode == 0, f"7: the restart exited {r.returncode}: {r.stdout[-400:]}")
    final = [x for x in _shard_rows() if x.get("status") == "ok"]
    ck(len(final) == 6200,
       f"7: the restart left {len(final)} of 6,200 unit-runs; a resume must complete the design")
    ck(not _dupes(),
       f"7: the restart double-counted {len(_dupes())} keys -- the done-set did not hold")
    import ext_collect as C
    C.RAW = SANDBOX / "raw"
    ck("COMPLETE" in r.stdout,
       "7: the restart did not report the benchmark complete against the frozen design")
    return {"inflight_at_kill": inflight_at_kill, "reported_before_kill": reported,
            "survived_the_kill": killed_total, "after_restart": len(final),
            "records_the_restart_added": len(final) - killed_total,
            "duplicate_keys": len(_dupes())}


# ---------------------------------------------------------------- 8. the flags are independent
def test_flag_independence() -> Dict[str, Any]:
    import ext_stop_control as ESC
    import stop_control as MSC
    ck(ESC.CONTROL != MSC.CONTROL,
       "8: the extension and the main collection share one control file, so stopping one would "
       "stop the other")
    ck(ESC.STATUS != MSC.STATUS, "8: the two collections share one status file")
    main_before = MSC.read_state().get("state")
    # writing the extension flag must not touch the main one. Use the sandbox so the real
    # extension flag is not left set by a test.
    p = SANDBOX / "CONTROL.json"
    p.write_text(json.dumps({"state": "stop", "by": "test"}) + "\n", encoding="utf-8")
    ck(MSC.read_state().get("state") == main_before,
       "8: setting the extension stop flag changed the main collection's state")
    p.write_text(json.dumps({"state": "run", "by": "test"}) + "\n", encoding="utf-8")
    return {"extension_control": str(ESC.CONTROL.relative_to(ROOT)),
            "main_control": str(MSC.CONTROL.relative_to(ROOT)),
            "main_state_unchanged": main_before}


PROBE = r'''
# The parent probes the endpoints (which installs the registry), then a worker initializer runs.
# Under `fork` the parent's probe goes through `_ensure_installed`, so the child skips the install,
# and `_ensure_endpoints` treats a duplicate-alias error as an inherited registry.
import concurrent.futures as cf, os, sys
sys.path[:0] = [os.environ["EXT_HERE"]]
import ext_collect as EC
import ext_unit_driver as EUD

ready, _detail = EC.endpoints_ready("gpqa")          # the parent-side install, as production does
import collect_all_matching as C
assert getattr(C, "_INSTALLED", False), "the parent probe left _INSTALLED False -- forks will die"

# the worker initializer must survive, twice, in this process and in forked children
EC._worker_init()
EC._worker_init()

# The hostile case, reproduced faithfully: the registry is full and BOTH guard flags are False.
# That is exactly the state a fork inherits from a parent that installed the registry some other
# way -- `unit_driver._READY` and `collect_all_matching._INSTALLED` are the two flags a fork would
# have carried, and with the registry already built, re-installing raises `duplicate alias`.
import unit_driver as UD
C._INSTALLED = False
UD._READY = False
EUD._READY.clear()
note = EUD._ensure_endpoints()
assert "inherited" in note, "a full registry with both flags lost was not recognised: %r" % note
# and it must be usable afterwards, not merely non-fatal
from consensus_delivery.registry_live import REGISTRY
assert REGISTRY.get("llama3-8b-g1") is not None, "the registry is unusable after recovery"

def _probe(_i):
    EC._worker_init()
    import collect_all_matching as C2
    return getattr(C2, "_INSTALLED", False)

with cf.ProcessPoolExecutor(max_workers=4, initializer=EC._worker_init) as ex:
    got = list(ex.map(_probe, range(8)))
assert all(got), "a worker finished initialization without the registry installed: %r" % got
print("PROBE OK workers=%d parent_ready=%s" % (len(got), ready))
'''


def test_worker_init_after_parent_probe() -> Dict[str, Any]:
    """Regression: the parent's endpoint probe must not kill every forked worker."""
    env = dict(os.environ)
    env["EXT_HERE"] = str(HERE)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), str(ROOT / "code"), str(ROOT / "experiments/06_cost_resources/code"),
         env.get("PYTHONPATH", "")])
    env.pop("PAM_FAKE_RUN", None)
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=str(HERE), env=env,
                       capture_output=True, text=True, timeout=600)
    ck(r.returncode == 0,
       "9: a worker initializer failed after the parent probed the endpoints -- this is the "
       "BrokenProcessPool that produced 38 errors and 0 ok records: "
       + (r.stderr or r.stdout)[-400:])
    ck("PROBE OK" in r.stdout, "9: the probe did not reach its assertions: " + r.stdout[-200:])
    return {"stdout": r.stdout.strip()[-120:], "returncode": r.returncode}


def main() -> int:
    for _p in (HERE, ROOT / "code", ROOT / "experiments/06_cost_resources/code"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    report: Dict[str, Any] = {"model_calls": 0, "sandbox": str(SANDBOX.relative_to(ROOT))}
    print("== 1. the dependency gate")
    report["gate"] = test_gate()
    print("   real gate open: %s | --run with it closed exits %s (not cached: %s)"
          % (report["gate"]["real_gate_open"], report["gate"]["closed_run_exit"],
             report["gate"]["second_closed_run_exit"]))
    print("== 2-5. stop: halt dispatch, immediate return, durable results, no self-restart")
    report["stop"] = test_stop()
    s = report["stop"]
    print("   stopped with %s items still queued; %s ok records on disk; stop returned in %ss; "
          "second --run exit %s; duplicates %s"
          % (f"{s['queued_at_stop']:,}", f"{s['ok_on_disk']:,}", s["stop_request_seconds"],
             s["second_run_exit"], s["duplicate_keys"]))
    print("== 6. resume skips finished work")
    report["resume"] = test_resume(report["stop"]["ok_on_disk"])
    print("   --plan reported %s pending; resumed run added %s; duplicates %s"
          % (f"{report['resume']['pending_reported']:,}",
             f"{report['resume']['new_records']:,}", report["resume"]["duplicate_keys"]))
    print("== 7. SIGKILL mid-flight, then restart")
    report["hard_kill"] = test_hard_kill()
    h = report["hard_kill"]
    print("   killed with %s in flight; %s survived; restart added %s; final %s/6,200; "
          "duplicates %s" % (h["inflight_at_kill"], f"{h['survived_the_kill']:,}",
                             f"{h['records_the_restart_added']:,}", f"{h['after_restart']:,}",
                             h["duplicate_keys"]))
    print("== 8. the two control flags are independent")
    report["flags"] = test_flag_independence()
    print("   %s vs %s" % (report["flags"]["extension_control"],
                           report["flags"]["main_control"]))
    print("== 9. worker initialization survives the parent's endpoint probe (regression)")
    report["worker_init"] = test_worker_init_after_parent_probe()
    print("   %s" % report["worker_init"]["stdout"])

    report["checks"] = N["n"]
    report["failures"] = FAIL
    out = ROOT / "results/benchmark_extension_2026-10-20/ACCEPTANCE_stop_offline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print()
    for f in FAIL:
        print("  FAIL " + f)
    print("ext_test_stop_offline: %s (%d checks) -> %s"
          % ("PASS" if not FAIL else "FAIL", N["n"], out.relative_to(ROOT)))
    if not FAIL and SANDBOX.exists():
        shutil.rmtree(SANDBOX)                 # the sandbox is scaffolding, not evidence
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
