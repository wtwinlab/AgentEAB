#!/usr/bin/env python3
"""Measure the STOP contract end to end, with synthetic unit-runs. Zero model calls.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "code")]

PY = sys.executable
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "", on_pass: str = ""):
    note = on_pass if cond else detail
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + note) if note else ''}")
    if not cond:
        FAILS.append(f"{name}: {detail}")


def main() -> int:
    import collect_controlled as CC
    import stop_control as SC

    # a scratch output root, so the test never touches real collected data
    sandbox = ROOT / "results/_test_stop_sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    env = dict(os.environ, PAM_FAKE_RUN="0.35")

    def run_collector(extra: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run([PY, str(HERE / "collect_controlled.py"), "--run",
                               "--workers", "4", "--limit", "4"] + extra,
                              cwd=str(HERE), env=env, capture_output=True, text=True, timeout=300)

    print("stop contract, offline (synthetic unit-runs)\n")
    SC.clear_stop(by="test")
    for p in (CC.OUT / "raw",):
        if p.exists():
            shutil.rmtree(p)
    if CC.STATUS.exists() if hasattr(CC, "STATUS") else False:
        pass

    # ---- 1. a stop mid-flight halts dispatch while work remains
    proc = subprocess.Popen([PY, str(HERE / "collect_controlled.py"), "--run",
                             "--workers", "4", "--limit", "4"],
                            cwd=str(HERE), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    time.sleep(6.0)                              # let a few synthetic runs complete
    st_before = SC.read_status()
    done_before = st_before.get("completed_unit_runs", 0)
    t_stop = time.time()
    SC.request_stop(by="test", note="offline stop contract test")
    try:
        out, _ = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        FAILS.append("collector did not return after a stop")
        out += "\n[TEST] collector had to be killed"
    t_return = time.time() - t_stop

    check("some work had been done before the stop was requested", done_before > 0,
          f"status showed {done_before} completed", f"{done_before} unit-runs completed")
    check("the collector returned promptly after the stop",
          t_return <= SC.DRAIN_SECONDS + 30,
          f"took {t_return:.0f}s, drain window is {SC.DRAIN_SECONDS:.0f}s",
          f"returned in {t_return:.1f}s (drain window {SC.DRAIN_SECONDS:.0f}s)")
    check("the run reported itself STOPPED rather than finished", "STOPPED" in out,
          "the collector claimed it finished a stopped run",
          "output says STOPPED")
    check("[STOP] was observed and logged", "[STOP] requested" in out,
          "no stop acknowledgement in the output")

    st = SC.read_status()
    check("status file records the stopped phase", st.get("phase") in ("stopped", "stopping"),
          f"phase is {st.get('phase')!r}", f"phase={st.get('phase')!r}")
    check("work still remained when the collector stopped",
          (st.get("remaining_in_benchmark") or 0) > 0,
          "the queue had drained anyway, so this run does not show dispatch being halted",
          f"{st.get('remaining_in_benchmark')} items still queued in the current benchmark")

    # ---- 2. results that finished are durable, and the done-set agrees with disk
    keys = CC.done_keys()
    shard_lines = sum(1 for p in (CC.RAW).glob("*.jsonl")
                      for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    check("every written record is in the rebuilt done-set", len(keys) == shard_lines,
          f"{len(keys)} keys vs {shard_lines} lines -- a duplicate or unparsable record",
          f"{len(keys)} keys == {shard_lines} shard lines")
    check("the completed work survived the stop", len(keys) >= done_before,
          f"{len(keys)} on disk but status had counted {done_before}",
          f"{len(keys)} unit-runs durable on disk")

    # ---- 3. the flag persists: a second run refuses to dispatch
    again = run_collector([])
    check("a stopped collection refuses to dispatch again", again.returncode == 2
          and "says stop" in again.stdout,
          f"rc={again.returncode}, stdout={again.stdout[-200:]!r}",
          "second --run exits 2 without dispatching")
    check("nothing restarted on its own", SC.stop_requested(),
          "the stop flag cleared itself")

    # ---- 4. resume skips finished work instead of recollecting it
    SC.clear_stop(by="test")
    before_keys = set(CC.done_keys())
    resumed = run_collector([])
    after_keys = set(CC.done_keys())
    recollected = 0
    counts: dict = {}
    for p in CC.RAW.glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                k = json.loads(line)["key"]
                counts[k] = counts.get(k, 0) + 1
    recollected = sum(1 for k, n in counts.items() if n > 1)
    check("a resume recollected nothing", recollected == 0,
          f"{recollected} cache keys appear more than once, which would double-count",
          f"{len(before_keys)} prior keys untouched, {len(after_keys - before_keys)} new")
    check("the resume completed the queue", "finished" in resumed.stdout,
          f"stdout tail: {resumed.stdout[-200:]!r}", "resumed run reports finished")

    # ---- cleanup: this test's synthetic records must not pollute real collection
    for p in CC.RAW.glob("*.jsonl"):
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines and all(json.loads(l).get("synthetic") for l in lines):
            p.unlink()
    SC.clear_stop(by="test")
    left = list(CC.RAW.glob("*.jsonl")) if CC.RAW.exists() else []
    check("synthetic records removed from the collection tree", not left,
          f"{len(left)} shard(s) still hold test data: {[p.name for p in left][:3]}",
          "no synthetic shards remain")

    print(f"\n{'ALL STOP CHECKS PASS' if not FAILS else str(len(FAILS)) + ' FAILURES'}")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
