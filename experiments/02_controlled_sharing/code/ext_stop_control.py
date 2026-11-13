#!/usr/bin/env python3
"""STOP / STATUS / RESUME for the **benchmark extension** collection, and the dependency gate.

    python3 ext_stop_control.py stop      # request a stop; returns immediately
    python3 ext_stop_control.py status    # benchmark, in flight, done, spend, ETA
    python3 ext_stop_control.py resume    # clears the flag; nothing is started
    python3 ext_stop_control.py gate      # re-evaluate the three gate conditions, right now

The contract a stop honours, in the order it takes effect
---------------------------------------------------------
1. **no new dispatch** -- the collector stops submitting within one poll (<= 2 s), while items are
   still queued. "Waiting for the current benchmark to finish" is not a stop;
2. **no auto-restart** -- the flag is on disk, so a supervisor or a careless re-run cannot revive
   the collection: `ext_collect.py --run` exits 2 while the flag is set. Resuming is a separate act;
3. **bounded drain** -- in-flight unit-runs get `DRAIN_SECONDS` to return and be written; after that
   their workers are cancelled. Nothing waits for a debate round or a benchmark;
4. **durable results** -- a unit-run is appended and fsynced before it counts, so a stop never
   costs a finished result and a resume never re-runs one;
5. **release only this project's own resources** -- this collection starts no container, so there is
   nothing of its own to tear down; the shared `cx-*` endpoints belong to the main agent and are
   left running. A stop that killed them would stop someone else's work.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]

RUN_ID = "benchmark_extension_2026-10-20"
OUT = ROOT / "workspace" / RUN_ID
CONTROL = OUT / "CONTROL.json"
STATUS = OUT / "STATUS.json"
GATE = OUT / "GATE.json"
MANIFEST = OUT / "manifest.jsonl"

#: the main agent's own control file, read-only from here
MAIN_OUT = ROOT / "workspace/controlled_sharing"
MAIN_CONTROL = MAIN_OUT / "CONTROL.json"
SYNTH = ROOT / "analysis/controlled_sharing/rq_results.json"

#: the five benchmarks the gate requires to be complete before any extension GPU call
GATE_BENCHMARKS = ["gsm8k", "strategyqa", "mmlu_pro", "geographic_settlements", "wdc_products"]

DRAIN_SECONDS = 90.0


def _write_atomic(path: pathlib.Path, blob: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- stop flag
def read_state() -> Dict[str, Any]:
    """Absent file means `run`: a fresh tree is not stopped."""
    try:
        return json.loads(CONTROL.read_text(encoding="utf-8"))
    except Exception:
        return {"state": "run"}


def stop_requested() -> bool:
    return read_state().get("state") == "stop"


def request_stop(by: str = "operator", note: str = "") -> Dict[str, Any]:
    blob = {"state": "stop", "requested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "by": by, "note": note, "run_id": RUN_ID,
            "contract": ["stop dispatch within one poll (<=2s), with items still queued",
                         "no auto-restart: --run exits 2 while the flag is set",
                         f"bounded drain <= {DRAIN_SECONDS:.0f}s, then cancel",
                         "completed unit-runs already fsynced; a resume skips them",
                         "release only this collection's own resources (it starts no container)"]}
    _write_atomic(CONTROL, blob)
    return blob


def clear_stop(by: str = "operator") -> Dict[str, Any]:
    blob = {"state": "run", "cleared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "by": by, "run_id": RUN_ID,
            "note": "clearing the flag only permits a new run; it does not start one"}
    _write_atomic(CONTROL, blob)
    return blob


def write_status(blob: Dict[str, Any]) -> None:
    blob = dict(blob)
    blob["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    blob["run_id"] = RUN_ID
    _write_atomic(STATUS, blob)


def read_status() -> Dict[str, Any]:
    try:
        return json.loads(STATUS.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------- the dependency gate
def gate_state() -> Tuple[bool, List[Dict[str, Any]]]:
    """`(open, conditions)` recomputed from disk on every call -- never cached.
    """
    conds: List[Dict[str, Any]] = []

    # 1. all five benchmarks complete under the main collector's own definition
    detail: Dict[str, str] = {}
    all_done = True
    try:
        sys.path[:0] = [str(ROOT / "code"), str(HERE)]
        import collect_controlled as CC
        done = CC.done_keys()
        for b in GATE_BENCHMARKS:
            ok, have, need = CC.design_complete(b, done)
            detail[b] = f"{have:,}/{need:,}{' complete' if ok else ''}"
            all_done = all_done and ok
    except Exception as exc:                                              # noqa: BLE001
        all_done = False
        detail["error"] = f"{type(exc).__name__}: {exc}"[:200]
    conds.append({"condition": "collect_controlled.design_complete for all 5 benchmarks",
                  "satisfied": all_done, "detail": detail})

    # 2. the synthesis result exists AND covers the five benchmarks
    cov: Any = "absent"
    ok2 = False
    if SYNTH.exists():
        try:
            blob = json.loads(SYNTH.read_text(encoding="utf-8"))
            names = set()
            for key in ("benchmarks", "per_benchmark", "by_benchmark"):
                v = blob.get(key)
                if isinstance(v, dict):
                    names |= set(v)
                elif isinstance(v, list):
                    names |= {str(x) for x in v}
            if not names:                       # fall back to a textual scan, then say so
                text = SYNTH.read_text(encoding="utf-8")
                names = {b for b in GATE_BENCHMARKS if f'"{b}"' in text}
            missing = [b for b in GATE_BENCHMARKS if b not in names]
            ok2 = not missing
            cov = {"found": sorted(names & set(GATE_BENCHMARKS)), "missing": missing}
        except Exception as exc:                                          # noqa: BLE001
            cov = f"unreadable: {type(exc).__name__}"
    conds.append({"condition": "analysis/controlled_sharing/rq_results.json covers all 5",
                  "satisfied": ok2, "detail": cov})

    # 3. the main agent's explicit hand-over
    g: Any = "absent"
    ok3 = False
    if GATE.exists():
        try:
            g = json.loads(GATE.read_text(encoding="utf-8"))
            ok3 = str(g.get("state")).lower() == "open"
        except Exception as exc:                                          # noqa: BLE001
            g = f"unreadable: {type(exc).__name__}"
    conds.append({"condition": f"{GATE.relative_to(ROOT)} says state=open",
                  "satisfied": ok3, "detail": g})

    return all(c["satisfied"] for c in conds), conds


def require_gate() -> Tuple[bool, List[Dict[str, Any]]]:
    """Same as `gate_state`, and it also records the evaluation so a wait is auditable."""
    open_, conds = gate_state()
    _write_atomic(OUT / "GATE_CHECKS.json", {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "open": open_, "conditions": conds,
        "note": "re-evaluated from disk at this instant; never cached across launches"})
    return open_, conds


# ---------------------------------------------------------------- status rendering
def manifest_summary() -> Dict[str, Any]:
    """Counts and a measured ETA from `manifest.jsonl`, which the collector keeps current."""
    if not MANIFEST.exists():
        return {}
    by: Dict[str, int] = {}
    tok = calls = 0
    first = last = None
    out_head: Dict[str, Any] = {}
    for line in MANIFEST.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("_header"):
            out_head = r
            continue
        by[r.get("status", "?")] = by.get(r.get("status", "?"), 0) + 1
        tok += r.get("completion_tokens") or 0
        calls += r.get("n_calls") or 0
        t = r.get("utc")
        if t:
            first = t if first is None or t < first else first
            last = t if last is None or t > last else last
    out: Dict[str, Any] = {"by_status": by, "output_tokens": tok, "calls": calls,
                           "first_record_utc": first, "last_record_utc": last,
                           "phase": out_head.get("phase"),
                           "eta_from_manifest": out_head.get("eta_utc")}
    st = read_status()
    rate = st.get("unit_runs_per_second")
    pending = by.get("to-collect", 0) + by.get("running", 0)
    if rate and pending:
        out["eta_seconds"] = round(pending / rate, 1)
        out["eta_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime(time.time() + pending / rate))
        out["eta_basis"] = f"measured {rate:.2f} unit-runs/s over this run"
    elif pending:
        out["eta_seconds"] = None
        out["eta_basis"] = "no throughput measured yet in this run"
    return out


def main() -> int:
    cmd = (sys.argv[1] if len(sys.argv) > 1 else "status").lower()
    args = sys.argv[2:]
    if cmd == "stop":
        blob = request_stop(by=os.environ.get("USER", "operator"),
                            note=" ".join(a for a in args if a != "--both"))
        print("EXTENSION STOP requested at %s" % blob["requested_utc"])
        for line in blob["contract"]:
            print("  - " + line)
        if "--both" in args:
            try:
                sys.path.insert(0, str(HERE))
                import stop_control as SC
                b2 = SC.request_stop(by=os.environ.get("USER", "operator"),
                                     note="stopped together with the extension")
                print("\nMAIN STOP also requested at %s" % b2["requested_utc"])
            except Exception as exc:                                      # noqa: BLE001
                print("\ncould not stop the main collection: %s" % exc)
                return 1
        else:
            print("\nthis does NOT stop the main collection; add --both, or run "
                  "`python3 stop_control.py stop`")
        return 0
    if cmd == "resume":
        blob = clear_stop(by=os.environ.get("USER", "operator"))
        print("extension stop flag cleared at %s -- no run was started" % blob["cleared_utc"])
        return 0
    if cmd == "gate":
        open_, conds = require_gate()
        print("GATE: %s" % ("OPEN" if open_ else "CLOSED"))
        for c in conds:
            print("  [%s] %s" % ("x" if c["satisfied"] else " ", c["condition"]))
            print("        %s" % json.dumps(c["detail"])[:400])
        if not open_:
            print("\nGPU collection must wait. Offline work is unaffected.")
        return 0 if open_ else 3

    st = read_state()
    print("extension control: %s" % st.get("state"))
    if st.get("state") == "stop":
        print("  requested %s by %s" % (st.get("requested_utc"), st.get("by")))
    try:
        main_state = json.loads(MAIN_CONTROL.read_text(encoding="utf-8")).get("state")
    except Exception:
        main_state = "unknown"
    print("main control:      %s  (separate flag; stopping one does not stop the other)"
          % main_state)
    open_, conds = gate_state()
    print("gate:              %s" % ("OPEN" if open_ else "CLOSED -- "
                                     + ", ".join(c["condition"].split(" ")[0]
                                                 for c in conds if not c["satisfied"])))
    s = read_status()
    if s:
        print("status as of %s" % s.get("utc"))
        for k in ("phase", "benchmark", "benchmark_index", "workers_busy", "inflight",
                  "completed_unit_runs", "remaining_in_benchmark", "calls", "output_tokens",
                  "exec_queries", "exec_seconds", "elapsed_seconds", "unit_runs_per_second",
                  "benchmarks_complete"):
            if k in s:
                print("  %-24s %s" % (k, s[k]))
    else:
        print("status: no extension collector has reported yet")
    ms = manifest_summary()
    if ms:
        print("manifest: %s" % json.dumps(ms.get("by_status", {})))
        if ms.get("eta_basis"):
            print("  ETA %s (%s)" % (ms.get("eta_utc") or "unknown", ms["eta_basis"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
