#!/usr/bin/env python3
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

MAIN_OUT = ROOT / "workspace/controlled_sharing"
MAIN_CONTROL = MAIN_OUT / "CONTROL.json"
SYNTH = ROOT / "analysis/controlled_sharing/rq_results.json"

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


def read_state() -> Dict[str, Any]:
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


def gate_state() -> Tuple[bool, List[Dict[str, Any]]]:
    conds: List[Dict[str, Any]] = []

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
    except Exception as exc:
        all_done = False
        detail["error"] = f"{type(exc).__name__}: {exc}"[:200]
    conds.append({"condition": "collect_controlled.design_complete for all 5 benchmarks",
                  "satisfied": all_done, "detail": detail})

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
            if not names:
                text = SYNTH.read_text(encoding="utf-8")
                names = {b for b in GATE_BENCHMARKS if f'"{b}"' in text}
            missing = [b for b in GATE_BENCHMARKS if b not in names]
            ok2 = not missing
            cov = {"found": sorted(names & set(GATE_BENCHMARKS)), "missing": missing}
        except Exception as exc:
            cov = f"unreadable: {type(exc).__name__}"
    conds.append({"condition": "analysis/controlled_sharing/rq_results.json covers all 5",
                  "satisfied": ok2, "detail": cov})

    g: Any = "absent"
    ok3 = False
    if GATE.exists():
        try:
            g = json.loads(GATE.read_text(encoding="utf-8"))
            ok3 = str(g.get("state")).lower() == "open"
        except Exception as exc:
            g = f"unreadable: {type(exc).__name__}"
    conds.append({"condition": f"{GATE.relative_to(ROOT)} says state=open",
                  "satisfied": ok3, "detail": g})

    return all(c["satisfied"] for c in conds), conds


def require_gate() -> Tuple[bool, List[Dict[str, Any]]]:
    open_, conds = gate_state()
    _write_atomic(OUT / "GATE_CHECKS.json", {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "open": open_, "conditions": conds,
        "note": "re-evaluated from disk at this instant; never cached across launches"})
    return open_, conds


def manifest_summary() -> Dict[str, Any]:
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
            except Exception as exc:
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
