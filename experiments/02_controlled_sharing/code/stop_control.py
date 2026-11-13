#!/usr/bin/env python3
"""The STOP entry point for the controlled collection, and the state it reads and writes.

    python3 stop_control.py stop      # request a stop; returns immediately
    python3 stop_control.py status    # what the collector is doing right now
    python3 stop_control.py resume    # clears the stop flag; the collector does NOT self-restart
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any, Dict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "workspace/controlled_sharing"
CONTROL = OUT / "CONTROL.json"
STATUS = OUT / "STATUS.json"

#: how long in-flight requests may take to come back once a stop is requested
DRAIN_SECONDS = 90.0


def _write_atomic(path: pathlib.Path, blob: Dict[str, Any]) -> None:
    """Write via a temp file + rename so a reader never sees a half-written control file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_state() -> Dict[str, Any]:
    """The current control state. Absent file means `run` -- a fresh checkout is not stopped."""
    try:
        return json.loads(CONTROL.read_text(encoding="utf-8"))
    except Exception:
        return {"state": "run"}


def stop_requested() -> bool:
    return read_state().get("state") == "stop"


def request_stop(by: str = "operator", note: str = "") -> Dict[str, Any]:
    blob = {"state": "stop", "requested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "by": by, "note": note,
            "contract": ["stop dispatch", "no auto-restart", f"drain <= {DRAIN_SECONDS:.0f}s",
                         "save completed results", "release this project's services only"]}
    _write_atomic(CONTROL, blob)
    return blob


def clear_stop(by: str = "operator") -> Dict[str, Any]:
    blob = {"state": "run", "cleared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "by": by,
            "note": "clearing the flag only permits a new run; it does not start one"}
    _write_atomic(CONTROL, blob)
    return blob


def write_status(blob: Dict[str, Any]) -> None:
    blob = dict(blob)
    blob["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_atomic(STATUS, blob)


def read_status() -> Dict[str, Any]:
    try:
        return json.loads(STATUS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    cmd = (sys.argv[1] if len(sys.argv) > 1 else "status").lower()
    if cmd == "stop":
        blob = request_stop(by=os.environ.get("USER", "operator"),
                            note=" ".join(sys.argv[2:]))
        print("STOP requested at %s" % blob["requested_utc"])
        for line in blob["contract"]:
            print("  - " + line)
        print("\nthe collector stops dispatching within one poll (<=2 s) and drains for at most "
              f"{DRAIN_SECONDS:.0f}s")
        print("it will NOT restart on its own; `stop_control.py resume` only clears the flag")
        return 0
    if cmd == "resume":
        blob = clear_stop(by=os.environ.get("USER", "operator"))
        print("stop flag cleared at %s -- no run was started" % blob["cleared_utc"])
        return 0
    st = read_state()
    print("control: %s" % st.get("state"))
    if st.get("state") == "stop":
        print("  requested %s by %s" % (st.get("requested_utc"), st.get("by")))
    s = read_status()
    if not s:
        print("status: no collector has reported yet")
        return 0
    print("status as of %s" % s.get("utc"))
    for k in ("phase", "benchmark", "benchmark_index", "workers_busy", "inflight",
              "completed_unit_runs", "remaining_in_benchmark", "calls", "output_tokens",
              "elapsed_seconds", "benchmarks_complete"):
        if k in s:
            print("  %-24s %s" % (k, s[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
