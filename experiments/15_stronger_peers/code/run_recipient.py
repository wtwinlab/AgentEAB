r"""Qwen2.5-7B recipient calls for this round, through W3's validated client (`common.call`).

    python3 code/run_recipient.py --stage pilot --port 8051   # 36 calls -> raw/pilot.jsonl, then the gate
    python3 code/run_recipient.py --stage main  --port 8051   # 2,880 calls -> raw/recipient_qwen25-7b.jsonl

Decoding is the reused E1 recipient configuration: temperature 1.0, top_p 1.0,
max_tokens 1024, plus a per-request seed. A transport/HTTP error is retried once; every attempt is
appended, the failed one included. Resumable: units with an ok record are skipped.
"""
from __future__ import annotations

import argparse
import os
import collections
import concurrent.futures as cf
import json
import pathlib
import sys
import threading

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
for cand in (HERE, pathlib.Path(os.environ.get("W36_CODE", str(HERE)))):
    if (cand / "common.py").is_file():
        sys.path.insert(0, str(cand))
        break
import common as C                                                          # noqa: E402

MODEL = "qwen2.5-7b-instruct"
SNAPSHOT = "a09a35458c702b33eeacc393d103063234e8bc28"
DECODING = {"temperature": 1.0, "top_p": 1.0}
MAX_TOKENS = 1024
CONCURRENCY = 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("pilot", "main"), required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    src = WORK / "data" / f"requests_{a.stage}.jsonl"
    out = WORK / "raw" / ("pilot.jsonl" if a.stage == "pilot" else "recipient_qwen25-7b.jsonl")
    reqs = [json.loads(l) for l in src.open()]
    done = C.load_done(out)
    todo = [r for r in reqs if r["unit_id"] not in done]
    print(f"{a.stage}: {len(reqs)} requests, done {len(done)}, to do {len(todo)}", flush=True)
    if todo and not C.wait_ready(a.port):
        print(f"FATAL: port {a.port} never became ready", flush=True)
        return 1
    lock = threading.Lock()
    n = collections.Counter()

    def run(r):
        for attempt in (1, 2):
            rec = C.call(a.port, MODEL, r["messages"], MAX_TOKENS, DECODING, r["seed"], attempt,
                         timeout=600.0)
            o = dict(r)
            o.update(rec)
            o.update(model=MODEL, snapshot=SNAPSHOT, final_text=C.final_channel(rec),
                     finish_reason=C.finish_reason(rec), usage=C.usage(rec))
            with lock:
                C.append(out, o)
                n[f"attempt_{rec['status']}"] += 1
                if o["finish_reason"] == "length":
                    n["length"] += 1
            if rec["status"] == "ok":
                break
        with lock:
            n["units"] += 1
            if n["units"] % 200 == 0 or n["units"] == len(todo):
                print(f"  {n['units']}/{len(todo)} {dict(n)}", flush=True)

    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(run, todo))
    print(f"{a.stage} finished: {dict(n)}", flush=True)

    if a.stage == "pilot":
        last = {}
        for l in out.open():
            x = json.loads(l)
            last[x["unit_id"]] = x
        ok = [x for x in last.values() if x["status"] == "ok" and x["final_text"].strip()]
        tok_match = [x for x in last.values() if x["condition"] == "no_peers"
                     and x["usage"].get("prompt_tokens") == x["e1_prompt_tokens"]]
        n_np = sum(x["condition"] == "no_peers" for x in last.values())
        gate = dict(units=len(last), ok_nonempty=len(ok),
                    length=sum(x["finish_reason"] == "length" for x in last.values()),
                    no_peers_prompt_tokens_match_e1=f"{len(tok_match)}/{n_np}",
                    max_prompt_tokens=max((x["usage"].get("prompt_tokens") or 0)
                                          for x in last.values()))
        gate["pass"] = len(ok) >= 34 and len(tok_match) == n_np
        (WORK / "logs" / "pilot_gate.json").write_text(json.dumps(gate, indent=1) + "\n")
        print("PILOT GATE", json.dumps(gate), flush=True)
        return 0 if gate["pass"] else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
