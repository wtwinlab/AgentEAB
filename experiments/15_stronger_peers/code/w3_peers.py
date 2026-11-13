r"""W3 stage 1: generate the new strong peer source on the frozen task set.

    python3 code/w3_peers.py                 # -> raw/w3_peer_qwen3-32b.jsonl

One generation per (task, seat), 480 x 3 = 1,440. The prompt is taken **verbatim** from the
existing `no_peers` unit for that (task, seat, decode 0), so the new source answers exactly the
problem statement the old panel answered, under the same instruction. The old panel's three seats
shared that prompt too (verified: 960 of 960 (task, panel) groups have one prompt across seats), so
the three samples are separated by seed only, which is what the old panel's seats were.

Decoding is the recorded configuration of the original Qwen3-32B run, unchanged.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import json
import pathlib
import sys

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                          # noqa: E402

MODEL = "qwen3-32b"
BASE_SEED = 20261101
OUT = C.RAW / "w3_peer_qwen3-32b.jsonl"
CONCURRENCY = 40
INPUTS = C.input_path("qa_replay_inputs.jsonl")


def plan() -> list:
    """(task, seat) with the frozen no_peers prompt, one entry each."""
    by = {}
    with INPUTS.open() as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] != "no_peers" or r["decode"] != 0:
                continue
            k = (r["task"], r["agent"])
            if k in by:
                # the same prompt is reached through both old panels; assert rather than assume
                assert by[k]["rendered_sha"] == r["rendered_sha"], k
                continue
            by[k] = r
    items = []
    for (task, seat), r in sorted(by.items()):
        items.append(dict(unit_id=f"peer|{task}|s{seat}", task=task, seat=seat,
                          benchmark=r["benchmark"], answer_kind=r["answer_kind"],
                          messages=r["messages"], rendered_sha=r["rendered_sha"],
                          seed=BASE_SEED + seat))
    return items


def main() -> int:
    items = plan()
    done = C.load_done(OUT)
    todo = [i for i in items if i["unit_id"] not in done]
    print(f"planned {len(items)}  already done {len(done)}  to do {len(todo)}", flush=True)
    bybench = collections.Counter(i["benchmark"] for i in items)
    print(f"  per benchmark: {dict(bybench)}  seats {sorted({i['seat'] for i in items})}",
          flush=True)
    if not todo:
        return 0
    if not C.wait_ready(C.PORTS["qa"]):
        print("FATAL: qa endpoint never became ready")
        return 1

    lock = __import__("threading").Lock()
    n = {"ok": 0, "err": 0, "trunc": 0}

    def run(it):
        rec = None
        for attempt in (1, 2):
            rec = C.call(C.PORTS["qa"], MODEL, it["messages"], C.QA_MAX_TOKENS,
                         C.DECODING[MODEL], it["seed"], attempt)
            if rec["status"] == "ok":
                break
        out = dict(it)
        out.pop("messages")
        out.update(rec)
        out["model"] = MODEL
        out["snapshot"] = C.SNAPSHOT[MODEL]
        out["messages"] = it["messages"]
        out["final_text"] = C.final_channel(rec)
        out["finish_reason"] = C.finish_reason(rec)
        out["usage"] = C.usage(rec)
        with lock:
            C.append(OUT, out)
            if rec["status"] == "ok":
                n["ok"] += 1
                if out["finish_reason"] == "length":
                    n["trunc"] += 1
            else:
                n["err"] += 1
            t = n["ok"] + n["err"]
            if t % 50 == 0 or t == len(todo):
                print(f"  {t}/{len(todo)} ok={n['ok']} err={n['err']} truncated={n['trunc']}",
                      flush=True)

    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(run, todo))
    print(f"done ok={n['ok']} err={n['err']} truncated={n['trunc']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
