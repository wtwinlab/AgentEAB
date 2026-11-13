r"""W6 follow-up arms B and C.

    python3 code/w6_arms.py --arm B --pilot
    python3 code/w6_arms.py --arm C --pilot
    python3 code/w6_arms.py --arm B --main
    python3 code/w6_arms.py --arm C --main

B: per-question random permutation, thinking off, temperature 0, 256 tokens.
C: the identical permutation, thinking on, Qwen3 thinking-mode sampling, 32,768 tokens.
Prompt text, pool, feedback and parse rule are arm A's (`w6_selector.py`), imported, not copied.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import itertools
import json
import pathlib
import random
import sys
import threading

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                          # noqa: E402
import w6_selector as A                                                     # noqa: E402

PORTS = (8041, 8042)
SEED = 20261101
PILOT_N = 12
ARMS = {
    "B": dict(max_tokens=256,
              decoding={"temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}},
              concurrency=48, timeout=600.0),
    "C": dict(max_tokens=32768,
              decoding={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                        "chat_template_kwargs": {"enable_thinking": True}},
              concurrency=96, timeout=3600.0),
}


def permuted(item_a: dict, rec_by_model: dict) -> dict:
    """Rebuild arm A's prompt for one question with a seeded permutation of its candidates."""
    bench, qid = item_a["benchmark"], item_a["question_id"]
    models = [m for m in A.POOL if m in item_a["id_to_model"].values()]
    order = list(models)
    random.Random(f"{SEED}|{bench}|{qid}").shuffle(order)
    blocks, mapping = [], {}
    for i, m in enumerate(order):
        cid = f"C{i + 1}"
        mapping[cid] = m
        rec = rec_by_model[m]
        blocks.append(f"{cid}:\n  SQL: {(rec.get('sql') or '').strip()}\n"
                      f"  {A.feedback(rec.get('exec'), rec.get('exec_status'))}\n")
    head = item_a["prompt"][:item_a["prompt"].index("Candidates, each with the result of running it:")]
    tail = item_a["prompt"][item_a["prompt"].index("Reply with the identifier"):]
    prompt = (head + "Candidates, each with the result of running it:\n\n" + "\n".join(blocks)
              + "\n" + tail.replace(f"for example: {A.IDS[0]}", "for example: C1"))
    return dict(item_a, prompt=prompt, id_to_model=mapping, order=order,
                identity_permutation=(order == models))


def items() -> list:
    peers = {}
    for b in ("bird", "spider"):
        with C.input_path(f"peers_{b}.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                peers[(b, int(r["question_id"]), r["model"])] = r
    out = []
    for b in ("bird", "spider"):
        its, _ = A.load(b)
        for it in its:
            rm = {m: peers[(b, it["question_id"], m)] for m in it["id_to_model"].values()}
            out.append(permuted(it, rm))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--main", action="store_true")
    a = ap.parse_args()
    cfg = ARMS[a.arm]
    tag = "pilot" if a.pilot else "main"
    out = C.RAW / f"w6_arm{a.arm}_{tag}.jsonl"
    allit = items()
    if a.pilot:
        allit = [i for b in ("bird", "spider") for i in allit if i["benchmark"] == b][:0] + \
                [i for i in allit if i["benchmark"] == "bird"][:PILOT_N] + \
                [i for i in allit if i["benchmark"] == "spider"][:PILOT_N]
    (C.OUT / f"w6_arm{a.arm}_{tag}_idmap.json").write_text(json.dumps(
        {i["unit_id"]: i["id_to_model"] for i in allit}, indent=0) + "\n")
    done = C.load_done(out)
    todo = [i for i in allit if i["unit_id"] not in done]
    ident = sum(i["identity_permutation"] for i in allit)
    print(f"[{a.arm} {tag}] planned {len(allit)} done {len(done)} to do {len(todo)}; "
          f"permutation equals pool order for {ident}", flush=True)
    for p in PORTS:
        if not C.wait_ready(p):
            print(f"FATAL: port {p} not ready"); return 1
    lock = threading.Lock()
    n = collections.Counter()
    ports = itertools.cycle(PORTS)

    def run(it):
        with lock:
            port = next(ports)
        msgs = [{"role": "user", "content": it["prompt"]}]
        rec = C.call(port, A.MODEL, msgs, cfg["max_tokens"], cfg["decoding"], SEED, 1,
                     timeout=cfg["timeout"])
        if rec["status"] != "ok":
            rec = C.call(port, A.MODEL, msgs, cfg["max_tokens"], cfg["decoding"], SEED, 2,
                         timeout=cfg["timeout"])
        final = C.final_channel(rec)
        cid, why = A.parse(final, it["id_to_model"])
        msg = ((rec.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
        o = dict(it); o.update(rec)
        o.update(messages=msgs, arm=a.arm, model=A.MODEL, snapshot=C.SNAPSHOT[A.MODEL],
                 final_text=final, reasoning_text=msg.get("reasoning_content") or msg.get("reasoning"),
                 finish_reason=C.finish_reason(rec), usage=C.usage(rec), picked_id=cid,
                 picked_model=it["id_to_model"].get(cid) if cid else None,
                 nonselection_reason=why)
        with lock:
            C.append(out, o)
            n[rec["status"]] += 1
            n["picked" if cid else f"nonsel:{(why or 'error').split(':')[0]}"] += 1
            n[f"finish:{o['finish_reason']}"] += 1
            t = n["ok"] + n["error"]
            if t % 100 == 0 or t == len(todo):
                print(f"  {t}/{len(todo)} {dict(n)}", flush=True)

    with cf.ThreadPoolExecutor(max_workers=cfg["concurrency"]) as ex:
        list(ex.map(run, todo))
    print(f"[{a.arm} {tag}] done {dict(n)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
