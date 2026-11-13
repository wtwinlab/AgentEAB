r"""Collect one independent prediction per (dataset, split, pair, model, seed) for entity matching.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/collect_em.py --split valid --n 150 --seeds 1

This is the substrate for everything downstream. Each model answers each pair independently, so the
same saved table supports: the best-single-model baseline, independent sampling and voting,
ThriftLLM's budgeted selection and adaptive stopping replayed offline at any budget, and the
controlled sharing conditions that start from these frozen candidates. Collecting once and replaying
is what makes a budget grid affordable, and it is also what the ThriftLLM artifact itself expects --
its `main.py` consumes a prediction table rather than calling models.

No gold label is read anywhere in this file. `Matching.gold` exists for the evaluator and is never
called here, so an invalid output cannot be resolved by peeking at the answer.

Every attempt is recorded, including failures and unparseable replies. An unparseable reply is
stored with `decision: null` and its raw text, so the abstention rate is measurable rather than
hidden inside a wrong-answer count.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import queue
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import em_common as E

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")

MAX_TOKENS = 192
_lock = threading.Lock()
_n = collections.Counter()


def post(url: str, payload: Dict[str, Any], timeout: float = 180.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def append(path: str, obj: Dict[str, Any]) -> None:
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())


def done_ids(path: str) -> set:
    out = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        out.add(r["unit_id"])
                except Exception:
                    pass
    return out


def sample_pairs(ds: E.Matching, split: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """A stratified sample that preserves the natural positive prevalence.

    Stratification uses the label, which is an evaluator quantity, so it happens here at *sampling*
    time and never reaches a policy: the collector hands the model only the two records. Preserving
    prevalence matters because precision and the false-merge rate are meaningless at an inflated
    positive rate.
    """
    rows = ds.splits[split]
    if n >= len(rows):
        return list(rows)
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    frac = len(pos) / len(rows)
    k_pos = max(1, round(n * frac))
    rnd = random.Random(seed)
    return rnd.sample(pos, min(k_pos, len(pos))) + rnd.sample(neg, min(n - k_pos, len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="amazon_google,abt_buy,wdc_products")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    out = os.path.join(RAW, f"em_preds{a.tag}.jsonl")

    plan: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    for dname in a.datasets.split(","):
        ds = E.Matching(dname)
        rows = sample_pairs(ds, a.split, a.n, seed=20261013)
        meta[dname] = {"n_sampled": len(rows), "split": a.split,
                       "positive_rate": sum(int(r["label"]) for r in rows) / len(rows)}
        for row in rows:
            L, R = ds.pair_record(row)
            prompt = E.render(L, R)
            for model in E.POOL:
                for seed in range(a.seeds):
                    uid = hashlib.sha256(
                        f"{dname}|{a.split}|{ds.pair_key(row)}|{model}|{seed}".encode()
                    ).hexdigest()[:20]
                    plan.append({"unit_id": uid, "dataset": dname, "split": a.split,
                                 "pair_key": ds.pair_key(row), "model": model, "seed": seed,
                                 "prompt": prompt, "block": ds.entity_block(row),
                                 "left": L, "right": R})
    for d, m in meta.items():
        print(f"  {d}: {m['n_sampled']} pairs from {m['split']}, "
              f"positive rate {m['positive_rate']:.1%}")
    already = done_ids(out)
    todo = [j for j in plan if j["unit_id"] not in already]
    print(f"  planned {len(plan)} calls, done {len(already)}, todo {len(todo)}", flush=True)
    if not todo:
        return 0

    random.Random(20261013).shuffle(todo)
    q: "queue.Queue" = queue.Queue()
    for j in todo:
        q.put(j)
    t0 = time.time()

    def run(job) -> str:
        port = E.POOL[job["model"]]["port"]
        payload = {"model": job["model"], "temperature": a.temperature, "top_p": 1.0,
                   "max_tokens": MAX_TOKENS, "seed": 20261013 + job["seed"],
                   "messages": [{"role": "user", "content": job["prompt"]}]}
        head = {k: job[k] for k in ("unit_id", "dataset", "split", "pair_key", "model", "seed",
                                    "block")}
        head["prompt_sha"] = E.sha(job["prompt"])
        t1 = time.time()
        try:
            body = post(f"http://{E.HOST}:{port}/v1/chat/completions", payload)
        except Exception as exc:                                          # noqa: BLE001
            append(out, {**head, "status": "failed",
                         "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            return "failed"
        ch = (body.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content") or ""
        us = body.get("usage") or {}
        pt, ct = us.get("prompt_tokens") or 0, us.get("completion_tokens") or 0
        append(out, {
            **head, "status": "ok",
            "decision": E.parse_decision(text),           # None = abstention, label-independent
            "raw_reply": text, "finish_reason": ch.get("finish_reason"),
            "prompt_tokens": pt, "completion_tokens": ct,
            "cost_units": E.cost_of(job["model"], pt, ct),
            "latency_s": round(time.time() - t1, 3),
            "served_model": body.get("model"), "response_id": body.get("id"),
            "request": payload, "left": job["left"], "right": job["right"],
        })
        return "ok"

    def worker():
        while True:
            job = q.get()
            if job is None:
                q.task_done()
                return
            try:
                st = run(job)
            except Exception as exc:                                      # noqa: BLE001
                st = "failed"
                append(out, {"unit_id": job["unit_id"], "status": "failed",
                             "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            with _lock:
                _n[st] += 1
                k = _n["ok"] + _n["failed"]
            if k % 500 == 0:
                el = time.time() - t0
                print(f"  {k}/{len(todo)} ok={_n['ok']} failed={_n['failed']} "
                      f"{el/60:.1f}min {k/max(el,1)*60:.0f}/min", flush=True)
            q.task_done()

    th = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    for t in th:
        t.start()
    q.join()
    for _ in th:
        q.put(None)
    for t in th:
        t.join(timeout=5)
    print(f"  DONE ok={_n['ok']} failed={_n['failed']} wall={(time.time()-t0)/60:.1f}min")
    with open(os.path.join(RAW, f"em_preds{a.tag}.meta.json"), "w", encoding="utf-8") as f:
        json.dump({"datasets": meta, "pool": E.POOL, "split": a.split, "n_per_dataset": a.n,
                   "seeds": a.seeds, "temperature": a.temperature, "max_tokens": MAX_TOKENS,
                   "prompt_template_sha": E.sha(E.PROMPT)}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
