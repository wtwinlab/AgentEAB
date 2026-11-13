r"""Collect matching decisions on Geographic Settlements. Resumable, append-only, gold-free.

    PYTHONPATH=experiments/05_matching_downstream/code python3.13 \
      experiments/05_matching_downstream/code/collect_gs.py \
        --stage independent --split test --repeats 3

Two stages, as frozen. `independent` asks every pool model about every candidate pair, which gives
the frozen candidate set that everything downstream consumes. `share` re-asks one recipient with the
peer block varied and everything else byte-identical.

Nothing here reads a gold entity id or a gold pair label. `gs_common.Settlements.gold_pair` and
`gold_cluster` exist and are never called from this file; the collector only ever touches
`record_text`, which is the policy-visible projection.

What is preserved per call
--------------------------
The exact request including the rendered message, the complete reply, the parsed decision and its
ambiguity record, the parser version, token counts, finish reason, latency, cost, and for the
sharing stage the peer model list, their decisions, and the rendered peer block. Failed attempts are
written too, with their error, and are never later mistaken for abstentions.

Log-probabilities are requested with `logprobs: true, top_logprobs: 5`. Whatever the server returns
is stored verbatim under `logprobs`, together with a `logprob_schema` record saying what was asked
for and what arrived. Top-5 is **not** a distribution over the vocabulary and is never described as
one. If the server declines, the field records the refusal rather than being silently absent.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
import queue
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import gs_common as G

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(BASE, "raw")

_lock = threading.Lock()
_n: "collections.Counter[str]" = collections.Counter()


def post(url: str, payload: Dict[str, Any], timeout: float = 240.0) -> Dict[str, Any]:
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
    """Resume by stable unit id. Only `ok` rows count as done, so a failure is retried."""
    out = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") == "ok":
                    out.add(r["unit_id"])
    return out


def peer_block(cond: str, peers: List[Dict[str, Any]]) -> str:
    if cond == "candidate_only":
        return "\n".join(f"  system {i+1}: {p['decision'] or 'no answer'}"
                         for i, p in enumerate(peers))
    if cond == "full":
        return "\n\n".join(f"  system {i+1} said: {(p['raw_reply'] or '').strip()[:700]}"
                           for i, p in enumerate(peers))
    raise AssertionError(cond)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("independent", "share"), required=True)
    ap.add_argument("--split", default="test", choices=("dev", "test"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="pilot only; 0 = the frozen full set")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    ds = G.Settlements()
    rids = ds.records_in(a.split)
    cand = ds.block(rids, a.k)
    if a.limit:
        # deterministic pilot subset; never used for a confirmatory number
        rnd = random.Random(20261013)
        cand = sorted(rnd.sample(cand, min(a.limit, len(cand))))
    print(f"  {a.split}: {len(rids)} records, {len(cand)} candidate pairs"
          + (f" (pilot subset of {a.limit})" if a.limit else " (frozen full set)"))

    tag = a.tag or f"_{a.split}"
    if a.stage == "independent":
        out = os.path.join(RAW, f"gs_preds{tag}.jsonl")
        plan: List[Dict[str, Any]] = []
        for (x, y) in cand:
            A, B = ds.record_text(x), ds.record_text(y)
            prompt = G.render(A, B)
            for m in G.POOL:
                for rep in range(a.repeats):
                    plan.append({
                        "unit_id": hashlib.sha256(
                            f"gsi|{a.split}|{x}|{y}|{m}|{rep}".encode()).hexdigest()[:20],
                        "split": a.split, "left": x, "right": y,
                        "pair_key": ds.pair_key(x, y), "model": m, "repeat": rep,
                        "max_tokens": 192, "prompt": prompt})
    else:
        src = os.path.join(RAW, f"gs_preds{tag}.jsonl")
        cell: Dict[Tuple[str, str], Dict[str, Any]] = {}
        with open(src, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") == "ok":
                    cell[(r["pair_key"], r["model"])] = r          # last ok wins, deterministic
        out = os.path.join(RAW, f"gs_share{tag}.jsonl")
        plan = []
        for (x, y) in cand:
            pk = ds.pair_key(x, y)
            A, B = ds.record_text(x), ds.record_text(y)
            base = G.render(A, B)
            for rec in G.RECIPIENTS:
                peers = [cell[(pk, m)] for m in G.POOL
                         if m != rec and (pk, m) in cell]
                if len(peers) < 2:
                    continue
                for cond in G.CONDITIONS:
                    if cond == "no_peers":
                        body = base
                    else:
                        body = (base
                                + "\n\nOther systems have already judged this pair:\n"
                                + peer_block(cond, peers)
                                + "\n\nUsing this as additional information, give your own answer. "
                                  "Answer with exactly one word on the final line, either match or "
                                  "non-match.")
                    plan.append({
                        "unit_id": hashlib.sha256(
                            f"gss|{a.split}|{x}|{y}|{rec}|{cond}".encode()).hexdigest()[:20],
                        "split": a.split, "left": x, "right": y, "pair_key": pk,
                        "model": rec, "recipient": rec, "condition": cond,
                        "peer_models": [p["model"] for p in peers],
                        "peer_decisions": [p["decision"] for p in peers],
                        "peer_block_sha": G.sha(peer_block(cond, peers))
                                          if cond != "no_peers" else None,
                        "max_tokens": 256 if cond == "full" else 192,
                        "prompt": body})

    already = done_ids(out)
    todo = [j for j in plan if j["unit_id"] not in already]
    dupes = len(plan) - len({j["unit_id"] for j in plan})
    print(f"  planned {len(plan)} (duplicate ids {dupes}), already ok {len(already)}, "
          f"todo {len(todo)}", flush=True)
    if dupes:
        raise AssertionError("plan contains duplicate unit ids; stable-id construction is wrong")
    if not todo:
        return 0
    random.Random(7).shuffle(todo)
    q: "queue.Queue" = queue.Queue()
    for j in todo:
        q.put(j)
    t0 = time.time()

    def run(job: Dict[str, Any]) -> str:
        port = G.POOL[job["model"]]["port"]
        payload = {"model": job["model"], "temperature": 0.0, "top_p": 1.0,
                   "max_tokens": job["max_tokens"], "seed": 20261013,
                   "logprobs": True, "top_logprobs": 5,
                   "messages": [{"role": "user", "content": job["prompt"]}]}
        head = {k: v for k, v in job.items() if k != "prompt"}
        head["prompt_sha"] = G.sha(job["prompt"])
        head["prompt_chars"] = len(job["prompt"])
        head["parser_version"] = G.PARSER_VERSION
        head["request"] = {k: v for k, v in payload.items()}
        t = time.time()
        try:
            body = post(f"http://{G.HOST}:{port}/v1/chat/completions", payload)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode()[:300]
            except Exception:
                pass
            append(out, {**head, "status": "failed", "latency_s": round(time.time() - t, 3),
                         "error": f"HTTPError {exc.code}: {detail}"})
            return "failed"
        except Exception as exc:                                          # noqa: BLE001
            append(out, {**head, "status": "failed", "latency_s": round(time.time() - t, 3),
                         "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            return "failed"
        lat = round(time.time() - t, 3)
        ch = (body.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content") or ""
        lp = ch.get("logprobs")
        us = body.get("usage") or {}
        pt, ct = us.get("prompt_tokens") or 0, us.get("completion_tokens") or 0
        append(out, {**head, "status": "ok", "raw_reply": text,
                     "decision": G.parse_decision(text),
                     "ambiguity": G.parse_ambiguity(text),
                     "prompt_tokens": pt, "completion_tokens": ct,
                     "cost_units": G.cost_of(job["model"], pt, ct),
                     "finish_reason": ch.get("finish_reason"),
                     "truncated": ch.get("finish_reason") == "length",
                     "latency_s": lat, "served_model": body.get("model"),
                     "response_id": body.get("id"),
                     "logprobs": lp,
                     "logprob_schema": {
                         "requested": {"logprobs": True, "top_logprobs": 5},
                         "returned": "content" if (lp or {}).get("content") else
                                     ("empty" if lp is not None else "absent"),
                         "is_raw_logits": False,
                         "note": ("top-k alternatives at the sampled positions; NOT a distribution "
                                  "over the vocabulary and not a recoverable original decode")}})
        return "ok"

    def worker() -> None:
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
            if k % 2000 == 0:
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
    meta = {"stage": a.stage, "split": a.split, "repeats": a.repeats, "k": a.k,
            "limit": a.limit, "planned": len(plan), "todo": len(todo),
            "ok": _n["ok"], "failed": _n["failed"],
            "parser_version": G.PARSER_VERSION,
            "pool": {m: {k: v for k, v in d.items()} for m, d in G.POOL.items()},
            "prompt_sha": G.sha(G.PROMPT), "wall_min": round((time.time() - t0) / 60, 2)}
    with open(out.replace(".jsonl", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
