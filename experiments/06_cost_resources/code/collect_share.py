from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import os
import queue
import random
import re
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import em_common as E

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")

CONDITIONS = ("no_peers", "candidate_only", "candidate_evidence", "full")
RECIPIENTS = ("qwen2.5-7b", "qwen2.5-32b")
MAX_TOKENS = 192
_lock = threading.Lock()
_n = collections.Counter()


def post(url, payload, timeout=180.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def append(path, obj):
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())


def done_ids(path):
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


_NUM = re.compile(r"\d+(?:\.\d+)?")


def field_evidence(L: Dict[str, str], R: Dict[str, str]) -> str:
    lines = []
    for f in sorted(set(L) | set(R)):
        a, b = L.get(f), R.get(f)
        if not a or not b:
            lines.append(f"  {f}: present in only one record "
                         f"({'A' if a else 'B'})")
            continue
        if a.lower() == b.lower():
            lines.append(f"  {f}: identical in A and B")
            continue
        ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
        na, nb = set(_NUM.findall(a)), set(_NUM.findall(b))
        shared = sorted(na & nb)
        bits = [f"token overlap {ratio:.2f}"]
        if shared:
            bits.append(f"shared numbers {', '.join(shared[:4])}")
        if na - nb or nb - na:
            bits.append("differing numbers present")
        lines.append(f"  {f}: differs ({'; '.join(bits)})")
    return "\n".join(lines)


def render(cond: str, L, R, peers: List[Dict[str, Any]]) -> str:
    base = E.render(L, R)
    if cond == "no_peers":
        return base
    head = "\n\nOther systems have already judged this pair:\n"
    if cond == "candidate_only":
        body = "\n".join(f"  system {i+1}: {p['decision'] or 'no answer'}"
                         for i, p in enumerate(peers))
    elif cond == "candidate_evidence":
        body = "\n".join(f"  system {i+1}: {p['decision'] or 'no answer'}"
                         for i, p in enumerate(peers))
        body += ("\n\nField-level comparison of the two records (computed from the record text; "
                 "it shows where the strings agree or differ, not whether the pair matches):\n"
                 + field_evidence(L, R))
    elif cond == "full":
        body = "\n\n".join(f"  system {i+1} said: {(p['raw_reply'] or '').strip()[:700]}"
                           for i, p in enumerate(peers))
    else:
        raise AssertionError(cond)
    tail = ("\n\nUsing this as additional information, give your own answer. "
            "Answer with exactly one word on the final line, either match or non-match.")
    return base + head + body + tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=800)
    ap.add_argument("--datasets", default="amazon_google,abt_buy,wdc_products")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    out = os.path.join(RAW, f"em_share{a.tag}.jsonl")

    init: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    with open(os.path.join(RAW, "em_preds_test.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok" or r.get("seed") != 0:
                continue
            init[(r["dataset"], r["pair_key"], r["model"])] = r

    plan = []
    for dname in a.datasets.split(","):
        ds = E.Matching(dname)
        rows = ds.splits["test"]
        rnd = random.Random(20261013)
        pick = rows if a.pairs >= len(rows) else rnd.sample(rows, a.pairs)
        for row in pick:
            pk = ds.pair_key(row)
            L, R = ds.pair_record(row)
            for rec in RECIPIENTS:
                peers = [init[(dname, pk, m)] for m in E.POOL
                         if m != rec and (dname, pk, m) in init]
                if len(peers) < 4:
                    continue
                for cond in CONDITIONS:
                    uid = hashlib.sha256(
                        f"share|{dname}|{pk}|{rec}|{cond}".encode()).hexdigest()[:20]
                    plan.append({"unit_id": uid, "dataset": dname, "pair_key": pk,
                                 "recipient": rec, "condition": cond,
                                 "prompt": render(cond, L, R, peers),
                                 "block": ds.entity_block(row),
                                 "peer_models": [p["model"] for p in peers],
                                 "peer_decisions": [p["decision"] for p in peers],
                                 "own_initial": init.get((dname, pk, rec), {}).get("decision")})
    already = done_ids(out)
    todo = [j for j in plan if j["unit_id"] not in already]
    print(f"  planned {len(plan)}, done {len(already)}, todo {len(todo)}", flush=True)
    if not todo:
        return 0
    random.Random(1).shuffle(todo)
    q: "queue.Queue" = queue.Queue()
    for j in todo:
        q.put(j)
    t0 = time.time()

    def run(job):
        port = E.POOL[job["recipient"]]["port"]
        payload = {"model": job["recipient"], "temperature": 0.0, "top_p": 1.0,
                   "max_tokens": MAX_TOKENS, "seed": 20261013,
                   "messages": [{"role": "user", "content": job["prompt"]}]}
        head = {k: job[k] for k in ("unit_id", "dataset", "pair_key", "recipient", "condition",
                                    "block", "peer_models", "peer_decisions", "own_initial")}
        head["prompt_sha"] = E.sha(job["prompt"])
        try:
            body = post(f"http://{E.HOST}:{port}/v1/chat/completions", payload)
        except Exception as exc:
            append(out, {**head, "status": "failed",
                         "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            return "failed"
        ch = (body.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content") or ""
        us = body.get("usage") or {}
        pt, ct = us.get("prompt_tokens") or 0, us.get("completion_tokens") or 0
        append(out, {**head, "status": "ok", "decision": E.parse_decision(text),
                     "raw_reply": text, "prompt_tokens": pt, "completion_tokens": ct,
                     "cost_units": E.cost_of(job["recipient"], pt, ct),
                     "finish_reason": ch.get("finish_reason"), "request": payload})
        return "ok"

    def worker():
        while True:
            job = q.get()
            if job is None:
                q.task_done()
                return
            try:
                st = run(job)
            except Exception as exc:
                st = "failed"
                append(out, {"unit_id": job["unit_id"], "status": "failed",
                             "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            with _lock:
                _n[st] += 1
                k = _n["ok"] + _n["failed"]
            if k % 1000 == 0:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
