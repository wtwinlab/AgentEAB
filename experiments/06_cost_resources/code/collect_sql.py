r"""Text-to-SQL collection: independent candidates, then sharing conditions with execution feedback.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/collect_sql.py --stage initial --n 60 --tag _pilot

Two stages. `initial` has every model answer each question alone, which gives the frozen candidate
set. `share` re-asks one recipient with a peer block, varying only what that block contains.

Conditions, matching the plan's SQL branch
------------------------------------------
  no_peers            the recipient alone
  candidate_only      the peers' SQL statements
  candidate_feedback  the peers' SQL **plus what each one actually returned when executed**
  full                the peers' complete replies

`candidate_feedback` is the whole point of this workload: unlike QA, a data-processing agent can run
its candidate and look at the result. The feedback comes from executing the peer's own SQL read-only
and reporting errors, columns, row counts and a bounded row sample. It never contains the gold SQL,
the gold result, or any evaluator verdict, and nothing in this file calls `gold_sql` or
`execution_match`.

Sampling is stratified by database and difficulty so a result is not driven by one schema, and the
database is recorded with every row because inference clusters on it.
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
import urllib.request
from typing import Any, Dict, List, Optional

import em_common as E
import sql_common as S

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")

POOL = ("qwen2.5-7b", "llama-3-8b", "qwen2.5-14b", "qwen2.5-32b")
RECIPIENTS = ("qwen2.5-7b", "qwen2.5-32b")
CONDITIONS = ("no_peers", "candidate_only", "candidate_feedback", "full")
MAX_TOKENS = 400
_lock = threading.Lock()
_n = collections.Counter()


def post(url, payload, timeout=240.0):
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


def sample(qs: List[Dict[str, Any]], n: int, seed: int = 20261013) -> List[Dict[str, Any]]:
    """Stratified by (database, difficulty) so no single schema or stratum dominates."""
    if n >= len(qs):
        return qs
    by = collections.defaultdict(list)
    for q in qs:
        by[(q["db_id"], q.get("difficulty"))].append(q)
    rnd = random.Random(seed)
    keys = sorted(by)
    out, i = [], 0
    while len(out) < n:
        k = keys[i % len(keys)]
        if by[k]:
            out.append(by[k].pop(rnd.randrange(len(by[k]))))
        i += 1
        if i > 100000:
            break
    return out[:n]


def peer_block(cond: str, peers: List[Dict[str, Any]]) -> str:
    if cond == "candidate_only":
        return "\n\n".join(f"  system {i+1} proposed:\n    {p['sql'] or '(no query)'}"
                           for i, p in enumerate(peers))
    if cond == "candidate_feedback":
        out = []
        for i, p in enumerate(peers):
            fb = p.get("feedback_text") or "(not executed)"
            out.append(f"  system {i+1} proposed:\n    {p['sql'] or '(no query)'}\n"
                       f"    when this query was run on the database: {fb}")
        return "\n\n".join(out)
    if cond == "full":
        return "\n\n".join(f"  system {i+1} said: {(p['raw_reply'] or '').strip()[:900]}"
                           for i, p in enumerate(peers))
    raise AssertionError(cond)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("initial", "share"), required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tag", default="_pilot")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    qs = sample(S.load_questions(), a.n)
    byid = {q["question_id"]: q for q in qs}
    print(f"  {len(qs)} questions over {len({q['db_id'] for q in qs})} databases, "
          f"difficulty {dict(collections.Counter(q.get('difficulty') for q in qs))}")

    if a.stage == "initial":
        out = os.path.join(RAW, f"sql_initial{a.tag}.jsonl")
        plan = [{"unit_id": hashlib.sha256(f"sqli|{q['question_id']}|{m}".encode()).hexdigest()[:20],
                 "question_id": q["question_id"], "db_id": q["db_id"],
                 "difficulty": q.get("difficulty"), "model": m,
                 "prompt": S.render(q)} for q in qs for m in POOL]
    else:
        out = os.path.join(RAW, f"sql_share{a.tag}.jsonl")
        init: Dict[tuple, Dict[str, Any]] = {}
        with open(os.path.join(RAW, f"sql_initial{a.tag}.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") == "ok":
                    init[(r["question_id"], r["model"])] = r
        plan = []
        for q in qs:
            for rec in RECIPIENTS:
                peers = [init[(q["question_id"], m)] for m in POOL
                         if m != rec and (q["question_id"], m) in init]
                if len(peers) < 2:
                    continue
                for cond in CONDITIONS:
                    body = S.render(q) if cond == "no_peers" else (
                        S.render(q)
                        + "\n\nOther systems have already proposed queries for this question:\n"
                        + peer_block(cond, peers)
                        + "\n\nUsing this as additional information, give your own query. "
                          "Return only the SQL query, with no explanation and no code fence.")
                    plan.append({
                        "unit_id": hashlib.sha256(
                            f"sqls|{q['question_id']}|{rec}|{cond}".encode()).hexdigest()[:20],
                        "question_id": q["question_id"], "db_id": q["db_id"],
                        "difficulty": q.get("difficulty"), "model": rec, "condition": cond,
                        "prompt": body,
                        "peer_models": [p["model"] for p in peers],
                        "own_initial_sql": (init.get((q["question_id"], rec)) or {}).get("sql")})

    already = done_ids(out)
    todo = [j for j in plan if j["unit_id"] not in already]
    print(f"  planned {len(plan)}, done {len(already)}, todo {len(todo)}", flush=True)
    if not todo:
        return 0
    random.Random(7).shuffle(todo)
    q_: "queue.Queue" = queue.Queue()
    for j in todo:
        q_.put(j)
    t0 = time.time()

    def run(job):
        port = E.POOL[job["model"]]["port"]
        payload = {"model": job["model"], "temperature": 0.0, "top_p": 1.0,
                   "max_tokens": MAX_TOKENS, "seed": 20261013,
                   "messages": [{"role": "user", "content": job["prompt"]}]}
        head = {k: v for k, v in job.items() if k != "prompt"}
        head["prompt_sha"] = E.sha(job["prompt"])
        head["prompt_chars"] = len(job["prompt"])
        try:
            body = post(f"http://{E.HOST}:{port}/v1/chat/completions", payload)
        except Exception as exc:                                          # noqa: BLE001
            append(out, {**head, "status": "failed",
                         "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            return "failed"
        ch = (body.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content") or ""
        sql = S.parse_sql(text)
        # execution feedback on the model's OWN candidate: legitimate, and gold-free
        res = S.execute(job["db_id"], sql) if sql else {"status": "unparsed"}
        us = body.get("usage") or {}
        pt, ct = us.get("prompt_tokens") or 0, us.get("completion_tokens") or 0
        append(out, {**head, "status": "ok", "sql": sql, "raw_reply": text,
                     "exec": res, "feedback_text": (S.feedback_text(res)
                                                    if res.get("status") != "unparsed" else None),
                     "prompt_tokens": pt, "completion_tokens": ct,
                     "cost_units": E.cost_of(job["model"], pt, ct),
                     "finish_reason": ch.get("finish_reason")})
        return "ok"

    def worker():
        while True:
            job = q_.get()
            if job is None:
                q_.task_done()
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
            if k % 100 == 0:
                el = time.time() - t0
                print(f"  {k}/{len(todo)} ok={_n['ok']} failed={_n['failed']} "
                      f"{el/60:.1f}min {k/max(el,1)*60:.0f}/min", flush=True)
            q_.task_done()

    th = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    for t in th:
        t.start()
    q_.join()
    for _ in th:
        q_.put(None)
    for t in th:
        t.join(timeout=5)
    print(f"  DONE ok={_n['ok']} failed={_n['failed']} wall={(time.time()-t0)/60:.1f}min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
