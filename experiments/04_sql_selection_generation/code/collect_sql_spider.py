r"""P4: the BIRD evidence-versus-generation and feedback-decomposition conditions, run on Spider 1.0 dev.

    PYTHONPATH=code python3 code/collect_sql_spider.py --stage peers
    PYTHONPATH=code python3 code/collect_sql_spider.py --stage conditions
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
from typing import Any, Dict, List

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
sys.path.insert(0, os.path.join(STAGE, "code"))

import collect_sql_arms as CS            # noqa: E402  arm logic, unchanged
import spider_common as SP               # noqa: E402  benchmark adapter

DATA = os.path.join(STAGE, "data/spider")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["peers", "arms"])
    ap.add_argument("--arms", default=",".join(CS.ARMS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    if a.workers:
        CS.WORKERS = a.workers
    os.makedirs(DATA, exist_ok=True)

    qs = SP.load_questions()
    if a.limit:
        qs = qs[:a.limit]
    print(f"Spider dev: {len(qs)} questions, {len({q['db_id'] for q in qs})} databases, "
          f"stage={a.stage}", flush=True)

    peers_path = os.path.join(DATA, "peers.jsonl")
    arms_path = os.path.join(DATA, "arms.jsonl")
    stats: Dict[str, Any] = {"n_questions": len(qs), "benchmark": "spider_dev",
                             "provenance": SP.PROVENANCE}

    # ------------------------------------------------------------- peer + extra candidates
    if a.stage == "peers":
        have = CS.done_units(peers_path)
        jobs: List[Dict[str, Any]] = []
        for q in qs:
            pr = SP.render(q)
            for m in tuple(CS.PEER_MODELS) + tuple(CS.EXTRA_GENERATORS):
                u = CS.uid("spiderpeer", q["question_id"], m)
                if u not in have:
                    jobs.append({"unit_id": u, "q": q, "model": m, "prompt": pr,
                                 "role": "extra" if m in CS.EXTRA_GENERATORS else "peer"})
        print(f"  peers+extra: {len(jobs)} calls ({len(have)} present)", flush=True)
        random.Random(CS.SEED).shuffle(jobs)

        def dop(j: Dict[str, Any]) -> None:
            g = CS.post(j["model"], j["prompt"])
            sql = SP.parse_sql(g["text"])
            ex = SP.execute(j["q"]["db_id"], sql) if sql else {"status": "error",
                                                               "error": "no sql parsed"}
            CS.append(peers_path, {
                "unit_id": j["unit_id"], "question_id": j["q"]["question_id"],
                "db_id": j["q"]["db_id"], "difficulty": j["q"]["difficulty"],
                "model": j["model"], "role": j["role"], "status": "ok",
                "sql": sql, "raw_reply": g["text"], "exec": ex,
                "exec_status": ex.get("status"),
                "prompt_tokens": g["prompt_tokens"],
                "completion_tokens": g["completion_tokens"],
                "finish_reason": g["finish_reason"], "benchmark": "spider_dev"})
        stats["peers"] = CS.run(jobs, dop, "spider peers+extra") if jobs else {"n_ok": 0}

    # ------------------------------------------------------------- the recipient conditions
    if a.stage == "arms":
        want = [x for x in a.arms.split(",") if x in CS.ARMS]
        rows = [json.loads(l) for l in open(peers_path, encoding="utf-8")]
        by_q: Dict[int, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
        for r in rows:
            by_q[r["question_id"]][r["model"]] = r
        have = CS.done_units(arms_path)
        qmap = {q["question_id"]: q for q in qs}
        jobs = []
        skipped = collections.Counter()
        for qid, got in by_q.items():
            q = qmap.get(qid)
            if q is None:
                continue
            base = SP.render(q)
            peers = [got[m] for m in CS.PEER_MODELS if m in got and got[m].get("sql")]
            extra = [got[m] for m in CS.EXTRA_GENERATORS if m in got and got[m].get("sql")]
            if len(peers) < 2:
                skipped["fewer than two parsed peer queries"] += 1
                continue
            pick34 = CS.majority_pick(peers + extra)
            for rec in CS.RECIPIENTS:
                for arm in want:
                    for rep in range(CS.REPEATS):
                        u = CS.uid("spiderarm", qid, rec, arm, rep)
                        if u in have:
                            continue
                        jobs.append({"unit_id": u, "q": q, "recipient": rec, "arm": arm,
                                     "repeat": rep, "base": base, "peers": peers,
                                     "extra": extra, "pick34": pick34})
        print(f"  arms {want}: {len(jobs)} calls ({len(have)} present); "
              + ", ".join(f"{v} {k}" for k, v in skipped.items()), flush=True)
        random.Random(CS.SEED).shuffle(jobs)

        def doa(j: Dict[str, Any]) -> None:
            arm, q = j["arm"], j["q"]
            peers, extra, pick = j["peers"], j["extra"], j["pick34"]
            level = {"S2": "F3", "F3": "F3", "F1": "F1", "F2": "F2"}.get(arm)
            if arm == "S0":
                prompt = j["base"]
            elif arm == "S1":
                prompt = j["base"] + CS.PEER_FRAME.format(block=CS.peer_block(peers, None))
            elif arm in ("S2", "F1", "F2"):
                prompt = j["base"] + CS.PEER_FRAME.format(block=CS.peer_block(peers, level))
            elif arm in ("S3", "S4"):
                lvl = "F3" if arm == "S3" else None
                prompt = j["base"] + CS.PEER_FRAME.format(
                    block=CS.peer_block(peers + extra, lvl))
            else:
                raise AssertionError(arm)
            g = CS.post(j["recipient"], prompt, port=CS.PORTS[j["recipient"]])
            sql = SP.parse_sql(g["text"])
            ex = SP.execute(q["db_id"], sql) if sql else {"status": "error",
                                                          "error": "no sql parsed"}
            ev = (SP.execution_match(q["db_id"], sql, q["SQL"]) if sql
                  else {"match": False, "reason": "no sql parsed"})
            CS.append(arms_path, {
                "unit_id": j["unit_id"], "question_id": q["question_id"],
                "db_id": q["db_id"], "difficulty": q["difficulty"],
                "recipient": j["recipient"], "arm": arm, "repeat": j["repeat"],
                "role": "recipient", "status": "ok", "sql": sql, "raw_reply": g["text"],
                "exec": ex, "exec_status": ex.get("status"),
                "exec_match": bool(ev.get("match")), "eval_reason": ev.get("reason"),
                "n_peers": len(peers), "n_candidates": len(peers) + (
                    len(extra) if arm in ("S3", "S4") else 0),
                "frozen_pick_sql": (pick or {}).get("sql") if arm in ("S3", "S4") else None,
                "source_statuses": [p["exec_status"] for p in peers],
                "prompt_tokens": g["prompt_tokens"],
                "completion_tokens": g["completion_tokens"],
                "truncated": g["finish_reason"] == "length",
                "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                "benchmark": "spider_dev"})
        stats["arms"] = CS.run(jobs, doa, f"spider arms:{'+'.join(want)}") if jobs \
            else {"n_ok": 0}
        stats["skipped"] = dict(skipped)

    sp = os.path.join(STAGE, "logs", f"collect_spider_{a.stage}.json")
    json.dump(stats, open(sp, "w", encoding="utf-8"), indent=1, default=str)
    print(f"  wrote {sp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
