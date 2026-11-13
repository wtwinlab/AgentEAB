#!/usr/bin/env python3
"""Execution verdicts for **every** query the SQL records contain, not only the final answers.

    python3 ext_sql_verdicts.py            # writes analysis/sql_verdicts.json
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import hashlib
import json
import pathlib
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE, ROOT / "experiments/06_cost_resources/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ext_design as D                                                   # noqa: E402

OUT = D.OUT
RAW = OUT / "raw"
ANALYSIS = OUT / "analysis"
BENCH = "sql_bird_v2"
WORKERS = 48


def qkey(db: str, sql: str) -> str:
    return hashlib.sha256(f"{db}|{sql}".encode()).hexdigest()[:24]


def _one(args: Tuple[str, str, str]) -> Tuple[str, Dict[str, Any]]:
    """One (db, candidate, gold) comparison in this worker. Imported fresh, so no poison applies."""
    db, pred, gold = args
    try:
        import sql_common as SQ
        t0 = time.time()
        pr = SQ.execute(db, pred, max_rows=10000)
        if pr["status"] != "ok":
            import ext_eval_sql as EV
            out = {"correct": False,
                   "outcome": ("exec_timeout" if pr["status"] == "timeout"
                               else EV.classify_error(pr.get("error") or "")),
                   "queries": 1, "seconds": round(time.time() - t0, 3)}
            return qkey(db, pred), out
        gr = SQ.execute(db, gold, max_rows=10000)
        if gr["status"] != "ok":
            return qkey(db, pred), {"correct": None, "outcome": "gold_failed", "queries": 2,
                                    "seconds": round(time.time() - t0, 3)}
        same = SQ._norm_result(pr["rows"]) == SQ._norm_result(gr["rows"])
        return qkey(db, pred), {"correct": bool(same),
                                "outcome": "match" if same else "mismatch", "queries": 2,
                                "seconds": round(time.time() - t0, 3)}
    except Exception as exc:                                              # noqa: BLE001
        return qkey(db, pred), {"correct": False, "outcome": "other_exec_error",
                                "error": f"{type(exc).__name__}: {exc}"[:200], "queries": 0,
                                "seconds": 0.0}


def collect_candidates() -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    """Every distinct (db, query) that appears anywhere in the SQL records, with its gold."""
    import sql_tasks as ST
    import sql_common as SQ
    gold = {f"bird|{q['question_id']}": q["SQL"]
            for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text(encoding="utf-8"))}
    seen: Set[str] = set()
    out: List[Tuple[str, str, str]] = []
    where = collections.Counter()
    for p in sorted(RAW.glob(f"{BENCH}__*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok":
                continue
            db = ((r.get("config") or {}).get("unit_meta") or {}).get("db_id")
            g = gold.get(r.get("unit"))
            if not db or not g:
                continue
            rec = r.get("record") or {}
            cands: List[Tuple[str, Any]] = [("final", rec.get("majority_answer"))]
            cands += [("per_agent", a) for a in (rec.get("per_agent_final_answer") or [])]
            for c in rec.get("calls") or []:
                for f in ("extracted_answer", "prediction", "normalised_answer"):
                    if c.get(f) is not None:
                        cands.append(("per_call", c[f]))
                        break
            for src, a in cands:
                s = ST.unwrap_sql(a) if a not in (None, "", "None", "none") else ""
                if not s:
                    continue
                k = qkey(db, s)
                where[src] += 1
                if k in seen:
                    continue
                seen.add(k)
                out.append((db, s, g))
    return out, dict(where)


def main() -> int:
    import ext_collect as EC
    okc, have, need = EC.design_complete(BENCH)
    cands, where = collect_candidates()
    print("distinct (database, query) pairs to execute: %s   (occurrences by position: %s)"
          % (f"{len(cands):,}", where))
    t0 = time.time()
    verdicts: Dict[str, Dict[str, Any]] = {}
    cost = collections.Counter()
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, (k, v) in enumerate(ex.map(_one, cands, chunksize=8), 1):
            verdicts[k] = v
            cost["queries"] += v.get("queries", 0)
            cost["seconds"] += 0
            cost[v["outcome"]] += 1
            if i % 2000 == 0:
                print("  %s/%s   %.0fs" % (f"{i:,}", f"{len(cands):,}", time.time() - t0))
    el = time.time() - t0
    blob = {
        "benchmark": BENCH, "run_id": D.RUN_ID,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design_complete": okc, "design_have": have, "design_need": need,
        "what": "an execution verdict per distinct (database, query) appearing anywhere in the "
                "records -- final answers, per-agent final answers and per-call answers -- so the "
                "correction / regression / availability-loss / selection-miss diagnostics are "
                "execution-based rather than string comparisons against the reference query",
        "identity": "single-instance execution match (set equality of result tuples); a match is "
                    "evidence on this database instance, not proven semantic equivalence",
        "occurrences_by_position": where,
        "execution_cost": {"distinct_queries": len(cands), "queries_executed": cost["queries"],
                           "wall_seconds": round(el, 1), "workers": WORKERS,
                           "outcomes": {k: v for k, v in cost.items()
                                        if k not in ("queries", "seconds")},
                           "note": "execution cost, reported separately and never added to a "
                                   "token count. This is the diagnostics pass; the endpoint "
                                   "evaluation's own cost is in sql_execution.json"},
        "verdicts": verdicts,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    p = ANALYSIS / "sql_verdicts.json"
    p.write_text(json.dumps(blob, default=str) + "\n", encoding="utf-8")
    print("%s verdicts, %s queries, %.0fs -> %s"
          % (f"{len(verdicts):,}", f"{cost['queries']:,}", el, p.relative_to(ROOT)))
    print("outcomes: %s" % blob["execution_cost"]["outcomes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
