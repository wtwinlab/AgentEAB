#!/usr/bin/env python3
"""Endpoint evaluation for the SQL extension: read-only, process-isolated, cost kept separate.

    python3 ext_eval_sql.py                 # evaluate every delivered SQL answer on disk
    python3 ext_eval_sql.py --allow-partial
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import pathlib
import re
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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

#: per-query hard limit enforced by the parent, above sql_common's own 5 s statement timeout
HARD_TIMEOUT_S = 20.0
ROW_CAP = 10000

_MISSING = re.compile(r"no such (table|column|function)", re.I)
_SYNTAX = re.compile(r"(syntax error|incomplete input|unrecognized token)", re.I)
_TYPE = re.compile(r"(datatype mismatch|wrong number of arguments|misuse of aggregate|"
                   r"ambiguous column name|too many terms)", re.I)


def classify_error(msg: str) -> str:
    m = msg or ""
    if _SYNTAX.search(m):
        return "syntax_error"
    if _MISSING.search(m):
        return "missing_object"
    if _TYPE.search(m):
        return "type_or_arg_error"
    return "other_exec_error"


# ---------------------------------------------------------------- isolated execution
def _child(db_id: str, pred: str, gold: str, q: "mp.Queue") -> None:
    """Runs in its own process. Imports sql_common there, so the poison never applies here."""
    try:
        sys.path[:0] = [str(ROOT / "experiments/06_cost_resources/code")]
        import sql_common as SQ
        t0 = time.time()
        pr = SQ.execute(db_id, pred, max_rows=ROW_CAP)
        if pr["status"] != "ok":
            q.put({"outcome": ("exec_timeout" if pr["status"] == "timeout"
                               else classify_error(pr.get("error") or "")),
                   "error": (pr.get("error") or "")[:300],
                   "pred_seconds": pr.get("elapsed_s"), "gold_seconds": None,
                   "queries": 1, "seconds": round(time.time() - t0, 3)})
            return
        gr = SQ.execute(db_id, gold, max_rows=ROW_CAP)
        if gr["status"] != "ok":
            q.put({"outcome": "gold_failed", "error": (gr.get("error") or "")[:200],
                   "pred_seconds": pr.get("elapsed_s"), "gold_seconds": gr.get("elapsed_s"),
                   "queries": 2, "seconds": round(time.time() - t0, 3)})
            return
        same = SQ._norm_result(pr["rows"]) == SQ._norm_result(gr["rows"])
        q.put({"outcome": "match" if same else "mismatch", "error": None,
               "n_pred_rows": pr["n_rows_returned"], "n_gold_rows": gr["n_rows_returned"],
               "pred_seconds": pr.get("elapsed_s"), "gold_seconds": gr.get("elapsed_s"),
               "queries": 2, "seconds": round(time.time() - t0, 3)})
    except Exception as exc:                                              # noqa: BLE001
        q.put({"outcome": "other_exec_error", "error": f"{type(exc).__name__}: {exc}"[:300],
               "queries": 0, "seconds": 0.0})


def execute_isolated(db_id: str, pred: str, gold: str) -> Dict[str, Any]:
    """One comparison, in a process the parent can kill. Read-only, bounded, no gold in a prompt."""
    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue()
    p = ctx.Process(target=_child, args=(db_id, pred, gold, q), daemon=True)
    t0 = time.time()
    p.start()
    p.join(HARD_TIMEOUT_S)
    if p.is_alive():
        p.kill()
        p.join(5)
        return {"outcome": "exec_timeout", "error": f"killed after {HARD_TIMEOUT_S}s by the "
                                                    f"parent; the statement timeout did not fire",
                "queries": 1, "seconds": round(time.time() - t0, 3)}
    try:
        return q.get_nowait()
    except Exception:
        return {"outcome": "other_exec_error",
                "error": "the isolated executor produced no result", "queries": 0,
                "seconds": round(time.time() - t0, 3)}


# ---------------------------------------------------------------- the evaluation
def rows_on_disk() -> List[Dict[str, Any]]:
    out = []
    for p in sorted(RAW.glob(f"{BENCH}__*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true",
                    help="evaluate an unfinished benchmark; the output is marked incomplete and "
                         "must not be read as the benchmark's result")
    ap.add_argument("--limit", type=int, default=0, help="debugging only; marks the output")
    a = ap.parse_args()

    import ext_collect as EC
    okc, have, need = EC.design_complete(BENCH)
    if not okc and not a.allow_partial:
        print("sql_bird_v2 is %s/%s unit-runs; pass --allow-partial to evaluate anyway"
              % (f"{have:,}", f"{need:,}"))
        return 3

    import sql_tasks as ST
    gold = {f"bird|{q['question_id']}": q["SQL"]
            for q in json.loads((pathlib.Path(
                __import__("sql_common").BIRD) / "dev.json").read_text(encoding="utf-8"))}
    meta = {}
    for r in rows_on_disk():
        cfg = (r.get("config") or {}).get("unit_meta") or {}
        if cfg.get("db_id"):
            meta[r["unit"]] = cfg

    rows = rows_on_disk()
    if a.limit:
        rows = rows[: a.limit]
    # one execution per DISTINCT delivered query per unit: the same string from two conditions is the
    # same query on the same instance, so executing it twice would charge the benchmark twice for
    # one fact.
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    per_cell: Dict[Any, collections.Counter] = {}
    per_unit: List[Dict[str, Any]] = []
    cost = {"queries_executed": 0, "cache_hits": 0, "seconds": 0.0, "timeouts": 0,
            "processes_spawned": 0, "killed_by_parent": 0}
    t_all = time.time()

    for r in rows:
        cellk = (r.get("protocol"), r.get("arm"), r.get("backbone"), r.get("repeat"))
        c = per_cell.setdefault(cellk, collections.Counter())
        if r.get("status") != "ok":
            c[r.get("error_class") or "collection_error"] += 1
            continue
        rec = r.get("record") or {}
        ans = rec.get("majority_answer")
        pred = ST.unwrap_sql(ans) if ans not in (None, "", "None", "none") else ""
        if not pred:
            c["truncated_only" if (r.get("truncated_calls") or 0) else "declined"] += 1
            per_unit.append({"unit": r["unit"], **dict(zip(
                ("protocol", "arm", "backbone", "repeat"), cellk)), "outcome": "declined",
                "truncated_calls": r.get("truncated_calls") or 0})
            continue
        db = (meta.get(r["unit"]) or {}).get("db_id")
        g = gold.get(r["unit"])
        if not db or not g:
            c["no_gold"] += 1
            continue
        key = (db, pred)
        res = cache.get(key)
        if res is None:
            res = execute_isolated(db, pred, g)
            cache[key] = res
            cost["queries_executed"] += res.get("queries", 0)
            cost["processes_spawned"] += 1
            cost["seconds"] += res.get("seconds", 0.0)
            if res["outcome"] == "exec_timeout":
                cost["timeouts"] += 1
                if "killed after" in str(res.get("error")):
                    cost["killed_by_parent"] += 1
        else:
            cost["cache_hits"] += 1
        c[res["outcome"]] += 1
        per_unit.append({"unit": r["unit"], **dict(zip(
            ("protocol", "arm", "backbone", "repeat"), cellk)),
            "outcome": res["outcome"], "error": res.get("error"),
            "truncated_calls": r.get("truncated_calls") or 0,
            "db_id": db, "difficulty": (meta.get(r["unit"]) or {}).get("difficulty")})

    report: Dict[str, Any] = {
        "benchmark": BENCH, "run_id": D.RUN_ID,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design_complete": okc, "design_have": have, "design_need": need,
        "incomplete_warning": (None if okc else
                               "evaluated with --allow-partial: these numbers describe the "
                               "records collected so far, not the benchmark"),
        "limited_debug_run": bool(a.limit),
        "identity": "single-instance execution match (set equality of result tuples) -- the "
                    "benchmark's own notion, not proven semantic equivalence",
        "read_only": {"connection": "mode=ro + PRAGMA query_only=ON",
                      "statement_timeout_s": 5.0,
                      "parent_hard_timeout_s": HARD_TIMEOUT_S,
                      "row_cap": ROW_CAP,
                      "isolation": "one short-lived subprocess per distinct query, killed by the "
                                   "parent if it outlives the limit"},
        "execution_feedback_to_any_arm": False,
        "execution_cost": {**cost,
                           "wall_seconds_total": round(time.time() - t_all, 2),
                           "distinct_queries": len(cache),
                           "note": "execution cost is reported here and is NEVER added to a token "
                                   "count; a cached hit is one query asked twice, not two facts"},
        "per_cell": {},
        "per_protocol": {},
    }
    for cellk, c in sorted(per_cell.items()):
        scored = c["match"] + c["mismatch"]
        charged = scored + c["declined"] + c["truncated_only"]
        report["per_cell"]["|".join(str(x) for x in cellk)] = {
            "outcomes": dict(c), "n_scored": scored, "n_matched": c["match"],
            "execution_accuracy_delivered_only": round(c["match"] / scored, 6) if scored else None,
            "execution_accuracy_declines_charged": (round(c["match"] / charged, 6)
                                                    if charged else None)}
    for proto in D.PROTOS:
        ks = [k for k in report["per_cell"] if k.split("|")[0] == proto]
        accs = [report["per_cell"][k]["execution_accuracy_delivered_only"] for k in ks
                if report["per_cell"][k]["execution_accuracy_delivered_only"] is not None]
        if not accs:
            continue
        report["per_protocol"][proto] = {
            "cells": len(ks),
            "execution_accuracy_delivered_only_mean": round(statistics.fmean(accs), 6),
            "sd": round(statistics.stdev(accs), 6) if len(accs) > 1 else None}

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    p = ANALYSIS / ("sql_execution%s.json" % ("_partial" if not okc else ""))
    p.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    (ANALYSIS / "sql_per_unit.jsonl").write_text(
        "".join(json.dumps(x, default=str) + "\n" for x in per_unit), encoding="utf-8")

    tax = collections.Counter()
    for c in per_cell.values():
        tax.update(c)
    print("error taxonomy over %s unit-runs: %s" % (f"{sum(tax.values()):,}", dict(tax)))
    print("execution cost: %s queries in %d processes, %.1fs wall, %d timeouts, %s cache hits"
          % (f"{cost['queries_executed']:,}", cost["processes_spawned"],
             report["execution_cost"]["wall_seconds_total"], cost["timeouts"],
             f"{cost['cache_hits']:,}"))
    print("-> %s" % p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
