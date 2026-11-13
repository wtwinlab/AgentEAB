#!/usr/bin/env python3
"""Per-execution decomposition of SQL selection versus generation. Zero model calls.

    python3 sql_decompose.py --gold          # stage 1: one gold execution per question, cached
    python3 sql_decompose.py --candidates    # stage 2: label every candidate (reuse where exact)
    python3 sql_decompose.py --decompose     # stage 3: validate, then decompose the 8 differences
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "experiments/04_sql_selection_generation"
RECORDS = ROOT / "data/records"          # per-question candidates and recipient answers
OUT = ROOT / "analysis/sql_decomposition"
BENCH = ("bird", "spider")
EVAL_CAP = 10000          # the cap `execution_match` uses
BOOT = 5000
SEED = 20261025


def adapters(bench: str):
    """The benchmark's own execution module, imported exactly as its collector did."""
    for p in (str(ROOT / "experiments/06_cost_resources/code"), str(SRC / "code")):
        if p not in sys.path:
            sys.path.insert(0, p)
    if bench == "spider":
        import spider_common as A            # redirects sql_common.db_path to Spider, in-process
        return A
    import sql_common as A
    return A


def norm_rows(A, rows: Sequence[Sequence[Any]]) -> set:
    import sql_common as S
    return S._norm_result(rows)


def _rowkey(r) -> str:
    # a result set mixes None, int, float and str, which are not mutually orderable, so the
    # canonical order is by JSON text; the hash must not depend on Python's iteration order
    return json.dumps(list(r), default=str, ensure_ascii=False)


def sorted_rows(s: set) -> List[list]:
    return [list(r) for r in sorted(s, key=_rowkey)]


def rowset_hash(s: set) -> str:
    """Provenance only. NOT a match test -- see `same_rowset`."""
    return hashlib.sha256(json.dumps(sorted_rows(s), default=str,
                                     ensure_ascii=False).encode()).hexdigest()[:16]


def same_rowset(a: set, b: set) -> bool:
    """The evaluator's own test: Python set equality on coerced tuples.
    """
    return a == b


def rows_to_set(rows: Sequence[Sequence[Any]]) -> set:
    return {tuple(r) for r in rows}


def questions(bench: str) -> Dict[int, Dict[str, Any]]:
    """Questions with gold SQL, through the adapter's own `load_questions`.
    """
    A = adapters(bench)
    qs = A.load_questions()
    out = {}
    for i, q in enumerate(qs):
        qid = q.get("question_id", i)
        q.setdefault("question_id", qid)
        out[qid] = q
    return out


def jsonl_done(p: pathlib.Path, key) -> set:
    if not p.exists():
        return set()
    got = set()
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            got.add(key(json.loads(ln)))
        except Exception:
            continue
    return got


def stage_gold() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for bench in BENCH:
        A = adapters(bench)
        import sql_common as S
        qs = questions(bench)
        p = OUT / f"gold_{bench}.jsonl"
        done = jsonl_done(p, lambda r: r["question_id"])
        todo = [q for q in qs if q not in done]
        print("%s: %d questions, %d already cached, executing %d gold queries at cap %d"
              % (bench, len(qs), len(done), len(todo), EVAL_CAP))
        t0 = time.time()
        with open(p, "a", encoding="utf-8") as fh:
            for i, qid in enumerate(todo, 1):
                q = qs[qid]
                gold = q.get("SQL") or q.get("gold_sql") or ""
                res = S.execute(q["db_id"], gold, max_rows=EVAL_CAP)
                rec = {"question_id": qid, "db_id": q["db_id"], "status": res["status"],
                       "n_rows": res.get("n_rows_returned"),
                       "more_rows_exist": res.get("more_rows_exist"),
                       "elapsed_s": res.get("elapsed_s"), "error": res.get("error")}
                if res["status"] == "ok":
                    s = norm_rows(A, res["rows"])
                    rec["rowset_hash"] = rowset_hash(s)
                    rec["rows"] = sorted_rows(s)
                fh.write(json.dumps(rec, default=str) + "\n")
                if i % 200 == 0:
                    fh.flush()
                    print("   %d/%d  %.0fs" % (i, len(todo), time.time() - t0))
        bad = [r for r in read_gold(bench).values() if r["status"] != "ok"]
        trunc = [r for r in read_gold(bench).values() if r.get("more_rows_exist")]
        print("   cached: %d | gold not ok: %d | gold truncated at %d rows: %d"
              % (len(read_gold(bench)), len(bad), EVAL_CAP, len(trunc)))
    return 0


def read_gold(bench: str) -> Dict[int, Dict[str, Any]]:
    p = OUT / f"gold_{bench}.jsonl"
    out = {}
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[r["question_id"]] = r
    return out


def stage_candidates() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for bench in BENCH:
        A = adapters(bench)
        import sql_common as S
        gold = read_gold(bench)
        if not gold:
            raise SystemExit("run --gold first for %s" % bench)
        p = OUT / f"cand_{bench}.jsonl"
        done = jsonl_done(p, lambda r: (r["question_id"], r["model"]))
        rows = []
        for ln in (RECORDS / ("sql_" + bench) / "peers.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rows.append(json.loads(ln))
        todo = [r for r in rows if (r["question_id"], r["model"]) not in done]
        # `spider_common` redirects `sql_common.db_path` process-wide, so a wrong import order
        # would silently execute one benchmark's queries against the other's files.
        probe = rows[0]["db_id"]
        resolved = S.db_path(probe)
        want = "spider_src" if bench == "spider" else "dev_databases"
        if want not in resolved or not pathlib.Path(resolved).exists():
            raise SystemExit("db_path for %s resolves to %r, which is not this benchmark's tree"
                             % (bench, resolved))
        print("%s: %d candidates, %d cached, labelling %d  (db root ok: %s)"
              % (bench, len(rows), len(done), len(todo), pathlib.Path(resolved).parent.parent.name))
        n_reuse = n_exec = 0
        t0 = time.time()
        with open(p, "a", encoding="utf-8") as fh:
            for i, r in enumerate(todo, 1):
                qid = r["question_id"]
                g = gold.get(qid)
                sql = (r.get("sql") or "").strip()
                ex = r.get("exec") or {}
                rec = {"question_id": qid, "model": r["model"], "db_id": r["db_id"],
                       "has_sql": bool(sql)}
                if g is None or g["status"] != "ok":
                    rec.update(match=None, reason="gold_failed" if g else "gold_missing",
                               source="none")
                elif not sql:
                    rec.update(match=False, reason="no_sql", source="stored")
                elif ex.get("status") == "ok" and not ex.get("more_rows_exist"):
                    # the stored result is complete, so it decides set equality at any cap >= it
                    s = norm_rows(A, ex.get("rows") or [])
                    rec.update(match=same_rowset(s, rows_to_set(g["rows"])), reason="compared",
                               source="reused_stored_complete", n_pred_rows=len(s),
                               rowset_hash=rowset_hash(s))
                    n_reuse += 1
                else:
                    res = S.execute(r["db_id"], sql, max_rows=EVAL_CAP)
                    if res["status"] != "ok":
                        rec.update(match=False, reason=res["status"], source="reexecuted",
                                   error=(res.get("error") or "")[:200])
                    else:
                        s = norm_rows(A, res["rows"])
                        rec.update(match=same_rowset(s, rows_to_set(g["rows"])), reason="compared",
                                   source="reexecuted", n_pred_rows=len(s),
                                   rowset_hash=rowset_hash(s),
                                   pred_truncated=bool(res.get("more_rows_exist")))
                    n_exec += 1
                fh.write(json.dumps(rec, default=str) + "\n")
                if i % 500 == 0:
                    fh.flush()
                    print("   %d/%d  reused %d  executed %d  %.0fs"
                          % (i, len(todo), n_reuse, n_exec, time.time() - t0))
        print("   reused stored complete results: %d | re-executed: %d" % (n_reuse, n_exec))
    return 0


def read_cand(bench: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    p = OUT / f"cand_{bench}.jsonl"
    out = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[(r["question_id"], r["model"])] = r
    return out


def cluster_ci(per_q: List[Tuple[str, float]], reps: int = BOOT) -> Tuple[float, float]:
    """Database-level cluster bootstrap, the statistical unit the published analysis declares."""
    by = collections.defaultdict(list)
    for db, v in per_q:
        by[db].append(v)
    dbs = list(by)
    if len(dbs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(SEED)
    means = []
    for _ in range(reps):
        vals = []
        for _ in dbs:
            vals.extend(by[dbs[rng.randrange(len(dbs))]])
        means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--decompose", action="store_true")
    a = ap.parse_args()
    if a.gold:
        return stage_gold()
    if a.candidates:
        return stage_candidates()
    if a.decompose:
        import sql_decompose_report as R
        return R.run()
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
