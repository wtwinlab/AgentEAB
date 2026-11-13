r"""Text-to-SQL analysis: does execution feedback change what sharing is worth?

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/analyze_sql.py

Execution accuracy is the benchmark's own measure: a candidate is correct when its result set equals
the gold query's on the database instance. That is not semantic equivalence, and the floor is
measured rather than assumed — a constant `SELECT 1` coincides with gold on about 2.6% of questions.

Inference clusters on the **database**, not the question. Questions over one schema share its
quirks, its column names and its join structure, so treating 1,534 questions as 1,534 independent
units would overstate precision badly when there are only 11 schemas. Every interval here resamples
databases.

The contrast this workload exists for is `candidate_feedback` against `candidate_only`: both show
the peers' SQL, and only one adds what those queries actually returned. QA had no analogue, because
a QA answer cannot be run.

`M.C2` holds out whole databases. A model that has seen five schemas in development and is tested on
six unseen ones is the realistic deployment question, and it is a different question from average
accuracy over a pooled set.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import sql_common as S

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")
OUT = os.path.join(BASE, "analysis")
DRAWS = 4000
SEED = 20261013
CONDS = ("no_peers", "candidate_only", "candidate_feedback", "full")


def jl(name: str) -> List[Dict[str, Any]]:
    out = []
    with open(os.path.join(RAW, name), encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                out.append(r)
    return out


def boot_mean(vals: List[Tuple[str, float]], alpha=0.05, draws=DRAWS, seed=SEED):
    """Cluster bootstrap over databases."""
    by = collections.defaultdict(list)
    for db, y in vals:
        by[db].append(y)
    keys = list(by)
    flat = [y for k in keys for y in by[k]]
    point = sum(flat) / len(flat)
    rnd = random.Random(seed)
    sims = []
    for _ in range(draws):
        pool = []
        for _ in range(len(keys)):
            pool.extend(by[keys[rnd.randrange(len(keys))]])
        sims.append(sum(pool) / len(pool) if pool else 0.0)
    sims.sort()
    return {"n": len(flat), "n_databases": len(keys), "mean": point,
            "lo": sims[int(alpha / 2 * draws)],
            "hi": sims[min(draws - 1, int((1 - alpha / 2) * draws))]}


def boot_diff(pairs: List[Tuple[str, float, float]], alpha=0.05, draws=DRAWS, seed=SEED):
    by = collections.defaultdict(list)
    for db, a, b in pairs:
        by[db].append(a - b)
    keys = list(by)
    flat = [d for k in keys for d in by[k]]
    point = sum(flat) / len(flat)
    rnd = random.Random(seed)
    sims = []
    for _ in range(draws):
        pool = []
        for _ in range(len(keys)):
            pool.extend(by[keys[rnd.randrange(len(keys))]])
        sims.append(sum(pool) / len(pool) if pool else 0.0)
    sims.sort()
    c = [s - point for s in sims]
    p = 2.0 * min(sum(1 for z in c if z >= -point), sum(1 for z in c if z <= -point)) / draws
    return {"n": len(flat), "n_databases": len(keys), "mean": point,
            "lo": sims[int(alpha / 2 * draws)],
            "hi": sims[min(draws - 1, int((1 - alpha / 2) * draws))],
            "p": max(1.0 / draws, min(1.0, p))}


def score_all(rows: List[Dict[str, Any]], byid) -> Dict[str, float]:
    """Execution match per row, cached by (db, sql) so each distinct query runs once."""
    cache: Dict[Tuple[str, str], bool] = {}
    out = {}
    for r in rows:
        sql = r.get("sql")
        if not sql:
            out[r["unit_id"]] = 0.0
            continue
        k = (r["db_id"], sql)
        if k not in cache:
            q = byid[r["question_id"]]
            cache[k] = bool(S.execution_match(r["db_id"], sql, S.gold_sql(q))["match"])
        out[r["unit_id"]] = 1.0 if cache[k] else 0.0
    return out


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    byid = {q["question_id"]: q for q in S.load_questions()}
    init = jl("sql_initial_main.jsonl")
    share = jl("sql_share_main.jsonl")
    print(f"  initial {len(init)}, sharing {len(share)}")

    print("  scoring (each distinct query executed once)...", flush=True)
    si = score_all(init, byid)
    ss = score_all(share, byid)

    res: Dict[str, Any] = {"draws": DRAWS, "seed": SEED,
                           "metric": "BIRD execution accuracy: result-set equality on the instance",
                           "caveat": "constant-query false-match floor measured at ~2.6%",
                           "clustering": "databases, not questions",
                           "initial": {}, "sharing": {}, "contrasts": {},
                           "db_disjoint": {}, "exec_health": {}}

    print(f"\n=== independent candidates ===")
    print(f"  {'model':14s} {'n':>5s} {'EX':>6s} {'95% CI':>15s} {'err':>6s} {'timeout':>8s} {'unparsed':>9s}")
    for m in sorted({r["model"] for r in init}):
        s = [r for r in init if r["model"] == m]
        b = boot_mean([(r["db_id"], si[r["unit_id"]]) for r in s])
        st = collections.Counter((r.get("exec") or {}).get("status") for r in s)
        res["initial"][m] = {**b, "exec_status": dict(st),
                             "cost_units": sum(r["cost_units"] for r in s)}
        print(f"  {m:14s} {b['n']:5d} {b['mean']:6.3f} [{b['lo']:.3f},{b['hi']:.3f}] "
              f"{st.get('error',0):6d} {st.get('timeout',0):8d} {st.get('unparsed',0):9d}")

    idx = {(r["question_id"], r["model"], r["condition"]): r for r in share}
    print(f"\n=== sharing conditions ===")
    print(f"  {'recipient':14s} {'condition':20s} {'n':>5s} {'EX':>6s} {'95% CI':>15s} {'err':>6s}")
    for rec in sorted({r["model"] for r in share}):
        for c in CONDS:
            s = [r for r in share if r["model"] == rec and r["condition"] == c]
            if not s:
                continue
            b = boot_mean([(r["db_id"], ss[r["unit_id"]]) for r in s])
            st = collections.Counter((r.get("exec") or {}).get("status") for r in s)
            res["sharing"].setdefault(rec, {})[c] = {
                **b, "exec_status": dict(st),
                "mean_prompt_tokens": sum(r["prompt_tokens"] for r in s) / len(s),
                "cost_units": sum(r["cost_units"] for r in s)}
            print(f"  {rec:14s} {c:20s} {b['n']:5d} {b['mean']:6.3f} "
                  f"[{b['lo']:.3f},{b['hi']:.3f}] {st.get('error',0):6d}")

    print(f"\n=== contrasts, paired per question, clustered on databases ===")
    print(f"  {'recipient':14s} {'contrast':42s} {'dEX':>8s} {'95% CI':>18s} {'p':>7s}")
    for rec in sorted({r["model"] for r in share}):
        qids = sorted({r["question_id"] for r in share if r["model"] == rec
                       and all((r["question_id"], rec, c) in idx for c in CONDS)})
        for a_, b_ in (("candidate_feedback", "candidate_only"),
                       ("candidate_only", "no_peers"),
                       ("candidate_feedback", "no_peers"),
                       ("full", "no_peers"),
                       ("candidate_only", "full")):
            pr = [(idx[(q, rec, a_)]["db_id"], ss[idx[(q, rec, a_)]["unit_id"]],
                   ss[idx[(q, rec, b_)]["unit_id"]]) for q in qids]
            d = boot_diff(pr)
            res["contrasts"].setdefault(rec, {})[f"{a_} - {b_}"] = d
            print(f"  {rec:14s} {a_+' - '+b_:42s} {d['mean']:+8.4f} "
                  f"[{d['lo']:+.4f},{d['hi']:+.4f}] {d['p']:7.4f}")

    # ---- M.C2 database-disjoint: split the 11 schemas, report both halves
    dbs = sorted({r["db_id"] for r in init})
    half = {d: ("dev" if int(hashlib.sha256(d.encode()).hexdigest(), 16) % 2 == 0 else "test")
            for d in dbs}
    res["db_disjoint"]["split"] = half
    print(f"\n=== database-disjoint split ===")
    print(f"  dev schemas:  {[d for d in dbs if half[d]=='dev']}")
    print(f"  test schemas: {[d for d in dbs if half[d]=='test']}")
    print(f"  {'recipient':14s} {'condition':20s} {'dev EX':>8s} {'test EX':>9s} {'gap':>7s}")
    for rec in sorted({r["model"] for r in share}):
        for c in CONDS:
            s = [r for r in share if r["model"] == rec and r["condition"] == c]
            dv = [ss[r["unit_id"]] for r in s if half[r["db_id"]] == "dev"]
            te = [ss[r["unit_id"]] for r in s if half[r["db_id"]] == "test"]
            if not dv or not te:
                continue
            res["db_disjoint"].setdefault(rec, {})[c] = {
                "dev_ex": sum(dv) / len(dv), "test_ex": sum(te) / len(te),
                "n_dev": len(dv), "n_test": len(te)}
            print(f"  {rec:14s} {c:20s} {sum(dv)/len(dv):8.3f} {sum(te)/len(te):9.3f} "
                  f"{sum(te)/len(te)-sum(dv)/len(dv):+7.3f}")

    with open(os.path.join(OUT, "sql_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {OUT}/sql_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
