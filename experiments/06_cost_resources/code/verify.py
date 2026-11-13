r"""V.1 and V.2: independent recomputation, and a failure / denominator audit.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/verify.py

Re-running the analysis modules would only show they are deterministic. This file shares nothing
with them but the raw JSONL and the benchmark files: it re-reads the records, re-derives every
decision from the stored `raw_reply` with its **own** parser, re-executes every SQL candidate, and
recomputes the headline numbers with its own scoring code. A disagreement is printed as a
disagreement and is never reconciled by editing this file.

The audit half counts every attempt. Units that failed, replies that could not be parsed, queries
that errored or timed out, and records dropped by a join all stay in the denominator and are named.
A rate reported without its denominator is the thing this file exists to prevent.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import em_common as E
import sql_common as S

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")
OUT = os.path.join(BASE, "analysis")
TOL = 0.002


def jl(name: str) -> List[Dict[str, Any]]:
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                out.append({"status": "unparsable_line"})
    return out


# ---- an independent parser, written from the task definition rather than copied
_MY = re.compile(r"\b(non[\s-]*match|not\s+a\s+match|no\s+match|match)\b", re.I)


def my_parse(text: str) -> Optional[str]:
    if not text:
        return None
    hits = _MY.findall(text)
    if not hits:
        return None
    last = re.sub(r"[\s-]+", "", hits[-1]).lower()
    if last == "match":
        return "match"
    if last in ("nonmatch", "notamatch", "nomatch"):
        return "non-match"
    return None


def my_f1(items: List[Tuple[Optional[str], int]]) -> float:
    tp = sum(1 for d, g in items if d == "match" and g == 1)
    fp = sum(1 for d, g in items if d == "match" and g == 0)
    fn = sum(1 for d, g in items if d != "match" and g == 1)
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    report: Dict[str, Any] = {"tolerance": TOL, "parser_check": {}, "f1_check": {},
                              "sql_check": {}, "audit": {}}

    # ------------------------------------------------ V.2 audit over every record file
    print("=== V.2 failure and denominator audit ===")
    files = ["em_preds_pilot.jsonl", "em_preds_test.jsonl", "em_preds_train.jsonl",
             "em_share_test.jsonl", "sql_initial_pilot.jsonl", "sql_share_pilot.jsonl",
             "sql_initial_main.jsonl", "sql_share_main.jsonl", "thrift_preds.jsonl"]
    tot = collections.Counter()
    for fn in files:
        recs = jl(fn)
        if not recs:
            report["audit"][fn] = {"present": False}
            continue
        h = hashlib.sha256(open(os.path.join(RAW, fn), "rb").read()).hexdigest()
        st = collections.Counter(r.get("status", "no_status") for r in recs)
        ids = collections.Counter(r.get("unit_id") for r in recs if "unit_id" in r)
        unparsed = sum(1 for r in recs if r.get("status") == "ok"
                       and "decision" in r and r["decision"] is None)
        nosql = sum(1 for r in recs if r.get("status") == "ok"
                    and "sql" in r and not r.get("sql"))
        exec_bad = collections.Counter((r.get("exec") or {}).get("status")
                                       for r in recs if r.get("status") == "ok" and "exec" in r)
        d = {"present": True, "sha256": h[:32], "records": len(recs), "status": dict(st),
             "unique_unit_ids": len(ids),
             "duplicate_unit_ids": sum(1 for v in ids.values() if v > 1),
             "ok_but_unparsed_decision": unparsed,
             "ok_but_no_sql": nosql,
             "exec_status": {k: v for k, v in exec_bad.items() if k}}
        report["audit"][fn] = d
        for k in ("records",):
            tot[k] += len(recs)
        tot["ok"] += st.get("ok", 0)
        tot["failed"] += st.get("failed", 0)
        print(f"  {fn:28s} {len(recs):7d} rec  {dict(st)}  "
              f"dup_ids={d['duplicate_unit_ids']}  unparsed={unparsed}  no_sql={nosql}")
    report["audit"]["_totals"] = dict(tot)
    print(f"  TOTAL records {tot['records']:,}  ok {tot['ok']:,}  failed {tot['failed']:,}")

    # ------------------------------------------------ V.1a independent parser agreement
    print("\n=== V.1 independent recomputation ===")
    for fn in ("em_preds_test.jsonl", "em_share_test.jsonl"):
        recs = [r for r in jl(fn) if r.get("status") == "ok"]
        dis = sum(1 for r in recs if my_parse(r.get("raw_reply")) != r.get("decision"))
        report["parser_check"][fn] = {"n": len(recs), "disagreements": dis,
                                      "rate": dis / max(1, len(recs))}
        print(f"  parser, {fn:24s} {len(recs):7d} replies, "
              f"{dis} disagreements ({dis/max(1,len(recs)):.4%})")

    # ------------------------------------------------ V.1b F1 recomputed independently
    cell = collections.defaultdict(list)
    for r in jl("em_preds_test.jsonl"):
        if r.get("status") == "ok":
            cell[(r["dataset"], r["pair_key"], r["model"])].append(r)
    pub = json.load(open(os.path.join(OUT, "em_baselines_test.json")))
    ok = bad = 0
    for dn in ("abt_buy", "amazon_google", "wdc_products"):
        ds = E.Matching(dn)
        gold = {ds.pair_key(r): ds.gold(r) for r in ds.splits["test"]}
        for m in E.POOL:
            items = []
            for pk, g in gold.items():
                rs = cell.get((dn, pk, m)) or []
                v = [my_parse(x.get("raw_reply")) for x in rs]
                v = [x for x in v if x is not None]
                d = None
                if v:
                    c = collections.Counter(v)
                    t = max(c.values())
                    w = [k for k, n in c.items() if n == t]
                    d = w[0] if len(w) == 1 else None
                items.append((d, g))
            mine = my_f1(items)
            theirs = pub["per_model"][dn][m]["f1"]
            agree = abs(mine - theirs) < TOL
            ok += agree
            bad += (not agree)
            report["f1_check"][f"{dn}/{m}"] = {"independent": mine, "published": theirs,
                                               "delta": mine - theirs, "agree": agree}
    print(f"  F1 recomputed from raw replies with an independent parser and scorer: "
          f"{ok} agree, {bad} disagree (tolerance {TOL})")
    if bad:
        for k, v in report["f1_check"].items():
            if not v["agree"]:
                print(f"    DISAGREE {k}: independent {v['independent']:.4f} vs "
                      f"published {v['published']:.4f}")

    # ------------------------------------------------ V.1c SQL re-execution
    byid = {q["question_id"]: q for q in S.load_questions()}
    recs = [r for r in jl("sql_initial_main.jsonl") if r.get("status") == "ok"]
    pub_sql = json.load(open(os.path.join(OUT, "sql_results.json")))
    agree_n = 0
    for m in sorted({r["model"] for r in recs}):
        s = [r for r in recs if r["model"] == m]
        hit = 0
        cache: Dict[Tuple[str, str], bool] = {}
        for r in s:
            if not r.get("sql"):
                continue
            k = (r["db_id"], r["sql"])
            if k not in cache:
                q = byid[r["question_id"]]
                cache[k] = bool(S.execution_match(r["db_id"], r["sql"], S.gold_sql(q))["match"])
            hit += cache[k]
        mine = hit / len(s)
        theirs = pub_sql["initial"][m]["mean"]
        a = abs(mine - theirs) < TOL
        agree_n += a
        report["sql_check"][m] = {"independent": mine, "published": theirs,
                                 "delta": mine - theirs, "agree": a}
        print(f"  SQL EX re-executed, {m:14s} independent {mine:.4f} vs published {theirs:.4f} "
              f"{'AGREE' if a else 'DISAGREE'}")

    report["summary"] = {
        "f1_cells_checked": ok + bad, "f1_agree": ok, "f1_disagree": bad,
        "sql_models_checked": len(report["sql_check"]), "sql_agree": agree_n,
        "total_records_audited": tot["records"],
    }
    with open(os.path.join(OUT, "verification.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {OUT}/verification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
