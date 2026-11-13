#!/usr/bin/env python3
"""Validate the candidate labels, then decompose the 8 published differences exactly.

Called by `sql_decompose.py --decompose`. Zero model calls; no database access (the labels are
already cached by stages 1 and 2).
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import sql_decompose as D

OUT = D.OUT
POOLS = {"pool4": None, "pool6": None}      # filled from the collector's own model tuples


def pools_for(bench: str) -> Dict[str, Tuple[str, ...]]:
    D.adapters(bench)
    import collect_sql_arms as CS
    return {"pool4": tuple(CS.PEER_MODELS),
            "pool6": tuple(CS.PEER_MODELS) + tuple(CS.EXTRA_GENERATORS)}


def load_arms(bench: str):
    rows = collections.defaultdict(list)
    stored = collections.defaultdict(dict)
    for ln in (D.RECORDS / ("sql_" + bench) / "arms.jsonl").read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        rows[(r["question_id"], r["recipient"], r["arm"])].append(r)
        s = (r.get("sql") or "").strip()
        if s and r.get("exec_match") is not None:
            stored[r["question_id"]].setdefault(s, set()).add(bool(r["exec_match"]))
    return rows, stored


def run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"benchmarks": {}}
    per_exec_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []

    for bench in D.BENCH:
        pub = json.loads((D.SRC / "output" / f"sql_final_selection_{bench}.json")
                         .read_text(encoding="utf-8"))
        pools = pools_for(bench)
        import collect_sql_arms as CS
        qs = D.questions(bench)
        gold = D.read_gold(bench)
        cand = D.read_cand(bench)
        arms, stored = load_arms(bench)
        peers = collections.defaultdict(dict)
        for ln in (D.RECORDS / ("sql_" + bench) / "peers.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                peers[r["question_id"]][r["model"]] = r
        assigned = sorted(peers)

        # ---- validation A: my labels against the labels the collection already stored
        agree = disagree = 0
        mism = []
        for (qid, model), c in cand.items():
            s = (peers[qid][model].get("sql") or "").strip()
            v = stored.get(qid, {}).get(s)
            if not s or v is None or len(v) > 1 or c["match"] is None:
                continue
            if bool(c["match"]) == next(iter(v)):
                agree += 1
            else:
                disagree += 1
                mism.append({"benchmark": bench, "question_id": qid, "model": model,
                             "db_id": c["db_id"], "mine": bool(c["match"]),
                             "stored": next(iter(v)), "reason": c.get("reason"),
                             "source": c.get("source"), "sql": s[:300]})
        print("%s validation A: candidate labels vs stored exec_match -> %d agree, %d disagree"
              % (bench, agree, disagree))
        for m in mism[:6]:
            print("   MISMATCH q=%s model=%s mine=%s stored=%s reason=%s source=%s"
                  % (m["question_id"], m["model"], m["mine"], m["stored"], m["reason"], m["source"]))
        val_rows.extend(mism)

        bench_out: Dict[str, Any] = {"validation_candidate_labels":
                                     {"agree": agree, "disagree": disagree}, "pools": {}}

        for pool, models in pools.items():
            pubp = pub["pools"][pool]
            arm_name = pubp["matched_arm"]
            # ---- the selector's pick, by the collector's own function object
            picks: Dict[int, Optional[str]] = {}
            pick_model: Dict[int, Optional[str]] = {}
            for qid in assigned:
                cs = [peers[qid][m] for m in models if m in peers[qid]]
                p = CS.majority_pick(cs)
                picks[qid] = ((p or {}).get("sql") or "").strip() or None
                pick_model[qid] = (p or {}).get("model")
            # ---- the pick's correctness, from the candidate labels
            s_q: Dict[int, Optional[bool]] = {}
            for qid in assigned:
                m = pick_model.get(qid)
                c = cand.get((qid, m)) if m else None
                s_q[qid] = None if c is None else c["match"]
            n_corr = sum(1 for v in s_q.values() if v)
            print("   %s/%s selection correct: mine %d vs published %s  (%s)"
                  % (bench, pool, n_corr, pubp["n_correct"],
                     "MATCH" if n_corr == int(pubp["n_correct"]) else "**DIFFERS**"))

            # ---- coverage: did the pool hold any correct candidate?
            cov: Dict[int, Optional[bool]] = {}
            for qid in assigned:
                vals = [cand.get((qid, m), {}).get("match") for m in models if m in peers[qid]]
                if any(v is True for v in vals):
                    cov[qid] = True
                elif any(v is None for v in vals):
                    cov[qid] = None          # gold unscorable -> cannot decide
                else:
                    cov[qid] = False
            pool_out = {"matched_arm": arm_name, "n_candidates": len(models),
                        "selection_n_correct_mine": n_corr,
                        "selection_n_correct_published": int(pubp["n_correct"]),
                        "coverage_true": sum(1 for v in cov.values() if v is True),
                        "coverage_false": sum(1 for v in cov.values() if v is False),
                        "coverage_undetermined": sum(1 for v in cov.values() if v is None),
                        "recipients": {}}

            for recip in CS.RECIPIENTS:
                pubr = pubp["paired"][recip]
                M = 0
                R = G = both_ok = both_bad = 0
                Gc = collections.Counter()
                Rc = collections.Counter()
                per_db: List[Tuple[str, float]] = []
                n_expected = int(pubr["n_arm_rows_expected"])
                # The published condition accuracy divides by the EXPECTED cell count and scores a
                # missing cell as a failure (`missing_arm_cells_scored_as_failure`).
                reps_expected = max(1, round(n_expected / max(len(assigned), 1)))
                n_missing_counted = 0
                for qid in assigned:
                    rr = list(arms.get((qid, recip, arm_name)) or [])
                    while len(rr) < reps_expected:
                        rr.append({"exec_match": False, "repeat": None,
                                   "eval_reason": "missing_cell_scored_as_failure"})
                        n_missing_counted += 1
                    s = s_q.get(qid)
                    sv = 1 if s else 0
                    for row in rr:
                        m = 1 if row.get("exec_match") else 0
                        M += 1
                        if sv and not m:
                            R += 1
                            Rc[cov.get(qid)] += 1
                        elif m and not sv:
                            G += 1
                            Gc[cov.get(qid)] += 1
                        elif sv and m:
                            both_ok += 1
                        else:
                            both_bad += 1
                        per_db.append((qs[qid]["db_id"], float(sv - m)))
                        per_exec_rows.append({
                            "benchmark": bench, "pool": pool, "recipient": recip,
                            "arm": arm_name, "question_id": qid, "db_id": qs[qid]["db_id"],
                            "repeat": row.get("repeat"), "selection_correct": sv,
                            "generation_correct": m,
                            "pool_had_correct_candidate": cov.get(qid),
                            "pick_model": pick_model.get(qid),
                            "pick_reason": (cand.get((qid, pick_model.get(qid))) or {}).get("reason"),
                            "arm_eval_reason": row.get("eval_reason"),
                        })
                D_mine_used = (R - G) / M if M else float("nan")
                # published denominator: recomputed mean uses the rows actually present
                pubd = float(pubr["selection_minus_arm"])
                lo, hi = D.cluster_ci(per_db)
                rec_out = {
                    "arm": arm_name,
                    "arm_executions_in_frame": M, "arm_executions_expected": n_expected,
                    "missing_cells_counted_as_generation_failure": n_missing_counted,
                    "recoveries_R": R, "regressions_G": G,
                    "both_correct": both_ok, "both_wrong": both_bad,
                    "regressions_pool_had_correct": Gc.get(True, 0),
                    "regressions_pool_had_none": Gc.get(False, 0),
                    "regressions_undetermined": Gc.get(None, 0),
                    "recoveries_pool_had_correct": Rc.get(True, 0),
                    "recoveries_other": Rc.get(False, 0) + Rc.get(None, 0),
                    "D_from_decomposition": D_mine_used,
                    "D_published": pubd,
                    "D_abs_difference": abs(D_mine_used - pubd),
                    "adds_back_exactly": abs(D_mine_used - pubd) < 5e-6,
                    "cluster_ci_mine_lo": lo, "cluster_ci_mine_hi": hi,
                    "cluster_ci_published_lo": float(pubr["cluster_bootstrap_ci_lo"]),
                    "cluster_ci_published_hi": float(pubr["cluster_bootstrap_ci_hi"]),
                }
                pool_out["recipients"][recip] = rec_out
                print("      %-11s M=%d R=%d G=%d | D=(R-G)/M=%+.6f vs published %+.6f  %s"
                      % (recip, M, R, G, D_mine_used, pubd,
                         "EXACT" if rec_out["adds_back_exactly"] else "**OFF**"))
                print("                  G split: pool had a correct candidate %d | pool had none %d"
                      " | undetermined %d" % (rec_out["regressions_pool_had_correct"],
                                              rec_out["regressions_pool_had_none"],
                                              rec_out["regressions_undetermined"]))
                print("                  my cluster CI [%+.4f, %+.4f] vs published [%+.4f, %+.4f]"
                      % (lo, hi, rec_out["cluster_ci_published_lo"],
                         rec_out["cluster_ci_published_hi"]))
            bench_out["pools"][pool] = pool_out
        report["benchmarks"][bench] = bench_out

    with open(OUT / "per_execution.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_exec_rows[0]))
        w.writeheader()
        w.writerows(per_exec_rows)
    fields = ["benchmark", "question_id", "model", "db_id", "mine", "stored", "reason", "source",
              "sql"]
    with open(OUT / "label_mismatches.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(val_rows)
    report["evaluator_semantics"] = {
        "comparison": "set of coerced tuples (sql_common._norm_result): row order ignored, "
                      "duplicate rows collapsed, floats rounded to 6 places, blobs as '<n bytes>'",
        "cap_used_here": D.EVAL_CAP,
        "cap_the_collector_stored_candidates_at": 20,
        "reuse_rule": "stored candidate result reused only when status ok and more_rows_exist is "
                      "false, i.e. the stored rows are the complete set; otherwise re-executed at "
                      "the evaluator's cap",
        "failure_handling": "a candidate that errors or times out is match=False, as the published "
                            "evaluator does; the reason is kept per row so failure stays separable "
                            "from a wrong result",
        "sqlite_version": __import__("sqlite3").sqlite_version,
        "access": "mode=ro, PRAGMA query_only=ON, 5 s per-statement deadline",
    }
    (OUT / "decomposition.json").write_text(json.dumps(report, indent=1, default=str) + "\n",
                                            encoding="utf-8")
    print("\n-> %s" % (OUT / "decomposition.json"))
    print("   per_execution.csv rows: %s" % f"{len(per_exec_rows):,}")
    return 0
