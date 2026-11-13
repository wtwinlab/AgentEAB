r"""Replay a fixed final-answer-selection rule over the existing SQL candidate pool, offline.

    PYTHONPATH=code python3 code/sql_final_selection_replay.py --bench bird
    PYTHONPATH=code python3 code/sql_final_selection_replay.py --bench spider
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(STAGE, "code"))

import collect_sql_arms as CS                                       # noqa: E402

PEER_MODELS = CS.PEER_MODELS                                        # 4 peers
EXTRA = CS.EXTRA_GENERATORS                                         # 2 extra generators
RECIPIENTS = CS.RECIPIENTS
REPEATS = CS.REPEATS
BOOT = 2000
SEED = 20261015

POOLS = {"pool4": list(PEER_MODELS), "pool6": list(PEER_MODELS) + list(EXTRA)}
MATCHED_ARM = {"pool4": "S2", "pool6": "S3"}        # identical candidate set, as declared


def adapter(bench: str):
    if bench == "bench_unset":
        raise SystemExit("--bench is required")
    if bench == "spider":
        import spider_common as A                                   # noqa: PLC0415
        return A, os.path.join(STAGE, "data/spider")
    sys.path.insert(0, os.path.join(os.path.dirname(STAGE), "dataeng_2026-10-13/code"))
    import sql_common as A                                          # noqa: PLC0415
    return A, os.path.join(STAGE, "data/bird")


def load_candidates(data_dir: str) -> Dict[int, Dict[str, Dict[str, Any]]]:
    by_q: Dict[int, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
    with open(os.path.join(data_dir, "peers.jsonl"), encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            by_q[r["question_id"]][r["model"]] = r
    return by_q


def load_arms(data_dir: str) -> Dict[Tuple[int, str, str], List[Dict[str, Any]]]:
    out: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    with open(os.path.join(data_dir, "arms.jsonl"), encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            out[(r["question_id"], r["recipient"], r["arm"])].append(r)
    return out


def cluster_bootstrap(per_q: List[Tuple[str, float]], n: int = BOOT) -> Tuple[float, float]:
    """Resample databases with replacement; the statistic is the mean paired difference."""
    by_db: Dict[str, List[float]] = collections.defaultdict(list)
    for db, d in per_q:
        by_db[db].append(d)
    dbs = sorted(by_db)
    rng = random.Random(SEED)
    draws = []
    for _ in range(n):
        vals: List[float] = []
        for _ in range(len(dbs)):
            vals.extend(by_db[dbs[rng.randrange(len(dbs))]])
        if vals:
            draws.append(sum(vals) / len(vals))
    draws.sort()
    if not draws:
        return (0.0, 0.0)
    return (draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["bird", "spider"])
    a = ap.parse_args()
    A, data_dir = adapter(a.bench)

    qs = {q["question_id"]: q for q in A.load_questions()}
    cands = load_candidates(data_dir)
    arms = load_arms(data_dir)

    out: Dict[str, Any] = {
        "benchmark": a.bench,
        "selection_rules": "fixed in advance",
        "rule": "majority_pick: largest result-signature group, ties by lowest-sorted SQL text",
        "rule_implementation": "collect_sql_arms.majority_pick (the same function object the "
                               "collector used; collect_sql_spider imports it rather than copying)",
        "selector_inputs": ["candidate SQL text", "each candidate's own exec record"],
        "selector_never_reads": ["gold SQL", "the evaluator", "any accuracy"],
        "evaluator": "sql_common.execution_match, cap 10000 rows, 5s timeout — the evaluator the "
                     "arms were scored with",
        "deterministic_single_value": "one selection per question per pool; NOT replicated into "
                                      "repeats, because the rule has no stochastic component",
        "pools": {}, "verification": {}, "resources": {}}

    # ---------------------------------------------------------------- assignment, as the collector did
    assigned: List[int] = []
    pool_rows: Dict[str, Dict[int, List[Dict[str, Any]]]] = {k: {} for k in POOLS}
    for qid, got in cands.items():
        if qid not in qs:
            continue
        peers = [got[m] for m in PEER_MODELS if m in got and got[m].get("sql")]
        extra = [got[m] for m in EXTRA if m in got and got[m].get("sql")]
        if len(peers) < 2:                      # the collector's assignment condition
            continue
        assigned.append(qid)
        pool_rows["pool4"][qid] = peers
        pool_rows["pool6"][qid] = peers + extra
    assigned.sort()
    out["n_questions_assigned"] = len(assigned)
    print(f"  {a.bench}: {len(assigned)} assigned questions")

    # ---------------------------------------------------------------- provenance of the stored pick
    recomputed_pick6 = {qid: CS.majority_pick(pool_rows["pool6"][qid]) for qid in assigned}
    agree = disagree = no_stored = 0
    examples: List[Dict[str, Any]] = []
    for (qid, rec, arm), rows in arms.items():
        if arm not in ("S3", "S4") or qid not in recomputed_pick6:
            continue
        mine = (recomputed_pick6[qid] or {}).get("sql")
        for r in rows:
            stored = r.get("frozen_pick_sql")
            if stored is None:
                no_stored += 1
            elif stored == mine:
                agree += 1
            else:
                disagree += 1
                if len(examples) < 3:
                    examples.append({"question_id": qid, "arm": arm, "recipient": rec,
                                     "stored": stored[:160], "recomputed": (mine or "")[:160]})
    out["verification"]["frozen_pick_provenance"] = {
        "rows_compared": agree + disagree, "agree": agree, "disagree": disagree,
        "rows_without_stored_pick": no_stored,
        "examples_of_disagreement": examples,
        "meaning": "the stored frozen_pick_sql on S3/S4 rows is the pool-6 pick of this same rule; "
                   "agreement is what licenses calling it provenance. pool-4 has NO stored "
                   "counterpart, so its selection is newly computed and unverifiable against a "
                   "stored field."}
    print(f"  frozen_pick provenance: {agree} agree / {disagree} disagree "
          f"({no_stored} rows carry no stored pick)")

    # ---------------------------------------------------------------- selection exposure
    trunc_q = {k: 0 for k in POOLS}
    for k in POOLS:
        for qid in assigned:
            if any((c.get("exec") or {}).get("more_rows_exist") for c in pool_rows[k][qid]):
                trunc_q[k] += 1
    out["verification"]["questions_with_a_truncated_candidate_result"] = {
        k: {"n": v, "share": v / len(assigned)} for k, v in trunc_q.items()}

    # ---------------------------------------------------------------- score each pool
    t0 = time.time()
    n_exec = 0
    reuse_checked = reuse_agree = reuse_disagree = 0
    for pool in POOLS:
        picks: Dict[int, Optional[str]] = {}
        for qid in assigned:
            p = CS.majority_pick(pool_rows[pool][qid])
            picks[qid] = (p or {}).get("sql")
        per_q: Dict[int, int] = {}
        reasons: collections.Counter = collections.Counter()
        no_answer = 0
        for qid in assigned:
            sql = picks[qid]
            if not sql:
                per_q[qid] = 0
                no_answer += 1
                reasons["no_candidate_with_sql"] += 1
                continue
            ev = A.execution_match(qs[qid]["db_id"], sql, qs[qid]["SQL"])
            n_exec += 2                                   # the selected query and gold
            per_q[qid] = int(bool(ev.get("match")))
            reasons[ev.get("reason") or "?"] += 1
            # reuse agreement: a stored scored row with the identical SQL must agree
            for rec in RECIPIENTS:
                for arm in ("S1", "S2", "S3", "S4", "F1", "F2"):
                    for r in arms.get((qid, rec, arm), ()):
                        if r.get("sql") and r["sql"].strip() == sql.strip():
                            reuse_checked += 1
                            if bool(r["exec_match"]) == bool(ev.get("match")):
                                reuse_agree += 1
                            else:
                                reuse_disagree += 1
        acc = sum(per_q.values()) / len(assigned)
        entry: Dict[str, Any] = {
            "members": POOLS[pool], "n_candidates": len(POOLS[pool]),
            "selection_accuracy": acc,
            "n_correct": sum(per_q.values()), "denominator": len(assigned),
            "questions_with_no_candidate_sql": no_answer,
            "evaluation_reasons": dict(reasons),
            "recipient_generation_calls": 0,
            "matched_arm": MATCHED_ARM[pool], "paired": {}}

        # ------------------------------------------------------------ paired against the matched condition
        sa = json.load(open(os.path.join(STAGE, "output", f"sql_analysis_{a.bench}.json"),
                            encoding="utf-8"))
        for rec in RECIPIENTS:
            arm = MATCHED_ARM[pool]
            diffs: List[Tuple[str, float]] = []
            arm_mean_num = 0.0
            n_rows_used = 0
            for qid in assigned:
                rows = arms.get((qid, rec, arm), [])
                # the assigned denominator: a missing (unsendable) cell is a failure, not a drop
                got = [int(bool(r["exec_match"])) for r in rows]
                am = (sum(got) / REPEATS) if REPEATS else 0.0
                arm_mean_num += am
                n_rows_used += len(got)
                diffs.append((qs[qid]["db_id"], per_q[qid] - am))
            arm_mean = arm_mean_num / len(assigned)
            lo, hi = cluster_bootstrap(diffs)
            delta = acc - arm_mean
            ref = sa["by_recipient"][rec]["arms"][arm]["per_execution_mean_accuracy"]

            # An asymmetry that favours selection, bounded rather than argued about.
            n_to = sum(1 for qid in assigned for r in arms.get((qid, rec, arm), [])
                       if r.get("eval_reason") == "timeout")
            bias = n_to / (len(assigned) * REPEATS)
            entry["paired"][rec] = {
                "arm": arm,
                "arm_per_execution_mean_accuracy_recomputed": arm_mean,
                "arm_per_execution_mean_accuracy_in_analysis_json": ref,
                "recomputation_matches_analysis": abs(arm_mean - ref) < 1e-9,
                "selection_minus_arm": delta,
                "cluster_bootstrap_ci_lo": lo, "cluster_bootstrap_ci_hi": hi,
                "covers_zero": lo <= 0 <= hi,
                "arm_evaluator_timeout_rows": n_to,
                "arm_mean_if_every_evaluator_timeout_were_a_match": arm_mean + bias,
                "selection_minus_arm_worst_case_for_selection": delta - bias,
                "sign_survives_worst_case": (delta > 0) == (delta - bias > 0),
                "worst_case_note": "the arms were evaluated during a 24-worker collection and a few "
                                   "rows were scored 0 on an evaluator timeout; this replay was "
                                   "evaluated unloaded. Crediting every such row to the arm is the "
                                   "largest correction the asymmetry can produce.",
                "n_arm_rows_used": n_rows_used,
                "n_arm_rows_expected": len(assigned) * REPEATS,
                "missing_arm_cells_scored_as_failure": len(assigned) * REPEATS - n_rows_used,
                "recipient_calls_saved": n_rows_used,
                "prompt_tokens_saved": sum(
                    r.get("prompt_tokens", 0) for qid in assigned
                    for r in arms.get((qid, rec, arm), []))}
            print(f"  {pool} vs {arm} / {rec}: selection {acc:.4f} - arm {arm_mean:.4f} = "
                  f"{delta:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"worst case {delta - bias:+.4f} ({n_to} evaluator timeouts)")
        out["pools"][pool] = entry

    out["verification"]["reuse_agreement_with_stored_exec_match"] = {
        "rows_compared": reuse_checked, "agree": reuse_agree, "disagree": reuse_disagree,
        "meaning": "where a selected SQL string is identical to a stored scored row's SQL for the "
                   "same question, the fresh evaluation must reproduce that row's exec_match; a "
                   "disagreement would indicate evaluator nondeterminism",
        "investigated_disagreements": [
            {"benchmark": "bird", "question_id": 390, "db_id": "card_games", "arm": "S4",
             "recipient": "qwen2.5-7b", "repeat": 0,
             "stored": {"exec_match": False, "eval_reason": "timeout", "own_exec_status": "ok"},
             "fresh": {"match": True, "reason": "compared", "n_pred_rows": 157, "n_gold_rows": 157},
             "measured": "the query completes in 0.08s unloaded, three times in a row, and gold "
                         "returns the same 157 rows; so the 5s evaluator timeout is not a property "
                         "of the query",
             "root_cause_status": "NOT fully established. The timeout is machine-state dependent "
                                  "and the collection ran 24 workers against the same sqlite files; "
                                  "contention is the plausible mechanism but was not instrumented "
                                  "at the time, so it is not asserted as the cause",
             "impact": "the arms pay this penalty and the replay does not, which favours selection; "
                       "bounded per cell by arm_mean_if_every_evaluator_timeout_were_a_match"}]}

    # ---------------------------------------------------------------- resources
    pool_calls = sum(1 for qid in cands for _ in cands[qid])
    pool_tokens = sum(r.get("prompt_tokens", 0) for qid in cands for r in cands[qid].values())
    out["resources"] = {
        "new_model_calls": 0,
        "new_generation": "none — every candidate was already on disk",
        "sql_executions_performed_here": n_exec,
        "wall_seconds": time.time() - t0,
        "candidate_pool_calls": pool_calls,
        "candidate_pool_prompt_tokens": pool_tokens,
        "pool_is_not_free": "the pool is a sunk cost shared with every arm compared against; it is "
                            "attributed to neither condition, and 'zero recipient calls' describes "
                            "only the additional recipient generation avoided",
        "selection_execution_cost": "the candidate executions were already stored; this replay adds "
                                    "one execution of the selected query and one of gold per "
                                    "question, which is evaluation cost, not selection cost"}
    print(f"  executions here: {n_exec} in {out['resources']['wall_seconds']:.1f}s; "
          f"pool was {pool_calls} calls / {pool_tokens:,} prompt tokens (shared, not free)")

    p = os.path.join(STAGE, "output", f"sql_final_selection_{a.bench}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
