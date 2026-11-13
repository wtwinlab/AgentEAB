r"""P2a / P2b / P4 analysis for the SQL conditions, on BIRD and on Spider, by one code path.

    PYTHONPATH=code python3 code/analyse_sql_arms.py --benchmark bird
    PYTHONPATH=code python3 code/analyse_sql_arms.py --benchmark spider
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")

RECIPIENTS = ("qwen2.5-7b", "qwen2.5-32b")
ARMS = ("S0", "S1", "S2", "S3", "S4", "F1", "F2")
FAMILIES = {
    "P2a": (("S2", "S1"), ("S3", "S1"), ("S2", "S3"), ("S4", "S3")),
    "P2b": (("F1", "S1"), ("F2", "S1"), ("S2", "S1"), ("F2", "F1"), ("S2", "F2")),
}
N_REP = 2000
SEED = 20261015
ALPHA = 0.05


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps: Dict[str, float], alpha: float = ALPHA) -> Dict[str, Any]:
    order = sorted(ps, key=ps.get)
    m, out, prev = len(order), {}, 0.0
    for i, k in enumerate(order):
        adj = max(prev, min(1.0, (m - i) * ps[k]))
        prev = adj
        out[k] = {"p_raw": ps[k], "p_holm": adj, "survives": adj < alpha}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["bird", "spider"])
    a = ap.parse_args()
    data = os.path.join(STAGE, "data", a.benchmark)
    arms_rows = [json.loads(l) for l in open(os.path.join(data, "arms.jsonl"), encoding="utf-8")]
    peer_rows = [json.loads(l) for l in open(os.path.join(data, "peers.jsonl"), encoding="utf-8")]
    print(f"  {a.benchmark}: {len(arms_rows)} arm rows, {len(peer_rows)} peer rows")

    # ---------------------------------------------------------------- strata from the source queries
    peers_by_q: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in peer_rows:
        if r.get("role") == "peer":
            peers_by_q[r["question_id"]].append(r)
    stratum: Dict[int, str] = {}
    for qid, ps in peers_by_q.items():
        sts = {p.get("exec_status") for p in ps}
        stratum[qid] = ("some_peer_timeout" if "timeout" in sts
                        else "some_peer_error" if "error" in sts else "all_peers_ok")
    db_of = {r["question_id"]: r["db_id"] for r in arms_rows}

    rep: Dict[str, Any] = {
        "benchmark": a.benchmark,
        "estimand": "per-execution mean accuracy over the assigned questions: each question's three "
                    "executions are averaged first, then averaged over questions. Timeouts, "
                    "execution errors, unparsed replies and unsendable cells contribute 0.",
        "secondary_statistic": "majority_of_repeats_success — a repeat-stability diagnostic, NOT "
                               "per-execution accuracy and NOT a deployable answer-selection "
                               "policy (a policy would pick one query by majority SQL text or "
                               "result signature; this consults the evaluator per run)",
        "statistical_unit": "database-level cluster bootstrap; questions share databases, so "
                            "questions are not independent draws",
        "families_declared_before_inspection": {k: [f"{x}-{y}" for x, y in v]
                                                for k, v in FAMILIES.items()},
        "mcnemar_scope": "paired per-question correctness change only",
        "strata": dict(collections.Counter(stratum.values())),
        "by_recipient": {}}

    # ---------------------------------------------------------------- per condition, modal over repeats
    pvals: Dict[str, float] = {}
    for rc in RECIPIENTS:
        cell: Dict[Tuple[str, int], List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in arms_rows:
            if r["recipient"] == rc:
                cell[(r["arm"], r["question_id"])].append(r)
        # The assigned population is every question that met the assignment condition — at least two
        # peer queries that parsed.
        assigned = sorted({q for q, ps in peers_by_q.items()
                           if sum(1 for p in ps if p.get("sql")) >= 2})
        present = {arm: {q for (arm2, q) in cell if arm2 == arm} for arm in ARMS}
        unsendable = {arm: sorted(set(assigned) - present.get(arm, set())) for arm in ARMS}
        common = assigned
        entry: Dict[str, Any] = {
            "n_questions_assigned": len(assigned),
            "arms_with_unsendable_questions": {a_: v for a_, v in unsendable.items() if v},
            "unsendable_reason": "prompt exceeds the served models' 8,192-token context window; "
                                 "measured, server-reported, not inferred",
            "denominator_rule": "assigned population: an unsendable or failed cell counts as a "
                                "failure for that arm and is never removed from any arm",
            "arms": {}}

        def exec_rate(arm: str, q: int) -> float:
            """**Primary metric**: per-execution accuracy, the mean of this cell's executions.
            """
            rs = cell.get((arm, q))
            if not rs:
                return 0.0
            v = [bool(r.get("exec_match")) for r in rs]
            return sum(v) / len(v)

        def correct(arm: str, q: int) -> bool:
            """**Diagnostic only**: did a majority of this cell's repeats succeed?
            """
            rs = cell.get((arm, q))
            if not rs:
                return False
            v = [bool(r.get("exec_match")) for r in rs]
            return sum(v) * 2 > len(v)

        for arm in ARMS:
            rs = [r for (a_, _), v in cell.items() if a_ == arm for r in v]
            if not rs:
                continue
            rates = [exec_rate(arm, q) for q in common]
            acc = sum(rates) / max(len(rates), 1)
            maj = [correct(arm, q) for q in common]
            acc_majority = sum(1 for x in maj if x) / max(len(maj), 1)
            reasons = collections.Counter(r.get("eval_reason") for r in rs)
            estat = collections.Counter(r.get("exec_status") for r in rs)
            entry["arms"][arm] = {
                "per_execution_mean_accuracy": acc,
                "majority_of_repeats_success": acc_majority,
                "metric_note": "per_execution_mean_accuracy is primary and matches the "
                               "manuscript's metric; majority_of_repeats_success is a "
                               "repeat-stability diagnostic and is not a deployable policy",
                "accuracy_assigned": acc,
                "n_questions_assigned": len(common),
                "n_unsendable": len(unsendable.get(arm, [])),
                "executable_rate": sum(1 for r in rs if r.get("exec_status") == "ok") / len(rs),
                "exec_status_counts": dict(estat),
                "eval_reason_counts": dict(reasons),
                "calls": len(rs),
                "prompt_tokens_per_call": sum(r["prompt_tokens"] for r in rs) / len(rs),
                "completion_tokens_per_call": sum(r["completion_tokens"] for r in rs) / len(rs),
                "truncated": sum(1 for r in rs if r.get("truncated")),
                "mean_exec_seconds": st.mean(
                    [r["exec"].get("elapsed_s", 0.0) for r in rs if isinstance(r.get("exec"), dict)]
                    or [0.0]),
                "per_execution_accuracy_by_stratum": {
                    s: (lambda xs: sum(xs) / max(len(xs), 1))(
                        [exec_rate(arm, q) for q in common if stratum.get(q) == s])
                    for s in ("all_peers_ok", "some_peer_error", "some_peer_timeout")},
                "n_by_stratum": {s: sum(1 for q in common if stratum.get(q) == s)
                                 for s in ("all_peers_ok", "some_peer_error",
                                           "some_peer_timeout")}}

        # ------------------------------------------------- paired contrasts, database bootstrap
        dbs = sorted({db_of[q] for q in common if q in db_of})
        entry["n_questions_without_db"] = sum(1 for q in common if q not in db_of)
        by_db: Dict[str, List[int]] = collections.defaultdict(list)
        for q in common:
            by_db[db_of.get(q, "?")].append(q)
        rng = random.Random(SEED)
        draws = [[rng.choice(dbs) for _ in dbs] for _ in range(N_REP)]
        entry["n_databases"] = len(dbs)

        for fam, pairs in FAMILIES.items():
            for x, y in pairs:
                if x not in entry["arms"] or y not in entry["arms"]:
                    continue
                rx = {q: exec_rate(x, q) for q in common}
                ry = {q: exec_rate(y, q) for q in common}
                cx = {q: correct(x, q) for q in common}
                cy = {q: correct(y, q) for q in common}
                corr = sum(1 for q in common if cx[q] and not cy[q])
                harm = sum(1 for q in common if cy[q] and not cx[q])
                p = mcnemar_exact(corr, harm)
                key = f"{rc}|{fam}|{x}-{y}"
                pvals[key] = p
                diffs = []
                for sample in draws:
                    sx = sy = n = 0.0
                    for d in sample:
                        for q in by_db[d]:
                            n += 1
                            sx += rx[q]
                            sy += ry[q]
                    diffs.append((sx - sy) / max(n, 1))
                diffs.sort()
                entry.setdefault("contrasts", {})[f"{fam}|{x}-{y}"] = {
                    "metric": "per-execution mean accuracy",
                    "accuracy_delta": entry["arms"][x]["per_execution_mean_accuracy"]
                    - entry["arms"][y]["per_execution_mean_accuracy"],
                    "majority_diagnostic_delta":
                        entry["arms"][x]["majority_of_repeats_success"]
                        - entry["arms"][y]["majority_of_repeats_success"],
                    "ci_lo": diffs[int(ALPHA / 2 * N_REP)],
                    "ci_hi": diffs[int((1 - ALPHA / 2) * N_REP) - 1],
                    "covers_zero": diffs[int(ALPHA / 2 * N_REP)] <= 0
                    <= diffs[int((1 - ALPHA / 2) * N_REP) - 1],
                    "corrections_majority_diagnostic": corr,
                    "harms_majority_diagnostic": harm,
                    "mcnemar_p_on_majority_diagnostic_only": p,
                    "prompt_tokens_delta_per_call":
                        entry["arms"][x]["prompt_tokens_per_call"]
                        - entry["arms"][y]["prompt_tokens_per_call"],
                    "calls_delta": entry["arms"][x]["calls"] - entry["arms"][y]["calls"]}
        # ---- P2b.4: the paired contrast WITHIN each prespecified stratum.
        # Storing `source_statuses` and printing per-arm accuracy by stratum is not the stratified
        # comparison the plan asks for.
        strat_out: Dict[str, Any] = {}
        for sname in ("all_peers_ok", "some_peer_error", "some_peer_timeout"):
            qs_s = [q for q in common if stratum.get(q) == sname]
            # A bootstrap over a handful of questions in one or two databases returns a
            # zero-width interval that looks certain and means nothing: Spider's timeout stratium
            # has n=2 and produced [+0.5000, +0.5000].
            MIN_STRATUM = 20
            if len(qs_s) < MIN_STRATUM:
                per_arm_acc = {}
                for arm_ in ARMS:
                    if arm_ in entry["arms"] and qs_s:
                        per_arm_acc[arm_] = sum(exec_rate(arm_, q) for q in qs_s) / len(qs_s)
                strat_out[sname] = {
                    "n_questions": len(qs_s),
                    "per_arm_accuracy": per_arm_acc,
                    "no_interval_reason": f"fewer than {MIN_STRATUM} questions; a database-cluster "
                                          f"bootstrap over this many units returns a degenerate "
                                          f"interval, so none is reported"}
                continue
            dbs_s = sorted({db_of[q] for q in qs_s if q in db_of})
            by_db_s: Dict[str, List[int]] = collections.defaultdict(list)
            for q in qs_s:
                by_db_s[db_of.get(q, "?")].append(q)
            rng_s = random.Random(SEED)
            draws_s = [[rng_s.choice(dbs_s) for _ in dbs_s] for _ in range(N_REP)]
            inner: Dict[str, Any] = {"n_questions": len(qs_s), "n_databases": len(dbs_s)}
            for fam, pairs in FAMILIES.items():
                for x, y in pairs:
                    if x not in entry["arms"] or y not in entry["arms"]:
                        continue
                    rx = {q: exec_rate(x, q) for q in qs_s}
                    ry = {q: exec_rate(y, q) for q in qs_s}
                    cx = {q: correct(x, q) for q in qs_s}
                    cy = {q: correct(y, q) for q in qs_s}
                    corr = sum(1 for q in qs_s if cx[q] and not cy[q])
                    harm = sum(1 for q in qs_s if cy[q] and not cx[q])
                    acc_x = sum(rx.values()) / len(qs_s)
                    acc_y = sum(ry.values()) / len(qs_s)
                    dd = []
                    for sample in draws_s:
                        sx = sy = n = 0.0
                        for d_ in sample:
                            for q in by_db_s[d_]:
                                n += 1
                                sx += rx[q]
                                sy += ry[q]
                        dd.append((sx - sy) / max(n, 1))
                    dd.sort()
                    inner[f"{fam}|{x}-{y}"] = {
                        "accuracy_delta": acc_x - acc_y,
                        "ci_lo": dd[int(ALPHA / 2 * N_REP)],
                        "ci_hi": dd[int((1 - ALPHA / 2) * N_REP) - 1],
                        "covers_zero": dd[int(ALPHA / 2 * N_REP)] <= 0
                        <= dd[int((1 - ALPHA / 2) * N_REP) - 1],
                        "metric": "per-execution mean accuracy",
                        "corrections_majority_diagnostic": corr,
                        "harms_majority_diagnostic": harm,
                        "mcnemar_p_on_majority_diagnostic_only": mcnemar_exact(corr, harm)}
            strat_out[sname] = inner
        entry["stratified_contrasts"] = strat_out

        rep["by_recipient"][rc] = entry

    rep["holm"] = holm(pvals)

    # ---------------------------------------------------------------- source-side resources
    rep["source_pool_resources"] = {
        "calls": len(peer_rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in peer_rows),
        "completion_tokens": sum(r["completion_tokens"] for r in peer_rows),
        "note": "candidate generation, shared by every arm that uses peers; attributed to none of "
                "them individually"}

    print(f"\n=== {a.benchmark}: assigned-question execution accuracy ===")
    print(f"{'recipient':<13}{'arm':<5}{'acc':>8}{'exec ok':>9}{'ptok':>8}{'ctok':>7}"
          f"{'trunc':>7}{'exec s':>8}{'maj':>8}")
    for rc, e in rep["by_recipient"].items():
        for arm in ARMS:
            v = e["arms"].get(arm)
            if not v:
                continue
            print(f"{rc:<13}{arm:<5}{v['per_execution_mean_accuracy']:>8.4f}"
                  f"{v['executable_rate']:>9.4f}"
                  f"{v['prompt_tokens_per_call']:>8.0f}{v['completion_tokens_per_call']:>7.1f}"
                  f"{v['truncated']:>7}{v['mean_exec_seconds']:>8.3f}"
                  f"{v['majority_of_repeats_success']:>8.4f}")
    print(f"\n=== contrasts (database cluster bootstrap, {rep['by_recipient'][RECIPIENTS[0]]['n_databases']} units) ===")
    for rc, e in rep["by_recipient"].items():
        for k, c in e.get("contrasts", {}).items():
            h = rep["holm"].get(f"{rc}|{k.replace('|', '|')}", {})
            print(f"  {rc:<13}{k:<16} acc {c['accuracy_delta']:+.4f} "
                  f"[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}]"
                  f"{'  covers 0' if c['covers_zero'] else '          '}  "
                  f"+{c['corrections_majority_diagnostic']}/-{c['harms_majority_diagnostic']}  "
                  f"Δptok {c['prompt_tokens_delta_per_call']:+.0f}")
    print(f"\n  strata: {rep['strata']}")
    print("\n=== P2b.4 paired contrasts WITHIN each prespecified source-status stratum ===")
    for rc, e in rep["by_recipient"].items():
        for sname, blk in e.get("stratified_contrasts", {}).items():
            if blk.get("no_interval_reason"):
                print(f"  {rc} {sname}: n={blk['n_questions']} — no interval "
                      f"({blk['no_interval_reason'][:48]}...); per-arm accuracy "
                      f"{ {k: round(v, 3) for k, v in blk['per_arm_accuracy'].items()} }")
                continue
            keys = [k for k in blk if k.startswith("P2b|")]
            print(f"  {rc} {sname}: n={blk['n_questions']} over {blk['n_databases']} databases")
            for k in keys:
                v = blk[k]
                print(f"      {k:<16}{v['accuracy_delta']:+.4f} "
                      f"[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}]"
                      f"{'  covers 0' if v['covers_zero'] else '          '}  "
                      f"+{v['corrections_majority_diagnostic']}"
                      f"/-{v['harms_majority_diagnostic']}")

    p = os.path.join(STAGE, "output", f"sql_analysis_{a.benchmark}.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
