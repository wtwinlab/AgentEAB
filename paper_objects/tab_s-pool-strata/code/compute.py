"""BIRD generation gains split by whether the candidate pool holds a correct query.

    python3 code/compute.py --out DIR     # -> DIR/values.json, DIR/strata.csv

Input (inputs/pool_strata_questions.csv): one row per BIRD question and generator-pool comparison,
with the question's database, whether some candidate returns the reference result, the correctness
of the Nonempty vote, of the generated query (correct runs out of all runs) and, where logged, of
the LLM selector given the same inputs.

Each database is weighted equally. A stratum's contribution averages each question's paired
difference times the stratum indicator, so the two contributions add up to the total. Intervals:
database-cluster bootstrap (10,000 draws, seed 20261105) and question bootstrap of the same
database-weighted mean (10,000 draws, seed 20261109); each estimate draws from a fresh generator.
An estimate is detected when both 95% intervals exclude 0.
"""
import argparse
import collections
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
DRAWS = 10000
COMPARISONS = (            # table order
    ("deepseek-v4-pro", "four_model"), ("qwen3-32b", "four_model"), ("qwen3-32b", "qwen3_glm"),
    ("qwen2.5-7b", "four_model"), ("qwen2.5-32b", "four_model"), ("qwen2.5-7b", "six_candidates"),
    ("qwen2.5-32b", "six_candidates"), ("gemma-4-26b", "four_model"), ("gpt-oss-120b", "four_model"))
SHORT = {"deepseek-v4-pro": "ds", "qwen3-32b": "q3", "qwen2.5-7b": "q25_7b", "qwen2.5-32b": "q25_32b",
         "gemma-4-26b": "gemma", "gpt-oss-120b": "gptoss"}
PSHORT = {"four_model": "pool4", "qwen3_glm": "glm", "six_candidates": "pool6"}


def cluster_boot(values, clusters, seed=20261105):
    """Database-cluster bootstrap of the database-equal mean."""
    by = collections.defaultdict(list)
    for u in sorted(values):
        by[clusters[u]].append(values[u])
    cl = sorted(by)
    per = np.array([np.mean(by[c]) for c in cl])
    draws = np.random.default_rng(seed).integers(0, len(cl), size=(DRAWS, len(cl)))
    bs = per[draws].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return 100 * float(per.mean()), 100 * float(lo), 100 * float(hi)


def question_boot(values, clusters, seed=20261109):
    """Question bootstrap of the same database-equal mean (each question weighted 1 / its database size)."""
    qs = sorted(values)
    v = np.array([values[q] for q in qs], float)
    n = collections.Counter(clusters[q] for q in qs)
    w = np.array([1.0 / n[clusters[q]] for q in qs])
    idx = np.random.default_rng(seed).integers(0, len(v), size=(DRAWS, len(v)))
    b = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return 100 * float(lo), 100 * float(hi)


def estimate(values, clusters):
    pt, dlo, dhi = cluster_boot(values, clusters)
    qlo, qhi = question_boot(values, clusters)
    det = int((dlo > 0 or dhi < 0) and (qlo > 0 or qhi < 0))
    return dict(est=pt, db_lo=dlo, db_hi=dhi, q_lo=qlo, q_hi=qhi, detected=det)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    data = collections.defaultdict(dict)
    for r in csv.DictReader(open(INP / "pool_strata_questions.csv", encoding="utf-8")):
        data[(r["recipient"], r["pool"])][int(r["question_id"])] = dict(
            db=r["db_id"], ref=int(r["has_reference"]), oracle=int(r["correct_candidate"]),
            ne=int(r["nonempty"]), gen=int(r["generation_correct"]) / int(r["generation_runs"]),
            runs=int(r["generation_runs"]), sel=int(r["selector"]) if r["selector"] != "" else None)
    V, table = {}, []
    for rec, pool in COMPARISONS:
        X = data[(rec, pool)]
        cl = {q: X[q]["db"] for q in X}
        k = f"{SHORT[rec]}_{PSHORT[pool]}"
        res = {"total": estimate({q: X[q]["gen"] - X[q]["ne"] for q in X}, cl)}
        for s, name in ((1, "present"), (0, "absent")):
            res[name] = estimate({q: (X[q]["gen"] - X[q]["ne"]) * (X[q]["oracle"] == s) for q in X}, cl)
        for part, r in res.items():
            for f, v in r.items():
                V[f"{k}_{part}" + ("" if f == "est" else f"_{f}")] = v
        V[f"{k}_questions"] = len(X)
        V[f"{k}_generation_runs"] = sorted({X[q]["runs"] for q in X})[0]
        V[f"{k}_absent_questions"] = sum(1 for q in X if X[q]["oracle"] == 0)
        table.append(dict(recipient=rec, pool=pool, questions=len(X),
                          **{f"{p}_{f}": round(v, 4) for p, r in res.items() for f, v in r.items()}))
    absent = [V[f"{SHORT[r]}_{PSHORT[p]}_absent"] for r, p in COMPARISONS]
    present = [V[f"{SHORT[r]}_{PSHORT[p]}_present"] for r, p in COMPARISONS]
    V["comparisons"] = len(COMPARISONS)
    V["absent_min"], V["absent_max"] = min(absent), max(absent)
    V["present_min"], V["present_max"] = min(present), max(present)
    V["absent_detected_count"] = sum(V[f"{SHORT[r]}_{PSHORT[p]}_absent_detected"] for r, p in COMPARISONS)
    V["present_detected_positive_count"] = sum(
        int(V[f"{SHORT[r]}_{PSHORT[p]}_present_detected"] and V[f"{SHORT[r]}_{PSHORT[p]}_present"] > 0)
        for r, p in COMPARISONS)
    # DeepSeek-V4-Pro generation against its own selector, among questions with a correct candidate
    X = data[("deepseek-v4-pro", "four_model")]
    cl = {q: X[q]["db"] for q in X}
    r = estimate({q: (X[q]["gen"] - X[q]["sel"]) * (X[q]["oracle"] == 1) for q in X}, cl)
    V["ds_pool4_present_minus_selector"] = r["est"]
    V["ds_pool4_present_minus_selector_detected"] = r["detected"]
    V["ds_pool4_present_any_interval_excludes_zero"] = int(
        V["ds_pool4_present_db_lo"] > 0 or V["ds_pool4_present_db_hi"] < 0
        or V["ds_pool4_present_q_lo"] > 0 or V["ds_pool4_present_q_hi"] < 0)
    # sensitivity: drop the two questions without a completed reference result
    V["ds_pool4_absent_without_reference"] = sum(1 for q in X if X[q]["oracle"] == 0 and not X[q]["ref"])
    keep = [q for q in X if X[q]["ref"]]
    clk = {q: cl[q] for q in keep}
    V["ds_pool4_with_reference_total"] = cluster_boot({q: X[q]["gen"] - X[q]["ne"] for q in keep}, clk)[0]
    V["ds_pool4_with_reference_absent"] = cluster_boot(
        {q: (X[q]["gen"] - X[q]["ne"]) * (X[q]["oracle"] == 0) for q in keep}, clk)[0]
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n")
    with open(out / "strata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    main()
