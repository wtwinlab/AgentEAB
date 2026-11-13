#!/usr/bin/env python3
"""Generation minus selection across recipient families (Fig. fig:sql-family, Table tab:s-family-sql).

    python3 code/compute.py --out DIR      # writes DIR/values.json and DIR/plotted_values.csv

Reads only ../inputs/.

* Cross-family contrasts: Qwen3-32B, gpt-oss-120b and Gemma-4-26B generation (one per question)
  minus the Unfiltered, Valid and Nonempty winners of the four-candidate pool, on BIRD (1,534
  questions, 11 databases) and Spider (1,034, 20). Each database weighted equally: the estimate is
  the mean of per-database mean differences. The 95% interval resamples databases, 10,000 draws of
  numpy `default_rng(20261028)` per contrast; the bootstrap p-value is twice the smaller tail share
  at zero, floored at 1/10,000. Holm at 0.05 over the 18 contrasts jointly, over the 15 contrasts
  other than Gemma on Spider, and over the three Gemma-on-Spider contrasts.
* Execution feedback: Qwen2.5-7B generation with the candidate queries and their execution status,
  errors and result details, minus generation with the candidate queries alone, on the 1,532 BIRD
  questions that have all four input conditions; each question weighted equally. The interval
  resamples databases (4,000 draws of Python `random.Random(20261013)`, databases in order of first
  appearance over sorted question ids), as in the source analysis.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
BOOT, SEED = 10000, 20261028
RECIPIENTS = ("qwen3-32b", "gpt-oss-120b", "gemma-4-26b")
RULES = (("released", "unfiltered"), ("valid_majority", "valid"), ("nonempty_majority", "nonempty"))
SHORT = {"qwen3-32b": "qwen3", "gpt-oss-120b": "gptoss", "gemma-4-26b": "gemma"}


def db_cluster(by_db):
    keys = sorted(by_db)
    per = np.array([float(np.mean(by_db[k])) for k in keys])
    draws = per[np.random.default_rng(SEED).integers(0, len(per), size=(BOOT, len(per)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return float(per.mean()), float(lo), float(hi), float(min(1.0, max(p, 1.0 / BOOT)))


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, stop = [False] * len(ps), False
    for rank, i in enumerate(order):
        if not stop and ps[i] <= alpha / (len(ps) - rank):
            rej[i] = True
        else:
            stop = True
    return rej


def family(v, rows_out):
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    gen = collections.defaultdict(dict)
    with open(INP / "crossfamily_generation_selection_per_question.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            data[(r["benchmark"], r["recipient"], r["selector"])][r["db_id"]].append(float(r["difference"]))
            gen[(r["benchmark"], r["recipient"])][(r["db_id"], r["question_id"])] = int(r["generation_correct"])
    cells = []
    for b in ("bird", "spider"):
        for rec in RECIPIENTS:
            for sel, lab in RULES:
                est, lo, hi, p = db_cluster(data[(b, rec, sel)])
                k = "%s_%s_gen_minus_%s" % (b, SHORT[rec], lab)
                v[k], v[k + "_lo"], v[k + "_hi"], v[k + "_p"] = 100 * est, 100 * lo, 100 * hi, p
                cells.append((k, b, rec, p))
                rows_out.append([b, rec, sel, round(100 * est, 3), round(100 * lo, 3), round(100 * hi, 3)])
            by = collections.defaultdict(list)
            for (db, _q), y in gen[(b, rec)].items():
                by[db].append(y)
            v["%s_%s_generation_accuracy" % (b, SHORT[rec])] = 100 * float(np.mean([np.mean(x) for x in by.values()]))
    v["joint_family_contrasts"] = len(cells)
    for k, flag in zip([c[0] for c in cells], holm([c[3] for c in cells])):
        v[k + "_holm_joint"] = int(flag)
    sub = [c for c in cells if not (c[1] == "spider" and c[2] == "gemma-4-26b")]
    for k, flag in zip([c[0] for c in sub], holm([c[3] for c in sub])):
        v[k + "_holm_fifteen"] = int(flag)
    sub = [c for c in cells if c[1] == "spider" and c[2] == "gemma-4-26b"]
    for k, flag in zip([c[0] for c in sub], holm([c[3] for c in sub])):
        v[k + "_holm_three"] = int(flag)


def feedback(v):
    rows = []
    with open(INP / "bird_qwen25_7b_feedback_per_question.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["all_four_conditions_present"] == "1":
                rows.append(r)
    rows.sort(key=lambda r: int(r["question_id"]))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["db_id"]].append(float(r["candidate_feedback_correct"]) - float(r["candidate_only_correct"]))
    keys = list(by)
    flat = [d for k in keys for d in by[k]]
    point = sum(flat) / len(flat)
    rnd, sims = random.Random(20261013), []
    for _ in range(4000):
        pool = []
        for _ in range(len(keys)):
            pool.extend(by[keys[rnd.randrange(len(keys))]])
        sims.append(sum(pool) / len(pool))
    sims.sort()
    v["bird_qwen25_7b_feedback_minus_queries_only"] = 100 * point
    v["bird_qwen25_7b_feedback_minus_queries_only_lo"] = 100 * sims[int(0.025 * 4000)]
    v["bird_qwen25_7b_feedback_minus_queries_only_hi"] = 100 * sims[min(3999, int(0.975 * 4000))]
    v["bird_feedback_paired_questions"] = len(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    v, rows = {}, []
    family(v, rows)
    feedback(v)
    (out / "values.json").write_text(json.dumps(dict(sorted(v.items())), indent=1), encoding="utf-8")
    with open(out / "plotted_values.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "recipient", "rule", "est", "lo", "hi"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
