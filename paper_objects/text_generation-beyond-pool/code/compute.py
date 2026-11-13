#!/usr/bin/env python3
"""Split generation's gain over Nonempty voting by whether the four-model pool holds a correct query.

    python3 code/compute.py --out output     # -> output/values.json, output/contributions.csv

A question's pool holds a correct query (Oracle = 1) when at least one of the four candidates returns
the reference result. A correct generated query on an Oracle = 1 question returns the same rows as a
correct candidate; a correct generated query on an Oracle = 0 question returns a result no candidate
returned. Each contrast is split into the two strata so that the parts add up to the total: each
question's paired difference is multiplied by the stratum indicator and averaged with the same weights
as the total. Weights: each database equal. Intervals: 95% database-cluster bootstrap, 10,000 draws.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
DRAWS, SEED = 10000, 20261105
METHODS = ("Valid", "Nonempty", "Q3_S_al", "Q3_G", "DS_S_al", "DS_G")
CONTRASTS = (("DS_G", "Nonempty"), ("DS_G", "DS_S_al"), ("DS_S_al", "Nonempty"),
             ("Q3_G", "Nonempty"), ("Q3_G", "Q3_S_al"))


def cluster_boot(values: dict, clusters: dict):
    """Paired cluster bootstrap: per-database mean, databases weighted equally."""
    by = collections.defaultdict(list)
    for u in sorted(values):
        by[clusters[u]].append(values[u])
    cl = sorted(by)
    per = np.array([np.mean(by[c]) for c in cl])
    d = np.random.default_rng(SEED).integers(0, len(cl), size=(DRAWS, len(cl)))
    bs = per[d].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return 100 * float(per.mean()), 100 * float(lo), 100 * float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [{k: (v if k in ("benchmark", "db_id") else int(v)) for k, v in r.items()}
            for r in csv.DictReader(open(HERE / "inputs" / "pool_questions.csv", encoding="utf-8"))]
    V, table = {}, []
    for b in ("bird", "spider"):
        R = [r for r in rows if r["benchmark"] == b]
        cl = {r["question_id"]: r["db_id"] for r in R}
        V[f"{b}_questions"] = len(R)
        V[f"{b}_databases"] = len(set(cl.values()))
        for s in (1, 0):
            sub = [r for r in R if r["Oracle"] == s]
            V[f"{b}_oracle{s}_questions"] = len(sub)
            for m in METHODS:
                n = sum(r[m] for r in sub)
                V[f"{b}_oracle{s}_{m}_correct"] = n
                V[f"{b}_oracle{s}_{m}_pct"] = 100 * n / len(sub)
        for A, B in CONTRASTS:
            name = f"{b}_{A}_minus_{B}"
            t = cluster_boot({r["question_id"]: r[A] - r[B] for r in R}, cl)
            V[name], V[name + "_lo"], V[name + "_hi"] = t
            V[name + "_detected"] = int(t[1] > 0 or t[2] < 0)
            row = dict(benchmark=b, contrast=f"{A} - {B}", total=t[0], total_lo=t[1], total_hi=t[2])
            for s in (1, 0):
                c = cluster_boot({r["question_id"]: (r[A] - r[B]) * (r["Oracle"] == s) for r in R}, cl)
                key = f"{name}_oracle{s}"
                V[key], V[key + "_lo"], V[key + "_hi"] = c
                V[key + "_detected"] = int(c[1] > 0 or c[2] < 0)
                sub = [r for r in R if r["Oracle"] == s]
                V[key + "_gained"] = sum(r[A] and not r[B] for r in sub)
                V[key + "_lost"] = sum(r[B] and not r[A] for r in sub)
                row.update({f"oracle{s}": c[0], f"oracle{s}_lo": c[1], f"oracle{s}_hi": c[2],
                            f"oracle{s}_gained": V[key + "_gained"], f"oracle{s}_lost": V[key + "_lost"]})
            table.append(row)
        V[f"{b}_DS_G_minus_Nonempty_share_from_oracle0_pct"] = (
            100 * V[f"{b}_DS_G_minus_Nonempty_oracle0"] / V[f"{b}_DS_G_minus_Nonempty"])
    V = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sorted(V.items())}
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with open(out / "contributions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
