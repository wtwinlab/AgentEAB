r"""Interaction effect across Qwen2.5 backbone sizes on BIRD (Table tab:scale, Table tab:s-scale-contrasts).

    python3 code/compute.py --out DIR      -> DIR/values.json, DIR/scale_table.csv, DIR/scale_contrasts.csv

Input: inputs/debate_runs.csv, one row per debate run (model, protocol, condition, question, database,
repeat) with the correctness of the initial answers and of the last round, both read with the lenient
extractor and the execution vote, ties going to the first agent.

Estimands (pp):
  round change  E1 = last-round accuracy minus initial-answer accuracy, per question, repeats averaged;
  interaction effect  Delta_peer = E1 with peers minus E1 without peers, paired by question;
  per-model value of Du-MAD and MAD-M2 = per-question mean of the two protocols' values;
  scale contrast = larger minus smaller model on that per-question mean.
Intervals: 10,000 bootstrap draws over the 198 questions and, separately, over the 11 BIRD databases
(every question of a drawn database enters), seed 20261106 for both, percentile intervals. Two-sided
p = 2 min(share of draws <= 0, share >= 0), floored at 1/10,000. Holm correction runs over the five
Qwen2.5 contrasts. The equivalence cell (MAD-M2 at 72B) has a 90% interval over questions.
Questions bird|518 and bird|701 are excluded.
"""
import argparse
import collections
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
EXCL = {"bird|518", "bird|701"}
BOOT, SEED = 10000, 20261106
SIZES = [("7B", "qwen2.5-7b"), ("14B", "qwen25-14b-awq"), ("32B", "qwen25-32b-awq"), ("72B", "qwen25-72b-awq")]
PROTOS = [("dumad", "du_mad"), ("madm2", "madm2_objective"), ("bccs", "bccs")]
PAIRS = [("72b_7b", "qwen25-72b-awq", "qwen2.5-7b"), ("14b_7b", "qwen25-14b-awq", "qwen2.5-7b"),
         ("32b_14b", "qwen25-32b-awq", "qwen25-14b-awq"), ("72b_32b", "qwen25-72b-awq", "qwen25-32b-awq"),
         ("72b_14b", "qwen25-72b-awq", "qwen25-14b-awq"),
         ("gptoss_120b_20b", "gpt-oss-120b", "gpt-oss-20b"), ("llama_70b_8b", "llama33-70b", "llama3-8b")]
HOLM = ["72b_7b", "14b_7b", "32b_14b", "72b_32b", "72b_14b"]


def r2(v):
    return round(100 * float(v), 2)


def stats(d, db):
    """Estimate, question and database intervals, two-sided p (same estimator, seed and draws as the source analysis)."""
    us = sorted(d)
    x = np.array([d[u] for u in us])
    rng = np.random.default_rng(SEED)
    t = x[rng.integers(0, len(x), size=(BOOT, len(x)))].mean(axis=1)
    g = collections.defaultdict(list)
    for i, u in enumerate(us):
        g[db[u]].append(i)
    keys = sorted(g)
    rng = np.random.default_rng(SEED)
    c = np.array([x[[i for k in rng.integers(0, len(keys), size=len(keys)) for i in g[keys[k]]]].mean()
                  for _ in range(BOOT)])
    p = max(1 / BOOT, min(1.0, 2 * min((t <= 0).mean(), (t >= 0).mean())))
    return dict(n=len(us), est=r2(x.mean()), q_lo=r2(np.percentile(t, 2.5)), q_hi=r2(np.percentile(t, 97.5)),
                db_lo=r2(np.percentile(c, 2.5)), db_hi=r2(np.percentile(c, 97.5)), p=round(float(p), 5))


def per_question(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["question"]].append(r["last"] - r["initial"])
    return {u: float(np.mean(v)) for u, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    cells, db = collections.defaultdict(list), {}
    with open(INP / "debate_runs.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["question"] in EXCL:
                continue
            db[r["question"]] = r["database"]
            cells[(r["model"], r["protocol"], r["condition"])].append(
                dict(question=r["question"], repeat=int(r["repeat"]),
                     initial=float(r["initial_correct"]), last=float(r["last_correct"])))
    V = {"n_questions": len(db), "n_databases": len(set(db.values()))}
    dp = {}          # (model, protocol) -> per-question Delta_peer
    for m in {k[0] for k in cells}:
        for _, p in PROTOS:
            F, N = cells.get((m, p, "with_peers")), cells.get((m, p, "no_peers"))
            if F and N:
                ef, en = per_question(F), per_question(N)
                dp[(m, p)] = {u: ef[u] - en[u] for u in set(ef) & set(en)}
        if (m, "du_mad") in dp and (m, "madm2_objective") in dp:
            a, b = dp[(m, "du_mad")], dp[(m, "madm2_objective")]
            dp[(m, "mean")] = {u: float(np.mean([a[u], b[u]])) for u in set(a) & set(b)}
        elif (m, "du_mad") in dp:
            dp[(m, "mean")] = dp[(m, "du_mad")]
    table = []
    for size, m in SIZES:
        for key, p in PROTOS + [("mean", "mean")]:
            s = stats(dp[(m, p)], db)
            for f in ("est", "q_lo", "q_hi", "db_lo", "db_hi"):
                V[f"dpeer_{key}_{size}_{f}"] = s[f]
            table.append(dict(size=size, protocol=p, **s))
    con = {}
    for name, a, b in PAIRS:
        da, dbb = dp[(a, "mean")], dp[(b, "mean")]
        con[name] = stats({u: da[u] - dbb[u] for u in set(da) & set(dbb)}, db)
    order = sorted(HOLM, key=lambda n: con[n]["p"])
    run = 0.0
    for rank, n in enumerate(order):
        run = max(run, min(1.0, (len(HOLM) - rank) * con[n]["p"]))
        con[n]["p_holm"] = round(run, 5)
    rows = []
    for name, a, b in PAIRS:
        s = con[name]
        for f in ("est", "q_lo", "q_hi", "db_lo", "db_hi", "p", "p_holm"):
            if f in s:
                V[f"contrast_{name}_{f}"] = s[f]
        rows.append(dict(contrast=name, larger=a, smaller=b, **s))
    # equivalence cell: MAD-M2 at 72B, 90% interval over questions, margin +-2 pp; and its two repeats
    F = cells[("qwen25-72b-awq", "madm2_objective", "with_peers")]
    N = cells[("qwen25-72b-awq", "madm2_objective", "no_peers")]
    ef, en = per_question(F), per_question(N)
    d = np.array([ef[u] - en[u] for u in sorted(set(ef) & set(en))])
    b = d[np.random.default_rng(SEED).integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(b, [5, 95])
    V.update(equiv_madm2_72B_est=r2(d.mean()), equiv_madm2_72B_ci90_lo=r2(lo), equiv_madm2_72B_ci90_hi=r2(hi),
             equiv_madm2_72B_inside_margin=int(100 * lo > -2 and 100 * hi < 2))
    for rep in (1, 2):
        a = per_question([r for r in F if r["repeat"] == rep])
        c = per_question([r for r in N if r["repeat"] == rep])
        V[f"equiv_madm2_72B_repeat{rep}"] = r2(np.mean([a[u] - c[u] for u in sorted(set(a) & set(c))]))
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True))
    for name, data in (("scale_table.csv", table), ("scale_contrasts.csv", rows)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()) + (["p_holm"] if "p_holm" not in data[0] else []),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(data)


if __name__ == "__main__":
    main()
