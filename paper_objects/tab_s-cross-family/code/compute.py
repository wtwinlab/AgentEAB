r"""Cross-family BIRD debates (Table tab:s-cross-family and the prose of sec:scale and sec:s-spend-scale).

    python3 code/compute.py --out DIR      -> DIR/values.json, DIR/cross_family.csv, DIR/levels.csv

Input: inputs/debate_runs.csv, one row per debate run (model, protocol, condition, question, database,
repeat) with the correctness of the initial answers and of the last round, both read with the lenient
extractor and the execution vote, ties going to the first agent.

Per model (pp):
  later rounds  E1 = last-round accuracy minus initial-answer accuracy with peers, per question, repeats averaged;
  peer interaction effect  Delta_peer = E1 with peers minus E1 without peers, paired by question;
  both are the per-question mean of Du-MAD and MAD-M2 (MiniMax-M2.5 runs Du-MAD only, so its row is Du-MAD).
Intervals: 10,000 bootstrap draws over the 198 questions and, separately, over the 11 BIRD databases, seed
20261106 for both. Two-sided p = 2 min(share of draws <= 0, share >= 0), floored at 1/10,000. Holm
correction runs over the nine models of the cross-family family (models with both repeats on all 198
questions in both conditions). Accuracy levels are the mean over questions of per-question accuracy for the
initial answers and the last round, with and without peers, in Du-MAD and MAD-M2.
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
MODELS = [("agentar", "agentar-sql-32b"), ("gemma4", "gemma4-26b-a4b"), ("glm47", "glm47-flash"),
          ("gptoss120b", "gpt-oss-120b"), ("gptoss20b", "gpt-oss-20b"), ("llama3_8b", "llama3-8b"),
          ("llama33_70b", "llama33-70b"), ("minimax", "minimax-m25-awq"), ("mistral32", "mistral-small-32"),
          ("qwen25_7b", "qwen2.5-7b"), ("qwen25_14b", "qwen25-14b-awq"), ("qwen25_32b", "qwen25-32b-awq"),
          ("qwen25_72b", "qwen25-72b-awq"), ("qwen3_32b", "qwen3-32b"), ("qwen36", "qwen36-35b-a3b"),
          ("qwen38", "qwen38-27b")]
HOLM = ["qwen3-32b", "qwen36-35b-a3b", "qwen38-27b", "glm47-flash", "mistral-small-32", "gemma4-26b-a4b",
        "llama33-70b", "minimax-m25-awq", "agentar-sql-32b"]
IDENT = ("du_mad", "madm2_objective")


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


def per_question(rows, f):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["question"]].append(f(r))
    return {u: float(np.mean(v)) for u, v in by.items()}


def e1(r):
    return r["last"] - r["initial"]


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
    V = {"n_models": len(MODELS), "n_questions": len(db)}
    res, levels, complete = {}, [], {}
    for key, m in MODELS:
        e1f, dp, comp = {}, {}, []
        for p in IDENT:
            F, N = cells.get((m, p, "with_peers")), cells.get((m, p, "no_peers"))
            if not F:
                continue
            ef, en = per_question(F, e1), per_question(N, e1)
            e1f[p] = ef
            dp[p] = {u: ef[u] - en[u] for u in set(ef) & set(en)}
            comp.append(len({(r["question"], r["repeat"]) for r in F}) >= 396 and
                        len({(r["question"], r["repeat"]) for r in N}) >= 396)
            for cond, rows in (("with_peers", F), ("no_peers", N)):
                for rnd in ("initial", "last"):
                    lv = 100 * float(np.mean(list(per_question(rows, lambda r: r[rnd]).values())))
                    levels.append(dict(model=m, protocol=p, condition=cond, round=rnd, accuracy=round(lv, 2)))
            if m == "minimax-m25-awq" and p == "du_mad":
                s = stats(en, db)
                V.update(minimax_rounds_nopeer_est=s["est"], minimax_rounds_nopeer_q_lo=s["q_lo"],
                         minimax_rounds_nopeer_q_hi=s["q_hi"])
        if len(dp) == 2:
            a, b = e1f["du_mad"], e1f["madm2_objective"]
            ie = {u: float(np.mean([a[u], b[u]])) for u in set(a) & set(b)}
            a, b = dp["du_mad"], dp["madm2_objective"]
            idp = {u: float(np.mean([a[u], b[u]])) for u in set(a) & set(b)}
        else:
            ie, idp = e1f["du_mad"], dp["du_mad"]
        complete[m] = all(comp)
        res[m] = (stats(ie, db), stats(idp, db), idp)
        for name, s in (("rounds", res[m][0]), ("dpeer", res[m][1])):
            for f in ("est", "q_lo", "q_hi", "db_lo", "db_hi", "p"):
                V[f"{key}_{name}_{f}"] = s[f]
    fam = [m for m in HOLM if complete[m]]
    order = sorted(fam, key=lambda m: res[m][1]["p"])
    run, holm = 0.0, {}
    for rank, m in enumerate(order):
        run = max(run, min(1.0, (len(fam) - rank) * res[m][1]["p"]))
        holm[m] = round(run, 5)
    for key, m in MODELS:
        if m in holm:
            V[f"{key}_dpeer_p_holm"] = holm[m]
    V["holm_family_size"] = len(fam)
    V["holm_max_except_minimax"] = min(holm[m] for m in fam if m != "minimax-m25-awq")
    est = [res[m][1]["est"] for _, m in MODELS]
    V["dpeer_min"], V["dpeer_max"] = min(est), max(est)
    V["n_positive_dpeer_q_ci_excluding_zero"] = sum(res[m][1]["est"] > 0 and res[m][1]["q_lo"] > 0 for _, m in MODELS)
    V["n_detected_both_intervals"] = sum((s["q_lo"] > 0 or s["q_hi"] < 0) and (s["db_lo"] > 0 or s["db_hi"] < 0)
                                         for s in (res[m][1] for _, m in MODELS))
    # Llama-3.3-70B minus Llama-3-8B on the per-question Delta_peer
    a, b = res["llama33-70b"][2], res["llama3-8b"][2]
    s = stats({u: a[u] - b[u] for u in set(a) & set(b)}, db)
    V.update(llama_70b_minus_8b_est=s["est"], llama_70b_minus_8b_q_lo=s["q_lo"], llama_70b_minus_8b_db_lo=s["db_lo"])
    # accuracy levels (Du-MAD and MAD-M2, initial and last round, with and without peers)
    acc = [l["accuracy"] for l in levels]
    V["accuracy_min"], V["accuracy_max"] = min(acc), max(acc)
    V["n_models_above_50"] = len({l["model"] for l in levels if l["accuracy"] > 50})
    small = [l["accuracy"] for l in levels if l["model"] in ("llama3-8b", "qwen2.5-7b")]
    V["accuracy_7_8b_min"], V["accuracy_7_8b_max"] = min(small), max(small)
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True))
    with open(out / "cross_family.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "rounds", "rounds_q_lo", "rounds_q_hi", "rounds_db_lo", "rounds_db_hi", "dpeer", "dpeer_q_lo",
                    "dpeer_q_hi", "dpeer_db_lo", "dpeer_db_hi", "dpeer_p", "dpeer_p_holm"])
        for _, m in MODELS:
            a, b, _ = res[m]
            w.writerow([m, a["est"], a["q_lo"], a["q_hi"], a["db_lo"], a["db_hi"], b["est"], b["q_lo"], b["q_hi"],
                        b["db_lo"], b["db_hi"], b["p"], holm.get(m, "")])
    with open(out / "levels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(levels[0]))
        w.writeheader()
        w.writerows(levels)


if __name__ == "__main__":
    main()
