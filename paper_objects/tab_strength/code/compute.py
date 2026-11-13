#!/usr/bin/env python3
"""Selection and generation calls on a fixed SQL candidate pool (tab:strength and its supplementary tables).

    python3 code/compute.py --out DIR      # -> DIR/values.json

Reads only ../inputs. Every interval is a paired database-cluster bootstrap: databases are drawn with
replacement, each database's question mean enters with equal weight ("database_equal") unless a key
says otherwise, 10,000 draws, 2.5th and 97.5th percentiles. Seeds follow the analyses that produced
the printed numbers: 20261105 for the DeepSeek-V4-Pro, pool, decode and same-source comparisons,
20261106 for the difference between the two selector backbones, 20261101 (one generator shared by
BIRD and then Spider) for the selector controls. Holm correction at 0.05 runs within the families the
paper names, on the bootstrap p-value 2*min(P(b<=0), P(b>=0)).

Operations: Unfiltered (Original), Valid and Nonempty votes; B, the Qwen3-32B selector given an
execution summary without reasoning; C, the same selector with reasoning; S_al, the selector given
the generation prompt; Generation, the recipient writing its own query. A and B/C of the selector
controls are the three selector conditions of Supplementary Section "Decoding, Pools, and Selector
Inputs".
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
DRAWS = 10000
BENCH = ("bird", "spider")
POOLS = ("original", "qwen3_glm")
EXCLUDE_COMMON = {("bird", 518), ("bird", 701)}   # their four-model candidates carry no correctness label
PRICE_IN, PRICE_OUT = 0.08, 0.28                   # USD per million input / output tokens


def read(name):
    return list(csv.DictReader(open(INP / name, encoding="utf-8")))


def summarize(per, ww, draws):
    pt = float((ww * per).sum() / ww.sum())
    bs = (per[draws] * ww[draws]).sum(1) / ww[draws].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return dict(pt=100 * pt, lo=100 * float(lo), hi=100 * float(hi), p=p)


def cboot(values, clusters, seed, weighting="database_equal"):
    """Paired cluster bootstrap with a generator drawn afresh from `seed` for this call."""
    by = collections.defaultdict(list)
    for u in sorted(values):
        by[clusters[u]].append(values[u])
    cl = sorted(by)
    per = np.array([np.mean(by[c]) for c in cl])
    n = np.array([len(by[c]) for c in cl], float)
    ww = np.ones(len(cl)) if weighting == "database_equal" else n
    draws = np.random.default_rng(seed).integers(0, len(cl), size=(DRAWS, len(cl)))
    return summarize(per, ww, draws)


def holm(results):
    """results: list of dicts with key 'p'; sets 'holm' to 1/0."""
    srt = sorted(results, key=lambda r: round(r["p"], 4))
    stop = False
    for j, r in enumerate(srt):
        rej = (not stop) and round(r["p"], 4) <= 0.05 / (len(srt) - j)
        stop = stop or not rej
        r["holm"] = int(rej)


def put(V, key, r, ci=True):
    V[key] = r["pt"]
    if ci:
        V[key + "_lo"], V[key + "_hi"] = r["lo"], r["hi"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    V = {}

    # ---------------------------------------------------------------- per-question operation table
    ops = {}
    for r in read("pool_operations_per_question.csv"):
        k = (r["pool"], r["benchmark"], int(r["question_id"]))
        ops[k] = dict(db=r["db_id"], **{m: int(r[m]) for m in
                                       ("Oracle", "Original", "Valid", "Nonempty", "B", "C", "S_al", "Generation")})
    dec = {}
    for r in read("pool_decodes_per_question.csv"):
        k = (r["pool"], r["benchmark"], int(r["question_id"]))
        dec[k] = dict(db=r["db_id"], Nonempty=int(r["Nonempty"]), Oracle=int(r["Oracle"]),
                      **{f"{m}_{d}": float(r[f"{m}_{d}"]) for m in ("S_al", "Generation") for d in ("d1", "d2", "avg")})

    def full(pool, b):
        return sorted(q for (p, bb, q) in ops if p == pool and bb == b)

    def common(b):
        return sorted(q for (p, bb, q) in ops if p == "original" and bb == b
                      and ("qwen3_glm", b, q) in ops and (b, q) not in EXCLUDE_COMMON)

    # ---------------------------------------------------------------- tab:strength (DeepSeek-V4-Pro and Qwen3-32B)
    ds = {(r["benchmark"], int(r["question_id"])): r for r in read("deepseek_per_question.csv")}
    for b in BENCH:
        qs = full("original", b)
        cl = {q: ops[("original", b, q)]["db"] for q in qs}
        x = {}
        for q in qs:
            d, o = ds.get((b, q), {}), ops[("original", b, q)]
            x[q] = dict(DS_S_al=int(d.get("DS_S_al") or 0), DS_Generation=int(d.get("DS_Generation") or 0),
                        Q3_S_al=o["S_al"], Q3_Generation=o["Generation"], Valid=o["Valid"],
                        Nonempty=o["Nonempty"], Oracle=o["Oracle"])
        V[f"ds_{b}_questions"] = len(qs)
        for m in ("Oracle", "Valid", "Nonempty", "Q3_S_al", "Q3_Generation", "DS_S_al", "DS_Generation"):
            put(V, f"ds_{b}_level_{m}", cboot({q: x[q][m] for q in qs}, cl, 20261105))
        for model in ("DS", "Q3"):
            fam = []
            for a_, z_ in (("S_al", "Valid"), ("S_al", "Nonempty"), ("Generation", "Valid"),
                           ("Generation", "Nonempty"), ("Generation", "S_al")):
                A = f"{model}_{a_}"
                Z = z_ if z_ in ("Valid", "Nonempty") else f"{model}_{z_}"
                r = cboot({q: x[q][A] - x[q][Z] for q in qs}, cl, 20261105)
                r["key"] = f"ds_{b}_{model}_{a_}_minus_{z_}"
                fam.append(r)
            holm(fam)
            for r in fam:
                put(V, r["key"], r)
                V[r["key"] + "_holm"] = r["holm"]

        # tab:s-did: DeepSeek-V4-Pro minus Qwen3-32B gains on the same questions (seed 20261106)
        for a_, z_ in (("S_al", "Nonempty"), ("Generation", "Nonempty"), ("Generation", "S_al")):
            def gain(xx, m):
                return xx[f"{m}_{a_}"] - (xx[z_] if z_ == "Nonempty" else xx[f"{m}_{z_}"])
            r = cboot({q: gain(x[q], "DS") - gain(x[q], "Q3") for q in qs}, cl, 20261106)
            put(V, f"did_{b}_{a_}_minus_{z_}", r)

    # ---------------------------------------------------------------- tab:s-aligned (first decode, full population)
    for pool in POOLS:
        fam = []
        for b in BENCH:
            qs = full(pool, b)
            cl = {q: ops[(pool, b, q)]["db"] for q in qs}
            for z_ in ("Generation", "C", "Nonempty", "B"):
                r = cboot({q: ops[(pool, b, q)]["S_al"] - ops[(pool, b, q)][z_] for q in qs}, cl, 20261105)
                r["key"] = f"aligned_{pool}_{b}_S_al_minus_{z_}"
                if z_ != "B":
                    fam.append(r)
                put(V, r["key"], r)
        holm(fam)   # six-contrast Holm family per pool
        for r in fam:
            V[r["key"] + "_holm"] = r["holm"]

    # ---------------------------------------------------------------- tab:s-repeat and tab:s-repeat-within
    CON = (("Generation", "S_al"), ("Generation", "Nonempty"), ("S_al", "Nonempty"))

    def col(m, d):
        return m if m == "Nonempty" else f"{m}_{d}"
    within = []
    for b in BENCH:
        for pool in POOLS:
            qs = sorted(q for (p, bb, q) in dec if p == pool and bb == b)
            cl = {q: dec[(pool, b, q)]["db"] for q in qs}
            for d in ("d1", "d2", "avg"):
                for a_, z_ in CON:
                    r = cboot({q: dec[(pool, b, q)][col(a_, d)] - dec[(pool, b, q)][col(z_, d)] for q in qs},
                              cl, 20261105)
                    r["key"] = f"decode_{pool}_{b}_{d}_{a_}_minus_{z_}"
                    put(V, r["key"], r)
                    if pool == "qwen3_glm" and d == "avg":
                        within.append(r)
    holm(within)
    for r in within:
        V[r["key"] + "_holm"] = r["holm"]
    change = []
    for b in BENCH:
        qs = common(b)
        cl = {q: dec[("original", b, q)]["db"] for q in qs}
        V[f"common_{b}_questions"] = len(qs)
        for a_, z_ in CON:
            def f(q, a_=a_, z_=z_):
                n_, o_ = dec[("qwen3_glm", b, q)], dec[("original", b, q)]
                return (n_[col(a_, "avg")] - n_[col(z_, "avg")]) - (o_[col(a_, "avg")] - o_[col(z_, "avg")])
            r = cboot({q: f(q) for q in qs}, cl, 20261105)
            r["key"] = f"change_{b}_avg_{a_}_minus_{z_}"
            put(V, r["key"], r)
            change.append(r)
        for m in ("Nonempty", "S_al", "Generation"):
            r = cboot({q: dec[("qwen3_glm", b, q)][col(m, "avg")] - dec[("original", b, q)][col(m, "avg")]
                       for q in qs}, cl, 20261105)
            put(V, f"levelchange_{b}_avg_{m}", r)
    holm(change)
    for r in change:
        V[r["key"] + "_holm"] = r["holm"]

    # ---------------------------------------------------------------- tab:s-availability and pool levels (common population)
    for b in BENCH:
        qs = common(b)
        for pool in POOLS:
            cl = {q: ops[(pool, b, q)]["db"] for q in qs}
            o = {q: ops[(pool, b, q)] for q in qs}
            for name, fn in (("Oracle", lambda r: r["Oracle"]), ("Original", lambda r: r["Original"]),
                             ("Nonempty", lambda r: r["Nonempty"]), ("B", lambda r: r["B"]),
                             ("unreturned_Original", lambda r: int(r["Oracle"] and not r["Original"])),
                             ("unreturned_Nonempty", lambda r: int(r["Oracle"] and not r["Nonempty"])),
                             ("Nonempty_minus_Original", lambda r: r["Nonempty"] - r["Original"])):
                put(V, f"common_{pool}_{b}_{name}", cboot({q: fn(o[q]) for q in qs}, cl, 20261105), ci=False)
            V[f"common_{pool}_{b}_Nonempty_losses_vs_Original"] = sum(
                1 for q in qs if o[q]["Original"] and not o[q]["Nonempty"])
            for m in ("S_al", "Generation"):
                put(V, f"common_{pool}_{b}_{m}_avg",
                    cboot({q: dec[(pool, b, q)][f"{m}_avg"] for q in qs}, cl, 20261105), ci=False)

    # ---------------------------------------------------------------- tab:s-quality-cost (tokens per question, database-equal)
    tok = collections.defaultdict(dict)
    for r in read("call_tokens.csv"):
        tok[(r["pool"], r["benchmark"], r["operation"], r["decode"])][int(r["question_id"])] = (
            int(r["input"]), int(r["output"]))
    ratios = []
    for b in BENCH:
        qs = common(b)
        for pool in POOLS:
            dbs = sorted({ops[(pool, b, q)]["db"] for q in qs})
            price = {}
            for op, decs in (("B", ("d1",)), ("S_al", ("d1", "d2")), ("Generation", ("d1", "d2"))):
                mi, mo = [], []
                for d in decs:
                    t = tok[(pool, b, op, d)]
                    byi, byo = collections.defaultdict(list), collections.defaultdict(list)
                    for q in qs:
                        i, o_ = t.get(q, (0, 0))
                        byi[ops[(pool, b, q)]["db"]].append(i)
                        byo[ops[(pool, b, q)]["db"]].append(o_)
                    mi.append(float(np.mean([np.mean(byi[x]) for x in dbs])))
                    mo.append(float(np.mean([np.mean(byo[x]) for x in dbs])))
                mi, mo = float(np.mean(mi)), float(np.mean(mo))
                V[f"cost_{pool}_{b}_{op}_input"] = mi
                V[f"cost_{pool}_{b}_{op}_output"] = mo
                price[op] = 1000 * (mi * PRICE_IN + mo * PRICE_OUT) / 1e6
                V[f"cost_{pool}_{b}_{op}_usd_per_1000q"] = price[op]
            ratios.append(price["S_al"] / price["Generation"])
            V[f"cost_{pool}_{b}_B_over_Generation"] = price["B"] / price["Generation"]
    V["cost_S_al_over_Generation_min"] = min(ratios)
    V["cost_S_al_over_Generation_max"] = max(ratios)

    # ---------------------------------------------------------------- candidate acquisition (output tokens, millions)
    for r in read("candidate_acquisition.csv"):
        V[f"acquisition_{r['pool']}_{r['benchmark']}_output_M"] = int(r["output_tokens"]) / 1e6

    # ---------------------------------------------------------------- same-source samples: change from two to six candidates
    rows = read("same_source_samples_per_question.csv")
    for b in ("bird",):
        y = collections.defaultdict(dict)
        cl = {}
        for r in rows:
            if r["benchmark"] != b:
                continue
            q, k = int(r["question_id"]), int(r["k"])
            cl[q] = r["db_id"]
            y[k][q] = dict(Oracle=int(r["Oracle"]), Nonempty=int(r["Nonempty"]),
                           Generation=int(r["Generation"]) if r["Generation"] not in ("", None) else None)
        for name, fn in (("Oracle", lambda v: v["Oracle"]), ("Nonempty", lambda v: v["Nonempty"]),
                         ("Generation_minus_Nonempty", lambda v: v["Generation"] - v["Nonempty"])):
            r = cboot({q: fn(y[6][q]) - fn(y[2][q]) for q in y[2]}, cl, 20261105)
            put(V, f"samesource_{b}_k6_minus_k2_{name}", r)

    # ---------------------------------------------------------------- tab:s-selector-controls (conditions A, B, C)
    sc = read("selector_controls_per_question.csv")
    rng = np.random.default_rng(20261101)
    fam = []
    for b in BENCH:
        sub = [r for r in sc if r["benchmark"] == b]
        dbs = sorted({r["db_id"] for r in sub})
        by = {d: [r for r in sub if r["db_id"] == d] for d in dbs}
        draws = rng.integers(0, len(dbs), size=(DRAWS, len(dbs)))

        def stat(fn):
            per = np.array([np.mean([fn(r) for r in by[d]]) for d in dbs])
            return summarize(per, np.ones(len(dbs)), draws)
        for m in ("A", "B", "C", "Nonempty", "Generation"):
            put(V, f"controls_{b}_level_{m}", stat(lambda r, m=m: int(r[m])), ci=False)
        for x_, z_ in (("B", "A"), ("C", "B"), ("C", "Nonempty"), ("Generation", "C")):
            r = stat(lambda r, x_=x_, z_=z_: int(r[x_]) - int(r[z_]))
            r["key"] = f"controls_{b}_{x_}_minus_{z_}"
            put(V, r["key"], r)
            fam.append(r)
    holm(fam)   # eight-contrast family over both benchmarks
    for r in fam:
        V[r["key"] + "_holm"] = r["holm"]
    calls = read("selector_controls_calls.csv")
    for c in ("A", "B", "C"):
        cc = [r for r in calls if r["condition"] == c]
        V[f"controls_{c}_questions"] = len(cc)
        V[f"controls_{c}_input_total"] = sum(int(r["input"]) for r in cc)
        V[f"controls_{c}_output_total"] = sum(int(r["output"]) for r in cc)
        V[f"controls_{c}_reasoning_total"] = sum(int(r["reasoning"]) for r in cc)
        V[f"controls_{c}_invalid_choices"] = sum(1 for r in cc if not r["chosen_position"])
        for b in BENCH:
            cb = [r for r in cc if r["benchmark"] == b]
            shown = [r for r in cb if r["first_position_shown"] == "1"]
            V[f"controls_{c}_{b}_first_position_pct"] = 100 * sum(r["chosen_position"] == "C1" for r in shown) / len(shown)
            V[f"controls_{c}_{b}_mean_latency_s"] = float(np.mean([float(r["latency_s"]) for r in cb]))

    clean = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in sorted(V.items())}
    (out / "values.json").write_text(json.dumps(clean, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
