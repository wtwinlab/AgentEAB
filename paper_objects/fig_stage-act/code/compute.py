from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
DRAWS, SEED = 10000, 20261105
MIN_N = 30


def read(name):
    with open(INP / name, newline="") as f:
        return list(csv.DictReader(f))


def cluster_boot(values: dict, clusters: dict):
    by = collections.defaultdict(list)
    for u in sorted(values):
        by[clusters[u]].append(values[u])
    cl = sorted(by)
    per = np.array([np.mean(by[c]) for c in cl])
    n = np.array([len(by[c]) for c in cl], float)
    d = np.random.default_rng(SEED).integers(0, len(cl), size=(DRAWS, len(cl)))
    bs = (per[d] * n[d]).sum(1) / n[d].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def strat_boot(values: dict, strata: dict):
    g = collections.defaultdict(list)
    for k, v in values.items():
        g[strata[k]].append(v)
    rng = np.random.default_rng(SEED)
    g = {k: np.asarray(v, float) for k, v in sorted(g.items())}
    n = sum(len(v) for v in g.values())
    tot = np.zeros(DRAWS)
    for v in g.values():
        tot += v[rng.integers(0, len(v), size=(DRAWS, len(v)))].sum(1)
    lo, hi = np.percentile(tot / n, [2.5, 97.5])
    return float(lo), float(hi)


def decompose(units, boot):
    D = {k: ys - ya for k, _, _, ys, ya in units}
    cl = {k: c for k, c, _, _, _ in units}
    act = [u for u in units if u[2]]
    n, nA = len(units), len(act)
    assert all(D[k] == 0 for k, _, a, _, _ in units if not a)
    r = dict(n=n, n_A=nA, P_A=100 * nA / n,
             gains=sum(D[u[0]] > 0 for u in act), losses=sum(D[u[0]] < 0 for u in act),
             both_wrong=sum(1 for u in act if not u[3] and not u[4]),
             total=100 * sum(D.values()) / n, E_D_A=100 * sum(D[u[0]] for u in act) / nA)
    r["total_lo"], r["total_hi"] = (100 * x for x in boot(D, cl))
    r["P_A_lo"], r["P_A_hi"] = (100 * x for x in boot({k: float(a) for k, _, a, _, _ in units}, cl))
    if nA >= MIN_N:
        r["E_D_A_lo"], r["E_D_A_hi"] = (100 * x for x in boot({u[0]: D[u[0]] for u in act}, cl))
    return r


ROWS = {}
for mt in ("q3", "q72"):
    for b in ("bird", "spider"):
        U = [((int(r["question_id"]),), r["db_id"], int(r["A_result"]), int(r["Y_stage"]), int(r["Y_alt"]))
             for r in read(f"macsql_units_{mt}_{b}.csv")]
        ROWS[f"refiner_{mt}_{b}"] = decompose(U, cluster_boot)
U = [((int(r["question_id"]),), r["db_id"], int(r["A_result"]), int(r["Y_stage"]), int(r["Y_alt"]))
     for r in read("chess_units.csv") if r["alternative"] == "SC_official"]
ROWS["ut_sc"] = decompose(U, cluster_boot)
comem = read("comem_anchors.csv")
for mt in ("q3", "q7"):
    U = [((r["dataset"], r["id_left"]), r["dataset"], int(r["changed"]), int(r["selector_correct"]), int(r["ranker_correct"]))
         for r in comem if r["model"] == mt]
    ROWS[f"selector_{mt}"] = decompose(U, strat_boot)

SHARE = {}
mq = read("macsql_calls_per_question.csv")
for mt in ("q3", "q72"):
    for b in ("bird", "spider"):
        rs = [r for r in mq if r["model"] == mt and r["benchmark"] == b]
        st = [r for r in rs if r["agent"] == "refiner"]
        tok = lambda xs: sum(int(r["prompt_tokens"]) + int(r["completion_tokens"]) for r in xs)
        SHARE[f"refiner_{mt}_{b}"] = (100 * sum(int(r["calls"]) for r in st) / sum(int(r["calls"]) for r in rs),
                                      100 * tok(st) / tok(rs))
cq = read("chess_costs_per_question.csv")
ut = [r for r in cq if r["part"] == "ut_increment"]
tk = lambda xs: sum(int(r["prompt_tokens"]) + int(r["completion_tokens"]) for r in xs)
SHARE["ut_sc"] = (100 * sum(int(r["calls"]) for r in ut) / sum(int(r["calls"]) for r in cq), 100 * tk(ut) / tk(cq))
SHARE["selector_q3"] = SHARE["selector_q7"] = (100.0, 100.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    v = {}
    for k, r in ROWS.items():
        for f in ("n", "n_A", "P_A", "P_A_lo", "P_A_hi", "gains", "losses", "both_wrong", "total", "total_lo", "total_hi",
                  "E_D_A", "E_D_A_lo", "E_D_A_hi"):
            if f in r:
                v[f"{k}_{f}"] = round(float(r[f]), 6)
        v[f"{k}_call_share"] = round(SHARE[k][0], 6)
        v[f"{k}_token_share"] = round(SHARE[k][1], 6)
    (out / "values.json").write_text(json.dumps(v, indent=1, sort_keys=True) + "\n")

    spec = [("refiner_q3_bird", "Refiner, Qwen3, BIRD"), ("refiner_q72_bird", "Refiner, 72B, BIRD"),
            ("refiner_q3_spider", "Refiner, Qwen3, Spider"), ("refiner_q72_spider", "Refiner, 72B, Spider"),
            ("ut_sc", "Unit tester, Qwen3, BIRD"), ("selector_q3", "Selector, Qwen3, EM"), ("selector_q7", "Selector, 7B, EM")]
    with open(out / "plotted_values.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["label", "gain", "loss", "both", "pa", "total", "calls", "tokens"])
        for k, lab in spec:
            r = ROWS[k]
            w.writerow([lab, 100 * r["gains"] / r["n"], 100 * r["losses"] / r["n"], 100 * r["both_wrong"] / r["n"],
                        round(r["P_A"], 2), round(r["total"], 3), round(SHARE[k][0], 2), round(SHARE[k][1], 2)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
