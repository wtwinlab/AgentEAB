"""Peer content on BIRD: full peer responses, peers' extracted SQL only, and no peers.

    python3 code/compute.py --out DIR     # -> DIR/values.json, DIR/contrasts.json

Input (inputs/runs.csv): one row per debate run of the matched collection on the 198 sampled BIRD
questions: question, database, protocol (Du-MAD, MAD-M2), backbone (Llama-3-8B, Qwen2.5-7B),
condition (FULL = full peer responses, ANSWER_ONLY = each peer's extracted SQL in the protocol's answer
form, NOPEER = no peers), repeat, correctness of the execution vote over the initial answers (r0) and
the last round (last) with the lenient extractor and first-agent ties, correctness of the protocol's
final answer (final), and each agent's initial and last-round correctness (lenient extractor).

Vote-level contrasts compare the round-to-round change (last round minus initial answers) between two
conditions, paired per question; repeats are averaged within a question and configurations are averaged
per question. Conditional loss is the share of initially correct agent answers that are wrong in the last
round; conditional recovery is the share of initially wrong answers that are correct; both are ratios of
counts pooled over repeats and agents. Intervals: 95% percentile bootstrap, 10,000 draws, one generator
(seed 20261109) that first draws question weights and then database-cluster weights over the 11
databases. A contrast is detected when both intervals exclude 0.
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
B, SEED = 10000, 20261109
PROTOS = ("du_mad", "madm2_objective")
BBS = ("llama3-8b", "qwen2.5-7b")
CELLS = [(p, b) for p in PROTOS for b in BBS]
CONDS = ("FULL", "ANSWER_ONLY", "NOPEER")
CONTRASTS = (("ANSWER_ONLY", "FULL"), ("ANSWER_ONLY", "NOPEER"), ("FULL", "NOPEER"))


def load():
    rows = []
    with open(INP / "runs.csv") as f:
        for r in csv.DictReader(f):
            for k in ("r0", "last", "final"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load()
    units = sorted({r["unit"] for r in rows})
    db_of = {r["unit"]: r["db"] for r in rows}
    dbs = sorted({db_of[u] for u in units})
    uix = {u: i for i, u in enumerate(units)}
    rng = np.random.default_rng(SEED)
    WQ = rng.multinomial(len(units), [1 / len(units)] * len(units), size=B).astype(float)
    cnt_db = rng.multinomial(len(dbs), [1 / len(dbs)] * len(dbs), size=B).astype(float)
    WD = cnt_db[:, np.array([dbs.index(db_of[u]) for u in units])]

    def ci(est):
        lo, hi = np.percentile(est, [2.5, 97.5])
        return [round(100 * float(lo), 2), round(100 * float(hi), 2)]

    def mean_boot(x, mask):
        x = np.where(mask, x, 0.0)
        m = mask.astype(float)
        o = dict(est=round(100 * float(x.sum() / m.sum()), 2), n_questions=int(m.sum()))
        for tag, W in (("q_ci", WQ), ("db_ci", WD)):
            o[tag] = ci((W @ x) / np.maximum(W @ m, 1e-12))
        o["detected"] = all(not (o[t][0] <= 0 <= o[t][1]) for t in ("q_ci", "db_ci"))
        return o

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        per[(r["protocol"], r["backbone"], r["condition"])][r["unit"]].append(r)

    def vec(cell, cond, metric):
        x = np.full(len(units), np.nan)
        for u, rs in per[(cell[0], cell[1], cond)].items():
            if metric == "E1":
                x[uix[u]] = np.mean([r["last"] - r["r0"] for r in rs])
            else:
                x[uix[u]] = np.mean([r[metric] for r in rs])
        return x

    V, C = {}, {}
    V["n_questions"] = len(units)
    V["n_runs"] = len(rows)
    V["n_runs_per_condition"] = {c: sum(r["condition"] == c for r in rows) for c in CONDS}
    # vote-level levels and contrasts
    for metric in ("r0", "last", "final", "E1"):
        for cell in CELLS:
            for cond in CONDS:
                x = vec(cell, cond, metric)
                V[f"{metric}|{cell[0]}|{cell[1]}|{cond}"] = mean_boot(np.nan_to_num(x), np.isfinite(x))["est"]
        for a1, a2 in CONTRASTS:
            diffs = {cell: vec(cell, a1, metric) - vec(cell, a2, metric) for cell in CELLS}
            scopes = {f"{c[0]}|{c[1]}": [c] for c in CELLS}
            scopes.update({"DuMAD_2cell": CELLS[:2], "MADM2_2cell": CELLS[2:], "4cell": CELLS})
            for sc, cl in scopes.items():
                D = np.vstack([diffs[c] for c in cl])
                m = np.isfinite(D).all(0)
                b = mean_boot(np.nan_to_num(D.mean(0)), m)
                key = f"{metric}|{sc}|{a1}-{a2}"
                C[key] = b
                V[key] = b["est"]
                V[key + "_q_lo"], V[key + "_q_hi"] = b["q_ci"]
                V[key + "_db_lo"], V[key + "_db_hi"] = b["db_ci"]
                V[key + "_detected"] = b["detected"]
    # per-agent conditional loss and recovery
    for cell in CELLS:
        for cond in CONDS:
            cnt = collections.Counter()
            for rs in per[(cell[0], cell[1], cond)].values():
                for r in rs:
                    for x, y in zip(r["agents_r0"], r["agents_last"]):
                        cnt[("retained" if y == "1" else "lost") if x == "1" else
                            ("recovered" if y == "1" else "stayed_wrong")] += 1
            k = f"{cell[0]}|{cell[1]}|{cond}"
            V[f"cond_loss|{k}"] = round(100 * cnt["lost"] / (cnt["retained"] + cnt["lost"]), 4)
            V[f"cond_rec|{k}"] = round(100 * cnt["recovered"] / (cnt["recovered"] + cnt["stayed_wrong"]), 4)
            V[f"agent_counts|{k}"] = dict(cnt)
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n")
    (out / "contrasts.json").write_text(json.dumps(C, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
