"""Fig. fig:readout-cells: interaction effect per BIRD cell under each extractor, vote and tie rule.

    python3 code/compute.py --out output

Reads only ../inputs/per_run_scores.jsonl. For each of the six cells (Du-MAD, MAD-M2, BCCS with
Llama-3-8B or Qwen2.5-7B agents) and their equal-weight mean, the interaction effect is
(last round - initial answers) with peers minus the same difference without peers, in pp, with a
95% percentile bootstrap interval over the 198 questions (10,000 draws, seed 20261106, repeats
averaged within a question). Writes values.json and plotted_values.csv (the 84 plotted points).
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
BOOT, SEED = 10000, 20261106
INTER = ("du_mad", "madm2_objective", "bccs")
EXTS, IDS, RDS, ARMS = ("strict", "lenient"), ("string", "exec"), ("r0", "last"), ("NOPEER", "FULL")
CELLS = ["du_mad|qwen2.5-7b", "du_mad|llama3-8b", "madm2_objective|qwen2.5-7b", "madm2_objective|llama3-8b",
         "bccs|qwen2.5-7b", "bccs|llama3-8b", "mean"]
READOUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]           # (extractor, vote): protocol/lenient, string/exec
TIES = ("first", "last", "random")


def build(by_unit, tie):
    units = sorted(u for u, rs in by_unit.items() if {x["arm"] for x in rs} >= set(ARMS))
    Y = np.zeros((len(units), 2, 2, 2, 2))
    for ui, u in enumerate(units):
        for a, arm in enumerate(ARMS):
            rs = [x for x in by_unit[u] if x["arm"] == arm]
            for e, i, rd in itertools.product((0, 1), repeat=3):
                Y[ui, e, i, rd, a] = np.mean([x[f"{EXTS[e]}|{IDS[i]}|{RDS[rd]}|{tie}"] for x in rs])
    return units, Y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in (INP / "per_run_scores.jsonl").read_text().splitlines():
        r = json.loads(l)
        by[f"{r['protocol']}|{r['backbone']}"][r["unit"]].append(r)
    inter = sorted(c for c in by if c.split("|")[0] in INTER)
    V, rows = {}, []
    for tie in TIES:
        built = {c: build(by[c], tie) for c in inter}
        common = sorted(set.intersection(*[set(built[c][0]) for c in inter]))
        Ys = []
        for c in inter:
            pos = {u: k for k, u in enumerate(built[c][0])}
            Ys.append(built[c][1][[pos[u] for u in common]])
        built["mean"] = (common, np.mean(Ys, axis=0))
        for c in CELLS:
            units, Y = built[c]
            idx = np.random.default_rng(SEED).integers(0, len(units), size=(BOOT, len(units)))
            for e, i in READOUTS:
                d = (Y[:, e, i, 1, 1] - Y[:, e, i, 0, 1]) - (Y[:, e, i, 1, 0] - Y[:, e, i, 0, 0])
                b = d[idx].mean(axis=1)
                est, lo, hi = 100 * float(d.mean()), *(100 * np.percentile(b, [2.5, 97.5]))
                ro = f"{EXTS[e]}|{IDS[i]}"
                key = f"{c.replace('|', '_').replace('.', '').replace('-', '')}_{EXTS[e]}_{IDS[i]}_{tie}"
                V[key], V[key + "_lo"], V[key + "_hi"] = (round(float(x), 6) for x in (est, lo, hi))
                rows.append([ro, c, tie, round(est, 2), round(float(lo), 2), round(float(hi), 2)])
    order = {(ro, c, t): k for k, (ro, c, t) in enumerate(
        (f"{EXTS[e]}|{IDS[i]}", c, t) for e, i in READOUTS for c in CELLS for t in TIES)}
    rows.sort(key=lambda r: order[(r[0], r[1], r[2])])
    with open(out / "plotted_values.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["readout", "cell", "tie", "est", "lo", "hi"])
        w.writerows(rows)
    (out / "values.json").write_text(json.dumps(dict(sorted(V.items())), indent=1) + "\n")
    print(len(rows), "points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
