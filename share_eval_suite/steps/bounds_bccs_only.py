r"""Deployment bounds of BCCS debates from their first-divergence records alone (debates without replay
scores, such as the Qwen2.5-14B BCCS debates). Same estimands and bootstrap as `steps.bounds`; each
backbone is summarised on its own and never pooled with another.

    python3 -m share_eval_suite.steps.bounds_bccs_only --div output/share_eval_suite/replay_validity_backbones/bccs_divergence_runs.jsonl \
            --out output/share_eval_suite/replay_validity_backbones/bounds_bccs.json

Same outputs as `share_eval/replay_validity/bounds_bccs_only.py`.

A term at the initial answers takes the replay value of the run (R_r0, identified: the initial
answers precede every read); a term at the last round takes the run's bound [O_lo, O_hi] with replay
R_last. Repeats are averaged within a question and peer condition; questions with runs in both peer
conditions count; intervals resample questions (10,000 draws, seed 20261106, one index matrix per
backbone). Writes one JSON file (`--out`) and prints it.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

from share_eval_suite.steps.bounds import BOOT, SEED, summarise
from share_eval_suite import stats

sys.dont_write_bytecode = True

TERMS = {
    "Dpeer (lenient, exec)": [(1, "EI", "last", "FULL"), (-1, "EI", "r0", "FULL"), (-1, "EI", "last", "NOPEER"), (1, "EI", "r0", "NOPEER")],
    "FULL last - r0 (lenient, exec)": [(1, "EI", "last", "FULL"), (-1, "EI", "r0", "FULL")],
    "E+I vs ref (FULL, last)": [(1, "EI", "last", "FULL"), (-1, "ref", "last", "FULL")],
    "Dpeer (strict, string), first tie": [(1, "ref", "last", "FULL"), (-1, "ref", "r0", "FULL"), (-1, "ref", "last", "NOPEER"), (1, "ref", "r0", "NOPEER")],
}


def iv(x, mp, rnd):
    """(lower, upper, replay) of a run's outcome under reading `mp` at round `rnd`."""
    d = x[mp]
    return (d["R_r0"],) * 3 if rnd == "r0" else (d["O_lo"], d["O_hi"], d["R_last"])


def run(div, out) -> int:
    """Write the BCCS-only bounds of every backbone in `div` to `out`."""
    runs = [json.loads(line) for line in pathlib.Path(div).read_text().splitlines()]
    res = {}
    for bb in sorted({r["backbone"] for r in runs}):
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in runs:
            if r["backbone"] == bb:
                by[r["unit"]][r["arm"]].append(r)
        units = sorted(u for u, v in by.items() if {"FULL", "NOPEER"} <= set(v))
        idx = stats.resample_index(len(units), BOOT, SEED)
        res[bb] = {"n_tasks": len(units)}
        for name, terms in TERMS.items():
            E, L, H = (np.zeros(len(units)) for _ in range(3))
            for k, u in enumerate(units):
                for sg, mp, rnd, arm in terms:
                    vals = [iv(x, mp, rnd) for x in by[u][arm]]
                    l_, h_, R_ = (np.mean([v[j] for v in vals]) for j in range(3))
                    E[k] += sg * R_
                    L[k] += l_ if sg > 0 else -h_
                    H[k] += h_ if sg > 0 else -l_
            res[bb][name] = summarise(E, L, H, idx)
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="BCCS-only bounds of Lemma 1")
    ap.add_argument("--div", required=True, help="bccs_divergence_runs.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return run(a.div, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
