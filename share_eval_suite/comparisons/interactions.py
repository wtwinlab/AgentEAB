from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

import numpy as np

from share_eval_suite import stats
from share_eval_suite.comparisons import dependency

FACTORIAL = "tab_factorial"
FACTORS = ("E", "I", "R", "A")
CELL_SETS = {"six": ("du_mad", "madm2_objective", "bccs"), "four": ("du_mad", "madm2_objective")}
REF = (0, 0, 1, 1)
DRAWS, SEED = stats.DRAWS, stats.SEED_REPLAY


def outcome_array(cells: str = "six", tie: str = "first", inputs=None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    _, by = dependency._load_runs(dependency._dir(FACTORIAL, inputs))
    chosen = sorted(c for c in by if c[0] in CELL_SETS[cells])
    return dependency._pooled(by, chosen, tie)


def factorial_effects(Y: np.ndarray) -> Dict[str, Dict[str, float]]:
    n = Y.shape[0]
    idx = stats.resample_index(n, DRAWS, SEED)
    flat = Y.reshape(n, 16)
    signs = np.array(list(itertools.product((-1, 1), repeat=4)))
    res = {}
    for k in range(1, 5):
        for S in itertools.combinations(range(4), k):
            w = np.prod(signs[:, S], axis=1) * (2 ** k) / 16.0
            d = flat @ w
            b = d[idx].mean(axis=1)
            lo, hi = np.percentile(b, [2.5, 97.5])
            p = max(1.0 / DRAWS, min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))
            res["x".join(FACTORS[j] for j in S)] = dict(est=round(100 * float(d.mean()), 2),
                                                       lo=round(100 * float(lo), 2),
                                                       hi=round(100 * float(hi), 2), p=round(float(p), 4))
    return res


def _ledger(Y: np.ndarray):
    yref = Y[:, REF[0], REF[1], REF[2], REF[3]]
    single = {}
    for j in range(4):
        c = list(REF)
        c[j] = 1 - c[j]
        single[j] = Y[:, c[0], c[1], c[2], c[3]] - yref
    out = {}
    for k in range(1, 5):
        for S in itertools.combinations(range(4), k):
            c = list(REF)
            for j in S:
                c[j] = 1 - c[j]
            joint = Y[:, c[0], c[1], c[2], c[3]] - yref
            out["+".join(FACTORS[j] for j in S)] = (joint, sum(single[j] for j in S))
    return yref, out


def shared_reference_ledger(cells: str = "six", tie: str = "first", inputs=None) -> Dict[str, Any]:
    units, Y, _ = outcome_array(cells, tie, inputs)
    idx = stats.resample_index(len(units), DRAWS, SEED)
    yref, led = _ledger(Y)
    rows = {}
    for name, (joint, ssum) in led.items():
        j, d = dependency._est_ci(joint, idx), dependency._est_ci(joint - ssum, idx)
        rows[name] = dict(joint=j[0], joint_lo=j[1], joint_hi=j[2], sum_of_singles=100 * float(ssum.mean()),
                          joint_minus_sum=d[0], joint_minus_sum_lo=d[1], joint_minus_sum_hi=d[2])
    return dict(reference_level=100 * float(yref.mean()), n_questions=len(units), rows=rows)


def replay_ledger(group: str = "mean4", inputs=None) -> Dict[str, Dict[str, float]]:
    names = {"E": "E", "I": "I", "E+I": "EI", "E+I joint minus sum": "nonadd_EI",
             "E+I+R joint minus sum": "nonadd_EIR"}
    return {k: dependency.replay_bound(c, group, inputs) for k, c in names.items()}
