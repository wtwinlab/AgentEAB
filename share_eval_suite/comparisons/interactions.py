"""Joint stage choices on fixed BIRD debate responses.

Four two-level factors are read from the same logged responses: extractor (E: protocol | lenient), vote
(I: exact-match | execution), round (R: initial answers | last round) and peers (A: without | with). The
reference reading is the protocol extractor, the exact-match vote, the last round, with peers.

- `factorial_effects`: main and interaction effects with the difference-of-differences convention
  (a two-way effect is the effect of one factor at level 1 of the other minus at level 0, averaged over
  the remaining factors).
- `shared_reference_ledger`: from the reference reading, the change of switching one factor, the joint
  change of switching several, the sum of their single changes, and joint minus sum.
- `replay_ledger`: the same ledger rows as replay contrasts with database intervals and deployment
  bounds (see `dependency.replay_bounds`).

Estimates are in percentage points; question intervals resample the 198 questions (10,000 draws,
seed 20261106; repeats averaged within a question).
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

import numpy as np

from share_eval_suite import stats
from share_eval_suite.comparisons import dependency

FACTORIAL = "tab_factorial"
FACTORS = ("E", "I", "R", "A")
CELL_SETS = {"six": ("du_mad", "madm2_objective", "bccs"), "four": ("du_mad", "madm2_objective")}
REF = (0, 0, 1, 1)                      # protocol extractor, exact-match vote, last round, with peers
DRAWS, SEED = stats.DRAWS, stats.SEED_REPLAY


def outcome_array(cells: str = "six", tie: str = "first", inputs=None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Correctness per question under every reading, averaged over repeats and configurations.

    `cells` names the configurations: 'six' (Du-MAD, MAD-M2, BCCS with Llama-3-8B and Qwen2.5-7B; the
    replay group 'mean6') or 'four' (without BCCS; 'mean4', or 'both' in `peers`).
    Returns (questions, Y, F): Y[q, E, I, R, A] in [0, 1], F[q, A] the protocol's returned answer."""
    _, by = dependency._load_runs(dependency._dir(FACTORIAL, inputs))
    chosen = sorted(c for c in by if c[0] in CELL_SETS[cells])
    return dependency._pooled(by, chosen, tie)


def factorial_effects(Y: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Main and interaction effects of extractor, vote, round and peers over a question array Y.

    Keys are 'E', 'I', 'R', 'A', 'ExI', ..., 'ExIxRxA'; each value has `est`, `lo`, `hi` (pp, two
    decimals) and a two-sided bootstrap `p` (four decimals, floor 1/10,000). The effect of a factor set
    S is the contrast of the 16 cells with weights prod_{j in S} sign_j * 2^|S| / 16; intervals resample
    the questions (10,000 draws, seed 20261106, one index matrix shared by every effect)."""
    n = Y.shape[0]
    idx = stats.resample_index(n, DRAWS, SEED)
    flat = Y.reshape(n, 16)
    signs = np.array(list(itertools.product((-1, 1), repeat=4)))   # E, I, R, A in the order of the reshape
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
    """Reference level per question and, per switched factor set, (joint change, sum of single changes)."""
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
    """Shared-reference ledger over the pooled configurations.

    Returns `reference_level` (% correct under the reference reading), `n_questions` and `rows`: for each
    switched set ('E', 'I', 'R', 'A', 'E+I', ..., 'E+I+R+A') the joint change (`joint`, `joint_lo`,
    `joint_hi`), the sum of single changes (`sum_of_singles`) and joint minus sum (`joint_minus_sum`,
    with `_lo`, `_hi`)."""
    units, Y, _ = outcome_array(cells, tie, inputs)
    idx = stats.resample_index(len(units), DRAWS, SEED)       # shared by every ledger row
    yref, led = _ledger(Y)
    rows = {}
    for name, (joint, ssum) in led.items():
        j, d = dependency._est_ci(joint, idx), dependency._est_ci(joint - ssum, idx)
        rows[name] = dict(joint=j[0], joint_lo=j[1], joint_hi=j[2], sum_of_singles=100 * float(ssum.mean()),
                          joint_minus_sum=d[0], joint_minus_sum_lo=d[1], joint_minus_sum_hi=d[2])
    return dict(reference_level=100 * float(yref.mean()), n_questions=len(units), rows=rows)


def replay_ledger(group: str = "mean4", inputs=None) -> Dict[str, Dict[str, float]]:
    """Ledger rows as replay contrasts for one group of configurations ('mean4', 'mean6' or
    'protocol|backbone'): 'E' (lenient extractor), 'I' (execution vote), 'E+I' (both), 'E+I joint minus
    sum' and 'E+I+R joint minus sum' (also the initial answers)."""
    names = {"E": "E", "I": "I", "E+I": "EI", "E+I joint minus sum": "nonadd_EI",
             "E+I+R joint minus sum": "nonadd_EIR"}
    return {k: dependency.replay_bound(c, group, inputs) for k, c in names.items()}
