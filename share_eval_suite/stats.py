"""Estimators shared by the stage evaluation, the comparisons and the reports.

All bootstrap intervals are 95% percentile intervals. Every bootstrap draws from its own seeded
generator (`numpy.random.default_rng(seed)`), so a result depends only on its inputs and its seed.
Questions are resampled as units; callers average the repeats of a question before passing
per-question values.
"""
from __future__ import annotations

import collections
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta

DRAWS = 10000
SEED_REPLAY = 20261106          # extraction, voting and peer comparisons on the BIRD debates
SEED_STAGE = 20261105           # stage replacements


# ------------------------------------------------------------------ resampling indices
def resample_index(n: int, draws: int = DRAWS, seed: int = SEED_REPLAY) -> np.ndarray:
    """A (draws × n) matrix of indices drawn with replacement from range(n).

    One matrix is shared by every contrast on the same questions, so contrasts are resampled jointly.
    """
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def percentile_interval(boot: np.ndarray) -> Tuple[float, float]:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_p(boot: np.ndarray, floor: Optional[int] = None) -> float:
    """Two-sided bootstrap p-value: twice the smaller tail share at 0, capped at 1.

    `floor` (the number of draws) bounds the value below by 1/floor, as the replay analyses report it.
    """
    p = min(1.0, 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean())))
    return max(1.0 / floor, p) if floor else p


def mean_interval(d: np.ndarray, idx: np.ndarray) -> Tuple[float, float, float]:
    """Mean of per-question values `d` and its percentile interval under the index matrix `idx`."""
    b = d[idx].mean(axis=1)
    lo, hi = percentile_interval(b)
    return float(d.mean()), lo, hi


# ------------------------------------------------------------------ cluster and stratified bootstraps
def cluster_bootstrap(values: Mapping[Hashable, float], clusters: Mapping[Hashable, Hashable],
                      weighting: str = "unit", draws: int = DRAWS, seed: int = SEED_STAGE,
                      units: Optional[Iterable[Hashable]] = None) -> Dict[str, float]:
    """Paired cluster bootstrap over clusters (e.g. databases).

    Each cluster contributes the mean of its units. `weighting="unit"` weights clusters by their number
    of units (every question counts once); `"cluster_equal"` weights clusters equally.
    Returns point, lo, hi, p_boot, units, clusters.
    """
    units = sorted(values) if units is None else sorted(units)
    by = collections.defaultdict(list)
    for u in units:
        by[clusters[u]].append(values[u])
    cl = sorted(by)
    per = np.array([np.mean(by[c]) for c in cl])
    n = np.array([len(by[c]) for c in cl], float)
    w = np.ones(len(cl)) if weighting == "cluster_equal" else n
    point = float((w * per).sum() / w.sum())
    d = np.random.default_rng(seed).integers(0, len(cl), size=(draws, len(cl)))
    boot = (per[d] * w[d]).sum(1) / w[d].sum(1)
    lo, hi = percentile_interval(boot)
    return dict(point=point, lo=lo, hi=hi, p_boot=bootstrap_p(boot), units=len(units), clusters=len(cl))


def stratified_bootstrap(groups: Mapping[Hashable, Sequence[float]], draws: int = DRAWS,
                         seed: int = SEED_STAGE) -> Optional[Dict[str, float]]:
    """Units resampled within each stratum (strata fixed, e.g. datasets); pooled mean with unit weights.

    Strata are visited in sorted order and empty strata are skipped.
    """
    rng = np.random.default_rng(seed)
    g = {k: np.asarray(v, float) for k, v in sorted(groups.items()) if len(v)}
    if not g:
        return None
    n = sum(len(v) for v in g.values())
    tot = np.zeros(draws)
    for v in g.values():
        tot += v[rng.integers(0, len(v), size=(draws, len(v)))].sum(1)
    boot = tot / n
    lo, hi = percentile_interval(boot)
    return dict(point=float(sum(v.sum() for v in g.values()) / n), lo=lo, hi=hi, units=n, clusters=len(g))


def stratified_cluster_bootstrap(groups: Mapping[Hashable, Sequence[Sequence[float]]], draws: int = DRAWS,
                                 seed: int = SEED_STAGE) -> Optional[Dict[str, float]]:
    """Clusters (e.g. anchor records with their pairs) resampled within each stratum; pooled unit mean."""
    rng = np.random.default_rng(seed)
    tot = np.zeros(draws)
    cnt = np.zeros(draws)
    s = c = 0.0
    for k in sorted(groups):
        arrays = [np.asarray(a, float) for a in groups[k] if len(a)]
        if not arrays:
            continue
        sums = np.array([a.sum() for a in arrays])
        lens = np.array([len(a) for a in arrays], float)
        d = rng.integers(0, len(arrays), size=(draws, len(arrays)))
        tot += sums[d].sum(1)
        cnt += lens[d].sum(1)
        s += sums.sum()
        c += lens.sum()
    if not c:
        return None
    lo, hi = percentile_interval(tot / cnt)
    return dict(point=float(s / c), lo=lo, hi=hi, units=int(c), clusters=None)


# ------------------------------------------------------------------ proportions and multiple testing
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact binomial interval for k successes in n trials."""
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def holm(ps: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm step-down rejections, in the order of `ps`."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            out[i] = True
        else:
            break
    return out


def excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0
