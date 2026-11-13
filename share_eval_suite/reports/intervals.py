"""Confidence intervals: percentile bootstraps over questions, databases and dataset strata.

All intervals are 95% percentile intervals unless stated otherwise. A question bootstrap resamples
questions; a database bootstrap resamples whole databases (BIRD, Spider) with every question of a
drawn database carried along, either question-weighted or with each database weighted equally; a
stratified bootstrap keeps each dataset fixed and resamples its tasks (the entity-matching rows).
Estimates and interval ends are fractions, not percentages, unless a function says otherwise.
"""
from __future__ import annotations

import collections
import random
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Tuple

import numpy as np

from share_eval_suite import stats

DRAWS = stats.DRAWS
SEED = stats.SEED_STAGE          # the seed of the stage decomposition and the replacement table


@dataclass(frozen=True)
class Interval:
    """An estimate with its 95% interval."""
    estimate: float
    lo: float
    hi: float

    def scaled(self, k: float) -> "Interval":
        return Interval(k * self.estimate, k * self.lo, k * self.hi)

    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0


@dataclass(frozen=True)
class TwoIntervals:
    """An estimate with a question interval and a database interval (pp)."""
    estimate: float
    question: Interval
    database: Interval

    def detected(self) -> bool:
        """Both intervals exclude zero on the same side."""
        q, d = self.question, self.database
        return (q.lo > 0 and d.lo > 0) or (q.hi < 0 and d.hi < 0)


def cluster_bootstrap(values: Dict[Hashable, float], clusters: Dict[Hashable, str],
                      weighting: str = "question", draws: int = DRAWS, seed: int = SEED) -> Interval:
    """Paired database bootstrap of a mean over tasks.

    `values` maps each task to its value (e.g. stage minus rule, in {-1, 0, 1}); `clusters` maps each
    task to its database. Databases are drawn with replacement; `weighting` is "question" (every
    question weighs the same) or "database" (every database's mean weighs the same).
    """
    w = {"question": "unit", "database": "cluster_equal"}[weighting]
    r = stats.cluster_bootstrap(values, clusters, w, draws, seed)
    return Interval(r["point"], r["lo"], r["hi"])


def stratified_bootstrap(values: Dict[Hashable, float], strata: Dict[Hashable, str]) -> Interval:
    """Stratified bootstrap with each dataset held fixed: tasks are resampled within their dataset and
    the pooled mean weighs every task the same (10,000 draws, seed 20261105).

    Tasks enter each stratum in the order of `values`.
    """
    groups: Dict[str, list] = {}
    for k, v in values.items():
        groups.setdefault(strata[k], []).append(v)
    r = stats.stratified_bootstrap(groups, DRAWS, SEED)
    return Interval(r["point"], r["lo"], r["hi"])


def _database_resample(items: Iterable[Tuple[str, float]], draws: int, seed: int) -> Tuple[float, float, float]:
    """Question-weighted mean and the order-statistic interval of a database resample drawn with
    Python's `random.Random(seed)`: each draw picks as many databases as there are, in sorted order of
    database names, and every value of a drawn database enters; the interval ends are the
    int(0.025 m)-th and (int(0.975 m) - 1)-th of the m sorted draw means."""
    by: Dict[str, list] = collections.defaultdict(list)
    for db, y in items:
        by[db].append(y)
    keys = sorted(by)
    n = len(keys)
    tot = sum(len(v) for v in by.values())
    point = sum(sum(v) for v in by.values()) / tot
    rnd = random.Random(seed)
    sims: List[float] = []
    for _ in range(draws):
        s = c = 0.0
        for _ in range(n):
            k = keys[rnd.randrange(n)]
            s += sum(by[k])
            c += len(by[k])
        if c:
            sims.append(s / c)
    sims.sort()
    return point, sims[int(0.025 * len(sims))], sims[int(0.975 * len(sims)) - 1]


def database_resample_bootstrap(items: Iterable[Tuple[str, float]], draws: int = 5000,
                                seed: int = 20261026) -> Interval:
    """Question-weighted mean over (database, value) pairs; the interval resamples databases with
    Python's `random.Random(seed)`, databases in sorted order; the interval ends are the int(0.025 m)-th
    and (int(0.975 m) - 1)-th of the m sorted draw means (the convention of the execution-filter comparisons on fixed SQL pools)."""
    return Interval(*_database_resample(list(items), draws, seed))


_SCALE_DRAWS, _SCALE_SEED = 10000, stats.SEED_REPLAY


def _pp2(v) -> float:
    """A fraction in pp, rounded to two decimals."""
    return round(100 * float(v), 2)


def _question_database(values: Dict[Hashable, float], databases: Dict[Hashable, str]) -> dict:
    """Estimate, question interval, database interval (pp, two decimals) and the question p-value.

    Questions are taken in sorted order. The question bootstrap draws a (10,000 x n) index matrix from
    `default_rng(20261106)`; the database bootstrap draws, from a second `default_rng(20261106)`, one
    vector of database indices per draw (databases in sorted order) and averages every question of
    the drawn databases. p = 2 min(share of question draws <= 0, share >= 0), floored at 1/10,000 and
    rounded to five decimals.
    """
    us = sorted(values)
    x = np.array([values[u] for u in us])
    t = x[stats.resample_index(len(x), _SCALE_DRAWS, _SCALE_SEED)].mean(axis=1)
    g = collections.defaultdict(list)
    for i, u in enumerate(us):
        g[databases[u]].append(i)
    keys = sorted(g)
    rng = np.random.default_rng(_SCALE_SEED)
    c = np.array([x[[i for k in rng.integers(0, len(keys), size=len(keys)) for i in g[keys[k]]]].mean()
                  for _ in range(_SCALE_DRAWS)])
    p = stats.bootstrap_p(t, floor=_SCALE_DRAWS)
    return dict(n=len(us), est=_pp2(x.mean()), q_lo=_pp2(np.percentile(t, 2.5)), q_hi=_pp2(np.percentile(t, 97.5)),
                db_lo=_pp2(np.percentile(c, 2.5)), db_hi=_pp2(np.percentile(c, 97.5)), p=round(float(p), 5))


def question_and_database_bootstrap(values: Dict[Hashable, float],
                                    databases: Dict[Hashable, str]) -> Tuple[TwoIntervals, float]:
    """Question interval and database interval of a per-question mean, each from its own generator
    (10,000 draws, seed 20261106), in pp rounded to two decimals as in the backbone-scale table;
    returns the intervals and the two-sided bootstrap p-value over questions (floored at 1/10,000)."""
    s = _question_database(values, databases)
    return (TwoIntervals(s["est"], Interval(s["est"], s["q_lo"], s["q_hi"]),
                         Interval(s["est"], s["db_lo"], s["db_hi"])), s["p"])
