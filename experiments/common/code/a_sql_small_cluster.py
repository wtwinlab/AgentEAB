r"""A. SQL small-cluster sensitivity on the accepted unit outcomes. No model call.

    python3 code/a_sql_small_cluster.py

BIRD has 11 databases and Spider 20. A cluster-robust normal interval under-covers at that cluster
count, so each accepted paired contrast gets a restricted wild cluster bootstrap-t beside its
existing inference. The existing inference is not replaced.

Both accepted weightings are preserved and never mixed: the core Qwen2.5 comparison is
question-weighted, the cross-family comparison is equal-database-weighted. Repeats of a question are
averaged before the question enters any estimator.

The bootstrap is Cameron-Gelbach-Miller: the paired difference regressed on an intercept, which is
numerically the weighted paired mean. The equivalence is checked in code, not assumed. Rademacher
cluster weights with the null imposed by recentring; for G <= 12 clusters all 2^G sign assignments
are enumerated, which makes the p-value exact for that weight distribution -- not randomisation
inference, because sign-flipping databases is not how these data were generated.
"""
from __future__ import annotations

import collections
import csv
import itertools
import json
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]
UNITS = REPO / "data" / "derived" / "sql_decomposition" / "per_execution.csv"
OUT = WORK / "output"

#: Each row carries both outcomes for the same question, so the contrast is paired inside the row.
#: `arm` encodes the pool context: pool4 is observed under S2 and pool6 under S3, verified by
#: reproducing the published `net_pp` exactly before any sensitivity is computed.
ARM_FOR_POOL = {"pool4": "S2", "pool6": "S3"}
CONTRAST = "Generation - Selection"
DRAWS = 9999
SEED = 20261031
ENUMERATE_MAX_G = 12


def load_units() -> list:
    rows = []
    with open(UNITS) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def paired_per_question(rows: list) -> tuple:
    """Per-question paired difference, repeats averaged before the question enters an estimator."""
    acc = collections.defaultdict(list)
    meta = {}
    skipped = 0
    for r in rows:
        sc, gc = r["selection_correct"], r["generation_correct"]
        if sc in ("", "None", None) or gc in ("", "None", None):
            skipped += 1
            continue
        acc[r["question_id"]].append(float(int(gc)) - float(int(sc)))
        meta[r["question_id"]] = r["db_id"]
    return {k: (float(np.mean(v)), meta[k]) for k, v in acc.items()}, skipped


def weighted_mean_and_clusters(diffs: dict, weighting: str):
    """Return (effect, per-cluster arrays, per-cluster weights) under the stated weighting."""
    by_db = collections.defaultdict(list)
    for q, (d, db) in diffs.items():
        by_db[db].append(d)
    dbs = sorted(by_db)
    arrs = [np.array(by_db[db], dtype=float) for db in dbs]
    if weighting == "question":
        # every question counts once: cluster weight proportional to its question count
        w = np.array([len(a) for a in arrs], dtype=float)
    elif weighting == "equal_database":
        w = np.ones(len(arrs), dtype=float)
    else:
        raise ValueError(weighting)
    means = np.array([a.mean() for a in arrs])
    eff = float(np.sum(w * means) / np.sum(w))
    return eff, dbs, arrs, w, means


def wild_cluster_t(arrs, w, means, draws: int, rng):
    """Restricted wild cluster bootstrap-t for the weighted mean being zero.

    The intercept-only weighted regression of the paired difference has coefficient equal to the
    weighted mean, so the Wald statistic uses the cluster-robust variance of that weighted mean.
    Imposing the null means recentring the cluster means on zero, i.e. the restricted residual of
    cluster g is its own mean.
    """
    W = float(np.sum(w))
    eff = float(np.sum(w * means) / W)
    G = len(arrs)

    def stat(cluster_means):
        e = float(np.sum(w * cluster_means) / W)
        # cluster-robust variance of a weighted mean of cluster means
        v = float(np.sum((w ** 2) * ((cluster_means - e) ** 2))) / (W ** 2)
        if v <= 0:
            return None, e
        return e / np.sqrt(v), e

    t_obs, _ = stat(means)
    if t_obs is None:
        return dict(status="undefined_zero_variance", clusters=G, effect=round(100 * eff, 4))

    restricted = means.copy()                      # null imposed: residual = own mean
    ts = []
    if G <= ENUMERATE_MAX_G:
        signs = list(itertools.product((-1.0, 1.0), repeat=G))
        exact = True
    else:
        signs = rng.choice((-1.0, 1.0), size=(draws, G))
        exact = False
    for s in signs:
        s = np.asarray(s, dtype=float)
        t_b, _ = stat(restricted * s)
        if t_b is not None:
            ts.append(abs(t_b))
    ts = np.array(ts)
    p = float((ts >= abs(t_obs)).mean())
    # percentile-t interval from the same draws, inverted around the observed effect
    se = float(np.sqrt(np.sum((w ** 2) * ((means - eff) ** 2))) / W)
    q = float(np.percentile(ts, 95.0))
    return dict(status="ok", clusters=G, effect_pp=round(100 * eff, 4),
                t_obs=round(float(t_obs), 4), p_value=round(p, 5),
                draws_used=len(ts), weight_space_enumerated=exact,
                cluster_robust_se_pp=round(100 * se, 4),
                sym_t_crit_95=round(q, 4),
                ci_lo_pp=round(100 * (eff - q * se), 4),
                ci_hi_pp=round(100 * (eff + q * se), 4))


def main() -> int:
    rows = load_units()
    cells, checks = [], []
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["benchmark"], r["pool"], r["recipient"])].append(r)
    print(f"unit rows={len(rows)} cells={len(groups)}")

    for (bench, pool, recip), sub in sorted(groups.items()):
        arm = ARM_FOR_POOL.get(pool)
        sub = [r for r in sub if r["arm"] == arm]
        if not sub:
            continue
        diffs, skipped = paired_per_question(sub)
        if not diffs:
            continue
        common, dropped_q = sorted(diffs), skipped
        if True:
            treat, base = "Generation", "Selection"
            for weighting in ("question", "equal_database"):
                eff, dbs, arrs, w, means = weighted_mean_and_clusters(diffs, weighting)
                # equivalence check: intercept-only weighted least squares == weighted mean
                X = np.ones((len(means), 1))
                wls = float(np.linalg.lstsq(X * np.sqrt(w)[:, None],
                                            means * np.sqrt(w), rcond=None)[0][0])
                checks.append(dict(cell=f"{bench}|{pool}|{recip}|{treat}-{base}",
                                   weighting=weighting,
                                   weighted_mean=round(eff, 10), wls_intercept=round(wls, 10),
                                   agree=abs(eff - wls) < 1e-9))
                rng = np.random.default_rng(SEED)
                res = wild_cluster_t(arrs, w, means, DRAWS, rng)
                # most influential database: leave-one-out shift in the effect
                loo = []
                for i, db in enumerate(dbs):
                    keep = [j for j in range(len(dbs)) if j != i]
                    if not keep:
                        continue
                    wk = w[keep]
                    e2 = float(np.sum(wk * means[keep]) / np.sum(wk))
                    loo.append((abs(e2 - eff), db, e2))
                loo.sort(reverse=True)
                cells.append({
                    "benchmark": bench, "pool": pool, "recipient": recip,
                    "contrast": f"{treat} - {base}", "weighting": weighting,
                    "questions": len(common), "questions_unmatched_dropped": dropped_q,
                    "databases": len(dbs), **res,
                    "most_influential_db": loo[0][1] if loo else None,
                    "effect_pp_without_that_db": round(100 * loo[0][2], 4) if loo else None,
                    "influence_pp": round(100 * loo[0][0], 4) if loo else None,
                })

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "a_sql_small_cluster.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(cells[0]))
        w_.writeheader()
        w_.writerows(cells)
    bad = [c for c in checks if not c["agree"]]
    (OUT / "a_equivalence_check.json").write_text(
        json.dumps(dict(checks=len(checks), disagreements=bad,
                        note="intercept-only weighted least squares must equal the weighted paired "
                             "mean; the bootstrap-t is built on that formulation"),
                   indent=1) + "\n", encoding="utf-8")
    print(f"cells={len(cells)}  equivalence checks={len(checks)}  disagreements={len(bad)}")
    und = [c for c in cells if c.get("status") != "ok"]
    print(f"undefined cells: {len(und)}")
    enum = sum(1 for c in cells if c.get("weight_space_enumerated"))
    print(f"cells with the full 2^G weight space enumerated: {enum}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
