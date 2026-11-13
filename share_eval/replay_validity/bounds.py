r"""Manski-type bounds for the factorial's replayed contrasts, from first-divergence events.

    python3 -I share_eval/replay_validity/bounds.py [--div DIV] [--scores SCORES] [--questions dev.json] [--out FILE]

Du-MAD and MAD-M2 have no downstream read of the answer labels (no stop rule; peers are shown as raw
text), so their replays are pathwise identified: O = R. BCCS runs carry [O_lo, O_hi] per readout from
bccs_divergence.py. A contrast's per-run interval is formed by interval arithmetic; repeats are
averaged within a task, tasks equally; the bootstrap resamples tasks (10,000 draws, seed 20261106)
and reports the 2.5% quantile of the lower bound and the 97.5% quantile of the upper bound.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "output/share_eval/replay_validity"
BIRD_DEV = REPO / "data/records/dataeng_raw/sql/dev_20240627/dev.json"
BOOT, SEED = 10000, 20261106
INTERACTIVE = ("du_mad", "madm2_objective", "bccs")

# readout name -> (factorial score key at last, at r0, divergence map)
RD = {"ref": ("strict|string|last|first", "strict|string|r0|first", "ref"),
      "I": ("strict|exec|last|first", "strict|exec|r0|first", "I"),
      "E": ("lenient|string|last|first", "lenient|string|r0|first", "E"),
      "EI": ("lenient|exec|last|first", "lenient|exec|r0|first", "EI"),
      "ref_tl": ("strict|string|last|last", "strict|string|r0|last", "ref_tie_last")}


def interval(run, div, name, rnd):
    """(lo, hi, replay) of the online outcome of readout `name` at round `rnd` ('last' | 'r0' | 'final')."""
    if rnd == "final":
        v = run["final"]
        return v, v, v
    key = RD[name][0 if rnd == "last" else 1]
    R = run[key]
    if rnd == "r0" or run["protocol"] != "bccs":
        return R, R, R                                   # the initial answers precede every read; no reads in Du-MAD/MAD-M2
    d = div[(run["unit"], run["arm"], run["backbone"], int(run["repeat"]))][RD[name][2]]
    assert d["R_last"] == R, "replay score mismatch"
    return d["O_lo"], d["O_hi"], R


# contrasts: list of (sign, readout, round, arm)
CONTRASTS = {
    "E vs ref (FULL, last)": [(1, "E", "last", "FULL"), (-1, "ref", "last", "FULL")],
    "I vs ref (FULL, last)": [(1, "I", "last", "FULL"), (-1, "ref", "last", "FULL")],
    "E+I vs ref (FULL, last)": [(1, "EI", "last", "FULL"), (-1, "ref", "last", "FULL")],
    "R: ref r0 vs ref last (FULL)": [(1, "ref", "r0", "FULL"), (-1, "ref", "last", "FULL")],
    "E+I+R vs ref (FULL)": [(1, "EI", "r0", "FULL"), (-1, "ref", "last", "FULL")],
    "A: ref NOPEER vs ref FULL (last)": [(1, "ref", "last", "NOPEER"), (-1, "ref", "last", "FULL")],
    "E+I+R non-additivity (joint - sum of singles)": [(1, "EI", "r0", "FULL"), (-1, "E", "last", "FULL"),
                                                       (-1, "I", "last", "FULL"), (-1, "ref", "r0", "FULL"),
                                                       (1, "ref", "last", "FULL"), (1, "ref", "last", "FULL")],
    "ref vs released (FULL)": [(1, "ref", "last", "FULL"), (-1, "ref", "final", "FULL")],
    "released vs E+I r0 vote (FULL)": [(1, "ref", "final", "FULL"), (-1, "EI", "r0", "FULL")],
    "Dpeer (lenient, exec)": [(1, "EI", "last", "FULL"), (-1, "EI", "r0", "FULL"),
                              (-1, "EI", "last", "NOPEER"), (1, "EI", "r0", "NOPEER")],
    "FULL last - r0 (lenient, exec)": [(1, "EI", "last", "FULL"), (-1, "EI", "r0", "FULL")],
    "Dpeer (strict, string), first tie": [(1, "ref", "last", "FULL"), (-1, "ref", "r0", "FULL"),
                                          (-1, "ref", "last", "NOPEER"), (1, "ref", "r0", "NOPEER")],
    "Dpeer (strict, string), last tie": [(1, "ref_tl", "last", "FULL"), (-1, "ref_tl", "r0", "FULL"),
                                         (-1, "ref_tl", "last", "NOPEER"), (1, "ref_tl", "r0", "NOPEER")],
}


def task_arrays(runs, div, terms):
    """Per task: (replay estimate, lower bound, upper bound), repeats averaged within (task, arm)."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        by[r["unit"]][r["arm"]].append(r)
    units = sorted(u for u, a in by.items() if {"FULL", "NOPEER"} <= set(a))
    est, lo, hi = np.zeros(len(units)), np.zeros(len(units)), np.zeros(len(units))
    for k, u in enumerate(units):
        for sgn, name, rnd, arm in terms:
            vals = [interval(r, div, name, rnd) for r in by[u][arm]]
            l_, h_, R_ = (np.mean([v[j] for v in vals]) for j in range(3))
            est[k] += sgn * R_
            if sgn > 0:
                lo[k] += l_
                hi[k] += h_
            else:
                lo[k] -= h_
                hi[k] -= l_
    return units, est, lo, hi


def summarise(est, lo, hi, idx):
    b_est = est[idx].mean(axis=1)
    b_lo, b_hi = lo[idx].mean(axis=1), hi[idx].mean(axis=1)
    f = lambda x: round(100 * float(x), 2)                               # noqa: E731
    return dict(replay=f(est.mean()), replay_ci=[f(np.percentile(b_est, 2.5)), f(np.percentile(b_est, 97.5))],
                bound=[f(lo.mean()), f(hi.mean())],
                bound_ci=[f(np.percentile(b_lo, 2.5)), f(np.percentile(b_hi, 97.5))],
                width=f(hi.mean() - lo.mean()),
                sign_identified=bool(lo.mean() > 0 or hi.mean() < 0),
                sign_identified_ci=bool(np.percentile(b_lo, 2.5) > 0 or np.percentile(b_hi, 97.5) < 0))


def db_cluster_ci(units, est, draws=BOOT, questions=BIRD_DEV):
    """Replay estimate CI resampling the 11 BIRD databases (question-equal within a draw)."""
    sys.path.insert(0, str(REPO / "share_eval"))
    from _questions import db_of  # noqa: PLC0415
    dbmap = db_of(questions)
    dbs = sorted({dbmap[u] for u in units})
    members = {d: np.array([i for i, u in enumerate(units) if dbmap[u] == d]) for d in dbs}
    rng = np.random.default_rng(SEED)
    out = np.empty(draws)
    for b in range(draws):
        pick = rng.integers(0, len(dbs), size=len(dbs))
        ix = np.concatenate([members[dbs[j]] for j in pick])
        out[b] = est[ix].mean()
    return [round(100 * float(np.percentile(out, 2.5)), 2), round(100 * float(np.percentile(out, 97.5)), 2)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--div", default=str(OUT / "bccs_divergence_runs.jsonl"))
    ap.add_argument("--scores", default=str(REPO / "output/share_eval/replay/per_run_scores.jsonl"))
    ap.add_argument("--questions", default=str(BIRD_DEV), help="BIRD dev.json, or a CSV with columns unit, db_id")
    ap.add_argument("--out", default=str(OUT / "bounds.json"))
    ap.add_argument("--backbones", default="")
    a = ap.parse_args()
    keep_bb = {b for b in a.backbones.split(",") if b}
    div = {}
    for l in pathlib.Path(a.div).read_text().splitlines():
        d = json.loads(l)
        div[(d["unit"], d["arm"], d["backbone"], int(d["repeat"]))] = d
    runs = [json.loads(l) for l in pathlib.Path(a.scores).read_text().splitlines()]
    runs = [r for r in runs if r["protocol"] in INTERACTIVE and (not keep_bb or r["backbone"] in keep_bb)]
    cells = sorted({(r["protocol"], r["backbone"]) for r in runs})
    res = {}
    for cname, terms in CONTRASTS.items():
        per = {}
        arrays = {}
        for c in cells:
            rs = [r for r in runs if (r["protocol"], r["backbone"]) == c]
            units, est, lo, hi = task_arrays(rs, div, terms)
            arrays[c] = (units, est, lo, hi)
        u0 = arrays[cells[0]][0]
        assert all(arrays[c][0] == u0 for c in cells), "cells must share one task set"
        idx = np.random.default_rng(SEED).integers(0, len(u0), size=(BOOT, len(u0)))
        for c in cells:
            per[f"{c[0]}|{c[1]}"] = summarise(*arrays[c][1:], idx)
        for label, sub in (("mean of 6", cells), ("mean of 4 (Du-MAD, MAD-M2)", [c for c in cells if c[0] != "bccs"]),
                           ("mean of 2 (BCCS)", [c for c in cells if c[0] == "bccs"])):
            E = np.mean([arrays[c][1] for c in sub], axis=0)
            L = np.mean([arrays[c][2] for c in sub], axis=0)
            H = np.mean([arrays[c][3] for c in sub], axis=0)
            per[label] = summarise(E, L, H, idx)
            if label.startswith("mean of 4"):
                per[label]["replay_db_cluster_ci"] = db_cluster_ci(u0, E, questions=a.questions)
        res[cname] = per
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    for cname, per in res.items():
        print(f"== {cname}")
        for k in [c for c in per if c.startswith("bccs|")] + ["mean of 6", "mean of 4 (Du-MAD, MAD-M2)"]:
            v = per[k]
            print(f"   {k:28s} replay {v['replay']:+7.2f} {v['replay_ci']}  bound {v['bound']} ci {v['bound_ci']} "
                  f"w {v['width']:.1f} signID {v['sign_identified']}/{v['sign_identified_ci']} {v.get('replay_db_cluster_ci','')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
