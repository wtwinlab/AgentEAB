r"""Database-cluster intervals next to the question intervals of the factorial replay (no execution).

    python3 -m share_eval_suite.steps.db_cluster_bootstrap [--out output/share_eval_suite/replay] [--questions dev.json]

Same outputs as `share_eval/replay/db_cluster_bootstrap.py`.

Reads `<out>/per_run_scores.jsonl` (tie rule first; vote string or exec). Repeats are averaged within a
question and peer condition. For every (protocol, backbone) and for the equal-weight mean of the
interactive configurations on their common questions, each contrast (main effects, selected
interactions, peer effects, round changes, the returned answer against the initial lenient
execution vote, and the shared-reference ledger) gets two 95% percentile intervals over 10,000 draws
(seed 20261106 for both): the question interval resamples questions; the database interval resamples
the BIRD development databases with replacement and averages the questions of the drawn databases (a
question counts once per draw of its database). Writes `<out>/db_cluster_bootstrap.json` and `.md` and
refreshes `<out>/SHA256SUMS`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

from share_eval_suite import stats
from share_eval_suite.comparisons import dependency
from share_eval_suite.steps import _replay_common as C

sys.dont_write_bytecode = True

BOOT, SEED = stats.DRAWS, stats.SEED_REPLAY
INTER = dependency.INTER
EXTS, IDS = dependency.EXTS, dependency.IDS
F = ("E", "I", "R", "A")
NAMES = {"du_mad": "Du-MAD", "madm2_objective": "MAD-M²", "bccs": "BCCS", "moa": "MoA"}
TWO_AND_THREE_WAY = ("ExI", "RxA", "IxR", "ExR", "IxRxA", "ExRxA")
LEDGER_SETS = [(0,), (1,), (2,), (3,), (0, 1), (1, 2), (2, 3), (0, 1, 2), (0, 1, 2, 3)]


def contrasts(Y: np.ndarray, Fin: np.ndarray):
    """Per-question contrast vectors (fractions) of a question array Y[q, E, I, R, A], Fin[q, A]."""
    n = Y.shape[0]
    flat = Y.reshape(n, 16)
    signs = np.array(list(itertools.product((-1, 1), repeat=4)))
    out = {}
    for k in range(1, 5):
        for S in itertools.combinations(range(4), k):
            name = "x".join(F[j] for j in S)
            if k == 1 or name in TWO_AND_THREE_WAY:
                out[f"effect {name}"] = flat @ (np.prod(signs[:, S], axis=1) * 2 ** k / 16.0)
    for e, i in itertools.product((0, 1), repeat=2):
        out[f"Δpeer {EXTS[e]},{IDS[i]}"] = (Y[:, e, i, 1, 1] - Y[:, e, i, 0, 1]) - (Y[:, e, i, 1, 0] - Y[:, e, i, 0, 0])
    out["last−r0 lenient,exec FULL"] = Y[:, 1, 1, 1, 1] - Y[:, 1, 1, 0, 1]
    out["last−r0 lenient,exec NOPEER"] = Y[:, 1, 1, 1, 0] - Y[:, 1, 1, 0, 0]
    out["last−r0 strict,string FULL"] = Y[:, 0, 0, 1, 1] - Y[:, 0, 0, 0, 1]
    out["final − (lenient,exec,r0,FULL)"] = Fin[:, 1] - Y[:, 1, 1, 0, 1]
    ref = [0, 0, 1, 1]
    yref = Y[:, 0, 0, 1, 1]
    single = {}
    for j in range(4):
        c = list(ref)
        c[j] = 1 - c[j]
        single[j] = Y[:, c[0], c[1], c[2], c[3]] - yref
    for S in LEDGER_SETS:
        c = list(ref)
        for j in S:
            c[j] = 1 - c[j]
        joint = Y[:, c[0], c[1], c[2], c[3]] - yref
        tag = "+".join(F[j] for j in S)
        out[f"ledger joint {tag}"] = joint
        if len(S) > 1:
            out[f"ledger non-add. {tag}"] = joint - sum(single[j] for j in S)
    return out


def ci_q(d: np.ndarray, idx: np.ndarray):
    """Question interval: percentile interval of the mean over resampled questions."""
    return stats.percentile_interval(d[idx].mean(axis=1))


def ci_db(d: np.ndarray, dbi: np.ndarray, ndb: int, idx_db: np.ndarray):
    """Database interval: per draw, the sum over the drawn databases of their question sums divided by
    the total number of their questions."""
    sums = np.bincount(dbi, weights=d, minlength=ndb)
    cnts = np.bincount(dbi, minlength=ndb).astype(float)
    b = sums[idx_db].sum(axis=1) / cnts[idx_db].sum(axis=1)
    return stats.percentile_interval(b)


def _name(k: str) -> str:
    p, b = k.split("|")
    return f"{NAMES.get(p, p)} / {b}" if b else p


def run(out=C.OUTPUT / "replay", questions=C.BIRD_DEV) -> int:
    """Write `<out>/db_cluster_bootstrap.json` and `.md` from `<out>/per_run_scores.jsonl`."""
    out = pathlib.Path(out)
    dbof = C.question_databases(questions)
    _, runs = dependency._load_runs(out)

    cells = {key: dependency._build(by_unit) for key, by_unit in sorted(runs.items())}
    common = sorted(set.intersection(*[set(cells[k][0]) for k in cells if k[0] in INTER]))
    pooledY, pooledF = [], []
    for k in cells:
        if k[0] in INTER:
            pos = {u: i for i, u in enumerate(cells[k][0])}
            sel = [pos[u] for u in common]
            pooledY.append(cells[k][1][sel])
            pooledF.append(cells[k][2][sel])
    cells[("mean of 6", "")] = (common, np.mean(pooledY, axis=0), np.mean(pooledF, axis=0))

    res = {}
    for key, (units, Y, Fin) in cells.items():
        dbs = sorted({dbof[u] for u in units})
        dbi = np.array([dbs.index(dbof[u]) for u in units])
        idx_q = stats.resample_index(len(units), BOOT, SEED)
        idx_db = stats.resample_index(len(dbs), BOOT, SEED)
        rec = dict(n_questions=len(units), n_dbs=len(dbs),
                   questions_per_db={d: int((dbi == j).sum()) for j, d in enumerate(dbs)}, terms={})
        for name, d in contrasts(Y, Fin).items():
            lq, hq = ci_q(d, idx_q)
            ld, hd = ci_db(d, dbi, len(dbs), idx_db)
            sig_q, sig_d = bool(lq > 0 or hq < 0), bool(ld > 0 or hd < 0)
            rec["terms"][name] = dict(est=round(100 * d.mean(), 2), q_lo=round(100 * lq, 2), q_hi=round(100 * hq, 2),
                                      db_lo=round(100 * ld, 2), db_hi=round(100 * hd, 2),
                                      sig_question=sig_q, sig_db=sig_d, changes=sig_q != sig_d)
        res[f"{key[0]}|{key[1]}"] = rec

    blob = dict(note="DB-cluster vs question bootstrap; tie=first, identity exec_filt; repeats averaged within question; "
                     "DB bootstrap averages the questions of the resampled DB multiset", boot=BOOT, seed=SEED, cells=res)
    (out / "db_cluster_bootstrap.json").write_text(json.dumps(blob, indent=1))

    L = ["# DB-cluster bootstrap vs question bootstrap (7–8B BIRD factorial replay)\n",
         f"{BOOT} draws, seed {SEED}. pp. `q CI` resamples questions; `DB CI` resamples the "
         f"{res['mean of 6|']['n_dbs']} dev databases with replacement. ⚠ = significance (CI excludes 0) differs.\n",
         "Questions per DB (mean-of-6 set): " + ", ".join(f"{d} {n}" for d, n in res["mean of 6|"]["questions_per_db"].items()) + "\n"]
    order = ["mean of 6|"] + [k for k in res if k != "mean of 6|"]
    for k in order:
        r = res[k]
        L.append(f"\n## {_name(k)} (n = {r['n_questions']} questions, {r['n_dbs']} DBs)\n")
        L.append("| term | est | question CI | DB-cluster CI | change |")
        L.append("|---|---|---|---|---|")
        for t, v in r["terms"].items():
            L.append(f"| {t} | {v['est']:+.2f} | [{v['q_lo']:+.2f}, {v['q_hi']:+.2f}] | [{v['db_lo']:+.2f}, {v['db_hi']:+.2f}] | "
                     f"{'⚠ ' + ('loses' if v['sig_question'] else 'gains') + ' significance' if v['changes'] else ''} |")
    (out / "db_cluster_bootstrap.md").write_text("\n".join(L) + "\n")
    C.write_checksums(out)
    print("\n".join(L[:3]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Database-cluster and question intervals of the factorial replay")
    ap.add_argument("--out", default=str(C.OUTPUT / "replay"),
                    help="directory holding per_run_scores.jsonl; results are written there (default %(default)s)")
    ap.add_argument("--questions", default=str(C.BIRD_DEV),
                    help="BIRD dev.json, or a CSV with columns unit, db_id (default %(default)s)")
    a = ap.parse_args(argv)
    return run(out=a.out, questions=a.questions)


if __name__ == "__main__":
    raise SystemExit(main())
