r"""Database-cluster bootstrap next to the question-level bootstrap (zero model calls, no execution).

    python3 -I share_eval/replay/db_cluster_bootstrap.py [--out output/share_eval/replay] [--questions dev.json]

Reads <out>/per_run_scores.jsonl (written by factorial_replay.py; tie = first, identity in {string, exec}).
Repeats are averaged within a question. Question bootstrap: resample questions. DB-cluster bootstrap:
resample the BIRD dev databases with replacement and average the questions of the resampled set
(a question weighs once per draw of its database). 10,000 draws, seed 20261106 for both.
Writes <out>/db_cluster_bootstrap.{json,md} and refreshes <out>/SHA256SUMS.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIRD_DEV = REPO / "data/records/dataeng_raw/sql/dev_20240627/dev.json"
BOOT, SEED = 10000, 20261106
INTER = ("du_mad", "madm2_objective", "bccs")
EXTS, IDS, RDS, ARMS = ("strict", "lenient"), ("string", "exec"), ("r0", "last"), ("NOPEER", "FULL")
F = ("E", "I", "R", "A")
NAMES = {"du_mad": "Du-MAD", "madm2_objective": "MAD-M²", "bccs": "BCCS", "moa": "MoA"}


def contrasts(Y, Fin):
    """Per-question contrast vectors (pp-ready fractions)."""
    n = Y.shape[0]
    flat = Y.reshape(n, 16)
    signs = np.array(list(itertools.product((-1, 1), repeat=4)))
    out = {}
    for k in range(1, 5):
        for S in itertools.combinations(range(4), k):
            name = "x".join(F[j] for j in S)
            if k == 1 or name in ("ExI", "RxA", "IxR", "ExR", "IxRxA", "ExRxA"):
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
        c = list(ref); c[j] = 1 - c[j]
        single[j] = Y[:, c[0], c[1], c[2], c[3]] - yref
    for S in [(0,), (1,), (2,), (3,), (0, 1), (1, 2), (2, 3), (0, 1, 2), (0, 1, 2, 3)]:
        c = list(ref)
        for j in S:
            c[j] = 1 - c[j]
        joint = Y[:, c[0], c[1], c[2], c[3]] - yref
        tag = "+".join(F[j] for j in S)
        out[f"ledger joint {tag}"] = joint
        if len(S) > 1:
            out[f"ledger non-add. {tag}"] = joint - sum(single[j] for j in S)
    return out


def ci_q(d, idx):
    b = d[idx].mean(axis=1)
    return np.percentile(b, [2.5, 97.5])


def ci_db(d, dbi, ndb, idx_db):
    sums = np.bincount(dbi, weights=d, minlength=ndb)
    cnts = np.bincount(dbi, minlength=ndb).astype(float)
    b = sums[idx_db].sum(axis=1) / cnts[idx_db].sum(axis=1)
    return np.percentile(b, [2.5, 97.5])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "output/share_eval/replay"),
                    help="directory holding per_run_scores.jsonl; results are written there (default %(default)s)")
    ap.add_argument("--questions", default=str(BIRD_DEV),
                    help="BIRD dev.json, or a CSV with columns unit, db_id (default %(default)s)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    sys.path.insert(0, str(REPO / "share_eval"))
    from _questions import db_of  # noqa: PLC0415
    dbof = db_of(a.questions)

    runs = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in (out / "per_run_scores.jsonl").read_text().splitlines():
        r = json.loads(l)
        runs[(r["protocol"], r["backbone"])][r["unit"]].append(r)

    def build(by_unit):
        units = sorted(u for u, rs in by_unit.items() if {x["arm"] for x in rs} >= set(ARMS))
        Y = np.zeros((len(units), 2, 2, 2, 2))
        Fin = np.zeros((len(units), 2))
        for ui, u in enumerate(units):
            for a, arm in enumerate(ARMS):
                rs = [x for x in by_unit[u] if x["arm"] == arm]
                for e, i, rd in itertools.product((0, 1), repeat=3):
                    Y[ui, e, i, rd, a] = np.mean([x[f"{EXTS[e]}|{IDS[i]}|{RDS[rd]}|first"] for x in rs])
                Fin[ui, a] = np.mean([float(x["final"]) for x in rs])
        return units, Y, Fin

    cells = {}
    for key, by_unit in sorted(runs.items()):
        cells[key] = build(by_unit)
    common = sorted(set.intersection(*[set(cells[k][0]) for k in cells if k[0] in INTER]))
    pooledY, pooledF = [], []
    for k in cells:
        if k[0] in INTER:
            pos = {u: i for i, u in enumerate(cells[k][0])}
            sel = [pos[u] for u in common]
            pooledY.append(cells[k][1][sel]); pooledF.append(cells[k][2][sel])
    cells[("mean of 6", "")] = (common, np.mean(pooledY, axis=0), np.mean(pooledF, axis=0))

    res = {}
    for key, (units, Y, Fin) in cells.items():
        dbs = sorted({dbof[u] for u in units})
        dbi = np.array([dbs.index(dbof[u]) for u in units])
        rng = np.random.default_rng(SEED)
        idx_q = rng.integers(0, len(units), size=(BOOT, len(units)))
        idx_db = np.random.default_rng(SEED).integers(0, len(dbs), size=(BOOT, len(dbs)))
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

    def nm(k):
        p, b = k.split("|")
        return f"{NAMES.get(p, p)} / {b}" if b else p
    L = ["# DB-cluster bootstrap vs question bootstrap (7–8B BIRD factorial replay)\n",
         f"{BOOT} draws, seed {SEED}. pp. `q CI` resamples questions; `DB CI` resamples the "
         f"{res['mean of 6|']['n_dbs']} dev databases with replacement. ⚠ = significance (CI excludes 0) differs.\n",
         "Questions per DB (mean-of-6 set): " + ", ".join(f"{d} {n}" for d, n in res["mean of 6|"]["questions_per_db"].items()) + "\n"]
    order = ["mean of 6|"] + [k for k in res if k != "mean of 6|"]
    for k in order:
        r = res[k]
        L.append(f"\n## {nm(k)} (n = {r['n_questions']} questions, {r['n_dbs']} DBs)\n")
        L.append("| term | est | question CI | DB-cluster CI | change |")
        L.append("|---|---|---|---|---|")
        for t, v in r["terms"].items():
            L.append(f"| {t} | {v['est']:+.2f} | [{v['q_lo']:+.2f}, {v['q_hi']:+.2f}] | [{v['db_lo']:+.2f}, {v['db_hi']:+.2f}] | "
                     f"{'⚠ ' + ('loses' if v['sig_question'] else 'gains') + ' significance' if v['changes'] else ''} |")
    (out / "db_cluster_bootstrap.md").write_text("\n".join(L) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS"}
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in hashes.items()))
    print("\n".join(L[:3]))
    print("\n".join(l for l in L if "mean of 6" in l or l.startswith("| ")) if False else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
