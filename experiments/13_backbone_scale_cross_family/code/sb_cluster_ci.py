r"""Task-level and database-cluster bootstrap CIs for E1, Δpeer and the deployed FULL − NOPEER
difference, per (backbone, protocol). Zero model calls.

    python3 -I sb_cluster_ci.py [--backbones qwen2.5-7b-execid,...]

Reads output/interaction_matched_rows.jsonl (first tie). Task-level: 10,000 task resamples. DB
cluster: 10,000 resamples of the 11 BIRD databases with replacement, every task of a drawn database
included (the per-task value is first averaged over repeats). Seed 20261106. Writes
output/cluster_ci.json.
"""
import argparse
import collections
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "output"
EXCL = {"bird|518", "bird|701"}
E = "exec_filt|lenient"
BOOT, SEED = 10000, 20261106


def per_unit(rows, f):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["unit"]].append(f(r))
    return {u: float(np.mean(v)) for u, v in by.items()}


def ci(d, db):
    us = sorted(d)
    x = np.array([d[u] for u in us])
    rng = np.random.default_rng(SEED)
    t = x[rng.integers(0, len(x), size=(BOOT, len(x)))].mean(axis=1)
    groups = collections.defaultdict(list)
    for i, u in enumerate(us):
        groups[db[u]].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(SEED)
    c = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = [i for k in pick for i in groups[keys[k]]]
        c.append(x[idx].mean())
    return dict(n_tasks=len(us), n_dbs=len(keys), est=round(100 * x.mean(), 2),
                task_ci=[round(100 * v, 2) for v in np.percentile(t, [2.5, 97.5])],
                db_cluster_ci=[round(100 * v, 2) for v in np.percentile(c, [2.5, 97.5])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(REPO / "experiments/02_controlled_sharing/code"))
    import sql_common as SQ
    db = {f"bird|{q['question_id']}": q["db_id"] for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}
    rows = [json.loads(l) for l in (OUT / "interaction_matched_rows.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["unit"] not in EXCL]
    want = set(b for b in a.backbones.split(",") if b)
    cells = collections.defaultdict(list)
    for r in rows:
        if not want or r["backbone"] in want:
            cells[(r["backbone"], r["protocol"], r["arm"])].append(r)
    out = {}
    for (bb, proto, arm) in sorted(cells):
        if arm != "FULL":
            continue
        F, N = cells[(bb, proto, "FULL")], cells.get((bb, proto, "NOPEER"), [])
        e1f = per_unit(F, lambda r: float(r[f"last|{E}"]) - float(r[f"r0|{E}"]))
        rec = dict(E1_full=ci(e1f, db))
        if N:
            e1n = per_unit(N, lambda r: float(r[f"last|{E}"]) - float(r[f"r0|{E}"]))
            com = set(e1f) & set(e1n)
            rec["E1_nopeer"] = ci(e1n, db)
            rec["dpeer"] = ci({u: e1f[u] - e1n[u] for u in com}, db)
            ff = per_unit(F, lambda r: float(r["final"]))
            fn = per_unit(N, lambda r: float(r["final"]))
            rec["final_full_minus_nopeer"] = ci({u: ff[u] - fn[u] for u in set(ff) & set(fn)}, db)
            lf = per_unit(F, lambda r: float(r[f"last|{E}"]))
            ln = per_unit(N, lambda r: float(r[f"last|{E}"]))
            rec["last_vote_full_minus_nopeer"] = ci({u: lf[u] - ln[u] for u in set(lf) & set(ln)}, db)
        out[f"{bb}|{proto}"] = rec
        print(bb, proto, json.dumps(rec))
    (OUT / "cluster_ci.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
