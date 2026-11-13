r"""Robustness of the online bccs_execid Δpeer (qwen2.5-7b-execid): per-database breakdown, tie-break
sensitivity, and exclusion of runs whose stop fired on a single executing answer. Zero model calls.
    python3 -I sb_online_checks.py -> output/online_dpeer_checks.json"""
import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "output"
sys.path.insert(0, str(HERE))
import sb_q4 as Q                                                          # noqa: E402

BB = "qwen2.5-7b-execid"


def load(sfx):
    rows = [json.loads(l) for l in (OUT / f"interaction_matched_rows{sfx}.jsonl").read_text().splitlines()]
    return [r for r in rows if r["backbone"] == BB and r["unit"] not in Q.EXCL]


def dpeer(rows, keep=None):
    F = [r for r in rows if r["arm"] == "FULL" and (keep is None or keep(r))]
    N = [r for r in rows if r["arm"] == "NOPEER" and (keep is None or keep(r))]
    a, b = Q.per_unit(F, Q.e1), Q.per_unit(N, Q.e1)
    return {u: a[u] - b[u] for u in set(a) & set(b)}


def main():
    sys.path.insert(0, str(REPO / "experiments/02_controlled_sharing/code"))
    import sql_common as SQ
    db = {f"bird|{q['question_id']}": q["db_id"] for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}
    single = set()
    for p in (REPO / "data/records/backbone_scale_cross_family/raw").glob("*qwen25-7b-execid*"):
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if r.get("status") != "ok":
                continue
            if any(t["verdict"] == "full consensus" and t["n_executing"] == 1 for t in r["record"]["execid_trace"][::2]):
                single.add((r["unit"], r["arm"], int(r["repeat"])))
    out = {}
    rows = load("")
    d = dpeer(rows)
    out["first_tie"] = Q.stats(d, db)
    per_db = collections.defaultdict(list)
    for u, v in d.items():
        per_db[db[u]].append(v)
    out["per_database"] = {k: dict(n=len(v), est=round(100 * float(np.mean(v)), 2)) for k, v in sorted(per_db.items())}
    out["tie_last"] = Q.stats(dpeer(load("_tie_last")), db)
    out["tie_random"] = Q.stats(dpeer(load("_tie_random")), db)
    out["single_exec_stop_runs"] = len(single)
    out["excluding_single_exec_stop_runs"] = Q.stats(
        dpeer(rows, keep=lambda r: (r["unit"], r["arm"], int(r["repeat"])) not in single), db)
    (OUT / "online_dpeer_checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
