from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict

import numpy as np

from share_eval_suite import stats
from share_eval_suite.comparisons import dependency
from share_eval_suite.steps import _replay_common as C

sys.dont_write_bytecode = True

OUT = C.OUTPUT / "replay_validity"
BOOT, SEED = stats.DRAWS, stats.SEED_REPLAY
INTERACTIVE = dependency.INTER

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


def summarise(est: np.ndarray, lo: np.ndarray, hi: np.ndarray, idx: np.ndarray) -> Dict[str, Any]:
    b_est = est[idx].mean(axis=1)
    b_lo, b_hi = lo[idx].mean(axis=1), hi[idx].mean(axis=1)
    f = lambda x: round(100 * float(x), 2)
    return dict(replay=f(est.mean()), replay_ci=[f(np.percentile(b_est, 2.5)), f(np.percentile(b_est, 97.5))],
                bound=[f(lo.mean()), f(hi.mean())],
                bound_ci=[f(np.percentile(b_lo, 2.5)), f(np.percentile(b_hi, 97.5))],
                width=f(hi.mean() - lo.mean()),
                sign_identified=bool(lo.mean() > 0 or hi.mean() < 0),
                sign_identified_ci=bool(np.percentile(b_lo, 2.5) > 0 or np.percentile(b_hi, 97.5) < 0))


def read_divergence(path) -> Dict:
    div = {}
    for line in pathlib.Path(path).read_text().splitlines():
        d = json.loads(line)
        div[(d["unit"], d["arm"], d["backbone"], int(d["repeat"]))] = d
    return div


def run(div=OUT / "bccs_divergence_runs.jsonl", scores=C.OUTPUT / "replay" / "per_run_scores.jsonl",
        questions=C.BIRD_DEV, out=OUT / "bounds.json", backbones=()) -> int:
    keep_bb = {b for b in backbones if b}
    divs = read_divergence(div)
    runs = [json.loads(line) for line in pathlib.Path(scores).read_text().splitlines()]
    runs = [r for r in runs if r["protocol"] in INTERACTIVE and (not keep_bb or r["backbone"] in keep_bb)]
    cells = sorted({(r["protocol"], r["backbone"]) for r in runs})
    dbmap = None
    res = {}
    for cname, terms in CONTRASTS.items():
        per = {}
        arrays = {}
        for c in cells:
            rs = [r for r in runs if (r["protocol"], r["backbone"]) == c]
            arrays[c] = dependency._task_arrays(rs, divs, terms)
        u0 = arrays[cells[0]][0]
        assert all(arrays[c][0] == u0 for c in cells), "cells must share one task set"
        idx = stats.resample_index(len(u0), BOOT, SEED)
        for c in cells:
            per[f"{c[0]}|{c[1]}"] = summarise(*arrays[c][1:], idx)
        for label, sub in (("mean of 6", cells), ("mean of 4 (Du-MAD, MAD-M2)", [c for c in cells if c[0] != "bccs"]),
                           ("mean of 2 (BCCS)", [c for c in cells if c[0] == "bccs"])):
            E = np.mean([arrays[c][1] for c in sub], axis=0)
            L = np.mean([arrays[c][2] for c in sub], axis=0)
            H = np.mean([arrays[c][3] for c in sub], axis=0)
            per[label] = summarise(E, L, H, idx)
            if label.startswith("mean of 4"):
                if dbmap is None:
                    dbmap = C.question_databases(questions)
                lo_db, hi_db = dependency._db_ci(u0, E, dbmap)
                per[label]["replay_db_cluster_ci"] = [round(lo_db, 2), round(hi_db, 2)]
        res[cname] = per
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    for cname, per in res.items():
        print(f"== {cname}")
        for k in [c for c in per if c.startswith("bccs|")] + ["mean of 6", "mean of 4 (Du-MAD, MAD-M2)"]:
            v = per[k]
            print(f"   {k:28s} replay {v['replay']:+7.2f} {v['replay_ci']}  bound {v['bound']} ci {v['bound_ci']} "
                  f"w {v['width']:.1f} signID {v['sign_identified']}/{v['sign_identified_ci']} {v.get('replay_db_cluster_ci', '')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Deployment bounds of the replayed contrasts")
    ap.add_argument("--div", default=str(OUT / "bccs_divergence_runs.jsonl"))
    ap.add_argument("--scores", default=str(C.OUTPUT / "replay" / "per_run_scores.jsonl"))
    ap.add_argument("--questions", default=str(C.BIRD_DEV), help="BIRD dev.json, or a CSV with columns unit, db_id")
    ap.add_argument("--out", default=str(OUT / "bounds.json"))
    ap.add_argument("--backbones", default="")
    a = ap.parse_args(argv)
    return run(div=a.div, scores=a.scores, questions=a.questions, out=a.out, backbones=a.backbones.split(","))


if __name__ == "__main__":
    sys.exit(main())
