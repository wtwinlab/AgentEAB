"""Evaluate development-selected grouping across entity-disjoint folds of the saved candidate graphs."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]
OUT = WORK / "output"
NFS = WORK
CACHE = NFS / "cache"
STAGE = WORK
for _p in (str(NFS / "code"), str(STAGE / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import eb1_grouping as EB                                                  # noqa: E402
import metrics as M                                                        # noqa: E402

DATASETS = EB.DATASETS
FOLDS = 5
SALT = "cis2-grouping-resplit-20261028"


def side_entities(entry: Dict[str, Any]) -> Dict[str, List[str]]:
    by = collections.defaultdict(list)
    for node, ent in entry["entity_of"].items():
        by[ent].append(node)
    return by


def fold_of(ent: str) -> int:
    return int(hashlib.sha256((SALT + "|" + str(ent)).encode()).hexdigest(), 16) % FOLDS


def restrict(entry: Dict[str, Any], keep: set) -> Dict[str, Any]:
    """The same entry structure, cut down to one fold's nodes and its really-judged pairs."""
    nodes = sorted(keep)
    score = {k: v for k, v in entry["edge_score_pool_vote_fraction"].items()
             if set(k.split("\t")) <= keep}
    cells = {}
    for ck, cell in entry["cells"].items():
        dec = {k: d for k, d in cell["decisions"].items() if set(k.split("\t")) <= keep}
        cells[ck] = dict(cell, decisions=dec, n_pairs=len(dec))
    return dict(nodes=nodes,
                entity_of={n: e for n, e in entry["entity_of"].items() if n in keep},
                edge_score_pool_vote_fraction=score, cells=cells,
                resources_by_recipient_arm=entry.get("resources_by_recipient_arm", {}),
                candidates=[k for k in score])


def fold_coverage(sub: Dict[str, Any]) -> Dict[str, Any]:
    cand = [tuple(k.split("\t")) for k in sub["edge_score_pool_vote_fraction"]]
    cov = M.coverage(cand, sub["entity_of"], sub["nodes"])
    return dict(candidate_pairs=len(cand), **{k: cov[k] for k in sorted(cov)})


def load(dataset: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for side in ("dev", "test"):
        p = CACHE / f"{dataset}_{side}.json"
        out[side] = json.loads(p.read_text(encoding="utf-8"))
    return out


def run_dataset(dataset: str, insts) -> Dict[str, Any]:
    sides = load(dataset)
    folds: Dict[Tuple[str, int], Dict[str, Any]] = {}
    meta: List[Dict[str, Any]] = []
    for side, entry in sides.items():
        ents = side_entities(entry)
        assign = collections.defaultdict(set)
        for ent, nodes in ents.items():
            assign[fold_of(ent)].update(nodes)
        for f in sorted(assign):
            sub = restrict(entry, assign[f])
            if len(sub["nodes"]) < 10 or not sub["edge_score_pool_vote_fraction"]:
                meta.append(dict(side=side, fold=f, usable=False,
                                 reason="fold too small or no judged pair inside it",
                                 nodes=len(sub["nodes"])))
                continue
            cov = fold_coverage(sub)
            res = EB.evaluate(sub, insts, f"{dataset}|{side}f{f}")
            folds[(side, f)] = res
            meta.append(dict(side=side, fold=f, usable=True, nodes=len(sub["nodes"]),
                             entities=len({sub['entity_of'][n] for n in sub['nodes']}),
                             **cov))
            print(f"  {dataset} {side} fold {f}: {len(sub['nodes'])} nodes, "
                  f"{cov['candidate_pairs']} judged pairs, coverage "
                  f"{cov.get('coverage')}", flush=True)
    return dict(folds=folds, fold_meta=meta, sides=list(sides))


def selection_rows(dataset: str, run: Dict[str, Any]) -> List[Dict[str, Any]]:
    order_of = {k: i for i, k in enumerate(EB.RULE_ORDER)}
    rows: List[Dict[str, Any]] = []
    for side in run["sides"]:
        fs = sorted(f for (s, f) in run["folds"] if s == side)
        for sf in fs:
            for ef in fs:
                if sf == ef:
                    continue
                sel_cells = run["folds"][(side, sf)]
                ev_cells = run["folds"][(side, ef)]
                for ck in sorted(set(sel_cells) & set(ev_cells)):
                    dv, tv = sel_cells[ck]["rules"], ev_cells[ck]["rules"]
                    top = max(dv[k]["cluster_f1"] for k in dv)
                    chosen = sorted([k for k in dv if dv[k]["cluster_f1"] == top],
                                    key=lambda k: order_of[dv[k]["rule"]])[0]
                    best_ev = max(tv, key=lambda k: tv[k]["cluster_f1"])
                    row = dict(dataset=dataset, side=side, select_fold=sf, eval_fold=ef,
                               cell=ck, recipient=ev_cells[ck]["recipient"],
                               arm=ev_cells[ck]["arm"],
                               selected_rule=chosen,
                               selected_eval_f1=tv[chosen]["cluster_f1"],
                               retrospective_best_rule=best_ev,
                               retrospective_best_eval_f1=tv[best_ev]["cluster_f1"],
                               regret_pp=100 * (tv[best_ev]["cluster_f1"]
                                                - tv[chosen]["cluster_f1"]))
                    for k in sorted(tv):
                        row[f"eval_f1__{k}"] = tv[k]["cluster_f1"]
                        row[f"selected_minus__{k}"] = 100 * (tv[chosen]["cluster_f1"]
                                                             - tv[k]["cluster_f1"])
                    rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    insts = EB.rule_instances()
    names = [a.dataset] if a.dataset else list(DATASETS)

    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = dict(
        folds_per_side=FOLDS, salt=SALT, rule_instances=len(insts),
        legality=("re-partitions are entity-isolated and stay inside an original side, because no "
                  "pair spanning the original dev/test boundary was ever judged and a missing edge "
                  "must not be scored as a negative edge"),
        dependence=("folds recur across ordered (select, eval) pairs, so those pairs are not "
                    "independent samples; spreads are reported as ranges, not as intervals"),
        datasets={})
    for ds in names:
        print(f"== {ds}", flush=True)
        run = run_dataset(ds, insts)
        rows = selection_rows(ds, run)
        all_rows += rows
        if rows:
            reg = [r["regret_pp"] for r in rows]
            fixed = sorted({k[len("selected_minus__"):] for k in rows[0]
                            if k.startswith("selected_minus__")})
            vs_fixed = {}
            for k in fixed:
                d = [r[f"selected_minus__{k}"] for r in rows]
                vs_fixed[k] = dict(mean_pp=round(statistics.fmean(d), 4),
                                   min_pp=round(min(d), 4), max_pp=round(max(d), 4),
                                   share_selected_worse=round(
                                       sum(1 for x in d if x < 0) / len(d), 4))
            chosen_counts = collections.Counter(r["selected_rule"] for r in rows)
            summary["datasets"][ds] = dict(
                fold_meta=run["fold_meta"], ordered_pairs=len(rows),
                regret_mean_pp=round(statistics.fmean(reg), 4),
                regret_median_pp=round(statistics.median(reg), 4),
                regret_max_pp=round(max(reg), 4),
                regret_p90_pp=round(sorted(reg)[int(0.9 * len(reg))], 4),
                share_zero_regret=round(sum(1 for x in reg if x <= 1e-12) / len(reg), 4),
                selected_rule_counts=dict(chosen_counts),
                selected_minus_each_fixed_rule=vs_fixed)
        else:
            summary["datasets"][ds] = dict(fold_meta=run["fold_meta"], ordered_pairs=0,
                                           note="no usable fold pair")
    (OUT / "gr1_resplit.json").write_text(json.dumps(summary, indent=1, sort_keys=True),
                                          encoding="utf-8")
    if all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open(OUT / "gr1_resplit_rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
    for ds, d in summary["datasets"].items():
        if not d.get("ordered_pairs"):
            continue
        print(f"\n{ds}: {d['ordered_pairs']} ordered (select, eval) fold pairs")
        print(f"  regret pp  mean {d['regret_mean_pp']:.3f}  median "
              f"{d['regret_median_pp']:.3f}  p90 {d['regret_p90_pp']:.3f}  "
              f"max {d['regret_max_pp']:.3f}  zero-regret share {d['share_zero_regret']:.3f}")
        print(f"  rules chosen: {d['selected_rule_counts']}")
        print("  selected minus each fixed rule (pp): mean / min / max / share selected worse")
        for k, v in d["selected_minus_each_fixed_rule"].items():
            print(f"    {k:28s} {v['mean_pp']:+8.3f} {v['min_pp']:+8.3f} "
                  f"{v['max_pp']:+8.3f} {v['share_selected_worse']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
