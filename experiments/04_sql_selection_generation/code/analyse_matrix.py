r"""Analysis of the eight-arm matrix for one dataset: RQ1, RQ2, RQ3 and the RQ4 ledger.

    PYTHONPATH=experiments/06_cost_resources/code:experiments/05_matching_downstream/code:experiments/04_sql_selection_generation/code \
      python3 experiments/04_sql_selection_generation/code/analyse_matrix.py --dataset geographic_settlements
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
PROTO = os.path.join(ROOT, "data/records/matching_protocol")
OUT = os.path.join(STAGE, "output")

POOL = ("qwen2.5-7b", "llama-3-8b", "mistral-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b")
RECIPIENTS = ("qwen2.5-7b", "qwen2.5-32b")
ARMS = ("N", "D5", "E5", "F5", "UD1", "UF1", "AD1", "VD1")
PAIRED = [("D5", "N"), ("F5", "N"), ("D5", "F5"), ("E5", "D5"),
          ("UD1", "D5"), ("UF1", "UD1"), ("AD1", "UD1"), ("VD1", "UD1")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", default="dev,test")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(STAGE, "code"))
    import metrics as M                                                 # noqa: PLC0415

    blob = json.load(open(os.path.join(PROTO, f"{a.dataset}.json"), encoding="utf-8"))
    ent, split_of = blob["entity_of"], blob["split"]
    data = os.path.join(STAGE, "data", a.dataset)
    rep: Dict[str, Any] = {"dataset": a.dataset, "splits": {}}

    for sp in a.splits.split(","):
        ap_ = os.path.join(data, f"arms_{sp}.jsonl")
        pp = os.path.join(data, f"pool_{sp}.jsonl")
        if not (os.path.exists(ap_) and os.path.exists(pp)):
            print(f"  {a.dataset}/{sp}: not collected yet, skipped")
            continue
        arms = M.load_jsonl(ap_)
        pool = M.load_jsonl(pp)
        pv = M.pool_view(pool)
        nodes = sorted({r for r in ent if split_of[ent[r]] == sp})
        cands = [(x, y) for x, y in (tuple(c) for c in blob["candidates"])
                 if split_of[ent[x]] == sp and split_of[ent[y]] == sp]
        cov = M.coverage(cands, ent, nodes)

        entry: Dict[str, Any] = {
            "n_candidates": len(cands), "n_records": len(nodes), "coverage": cov,
            "n_arm_rows": len(arms), "n_pool_rows": len(pool),
            "arms_present": sorted({r["arm"] for r in arms}),
            "recipients_present": sorted({r["recipient"] for r in arms}),
            "by_recipient": {}}

        for rec in RECIPIENTS:
            got = sorted({r["arm"] for r in arms if r["recipient"] == rec})
            per_arm: Dict[str, Any] = {}
            for arm in got:
                # UD1/UF1 average over the three frozen peer draws; each draw is a policy sample,
                # not an independent pair
                draws = sorted({r.get("draw") for r in arms
                                if r["recipient"] == rec and r["arm"] == arm
                                and r.get("draw") is not None})
                if draws:
                    per_draw = []
                    for d in draws:
                        dec = M.arm_decisions(arms, rec, arm, draw=d)
                        per_draw.append(M.pair_scores(dec, ent))
                    base = {k: st.fmean([p[k] for p in per_draw])
                            for k in ("precision", "recall", "f1", "abstention_rate")}
                    base["n_draws"] = len(draws)
                    base["f1_per_draw"] = [p["f1"] for p in per_draw]
                    base["draw_spread"] = max(p["f1"] for p in per_draw) - min(
                        p["f1"] for p in per_draw)
                    dec = M.arm_decisions(arms, rec, arm, draw=draws[0])
                    base["n_pairs"] = per_draw[0]["n_pairs"]
                else:
                    dec = M.arm_decisions(arms, rec, arm)
                    base = M.pair_scores(dec, ent)

                # repeat-level spread: generation randomness, not a population CI
                byrep = collections.defaultdict(dict)
                for r in arms:
                    if r["recipient"] == rec and r["arm"] == arm and r.get("draw") in (None, 0):
                        byrep[r["repeat"]][(r["a"], r["b"])] = r["decision"]
                reps = [M.pair_scores(d, ent)["f1"] for d in byrep.values() if d]
                base["f1_per_repeat"] = reps
                base["repeat_spread"] = (max(reps) - min(reps)) if len(reps) > 1 else 0.0

                # scope B and C: links from this arm's modal decisions
                links = [(x, y) for (x, y), d in dec.items() if d == "match"]
                scored = [(x, y, M.vote_fraction(pv, x, y, POOL)) for x, y in links]
                cc = M.connected_components(nodes, links)
                ce = M.center_clustering(nodes, scored)
                base["scope_B_direct_links"] = {
                    "n_links": len(links),
                    **{k: v for k, v in M.pair_scores(
                        {**{c: None for c in
                            ((x, y) for x in nodes for y in nodes if False)},
                         **dec}, ent).items() if k in ("precision", "recall", "f1")},
                    "f1_over_full_universe": (2 * sum(1 for x, y in links if ent[x] == ent[y])
                                              / (len(links) + cov["gold_pairs_in_universe"]))
                    if (len(links) + cov["gold_pairs_in_universe"]) else 0.0}
                base["scope_C_connected_components"] = M.cluster_scores(cc, ent, nodes)
                base["scope_C_connected_components"]["amplification"] = M.amplification(
                    cc, nodes, len(links))
                base["scope_C_center"] = M.cluster_scores(ce, ent, nodes)
                base["scope_C_center"]["amplification"] = M.amplification(ce, nodes, len(links))
                per_arm[arm] = base

            # paired differences on the pairs both conditions decided
            diffs: Dict[str, Any] = {}
            for x, y in PAIRED:
                if x not in per_arm or y not in per_arm:
                    continue
                dx = M.arm_decisions(arms, rec, x, draw=0 if "f1_per_draw" in per_arm[x] else None)
                dy = M.arm_decisions(arms, rec, y, draw=0 if "f1_per_draw" in per_arm[y] else None)
                common = sorted(set(dx) & set(dy))
                sx = M.pair_scores({k: dx[k] for k in common}, ent)
                sy = M.pair_scores({k: dy[k] for k in common}, ent)
                corr = sum(1 for k in common
                           if dy[k] != ("match" if ent[k[0]] == ent[k[1]] else "non-match")
                           and dx[k] == ("match" if ent[k[0]] == ent[k[1]] else "non-match"))
                harm = sum(1 for k in common
                           if dy[k] == ("match" if ent[k[0]] == ent[k[1]] else "non-match")
                           and dx[k] != ("match" if ent[k[0]] == ent[k[1]] else "non-match"))
                diffs[f"{x}-{y}"] = {
                    "n_common_pairs": len(common),
                    "pair_f1_delta": sx["f1"] - sy["f1"],
                    "pair_f1": {x: sx["f1"], y: sy["f1"]},
                    "precision_delta": sx["precision"] - sy["precision"],
                    "recall_delta": sx["recall"] - sy["recall"],
                    "abstention_delta": sx["abstention_rate"] - sy["abstention_rate"],
                    "corrections": corr, "harms": harm, "net_corrections": corr - harm,
                    "cluster_f1_delta_connected_components":
                        per_arm[x]["scope_C_connected_components"]["f1"]
                        - per_arm[y]["scope_C_connected_components"]["f1"],
                    "cluster_f1_delta_center":
                        per_arm[x]["scope_C_center"]["f1"] - per_arm[y]["scope_C_center"]["f1"]}
            entry["by_recipient"][rec] = {"arms": per_arm, "paired_differences": diffs}

        entry["resources"] = {
            "by_role": M.resource_ledger(arms + pool),
            "by_arm": M.arm_resources(arms),
            "boundaries": {
                "recipient_margin": "recipient calls only, pool held fixed",
                "whole_workflow": "pool generation + verifier scoring + recipient calls",
                "benchmark_collection_total": "every call this stage made, including all repeats; "
                                              "not a deployment policy cost"}}
        rep["splits"][sp] = entry

        print(f"\n=== {a.dataset}/{sp}: {len(cands)} candidates, {len(nodes)} records, "
              f"coverage {cov['coverage']:.4f}")
        for rec, rv in entry["by_recipient"].items():
            print(f"  {rec}")
            print(f"    {'arm':5s}{'pairF1':>9s}{'absten':>8s}{'links':>8s}"
                  f"{'ccF1':>8s}{'ctrF1':>8s}{'amp':>7s}")
            for arm in ARMS:
                if arm not in rv["arms"]:
                    continue
                v = rv["arms"][arm]
                print(f"    {arm:5s}{v['f1']:9.4f}{v['abstention_rate']:8.3f}"
                      f"{v['scope_B_direct_links']['n_links']:8d}"
                      f"{v['scope_C_connected_components']['f1']:8.4f}"
                      f"{v['scope_C_center']['f1']:8.4f}"
                      f"{(v['scope_C_connected_components']['amplification'] or 0):7.1f}")
            for k, d in rv["paired_differences"].items():
                print(f"      {k:9s} pairF1 {d['pair_f1_delta']:+.4f}  "
                      f"ccF1 {d['cluster_f1_delta_connected_components']:+.4f}  "
                      f"centerF1 {d['cluster_f1_delta_center']:+.4f}  "
                      f"corr {d['corrections']} harm {d['harms']}")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"analysis_{a.dataset}.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
