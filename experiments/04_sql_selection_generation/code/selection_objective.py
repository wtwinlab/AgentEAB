r"""The operational experiment: does selecting a configuration by pair quality or by group quality
change what you deploy, and what you get?

    PYTHONPATH=...:experiments/04_sql_selection_generation/code \
      python3 experiments/04_sql_selection_generation/code/selection_objective.py \
        --dataset geographic_settlements
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
PROTO = os.path.join(ROOT, "data/records/matching_protocol")
OUT = os.path.join(STAGE, "output")
POOL = ("qwen2.5-7b", "llama-3-8b", "mistral-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b")
GROUPINGS = ("connected_components", "center")
NESTED_RESAMPLES = 40


def config_id(rec: str, arm: str, grp: str) -> str:
    return f"{rec}|{arm}|{grp}"


def evaluate(M, arms, pv, ent, nodes, cands, rec: str, arm: str, grp: str) -> Dict[str, Any]:
    """One configuration on one split: pair F1, cluster F1, and its measured resources."""
    # Evaluate the induced candidate graph: pair scores, grouping and resource
    # ties must use the same retained records.
    node_set = set(nodes)
    candidate_set = {tuple(pair) for pair in cands
                     if pair[0] in node_set and pair[1] in node_set}
    arms = [r for r in arms if (r["a"], r["b"]) in candidate_set]
    draws = sorted({r.get("draw") for r in arms
                    if r["recipient"] == rec and r["arm"] == arm and r.get("draw") is not None})
    dec = M.arm_decisions(arms, rec, arm, draw=draws[0] if draws else None)
    if not dec:
        return {}
    ps = M.pair_scores(dec, ent)
    links = [(x, y) for (x, y), d in dec.items() if d == "match"]
    if grp == "connected_components":
        part = M.connected_components(nodes, links)
    else:
        part = M.center_clustering(
            nodes, [(x, y, M.vote_fraction(pv, x, y, POOL)) for x, y in links])
    cs = M.cluster_scores(part, ent, nodes)
    rows = [r for r in arms if r["recipient"] == rec and r["arm"] == arm
            and (not draws or r.get("draw") == draws[0])]
    res = {"calls": len(rows),
           "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rows),
           "completion_tokens": sum(r.get("completion_tokens") or 0 for r in rows)}
    res["tokens"] = res["prompt_tokens"] + res["completion_tokens"]
    return {"pair_f1": ps["f1"], "pair_precision": ps["precision"], "pair_recall": ps["recall"],
            "abstention_rate": ps["abstention_rate"], "n_links": len(links),
            "cluster_f1": cs["f1"], "cluster_precision": cs["precision"],
            "cluster_recall": cs["recall"], "n_clusters": cs["n_clusters"],
            "largest_cluster": cs["largest_cluster"], "resources": res}


def pick(scores: Dict[str, Dict[str, Any]], key: str,
         feasible: Sequence[str]) -> Optional[str]:
    """argmax on `key`, ties by lower development tokens then configuration id."""
    cand = [c for c in feasible if c in scores and scores[c]]
    if not cand:
        return None
    return sorted(cand, key=lambda c: (-scores[c][key],
                                       scores[c]["resources"]["tokens"], c))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(STAGE, "code"))
    import metrics as M                                                 # noqa: PLC0415

    blob = json.load(open(os.path.join(PROTO, f"{a.dataset}.json"), encoding="utf-8"))
    ent, split_of = blob["entity_of"], blob["split"]
    allc = [tuple(c) for c in blob["candidates"]]
    data = os.path.join(STAGE, "data", a.dataset)

    per_split: Dict[str, Dict[str, Dict[str, Any]]] = {}
    ctx: Dict[str, Any] = {}
    for sp in ("dev", "test"):
        apth, ppth = os.path.join(data, f"arms_{sp}.jsonl"), os.path.join(data, f"pool_{sp}.jsonl")
        if not (os.path.exists(apth) and os.path.exists(ppth)):
            print(f"  {sp}: not collected yet")
            return 1
        arms, pool = M.load_jsonl(apth), M.load_jsonl(ppth)
        pv = M.pool_view(pool)
        nodes = sorted({r for r in ent if split_of[ent[r]] == sp})
        cands = [(x, y) for x, y in allc
                 if split_of[ent[x]] == sp and split_of[ent[y]] == sp]
        ctx[sp] = {"arms": arms, "pv": pv, "nodes": nodes, "cands": cands}
        got = sorted({(r["recipient"], r["arm"]) for r in arms})
        sc: Dict[str, Dict[str, Any]] = {}
        for rec, arm in got:
            for grp in GROUPINGS:
                sc[config_id(rec, arm, grp)] = evaluate(
                    M, arms, pv, ent, nodes, cands, rec, arm, grp)
        per_split[sp] = sc
        print(f"  {sp}: {len(sc)} configurations scored")

    dev, test = per_split["dev"], per_split["test"]
    configs = sorted(c for c in dev if dev[c] and c in test and test[c])
    rep: Dict[str, Any] = {
        "dataset": a.dataset, "n_configurations": len(configs), "configurations": configs,
        "design": {
            "space": "(recipient, arm, grouping rule), identical for both selection rules",
            "tie_rule": "lower development tokens, then configuration id ascending",
            "gold_use": "development gold selects; test gold scores the selected configuration once",
            "budgets": "unrestricted, plus 0.5x and 1.0x of the fixed-pool baseline's development cost"},
        "per_configuration": {c: {"dev": dev[c], "test": test[c]} for c in configs},
        "budgets": {}}

    # the budget reference: the fixed-pool baseline D5 on the stronger recipient, if present
    base = next((c for c in ("qwen2.5-32b|D5|connected_components",
                             "qwen2.5-7b|D5|connected_components") if c in dev), configs[0])
    ref = dev[base]["resources"]["tokens"]
    rep["budget_reference"] = {"configuration": base, "dev_tokens": ref}

    for lab, mult in (("unrestricted", None), ("0.5x", 0.5), ("1.0x", 1.0)):
        feas = configs if mult is None else [
            c for c in configs if dev[c]["resources"]["tokens"] <= mult * ref]
        if not feas:
            rep["budgets"][lab] = {"feasible": 0, "infeasible": True,
                                   "note": "no configuration fits this development budget; "
                                           "recorded infeasible, budget not re-chosen"}
            print(f"  budget {lab}: INFEASIBLE")
            continue
        A = pick(dev, "pair_f1", feas)
        B = pick(dev, "cluster_f1", feas)
        oracle_pair = pick(test, "pair_f1", feas)
        oracle_grp = pick(test, "cluster_f1", feas)
        e = {
            "feasible": len(feas), "feasible_configurations": feas,
            "rule_A_selected_by_dev_pair_f1": A,
            "rule_B_selected_by_dev_cluster_f1": B,
            "same_configuration": A == B,
            "dev": {"A_pair_f1": dev[A]["pair_f1"], "A_cluster_f1": dev[A]["cluster_f1"],
                    "B_pair_f1": dev[B]["pair_f1"], "B_cluster_f1": dev[B]["cluster_f1"]},
            "held_out": {
                "A_pair_f1": test[A]["pair_f1"], "A_cluster_f1": test[A]["cluster_f1"],
                "B_pair_f1": test[B]["pair_f1"], "B_cluster_f1": test[B]["cluster_f1"],
                "cluster_f1_B_minus_A": test[B]["cluster_f1"] - test[A]["cluster_f1"],
                "pair_f1_B_minus_A": test[B]["pair_f1"] - test[A]["pair_f1"]},
            "realised_test_resources": {
                "A": test[A]["resources"], "B": test[B]["resources"],
                "note": "reported, not equalised; the two configurations differ"},
            "diagnostic_test_oracle": {
                "best_test_pair_f1": oracle_pair,
                "best_test_cluster_f1": oracle_grp,
                "cluster_f1": test[oracle_grp]["cluster_f1"],
                "regret_of_B_vs_oracle": test[oracle_grp]["cluster_f1"] - test[B]["cluster_f1"],
                "label": "diagnostic ceiling only; selected with test gold, never deployable"}}
        rep["budgets"][lab] = e
        print(f"  budget {lab}: A={A} B={B} same={A == B}  "
              f"held-out cluster F1 A {test[A]['cluster_f1']:.4f} vs B {test[B]['cluster_f1']:.4f} "
              f"(B−A {e['held_out']['cluster_f1_B_minus_A']:+.4f})")

    # ---------------------------------------------------------------- nested estimate
    # re-run the whole selection on resampled development entities. This estimates the procedure,
    # not one draw of it.
    dev_ents = sorted({ent[r] for r in ctx["dev"]["nodes"]})
    rng = random.Random(20261015)
    agree, deltas, picks_A, picks_B = 0, [], collections.Counter(), collections.Counter()
    for _ in range(NESTED_RESAMPLES):
        keep = set(rng.sample(dev_ents, max(2, int(0.8 * len(dev_ents)))))
        sub_nodes = [r for r in ctx["dev"]["nodes"] if ent[r] in keep]
        sub_c = [(x, y) for x, y in ctx["dev"]["cands"] if ent[x] in keep and ent[y] in keep]
        sc: Dict[str, Dict[str, Any]] = {}
        for c in configs:
            rec, arm, grp = c.split("|")
            sc[c] = evaluate(M, ctx["dev"]["arms"], ctx["dev"]["pv"], ent,
                             sub_nodes, sub_c, rec, arm, grp)
        A = pick(sc, "pair_f1", configs)
        B = pick(sc, "cluster_f1", configs)
        if A and B:
            picks_A[A] += 1
            picks_B[B] += 1
            agree += int(A == B)
            deltas.append(test[B]["cluster_f1"] - test[A]["cluster_f1"])
    rep["nested_selection_estimate"] = {
        "resamples": NESTED_RESAMPLES, "development_entity_fraction": 0.8,
        "rules_agree_fraction": agree / max(1, len(deltas)),
        "held_out_cluster_f1_B_minus_A": {
            "mean": st.fmean(deltas) if deltas else None,
            "min": min(deltas) if deltas else None, "max": max(deltas) if deltas else None},
        "rule_A_selection_frequency": dict(picks_A.most_common()),
        "rule_B_selection_frequency": dict(picks_B.most_common()),
        "interpretation": "estimates the selection PROCEDURE over development resamples. Entity "
                          "resampling changes the predicted graph, so this is a range over "
                          "resamples, not a confidence interval for the fixed graph."}
    n = rep["nested_selection_estimate"]
    print(f"\n  nested over {NESTED_RESAMPLES} dev resamples: rules agree "
          f"{100 * n['rules_agree_fraction']:.0f}% of the time; held-out cluster F1 B−A mean "
          f"{n['held_out_cluster_f1_B_minus_A']['mean']}")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"selection_{a.dataset}.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
