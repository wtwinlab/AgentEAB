r"""RQ3-B (final answer use) and RQ4 (sharing versus extra generation at matched calls).

    PYTHONPATH=code python3 code/rq3b_rq4_analysis.py --dataset <name>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
sys.path.insert(0, os.path.join(STAGE, "code"))

import metrics as M                      # noqa: E402

POOL = ("qwen2.5-7b", "llama-3-8b", "mistral-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b")
RECIPIENTS = ("qwen2.5-7b", "qwen2.5-32b")


def pool_decisions(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Optional[str]]]:
    """Per pair: each model's modal decision over its repeats, tie -> abstain."""
    cell: Dict[Tuple[str, str, str], List[Optional[str]]] = collections.defaultdict(list)
    for r in rows:
        cell[(r["a"], r["b"], r["model"])].append(r["decision"])
    out: Dict[Tuple[str, str], Dict[str, Optional[str]]] = collections.defaultdict(dict)
    for (a, b, m), ds in cell.items():
        out[(a, b)][m] = M.modal_decision(ds)[0]
    return out


def apply_rule(votes: Dict[str, Optional[str]], rule: str,
               weights: Optional[Dict[str, float]] = None) -> Optional[str]:
    avail = {m: d for m, d in votes.items() if d is not None}
    if not avail:
        return None
    if rule == "unanimity_with_abstention":
        s = set(avail.values())
        return next(iter(s)) if len(s) == 1 else None
    if rule == "plurality":
        c = collections.Counter(avail.values())
    elif rule == "weighted_vote":
        c = collections.Counter()
        for m, d in avail.items():
            c[d] += (weights or {}).get(m, 1.0)
    else:
        raise AssertionError(rule)
    top = max(c.values())
    win = sorted(k for k, v in c.items() if abs(v - top) < 1e-12)
    return win[0] if len(win) == 1 else None          # tie -> abstain, same as every other rule


def fit_weights(pd_dev: Dict[Tuple[str, str], Dict[str, Optional[str]]],
                ent: Dict[str, str]) -> Dict[str, Any]:
    """Per-model dev accuracy -> weight 2*acc-1, clipped at 0. Gold used on dev only."""
    hit = collections.Counter()
    tot = collections.Counter()
    for (a, b), votes in pd_dev.items():
        g = "match" if ent[a] == ent[b] else "non-match"
        for m, d in votes.items():
            if d is None:
                continue
            tot[m] += 1
            hit[m] += d == g
    acc = {m: hit[m] / tot[m] for m in tot}
    return {"accuracy": acc, "weights": {m: max(0.0, 2 * a - 1) for m, a in acc.items()},
            "n_decisions": dict(tot)}


def score(dec: Dict[Tuple[str, str], Optional[str]], ent: Dict[str, str], nodes: List[str],
          pv: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Dict[str, Any]:
    base = M.pair_scores(dec, ent)
    links = [(x, y) for (x, y), v in dec.items() if v == "match"]
    scored = [(x, y, M.vote_fraction(pv, x, y, POOL)) for x, y in links]
    base["n_links"] = len(links)
    base["scope_C_connected_components"] = M.cluster_scores(
        M.connected_components(nodes, links), ent, nodes)
    base["scope_C_center"] = M.cluster_scores(M.center_clustering(nodes, scored), ent, nodes)
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    data = os.path.join(STAGE, "data", a.dataset)
    blob = json.load(open(os.path.join(STAGE, "data/protocol",
                                       f"{a.dataset.replace('_e5p','')}.json"), encoding="utf-8"))
    ent, sp = blob["entity_of"], blob["split"]

    rep: Dict[str, Any] = {"dataset": a.dataset, "model_calls_made_by_this_script": 0,
                           "rq3b": {}, "rq4": {}}

    pools, arms_rows, nodes_of = {}, {}, {}
    for split in ("dev", "test"):
        p = os.path.join(data, f"pool_{split}.jsonl")
        if not os.path.exists(p):
            continue
        pools[split] = M.load_jsonl(p)
        arms_rows[split] = [r for r in M.load_jsonl(os.path.join(data, f"arms_{split}.jsonl"))
                            if r.get("role") == "recipient"]
        nodes_of[split] = [str(n) for n in blob["universe"]
                           if sp.get(ent.get(str(n))) == split]
    if "dev" not in pools:
        print("  no dev pool; nothing to do")
        return 1

    pd_of = {s: pool_decisions(rows) for s, rows in pools.items()}
    w = fit_weights(pd_of["dev"], ent)
    rep["rq3b"]["weights_fitted_on_dev"] = w
    print(f"  dev-fitted model weights: "
          + ", ".join(f"{m} {w['weights'][m]:.3f} (acc {w['accuracy'][m]:.4f})"
                      for m in sorted(w["weights"])))

    # ------------------------------------------------------------------ RQ3-B
    for split in pd_of:
        pv = M.pool_view(pools[split])
        nodes = nodes_of[split]
        ent_rows = {}
        for rule in ("plurality", "weighted_vote", "unanimity_with_abstention"):
            dec = {k: apply_rule(v, rule, w["weights"]) for k, v in pd_of[split].items()}
            s = score(dec, ent, nodes, pv)
            s["additional_recipient_calls"] = 0
            s["rule"] = rule
            ent_rows[rule] = s
        # reference: the best single pool model, and D5, on the same pairs
        for m in POOL:
            dec = {k: v.get(m) for k, v in pd_of[split].items()}
            ent_rows[f"single_model:{m}"] = score(dec, ent, nodes, pv)
        for rc in RECIPIENTS:
            d5 = M.arm_decisions(arms_rows[split], rc, "D5")
            if d5:
                ent_rows[f"D5:{rc}"] = score(
                    {k: d5[k] for k in pd_of[split] if k in d5}, ent, nodes, pv)
        rep["rq3b"][split] = {k: {kk: vv for kk, vv in v.items()
                                  if kk in ("f1", "precision", "recall", "abstention_rate",
                                            "n_links", "rule", "additional_recipient_calls")}
                              | {"cluster_f1_cc": v["scope_C_connected_components"]["f1"],
                                 "cluster_f1_center": v["scope_C_center"]["f1"]}
                              for k, v in ent_rows.items()}

    # ------------------------------------------------------------------ RQ4
    for split in pd_of:
        pv = M.pool_view(pools[split])
        nodes = nodes_of[split]
        out: Dict[str, Any] = {}
        for rc in RECIPIENTS:
            for arm, label in (("D5", "sharing"), ("N", "extra_generation")):
                byrep: Dict[Any, Dict[Tuple[str, str], Optional[str]]] = collections.defaultdict(
                    dict)
                rr = [r for r in arms_rows[split]
                      if r["recipient"] == rc and r["arm"] == arm and r.get("draw") in (None, 0)]
                for r in rr:
                    byrep[r["repeat"]][(r["a"], r["b"])] = r["decision"]
                if not byrep:
                    continue
                common = sorted(set.intersection(*(set(d) for d in byrep.values())))
                per_rep = [score({k: d[k] for k in common}, ent, nodes, pv)
                           for d in byrep.values()]
                modal = {k: M.modal_decision([d[k] for d in byrep.values()])[0] for k in common}
                s3 = score(modal, ent, nodes, pv)
                tok_in = sum(r["prompt_tokens"] for r in rr) / max(len(rr), 1)
                tok_out = sum(r["completion_tokens"] for r in rr) / max(len(rr), 1)
                out[f"{rc}|{label}|1_call"] = {
                    "arm": arm, "additional_recipient_calls": 1,
                    "pair_f1_mean_over_repeats": st.mean(p["f1"] for p in per_rep),
                    "pair_f1_per_repeat": [p["f1"] for p in per_rep],
                    "pair_f1_spread": max(p["f1"] for p in per_rep)
                    - min(p["f1"] for p in per_rep),
                    "cluster_f1_cc_mean": st.mean(
                        p["scope_C_connected_components"]["f1"] for p in per_rep),
                    "prompt_tokens_per_call": tok_in, "completion_tokens_per_call": tok_out}
                out[f"{rc}|{label}|3_calls"] = {
                    "arm": arm, "additional_recipient_calls": 3,
                    "aggregation": "modal over 3, tie -> abstain",
                    "pair_f1": s3["f1"], "precision": s3["precision"], "recall": s3["recall"],
                    "cluster_f1_cc": s3["scope_C_connected_components"]["f1"],
                    "cluster_f1_center": s3["scope_C_center"]["f1"],
                    "prompt_tokens_per_call": tok_in, "completion_tokens_per_call": tok_out,
                    "total_prompt_tokens": tok_in * 3, "total_completion_tokens": tok_out * 3}
        pool_calls = len(pools[split])
        pool_in = sum(r.get("prompt_tokens", 0) for r in pools[split])
        out["shared_pool_sunk_cost"] = {
            "calls": pool_calls, "prompt_tokens": pool_in,
            "note": "six models x three repeats, common to both sides, attributed to neither"}
        for rc in RECIPIENTS:
            for b in ("1_call", "3_calls"):
                ks, ke = f"{rc}|sharing|{b}", f"{rc}|extra_generation|{b}"
                if ks in out and ke in out:
                    kf = "pair_f1" if b == "3_calls" else "pair_f1_mean_over_repeats"
                    out[f"{rc}|sharing_minus_extra_generation|{b}"] = {
                        "pair_f1_delta": out[ks][kf] - out[ke][kf],
                        "prompt_tokens_delta_per_call":
                            out[ks]["prompt_tokens_per_call"] - out[ke]["prompt_tokens_per_call"],
                        "equal_calls_not_equal_tokens": True}
        rep["rq4"][split] = out

    print("\n=== RQ3-B: final answer use over the fixed six-model pool (0 extra calls) ===")
    for split, rows in rep["rq3b"].items():
        if split == "weights_fitted_on_dev":
            continue
        print(f"  -- {split}")
        print(f"     {'rule/reference':<34}{'pairF1':>9}{'abst':>7}{'ccF1':>9}{'centerF1':>10}")
        for k, v in rows.items():
            print(f"     {k:<34}{v['f1']:>9.4f}{v['abstention_rate']:>7.3f}"
                  f"{v['cluster_f1_cc']:>9.4f}{v['cluster_f1_center']:>10.4f}")
    print("\n=== RQ4: sharing vs extra generation at matched additional recipient calls ===")
    for split, rows in rep["rq4"].items():
        print(f"  -- {split}")
        for k, v in rows.items():
            if "sharing_minus" in k:
                print(f"     {k:<48} pairF1 {v['pair_f1_delta']:+.4f}  "
                      f"Δptok/call {v['prompt_tokens_delta_per_call']:+.0f}")

    p = os.path.join(STAGE, "output", f"rq3b_rq4_{a.dataset}.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
