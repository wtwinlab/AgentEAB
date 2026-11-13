r"""Nested randomness diagnostic: how much does the single initial pool determine the results?
"""
from __future__ import annotations

import collections
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
sys.path.insert(0, os.path.join(STAGE, "code"))

import metrics as M                      # noqa: E402

POOL = ("qwen2.5-7b", "llama-3-8b", "mistral-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b")
RECIPIENTS = ("qwen2.5-7b", "qwen2.5-32b")
import argparse

NATIVE_OF = {"musicbrainz20k": "musicbrainz20k_e5p",
             "geographic_settlements": "geographic_settlements",
             "wdc_products": "wdc_products"}


def modal(ds: List[Optional[str]]) -> Optional[str]:
    v = [d for d in ds if d is not None]
    if not v:
        return None
    c = collections.Counter(v)
    top = max(c.values())
    w = [k for k, n in c.items() if n == top]
    return w[0] if len(w) == 1 else None


def majority_side(votes: Dict[str, Optional[str]]) -> Optional[str]:
    v = [d for d in votes.values() if d]
    if not v:
        return None
    c = collections.Counter(v)
    top = max(c.values())
    return sorted(k for k, n in c.items() if n == top)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musicbrainz20k", choices=list(NATIVE_OF))
    a = ap.parse_args()
    DATASET = a.dataset
    NATIVE = NATIVE_OF[DATASET]
    d = os.path.join(STAGE, "data", DATASET)
    p2 = os.path.join(d, "pool2_dev.jsonl")
    if not os.path.exists(p2):
        print("  pool2 not collected")
        return 1
    pool2 = M.load_jsonl(p2)
    pool1 = M.load_jsonl(os.path.join(d, "pool_dev.jsonl"))
    pairs = sorted({(r["a"], r["b"]) for r in pool2})
    blob = json.load(open(os.path.join(STAGE, "data/protocol", f"{DATASET}.json"),
                          encoding="utf-8"))
    ent = blob["entity_of"]

    def view(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Optional[str]]]:
        cell: Dict[Tuple[str, str, str], List[Optional[str]]] = collections.defaultdict(list)
        for r in rows:
            cell[(r["a"], r["b"], r["model"])].append(r["decision"])
        out: Dict[Tuple[str, str], Dict[str, Optional[str]]] = collections.defaultdict(dict)
        for (a, b, m), ds in cell.items():
            out[(a, b)][m] = modal(ds)
        return out

    v1, v2 = view(pool1), view(pool2)
    rep: Dict[str, Any] = {
        "dataset": DATASET, "n_pairs": len(pairs),
        "n_pairs_in_full_dev": len({(r["a"], r["b"]) for r in pool1}),
        "sampling": "frozen 10% of dev candidate pairs, seed 20261016",
        "not_additional_sample_size": True,
        "pool2_calls": len(pool2),
        "input_divergence": {}, "outcome_sensitivity": {}}

    # ------------------------------------------------------------------ 1. input divergence
    per_model = {}
    for m in POOL:
        agree = tot = 0
        for k in pairs:
            a, b = v1.get(k, {}).get(m), v2.get(k, {}).get(m)
            if k in v1 and k in v2 and m in v1[k] and m in v2[k]:
                tot += 1
                agree += a == b
        per_model[m] = {"n": tot, "agreement": agree / max(tot, 1)}
    maj_agree = sum(1 for k in pairs
                    if majority_side(v1.get(k, {})) == majority_side(v2.get(k, {})))
    vf_diff = []
    for k in pairs:
        d1 = [x for x in v1.get(k, {}).values() if x]
        d2 = [x for x in v2.get(k, {}).values() if x]
        if d1 and d2:
            f1 = sum(1 for x in d1 if x == "match") / len(d1)
            f2 = sum(1 for x in d2 if x == "match") / len(d2)
            vf_diff.append(abs(f1 - f2))
    rep["input_divergence"] = {
        "per_model_modal_agreement": per_model,
        "majority_side_agreement": maj_agree / max(len(pairs), 1),
        "vote_fraction_mean_abs_difference": sum(vf_diff) / max(len(vf_diff), 1),
        "vote_fraction_exactly_equal": sum(1 for x in vf_diff if x == 0) / max(len(vf_diff), 1),
        "note": "the vote fraction orders edges for Center clustering, so its stability bounds how "
                "much the alternative grouping rule depends on the pool draw"}
    print(f"=== input divergence over {len(pairs)} pairs "
          f"({100*len(pairs)/rep['n_pairs_in_full_dev']:.1f}% of dev) ===")
    for m, s in per_model.items():
        print(f"  {m:<14} modal agreement {s['agreement']:.4f}  (n={s['n']})")
    print(f"  AD1 majority side agreement: {rep['input_divergence']['majority_side_agreement']:.4f}")
    print(f"  vote fraction: mean |Δ| {rep['input_divergence']['vote_fraction_mean_abs_difference']:.4f}, "
          f"exactly equal {100*rep['input_divergence']['vote_fraction_exactly_equal']:.1f}%")

    # ------------------------------------------------------------------ 2. outcome sensitivity
    na = os.path.join(d, "nestedarms_dev.jsonl")
    if not os.path.exists(na):
        rep["outcome_sensitivity"] = {"status": "D5-on-pool2 not collected yet"}
        print("\n  D5 on pool2 not collected yet; outcome sensitivity pending")
    else:
        rows2 = M.load_jsonl(na)
        native = [r for r in M.load_jsonl(os.path.join(STAGE, "data", NATIVE, "arms_dev.jsonl"))
                  if r.get("role") == "recipient" and r.get("arm") == "D5"]
        print(f"\n=== outcome sensitivity: D5 under two independent pools, same {len(pairs)} pairs ===")
        for rc in RECIPIENTS:
            d1 = M.arm_decisions([r for r in native if r["recipient"] == rc], rc, "D5")
            d2 = M.arm_decisions([r for r in rows2 if r["recipient"] == rc], rc, "D5p2")
            common = sorted(set(d1) & set(d2) & set(pairs))
            if not common:
                continue
            s1 = M.pair_scores({k: d1[k] for k in common}, ent)
            s2 = M.pair_scores({k: d2[k] for k in common}, ent)
            flip = sum(1 for k in common if d1[k] != d2[k])
            corr = sum(1 for k in common
                       if (d2[k] == ("match" if ent[k[0]] == ent[k[1]] else "non-match"))
                       and (d1[k] != ("match" if ent[k[0]] == ent[k[1]] else "non-match")))
            harm = sum(1 for k in common
                       if (d1[k] == ("match" if ent[k[0]] == ent[k[1]] else "non-match"))
                       and (d2[k] != ("match" if ent[k[0]] == ent[k[1]] else "non-match")))
            rep["outcome_sensitivity"][rc] = {
                "n_pairs": len(common),
                "pair_f1_pool1": s1["f1"], "pair_f1_pool2": s2["f1"],
                "pair_f1_difference": s2["f1"] - s1["f1"],
                "decision_flips": flip, "flip_rate": flip / len(common),
                "pool2_corrections": corr, "pool2_harms": harm}
            print(f"  {rc}: D5 pair F1 {s1['f1']:.4f} (pool 1) vs {s2['f1']:.4f} (pool 2), "
                  f"Δ {s2['f1']-s1['f1']:+.4f}; {flip} of {len(common)} decisions flip "
                  f"({100*flip/len(common):.1f}%), +{corr}/-{harm}")

    p = os.path.join(STAGE, "output", f"nested_pool_analysis_{DATASET}.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
