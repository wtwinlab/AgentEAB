r"""The common input-budget control: does the content advantage survive a shared input ceiling?
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
sys.path.insert(0, os.path.join(STAGE, "code"))

import metrics as M                      # noqa: E402

POOL = ("qwen2.5-7b", "llama-3-8b", "mistral-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b")
RECIPIENTS = ("qwen2.5-7b", "qwen2.5-32b")
# dataset -> (protocol name, directory holding the native conditions, native E5 condition label)
DATASETS = [("musicbrainz20k", "musicbrainz20k_e5p", "E5"),
            ("geographic_settlements", "geographic_settlements", "E5"),
            ("wdc_products", "wdc_products", "E5")]
FIELD = re.compile(r"-\s*([a-z_ ]+?)\s*(?:token overlap)?\s*:")


def main() -> int:
    rep: Dict[str, Any] = {
        "cap_block_chars": 380, "cap_tokens_nominal": 128,
        "chars_per_token_measured": 2.971,
        "cap_chosen_on": "dev, from the E5 block's median natural length; never from a test outcome",
        "control_type": "common input budget. Truncation drops whole trailing lines, so which "
                        "comparisons survive is decided by field order: this bounds the input-size "
                        "question and does not isolate content.",
        "reuse_rule": "pairs already within the ceiling reuse the stored arm, because the capped "
                      "payload would be byte-identical",
        "datasets": {}}

    for proto, native_dir, e5_label in DATASETS:
        cpath = os.path.join(STAGE, "data", proto, "controls_dev.jsonl")
        if not os.path.exists(cpath):
            rep["datasets"][proto] = {"status": "not collected"}
            print(f"  {proto}: controls not collected")
            continue
        ctl = M.load_jsonl(cpath)
        native = [r for r in M.load_jsonl(os.path.join(STAGE, "data", native_dir,
                                                       "arms_dev.jsonl"))
                  if r.get("role") == "recipient"]
        pool = M.load_jsonl(os.path.join(STAGE, "data", proto, "pool_dev.jsonl"))
        pv = M.pool_view(pool)
        blob = json.load(open(os.path.join(STAGE, "data/protocol", f"{proto}.json"),
                              encoding="utf-8"))
        ent, sp = blob["entity_of"], blob["split"]
        nodes = [str(n) for n in blob["universe"] if sp.get(ent.get(str(n))) == "dev"]

        entry: Dict[str, Any] = {"n_control_rows": len(ctl), "by_recipient": {},
                                 "truncation": {}}
        # which fields get dropped, over all control rows
        for arm in ("E5cap", "F5cap"):
            rows = [r for r in ctl if r["arm"] == arm]
            if not rows:
                continue
            dropped = collections.Counter()
            for r in rows:
                for ln in r.get("dropped_lines", []):
                    m = FIELD.match(ln.strip())
                    dropped[m.group(1).strip() if m else "other"] += 1
            entry["truncation"][arm] = {
                "rows_collected": len(rows),
                "mean_block_chars_before": sum(r["block_chars_before_cap"] for r in rows)
                / len(rows),
                "mean_block_chars_after": sum(r["block_chars_after_cap"] for r in rows) / len(rows),
                "mean_lines_dropped": sum(r["n_dropped_lines"] for r in rows) / len(rows),
                "dropped_line_leading_field_counts": dict(dropped.most_common(12)),
                "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / len(rows)}

        for rc in RECIPIENTS:
            # native conditions
            dec_native = {a: M.arm_decisions(native, rc, a) for a in ("D5", e5_label, "F5")}
            # capped conditions: collected cell where present, else the stored native condition
            out: Dict[str, Any] = {}
            for cap_arm, base_arm in (("E5cap", e5_label), ("F5cap", "F5")):
                collected = M.arm_decisions([r for r in ctl if r["arm"] == cap_arm], rc, cap_arm)
                base = dec_native[base_arm]
                merged = dict(base)
                merged.update(collected)
                n_coll = len(set(collected) & set(base))
                out[cap_arm] = {"decisions": merged, "n_from_capped_collection": n_coll,
                                "n_reused_within_cap": len(base) - n_coll,
                                "frac_truncated": n_coll / max(len(base), 1)}
            common = sorted(set(dec_native["D5"]) & set(dec_native[e5_label])
                            & set(dec_native["F5"]) & set(out["E5cap"]["decisions"])
                            & set(out["F5cap"]["decisions"]))
            scored: Dict[str, Any] = {}
            for name, dec in (("D5", dec_native["D5"]), ("E5_native", dec_native[e5_label]),
                              ("F5_native", dec_native["F5"]),
                              ("E5cap", out["E5cap"]["decisions"]),
                              ("F5cap", out["F5cap"]["decisions"])):
                d = {k: dec[k] for k in common}
                base = M.pair_scores(d, ent)
                links = [(x, y) for (x, y), v in d.items() if v == "match"]
                sc = [(x, y, M.vote_fraction(pv, x, y, POOL)) for x, y in links]
                base["cluster_f1_cc"] = M.cluster_scores(
                    M.connected_components(nodes, links), ent, nodes)["f1"]
                base["cluster_f1_center"] = M.cluster_scores(
                    M.center_clustering(nodes, sc), ent, nodes)["f1"]
                scored[name] = {k: base[k] for k in
                                ("f1", "precision", "recall", "cluster_f1_cc",
                                 "cluster_f1_center")}
            deltas = {
                "E5cap-D5": scored["E5cap"]["f1"] - scored["D5"]["f1"],
                "E5native-D5": scored["E5_native"]["f1"] - scored["D5"]["f1"],
                "E5cap-E5native": scored["E5cap"]["f1"] - scored["E5_native"]["f1"],
                "F5cap-D5": scored["F5cap"]["f1"] - scored["D5"]["f1"],
                "F5native-D5": scored["F5_native"]["f1"] - scored["D5"]["f1"],
                "F5cap-F5native": scored["F5cap"]["f1"] - scored["F5_native"]["f1"]}
            entry["by_recipient"][rc] = {
                "n_pairs": len(common), "arms": scored, "deltas": deltas,
                "assembly": {k: {kk: vv for kk, vv in v.items() if kk != "decisions"}
                             for k, v in out.items()}}
        rep["datasets"][proto] = entry

    print("=== common input-budget control (380-char block ceiling, ~128 tokens) ===")
    for proto, e in rep["datasets"].items():
        if e.get("status"):
            print(f"  {proto}: {e['status']}")
            continue
        print(f"\n  -- {proto}")
        for arm, t in e["truncation"].items():
            print(f"     {arm}: {t['rows_collected']} rows collected, block "
                  f"{t['mean_block_chars_before']:.0f} -> {t['mean_block_chars_after']:.0f} chars, "
                  f"{t['mean_lines_dropped']:.1f} lines dropped, {t['mean_prompt_tokens']:.0f} ptok")
            print(f"        dropped leading fields: {t['dropped_line_leading_field_counts']}")
        for rc, v in e["by_recipient"].items():
            a = v["arms"]
            print(f"     {rc}: D5 {a['D5']['f1']:.4f} | E5 native {a['E5_native']['f1']:.4f} "
                  f"capped {a['E5cap']['f1']:.4f} | F5 native {a['F5_native']['f1']:.4f} "
                  f"capped {a['F5cap']['f1']:.4f}")
            print(f"        truncated share: E5 {v['assembly']['E5cap']['frac_truncated']:.3f}, "
                  f"F5 {v['assembly']['F5cap']['frac_truncated']:.3f}")
            print(f"        deltas: " + ", ".join(f"{k} {x:+.4f}" for k, x in v["deltas"].items()))

    p = os.path.join(STAGE, "output", "controls_analysis.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
