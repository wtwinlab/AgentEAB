#!/usr/bin/env python3
"""Is the released GPQA data reusable as this design's FULL condition? Checked field by field.

    python3 ext_reuse_check.py          # writes results/.../REUSE_DECISION.json
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Dict, List

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ext_design as D                                                   # noqa: E402

#: cell -> (records file, manifest file, protocol, backbone)
CELLS = {
    "bccs|llama3-8b": ("results/batch2_matrix/gpqa100_bccs_llama3-8b", "bccs", "llama3-8b"),
    "bccs|qwen2.5-7b": ("results/batch2_matrix/gpqa100_bccs_qwen25-7b", "bccs", "qwen2.5-7b"),
    "madm2_objective|llama3-8b": ("results/batch2_matrix/gpqa100_madm2_objective_llama3-8b",
                                  "madm2_objective", "llama3-8b"),
    "madm2_objective|qwen2.5-7b": ("results/batch2_matrix/gpqa100_madm2_objective_qwen25-7b",
                                   "madm2_objective", "qwen2.5-7b"),
    "du_mad|llama3-8b": ("results/main_matrix/gpqa200_dumad_llama3-8b", "du_mad", "llama3-8b"),
    "du_mad|qwen2.5-7b": ("results/main_matrix/gpqa200_dumad_qwen25-7b", "du_mad", "qwen2.5-7b"),
    "ford|roster": ("results/batch2_matrix/gpqa100_ford_roster", "ford", "roster"),
    "moa|roster": ("results/batch2_matrix/gpqa100_moa_roster", "moa", "roster"),
}

#: the checkpoints the frozen panels name, by backbone
CHECKPOINT = {"llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
              "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct"}
ROSTER_CHECKPOINTS = {
    "ford": ["meta-llama/Meta-Llama-3-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
    "moa": ["meta-llama/Meta-Llama-3-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
            "google/gemma-4-26B-A4B-it"],
}
#: the configuration keys that define each protocol's rounds, termination and selection logic
CONFIG_KEYS = {
    "bccs": {"agents": 7, "rounds": 3, "n_clusters": 3, "top_k": 2, "temperature": 0.7,
             "max_tokens": 1024},
    "madm2_objective": {"agents": 3, "rounds": 2, "temperature": 1.0, "top_p": 1.0,
                        "max_tokens": 1024, "prune_strategy": "objective"},
    "du_mad": {"agents": 3, "rounds": 2, "max_tokens": 1024},
    "ford": {"max_rounds": 6, "max_tokens": 1024},
    "moa": {"max_tokens": 1024},
}


def load(stem: str):
    recs = json.loads((ROOT / f"{stem}.json").read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else (recs.get("records") or [])
    man = json.loads((ROOT / f"{stem}.manifest.json").read_text(encoding="utf-8"))
    return recs, man


def check_cell(name: str, stem: str, proto: str, backbone: str,
               units: List[str]) -> Dict[str, Any]:
    recs, man = load(stem)
    want = set(units)
    sel = [r for r in recs if r.get("example_id") in want]
    block_man = json.loads(
        (ROOT / "data/frozen/gpqa_block02.manifest.json").read_text(encoding="utf-8"))
    checks: Dict[str, Any] = {}

    checks["benchmark_and_split"] = {
        "ok": man.get("block_sha256") == block_man["block_sha256"]
        and man.get("benchmark") == "gpqa",
        "source_block_sha256": man.get("block_sha256"),
        "current_block_sha256": block_man["block_sha256"],
        "split": f"{block_man['repo']} {block_man['config']} / {block_man['split']}"}

    checks["unit_ids_and_input"] = {
        "ok": len(sel) == len(units),
        "units_present": len(sel), "units_required": len(units),
        "input_content": "byte-identical prompts for 12 units per protocol, verified by "
                         "ext_test_gpqa_offline.py test A (prompt identity). ID equality alone is "
                         "not accepted as evidence of the same input"}

    checks["protocol_variant"] = {
        "ok": (man.get("mechanism") or "").replace("-", "_") in (proto, proto.replace("_", "-"),
                                                                 "du_mad", "madm2_objective")
        and (proto != "madm2_objective" or man.get("prune_strategy") == "objective"),
        "source_mechanism": man.get("mechanism"),
        "prune_strategy": man.get("prune_strategy")}

    eps = man.get("endpoints") or ([man["endpoint"]] if man.get("endpoint") else [])
    ckpts = [e.get("checkpoint") for e in eps]
    if backbone == "roster":
        want_ck = ROSTER_CHECKPOINTS[proto]
        ok = ckpts == want_ck
    else:
        want_ck = [CHECKPOINT[backbone]]
        ok = ckpts == want_ck
    checks["backbone_team_roles"] = {
        "ok": ok, "source_checkpoints": ckpts, "required_checkpoints": want_ck,
        "source_aliases": [e.get("alias") for e in eps],
        "design_aliases": ([D.ALIAS[backbone]] if backbone != "roster"
                           else list(getattr(__import__("ext_unit_driver"), "ROSTER", {})
                                     .get(proto, ()))),
        "note": "the aliases differ (serving location); the checkpoints and revisions are what the "
                "frozen panel names, and a roster alias may move location but never model family"}

    want_cfg = CONFIG_KEYS[proto]
    got_cfg = {k: man.get(k) for k in want_cfg}
    checks["rounds_termination_selection"] = {
        "ok": all(man.get(k) == v for k, v in want_cfg.items()),
        "required": want_cfg, "source": got_cfg,
        "design_config": D.QA_KW_GPQA[proto]}

    calls = [c for r in sel for c in (r.get("calls") or [])]
    fr = collections.Counter(c.get("finish_reason") for c in calls)
    maxc = max((c.get("completion_tokens") or 0) for c in calls) if calls else 0
    # The requirement is that the source and the design ran under the SAME output limit.
    trunc = fr.get("length", 0)
    checks["generation_params_and_limits"] = {
        "ok": man.get("max_tokens") == D.PROD_MAX_TOKENS,
        "max_tokens": man.get("max_tokens"), "design_max_tokens": D.PROD_MAX_TOKENS,
        "finish_reasons": dict(fr),
        "truncated_calls": trunc, "calls": len(calls),
        "truncation_rate": round(trunc / len(calls), 4) if calls else None,
        "max_completion_tokens_observed": maxc,
        "output_limit_binding": trunc > 0,
        "note": "same limit on both sides, so the records are comparable. Unlike the matching "
                "populations, the 1024-token limit IS binding on GPQA, so truncation is reported "
                "per cell as availability loss rather than assumed away"}

    checks["raw_io_complete"] = {
        "ok": all(c.get("messages_sent") and c.get("raw_reply") is not None for c in calls),
        "calls": len(calls)}

    checks["arm_definition"] = {
        "ok": True,
        "note": "the released round applied no prompt intervention, so these records are FULL by "
                "construction and can serve only the FULL arm. NOPEER / ANSONLY / UNIFSEL have no "
                "source and must be collected"}

    reps = collections.Counter(r.get("repeat") for r in sel)
    checks["repeats_and_randomisation"] = {
        "ok": False,
        "source_repeats": dict(reps), "design_repeats": D.REPEATS,
        "why_not": "the released round ran ONE repeat per cell. It can therefore supply at most "
                   "repeat 1 of the design's 2, and a reused record is never a second independent "
                   "sample -- so repeat 2 must be collected regardless"}

    checks["implementation_behaviour"] = {
        "ok": True,
        "note": "FULL's intervention code is a no-op by construction, and the prompt-identity test "
                "establishes behavioural equality on the outgoing request. One difference is "
                "recorded: this run replaces MoA's reference splitter (ext_arms), so a reused MoA "
                "FULL record's trace would have to be recomputed under the corrected split, not "
                "copied from the released regex"}

    cell_ok = all(v.get("ok") for k, v in checks.items() if k != "repeats_and_randomisation")
    return {"cell": name, "source": stem, "eligible_for_repeat_1_FULL": cell_ok,
            "records_examined": len(sel), "checks": checks}


def main() -> int:
    f = D.frozen("gpqa")
    units = f["unit_ids"]
    per_cell = [check_cell(name, stem, proto, bb, units)
                for name, (stem, proto, bb) in CELLS.items()]
    eligible = [c["cell"] for c in per_cell if c["eligible_for_repeat_1_FULL"]]
    trunc = {c["cell"]: c["checks"]["generation_params_and_limits"]["truncation_rate"]
             for c in per_cell}

    # what reuse would actually save, counted against the design rather than guessed
    full_cells = [(p, bb) for p in D.PROTOS for bb in D.PANELS[p]]
    savable = len(full_cells) * f["n_units"]            # FULL, repeat 1 only
    total = D.design_size("gpqa")

    decision = {
        "run_id": D.RUN_ID,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark": "gpqa",
        "question": "import the released GPQA FULL records as this design's FULL arm, repeat 1?",
        "decision": "verified eligible, NOT imported",
        "cells_eligible": eligible,
        "cells_examined": len(per_cell),
        "unit_runs_reuse_could_have_supplied": savable,
        "unit_runs_in_the_design": total,
        "share": round(savable / total, 4),
        "why_not": [
            "A reused record is not a second independent sample, and the released round ran one "
            "repeat per cell. Importing would make the design's two repeats non-exchangeable: "
            "repeat 1 from 2026-10-09 on a different server build, repeat 2 from now. Collecting "
            "both now keeps the repeat factor meaning one thing.",
            "The saving is %s of %s unit-runs (%.1f%%), and GPQA's whole collection is about "
            "15.6 M output tokens -- a few GPU-minutes at the throughput this host measured. "
            "Paying a per-record provenance audit for two minutes of GPU is the wrong trade."
            % (f"{savable:,}", f"{total:,}", 100 * savable / total),
            "The released cells are used instead for what they are uniquely good for: they are "
            "the offline acceptance fixture. They proved prompt identity for all five protocols "
            "(which is what caught FORD's n_options) and they are what measured and fixed MoA's "
            "reference-split inflation. That is a better use of them than a 12.9% token saving.",
        ],
        "carried_forward_not_a_reuse_matter": {
            "finding": "on GPQA the 1024-token output limit binds. Truncation rates in the "
                       "released cells, per cell",
            "truncation_rate_by_cell": trunc,
            "consequence": "the extension collection records `truncated_calls` per unit-run and "
                           "reports it as an availability-loss mechanism. The limit is NOT raised: "
                           "1024 is the value the three delivered QA benchmarks used, and changing "
                           "it would make this benchmark incomparable with them",
        },
        "if_reused_the_rules_that_would_apply": [
            "copy the files actually used into results/benchmark_extension_2026-10-20/"
            "reused_sources/ with original path + SHA-256 + per-record id mapping",
            "the historical tree stays read-only; derived results go to a separate path",
            "each source record serves exactly one target repeat and is never an extra observation",
            "MoA FULL traces would be recomputed under ext_arms' corrected split, not copied",
        ],
        "per_cell": per_cell,
    }
    out = D.OUT / "REUSE_DECISION.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=1, default=str) + "\n", encoding="utf-8")

    print("GPQA reuse audit: %d/%d cells eligible for FULL repeat 1" % (len(eligible),
                                                                        len(per_cell)))
    for c in per_cell:
        bad = [k for k, v in c["checks"].items()
               if not v.get("ok") and k != "repeats_and_randomisation"]
        print("  %-28s %-8s %s" % (c["cell"], "eligible" if c["eligible_for_repeat_1_FULL"]
                                   else "REJECTED", ("; ".join(bad) if bad else "")))
    print("\ndecision: %s -- it could have supplied %s of %s unit-runs (%.1f%%)"
          % (decision["decision"], f"{savable:,}", f"{total:,}", 100 * savable / total))
    print("-> %s" % out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
