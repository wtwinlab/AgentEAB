#!/usr/bin/env python3
"""Cost-quality trade-offs with explicit direction, units and stability. Zero model calls.

    python3 analyze_cost_tradeoff.py
"""
from __future__ import annotations

import collections
import csv
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

OUT = ROOT / "analysis/controlled_sharing"
#: below this many output tokens per unit, a quality-per-token ratio is not reported: the
#: denominator is too close to zero for the quotient to mean anything.
MIN_ABS_OUT_TOK = 25.0


def main() -> int:
    j = json.loads((OUT / "rq_results.json").read_text(encoding="utf-8"))
    cost = {(c["benchmark"], c["protocol"], c["backbone"], c["arm"]): c for c in j["cost"]}
    rows: List[Dict[str, Any]] = []
    for e in j["effects"]:
        if not e["convention"].startswith("abstention"):
            continue
        arm = e["contrast"].split()[0]
        k_full = (e["benchmark"], e["protocol"], e["backbone"], "FULL")
        k_arm = (e["benchmark"], e["protocol"], e["backbone"], arm)
        cf, ca = cost.get(k_full), cost.get(k_arm)
        if not cf or not ca:
            continue
        # rq_effects is `arm - FULL`; everything below is `FULL - arm`
        dq = -100.0 * e["effect"]
        dq_lo, dq_hi = -100.0 * e["hi"], -100.0 * e["lo"]
        d_out = cf["out_tok_per_unit"] - ca["out_tok_per_unit"]
        d_in = cf["in_tok_per_unit"] - ca["in_tok_per_unit"]
        d_calls = cf["calls_per_unit"] - ca["calls_per_unit"]
        q_sig = (dq_lo > 0) or (dq_hi < 0)
        cheaper_all = (d_out <= 0) and (d_in <= 0) and (d_calls <= 0)
        dearer_any = (d_out > 0) or (d_in > 0) or (d_calls > 0)
        if not q_sig:
            verdict = "quality not distinguished: no dominance claim"
        elif dq_lo > 0 and cheaper_all:
            verdict = "FULL dominates: better quality, cheaper on calls, input and output"
        elif dq_hi < 0 and (d_out >= 0 and d_in >= 0 and d_calls >= 0):
            verdict = "arm dominates: better quality and not more expensive"
        elif dq_lo > 0 and dearer_any:
            verdict = "FULL better, costs more on at least one axis: trade-off"
        else:
            verdict = "arm better, costs more on at least one axis: trade-off"
        ratio = (dq / (d_out / 1000.0)) if abs(d_out) >= MIN_ABS_OUT_TOK else None
        rows.append({
            "source": e.get("source"),
            "benchmark": e["benchmark"], "protocol": e["protocol"], "backbone": e["backbone"],
            "arm": arm, "dimension": e["dimension"], "metric": e["metric"],
            "degenerate_neither_delivered": e.get("degenerate_neither_delivered"),
            "n_delivered_full": e.get("n_delivered_full"),
            "n_delivered_arm": e.get("n_delivered_arm"),
            # When one condition delivers nothing at all, the entire quality difference is the
            # delivery difference: the protocol failed to produce a usable answer there, so the
            # cell says something about format compliance or budget, not about answer quality.
            "effect_is_entirely_delivery": bool(e.get("n_delivered_full") == 0
                                                or e.get("n_delivered_arm") == 0),
            "quality_FULL_minus_arm_pp": dq,
            "quality_ci_lo_pp": dq_lo, "quality_ci_hi_pp": dq_hi,
            "quality_ci_excludes_zero": q_sig,
            "holm_dimension_reject": e.get("holm_dimension_reject"),
            "d_calls_per_unit_FULL_minus_arm": d_calls,
            "d_out_tok_per_unit_FULL_minus_arm": d_out,
            "d_in_tok_per_unit_FULL_minus_arm": d_in,
            "pp_per_1k_out_tok": ratio,
            "ratio_reported": ratio is not None,
            "ratio_withheld_reason": (None if ratio is not None else
                                      "|d out tok/unit| = %.1f < %.0f floor" % (abs(d_out),
                                                                                MIN_ABS_OUT_TOK)),
            "verdict": verdict,
        })

    with open(OUT / "cost_tradeoff.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("CONVENTION: every delta is FULL minus arm.")
    print("  quality_FULL_minus_arm_pp > 0  -> the released sharing condition is BETTER")
    print("  d_*_FULL_minus_arm        < 0  -> the released sharing condition is CHEAPER")
    print("  a ratio is withheld when |d out tok/unit| < %.0f\n" % MIN_ABS_OUT_TOK)

    print("=== access (FULL vs NOPEER): dominance verdicts over all %d cells ==="
          % sum(1 for r in rows if r["arm"] == "NOPEER"))
    v = collections.Counter(r["verdict"] for r in rows if r["arm"] == "NOPEER")
    for k, n in v.most_common():
        print("   %-62s %d" % (k, n))
    print("\n=== every cell where one side dominates on all three resource axes ===")
    print("%-9s %-22s %-16s %-11s %9s %20s %8s %9s %9s"
          % ("arm", "benchmark", "protocol", "backbone", "dQ pp", "dQ 95% CI", "dcalls", "dout",
             "din"))
    for side in ("FULL dominates", "arm dominates"):
        sel = sorted([x for x in rows if x["verdict"].startswith(side)],
                     key=lambda x: -abs(x["quality_FULL_minus_arm_pp"]))
        print("   -- %s (%d cells)" % (side, len(sel)))
        for r in sel:
            print("%-9s %-22s %-16s %-11s %+9.2f [%+7.2f,%+7.2f] %+8.1f %+9.0f %+9.0f%s" % (
                r["arm"], r["benchmark"][:20], r["protocol"], r["backbone"],
                r["quality_FULL_minus_arm_pp"], r["quality_ci_lo_pp"], r["quality_ci_hi_pp"],
                r["d_calls_per_unit_FULL_minus_arm"], r["d_out_tok_per_unit_FULL_minus_arm"],
                r["d_in_tok_per_unit_FULL_minus_arm"],
                "   <- entirely a DELIVERY difference (%d vs %d delivered)"
                % (r["n_delivered_full"], r["n_delivered_arm"])
                if r["effect_is_entirely_delivery"] else ""))

    print("\n=== quality-per-token ratio, reported WITHIN a metric and never across metrics ===")
    print("   a point of QA accuracy, a point of pair F1 and a point of SQL execution accuracy are")
    print("   different quantities, so these ratios are grouped by metric: a spread computed across")
    print("   metrics would be comparing numerators that do not share a unit.")
    rep = [r for r in rows if r["ratio_reported"] and r["quality_ci_excludes_zero"]
           and r["arm"] == "NOPEER"]
    print("\n   reportable access cells: %d of %d (the rest have |d out tok| below the floor or "
          "an undistinguished quality difference)" % (len(rep), sum(1 for r in rows
                                                                    if r["arm"] == "NOPEER")))
    for metric in sorted({r["metric"] for r in rep}):
        sel = sorted([r for r in rep if r["metric"] == metric],
                     key=lambda x: x["pp_per_1k_out_tok"])
        print("\n   --- %s (%d cells) ---" % (metric, len(sel)))
        for r in sel:
            print("   %-22s %-16s %-11s %+9.2f pp / %+6.0f tok = %+9.1f pp per 1k out tok" % (
                r["benchmark"][:20], r["protocol"], r["backbone"],
                r["quality_FULL_minus_arm_pp"], r["d_out_tok_per_unit_FULL_minus_arm"],
                r["pp_per_1k_out_tok"]))
        pos = [r["pp_per_1k_out_tok"] for r in sel if r["pp_per_1k_out_tok"] > 0]
        neg = [r for r in sel if r["pp_per_1k_out_tok"] < 0]
        if len(pos) >= 2:
            print("      positive-ratio spread within this metric: %.1f to %.1f pp per 1k output "
                  "tokens (%.1fx)" % (min(pos), max(pos), max(pos) / min(pos)))
        elif len(pos) == 1:
            print("      only one positive ratio (%.1f), so there is no spread to report"
                  % pos[0])
        else:
            print("      no positive ratio: in every reportable cell of this metric one side is "
                  "both better and cheaper, so no price exists")
        if neg:
            print("      %d cell(s) with a negative ratio = one side both better and cheaper; "
                  "a 'price per quality point' is undefined there" % len(neg))
    print("\n   NOTE: a ratio is a quality-efficiency figure, not a price -- no token price is used")
    print("   anywhere in this study. Each row states its own denominator so near-zero cases are "
          "visible.")

    blob = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "convention": "all deltas are FULL minus arm; quality in percentage points",
            "min_abs_out_tok_for_ratio": MIN_ABS_OUT_TOK,
            "supersedes": "breakeven.csv",
            "metric_separation": "ratios and spreads are computed within a metric only; QA "
                                 "accuracy, pair F1(match) and SQL execution accuracy are not "
                                 "pooled, and no combined quality score is formed",
            "resource_axes": "calls/unit, input tokens/unit, output tokens/unit -- model resources "
                             "only. BIRD SQL additionally incurs query-execution cost, accounted "
                             "separately in experiments/02_controlled_sharing/extension_analysis/"
                             "sql_execution.json and NOT folded into these axes, so a dominance "
                             "verdict on SQL is a statement about model resources",
            "rows": rows}
    (OUT / "cost_tradeoff.json").write_text(json.dumps(blob, indent=1, default=str) + "\n",
                                            encoding="utf-8")
    print("\n-> %s" % (OUT / "cost_tradeoff.csv"))
    print("   sha256 %s" % hashlib.sha256((OUT / "cost_tradeoff.json").read_bytes()
                                          ).hexdigest()[:32])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
