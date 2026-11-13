#!/usr/bin/env python3
"""Availability loss and selection miss on a COMMON denominator. Zero model calls.

    python3 analyze_loss_vs_miss.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

TOTALS: Dict[Any, int] = {}

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "analysis/controlled_sharing"


def executions_with_gold() -> Dict[Any, int]:
    """FULL executions per cell that have a reference answer: the common denominator.
    """
    import inputs as IN
    out: Dict[Any, int] = collections.Counter()
    for bench in IN.BENCHMARKS:
        gold = IN.gold_for(bench)
        rows, _dropped = IN.read(bench)
        for r in rows:
            if r.get("arm") != "FULL" or r.get("unit") not in gold:
                continue
            out[(r["protocol"], r["backbone"], r["benchmark"])] += 1
    return out


def main() -> int:
    global TOTALS
    TOTALS = executions_with_gold()
    j = json.loads((OUT / "rq_results.json").read_text(encoding="utf-8"))
    # the FULL condition's diagnostics, deduplicated: the same FULL cell appears once per pairing
    seen = set()
    rows: List[Dict[str, Any]] = []
    no_opportunity: List[Any] = []
    for d in j["diagnostics"]:
        if d["condition"] != "FULL":
            continue
        k = (d["protocol"], d["backbone"], d["benchmark"])
        if k in seen:
            continue
        seen.add(k)
        A = d["n_with_initial_correct"] or 0
        O = d["n_correct_at_selection"] or 0
        loss_rate = d["availability_loss"]
        miss_rate = d["selection_miss"]
        # A rate over an empty opportunity set is NaN, not zero: a cell in which no execution ever
        # held a correct candidate has no loss rate to report rather than a loss rate of 0%.
        def _n(x: Any) -> Optional[float]:
            try:
                v = float(x)
            except (TypeError, ValueError):
                return None
            return None if v != v else v
        loss_rate, miss_rate = _n(loss_rate), _n(miss_rate)
        if loss_rate is None and miss_rate is None:
            no_opportunity.append((d["benchmark"], d["protocol"], d["backbone"]))
            continue
        L = round((loss_rate or 0.0) * A)
        B = round((miss_rate or 0.0) * O)
        # The common denominator is counted from the records, not inferred from the conditioned
        # sets: every FULL execution of this cell that has a reference answer, whether or not a
        # correct candidate ever existed.
        total = TOTALS.get((d["protocol"], d["backbone"], d["benchmark"]), 0)
        rows.append({
            "benchmark": d["benchmark"], "protocol": d["protocol"], "backbone": d["backbone"],
            "n_A_initially_had_correct": A, "n_L_lost_before_selection": L,
            "loss_rate_conditional_on_A": loss_rate,
            "n_O_correct_at_selection": O, "n_B_missed_by_the_rule": B,
            "miss_rate_conditional_on_O": miss_rate,
            "common_denominator_used": total,
            "loss_share_of_common": (L / total) if total else None,
            "miss_share_of_common": (B / total) if total else None,
            "denominators_differ": A != O,
        })

    with open(OUT / "loss_vs_miss.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("The two conditional rates are NOT comparable: loss conditions on A (initially had a")
    print("correct candidate), miss conditions on O (still had one at selection), and O is itself")
    print("shrunk by whatever L removed. The shares below use a common denominator.\n")
    print("%-22s %-16s %-11s %7s %7s %10s %10s %11s %11s" % (
        "benchmark", "protocol", "backbone", "|A|", "|O|", "loss|A", "miss|O",
        "loss share", "miss share"))
    for r in sorted(rows, key=lambda x: -(x["miss_share_of_common"] or 0))[:14]:
        print("%-22s %-16s %-11s %7d %7d %9.1f%% %9.1f%% %10.1f%% %10.1f%%" % (
            r["benchmark"][:20], r["protocol"], r["backbone"],
            r["n_A_initially_had_correct"], r["n_O_correct_at_selection"],
            100 * (r["loss_rate_conditional_on_A"] or 0),
            100 * (r["miss_rate_conditional_on_O"] or 0),
            100 * (r["loss_share_of_common"] or 0), 100 * (r["miss_share_of_common"] or 0)))

    if no_opportunity:
        print("\n   cells with NO opportunity set at all (never held a correct candidate, so "
              "neither rate is defined): %d" % len(no_opportunity))
        for b, pr, bb in no_opportunity:
            print("      %-14s %-16s %-11s" % (b, pr, bb))

    nd = sum(1 for r in rows if r["denominators_differ"])
    gt_cond = sum(1 for r in rows if (r["miss_rate_conditional_on_O"] or 0)
                  > (r["loss_rate_conditional_on_A"] or 0))
    gt_share = sum(1 for r in rows if (r["miss_share_of_common"] or 0)
                   > (r["loss_share_of_common"] or 0))
    print("\n   cells: %d | denominators differ in %d of them" % (len(rows), nd))
    print("   miss > loss using the CONDITIONAL rates (not a valid comparison): %d of %d"
          % (gt_cond, len(rows)))
    print("   miss > loss using the COMMON denominator (valid): %d of %d" % (gt_share, len(rows)))
    shares = sorted((r["miss_share_of_common"] or 0) for r in rows)
    print("   miss share of all executions: median %.1f%%, max %.1f%%"
          % (100 * shares[len(shares) // 2], 100 * shares[-1]))
    print("\n   The miss share is an UPPER BOUND on what a perfect oracle selector could recover.")
    print("   No fixed-candidate replay was run in this study, so there is no realisable-gain")
    print("   figure here to compare against the paper's 3.15 pp recovery rate.")

    blob = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "definitions": {
                "loss_rate_conditional_on_A": "|L|/|A|, A = executions that initially had a "
                                              "verified correct candidate",
                "miss_rate_conditional_on_O": "|B|/|O|, O = executions that still had a correct "
                                              "candidate at selection (the paper excludes the "
                                              "others from this denominator)",
                "shares": "the same counts divided by a common denominator, which is what the "
                          "paper prescribes for comparing contributions to overall failure",
                "interpretation": "the miss share bounds oracle-selector recovery from above; it "
                                  "is not a gain any actual rule achieves"},
            "cells": len(rows), "miss_gt_loss_conditional": gt_cond,
            "miss_gt_loss_common_denominator": gt_share, "rows": rows}
    (OUT / "loss_vs_miss.json").write_text(json.dumps(blob, indent=1, default=str) + "\n",
                                           encoding="utf-8")
    print("\n-> %s" % (OUT / "loss_vs_miss.csv"))
    print("   sha256 %s" % hashlib.sha256((OUT / "loss_vs_miss.json").read_bytes()
                                          ).hexdigest()[:32])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
