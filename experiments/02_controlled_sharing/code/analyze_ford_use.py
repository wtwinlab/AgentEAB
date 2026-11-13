#!/usr/bin/env python3
"""FORD's fourth dimension: the final-answer rule, recomputed from traces. Zero model calls.

    python3 analyze_ford_use.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import inputs as IN                                                      # noqa: E402

OUT = IN.OUT


def rules(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], int]:
    """(released rule's answer, all-round majority, n predictions) -- the collector's own function."""
    import arm_transform as AT
    got = AT.use_arm_final_answer("ford", record)
    return got["released_rule_answer"], got["all_round_majority"], got["n_predictions"]


def main() -> int:
    import analyze_research_questions as ARQ
    print(IN.ext_gate(require=True), "\n")
    rows: List[Dict[str, Any]] = []
    recomputed = stored_ok = stored_bad = 0

    for bench in IN.BENCHMARKS:
        kind = IN.kind_for(bench)
        gold = IN.gold_for(bench)
        recs, _drop = IN.read(bench, apply_transform=False)
        ford = [r for r in recs if r["protocol"] == "ford"]
        if not ford:
            continue
        by_arm: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in ford:
            by_arm[r["arm"]].append(r)

        for arm, rs in sorted(by_arm.items()):
            # per (unit, repeat): was each rule right?
            per: Dict[Tuple[str, int], Dict[str, Optional[bool]]] = {}
            agree = 0
            npred: List[int] = []
            for r in rs:
                rec = r.get("record") or {}
                rel, maj, n = rules(rec)
                npred.append(n)
                st = r.get("ford_use_contrast")
                if st is None:
                    recomputed += 1
                elif (st.get("released_rule_answer") == rel
                        and st.get("all_round_majority") == maj):
                    stored_ok += 1
                else:
                    stored_bad += 1
                agree += 1 if rel == maj else 0
                per[(r["unit"], r["repeat"])] = {
                    "released": IN.score_answer(bench, r, rel, gold, r["unit"], kind),
                    "majority": IN.score_answer(bench, r, maj, gold, r["unit"], kind),
                    "rel_raw": rel, "maj_raw": maj,
                }
            if not per:
                continue

            if IN.is_matching(bench):
                # F1 is set-level, so the difference is between two recomputed scores, not an
                # average of per-unit differences
                trips = [(u, p["maj_raw"], p["rel_raw"], gold.get(u))
                         for (u, _rp), p in per.items() if gold.get(u) is not None]
                trips = [(u, ARQ.norm(a, kind), ARQ.norm(b, kind), g) for u, a, b, g in trips]
                est = ARQ.paired_f1(trips, charge=True)
                metric = IN.METRIC[bench]
                level_rel, level_maj = est.get("f1_0"), est.get("f1_1")
            else:
                bu: Dict[str, Dict[str, List[float]]] = collections.defaultdict(
                    lambda: {"0": [], "1": []})
                for (u, _rp), p in per.items():
                    for tag, key in (("1", "majority"), ("0", "released")):
                        v = p[key]
                        bu[u][tag].append(1.0 if v else 0.0)      # charged: unscorable = failure
                est = ARQ.paired_effect(bu)
                metric = IN.METRIC[bench]
                level_rel = sum(1 for p in per.values() if p["released"]) / len(per)
                level_maj = sum(1 for p in per.values() if p["majority"]) / len(per)

            rows.append({
                "source": IN.source_for(bench).tag, "benchmark": bench, "metric": metric,
                "protocol": "ford", "backbone": rs[0]["backbone"], "arm_context": arm,
                "dimension": "selection/use (final-answer rule)",
                "contrast": "all-round majority vs released two-consecutive rule",
                "convention": "abstention charged as failure",
                "unit_runs": len(per), "n_units": est["n_units"],
                "rules_agree_rate": agree / len(rs),
                "mean_predictions_per_run": sum(npred) / len(npred),
                "level_released_rule": level_rel, "level_all_round_majority": level_maj,
                "effect_majority_minus_released": est["effect"],
                "lo": est["lo"], "hi": est["hi"], "p_boot": est.get("p_boot"),
                "ci_excludes_zero": (est["lo"] > 0 or est["hi"] < 0),
                "extra_model_calls": 0,
            })

    print("recomputation vs the collector's stored field:")
    print("   stored and reproduced exactly : %s" % f"{stored_ok:,}")
    print("   stored and NOT reproduced     : %s" % f"{stored_bad:,}")
    print("   absent, recomputed from traces: %s  (the A-group-imported FULL arms of WDC and "
          "Settlements, which never passed through the collector's hook)" % f"{recomputed:,}")

    print("\n=== FORD's final-answer rule, per benchmark (charged convention) ===")
    print("%-14s %-11s %-9s %7s %9s %9s %9s %22s %7s"
          % ("benchmark", "backbone", "context", "runs", "agree", "released", "majority",
             "majority - released", "CI≠0"))
    for r in rows:
        print("%-14s %-11s %-9s %7d %8.1f%% %9.4f %9.4f  %+8.4f [%+6.4f,%+6.4f] %7s"
              % (r["benchmark"][:14], r["backbone"], r["arm_context"], r["unit_runs"],
                 100 * r["rules_agree_rate"], r["level_released_rule"],
                 r["level_all_round_majority"], r["effect_majority_minus_released"],
                 r["lo"], r["hi"], "yes" if r["ci_excludes_zero"] else "no"))

    full = [r for r in rows if r["arm_context"] == "FULL"]
    sig = [r for r in full if r["ci_excludes_zero"]]
    print("\n   on the FULL arm, %d of %d benchmark x backbone cells show a distinguishable "
          "difference" % (len(sig), len(full)))
    for r in sig:
        print("      %-14s %-11s %+.4f [%+.4f, %+.4f] (%s)"
              % (r["benchmark"], r["backbone"], r["effect_majority_minus_released"],
                 r["lo"], r["hi"], r["metric"]))
    print("   every cell costs 0 extra model calls: both rules read the same calls.")

    with open(OUT / "ford_use_contrast.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    blob = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "why_not_a_prompt_arm": "FORD renders one peer, so UNIFSEL has nothing to choose "
                                    "between; both designs exclude the cell, which is why each "
                                    "benchmark has 19 (protocol, arm) combinations, not 20. No "
                                    "FORD/UNIFSEL prompt arm was ever collected in either design.",
            "what_is_measured": "the released stop-on-two-consecutive-agreeing rule against the "
                                "majority over every round's predictions, both read off the same "
                                "calls",
            "recomputation": {"stored_and_reproduced": stored_ok,
                              "stored_and_not_reproduced": stored_bad,
                              "absent_and_recomputed": recomputed},
            "sql_note": "on BIRD the majority is taken over query strings and only the winner is "
                        "executed; a majority over execution verdicts would be a rule that reads "
                        "the gold",
            "separate_from_the_161": "this contrast is NOT in the 161-test families: it is a "
                                     "different intervention (a decision rule, not a prompt edit) "
                                     "on a different estimand, and folding it in would pad a "
                                     "family with a non-comparable hypothesis",
            "rows": rows}
    (OUT / "ford_use_contrast.json").write_text(json.dumps(blob, indent=1, default=str) + "\n",
                                                encoding="utf-8")
    print("\n-> %s" % (OUT / "ford_use_contrast.csv"))
    print("   sha256 %s" % hashlib.sha256((OUT / "ford_use_contrast.csv").read_bytes()
                                          ).hexdigest()[:32])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
