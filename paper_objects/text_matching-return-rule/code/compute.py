"""Return rules on stored matching runs (matching part of Section sec:qa-vote-replay).

    python3 code/compute.py --out DIR      # -> DIR/values.json, DIR/sharing_effects.csv

Inputs (inputs/):
  return_rule_runs.csv   one row per admissible run of Du-MAD, MAD-M2 and BCCS on WDC and
                         Settlements (conditions FULL, NOPEER, ANSONLY; two backbones, two repeats
                         per matching unit): the answer the protocol returns and the all-round
                         majority over every call's answer, with ties going to the first agent,
                         to the last agent, or kept at the protocol's answer
  return_rule_units.csv  one row per matching unit (candidate pair): reference label

Sharing effect = pair F1 (match class) with full sharing minus pair F1 in the other condition,
pooled over the runs of the units present in both conditions; a run without an answer predicts
non-match. The effect is computed under the protocol's own return rule and under each majority
variant.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
RULES = (("released", "protocol"), ("majority_first", "majority_first"),
         ("majority_last", "majority_last"), ("majority_keep", "majority_kept"))
PROTOS = {"du_mad": "dumad", "madm2_objective": "madm2", "bccs": "bccs"}
BENCH = {"wdc_products": "wdc", "geographic_settlements": "settlements"}


def rd(name):
    with open(INP / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    units = collections.defaultdict(list)
    gold = {}
    for r in rd("return_rule_units.csv"):
        units[r["benchmark"]].append(r["unit"])
        gold[(r["benchmark"], r["unit"])] = r["gold"] == "match"
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rd("return_rule_runs.csv"):
        by[(r["benchmark"], r["protocol"], r["condition"])][r["unit"]].append(r)

    def level(bench, rows_by_unit, keep, col):
        tp = fp = fn = 0
        for u in keep:
            g = gold[(bench, u)]
            for r in rows_by_unit[u]:
                pred = r[col] == "match"
                tp += pred and g
                fp += pred and not g
                fn += (not pred) and g
        return f1(tp, fp, fn)

    vals, table = {}, []
    for bench, bt in BENCH.items():
        for proto, pt in PROTOS.items():
            full = by[(bench, proto, "FULL")]
            for cond in ("NOPEER", "ANSONLY"):
                other = by[(bench, proto, cond)]
                keep = [u for u in units[bench] if full.get(u) and other.get(u)]
                rec = dict(benchmark=bench, protocol=proto, condition=cond, units=len(keep))
                for col, name in RULES:
                    e = 100 * (level(bench, full, keep, col) - level(bench, other, keep, col))
                    rec[f"effect_{name}_pp"] = e
                    vals[f"{bt}_{pt}_{cond.lower()}_effect_{name}_pp"] = e
                for col, name in RULES[1:]:
                    rec[f"move_{name}_pp"] = rec[f"effect_{name}_pp"] - rec["effect_protocol_pp"]
                table.append(rec)
    moves = [abs(r[f"move_{n}_pp"]) for r in table for _, n in RULES[1:]]
    vals["sharing_effects"] = len(table)
    vals["max_abs_move_pp"] = max(moves)
    vals["min_abs_move_pp"] = min(moves)
    vals["max_abs_move_ties_first_pp"] = max(abs(r["move_majority_first_pp"]) for r in table)
    vals["min_abs_move_ties_first_pp"] = min(abs(r["move_majority_first_pp"]) for r in table)
    with open(out / "sharing_effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    (out / "values.json").write_text(json.dumps(vals, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
