#!/usr/bin/env python3
"""Group A: the whole-protocol matrix. Quality, delivery, abstention, coverage and cost. Zero calls.

    python3 analyze_group_a.py --inventory    # coverage and the comparable set, no scoring
    python3 analyze_group_a.py --full         # also score quality and cost
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SOURCE_RUN = "protocol_all_matching_20261018T192619Z"
SRC = ROOT / "workspace" / SOURCE_RUN / "raw"
OUT = ROOT / "analysis/controlled_sharing"
OKMARK = '"status": "ok"'
POPS = ("wdc_products", "geographic_settlements", "musicbrainz20k")
#: 10 conditions, 9 families: the two MAD-M^2 entries are pruning variants of one released protocol
ARMS = ["ford", "moa", "reconcile", "du_mad", "liang_mad", "dmad", "dov_vote",
        "madm2_objective", "madm2_naive", "bccs"]
FAMILY = {"ford": "FORD", "moa": "MoA", "reconcile": "ReConcile", "du_mad": "Du-MAD",
          "liang_mad": "Liang-MAD", "dmad": "DMAD", "dov_vote": "Debate-or-Vote",
          "madm2_objective": "MAD-M2", "madm2_naive": "MAD-M2", "bccs": "BCCS"}
HETERO = {"ford", "moa", "reconcile"}
CKPT = {"llama3-8b-instruct": "llama3-8b", "qwen2.5-7b-instruct": "qwen2.5-7b",
        "gemma4-26b-a4b-it": "gemma4"}
BOOT = 2000
SEED = 20261019


def norm_answer(a: Any) -> Optional[str]:
    s = str(a if a is not None else "").strip().lower().replace("_", "-")
    if s in ("", "none", "null"):
        return None
    return s


def gold_map(pop: str) -> Dict[str, str]:
    """`match`/`non-match` per unit. The population stores 0/1; `eval_matching.py` sets the convention
`pred = 1 if ans == "match" else 0`, so the gold is mapped into that vocabulary."""
    import collect_all_matching as C
    C._ensure_installed()
    rows, _sha = C.load_population(pop)
    out = {}
    for row in rows:
        k = row.get("unit_id") if isinstance(row, dict) else row[0]
        g = row.get("gold") if isinstance(row, dict) else (row[2] if len(row) > 2 else None)
        if k is None or g is None:
            continue
        out[k] = "match" if str(g).strip() in ("1", "true", "True", "match") else "non-match"
    return out


def scan() -> Tuple[Dict[Tuple[str, str, str, int], Dict[str, Any]], Dict[str, Any]]:
    """`cells[(pop, condition, checkpoint, repeat)] = {...}` plus a problems report.
    """
    cells: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    problems = {"non_ok_records": collections.Counter(),
                "unparseable_lines": 0,
                "mixed_checkpoint_records": 0,
                "unknown_checkpoint": collections.Counter(),
                "duplicate_unit_in_cell": collections.Counter()}
    for shard in sorted(glob.glob(str(SRC / "*.jsonl"))):
        name = pathlib.Path(shard).name[:-6].split("__")
        if len(name) < 4:
            continue
        pop, arm = name[0], name[1]
        if pop not in POPS or arm not in ARMS:
            continue
        for line in open(shard, encoding="utf-8"):
            if not line.strip():
                continue
            if OKMARK not in line:
                try:
                    r = json.loads(line)
                    problems["non_ok_records"][(pop, arm, r.get("status"))] += 1
                except Exception:
                    problems["unparseable_lines"] += 1
                continue
            try:
                r = json.loads(line)
            except Exception:
                problems["unparseable_lines"] += 1
                continue
            parts = (r.get("unit_id") or "").split("|")
            if len(parts) < 6:
                continue
            unit = "|".join(parts[3:])
            rec = r.get("record") or {}
            calls = rec.get("calls") or []
            served = {c.get("served_model") for c in calls if c.get("served_model")}
            if arm in HETERO:
                ck = "roster"
            elif len(served) != 1:
                problems["mixed_checkpoint_records"] += 1
                continue
            else:
                ck = CKPT.get(next(iter(served)), "")
                if not ck:
                    problems["unknown_checkpoint"][next(iter(served))] += 1
                    continue
            key = (pop, arm, ck, r.get("repeat"))
            d = cells.setdefault(key, {
                "units": {}, "calls": 0, "in_tok": 0, "out_tok": 0, "wall": 0.0,
                "truncated": 0, "retry": 0, "n": 0, "teams": set(),
                "stop_reasons": collections.Counter()})
            if unit in d["units"]:
                problems["duplicate_unit_in_cell"][key] += 1
                continue
            d["units"][unit] = norm_answer(rec.get("majority_answer"))
            d["n"] += 1
            d["calls"] += r.get("n_calls") or len(calls)
            d["in_tok"] += r.get("prompt_tokens_total") or 0
            d["out_tok"] += r.get("completion_tokens_total") or 0
            d["wall"] += r.get("wall_s") or 0.0
            d["truncated"] += r.get("truncated_calls") or 0
            d["retry"] += r.get("retry_calls") or 0
            d["teams"].add(r.get("team"))
            d["stop_reasons"][(rec.get("native_stop_reason") or "?")[:40]] += 1
    return cells, problems


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(units: Dict[str, Optional[str]], gold: Dict[str, str], on: List[str],
          charge_abstentions: bool) -> Dict[str, Any]:
    tp = fp = fn = tn = delivered = abstained = 0
    for u in on:
        g = gold.get(u)
        if g is None:
            continue
        a = units.get(u)
        if a is None:
            abstained += 1
            if not charge_abstentions:
                continue
            a = "non-match"          # charged: an abstention counts as a negative prediction
        else:
            delivered += 1
        pred = 1 if a == "match" else 0
        truth = 1 if g == "match" else 0
        if pred and truth:
            tp += 1
        elif pred:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    p, r, f = prf(tp, fp, fn)
    n = tp + fp + fn + tn
    return {"precision": p, "recall": r, "f1": f, "n_scored": n,
            "delivered": delivered, "abstained": abstained,
            "accuracy": (tp + tn) / n if n else float("nan")}


def f1_ci(units: Dict[str, Optional[str]], gold: Dict[str, str], on: List[str],
          charge: bool, reps: int = BOOT) -> Tuple[float, float]:
    rng = random.Random(SEED)
    vals = []
    for _ in range(reps):
        sel = [on[rng.randrange(len(on))] for _ in on]
        vals.append(score(units, gold, sel, charge)["f1"])
    vals.sort()
    if len(vals) < 20:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cells, problems = scan()
    print("scanned %d cells in %.0fs" % (len(cells), time.time() - t0))

    # ---------------- coverage inventory
    inv = []
    for (pop, arm, ck, rep), d in sorted(cells.items()):
        inv.append({"population": pop, "arm": arm, "family": FAMILY[arm],
                    "team_kind": "roster" if arm in HETERO else "homogeneous",
                    "checkpoint": ck, "repeat": rep, "units": d["n"],
                    "teams": sorted(x for x in d["teams"] if x),
                    "calls": d["calls"], "in_tok": d["in_tok"], "out_tok": d["out_tok"],
                    "truncated_calls": d["truncated"], "retry_calls": d["retry"]})
    with open(OUT / "group_a_coverage.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in inv[0] if k != "teams"])
        w.writeheader()
        for row in inv:
            w.writerow({k: v for k, v in row.items() if k != "teams"})

    print("\n=== configurations present, per population x arm (units per repeat) ===")
    for pop in POPS:
        print("  -- %s --" % pop)
        for arm in ARMS:
            rows = [r for r in inv if r["population"] == pop and r["arm"] == arm]
            if not rows:
                print("     %-18s ABSENT" % arm)
                continue
            by = collections.defaultdict(dict)
            for r in rows:
                by[r["checkpoint"]][r["repeat"]] = r["units"]
            print("     %-18s %s" % (arm, {k: dict(sorted(v.items())) for k, v in sorted(by.items())}))

    print("\n=== problems, listed separately and never folded into a denominator ===")
    print("   unparseable lines: %d" % problems["unparseable_lines"])
    print("   records with a mixed checkpoint (excluded): %d" % problems["mixed_checkpoint_records"])
    print("   unknown checkpoints: %s" % dict(problems["unknown_checkpoint"]))
    print("   duplicate unit inside one cell (first kept): %d cells, %d records"
          % (len(problems["duplicate_unit_in_cell"]), sum(problems["duplicate_unit_in_cell"].values())))
    nok = collections.Counter()
    for (pop, arm, st), c in problems["non_ok_records"].items():
        nok[(arm, st)] += c
    print("   non-ok records by (arm, status): %s" % dict(nok.most_common(10)))

    blob = {"source_run": SOURCE_RUN, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                    time.gmtime()),
            "coverage": inv,
            "problems": {"unparseable_lines": problems["unparseable_lines"],
                         "mixed_checkpoint_records": problems["mixed_checkpoint_records"],
                         "unknown_checkpoint": dict(problems["unknown_checkpoint"]),
                         "duplicate_unit_records": sum(problems["duplicate_unit_in_cell"].values()),
                         "non_ok_by_arm_status": {f"{k[0]}|{k[1]}": v for k, v in nok.items()}}}

    if not a.full:
        (OUT / "group_a_inventory.json").write_text(json.dumps(blob, indent=1, default=str) + "\n",
                                                    encoding="utf-8")
        print("\n--inventory only; scoring skipped. -> %s" % (OUT / "group_a_inventory.json"))
        return 0

    # ---------------- comparable set and scoring
    results = []
    for pop in POPS:
        gold = gold_map(pop)
        # the comparable set: units every condition covered, at the repeat level each condition actually has
        per_arm_units = {}
        for arm in ARMS:
            us = set()
            first = True
            for (p2, a2, ck, rep), d in cells.items():
                if p2 != pop or a2 != arm:
                    continue
                us = set(d["units"]) if first else (us & set(d["units"]))
                first = False
            per_arm_units[arm] = us
        # Two intersections, both named.
        pop_size = max((len(v) for v in per_arm_units.values()), default=0)
        well_covered = [a_ for a_ in ARMS
                        if per_arm_units[a_] and len(per_arm_units[a_]) >= 0.9 * pop_size]
        thin = [a_ for a_ in ARMS if a_ not in well_covered]
        inter_all = set.intersection(*[per_arm_units[a_] for a_ in ARMS if per_arm_units[a_]]) \
            if all(per_arm_units[a_] for a_ in ARMS) else set()
        inter_major = set.intersection(*[per_arm_units[a_] for a_ in well_covered]) \
            if well_covered else set()
        inter_all = sorted(u for u in inter_all if u in gold)
        inter_major = sorted(u for u in inter_major if u in gold)
        use_major = len(inter_all) < 0.5 * pop_size and len(inter_major) > len(inter_all)
        inter = inter_major if use_major else inter_all
        scope = ("intersection of the %d arms covering >=90%% of the population"
                 % len(well_covered)) if use_major else "intersection of all 10 arms"
        print("\n=== %s (population %d) ===" % (pop, pop_size))
        print("     all-10-arm intersection: %d units" % len(inter_all))
        print("     >=90%%-coverage arms (%d): %s" % (len(well_covered), ", ".join(well_covered)))
        if thin:
            print("     EXCLUDED as thin, reported separately: %s"
                  % ", ".join("%s(%d)" % (a_, len(per_arm_units[a_])) for a_ in thin))
        print("     scoring on: %s -> %d units" % (scope, len(inter)))
        for arm in ARMS:
            print("     %-18s own coverage %6d units" % (arm, len(per_arm_units[arm])))
        for (p2, arm, ck, rep), d in sorted(cells.items()):
            if p2 != pop:
                continue
            on = inter if inter else sorted(u for u in d["units"] if u in gold)
            thin_arm = arm not in well_covered
            if not on:
                continue
            deliv = score(d["units"], gold, on, charge_abstentions=False)
            charged = score(d["units"], gold, on, charge_abstentions=True)
            lo, hi = f1_ci(d["units"], gold, on, charge=False)
            results.append({
                "population": pop, "arm": arm, "family": FAMILY[arm],
                "team_kind": "roster" if arm in HETERO else "homogeneous",
                "checkpoint": ck, "repeat": rep,
                "scored_on": scope if inter else "its own units (no usable intersection)",
                "arm_coverage_units": len(per_arm_units[arm]),
                "thin_coverage": thin_arm,
                "n_units_scored": len(on),
                "f1_delivered": deliv["f1"], "f1_lo": lo, "f1_hi": hi,
                "precision_delivered": deliv["precision"], "recall_delivered": deliv["recall"],
                "f1_abstentions_charged": charged["f1"],
                "delivered": deliv["delivered"], "abstained": deliv["abstained"],
                "delivery_rate": deliv["delivered"] / len(on),
                "calls_per_unit": d["calls"] / max(d["n"], 1),
                "in_tok_per_unit": d["in_tok"] / max(d["n"], 1),
                "out_tok_per_unit": d["out_tok"] / max(d["n"], 1),
                "wall_s_per_unit": d["wall"] / max(d["n"], 1),
                "truncated_calls": d["truncated"],
            })
    with open(OUT / "group_a_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    blob["results"] = results
    (OUT / "group_a_results.json").write_text(json.dumps(blob, indent=1, default=str) + "\n",
                                              encoding="utf-8")
    sha = hashlib.sha256((OUT / "group_a_results.json").read_bytes()).hexdigest()

    print("\n=== per arm, averaged over the repeats of each checkpoint (intersection units) ===")
    print("%-18s %-12s %-13s %8s %8s %8s %9s %9s" % ("arm", "checkpoint", "population",
          "F1 deliv", "F1 chg", "deliv%", "calls/u", "out tok/u"))
    agg = collections.defaultdict(list)
    for r in results:
        agg[(r["arm"], r["checkpoint"], r["population"])].append(r)
    for k in sorted(agg):
        rs = agg[k]
        print("%-18s %-12s %-13s %8.4f %8.4f %7.1f%% %9.1f %9.0f"
              % (k[0], k[1], k[2][:12],
                 sum(x["f1_delivered"] for x in rs) / len(rs),
                 sum(x["f1_abstentions_charged"] for x in rs) / len(rs),
                 100 * sum(x["delivery_rate"] for x in rs) / len(rs),
                 sum(x["calls_per_unit"] for x in rs) / len(rs),
                 sum(x["out_tok_per_unit"] for x in rs) / len(rs)))
    print("\n-> %s" % (OUT / "group_a_results.json"))
    print("   sha256 %s" % sha[:32])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
