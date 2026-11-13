#!/usr/bin/env python3
"""Endpoint evaluation for the three common matching populations. Reference labels enter only here.

    python3 eval_matching.py                      # complete cells only
    python3 eval_matching.py --allow-partial      # include short cells, paired by intersection
    python3 eval_matching.py --protocol ford      # one workflow
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "code"),
                str(ROOT / "experiments/04_sql_selection_generation/code"),
                str(ROOT / "experiments/06_cost_resources/code"),
                str(ROOT / "experiments/05_matching_downstream/code"),
                str(HERE)]

NR = ROOT / "results/protocol_all_matching_20261018T192619Z"
RAW = NR / "raw"
POPS = {"wdc_products": 2615, "geographic_settlements": 4719, "musicbrainz20k": 32199}
HET = {"ford", "moa", "reconcile"}
ORDER = ["ford", "moa", "reconcile", "du_mad", "liang_mad", "madm2_objective",
         "madm2_naive", "bccs", "dov_vote", "dmad"]

#: per-call field that holds a workflow's OWN extracted answer. Resolved per workflow because the
#: adapters do not share one name; a single guessed key silently reports 0% contract satisfaction.
CONTRACT_KEY = {
    "bccs": ("extracted_answer",), "madm2_naive": ("extracted_answer",),
    "madm2_objective": ("extracted_answer",), "du_mad": ("extracted_answer",),
    "dov_vote": ("vote_released", "vote_strict"), "ford": ("prediction",),
    "moa": ("extracted_answer", "answer"), "reconcile": ("answer",),
    "dmad": ("normalised_answer", "parsed_answer"),
    # Liang MAD has NO per-call answer field: its answer is the moderator's JSON at the record
    # level, so a per-call contract rate does not exist for it.
    "liang_mad": (),
}


def gold_for(pop):
    proto = (ROOT / "data/records/matching_protocol" /
             f"{pop}.json")
    blob = json.loads(proto.read_text(encoding="utf-8"))
    return blob["entity_of"], blob["split"]


#: set from --relabel; `None` reads the stored answer, which is what collection's parser produced
RELABEL = None


def _relabeller():
    """`rescore_v2.recompute`, loaded lazily so the as-collected path needs no adapter import."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rescore_v2", HERE / "rescore_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.recompute, mod.AS_COLLECTED


def availability(env):
    st = env.get("status")
    if st != "ok":
        e = str(env.get("error") or env.get("reason") or st)
        return ("transport_failed" if "TransportFailure" in e else "collection_error"), None
    rec = env.get("record") or {}
    ans = rec.get("majority_answer")
    if RELABEL is not None:
        recompute, as_collected = RELABEL
        proto = env.get("protocol")
        if proto not in as_collected:
            try:
                ans = recompute(proto, rec)
            except Exception:
                # a unit whose rule cannot be re-applied keeps its collected answer, and the
                # fallback is counted rather than silently equated with a successful re-parse
                ans = rec.get("majority_answer")
    return ("delivered", ans) if ans not in (None, "") else ("declined", None)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (round(p, 6), round(r, 6),
            round(2 * p * r / (p + r), 6) if p + r else 0.0)


def contract_rate(env, proto):
    keys = CONTRACT_KEY.get(proto, ("extracted_answer",))
    if not keys:                     # no per-call answer field exists for this workflow
        return None, 0
    calls = (env.get("record") or {}).get("calls") or []
    if not calls:
        return None, 0
    got = 0
    for c in calls:
        v = next((c.get(k) for k in keys if c.get(k) not in (None, "", "None")), None)
        got += 1 if v is not None else 0
    return got, len(calls)


def load(protocols=None):
    cells = collections.defaultdict(dict)
    for f in sorted(RAW.glob("*.jsonl")):
        parts = f.stem.split("__")
        if len(parts) < 4:
            continue
        if protocols and parts[1] not in protocols:
            continue
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = (r.get("population") or parts[0], r.get("protocol") or parts[1],
                   r.get("team") or parts[2], r.get("repeat", parts[3]))
            u = r["unit_id"].split("|", 3)[3]
            cur = cells[key].get(u)
            if cur is None or (cur.get("status") != "ok" and r.get("status") == "ok"):
                cells[key][u] = r
    return cells


def score_cell(units, pop, ent, split, proto):
    av = collections.Counter()
    deliv = collections.Counter()
    charged = collections.Counter()
    cost = collections.defaultdict(list)
    cg, ct = 0, 0
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = 0
    for k, env in units.items():
        state, ans = availability(env)
        av[state] += 1
        a, b = k.split("|")[-2:]
        g = 1 if (ent.get(a) == ent.get(b)) else 0
        if state == "ok":
            pass
        if state == "delivered":
            pred = 1 if ans == "match" else 0
            cell = "tp" if (pred and g) else "fp" if pred else "fn" if g else "tn"
            deliv[cell] += 1
            charged[cell] += 1
            if pred:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
                edges += 1
        elif state == "declined":
            charged["fn" if g else "tn"] += 1
        if env.get("status") == "ok":
            cost["calls"].append(env["n_calls"])
            cost["out"].append(env["completion_tokens_total"])
            cost["inp"].append(env["prompt_tokens_total"])
            cost["wall"].append(env["wall_s"])
            rec = env.get("record") or {}
            rr = rec.get("rounds_run") or len(rec.get("rounds_detail") or []) or rec.get("rounds")
            if rr:
                cost["rounds"].append(rr)
            got, tot = contract_rate(env, proto)
            if tot:
                cg += got
                ct += tot
    pd_, rd, fd = prf(deliv["tp"], deliv["fp"], deliv["fn"])
    pc, rc, fc = prf(charged["tp"], charged["fp"], charged["fn"])
    comps = collections.Counter(find(x) for x in parent)
    n = sum(av.values())
    return {
        "n_units": n,
        "availability": dict(av),
        "delivered_share": round(av["delivered"] / n, 6) if n else None,
        "declined_share": round(av["declined"] / n, 6) if n else None,
        "transport_failed_share": round(av["transport_failed"] / n, 6) if n else None,
        "pair_delivered_only": {**deliv, "precision": pd_, "recall": rd, "f1": fd,
                                "n": sum(deliv.values())},
        "pair_declines_charged": {**charged, "precision": pc, "recall": rc, "f1": fc,
                                  "n": sum(charged.values())},
        "contract_satisfaction": round(cg / ct, 6) if ct else None,
        "contract_key": list(CONTRACT_KEY.get(proto, ("extracted_answer",))),
        "cost": {k: round(statistics.fmean(v), 3) for k, v in cost.items() if v},
        "induced_grouping": {"predicted_match_edges": edges, "records_touched": len(parent),
                             "components": len(comps),
                             "largest_component": max(comps.values()) if comps else 0,
                             "complete_population": n >= POPS[pop]},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--protocol", action="append")
    ap.add_argument("--out", default=str(NR / "analysis/matching_evaluation.json"))
    ap.add_argument("--relabel", choices=("v2", "as-collected"), default="v2",
                    help="v2 re-derives the delivered answer from the preserved replies with the "
                         "corrected parser and re-applies each protocol's released terminal rule; "
                         "as-collected reproduces the pre-correction numbers for audit")
    a = ap.parse_args()
    global RELABEL
    RELABEL = _relabeller() if a.relabel == "v2" else None

    cells = load(set(a.protocol) if a.protocol else None)
    gold = {p: gold_for(p) for p in POPS}
    report = {"mode": "partial_included_paired" if a.allow_partial else "complete_cells_only",
              "relabel": a.relabel,
              "relabel_note": (
                  "delivered answers re-derived from the preserved replies with the corrected "
                  "label parser, each protocol's released terminal rule re-applied; FORD, Du-MAD, "
                  "DoV and BCCS read as collected, see rescore_v2.AS_COLLECTED"
                  if a.relabel == "v2" else
                  "answers exactly as collection's parser produced them, kept for audit"),
              "cells_excluded": [], "per_cell": {}, "by_protocol": {}}

    use = {}
    for key, units in cells.items():
        pop = key[0]
        n_ok = sum(1 for v in units.values() if v.get("status") == "ok")
        if n_ok >= POPS[pop] or a.allow_partial:
            use[key] = units
        else:
            report["cells_excluded"].append(
                {"cell": "|".join(map(str, key)), "units": len(units),
                 "expected": POPS[pop], "included": False})

    keep = None
    if a.allow_partial:
        per_proto = collections.defaultdict(set)
        for (pop, pr, _t, _r), u in use.items():
            per_proto[(pop, pr)] |= set(u)
        by_pop = collections.defaultdict(list)
        for (pop, pr), s in per_proto.items():
            by_pop[pop].append(s)
        keep = {pop: set.intersection(*ss) if ss else set() for pop, ss in by_pop.items()}

    for key, units in sorted(use.items()):
        pop, pr = key[0], key[1]
        u = {k: v for k, v in units.items() if keep is None or k in keep.get(pop, set())}
        if not u:
            continue
        ent, split = gold[pop]
        report["per_cell"]["|".join(map(str, key))] = score_cell(u, pop, ent, split, pr)

    for pr in ORDER:
        ks = [k for k in report["per_cell"] if k.split("|")[1] == pr]
        if not ks:
            continue
        agg = {}
        for pop in POPS:
            pk = [k for k in ks if k.split("|")[0] == pop]
            if not pk:
                continue
            f1 = [report["per_cell"][k]["pair_delivered_only"]["f1"] for k in pk]
            fc = [report["per_cell"][k]["pair_declines_charged"]["f1"] for k in pk]
            dec = [report["per_cell"][k]["declined_share"] for k in pk]
            tf = [report["per_cell"][k]["transport_failed_share"] for k in pk]
            con = [report["per_cell"][k]["contract_satisfaction"] for k in pk
                   if report["per_cell"][k]["contract_satisfaction"] is not None]
            cal = [report["per_cell"][k]["cost"].get("calls") for k in pk
                   if report["per_cell"][k]["cost"].get("calls")]
            out = [report["per_cell"][k]["cost"].get("out") for k in pk
                   if report["per_cell"][k]["cost"].get("out")]
            agg[pop] = {
                "cells": len(pk),
                "units_per_cell": [report["per_cell"][k]["n_units"] for k in pk],
                "pair_f1_delivered_only": {
                    "mean": round(statistics.fmean(f1), 6),
                    "sd": round(statistics.stdev(f1), 6) if len(f1) > 1 else None},
                "pair_f1_declines_charged": {"mean": round(statistics.fmean(fc), 6)},
                "declined_share": round(statistics.fmean(dec), 6),
                "transport_failed_share": round(statistics.fmean(tf), 6),
                "contract_satisfaction": round(statistics.fmean(con), 6) if con else None,
                "calls_per_unit": round(statistics.fmean(cal), 3) if cal else None,
                "out_tokens_per_unit": round(statistics.fmean(out), 1) if out else None,
            }
        report["by_protocol"][pr] = agg

    op = pathlib.Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"mode: {report['mode']}   scored cells: {len(report['per_cell'])}   "
          f"excluded: {len(report['cells_excluded'])}")
    hdr = (f"\n{'workflow':17s}{'population':24s}{'cells':>6s}{'units':>8s}"
           f"{'F1 deliv':>10s}{'F1 charged':>12s}{'declined':>10s}{'transport':>11s}"
           f"{'contract':>10s}{'calls/u':>9s}")
    print(hdr)
    for pr, agg in report["by_protocol"].items():
        for pop, s in agg.items():
            print(f"  {pr:15s}{pop:24s}{s['cells']:6d}{min(s['units_per_cell']):8,d}"
                  f"{s['pair_f1_delivered_only']['mean']:10.4f}"
                  f"{s['pair_f1_declines_charged']['mean']:12.4f}"
                  f"{s['declined_share']:10.4f}{s['transport_failed_share']:11.4f}"
                  f"{(s['contract_satisfaction'] if s['contract_satisfaction'] is not None else 0):10.4f}"
                  f"{(s['calls_per_unit'] or 0):9.2f}")
    print(f"\n-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
