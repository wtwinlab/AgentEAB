#!/usr/bin/env python3
"""Re-score the matching cells with the corrected label parser, from preserved raw output only.
"""
from __future__ import annotations

import argparse
import collections
import glob
import importlib.util
import json
import os
import pathlib
import statistics
import sys

NR = pathlib.Path(os.environ.get("RUN_DIR")
                  or (pathlib.Path(__file__).resolve().parents[3]
                      / "data" / "records" / "controlled_sharing"))
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "code"), str(ROOT / "experiments/06_cost_resources/code")]

import matching_tasks as MT  # noqa: E402

_spec = importlib.util.spec_from_file_location("alp", HERE / "audit_label_parse.py")
alp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(alp)
v2 = alp.to_label_v2

_spec2 = importlib.util.spec_from_file_location("em", HERE / "eval_matching.py")
em = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(em)

from consensus_delivery.adapters import (debate_or_vote, dmad, du_mad, mad_m2,  # noqa: E402
                                         reconcile)

POPS = {"wdc_products": 2615, "geographic_settlements": 4719, "musicbrainz20k": 32199}
#: left as collected, with the reason carried into the report
AS_COLLECTED = {
    "ford": "the contract field changes on 0 of 45,221 calls",
    "du_mad": "the contract field changes on 0 of 264,024 calls",
    "dov_vote": "the contract field changes on 0 of 391,300 calls",
    "bccs": "changes 0.0007-0.0033 of calls; its released vote weights by token beliefs that the "
            "stored reply text does not carry, so re-voting would substitute a different rule",
}


def last_round(calls):
    rounds = [c.get("round") for c in calls if c.get("round") is not None]
    if not rounds:
        return calls
    top = max(rounds)
    return [c for c in calls if c.get("round") == top]


def recompute(proto, rec):
    """The protocol's released terminal rule, over labels re-extracted with the corrected parser."""
    calls = rec.get("calls") or []
    if not calls:
        return None
    tid = rec.get("example_id") or ""

    if proto == "moa":
        agg = [c for c in calls if c.get("role") == "aggregator"] or calls[-1:]
        return v2(agg[-1].get("raw_reply")) or None

    if proto in ("madm2_naive", "madm2_objective"):
        answers = [v2(MT._after_answer(c.get("raw_reply"))) or None for c in last_round(calls)]
        _reached, winner = mad_m2.if_reach_consensus(answers)
        return winner

    if proto == "dmad":
        answers = [v2(dmad.parse_response(c.get("raw_reply") or "")[1]) for c in last_round(calls)]
        return dmad.majority(answers, tid)["answer"] or None

    if proto == "dov_vote":
        votes = [v2(debate_or_vote.bracketed(c.get("raw_reply") or "") or "")
                 for c in last_round(calls)]
        return debate_or_vote.tally(votes, tid)["answer"] or None

    if proto == "du_mad":
        answers = [v2(du_mad.boxed_group(c.get("raw_reply") or "") or "") or None
                   for c in last_round(calls)]
        return du_mad.most_frequent(answers)

    if proto == "reconcile":
        lr = last_round(calls)
        answers = [v2((c.get("parsed_json") or {}).get("answer")) for c in lr]
        confs = [(c.get("parsed_json") or {}).get("confidence_level") for c in lr]
        _tally, winner, _unusable = reconcile.weighted_vote(answers, confs)
        return winner or None

    if proto == "liang_mad":
        mod = [c for c in calls if c.get("role") == "moderator"]
        if mod:
            ans, _side = alp.moderator_answer(mod[-1].get("raw_reply") or "")
            lab = v2(ans)
            if lab:
                return lab
        # the released judge fallback runs when the moderator states no preference; its own reply is
        # the last call in that case
        return v2(calls[-1].get("raw_reply")) or None

    return None


def prf(pairs):
    tp = sum(1 for g, p in pairs if g == 1 and p == 1)
    fp = sum(1 for g, p in pairs if g == 0 and p == 1)
    fn = sum(1 for g, p in pairs if g == 1 and p == 0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", action="append")
    ap.add_argument("--population", action="append")
    a = ap.parse_args()

    gold = {}
    cells = collections.defaultdict(lambda: {"before": [], "after": [],
                                             "dec_before": 0, "dec_after": 0, "n": 0,
                                             "changed": 0, "agreement_changed": 0})
    for f in sorted(glob.glob(str(NR / "raw/*.jsonl"))):
        parts = pathlib.Path(f).stem.split("__")
        if len(parts) < 4:
            continue
        pop, proto, team, rep = parts[0], parts[1], parts[2], parts[3]
        if a.protocol and proto not in a.protocol:
            continue
        if a.population and pop not in a.population:
            continue
        if proto in AS_COLLECTED:
            continue
        if pop not in gold:
            gold[pop] = em.gold_for(pop)[0]
        ent = gold[pop]
        key = (pop, proto, team, rep)
        c = cells[key]
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok":
                continue
            rec = r.get("record") or {}
            u = r["unit_id"].split("|", 3)[3]
            bits = u.split("|")
            if len(bits) < 3 or bits[1] not in ent or bits[2] not in ent:
                continue
            g = 1 if ent[bits[1]] == ent[bits[2]] else 0
            c["n"] += 1
            old = rec.get("majority_answer")
            new = recompute(proto, rec)
            if old != new:
                c["changed"] += 1
            if old in ("match", "non-match"):
                c["before"].append((g, 1 if old == "match" else 0))
            else:
                c["dec_before"] += 1
            if new in ("match", "non-match"):
                c["after"].append((g, 1 if new == "match" else 0))
            else:
                c["dec_after"] += 1

    out = {"per_cell": {}, "by_protocol": {}, "left_as_collected": AS_COLLECTED}
    print(f"  {'cell':56s}{'n':>7s}{'F1 before':>11s}{'F1 after':>10s}{'delta':>9s}"
          f"{'dec before':>12s}{'dec after':>11s}")
    for key in sorted(cells):
        c = cells[key]
        if not c["n"]:
            continue
        pb, rb, fb = prf(c["before"])
        pa, ra, fa = prf(c["after"])
        name = "|".join(key)
        out["per_cell"][name] = {
            "units": c["n"], "labels_changed": c["changed"],
            "before": {"precision": pb, "recall": rb, "f1": fb,
                       "declined": round(c["dec_before"] / c["n"], 4)},
            "after": {"precision": pa, "recall": ra, "f1": fa,
                      "declined": round(c["dec_after"] / c["n"], 4)},
            "f1_delta": round(fa - fb, 4)}
        print(f"  {name:56s}{c['n']:7,d}{fb:11.4f}{fa:10.4f}{fa - fb:+9.4f}"
              f"{c['dec_before'] / c['n']:12.4f}{c['dec_after'] / c['n']:11.4f}")

    for proto in sorted({k.split("|")[1] for k in out["per_cell"]}):
        for pop in POPS:
            ks = [k for k in out["per_cell"]
                  if k.split("|")[1] == proto and k.split("|")[0] == pop
                  and out["per_cell"][k]["units"] >= POPS[pop]]
            if not ks:
                continue
            fb = [out["per_cell"][k]["before"]["f1"] for k in ks]
            fa = [out["per_cell"][k]["after"]["f1"] for k in ks]
            out["by_protocol"][f"{proto}|{pop}"] = {
                "complete_cells": len(ks),
                "f1_before": round(statistics.fmean(fb), 4),
                "f1_after": round(statistics.fmean(fa), 4),
                "f1_delta": round(statistics.fmean(fa) - statistics.fmean(fb), 4),
                "sd_after": round(statistics.stdev(fa), 4) if len(fa) > 1 else None,
                "declined_after": round(statistics.fmean(
                    [out["per_cell"][k]["after"]["declined"] for k in ks]), 4)}

    print(f"\n  {'workflow|population':44s}{'cells':>7s}{'before':>9s}{'after':>9s}{'delta':>9s}")
    for k, s in out["by_protocol"].items():
        print(f"  {k:44s}{s['complete_cells']:7d}{s['f1_before']:9.4f}{s['f1_after']:9.4f}"
              f"{s['f1_delta']:+9.4f}")

    p = NR / "analysis/rescore_v2.json"
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
