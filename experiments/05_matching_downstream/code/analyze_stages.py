r"""Stages A, B and C on the declared test universe, plus topology and the sharing contrasts.

    PYTHONPATH=experiments/05_matching_downstream/code python3.13 \
      experiments/05_matching_downstream/code/analyze_stages.py

The three stages exist because the WDC comparison moved two things at once. Here they are separated
by construction:

  A  the 4,719 candidate pairs only; precision, recall, F1, abstention, counts
  B  the fixed 1,115,271-pair universe over the 1,494 test records, positives = asserted links only
  C  the same universe, positives = every pair implied by connected components

A to B is a **scoring-universe change**. B to C is a **merge-operator effect**. They are never added
together or reported as one number.

Stage A resamples gold entities, because a candidate pair belongs to the pair of clusters it joins
and a record sits in many pairs. Stages B and C get **no interval**: closure couples the entities a
method merges, so an entity bootstrap would no longer describe the estimator. That is a declared
limit from the freeze, not an omission.

Gold is read only here, by the evaluator. Nothing in this file feeds a prompt or a policy.
"""
from __future__ import annotations

import collections
import csv
import itertools
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gs_common as G

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(BASE, "raw")
OUT = os.path.join(BASE, "analysis")
DRAWS = 4000
SEED = 20261013


class DSU:
    def __init__(self) -> None:
        self.p: Dict[int, int] = {}

    def find(self, x: int) -> int:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def components(nodes: Sequence[int], edges: Iterable[Tuple[int, int]]) -> Dict[int, List[int]]:
    d = DSU()
    for n in nodes:
        d.find(n)
    for a, b in edges:
        d.union(a, b)
    out: Dict[int, List[int]] = collections.defaultdict(list)
    for n in nodes:
        out[d.find(n)].append(n)
    return dict(out)


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r,
            "f1": (2 * p * r / (p + r)) if p + r else 0.0}


def modal(vals: List[Optional[str]]) -> Optional[str]:
    v = [x for x in vals if x is not None]
    if not v:
        return None
    c = collections.Counter(v)
    t = max(c.values())
    w = [k for k, n in c.items() if n == t]
    return w[0] if len(w) == 1 else None


def jl(name: str) -> List[Dict[str, Any]]:
    out = []
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


# ---------------------------------------------------------------- the three stages
def stage_ABC(ds: G.Settlements, nodes: List[int], cand: List[Tuple[int, int]],
              asserted: Set[str], gold_pairs_total: int) -> Dict[str, Any]:
    """One method's three readings. `asserted` is the set of pair keys it predicted positive."""
    # A: candidate pairs only
    tp = fp = fn = 0
    for (x, y) in cand:
        pk = ds.pair_key(x, y)
        g = ds.gold_pair(x, y)
        pos = pk in asserted
        tp += pos and g == 1
        fp += pos and g == 0
        fn += (not pos) and g == 1
    A = {**prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn,
         "n_scored": len(cand), "gold_positives_in_scope": tp + fn}

    # B: the full universe, direct links only. Every pair not asserted is predicted negative, so
    # the only false negatives are gold pairs that were not asserted anywhere in the universe.
    btp = sum(1 for pk in asserted
              if ds.gold_pair(*(int(z) for z in pk.split("|"))) == 1)
    bfp = len(asserted) - btp
    bfn = gold_pairs_total - btp
    B = {**prf(btp, bfp, bfn), "tp": btp, "fp": bfp, "fn": bfn,
         "n_scored": len(nodes) * (len(nodes) - 1) // 2,
         "predicted_positive": len(asserted), "gold_positives_in_scope": gold_pairs_total}

    # C: the same universe after closure
    edges = [tuple(int(z) for z in pk.split("|")) for pk in asserted]
    comp = components(nodes, edges)
    ctp = cn = 0
    for mem in comp.values():
        if len(mem) < 2:
            continue
        cn += len(mem) * (len(mem) - 1) // 2
        by = collections.Counter(ds.gold_cluster(m) for m in mem)
        ctp += sum(v * (v - 1) // 2 for v in by.values())
    cfp = cn - ctp
    cfn = gold_pairs_total - ctp
    C = {**prf(ctp, cfp, cfn), "tp": ctp, "fp": cfp, "fn": cfn,
         "n_scored": B["n_scored"], "predicted_positive": cn,
         "gold_positives_in_scope": gold_pairs_total}

    sizes = sorted((len(m) for m in comp.values()), reverse=True)
    topo = {"asserted_links": len(asserted),
            "implied_pairs": cn,
            "closure_amplification": cn / len(asserted) if asserted else 0.0,
            "n_components": len(comp),
            "n_multi_record_components": sum(1 for s in sizes if s > 1),
            "largest_component": sizes[0] if sizes else 0,
            "component_size_distribution": dict(
                sorted(collections.Counter(sizes).items())),
            "n_nodes": len(nodes)}
    return {"A": A, "B": B, "C": C, "topology": topo,
            "universe_change_A_to_B_f1": B["f1"] - A["f1"],
            "operator_effect_B_to_C_f1": C["f1"] - B["f1"]}


def boot_stageA_f1(ds: G.Settlements, cand: List[Tuple[int, int]], asserted: Set[str],
                   draws: int = DRAWS, seed: int = SEED) -> Dict[str, float]:
    """Cluster bootstrap for stage A only. The unit is the unordered pair of gold clusters."""
    by: Dict[Tuple[int, int], List[Tuple[bool, int]]] = collections.defaultdict(list)
    for (x, y) in cand:
        ca, cb = ds.gold_cluster(x), ds.gold_cluster(y)
        key = (ca, cb) if ca <= cb else (cb, ca)
        by[key].append((ds.pair_key(x, y) in asserted, ds.gold_pair(x, y)))
    keys = list(by)
    rnd = random.Random(seed)
    sims = []
    for _ in range(draws):
        tp = fp = fn = 0
        for _ in range(len(keys)):
            for pos, g in by[keys[rnd.randrange(len(keys))]]:
                tp += pos and g == 1
                fp += pos and g == 0
                fn += (not pos) and g == 1
        sims.append(prf(tp, fp, fn)["f1"])
    sims.sort()
    return {"lo": sims[int(0.025 * draws)], "hi": sims[min(draws - 1, int(0.975 * draws))],
            "n_cluster_pairs": len(keys), "draws": draws}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    ds = G.Settlements()
    nodes = ds.records_in("test")
    cand = ds.block(nodes, 5)
    gold_total = sum(len(m) * (len(m) - 1) // 2 for m in ds.clusters_in("test").values())
    print(f"  test universe: {len(nodes)} records, "
          f"{len(nodes)*(len(nodes)-1)//2:,} pairs, {gold_total} gold pairs, "
          f"{len(cand)} candidates")

    recs = jl("gs_preds_test.jsonl")
    ok = [r for r in recs if r.get("status") == "ok"]
    failed = [r for r in recs if r.get("status") != "ok"]
    cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in ok:
        cell[(r["pair_key"], r["model"])].append(r)
    dec = {k: modal([x["decision"] for x in v]) for k, v in cell.items()}
    models = list(G.POOL)

    res: Dict[str, Any] = {
        "workload": "Geographic Settlements (Leipzig DBS), test split, entity-disjoint",
        "universe": {"records": len(nodes),
                     "all_record_pairs": len(nodes) * (len(nodes) - 1) // 2,
                     "gold_pairs": gold_total, "candidate_pairs": len(cand)},
        "records_collected": len(recs), "records_ok": len(ok), "records_failed": len(failed),
        "failures": [{k: v for k, v in r.items() if k in
                      ("unit_id", "pair_key", "model", "repeat", "error")} for r in failed[:50]],
        "parser_version": G.PARSER_VERSION,
        "interval_note": ("stage A resamples gold-cluster pairs; stages B and C carry no interval "
                          "because closure couples the entities a method merges, which an entity "
                          "bootstrap would not describe"),
        "methods": {}}

    print(f"  collected {len(recs)} records, {len(ok)} ok, {len(failed)} failed")
    print(f"\n{'method':22s}{'A:F1':>8s}{'A 95% CI':>18s}{'B:F1':>8s}{'C:F1':>8s}"
          f"{'A→B':>9s}{'B→C':>9s}{'links':>8s}{'implied':>10s}{'ampl':>9s}{'abst':>7s}")
    for m in models + ["vote:majority"]:
        if m == "vote:majority":
            asserted = {ds.pair_key(x, y) for (x, y) in cand
                        if modal([dec.get((ds.pair_key(x, y), mm)) for mm in models]) == "match"}
            ab = sum(1 for (x, y) in cand
                     if modal([dec.get((ds.pair_key(x, y), mm)) for mm in models]) is None)
        else:
            asserted = {ds.pair_key(x, y) for (x, y) in cand
                        if dec.get((ds.pair_key(x, y), m)) == "match"}
            ab = sum(1 for (x, y) in cand if dec.get((ds.pair_key(x, y), m)) is None)
        s = stage_ABC(ds, nodes, cand, asserted, gold_total)
        s["abstentions"] = ab
        s["abstention_rate"] = ab / len(cand)
        s["A"]["ci"] = boot_stageA_f1(ds, cand, asserted)
        name = m if m == "vote:majority" else f"single:{m}"
        res["methods"][name] = s
        t = s["topology"]
        print(f"{name:22s}{s['A']['f1']:8.4f}  [{s['A']['ci']['lo']:.4f},{s['A']['ci']['hi']:.4f}]"
              f"{s['B']['f1']:8.4f}{s['C']['f1']:8.4f}"
              f"{s['universe_change_A_to_B_f1']:+9.4f}{s['operator_effect_B_to_C_f1']:+9.4f}"
              f"{t['asserted_links']:8d}{t['implied_pairs']:10d}"
              f"{t['closure_amplification']:9.2f}{s['abstention_rate']:7.1%}")

    # gold-edge replay: the fixed-blocking reference, valid only under a zero-false-positive
    # edge constraint
    gasserted = {ds.pair_key(x, y) for (x, y) in cand if ds.gold_pair(x, y) == 1}
    s = stage_ABC(ds, nodes, cand, gasserted, gold_total)
    s["note"] = ("gold-edge replay over the frozen candidate set. An upper bound only under a "
                 "zero-false-positive edge constraint: a method asserting false bridges can exceed "
                 "its recall, as WDC's llama-3-8b did")
    res["methods"]["GOLD_edge_replay"] = s
    t = s["topology"]
    print(f"{'GOLD_edge_replay':22s}{s['A']['f1']:8.4f}{'':18s}{s['B']['f1']:8.4f}{s['C']['f1']:8.4f}"
          f"{s['universe_change_A_to_B_f1']:+9.4f}{s['operator_effect_B_to_C_f1']:+9.4f}"
          f"{t['asserted_links']:8d}{t['implied_pairs']:10d}{t['closure_amplification']:9.2f}")

    # ---- ranks with ties at each stage
    print("\n=== ranks with ties (1 = best) ===")
    order: Dict[str, Dict[str, int]] = {}
    meth = [k for k in res["methods"] if k != "GOLD_edge_replay"]
    for st in ("A", "B", "C"):
        vals = sorted({round(res["methods"][k][st]["f1"], 10) for k in meth}, reverse=True)
        rank = {v: i + 1 for i, v in enumerate(vals)}
        order[st] = {k: rank[round(res["methods"][k][st]["f1"], 10)] for k in meth}
    print(f"  {'method':22s}{'rank A':>8s}{'rank B':>8s}{'rank C':>8s}  changed?")
    for k in sorted(meth, key=lambda k: order["A"][k]):
        ch = len({order['A'][k], order['B'][k], order['C'][k]}) > 1
        print(f"  {k:22s}{order['A'][k]:8d}{order['B'][k]:8d}{order['C'][k]:8d}  "
              f"{'YES' if ch else '-'}")
    res["ranks"] = order
    res["rank_changes"] = {k: {"A": order["A"][k], "B": order["B"][k], "C": order["C"][k],
                               "changed": len({order['A'][k], order['B'][k],
                                               order['C'][k]}) > 1} for k in meth}

    # ---- long-form per-method CSV
    rows: List[Dict[str, Any]] = []
    for k, s in res["methods"].items():
        for st in ("A", "B", "C"):
            d = s[st]
            rows.append({"method": k, "stage": st,
                         "stage_name": {"A": "candidate pairs",
                                        "B": "all record pairs, direct links only",
                                        "C": "all record pairs, after closure"}[st],
                         "f1": d["f1"], "precision": d["precision"], "recall": d["recall"],
                         "tp": d["tp"], "fp": d["fp"], "fn": d["fn"],
                         "n_scored": d["n_scored"],
                         "gold_positives_in_scope": d["gold_positives_in_scope"],
                         "predicted_positive": d.get("predicted_positive", d["tp"] + d["fp"]),
                         "ci_lo": (d.get("ci") or {}).get("lo"),
                         "ci_hi": (d.get("ci") or {}).get("hi"),
                         "rank": (res["ranks"][st].get(k) if k != "GOLD_edge_replay" else None),
                         "abstentions": s.get("abstentions"),
                         "asserted_links": s["topology"]["asserted_links"],
                         "implied_pairs": s["topology"]["implied_pairs"],
                         "closure_amplification": s["topology"]["closure_amplification"],
                         "largest_component": s["topology"]["largest_component"],
                         "n_multi_record_components": s["topology"]["n_multi_record_components"]})
    cp = os.path.join(OUT, "stages_long.csv")
    cols = list(rows[0])
    with open(cp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(os.path.join(OUT, "stages.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {OUT}/stages.json and {cp} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
