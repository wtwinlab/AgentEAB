r"""M.B: does a better pairwise decision produce a better linkage result?

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/analyze_pipeline.py

The pipeline is fixed except for the scoring step: blocking is the benchmark's own candidate pair
set, the merge rule is connected components over predicted matches, and only the pairwise decision
varies. That isolates the question the plan asks — whether a local improvement in pair scoring
survives aggregation into entities.

Where cluster evaluation is defined, and where it is not
-------------------------------------------------------
WDC Products ships `cluster_id_left/right`, real entity identifiers, so predicted clusters can be
compared with true ones. Amazon-Google and Abt-Buy ship labelled pairs only. For those two we report
the **link-level** consequences of merging — how many predicted links are spurious, how many true
links are missed, and how the transitive closure inflates both — and we do **not** invent cluster
ground truth, because a set of labelled pairs does not determine the entity partition.

Transitive closure is where pairwise error becomes pipeline error: one false merge can join two
otherwise correct clusters, so a precision loss at the pair level is amplified. That amplification
is the quantity of interest and it is reported as a measured ratio, not asserted.
"""
from __future__ import annotations

import collections
import json
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import em_common as E

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")
OUT = os.path.join(BASE, "analysis")
DRAWS = 2000
SEED = 20261013


class DSU:
    def __init__(self):
        self.p: Dict[str, str] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def clusters_from(pairs: List[Tuple[str, str]], nodes: Set[str]) -> Dict[str, int]:
    d = DSU()
    for n in nodes:
        d.find(n)
    for a, b in pairs:
        d.union(a, b)
    roots = {}
    out = {}
    for n in nodes:
        r = d.find(n)
        if r not in roots:
            roots[r] = len(roots)
        out[n] = roots[r]
    return out


def pairwise_cluster_scores(pred: Dict[str, int], gold: Dict[str, int]) -> Dict[str, float]:
    """Compare two partitions by the pairs they imply, over the nodes both define."""
    nodes = sorted(set(pred) & set(gold))
    bp = collections.defaultdict(list)
    bg = collections.defaultdict(list)
    for n in nodes:
        bp[pred[n]].append(n)
        bg[gold[n]].append(n)

    def ipairs(groups):
        s = set()
        for g in groups.values():
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    s.add((g[i], g[j]) if g[i] < g[j] else (g[j], g[i]))
        return s
    P, G = ipairs(bp), ipairs(bg)
    tp = len(P & G)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(G) if G else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"cluster_precision": prec, "cluster_recall": rec, "cluster_f1": f1,
            "n_nodes": len(nodes), "n_pred_clusters": len(bp), "n_gold_clusters": len(bg),
            "n_pred_intra_pairs": len(P), "n_gold_intra_pairs": len(G)}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    # frozen per-model decisions, modal over seeds
    cell = collections.defaultdict(list)
    with open(os.path.join(RAW, "em_preds_test.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                cell[(r["dataset"], r["pair_key"], r["model"])].append(r)

    def modal(rs):
        v = [x["decision"] for x in rs if x["decision"] is not None]
        if not v:
            return None
        c = collections.Counter(v)
        top = max(c.values())
        w = [k for k, n in c.items() if n == top]
        return w[0] if len(w) == 1 else None
    pred = {k: modal(v) for k, v in cell.items()}
    models = list(E.POOL)

    def majority(ds):
        v = [d for d in ds if d is not None]
        if not v:
            return None
        c = collections.Counter(v)
        t = max(c.values())
        w = [k for k, n in c.items() if n == t]
        return w[0] if len(w) == 1 else None

    res: Dict[str, Any] = {
        "pipeline": "benchmark blocking -> pairwise decision -> connected-component merge",
        "note": ("cluster metrics are computed only where the benchmark supplies entity ids; "
                 "a set of labelled pairs does not determine the entity partition"),
        "datasets": {}}

    for dn in ("abt_buy", "amazon_google", "wdc_products"):
        ds = E.Matching(dn)
        rows = ds.splits["test"]
        entry: Dict[str, Any] = {"has_cluster_gold": ds.has_clusters}
        # node ids must be globally unique across the two tables
        def nid(row, side):
            if dn == "wdc_products":
                return f"{side}:{row['id_left'] if side=='L' else row['id_right']}"
            return f"{side}:{row['ltable_id'] if side=='L' else row['rtable_id']}"
        nodes = set()
        gold_pairs, pk_of = [], {}
        for row in rows:
            a, b = nid(row, "L"), nid(row, "R")
            nodes |= {a, b}
            pk_of[ds.pair_key(row)] = (a, b)
            if ds.gold(row) == 1:
                gold_pairs.append((a, b))

        methods: Dict[str, List[Tuple[str, str]]] = {}
        for m in models:
            methods[f"single:{m}"] = [pk_of[p] for p in pk_of
                                      if pred.get((dn, p, m)) == "match"]
        methods["vote:majority"] = [pk_of[p] for p in pk_of
                                    if majority([pred.get((dn, p, m)) for m in models]) == "match"]
        methods["GOLD"] = gold_pairs

        print(f"\n=== {dn} === {len(nodes)} records, {len(gold_pairs)} true links, "
              f"cluster gold: {ds.has_clusters}")
        print(f"  {'method':22s} {'links':>6s} {'merged':>7s} {'closure':>8s} "
              + ("{:>9s} {:>9s} {:>9s}".format('cl.prec', 'cl.rec', 'cl.F1')
                 if ds.has_clusters else ""))
        gold_cl = clusters_from(gold_pairs, nodes)
        for name, links in methods.items():
            cl = clusters_from(links, nodes)
            sizes = collections.Counter(cl.values())
            merged = sum(1 for v in sizes.values() if v > 1)
            biggest = max(sizes.values()) if sizes else 0
            # closure amplification: implied intra-cluster pairs per asserted link
            implied = sum(v * (v - 1) // 2 for v in sizes.values())
            amp = implied / max(1, len(links))
            row: Dict[str, Any] = {"asserted_links": len(links), "merged_clusters": merged,
                                   "largest_cluster": biggest,
                                   "implied_intra_pairs": implied,
                                   "closure_amplification": amp}
            line = f"  {name:22s} {len(links):6d} {merged:7d} {amp:8.2f}"
            if ds.has_clusters:
                # gold entity ids, mapped onto the same node space
                gmap = {}
                for r0 in rows:
                    gmap[nid(r0, "L")] = f"c{r0['cluster_id_left']}"
                    gmap[nid(r0, "R")] = f"c{r0['cluster_id_right']}"
                idx = {c: i for i, c in enumerate(sorted(set(gmap.values())))}
                true_part = {n: idx[gmap[n]] for n in nodes if n in gmap}
                sc = pairwise_cluster_scores(cl, true_part)
                row.update(sc)
                line += (f" {sc['cluster_precision']:9.3f} {sc['cluster_recall']:9.3f} "
                         f"{sc['cluster_f1']:9.3f}")
            entry.setdefault("methods", {})[name] = row
            print(line)
        res["datasets"][dn] = entry

    # does a pairwise gain survive the merge? compare rank order at both levels
    print("\n=== pairwise rank vs pipeline rank (WDC, where cluster gold exists) ===")
    try:
        base = json.load(open(os.path.join(OUT, "em_method_comparison.json")))
        pw = {k: v["f1"] for k, v in base["datasets"]["wdc_products"]["methods"].items()
              if k.startswith("single:") or k == "vote:all6"}
        cl = {k: v.get("cluster_f1") for k, v in res["datasets"]["wdc_products"]["methods"].items()
              if k != "GOLD"}
        common = [k for k in pw if k.replace("vote:all6", "vote:majority") in cl]
        print(f"  {'method':22s} {'pair F1':>8s} {'cluster F1':>11s}")
        for k in sorted(common, key=lambda k: -pw[k]):
            kk = k.replace("vote:all6", "vote:majority")
            print(f"  {kk:22s} {pw[k]:8.3f} {cl[kk]:11.3f}")
        order_pw = [k.replace("vote:all6", "vote:majority") for k in sorted(common, key=lambda k: -pw[k])]
        order_cl = sorted(order_pw, key=lambda k: -(cl[k] or 0))
        res["rank_agreement_wdc"] = {"pairwise_order": order_pw, "cluster_order": order_cl,
                                     "same_top": order_pw[0] == order_cl[0]}
        print(f"  same best method at both levels: {order_pw[0] == order_cl[0]}")
    except Exception as exc:                                              # noqa: BLE001
        print(f"  (rank comparison unavailable: {type(exc).__name__})")

    with open(os.path.join(OUT, "pipeline_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {OUT}/pipeline_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
