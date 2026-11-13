r"""Grouping: why an edge-level divergence share does not bound cluster F1, and a relation-level bound
that does (zero model calls).

    python3 -I share_eval/replay_validity/grouping_bound.py [--sharing data/sharing_long.csv] [--out DIR]

Bound (grouping corollary of Lemma 1 of the paper). Let a relation set R (pairs implied by the groups) change to R'
by adding a and removing r relations after transitive closure; P' = |R'|, G = |gold relations|. Then
    |F1(R') − F1(R)| ≤ (2·max(a, r) + min(a, r)) / (P' + G).
Proof sketch: F1 = 2T/S with S = P + G and T ≤ S/2; T' = T + τ_a − τ_r with 0 ≤ τ_a ≤ a, 0 ≤ τ_r ≤ r;
F1' − F1 = 2[(τ_a − τ_r)S − T(a − r)]/(S S'), bounded above by (2a + r)/S' and below by −(2r + a)/S'.

Checks: (1) a synthetic single bridge edge between two components of sizes s1, s2 adds s1·s2 relations,
so an edge-level share 1/E can sit far below the F1 change; (2) the paper's Settlements example
(Qwen2.5-32B, decisions shared vs no peers; data/sharing_long.csv, stage C = cluster level).
"""
from __future__ import annotations

import csv
import itertools
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def closure_pairs(nodes, edges):
    parent = {v: v for v in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in edges:
        parent[find(u)] = find(v)
    comp = {}
    for v in nodes:
        comp.setdefault(find(v), []).append(v)
    return {tuple(sorted(p)) for c in comp.values() for p in itertools.combinations(c, 2)}


def f1(pred, gold):
    t = len(pred & gold)
    return 2 * t / (len(pred) + len(gold)) if (pred or gold) else 1.0


def bound(a, r, p_after, g):
    return (2 * max(a, r) + min(a, r)) / (p_after + g)


def one_sided(a, r, p_after, g):
    """[lower, upper] on F1' − F1: [−(2r + a)/S', (2a + r)/S']."""
    s = p_after + g
    return [round(-100 * (2 * r + a) / s, 3), round(100 * (2 * a + r) / s, 3)]


def synthetic(s1=10, s2=10, extra_singletons=200):
    nodes = [f"a{i}" for i in range(s1)] + [f"b{i}" for i in range(s2)] + [f"c{i}" for i in range(extra_singletons)]
    chain_a = [(f"a{i}", f"a{i+1}") for i in range(s1 - 1)]
    chain_b = [(f"b{i}", f"b{i+1}") for i in range(s2 - 1)]
    gold = closure_pairs(nodes, chain_a + chain_b)
    before = closure_pairs(nodes, chain_a + chain_b)
    after = closure_pairs(nodes, chain_a + chain_b + [("a0", "b0")])
    n_edges_judged = len(chain_a) + len(chain_b) + 1
    a, r = len(after - before), len(before - after)
    d = f1(after, gold) - f1(before, gold)
    return dict(sizes=[s1, s2], edges_judged=n_edges_judged, edges_changed=1, edge_share=round(1 / n_edges_judged, 4),
                relations_added=a, relations_removed=r, delta_f1_pp=round(100 * d, 2),
                relation_bound_pp=round(100 * bound(a, r, len(after), len(gold)), 2),
                relation_interval_pp=one_sided(a, r, len(after), len(gold)),
                edge_share_bound_violated=bool(abs(d) > 1 / n_edges_judged),
                relation_bound_holds=bool(abs(d) <= bound(a, r, len(after), len(gold)) + 1e-12))


def settlements(p):
    lv = {}
    for r in csv.DictReader(open(p)):
        if r["recipient"] == "qwen2.5-32b" and r["row_kind"] == "level":
            lv[(r["condition"], r["stage"])] = r
    b, c = lv[("no_peers", "C")], lv[("candidate_only", "C")]
    tp0, fp0, fn0 = int(b["tp"]), int(b["fp"]), int(b["fn"])
    tp1, fp1 = int(c["tp"]), int(c["fp"])
    G = tp0 + fn0
    P0, P1 = tp0 + fp0, tp1 + fp1
    f_0, f_1 = 2 * tp0 / (P0 + G), 2 * tp1 / (P1 + G)
    # relations added/removed after closure (5 true + 142 false added, 0 removed)
    a, r = (tp1 - tp0) + (fp1 - fp0), 0
    ea = lv[("no_peers", "A")]
    n_judged = int(ea["n_scored"])
    changed_judgments = 23                                  # changed pair judgments of this comparison (Section VI-F)
    return dict(source=p.name, gold_relations=G, pred_relations_before=P0, pred_relations_after=P1,
                f1_before=round(f_0, 5), f1_after=round(f_1, 5), delta_f1_pp=round(100 * (f_1 - f_0), 3),
                relations_added=a, relations_removed=r, judged_pairs=n_judged, changed_judgments=changed_judgments,
                edge_share_pp=round(100 * changed_judgments / n_judged, 3),
                relation_bound_pp=round(100 * bound(a, r, P1, G), 3),
                relation_interval_pp=one_sided(a, r, P1, G),
                edge_share_bound_violated=bool(abs(f_1 - f_0) > changed_judgments / n_judged),
                relation_bound_holds=bool(abs(f_1 - f_0) <= bound(a, r, P1, G)))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="relation-level bound for grouping (Lemma 1, grouping corollary)")
    ap.add_argument("--sharing", default=str(HERE / "data/sharing_long.csv"),
                    help="Settlements grouping levels per condition and stage (default %(default)s)")
    ap.add_argument("--out", default=str(REPO / "output/share_eval/replay_validity"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = dict(synthetic_bridge=synthetic(), synthetic_bridge_large=synthetic(30, 40, 500),
               settlements=settlements(pathlib.Path(a.sharing)))
    (out / "grouping_bound.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
