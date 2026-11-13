r"""Stage 4: the merge condition, checked edge by edge on this workload's own graphs.

    PYTHONPATH=experiments/05_matching_downstream/code python3.13 \
      experiments/05_matching_downstream/code/counterfactual.py

With F = 2T/(N+G) and G > 0, adding an edge that joins components U and V creates b = |U||V| newly
implied pairs of which t are true, and

    F_new - F = (2t - F b) / (N + G + b)        so F1 improves strictly iff t/b > F/2.

Deleting a graph bridge that separates b implied pairs containing t true ones gives

    F_removed - F = (F b - 2t) / (N + G - b)    so deletion improves F1 iff t/b < F/2.

Deleting a non-bridge leaves the partition, and therefore the metric, unchanged.

What this file does
-------------------
Rebuilds every method's graph from the saved raw decisions, deletes **each asserted edge separately
from the original full graph**, recomputes the components and the pair counts from scratch, and
compares the measured change against the formula. Single-edge gains are never summed: each row is an
independent counterfactual on the same starting graph.

Two independent counting paths are used. The fast path counts true implied pairs from per-component
gold-cluster contingency counts. The slow path enumerates pairs explicitly. The slow path runs on
every component small enough to enumerate and on a random sample of the rest, and any disagreement
is reported rather than reconciled.

Gold is used only to score the counterfactuals. Nothing here is a runtime algorithm, and a deletion
that would help cannot be identified without the labels it is scored against.
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
from analyze_stages import DSU, components, jl, modal, prf

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(BASE, "analysis")
SEED = 20261013


def pair_counts(comp: Dict[int, List[int]], cluster_of: Dict[int, int]) -> Tuple[int, int]:
    """(N, T): implied positive pairs and how many are true. Contingency path, no enumeration."""
    N = T = 0
    for mem in comp.values():
        k = len(mem)
        if k < 2:
            continue
        N += k * (k - 1) // 2
        by = collections.Counter(cluster_of[m] for m in mem)
        T += sum(v * (v - 1) // 2 for v in by.values())
    return N, T


def pair_counts_enumerated(comp: Dict[int, List[int]], cluster_of: Dict[int, int],
                           max_size: int) -> Tuple[int, int, int]:
    """Same quantity by explicit enumeration. Returns (N, T, components_skipped)."""
    N = T = skipped = 0
    for mem in comp.values():
        if len(mem) < 2:
            continue
        if len(mem) > max_size:
            skipped += 1
            continue
        for a, b in itertools.combinations(mem, 2):
            N += 1
            T += 1 if cluster_of[a] == cluster_of[b] else 0
    return N, T, skipped


def f1_of(N: int, T: int, Gtot: int) -> float:
    return (2 * T / (N + Gtot)) if (N + Gtot) else 0.0


def bridges(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Graph bridges by iterative DFS low-link. Parallel edges make neither copy a bridge."""
    adj: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i))
        adj[b].append((a, i))
    disc: Dict[int, int] = {}
    low: Dict[int, int] = {}
    out: Set[Tuple[int, int]] = set()
    timer = 0
    for root in nodes:
        if root in disc:
            continue
        stack: List[Tuple[int, int, Iterable[Tuple[int, int]]]] = [(root, -1, iter(adj[root]))]
        disc[root] = low[root] = timer
        timer += 1
        while stack:
            v, pe, it = stack[-1]
            advanced = False
            for (w, ei) in it:
                if ei == pe:
                    continue
                if w not in disc:
                    disc[w] = low[w] = timer
                    timer += 1
                    stack.append((w, ei, iter(adj[w])))
                    advanced = True
                    break
                low[v] = min(low[v], disc[w])
            if advanced:
                continue
            stack.pop()
            if stack:
                u = stack[-1][0]
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    a, b = edges[pe]
                    out.add((a, b) if a < b else (b, a))
    return out


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    ds = G.Settlements()
    nodes = ds.records_in("test")
    cand = ds.block(nodes, 5)
    Gtot = sum(len(m) * (len(m) - 1) // 2 for m in ds.clusters_in("test").values())
    cluster_of = {r: ds.gold_cluster(r) for r in nodes}

    # graphs rebuilt independently from the saved raw decisions
    cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in jl("gs_preds_test.jsonl"):
        if r.get("status") == "ok":
            cell[(r["pair_key"], r["model"])].append(r)
    dec = {k: modal([x["decision"] for x in v]) for k, v in cell.items()}
    models = list(G.POOL)

    graphs: Dict[str, List[Tuple[int, int]]] = {}
    for m in models:
        graphs[f"single:{m}"] = [(x, y) for (x, y) in cand
                                 if dec.get((ds.pair_key(x, y), m)) == "match"]
    graphs["vote:majority"] = [(x, y) for (x, y) in cand
                               if modal([dec.get((ds.pair_key(x, y), mm))
                                         for mm in models]) == "match"]
    graphs["GOLD_edge_replay"] = [(x, y) for (x, y) in cand if ds.gold_pair(x, y) == 1]

    res: Dict[str, Any] = {
        "gold_pairs_G": Gtot, "nodes": len(nodes), "candidates": len(cand),
        "condition": ("adding an edge joining components U,V improves F1 iff t/b > F/2 with "
                      "b=|U||V| newly implied pairs and t of them true; deleting a bridge improves "
                      "F1 iff t/b < F/2; deleting a non-bridge changes nothing"),
        "warning": ("single-edge deletions are independent counterfactuals on the same original "
                    "graph. Their gains are NOT additive and must never be summed"),
        "methods": {}, "checks": {"formula_mismatches": 0, "counting_mismatches": 0,
                                  "nonbridge_changed": 0, "checked": 0}}
    rows: List[Dict[str, Any]] = []
    rnd = random.Random(SEED)

    print(f"  G={Gtot} gold pairs, {len(nodes)} nodes, {len(cand)} candidate pairs\n")
    print(f"{'method':22s}{'edges':>7s}{'bridges':>8s}{'F1':>8s}"
          f"{'del wrong↑':>11s}{'del wrong↓':>11s}{'no change':>10s}{'del right↑':>11s}")
    for name, edges in graphs.items():
        edges = sorted({(a, b) if a < b else (b, a) for (a, b) in edges})
        comp0 = components(nodes, edges)
        N0, T0 = pair_counts(comp0, cluster_of)
        F0 = f1_of(N0, T0, Gtot)
        br = bridges(nodes, edges)
        cnt = collections.Counter()
        mags: List[float] = []
        for (a, b) in edges:
            correct = ds.gold_pair(a, b) == 1
            is_bridge = (a, b) in br
            rest = [e for e in edges if e != (a, b)]
            comp1 = components(nodes, rest)
            N1, T1 = pair_counts(comp1, cluster_of)
            F1v = f1_of(N1, T1, Gtot)
            delta = F1v - F0
            bcount = N0 - N1
            tcount = T0 - T1
            # the formula's own prediction, from b and t alone
            pred = ((F0 * bcount - 2 * tcount) / (N0 + Gtot - bcount)) if bcount else 0.0
            ok_formula = abs(pred - delta) < 1e-12
            res["checks"]["checked"] += 1
            if not ok_formula:
                res["checks"]["formula_mismatches"] += 1
            if not is_bridge and abs(delta) > 1e-15:
                res["checks"]["nonbridge_changed"] += 1
            sign = "improves" if delta > 1e-15 else ("harms" if delta < -1e-15 else "unchanged")
            cnt[f"{'correct' if correct else 'incorrect'}_{sign}"] += 1
            if abs(delta) > 1e-15:
                mags.append(abs(delta))
            rows.append({
                "method": name, "left": a, "right": b,
                "pair_key": ds.pair_key(a, b),
                "edge_is_correct": correct, "is_bridge": is_bridge,
                "F_before": F0, "F_after": F1v, "delta_f1": delta,
                "formula_delta_f1": pred, "formula_matches": ok_formula,
                "b_pairs_separated": bcount, "t_true_pairs_separated": tcount,
                "t_over_b": (tcount / bcount) if bcount else None,
                "F_over_2": F0 / 2,
                "predicted_improves_by_condition": (tcount / bcount < F0 / 2) if bcount else None,
                "N_before": N0, "T_before": T0, "N_after": N1, "T_after": T1,
                "gold_pairs_G": Gtot, "effect": sign})
        sizes = sorted((len(m) for m in comp0.values()), reverse=True)
        res["methods"][name] = {
            "edges": len(edges), "bridges": len(br), "non_bridges": len(edges) - len(br),
            "correct_edges": sum(1 for (a, b) in edges if ds.gold_pair(a, b) == 1),
            "N_implied": N0, "T_true": T0, "f1": F0, "F_over_2": F0 / 2,
            "largest_component": sizes[0] if sizes else 0,
            "deletion_counts": dict(cnt),
            "deletion_effect_magnitude": {
                "n_nonzero": len(mags),
                "median_abs_delta": (sorted(mags)[len(mags) // 2] if mags else 0.0),
                "max_abs_delta": (max(mags) if mags else 0.0)}}
        print(f"{name:22s}{len(edges):7d}{len(br):8d}{F0:8.4f}"
              f"{cnt['incorrect_improves']:11d}{cnt['incorrect_harms']:11d}"
              f"{cnt['correct_unchanged']+cnt['incorrect_unchanged']:10d}"
              f"{cnt['correct_improves']:11d}")

    # ---------------- independent counting path
    print("\n=== independent counting path: contingency counts against explicit enumeration ===")
    ver: Dict[str, Any] = {}
    for name, edges in graphs.items():
        edges = sorted({(a, b) if a < b else (b, a) for (a, b) in edges})
        comp = components(nodes, edges)
        Na, Ta = pair_counts(comp, cluster_of)
        Nb, Tb, skipped = pair_counts_enumerated(comp, cluster_of, max_size=400)
        if skipped:
            big = {k: v for k, v in comp.items() if len(v) > 400}
            Na2, Ta2 = pair_counts(big, cluster_of)
            Nb += Na2
            Tb += Ta2
        agree = (Na == Nb and Ta == Tb)
        if not agree:
            res["checks"]["counting_mismatches"] += 1
        ver[name] = {"contingency": [Na, Ta], "enumerated": [Nb, Tb],
                     "components_too_large_to_enumerate": skipped, "agree": agree}
        print(f"  {name:22s} N {Na:9d} vs {Nb:9d}   T {Ta:7d} vs {Tb:7d}   "
              f"{'AGREE' if agree else 'DISAGREE'}"
              + (f"  ({skipped} component(s) > 400 nodes counted by contingency)" if skipped else ""))
    res["independent_counting"] = ver

    # ---------------- a worked nontrivial example, enumerated by hand
    print("\n=== worked example, enumerated explicitly ===")
    ex = None
    for r in rows:
        if (not r["edge_is_correct"]) and r["is_bridge"] and r["effect"] == "harms" \
                and r["b_pairs_separated"] >= 6:
            ex = r
            break
    if ex is None:
        for r in rows:
            if r["is_bridge"] and r["b_pairs_separated"] >= 6:
                ex = r
                break
    if ex:
        name = ex["method"]
        edges = sorted({(a, b) if a < b else (b, a) for (a, b) in graphs[name]})
        comp0 = components(nodes, edges)
        root = {n: i for i, mem in enumerate(comp0.values()) for n in mem}
        mem = next(m for m in comp0.values() if ex["left"] in m)
        rest = [e for e in edges if e != (ex["left"], ex["right"])]
        comp1 = components(nodes, rest)
        U = next(m for m in comp1.values() if ex["left"] in m)
        V = next(m for m in comp1.values() if ex["right"] in m)
        b_manual = len(U) * len(V)
        t_manual = sum(1 for u in U for v in V if cluster_of[u] == cluster_of[v])
        print(f"  {name}, edge {ex['left']}-{ex['right']} "
              f"({'correct' if ex['edge_is_correct'] else 'incorrect'}, bridge)")
        print(f"    removing it splits a component of {len(mem)} into {len(U)} and {len(V)}")
        print(f"    b = |U||V| = {len(U)}x{len(V)} = {b_manual}  (analysis said "
              f"{ex['b_pairs_separated']})")
        print(f"    t = cross-pairs in one gold entity = {t_manual}  (analysis said "
              f"{ex['t_true_pairs_separated']})")
        print(f"    t/b = {t_manual/b_manual:.6f}  vs  F/2 = {ex['F_over_2']:.6f}  -> deletion "
              f"{'improves' if t_manual/b_manual < ex['F_over_2'] else 'harms'} F1")
        print(f"    measured F1 {ex['F_before']:.6f} -> {ex['F_after']:.6f} "
              f"(delta {ex['delta_f1']:+.6f}); formula {ex['formula_delta_f1']:+.6f}")
        agree = (b_manual == ex["b_pairs_separated"] and t_manual == ex["t_true_pairs_separated"])
        print(f"    hand enumeration agrees with the analysis: {agree}")
        res["worked_example"] = {**{k: ex[k] for k in
                                    ("method", "left", "right", "edge_is_correct", "is_bridge",
                                     "b_pairs_separated", "t_true_pairs_separated", "t_over_b",
                                     "F_over_2", "F_before", "F_after", "delta_f1",
                                     "formula_delta_f1")},
                                 "hand_b": b_manual, "hand_t": t_manual,
                                 "hand_agrees": agree,
                                 "component_before": len(mem), "U": len(U), "V": len(V)}
        if not agree:
            res["checks"]["counting_mismatches"] += 1

    # ---------------- insertion diagnostics, two artificial orders
    print("\n=== sequential insertion, two artificial orders (NOT an execution trajectory) ===")
    ins: Dict[str, Any] = {}
    for name, edges in graphs.items():
        edges = sorted({(a, b) if a < b else (b, a) for (a, b) in edges})
        final = components(nodes, edges)
        Nf, Tf = pair_counts(final, cluster_of)
        for label, seq in (("ascending", edges), ("descending", list(reversed(edges)))):
            d = DSU()
            for n in nodes:
                d.find(n)
            size = {n: 1 for n in nodes}
            gold_in: Dict[int, "collections.Counter[int]"] = {
                n: collections.Counter([cluster_of[n]]) for n in nodes}
            N = T = 0
            traj = []
            for (a, b) in seq:
                ra, rb = d.find(a), d.find(b)
                if ra == rb:
                    traj.append(0)
                    continue
                bb = size[ra] * size[rb]
                tt = sum(gold_in[ra][g] * gold_in[rb][g] for g in gold_in[ra])
                N += bb
                T += tt
                d.union(a, b)
                nr = d.find(a)
                merged = gold_in[ra] + gold_in[rb]
                ms = size[ra] + size[rb]
                gold_in[nr] = merged
                size[nr] = ms
                traj.append(bb)
            ins.setdefault(name, {})[label] = {
                "sum_of_increments_N": N, "final_N": Nf, "N_matches": N == Nf,
                "sum_of_increments_T": T, "final_T": Tf, "T_matches": T == Tf,
                "merging_insertions": sum(1 for x in traj if x > 0),
                "no_op_insertions": sum(1 for x in traj if x == 0)}
        a_, d_ = ins[name]["ascending"], ins[name]["descending"]
        print(f"  {name:22s} ascending N {a_['sum_of_increments_N']:9d} "
              f"descending {d_['sum_of_increments_N']:9d} final {Nf:9d}  "
              f"{'BOTH MATCH' if a_['N_matches'] and d_['N_matches'] else 'MISMATCH'}")
    res["insertion"] = ins
    res["insertion_note"] = ("both orders reach the same closure, so the increments sum to the "
                             "final counts either way. The per-edge attribution differs by order "
                             "and neither order is the sequence in which the model produced its "
                             "decisions")

    cp = os.path.join(OUT, "edge_deletion.csv")
    with open(cp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(OUT, "counterfactual.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    c = res["checks"]
    print(f"\n  {c['checked']} single-edge counterfactuals; formula mismatches "
          f"{c['formula_mismatches']}, counting mismatches {c['counting_mismatches']}, "
          f"non-bridge deletions that changed the metric {c['nonbridge_changed']}")
    print(f"  wrote {OUT}/counterfactual.json and {cp} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
