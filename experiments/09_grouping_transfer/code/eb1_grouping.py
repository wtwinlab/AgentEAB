from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]
CACHE = WORK / "cache"
OUT = WORK / "output"
STAGE = WORK
sys.path.insert(0, str(STAGE / "code"))
import metrics as M

DATASETS = ("wdc_products_E5f", "geographic_settlements_E5f", "musicbrainz20k_e5p")
RULE_ORDER = ("connected_components", "center", "merge_center", "star_disjoint",
              "correlation_greedy", "markov")
THETA_GRID = tuple(i / 6 for i in range(6))
INFLATION_GRID = (1.4, 1.8, 2.0, 2.5, 3.0)
MCL_MAX_ITER = 100
MCL_PRUNE = 1e-5
MCL_TOL = 1e-6


def connected_components(nodes: Sequence[str],
                         edges: Sequence[Tuple[str, str, float]]) -> Dict[str, int]:
    return M.connected_components(nodes, [(a, b) for a, b, _ in edges])


def center(nodes: Sequence[str],
           edges: Sequence[Tuple[str, str, float]]) -> Dict[str, int]:
    return M.center_clustering(nodes, edges)


def merge_center(nodes: Sequence[str],
                 edges: Sequence[Tuple[str, str, float]]) -> Dict[str, int]:
    order = sorted(edges, key=lambda e: (-e[2], min(e[0], e[1]), max(e[0], e[1])))
    cluster_of: Dict[str, str] = {}
    is_centre: Set[str] = set()
    members: Dict[str, List[str]] = {}

    def absorb(into: str, other: str) -> None:
        for m in members[other]:
            cluster_of[m] = into
        members[into].extend(members[other])
        del members[other]
        is_centre.discard(other)

    for a, b, _ in order:
        ca, cb = cluster_of.get(a), cluster_of.get(b)
        if ca is None and cb is None:
            is_centre.add(a)
            cluster_of[a] = cluster_of[b] = a
            members[a] = [a, b]
        elif ca is not None and cb is None:
            cluster_of[b] = ca
            members[ca].append(b)
        elif ca is None and cb is not None:
            cluster_of[a] = cb
            members[cb].append(a)
        elif ca != cb:
            a_c, b_c = a in is_centre, b in is_centre
            if not (a_c or b_c):
                continue
            if a_c and b_c:
                keep, drop = (ca, cb) if a <= b else (cb, ca)
            elif a_c:
                keep, drop = ca, cb
            else:
                keep, drop = cb, ca
            absorb(keep, drop)

    lab: Dict[str, int] = {}
    out: Dict[str, int] = {}
    for n in nodes:
        c = cluster_of.get(n, n)
        if c not in lab:
            lab[c] = len(lab)
        out[n] = lab[c]
    return out


def star_disjoint(nodes: Sequence[str],
                  edges: Sequence[Tuple[str, str, float]]) -> Dict[str, int]:
    adj: Dict[str, List[str]] = collections.defaultdict(list)
    for a, b, _ in edges:
        adj[a].append(b)
        adj[b].append(a)
    deg = {n: len(adj.get(n, ())) for n in nodes}
    order = sorted(nodes, key=lambda n: (-deg[n], n))
    marked: Set[str] = set()
    lab: Dict[str, int] = {}
    out: Dict[str, int] = {}
    for c in order:
        if c in marked:
            continue
        cid = len(lab)
        lab[c] = cid
        out[c] = cid
        marked.add(c)
        for v in sorted(adj.get(c, ())):
            if v not in marked:
                marked.add(v)
                out[v] = cid
    for n in nodes:
        if n not in out:
            out[n] = len(lab) + 1
            lab[n] = out[n]
    return out


def correlation_greedy(nodes: Sequence[str], edges: Sequence[Tuple[str, str, float]],
                       theta: float) -> Dict[str, int]:
    import heapq
    idx = {n: i for i, n in enumerate(nodes)}
    size = [1] * len(nodes)
    members: Dict[int, List[int]] = {i: [i] for i in range(len(nodes))}
    alive = set(range(len(nodes)))
    w: Dict[Tuple[int, int], float] = collections.defaultdict(float)
    nbr: Dict[int, Set[int]] = collections.defaultdict(set)
    for a, b, s in edges:
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        w[(min(ia, ib), max(ia, ib))] += s
        nbr[ia].add(ib)
        nbr[ib].add(ia)

    def gain(i: int, j: int) -> float:
        return w.get((min(i, j), max(i, j)), 0.0) - theta * size[i] * size[j]

    heap = [(-gain(i, j), min(i, j), max(i, j))
            for i in alive for j in nbr[i] if j > i]
    heapq.heapify(heap)

    while heap:
        negg, i, j = heapq.heappop(heap)
        if i not in alive or j not in alive:
            continue
        cur = gain(i, j)
        if cur <= 1e-15:
            continue
        if abs(cur - (-negg)) > 1e-12:
            heapq.heappush(heap, (-cur, i, j))
            continue
        members[i].extend(members.pop(j))
        size[i] += size[j]
        alive.discard(j)
        for k2 in list(nbr[j]):
            if k2 == i or k2 not in alive:
                nbr[k2].discard(j)
                continue
            ki, kj = (min(i, k2), max(i, k2)), (min(j, k2), max(j, k2))
            w[ki] = w.get(ki, 0.0) + w.get(kj, 0.0)
            w.pop(kj, None)
            nbr[k2].discard(j)
            nbr[k2].add(i)
            nbr[i].add(k2)
        nbr[i].discard(j)
        w.pop((min(i, j), max(i, j)), None)
        nbr.pop(j, None)
        for k2 in nbr[i]:
            if k2 in alive:
                heapq.heappush(heap, (-gain(i, k2), min(i, k2), max(i, k2)))

    out: Dict[str, int] = {}
    for cid, i in enumerate(sorted(alive)):
        for m in members[i]:
            out[nodes[m]] = cid
    return out


def _correlation_rescan(nodes: Sequence[str], edges: Sequence[Tuple[str, str, float]],
                        theta: float) -> Dict[str, int]:
    idx = {n: i for i, n in enumerate(nodes)}
    comp: Dict[int, Set[int]] = {i: {i} for i in range(len(nodes))}
    w: Dict[Tuple[int, int], float] = collections.defaultdict(float)
    for a, b, s in edges:
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        w[(min(ia, ib), max(ia, ib))] += s
    while True:
        best, bestg = None, 1e-15
        for i in sorted(comp):
            for j in sorted(comp):
                if j <= i:
                    continue
                tot = sum(w.get((min(x, y), max(x, y)), 0.0)
                          for x in comp[i] for y in comp[j])
                g = tot - theta * len(comp[i]) * len(comp[j])
                if g > bestg + 1e-15:
                    best, bestg = (i, j), g
        if best is None:
            break
        i, j = best
        comp[i] |= comp.pop(j)
    out: Dict[str, int] = {}
    for cid, i in enumerate(sorted(comp)):
        for m in comp[i]:
            out[nodes[m]] = cid
    return out


def markov(nodes: Sequence[str], edges: Sequence[Tuple[str, str, float]],
           inflation: float) -> Dict[str, int]:
    adj: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
    for a, b, s in edges:
        adj[a][b] = max(adj[a].get(b, 0.0), s)
        adj[b][a] = max(adj[b].get(a, 0.0), s)
    comp = M.connected_components(nodes, [(a, b) for a, b, _ in edges])
    by_comp: Dict[int, List[str]] = collections.defaultdict(list)
    for n in nodes:
        by_comp[comp[n]].append(n)

    out: Dict[str, int] = {}
    nxt = 0
    for mem in by_comp.values():
        if len(mem) == 1:
            out[mem[0]] = nxt
            nxt += 1
            continue
        mem = sorted(mem)
        pos = {n: i for i, n in enumerate(mem)}
        k = len(mem)
        mat = [[0.0] * k for _ in range(k)]
        for n in mem:
            mat[pos[n]][pos[n]] = 1.0
            for m, s in adj.get(n, {}).items():
                if m in pos:
                    mat[pos[n]][pos[m]] = max(mat[pos[n]][pos[m]], s)
        for c in range(k):
            tot = sum(mat[r][c] for r in range(k))
            if tot:
                for r in range(k):
                    mat[r][c] /= tot
        for _ in range(MCL_MAX_ITER):
            sq = [[0.0] * k for _ in range(k)]
            for r in range(k):
                row = mat[r]
                for c in range(k):
                    v = 0.0
                    for t in range(k):
                        a = row[t]
                        if a:
                            b = mat[t][c]
                            if b:
                                v += a * b
                    sq[r][c] = v
            change = 0.0
            for c in range(k):
                col = [sq[r][c] ** inflation for r in range(k)]
                tot = sum(col)
                if tot:
                    col = [v / tot for v in col]
                for r in range(k):
                    v = col[r] if col[r] >= MCL_PRUNE else 0.0
                    change += abs(v - mat[r][c])
                    sq[r][c] = v
            for c in range(k):
                tot = sum(sq[r][c] for r in range(k))
                if tot:
                    for r in range(k):
                        sq[r][c] /= tot
            mat = sq
            if change < MCL_TOL:
                break
        pat = [(mem[r], mem[c]) for r in range(k) for c in range(k)
               if r != c and mat[r][c] > MCL_PRUNE]
        sub = M.connected_components(mem, pat)
        remap: Dict[int, int] = {}
        for n in mem:
            lab = sub[n]
            if lab not in remap:
                remap[lab] = nxt
                nxt += 1
            out[n] = remap[lab]
    return out


def rule_instances() -> List[Tuple[str, Optional[float], Any]]:
    inst: List[Tuple[str, Optional[float], Any]] = [
        ("connected_components", None, lambda n, e: connected_components(n, e)),
        ("center", None, lambda n, e: center(n, e)),
        ("merge_center", None, lambda n, e: merge_center(n, e)),
        ("star_disjoint", None, lambda n, e: star_disjoint(n, e)),
    ]
    for t in THETA_GRID:
        inst.append(("correlation_greedy", t,
                     (lambda tt: (lambda n, e: correlation_greedy(n, e, tt)))(t)))
    for f in INFLATION_GRID:
        inst.append(("markov", f, (lambda ff: (lambda n, e: markov(n, e, ff)))(f)))
    return inst


def key_of(rid: str, par: Optional[float]) -> str:
    return rid if par is None else "%s@%.4f" % (rid, par)


def selftest() -> int:
    ok = True

    nodes = ["a", "b", "c"]
    edges = [("a", "b", 0.9), ("b", "c", 0.8)]
    exp = {"connected_components": 1, "center": 2, "merge_center": 1, "star_disjoint": 1}
    for rid, par, fn in rule_instances():
        if par is not None:
            continue
        p = fn(nodes, edges)
        got = len(set(p.values()))
        if got != exp[rid]:
            print("  FAIL %-22s path a-b-c: %d clusters, expected %d" % (rid, got, exp[rid]))
            ok = False
        else:
            print("  ok   %-22s path a-b-c -> %d cluster(s)" % (rid, got))

    nodes = ["c1", "m1", "c2", "m2"]
    edges = [("c1", "m1", 0.9), ("c2", "m2", 0.8), ("m1", "m2", 0.1)]
    ncc = len(set(connected_components(nodes, edges).values()))
    nmc = len(set(merge_center(nodes, edges).values()))
    nce = len(set(center(nodes, edges).values()))
    if not (ncc == 1 and nmc == 2 and nce == 2):
        print("  FAIL two bridged stars: cc %d, merge_center %d, center %d; expected 1, 2, 2"
              % (ncc, nmc, nce))
        ok = False
    else:
        print("  ok   two stars bridged between non-centres: cc 1, merge_center 2, center 2")

    nodes = ["c1", "m1", "c2", "m2"]
    edges = [("c1", "m1", 0.9), ("c2", "m2", 0.8), ("m1", "c2", 0.1)]
    nmc = len(set(merge_center(nodes, edges).values()))
    nce = len(set(center(nodes, edges).values()))
    if not (nmc == 1 and nce == 2):
        print("  FAIL bridge touching a centre: merge_center %d, center %d; expected 1, 2"
              % (nmc, nce))
        ok = False
    else:
        print("  ok   bridge touching a centre: merge_center 1, center 2")

    nodes = ["a", "b", "c"]
    edges = [("a", "b", 0.5), ("b", "c", 0.5)]
    lo = correlation_greedy(nodes, edges, 0.0)
    hi = correlation_greedy(nodes, edges, 0.9)
    if len(set(lo.values())) != 1 or len(set(hi.values())) != 3:
        print("  FAIL correlation_greedy theta response: %d then %d clusters"
              % (len(set(lo.values())), len(set(hi.values()))))
        ok = False
    else:
        print("  ok   correlation_greedy theta=0 -> 1 cluster, theta=0.9 -> 3 singletons")

    nodes = ["a", "b", "c"]
    edges = [("a", "b", 0.34), ("b", "c", 0.34)]
    mid = correlation_greedy(nodes, edges, 1 / 3)
    print("  info correlation_greedy theta=1/3 on a weak chain -> %d clusters"
          % len(set(mid.values())))

    nodes = ["a", "b", "c", "x", "y", "z"]
    edges = [("a", "b", 1.0), ("b", "c", 1.0), ("a", "c", 1.0),
             ("x", "y", 1.0), ("y", "z", 1.0), ("x", "z", 1.0),
             ("c", "x", 0.2)]
    m14 = len(set(markov(nodes, edges, 1.4).values()))
    m30 = len(set(markov(nodes, edges, 3.0).values()))
    print("  info markov inflation 1.4 -> %d clusters, 3.0 -> %d clusters" % (m14, m30))
    if m30 < m14:
        print("  FAIL higher inflation produced fewer clusters, which inverts the mechanism")
        ok = False
    else:
        print("  ok   markov granularity is monotone in inflation on this graph")

    for rid, par, fn in rule_instances():
        p = fn(nodes, edges)
        if set(p) != set(nodes):
            print("  FAIL %s does not cover the node set" % key_of(rid, par))
            ok = False
    if ok:
        print("  ok   every rule returns a partition covering the node set")

    import random
    rnd = random.Random(20261027)
    bad = 0
    for trial in range(60):
        n = rnd.randrange(3, 11)
        ns = ["r%02d" % i for i in range(n)]
        es = []
        for i in range(n):
            for j in range(i + 1, n):
                if rnd.random() < 0.35:
                    es.append((ns[i], ns[j], rnd.choice(THETA_GRID[1:]) ))
        th = rnd.choice(THETA_GRID)
        p1 = correlation_greedy(ns, es, th)
        p2 = _correlation_rescan(ns, es, th)
        g1 = {frozenset(k for k in p1 if p1[k] == c) for c in set(p1.values())}
        g2 = {frozenset(k for k in p2 if p2[k] == c) for c in set(p2.values())}
        if g1 != g2:
            bad += 1
            if bad <= 2:
                print("  FAIL correlation_greedy heap != rescan, trial %d, theta %.4f, "
                      "%d nodes %d edges" % (trial, th, n, len(es)))
    if bad:
        print("  FAIL heap and rescan disagree on %d of 60 random graphs" % bad)
        ok = False
    else:
        print("  ok   correlation_greedy heap agrees with the exhaustive rescan, 60/60 "
              "random graphs")
    return 0 if ok else 1


def evaluate(entry: Dict[str, Any], insts: Sequence[Tuple[str, Optional[float], Any]],
             tag: str) -> Dict[str, Dict[str, Any]]:
    nodes: List[str] = entry["nodes"]
    ent: Dict[str, str] = entry["entity_of"]
    score: Dict[str, float] = entry["edge_score_pool_vote_fraction"]
    res: Dict[str, Dict[str, int]] = entry["resources_by_recipient_arm"]
    out: Dict[str, Dict[str, Any]] = {}
    for cellkey, cell in sorted(entry["cells"].items()):
        links = [k for k, d in cell["decisions"].items() if d == "match"]
        edges = []
        for k in links:
            a, b = k.split("\t")
            edges.append((a, b, score.get(k, 0.0)))
        comp = M.connected_components(nodes, [(a, b) for a, b, _ in edges])
        sizes = collections.Counter(comp.values())
        per: Dict[str, Any] = {}
        for rid, par, fn in insts:
            t0 = time.perf_counter()
            part = fn(nodes, edges)
            cpu = time.perf_counter() - t0
            cs = M.cluster_scores(part, ent, nodes)
            per[key_of(rid, par)] = {
                "rule": rid, "parameter": par,
                "cluster_f1": cs["f1"], "cluster_precision": cs["precision"],
                "cluster_recall": cs["recall"], "n_clusters": cs["n_clusters"],
                "largest_cluster": cs["largest_cluster"], "singletons": cs["singletons"],
                "predicted_intra_pairs": cs["predicted_intra_pairs"],
                "true_intra_pairs": cs["true_intra_pairs"],
                "cpu_seconds": cpu}
        rkey = "%s|%s" % (cell["recipient"], cell["arm"])
        out[cellkey] = {
            "recipient": cell["recipient"], "arm": cell["arm"], "draw": cell["draw"],
            "n_links": len(edges), "n_pairs_decided": cell["n_pairs"],
            "largest_link_component": max(sizes.values()) if sizes else 0,
            "n_link_components_over_1": sum(1 for v in sizes.values() if v > 1),
            "gold_pairs_in_universe": M.cluster_scores(
                {n: i for i, n in enumerate(nodes)}, ent, nodes)["gold_pairs_in_universe"],
            "resources": res.get(rkey, {}),
            "rules": per}
        print("    %-14s %-26s links %6d  cc %.4f  center %.4f  mc %.4f  star %.4f"
              % (tag, cellkey, len(edges),
                 per["connected_components"]["cluster_f1"], per["center"]["cluster_f1"],
                 per["merge_center"]["cluster_f1"], per["star_disjoint"]["cluster_f1"]),
              flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print("=== self-test: counterexamples separating the rules ===")
        return selftest()

    OUT.mkdir(parents=True, exist_ok=True)
    insts = rule_instances()
    keys = [key_of(r, p) for r, p, _ in insts]
    print("rule instances: %d  (%s)" % (len(insts), ", ".join(keys)))
    todo = [a.dataset] if a.dataset else list(DATASETS)

    report: Dict[str, Any] = {"rule_instances": keys,
                              "theta_grid": list(THETA_GRID),
                              "inflation_grid": list(INFLATION_GRID),
                              "domains": {}}
    rows: List[Dict[str, Any]] = []
    sel_rows: List[Dict[str, Any]] = []

    for ds in todo:
        dev_p, test_p = CACHE / ("%s_dev.json" % ds), CACHE / ("%s_test.json" % ds)
        if not (dev_p.exists() and test_p.exists()):
            print("absent, skipped: %s" % ds)
            continue
        dev = json.loads(dev_p.read_text(encoding="utf-8"))
        test = json.loads(test_p.read_text(encoding="utf-8"))

        de, te = set(dev["entity_of"].values()), set(test["entity_of"].values())
        dn, tn = set(dev["nodes"]), set(test["nodes"])
        assert not (de & te), "%s: %d entities appear in both splits" % (ds, len(de & te))
        assert not (dn & tn), "%s: %d records appear in both splits" % (ds, len(dn & tn))
        print("\n===== %s: %d dev entities, %d test entities, 0 shared; "
              "%d dev records, %d test records, 0 shared ====="
              % (ds, len(de), len(te), len(dn), len(tn)), flush=True)

        d_eval = evaluate(dev, insts, "dev")
        t_eval = evaluate(test, insts, "test")

        order_of = {k: i for i, k in enumerate(keys)}
        per_cell: Dict[str, Any] = {}
        for cellkey in sorted(set(d_eval) & set(t_eval)):
            dv, tv = d_eval[cellkey], t_eval[cellkey]
            best_dev = max(dv["rules"], key=lambda k: (dv["rules"][k]["cluster_f1"],
                                                       -order_of[k]))
            top = max(dv["rules"][k]["cluster_f1"] for k in dv["rules"])
            tied = sorted([k for k in dv["rules"]
                           if dv["rules"][k]["cluster_f1"] == top], key=lambda k: order_of[k])
            chosen = tied[0]
            best_test = max(tv["rules"], key=lambda k: tv["rules"][k]["cluster_f1"])
            identical_cost = len({(json.dumps(tv["resources"], sort_keys=True))}) == 1
            per_cell[cellkey] = {
                "recipient": tv["recipient"], "arm": tv["arm"], "draw": tv["draw"],
                "dev_tie_set_size": len(tied), "dev_tie_set": tied,
                "selected": chosen,
                "selected_dev_cluster_f1": dv["rules"][chosen]["cluster_f1"],
                "selected_test_cluster_f1": tv["rules"][chosen]["cluster_f1"],
                "retrospective_best_rule": best_test,
                "retrospective_best_test_cluster_f1": tv["rules"][best_test]["cluster_f1"],
                "regret_pp": 100 * (tv["rules"][best_test]["cluster_f1"]
                                    - tv["rules"][chosen]["cluster_f1"]),
                "fixed_rule_regret_pp": {
                    k: 100 * (tv["rules"][best_test]["cluster_f1"] - tv["rules"][k]["cluster_f1"])
                    for k in keys},
                "resources": tv["resources"],
                "resources_identical_across_rules": identical_cost,
                "cpu_seconds_by_rule": {k: tv["rules"][k]["cpu_seconds"] for k in keys},
            }
            sel_rows.append({
                "dataset": ds, "cell": cellkey, "recipient": tv["recipient"], "arm": tv["arm"],
                "draw": tv["draw"], "n_links_test": tv["n_links"],
                "dev_tie_set_size": len(tied), "selected": chosen,
                "selected_dev_cluster_f1": dv["rules"][chosen]["cluster_f1"],
                "selected_test_cluster_f1": tv["rules"][chosen]["cluster_f1"],
                "retrospective_best_rule": best_test,
                "retrospective_best_test_cluster_f1": tv["rules"][best_test]["cluster_f1"],
                "regret_pp": per_cell[cellkey]["regret_pp"],
                "calls": tv["resources"].get("calls"),
                "prompt_tokens": tv["resources"].get("prompt_tokens"),
                "completion_tokens": tv["resources"].get("completion_tokens")})
            for k in keys:
                rows.append({
                    "dataset": ds, "cell": cellkey, "recipient": tv["recipient"],
                    "arm": tv["arm"], "draw": tv["draw"], "rule": k,
                    "rule_family": tv["rules"][k]["rule"],
                    "parameter": tv["rules"][k]["parameter"],
                    "dev_cluster_f1": dv["rules"][k]["cluster_f1"],
                    "test_cluster_f1": tv["rules"][k]["cluster_f1"],
                    "test_cluster_precision": tv["rules"][k]["cluster_precision"],
                    "test_cluster_recall": tv["rules"][k]["cluster_recall"],
                    "test_n_clusters": tv["rules"][k]["n_clusters"],
                    "test_largest_cluster": tv["rules"][k]["largest_cluster"],
                    "test_singletons": tv["rules"][k]["singletons"],
                    "test_n_links": tv["n_links"],
                    "cpu_seconds_test": tv["rules"][k]["cpu_seconds"]})

        report["domains"][ds] = {
            "dev_entities": len(de), "test_entities": len(te),
            "dev_records": len(dn), "test_records": len(tn),
            "entity_overlap": 0, "record_overlap": 0,
            "candidate_coverage_dev": dev["coverage"]["coverage"],
            "candidate_coverage_test": test["coverage"]["coverage"],
            "attainable_direct_link_f1_test": test["coverage"]["attainable_direct_link_f1"],
            "dev": d_eval, "test": t_eval, "selection": per_cell}

        cells = sorted(per_cell)
        print("\n  --- %s: development selection over %d cells ---" % (ds, len(cells)))
        print("  %-26s %-24s %9s %9s %9s %8s"
              % ("cell", "selected rule", "dev F1", "test F1", "best F1", "regret"))
        for c in cells:
            v = per_cell[c]
            print("  %-26s %-24s %9.4f %9.4f %9.4f %+7.2f"
                  % (c, v["selected"], v["selected_dev_cluster_f1"],
                     v["selected_test_cluster_f1"],
                     v["retrospective_best_test_cluster_f1"], v["regret_pp"]))
        mean_regret = sum(per_cell[c]["regret_pp"] for c in cells) / len(cells)
        print("  mean regret of the development choice: %+.3f pp; worst %+.3f pp"
              % (mean_regret, max(per_cell[c]["regret_pp"] for c in cells)))
        print("  %-24s %12s %12s" % ("always this rule", "mean regret", "worst regret"))
        for k in keys:
            vals = [per_cell[c]["fixed_rule_regret_pp"][k] for c in cells]
            print("  %-24s %+11.3f %+12.3f" % (k, sum(vals) / len(vals), max(vals)))
        chosen_count = collections.Counter(per_cell[c]["selected"] for c in cells)
        print("  rules the development split actually chose: %s" % dict(chosen_count))
        report["domains"][ds]["summary"] = {
            "n_cells": len(cells),
            "mean_regret_pp": mean_regret,
            "worst_regret_pp": max(per_cell[c]["regret_pp"] for c in cells),
            "fixed_rule_mean_regret_pp": {
                k: sum(per_cell[c]["fixed_rule_regret_pp"][k] for c in cells) / len(cells)
                for k in keys},
            "fixed_rule_worst_regret_pp": {
                k: max(per_cell[c]["fixed_rule_regret_pp"][k] for c in cells) for k in keys},
            "selected_rule_counts": dict(chosen_count)}

    with open(OUT / "eb1_grouping.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, sort_keys=True, default=str)
    for name, data in (("eb1_rule_cells.csv", rows), ("eb1_selection.csv", sel_rows)):
        if not data:
            continue
        cols: List[str] = []
        for r in data:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
        print("wrote output/%s (%d rows)" % (name, len(data)))
    print("wrote output/eb1_grouping.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
