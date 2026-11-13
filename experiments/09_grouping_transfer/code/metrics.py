"""Pair, coverage and grouping metrics used by the matching experiments."""
from __future__ import annotations

import collections
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# --------------------------------------------------------------------- decisions
def modal_decision(decs: Sequence[Optional[str]]) -> Tuple[Optional[str], str]:
    """Modal decision over repeats; a tie abstains. Returns (decision, reason)."""
    v = [d for d in decs if d is not None]
    if not v:
        return None, "no_parsed_decision"
    c = collections.Counter(v)
    top = max(c.values())
    w = [k for k, n in c.items() if n == top]
    if len(w) != 1:
        return None, "tie_abstain"
    return w[0], "modal"


def pair_scores(dec: Dict[Tuple[str, str], Optional[str]],
                ent: Dict[str, str]) -> Dict[str, Any]:
    """Positive-class F1 over the assigned candidate set."""
    tp = fp = fn = tn = ab = 0
    ab_pos = 0
    for (a, b), d in dec.items():
        gold = 1 if ent[a] == ent[b] else 0
        if d == "match":
            tp += gold
            fp += 1 - gold
        elif d == "non-match":
            fn += gold
            tn += 1 - gold
        else:
            ab += 1
            ab_pos += gold
            fn += gold                  # an abstention yields no positive, so it costs recall
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"n_pairs": len(dec), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "abstentions": ab, "abstentions_on_gold_positive": ab_pos,
            "abstention_rate": ab / len(dec) if dec else 0.0,
            "precision": P, "recall": R,
            "f1": (2 * P * R / (P + R)) if P + R else 0.0}


# --------------------------------------------------------------------- grouping
def connected_components(nodes: Iterable[str],
                         edges: Iterable[Tuple[str, str]]) -> Dict[str, int]:
    parent: Dict[str, str] = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a not in parent or b not in parent:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    lab: Dict[str, int] = {}
    out: Dict[str, int] = {}
    for n in parent:
        r = find(n)
        if r not in lab:
            lab[r] = len(lab)
        out[n] = lab[r]
    return out


def center_clustering(nodes: Iterable[str],
                      scored_edges: Sequence[Tuple[str, str, float]]) -> Dict[str, int]:
    """Center clustering, Hassanzadeh et al. PVLDB 2009, exactly as frozen in the protocol.

    Edges descending by score, ties by (min id, max id) ascending. A record is assigned once and
    never reassigned; two centres are never merged.
    """
    ns = list(nodes)
    order = sorted(scored_edges, key=lambda e: (-e[2], min(e[0], e[1]), max(e[0], e[1])))
    centre_of: Dict[str, str] = {}
    is_centre: Set[str] = set()
    for a, b, _ in order:
        ia, ib = a in centre_of, b in centre_of
        if not ia and not ib:
            is_centre.add(a)
            centre_of[a] = a
            centre_of[b] = a
        elif a in is_centre and not ib:
            centre_of[b] = a
        elif b in is_centre and not ia:
            centre_of[a] = b
    lab: Dict[str, int] = {}
    out: Dict[str, int] = {}
    for n in ns:
        c = centre_of.get(n, n)
        if c not in lab:
            lab[c] = len(lab)
        out[n] = lab[c]
    return out


def cluster_scores(part: Dict[str, int], ent: Dict[str, str],
                   nodes: Sequence[str]) -> Dict[str, Any]:
    """Pairwise cluster F1 = 2T/(N+G) over the record universe `nodes`."""
    by_cluster = collections.defaultdict(list)
    for n in nodes:
        by_cluster[part[n]].append(n)
    N = T = 0
    sizes = []
    for mem in by_cluster.values():
        k = len(mem)
        sizes.append(k)
        N += k * (k - 1) // 2
        c = collections.Counter(ent[m] for m in mem)
        T += sum(v * (v - 1) // 2 for v in c.values())
    ge = collections.Counter(ent[n] for n in nodes)
    G = sum(v * (v - 1) // 2 for v in ge.values())
    P = T / N if N else 0.0
    R = T / G if G else 0.0
    return {"n_records": len(nodes), "n_clusters": len(by_cluster),
            "predicted_intra_pairs": N, "true_intra_pairs": T, "gold_pairs_in_universe": G,
            "precision": P, "recall": R,
            "f1": (2 * T / (N + G)) if (N + G) else 0.0,
            "largest_cluster": max(sizes) if sizes else 0,
            "singletons": sum(1 for s in sizes if s == 1),
            "cluster_size_distribution": dict(sorted(collections.Counter(sizes).items()))}


def amplification(part: Dict[str, int], nodes: Sequence[str],
                  n_links: int) -> Optional[float]:
    """Predicted intra-cluster pairs per asserted link: how far closure propagated."""
    by = collections.Counter(part[n] for n in nodes)
    N = sum(k * (k - 1) // 2 for k in by.values())
    return (N / n_links) if n_links else None


def coverage(cand: Sequence[Tuple[str, str]], ent: Dict[str, str],
             nodes: Sequence[str]) -> Dict[str, Any]:
    """What fraction of the universe's gold pairs retrieval even proposed."""
    ge = collections.Counter(ent[n] for n in nodes)
    G = sum(v * (v - 1) // 2 for v in ge.values())
    pos = sum(1 for a, b in cand if ent[a] == ent[b])
    return {"candidates": len(cand), "candidate_positives": pos,
            "gold_pairs_in_universe": G,
            "coverage": pos / G if G else None,
            "attainable_direct_link_f1": (2 * pos / (pos + G)) if (pos + G) else None}


# --------------------------------------------------------------------- resources
def resource_ledger(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Summed from stored calls, by role. Never derived from a price formula."""
    agg: Dict[str, Dict[str, float]] = collections.defaultdict(
        lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "truncated": 0})
    for r in rows:
        k = r.get("role", "unknown")
        a = agg[k]
        a["calls"] += 1
        a["prompt_tokens"] += r.get("prompt_tokens") or 0
        a["completion_tokens"] += r.get("completion_tokens") or 0
        a["truncated"] += 1 if r.get("truncated") else 0
    out = {k: dict(v) for k, v in agg.items()}
    tot = {"calls": sum(v["calls"] for v in agg.values()),
           "prompt_tokens": sum(v["prompt_tokens"] for v in agg.values()),
           "completion_tokens": sum(v["completion_tokens"] for v in agg.values())}
    out["total"] = tot
    return out


def arm_resources(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per (recipient, arm): calls and tokens, which is the RQ4 marginal view."""
    agg: Dict[Tuple[str, str], Dict[str, float]] = collections.defaultdict(
        lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "truncated": 0})
    for r in rows:
        a = agg[(r["recipient"], r["arm"])]
        a["calls"] += 1
        a["prompt_tokens"] += r.get("prompt_tokens") or 0
        a["completion_tokens"] += r.get("completion_tokens") or 0
        a["truncated"] += 1 if r.get("truncated") else 0
    return {f"{k[0]}|{k[1]}": dict(v) for k, v in agg.items()}


# --------------------------------------------------------------------- loading
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def arm_decisions(rows: Sequence[Dict[str, Any]], recipient: str, arm: str,
                  draw: Optional[int] = None) -> Dict[Tuple[str, str], Optional[str]]:
    """Modal decision per pair for one (recipient, arm[, draw]) cell."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["recipient"] != recipient or r["arm"] != arm:
            continue
        if draw is not None and r.get("draw") != draw:
            continue
        by[(r["a"], r["b"])].append(r["decision"])
    return {k: modal_decision(v)[0] for k, v in by.items()}


def pool_view(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    cell = collections.defaultdict(list)
    for r in rows:
        cell[(r["a"], r["b"], r["model"])].append(r)
    out = {}
    for k, rs in cell.items():
        d, why = modal_decision([x["decision"] for x in rs])
        rep = next((x["raw_reply"] for x in rs if x["decision"] == d), None)
        out[k] = {"model": k[2], "decision": d, "raw_reply": rep, "reason": why,
                  "n_repeats": len(rs)}
    return out


def vote_fraction(pv: Dict[Tuple[str, str, str], Dict[str, Any]],
                  a: str, b: str, models: Sequence[str]) -> float:
    """The frozen edge score: share of pool models whose modal decision is match."""
    ds = [pv[(a, b, m)]["decision"] for m in models if (a, b, m) in pv]
    ds = [d for d in ds if d is not None]
    if not ds:
        return 0.0
    return sum(1 for d in ds if d == "match") / len(ds)
