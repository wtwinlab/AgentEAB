r"""Independent recomputation. Shares nothing with the analysis modules but the raw JSONL and the dataset.

    PYTHONPATH=experiments/05_matching_downstream/code python3.13 \
      experiments/05_matching_downstream/code/recheck.py

Re-running `analyze_stages.py` would only show it is deterministic. This file reimplements every
headline quantity along a different path:

  * a parser written from the task statement rather than imported
  * the gold partition rebuilt by reading the two JSON files directly, not through `Settlements`
  * blocking reimplemented with explicit set operations instead of an inverted index
  * components found by breadth-first search instead of union-find
  * implied true pairs counted by explicit enumeration instead of contingency products
  * the universe counted by `comb(n, 2)` rather than by a loop

A disagreement is printed as a disagreement. The exit code is non-zero if anything fails, so this is
a gate rather than a report.
"""
from __future__ import annotations

import collections
import itertools
import json
import math
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(BASE, "raw")
OUT = os.path.join(BASE, "analysis")
DATA = os.path.join(BASE, "data", "gs", "geographicalSettelments")

fails: List[str] = []
checks = 0


def eq(label: str, got: Any, want: Any, tol: float = 1e-9) -> None:
    global checks
    checks += 1
    ok = (abs(got - want) <= tol) if isinstance(got, float) or isinstance(want, float) \
        else got == want
    if not ok:
        fails.append(f"{label}: independent {got!r} vs published {want!r}")


# ---------------------------------------------------------------- an independent parser
def my_parse(text: str) -> Optional[str]:
    """Written from the instruction "one word on the final line, either match or non-match"."""
    if not text:
        return None
    for ln in reversed([l.strip() for l in text.splitlines() if l.strip()]):
        t = re.sub(r"[^a-z]", "", ln.lower())
        if t in ("match",):
            return "match"
        if t in ("nonmatch",):
            return "non-match"
        break
    low = (text or "").lower()
    neg = [m.end() for m in re.finditer(
        r"non[\s-]?match|no match|not a match|not match|do(?:es)?n?'?o?t? ?not match|"
        r"do(?:es)? not match|don't match|doesn't match", low)]
    pos = [m.end() for m in re.finditer(r"(?<![a-z])match(?![a-z])", low)]
    pos = [p for p in pos if not any(abs(p - n) <= 1 for n in neg)]
    if not neg and not pos:
        return None
    return "non-match" if (neg and (not pos or max(neg) >= max(pos))) else "match"


def my_gold() -> Tuple[Dict[int, Dict[str, Any]], Dict[int, int], Dict[int, List[int]]]:
    rec = {}
    for line in open(os.path.join(DATA, "settlements.json"), encoding="utf-8"):
        o = json.loads(line)
        rec[int(o["id"])] = o["data"]
    raw = [json.loads(l) for l in
           open(os.path.join(DATA, "combinedSettlements(PerfectMatch).json"), encoding="utf-8")]
    seen = collections.Counter()
    for c in raw:
        for x in c["data"]["clusteredVertices"]:
            seen[int(x)] += 1
    clusters = {}
    for c in raw:
        mem = sorted(int(x) for x in c["data"]["clusteredVertices"])
        if len(mem) == 1 and seen[mem[0]] > 1:
            continue
        clusters[int(c["id"])] = mem
    cof = {x: cid for cid, mem in clusters.items() for x in mem}
    return rec, cof, clusters


def my_norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def my_grams(s: str) -> Set[str]:
    t = f"  {s} "
    return {t[i:i + 3] for i in range(max(0, len(t) - 2))} if s else set()


def my_block(rec: Dict[int, Dict[str, Any]], rids: List[int], k: int) -> Set[Tuple[int, int]]:
    """Explicit all-pairs Jaccard, no inverted index. O(n^2) and that is the point."""
    gr = {r: my_grams(my_norm(rec[r].get("label", ""))) for r in rids}
    out: Set[Tuple[int, int]] = set()
    for r in rids:
        sc = []
        for o in rids:
            if o == r:
                continue
            u = gr[r] | gr[o]
            if not u:
                continue
            j = len(gr[r] & gr[o]) / len(u)
            if j > 0:
                sc.append((-j, o))
        sc.sort()
        for _j, o in sc[:k]:
            out.add((r, o) if r < o else (o, r))
    return out


def my_components(nodes: List[int], edges: Set[Tuple[int, int]]) -> List[List[int]]:
    """Breadth-first search, not union-find."""
    adj: Dict[int, List[int]] = collections.defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen: Set[int] = set()
    out = []
    for n in nodes:
        if n in seen:
            continue
        comp = [n]
        seen.add(n)
        qi = 0
        while qi < len(comp):
            v = comp[qi]
            qi += 1
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    comp.append(w)
        out.append(sorted(comp))
    return out


def my_f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


def main() -> int:
    global checks
    rec, cof, clusters = my_gold()
    pub_stages = json.load(open(os.path.join(OUT, "stages.json"), encoding="utf-8"))
    pub_cf = json.load(open(os.path.join(OUT, "counterfactual.json"), encoding="utf-8"))

    print("=== dataset and partition, rebuilt from the two JSON files ===")
    eq("records", len(rec), 3054)
    eq("clusters after the redundant-singleton rule", len(clusters), 819)
    gp_all = sum(len(m) * (len(m) - 1) // 2 for m in clusters.values())
    eq("gold pairs over the whole dataset", gp_all, 4391)
    print(f"  {len(rec)} records, {len(clusters)} clusters, {gp_all} gold pairs")

    import hashlib
    def split_of(cid: int) -> str:
        h = hashlib.sha256(f"gs-entity-split-2026-10-13|{cid}".encode()).hexdigest()
        return "dev" if int(h, 16) % 2 == 0 else "test"
    test_cl = {c: m for c, m in clusters.items() if split_of(c) == "test"}
    nodes = sorted(x for m in test_cl.values() for x in m)
    Gtot = sum(len(m) * (len(m) - 1) // 2 for m in test_cl.values())
    eq("test records", len(nodes), pub_stages["universe"]["records"])
    eq("test all-record-pair universe", math.comb(len(nodes), 2),
       pub_stages["universe"]["all_record_pairs"])
    eq("test gold pairs", Gtot, pub_stages["universe"]["gold_pairs"])
    print(f"  test: {len(nodes)} records, {math.comb(len(nodes),2):,} pairs, {Gtot} gold pairs")

    print("\n=== blocking, reimplemented as explicit all-pairs Jaccard ===")
    cand = my_block(rec, nodes, 5)
    eq("candidate pairs", len(cand), pub_stages["universe"]["candidate_pairs"])
    hit = sum(1 for (a, b) in cand if cof[a] == cof[b])
    print(f"  {len(cand)} candidates, blocking recall {hit/Gtot:.4f} ({hit}/{Gtot})")
    eq("blocking recall matches the freeze", round(hit / Gtot, 4), 0.9935, tol=5e-5)

    print("\n=== parser: an independent reading of every stored reply ===")
    recs = [json.loads(l) for l in open(os.path.join(RAW, "gs_preds_test.jsonl"),
                                        encoding="utf-8")]
    ok = [r for r in recs if r.get("status") == "ok"]
    eq("records collected", len(recs), pub_stages["records_collected"])
    eq("records ok", len(ok), pub_stages["records_ok"])
    dis = sum(1 for r in ok if my_parse(r.get("raw_reply")) != r.get("decision"))
    print(f"  {len(ok):,} replies, {dis} disagreements with the stored decision "
          f"({dis/len(ok):.4%})")
    checks += 1
    if dis / len(ok) > 0.01:
        fails.append(f"independent parser disagrees on {dis/len(ok):.2%} of replies, over 1%")

    print("\n=== the three stages, recomputed with BFS components and enumerated pairs ===")
    cell: Dict[Tuple[str, str], List[Optional[str]]] = collections.defaultdict(list)
    for r in ok:
        cell[(r["pair_key"], r["model"])].append(r["decision"])

    def modal(v: List[Optional[str]]) -> Optional[str]:
        w = [x for x in v if x is not None]
        if not w:
            return None
        c = collections.Counter(w)
        t = max(c.values())
        top = [k for k, n in c.items() if n == t]
        return top[0] if len(top) == 1 else None

    models = sorted({r["model"] for r in ok})
    print(f"  {'method':22s}{'A:F1':>9s}{'B:F1':>9s}{'C:F1':>9s}  agree with stages.json")
    for m in models + ["vote:majority"]:
        if m == "vote:majority":
            ass = {f"{a}|{b}" for (a, b) in cand
                   if modal([modal(cell.get((f"{a}|{b}", mm), [])) for mm in models]) == "match"}
            name = "vote:majority"
        else:
            ass = {f"{a}|{b}" for (a, b) in cand if modal(cell.get((f"{a}|{b}", m), [])) == "match"}
            name = f"single:{m}"
        tp = sum(1 for (a, b) in cand if f"{a}|{b}" in ass and cof[a] == cof[b])
        fp = sum(1 for (a, b) in cand if f"{a}|{b}" in ass and cof[a] != cof[b])
        fn = sum(1 for (a, b) in cand if f"{a}|{b}" not in ass and cof[a] == cof[b])
        A = my_f1(tp, fp, fn)
        B = my_f1(tp, len(ass) - tp, Gtot - tp)
        comps = my_components(nodes, {tuple(int(z) for z in k.split("|")) for k in ass})
        cn = ct = 0
        for comp in comps:
            if len(comp) < 2:
                continue
            for x, y in itertools.combinations(comp, 2):
                cn += 1
                ct += 1 if cof[x] == cof[y] else 0
        C = my_f1(ct, cn - ct, Gtot - ct)
        pub = pub_stages["methods"][name]
        a_ok = abs(A - pub["A"]["f1"]) < 1e-9
        b_ok = abs(B - pub["B"]["f1"]) < 1e-9
        c_ok = abs(C - pub["C"]["f1"]) < 1e-9
        eq(f"{name} stage A", A, pub["A"]["f1"])
        eq(f"{name} stage B", B, pub["B"]["f1"])
        eq(f"{name} stage C", C, pub["C"]["f1"])
        eq(f"{name} asserted links", len(ass), pub["topology"]["asserted_links"])
        eq(f"{name} implied pairs", cn, pub["topology"]["implied_pairs"])
        print(f"  {name:22s}{A:9.4f}{B:9.4f}{C:9.4f}  "
              f"{'all three' if a_ok and b_ok and c_ok else 'MISMATCH'}")

    print("\n=== the merge condition, re-derived on a sample of deletions ===")
    import csv as _csv
    rows = list(_csv.DictReader(open(os.path.join(OUT, "edge_deletion.csv"))))
    eq("deletion rows", len(rows), pub_cf["checks"]["checked"])
    sample = [r for r in rows if r["is_bridge"] == "True"][:400]
    bad = 0
    for r in sample:
        b = int(r["b_pairs_separated"])
        t = int(r["t_true_pairs_separated"])
        F = float(r["F_before"])
        N = int(r["N_before"])
        pred = (F * b - 2 * t) / (N + Gtot - b)
        if abs(pred - float(r["delta_f1"])) > 1e-10:
            bad += 1
        improves = (t / b) < (F / 2) if b else False
        if improves != (float(r["delta_f1"]) > 1e-15):
            bad += 1
    checks += 1
    if bad:
        fails.append(f"the deletion formula disagrees on {bad} of {len(sample)} sampled bridges")
    print(f"  {len(sample)} bridge deletions re-derived from b, t, F and N: {bad} disagreements")
    eq("published formula mismatches", pub_cf["checks"]["formula_mismatches"], 0)
    eq("published counting mismatches", pub_cf["checks"]["counting_mismatches"], 0)
    eq("published non-bridge deletions that changed the metric",
       pub_cf["checks"]["nonbridge_changed"], 0)

    sp = os.path.join(OUT, "sharing.json")
    if os.path.exists(sp):
        print("\n=== sharing levels, recomputed ===")
        pub_sh = json.load(open(sp, encoding="utf-8"))
        sh = [json.loads(l) for l in open(os.path.join(RAW, "gs_share_test.jsonl"),
                                          encoding="utf-8")]
        sok = [r for r in sh if r.get("status") == "ok"]
        eq("sharing records", len(sh), pub_sh["records"])
        eq("sharing ok", len(sok), pub_sh["ok"])
        idx = {(r["pair_key"], r["recipient"], r["condition"]): r for r in sok}
        conds = sorted({r["condition"] for r in sok})
        recips = sorted({r["recipient"] for r in sok})
        common = [(a, b) for (a, b) in cand
                  if all((f"{a}|{b}", rr, c) in idx for rr in recips for c in conds)]
        eq("sharing common population", len(common), pub_sh["common_population"])
        dis2 = sum(1 for r in sok if my_parse(r.get("raw_reply")) != r.get("decision"))
        print(f"  {len(sok):,} sharing replies, {dis2} parser disagreements "
              f"({dis2/max(1,len(sok)):.4%})")
        for rr in recips:
            for c in conds:
                ass = {f"{a}|{b}" for (a, b) in common
                       if idx[(f"{a}|{b}", rr, c)]["decision"] == "match"}
                tp = sum(1 for (a, b) in common if f"{a}|{b}" in ass and cof[a] == cof[b])
                fp = sum(1 for (a, b) in common if f"{a}|{b}" in ass and cof[a] != cof[b])
                fn = sum(1 for (a, b) in common if f"{a}|{b}" not in ass and cof[a] == cof[b])
                eq(f"{rr}/{c} stage A", my_f1(tp, fp, fn), pub_sh["levels"][rr][c]["A"]["f1"])
        print(f"  all {len(recips)*len(conds)} recipient x condition stage-A values agree")

    print(f"\n{'=' * 78}\n  {checks} independent checks, {len(fails)} failed")
    for f in fails:
        print(f"    FAIL {f}")
    with open(os.path.join(OUT, "recheck.json"), "w", encoding="utf-8") as f:
        json.dump({"checks": checks, "failures": fails,
                   "independent_paths": [
                       "parser written from the task statement",
                       "gold partition read directly from the two JSON files",
                       "blocking as explicit all-pairs Jaccard, no inverted index",
                       "components by breadth-first search, not union-find",
                       "implied true pairs by explicit enumeration, not contingency products",
                       "universe size by comb(n,2)",
                       "the deletion formula re-derived from b, t, F and N"]},
                  f, indent=2)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
