#!/usr/bin/env python3
"""Downstream grouping: does an intervention that changes pair judgments change the groups?

    python3 analyze_downstream.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import itertools
import json
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "analysis/controlled_sharing"
RAW = ROOT / "data/records/controlled_sharing"
MATCHING = ("wdc_products", "geographic_settlements")
PROTOS = ["ford", "moa", "du_mad", "madm2_objective", "bccs"]
BOOT = 1000
SEED = 20261019


class DSU:
    def __init__(self):
        self.p: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def groups_from_edges(records: Set[str], edges: Set[Tuple[str, str]]) -> List[Set[str]]:
    d = DSU()
    for r in records:
        d.find(r)
    for a, b in edges:
        d.union(a, b)
    by: Dict[str, Set[str]] = collections.defaultdict(set)
    for r in records:
        by[d.find(r)].add(r)
    return list(by.values())


def implied_pairs(groups: List[Set[str]]) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for g in groups:
        for a, b in itertools.combinations(sorted(g), 2):
            out.add((a, b))
    return out


def cluster_f1(pairs: Set[Tuple[str, str]], ref: Set[Tuple[str, str]]) -> Tuple[float, int, int]:
    """F1 = 2T/(N+G) with N = implied pairs, T = true among them, G = reference pairs."""
    t = len(pairs & ref)
    n, g = len(pairs), len(ref)
    return (2 * t / (n + g) if (n + g) else 0.0), t, n


def main() -> int:
    import answer_extract as AX
    import collect_controlled as CC
    import collect_all_matching as C
    C._ensure_installed()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for bench in MATCHING:
        units, sha = CC.frozen_units(bench)
        frozen = set(units)
        # gold per pair, and the reference pair set restricted to this sub-universe
        pop, _ = C.load_population(bench)
        gold: Dict[str, str] = {}
        parts: Dict[str, Tuple[str, str]] = {}
        for row in pop:
            k = row.get("unit_id") if isinstance(row, dict) else row[0]
            g = row.get("gold") if isinstance(row, dict) else (row[2] if len(row) > 2 else None)
            if k not in frozen or g is None:
                continue
            gold[k] = "match" if str(g).strip() in ("1", "true", "True", "match") else "non-match"
            seg = k.split("|")
            if len(seg) >= 3:
                parts[k] = (seg[1], seg[2])

        records = {r for k in parts for r in parts[k]}
        ref = {tuple(sorted(parts[k])) for k in parts if gold.get(k) == "match"}
        print("\n=== %s: sub-universe of the frozen sample ===" % bench)
        print("   frozen pairs %s | records touched %s | reference match pairs %s"
              % (f"{len(parts):,}", f"{len(records):,}", f"{len(ref):,}"))

        # answers per (protocol, condition, backbone, unit, repeat)
        pred: Dict[Tuple[str, str, str, int], Dict[str, Optional[str]]] = \
            collections.defaultdict(dict)
        for p in sorted(RAW.glob(f"{bench}__*.jsonl")):
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") != "ok" or r["unit"] not in parts:
                    continue
                a = (r.get("record") or {}).get("majority_answer")
                a = str(a).strip().lower().replace("_", "-") if a not in (
                    None, "", "None", "none") else None
                pred[(r["protocol"], r["arm"], r["backbone"], r["repeat"])][r["unit"]] = a

        for proto in PROTOS:
            backbones = sorted({k[2] for k in pred if k[0] == proto})
            for bb in backbones:
                for arm in ("NOPEER", "ANSONLY", "UNIFSEL"):
                    for rep in sorted({k[3] for k in pred if k[0] == proto and k[2] == bb}):
                        base = pred.get((proto, "FULL", bb, rep))
                        cur = pred.get((proto, arm, bb, rep))
                        if not base or not cur:
                            continue
                        shared = sorted(set(base) & set(cur))
                        if len(shared) < 50:
                            continue
                        # edges: a predicted match on a shared pair. A non-delivered answer is not
                        # a predicted match -- it predicts nothing, which for grouping means no edge.
                        e0 = {tuple(sorted(parts[u])) for u in shared if base[u] == "match"}
                        e1 = {tuple(sorted(parts[u])) for u in shared if cur[u] == "match"}
                        recs = {r for u in shared for r in parts[u]}
                        refs = {tuple(sorted(parts[u])) for u in shared
                                if gold.get(u) == "match"}
                        P0 = implied_pairs(groups_from_edges(recs, e0))
                        P1 = implied_pairs(groups_from_edges(recs, e1))
                        f0, t0, n0 = cluster_f1(P0, refs)
                        f1_, t1, n1 = cluster_f1(P1, refs)
                        dplus, dminus = P1 - P0, P0 - P1
                        aT, aF = len(dplus & refs), len(dplus - refs)
                        rT, rF = len(dminus & refs), len(dminus - refs)
                        # pair-level F1 on the same shared pairs, for the local-vs-downstream check
                        def pair_f1(src) -> float:
                            tp = fp = fn = 0
                            for u in shared:
                                p_ = 1 if src[u] == "match" else 0
                                g_ = 1 if gold.get(u) == "match" else 0
                                if p_ and g_:
                                    tp += 1
                                elif p_:
                                    fp += 1
                                elif g_:
                                    fn += 1
                            pr = tp / (tp + fp) if tp + fp else 0.0
                            rc = tp / (tp + fn) if tp + fn else 0.0
                            return 2 * pr * rc / (pr + rc) if pr + rc else 0.0
                        pf0, pf1 = pair_f1(base), pair_f1(cur)
                        rows.append({
                            "benchmark": bench, "protocol": proto, "backbone": bb, "arm": arm,
                            "repeat": rep, "shared_pairs": len(shared),
                            "records_in_subuniverse": len(recs),
                            "reference_pairs": len(refs),
                            "pair_f1_full": pf0, "pair_f1_arm": pf1,
                            "pair_f1_delta": pf1 - pf0,
                            "cluster_f1_full": f0, "cluster_f1_arm": f1_,
                            "cluster_f1_delta": f1_ - f0,
                            "implied_pairs_full": n0, "implied_pairs_arm": n1,
                            "added_true": aT, "added_false": aF,
                            "removed_true": rT, "removed_false": rF,
                            "edges_full": len(e0), "edges_arm": len(e1),
                            "local_and_downstream_disagree":
                                (pf1 - pf0) * (f1_ - f0) < 0})

    with open(OUT / "downstream_grouping.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    blob = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": "frozen 1,000-pair sub-universe per matching benchmark; absolute cluster F1 "
                     "is not comparable to a full-candidate-set figure, only the paired "
                     "between-arm difference on the identical sub-universe",
            "grouping_rule": "connected components over predicted match edges, held fixed",
            "rows": rows}
    (OUT / "downstream_grouping.json").write_text(json.dumps(blob, indent=1, default=str) + "\n",
                                                  encoding="utf-8")

    print("\n=== where local pair quality and downstream grouping move in OPPOSITE directions ===")
    opp = [r for r in rows if r["local_and_downstream_disagree"]]
    print("   %d of %d (protocol, backbone, arm, repeat) cells" % (len(opp), len(rows)))
    print("%-24s %-16s %-12s %-8s %4s %12s %12s" % ("benchmark", "protocol", "backbone", "arm",
                                                     "rep", "d pair F1", "d cluster F1"))
    for r in sorted(opp, key=lambda x: -abs(x["cluster_f1_delta"]))[:14]:
        print("%-24s %-16s %-12s %-8s %4d %+12.4f %+12.4f"
              % (r["benchmark"][:22], r["protocol"], r["backbone"], r["arm"], r["repeat"],
                 r["pair_f1_delta"], r["cluster_f1_delta"]))
    print("\n=== largest downstream changes overall ===")
    for r in sorted(rows, key=lambda x: -abs(x["cluster_f1_delta"]))[:10]:
        print("%-24s %-16s %-12s %-8s rep%d  pair %+0.4f  cluster %+0.4f  "
              "(+T%d +F%d -T%d -F%d)"
              % (r["benchmark"][:22], r["protocol"], r["backbone"], r["arm"], r["repeat"],
                 r["pair_f1_delta"], r["cluster_f1_delta"],
                 r["added_true"], r["added_false"], r["removed_true"], r["removed_false"]))
    print("\n-> %s" % (OUT / "downstream_grouping.json"))
    print("   sha256 %s" % hashlib.sha256((OUT / "downstream_grouping.json").read_bytes()
                                          ).hexdigest()[:32])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
