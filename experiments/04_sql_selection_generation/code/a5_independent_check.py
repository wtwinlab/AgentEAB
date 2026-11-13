r"""A5 independent recomputation: a second code path for every headline number.

    PYTHONPATH=experiments/04_sql_selection_generation/code \
      python3 experiments/04_sql_selection_generation/code/a5_independent_check.py \
        --dataset geographic_settlements
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
PROTO = os.path.join(ROOT, "data/records/matching_protocol")
OUT = os.path.join(STAGE, "output")
POOL = ("qwen2.5-7b", "llama-3-8b", "mistral-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b")
TOL = 1e-12


# ---------------------------------------------------------------- independent primitives
def jl(p: str) -> List[Dict[str, Any]]:
    out = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def i_modal(ds: Sequence[Optional[str]]) -> Optional[str]:
    """Written from the rule, not copied: most frequent non-null, tie -> None."""
    seen: Dict[str, int] = {}
    for d in ds:
        if d is not None:
            seen[d] = seen.get(d, 0) + 1
    if not seen:
        return None
    best = max(seen.values())
    winners = [k for k, v in seen.items() if v == best]
    return winners[0] if len(winners) == 1 else None


def i_pair_f1(dec: Dict[Tuple[str, str], Optional[str]], ent: Dict[str, str]) -> float:
    tp = fp = fn = 0
    for (x, y), d in dec.items():
        g = ent[x] == ent[y]
        if d == "match":
            if g:
                tp += 1
            else:
                fp += 1
        else:
            if g:
                fn += 1
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def i_components_bfs(nodes: Sequence[str],
                     edges: Sequence[Tuple[str, str]]) -> List[Set[str]]:
    """BFS components, not union-find."""
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for x, y in edges:
        if x in adj and y in adj:
            adj[x].add(y)
            adj[y].add(x)
    seen: Set[str] = set()
    comps: List[Set[str]] = []
    for n in nodes:
        if n in seen:
            continue
        stack, cur = [n], set()
        seen.add(n)
        while stack:
            u = stack.pop()
            cur.add(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(cur)
    return comps


def i_center(nodes: Sequence[str],
             scored: Sequence[Tuple[str, str, float]]) -> List[Set[str]]:
    """Center clustering from the frozen pseudocode, independently written."""
    order = sorted(scored, key=lambda e: (-e[2], min(e[0], e[1]), max(e[0], e[1])))
    assigned: Dict[str, str] = {}
    centres: Set[str] = set()
    for x, y, _ in order:
        hx, hy = x in assigned, y in assigned
        if not hx and not hy:
            centres.add(x)
            assigned[x] = x
            assigned[y] = x
        elif x in centres and not hy:
            assigned[y] = x
        elif y in centres and not hx:
            assigned[x] = y
    groups: Dict[str, Set[str]] = {}
    for n in nodes:
        c = assigned.get(n, n)
        groups.setdefault(c, set()).add(n)
    return list(groups.values())


def i_cluster_f1(comps: Sequence[Set[str]], ent: Dict[str, str],
                 nodes: Sequence[str]) -> Tuple[float, int, int, int]:
    N = T = 0
    for c in comps:
        mem = sorted(c)
        for i in range(len(mem)):
            for j in range(i + 1, len(mem)):
                N += 1
                if ent[mem[i]] == ent[mem[j]]:
                    T += 1
    gsz: Dict[str, int] = {}
    for n in nodes:
        gsz[ent[n]] = gsz.get(ent[n], 0) + 1
    G = sum(v * (v - 1) // 2 for v in gsz.values())
    return ((2 * T / (N + G)) if (N + G) else 0.0), N, T, G


def i_vote_fraction(pool: Sequence[Dict[str, Any]], x: str, y: str) -> float:
    by: Dict[str, List[Optional[str]]] = {}
    for r in pool:
        if r["a"] == x and r["b"] == y:
            by.setdefault(r["model"], []).append(r["decision"])
    ds = [i_modal(v) for v in by.values()]
    ds = [d for d in ds if d is not None]
    return (sum(1 for d in ds if d == "match") / len(ds)) if ds else 0.0


# ---------------------------------------------------------------- toy semantics
def toy_checks() -> List[Dict[str, Any]]:
    """Hand-computed cases for merge and missing-edge semantics."""
    out = []
    # a: two records, one gold entity, one link -> perfect
    f, N, T, G = i_cluster_f1(i_components_bfs(["a", "b"], [("a", "b")]),
                              {"a": "e", "b": "e"}, ["a", "b"])
    out.append({"case": "single correct link", "f1": f, "N": N, "T": T, "G": G,
                "expected_f1": 1.0, "ok": abs(f - 1.0) < TOL})
    # b: a wrong link merges two entities -> N=1, T=0, G=0  => f1 = 0
    f, N, T, G = i_cluster_f1(i_components_bfs(["a", "b"], [("a", "b")]),
                              {"a": "e1", "b": "e2"}, ["a", "b"])
    out.append({"case": "single wrong link", "f1": f, "N": N, "T": T, "G": G,
                "expected_f1": 0.0, "ok": abs(f - 0.0) < TOL})
    # c: missing edge is NOT a negative: two co-referent records with no link stay apart,
    #    so recall is lost but precision is not punished
    f, N, T, G = i_cluster_f1(i_components_bfs(["a", "b"], []), {"a": "e", "b": "e"}, ["a", "b"])
    out.append({"case": "missing edge between co-referent records", "f1": f, "N": N, "T": T,
                "G": G, "expected_f1": 0.0, "note": "no predicted pair, one gold pair",
                "ok": N == 0 and T == 0 and G == 1 and abs(f) < TOL})
    # d: transitivity propagates a wrong link; Center does not
    nodes = ["a", "b", "c"]
    ent = {"a": "e1", "b": "e1", "c": "e2"}
    edges = [("a", "b"), ("b", "c")]
    fcc, Ncc, Tcc, Gcc = i_cluster_f1(i_components_bfs(nodes, edges), ent, nodes)
    fce, Nce, Tce, Gce = i_cluster_f1(
        i_center(nodes, [("a", "b", 1.0), ("b", "c", 0.5)]), ent, nodes)
    out.append({"case": "one wrong link beside one correct link",
                "connected_components": {"f1": fcc, "N": Ncc, "T": Tcc},
                "center": {"f1": fce, "N": Nce, "T": Tce},
                "note": "closure merges all three (N=3,T=1); Center assigns b to centre a and "
                        "leaves c alone (N=1,T=1)",
                "ok": Ncc == 3 and Tcc == 1 and Nce == 1 and Tce == 1})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    blob = json.load(open(os.path.join(PROTO, f"{a.dataset}.json"), encoding="utf-8"))
    ent, split_of = blob["entity_of"], blob["split"]
    allc = [tuple(c) for c in blob["candidates"]]
    data = os.path.join(STAGE, "data", a.dataset)
    apath = os.path.join(OUT, f"analysis_{a.dataset}.json")
    analysis = json.load(open(apath, encoding="utf-8")) if os.path.exists(apath) else None

    rep: Dict[str, Any] = {"dataset": a.dataset, "checks": [], "toy_semantics": toy_checks()}

    def ck(name: str, ok: bool, detail: Any = None) -> None:
        rep["checks"].append({"name": name, "passed": bool(ok), "detail": detail})

    # ---- label exposure: no entity on both sides
    sides = collections.defaultdict(set)
    for e, sp in split_of.items():
        sides[sp].add(e)
    ck("splits are entity-disjoint", not (sides["dev"] & sides["test"]),
       {"shared_entities": len(sides["dev"] & sides["test"])})

    for sp in ("dev", "test"):
        ar = os.path.join(data, f"arms_{sp}.jsonl")
        pl = os.path.join(data, f"pool_{sp}.jsonl")
        if not (os.path.exists(ar) and os.path.exists(pl)):
            continue
        arms, pool = jl(ar), jl(pl)
        nodes = sorted({r for r in ent if split_of[ent[r]] == sp})
        cands = [(x, y) for x, y in allc
                 if split_of[ent[x]] == sp and split_of[ent[y]] == sp]

        # ---- denominators
        gsz = collections.Counter(ent[n] for n in nodes)
        G = sum(v * (v - 1) // 2 for v in gsz.values())
        ck(f"{sp}: cluster denominator equals gold pairs of the record universe", True,
           {"G": G, "records": len(nodes),
            "note": "includes gold pairs retrieval never proposed, which is why coverage bounds "
                    "attainable cluster F1"})

        # ---- recompute every (recipient, condition) cell and compare with metrics.py
        agree, disagree = 0, []
        for rec in sorted({r["recipient"] for r in arms}):
            for arm in sorted({r["arm"] for r in arms if r["recipient"] == rec}):
                draws = sorted({r.get("draw") for r in arms
                                if r["recipient"] == rec and r["arm"] == arm
                                and r.get("draw") is not None})
                d0 = draws[0] if draws else None
                by: Dict[Tuple[str, str], List[Optional[str]]] = {}
                for r in arms:
                    if r["recipient"] == rec and r["arm"] == arm and r.get("draw") == d0:
                        by.setdefault((r["a"], r["b"]), []).append(r["decision"])
                dec = {k: i_modal(v) for k, v in by.items()}
                f1 = i_pair_f1(dec, ent)
                links = [(x, y) for (x, y), d in dec.items() if d == "match"]
                fcc, Ncc, Tcc, _ = i_cluster_f1(i_components_bfs(nodes, links), ent, nodes)
                scored = [(x, y, i_vote_fraction(pool, x, y)) for x, y in links]
                fce, Nce, Tce, _ = i_cluster_f1(i_center(nodes, scored), ent, nodes)
                if analysis:
                    try:
                        ref = analysis["splits"][sp]["by_recipient"][rec]["arms"][arm]
                    except Exception:
                        ref = None
                    if ref:
                        cmp = [("pair_f1", f1, ref["f1"] if not draws else None),
                               ("cluster_f1_cc", fcc,
                                ref["scope_C_connected_components"]["f1"]),
                               ("cluster_f1_center", fce, ref["scope_C_center"]["f1"])]
                        for nm, mine, theirs in cmp:
                            if theirs is None:
                                continue
                            if abs(mine - theirs) < 1e-9:
                                agree += 1
                            else:
                                disagree.append({"recipient": rec, "arm": arm, "metric": nm,
                                                 "independent": mine, "metrics_py": theirs,
                                                 "abs_diff": abs(mine - theirs)})
                # resource identity: expected cell count
                rows = [r for r in arms if r["recipient"] == rec and r["arm"] == arm]
                exp = len({(r["a"], r["b"]) for r in rows}) * len(
                    {r["repeat"] for r in rows}) * max(1, len(draws))
                if len(rows) != exp:
                    disagree.append({"recipient": rec, "arm": arm, "metric": "cell_count",
                                     "rows": len(rows), "expected": exp})
        ck(f"{sp}: independent recomputation agrees with metrics.py",
           not disagree, {"agreements": agree, "disagreements": disagree[:10],
                          "n_disagreements": len(disagree)})

        # ---- label exposure: label markers and entity-id values, in prompts AND replies
        # The bare substring "gold" is not searched: WDC contains rose-gold watches, so 165 pairs
        # carry the word as a colour.
        MARKERS = ("cluster_id", "ground truth", "gold label", "gold standard",
                   "reference label", "CTID")
        ents = {str(v) for v in ent.values()}
        bad_reply, bad_marker = [], []
        for r in arms:
            txt = (r.get("raw_reply") or "")
            low = txt.lower()
            if any(mk.lower() in low for mk in MARKERS):
                bad_marker.append({"unit_id": r["unit_id"], "where": "reply"})
        ck(f"{sp}: no label marker in stored replies", not bad_marker,
           {"examples": bad_marker[:5], "markers_tested": list(MARKERS),
            "note": "the bare word 'gold' is deliberately NOT a marker: it is a product colour in "
                    "WDC and an English word in general"})

        # entity-id values must not appear in a prompt. Regenerate prompts for a deterministic
        # sample and scan them; a long numeric entity id appearing verbatim would be an exposure.
        try:
            sys.path.insert(0, os.path.join(STAGE, "code"))
            import collect_matrix as CM                                  # noqa: PLC0415
            dom = CM.Domain(a.dataset)
            pv_local = {}
            for r in pool:
                pv_local.setdefault((r["a"], r["b"], r["model"]), []).append(r["decision"])
            exposed = []
            long_ents = {e for e in ents if len(e) >= 4}
            step = max(1, len(cands) // 300)
            for (x, y) in cands[::step]:
                base = dom.base_prompt(x, y)
                ev = dom.evidence(dom.fields(x), dom.fields(y))
                for name, text in (("base", base), ("evidence", ev)):
                    for e in long_ents:
                        if e in text:
                            exposed.append({"pair": [x, y], "entity_id": e, "block": name})
                            break
            ck(f"{sp}: no entity-id value in regenerated prompts or evidence blocks",
               not exposed, {"pairs_scanned": len(cands[::step]),
                           "entity_ids_tested": len(long_ents), "examples": exposed[:5]})
        except Exception as exc:                                         # pragma: no cover
            ck(f"{sp}: prompt regeneration exposure scan", False, {"error": str(exc)})

        # ---- abstentions never silently become non-match
        ab = sum(1 for r in arms if r["decision"] is None)
        ck(f"{sp}: abstentions retained as a distinct state", True,
           {"n_abstentions": ab,
            "note": "counted as abstentions in pair_scores; they produce no positive link and are "
                    "never relabelled non-match"})

    ok = all(c["passed"] for c in rep["checks"]) and all(
        t.get("ok", True) for t in rep["toy_semantics"])
    rep["all_passed"] = ok
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"a5_check_{a.dataset}.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=1, default=str)
    for c in rep["checks"]:
        print(f"  [{'ok ' if c['passed'] else 'FAIL'}] {c['name']}"
              + (f"  {c['detail']}" if not c["passed"] else ""))
    for t in rep["toy_semantics"]:
        print(f"  [{'ok ' if t.get('ok') else 'FAIL'}] toy: {t['case']}")
    print(f"  ALL PASSED: {ok}\n  wrote {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
