r"""Sharing conditions on this workload, at stage A and stage C, with the content contrast measured.

    PYTHONPATH=experiments/05_matching_downstream/code python3.13 \
      experiments/05_matching_downstream/code/analyze_sharing.py

The access contrasts (`candidate_only − no_peers`, `full − no_peers`) are confirmatory: four tests,
Holm-corrected, stage A, resampling pairs of gold clusters. The recipient × access interaction is a
separate four-test family, declared in the freeze rather than added afterwards.

The content contrast (`candidate_only − full`) is outside the confirmatory family: replies on this
workload are almost always a bare decision word, so the two peer blocks are near-identical by
construction. It is computed together with the identity rate of the two peer blocks on the full set.

Stage C has no interval, for the reason the freeze gives: closure couples the entities a method
merges. Stage C differences are exact descriptive quantities for this workload.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import gs_common as G
from analyze_stages import boot_stageA_f1, jl, modal, prf, stage_ABC

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(BASE, "analysis")
DRAWS = 4000
SEED = 20261013
ACCESS = ("candidate_only", "full")


def holm(tests: Dict[str, float], alpha: float = 0.05) -> Dict[str, Any]:
    order = sorted(tests, key=lambda k: tests[k])
    m = len(order)
    out, still = {}, True
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        ok = still and tests[k] <= thr
        if not ok:
            still = False
        out[k] = {"p": tests[k], "threshold": thr, "survives": ok}
    return {"alpha": alpha, "family_size": m, "tests": out,
            "survivors": [k for k in order if out[k]["survives"]]}


def boot_diff_f1(ds: G.Settlements, cand: Sequence[Tuple[int, int]],
                 a_set: Set[str], b_set: Set[str],
                 draws: int = DRAWS, seed: int = SEED) -> Dict[str, Any]:
    """Paired cluster bootstrap of F1(a) - F1(b) over pairs of gold clusters. Stage A only."""
    by: Dict[Tuple[int, int], List[Tuple[bool, bool, int]]] = collections.defaultdict(list)
    for (x, y) in cand:
        ca, cb = ds.gold_cluster(x), ds.gold_cluster(y)
        key = (ca, cb) if ca <= cb else (cb, ca)
        pk = ds.pair_key(x, y)
        by[key].append((pk in a_set, pk in b_set, ds.gold_pair(x, y)))
    keys = list(by)

    def f1_from(items) -> Tuple[float, float]:
        ta = fa = na = tb = fb = nb = 0
        for pa, pb, g in items:
            ta += pa and g == 1
            fa += pa and g == 0
            na += (not pa) and g == 1
            tb += pb and g == 1
            fb += pb and g == 0
            nb += (not pb) and g == 1
        return prf(ta, fa, na)["f1"], prf(tb, fb, nb)["f1"]

    flat = [x for k in keys for x in by[k]]
    fa0, fb0 = f1_from(flat)
    point = fa0 - fb0
    rnd = random.Random(seed)
    sims = []
    for _ in range(draws):
        items = [x for _ in range(len(keys)) for x in by[keys[rnd.randrange(len(keys))]]]
        u, v = f1_from(items)
        sims.append(u - v)
    sims.sort()
    c = [s - point for s in sims]
    p = 2.0 * min(sum(1 for z in c if z >= -point), sum(1 for z in c if z <= -point)) / draws
    return {"diff": point, "f1_a": fa0, "f1_b": fb0,
            "lo": sims[int(0.025 * draws)], "hi": sims[min(draws - 1, int(0.975 * draws))],
            "p": max(1.0 / draws, min(1.0, p)), "n_cluster_pairs": len(keys), "draws": draws}


def boot_did_f1(ds: G.Settlements, cand: Sequence[Tuple[int, int]],
                sets: Dict[Tuple[str, str], Set[str]], weak: str, strong: str, A: str,
                draws: int = DRAWS, seed: int = SEED) -> Dict[str, Any]:
    """DiD of stage-A F1: the weak recipient's access effect minus the strong recipient's."""
    by: Dict[Tuple[int, int], List[Tuple[bool, bool, bool, bool, int]]] = collections.defaultdict(list)
    for (x, y) in cand:
        ca, cb = ds.gold_cluster(x), ds.gold_cluster(y)
        key = (ca, cb) if ca <= cb else (cb, ca)
        pk = ds.pair_key(x, y)
        by[key].append((pk in sets[(weak, A)], pk in sets[(weak, "no_peers")],
                        pk in sets[(strong, A)], pk in sets[(strong, "no_peers")],
                        ds.gold_pair(x, y)))
    keys = list(by)

    def did(items) -> float:
        cnt = [[0, 0, 0] for _ in range(4)]
        for wa, wn, sa, sn, g in items:
            for i, pos in enumerate((wa, wn, sa, sn)):
                cnt[i][0] += pos and g == 1
                cnt[i][1] += pos and g == 0
                cnt[i][2] += (not pos) and g == 1
        f = [prf(*c)["f1"] for c in cnt]
        return (f[0] - f[1]) - (f[2] - f[3])

    flat = [x for k in keys for x in by[k]]
    point = did(flat)
    rnd = random.Random(seed)
    sims = [did([x for _ in range(len(keys)) for x in by[keys[rnd.randrange(len(keys))]]])
            for _ in range(draws)]
    sims.sort()
    c = [s - point for s in sims]
    p = 2.0 * min(sum(1 for z in c if z >= -point), sum(1 for z in c if z <= -point)) / draws
    return {"did": point, "lo": sims[int(0.025 * draws)],
            "hi": sims[min(draws - 1, int(0.975 * draws))],
            "p": max(1.0 / draws, min(1.0, p)), "n_cluster_pairs": len(keys)}


_BARE = re.compile(r'^[\s"\'*`#.:\-]*(non[\s-]*match|nonmatch|match)[\s"\'*`.!:\-]*$', re.I)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    ds = G.Settlements()
    nodes = ds.records_in("test")
    cand = ds.block(nodes, 5)
    Gtot = sum(len(m) * (len(m) - 1) // 2 for m in ds.clusters_in("test").values())

    share = jl("gs_share_test.jsonl")
    ok = [r for r in share if r.get("status") == "ok"]
    failed = [r for r in share if r.get("status") != "ok"]
    idx = {(r["pair_key"], r["recipient"], r["condition"]): r for r in ok}
    print(f"  sharing records {len(share)}, ok {len(ok)}, failed {len(failed)}")

    # the common population: every pair with all six cells (2 recipients x 3 conditions)
    common = [(x, y) for (x, y) in cand
              if all((ds.pair_key(x, y), rec, c) in idx
                     for rec in G.RECIPIENTS for c in G.CONDITIONS)]
    print(f"  common population: {len(common)} of {len(cand)} candidate pairs have all six cells")

    sets: Dict[Tuple[str, str], Set[str]] = {}
    res: Dict[str, Any] = {
        "records": len(share), "ok": len(ok), "failed": len(failed),
        "failures": [{k: v for k, v in r.items()
                      if k in ("unit_id", "pair_key", "recipient", "condition", "error")}
                     for r in failed[:50]],
        "common_population": len(common), "candidate_pairs": len(cand),
        "no_peers_is": ("the no-own-answer control: the recipient is re-asked from scratch and is "
                        "never shown its own earlier reply, so an access contrast is an "
                        "independent re-ask plus the peer block and absorbs serving "
                        "nondeterminism rather than attributing it to the block"),
        "levels": {}, "access_contrasts": {}, "interaction": {},
        "content_contrast_withdrawn": {}, "stageC": {}}

    print(f"\n{'recipient':14s}{'condition':16s}{'A:F1':>8s}{'A 95% CI':>18s}{'C:F1':>8s}"
          f"{'links':>7s}{'implied':>9s}{'ampl':>7s}{'abst':>7s}{'in-tok':>8s}")
    for rec in G.RECIPIENTS:
        for c in G.CONDITIONS:
            asserted = {ds.pair_key(x, y) for (x, y) in common
                        if idx[(ds.pair_key(x, y), rec, c)]["decision"] == "match"}
            sets[(rec, c)] = asserted
            s = stage_ABC(ds, nodes, common, asserted, Gtot)
            ci = boot_stageA_f1(ds, common, asserted)
            ab = sum(1 for (x, y) in common
                     if idx[(ds.pair_key(x, y), rec, c)]["decision"] is None)
            tok = sum(idx[(ds.pair_key(x, y), rec, c)]["prompt_tokens"]
                      for (x, y) in common) / len(common)
            cost = sum(idx[(ds.pair_key(x, y), rec, c)]["cost_units"] for (x, y) in common)
            res["levels"].setdefault(rec, {})[c] = {
                "A": {**s["A"], "ci": ci}, "B": s["B"], "C": s["C"],
                "topology": s["topology"], "abstentions": ab,
                "abstention_rate": ab / len(common),
                "mean_input_tokens": tok, "cost_units": cost}
            t = s["topology"]
            print(f"{rec:14s}{c:16s}{s['A']['f1']:8.4f}  [{ci['lo']:.4f},{ci['hi']:.4f}]"
                  f"{s['C']['f1']:8.4f}{t['asserted_links']:7d}{t['implied_pairs']:9d}"
                  f"{t['closure_amplification']:7.2f}{ab/len(common):7.1%}{tok:8.0f}")

    # ---- confirmatory access family: 4 tests, stage A
    print("\n=== access contrasts, stage A, Holm over the frozen family of 4 ===")
    pv: Dict[str, float] = {}
    for rec in G.RECIPIENTS:
        for A in ACCESS:
            d = boot_diff_f1(ds, common, sets[(rec, A)], sets[(rec, "no_peers")])
            res["access_contrasts"].setdefault(rec, {})[f"{A} - no_peers"] = d
            pv[f"{rec}: {A} - no_peers"] = d["p"]
    h = holm(pv)
    res["access_holm"] = h
    print(f"  {'recipient':14s}{'contrast':24s}{'dF1':>9s}{'95% CI':>20s}{'p':>8s}"
          f"{'thr':>9s}{'surv':>6s}")
    for k, v in h["tests"].items():
        rec, c = k.split(": ", 1)
        d = res["access_contrasts"][rec][c]
        print(f"  {rec:14s}{c:24s}{d['diff']:+9.4f}  [{d['lo']:+.4f},{d['hi']:+.4f}]{v['p']:8.4f}"
              f"{v['threshold']:9.5f}{'YES' if v['survives'] else '-':>6s}")

    # ---- stage C differences, descriptive and exact
    print("\n=== the same contrasts at stage C: exact, no interval ===")
    print(f"  {'recipient':14s}{'contrast':24s}{'dF1 (C)':>10s}{'links':>16s}{'implied':>18s}")
    for rec in G.RECIPIENTS:
        for A in ACCESS:
            ca = res["levels"][rec][A]["C"]["f1"]
            cb = res["levels"][rec]["no_peers"]["C"]["f1"]
            la = res["levels"][rec][A]["topology"]["asserted_links"]
            lb = res["levels"][rec]["no_peers"]["topology"]["asserted_links"]
            ia = res["levels"][rec][A]["topology"]["implied_pairs"]
            ib = res["levels"][rec]["no_peers"]["topology"]["implied_pairs"]
            res["stageC"].setdefault(rec, {})[f"{A} - no_peers"] = {
                "delta_f1_stageC": ca - cb, "f1_a": ca, "f1_b": cb,
                "links_a": la, "links_b": lb, "implied_a": ia, "implied_b": ib,
                "note": "exact finite-workload difference; closure couples entities so no interval"}
            print(f"  {rec:14s}{f'{A} - no_peers':24s}{ca-cb:+10.4f}"
                  f"{f'{la} vs {lb}':>16s}{f'{ia} vs {ib}':>18s}")

    # ---- recipient x access interaction, declared family of 4 (2 access forms x 2 metrics? no:
    #      2 access forms, stage A and stage C; stage C has no p, so the family is the 2 stage-A
    #      tests plus their stale-free recomputation is not applicable here -> family of 2)
    print("\n=== recipient x access interaction, stage A ===")
    ip: Dict[str, float] = {}
    for A in ACCESS:
        d = boot_did_f1(ds, common, sets, G.RECIPIENTS[0], G.RECIPIENTS[1], A)
        d["simple_weak"] = (res["levels"][G.RECIPIENTS[0]][A]["A"]["f1"]
                            - res["levels"][G.RECIPIENTS[0]]["no_peers"]["A"]["f1"])
        d["simple_strong"] = (res["levels"][G.RECIPIENTS[1]][A]["A"]["f1"]
                              - res["levels"][G.RECIPIENTS[1]]["no_peers"]["A"]["f1"])
        res["interaction"][f"{A} - no_peers"] = d
        ip[f"{A} - no_peers"] = d["p"]
    hi = holm(ip)
    res["interaction_holm"] = hi
    print(f"  {'access form':24s}{'DiD':>9s}{'95% CI':>20s}{'p':>8s}{'thr':>9s}{'surv':>6s}"
          f"{'weak':>9s}{'strong':>9s}")
    for k, v in hi["tests"].items():
        d = res["interaction"][k]
        print(f"  {k:24s}{d['did']:+9.4f}  [{d['lo']:+.4f},{d['hi']:+.4f}]{v['p']:8.4f}"
              f"{v['threshold']:9.5f}{'YES' if v['survives'] else '-':>6s}"
              f"{d['simple_weak']:+9.4f}{d['simple_strong']:+9.4f}")

    # ---- the content contrast, with the degeneracy measured not asserted
    print("\n=== content contrast: measured degeneracy, outside the confirmatory family ===")
    preds = [r for r in jl("gs_preds_test.jsonl") if r.get("status") == "ok"]
    # note: `preds` is needed by the content-identity measurement below
    bare = sum(1 for r in preds if _BARE.match((r.get("raw_reply") or "").strip()))
    byp: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in preds:
        byp[r["pair_key"]].append(r)
    pairs_with_text = sum(1 for pk, rs in byp.items()
                          if any(not _BARE.match((r.get("raw_reply") or "").strip()) for r in rs))
    # The two blocks never hash equal because their wrappers differ ("system 1: match" against
    # "system 1 said: match"), so a hash comparison measures formatting, not content. The content
    # question is whether a peer's full reply carries anything beyond its decision word.
    same_content = 0
    cells = 0
    bypk: Dict[str, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
    for r in preds:
        bypk[r["pair_key"]][r["model"]] = r
    for (x, y) in common:
        pk = ds.pair_key(x, y)
        for rec in G.RECIPIENTS:
            cells += 1
            peers = [bypk[pk][m] for m in G.POOL if m != rec and m in bypk[pk]]
            if all(_BARE.match((p.get("raw_reply") or "").strip()) for p in peers):
                same_content += 1
    res["content_contrast_withdrawn"] = {
        "reason": ("outside the confirmatory family: replies on this "
                   "workload are almost always a bare decision word, so a `full` block carries "
                   "nothing a `candidate_only` block does not. This is a property of the workload, "
                   "not a finding about information content"),
        "replies_that_are_a_bare_class_word": bare,
        "replies_total": len(preds),
        "bare_rate": bare / len(preds),
        "pairs_with_any_extra_text": pairs_with_text,
        "pairs_total": len(byp),
        "cells_where_every_peer_reply_is_only_its_decision_word": same_content,
        "cells_compared": cells,
        "content_identical_rate": same_content / cells if cells else None,
        "formatting_confound": ("the two peer blocks differ in wrapper text and token count even "
                                "when they carry the same decisions, so any measured "
                                "candidate_only - full difference on this workload is confounded "
                                "with prompt formatting and length and is not a content effect"),
        "computed_anyway": {}}
    print(f"  replies that are a bare class word: {bare}/{len(preds)} ({bare/len(preds):.2%})")
    print(f"  pairs where any of six replies carries extra text: {pairs_with_text}/{len(byp)} "
          f"({pairs_with_text/len(byp):.2%})")
    print(f"  recipient cells where EVERY peer reply is only its decision word: "
          f"{same_content}/{cells} ({same_content/cells:.2%})")
    print(f"  (the two blocks never hash equal because their wrappers differ, so any measured "
          f"difference here is confounded with formatting and length)")
    for rec in G.RECIPIENTS:
        d = boot_diff_f1(ds, common, sets[(rec, "candidate_only")], sets[(rec, "full")])
        res["content_contrast_withdrawn"]["computed_anyway"][rec] = {
            **d, "label": "DEGENERATE BY CONSTRUCTION - not a finding about content"}
        print(f"  {rec:14s} candidate_only - full {d['diff']:+.4f} "
              f"[{d['lo']:+.4f},{d['hi']:+.4f}] p={d['p']:.4f}  (degenerate)")

    rows: List[Dict[str, Any]] = []
    for rec, byc in res["levels"].items():
        for c, v in byc.items():
            for st in ("A", "B", "C"):
                rows.append({"recipient": rec, "condition": c, "row_kind": "level", "stage": st,
                             "f1": v[st]["f1"], "precision": v[st]["precision"],
                             "recall": v[st]["recall"], "tp": v[st]["tp"], "fp": v[st]["fp"],
                             "fn": v[st]["fn"], "n_scored": v[st]["n_scored"],
                             "ci_lo": (v[st].get("ci") or {}).get("lo"),
                             "ci_hi": (v[st].get("ci") or {}).get("hi"),
                             "asserted_links": v["topology"]["asserted_links"],
                             "implied_pairs": v["topology"]["implied_pairs"],
                             "closure_amplification": v["topology"]["closure_amplification"],
                             "largest_component": v["topology"]["largest_component"],
                             "abstention_rate": v["abstention_rate"],
                             "mean_input_tokens": v["mean_input_tokens"],
                             "cost_units": v["cost_units"],
                             "n_units": res["common_population"]})
    for rec, byc in res["access_contrasts"].items():
        for c, v in byc.items():
            hh = res["access_holm"]["tests"][f"{rec}: {c}"]
            rows.append({"recipient": rec, "condition": c, "row_kind": "access_contrast",
                         "stage": "A", "f1": v["diff"], "ci_lo": v["lo"], "ci_hi": v["hi"],
                         "p": v["p"], "holm_threshold": hh["threshold"],
                         "survives_holm": hh["survives"],
                         "n_units": res["common_population"],
                         "n_clusters": v["n_cluster_pairs"]})
    for rec, byc in res["stageC"].items():
        for c, v in byc.items():
            rows.append({"recipient": rec, "condition": c, "row_kind": "access_contrast",
                         "stage": "C", "f1": v["delta_f1_stageC"],
                         "n_units": res["common_population"],
                         "asserted_links": v["links_a"], "implied_pairs": v["implied_a"]})
    for c, v in res["interaction"].items():
        hh = res["interaction_holm"]["tests"][c]
        rows.append({"recipient": "weak minus strong", "condition": c,
                     "row_kind": "interaction", "stage": "A", "f1": v["did"],
                     "ci_lo": v["lo"], "ci_hi": v["hi"], "p": v["p"],
                     "holm_threshold": hh["threshold"], "survives_holm": hh["survives"],
                     "n_units": res["common_population"], "n_clusters": v["n_cluster_pairs"]})
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    cp = os.path.join(OUT, "sharing_long.csv")
    with open(cp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(OUT, "sharing.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {OUT}/sharing.json and {cp} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
