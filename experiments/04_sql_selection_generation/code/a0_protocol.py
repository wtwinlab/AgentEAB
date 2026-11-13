r"""A0 protocol construction: record universe, entity-disjoint split and candidate set, per dataset.

    PYTHONPATH=experiments/06_cost_resources/code:experiments/05_matching_downstream/code \
      python3 experiments/04_sql_selection_generation/code/a0_protocol.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import re
import statistics as st
import sys
import unicodedata
from typing import Any, Dict, List, Sequence, Set, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
OUT = os.path.join(STAGE, "output")
PROTO = os.path.join(ROOT, "data/records/matching_protocol")

SPLIT_SALT = {"wdc_products": "3dq-wdc-entity-split-2026-10-15",
              "musicbrainz20k": "3dq-mb20k-entity-split-2026-10-15"}
DEV_FRACTION = {"wdc_products": 0.50, "musicbrainz20k": 0.20}
BLOCK_NGRAM = 3
#: k selected on DEVELOPMENT ONLY by the rule Geographic Settlements' own freeze used -- the
#: smallest dev-tested k whose dev blocking recall reaches 0.90 (ISSUE-008).
BLOCK_K = {"geographic_settlements": 5, "musicbrainz20k": 3}
RECALL_TARGET = 0.90


# --------------------------------------------------------------------- shared blocker
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def ngrams(s: str, n: int = BLOCK_NGRAM) -> Set[str]:
    t = norm(s).replace(" ", "")
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))} or {t}


def topk_jaccard(keys: Sequence[str], texts: Dict[str, str], k: int) -> List[Tuple[str, str]]:
    """Top-k character-3-gram Jaccard neighbours per record, symmetrised.
    """
    grams = {r: ngrams(texts[r]) for r in keys}
    inv: Dict[str, List[str]] = collections.defaultdict(list)
    for r in keys:
        for g in grams[r]:
            inv[g].append(r)
    out: Set[Tuple[str, str]] = set()
    for r in keys:
        cand = collections.Counter()
        for g in grams[r]:
            for o in inv[g]:
                if o != r:
                    cand[o] += 1
        if not cand:
            continue
        scored = []
        gr = grams[r]
        for o, shared in cand.most_common(200):
            go = grams[o]
            u = len(gr | go)
            scored.append((shared / u if u else 0.0, o))
        scored.sort(key=lambda t: (-t[0], t[1]))
        for _, o in scored[:k]:
            out.add((r, o) if r < o else (o, r))
    return sorted(out)


def split_of(dataset: str, entity: str) -> str:
    h = int(hashlib.sha256(f"{SPLIT_SALT[dataset]}|{entity}".encode()).hexdigest(), 16)
    return "dev" if (h % 10000) < DEV_FRACTION[dataset] * 10000 else "test"


def stats(name: str, universe: List[str], ent_of: Dict[str, str], cands: List[Tuple[str, str]],
          split_assign: Dict[str, str]) -> Dict[str, Any]:
    by_split: Dict[str, Dict[str, Any]] = {}
    for sp in ("dev", "test"):
        recs = [r for r in universe if split_assign[ent_of[r]] == sp]
        ents = {ent_of[r] for r in recs}
        sizes = collections.Counter(ent_of[r] for r in recs)
        gold_pairs = sum(n * (n - 1) // 2 for n in sizes.values())
        cs = [(a, b) for a, b in cands
              if split_assign[ent_of[a]] == sp and split_assign[ent_of[b]] == sp]
        pos = sum(1 for a, b in cs if ent_of[a] == ent_of[b])
        by_split[sp] = {
            "records": len(recs), "entities": len(ents),
            "all_record_pair_universe": len(recs) * (len(recs) - 1) // 2,
            "gold_positive_pairs": gold_pairs,
            "candidates": len(cs), "candidate_positives": pos,
            "blocking_recall": pos / gold_pairs if gold_pairs else None,
            "candidate_positive_rate": pos / len(cs) if cs else None,
            "entity_size_distribution": dict(sorted(collections.Counter(sizes.values()).items()))}
    cross = sum(1 for a, b in cands if split_assign[ent_of[a]] != split_assign[ent_of[b]])
    return {"universe_records": len(universe), "entities": len({ent_of[r] for r in universe}),
            "candidates_total": len(cands), "candidates_crossing_the_split_excluded": cross,
            "by_split": by_split}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PROTO, exist_ok=True)
    rep: Dict[str, Any] = {
        "stage": "A0 protocol",
        "retrieval_rule": {
            "family": "character 3-gram Jaccard, top-k per record, symmetrised",
            "k_per_dataset_selected_on_dev": BLOCK_K,
            "k_selection_rule": "smallest dev-tested k reaching 0.90 dev blocking recall",
            "normalisation": "NFKD, combining marks stripped, lowercased, non-alphanumeric to "
                             "space, whitespace collapsed",
            "gold_visible_to_blocker": False,
            "note": "the low/high-k extension draws from the same pre-generated pool; positives "
                    "missed by retrieval are never injected from gold"},
        "split_rule": {"form": "sha256(salt|entity_id) % 10000 < fraction*10000",
                       "salts": SPLIT_SALT, "dev_fraction": DEV_FRACTION,
                       "entity_disjoint": True,
                       "gold_use": "split construction and evaluation only; never in a prompt or "
                                   "the blocker"},
        "datasets": {}}

    # ------------------------------------------------------------------ WDC Products
    sys.path.insert(0, os.path.join(ROOT, "experiments/06_cost_resources/code"))
    import em_common as E                                                # noqa: PLC0415
    m = E.Matching("wdc_products")
    texts: Dict[str, str] = {}
    ent_of: Dict[str, str] = {}
    for sp, rows in m.splits.items():
        for r in rows:
            L, R = m.pair_record(r)
            for side, rec in (("left", L), ("right", R)):
                rid = f"{side[0].upper()}:{r.get(f'id_{side}')}"
                cid = r.get(f"cluster_id_{side}")
                if cid is None:
                    continue
                ent_of[rid] = str(cid)
                if rid not in texts:
                    texts[rid] = " ".join(str(v) for v in rec.values() if v)
    universe = sorted(texts)
    assign = {e: split_of("wdc_products", e) for e in set(ent_of.values())}
    # WDC's candidate set is the benchmark's own labelled pair set, not retrieval-generated: no
    # top-k lexical rule reaches usable recall on its 11.6-records-per-entity universe
    # (ISSUE-007).
    cands = []
    for sp, rows in m.splits.items():
        for r in rows:
            a = f"L:{r.get('id_left')}"
            b = f"R:{r.get('id_right')}"
            if a in ent_of and b in ent_of and assign[ent_of[a]] == assign[ent_of[b]]:
                cands.append((a, b) if a < b else (b, a))
    cands = sorted(set(cands))
    s = stats("wdc_products", universe, ent_of, cands, assign)
    s["candidate_construction"] = (
        "the benchmark's own labelled candidate pairs, restricted to pairs whose both endpoint "
        "records fall in the same entity-disjoint half. Not retrieval-generated: see ISSUE-007. "
        "Consequence: WDC does not take part in the candidate-coverage (low/high-k) experiment, "
        "which Geographic Settlements and MusicBrainz carry instead.")
    s["blocking_recall_is_not_defined_here"] = (
        "the ratio reported under blocking_recall is the share of in-half gold pairs the "
        "benchmark's own pair sample happens to contain, not the recall of a retrieval rule")
    s["universe_provenance"] = (
        "the records this project has actually observed: every left/right record appearing in the "
        "shipped WDC train/valid/test candidate pairs. This is a subset of the real product "
        "catalogue and is reported as the observed record universe, never as the complete corpus.")
    s["why_not_the_shipped_split"] = (
        "all 500 entities appear in all three shipped splits, so they cannot support an "
        "entity-level selection experiment (ISSUE-001). The earlier salted partition dropped 43.9% "
        "of pairs and is a different protocol; this rebuild is used instead and the earlier numbers "
        "stay valid for what they measured.")
    rep["datasets"]["wdc_products"] = s
    json.dump({"universe": universe, "entity_of": ent_of, "split": assign,
               "candidates": [list(c) for c in cands]},
              open(os.path.join(PROTO, "wdc_products.json"), "w", encoding="utf-8"))

    # ------------------------------------------------------------------ Geographic Settlements
    sys.path.insert(0, os.path.join(ROOT, "experiments/05_matching_downstream/code"))
    import gs_common as G                                                # noqa: PLC0415
    gs = G.Settlements()
    gs_ent: Dict[str, str] = {}
    gs_assign: Dict[str, str] = {}
    for sp in ("dev", "test"):
        for rid in gs.records_in(sp):
            c = str(gs.gold_cluster(rid))
            gs_ent[str(rid)] = c
            gs_assign[c] = sp
    gs_univ = sorted(gs_ent, key=lambda x: int(x))
    gs_cands: List[Tuple[str, str]] = []
    for sp in ("dev", "test"):
        for a, b in gs.block(gs.records_in(sp), BLOCK_K["geographic_settlements"]):
            gs_cands.append((str(a), str(b)) if int(a) < int(b) else (str(b), str(a)))
    gsd = stats("gs", gs_univ, gs_ent, gs_cands, gs_assign)
    gsd["universe_provenance"] = ("3,054 records over 819 gold clusters from four real catalogues; "
                                  "the split and blocker are reused unchanged from "
                                  "downstream_replication_2026-10-13/FREEZE.md")
    gsd["retrieval_rule_used"] = ("the study's own frozen character-3-gram Jaccard at "
                                  "k=5, reused unchanged per DECISIONS D3")
    gsd["selected_k"] = BLOCK_K["geographic_settlements"]
    gsd["prior_exposure"] = ("test outcomes have been examined; the dev split has never been "
                             "collected or examined")
    rep["datasets"]["geographic_settlements"] = gsd
    json.dump({"universe": gs_univ, "entity_of": gs_ent, "split": gs_assign,
               "candidates": [list(c) for c in gs_cands]},
              open(os.path.join(PROTO, "geographic_settlements.json"), "w", encoding="utf-8"))

    # ------------------------------------------------------------------ MusicBrainz 20K
    mb_csv = os.path.join(STAGE, "data/musicbrainz/musicbrainz-20-A01.csv.dapo")
    with open(mb_csv, encoding="utf-8", newline="") as f:
        mrows = list(csv.DictReader(f))
    SEM = ("title", "artist", "album", "year", "length", "number", "language")
    mb_texts = {r["TID"]: " ".join(str(r.get(c) or "") for c in ("title", "artist", "album"))
                for r in mrows}
    mb_ent = {r["TID"]: r["CID"] for r in mrows}
    mb_univ = sorted(mb_texts, key=lambda x: int(x))
    mb_assign = {e: split_of("musicbrainz20k", e) for e in set(mb_ent.values())}
    # retrieval runs WITHIN each split: a deployed matcher never retrieves across an evaluation
    # boundary, and blocking over the union would put dev records into test candidate lists
    mb_cands = []
    for sp in ("dev", "test"):
        sub = [r for r in mb_univ if mb_assign[mb_ent[r]] == sp]
        mb_cands.extend(topk_jaccard(sub, {r: mb_texts[r] for r in sub},
                                     BLOCK_K["musicbrainz20k"]))
    mb_cands = sorted(set(mb_cands))
    mbd = stats("mb", mb_univ, mb_ent, mb_cands, mb_assign)
    mbd["universe_provenance"] = ("all 19,375 records of musicbrainz-20-A01, the full published "
                                  "release; no sampling")
    mbd["split_label"] = ("a new entity-disjoint 20/80 protocol defined by this study on a published "
                          "dataset, not the dataset's own split")
    mbd["blocking_text_fields"] = ["title", "artist", "album"]
    mbd["selected_k"] = BLOCK_K["musicbrainz20k"]
    mbd["k_selection_rule"] = ("smallest dev-tested k reaching %.2f dev blocking "
                              "recall; see ISSUE-008" % RECALL_TARGET)
    mbd["model_visible_fields"] = list(SEM)
    mbd["excluded_from_model_input"] = ["TID", "CID", "CTID", "SourceID", "id"]
    mbd["prior_exposure"] = "none; test is unseen and carries the confirmatory load"
    rep["datasets"]["musicbrainz20k"] = mbd
    json.dump({"universe": mb_univ, "entity_of": mb_ent, "split": mb_assign,
               "candidates": [list(c) for c in mb_cands]},
              open(os.path.join(PROTO, "musicbrainz20k.json"), "w", encoding="utf-8"))

    # ------------------------------------------------------------------ report
    for name, d in rep["datasets"].items():
        print(f"\n=== {name}: universe {d['universe_records']} records, {d['entities']} entities, "
              f"{d['candidates_total']} candidates "
              f"({d['candidates_crossing_the_split_excluded']} cross-split excluded)")
        for sp, v in d["by_split"].items():
            print(f"   {sp:4s} {v['records']:6d} records  {v['entities']:5d} entities  "
                  f"{v['candidates']:6d} candidates  {v['candidate_positives']:5d} positive  "
                  f"recall {v['blocking_recall'] if v['blocking_recall'] is None else round(v['blocking_recall'],4)}  "
                  f"pos-rate {v['candidate_positive_rate'] if v['candidate_positive_rate'] is None else round(v['candidate_positive_rate'],4)}")

    p = os.path.join(OUT, "a0_protocol.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n  wrote {p} and data/protocol/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
