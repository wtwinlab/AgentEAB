r"""A0 vetting for MusicBrainz 20K: eligibility, label integrity and gold-label exposure check.

    python3 experiments/04_sql_selection_generation/code/a0_vet_musicbrainz.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import re
import statistics as st
from typing import Any, Dict, List

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
DATA = os.path.join(STAGE, "data/musicbrainz")
OUT = os.path.join(STAGE, "output")

CSV_FILE = os.path.join(DATA, "musicbrainz-20-A01.csv.dapo")
PUBLISHED = {"records": 19375, "clusters": 10000, "matching_pairs": 16250, "sources": 5}

#: columns that encode the gold entity relation and must never reach a model prompt
GOLD_COLUMNS = ("CID", "CTID")
#: bookkeeping columns: kept in logs, not shown to a model
ID_COLUMNS = ("TID", "SourceID", "id")
#: the semantic fields a model may see
SEMANTIC_COLUMNS = ("number", "title", "length", "artist", "album", "year", "language")


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    with open(CSV_FILE, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())

    rep: Dict[str, Any] = {
        "dataset": "MusicBrainz 20K (musicbrainz-20-A01)",
        "provenance": {
            "page": "https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution",
            "readme": "https://dbs.uni-leipzig.de/files/datasets/saeedi/musicBrainz_readme.txt",
            "data": "https://dbs.uni-leipzig.de/files/datasets/saeedi/musicbrainz-20-A01.csv.dapo",
            "paper": "Saeedi, Peukert, Rahm, Scalable Matching and Clustering of Entities with "
                     "FAMER, CSIMQ 16 (2018) 61-83, DOI 10.7250/csimq.2018-16.04",
            "construction": "real MusicBrainz song records; duplicates and attribute perturbations "
                            "produced by the DAPO generator (publisher's own .dapo marker). A "
                            "generator-perturbed benchmark over real records, not a natural error "
                            "collection.",
            "file_sha256": sha(CSV_FILE),
            "file_bytes": os.path.getsize(CSV_FILE),
            "readme_sha256": sha(os.path.join(DATA, "musicBrainz_readme.txt"))},
        "columns": cols,
        "column_roles": {"gold_only": list(GOLD_COLUMNS), "logs_only": list(ID_COLUMNS),
                         "model_visible": list(SEMANTIC_COLUMNS)}}

    # ---------------------------------------------------------------- 1. ids and labels
    tids = [r["TID"] for r in rows]
    cids = [r["CID"] for r in rows]
    clusters = collections.defaultdict(list)
    for r in rows:
        clusters[r["CID"]].append(r)
    sizes = [len(v) for v in clusters.values()]
    pos_pairs = sum(n * (n - 1) // 2 for n in sizes)

    rep["integrity"] = {
        "n_records": len(rows),
        "tid_unique": len(set(tids)) == len(tids),
        "n_distinct_tid": len(set(tids)),
        "every_record_has_cid": all(r.get("CID") not in (None, "") for r in rows),
        "n_clusters": len(clusters),
        "cluster_size_min": min(sizes), "cluster_size_max": max(sizes),
        "cluster_size_mean": st.fmean(sizes),
        "cluster_size_distribution": dict(sorted(collections.Counter(sizes).items())),
        "n_singleton_clusters": sum(1 for n in sizes if n == 1),
        "gold_positive_pairs": pos_pairs,
        "ctid_within_cluster_unique": all(
            len({r["CTID"] for r in v}) == len(v) for v in clusters.values()),
        "ctid_starts_at_1_and_is_contiguous": all(
            sorted(int(r["CTID"]) for r in v) == list(range(1, len(v) + 1))
            for v in clusters.values())}

    # ---------------------------------------------------------------- 2. published stats
    srcs = collections.Counter(r["SourceID"] for r in rows)
    rep["published_comparison"] = {
        "published": PUBLISHED,
        "measured": {"records": len(rows), "clusters": len(clusters),
                     "matching_pairs": pos_pairs, "sources": len(srcs)},
        "matches": {k: (PUBLISHED[k] == v) for k, v in
                    {"records": len(rows), "clusters": len(clusters),
                     "matching_pairs": pos_pairs, "sources": len(srcs)}.items()}}
    rep["sources"] = {"n_sources": len(srcs), "records_per_source": dict(sorted(srcs.items()))}

    # ---------------------------------------------------------------- 3. duplicate-free sources?
    viol = []
    for cid, v in clusters.items():
        c = collections.Counter(r["SourceID"] for r in v)
        bad = {s: n for s, n in c.items() if n > 1}
        if bad:
            viol.append({"CID": cid, "duplicate_sources": bad, "cluster_size": len(v)})
    rep["duplicate_free_sources"] = {
        "readme_claim": "SourceID identifies to which source a record belongs (there are five "
                        "sources). The sources are deduplicated.",
        "n_clusters_violating": len(viol),
        "holds": len(viol) == 0,
        "examples": viol[:5],
        "implication": "if it holds, a grouping method that assumes at most one record per source "
                       "per entity is applicable here; if it does not, such a constraint may not be "
                       "imposed"}

    # ---------------------------------------------------------------- 4. gold-label exposure check
    exposure: Dict[str, Any] = {}
    # (a) do CID/CTID encode the relation? yes by definition - recorded, not tested
    exposure["CID"] = {"encodes_entity": True, "reason": "cluster id; equal CID means duplicate",
                   "exposed_to_model": False}
    exposure["CTID"] = {"encodes_entity": True,
                    "reason": "position within the cluster; only meaningful given CID",
                    "exposed_to_model": False}
    # (b) does the source-native `id` reveal co-membership?
    by_id = collections.defaultdict(set)
    for r in rows:
        by_id[r["id"]].add(r["CID"])
    shared_id = {k: v for k, v in by_id.items() if len(v) > 1}
    same_id_same_cid = sum(1 for k, v in by_id.items() if len(v) == 1 and
                           sum(1 for r in rows if r["id"] == k) > 1)
    # prefix structure: do records in one cluster share an id prefix more than chance?
    def prefix(s: str, n: int = 6) -> str:
        return re.sub(r"[^A-Za-z]", "", s)[:n]
    same_pref_in_cluster = 0
    pairs_checked = 0
    for v in clusters.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                pairs_checked += 1
                if prefix(v[i]["id"]) and prefix(v[i]["id"]) == prefix(v[j]["id"]):
                    same_pref_in_cluster += 1
    exposure["id"] = {
        "encodes_entity": None,
        "n_distinct_id": len(by_id),
        "ids_reused_across_different_clusters": len(shared_id),
        "ids_reused_within_one_cluster_only": same_id_same_cid,
        "within_cluster_pairs_sharing_an_alpha_prefix": same_pref_in_cluster,
        "within_cluster_pairs_checked": pairs_checked,
        "within_cluster_prefix_share_rate": (same_pref_in_cluster / pairs_checked)
        if pairs_checked else None,
        "decision": "kept out of model input regardless: the readme says its uniqueness is not "
                    "guaranteed and that it can be ignored, so it carries no semantic value worth "
                    "the exposure risk",
        "exposed_to_model": False}
    exposure["SourceID"] = {
        "encodes_entity": False,
        "reason": "which catalogue a record came from; not the entity relation",
        "exposed_to_model": False,
        "decision": "kept in logs. It is not gold, but it is provenance metadata a deployed matcher "
                    "would not normally paste into a record description, and the duplicate-free-"
                    "source property makes it weak evidence about co-membership, so it stays out."}
    rep["gold_label_exposure"] = exposure

    # ---------------------------------------------------------------- 5. fields
    fields: Dict[str, Any] = {}
    for c in SEMANTIC_COLUMNS:
        vals = [r.get(c, "") for r in rows]
        nonempty = [v for v in vals if v not in (None, "")]
        fields[c] = {
            "missing": len(vals) - len(nonempty),
            "missing_rate": (len(vals) - len(nonempty)) / len(vals),
            "n_distinct": len(set(nonempty)),
            "max_len": max((len(v) for v in nonempty), default=0),
            "example": nonempty[0] if nonempty else None,
            "non_ascii_values": sum(1 for v in nonempty if any(ord(ch) > 127 for ch in v))}
    rep["semantic_fields"] = fields
    rep["encoding"] = {"read_as": "utf-8",
                       "rows_with_any_non_ascii": sum(
                           1 for r in rows if any(
                               any(ord(ch) > 127 for ch in str(r.get(c) or ""))
                               for c in SEMANTIC_COLUMNS))}

    # ---------------------------------------------------------------- verdict
    ok = (rep["integrity"]["tid_unique"]
          and rep["integrity"]["every_record_has_cid"]
          and all(rep["published_comparison"]["matches"].values()))
    rep["eligibility_verdict"] = {
        "record_ids_unique": rep["integrity"]["tid_unique"],
        "one_interpretable_gold_entity_per_record": rep["integrity"]["every_record_has_cid"],
        "published_statistics_reproduce": all(rep["published_comparison"]["matches"].values()),
        "entity_disjoint_split_constructible": True,
        "distinct_structural_regime": True,
        "eligible": bool(ok),
        "note": "eligibility is decided on these criteria only, never on performance"}

    p = os.path.join(OUT, "a0_musicbrainz_vet.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=1, default=str)

    i, pc = rep["integrity"], rep["published_comparison"]
    print(f"records {i['n_records']} (TID unique: {i['tid_unique']}), clusters {i['n_clusters']}, "
          f"gold positive pairs {i['gold_positive_pairs']}, sources {rep['sources']['n_sources']}")
    print(f"published comparison: {pc['matches']}")
    print(f"cluster sizes {i['cluster_size_min']}-{i['cluster_size_max']}, "
          f"distribution {i['cluster_size_distribution']}, singletons {i['n_singleton_clusters']}")
    print(f"duplicate-free sources holds: {rep['duplicate_free_sources']['holds']} "
          f"({rep['duplicate_free_sources']['n_clusters_violating']} violating clusters)")
    print(f"CTID contiguous within clusters: {i['ctid_starts_at_1_and_is_contiguous']}")
    L = rep["gold_label_exposure"]["id"]
    print(f"id column: {L['n_distinct_id']} distinct, reused across clusters "
          f"{L['ids_reused_across_different_clusters']}, within-cluster alpha-prefix share "
          f"{L['within_cluster_prefix_share_rate']:.4f} -> excluded from model input")
    print("field missingness: " + ", ".join(
        f"{k} {v['missing_rate']*100:.1f}%" for k, v in rep["semantic_fields"].items()))
    print(f"ELIGIBLE: {rep['eligibility_verdict']['eligible']}")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
