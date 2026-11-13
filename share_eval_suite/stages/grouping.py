"""Entity matching after the final decision: pair judgments, grouping, and the relations groups imply.

A matching workflow returns one judgment per candidate pair. Downstream, a grouping rule turns the
judged matches into record groups, and every two records in one group become related, directly (the
pair was judged a match) or indirectly (they were joined through other records). A few changed pair
judgments can therefore add many incorrect indirect relations while pair F1 barely moves. This module
scores judgments at three levels (candidate pairs, direct links, group relations), measures what a
change in the judgments adds through grouping, and gives the grouping-rule, peer-content and
return-rule comparisons on matching runs.

A judgment is a match only if it reads "match"; a missing or unreadable judgment counts as a non-match.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from share_eval_suite import paths

MERGE = "fig_merge-validation"
RULES = "text_grouping-rules"
CONTENT = "text_matching-content"
RETURN_RULE = "text_matching-return-rule"

#: the six single models and the vote over their judgments, in figure order
MODELS = ("qwen2.5-7b", "llama-3-8b", "mistral-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b")
METHODS = tuple(f"single:{m}" for m in MODELS) + ("vote:majority",)

#: grouping-rule selection: datasets, development-size sampling
_RULE_DATASETS = {"wdc_products_E5f": "wdc", "geographic_settlements_E5f": "settlements",
                  "musicbrainz20k_e5p": "musicbrainz"}
_DEVSIZE_SEED = 20261105
_DEVSIZE_REPS = 200          # development-entity samples per fraction
_DIAG_REPS = 50              # samples for the false-pair / recall diagnostic
_DEVSIZE_FRACS = (0.05, 0.20, 0.30, 0.50, 1.00)
_DIAG_FRACS = (0.10, 0.50, 1.00)

#: peer inputs of the matching recipients
_MENU = ("no_peer", "peer_decision", "decision_plus_fields", "full_explanation")
_MENU_ORDER = {a: i for i, a in enumerate(sorted(_MENU))}
_RECIPIENTS = {"gemma-4-26b": "gemma", "gpt-oss-120b": "gptoss", "qwen3-32b": "qwen3"}

#: return rules: column of the run table, name in the result keys
_RETURN_RULES = (("released", "protocol"), ("majority_first", "majority_first"),
                 ("majority_last", "majority_last"), ("majority_keep", "majority_kept"))
_RETURN_PROTOCOLS = {"du_mad": "dumad", "madm2_objective": "madm2", "bccs": "bccs"}
_RETURN_BENCHMARKS = {"wdc_products": "wdc", "geographic_settlements": "settlements"}


def _dir(obj: str, inputs_dir: Optional[pathlib.Path]) -> pathlib.Path:
    return pathlib.Path(inputs_dir) if inputs_dir is not None else paths.inputs(obj)


def _rd(inp: pathlib.Path, name: str) -> List[Dict[str, str]]:
    with open(inp / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f1(tp: float, fp: float, fn: float) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _modal(vals: Iterable[str]) -> Optional[str]:
    """The most frequent non-empty judgment; `None` when there is none or the top count is tied."""
    v = [x for x in vals if x]
    if not v:
        return None
    c = collections.Counter(v)
    t = max(c.values())
    w = [k for k, n in c.items() if n == t]
    return w[0] if len(w) == 1 else None


def _components(nodes: Sequence[str], edges: Iterable[Tuple[str, str]]) -> List[List[str]]:
    """Connected components by union-find; groups listed in order of their first record."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    out = collections.defaultdict(list)
    for n in nodes:
        out[find(n)].append(n)
    return list(out.values())


def _intra(groups: Iterable[Sequence[str]], ent: Mapping[str, str]) -> Tuple[int, int]:
    """Predicted relations implied by a grouping, and how many of them are true."""
    n = t = 0
    for mem in groups:
        k = len(mem)
        n += k * (k - 1) // 2
        t += sum(v * (v - 1) // 2 for v in collections.Counter(ent[m] for m in mem).values())
    return n, t


def _gold_pairs_total(nodes: Iterable[str], ent: Mapping[str, str]) -> int:
    """True relations over all record pairs: pairs of records with the same reference entity."""
    return sum(v * (v - 1) // 2 for v in collections.Counter(ent[n] for n in nodes).values())


def _three_levels(pairs, gold_of, links, nodes, ent, n_gold) -> Dict[str, Any]:
    """F1 on the candidate pairs (A), of the direct links over all pairs (B), and of the relations
    connected components imply over all pairs (C)."""
    tp = sum(1 for p in pairs if p in links and gold_of[p])
    fp = sum(1 for p in pairs if p in links and not gold_of[p])
    fn = sum(1 for p in pairs if p not in links and gold_of[p])
    a = _f1(tp, fp, fn)
    b = _f1(tp, len(links) - tp, n_gold - tp)
    groups = _components(nodes, links)
    n, t = _intra(groups, ent)
    c = _f1(t, n - t, n_gold - t)
    return dict(A=a, B=b, C=c, tpA=tp, fpA=fp, tpC=t, fpC=n - t, groups=groups)


def _bridges(groups, new_links, ent, F) -> List[Dict[str, Any]]:
    """Each new link added alone to the given groups. A link that joins groups U and V adds |U||V|
    relations, t of them true; it raises cluster F1 exactly when t / (|U||V|) > F / 2."""
    gid = {n: i for i, mem in enumerate(groups) for n in mem}
    cnt = [collections.Counter(ent[n] for n in mem) for mem in groups]
    out = []
    for a, b in new_links:
        if gid[a] == gid[b]:
            continue
        U, V = cnt[gid[a]], cnt[gid[b]]
        bb = sum(U.values()) * sum(V.values())
        tt = sum(U[e] * V[e] for e in U)
        out.append(dict(correct=ent[a] == ent[b], b=bb, t=tt, harmful=tt / bb < F / 2))
    return out


def _method_levels(rows, pairs, gold_of, nodes, ent, G) -> Dict[str, Dict[str, Any]]:
    """Three levels of each single model and of the vote (modal judgment of the six models)."""
    lv = {}
    for m in METHODS:
        if m == "vote:majority":
            dec = {(r["left"], r["right"]): _modal([r[x] for x in METHODS[:-1]]) for r in rows}
        else:
            dec = {(r["left"], r["right"]): (r[m] or None) for r in rows}
        links = {p for p, d in dec.items() if d == "match"}
        lv[m] = _three_levels(pairs, gold_of, links, nodes, ent, G)
    return lv


# ------------------------------------------------------------------ three evaluation levels
@dataclass
class Levels:
    """F1 of one set of judgments at three levels, and the groups connected components form.

    `pair_f1`: F1 on the candidate pairs; `direct_f1`: F1 of the judged matches (direct links) over all
    record pairs; `group_f1`: cluster F1, the F1 of the relations the groups imply over all record pairs.
    """
    pair_f1: float
    direct_f1: float
    group_f1: float
    groups: List[List[str]]


def connected_components(records: Sequence[str], links: Iterable[Tuple[str, str]]) -> List[List[str]]:
    """Groups of records joined by judged matches (connected components)."""
    return _components(list(records), list(links))


def relations(groups: Iterable[Sequence[str]]) -> set:
    """Every record pair inside a group, as sorted tuples (record ids compared as integers)."""
    out = set()
    for mem in groups:
        ms = sorted(mem, key=int)
        out.update((ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms)))
    return out


def evaluation_levels(pairs: Sequence[Tuple[str, str]], gold: Mapping[Tuple[str, str], bool],
                      links: Iterable[Tuple[str, str]], records: Sequence[str],
                      entity: Mapping[str, str]) -> Levels:
    """Score judged matches `links` over candidate `pairs` with reference labels `gold`, against the
    reference entity of every record."""
    n_gold = _gold_pairs_total(list(records), entity)
    r = _three_levels(list(pairs), dict(gold), set(links), list(records), entity, n_gold)
    return Levels(r["A"], r["B"], r["C"], r["groups"])


# ------------------------------------------------------------------ Settlements judgments
@dataclass
class SettlementsJudgments:
    """Settlements test records with their reference entities and the judged candidate pairs."""
    records: List[str]
    entity: Dict[str, str]
    rows: List[Dict[str, str]]

    @property
    def pairs(self) -> List[Tuple[str, str]]:
        return [(r["left"], r["right"]) for r in self.rows]

    @property
    def gold(self) -> Dict[Tuple[str, str], bool]:
        return {(r["left"], r["right"]): r["gold"] == "1" for r in self.rows}


def load_settlements(inputs_dir: Optional[pathlib.Path] = None) -> SettlementsJudgments:
    """The Settlements records and candidate pairs (each pair: reference label, each model's decision
    and each recipient's decision with and without peers)."""
    inp = _dir(MERGE, inputs_dir)
    recs = _rd(inp, "settlements_records.csv")
    rows = _rd(inp, "settlements_pairs.csv")
    return SettlementsJudgments([r["record"] for r in recs], {r["record"]: r["entity"] for r in recs}, rows)


@dataclass
class GroupingEffect:
    """What one change of peer input does to the judgments and to the groups they form.

    Changes are with-peers minus without-peers, in percentage points. Changed judgments are counted on
    the pairs that carry every recipient and peer-input decision; a changed judgment is correct when it
    moves to the reference label. Added and removed relations are record pairs that become, or stop
    being, in one group under connected components; an added relation is direct when the pair itself
    is judged a match, otherwise indirect.
    """
    candidate_pairs: int
    pair_f1_change_pp: float
    group_f1_change_pp: float
    changed_judgments: int
    changed_correct: int
    changed_incorrect: int
    added_relations: int
    added_true: int
    added_false: int
    added_direct: int
    added_indirect_false: int
    removed_relations: int


def sharing_effect_on_groups(recipient: str = "qwen2.5-32b", with_peers: str = "candidate_only",
                             without_peers: str = "no_peers",
                             data: Optional[SettlementsJudgments] = None) -> GroupingEffect:
    """Compare one recipient's Settlements judgments with peers (`candidate_only`: the peers' decisions;
    `full`: their full replies) against its judgments without peers, at the pair level and through
    connected-components grouping. Levels are scored on the pairs that carry all six recipient x
    peer-input decisions, against all Settlements records."""
    d = data or load_settlements()
    common = [r for r in d.rows if r["share_complete"] == "1"]
    cpairs = [(r["left"], r["right"]) for r in common]
    gold = d.gold

    def arm(cond):
        col = f"share:{recipient}:{cond}"
        links = {(r["left"], r["right"]) for r in common if r[col] == "match"}
        dec = {(r["left"], r["right"]): r[col] for r in common}
        return links, dec, evaluation_levels(cpairs, gold, links, d.records, d.entity)

    la, da, A = arm(with_peers)
    lb, db, B = arm(without_peers)
    n2m = [p for p in cpairs if da[p] == "match" and db[p] != "match"]
    m2n = [p for p in cpairs if da[p] != "match" and db[p] == "match"]
    ra, rb = relations(A.groups), relations(B.groups)
    added, removed = ra - rb, rb - ra
    direct = {tuple(sorted(p, key=int)) for p in la}
    ent = d.entity
    return GroupingEffect(
        candidate_pairs=len(cpairs),
        pair_f1_change_pp=100 * (A.pair_f1 - B.pair_f1),
        group_f1_change_pp=100 * (A.group_f1 - B.group_f1),
        changed_judgments=len(n2m) + len(m2n),
        changed_correct=sum(1 for p in n2m if gold[p]) + sum(1 for p in m2n if not gold[p]),
        changed_incorrect=sum(1 for p in n2m if not gold[p]) + sum(1 for p in m2n if gold[p]),
        added_relations=len(added),
        added_true=sum(1 for p in added if ent[p[0]] == ent[p[1]]),
        added_false=sum(1 for p in added if ent[p[0]] != ent[p[1]]),
        added_direct=sum(1 for p in added if p in direct),
        added_indirect_false=sum(1 for p in added if p not in direct and ent[p[0]] != ent[p[1]]),
        removed_relations=len(removed),
    )


# ------------------------------------------------------------------ merge validation
def _merge_wdc(inp: pathlib.Path, vals: Dict[str, Any]) -> List[List[float]]:
    """WDC: true relations, those among the candidate pairs, the direct-link F1 cap, and the three
    levels of each method (figure rows)."""
    rows = _rd(inp, "wdc_pairs.csv")
    ent = {}
    for r in rows:
        ent[r["left"]] = r["left_entity"]
        ent[r["right"]] = r["right_entity"]
    nodes = sorted(ent)
    pairs = [(r["left"], r["right"]) for r in rows]
    gold_of = {(r["left"], r["right"]): r["gold"] == "1" for r in rows}
    G = _gold_pairs_total(nodes, ent)
    pos = sum(gold_of.values())
    vals["wdc_true_matches"] = G
    vals["wdc_true_matches_in_candidates"] = pos
    # direct links found only among the candidates: recall at most pos / G, precision at most 1
    r_ = pos / G
    vals["wdc_direct_link_f1_cap"] = 2 * r_ / (1 + r_)
    lv = _method_levels(rows, pairs, gold_of, nodes, ent, G)
    return [[lv[m]["A"], lv[m]["B"], lv[m]["C"]] for m in METHODS]


def _second_collection(inp: pathlib.Path, vals: Dict[str, Any], nodes, ent) -> None:
    """Qwen2.5-32B, peer decisions against no peers, in the second judgment collection on the same
    Settlements records and pairs: changed judgments and the relations they add."""
    rows = _rd(inp, "settlements_second_collection_q32.csv")
    base = {(r["left"], r["right"]) for r in rows if r["no_peers"] == "match"}
    alt = {(r["left"], r["right"]) for r in rows if r["peer_decisions"] == "match"}
    vals["second_q32_changed_judgments"] = len(base ^ alt)
    added = relations(_components(nodes, alt)) - relations(_components(nodes, base))
    direct = {tuple(sorted(p, key=int)) for p in alt}
    vals["second_q32_added_relations"] = len(added)
    vals["second_q32_added_indirect_false"] = sum(1 for p in added if p not in direct and ent[p[0]] != ent[p[1]])


def _merge_settlements(inp: pathlib.Path, vals: Dict[str, Any]) -> List[List[float]]:
    """Settlements: three levels of each method; with vs without peers through grouping for two
    recipients and two peer inputs; the Qwen2.5-32B peer-decision contrast in detail (changed
    judgments, added and removed relations, merges, bridging links)."""
    recs = _rd(inp, "settlements_records.csv")
    nodes = [r["record"] for r in recs]
    ent = {r["record"]: r["entity"] for r in recs}
    rows = _rd(inp, "settlements_pairs.csv")
    pairs = [(r["left"], r["right"]) for r in rows]
    gold_of = {(r["left"], r["right"]): r["gold"] == "1" for r in rows}
    G = _gold_pairs_total(nodes, ent)
    vals["settlements_test_records"] = len(nodes)
    vals["settlements_candidate_pairs"] = len(pairs)
    vals["settlements_candidate_coverage_pct"] = 100 * sum(gold_of.values()) / G
    lv = _method_levels(rows, pairs, gold_of, nodes, ent, G)
    fig = [[lv[m]["A"], lv[m]["B"], lv[m]["C"]] for m in METHODS]
    # with vs without peers, on the pairs that carry all six recipient x peer-input decisions
    common = [r for r in rows if r["share_complete"] == "1"]
    cpairs = [(r["left"], r["right"]) for r in common]
    vals["settlements_sharing_pairs"] = len(cpairs)
    sh = {}
    for rc in ("qwen2.5-7b", "qwen2.5-32b"):
        for c in ("no_peers", "candidate_only", "full"):
            col = f"share:{rc}:{c}"
            links = {(r["left"], r["right"]) for r in common if r[col] == "match"}
            sh[(rc, c)] = dict(links=links, dec={(r["left"], r["right"]): r[col] for r in common},
                               **_three_levels(cpairs, gold_of, links, nodes, ent, G))
    for rc in ("qwen2.5-7b", "qwen2.5-32b"):
        for c in ("candidate_only", "full"):
            tag = f"{rc.replace('qwen2.5-', 'q')}_{'decisions' if c == 'candidate_only' else 'full'}"
            vals[f"share_{tag}_pair_pp"] = 100 * (sh[(rc, c)]["A"] - sh[(rc, "no_peers")]["A"])
            vals[f"share_{tag}_cluster_pp"] = 100 * (sh[(rc, c)]["C"] - sh[(rc, "no_peers")]["C"])
    # Qwen2.5-32B, peer decisions vs no peers: which judgments change and what grouping adds
    A, B = sh[("qwen2.5-32b", "candidate_only")], sh[("qwen2.5-32b", "no_peers")]
    n2m = [p for p in cpairs if A["dec"][p] == "match" and B["dec"][p] != "match"]
    m2n = [p for p in cpairs if A["dec"][p] != "match" and B["dec"][p] == "match"]
    vals["q32_changed_judgments"] = len(n2m) + len(m2n)
    vals["q32_nonmatch_to_match"] = len(n2m)
    vals["q32_match_to_nonmatch"] = len(m2n)
    vals["q32_changed_correct"] = sum(1 for p in n2m if gold_of[p]) + sum(1 for p in m2n if not gold_of[p])
    vals["q32_changed_incorrect"] = sum(1 for p in n2m if not gold_of[p]) + sum(1 for p in m2n if gold_of[p])
    ra, rb = relations(A["groups"]), relations(B["groups"])
    added, removed = ra - rb, rb - ra
    direct = {tuple(sorted(p, key=int)) for p in A["links"]}
    vals["q32_added_relations"] = len(added)
    vals["q32_added_true"] = sum(1 for p in added if ent[p[0]] == ent[p[1]])
    vals["q32_added_false"] = sum(1 for p in added if ent[p[0]] != ent[p[1]])
    vals["q32_removed_relations"] = len(removed)
    vals["q32_removed_true"] = sum(1 for p in removed if ent[p[0]] == ent[p[1]])
    vals["q32_removed_false"] = sum(1 for p in removed if ent[p[0]] != ent[p[1]])
    vals["q32_added_direct"] = sum(1 for p in added if p in direct)
    vals["q32_added_indirect"] = sum(1 for p in added if p not in direct)
    vals["q32_added_indirect_false"] = sum(1 for p in added if p not in direct and ent[p[0]] != ent[p[1]])
    # groups with peers that join several groups without peers
    gid_b = {n: i for i, mem in enumerate(B["groups"]) for n in mem}
    merges = []
    for mem in A["groups"]:
        before = {gid_b[n] for n in mem}
        if len(before) > 1:
            ms = set(mem)
            add = [p for p in added if p[0] in ms]
            merges.append(dict(size=len(mem), groups_before=len(before),
                               entities=len({ent[n] for n in mem}),
                               added_false_indirect=sum(1 for p in add if p not in direct
                                                        and ent[p[0]] != ent[p[1]]),
                               false_relations_in_group=sum(1 for p in relations([mem])
                                                            if ent[p[0]] != ent[p[1]])))
    gid_a = {n: i for i, mem in enumerate(A["groups"]) for n in mem}
    vals["q32_merges"] = len(merges)
    vals["q32_splits"] = sum(1 for mem in B["groups"] if len({gid_a[n] for n in mem}) > 1)
    big = max(merges, key=lambda m: m["size"])
    vals["q32_largest_merge_size"] = big["size"]
    vals["q32_largest_merge_groups_before"] = big["groups_before"]
    vals["q32_largest_merge_entities"] = big["entities"]
    vals["q32_largest_merge_added_false_indirect"] = big["added_false_indirect"]
    vals["q32_largest_merge_false_relations"] = big["false_relations_in_group"]
    _second_collection(inp, vals, nodes, ent)
    # each new match link added alone to the no-peer groups
    br = _bridges(B["groups"], n2m, ent, B["C"])
    inc = [x for x in br if not x["correct"]]
    vals["q32_bridging_links"] = len(br)
    vals["q32_bridging_links_incorrect"] = len(inc)
    vals["q32_incorrect_bridges_below_condition"] = sum(1 for x in inc if x["harmful"])
    vals["q32_incorrect_bridges_above_condition"] = sum(1 for x in inc if not x["harmful"])
    vals["q32_incorrect_bridges_alone_relations"] = sum(x["b"] for x in inc)
    vals["q32_incorrect_bridges_alone_false"] = sum(x["b"] - x["t"] for x in inc)
    vals["q32_bridges_alone_relations"] = sum(x["b"] for x in br)
    vals["q32_largest_single_bridge_relations"] = max(x["b"] for x in br)
    return fig


def _merge_grouping(inp: pathlib.Path, vals: Dict[str, Any]) -> None:
    """Grouping-rule matrix on the test side: connected components against center clustering, and the
    sign of the sharing effect on pair F1 and on cluster F1 under each of the 15 grouping settings."""
    lv = [r for r in _rd(inp, "config_levels.csv") if r["split"] == "test"]
    rules = [k[len("cluster_f1__"):] for k in lv[0] if k.startswith("cluster_f1__")]
    # one configuration per recipient x condition (first draw for the random-peer conditions)
    cmp_ = collections.defaultdict(list)
    for r in lv:
        if r["draw"] in ("", "0"):
            cmp_[r["dataset"]].append(100 * (float(r["cluster_f1__connected_components"])
                                             - float(r["cluster_f1__center"])))
    vals["cc_vs_center_comparisons"] = sum(len(v) for v in cmp_.values())
    for ds, v in cmp_.items():
        t = ds.lower()
        vals[f"cc_minus_center_{t}_n"] = len(v)
        vals[f"cc_minus_center_{t}_min_pp"] = min(v)
        vals[f"cc_minus_center_{t}_max_pp"] = max(v)
        vals[f"cc_minus_center_{t}_negative"] = sum(1 for x in v if x < 0)
    # with peers minus no peers, grouping setting held fixed
    base = {(r["dataset"], r["recipient"]): r for r in lv if r["arm"] == "N"}

    def sgn(x, tol):
        return 0 if abs(x) <= tol else (1 if x > 0 else -1)
    combos = []
    for r in lv:
        if r["arm"] == "N":
            continue
        b = base[(r["dataset"], r["recipient"])]
        dp = 100 * (float(r["pair_f1"]) - float(b["pair_f1"]))
        for k in rules:
            dc = 100 * (float(r[f"cluster_f1__{k}"]) - float(b[f"cluster_f1__{k}"]))
            combos.append(dict(dataset=r["dataset"], config=r["cell"], rule=k,
                               family=k.split("@")[0], d_pair_pp=dp, d_cluster_pp=dc,
                               exact=(sgn(dp, 1e-9), sgn(dc, 1e-9)),
                               band=(sgn(dp, 0.10), sgn(dc, 0.10))))

    def rev(c, f):
        return c[f] in ((1, -1), (-1, 1))
    vals["sharing_configurations"] = len({(c["dataset"], c["config"]) for c in combos})
    vals["combinations"] = len(combos)
    vals["both_up"] = sum(1 for c in combos if c["exact"] == (1, 1))
    vals["both_down"] = sum(1 for c in combos if c["exact"] == (-1, -1))
    vals["reversals"] = sum(1 for c in combos if rev(c, "exact"))
    vals["pair_change_cluster_unchanged"] = sum(1 for c in combos if c["exact"][0] != 0 and c["exact"][1] == 0)
    vals["stable_same_direction"] = vals["both_up"] + vals["both_down"]
    vals["configurations_with_reversal"] = len({(c["dataset"], c["config"]) for c in combos if rev(c, "exact")})
    vals["reversals_band010"] = sum(1 for c in combos if rev(c, "band"))
    up_dn = [c["d_cluster_pp"] for c in combos if c["exact"] == (1, -1)]
    dn_up = [c["d_cluster_pp"] for c in combos if c["exact"] == (-1, 1)]
    vals["pairup_groupdown_cluster_median_pp"] = statistics.median(up_dn)
    vals["pairup_groupdown_cluster_min_pp"] = min(up_dn)
    vals["pairup_groupdown_cluster_max_pp"] = max(up_dn)
    vals["pairdown_groupup_cluster_max_pp"] = max(dn_up)
    for ds in ("WDC", "Settlements", "MusicBrainz"):
        cs = [c for c in combos if c["dataset"] == ds]
        t = ds.lower()
        vals[f"rev_{t}_n"] = len(cs)
        vals[f"rev_{t}_pairup_groupdown"] = sum(1 for c in cs if c["exact"] == (1, -1))
        vals[f"rev_{t}_pairdown_groupup"] = sum(1 for c in cs if c["exact"] == (-1, 1))
        vals[f"rev_{t}_band010"] = sum(1 for c in cs if rev(c, "band"))
    for fam in ("center", "connected_components", "merge_center", "star_disjoint",
                "correlation_greedy", "markov"):
        cs = [c for c in combos if c["family"] == fam]
        vals[f"rev_{fam}_n"] = len(cs)
        vals[f"rev_{fam}_pairup_groupdown"] = sum(1 for c in cs if c["exact"] == (1, -1))
        vals[f"rev_{fam}_pairdown_groupup"] = sum(1 for c in cs if c["exact"] == (-1, 1))
    cp = [c for c in combos if c["dataset"] == "Settlements" and c["config"] == "qwen2.5-32b|D5|none"
          and c["rule"] == "connected_components"][0]
    vals["entity_disjoint_counterpart_cluster_pp"] = cp["d_cluster_pp"]
    vals["entity_disjoint_counterpart_pair_pp"] = cp["d_pair_pp"]
    # configurations whose sharing effect on cluster F1 changes sign across the 15 grouping settings
    eff = collections.defaultdict(list)
    for c in combos:
        eff[(c["dataset"], c["config"])].append(c["d_cluster_pp"])
    vals["configurations_sign_differs_across_settings"] = sum(
        1 for v in eff.values() if min(v) < -1e-7 and max(v) > 1e-7)


def merge_validation(inputs_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Pair, direct-link and cluster F1 of each model and of the vote on WDC and Settlements; the
    Settlements peer-input contrast through grouping; and the grouping-rule matrix (connected
    components against center clustering, pair-up/group-down reversals over 15 grouping settings).
    Keys are those of the paper's merge-validation figure and the text around it."""
    inp = _dir(MERGE, inputs_dir)
    vals: Dict[str, Any] = {}
    wdc = _merge_wdc(inp, vals)
    settlements = _merge_settlements(inp, vals)
    _merge_grouping(inp, vals)
    # figure cells: method x level
    for i, meth in enumerate(METHODS):
        short = meth.split(":")[1]
        for j, lvl in enumerate(("pairs", "direct", "groups")):
            vals[f"fig_wdc_{short}_{lvl}"] = wdc[i][j]
            vals[f"fig_settlements_{short}_{lvl}"] = settlements[i][j]
    return vals


# ------------------------------------------------------------------ grouping-rule selection
def _menu_and_regret(inp: pathlib.Path, vals: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Menu size; per dataset the test regret (best setting minus selected setting, pp) of the setting
    selected on development data and of always using one rule; the selected rule families."""
    rc = _rd(inp, "rule_cells.csv")
    sel = _rd(inp, "selection.csv")
    rules = list(dict.fromkeys(r["rule"] for r in rc))
    vals["menu_settings"] = len(rules)
    vals["menu_families"] = len({r.split("@")[0] for r in rules})
    test: Dict[Tuple[str, str], Dict[str, float]] = collections.defaultdict(dict)
    for r in rc:
        test[(r["dataset"], r["cell"])][r["rule"]] = float(r["test_cluster_f1"])
    for ds, t in _RULE_DATASETS.items():
        cells = [k for k in test if k[0] == ds]
        vals[f"{t}_configurations"] = len(cells)
        s = [r for r in sel if r["dataset"] == ds]
        reg = []
        for r in s:
            best = max(test[(ds, r["cell"])].values())
            g = 100 * (best - test[(ds, r["cell"])][r["selected"]])
            if abs(g - float(r["regret_pp"])) >= 1e-9:
                raise ValueError(f"selection table regret differs for {ds} {r['cell']}")
            reg.append(g)
        vals[f"{t}_selected_mean_regret_pp"] = statistics.mean(reg)
        vals[f"{t}_selected_max_regret_pp"] = max(reg)
        for fixed in ("connected_components", "center"):
            vals[f"{t}_fixed_{fixed}_mean_regret_pp"] = statistics.mean(
                100 * (max(test[c].values()) - test[c][fixed]) for c in cells)
        fam = collections.Counter(r["selected"].split("@")[0] for r in s)
        for f in ("connected_components", "correlation_greedy", "markov", "center",
                  "merge_center", "star_disjoint"):
            vals[f"{t}_selected_{f}"] = fam.get(f, 0)
    vals["selected_mean_regret_max_over_datasets_pp"] = max(
        vals[f"{t}_selected_mean_regret_pp"] for t in _RULE_DATASETS.values())
    return test


def _timing_and_sizes(inp: pathlib.Path, vals: Dict[str, Any]) -> None:
    """Seconds of the grouping algorithm per configuration, summed over all settings (development
    side), and the development entity and record counts."""
    g = json.loads((inp / "grouping.json").read_text(encoding="utf-8"))
    allt = []
    for ds, t in _RULE_DATASETS.items():
        d = g["domains"][ds]
        secs = [sum(x["cpu_seconds"] for x in c["rules"].values()) for c in d["dev"].values()]
        vals[f"{t}_dev_seconds_all_settings_min"] = min(secs)
        vals[f"{t}_dev_seconds_all_settings_max"] = max(secs)
        vals[f"{t}_dev_entities"] = d["dev_entities"]
        vals[f"{t}_dev_records"] = d["dev_records"]
        allt += secs
    vals["dev_seconds_all_settings_min"] = min(allt)
    vals["dev_seconds_all_settings_max"] = max(allt)


def _development_size(inp: pathlib.Path, vals: Dict[str, Any], test) -> None:
    """Settlements: development cluster F1 scored only on the records of a sample of development
    entities (groups kept from the complete development graph); the setting with the highest score
    (ties to the first setting in menu order) and its test regret, averaged over 200 samples per
    fraction (one sample at fraction 1.00) and the 24 configurations. Each sample draws entities without replacement with Python
    `random.Random` seeded by `"20261105|<dataset>|<fraction>|<sample>"`. Also the false pairs and
    recall of connected components on sampled entities (50 samples per fraction, one at 1.00)."""
    ds = "geographic_settlements_E5f"
    rows = _rd(inp, "settlements_dev_groups.csv")
    cols = [c for c in rows[0] if "||" in c]
    cells = list(dict.fromkeys(c.split("||")[0] for c in cols))
    keys = list(dict.fromkeys(c.split("||")[1] for c in cols))
    ent_ids = sorted({r["entity"] for r in rows})
    eidx = {e: i for i, e in enumerate(ent_ids)}
    rec_e = np.array([eidx[r["entity"]] for r in rows])
    size = np.bincount(rec_e, minlength=len(ent_ids))
    # predicted relations of every configuration x setting, as pairs of reference-entity indices
    ci, ei, ej = [], [], []
    for k, c in enumerate(cols):
        by = collections.defaultdict(list)
        for i, r in enumerate(rows):
            by[r[c]].append(rec_e[i])
        for mem in by.values():
            for a in range(len(mem)):
                for b in range(a + 1, len(mem)):
                    ci.append(k)
                    ei.append(mem[a])
                    ej.append(mem[b])
    ci, ei, ej = np.array(ci), np.array(ei), np.array(ej)
    same = ei == ej
    ncol = len(cols)
    order = {k: i for i, k in enumerate(keys)}
    vals["settlements_dev_entities_in_groups_table"] = len(ent_ids)

    def scores(pick):
        # predicted relations N, true relations T among the sampled entities' records; gold pairs G
        s = np.zeros(len(ent_ids), dtype=bool)
        s[list(pick)] = True
        vis = s[ei] & s[ej]
        N = np.bincount(ci[vis], minlength=ncol)
        T = np.bincount(ci[vis & same], minlength=ncol)
        G = int((size[s] * (size[s] - 1) // 2).sum())
        return N, T, G

    def draw(frac, rep):
        k = len(ent_ids) if frac >= 1.0 else max(2, round(frac * len(ent_ids)))
        rng = random.Random(f"{_DEVSIZE_SEED}|{ds}|{frac}|{rep}")
        pick = set(ent_ids) if k == len(ent_ids) else set(rng.sample(ent_ids, k))
        return k, {eidx[e] for e in pick}

    for frac in _DEVSIZE_FRACS:
        regs = []
        for rep in (range(1) if frac >= 1.0 else range(_DEVSIZE_REPS)):
            k, pick = draw(frac, rep)
            N, T, G = scores(pick)
            for j, cell in enumerate(cells):
                cf = {}
                for kk in keys:
                    col = j * len(keys) + order[kk]
                    n, t = int(N[col]), int(T[col])
                    cf[kk] = (2 * t / (n + G)) if (n + G) else 0.0
                top = max(cf.values())
                chosen = sorted([kk for kk in cf if cf[kk] == top], key=lambda kk: order[kk])[0]
                tt = test[(ds, cell)]
                regs.append(100 * (max(tt.values()) - tt[chosen]))
        vals[f"settlements_devsize_{k}_entities_mean_regret_pp"] = statistics.mean(regs)
        vals[f"settlements_devsize_frac{frac:.2f}_entities"] = k
    # connected components scored on sampled entities: false pairs and recall
    cc = [j * len(keys) + order["connected_components"] for j in range(len(cells))]
    for frac in _DIAG_FRACS:
        fp, rec = [], []
        for rep in (range(1) if frac >= 1.0 else range(_DIAG_REPS)):
            _, pick = draw(frac, rep)
            N, T, G = scores(pick)
            for col in cc:
                fp.append(int(N[col]) - int(T[col]))
                rec.append(int(T[col]) / G if G else 0.0)
        vals[f"settlements_cc_false_pairs_frac{frac:.2f}"] = statistics.mean(fp)
        vals[f"settlements_cc_recall_frac{frac:.2f}"] = statistics.mean(rec)


def grouping_rule_selection(inputs_dir: Optional[pathlib.Path] = None,
                            development_size: bool = True) -> Dict[str, Any]:
    """Grouping settings chosen on development data from fixed judgments: the test gap of the chosen
    setting to the best one per dataset, the gap of always using one rule, the seconds per
    configuration, and (with `development_size`) the gap when fewer development entities are scored."""
    inp = _dir(RULES, inputs_dir)
    vals: Dict[str, Any] = {}
    test = _menu_and_regret(inp, vals)
    _timing_and_sizes(inp, vals)
    if development_size:
        _development_size(inp, vals, test)
    return vals


# ------------------------------------------------------------------ peer content in matching
def _prf(rows: Sequence[Mapping[str, str]]) -> Dict[str, float]:
    """Pair F1 and recall on the match class (percent) and the counts; an unreadable decision (`""`)
    counts as a non-match."""
    tp = sum(1 for r in rows if r["pred"] == "match" and r["gold"] == "match")
    fp = sum(1 for r in rows if r["pred"] == "match" and r["gold"] != "match")
    fn = sum(1 for r in rows if r["pred"] != "match" and r["gold"] == "match")
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(f1=100 * (2 * p * rc / (p + rc) if p + rc else 0.0), recall=100 * rc,
                tp=tp, fp=fp, fn=fn, n=len(rows))


def _matching(inp: pathlib.Path, name: str, V: Dict[str, Any]):
    """Test pair F1 of each recipient under each peer input; the input selected by development pair F1
    (ties to fewer mean prompt tokens, then input name) and its gaps to the full explanation and to
    the best input on test."""
    rows = collections.defaultdict(list)
    with open(inp / f"{name}_pairs.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["recipient"], r["split"], r["input"])].append(r)
    tag = "ag" if name == "amazon_google" else "da"
    for m in _RECIPIENTS:
        dev = {a: _prf(rows[(m, "dev", a)]) for a in _MENU}
        tok = {a: sum(int(r["prompt_tokens"]) for r in rows[(m, "dev", a)]) / len(rows[(m, "dev", a)])
               for a in _MENU}
        best = max(d["f1"] for d in dev.values())
        chosen = sorted([a for a in _MENU if dev[a]["f1"] == best], key=lambda a: (tok[a], _MENU_ORDER[a]))[0]
        test = {a: _prf(rows[(m, "test", a)]) for a in _MENU}
        for a in _MENU:
            k = f"{tag}_{_RECIPIENTS[m]}_{a}"
            V[k + "_f1"] = test[a]["f1"]
            V[k + "_pairs"] = test[a]["n"]
            if tag == "ag":
                for x in ("tp", "fp", "fn", "recall"):
                    V[f"{k}_{x}"] = test[a][x]
        r_ = _RECIPIENTS[m]
        V[f"{tag}_{r_}_selected_is_no_peer"] = int(chosen == "no_peer")
        V[f"{tag}_{r_}_selected_is_peer_decision"] = int(chosen == "peer_decision")
        V[f"{tag}_{r_}_dev_and_test_best_is_no_peer"] = int(
            chosen == "no_peer" and max(test, key=lambda a: test[a]["f1"]) == "no_peer")
        V[f"{tag}_{r_}_selected_minus_full"] = test[chosen]["f1"] - test["full_explanation"]["f1"]
        V[f"{tag}_{r_}_best_minus_selected"] = max(t["f1"] for t in test.values()) - test[chosen]["f1"]
        V[f"{tag}_{r_}_decision_minus_fields"] = test["peer_decision"]["f1"] - test["decision_plus_fields"]["f1"]
    return rows


def _amazon_google_qwen3(rows, V: Dict[str, Any]) -> None:
    """Paired changes of the Qwen3 judgments: decision with field comparisons against no peers."""
    def ok(r):
        return (r["pred"] or "non-match") == r["gold"]
    base = {r["pair"]: r for r in rows[("qwen3-32b", "test", "no_peer")]}
    trt = {r["pair"]: r for r in rows[("qwen3-32b", "test", "decision_plus_fields")]}
    pairs = sorted(set(base) & set(trt))
    c2w = [p for p in pairs if ok(base[p]) and not ok(trt[p])]
    w2c = [p for p in pairs if not ok(base[p]) and ok(trt[p])]
    V["ag_fields_test_pairs"] = len(pairs)
    V["ag_fields_correct_to_wrong"] = len(c2w)
    V["ag_fields_wrong_to_correct"] = len(w2c)
    V["ag_fields_c2w_true_match_to_nonmatch"] = sum(
        1 for p in c2w if base[p]["gold"] == "match" and base[p]["pred"] == "match" and trt[p]["pred"] == "non-match")
    V["ag_fields_c2w_to_unreadable"] = sum(1 for p in c2w if trt[p]["pred"] == "")
    V["ag_fields_c2w_follow_peer"] = sum(
        1 for p in c2w if (trt[p]["pred"] or "non-match") == trt[p]["shown_peer_decision"])
    V["ag_fields_w2c_follow_peer"] = sum(
        1 for p in w2c if (trt[p]["pred"] or "non-match") == trt[p]["shown_peer_decision"])
    V["ag_true_matches"] = sum(1 for r in trt.values() if r["gold"] == "match")
    V["ag_true_nonmatches"] = sum(1 for r in trt.values() if r["gold"] != "match")
    V["ag_peer_rejects_true_matches"] = sum(
        1 for r in trt.values() if r["gold"] == "match" and r["shown_peer_decision"] != "match")


def matching_peer_content(inputs_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Pair F1 of each recipient on Amazon-Google and DBLP-ACM under each peer input (no peers, the peer
    decision, the decision with field comparisons, the full explanation), the input chosen on
    development data, and the paired judgment changes of Qwen3 when field comparisons are added."""
    inp = _dir(CONTENT, inputs_dir)
    vals: Dict[str, Any] = {}
    ag = _matching(inp, "amazon_google", vals)
    _matching(inp, "dblp_acm", vals)
    _amazon_google_qwen3(ag, vals)
    vals["shared_content_test_pairs"] = vals["da_qwen3_no_peer_pairs"] + vals["ag_qwen3_no_peer_pairs"]
    return vals


# ------------------------------------------------------------------ return rules
def return_rule_effects(inputs_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """The matching sharing effect (pair F1 with peers minus without) under the protocol's return rule
    and under the all-round majority with ties to the first agent, to the last agent, or kept, and how
    far the effect moves.

    The effect is pair F1 (match class, pp) with full sharing minus pair F1 without peers or with
    answer-only peers, pooled over the runs (two backbones, two repeats) of the matching units present
    in both conditions; a run without an answer predicts non-match."""
    inp = _dir(RETURN_RULE, inputs_dir)
    units = collections.defaultdict(list)
    gold = {}
    for r in _rd(inp, "return_rule_units.csv"):
        units[r["benchmark"]].append(r["unit"])
        gold[(r["benchmark"], r["unit"])] = r["gold"] == "match"
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in _rd(inp, "return_rule_runs.csv"):
        by[(r["benchmark"], r["protocol"], r["condition"])][r["unit"]].append(r)

    def level(bench, rows_by_unit, keep, col):
        tp = fp = fn = 0
        for u in keep:
            g = gold[(bench, u)]
            for r in rows_by_unit[u]:
                pred = r[col] == "match"
                tp += pred and g
                fp += pred and not g
                fn += (not pred) and g
        return _f1(tp, fp, fn)

    vals: Dict[str, Any] = {}
    table = []
    for bench, bt in _RETURN_BENCHMARKS.items():
        for proto, pt in _RETURN_PROTOCOLS.items():
            full = by[(bench, proto, "FULL")]
            for cond in ("NOPEER", "ANSONLY"):
                other = by[(bench, proto, cond)]
                keep = [u for u in units[bench] if full.get(u) and other.get(u)]
                rec = {}
                for col, name in _RETURN_RULES:
                    e = 100 * (level(bench, full, keep, col) - level(bench, other, keep, col))
                    rec[name] = e
                    vals[f"{bt}_{pt}_{cond.lower()}_effect_{name}_pp"] = e
                # how far each majority variant moves the effect from the protocol's return rule
                table.append({name: rec[name] - rec["protocol"] for _, name in _RETURN_RULES[1:]})
    moves = [abs(r[n]) for r in table for _, n in _RETURN_RULES[1:]]
    vals["sharing_effects"] = len(table)
    vals["max_abs_move_pp"] = max(moves)
    vals["min_abs_move_pp"] = min(moves)
    vals["max_abs_move_ties_first_pp"] = max(abs(r["majority_first"]) for r in table)
    vals["min_abs_move_ties_first_pp"] = min(abs(r["majority_first"]) for r in table)
    return dict(sorted(vals.items()))
