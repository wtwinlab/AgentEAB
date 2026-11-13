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

MODELS = ("qwen2.5-7b", "llama-3-8b", "mistral-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b")
METHODS = tuple(f"single:{m}" for m in MODELS) + ("vote:majority",)

_RULE_DATASETS = {"wdc_products_E5f": "wdc", "geographic_settlements_E5f": "settlements",
                  "musicbrainz20k_e5p": "musicbrainz"}
_DEVSIZE_SEED = 20261105
_DEVSIZE_REPS = 200
_DIAG_REPS = 50
_DEVSIZE_FRACS = (0.05, 0.20, 0.30, 0.50, 1.00)
_DIAG_FRACS = (0.10, 0.50, 1.00)

_MENU = ("no_peer", "peer_decision", "decision_plus_fields", "full_explanation")
_MENU_ORDER = {a: i for i, a in enumerate(sorted(_MENU))}
_RECIPIENTS = {"gemma-4-26b": "gemma", "gpt-oss-120b": "gptoss", "qwen3-32b": "qwen3"}

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
    v = [x for x in vals if x]
    if not v:
        return None
    c = collections.Counter(v)
    t = max(c.values())
    w = [k for k, n in c.items() if n == t]
    return w[0] if len(w) == 1 else None


def _components(nodes: Sequence[str], edges: Iterable[Tuple[str, str]]) -> List[List[str]]:
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
    n = t = 0
    for mem in groups:
        k = len(mem)
        n += k * (k - 1) // 2
        t += sum(v * (v - 1) // 2 for v in collections.Counter(ent[m] for m in mem).values())
    return n, t


def _gold_pairs_total(nodes: Iterable[str], ent: Mapping[str, str]) -> int:
    return sum(v * (v - 1) // 2 for v in collections.Counter(ent[n] for n in nodes).values())


def _three_levels(pairs, gold_of, links, nodes, ent, n_gold) -> Dict[str, Any]:
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
    lv = {}
    for m in METHODS:
        if m == "vote:majority":
            dec = {(r["left"], r["right"]): _modal([r[x] for x in METHODS[:-1]]) for r in rows}
        else:
            dec = {(r["left"], r["right"]): (r[m] or None) for r in rows}
        links = {p for p, d in dec.items() if d == "match"}
        lv[m] = _three_levels(pairs, gold_of, links, nodes, ent, G)
    return lv


@dataclass
class Levels:
    pair_f1: float
    direct_f1: float
    group_f1: float
    groups: List[List[str]]


def connected_components(records: Sequence[str], links: Iterable[Tuple[str, str]]) -> List[List[str]]:
    return _components(list(records), list(links))


def relations(groups: Iterable[Sequence[str]]) -> set:
    out = set()
    for mem in groups:
        ms = sorted(mem, key=int)
        out.update((ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms)))
    return out


def evaluation_levels(pairs: Sequence[Tuple[str, str]], gold: Mapping[Tuple[str, str], bool],
                      links: Iterable[Tuple[str, str]], records: Sequence[str],
                      entity: Mapping[str, str]) -> Levels:
    n_gold = _gold_pairs_total(list(records), entity)
    r = _three_levels(list(pairs), dict(gold), set(links), list(records), entity, n_gold)
    return Levels(r["A"], r["B"], r["C"], r["groups"])


@dataclass
class SettlementsJudgments:
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
    inp = _dir(MERGE, inputs_dir)
    recs = _rd(inp, "settlements_records.csv")
    rows = _rd(inp, "settlements_pairs.csv")
    return SettlementsJudgments([r["record"] for r in recs], {r["record"]: r["entity"] for r in recs}, rows)


@dataclass
class GroupingEffect:
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


def _merge_wdc(inp: pathlib.Path, vals: Dict[str, Any]) -> List[List[float]]:
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
    r_ = pos / G
    vals["wdc_direct_link_f1_cap"] = 2 * r_ / (1 + r_)
    lv = _method_levels(rows, pairs, gold_of, nodes, ent, G)
    return [[lv[m]["A"], lv[m]["B"], lv[m]["C"]] for m in METHODS]


def _second_collection(inp: pathlib.Path, vals: Dict[str, Any], nodes, ent) -> None:
    rows = _rd(inp, "settlements_second_collection_q32.csv")
    base = {(r["left"], r["right"]) for r in rows if r["no_peers"] == "match"}
    alt = {(r["left"], r["right"]) for r in rows if r["peer_decisions"] == "match"}
    vals["second_q32_changed_judgments"] = len(base ^ alt)
    added = relations(_components(nodes, alt)) - relations(_components(nodes, base))
    direct = {tuple(sorted(p, key=int)) for p in alt}
    vals["second_q32_added_relations"] = len(added)
    vals["second_q32_added_indirect_false"] = sum(1 for p in added if p not in direct and ent[p[0]] != ent[p[1]])


def _merge_settlements(inp: pathlib.Path, vals: Dict[str, Any]) -> List[List[float]]:
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
    lv = [r for r in _rd(inp, "config_levels.csv") if r["split"] == "test"]
    rules = [k[len("cluster_f1__"):] for k in lv[0] if k.startswith("cluster_f1__")]
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
    eff = collections.defaultdict(list)
    for c in combos:
        eff[(c["dataset"], c["config"])].append(c["d_cluster_pp"])
    vals["configurations_sign_differs_across_settings"] = sum(
        1 for v in eff.values() if min(v) < -1e-7 and max(v) > 1e-7)


def merge_validation(inputs_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    inp = _dir(MERGE, inputs_dir)
    vals: Dict[str, Any] = {}
    wdc = _merge_wdc(inp, vals)
    settlements = _merge_settlements(inp, vals)
    _merge_grouping(inp, vals)
    for i, meth in enumerate(METHODS):
        short = meth.split(":")[1]
        for j, lvl in enumerate(("pairs", "direct", "groups")):
            vals[f"fig_wdc_{short}_{lvl}"] = wdc[i][j]
            vals[f"fig_settlements_{short}_{lvl}"] = settlements[i][j]
    return vals


def _menu_and_regret(inp: pathlib.Path, vals: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, float]]:
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
    ds = "geographic_settlements_E5f"
    rows = _rd(inp, "settlements_dev_groups.csv")
    cols = [c for c in rows[0] if "||" in c]
    cells = list(dict.fromkeys(c.split("||")[0] for c in cols))
    keys = list(dict.fromkeys(c.split("||")[1] for c in cols))
    ent_ids = sorted({r["entity"] for r in rows})
    eidx = {e: i for i, e in enumerate(ent_ids)}
    rec_e = np.array([eidx[r["entity"]] for r in rows])
    size = np.bincount(rec_e, minlength=len(ent_ids))
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
    inp = _dir(RULES, inputs_dir)
    vals: Dict[str, Any] = {}
    test = _menu_and_regret(inp, vals)
    _timing_and_sizes(inp, vals)
    if development_size:
        _development_size(inp, vals, test)
    return vals


def _prf(rows: Sequence[Mapping[str, str]]) -> Dict[str, float]:
    tp = sum(1 for r in rows if r["pred"] == "match" and r["gold"] == "match")
    fp = sum(1 for r in rows if r["pred"] == "match" and r["gold"] != "match")
    fn = sum(1 for r in rows if r["pred"] != "match" and r["gold"] == "match")
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(f1=100 * (2 * p * rc / (p + rc) if p + rc else 0.0), recall=100 * rc,
                tp=tp, fp=fp, fn=fn, n=len(rows))


def _matching(inp: pathlib.Path, name: str, V: Dict[str, Any]):
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
    inp = _dir(CONTENT, inputs_dir)
    vals: Dict[str, Any] = {}
    ag = _matching(inp, "amazon_google", vals)
    _matching(inp, "dblp_acm", vals)
    _amazon_google_qwen3(ag, vals)
    vals["shared_content_test_pairs"] = vals["da_qwen3_no_peer_pairs"] + vals["ag_qwen3_no_peer_pairs"]
    return vals


def return_rule_effects(inputs_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
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
                table.append({name: rec[name] - rec["protocol"] for _, name in _RETURN_RULES[1:]})
    moves = [abs(r[n]) for r in table for _, n in _RETURN_RULES[1:]]
    vals["sharing_effects"] = len(table)
    vals["max_abs_move_pp"] = max(moves)
    vals["min_abs_move_pp"] = min(moves)
    vals["max_abs_move_ties_first_pp"] = max(abs(r["majority_first"]) for r in table)
    vals["min_abs_move_ties_first_pp"] = min(abs(r["majority_first"]) for r in table)
    return dict(sorted(vals.items()))
