"""Pair judgments and the groups they form: Fig. merge-validation and the numbers that discuss it.

    python3 code/compute.py --out DIR      # -> DIR/values.json, DIR/plot_data.json, DIR/*.csv

Inputs (inputs/):
  settlements_records.csv  Settlements test records and their reference entities
  settlements_pairs.csv    Settlements candidate pairs: reference label, each model's decision
                           (modal over its repeats), each recipient x sharing-condition decision
  wdc_pairs.csv            WDC test pairs: records, reference entities, each model's decision
  config_levels.csv        per matching configuration: pair F1 and cluster F1 under 15 grouping
                           settings (entity-disjoint test and development sides)

Three evaluation levels (Fig. (a,b)): F1 on the supplied candidate pairs, F1 of the direct links
over all record pairs, and F1 of the relations implied by connected components over all record
pairs. A decision is a match only if it reads "match"; no decision counts as non-match.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
MODELS = ["qwen2.5-7b", "llama-3-8b", "mistral-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b"]
METHODS = [f"single:{m}" for m in MODELS] + ["vote:majority"]
FIG_ORDER = ["single:qwen2.5-7b", "single:llama-3-8b", "single:mistral-7b", "single:qwen2.5-14b",
             "single:qwen2.5-32b", "single:qwen2.5-72b", "vote:majority"]


def rd(name):
    with open(INP / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def modal(vals):
    v = [x for x in vals if x]
    if not v:
        return None
    c = collections.Counter(v)
    t = max(c.values())
    w = [k for k, n in c.items() if n == t]
    return w[0] if len(w) == 1 else None


def components(nodes, edges):
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


def intra(groups, ent):
    """Predicted and true relations implied by a grouping."""
    n = t = 0
    for mem in groups:
        k = len(mem)
        n += k * (k - 1) // 2
        t += sum(v * (v - 1) // 2 for v in collections.Counter(ent[m] for m in mem).values())
    return n, t


def three_levels(pairs, gold_of, links, nodes, ent, n_gold):
    """F1 on candidate pairs, on direct links over all pairs, on implied relations over all pairs."""
    tp = sum(1 for p in pairs if p in links and gold_of[p])
    fp = sum(1 for p in pairs if p in links and not gold_of[p])
    fn = sum(1 for p in pairs if p not in links and gold_of[p])
    a = f1(tp, fp, fn)
    b = f1(tp, len(links) - tp, n_gold - tp)
    groups = components(nodes, links)
    n, t = intra(groups, ent)
    c = f1(t, n - t, n_gold - t)
    return dict(A=a, B=b, C=c, tpA=tp, fpA=fp, tpC=t, fpC=n - t, groups=groups)


def bridges(groups, new_links, ent, F):
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


def second_collection(vals, nodes, ent):
    """The same contrast (Qwen2.5-32B, peer decisions minus no peers) in the second judgment collection
    on the same Settlements records and candidate pairs: changed judgments and the relations they add
    under connected components."""
    rows = rd("settlements_second_collection_q32.csv")
    base = {(r["left"], r["right"]) for r in rows if r["no_peers"] == "match"}
    alt = {(r["left"], r["right"]) for r in rows if r["peer_decisions"] == "match"}
    vals["second_q32_changed_judgments"] = len(base ^ alt)

    def rel(links):
        s = set()
        for mem in components(nodes, links):
            ms = sorted(mem, key=int)
            s.update((ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms)))
        return s
    added = rel(alt) - rel(base)
    direct = {tuple(sorted(p, key=int)) for p in alt}
    vals["second_q32_added_relations"] = len(added)
    vals["second_q32_added_indirect_false"] = sum(1 for p in added if p not in direct and ent[p[0]] != ent[p[1]])


def gold_pairs_total(nodes, ent):
    return sum(v * (v - 1) // 2 for v in collections.Counter(ent[n] for n in nodes).values())


# ------------------------------------------------------------------ Settlements, full population
def settlements(vals, plot):
    recs = rd("settlements_records.csv")
    nodes = [r["record"] for r in recs]
    ent = {r["record"]: r["entity"] for r in recs}
    rows = rd("settlements_pairs.csv")
    pairs = [(r["left"], r["right"]) for r in rows]
    gold_of = {(r["left"], r["right"]): r["gold"] == "1" for r in rows}
    G = gold_pairs_total(nodes, ent)
    vals["settlements_test_records"] = len(nodes)
    vals["settlements_candidate_pairs"] = len(pairs)
    vals["settlements_candidate_coverage_pct"] = 100 * sum(gold_of.values()) / G
    # (b): each model alone and the six-model vote
    lv = {}
    for m in METHODS:
        if m == "vote:majority":
            dec = {(r["left"], r["right"]): modal([r[x] for x in METHODS[:-1]]) for r in rows}
        else:
            dec = {(r["left"], r["right"]): (r[m] or None) for r in rows}
        links = {p for p, d in dec.items() if d == "match"}
        lv[m] = three_levels(pairs, gold_of, links, nodes, ent, G)
    plot["settlements"] = [[lv[m]["A"], lv[m]["B"], lv[m]["C"]] for m in FIG_ORDER]
    # (c): sharing, on the pairs that carry all six recipient x condition decisions
    common = [r for r in rows if r["share_complete"] == "1"]
    cpairs = [(r["left"], r["right"]) for r in common]
    vals["settlements_sharing_pairs"] = len(cpairs)
    sh = {}
    for rc in ("qwen2.5-7b", "qwen2.5-32b"):
        for c in ("no_peers", "candidate_only", "full"):
            col = f"share:{rc}:{c}"
            links = {(r["left"], r["right"]) for r in common if r[col] == "match"}
            sh[(rc, c)] = dict(links=links, dec={(r["left"], r["right"]): r[col] for r in common},
                               **three_levels(cpairs, gold_of, links, nodes, ent, G))
    plot["sharing"] = []
    for rc in ("qwen2.5-7b", "qwen2.5-32b"):
        for c in ("candidate_only", "full"):
            dp = 100 * (sh[(rc, c)]["A"] - sh[(rc, "no_peers")]["A"])
            dc = 100 * (sh[(rc, c)]["C"] - sh[(rc, "no_peers")]["C"])
            plot["sharing"].append(dict(recipient=rc, form=c, pair_pp=dp, cluster_pp=dc))
            tag = f"{rc.replace('qwen2.5-', 'q')}_{'decisions' if c == 'candidate_only' else 'full'}"
            vals[f"share_{tag}_pair_pp"] = dp
            vals[f"share_{tag}_cluster_pp"] = dc
    # Qwen2.5-32B, shared decisions vs no peers: which judgments change and what grouping adds
    A, B = sh[("qwen2.5-32b", "candidate_only")], sh[("qwen2.5-32b", "no_peers")]
    n2m = [p for p in cpairs if A["dec"][p] == "match" and B["dec"][p] != "match"]
    m2n = [p for p in cpairs if A["dec"][p] != "match" and B["dec"][p] == "match"]
    vals["q32_changed_judgments"] = len(n2m) + len(m2n)
    vals["q32_nonmatch_to_match"] = len(n2m)
    vals["q32_match_to_nonmatch"] = len(m2n)
    vals["q32_changed_correct"] = sum(1 for p in n2m if gold_of[p]) + sum(1 for p in m2n if not gold_of[p])
    vals["q32_changed_incorrect"] = sum(1 for p in n2m if not gold_of[p]) + sum(1 for p in m2n if gold_of[p])

    def rel(groups):
        s = set()
        for mem in groups:
            ms = sorted(mem, key=int)
            s.update((ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms)))
        return s
    ra, rb = rel(A["groups"]), rel(B["groups"])
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
                               false_relations_in_group=sum(1 for p in rel([mem])
                                                            if ent[p[0]] != ent[p[1]])))
    gid_a = {n: i for i, mem in enumerate(A["groups"]) for n in mem}
    splits = sum(1 for mem in B["groups"] if len({gid_a[n] for n in mem}) > 1)
    vals["q32_merges"] = len(merges)
    vals["q32_splits"] = splits
    big = max(merges, key=lambda m: m["size"])
    vals["q32_largest_merge_size"] = big["size"]
    vals["q32_largest_merge_groups_before"] = big["groups_before"]
    vals["q32_largest_merge_entities"] = big["entities"]
    vals["q32_largest_merge_added_false_indirect"] = big["added_false_indirect"]
    vals["q32_largest_merge_false_relations"] = big["false_relations_in_group"]
    second_collection(vals, nodes, ent)
    # each changed link added alone to the no-peer groups: does it join two groups, and what does it add?
    br = bridges(B["groups"], n2m, ent, B["C"])
    vals["q32_bridging_links"] = len(br)
    vals["q32_bridging_links_incorrect"] = sum(1 for x in br if not x["correct"])
    vals["q32_incorrect_bridges_below_condition"] = sum(1 for x in br if not x["correct"] and x["harmful"])
    vals["q32_incorrect_bridges_above_condition"] = sum(1 for x in br if not x["correct"] and not x["harmful"])
    inc = [x for x in br if not x["correct"]]
    vals["q32_incorrect_bridges_alone_relations"] = sum(x["b"] for x in inc)
    vals["q32_incorrect_bridges_alone_false"] = sum(x["b"] - x["t"] for x in inc)
    vals["q32_bridges_alone_relations"] = sum(x["b"] for x in br)
    vals["q32_largest_single_bridge_relations"] = max(x["b"] for x in br)
    with open(OUT / "settlements_q32_merges.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(merges[0]))
        w.writeheader()
        w.writerows(sorted(merges, key=lambda m: -m["size"]))
    return lv


# ------------------------------------------------------------------ WDC
def wdc(vals, plot):
    rows = rd("wdc_pairs.csv")
    ent = {}
    for r in rows:
        ent[r["left"]] = r["left_entity"]
        ent[r["right"]] = r["right_entity"]
    nodes = sorted(ent)
    pairs = [(r["left"], r["right"]) for r in rows]
    gold_of = {(r["left"], r["right"]): r["gold"] == "1" for r in rows}
    G = gold_pairs_total(nodes, ent)
    pos = sum(gold_of.values())
    vals["wdc_true_matches"] = G
    vals["wdc_true_matches_in_candidates"] = pos
    r_ = pos / G
    vals["wdc_direct_link_f1_cap"] = 2 * r_ / (1 + r_)
    lv = {}
    for m in METHODS:
        if m == "vote:majority":
            dec = {(r["left"], r["right"]): modal([r[x] for x in METHODS[:-1]]) for r in rows}
        else:
            dec = {(r["left"], r["right"]): (r[m] or None) for r in rows}
        links = {p for p, d in dec.items() if d == "match"}
        lv[m] = three_levels(pairs, gold_of, links, nodes, ent, G)
    plot["wdc"] = [[lv[m]["A"], lv[m]["B"], lv[m]["C"]] for m in FIG_ORDER]


# ------------------------------------------------------------------ grouping matrix (test side)
def grouping(vals):
    lv = [r for r in rd("config_levels.csv") if r["split"] == "test"]
    rules = [k[len("cluster_f1__"):] for k in lv[0] if k.startswith("cluster_f1__")]
    # center clustering vs connected components, one configuration per recipient x condition
    # (first random-peer draw for the two random-peer conditions)
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
    # sharing minus no peers, grouping setting held fixed
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
    rev = lambda c, f: c[f] in ((1, -1), (-1, 1))
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
    with open(OUT / "sharing_grouping_combinations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "config", "rule", "family", "d_pair_pp",
                                          "d_cluster_pp", "exact", "band"])
        w.writeheader()
        for c in combos:
            w.writerow({**c, "exact": "%d/%d" % c["exact"], "band": "%d/%d" % c["band"]})


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    OUT = pathlib.Path(ap.parse_args().out)
    OUT.mkdir(parents=True, exist_ok=True)
    vals, plot = {}, {"methods": FIG_ORDER}
    wdc(vals, plot)
    settlements(vals, plot)
    grouping(vals)
    for i, m in enumerate(FIG_ORDER):
        short = m.split(":")[1]
        for j, lvl in enumerate(("pairs", "direct", "groups")):
            vals[f"fig_wdc_{short}_{lvl}"] = plot["wdc"][i][j]
            vals[f"fig_settlements_{short}_{lvl}"] = plot["settlements"][i][j]
    (OUT / "plot_data.json").write_text(json.dumps(plot, indent=1) + "\n", encoding="utf-8")
    (OUT / "values.json").write_text(json.dumps(vals, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
