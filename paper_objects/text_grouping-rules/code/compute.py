"""Grouping rules chosen on development data from fixed pair judgments (Section sec:grouping-rules).

    python3 code/compute.py --out DIR      # -> DIR/values.json, DIR/development_size.csv

Inputs (inputs/):
  rule_cells.csv              per dataset x configuration x grouping setting: development and test
                              cluster F1 (24 configurations per dataset, 15 settings)
  selection.csv               per configuration: the setting selected by development cluster F1,
                              its test cluster F1, and the gap to the best setting on test (regret)
  grouping.json               per configuration and setting on the development side: logged
                              seconds of the grouping algorithm; development entity and record counts
  settlements_dev_groups.csv  Settlements development records: reference entity and group id under
                              every configuration x setting, computed on the complete development
                              graph

The development-size analysis scores development cluster F1 only over the records of a seeded
sample of development entities, keeps the groups of the complete development graph, selects the
setting with the highest score (ties to the setting listed first in the menu order) and reports the
test regret of that setting, averaged over 200 samples and the 24 configurations.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random
import statistics

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
SHORT = {"wdc_products_E5f": "wdc", "geographic_settlements_E5f": "settlements",
         "musicbrainz20k_e5p": "musicbrainz"}
SEED = 20261105
REPS = 200          # development-size samples per fraction
DIAG_REPS = 50      # samples for the precision / recall diagnostic
FRACS = (0.05, 0.20, 0.30, 0.50, 1.00)
DIAG_FRACS = (0.10, 0.50, 1.00)


def rd(name):
    with open(INP / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def menu_and_regret(vals):
    rc = rd("rule_cells.csv")
    sel = rd("selection.csv")
    rules = list(dict.fromkeys(r["rule"] for r in rc))
    vals["menu_settings"] = len(rules)
    vals["menu_families"] = len({r.split("@")[0] for r in rules})
    test = collections.defaultdict(dict)
    for r in rc:
        test[(r["dataset"], r["cell"])][r["rule"]] = float(r["test_cluster_f1"])
    for ds, t in SHORT.items():
        cells = [k for k in test if k[0] == ds]
        vals[f"{t}_configurations"] = len(cells)
        s = [r for r in sel if r["dataset"] == ds]
        reg = []
        for r in s:
            best = max(test[(ds, r["cell"])].values())
            g = 100 * (best - test[(ds, r["cell"])][r["selected"]])
            assert abs(g - float(r["regret_pp"])) < 1e-9
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
        vals[f"{t}_selected_mean_regret_pp"] for t in SHORT.values())
    return test


def timing_and_sizes(vals):
    g = json.loads((INP / "grouping.json").read_text(encoding="utf-8"))
    allt = []
    for ds, t in SHORT.items():
        d = g["domains"][ds]
        secs = [sum(x["cpu_seconds"] for x in c["rules"].values()) for c in d["dev"].values()]
        vals[f"{t}_dev_seconds_all_settings_min"] = min(secs)
        vals[f"{t}_dev_seconds_all_settings_max"] = max(secs)
        vals[f"{t}_dev_entities"] = d["dev_entities"]
        vals[f"{t}_dev_records"] = d["dev_records"]
        allt += secs
    vals["dev_seconds_all_settings_min"] = min(allt)
    vals["dev_seconds_all_settings_max"] = max(allt)


def development_size(vals, test, out):
    ds = "geographic_settlements_E5f"
    rows = rd("settlements_dev_groups.csv")
    cols = [c for c in rows[0] if "||" in c]
    cells = list(dict.fromkeys(c.split("||")[0] for c in cols))
    keys = list(dict.fromkeys(c.split("||")[1] for c in cols))
    ent_ids = sorted({r["entity"] for r in rows})
    eidx = {e: i for i, e in enumerate(ent_ids)}
    rec_e = np.array([eidx[r["entity"]] for r in rows])
    size = np.bincount(rec_e, minlength=len(ent_ids))
    # predicted relations of every column, as pairs of reference-entity indices
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
        rng = random.Random(f"{SEED}|{ds}|{frac}|{rep}")
        pick = set(ent_ids) if k == len(ent_ids) else set(rng.sample(ent_ids, k))
        return k, {eidx[e] for e in pick}

    table = []
    for frac in FRACS:
        regs = []
        for rep in (range(1) if frac >= 1.0 else range(REPS)):
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
        m = statistics.mean(regs)
        table.append(dict(fraction=frac, development_entities=k, samples=REPS if frac < 1 else 1,
                          mean_test_regret_pp=m))
        vals[f"settlements_devsize_{k}_entities_mean_regret_pp"] = m
        vals[f"settlements_devsize_frac{frac:.2f}_entities"] = k
    with open(out / "development_size.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    # what entity sampling hides: connected components scored on sampled entities
    cc = [j * len(keys) + order["connected_components"] for j in range(len(cells))]
    for frac in DIAG_FRACS:
        fp, rec = [], []
        for rep in (range(1) if frac >= 1.0 else range(DIAG_REPS)):
            _, pick = draw(frac, rep)
            N, T, G = scores(pick)
            for col in cc:
                fp.append(int(N[col]) - int(T[col]))
                rec.append(int(T[col]) / G if G else 0.0)
        vals[f"settlements_cc_false_pairs_frac{frac:.2f}"] = statistics.mean(fp)
        vals[f"settlements_cc_recall_frac{frac:.2f}"] = statistics.mean(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    vals = {}
    test = menu_and_regret(vals)
    timing_and_sizes(vals)
    development_size(vals, test, out)
    (out / "values.json").write_text(json.dumps(vals, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
