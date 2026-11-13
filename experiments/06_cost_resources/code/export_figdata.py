r"""V.3: tidy plotting data, one long-format CSV per panel.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/export_figdata.py

The figures themselves are the user's; this writes the data they need in a form that does not
require reading the analysis JSON's nesting. Every row carries its own denominator, its clustering
unit, and whether the point is a per-call-unit cost or a token cost, because those three are the
things a plot most easily hides.

Panels, matching the checklist's V.3 list
-----------------------------------------
  frontier.csv       quality against realised cost, per dataset, with each method's own budget knob
  content.csv        sharing conditions: levels and paired contrasts, matching and SQL side by side
  transitions.csv    what sharing changed, as counts of before/after decision pairs
  budget.csv         the three native budget controls, each on its own axis
  pipeline.csv       pairwise decision quality against post-merge linkage quality
  amortisation.csv   LOTUS reference and proxy token split as the batch target moves
  flips.csv          direction of change under sharing: wrong-to-right against right-to-wrong
  families.csv       the five pre-specified contrast families with their Holm thresholds

Token costs and per-call costs are never added together. A LOTUS point and a ThriftLLM point sit in
different `cost_kind` rows for exactly that reason.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(BASE, "analysis")
FIG = os.path.join(BASE, "figdata")


def J(name: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write(name: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"  {name:20s} (no rows; source analysis missing)")
        return
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    p = os.path.join(FIG, name)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  {name:20s} {len(rows):5d} rows, {len(cols)} columns")


def family(method: str) -> str:
    for pre, fam in (("single:", "single model"), ("vote:", "ensemble vote"),
                     ("thriftllm", "ThriftLLM"), ("lotus:", "LOTUS"),
                     ("select:", "selected subset")):
        if method.startswith(pre):
            return fam
    return "other"


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    print(f"  writing to {FIG}")

    # ---- frontier
    bf = J("budget_frontier.json")
    rows: List[Dict[str, Any]] = []
    if bf:
        for dn, d in bf["datasets"].items():
            front = {p["method"] for p in d.get("pareto_frontier", [])}
            best = (d.get("best") or {}).get("method")
            cheap = (d.get("cheapest_within_2pp_of_best") or {}).get("method")
            for p in d["points"]:
                rows.append({
                    "dataset": dn, "method": p["method"], "family": family(p["method"]),
                    "budget_control": p.get("control"), "f1": p.get("f1"),
                    "ci_lo": p.get("lo"), "ci_hi": p.get("hi"),
                    "cost_kind": "per_call_units" if p.get("cost_units") is not None else "tokens",
                    "cost_units": p.get("cost_units"), "cost_tokens": p.get("cost_tokens"),
                    "x_ref": p.get("x_ref"), "calls": p.get("calls"),
                    "reference_cost_units": d["reference_cost_units"],
                    "on_pareto_frontier": p["method"] in front,
                    "is_best": p["method"] == best,
                    "is_cheapest_within_2pp": p["method"] == cheap})
    write("frontier.csv", rows)

    # ---- content: sharing levels and contrasts, both workloads
    sh, sq = J("em_sharing.json"), J("sql_results.json")
    rows = []
    if sh:
        for dn, byrec in sh["levels"].items():
            for rec, byc in byrec.items():
                for c, v in byc.items():
                    cst = ((sh.get("cost") or {}).get(dn, {}).get(rec, {}).get(c) or {})
                    rows.append({"workload": "matching", "dataset": dn, "recipient": rec,
                                 "condition": c, "row_kind": "level", "metric": "f1",
                                 "value": v.get("f1"), "ci_lo": None, "ci_hi": None,
                                 "p": None, "n": v.get("n"), "n_clusters": None,
                                 "cluster_unit": "entity-connected block",
                                 "precision": v.get("precision"), "recall": v.get("recall"),
                                 "abstention_rate": v.get("abstention_rate"),
                                 "cost_units": cst.get("total_cost_units"),
                                 "mean_prompt_tokens": v.get("mean_input_tokens")})
        for kind, key in (("contrast", "contrasts"),
                          ("contrast_stale_excluded", "contrasts_stale_excluded")):
            for dn, byrec in (sh.get(key) or {}).items():
                for rec, byc in byrec.items():
                    for c, v in byc.items():
                        rows.append({"workload": "matching", "dataset": dn, "recipient": rec,
                                     "condition": c, "row_kind": kind, "metric": "d_f1",
                                     "value": v.get("diff"), "ci_lo": v.get("lo"),
                                     "ci_hi": v.get("hi"), "p": v.get("p"),
                                     "n": v.get("n_pairs"), "n_clusters": v.get("n_blocks"),
                                     "cluster_unit": "entity-connected block",
                                     "n_pairs_dropped": v.get("n_pairs_dropped")})
    if sq:
        for rec, byc in (sq.get("sharing") or {}).items():
            for c, v in byc.items():
                rows.append({"workload": "text-to-SQL", "dataset": "BIRD dev", "recipient": rec,
                             "condition": c, "row_kind": "level", "metric": "execution_accuracy",
                             "value": v.get("mean"), "ci_lo": v.get("lo"), "ci_hi": v.get("hi"),
                             "p": None, "n": v.get("n"), "n_clusters": v.get("n_databases"),
                             "cluster_unit": "database",
                             "cost_units": v.get("cost_units"),
                             "mean_prompt_tokens": v.get("mean_prompt_tokens")})
        for rec, byc in (sq.get("contrasts") or {}).items():
            for c, v in byc.items():
                rows.append({"workload": "text-to-SQL", "dataset": "BIRD dev", "recipient": rec,
                             "condition": c, "row_kind": "contrast",
                             "metric": "d_execution_accuracy", "value": v.get("mean"),
                             "ci_lo": v.get("lo"), "ci_hi": v.get("hi"), "p": v.get("p"),
                             "n": v.get("n"), "n_clusters": v.get("n_databases"),
                             "cluster_unit": "database"})
        for rec, byc in (sq.get("db_disjoint") or {}).items():
            if rec == "split":
                continue
            for c, v in byc.items():
                for half in ("dev", "test"):
                    rows.append({"workload": "text-to-SQL", "dataset": f"BIRD {half} schemas",
                                 "recipient": rec, "condition": c, "row_kind": "db_disjoint",
                                 "metric": "execution_accuracy", "value": v.get(f"{half}_ex"),
                                 "n": v.get(f"n_{half}"), "cluster_unit": "database"})
    write("content.csv", rows)

    # ---- transitions
    rows = []
    if sh:
        for dn, byrec in (sh.get("transitions") or {}).items():
            for rec, byc in byrec.items():
                for c, v in byc.items():
                    if not isinstance(v, dict):
                        continue
                    for k, n in v.items():
                        rows.append({"workload": "matching", "dataset": dn, "recipient": rec,
                                     "condition": c, "transition": k, "count": n})
    write("transitions.csv", rows)

    # ---- budget: each control on its own axis
    rows = []
    cmp_ = J("em_method_comparison.json")
    sel = J("em_select_agg.json")
    sw = J("lotus_target_sweep.json")
    if cmp_:
        for dn, d in cmp_["datasets"].items():
            for m, v in d["methods"].items():
                if not m.startswith("thriftllm"):
                    continue
                rows.append({"dataset": dn, "control": "ThriftLLM per-query cost cap B",
                             "setting": m.split("=")[-1], "metric": "f1", "value": v["f1"],
                             "ci_lo": v["lo"], "ci_hi": v["hi"], "cost_kind": "per_call_units",
                             "cost": v["cost_units"]})
    if sel:
        for dn, byk in (sel.get("selection") or {}).items():
            for k, rules in byk.items():
                for rule, v in rules.items():
                    rows.append({"dataset": dn, "control": "ensemble model count k",
                                 "setting": f"k={k}:{rule}", "metric": "f1",
                                 "value": v.get("f1"), "ci_lo": v.get("lo"), "ci_hi": v.get("hi"),
                                 "cost_kind": "per_call_units", "cost": v.get("cost_units"),
                                 "models": "|".join(v.get("models") or [])})
    if sw:
        for dn, byt in sw.items():
            for t, v in byt.items():
                rows.append({"dataset": dn, "control": "LOTUS batch recall/precision target",
                             "setting": t, "metric": "f1", "value": v["f1"],
                             "cost_kind": "tokens", "cost": v["total"],
                             "ratio_vs_direct": v["ratio_vs_direct"],
                             "reference_fidelity": v.get("reference_fidelity")})
    write("budget.csv", rows)

    # ---- pipeline transfer
    rows = []
    pl = J("pipeline_results.json")
    if pl:
        for dn, d in pl["datasets"].items():
            pw = ((cmp_ or {}).get("datasets", {}).get(dn, {}).get("methods") or {})
            for m, v in d["methods"].items():
                key = "vote:all6" if m == "vote:majority" else m
                rows.append({"dataset": dn, "method": m, "family": family(m),
                             "has_cluster_gold": d["has_cluster_gold"],
                             "pair_f1": (pw.get(key) or {}).get("f1"),
                             "asserted_links": v.get("asserted_links"),
                             "merged_clusters": v.get("merged_clusters"),
                             "largest_cluster": v.get("largest_cluster"),
                             "implied_intra_pairs": v.get("implied_intra_pairs"),
                             "closure_amplification": v.get("closure_amplification"),
                             "cluster_precision": v.get("cluster_precision"),
                             "cluster_recall": v.get("cluster_recall"),
                             "cluster_f1": v.get("cluster_f1")})
    write("pipeline.csv", rows)

    # ---- LOTUS amortisation
    rows = []
    lo = J("lotus_results.json")
    if lo:
        for dn, d in lo.items():
            for mode in ("direct", "cascade"):
                v = d.get(mode) or {}
                rows.append({"dataset": dn, "mode": mode, "target": (
                    (v.get("cascade_args") or {}).get("recall_target") if mode == "cascade"
                    else None),
                    "f1": v.get("f1"), "ci_lo": v.get("lo"), "ci_hi": v.get("hi"),
                    "ref_tokens": v.get("ref_tokens"), "proxy_tokens": v.get("proxy_tokens"),
                    "total_tokens": v.get("total_tokens"), "wall_s": v.get("wall_s"),
                    "n": v.get("n"), "n_blocks": v.get("n_blocks"),
                    "reference_fidelity": (d.get("reference_fidelity") or {}).get("agreement")
                                           if mode == "cascade" else None})
            if sw and dn in sw:
                for t, v in sw[dn].items():
                    rows.append({"dataset": dn, "mode": "cascade_sweep", "target": t,
                                 "f1": v["f1"], "ref_tokens": v["ref_tokens"],
                                 "proxy_tokens": v["proxy_tokens"], "total_tokens": v["total"],
                                 "n": v["n"], "reference_fidelity": v.get("reference_fidelity")})
    write("amortisation.csv", rows)

    # ---- flip direction
    rows = []
    orl = J("offline_reuse.json")
    if orl:
        q = orl["questions"]
        for rec, byc in (q.get("sharing_flips") or {}).items():
            for c, v in byc.items():
                rows.append({"workload": "matching", "recipient": rec, "condition": c, **v})
        for rec, byc in (q.get("sql_text_change_vs_accuracy") or {}).items():
            if not isinstance(byc, dict):
                continue
            for c, v in byc.items():
                if isinstance(v, dict):
                    rows.append({"workload": "text-to-SQL", "recipient": rec,
                                 "condition": c, **v})
    write("flips.csv", rows)

    # ---- the five pre-specified families, with what survives their own correction
    rows = []
    mp = J("multiplicity.json")
    if mp:
        for dn, d in mp["datasets"].items():
            for fid, fam in d["families"].items():
                h = fam.get("holm")
                if not h:
                    rows.append({"dataset": dn, "family": fid, "family_name": fam.get("name"),
                                 "test": None, "unavailable": fam.get("unavailable")})
                    continue
                for k, v in h["tests"].items():
                    t = fam["tests"].get(k) or {}
                    rows.append({
                        "dataset": dn, "family": fid, "family_name": fam["name"], "test": k,
                        "effect": t.get("diff", t.get("f1_score_based")),
                        "ci_lo": t.get("lo"), "ci_hi": t.get("hi"),
                        "p": v["p"], "holm_threshold": v["threshold"],
                        "survives_holm": v["survives"], "family_size": h["family_size"],
                        "n_blocks": t.get("n_blocks"),
                        "validation_pick": d["model_chosen_on_validation"],
                        "test_maximal_model": d["model_best_on_test"]})
        sfam = mp.get("sql")
        if sfam:
            for k, v in sfam["holm"]["tests"].items():
                t = sfam["tests"][k]
                rows.append({"dataset": "BIRD dev", "family": "sql",
                             "family_name": sfam["name"], "test": k, "effect": t["mean"],
                             "ci_lo": t["lo"], "ci_hi": t["hi"], "p": v["p"],
                             "holm_threshold": v["threshold"], "survives_holm": v["survives"],
                             "family_size": sfam["holm"]["family_size"],
                             "n_blocks": t.get("n_databases")})
    write("families.csv", rows)

    spec = {
        "written_by": "code/export_figdata.py",
        "confirmatory_vs_descriptive": ("families.csv is the confirmatory record; the maxima "
                                        "labelled in frontier.csv are descriptive and were chosen "
                                        "after seeing test F1"),
        "cost_axes": ("per_call_units and tokens are different axes and are never summed; a LOTUS "
                      "point and a ThriftLLM point appear in separate cost_kind rows"),
        "clustering": ("matching resamples entity-connected blocks; SQL resamples databases. "
                       "Recipients and repeat decodes are not independent units"),
        "intervals": "95% cluster bootstrap, 4000 draws, seed 20261013",
        "files": sorted(os.listdir(FIG))}
    with open(os.path.join(FIG, "README.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, sort_keys=True)
    print(f"\n  wrote {FIG}/README.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
