r"""Supplementary Table tab:s-reader-sensitivity (sec:s-reader-sensitivity), the extractor audits
(sec:s-reader-audit, sec:evaluation-checks), and the extractor-robustness sentences of Section VI-C.

    python3 code/compute.py --out output

Reads only ../inputs. Writes <out>/values.json, <out>/reader_sensitivity.json (every estimate with both 95%
intervals and the number of votes each reading changes) and <out>/audits.json.
"""
import argparse
import collections
import json
import math
import pathlib
import sys

import numpy as np
from scipy.stats import beta

HERE = pathlib.Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402

INP = HERE.parent / "inputs"
READOUTS = {"DuMAD_2cell": E.DU, "cell_du_mad|llama3-8b": [E.DU[0]], "cell_du_mad|qwen2.5-7b": [E.DU[1]],
            "MADM2_2cell": E.M2, "cell_madm2_objective|llama3-8b": [E.M2[0]],
            "cell_madm2_objective|qwen2.5-7b": [E.M2[1]], "4cell": E.C4}
Z = 1.959964


def wilson(k, n):
    p = k / n
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return round(100 * (c - h), 1), round(100 * min(1.0, c + h), 1)


def clopper_pearson(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return round(100 * float(lo), 1), round(100 * float(hi), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    R = E.Rows(INP)
    RD = E.Reader(R, INP)
    V, full = {}, {}

    def put(key, b, ci=True):
        V[key] = b["est"]
        if ci:
            V[key + "_q_lo"], V[key + "_q_hi"] = b["q_ci"]
            V[key + "_db_lo"], V[key + "_db_hi"] = b["db_ci"]
        full[key] = {k: b[k] for k in ("est", "q_ci", "db_ci", "n_questions", "detected_both") if k in b}
        if "n_votes_changed" in b:
            full[key]["n_votes_changed"] = b["n_votes_changed"]

    # ---- Table: lenient extractor, own extractor only, joint worst case (extractor's candidates P, broad set B)
    for name, cells in READOUTS.items():
        for metric in ("last", "E1"):
            h = R.boot(R.fulldev_items(cells, f"h_{metric}"))
            put(f"{name}|{metric}|lenient", h)
            put(f"{name}|{metric}|own", R.boot(R.fulldev_items(cells, f"a_{metric}")))
            dirs = [-1, 1] if h["est"] == 0 else [int(np.sign(h["est"]))]
            for var in RD.VARIANTS:
                for dr in dirs:
                    w = RD.worst_amb(cells, metric, dr, var)
                    tag = "" if len(dirs) == 1 else ("|raise" if dr < 0 else "|lower")
                    put(f"{name}|{metric}|joint{var}{tag}", w)
                    V[f"{name}|{metric}|joint{var}{tag}_n_votes"] = w["n_votes_changed"]

    # ---- ambiguity counts (Du-MAD cells, recovered replies)
    cnt = {}
    for c in E.C4:
        sr = RD.side_rows([c])
        for side, rows in sr.items():
            for rd in RD.ROUNDS:
                t = collections.Counter()
                for r in rows:
                    rec, amb, piv1 = set(r[f"_rec_{rd}"]), set(r[f"_amb_{rd}"]), set(r[f"_piv1_{rd}"])
                    t["recovered"] += len(rec)
                    t["ambiguous"] += len(amb)
                    t["pivotal1"] += len(piv1)
                    t["pivotal1_ambiguous"] += len(piv1 & amb)
                    t["runs_dependent"] += bool(piv1)
                    t["runs_dependent_all_unambiguous"] += bool(piv1) and not (piv1 & amb)
                d = dict(t)
                d["ambiguous_share_pct"] = round(100 * t["ambiguous"] / t["recovered"], 2) if t["recovered"] else None
                cnt[f"{c[0]}|{c[1]}|{side}|{rd}"] = d
    full["ambiguity_counts"] = cnt
    du = {k: v for k, v in cnt.items() if k.startswith("du_mad")}
    V["du_ambiguous_share_max_r0"] = max(v["ambiguous_share_pct"] for k, v in du.items() if k.endswith("|r0"))
    V["du_ambiguous_share_max_last"] = max(v["ambiguous_share_pct"] for k, v in du.items() if k.endswith("|last"))
    V["du_last_without_pivotal1"] = sum(v["pivotal1"] for k, v in du.items() if k.endswith("without|last"))
    V["du_last_without_pivotal1_ambiguous"] = sum(v["pivotal1_ambiguous"] for k, v in du.items() if k.endswith("without|last"))
    V["du_last_with_runs_dependent"] = sum(v["runs_dependent"] for k, v in du.items() if k.endswith("|with|last"))
    V["du_last_with_runs_dependent_all_unambiguous"] = sum(
        v["runs_dependent_all_unambiguous"] for k, v in du.items() if k.endswith("|with|last"))

    # ---- audit-rate sensitivity: single-removal dependent votes flipped at 1/50 and 12/450, 1,000 draws
    rng = np.random.default_rng(E.SEED)
    pert = {}
    for metric in ("last", "E1"):
        for pn, p in (("human_1of50", 1 / 50), ("agent_12of450", 12 / 450)):
            for directed in (False, True):
                pert[f"DuMAD_2cell|{metric}|{pn}|{'directed_shrink' if directed else 'undirected'}"] = round(
                    RD.perturb_means(E.DU, metric, p, rng, directed, 1000), 3)
    shift = {k: round(abs(v - V[f"DuMAD_2cell|{k.split('|')[1]}|lenient"]), 3) for k, v in pert.items()}
    full["audit_rate_sensitivity_mean_estimates"] = pert
    full["audit_rate_sensitivity_shift"] = shift
    assert max(shift.values()) < 0.1, shift
    V["perturb_draws"] = 1000

    # ---- first audit: human annotator and two agent judges
    A1 = INP / "audit_first"
    sk = json.loads((A1 / "sample_key.json").read_text())
    hl = {x["item_id"]: x["label"] for x in json.loads((A1 / "human_labels.json").read_text())["items"]}
    V["audit1_population"] = sk["population_total"]
    V["audit1_n"] = len(hl)
    V["audit1_yes"] = sum(v == "yes" for v in hl.values())
    V["audit1_cp_lo"], V["audit1_cp_hi"] = clopper_pearson(V["audit1_yes"], V["audit1_n"])
    key = {x["judge_id"]: x for x in json.loads((A1 / "agent_judge_key.json").read_text())["items"]}
    items = {d["id"]: d for d in map(json.loads, (A1 / "agent_judge_items.jsonl").open())}
    jo = {d["id"]: d["label"] for d in map(json.loads, (A1 / "agent_judge_claude_opus5.5_high.jsonl").open())}
    jg = {d["id"]: d["label"] for d in map(json.loads, (A1 / "agent_judge_gpt6_extra_high.jsonl").open())}
    shared = [j for j in key if key[j]["src"] == "human50"]
    new = [j for j in key if key[j]["src"] == "new"]
    V["audit1_agree_gpt6"] = sum(jg[j] == hl[key[j]["ref"]] for j in shared)
    V["audit1_agree_opus"] = sum(jo[j] == hl[key[j]["ref"]] for j in shared)
    V["audit1_n_further"] = len(new)
    # accepted: both judges say yes; a query carrying the fence tag "boxed" counts as the same located query
    boxed = {j for j in new if items[j]["recovered_sql"].lstrip().lower().startswith("boxed")}
    acc = [j for j in new if jg[j] == "yes" and (jo[j] == "yes" or j in boxed)]
    V["audit1_further_accepted"] = len(acc)
    V["audit1_further_accepted_pct"] = round(100 * len(acc) / len(new), 1)
    V["audit1_further_cp_lo"], V["audit1_further_cp_hi"] = clopper_pearson(len(acc), len(new))
    V["audit1_rate_no"] = V["audit1_n"] - V["audit1_yes"]
    V["audit1_rate_further_no"] = len(new) - len(acc)

    # ---- second audit: vote-deciding recovered queries in the Du-MAD runs on the 1,531 questions
    A2 = INP / "audit_second"
    k2 = json.loads((A2 / "sample_key.json").read_text())
    l2 = {x["item_id"]: x["label"] for x in json.loads((A2 / "human_labels.json").read_text())["items"]}
    V["audit2_population"] = k2["population_total"]
    for s, n in k2["population"].items():
        V[f"audit2_population|{s}"] = n
    by = collections.defaultdict(list)
    for it in k2["items"]:
        by[f"{it['backbone']}|{it['condition']}"].append(l2[it["item_id"]])
    for s, labs in by.items():
        V[f"audit2_n|{s}"] = len(labs)
        V[f"audit2_yes|{s}"] = sum(x == "yes" for x in labs)
        V[f"audit2_wilson_lo|{s}"] = wilson(V[f"audit2_yes|{s}"], len(labs))[0]
    V["audit2_n"] = len(l2)
    V["audit2_yes"] = sum(x == "yes" for x in l2.values())
    V["audit2_wilson_lo"], V["audit2_wilson_hi"] = wilson(V["audit2_yes"], V["audit2_n"])
    audits = {k: v for k, v in V.items() if k.startswith("audit")}

    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True))
    (out / "reader_sensitivity.json").write_text(json.dumps(full, indent=1, sort_keys=True))
    (out / "audits.json").write_text(json.dumps(audits, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
