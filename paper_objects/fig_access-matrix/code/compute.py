"""Fig. fig:access-matrix: sharing effect on the protocol's final answer, with peers minus without (pp),
and the numbers that discuss it in Section V-A and Supplementary Section sec:s-ford.

    python3 code/compute.py --out output

Reads only ../inputs. Writes values.json and plotted_values.csv (8 protocol-backbone rows x 7
benchmarks; BCCS on WDC and Settlements has no estimate). Each plotted value is the paired effect
of the source table with its sign turned to "with peers minus without":
  - QA, GPQA and BIRD: rq_effects.csv, "NOPEER vs FULL", access, unanswered tasks counted as wrong;
  - WDC and Settlements (FORD, MoA, Du-MAD, MAD-M2): t3_contrasts.csv, corrected labels, all pairs.
A cell carries a dagger when its unresolved-reading range (u4_residue.csv) has positive width, and
the mark "0.0§" when the protocol delivers no answer with or without peers.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
BENCH = ["gsm8k", "strategyqa", "mmlu_pro", "gpqa", "wdc_products", "geographic_settlements", "sql_bird_v2"]
KEYS = [("ford", "roster"), ("moa", "roster"), ("du_mad", "llama3-8b"), ("du_mad", "qwen2.5-7b"),
        ("madm2_objective", "llama3-8b"), ("madm2_objective", "qwen2.5-7b"), ("bccs", "llama3-8b"), ("bccs", "qwen2.5-7b")]
SHORT = {"ford": "ford", "moa": "moa", "du_mad": "dumad", "madm2_objective": "madm2", "bccs": "bccs",
         "roster": "", "llama3-8b": "_l8", "qwen2.5-7b": "_q7"}
MATCHING = BENCH[4:6]


def k(x):
    return (x["benchmark"], x["protocol"], x["backbone"], x["dimension"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    rq = [x for x in csv.DictReader(open(INP / "rq_effects.csv")) if x["convention"] == "abstention charged as failure"]
    fixed = {k(x): x for x in csv.DictReader(open(INP / "t3_contrasts.csv"))
             if x["labels"] == "corrected" and x["paired_drop"] == "False"}
    residue = {k(x): x for x in csv.DictReader(open(INP / "u4_residue.csv"))}
    V, rows, plotted = {}, [], []
    for pr, bb in KEYS:
        for b in BENCH:
            key = (b, pr, bb, "access")
            row = next(x for x in rq if k(x) == key)
            dag = False
            undelivered = b not in MATCHING and int(row["n_delivered_arm"]) == 0 and int(row["n_delivered_full"]) == 0
            val, lo, hi = -100 * float(row["effect"]), -100 * float(row["hi"]), -100 * float(row["lo"])
            if b in MATCHING:
                if pr == "bccs":
                    val = lo = hi = float("nan")
                else:
                    z = fixed[key]
                    val, lo, hi = -float(z["effect_pp"]), -float(z["hi_pp"]), -float(z["lo_pp"])
                    if key in residue:
                        dag = float(residue[key]["residue_width_pp"]) > 0
            tag = f"{SHORT[pr]}{SHORT[bb]}_{b}"
            rows.append([pr, bb, b, None if math.isnan(val) else round(val, 6), None if math.isnan(lo) else round(lo, 6),
                         None if math.isnan(hi) else round(hi, 6), dag, undelivered])
            if not math.isnan(val):
                V[f"effect_{tag}"], V[f"effect_{tag}_lo"], V[f"effect_{tag}_hi"] = (round(v, 6) for v in (val, lo, hi))
                plotted.append(val)
            if b == "sql_bird_v2" and pr == "du_mad" and bb == "llama3-8b":
                V["dumad_l8_bird_paired_runs"] = int(row["n_shared_unit_runs"])
                V["dumad_l8_bird_delivered_with_peers"] = int(row["n_delivered_full"])
                V["dumad_l8_bird_delivered_without_peers"] = int(row["n_delivered_arm"])
    V["effect_min"], V["effect_max"] = round(min(plotted), 6), round(max(plotted), 6)
    V["n_cells_with_estimate"] = len(plotted)
    # unresolved-reading ranges for MAD-M2 / Llama-3-8B on matching (with peers minus without)
    for b in MATCHING:
        z = residue[(b, "madm2_objective", "llama3-8b", "access")]
        V[f"residue_madm2_l8_{b}_lo"] = round(-float(z["residue_bound_hi_pp"]), 6)
        V[f"residue_madm2_l8_{b}_hi"] = round(-float(z["residue_bound_lo_pp"]), 6)
    # FORD: share of runs on which the stopping rule never fires (%)
    for x in csv.DictReader(open(INP / "ford_stop_firing.csv")):
        if x["repeat"] == "all":
            V[f"ford_nonfiring_{x['benchmark']}_{x['arm']}"] = round(100 * (int(x["n"]) - int(x["fired"])) / int(x["n"]), 6)
    qa_bird = [V[f"ford_nonfiring_{b}_NOPEER"] for b in ("gsm8k", "strategyqa", "mmlu_pro", "gpqa", "sql_bird_v2")]
    V["ford_nonfiring_qa_bird_nopeer_min"], V["ford_nonfiring_qa_bird_nopeer_max"] = min(qa_bird), max(qa_bird)
    # FORD on WDC: tied runs and the sharing effect under the all-round majority
    for x in json.loads((INP / "ford_tie_identity.json").read_text())["rows"]:
        if x["benchmark"] == "wdc_products":
            V[f"ford_wdc_tied_runs_{x['arm']}"] = x["tied_runs"]
            V[f"ford_wdc_runs_{x['arm']}"] = x["runs"]
            V[f"ford_wdc_rule_fired_{x['arm']}"] = x["rule_fired"]
    for x in csv.DictReader(open(INP / "matching_return_rule.csv")):
        if x["protocol"] == "ford" and x["benchmark"] == "wdc_products" and x["arm"] == "NOPEER":
            V["ford_wdc_effect_released"] = float(x["sharing_effect_released_pp"])
            V["ford_wdc_effect_majority_tie_first"] = float(x["sharing_effect_all_round_majority_pp"])
            V["ford_wdc_effect_majority_tie_last"] = float(x["sharing_effect_all_round_majority_tie_last_pp"])
            V["ford_wdc_effect_majority_tie_keep"] = float(x["sharing_effect_all_round_majority_tie_keep_pp"])
    # FORD on WDC, per run: runs paired by pair and repeat, split by whether the no-peer run ends in a
    # tie (as many "match" as "non-match" predictions); accuracy parts are shares of all paired runs
    wdc = {}
    for x in csv.DictReader(open(INP / "ford_wdc_runs.csv")):
        wdc[(x["condition"], x["pair"], x["repeat"])] = {c: int(x[c]) for c in
                                                        ("reference_match", "final_match", "tied", "calls")}
    paired = sorted({(p, r) for c, p, r in wdc if (("with_peers", p, r) in wdc and ("no_peers", p, r) in wdc)})
    V["ford_wdc_paired_runs"] = len(paired)

    def f1(rs):
        tp = sum(r["reference_match"] and r["final_match"] for r in rs)
        fp = sum((not r["reference_match"]) and r["final_match"] for r in rs)
        fn = sum(r["reference_match"] and not r["final_match"] for r in rs)
        return 200 * tp / (2 * tp + fp + fn)

    def correct(rs):
        return sum(r["reference_match"] == r["final_match"] for r in rs)

    strata = {"all": paired,
              "tied": [u for u in paired if wdc[("no_peers",) + u]["tied"]],
              "nontied": [u for u in paired if not wdc[("no_peers",) + u]["tied"]]}
    for nm, us in strata.items():
        w = [wdc[("with_peers",) + u] for u in us]
        n = [wdc[("no_peers",) + u] for u in us]
        V[f"ford_wdc_{nm}_runs"] = len(us)
        V[f"ford_wdc_{nm}_f1_gain"] = round(f1(w) - f1(n), 6)
        V[f"ford_wdc_{nm}_accuracy_part"] = round(100 * (correct(w) - correct(n)) / len(paired), 6)
        V[f"ford_wdc_{nm}_calls_with_peers"] = round(sum(r["calls"] for r in w) / len(us), 6)
        V[f"ford_wdc_{nm}_calls_no_peers"] = round(sum(r["calls"] for r in n) / len(us), 6)
    # FORD on GSM8K: mean calls and output tokens per task, with peers minus without
    gsm = collections.defaultdict(list)
    for x in csv.DictReader(open(INP / "ford_gsm8k_runs.csv")):
        gsm[x["condition"]].append(x)
    for c in ("calls", "output_tokens"):
        m = {k: sum(int(x[c]) for x in v) / len(v) for k, v in gsm.items()}
        V[f"ford_gsm8k_{c}_fewer_with_peers"] = round(m["no_peers"] - m["with_peers"], 6)
    with open(out / "plotted_values.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protocol", "backbone", "benchmark", "with_minus_without_pp", "lo_pp", "hi_pp", "unresolved_readings",
                    "no_answer_in_either_condition"])
        w.writerows(rows)
    (out / "values.json").write_text(json.dumps(dict(sorted(V.items())), indent=1) + "\n")
    print(len(plotted), "plotted cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
