r"""Measurement-sensitivity census: naive vs controlled reading of every eligible cell (zero model calls).

    python3 -I share_eval/replay_validity/sensitivity_census.py [--replay DIR] [--out DIR]

Families S1–S7 (Table S31 of the paper) and their inputs (read-only):
  S1–S3  <replay>/per_run_scores.jsonl and factorial_replay.json (BIRD debates, Llama-3-8B and Qwen2.5-7B, 198 questions)
  S4     data/match_qa_audit.json        (matching and QA: protocol's extractor vs lenient extractor, per cell)
  S5     data/return_rule_cells.csv      (matching and QA: protocol's return rule vs all-round majority, per cell)
  S6     ReConcile confidence parse on matching, values listed below (S6)
  S7     pair-F1 versus cluster-F1 direction counts over the 990 grouping configuration cells (S7)
Writes <out>/sensitivity_census.{json,csv}.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
REPLAY = REPO / "output/share_eval/replay"
MATCH_QA = HERE / "data/match_qa_audit.json"
RETURN_RULE = HERE / "data/return_rule_cells.csv"
BOOT, SEED = 10000, 20261106


def classify(n_est, n_lo, n_hi, c_est, c_lo, c_hi):
    def sig(lo, hi):
        return None if lo is None or hi is None else (lo > 0 or hi < 0)
    sn, sc = sig(n_lo, n_hi), sig(c_lo, c_hi)
    sign_change = n_est != 0 and c_est != 0 and np.sign(n_est) != np.sign(c_est)
    out = dict(sign_change=bool(sign_change))
    if sn is None or sc is None:
        out["significance"] = "no interval"
    elif sn and not sc:
        out["significance"] = "lost"
    elif sc and not sn:
        out["significance"] = "gained"
    else:
        out["significance"] = "same"
    out["stable"] = (not sign_change) and out["significance"] == "same"
    return out


def boot(d, idx):
    b = d[idx].mean(axis=1)
    return (round(100 * float(d.mean()), 2), round(100 * float(np.percentile(b, 2.5)), 2),
            round(100 * float(np.percentile(b, 97.5)), 2))


# ------------------------------------------------------------------ S1–S3 (BIRD per-task data)
def bird_families():
    runs = [json.loads(l) for l in (REPLAY / "per_run_scores.jsonl").read_text().splitlines()]
    cells = sorted({(r["protocol"], r["backbone"]) for r in runs})
    rows = []
    tie_rows = []
    store = collections.defaultdict(dict)
    for c in cells:
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in runs:
            if (r["protocol"], r["backbone"]) == c:
                by[r["unit"]][r["arm"]].append(r)
        units = sorted(u for u, v in by.items() if {"FULL", "NOPEER"} <= set(v))
        idx = np.random.default_rng(SEED).integers(0, len(units), size=(BOOT, len(units)))

        def m(u, arm, key):
            return float(np.mean([x[key] for x in by[u][arm]]))
        fam = {}
        fam["S1"] = (np.array([m(u, "FULL", "final") - m(u, "NOPEER", "final") for u in units]),
                     np.array([m(u, "FULL", "lenient|exec|last|first") - m(u, "NOPEER", "lenient|exec|last|first") for u in units]))
        if c[0] != "moa":
            dp = lambda u, k: (m(u, "FULL", f"{k}|last|first") - m(u, "FULL", f"{k}|r0|first")
                               - m(u, "NOPEER", f"{k}|last|first") + m(u, "NOPEER", f"{k}|r0|first"))  # noqa: E731
            fam["S2"] = (np.array([dp(u, "strict|string") for u in units]), np.array([dp(u, "lenient|exec") for u in units]))
        fam["S3"] = (np.array([m(u, "FULL", "final") - m(u, "FULL", "strict|string|r0|first") for u in units]),
                     np.array([m(u, "FULL", "final") - m(u, "FULL", "lenient|exec|r0|first") for u in units]))
        naive_undef = all(m(u, a, "final") == 0 for u in units for a in ("FULL", "NOPEER"))
        for f, v in fam.items():
            if c[0] != "moa":
                store[f][c] = (units, v)
        for f, (nv, cv) in fam.items():
            ne, nl, nh = boot(nv, idx)
            ce, cl, ch = boot(cv, idx)
            de, dl, dh = boot(cv - nv, idx)
            rows.append(dict(family=f, cell=f"{c[0]}|{c[1]}", dataset="BIRD", n=len(units), metric="EX pp",
                             naive=ne, naive_ci=[nl, nh], controlled=ce, controlled_ci=[cl, ch],
                             controlled_minus_naive=de, cmn_ci=[dl, dh],
                             naive_undefined=bool(naive_undef and f != "S3"),
                             label="measurement correction", **classify(ne, nl, nh, ce, cl, ch)))
        # tie-mass check: only tied runs can change under a different tie rule
        for key in ("strict|string|r0", "lenient|exec|r0", "strict|string|last"):
            d_first_last = np.array([m(u, "FULL", f"{key}|first") - m(u, "FULL", f"{key}|last") for u in units])
            tie_rows.append(dict(cell=f"{c[0]}|{c[1]}", readout=key, moved_pp=round(100 * float(np.abs(d_first_last).mean()), 2),
                                 mean_change_first_minus_last_pp=round(100 * float(d_first_last.mean()), 2)))
    pooled = []
    for f, per in store.items():
        u0 = next(iter(per.values()))[0]
        assert all(v[0] == u0 for v in per.values())
        nv = np.mean([v[1][0] for v in per.values()], axis=0)
        cv = np.mean([v[1][1] for v in per.values()], axis=0)
        idx = np.random.default_rng(SEED).integers(0, len(u0), size=(BOOT, len(u0)))
        ne, nl, nh = boot(nv, idx)
        ce, cl, ch = boot(cv, idx)
        de, dl, dh = boot(cv - nv, idx)
        pooled.append(dict(family=f, cell="mean of 6 interactive cells (summary, not an extra cell)", dataset="BIRD",
                           n=len(u0), naive=ne, naive_ci=[nl, nh], controlled=ce, controlled_ci=[cl, ch],
                           controlled_minus_naive=de, cmn_ci=[dl, dh], **classify(ne, nl, nh, ce, cl, ch)))
    return rows, tie_rows, pooled


def tie_shares():
    fac = json.loads((REPLAY / "factorial_replay.json").read_text())
    out = {}
    for cell, v in fac["cells"].items():
        for k, t in (v.get("ties") or {}).items():
            out[(cell, k)] = t["tie_share"]
    return out


# ------------------------------------------------------------------ S4 (matching + QA audit)
def s4():
    d = json.loads(MATCH_QA.read_text())
    rows = []
    for k, v in sorted(d["cells"].items()):
        bench, proto, arm, bb = k.split("|")
        n, c = v.get("final-r0|contract"), v.get("final-r0|lenient")
        if not n or not c:
            continue
        metric = "pair F1 pp" if bench in ("wdc_products", "geographic_settlements") else "accuracy pp"
        undef = v["r0_extracted|contract"] == 0 and v["final_present|contract"] == 0
        negctl = v["r0_extracted|contract"] >= 0.995 and v["final_present|contract"] >= 0.995
        rows.append(dict(family="S4", cell=f"{proto}|{bb}|{arm}", dataset=bench, n=n["n_units"], metric=metric,
                         naive=n["effect_pp"], naive_ci=[n["lo_pp"], n["hi_pp"]],
                         controlled=c["effect_pp"], controlled_ci=[c["lo_pp"], c["hi_pp"]],
                         controlled_minus_naive=round(c["effect_pp"] - n["effect_pp"], 2), cmn_ci=None,
                         naive_undefined=bool(undef), negative_control=bool(negctl),
                         contract_extraction_r0=v["r0_extracted|contract"], contract_final_present=v["final_present|contract"],
                         label="measurement correction", **classify(n["effect_pp"], n["lo_pp"], n["hi_pp"], c["effect_pp"], c["lo_pp"], c["hi_pp"])))
    return rows


# ------------------------------------------------------------------ S5 (return rule on matching / QA)
def s5():
    rows = []
    with open(RETURN_RULE) as fh:
        for r in csv.DictReader(fh):
            f = lambda k: float(r[k])                                    # noqa: E731
            for tie, suf in (("first", ""), ("last", "_tie_last"), ("keep", "_tie_keep")):
                ne, nl, nh = f("sharing_effect_released_pp"), f("lo_released_pp"), f("hi_released_pp")
                ce = f(f"sharing_effect_all_round_majority{suf}_pp")
                cl, ch = f(f"lo_all_round_majority{suf}_pp"), f(f"hi_all_round_majority{suf}_pp")
                rows.append(dict(family="S5", tie=tie, cell=f"{r['protocol']}|{r['dimension']}|{r['arm']}", dataset=r["benchmark"],
                                 n=int(r["n_units_released"]), metric=r["metric"], naive=ne, naive_ci=[nl, nh],
                                 controlled=ce, controlled_ci=[cl, ch], controlled_minus_naive=round(ce - ne, 3), cmn_ci=None,
                                 tie_share_arm=f("share_majority_tied_arm"), tie_share_full=f("share_majority_tied_full"),
                                 label="estimand change (return operation)", naive_undefined=False,
                                 **classify(ne, nl, nh, ce, cl, ch)))
    return rows


# ------------------------------------------------------------------ S6 (ReConcile parse), transcribed
S6_SOURCE = "ReConcile confidence parse: declines charged vs delivered only, before and after the parse, per matching dataset"
S6 = [("wdc_products", 0.8746, 0.1334, 0.8714, 0.8714, False),
      ("geographic_settlements", 0.9995, 0.4459, 0.9958, 0.9958, False),
      ("musicbrainz (local ~15% subset)", 0.9964, 0.2793, 0.9960, 0.9960, True)]


def s6():
    rows = []
    for ds, dl0, ch0, dl2, ch2, partial in S6:
        ne, ce = round(100 * (ch0 - dl0), 2), round(100 * (ch2 - dl2), 2)
        rows.append(dict(family="S6", cell="reconcile|roster|FULL", dataset=ds, n=None, metric="pair F1 pp (declines charged − delivered only)",
                         naive=ne, naive_ci=None, controlled=ce, controlled_ci=None, controlled_minus_naive=round(ce - ne, 2),
                         cmn_ci=None, partial_population=partial, label="measurement correction", naive_undefined=False,
                         sign_change=False, significance="no interval", stable=False,
                         note="naive effect vanishes (controlled = 0); no interval in source"))
    return rows


S7 = dict(source="direction tables of the 990 grouping configuration cells (test split, exact categories)",
          cells=990, both_up=655, both_down=272, pair_up_cluster_down=20, pair_down_cluster_up=14,
          pair_up_cluster_0=15, pair_down_cluster_0=14, reversals=34)


def main() -> int:
    import argparse
    global REPLAY, MATCH_QA, RETURN_RULE
    ap = argparse.ArgumentParser(description="measurement-sensitivity census (Table S31)")
    ap.add_argument("--replay", default=str(REPLAY), help="share_eval/replay output directory (default %(default)s)")
    ap.add_argument("--match-qa", default=str(MATCH_QA), help="per-cell extractor comparison on matching and QA")
    ap.add_argument("--return-rule", default=str(RETURN_RULE), help="per-cell return-rule comparison on matching and QA")
    ap.add_argument("--out", default=str(REPO / "output/share_eval/replay_validity"))
    a = ap.parse_args()
    REPLAY, MATCH_QA, RETURN_RULE = pathlib.Path(a.replay), pathlib.Path(a.match_qa), pathlib.Path(a.return_rule)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bird, tie_rows, pooled = bird_families()
    ts = tie_shares()
    for t in tie_rows:
        rd = t["readout"].rsplit("|", 1)
        t["tie_share_FULL"] = ts.get((t["cell"], f"{t['readout']}|FULL"))
        t["bound_holds"] = t["tie_share_FULL"] is None or t["moved_pp"] <= 100 * t["tie_share_FULL"] + 1e-9
    rows = bird + s4() + s5() + s6()
    summ = {}
    for fam in ("S1", "S2", "S3", "S4", "S5", "S6"):
        for tie in ((None,) if fam != "S5" else ("first", "last", "keep")):
            rs = [r for r in rows if r["family"] == fam and (tie is None or r.get("tie") == tie)]
            elig = [r for r in rs if not r.get("naive_undefined")]
            key = fam if tie is None else f"{fam}|tie={tie}"
            summ[key] = dict(rows=len(rs), naive_undefined=len(rs) - len(elig), eligible=len(elig),
                             sign_change=sum(r["sign_change"] for r in elig),
                             sig_lost=sum(r["significance"] == "lost" for r in elig),
                             sig_gained=sum(r["significance"] == "gained" for r in elig),
                             no_interval=sum(r["significance"] == "no interval" for r in elig),
                             changed=sum((r["sign_change"] or r["significance"] in ("lost", "gained")) for r in elig),
                             stable=sum(r["stable"] for r in elig),
                             stable_rows=[f"{r['cell']}|{r['dataset']}" for r in elig if r["stable"]],
                             label=rs[0]["label"] if rs else None)
            if fam == "S4":
                nc = [r for r in elig if r.get("negative_control")]
                summ[key]["negative_control"] = dict(n=len(nc), max_abs_difference_pp=max((abs(r["controlled_minus_naive"]) for r in nc), default=None),
                                                     any_changed=any(not r["stable"] for r in nc),
                                                     cells=[f"{r['cell']}|{r['dataset']}" for r in nc])
    res = dict(summary=summ, pooled_BIRD=pooled, S7=S7, S6_source=S6_SOURCE, tie_mass_check=tie_rows, rows=rows)
    (out / "sensitivity_census.json").write_text(json.dumps(res, indent=1))
    keys = ["family", "tie", "cell", "dataset", "n", "metric", "naive", "naive_ci", "controlled", "controlled_ci",
            "controlled_minus_naive", "cmn_ci", "sign_change", "significance", "stable", "naive_undefined", "label"]
    with open(out / "sensitivity_census.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for k, v in summ.items():
        print(k, {kk: vv for kk, vv in v.items() if kk not in ("stable_rows",)})
    for t in pooled:
        print("pooled", t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
