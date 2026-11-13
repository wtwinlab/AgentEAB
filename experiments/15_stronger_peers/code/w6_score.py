r"""W6 scoring: the LLM selector beside Original / Valid / Nonempty / generation / oracle.

    python3 code/w6_score.py      # -> output/w6_levels.csv, w6_contrasts.csv, w6_decomposition.csv,
                                  #    w6_cost.json, w6_per_question.csv

Family, weighting and interval are fixed in advance.
Rule and generation outcomes are read from the existing cross-family per-question output, not
recomputed; the selector is scored from the existing candidate labels; the oracle is used for
scoring only.
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
WORK = HERE.parent
REPO = WORK.parents[1]
OUT = WORK / "output"
POOL = ("qwen2.5-7b", "llama-3-8b", "qwen2.5-14b", "qwen2.5-32b")
RULES = {"Original": "released", "Valid": "valid_majority", "Nonempty": "nonempty_majority"}
RECIPIENT = "qwen3-32b"
DRAWS, SEED = 10000, 20261101
CONTRASTS = (("Selector", "Original"), ("Selector", "Valid"), ("Selector", "Nonempty"),
             ("Generation", "Selector"))


def load():
    labels = {}
    for b in ("bird", "spider"):
        for line in open(REPO / "data/derived/sql_decomposition" / f"cand_{b}.jsonl"):
            r = json.loads(line)
            labels[(b, int(r["question_id"]), r["model"])] = r
    cf = collections.defaultdict(dict)
    for r in csv.DictReader(open(REPO / "data/derived/sql_rule_pairing/output"
                                 / "crossfamily_per_question.csv")):
        if r["recipient"] != RECIPIENT:
            continue
        k = (r["benchmark"], int(r["question_id"]))
        cf[k]["db_id"] = r["db_id"]
        cf[k][r["selector"]] = int(r["selection_correct"])
        cf[k]["generation"] = int(r["generation_correct"])
    sel = {}
    for line in open(WORK / "raw" / "w6_selector_main.jsonl"):
        r = json.loads(line)
        sel[(r["benchmark"], int(r["question_id"]))] = r
    return labels, cf, sel


def main() -> int:
    labels, cf, sel = load()
    rows, missing = [], []
    for (b, q), c in sorted(cf.items()):
        s = sel.get((b, q))
        if s is None:
            missing.append(dict(benchmark=b, question_id=q, reason="no selector record"))
        pick = s.get("picked_model") if s else None
        lab = labels.get((b, q, pick)) if pick else None
        oracle = int(any(bool((labels.get((b, q, m)) or {}).get("match")) for m in POOL))
        rows.append(dict(
            benchmark=b, question_id=q, db_id=c["db_id"],
            Original=c["released"], Valid=c["valid_majority"], Nonempty=c["nonempty_majority"],
            Selector=int(bool(lab and lab.get("match"))), Generation=c["generation"],
            Oracle=oracle, selector_pick=pick,
            selector_nonselection=(s or {}).get("nonselection_reason") if s else "missing",
            selector_finish=(s or {}).get("finish_reason")))
    OUT.mkdir(exist_ok=True)
    with open(OUT / "w6_per_question.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # sanity on the read-in rules: a rule can only be right when the pool has a right candidate
    viol = {k: sum(1 for r in rows if r[k] and not r["Oracle"])
            for k in ("Original", "Valid", "Nonempty", "Selector")}

    rng = np.random.default_rng(SEED)
    METHODS = ("Oracle", "Original", "Valid", "Nonempty", "Selector", "Generation")
    levels, contrasts = [], []
    for b in ("bird", "spider"):
        sub = [r for r in rows if r["benchmark"] == b]
        dbs = sorted({r["db_id"] for r in sub})
        by_db = {d: [r for r in sub if r["db_id"] == d] for d in dbs}
        draws = rng.integers(0, len(dbs), size=(DRAWS, len(dbs)))

        def score(fn, weighting):
            per = np.array([np.mean([fn(r) for r in by_db[d]]) for d in dbs])
            n = np.array([len(by_db[d]) for d in dbs], dtype=float)
            w = np.ones(len(dbs)) if weighting == "database_equal" else n
            point = float(np.sum(w * per) / np.sum(w))
            boot = (per[draws] * w[draws]).sum(1) / w[draws].sum(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            return point, float(lo), float(hi), boot

        for weighting in ("database_equal", "question"):
            for m in METHODS:
                p, lo, hi, _ = score(lambda r, m=m: r[m], weighting)
                levels.append(dict(benchmark=b, weighting=weighting, method=m,
                                   questions=len(sub), databases=len(dbs),
                                   accuracy_pct=round(100 * p, 2),
                                   lo_pct=round(100 * lo, 2), hi_pct=round(100 * hi, 2)))
            for a, c in CONTRASTS:
                p, lo, hi, boot = score(lambda r, a=a, c=c: r[a] - r[c], weighting)
                pv = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
                contrasts.append(dict(benchmark=b, weighting=weighting, contrast=f"{a} - {c}",
                                      effect_pp=round(100 * p, 2), lo_pp=round(100 * lo, 2),
                                      hi_pp=round(100 * hi, 2), p_boot=round(pv, 4),
                                      ci_excludes_zero=bool(lo > 0 or hi < 0)))
    # Holm over the frozen 8-contrast family, primary weighting only
    fam = sorted([c for c in contrasts if c["weighting"] == "database_equal"],
                 key=lambda c: c["p_boot"])
    m = len(fam); stop = False
    for i, c in enumerate(fam):
        thr = 0.05 / (m - i)
        rej = (not stop) and c["p_boot"] <= thr
        stop = stop or not rej
        c["holm_reject_in_w6_family"] = rej
        c["holm_family_size"] = m
    for c in contrasts:
        c.setdefault("holm_reject_in_w6_family", None); c.setdefault("holm_family_size", None)

    # where the gap goes: correct candidate available but not returned, same basis
    decomp = []
    for b in ("bird", "spider"):
        sub = [r for r in rows if r["benchmark"] == b]
        n = len(sub)
        for meth in ("Original", "Valid", "Nonempty", "Selector"):
            avail_missed = sum(1 for r in sub if r["Oracle"] and not r[meth])
            decomp.append(dict(benchmark=b, method=meth, weighting="question", questions=n,
                               oracle_pct=round(100 * sum(r["Oracle"] for r in sub) / n, 2),
                               returned_correct_pct=round(100 * sum(r[meth] for r in sub) / n, 2),
                               available_not_returned_pct=round(100 * avail_missed / n, 2),
                               share_of_gap_recovered_vs_original=None))
        o = next(d for d in decomp if d["benchmark"] == b and d["method"] == "Original")
        for d in decomp:
            if d["benchmark"] == b and o["available_not_returned_pct"]:
                d["share_of_gap_recovered_vs_original"] = round(
                    1 - d["available_not_returned_pct"] / o["available_not_returned_pct"], 4)

    # selector cost, as recorded
    cost = {}
    for b in ("bird", "spider"):
        rs = [json.loads(l) for l in open(WORK / "raw" / "w6_selector_main.jsonl")]
        rs = [r for r in rs if r["benchmark"] == b]
        u = [r.get("usage") or {} for r in rs]
        cost[b] = dict(calls=len(rs),
                       prompt_tokens=sum(x.get("prompt_tokens", 0) for x in u),
                       completion_tokens=sum(x.get("completion_tokens", 0) for x in u),
                       mean_prompt_tokens=round(np.mean([x.get("prompt_tokens", 0) for x in u]), 1),
                       mean_latency_s=round(float(np.mean([r["latency_s"] for r in rs])), 3),
                       nonselections=sum(1 for r in rs if not r.get("picked_id")),
                       truncated=sum(1 for r in rs if r.get("finish_reason") == "length"),
                       picked_model=dict(collections.Counter(r.get("picked_model") for r in rs)))
    cost["note"] = ("One extra selection call per question, over candidates already generated and "
                    "executed. Distinct from a regeneration call, whose cost is a new SQL "
                    "generation by the recipient; that cost is in the cross-family record, not here.")

    for name, data in (("w6_levels.csv", levels), ("w6_contrasts.csv", contrasts),
                       ("w6_decomposition.csv", decomp)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (OUT / "w6_cost.json").write_text(json.dumps(dict(cost=cost, missing=missing,
                                                      oracle_violations=viol), indent=1) + "\n")
    print("oracle violations (rule right while pool has no right candidate):", viol)
    print("missing selector records:", len(missing))
    for L in levels:
        if L["weighting"] == "database_equal":
            print(f"  {L['benchmark']:6} {L['method']:10} {L['accuracy_pct']:6.2f} "
                  f"[{L['lo_pct']}, {L['hi_pct']}]")
    for c in contrasts:
        print(f"  {c['benchmark']:6} {c['weighting']:14} {c['contrast']:22} {c['effect_pp']:+6.2f} "
              f"[{c['lo_pp']:+.2f},{c['hi_pp']:+.2f}] p={c['p_boot']} holm={c['holm_reject_in_w6_family']}")
    for d in decomp:
        print(f"  {d['benchmark']:6} {d['method']:9} oracle {d['oracle_pct']} returned "
              f"{d['returned_correct_pct']} missed {d['available_not_returned_pct']} "
              f"recovered {d['share_of_gap_recovered_vs_original']}")
    print(json.dumps({k: v for k, v in cost.items() if k != 'note'}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
