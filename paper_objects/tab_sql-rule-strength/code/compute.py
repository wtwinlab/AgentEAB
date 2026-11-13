#!/usr/bin/env python3
"""Execution filters on fixed SQL pools: Table tab:sql-rule-strength and its supplementary tables.

    python3 code/compute.py --out DIR      # writes DIR/values.json

Reads only ../inputs/. Four groups of numbers:

1. Qwen2.5 pools (four and six candidates; Qwen2.5-7B and Qwen2.5-32B recipients, three
   generation runs each). Each run of a question is one observation; the three selection rules
   are deterministic per question, so their outcome repeats across runs. Differences are question
   weighted; 95% intervals resample the databases (5,000 draws, Python `random.Random(20261026)`,
   databases in sorted order, every run of a drawn database carried along; 2.5th and 97.5th
   order statistics as in the stress-test analysis of the selection rules).
2. Execution-vote winners that return an empty result: the pools above and the initial answers
   of the Llama-3-8B and Qwen2.5-7B BIRD debates (lenient extractor, failed executions dropped,
   ties to the first answer). Debate shares of questions average the runs of a question first.
3. Spider test suites: every output scored on the single development database and on all
   test-suite databases of its question. Each question weighted equally; the detection flags use
   the question bootstrap and the 20-database bootstrap (10,000 draws, numpy seed 20261106).
4. The Qwen2.5-72B recipient on the four-candidate pool, compared per question (its single run
   against the mean of the three runs of the other recipients).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"

DRAWS, SEED = 5000, 20261026
TS_BOOT, TS_SEED = 10000, 20261106


def read(name):
    with open(INP / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def db_boot(items, draws=DRAWS, seed=SEED):
    """Point estimate and 95% interval of a mean, resampling databases (stress-test convention)."""
    by = collections.defaultdict(list)
    for db, y in items:
        by[db].append(y)
    keys = sorted(by)
    n = len(keys)
    tot = sum(len(v) for v in by.values())
    point = sum(sum(v) for v in by.values()) / tot
    rnd = random.Random(seed)
    sims = []
    for _ in range(draws):
        s = c = 0.0
        for _ in range(n):
            k = keys[rnd.randrange(n)]
            s += sum(by[k])
            c += len(by[k])
        if c:
            sims.append(s / c)
    sims.sort()
    return point, sims[int(0.025 * len(sims))], sims[int(0.975 * len(sims)) - 1]


def pools(v):
    pq = {(r["benchmark"], r["pool"], int(r["question_id"])): r for r in read("qwen25_pools_per_question.csv")}
    ex = read("qwen25_pools_per_execution.csv")
    rule = {"unfiltered": "unfiltered_correct", "valid": "valid_correct", "nonempty": "nonempty_correct"}

    def y(e, m):
        if m == "generation":
            return int(e["generation_correct"])
        return int(pq[(e["benchmark"], e["pool"], int(e["question_id"]))][rule[m]])

    cells = collections.defaultdict(list)
    for e in ex:
        cells[(e["benchmark"], e["pool"], e["recipient"])].append(e)
    tag = {"qwen2.5-7b": "7b", "qwen2.5-32b": "32b"}
    for (b, pool, rec), R in sorted(cells.items()):
        stem = "%s_%s_%s" % (b, pool, tag[rec])
        # generation minus each selection rule, all questions
        for m in ("unfiltered", "valid", "nonempty"):
            p, lo, hi = db_boot([(e["db_id"], float(y(e, "generation") - y(e, m))) for e in R])
            v["%s_gen_minus_%s" % (stem, m)] = 100 * p
            v["%s_gen_minus_%s_lo" % (stem, m)] = 100 * lo
            v["%s_gen_minus_%s_hi" % (stem, m)] = 100 * hi
        if rec != "qwen2.5-32b":
            continue
        # rule-to-rule differences (identical for both recipients: same pool, same runs)
        st = {e["question_id"]: pq[(b, pool, int(e["question_id"]))]["reference_stratum"] for e in R}
        for a, c in (("unfiltered", "valid"), ("valid", "nonempty")):
            for stratum in ("all", "gold_nonempty", "gold_empty"):
                sub = [e for e in R if stratum == "all" or st[e["question_id"]] == stratum]
                if not sub:
                    continue
                key = "%s_%s_%s_minus_%s_%s" % (b, pool, c, a, stratum)
                if stratum == "all" or b == "spider":
                    p, lo, hi = db_boot([(e["db_id"], float(y(e, c) - y(e, a))) for e in sub])
                    v[key], v[key + "_lo"], v[key + "_hi"] = 100 * p, 100 * lo, 100 * hi
                else:
                    v[key] = 100 * sum(y(e, c) - y(e, a) for e in sub) / len(sub)
        if b == "spider":
            ne = [e for e in R if st[e["question_id"]] == "gold_nonempty"]
            em = [e for e in R if st[e["question_id"]] == "gold_empty"]
            dn = sum(y(e, "nonempty") - y(e, "valid") for e in ne) / len(ne)
            de = sum(y(e, "nonempty") - y(e, "valid") for e in em) / len(em)
            q = len(em) / (len(em) + len(ne))
            v["%s_%s_empty_reference_share" % (b, pool)] = 100 * q
            v["%s_%s_empty_reference_questions" % (b, pool)] = len({e["question_id"] for e in em})
            # Nonempty minus Valid at prevalence p: (1-p) dn + p de; zero at p = dn / (dn - de)
            v["%s_%s_nonempty_break_even_share" % (b, pool)] = 100 * dn / (dn - de)
        # executions where generation succeeds and Unfiltered fails, and the pool holds a correct query
        for r2 in ("qwen2.5-7b", "qwen2.5-32b"):
            R2 = cells[(b, pool, r2)]
            reg = [e for e in R2 if y(e, "generation") == 1 and y(e, "unfiltered") == 0]
            av = sum(int(pq[(b, pool, int(e["question_id"]))]["pool_has_correct"]) for e in reg)
            v["%s_%s_%s_gen_only_pool_has_correct_share" % (b, pool, tag[r2])] = 100 * av / len(reg)
        if b == "spider" and pool == "pool6":
            v["spider_pool6_unfiltered_accuracy"] = 100 * sum(y(e, "unfiltered") for e in R) / len(R)
            v["spider_pool6_32b_generation_accuracy"] = 100 * sum(y(e, "generation") for e in R) / len(R)
            v["spider_pool6_32b_unfiltered_only_runs"] = sum(1 for e in R if y(e, "unfiltered") > y(e, "generation"))
            v["spider_pool6_32b_generation_only_runs"] = sum(1 for e in R if y(e, "unfiltered") < y(e, "generation"))
            v["spider_pool6_32b_runs"] = len(R)
            v["spider_pool6_32b_differing_share"] = 100 * (v["spider_pool6_32b_unfiltered_only_runs"]
                                                           + v["spider_pool6_32b_generation_only_runs"]) / len(R)
    v["bird_gen_only_pool_has_correct_share_min"] = min(x for k, x in v.items()
                                                        if k.startswith("bird") and k.endswith("_gen_only_pool_has_correct_share"))
    v["bird_gen_only_pool_has_correct_share_max"] = max(x for k, x in v.items()
                                                        if k.startswith("bird") and k.endswith("_gen_only_pool_has_correct_share"))

    # empty winners and unexecutable winners, one decision per (benchmark, pool, question)
    byp = collections.defaultdict(list)
    for r in pq.values():
        byp[(r["benchmark"], r["pool"])].append(r)
    for (b, pool), rows in sorted(byp.items()):
        n = len(rows)
        v["%s_%s_questions" % (b, pool)] = n
        v["%s_%s_unfiltered_winner_unexecutable_share" % (b, pool)] = 100 * sum(int(r["unfiltered_winner_failed"]) for r in rows) / n
        for m in ("valid", "nonempty"):
            emp = [r for r in rows if r["%s_winner_empty" % m] == "1"]
            wrong = [r for r in rows if r["%s_correct" % m] == "0"]
            wrong_emp = [r for r in wrong if r["%s_winner_empty" % m] == "1"]
            v["%s_%s_%s_empty_winner_share_of_questions" % (b, pool, m)] = 100 * len(emp) / n
            v["%s_%s_%s_empty_share_of_wrong_winners" % (b, pool, m)] = 100 * len(wrong_emp) / len(wrong)
            v["%s_%s_%s_empty_winners" % (b, pool, m)] = len(emp)
            v["%s_%s_%s_empty_winners_correct" % (b, pool, m)] = sum(1 for r in emp if r["%s_correct" % m] == "1")
    # the same share on the BIRD questions scored in both pools (1,532), as Section VII-B prints it
    paired = {q for (bb, pl, q) in pq if bb == "bird" and pl == "pool4"} & {q for (bb, pl, q) in pq if bb == "bird" and pl == "pool6"}
    v["bird_paired_questions"] = len(paired)
    for pool in ("pool4", "pool6"):
        v["bird_%s_unfiltered_winner_unexecutable_share_paired" % pool] = (
            100 * sum(int(pq[("bird", pool, q)]["unfiltered_winner_failed"]) for q in paired) / len(paired))


def debates(v):
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in read("debate_initial_vote_winners.csv"):
        cells[(r["protocol"], r["backbone"])][r["question"]].append(r)
    name = {"du_mad": "dumad", "madm2_objective": "madm2", "bccs": "bccs", "moa": "moa"}
    for (p, bb), by in sorted(cells.items()):
        stem = "debate_%s_%s" % (name[p], bb.replace(".", "").replace("-", ""))
        per_q = [np.mean([float(x["winner_empty"]) for x in runs]) for runs in by.values()]
        v[stem + "_empty_winner_share_of_questions"] = 100 * float(np.mean(per_q))
        runs = [x for rr in by.values() for x in rr]
        wrong = [x for x in runs if x["winner_exists"] == "1" and x["winner_correct"] == "0"]
        v[stem + "_empty_share_of_wrong_winners"] = 100 * sum(int(x["winner_empty"]) for x in wrong) / len(wrong)
    bird_valid = [v["bird_pool4_valid_empty_share_of_wrong_winners"], v["bird_pool6_valid_empty_share_of_wrong_winners"]]
    deb = [x for k, x in v.items() if k.startswith("debate_") and k.endswith("_empty_share_of_wrong_winners")
           and "_moa_" not in k]
    v["bird_vote_empty_share_of_wrong_winners_min"] = min(bird_valid + deb)
    v["bird_vote_empty_share_of_wrong_winners_max"] = max(bird_valid + deb)


def _det(x, dbs):
    """Both 95% intervals (questions; databases) exclude zero, as in the source analysis."""
    bq = x[np.random.default_rng(TS_SEED).integers(0, len(x), size=(TS_BOOT, len(x)))].mean(axis=1)
    keys = sorted(set(dbs))
    mem = {d: np.array([i for i, dd in enumerate(dbs) if dd == d]) for d in keys}
    rng = np.random.default_rng(TS_SEED)
    bd = np.empty(TS_BOOT)
    for b in range(TS_BOOT):
        pick = rng.integers(0, len(keys), size=len(keys))
        bd[b] = x[np.concatenate([mem[keys[j]] for j in pick])].mean()
    ql, qh = np.percentile(bq, [2.5, 97.5])
    dl, dh = np.percentile(bd, [2.5, 97.5])
    return int((ql > 0 and dl > 0) or (qh < 0 and dh < 0))


def test_suite(v, flags):
    rows = read("spider_test_suite_per_question.csv")
    dbs = [r["db_id"] for r in rows]
    nts = [int(r["test_suite_databases"]) for r in rows]
    v["spider_test_suite_databases_min"] = min(nts)
    v["spider_test_suite_databases_max"] = max(nts)
    v["spider_test_suite_databases_mean"] = float(np.mean(nts))

    def col(name, mode):
        if "qwen25_generation" in name:
            return np.array([int(r["%s_%s_of6" % (name, mode)]) / 6 for r in rows])
        return np.array([int(r["%s_%s" % (name, mode)]) for r in rows], dtype=float)

    systems = ["pool4_unfiltered", "pool4_valid", "pool4_nonempty", "pool4_qwen25_generation",
               "pool6_unfiltered", "pool6_valid", "pool6_nonempty", "pool6_qwen25_generation",
               "pool4_qwen3_selector", "pool4_qwen3_generation", "pool4_deepseek_selector",
               "pool4_deepseek_generation"]
    drops = []
    for s in systems:
        si, ts = col(s, "single"), col(s, "suite")
        v["ts_%s_single" % s] = 100 * float(si.mean())
        v["ts_%s_suite" % s] = 100 * float(ts.mean())
        v["ts_%s_drop" % s] = 100 * float((si - ts).mean())
        drops.append(v["ts_%s_drop" % s])
    v["ts_drop_min"], v["ts_drop_max"] = min(drops), max(drops)
    pairs = [("pool4_valid", "pool4_unfiltered"), ("pool6_valid", "pool6_unfiltered"),
             ("pool4_nonempty", "pool4_valid"), ("pool6_nonempty", "pool6_valid"),
             ("pool4_qwen25_generation", "pool4_valid"), ("pool6_qwen25_generation", "pool6_valid"),
             ("pool4_qwen3_selector", "pool4_valid"), ("pool4_deepseek_selector", "pool4_valid"),
             ("pool4_deepseek_selector", "pool4_nonempty"), ("pool4_deepseek_generation", "pool4_deepseek_selector")]
    for a, b in pairs:
        for mode in ("single", "suite"):
            d = col(a, mode) - col(b, mode)
            v["ts_%s_minus_%s_%s" % (a, b, mode)] = 100 * float(d.mean())
            flags["ts_%s_minus_%s_%s_detected" % (a, b, mode)] = _det(d, dbs)
    for pool in ("pool4", "pool6"):
        g = sum(int(r["%s_multi_sql_groups" % pool]) for r in rows)
        s = sum(int(r["%s_groups_split_on_suite" % pool]) for r in rows)
        v["ts_%s_vote_groups_split_share" % pool] = 100 * s / g
        si, ts = col(pool + "_valid", "single"), col(pool + "_valid", "suite")
        v["ts_%s_valid_single_correct_suite_wrong_share" % pool] = 100 * float(((si == 1) & (ts == 0)).mean())


def large_recipient(v):
    """Qwen2.5-72B generation on the four-candidate pool, per question (point estimates)."""
    pq = {(r["benchmark"], int(r["question_id"])): r for r in read("qwen25_pools_per_question.csv")
          if r["pool"] == "pool4"}
    g = collections.defaultdict(list)
    for e in read("qwen25_pools_per_execution.csv"):
        if e["pool"] == "pool4":
            g[(e["benchmark"], e["recipient"], int(e["question_id"]))].append(int(e["generation_correct"]))
    big = {(r["benchmark"], int(r["question_id"])): int(r["generation_correct"])
           for r in read("qwen25_72b_generation_per_question.csv")}
    for b in ("bird", "spider"):
        qs = sorted(q for (bb, q) in big if bb == b and (b, q) in pq)
        v["%s_pool4_72b_minus_valid" % b] = 100 * float(np.mean([big[(b, q)] - int(pq[(b, q)]["valid_correct"]) for q in qs]))
        v["%s_pool4_72b_minus_32b" % b] = 100 * float(np.mean(
            [big[(b, q)] - np.mean(g[(b, "qwen2.5-32b", q)]) for q in qs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    v, flags = {}, {}
    pools(v)
    debates(v)
    test_suite(v, flags)
    large_recipient(v)
    v.update(flags)
    (out / "values.json").write_text(json.dumps(dict(sorted(v.items())), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
