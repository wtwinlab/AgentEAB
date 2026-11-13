r"""Final aggregation incl. the information-aligned selector S_al (P0/P1). Offline, no DB, no model.

    python3 code/score.py          # B/C/G families and per_question.csv (run first)
    python3 code/score_sal.py      # adds S_al; P0 / P1 / P1-change families; P2 trigger;
                                   # database-equal tokens and price scenarios for every operation

Writes output/per_question_all.csv, levels_all.csv, sal_contrasts.csv, sal_cross_pool.csv,
cost_dbequal.csv, sal_prompt_token_delta.csv, p2_trigger.json, summary_sal.json.
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
import score as S0                                                          # noqa: E402

WORK, OUT, REPO, W3 = S0.WORK, S0.OUT, S0.REPO, S0.W3
PICK = re.compile(r"\bsystem\s+([1-9][0-9]*)\b", re.I)
METHODS = ("Oracle", "Original", "Valid", "Nonempty", "B", "C", "S_al", "Generation")
CON = (("S_al", "Generation"), ("S_al", "C"), ("S_al", "Nonempty"))
CHG = (("S_al", "Generation"), ("S_al", "Nonempty"))
ORDER_SET = ("Nonempty", "C", "S_al", "Generation")
DEC = REPO / "data/derived/sql_decomposition"


def parse(text, mapping):
    m = PICK.search(text or "")
    if not m:
        return None, "no_identifier"
    cid = f"system {int(m.group(1))}"
    return (mapping[cid], None) if cid in mapping else (None, "out_of_pool")


def last(path):
    d = {}
    for r in map(json.loads, open(path)):
        if r["unit_id"] not in d or r["status"] == "ok":
            d[r["unit_id"]] = r
    return d


def toks(r):
    u = (r or {}).get("usage") or {}
    return (u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0,
            (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)


def main() -> int:
    rows = {}
    for r in csv.DictReader(open(OUT / "per_question.csv")):
        k = (r["pool"], r["benchmark"], int(r["question_id"]))
        rows[k] = dict(r)
        for m in METHODS:
            if m in r:
                rows[k][m] = int(r[m])
    labels = {}
    for b in S0.BENCH:
        for r in map(json.loads, open(DEC / f"cand_{b}.jsonl")):
            labels[("original", b, int(r["question_id"]), r["model"])] = r.get("match") is True
        for r in map(json.loads, open(WORK / f"data/newpool_labels_{b}.jsonl")):
            labels[("qwen3_glm", b, r["question_id"], r["model"])] = bool(r["match"])
    health = collections.Counter()
    tok = collections.defaultdict(dict)        # (pool, bench, op) -> {qid: (in, out, reas)}
    finish = collections.defaultdict(dict)
    for s, pool in (("P0", "original"), ("P1", "qwen3_glm")):
        path = WORK / f"raw/main_Sal_{s}.jsonl"
        health[f"Sal_{s}_attempts"] = sum(1 for _ in open(path))
        for r in last(path).values():
            pm, why = parse(r["final_text"], r["id_to_model"]) if r["status"] == "ok" else (None, "error")
            k = (pool, r["benchmark"], r["question_id"])
            rows[k]["S_al"] = int(bool(pm) and labels.get((pool, r["benchmark"], r["question_id"], pm), False))
            rows[k]["S_al_pick"] = pm
            rows[k]["S_al_finish"] = r["finish_reason"]
            health[f"Sal_{s}_nonselection:{why}"] += pm is None
            health[f"Sal_{s}_length"] += r["finish_reason"] == "length"
            tok[(pool, r["benchmark"], "S_al")][r["question_id"]] = toks(r)
            finish[(pool, r["benchmark"], "S_al")][r["question_id"]] = (r["finish_reason"], r["status"])
    for k, r in rows.items():
        r.setdefault("S_al", 0)
        health["missing_S_al"] += "S_al_pick" not in r
    # tokens per question for every model-call operation
    for arm in ("B", "C"):
        for r in map(json.loads, open(W3 / f"raw/w6_arm{arm}_main.jsonl")):
            tok[("original", r["benchmark"], arm)][int(r["question_id"])] = toks(r)
            finish[("original", r["benchmark"], arm)][int(r["question_id"])] = (r["finish_reason"], r["status"])
        for r in last(WORK / f"raw/main_{arm}.jsonl").values():
            tok[("qwen3_glm", r["benchmark"], arm)][r["question_id"]] = toks(r)
            finish[("qwen3_glm", r["benchmark"], arm)][r["question_id"]] = (r["finish_reason"], r["status"])
    gen_prompt_tok = {}
    for b in S0.BENCH:
        units = {r["unit"]: r["question_id"] for r in map(json.loads, open(S0.GEN_RAW.parent.parent / "data" / f"sql_{b}_inputs.jsonl"))
                 if r["recipient"] == "qwen3-32b"}
        for r in map(json.loads, open(S0.GEN_RAW / f"sql_{b}_qwen3-32b.jsonl")):
            q = units[r["unit_id"]]
            tok[("original", b, "Generation")][q] = toks({"usage": r["response"]["usage"]})
            finish[("original", b, "Generation")][q] = (r["response"]["choices"][0]["finish_reason"], r["status"])
    for r in last(WORK / "raw/main_G.jsonl").values():
        tok[("qwen3_glm", r["benchmark"], "Generation")][r["question_id"]] = toks(r)
        finish[("qwen3_glm", r["benchmark"], "Generation")][r["question_id"]] = (r["finish_reason"], r["status"])

    with open(OUT / "per_question_all.csv", "w", newline="") as f:
        keys = list({k: None for r in rows.values() for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])

    rng = np.random.default_rng(S0.SEED)

    def boot(by, dbs, wt, draws):
        per = np.array([np.mean(by[d]) for d in dbs]); n = np.array([len(by[d]) for d in dbs], float)
        ww = np.ones(len(dbs)) if wt == "database_equal" else n
        pt = float((ww * per).sum() / ww.sum()); bs = (per[draws] * ww[draws]).sum(1) / ww[draws].sum(1)
        lo, hi = np.percentile(bs, [2.5, 97.5]); p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
        return pt, float(lo), float(hi), p

    levels, con, chg, cost, delta = [], [], [], [], []
    for b in S0.BENCH:
        common = sorted(q for (p, bb, q) in rows if p == "original" and bb == b
                        and ("qwen3_glm", b, q) in rows and (b, q) not in S0.EXCLUDE_COMMON)
        for pool in S0.POOLS:
            for pop in ("full", "common"):
                qs = common if pop == "common" else sorted(q for (p, bb, q) in rows if p == pool and bb == b)
                dbs = sorted({rows[(pool, b, q)]["db_id"] for q in qs})
                draws = np.random.default_rng(S0.SEED).integers(0, len(dbs), size=(S0.DRAWS, len(dbs)))
                for wt in ("database_equal", "question"):
                    for m in METHODS + tuple(f"unreturned_by_{x}" for x in ("Nonempty", "B", "C", "S_al")):
                        by = collections.defaultdict(list)
                        for q in qs:
                            r = rows[(pool, b, q)]
                            v = (r["Oracle"] and not r[m.split("_by_")[1]]) if m.startswith("unreturned") else r[m]
                            by[r["db_id"]].append(float(v))
                        pt, lo, hi, _ = boot(by, dbs, wt, draws)
                        levels.append(dict(benchmark=b, pool=pool, population=pop, weighting=wt, method=m,
                                           questions=len(qs), databases=len(dbs), pct=round(100 * pt, 2),
                                           lo=round(100 * lo, 2), hi=round(100 * hi, 2)))
                    if pop == "full":
                        for a, z in CON:
                            by = collections.defaultdict(list)
                            for q in qs:
                                r = rows[(pool, b, q)]; by[r["db_id"]].append(float(r[a] - r[z]))
                            pt, lo, hi, p = boot(by, dbs, wt, draws)
                            con.append(dict(benchmark=b, pool=pool, population=pop, weighting=wt, contrast=f"{a} - {z}",
                                            questions=len(qs), effect_pp=round(100 * pt, 2), lo_pp=round(100 * lo, 2),
                                            hi_pp=round(100 * hi, 2), p_boot=round(p, 4),
                                            family=("P0" if pool == "original" else "P1") if wt == "database_equal" else None))
                    # database-equal tokens / price per question, every operation
                    if wt == "database_equal" and pop == "common":
                        for op in ("B", "C", "S_al", "Generation"):
                            t = tok[(pool, b, op)]
                            byi, byo, byr = (collections.defaultdict(list) for _ in range(3))
                            for q in qs:
                                d = rows[(pool, b, q)]["db_id"]
                                x = t.get(q, (0, 0, 0))
                                byi[d].append(x[0]); byo[d].append(x[1]); byr[d].append(x[2])
                            mi = float(np.mean([np.mean(byi[d]) for d in dbs])); mo = float(np.mean([np.mean(byo[d]) for d in dbs]))
                            mr = float(np.mean([np.mean(byr[d]) for d in dbs]))
                            fin = finish[(pool, b, op)]
                            row = dict(benchmark=b, pool=pool, population=pop, weighting="database_equal", operation=op,
                                       calls=sum(1 for q in qs if q in t), mean_input=round(mi, 1), mean_output=round(mo, 1),
                                       mean_reasoning=round(mr, 1),
                                       truncated_length=sum(fin.get(q, (None,))[0] == "length" for q in qs),
                                       failed=sum(fin.get(q, (None, "missing"))[1] != "ok" for q in qs))
                            for name, (pi, po) in S0.PRICE.items():
                                row[f"usd_per_1000q_{name}"] = round(1000 * (mi * pi + mo * po) / 1e6, 4)
                            cost.append(row)
                        cost.append(dict(benchmark=b, pool=pool, population=pop, weighting="database_equal",
                                         operation="rules", calls=0, mean_input=0, mean_output=0, mean_reasoning=0,
                                         truncated_length=0, failed=0, usd_per_1000q_deepinfra=0, usd_per_1000q_blended=0))
            # S_al prompt vs matched generation prompt tokens
            d = [tok[(pool, b, "S_al")][q][0] - tok[(pool, b, "Generation")][q][0]
                 for q in tok[(pool, b, "S_al")] if q in tok[(pool, b, "Generation")]]
            delta.append(dict(benchmark=b, pool=pool, n=len(d), mean_prompt_token_delta=round(float(np.mean(d)), 2),
                              min=int(min(d)), max=int(max(d))))
        dbs = sorted({rows[("original", b, q)]["db_id"] for q in common})
        draws = np.random.default_rng(S0.SEED).integers(0, len(dbs), size=(S0.DRAWS, len(dbs)))
        for wt in ("database_equal", "question"):
            for a, z in CHG:
                by = collections.defaultdict(list)
                for q in common:
                    n_, o_ = rows[("qwen3_glm", b, q)], rows[("original", b, q)]
                    by[o_["db_id"]].append(float((n_[a] - n_[z]) - (o_[a] - o_[z])))
                pt, lo, hi, p = boot(by, dbs, wt, draws)
                chg.append(dict(benchmark=b, weighting=wt, contrast=f"{a} - {z}", questions=len(common),
                                change_pp=round(100 * pt, 2), lo_pp=round(100 * lo, 2), hi_pp=round(100 * hi, 2),
                                p_boot=round(p, 4), family="P1_change" if wt == "database_equal" else None))

    def holm(rs):
        srt = sorted(rs, key=lambda r: r["p_boot"]); m, stop = len(srt), False
        for j, r in enumerate(srt):
            rej = (not stop) and r["p_boot"] <= 0.05 / (m - j)
            stop = stop or not rej
            r["holm_reject"], r["holm_family_size"] = rej, m
    for fam in ("P0", "P1"):
        holm([c for c in con if c["family"] == fam])
    holm([c for c in chg if c["family"] == "P1_change"])
    for rs in (con, chg):
        for r in rs:
            r.setdefault("holm_reject", None); r.setdefault("holm_family_size", None)

    # P2 trigger, full population, database-equal
    trig = []
    lv = {(l["benchmark"], l["pool"], l["method"]): l["pct"] for l in levels
          if l["population"] == "common" and l["weighting"] == "database_equal"}
    for b in S0.BENCH:
        orders = {}
        for pool in S0.POOLS:
            orders[pool] = sorted(ORDER_SET, key=lambda m: -lv[(b, pool, m)])
        differ = orders["original"] != orders["qwen3_glm"]
        for pool in S0.POOLS:
            c = next(x for x in con if x["benchmark"] == b and x["pool"] == pool and x["family"]
                     and x["contrast"] == "S_al - Generation")
            near0 = (c["lo_pp"] <= 0 <= c["hi_pp"]) and abs(c["effect_pp"]) < 3
            o = orders[pool]
            gaps = [lv[(b, pool, o[i])] - lv[(b, pool, o[i + 1])] for i in range(len(o) - 1)]
            unstable = differ and min(gaps) < 1
            trig.append(dict(benchmark=b, pool=pool, s_al_minus_gen=c["effect_pp"], ci=[c["lo_pp"], c["hi_pp"]],
                             near_zero=near0, ordering=" > ".join(o), orderings_differ_across_pools=differ,
                             min_adjacent_gap_pp=round(min(gaps), 2), ordering_unstable=unstable,
                             triggered=bool(near0 or unstable)))
    (OUT / "p2_trigger.json").write_text(json.dumps(trig, indent=1) + "\n")

    for name, rs in (("levels_all.csv", levels), ("sal_contrasts.csv", con), ("sal_cross_pool.csv", chg),
                     ("cost_dbequal.csv", cost), ("sal_prompt_token_delta.csv", delta)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list({k: None for r in rs for k in r})); w.writeheader(); w.writerows(rs)
    summ = dict(P0=[c for c in con if c["family"] == "P0"], P1=[c for c in con if c["family"] == "P1"],
                P1_change=[c for c in chg if c["family"] == "P1_change"], p2_trigger=trig,
                prompt_token_delta=delta, health=dict(health))
    (OUT / "summary_sal.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
    print(json.dumps(summ, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
