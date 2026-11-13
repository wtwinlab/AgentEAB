r"""Aggregate both pools. Offline: no model call, no database (run score_generation.py first).

    python3 code/score.py

Database-equal primary, question-weighted beside it; cluster
bootstrap over databases, 10,000 draws, seed 20261105. Primary family: 8 cross-pool changes of
within-pool contrasts (common population). Secondary family: the same 8 contrasts inside the
Qwen3/GLM pool.

Writes output/per_question.csv, levels.csv, within_pool_contrasts.csv, cross_pool_changes.csv,
ordering.csv, cost.csv, health.json, summary.json.
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
W3 = REPO / "experiments/15_stronger_peers"
sys.path.insert(0, str(W3 / "code"))
import w6_selector as Asel                                                 # noqa: E402

OUT = WORK / "output"
DRAWS, SEED = 10000, 20261105
BENCH = ("bird", "spider")
POOLS = ("original", "qwen3_glm")
METHODS = ("Oracle", "Original", "Valid", "Nonempty", "B", "C", "Generation")
CONTRASTS = (("B", "Nonempty"), ("C", "Nonempty"), ("Generation", "Nonempty"), ("Generation", "C"))
EXCLUDE_COMMON = {("bird", 518), ("bird", 701)}        # original-pool candidates unlabelled
PRICE = dict(deepinfra=(0.08, 0.28), blended=(0.90, 0.90))   # USD per 1M input / output tokens
STRONG_RULES = REPO / "data/derived/sql_rule_pairing/output/strongpool_per_question.csv"
GEN_RAW = REPO / "experiments/08_cross_family/output/raw"
PEERS = REPO / "data/records/three_domain_four_question"


def last_by_unit(path):
    last = {}
    for r in map(json.loads, open(path)):
        if r["unit_id"] not in last or r["status"] == "ok":
            last[r["unit_id"]] = r
    return last


def usage_row(r, arm):
    u = (r or {}).get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    return dict(input=u.get("prompt_tokens") or 0, output=u.get("completion_tokens") or 0,
                reasoning=det.get("reasoning_tokens") or 0,
                finish=(r or {}).get("finish_reason"), status=(r or {}).get("status", "missing"))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rows = {}            # (pool, bench, qid) -> dict
    cost = collections.defaultdict(list)    # (pool, bench, op) -> usage rows
    health = collections.Counter()

    # ---- original pool: reused per-question outcomes and raw usage
    for r in csv.DictReader(open(W3 / "output/w6_arms_per_question.csv")):
        k = ("original", r["benchmark"], int(r["question_id"]))
        rows[k] = dict(pool="original", benchmark=r["benchmark"], question_id=k[2], db_id=r["db_id"],
                       **{m: int(r[m]) for m in METHODS},
                       B_pick=r["B_pick"], C_pick=r["C_pick"], B_finish=r["B_finish"], C_finish=r["C_finish"])
    for arm in ("B", "C"):
        for r in map(json.loads, open(W3 / f"raw/w6_arm{arm}_main.jsonl")):
            cost[("original", r["benchmark"], arm)].append(usage_row(r, arm))
    for b in BENCH:
        for r in map(json.loads, open(GEN_RAW / f"sql_{b}_qwen3-32b.jsonl")):
            u = r["response"]["usage"] if r.get("response") else {}
            det = u.get("completion_tokens_details") or {}
            cost[("original", b, "Generation")].append(dict(
                input=u.get("prompt_tokens") or 0, output=u.get("completion_tokens") or 0,
                reasoning=det.get("reasoning_tokens") or 0,
                finish=r["response"]["choices"][0]["finish_reason"], status=r["status"]))

    # ---- Qwen3/GLM pool
    labels = {}
    for b in BENCH:
        for r in map(json.loads, open(WORK / f"data/newpool_labels_{b}.jsonl")):
            labels[(b, r["question_id"], r["model"])] = bool(r["match"])
    strong = {(r["benchmark"], int(r["question_id"])): r for r in csv.DictReader(open(STRONG_RULES))}
    gen = {}
    for b in BENCH:
        for r in map(json.loads, open(OUT / f"newpool_generation_{b}.jsonl")):
            gen[(b, r["question_id"])] = r
    picks = {}
    for arm in ("B", "C"):
        last = last_by_unit(WORK / f"raw/main_{arm}.jsonl")
        health[f"{arm}_attempts"] = sum(1 for _ in open(WORK / f"raw/main_{arm}.jsonl"))
        for r in last.values():
            cid, why = Asel.parse(r["final_text"], r["id_to_model"]) if r["status"] == "ok" else (None, "error")
            picks[(arm, r["benchmark"], r["question_id"])] = (r["id_to_model"].get(cid) if cid else None, why, r)
            cost[("qwen3_glm", r["benchmark"], arm)].append(usage_row(r, arm))
    lastG = last_by_unit(WORK / "raw/main_G.jsonl")
    health["G_attempts"] = sum(1 for _ in open(WORK / "raw/main_G.jsonl"))
    for r in lastG.values():
        cost[("qwen3_glm", r["benchmark"], "Generation")].append(usage_row(r, "G"))
    for (b, q), s in strong.items():
        row = dict(pool="qwen3_glm", benchmark=b, question_id=q, db_id=s["db_id"],
                   Oracle=int(s["pool_had_correct"]), Original=int(s["released_correct"]),
                   Valid=int(s["valid_majority_correct"]), Nonempty=int(s["nonempty_majority_correct"]))
        for arm in ("B", "C"):
            pm, why, rec = picks.get((arm, b, q), (None, "missing", None))
            row[arm] = int(bool(pm) and labels.get((b, q, pm), False))
            row[f"{arm}_pick"] = pm
            row[f"{arm}_finish"] = (rec or {}).get("finish_reason")
            health[f"{arm}_nonselection:{(why or '').split(':')[0]}"] += pm is None
        g = gen.get((b, q))
        row["Generation"] = int(g["correct"]) if g else 0
        health["G_missing"] += g is None
        health["G_unparsed"] += bool(g) and not g["parsed"]
        rows[("qwen3_glm", b, q)] = row
    for k, r in rows.items():
        health[f"{k[0]}_violation_op_correct_without_oracle"] += sum(
            r[m] and not r["Oracle"] for m in ("Original", "Valid", "Nonempty", "B", "C"))

    with open(OUT / "per_question.csv", "w", newline="") as f:
        keys = list({k: None for r in rows.values() for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])

    rng = np.random.default_rng(SEED)
    levels, within, change, order = [], [], [], []

    def stat(vals_by_db, dbs, wt, draws):
        per = np.array([np.mean(vals_by_db[d]) for d in dbs])
        n = np.array([len(vals_by_db[d]) for d in dbs], float)
        ww = np.ones(len(dbs)) if wt == "database_equal" else n
        pt = float((ww * per).sum() / ww.sum())
        bs = (per[draws] * ww[draws]).sum(1) / ww[draws].sum(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
        return pt, float(lo), float(hi), p

    for b in BENCH:
        common = sorted(q for (p, bb, q) in rows if p == "original" and bb == b
                        and ("qwen3_glm", b, q) in rows and (b, q) not in EXCLUDE_COMMON)
        dbs = sorted({rows[("original", b, q)]["db_id"] for q in common})
        draws = rng.integers(0, len(dbs), size=(DRAWS, len(dbs)))
        for pool in POOLS:
            for popname in ("full", "common"):
                qs = common if popname == "common" else sorted(q for (p, bb, q) in rows if p == pool and bb == b)
                pdbs = sorted({rows[(pool, b, q)]["db_id"] for q in qs})
                pdraws = draws if pdbs == dbs else rng.integers(0, len(pdbs), size=(DRAWS, len(pdbs)))
                for wt in ("database_equal", "question"):
                    for m in METHODS + ("unreturned_by_Nonempty", "unreturned_by_B", "unreturned_by_C"):
                        by = collections.defaultdict(list)
                        for q in qs:
                            r = rows[(pool, b, q)]
                            v = (r["Oracle"] and not r[m.split("_by_")[1]]) if m.startswith("unreturned") else r[m]
                            by[r["db_id"]].append(float(v))
                        pt, lo, hi, _ = stat(by, pdbs, wt, pdraws)
                        levels.append(dict(benchmark=b, pool=pool, population=popname, weighting=wt, method=m,
                                           questions=len(qs), databases=len(pdbs), accuracy_pct=round(100 * pt, 2),
                                           lo_pct=round(100 * lo, 2), hi_pct=round(100 * hi, 2)))
        for wt in ("database_equal", "question"):
            for pool in POOLS:
                for a, z in CONTRASTS:
                    by = collections.defaultdict(list)
                    for q in common:
                        r = rows[(pool, b, q)]
                        by[r["db_id"]].append(float(r[a] - r[z]))
                    pt, lo, hi, p = stat(by, dbs, wt, draws)
                    within.append(dict(benchmark=b, pool=pool, population="common", weighting=wt,
                                       contrast=f"{a} - {z}", questions=len(common), databases=len(dbs),
                                       effect_pp=round(100 * pt, 2), lo_pp=round(100 * lo, 2),
                                       hi_pp=round(100 * hi, 2), p_boot=round(p, 4),
                                       ci_excludes_zero=bool(lo > 0 or hi < 0)))
            for a, z in CONTRASTS:
                by = collections.defaultdict(list)
                for q in common:
                    n_, o_ = rows[("qwen3_glm", b, q)], rows[("original", b, q)]
                    by[o_["db_id"]].append(float((n_[a] - n_[z]) - (o_[a] - o_[z])))
                pt, lo, hi, p = stat(by, dbs, wt, draws)
                change.append(dict(benchmark=b, weighting=wt, contrast=f"{a} - {z}",
                                   questions=len(common), databases=len(dbs),
                                   change_pp=round(100 * pt, 2), lo_pp=round(100 * lo, 2),
                                   hi_pp=round(100 * hi, 2), p_boot=round(p, 4),
                                   ci_excludes_zero=bool(lo > 0 or hi < 0)))

    def holm(rs):
        srt = sorted(rs, key=lambda r: r["p_boot"]); m, stop = len(srt), False
        for j, r in enumerate(srt):
            rej = (not stop) and r["p_boot"] <= 0.05 / (m - j)
            stop = stop or not rej
            r["holm_reject"], r["holm_family_size"] = rej, m
    holm([c for c in change if c["weighting"] == "database_equal"])
    holm([c for c in within if c["weighting"] == "database_equal" and c["pool"] == "qwen3_glm"])
    for rs in (change, within):
        for r in rs:
            r.setdefault("holm_reject", None); r.setdefault("holm_family_size", None)

    for b in BENCH:
        for pool in POOLS:
            lv = {l["method"]: l["accuracy_pct"] for l in levels if l["benchmark"] == b and l["pool"] == pool
                  and l["population"] == "common" and l["weighting"] == "database_equal"
                  and l["method"] in METHODS and l["method"] != "Oracle"}
            ranked = sorted(lv, key=lambda m: -lv[m])
            order.append(dict(benchmark=b, pool=pool, ranking=" > ".join(f"{m} {lv[m]:.2f}" for m in ranked)))

    costrows = []
    for (pool, b, op), us in sorted(cost.items()):
        n = len(us)
        mi, mo = np.mean([u["input"] for u in us]), np.mean([u["output"] for u in us])
        row = dict(pool=pool, benchmark=b, operation=op, calls=n,
                   mean_input=round(float(mi), 1), mean_output=round(float(mo), 1),
                   mean_reasoning=round(float(np.mean([u["reasoning"] for u in us])), 1),
                   truncated_length=sum(u["finish"] == "length" for u in us),
                   failed=sum(u["status"] != "ok" for u in us))
        for name, (pi, po) in PRICE.items():
            row[f"usd_per_1000q_{name}"] = round(1000 * (mi * pi + mo * po) / 1e6, 4)
        costrows.append(row)
    for pool in POOLS:
        for b in BENCH:
            costrows.append(dict(pool=pool, benchmark=b, operation="rules (Original/Valid/Nonempty)", calls=0,
                                 mean_input=0, mean_output=0, mean_reasoning=0, truncated_length=0, failed=0,
                                 usd_per_1000q_deepinfra=0, usd_per_1000q_blended=0))
    # candidate acquisition, reported apart
    acq = []
    for b in BENCH:
        ci = co = n = 0
        for r in map(json.loads, open(PEERS / b / "peers.jsonl")):
            if r.get("role") in ("peer", "extra") and r["model"] in Asel.POOL:
                n += 1; ci += r.get("prompt_tokens") or 0; co += r.get("completion_tokens") or 0
        lg = json.loads((WORK / f"logs/build_{b}.json").read_text())["newpool_acquisition"]
        acq.append(dict(benchmark=b, pool="original", candidate_calls=n, input_tokens=ci, output_tokens=co))
        acq.append(dict(benchmark=b, pool="qwen3_glm", candidate_calls=lg["calls"], input_tokens=lg["input"],
                        output_tokens=lg["output"]))

    for name, rs in (("levels.csv", levels), ("within_pool_contrasts.csv", within),
                     ("cross_pool_changes.csv", change), ("ordering.csv", order), ("cost.csv", costrows),
                     ("acquisition.csv", acq)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list({k: None for r in rs for k in r})); w.writeheader(); w.writerows(rs)
    (OUT / "health.json").write_text(json.dumps(dict(health), indent=1) + "\n")
    summ = dict(primary_family=[c for c in change if c["weighting"] == "database_equal"],
                secondary_family=[c for c in within if c["weighting"] == "database_equal" and c["pool"] == "qwen3_glm"],
                ordering=order, health=dict(health))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
