"""Compare SQL generation with fixed-pool selectors using saved responses and read-only benchmark databases. No model calls."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]
OUT = WORK / "output"
RAW = OUT / "raw"
DATA = WORK / "data"
#: previous round, read-only: its Spider arms are the comparison partners for E1
PREV = WORK
EXTRA_DATA = []
EXTRA_RAW = []
STAGE = REPO / "experiments/04_sql_selection_generation"
for _p in (str(REPO / "analysis/selectors_and_dependence/code"),
           str(REPO / "analysis/framework_value/code"),
           str(STAGE / "code"), str(REPO / "experiments/06_cost_resources/code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import b1_selectors as B1                                                  # noqa: E402
import collect_sql_arms as A                                              # noqa: E402

BOOT_DRAWS = 10000
BOOT_SEED = 20261028
SELECTOR_NAMES = ("released", "valid_majority", "nonempty_majority")


def sqlmod(bench: str):
    if bench == "bird":
        import sql_common as S
        return S
    import spider_common as S
    return S


def db_cluster_boot(by_db: Dict[str, List[float]], rng) -> Dict[str, float]:
    keys = sorted(by_db)
    per = np.array([float(np.mean(by_db[k])) for k in keys])
    draws = per[rng.integers(0, len(per), size=(BOOT_DRAWS, len(per)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return dict(mean=float(per.mean()), lo=float(lo), hi=float(hi), n_databases=len(keys),
                p=float(min(1.0, max(p, 1.0 / BOOT_DRAWS))))


def loo(by_db: Dict[str, List[float]]) -> Dict[str, float]:
    keys = sorted(by_db)
    vals = {}
    for drop in keys:
        rest = [float(np.mean(by_db[k])) for k in keys if k != drop]
        vals[drop] = float(np.mean(rest)) if rest else float("nan")
    return dict(min=min(vals.values()), max=max(vals.values()),
                most_influential_database=min(vals, key=lambda k: vals[k]))


def holm(ps: List[float], alpha: float = 0.05) -> List[bool]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out = [False] * m
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break
    return out


def score_benchmark(bench: str) -> Dict[str, Any]:
    S = sqlmod(bench)
    # the combined file plus any per-recipient file, so a recipient added later is still joined
    inputs: Dict[str, Dict[str, Any]] = {}
    for d in [DATA] + EXTRA_DATA:
        for p in sorted(d.glob(f"sql_{bench}_*.jsonl")):
            for l in open(p):
                r = json.loads(l)
                inputs[r["unit"]] = r
    peers = B1.load_peers(bench)
    labels = B1.load_cand_labels(bench)

    # ---------------------------------------------------- selector side, same candidate pool
    sel_by_db: Dict[str, Dict[str, List[float]]] = {s: collections.defaultdict(list)
                                                    for s in SELECTOR_NAMES}
    sel_per_q: Dict[str, Dict[int, Optional[bool]]] = {s: {} for s in SELECTOR_NAMES}
    db_of: Dict[int, str] = {}
    for u in inputs.values():
        db_of[u["question_id"]] = u["db_id"]
    for qid in sorted(set(db_of)):
        got = peers.get(qid) or {}
        live = [got[m] for m in A.PEER_MODELS if m in got and (got[m].get("sql") or "").strip()]
        if len(live) < 2:
            continue
        for s in SELECTOR_NAMES:
            pick = B1.SELECTORS[s](live, None)
            ok = labels.get((qid, pick["model"])) if pick else None
            sel_per_q[s][qid] = ok
            sel_by_db[s][db_of[qid]].append(float(ok is True))

    # ---------------------------------------------------- generation side, executed locally
    gen: Dict[str, Dict[int, Dict[str, Any]]] = {}
    health: List[Dict[str, Any]] = []
    seen_models = set()
    raw_paths = [q for d in [RAW] + EXTRA_RAW for q in sorted(d.glob(f"sql_{bench}_*.jsonl"))]
    for path in raw_paths:
        model = path.name[len(f"sql_{bench}_"):-len(".jsonl")]
        if model in seen_models:
            continue
        seen_models.add(model)
        rows: Dict[int, Dict[str, Any]] = {}
        statuses = collections.Counter()
        for line in open(path):
            r = json.loads(line)
            statuses[r["status"]] += 1
            if r.get("status") != "ok":
                continue
            u = inputs.get(r["unit_id"])
            if u is None:
                continue
            ch = r["response"]["choices"][0]
            content = ch["message"].get("content") or ""
            sql = S.parse_sql(content)
            ex = (S.execute(u["db_id"], sql) if sql
                  else {"status": "error", "error": "no sql parsed"})
            ok, reason = None, None
            if sql:
                try:
                    q = next(x for x in S.load_questions()
                             if int(x["question_id"]) == u["question_id"])
                    v = S.execution_match(u["db_id"], sql, q["SQL"])
                    ok, reason = bool(v["match"]), v.get("reason")
                except Exception as exc:                               # noqa: BLE001
                    reason = f"scorer_error: {type(exc).__name__}: {exc}"
            usage = r["response"].get("usage") or {}
            det = usage.get("completion_tokens_details") or {}
            rows[u["question_id"]] = dict(
                question_id=u["question_id"], db_id=u["db_id"], sql=sql,
                parsed=sql is not None, exec_status=ex.get("status"),
                correct=bool(ok is True), scorer_reason=reason,
                truncated=ch.get("finish_reason") == "length",
                empty_final=not content.strip(),
                prompt_tokens=usage.get("prompt_tokens") or 0,
                completion_tokens=usage.get("completion_tokens") or 0,
                reasoning_tokens=det.get("reasoning_tokens") or 0,
                latency_s=r.get("latency_s") or 0.0)
        gen[model] = rows
        if rows:
            rs = list(rows.values())
            health.append(dict(
                benchmark=bench, recipient=model, requests=len(rs),
                statuses=dict(statuses),
                unparsed_share=round(sum(1 for r in rs if not r["parsed"]) / len(rs), 5),
                exec_error_share=round(sum(1 for r in rs
                                           if r["exec_status"] != "ok") / len(rs), 5),
                truncated_share=round(sum(1 for r in rs if r["truncated"]) / len(rs), 5),
                empty_final_share=round(sum(1 for r in rs if r["empty_final"]) / len(rs), 5),
                mean_prompt_tokens=round(float(np.mean([r["prompt_tokens"] for r in rs])), 1),
                mean_completion_tokens=round(float(np.mean([r["completion_tokens"] for r in rs])), 1),
                mean_reasoning_tokens=round(float(np.mean([r["reasoning_tokens"] for r in rs])), 1),
                mean_latency_s=round(float(np.mean([r["latency_s"] for r in rs])), 3)))
        print(f"{bench} {model}: {len(rows)} scored, statuses={dict(statuses)}", flush=True)

    # ---------------------------------------------------- levels and paired contrasts
    levels, contrasts = [], []
    for s in SELECTOR_NAMES:
        if not sel_by_db[s]:
            continue
        rng = np.random.default_rng(BOOT_SEED)
        v = db_cluster_boot(sel_by_db[s], rng)
        levels.append(dict(benchmark=bench, method=f"select:{s}", extra_model_calls=0,
                           questions=sum(len(x) for x in sel_by_db[s].values()),
                           databases=v["n_databases"],
                           accuracy=round(100 * v["mean"], 3),
                           lo=round(100 * v["lo"], 3), hi=round(100 * v["hi"], 3)))
    for model, rows in sorted(gen.items()):
        if not rows:
            continue
        by_db = collections.defaultdict(list)
        for r in rows.values():
            by_db[r["db_id"]].append(float(r["correct"]))
        rng = np.random.default_rng(BOOT_SEED)
        v = db_cluster_boot(by_db, rng)
        levels.append(dict(benchmark=bench, method=f"generate:{model}",
                           extra_model_calls=len(rows), questions=len(rows),
                           databases=v["n_databases"],
                           accuracy=round(100 * v["mean"], 3),
                           lo=round(100 * v["lo"], 3), hi=round(100 * v["hi"], 3)))
        for s in SELECTOR_NAMES:
            per = sel_per_q[s]
            shared = sorted(set(rows) & set(per))
            if not shared:
                continue
            d_by_db = collections.defaultdict(list)
            ben = harm = 0
            for qid in shared:
                a = float(rows[qid]["correct"])
                b = float(per[qid] is True)
                d_by_db[rows[qid]["db_id"]].append(a - b)
                ben += int(a > b)
                harm += int(b > a)
            rng = np.random.default_rng(BOOT_SEED)
            v = db_cluster_boot(d_by_db, rng)
            l = loo(d_by_db)
            contrasts.append(dict(
                benchmark=bench, recipient=model, contrast=f"generate - select:{s}",
                questions=len(shared), databases=v["n_databases"],
                effect_pp=round(100 * v["mean"], 3),
                lo_pp=round(100 * v["lo"], 3), hi_pp=round(100 * v["hi"], 3),
                ci_half_width_pp=round(100 * (v["hi"] - v["lo"]) / 2, 3),
                ci_excludes_zero=bool(v["lo"] > 0 or v["hi"] < 0), p_boot=v["p"],
                benefited_questions=ben, harmed_questions=harm,
                unchanged_questions=len(shared) - ben - harm,
                residual_errors=sum(1 for q in shared if not rows[q]["correct"]),
                loo_min_pp=round(100 * l["min"], 3), loo_max_pp=round(100 * l["max"], 3),
                loo_most_influential_database=l["most_influential_database"]))
    return dict(levels=levels, contrasts=contrasts, health=health)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    benches = [a.benchmark] if a.benchmark else ["bird", "spider"]
    levels, contrasts, health = [], [], []
    for b in benches:
        if not [q for d in [DATA] + EXTRA_DATA for q in d.glob(f"sql_{b}_*.jsonl")]:
            print(f"skip {b}: no input file in either round")
            continue
        if not [q for d in [RAW] + EXTRA_RAW for q in d.glob(f"sql_{b}_*.jsonl")]:
            print(f"skip {b}: no raw output yet")
            continue
        r = score_benchmark(b)
        levels += r["levels"]
        contrasts += r["contrasts"]
        health += r["health"]
    # Multiplicity is corrected inside each DECLARED family, not across everything that happens to
    # be in one run. E1 added a new recipient on Spider; its plan and this round's frozen design
    # both say the new family must not revise the previous round's family, so the previous 15
    # contrasts keep their own correction and the 3 new ones form this round's family. The merged
    # 18-contrast version is computed too, but only as a reported sensitivity.
    if contrasts:
        for c in contrasts:
            c["family"] = ("e1_gemma_spider" if (c["benchmark"] == "spider"
                                                 and c["recipient"] == "gemma-4-26b")
                           else "prior_round_15")
        for fam in sorted({c["family"] for c in contrasts}):
            members = [c for c in contrasts if c["family"] == fam]
            for c, f in zip(members, holm([c["p_boot"] for c in members])):
                c["holm_significant_in_declared_family"] = bool(f)
                c["declared_family_size"] = len(members)
        for c, f in zip(contrasts, holm([c["p_boot"] for c in contrasts])):
            c["holm_merged_sensitivity_only"] = bool(f)
            c["merged_sensitivity_family_size"] = len(contrasts)
    summary = dict(levels=levels, contrasts=contrasts, health=health,
                   design=dict(selectors=list(SELECTOR_NAMES), independent_unit="database",
                               boot_draws=BOOT_DRAWS, boot_seed=BOOT_SEED,
                               same_pool=("both sides read the identical stored candidate pool; "
                                          "selection spends no extra model call, generation spends "
                                          "one per question"),
                               counted_separately="unparsed, execution error, wrong",
                               multiplicity=("Holm inside each declared family: "
                                             "prior_round_15 keeps the previous round's family, "
                                             "e1_gemma_spider is this round's new family of 3. "
                                             "The merged 18-contrast correction is a reported "
                                             "sensitivity and is not the main convention.")))
    (OUT / "sq2_scored.json").write_text(json.dumps(summary, indent=1, sort_keys=True),
                                         encoding="utf-8")
    for name, rows in (("sq2_levels", levels), ("sq2_contrasts", contrasts),
                       ("sq2_health", health)):
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print("\nlevels")
    for r in levels:
        print(f"  {r['benchmark']:8s} {r['method']:28s} acc {r['accuracy']:6.2f} "
              f"[{r['lo']:.2f},{r['hi']:.2f}] n={r['questions']} dbs={r['databases']} "
              f"extra_calls={r['extra_model_calls']}")
    print("\ncontrasts")
    for r in contrasts:
        hk = [k for k in r if k.startswith("holm_")][0]
        print(f"  {r['benchmark']:8s} {r['recipient']:14s} {r['contrast']:34s} "
              f"{r['effect_pp']:+7.2f} pp [{r['lo_pp']:+.2f},{r['hi_pp']:+.2f}] "
              f"holm={r[hk]} loo [{r['loo_min_pp']:+.2f},{r['loo_max_pp']:+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
