r"""Cost table in comparable units for every replacement contrast (offline; reads recorded logs only).

    python3 share_eval/stage_replacement/costs_comparable.py [--out DIR] ...   # -> <out>/costs_comparable.csv
                                                                              #    (needs decomposition.csv, stage_gpu_seconds.csv)

Per contrast: common prefix; the stage's incremental LLM calls / prompt / completion tokens over its alternative
(the alternative makes 0 LLM calls in every system); the alternative's own non-LLM overhead; recorded seconds
(descriptive only: different hardware, server concurrency and measurement points, never summed into an end-to-end
runtime); estimated GPU-seconds on the same host from throughput.py; and, only where the total's 95% CI excludes 0, accuracy gain per 1k incremental tokens. The tokens are
the observed totals of this run, treated as fixed; the gain interval is the total's interval rescaled.
"""
import collections
import csv
import glob
import json

import pathlib as _pl, sys as _sys
_sys.dont_write_bytecode = True
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import common as K  # noqa: E402

K.configure(__doc__.splitlines()[0])

DEC = {(r["system"], r["cell"], r["alternative"]): r for r in csv.DictReader(open(K.OUT / "decomposition.csv"))
       if "primary" in r["A_definition"]}
ROWS = []


def gain(row, n, incr_tokens):
    tot, lo, hi = float(row["total_pp"]), float(row["total_lo"]), float(row["total_hi"])
    excl = lo > 0 or hi < 0
    extra = int(row["stage_correct"]) - int(row["alt_correct"])
    per_unit_k = incr_tokens / n / 1000.0
    out = dict(total_pp=tot, total_lo=lo, total_hi=hi, total_ci_excludes_0=excl, extra_correct=extra)
    if excl and incr_tokens:
        out.update(extra_correct_per_1M_incr_tokens=round(extra / incr_tokens * 1e6, 3),
                   pp_per_1k_incr_tokens_per_unit=round(tot / per_unit_k, 3),
                   pp_per_1k_lo=round(lo / per_unit_k, 3), pp_per_1k_hi=round(hi / per_unit_k, 3))
    else:
        out.update(extra_correct_per_1M_incr_tokens="not reported (total CI includes 0)",
                   pp_per_1k_incr_tokens_per_unit="", pp_per_1k_lo="", pp_per_1k_hi="")
    return out


# ---------------------------------------------------------------- MAC-SQL
for mt, mname in (("q3", "Qwen3-32B"), ("q72", "Qwen2.5-72B-AWQ")):
    for b in ("bird", "spider"):
        X = json.load(open(K.EXT / f"output/macsql_{mt}_{b}.json"))
        U = X["units"]
        cell = f"{mname}, {'BIRD dev' if b == 'bird' else 'Spider dev'}"
        agg = collections.defaultdict(collections.Counter)
        for u in U:
            for s, v in u["calls_by_stage"].items():
                agg[s].update(v)
        pre = agg["selector"] + agg["decomposer"]
        st = agg["refiner"]
        it = st["prompt_tokens"] + st["completion_tokens"]
        ROWS.append(dict(system="MAC-SQL", cell=cell, stage="Refiner", alternative="Keep (Decomposer final_sql)", n_units=len(U),
                         prefix_llm_calls=pre["calls"], prefix_prompt_tokens=pre["prompt_tokens"], prefix_completion_tokens=pre["completion_tokens"],
                         prefix_non_llm="Selector + Decomposer only; 6 server pings per cell excluded",
                         incr_llm_calls=st["calls"], incr_prompt_tokens=st["prompt_tokens"], incr_completion_tokens=st["completion_tokens"],
                         incr_total_tokens=it, incr_tokens_per_unit=round(it / len(U), 1),
                         stage_non_llm_overhead=f"executes the Decomposer SQL on every question for its trigger: {len(U)} read-only executions, "
                                                f"{round(sum(u['keep_seconds'] for u in U), 1)} s re-measured at extraction (30 s cap; not the run's own timing); "
                                                f"its re-executions of refined SQL are not separately recorded",
                         alt_llm_calls=0, alt_sql_executions=0, alt_sql_seconds=0.0, alt_model_forward_passes=0,
                         recorded_stage_llm_latency_s=round(st["latency_ms"] / 1000, 1),
                         **gain(DEC[("MAC-SQL", cell, "Keep (Decomposer final_sql)")], len(U), it)))

# ---------------------------------------------------------------- CHESS
cq = list(csv.DictReader(open(K.CHESS / "output/costs_per_question.csv")))
P = json.load(open(K.CHESS / "output/offline/per_question.json"))
pre, ut = collections.Counter(), collections.Counter()
for r in cq:
    tgt = ut if r["part"] == "ut_increment" else pre
    for k in ("calls", "prompt_tokens", "completion_tokens"):
        tgt[k] += int(r[k])
    tgt["latency_s"] += float(r["latency_s"])
n_ex = sum(q["b1"]["n"] for q in P)
ex_s = round(sum(q["rule_execution_seconds"] for q in P), 2)
it = ut["prompt_tokens"] + ut["completion_tokens"]
for alt in ("SC_official", "Valid_vote", "Nonempty_vote"):
    ROWS.append(dict(system="CHESS B1", cell="Qwen3-32B, BIRD SDS 147", stage="Unit Tester (from its first call)", alternative=alt, n_units=len(P),
                     prefix_llm_calls=pre["calls"], prefix_prompt_tokens=pre["prompt_tokens"], prefix_completion_tokens=pre["completion_tokens"],
                     prefix_non_llm="IR + CG + revise incl. agent tool choice; recorded LLM latency " + str(round(pre["latency_s"], 1)) + " s",
                     incr_llm_calls=ut["calls"], incr_prompt_tokens=ut["prompt_tokens"], incr_completion_tokens=ut["completion_tokens"],
                     incr_total_tokens=it, incr_tokens_per_unit=round(it / len(P), 1),
                     stage_non_llm_overhead="UT's own candidate executions are not separately recorded",
                     alt_llm_calls=0, alt_sql_executions=n_ex, alt_sql_seconds=ex_s, alt_model_forward_passes=0,
                     recorded_stage_llm_latency_s=round(ut["latency_s"], 1),
                     **gain(DEC[("CHESS B1", "Qwen3-32B, BIRD SDS 147", alt)], len(P), it)))

# ---------------------------------------------------------------- ComEM
for mt, mname in (("q3", "Qwen3-32B"), ("q7", "Qwen2.5-7B")):
    c = collections.Counter()
    for f in sorted(glob.glob(str(K.EXT / f"output/comem_{mt}/*_test.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            s = r.get("select") or {}
            c["n"] += 1; c["fwd"] += r["ranker_forward"]; c["rsec"] += r["ranker_seconds"]
            c["extra"] += r.get("control_extra_forward") or 0
            c["calls"] += bool(s); c["lat"] += s.get("latency_s") or 0
            c["pt"] += (s.get("usage") or {}).get("prompt_tokens") or 0
            c["ct"] += (s.get("usage") or {}).get("completion_tokens") or 0
    cell = f"{mname} selector, 8 datasets x 400"
    it = c["pt"] + c["ct"]
    ROWS.append(dict(system="ComEM", cell=cell, stage="Selecting (selector over top-4)", alternative="Ranker_single", n_units=c["n"],
                     prefix_llm_calls=0, prefix_prompt_tokens=0, prefix_completion_tokens=0,
                     prefix_non_llm=f"BM25 blocking + Flan-T5-XL ranking, shared by both branches: {c['fwd']} test forward passes, "
                                    f"{round(c['rsec'], 1)} s recorded",
                     incr_llm_calls=c["calls"], incr_prompt_tokens=c["pt"], incr_completion_tokens=c["ct"],
                     incr_total_tokens=it, incr_tokens_per_unit=round(it / c["n"], 1), stage_non_llm_overhead="none",
                     alt_llm_calls=0, alt_sql_executions=0, alt_sql_seconds=0.0, alt_model_forward_passes=c["extra"],
                     recorded_stage_llm_latency_s=round(c["lat"], 1),
                     **gain(DEC[("ComEM", cell, "Ranker_single")], c["n"], it)))

# estimated GPU-seconds on the same host (code/throughput.py must run first); the alternative's GPU-seconds are 0
G = {(r["system"], r["cell"]): r for r in csv.DictReader(open(K.OUT / "stage_gpu_seconds.csv"))}
for r in ROWS:
    g = G[(r["system"], r["cell"])]
    r.update(est_gpu_s_point_single_stream=float(g["gpu_s_point"]), est_gpu_s_low_fastest_observed=float(g["gpu_s_low"]),
             est_gpu_s_high_slowest_single_stream=float(g["gpu_s_high"]), attributed_server_s=float(g["attributed_server_s"]),
             stage_conc_median=g["stage_conc_median"], prefix_conc_median=g["prefix_conc_median"], alt_est_gpu_s=0.0,
             latency_crosscheck=("comparable (stage and prefix at median concurrency 1)" if g["stage_conc_median"] == "1.0"
                                 and g["prefix_conc_median"] in ("1.0", "") else "not comparable (batched; use attributed_server_s)"))
    if g.get("prefix_gpu_s_point"):
        for k, src in (("point", "gpu_s_point"), ("low", "gpu_s_low"), ("high", "gpu_s_high"), ("attributed", "attributed_server_s")):
            st = float(g[src]); pre = float(g["prefix_" + src]) if k != "attributed" else float(g["prefix_attributed_server_s"])
            r[f"stage_share_of_llm_gpu_s_{k}_pct"] = round(100 * st / (st + pre), 1)
    r["direction_robust_across_range"] = float(g["gpu_s_low"]) > 0 and float(g["gpu_s_high"]) > 0
FIELDS = list({k: None for r in ROWS for k in r})
with open(K.OUT / "costs_comparable.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(ROWS)
for r in ROWS:
    print({k: r[k] for k in ("system", "cell", "alternative", "prefix_llm_calls", "prefix_prompt_tokens", "prefix_completion_tokens",
                             "incr_llm_calls", "incr_prompt_tokens", "incr_completion_tokens", "incr_tokens_per_unit", "alt_sql_executions",
                             "alt_sql_seconds", "alt_model_forward_passes", "recorded_stage_llm_latency_s", "total_ci_excludes_0", "extra_correct",
                             "extra_correct_per_1M_incr_tokens", "pp_per_1k_incr_tokens_per_unit", "pp_per_1k_lo", "pp_per_1k_hi")})
    print("   ", r["prefix_non_llm"], "|", r["stage_non_llm_overhead"])
