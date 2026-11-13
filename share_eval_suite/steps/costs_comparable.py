"""Cost table in comparable units for every stage replacement, next to its quality change."""
from __future__ import annotations

import collections
import json
import pathlib
import sys
from typing import Optional, Sequence

from share_eval_suite import stats
from share_eval_suite.steps import _replacement_common as C
from share_eval_suite.steps.decompose import CHESS_ALTS, CHESS_CELL, MODELS_EM, MODELS_SQL, macsql_cell, stage_aggregate

KEEP = "Keep (Decomposer final_sql)"


def gain(row: dict, n: int, incr_tokens: int) -> dict:
    tot, lo, hi = float(row["total_pp"]), float(row["total_lo"]), float(row["total_hi"])
    excl = stats.excludes_zero(lo, hi)
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


def run(out=C.OUT, external_run=C.EXT, chess_run=C.CHESS, verbose: bool = True) -> list:
    out, ext, chess = pathlib.Path(out), pathlib.Path(external_run), pathlib.Path(chess_run)
    out.mkdir(parents=True, exist_ok=True)
    dec = {(r["system"], r["cell"], r["alternative"]): r for r in C.read_table(out / "decomposition.csv")
           if "primary" in r["A_definition"]}
    rows = []

    for mt, mname in MODELS_SQL:
        for b in ("bird", "spider"):
            U = json.load(open(ext / f"output/macsql_{mt}_{b}.json"))["units"]
            cell = macsql_cell(mname, b)
            agg = stage_aggregate(U)
            pre = agg["selector"] + agg["decomposer"]
            st = agg["refiner"]
            it = st["prompt_tokens"] + st["completion_tokens"]
            rows.append(dict(system="MAC-SQL", cell=cell, stage="Refiner", alternative=KEEP, n_units=len(U),
                             prefix_llm_calls=pre["calls"], prefix_prompt_tokens=pre["prompt_tokens"],
                             prefix_completion_tokens=pre["completion_tokens"],
                             prefix_non_llm="Selector + Decomposer only; 6 server pings per cell excluded",
                             incr_llm_calls=st["calls"], incr_prompt_tokens=st["prompt_tokens"],
                             incr_completion_tokens=st["completion_tokens"],
                             incr_total_tokens=it, incr_tokens_per_unit=round(it / len(U), 1),
                             stage_non_llm_overhead=f"executes the Decomposer SQL on every question for its trigger: {len(U)} read-only executions, "
                                                    f"{round(sum(u['keep_seconds'] for u in U), 1)} s re-measured at extraction (30 s cap; not the run's own timing); "
                                                    f"its re-executions of refined SQL are not separately recorded",
                             alt_llm_calls=0, alt_sql_executions=0, alt_sql_seconds=0.0, alt_model_forward_passes=0,
                             recorded_stage_llm_latency_s=round(st["latency_ms"] / 1000, 1),
                             **gain(dec[("MAC-SQL", cell, KEEP)], len(U), it)))

    cq = C.read_table(chess / "output/costs_per_question.csv")
    P = json.load(open(chess / "output/offline/per_question.json"))
    pre, ut = collections.Counter(), collections.Counter()
    for r in cq:
        tgt = ut if r["part"] == "ut_increment" else pre
        for k in ("calls", "prompt_tokens", "completion_tokens"):
            tgt[k] += int(r[k])
        tgt["latency_s"] += float(r["latency_s"])
    n_ex = sum(q["b1"]["n"] for q in P)
    ex_s = round(sum(q["rule_execution_seconds"] for q in P), 2)
    it = ut["prompt_tokens"] + ut["completion_tokens"]
    for alt in CHESS_ALTS:
        rows.append(dict(system="CHESS B1", cell=CHESS_CELL, stage="Unit Tester (from its first call)", alternative=alt,
                         n_units=len(P), prefix_llm_calls=pre["calls"], prefix_prompt_tokens=pre["prompt_tokens"],
                         prefix_completion_tokens=pre["completion_tokens"],
                         prefix_non_llm="IR + CG + revise incl. agent tool choice; recorded LLM latency "
                                        + str(round(pre["latency_s"], 1)) + " s",
                         incr_llm_calls=ut["calls"], incr_prompt_tokens=ut["prompt_tokens"],
                         incr_completion_tokens=ut["completion_tokens"],
                         incr_total_tokens=it, incr_tokens_per_unit=round(it / len(P), 1),
                         stage_non_llm_overhead="UT's own candidate executions are not separately recorded",
                         alt_llm_calls=0, alt_sql_executions=n_ex, alt_sql_seconds=ex_s, alt_model_forward_passes=0,
                         recorded_stage_llm_latency_s=round(ut["latency_s"], 1),
                         **gain(dec[("CHESS B1", CHESS_CELL, alt)], len(P), it)))

    for mt, mname in MODELS_EM:
        c = collections.Counter()
        for f in sorted((ext / f"output/comem_{mt}").glob("*_test.jsonl"), key=str):
            for line in open(f):
                r = json.loads(line)
                s = r.get("select") or {}
                c["n"] += 1
                c["fwd"] += r["ranker_forward"]
                c["rsec"] += r["ranker_seconds"]
                c["extra"] += r.get("control_extra_forward") or 0
                c["calls"] += bool(s)
                c["lat"] += s.get("latency_s") or 0
                c["pt"] += (s.get("usage") or {}).get("prompt_tokens") or 0
                c["ct"] += (s.get("usage") or {}).get("completion_tokens") or 0
        cell = f"{mname} selector, 8 datasets x 400"
        it = c["pt"] + c["ct"]
        rows.append(dict(system="ComEM", cell=cell, stage="Selecting (selector over top-4)", alternative="Ranker_single",
                         n_units=c["n"], prefix_llm_calls=0, prefix_prompt_tokens=0, prefix_completion_tokens=0,
                         prefix_non_llm=f"BM25 blocking + Flan-T5-XL ranking, shared by both branches: {c['fwd']} test forward passes, "
                                        f"{round(c['rsec'], 1)} s recorded",
                         incr_llm_calls=c["calls"], incr_prompt_tokens=c["pt"], incr_completion_tokens=c["ct"],
                         incr_total_tokens=it, incr_tokens_per_unit=round(it / c["n"], 1), stage_non_llm_overhead="none",
                         alt_llm_calls=0, alt_sql_executions=0, alt_sql_seconds=0.0, alt_model_forward_passes=c["extra"],
                         recorded_stage_llm_latency_s=round(c["lat"], 1),
                         **gain(dec[("ComEM", cell, "Ranker_single")], c["n"], it)))

    G = {(r["system"], r["cell"]): r for r in C.read_table(out / "stage_gpu_seconds.csv")}
    for r in rows:
        g = G[(r["system"], r["cell"])]
        r.update(est_gpu_s_point_single_stream=float(g["gpu_s_point"]), est_gpu_s_low_fastest_observed=float(g["gpu_s_low"]),
                 est_gpu_s_high_slowest_single_stream=float(g["gpu_s_high"]), attributed_server_s=float(g["attributed_server_s"]),
                 stage_conc_median=g["stage_conc_median"], prefix_conc_median=g["prefix_conc_median"], alt_est_gpu_s=0.0,
                 latency_crosscheck=("comparable (stage and prefix at median concurrency 1)" if g["stage_conc_median"] == "1.0"
                                     and g["prefix_conc_median"] in ("1.0", "") else "not comparable (batched; use attributed_server_s)"))
        if g.get("prefix_gpu_s_point"):
            for k, src in (("point", "gpu_s_point"), ("low", "gpu_s_low"), ("high", "gpu_s_high"),
                           ("attributed", "attributed_server_s")):
                st = float(g[src])
                pre_ = float(g["prefix_" + src])
                r[f"stage_share_of_llm_gpu_s_{k}_pct"] = round(100 * st / (st + pre_), 1)
        r["direction_robust_across_range"] = float(g["gpu_s_low"]) > 0 and float(g["gpu_s_high"]) > 0
    C.write_table(out / "costs_comparable.csv", rows)
    if verbose:
        for r in rows:
            print({k: r[k] for k in ("system", "cell", "alternative", "incr_llm_calls", "incr_total_tokens",
                                     "total_ci_excludes_0", "extra_correct", "pp_per_1k_incr_tokens_per_unit")})
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = C.parse_args(__doc__.splitlines()[0], argv)
    run(a.out, a.external_run, a.chess_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
