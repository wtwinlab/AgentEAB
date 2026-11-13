"""Same-host throughput and estimated GPU-seconds per stage, from the per-call proxy records.

Same outputs as `share_eval/stage_replacement/throughput.py`.

    python3 -m share_eval_suite.steps.throughput [--out DIR] [--external-run DIR] [--chess-run DIR]
    # -> <out>/throughput_by_server.csv, <out>/stage_gpu_seconds.csv, <out>/throughput.json

Every LLM call is logged by the proxy with its start and end time, latency and token usage. Servers (one
vLLM instance each, same host and cards): CHESS Qwen3-32B chess_g1..g3; MAC-SQL Qwen3-32B q3_g1..g3
(ComEM's Qwen3-32B selector shares q3_g3) and Qwen2.5-72B-AWQ q72_g0..g3; ComEM Qwen2.5-7B q7_g3.

Per server:
  * concurrency of a call = the calls of the same server in flight during it, itself included;
  * single-stream rates: least squares of latency = t0 + p/Rp + c/Rd over calls with concurrency 1
    (at least 30 such calls) -> prefill rate Rp and decoding rate Rd;
  * peak batched rates: 60-s bins, prompt tokens in the bin where the call starts, completion tokens
    spread uniformly over the call; the busiest bin (Rp_peak, Rd_peak) and the median busy bin.
Estimated GPU-seconds of a stage = p/Rp + c/Rd at its model's rates: the point uses the median
single-stream fit, the low end the fastest rate seen for the model (single-stream fit or busiest bin),
the high end the slowest single-stream fit. Attributed server-seconds split every instant a server has
n calls in flight 1/n per call and sum the stage's share.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from share_eval_suite.reports import cost
from share_eval_suite.steps import _replacement_common as C

REFINER_SIG = "When executing SQL below, some errors occurred"
BIN = cost.BIN
MIN_SINGLE = cost.MIN_SINGLE
STAGES = [("MAC-SQL", "Qwen3-32B, BIRD dev", "MACSQL_q3_bird_refiner", "MACSQL_q3_bird_prefix", "Qwen3-32B"),
          ("MAC-SQL", "Qwen3-32B, Spider dev", "MACSQL_q3_spider_refiner", "MACSQL_q3_spider_prefix", "Qwen3-32B"),
          ("MAC-SQL", "Qwen2.5-72B-AWQ, BIRD dev", "MACSQL_q72_bird_refiner", "MACSQL_q72_bird_prefix", "Qwen2.5-72B-AWQ"),
          ("MAC-SQL", "Qwen2.5-72B-AWQ, Spider dev", "MACSQL_q72_spider_refiner", "MACSQL_q72_spider_prefix", "Qwen2.5-72B-AWQ"),
          ("CHESS B1", "Qwen3-32B, BIRD SDS 147", "CHESS_UT", "CHESS_prefix", "Qwen3-32B"),
          ("ComEM", "Qwen3-32B selector, 8 datasets x 400", "ComEM_selector@q3_g3", None, "Qwen3-32B"),
          ("ComEM", "Qwen2.5-7B selector, 8 datasets x 400", "ComEM_selector@q7_g3", None, "Qwen2.5-7B")]


def servers(ext: pathlib.Path, chess: pathlib.Path) -> Dict[str, tuple]:
    """Server -> (model, proxy record files)."""
    raw = ext / "raw"
    return {
        "chess_g1": ("Qwen3-32B", [chess / "raw/formal_s1.jsonl"]),
        "chess_g2": ("Qwen3-32B", [chess / "raw/formal_s2.jsonl"]),
        "chess_g3": ("Qwen3-32B", [chess / "raw/formal_s3.jsonl"]),
        "q3_g1": ("Qwen3-32B", [raw / "raw_a/q3_bird_s1.jsonl", raw / "raw_a/q3_spider_s1.jsonl"]),
        "q3_g2": ("Qwen3-32B", [raw / "raw_a/q3_bird_s2.jsonl", raw / "raw_a/q3_spider_s2.jsonl"]),
        "q3_g3": ("Qwen3-32B", [raw / "raw_a/q3_bird_s3.jsonl", raw / "raw_a/q3_spider_s3.jsonl", raw / "raw_a/q3_comem.jsonl"]),
        "q72_g0": ("Qwen2.5-72B-AWQ", [raw / f"raw_b/q72_spider_s{k}.jsonl" for k in (1, 2, 3)]),
        "q72_g1": ("Qwen2.5-72B-AWQ", [raw / "raw_b/q72_bird_s1.jsonl"]),
        "q72_g2": ("Qwen2.5-72B-AWQ", [raw / "raw_b/q72_bird_s2.jsonl"]),
        "q72_g3": ("Qwen2.5-72B-AWQ", [raw / "raw_b/q72_bird_s3.jsonl"]),
        "q7_g3": ("Qwen2.5-7B", [raw / "raw_b/q7_comem.jsonl"]),
    }


def load_calls(server: str, files) -> List[dict]:
    """Successful calls with token usage; refused calls and non-200 responses are dropped."""
    calls = []
    for f in files:
        src = pathlib.Path(f).stem
        for line in open(f):
            r = json.loads(line)
            if r.get("refused") or r.get("status") != 200 or not r.get("usage"):
                continue
            msgs = (r.get("request") or {}).get("messages") or [{}]
            first = msgs[0].get("content") or ""
            if isinstance(first, list):
                first = " ".join(x.get("text", "") for x in first if isinstance(x, dict))
            calls.append(dict(server=server, src=src, q=r.get("question") or "", tool=r.get("tool") or "",
                              s=float(r["ts_start"]), e=float(r["ts_end"]), L=float(r["latency_s"]),
                              p=r["usage"].get("prompt_tokens") or 0, c=r["usage"].get("completion_tokens") or 0,
                              refiner=REFINER_SIG in first, ping="Hello world!" in first))
    return calls


def attribute(calls: List[dict]) -> None:
    """Attributed server-seconds of each call: every instant with n calls in flight is split 1/n per call."""
    ev = sorted([(c["s"], 1, i) for i, c in enumerate(calls)] + [(c["e"], 0, i) for i, c in enumerate(calls)])
    active, t_prev = set(), None
    for c in calls:
        c["attr_s"] = 0.0
    for t, kind, i in ev:
        if active and t_prev is not None and t > t_prev:
            share = (t - t_prev) / len(active)
            for j in active:
                calls[j]["attr_s"] += share
        t_prev = t
        (active.add if kind else active.discard)(i)


def fit_single(calls: List[dict]) -> Optional[dict]:
    """Least squares of latency on prompt and completion tokens over calls with concurrency 1. Rp, Rd
    are None (and t0, a, b unrounded) when a slope is not positive; None below 30 such calls."""
    one = [c for c in calls if c["conc"] == 1 and c["L"] > 0]
    if len(one) < MIN_SINGLE:
        return None
    X = np.array([[1.0, c["p"], c["c"]] for c in one])
    y = np.array([c["L"] for c in one])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    t0, a, b = coef
    if a <= 0 or b <= 0:
        return dict(n=len(one), t0=float(t0), a=float(a), b=float(b), Rp=None, Rd=None)
    resid = y - X @ coef
    return dict(n=len(one), t0=round(float(t0), 4), Rp=round(float(1 / a), 1), Rd=round(float(1 / b), 2),
                r2=round(1 - float(resid.var() / y.var()), 3))


def bins(calls: List[dict]) -> dict:
    """Busiest-bin and median busy-bin prefill and decoding rates (tokens/s) over 60-s bins."""
    t0 = min(c["s"] for c in calls)
    P, Cc = collections.Counter(), collections.Counter()
    for c in calls:
        P[int((c["s"] - t0) // BIN)] += c["p"]
        dur = max(c["e"] - c["s"], 1e-6)
        b0, b1 = int((c["s"] - t0) // BIN), int((c["e"] - t0) // BIN)
        for k in range(b0, b1 + 1):
            lo, hi = max(c["s"], t0 + k * BIN), min(c["e"], t0 + (k + 1) * BIN)
            if hi > lo:
                Cc[k] += c["c"] * (hi - lo) / dur
    keys = sorted(set(P) | set(Cc))
    pr = np.array([P[k] / BIN for k in keys])
    cr = np.array([Cc[k] / BIN for k in keys])
    busy = cr > 0
    return dict(bins=len(keys), Rp_peak=round(float(pr.max()), 1), Rd_peak=round(float(cr.max()), 1),
                Rp_median_busy=round(float(np.median(pr[busy])), 1) if busy.any() else None,
                Rd_median_busy=round(float(np.median(cr[busy])), 1) if busy.any() else None)


def model_rates(per_server: List[dict]) -> Dict[str, dict]:
    """Per model: median, slowest single-stream fit, and fastest rate of any form (fit or busiest bin)."""
    rates = collections.defaultdict(lambda: dict(Rp=[], Rd=[], Rp_peak=[], Rd_peak=[]))
    for r in per_server:
        m = rates[r["model"]]
        if r["single_Rp_tok_s"] and r["single_Rd_tok_s"]:
            m["Rp"].append(r["single_Rp_tok_s"])
            m["Rd"].append(r["single_Rd_tok_s"])
        m["Rp_peak"].append(r["Rp_peak"])
        m["Rd_peak"].append(r["Rd_peak"])
    return {m: dict(Rp_single_median=float(np.median(v["Rp"])), Rd_single_median=float(np.median(v["Rd"])),
                    Rp_min=float(min(v["Rp"])), Rd_min=float(min(v["Rd"])),
                    Rp_max=float(max(v["Rp"] + v["Rp_peak"])), Rd_max=float(max(v["Rd"] + v["Rd_peak"])),
                    valid_single_fits=len(v["Rp"]))
            for m, v in rates.items()}


def gpu_s(rates: Dict[str, dict], model: str, p, c) -> dict:
    r = rates[model]
    return dict(gpu_s_point=round(p / r["Rp_single_median"] + c / r["Rd_single_median"], 1),
                gpu_s_low=round(p / r["Rp_max"] + c / r["Rd_max"], 1),
                gpu_s_high=round(float(p / r["Rp_min"] + c / r["Rd_min"]), 1))


def run(out=C.OUT, external_run=C.EXT, chess_run=C.CHESS, verbose: bool = True) -> dict:
    """Write the per-server throughput, the stage GPU-seconds and their JSON; returns the JSON object."""
    out, ext, chess = pathlib.Path(out), pathlib.Path(external_run), pathlib.Path(chess_run)
    out.mkdir(parents=True, exist_ok=True)
    per_server, all_calls = [], []
    for name, (model, files) in servers(ext, chess).items():
        calls = load_calls(name, files)
        cost._label(calls)
        cost._concurrency(calls)
        attribute(calls)
        fs, bn = fit_single(calls), bins(calls)
        conc = np.array([c["conc"] for c in calls])
        per_server.append(dict(server=name, model=model, calls=len(calls), conc1_calls=int((conc == 1).sum()),
                               conc_median=float(np.median(conc)), conc_p90=float(np.percentile(conc, 90)),
                               conc_max=int(conc.max()),
                               single_Rp_tok_s=(fs or {}).get("Rp"), single_Rd_tok_s=(fs or {}).get("Rd"),
                               single_t0_s=(fs or {}).get("t0"), single_r2=(fs or {}).get("r2"),
                               single_fit_n=(fs or {}).get("n"), **bn,
                               wall_s=round(max(c["e"] for c in calls) - min(c["s"] for c in calls), 1),
                               attributed_s=round(sum(c["attr_s"] for c in calls), 1)))
        all_calls += calls
        if verbose:
            print(per_server[-1], flush=True)
    rates = model_rates(per_server)

    by = collections.defaultdict(list)
    for c in all_calls:
        by[c["stage"]].append(c)
        by[f"{c['stage']}@{c['server']}"].append(c)
    stages = []
    for system, cell, st, pre, model in STAGES:
        S = by[st]
        p, c = sum(x["p"] for x in S), sum(x["c"] for x in S)
        row = dict(system=system, cell=cell, model=model, stage_label=st, proxy_calls=len(S), proxy_prompt_tokens=p,
                   proxy_completion_tokens=c, **gpu_s(rates, model, p, c),
                   attributed_server_s=round(sum(x["attr_s"] for x in S), 1),
                   recorded_latency_sum_s=round(sum(x["L"] for x in S), 1),
                   stage_conc_median=float(np.median([x["conc"] for x in S])) if S else None,
                   prefix_conc_median=float(np.median([x["conc"] for x in by[pre]])) if pre and by[pre] else None)
        if pre:
            P_ = by[pre]
            pp, pc = sum(x["p"] for x in P_), sum(x["c"] for x in P_)
            row.update(prefix_proxy_calls=len(P_), **{f"prefix_{k}": v for k, v in gpu_s(rates, model, pp, pc).items()},
                       prefix_attributed_server_s=round(sum(x["attr_s"] for x in P_), 1))
        stages.append(row)
        if verbose:
            print(row, flush=True)

    C.write_table(out / "throughput_by_server.csv", per_server)
    C.write_table(out / "stage_gpu_seconds.csv", stages)
    obj = dict(model_rates=rates, servers=per_server, stages=stages)
    C.write_json(out / "throughput.json", obj)
    return obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = C.parse_args(__doc__.splitlines()[0], argv)
    run(a.out, a.external_run, a.chess_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
