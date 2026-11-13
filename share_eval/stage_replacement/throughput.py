r"""Same-host throughput and estimated GPU-seconds per stage (offline; reads the per-call proxy logs only).

    python3 share_eval/stage_replacement/throughput.py [--out DIR] [--external-run DIR] [--chess-run DIR]
    # -> <out>/throughput_by_server.csv, <out>/stage_gpu_seconds.csv, <out>/throughput.json

Throughput is measured from the per-call proxy records: every LLM call is logged with ts_start, ts_end,
latency_s and usage.

Servers (one vLLM instance each; all on the same host and H200 cards; vllm/vllm-openai v0.32.0, --max-num-seqs 96):
  CHESS     Qwen3-32B  chess_g1..g3      <- raw/formal_s{1,2,3}.jsonl
  MAC-SQL   Qwen3-32B  q3_g1..g3          <- raw_a/q3_{bird,spider}_s{g}.jsonl (+ q3_comem on g3)
            72B-AWQ    q72_g0             <- raw_b/q72_spider_s{1,2,3}.jsonl;  q72_g1..g3 <- raw_b/q72_bird_s{g}.jsonl
  ComEM     Qwen3-32B  q3_g3 (shared with MAC-SQL shard 3);  Qwen2.5-7B q7_g3 <- raw_b/q7_comem.jsonl
Measured per server:
  * concurrency of each call = other calls of the same server overlapping its [ts_start, ts_end];
  * single-stream rates: OLS latency = t0 + p/Rp + c/Rd on concurrency-1 calls (calls with no overlap) -> Rp, Rd;
  * peak batched rates: 60-s bins; prompt tokens put in the start bin, completion tokens spread uniformly over the
    call; the maximum bin rate of each (Rp_peak, Rd_peak) and the median busy-bin rates.
Estimated GPU-seconds of a stage = p/Rp + c/Rd with that model's rates. Low-throughput end (upper GPU-s):
the slowest single-stream fit of that model; high-throughput end (lower GPU-s): the fastest peak bins of that model.
Point: the median single-stream fit (no batching). Assumptions: prefill and decode do not overlap; prompt tokens
include prefix-cache hits (vLLM prefix caching on by default), which lowers real prefill cost; one call = one stream.
Cross-check (attributed server-seconds): every instant a server has n calls in flight is split 1/n per call; the
stage's share is summed. It is reported with the stage's and prefix's median concurrency.
"""
import bisect
import collections
import csv
import json

import numpy as np

import pathlib as _pl, sys as _sys
_sys.dont_write_bytecode = True
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import common as K  # noqa: E402

K.configure(__doc__.splitlines()[0])

EXTR = K.EXT / "raw"
SERVERS = {
    "chess_g1": ("Qwen3-32B", [K.CHESS / "raw/formal_s1.jsonl"]),
    "chess_g2": ("Qwen3-32B", [K.CHESS / "raw/formal_s2.jsonl"]),
    "chess_g3": ("Qwen3-32B", [K.CHESS / "raw/formal_s3.jsonl"]),
    "q3_g1": ("Qwen3-32B", [EXTR / "raw_a/q3_bird_s1.jsonl", EXTR / "raw_a/q3_spider_s1.jsonl"]),
    "q3_g2": ("Qwen3-32B", [EXTR / "raw_a/q3_bird_s2.jsonl", EXTR / "raw_a/q3_spider_s2.jsonl"]),
    "q3_g3": ("Qwen3-32B", [EXTR / "raw_a/q3_bird_s3.jsonl", EXTR / "raw_a/q3_spider_s3.jsonl", EXTR / "raw_a/q3_comem.jsonl"]),
    "q72_g0": ("Qwen2.5-72B-AWQ", [EXTR / f"raw_b/q72_spider_s{k}.jsonl" for k in (1, 2, 3)]),
    "q72_g1": ("Qwen2.5-72B-AWQ", [EXTR / "raw_b/q72_bird_s1.jsonl"]),
    "q72_g2": ("Qwen2.5-72B-AWQ", [EXTR / "raw_b/q72_bird_s2.jsonl"]),
    "q72_g3": ("Qwen2.5-72B-AWQ", [EXTR / "raw_b/q72_bird_s3.jsonl"]),
    "q7_g3": ("Qwen2.5-7B", [EXTR / "raw_b/q7_comem.jsonl"]),
}
REFINER_SIG = "When executing SQL below, some errors occurred"
UT_TOOLS = ("agent_tool_choice:unit_tester", "generate_unit_test", "evaluate")
BIN = 60.0


def load(server, files):
    calls = []
    for f in files:
        src = f.stem
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


def label(calls):
    """Stage labels: CHESS UT increment (extract_b1 rule), MAC-SQL Refiner (prompt signature), ComEM selector (all)."""
    ut_start = {}
    for c in calls:
        if c["src"].startswith("formal_s") and c["tool"] in UT_TOOLS:
            ut_start[c["q"]] = min(ut_start.get(c["q"], c["s"]), c["s"])
    for c in calls:
        if c["src"].startswith("formal_s"):
            c["stage"] = "CHESS_UT" if (c["q"] in ut_start and c["s"] >= ut_start[c["q"]]) else "CHESS_prefix"
        elif "comem" in c["src"]:
            c["stage"] = "ComEM_selector"
        elif c["ping"]:
            c["stage"] = "ping"
        else:
            cell = c["src"].rsplit("_s", 1)[0]          # q3_bird / q72_spider ...
            c["stage"] = f"MACSQL_{cell}_" + ("refiner" if c["refiner"] else "prefix")


def concurrency_and_share(calls):
    S = sorted(c["s"] for c in calls); E = sorted(c["e"] for c in calls)
    for c in calls:
        c["conc"] = bisect.bisect_left(S, c["e"]) - bisect.bisect_right(E, c["s"])   # includes itself
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


def fit_single(calls):
    one = [c for c in calls if c["conc"] == 1 and c["L"] > 0]
    if len(one) < 30:
        return None
    X = np.array([[1.0, c["p"], c["c"]] for c in one]); y = np.array([c["L"] for c in one])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    t0, a, b = coef
    if a <= 0 or b <= 0:
        return dict(n=len(one), t0=float(t0), a=float(a), b=float(b), Rp=None, Rd=None)
    resid = y - X @ coef
    return dict(n=len(one), t0=round(float(t0), 4), Rp=round(float(1 / a), 1), Rd=round(float(1 / b), 2),
                r2=round(1 - float(resid.var() / y.var()), 3))


def bins(calls):
    t0 = min(c["s"] for c in calls)
    P, C = collections.Counter(), collections.Counter()
    for c in calls:
        P[int((c["s"] - t0) // BIN)] += c["p"]
        dur = max(c["e"] - c["s"], 1e-6)
        b0, b1 = int((c["s"] - t0) // BIN), int((c["e"] - t0) // BIN)
        for k in range(b0, b1 + 1):
            lo, hi = max(c["s"], t0 + k * BIN), min(c["e"], t0 + (k + 1) * BIN)
            if hi > lo:
                C[k] += c["c"] * (hi - lo) / dur
    keys = sorted(set(P) | set(C))
    pr = np.array([P[k] / BIN for k in keys]); cr = np.array([C[k] / BIN for k in keys])
    busy = cr > 0
    return dict(bins=len(keys), Rp_peak=round(float(pr.max()), 1), Rd_peak=round(float(cr.max()), 1),
                Rp_median_busy=round(float(np.median(pr[busy])), 1) if busy.any() else None,
                Rd_median_busy=round(float(np.median(cr[busy])), 1) if busy.any() else None)


per_server, all_calls = [], []
for name, (model, files) in SERVERS.items():
    calls = load(name, files)
    label(calls)
    concurrency_and_share(calls)
    fs, bn = fit_single(calls), bins(calls)
    conc = np.array([c["conc"] for c in calls])
    per_server.append(dict(server=name, model=model, calls=len(calls), conc1_calls=int((conc == 1).sum()),
                           conc_median=float(np.median(conc)), conc_p90=float(np.percentile(conc, 90)), conc_max=int(conc.max()),
                           single_Rp_tok_s=(fs or {}).get("Rp"), single_Rd_tok_s=(fs or {}).get("Rd"), single_t0_s=(fs or {}).get("t0"),
                           single_r2=(fs or {}).get("r2"), single_fit_n=(fs or {}).get("n"), **bn,
                           wall_s=round(max(c["e"] for c in calls) - min(c["s"] for c in calls), 1),
                           attributed_s=round(sum(c["attr_s"] for c in calls), 1)))
    all_calls += calls
    print(per_server[-1], flush=True)

rates = collections.defaultdict(lambda: dict(Rp=[], Rd=[], Rp_peak=[], Rd_peak=[]))
for r in per_server:
    m = rates[r["model"]]
    if r["single_Rp_tok_s"] and r["single_Rd_tok_s"]:
        m["Rp"].append(r["single_Rp_tok_s"]); m["Rd"].append(r["single_Rd_tok_s"])
    m["Rp_peak"].append(r["Rp_peak"]); m["Rd_peak"].append(r["Rd_peak"])
# Low end of throughput = the slowest valid single-stream fit of the model. High end = the fastest rate observed for
# the model in any form (single-stream fit or a peak batched bin). Peak bins measure offered load, so on servers that
# ran one stream at a time they stay below the single-stream rate and do not set the high end.
MODEL_RATES = {m: dict(Rp_single_median=float(np.median(v["Rp"])), Rd_single_median=float(np.median(v["Rd"])),
                       Rp_min=float(min(v["Rp"])), Rd_min=float(min(v["Rd"])),
                       Rp_max=float(max(v["Rp"] + v["Rp_peak"])), Rd_max=float(max(v["Rd"] + v["Rd_peak"])),
                       valid_single_fits=len(v["Rp"]))
               for m, v in rates.items()}


def gpu_s(model, p, c):
    r = MODEL_RATES[model]
    return dict(gpu_s_point=round(p / r["Rp_single_median"] + c / r["Rd_single_median"], 1),
                gpu_s_low=round(p / r["Rp_max"] + c / r["Rd_max"], 1),
                gpu_s_high=round(float(p / r["Rp_min"] + c / r["Rd_min"]), 1))


STAGES = [("MAC-SQL", "Qwen3-32B, BIRD dev", "MACSQL_q3_bird_refiner", "MACSQL_q3_bird_prefix", "Qwen3-32B"),
          ("MAC-SQL", "Qwen3-32B, Spider dev", "MACSQL_q3_spider_refiner", "MACSQL_q3_spider_prefix", "Qwen3-32B"),
          ("MAC-SQL", "Qwen2.5-72B-AWQ, BIRD dev", "MACSQL_q72_bird_refiner", "MACSQL_q72_bird_prefix", "Qwen2.5-72B-AWQ"),
          ("MAC-SQL", "Qwen2.5-72B-AWQ, Spider dev", "MACSQL_q72_spider_refiner", "MACSQL_q72_spider_prefix", "Qwen2.5-72B-AWQ"),
          ("CHESS B1", "Qwen3-32B, BIRD SDS 147", "CHESS_UT", "CHESS_prefix", "Qwen3-32B"),
          ("ComEM", "Qwen3-32B selector, 8 datasets x 400", "ComEM_selector@q3_g3", None, "Qwen3-32B"),
          ("ComEM", "Qwen2.5-7B selector, 8 datasets x 400", "ComEM_selector@q7_g3", None, "Qwen2.5-7B")]
by = collections.defaultdict(list)
for c in all_calls:
    by[c["stage"]].append(c); by[f"{c['stage']}@{c['server']}"].append(c)
out = []
for system, cell, st, pre, model in STAGES:
    S = by[st]
    p, c = sum(x["p"] for x in S), sum(x["c"] for x in S)
    row = dict(system=system, cell=cell, model=model, stage_label=st, proxy_calls=len(S), proxy_prompt_tokens=p,
               proxy_completion_tokens=c, **gpu_s(model, p, c),
               attributed_server_s=round(sum(x["attr_s"] for x in S), 1), recorded_latency_sum_s=round(sum(x["L"] for x in S), 1),
               stage_conc_median=float(np.median([x["conc"] for x in S])) if S else None,
               prefix_conc_median=float(np.median([x["conc"] for x in by[pre]])) if pre and by[pre] else None)
    if pre:
        P_ = by[pre]
        pp, pc = sum(x["p"] for x in P_), sum(x["c"] for x in P_)
        row.update(prefix_proxy_calls=len(P_), **{f"prefix_{k}": v for k, v in gpu_s(model, pp, pc).items()},
                   prefix_attributed_server_s=round(sum(x["attr_s"] for x in P_), 1))
    out.append(row)
    print(row, flush=True)


def write(name, rows):
    with open(K.OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list({k: None for r in rows for k in r})); w.writeheader(); w.writerows(rows)


write("throughput_by_server.csv", per_server); write("stage_gpu_seconds.csv", out)
(K.OUT / "throughput.json").write_text(json.dumps(dict(model_rates=MODEL_RATES, servers=per_server, stages=out), indent=1) + "\n")
print(json.dumps(MODEL_RATES, indent=1))
