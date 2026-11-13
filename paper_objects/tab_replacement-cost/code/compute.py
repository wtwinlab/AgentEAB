r"""Table replacement-cost (stage rows), the CHESS tables S-chess and S-chess-resources, and Table S-exec-cost.

    python3 code/compute.py --out output        # -> output/values.json, output/throughput_by_server.csv

Each stage is compared with a rule that makes no LLM call on the same logged input.
  * Quality change: stage minus rule (pp); a positive value is accuracy given up by the replacement.
    Percentile bootstrap, 95%, 10,000 draws, seed 20261105.
    SQL rows resample databases (question weights; database weights where stated); ComEM rows resample
    query records within each of the 8 datasets.
  * LLM work removed: the stage's own calls and prompt/completion tokens (the rule makes no call).
  * Additional correct tasks per million stage tokens: (stage correct - rule correct) / (stage tokens / 10^6).
  * Estimated GPU-seconds: p / Rp + c / Rd per stage, with prefill (Rp) and decode (Rd) rates fitted to the
    proxy-logged calls of each vLLM server (one H200 host): OLS of latency = t0 + p/Rp + c/Rd on calls with no
    overlapping call. Point = median single-stream fit of the model; low end = fastest rate seen for the model
    (single-stream fit or busiest 60-s bin); high end = slowest single-stream fit. The share is the stage's
    estimate over stage + rest of the pipeline.
  * Execution cost: logged single-threaded SQLite seconds per question of the candidates an execution vote executes.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import datetime
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
DRAWS, SEED = 10000, 20261105
V: dict = {}


def read(name):
    with open(INP / name, newline="") as f:
        return list(csv.DictReader(f))


def put(key, x):
    V[key] = round(float(x), 6)


def cluster_boot(values: dict, clusters: dict, weighting="unit"):
    """Paired cluster bootstrap: per-cluster means; unit (question) or cluster-equal (database) weights."""
    by = collections.defaultdict(list)
    for u in sorted(values):
        by[clusters[u]].append(values[u])
    cl = sorted(by)
    per = np.array([np.mean(by[c]) for c in cl])
    n = np.array([len(by[c]) for c in cl], float)
    w = np.ones(len(cl)) if weighting == "cluster_equal" else n
    d = np.random.default_rng(SEED).integers(0, len(cl), size=(DRAWS, len(cl)))
    bs = (per[d] * w[d]).sum(1) / w[d].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float((w * per).sum() / w.sum()), float(lo), float(hi)


def strat_boot(values: dict, strata: dict):
    g = collections.defaultdict(list)
    for k, v in values.items():
        g[strata[k]].append(v)
    rng = np.random.default_rng(SEED)
    g = {k: np.asarray(v, float) for k, v in sorted(g.items())}
    n = sum(len(v) for v in g.values())
    tot = np.zeros(DRAWS)
    for v in g.values():
        tot += v[rng.integers(0, len(v), size=(DRAWS, len(v)))].sum(1)
    lo, hi = np.percentile(tot / n, [2.5, 97.5])
    return float(sum(v.sum() for v in g.values()) / n), float(lo), float(hi)


def contrast(key, D, cl, boot, n_extra=None):
    """D = stage - rule per task; the table prints stage - rule."""
    pt, lo, hi = boot(D, cl)
    put(f"{key}_delta", 100 * pt)
    put(f"{key}_delta_lo", 100 * lo)
    put(f"{key}_delta_hi", 100 * hi)
    put(f"{key}_extra_correct", sum(D.values()))


# ================================================================ quality change
for mt in ("q3", "q72"):
    for b in ("bird", "spider"):
        rs = read(f"macsql_units_{mt}_{b}.csv")
        D = {(int(r["question_id"]),): int(r["Y_stage"]) - int(r["Y_alt"]) for r in rs}
        cl = {(int(r["question_id"]),): r["db_id"] for r in rs}
        put(f"refiner_{mt}_{b}_tasks", len(rs))
        contrast(f"refiner_{mt}_{b}", D, cl, cluster_boot)

comem = read("comem_anchors.csv")
for mt in ("q3", "q7"):
    rs = [r for r in comem if r["model"] == mt]
    D = {(r["dataset"], r["id_left"]): int(r["selector_correct"]) - int(r["ranker_correct"]) for r in rs}
    cl = {(r["dataset"], r["id_left"]): r["dataset"] for r in rs}
    put(f"selector_{mt}_tasks", len(rs))
    contrast(f"selector_{mt}", D, cl, strat_boot)

# CHESS: Unit Tester (returned answer) against the three voting rules
cq = read("chess_questions.csv")
DB = {(int(r["question_id"]),): r["db_id"] for r in cq}
Y = {(int(r["question_id"]),): {k: int(r[k]) for k in ("UT", "UT_fail0", "SC_official", "Valid_vote", "Nonempty_vote", "Any_correct", "ut_entered")}
     for r in cq}
keys = sorted(Y)
put("chess_tasks", len(keys))
put("chess_ut_entered", sum(Y[k]["ut_entered"] for k in keys))
for b in ("UT", "UT_fail0", "SC_official", "Valid_vote", "Nonempty_vote", "Any_correct"):
    put(f"chess_{b}_correct", sum(Y[k][b] for k in keys))
    put(f"chess_{b}_acc", 100 * cluster_boot({k: Y[k][b] for k in keys}, DB)[0])
RULES = ("SC_official", "Valid_vote", "Nonempty_vote")
for wt, wn in (("unit", "q"), ("cluster_equal", "db")):
    for ut, pop in (("UT", "all"), ("UT_fail0", "all"), ("UT", "entered")):
        kk = keys if pop == "all" else [k for k in keys if Y[k]["ut_entered"]]
        for b in RULES:
            pt, lo, hi = cluster_boot({k: Y[k][ut] - Y[k][b] for k in kk}, DB, wt)
            tag = f"chess_{ut}_minus_{b}_{pop}_{wn}"
            put(tag, 100 * pt)
            put(tag + "_lo", 100 * lo)
            put(tag + "_hi", 100 * hi)
            if (ut, pop, wn) == ("UT", "all", "q"):
                put(f"chess_ut_only_correct_{b}", sum(1 for k in kk if Y[k][ut] and not Y[k][b]))
                put(f"chess_rule_only_correct_{b}", sum(1 for k in kk if Y[k][b] and not Y[k][ut]))
# main table: the self-consistency row, stage minus rule
put("ut_sc_delta", V["chess_UT_minus_SC_official_all_q"])
put("ut_sc_delta_lo", V["chess_UT_minus_SC_official_all_q_lo"])
put("ut_sc_delta_hi", V["chess_UT_minus_SC_official_all_q_hi"])
for b in RULES:
    put(f"chess_same_result_{b}", sum(int(r[f"same_result_{b}"]) for r in cq if r[f"same_result_{b}"] != ""))
    put(f"chess_same_sql_{b}", sum(int(r[f"same_sql_{b}"]) for r in cq))
put("chess_flag_sc_equals_valid", int(all(Y[k]["SC_official"] == Y[k]["Valid_vote"] for k in keys)))

# ================================================================ calls and tokens
mq = read("macsql_calls_per_question.csv")
for mt in ("q3", "q72"):
    for b in ("bird", "spider"):
        st = [r for r in mq if r["model"] == mt and r["benchmark"] == b and r["agent"] == "refiner"]
        p, c = sum(int(r["prompt_tokens"]) for r in st), sum(int(r["completion_tokens"]) for r in st)
        put(f"refiner_{mt}_{b}_calls", sum(int(r["calls"]) for r in st))
        put(f"refiner_{mt}_{b}_tok_in_k", p / 1e3)
        put(f"refiner_{mt}_{b}_tok_out_k", c / 1e3)
        put(f"refiner_{mt}_{b}_per_million_tokens", V[f"refiner_{mt}_{b}_extra_correct"] / ((p + c) / 1e6))
for mt in ("q3", "q7"):
    rs = [r for r in comem if r["model"] == mt]
    p, c = sum(int(r["prompt_tokens"]) for r in rs), sum(int(r["completion_tokens"]) for r in rs)
    put(f"selector_{mt}_calls", sum(int(r["selector_calls"]) for r in rs))
    put(f"selector_{mt}_tok_in_k", p / 1e3)
    put(f"selector_{mt}_tok_out_k", c / 1e3)
    put(f"selector_{mt}_per_million_tokens", V[f"selector_{mt}_extra_correct"] / ((p + c) / 1e6))
    put(f"selector_{mt}_calls_per_task", sum(int(r["selector_calls"]) for r in rs) / len(rs))

# CHESS: the Unit Tester's increment starts at the first call of any Unit Tester tool for that question
calls = read("llm_calls.csv")
UT_TOOLS = ("agent_tool_choice:unit_tester", "generate_unit_test", "evaluate")
ch = [r for r in calls if r["source"].startswith("formal_s")]
first = {}
for r in sorted(ch, key=lambda r: float(r["ts_start"])):
    if r["tool"] in UT_TOOLS:
        first.setdefault(r["question"], float(r["ts_start"]))
part = {id(r): ("ut" if r["question"] in first and float(r["ts_start"]) >= first[r["question"]] else "prefix") for r in ch}
for nm in ("prefix", "ut"):
    xs = [r for r in ch if part[id(r)] == nm]
    put(f"chess_{nm}_calls", len(xs))
    put(f"chess_{nm}_tok_in", sum(int(r["prompt_tokens"]) for r in xs))
    put(f"chess_{nm}_tok_out", sum(int(r["completion_tokens"]) for r in xs))
put("chess_all_calls", len(ch))
put("chess_ut_call_share", 100 * V["chess_ut_calls"] / len(ch))
put("chess_ut_completion_share", 100 * V["chess_ut_tok_out"] / (V["chess_ut_tok_out"] + V["chess_prefix_tok_out"]))
put("chess_ut_tok_in_M", V["chess_ut_tok_in"] / 1e6)
put("chess_ut_tok_out_M", V["chess_ut_tok_out"] / 1e6)
utc = [r for r in ch if part[id(r)] == "ut"]
put("chess_ut_tool_choice_calls", sum(1 for r in utc if r["tool"].startswith("agent_tool_choice")))
put("chess_ut_test_generation_calls", sum(1 for r in utc if r["tool"] == "generate_unit_test"))
put("chess_ut_evaluation_calls", sum(1 for r in utc if r["tool"] == "evaluate"))
put("chess_non200_calls", sum(1 for r in ch if r["status"] != "200"))
wall = [int(r["wall_s"]) for r in read("chess_wall_time.csv")]
put("chess_wall_median_s", float(np.median(wall)))
put("chess_wall_max_s", max(wall))

# ================================================================ estimated GPU-seconds from the proxy-logged calls
SERVERS = ("chess_g1", "chess_g2", "chess_g3", "q3_g1", "q3_g2", "q3_g3", "q72_g0", "q72_g1", "q72_g2", "q72_g3", "q7_g3")
BIN = 60.0


def server_calls(name):
    out = []
    for r in calls:
        if r["server"] != name or r["status"] != "200" or not int(r["has_usage"]):
            continue
        out.append(dict(model=r["model"], src=r["source"], q=r["question"], tool=r["tool"], s=float(r["ts_start"]),
                        e=float(r["ts_end"]), L=float(r["latency_s"]), p=int(r["prompt_tokens"]), c=int(r["completion_tokens"]),
                        refiner=int(r["refiner_prompt"]), ping=int(r["ping"])))
    return out


def label(cs):
    ut_start = {}
    for c in cs:
        if c["src"].startswith("formal_s") and c["tool"] in UT_TOOLS:
            ut_start[c["q"]] = min(ut_start.get(c["q"], c["s"]), c["s"])
    for c in cs:
        if c["src"].startswith("formal_s"):
            c["stage"] = "CHESS_UT" if (c["q"] in ut_start and c["s"] >= ut_start[c["q"]]) else "CHESS_prefix"
        elif "comem" in c["src"]:
            c["stage"] = "ComEM_selector"
        elif c["ping"]:
            c["stage"] = "ping"
        else:
            c["stage"] = f"MACSQL_{c['src'].rsplit('_s', 1)[0]}_" + ("refiner" if c["refiner"] else "prefix")


def concurrency(cs):
    S, E = sorted(c["s"] for c in cs), sorted(c["e"] for c in cs)
    for c in cs:
        c["conc"] = bisect.bisect_left(S, c["e"]) - bisect.bisect_right(E, c["s"])   # includes the call itself


def fit_single(cs):
    one = [c for c in cs if c["conc"] == 1 and c["L"] > 0]
    if len(one) < 30:
        return None
    X = np.array([[1.0, c["p"], c["c"]] for c in one])
    y = np.array([c["L"] for c in one])
    (t0, a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    if a <= 0 or b <= 0:
        return None
    return dict(Rp=round(float(1 / a), 1), Rd=round(float(1 / b), 2))


def peak_bins(cs):
    t0 = min(c["s"] for c in cs)
    P, C = collections.Counter(), collections.Counter()
    for c in cs:
        P[int((c["s"] - t0) // BIN)] += c["p"]
        dur = max(c["e"] - c["s"], 1e-6)
        for k in range(int((c["s"] - t0) // BIN), int((c["e"] - t0) // BIN) + 1):
            lo, hi = max(c["s"], t0 + k * BIN), min(c["e"], t0 + (k + 1) * BIN)
            if hi > lo:
                C[k] += c["c"] * (hi - lo) / dur
    ks = sorted(set(P) | set(C))
    return round(float(max(P[k] / BIN for k in ks)), 1), round(float(max(C[k] / BIN for k in ks)), 1)


rates = collections.defaultdict(lambda: dict(Rp=[], Rd=[], Rp_peak=[], Rd_peak=[]))
by_stage = collections.defaultdict(list)
srv_rows = []
for name in SERVERS:
    cs = server_calls(name)
    label(cs)
    concurrency(cs)
    fs = fit_single(cs)
    rpk, rdk = peak_bins(cs)
    m = rates[cs[0]["model"]]
    if fs:
        m["Rp"].append(fs["Rp"]); m["Rd"].append(fs["Rd"])
    m["Rp_peak"].append(rpk); m["Rd_peak"].append(rdk)
    srv_rows.append(dict(server=name, model=cs[0]["model"], calls=len(cs), single_Rp=(fs or {}).get("Rp"),
                         single_Rd=(fs or {}).get("Rd"), Rp_peak=rpk, Rd_peak=rdk))
    for c in cs:
        by_stage[c["stage"]].append(c)
        by_stage[f"{c['stage']}@{name}"].append(c)
RATES = {m: dict(Rp=float(np.median(v["Rp"])), Rd=float(np.median(v["Rd"])), Rp_min=min(v["Rp"]), Rd_min=min(v["Rd"]),
                 Rp_max=max(v["Rp"] + v["Rp_peak"]), Rd_max=max(v["Rd"] + v["Rd_peak"])) for m, v in rates.items()}


def gpu_s(model, p, c):
    r = RATES[model]
    return (round(p / r["Rp"] + c / r["Rd"], 1), round(p / r["Rp_max"] + c / r["Rd_max"], 1),
            round(p / r["Rp_min"] + c / r["Rd_min"], 1))


STAGES = (("refiner_q3_bird", "MACSQL_q3_bird_refiner", "MACSQL_q3_bird_prefix", "Qwen3-32B"),
          ("refiner_q3_spider", "MACSQL_q3_spider_refiner", "MACSQL_q3_spider_prefix", "Qwen3-32B"),
          ("refiner_q72_bird", "MACSQL_q72_bird_refiner", "MACSQL_q72_bird_prefix", "Qwen2.5-72B-AWQ"),
          ("refiner_q72_spider", "MACSQL_q72_spider_refiner", "MACSQL_q72_spider_prefix", "Qwen2.5-72B-AWQ"),
          ("ut", "CHESS_UT", "CHESS_prefix", "Qwen3-32B"))
for key, st, pre, model in STAGES:
    tok = lambda xs: (sum(x["p"] for x in xs), sum(x["c"] for x in xs))  # noqa: E731
    s_pt, s_lo, s_hi = gpu_s(model, *tok(by_stage[st]))
    p_pt = gpu_s(model, *tok(by_stage[pre]))[0]
    put(f"{key}_gpu_s", s_pt)
    put(f"{key}_gpu_s_low", s_lo)
    put(f"{key}_gpu_s_high", s_hi)
    put(f"{key}_gpu_share", 100 * s_pt / (s_pt + p_pt))
for m, r in RATES.items():
    put(f"rate_{m}_decode_single_median", r["Rd"])
    put(f"rate_{m}_decode_single_min", min(rates[m]["Rd"]))     # range of single-stream decoding fits
    put(f"rate_{m}_decode_single_max", max(rates[m]["Rd"]))

# ================================================================ Table S-exec-cost: per question
q3 = "Qwen3-32B"
pq = read("pool4_questions.csv")
sel, gen = read("pool4_selector_calls.csv"), read("pool4_generation_calls.csv")
low_ends = []
for b in ("bird", "spider"):
    allq = [r for r in pq if r["benchmark"] == b]
    sc = [r for r in allq if r["in_scored_set"] == "1"]
    put(f"pool_{b}_questions", len(sc))
    for col, nm in (("Valid", "valid"), ("Nonempty", "nonempty"), ("S_al", "q3_selector"), ("Generation", "q3_generation"),
                    ("DS_S_al", "ds_selector"), ("DS_Generation", "ds_generation")):
        put(f"pool_{b}_{nm}_ex", 100 * np.mean([int(r[col]) for r in sc]))
    put(f"pool_{b}_sql_s", np.mean([float(r["pool_sql_seconds"]) for r in allq]))
    put(f"pool_{b}_ds_selector_tokens", round(float(np.mean([int(r["DS_S_al_in"]) + int(r["DS_S_al_out"]) for r in sc]))))
    put(f"pool_{b}_ds_generation_tokens", round(float(np.mean([int(r["DS_G_in"]) + int(r["DS_G_out"]) for r in sc]))))
    for nm, rows in (("q3_selector", [r for r in sel if r["benchmark"] == b]), ("q3_generation", [r for r in gen if r["benchmark"] == b])):
        p = float(np.mean([int(r["prompt_tokens"]) for r in rows]))
        c = float(np.mean([int(r["completion_tokens"]) for r in rows]))
        pt, lo, hi = gpu_s(q3, p, c)
        put(f"pool_{b}_{nm}_tokens", round(p + c))
        put(f"pool_{b}_{nm}_gpu_s", pt)
        put(f"pool_{b}_{nm}_gpu_s_batched", lo)
        low_ends.append(lo)
# CHESS
ex = collections.defaultdict(float)
for r in read("chess_executions.csv"):
    ex[r["question_id"]] += float(r["seconds"] or 0)
put("chess_sql_s", np.mean(list(ex.values())))
put("chess_sql_s_max", max(ex.values()))
put("chess_ut_calls_per_q", V["chess_ut_calls"] / V["chess_tasks"])
put("chess_ut_tokens_per_q", (V["chess_ut_tok_in"] + V["chess_ut_tok_out"]) / V["chess_tasks"])
put("chess_ut_tokens_per_q_k", V["chess_ut_tokens_per_q"] / 1e3)
put("chess_ut_gpu_s_per_q", V["ut_gpu_s"] / V["chess_tasks"])
put("chess_ut_gpu_s_per_q_batched", V["ut_gpu_s_low"] / V["chess_tasks"])
low_ends.append(V["chess_ut_gpu_s_per_q_batched"])
put("batched_gpu_s_per_q_min", min(low_ends))
put("batched_gpu_s_per_q_max", max(low_ends))
# Llama-3-8B / Qwen2.5-7B BIRD debates with peers: per protocol and model; for accuracy the
# two decodes of a question are averaged first, for cost every decode counts once
deb = collections.defaultdict(list)
for r in read("debate_decodes.csv"):
    deb[(r["protocol"], r["backbone"])].append(r)
cell = {}
for k, rs in sorted(deb.items()):
    byq = collections.defaultdict(list)
    for r in rs:
        byq[r["unit"]].append(r)
    acc = lambda col: 100 * float(np.mean([np.mean([float(x[col]) for x in v]) for v in byq.values()]))  # noqa: E731
    cell[k] = dict(vote=acc("vote_initial_correct"), final=acc("final_correct"),
                   sql=float(np.mean([float(r["r0_sql_seconds"]) for r in rs])),
                   calls=float(np.mean([int(r["later_calls"]) for r in rs])),
                   tokens=round(float(np.mean([int(r["later_prompt_tokens"]) for r in rs])))
                   + round(float(np.mean([int(r["later_completion_tokens"]) for r in rs]))))
for f in ("vote", "final", "sql", "calls", "tokens"):
    put(f"debate_{f}_min", min(c[f] for c in cell.values()))
    put(f"debate_{f}_max", max(c[f] for c in cell.values()))
put("debate_tokens_min_k", V["debate_tokens_min"] / 1e3)
put("debate_tokens_max_k", V["debate_tokens_max"] / 1e3)
put("debate_cells", len(cell))
put("debate_questions", len({r["unit"] for rs in deb.values() for r in rs}))
# ranges stated in the body (Section replacement-cost)
put("exec_sql_s_per_q_min", min(V["debate_sql_min"], V["pool_bird_sql_s"], V["pool_spider_sql_s"], V["chess_sql_s"]))
put("exec_sql_s_per_q_max", max(V["debate_sql_max"], V["pool_bird_sql_s"], V["pool_spider_sql_s"], V["chess_sql_s"]))
# CHESS server occupancy: three Qwen3-32B replicas, from server start to removal, loading and idle time included
occ = [(datetime.datetime.fromisoformat(r["removed_utc"].replace("Z", "+00:00"))
        - datetime.datetime.fromisoformat(r["started_utc"].replace("Z", "+00:00"))).total_seconds()
       for r in read("chess_server_occupancy.csv")]
put("chess_server_gpu_hours", sum(occ) / 3600)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n")
    with open(out / "throughput_by_server.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srv_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(srv_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
