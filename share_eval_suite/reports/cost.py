"""Cost reports: LLM calls, prompt and completion tokens, and estimated GPU-seconds.

A stage's work is what replacing it by a rule removes: its own calls and tokens (the rule makes no
call). GPU-seconds are estimated per stage as p / Rp + c / Rd, with prefill (Rp) and decoding (Rd)
rates fitted to the proxy-logged calls of each vLLM server: least squares of latency = t0 + p/Rp + c/Rd
on calls that overlap no other call. The point uses the median single-stream fit of the model, the low
end the fastest rate seen for the model (single-stream fit or busiest 60-s bin), the high end the
slowest single-stream fit. Default inputs: `paper_objects/tab_replacement-cost/inputs`; the debate
costs read `paper_objects/fig_debate-budget/inputs`.
"""
from __future__ import annotations

import bisect
import collections
import functools
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from share_eval_suite.comparisons import replacement as R

_OBJ = "tab_replacement-cost"
_GPU_MODEL = {"q3": "Qwen3-32B", "q72": "Qwen2.5-72B-AWQ"}
UT_TOOLS = ("agent_tool_choice:unit_tester", "generate_unit_test", "evaluate")
# the vLLM servers of the proxy log (CHESS, MAC-SQL with Qwen3-32B and Qwen2.5-72B, ComEM with Qwen2.5-7B)
SERVERS = ("chess_g1", "chess_g2", "chess_g3", "q3_g1", "q3_g2", "q3_g3", "q72_g0", "q72_g1", "q72_g2", "q72_g3", "q7_g3")
BIN = 60.0               # seconds per bin of the busiest-bin rates
MIN_SINGLE = 30          # no single-stream fit for a server with fewer calls that overlap no other call


def _inputs(inputs) -> pathlib.Path:
    return R._inp(_OBJ, inputs)


@dataclass(frozen=True)
class StageWork:
    """The LLM work of one stage over all tasks of its row."""
    calls: int
    prompt_tokens: int
    completion_tokens: int

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _unit_tester_partition(rows):
    """CHESS calls split at the first call of any Unit Tester tool for the question (that call and every
    later one are the Unit Tester's increment). Only the CHESS runs (sources `formal_s*`) count."""
    tools = UT_TOOLS
    ch = [r for r in rows if r["source"].startswith("formal_s")]
    first = {}
    for r in sorted(ch, key=lambda r: float(r["ts_start"])):
        if r["tool"] in tools:
            first.setdefault(r["question"], float(r["ts_start"]))
    ut = [r for r in ch if r["question"] in first and float(r["ts_start"]) >= first[r["question"]]]
    return ch, ut


def stage_work(stage: str, model: str = "qwen3-32b", benchmark: Optional[str] = "bird", inputs=None) -> StageWork:
    """Calls and tokens of the Refiner, the Unit Tester's increment, or the ComEM selector."""
    cell = R._cell(stage, model, benchmark)
    inp = _inputs(inputs)
    if stage == "refiner":
        st = [r for r in R._read(_OBJ, "macsql_calls_per_question.csv", inp)
              if r["model"] == R.MODELS[model] and r["benchmark"] == benchmark and r["agent"] == "refiner"]
        calls = sum(int(r["calls"]) for r in st)
    elif stage == "selector":
        st = [r for r in R._read(_OBJ, "comem_anchors.csv", inp) if r["model"] == cell.split("_")[1]]
        calls = sum(int(r["selector_calls"]) for r in st)
    else:
        _, st = _unit_tester_partition(R._read(_OBJ, "llm_calls.csv", inp))
        calls = len(st)
    return StageWork(calls, sum(int(r["prompt_tokens"]) for r in st), sum(int(r["completion_tokens"]) for r in st))


def unit_tester_calls(inputs=None) -> Dict[str, float]:
    """The Unit Tester's calls by kind and its share (%) of the CHESS calls and completion tokens."""
    ch, ut = _unit_tester_partition(R._read(_OBJ, "llm_calls.csv", _inputs(inputs)))
    out_ut = sum(int(r["completion_tokens"]) for r in ut)
    return dict(all_calls=len(ch), unit_tester_calls=len(ut),
                tool_choice_calls=sum(1 for r in ut if r["tool"].startswith("agent_tool_choice")),
                test_generation_calls=sum(1 for r in ut if r["tool"] == "generate_unit_test"),
                evaluation_calls=sum(1 for r in ut if r["tool"] == "evaluate"),
                call_share=100 * len(ut) / len(ch),
                completion_share=100 * out_ut / sum(int(r["completion_tokens"]) for r in ch))


# ---------------------------------------------------------------- GPU-seconds
def _server_calls(calls: List[dict], name: str) -> List[dict]:
    """Successful calls of one server that report token usage."""
    out = []
    for r in calls:
        if r["server"] != name or r["status"] != "200" or not int(r["has_usage"]):
            continue
        out.append(dict(model=r["model"], src=r["source"], q=r["question"], tool=r["tool"], s=float(r["ts_start"]),
                        e=float(r["ts_end"]), L=float(r["latency_s"]), p=int(r["prompt_tokens"]),
                        c=int(r["completion_tokens"]), refiner=int(r["refiner_prompt"]), ping=int(r["ping"])))
    return out


def _label(cs: List[dict]) -> None:
    """Stage of each call: the CHESS Unit Tester increment or prefix, the ComEM selector, a ping, or the
    MAC-SQL Refiner or prefix of a model and benchmark (e.g. MACSQL_q3_bird_refiner)."""
    ut_start: Dict[str, float] = {}
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


def _concurrency(cs: List[dict]) -> None:
    """Number of calls of the server in flight during each call, the call itself included."""
    S, E = sorted(c["s"] for c in cs), sorted(c["e"] for c in cs)
    for c in cs:
        c["conc"] = bisect.bisect_left(S, c["e"]) - bisect.bisect_right(E, c["s"])


def _fit_single(cs: List[dict]) -> Optional[Dict[str, float]]:
    """Least squares of latency = t0 + p/Rp + c/Rd on calls that overlap no other call; Rp (prefill,
    tokens/s, one decimal) and Rd (decoding, two decimals), or None below 30 such calls or for a
    nonpositive slope."""
    one = [c for c in cs if c["conc"] == 1 and c["L"] > 0]
    if len(one) < MIN_SINGLE:
        return None
    X = np.array([[1.0, c["p"], c["c"]] for c in one])
    y = np.array([c["L"] for c in one])
    (t0, a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    if a <= 0 or b <= 0:
        return None
    return dict(Rp=round(float(1 / a), 1), Rd=round(float(1 / b), 2))


def _peak_bins(cs: List[dict]):
    """Prefill and decoding rates (tokens/s, one decimal) of the busiest 60-s bin: prompt tokens count in
    the bin where the call starts, completion tokens are spread over the call's duration."""
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


def _gpu_s(rates: Dict[str, float], p, c):
    """GPU-seconds p / Rp + c / Rd at the point, fastest and slowest rates, each rounded to 0.1 s."""
    return (round(p / rates["Rp"] + c / rates["Rd"], 1), round(p / rates["Rp_max"] + c / rates["Rd_max"], 1),
            round(p / rates["Rp_min"] + c / rates["Rd_min"], 1))


@functools.lru_cache(maxsize=4)
def _gpu_tables(inp: str):
    """Rates per model (point = median single-stream fit; Rp_min/Rd_min the slowest fit; Rp_max/Rd_max
    the fastest fit or busiest bin), the raw fits and bins per model, and the calls per stage."""
    calls = R._read(_OBJ, "llm_calls.csv", inp)
    rates = collections.defaultdict(lambda: dict(Rp=[], Rd=[], Rp_peak=[], Rd_peak=[]))
    by_stage = collections.defaultdict(list)
    for name in SERVERS:
        cs = _server_calls(calls, name)
        _label(cs)
        _concurrency(cs)
        fs = _fit_single(cs)
        rpk, rdk = _peak_bins(cs)
        m = rates[cs[0]["model"]]
        if fs:
            m["Rp"].append(fs["Rp"]); m["Rd"].append(fs["Rd"])
        m["Rp_peak"].append(rpk); m["Rd_peak"].append(rdk)
        for c in cs:
            by_stage[c["stage"]].append(c)
    RATES = {m: dict(Rp=float(np.median(v["Rp"])), Rd=float(np.median(v["Rd"])), Rp_min=min(v["Rp"]), Rd_min=min(v["Rd"]),
                     Rp_max=max(v["Rp"] + v["Rp_peak"]), Rd_max=max(v["Rd"] + v["Rd_peak"])) for m, v in rates.items()}
    return RATES, {m: dict(v) for m, v in rates.items()}, dict(by_stage)


def decoding_rates(inputs=None) -> Dict[str, Dict[str, float]]:
    """Single-stream decoding rates per model (tokens/s): median, lowest and highest server fit."""
    _, raw, _ = _gpu_tables(str(_inputs(inputs)))
    return {m: dict(median=float(np.median(v["Rd"])), min=min(v["Rd"]), max=max(v["Rd"])) for m, v in raw.items()}


@dataclass(frozen=True)
class GpuSeconds:
    """Estimated GPU-seconds of a stage (point, low and high end) and its share (%) of stage plus the
    rest of the pipeline at the point rates."""
    point: float
    low: float
    high: float
    share: float


def stage_gpu_seconds(stage: str, model: str = "qwen3-32b", benchmark: Optional[str] = "bird",
                      inputs=None) -> GpuSeconds:
    """Estimated GPU-seconds of the Refiner or of the Unit Tester's increment."""
    cell = R._cell(stage, model, benchmark)
    if stage == "selector":
        raise ValueError("GPU-seconds are estimated for the SQL stages")
    RATES, _, by_stage = _gpu_tables(str(_inputs(inputs)))
    if stage == "refiner":
        st, pre = f"MACSQL_{R.MODELS[model]}_{benchmark}_refiner", f"MACSQL_{R.MODELS[model]}_{benchmark}_prefix"
    else:
        st, pre = "CHESS_UT", "CHESS_prefix"
    gm = _GPU_MODEL[cell.split("_")[1] if stage == "refiner" else "q3"]
    tok = lambda xs: (sum(x["p"] for x in xs), sum(x["c"] for x in xs))  # noqa: E731
    s_pt, s_lo, s_hi = _gpu_s(RATES[gm], *tok(by_stage[st]))
    p_pt = _gpu_s(RATES[gm], *tok(by_stage[pre]))[0]
    return GpuSeconds(s_pt, s_lo, s_hi, 100 * s_pt / (s_pt + p_pt))


# ---------------------------------------------------------------- debates at a call budget
def debate_cost(protocol: str, backbone: str, config: str, inputs=None) -> Dict[str, float]:
    """Mean calls and tokens per run of a BIRD debate (`config` is agents x rounds, e.g. "3x2")."""
    from share_eval_suite.comparisons import computation as C
    V = C._budget_values(str(C._inp("fig_debate-budget", inputs)))
    k = f"{C.PROTOCOLS[protocol]}_{C.BACKBONES[backbone]}_{config}"
    return dict(calls=V[f"{k}_calls"], tokens=V[f"{k}_tokens"], failed_runs=V[f"{k}_failed"])
