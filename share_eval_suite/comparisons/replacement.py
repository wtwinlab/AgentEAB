from __future__ import annotations

import csv
import pathlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from share_eval_suite import paths, stats
from share_eval_suite.reports.intervals import Interval, database_resample_bootstrap

STAGES = ("refiner", "unit_tester", "selector")
MODELS = {"qwen3-32b": "q3", "qwen2.5-72b": "q72", "qwen2.5-7b": "q7"}
VOTES = {"self_consistency": "SC_official", "valid": "Valid_vote", "nonempty": "Nonempty_vote"}
MIN_CHANGED = 30

Unit = Tuple[tuple, str, int, int, int]


def _inp(obj: str, inputs) -> pathlib.Path:
    return pathlib.Path(inputs) if inputs else paths.inputs(obj)


def _read(obj: str, name: str, inputs) -> List[dict]:
    with open(_inp(obj, inputs) / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _cell(stage: str, model: str, benchmark: Optional[str]) -> str:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    m = MODELS[model]
    if stage == "refiner":
        if m not in ("q3", "q72") or benchmark not in ("bird", "spider"):
            raise ValueError("the Refiner rows are Qwen3-32B or Qwen2.5-72B on BIRD or Spider")
        return f"refiner_{m}_{benchmark}"
    if stage == "unit_tester":
        if m != "q3" or benchmark not in (None, "bird"):
            raise ValueError("the Unit Tester row is Qwen3-32B on BIRD")
        return "ut_sc"
    if m not in ("q3", "q7"):
        raise ValueError("the selector rows are Qwen3-32B or Qwen2.5-7B")
    return f"selector_{m}"


@dataclass(frozen=True)
class StageDecomposition:
    n: int
    n_changed: int
    change_rate: Interval
    gains: int
    losses: int
    both_wrong: int
    gain_if_changed: float
    gain_if_changed_interval: Optional[Tuple[float, float]]
    total: Interval

    def as_dict(self) -> dict:
        return asdict(self)


def stage_units(stage: str, model: str = "qwen3-32b", benchmark: Optional[str] = "bird",
                inputs=None) -> List[Unit]:
    cell = _cell(stage, model, benchmark)
    inp = _inp("fig_stage-act", inputs)
    if stage == "refiner":
        rows = _read("fig_stage-act", f"macsql_units_{MODELS[model]}_{benchmark}.csv", inp)
        return [((int(r["question_id"]),), r["db_id"], int(r["A_result"]), int(r["Y_stage"]), int(r["Y_alt"]))
                for r in rows]
    if stage == "unit_tester":
        rows = _read("fig_stage-act", "chess_units.csv", inp)
        return [((int(r["question_id"]),), r["db_id"], int(r["A_result"]), int(r["Y_stage"]), int(r["Y_alt"]))
                for r in rows if r["alternative"] == "SC_official"]
    mt = cell.split("_")[1]
    return [((r["dataset"], r["id_left"]), r["dataset"], int(r["changed"]), int(r["selector_correct"]),
             int(r["ranker_correct"]))
            for r in _read("fig_stage-act", "comem_anchors.csv", inp) if r["model"] == mt]


def _database_boot(values: dict, clusters: dict) -> Tuple[float, float]:
    r = stats.cluster_bootstrap(values, clusters, "unit", stats.DRAWS, stats.SEED_STAGE)
    return r["lo"], r["hi"]


def _dataset_boot(values: dict, strata: dict) -> Tuple[float, float]:
    g: Dict[str, list] = {}
    for k, v in values.items():
        g.setdefault(strata[k], []).append(v)
    r = stats.stratified_bootstrap(g, stats.DRAWS, stats.SEED_STAGE)
    return r["lo"], r["hi"]


def _decompose(units: List[Unit], boot) -> dict:
    D = {k: ys - ya for k, _, _, ys, ya in units}
    cl = {k: c for k, c, _, _, _ in units}
    act = [u for u in units if u[2]]
    n, nA = len(units), len(act)
    assert all(D[k] == 0 for k, _, a, _, _ in units if not a)
    r = dict(n=n, n_A=nA, P_A=100 * nA / n,
             gains=sum(D[u[0]] > 0 for u in act), losses=sum(D[u[0]] < 0 for u in act),
             both_wrong=sum(1 for u in act if not u[3] and not u[4]),
             total=100 * sum(D.values()) / n,
             E_D_A=100 * sum(D[u[0]] for u in act) / nA if nA else float("nan"))
    r["total_lo"], r["total_hi"] = (100 * x for x in boot(D, cl))
    r["P_A_lo"], r["P_A_hi"] = (100 * x for x in boot({k: float(a) for k, _, a, _, _ in units}, cl))
    if nA >= MIN_CHANGED:
        r["E_D_A_lo"], r["E_D_A_hi"] = (100 * x for x in boot({u[0]: D[u[0]] for u in act}, cl))
    return r


def decompose(units: List[Unit], resample: str = "database") -> StageDecomposition:
    boot = {"database": _database_boot, "dataset": _dataset_boot}[resample]
    r = _decompose(units, boot)
    cond = (r["E_D_A_lo"], r["E_D_A_hi"]) if "E_D_A_lo" in r else None
    return StageDecomposition(
        n=r["n"], n_changed=r["n_A"], change_rate=Interval(r["P_A"], r["P_A_lo"], r["P_A_hi"]),
        gains=int(r["gains"]), losses=int(r["losses"]), both_wrong=int(r["both_wrong"]),
        gain_if_changed=r["E_D_A"], gain_if_changed_interval=cond,
        total=Interval(r["total"], r["total_lo"], r["total_hi"]))


def stage_decomposition(stage: str, model: str = "qwen3-32b", benchmark: Optional[str] = "bird",
                        inputs=None) -> StageDecomposition:
    return decompose(stage_units(stage, model, benchmark, inputs),
                     "dataset" if stage == "selector" else "database")


def stage_share(stage: str, model: str = "qwen3-32b", benchmark: Optional[str] = "bird",
                inputs=None) -> Dict[str, float]:
    cell = _cell(stage, model, benchmark)
    inp = _inp("fig_stage-act", inputs)
    tok = lambda xs: sum(int(r["prompt_tokens"]) + int(r["completion_tokens"]) for r in xs)
    if stage == "refiner":
        rs = [r for r in _read("fig_stage-act", "macsql_calls_per_question.csv", inp)
              if r["model"] == MODELS[model] and r["benchmark"] == benchmark]
        st = [r for r in rs if r["agent"] == "refiner"]
    elif stage == "unit_tester":
        rs = _read("fig_stage-act", "chess_costs_per_question.csv", inp)
        st = [r for r in rs if r["part"] == "ut_increment"]
    else:
        return dict(calls=100.0, tokens=100.0)
    return dict(calls=100 * sum(int(r["calls"]) for r in st) / sum(int(r["calls"]) for r in rs),
                tokens=100 * tok(st) / tok(rs))


@dataclass(frozen=True)
class StageMinusRule:
    tasks: int
    change: Interval
    extra_correct: int


def stage_minus_rule(stage: str, model: str = "qwen3-32b", benchmark: Optional[str] = "bird",
                     rule: str = "self_consistency", weighting: str = "question", population: str = "all",
                     inputs=None) -> StageMinusRule:
    _cell(stage, model, benchmark)
    inp = _inp("tab_replacement-cost", inputs)
    if stage == "refiner":
        rs = _read("tab_replacement-cost", f"macsql_units_{MODELS[model]}_{benchmark}.csv", inp)
        D = {(int(r["question_id"]),): int(r["Y_stage"]) - int(r["Y_alt"]) for r in rs}
        cl = {(int(r["question_id"]),): r["db_id"] for r in rs}
        b = stats.cluster_bootstrap(D, cl, "unit", stats.DRAWS, stats.SEED_STAGE)
    elif stage == "selector":
        rs = [r for r in _read("tab_replacement-cost", "comem_anchors.csv", inp) if r["model"] == MODELS[model]]
        D = {(r["dataset"], r["id_left"]): int(r["selector_correct"]) - int(r["ranker_correct"]) for r in rs}
        g: Dict[str, list] = {}
        for k, v in D.items():
            g.setdefault(k[0], []).append(v)
        b = stats.stratified_bootstrap(g, stats.DRAWS, stats.SEED_STAGE)
    else:
        cq = _read("tab_replacement-cost", "chess_questions.csv", inp)
        keep = [r for r in cq if population == "all" or int(r["ut_entered"])]
        D = {(int(r["question_id"]),): int(r["UT"]) - int(r[VOTES[rule]]) for r in keep}
        cl = {(int(r["question_id"]),): r["db_id"] for r in keep}
        w = {"question": "unit", "database": "cluster_equal"}[weighting]
        b = stats.cluster_bootstrap(D, cl, w, stats.DRAWS, stats.SEED_STAGE)
    pt, lo, hi = b["point"], b["lo"], b["hi"]
    return StageMinusRule(tasks=len(D), change=Interval(100 * pt, 100 * lo, 100 * hi),
                          extra_correct=int(sum(D.values())))


@dataclass(frozen=True)
class ReplacementCost:
    tasks: int
    calls: int
    tokens_in: int
    tokens_out: int
    gpu_seconds_share: Optional[float]
    change: Interval
    extra_correct_per_million_tokens: Optional[float]


def replacement_cost(stage: str, model: str = "qwen3-32b", benchmark: Optional[str] = "bird",
                     inputs=None) -> ReplacementCost:
    from share_eval_suite.reports import cost
    q = stage_minus_rule(stage, model, benchmark, inputs=inputs)
    w = cost.stage_work(stage, model, benchmark, inputs)
    gpu = cost.stage_gpu_seconds(stage, model, benchmark, inputs).share if stage != "selector" else None
    per_m = None if stage == "unit_tester" else q.extra_correct / ((w.prompt_tokens + w.completion_tokens) / 1e6)
    return ReplacementCost(tasks=q.tasks, calls=w.calls, tokens_in=w.prompt_tokens, tokens_out=w.completion_tokens,
                           gpu_seconds_share=gpu, change=q.change, extra_correct_per_million_tokens=per_m)


_SQL_RULES = {"unfiltered": "unfiltered_correct", "valid": "valid_correct", "nonempty": "nonempty_correct"}
_RECIPIENTS = {"qwen2.5-7b": "qwen2.5-7b", "qwen2.5-32b": "qwen2.5-32b"}


def _pool_rows(benchmark, pool, recipient, inputs):
    pq = {(r["benchmark"], r["pool"], int(r["question_id"])): r
          for r in _read("tab_sql-rule-strength", "qwen25_pools_per_question.csv", inputs)}
    runs = [e for e in _read("tab_sql-rule-strength", "qwen25_pools_per_execution.csv", inputs)
            if (e["benchmark"], e["pool"], e["recipient"]) == (benchmark, pool, _RECIPIENTS[recipient])]

    def y(e, m):
        if m == "generation":
            return int(e["generation_correct"])
        return int(pq[(e["benchmark"], e["pool"], int(e["question_id"]))][_SQL_RULES[m]])
    return pq, runs, y


def execution_filter_contrast(benchmark: str, pool: str, first: str, second: str, recipient: str = "qwen2.5-32b",
                              stratum: str = "all", inputs=None) -> Interval:
    pq, runs, y = _pool_rows(benchmark, pool, recipient, inputs)
    if stratum != "all":
        runs = [e for e in runs if pq[(benchmark, pool, int(e["question_id"]))]["reference_stratum"] == stratum]
    return database_resample_bootstrap([(e["db_id"], float(y(e, first) - y(e, second))) for e in runs]).scaled(100)


def empty_answer_break_even(benchmark: str = "spider", pool: str = "pool4", inputs=None) -> Dict[str, float]:
    pq, runs, y = _pool_rows(benchmark, pool, "qwen2.5-32b", inputs)
    st = {e["question_id"]: pq[(benchmark, pool, int(e["question_id"]))]["reference_stratum"] for e in runs}
    ne = [e for e in runs if st[e["question_id"]] == "gold_nonempty"]
    em = [e for e in runs if st[e["question_id"]] == "gold_empty"]
    dn = sum(y(e, "nonempty") - y(e, "valid") for e in ne) / len(ne)
    de = sum(y(e, "nonempty") - y(e, "valid") for e in em) / len(em)
    return dict(nonempty_answer_change=100 * dn, empty_answer_change=100 * de,
                empty_answer_share=100 * len(em) / (len(em) + len(ne)), break_even_share=100 * dn / (dn - de))


__all__ = ["STAGES", "MODELS", "VOTES", "StageDecomposition", "StageMinusRule", "ReplacementCost", "stage_units",
           "decompose", "stage_decomposition", "stage_share", "stage_minus_rule", "replacement_cost",
           "execution_filter_contrast", "empty_answer_break_even"]
