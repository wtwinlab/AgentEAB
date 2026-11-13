"""Stage replacements: an LLM stage against a rule that makes no LLM call, on the same logged input.

Three stages are compared with their rules: the MAC-SQL Refiner against the Decomposer's query, the
CHESS Unit Tester against a vote over the candidates (self-consistency, Valid or Nonempty), and the
ComEM selector against the ranker's own decision. For each task u,

    D_u = Y_u(stage) - Y_u(rule) in {-1, 0, 1},   A_u = 1 when the two outputs fall in different
    scorer classes (different execution results for SQL; a different decision for matching),

and the stage-minus-rule change decomposes as Eq. 4:  mean D = P(A) * E[D | A], the change rate
times the conditional gain (corrections minus regressions over the changed tasks). A correction is a
task the stage gets right and the rule wrong; a regression is the reverse. SQL rows resample
databases (question weights); matching rows resample query records within each of the 8 datasets.
The replacement also removes the stage's calls and tokens (see `share_eval_suite.reports.cost`).

Models are named "qwen3-32b", "qwen2.5-72b" and "qwen2.5-7b"; benchmarks "bird" and "spider".
Values are in percent or percentage points (pp) unless a field says it is a count.
"""
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
MIN_CHANGED = 30          # no interval for the gain when the stage changes the result below 30 tasks

Unit = Tuple[tuple, str, int, int, int]      # (task key, cluster, A, Y_stage, Y_rule)


def _inp(obj: str, inputs) -> pathlib.Path:
    """The inputs directory: `inputs` when given, else the scored tables shipped for `obj`."""
    return pathlib.Path(inputs) if inputs else paths.inputs(obj)


def _read(obj: str, name: str, inputs) -> List[dict]:
    """Rows of the table `name` (CSV with a header) from `inputs` or from the tables shipped for `obj`."""
    with open(_inp(obj, inputs) / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _cell(stage: str, model: str, benchmark: Optional[str]) -> str:
    """The key of a stage row in the shipped values, e.g. refiner_q3_bird, ut_sc, selector_q3."""
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


# ---------------------------------------------------------------- Eq. 4 decomposition
@dataclass(frozen=True)
class StageDecomposition:
    """Stage minus rule over n tasks, split into the change rate and the conditional gain. Rates and
    changes are in percent / pp; `gains` (corrections), `losses` (regressions) and `both_wrong` count
    changed tasks."""
    n: int
    n_changed: int
    change_rate: Interval                    # P(A), %
    gains: int                               # corrections
    losses: int                              # regressions
    both_wrong: int                          # changed tasks on which stage and rule are both wrong
    gain_if_changed: float                   # conditional gain E[D | A], pp (nan when no task changes)
    gain_if_changed_interval: Optional[Tuple[float, float]]   # only with at least 30 changed tasks
    total: Interval                          # mean D = stage minus rule, pp

    def as_dict(self) -> dict:
        return asdict(self)


def stage_units(stage: str, model: str = "qwen3-32b", benchmark: Optional[str] = "bird",
                inputs=None) -> List[Unit]:
    """The paired tasks of one stage row: (task key, cluster, A, Y_stage, Y_rule).

    The cluster is the database for SQL rows and the dataset for matching rows. Unit Tester rows
    compare with the self-consistency vote. Default inputs: `paper_objects/fig_stage-act/inputs`.
    """
    cell = _cell(stage, model, benchmark)
    inp = _inp("fig_stage-act", inputs)
    # A is the change of execution-result class (SQL) or of the decision (matching)
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
    """Interval ends of a question-weighted mean under the database bootstrap (seed 20261105)."""
    r = stats.cluster_bootstrap(values, clusters, "unit", stats.DRAWS, stats.SEED_STAGE)
    return r["lo"], r["hi"]


def _dataset_boot(values: dict, strata: dict) -> Tuple[float, float]:
    """Interval ends of the pooled mean with tasks resampled within each dataset (seed 20261105);
    tasks enter their dataset in the order of `values`."""
    g: Dict[str, list] = {}
    for k, v in values.items():
        g.setdefault(strata[k], []).append(v)
    r = stats.stratified_bootstrap(g, stats.DRAWS, stats.SEED_STAGE)
    return r["lo"], r["hi"]


def _decompose(units: List[Unit], boot) -> dict:
    """Eq. 4 on (key, cluster, A, Y_stage, Y_rule) units; rates and changes in percent / pp."""
    D = {k: ys - ya for k, _, _, ys, ya in units}
    cl = {k: c for k, c, _, _, _ in units}
    act = [u for u in units if u[2]]
    n, nA = len(units), len(act)
    assert all(D[k] == 0 for k, _, a, _, _ in units if not a)       # A = 0 implies D = 0
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
    """Eq. 4 on paired tasks; `resample` is "database" (cluster bootstrap, question weights) or
    "dataset" (tasks resampled within each dataset). 10,000 draws, seed 20261105."""
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
    """Eq. 4 for one stage row (Fig. 6, Table S24)."""
    return decompose(stage_units(stage, model, benchmark, inputs),
                     "dataset" if stage == "selector" else "database")


def stage_share(stage: str, model: str = "qwen3-32b", benchmark: Optional[str] = "bird",
                inputs=None) -> Dict[str, float]:
    """The stage's share (%) of its pipeline's LLM calls and of its prompt plus completion tokens."""
    cell = _cell(stage, model, benchmark)
    inp = _inp("fig_stage-act", inputs)
    tok = lambda xs: sum(int(r["prompt_tokens"]) + int(r["completion_tokens"]) for r in xs)  # noqa: E731
    if stage == "refiner":
        rs = [r for r in _read("fig_stage-act", "macsql_calls_per_question.csv", inp)
              if r["model"] == MODELS[model] and r["benchmark"] == benchmark]
        st = [r for r in rs if r["agent"] == "refiner"]
    elif stage == "unit_tester":
        rs = _read("fig_stage-act", "chess_costs_per_question.csv", inp)
        st = [r for r in rs if r["part"] == "ut_increment"]
    else:
        return dict(calls=100.0, tokens=100.0)      # the selector is ComEM's only LLM call
    return dict(calls=100 * sum(int(r["calls"]) for r in st) / sum(int(r["calls"]) for r in rs),
                tokens=100 * tok(st) / tok(rs))


# ---------------------------------------------------------------- stage minus rule (Table VII)
@dataclass(frozen=True)
class StageMinusRule:
    """Quality change of keeping the stage instead of the rule (pp); positive means the replacement
    gives up accuracy. `extra_correct` is the net count: corrections minus regressions, negative when
    the rule is better."""
    tasks: int
    change: Interval
    extra_correct: int


def stage_minus_rule(stage: str, model: str = "qwen3-32b", benchmark: Optional[str] = "bird",
                     rule: str = "self_consistency", weighting: str = "question", population: str = "all",
                     inputs=None) -> StageMinusRule:
    """Stage minus rule with its 95% interval (Table VII).

    SQL rows resample databases with question weights (`weighting="database"` weighs databases
    equally); matching rows resample query records within each dataset. For the Unit Tester, `rule`
    is the vote it is compared with ("self_consistency", "valid", "nonempty") and `population` is
    "all" questions or those on which the Unit Tester is "entered". `weighting`, `rule` and
    `population` are ignored for the Refiner and the selector; `benchmark` is ignored for the selector.
    Default inputs: `paper_objects/tab_replacement-cost/inputs`.
    """
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
        for k, v in D.items():             # tasks enter their dataset in file order
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
    """One row of Table VII: quality change and the LLM work the replacement removes.

    `gpu_seconds_share` is the stage's % of the pipeline's estimated GPU-seconds (SQL rows; None for
    the selector). `extra_correct_per_million_tokens` is the net count of corrections per million
    removed tokens (None for the Unit Tester)."""
    tasks: int
    calls: int
    tokens_in: int
    tokens_out: int
    gpu_seconds_share: Optional[float]
    change: Interval                         # stage minus rule, pp
    extra_correct_per_million_tokens: Optional[float]


def replacement_cost(stage: str, model: str = "qwen3-32b", benchmark: Optional[str] = "bird",
                     inputs=None) -> ReplacementCost:
    """Quality change and removed calls, tokens and GPU-seconds of replacing a stage by its rule."""
    from share_eval_suite.reports import cost
    q = stage_minus_rule(stage, model, benchmark, inputs=inputs)
    w = cost.stage_work(stage, model, benchmark, inputs)
    gpu = cost.stage_gpu_seconds(stage, model, benchmark, inputs).share if stage != "selector" else None
    per_m = None if stage == "unit_tester" else q.extra_correct / ((w.prompt_tokens + w.completion_tokens) / 1e6)
    return ReplacementCost(tasks=q.tasks, calls=w.calls, tokens_in=w.prompt_tokens, tokens_out=w.completion_tokens,
                           gpu_seconds_share=gpu, change=q.change, extra_correct_per_million_tokens=per_m)


# ---------------------------------------------------------------- execution filters on fixed SQL pools
_SQL_RULES = {"unfiltered": "unfiltered_correct", "valid": "valid_correct", "nonempty": "nonempty_correct"}
_RECIPIENTS = {"qwen2.5-7b": "qwen2.5-7b", "qwen2.5-32b": "qwen2.5-32b"}


def _pool_rows(benchmark, pool, recipient, inputs):
    pq = {(r["benchmark"], r["pool"], int(r["question_id"])): r
          for r in _read("tab_sql-rule-strength", "qwen25_pools_per_question.csv", inputs)}
    runs = [e for e in _read("tab_sql-rule-strength", "qwen25_pools_per_execution.csv", inputs)
            if (e["benchmark"], e["pool"], e["recipient"]) == (benchmark, pool, _RECIPIENTS[recipient])]

    def y(e, m):
        # generation is scored per run; a vote over the pool is fixed per question
        if m == "generation":
            return int(e["generation_correct"])
        return int(pq[(e["benchmark"], e["pool"], int(e["question_id"]))][_SQL_RULES[m]])
    return pq, runs, y


def execution_filter_contrast(benchmark: str, pool: str, first: str, second: str, recipient: str = "qwen2.5-32b",
                              stratum: str = "all", inputs=None) -> Interval:
    """First minus second on a fixed SQL pool, in pp (Table VIII).

    `first` and `second` are "generation" (the recipient writes its own query) or a vote over the
    pool: "unfiltered", "valid" (executable candidates), "nonempty" (candidates with a nonempty
    result). `pool` is "pool4" or "pool6" (four or six candidates). Each generation run of a question
    is one observation; the interval resamples databases (5,000 draws, Python `random.Random(20261026)`).
    `stratum` restricts to questions whose reference answer is "gold_nonempty" or "gold_empty".
    """
    pq, runs, y = _pool_rows(benchmark, pool, recipient, inputs)
    if stratum != "all":
        runs = [e for e in runs if pq[(benchmark, pool, int(e["question_id"]))]["reference_stratum"] == stratum]
    return database_resample_bootstrap([(e["db_id"], float(y(e, first) - y(e, second))) for e in runs]).scaled(100)


def empty_answer_break_even(benchmark: str = "spider", pool: str = "pool4", inputs=None) -> Dict[str, float]:
    """Nonempty minus Valid on nonempty-answer and empty-answer questions (pp), the share of
    empty-answer questions (%), and the share at which the two rules break even (%)."""
    pq, runs, y = _pool_rows(benchmark, pool, "qwen2.5-32b", inputs)
    st = {e["question_id"]: pq[(benchmark, pool, int(e["question_id"]))]["reference_stratum"] for e in runs}
    # Nonempty minus Valid at an empty-answer share q is (1 - q) dn + q de, zero at q = dn / (dn - de)
    ne = [e for e in runs if st[e["question_id"]] == "gold_nonempty"]
    em = [e for e in runs if st[e["question_id"]] == "gold_empty"]
    dn = sum(y(e, "nonempty") - y(e, "valid") for e in ne) / len(ne)
    de = sum(y(e, "nonempty") - y(e, "valid") for e in em) / len(em)
    return dict(nonempty_answer_change=100 * dn, empty_answer_change=100 * de,
                empty_answer_share=100 * len(em) / (len(em) + len(ne)), break_even_share=100 * dn / (dn - de))


__all__ = ["STAGES", "MODELS", "VOTES", "StageDecomposition", "StageMinusRule", "ReplacementCost", "stage_units",
           "decompose", "stage_decomposition", "stage_share", "stage_minus_rule", "replacement_cost",
           "execution_filter_contrast", "empty_answer_break_even"]
