"""Candidate availability across Generation, Revision, Extraction and Final decision.

For each run the stage evaluation asks whether a correct answer is still present: among the initial
candidates (C0), among the last-round candidates (CT), among the last-round candidates read by the
protocol extractor (CD), and in the vote (Y). Per run Y <= CD <= CT, and

    1 - Y = (1 - C0) + (C0 - CT) + (CT - CD) + (CD - Y)

splits the error into missing generation, revision, extraction and decision. A second view splits a
gain between two methods by whether the candidate pool holds a correct answer: questions where it does
(the gain could come from a better choice among the candidates) and questions where it does not (the
gain needs an answer that no candidate gave).
"""
from __future__ import annotations

import collections
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from share_eval_suite import paths
from share_eval_suite.records import readers

FLOW_OBJECT = "tab_candidate-flow"
STRATA_OBJECT = "tab_s-pool-strata"
POOL_OBJECT = "text_generation-beyond-pool"

STAGE_FLAGS = ("C0", "CT", "CD", "Y", "Yx", "loss", "recovery", "returned")
MEAN_FLAGS = ("C0", "CT", "CD", "Y", "Yx", "loss", "recovery")
PROTOCOLS = ("Du-MAD", "MAD-M2", "BCCS")
BACKBONES = ("llama3-8b", "qwen2.5-7b")
BASELINE_READING = "protocol_exact_first"
POOL_COMPARISONS = (       # table order
    ("deepseek-v4-pro", "four_model"), ("qwen3-32b", "four_model"), ("qwen3-32b", "qwen3_glm"),
    ("qwen2.5-7b", "four_model"), ("qwen2.5-32b", "four_model"), ("qwen2.5-7b", "six_candidates"),
    ("qwen2.5-32b", "six_candidates"), ("gemma-4-26b", "four_model"), ("gpt-oss-120b", "four_model"))
DRAWS = 10000
SEED_DB = 20261105         # database bootstrap of the pool strata and of the beyond-pool split
SEED_QUESTION = 20261109   # question bootstrap of the pool strata


def configurations() -> List[Tuple[str, str]]:
    """The six (protocol, backbone) configurations of the BIRD debates with peers, in table order."""
    return [(p, b) for p in PROTOCOLS for b in BACKBONES]


def run_flags(row: Mapping[str, Any]) -> Dict[str, int]:
    """Stage flags of one run: C0, CT, CD, the baseline vote Y (protocol extractor, exact-match vote,
    first-agent ties), Yx (lenient extractor, execution vote), loss (C0 and not CT), recovery (CT and
    not C0), the returned answer, and whether CT >= CD >= Y holds."""
    r = {k: str(v) for k, v in row.items()}
    c0, ct, cd = int(r["C0"]), int(r["CT"]), int(r["CD"])
    y, yx = int(r[BASELINE_READING]), int(r["lenient_execution_first"])
    return dict(C0=c0, CT=ct, CD=cd, Y=y, Yx=yx, loss=int(c0 == 1 and ct == 0),
                recovery=int(c0 == 0 and ct == 1), returned=int(r["final"]),
                ordered=int(ct >= cd >= y))


def _by_configuration(runs):
    # runs grouped by configuration and question; the database of each question
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    db = {}
    for r in runs:
        by[(r["protocol"], r["backbone"])][r["question"]].append(r)
        db[r["question"]] = r["db_id"]
    return by, db


def candidate_flow(runs: Optional[Iterable[Mapping[str, Any]]] = None,
                   configs: Optional[Sequence[Tuple[str, str]]] = None) -> Dict[str, Any]:
    """Candidate availability per stage, in % of questions.

    `runs` are candidate-flow rows (default: the shipped BIRD debates with peers). Repeats are averaged
    within a question, on the questions present in every configuration; the mean gives each
    configuration equal weight per question. Returns `questions`, `runs`, `per_configuration`
    ({(protocol, backbone): {flag: %}} with the flags of `run_flags`), `mean` (the flow flags averaged
    over configurations), `stage_losses` (the generation, revision, extraction and decision
    terms of the decomposition, and the gain of Yx over Y, in pp), `identity_holds` and `ordered_every_run`.
    """
    runs = readers.read_candidate_flow_runs() if runs is None else list(runs)
    configs = list(configs or configurations())
    by, _ = _by_configuration(runs)
    units = sorted(set.intersection(*[set(by[c]) for c in configs]))
    per, stack = {}, {t: [] for t in STAGE_FLAGS}
    identity = ordered = True
    for c in configs:
        flags = {u: [run_flags(x) for x in by[c][u]] for u in units}
        arr = {t: np.array([np.mean([f[t] for f in flags[u]]) for u in units], dtype=float) for t in STAGE_FLAGS}
        ordered &= all(f["ordered"] for u in units for f in flags[u])
        lhs = 1 - arr["Y"]
        rhs = (1 - arr["C0"]) + (arr["C0"] - arr["CT"]) + (arr["CT"] - arr["CD"]) + (arr["CD"] - arr["Y"])
        identity &= bool(np.allclose(lhs, rhs, atol=1e-12))
        per[c] = {t: 100 * float(arr[t].mean()) for t in STAGE_FLAGS}
        for t in STAGE_FLAGS:
            stack[t].append(arr[t])
    mean = {t: 100 * float(np.mean(stack[t], axis=0).mean()) for t in MEAN_FLAGS}
    losses = {"generation_missing": 100 - mean["C0"],
              "revision_net": mean["C0"] - mean["CT"],
              "extraction_gap": mean["CT"] - mean["CD"],
              "vote_gap": mean["CD"] - mean["Y"],
              "Yx_minus_Y": mean["Yx"] - mean["Y"]}
    n_runs = sum(len(by[c][u]) for c in configs for u in units)
    return {"questions": len(units), "runs": n_runs, "per_configuration": per, "mean": mean,
            "stage_losses": losses, "identity_holds": bool(identity), "ordered_every_run": bool(ordered)}


# ------------------------------------------------------------------ gains split by pool availability
def _db_equal_bootstrap(values, clusters, seed=SEED_DB):
    """Database bootstrap of the mean with each database weighted equally: per-database means, the
    databases resampled with replacement (10,000 draws). Returns the mean and its 95% percentile
    interval, in pp."""
    by = collections.defaultdict(list)
    for u in sorted(values):
        by[clusters[u]].append(values[u])
    cl = sorted(by)
    per = np.array([np.mean(by[c]) for c in cl])
    draws = np.random.default_rng(seed).integers(0, len(cl), size=(DRAWS, len(cl)))
    bs = per[draws].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return 100 * float(per.mean()), 100 * float(lo), 100 * float(hi)


def _question_bootstrap(values, clusters, seed=SEED_QUESTION):
    """Question bootstrap of the same database-equal mean: each question weighted by 1 / the number of
    questions of its database, questions resampled with replacement (10,000 draws). Returns the 95%
    percentile interval, in pp."""
    qs = sorted(values)
    v = np.array([values[q] for q in qs], float)
    n = collections.Counter(clusters[q] for q in qs)
    w = np.array([1.0 / n[clusters[q]] for q in qs])
    idx = np.random.default_rng(seed).integers(0, len(v), size=(DRAWS, len(v)))
    b = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return 100 * float(lo), 100 * float(hi)


def _strata_estimate(values, clusters):
    # database-equal mean with database (seed 20261105) and question (seed 20261109) intervals
    pt, dlo, dhi = _db_equal_bootstrap(values, clusters)
    qlo, qhi = _question_bootstrap(values, clusters)
    det = int((dlo > 0 or dhi < 0) and (qlo > 0 or qhi < 0))
    return dict(est=pt, db_lo=dlo, db_hi=dhi, q_lo=qlo, q_hi=qhi, detected=det)


def split_by_availability(difference: Mapping[Any, float], available: Mapping[Any, int],
                          clusters: Mapping[Any, str]) -> Dict[str, Dict[str, float]]:
    """Split a per-question paired difference by whether the candidate pool holds a correct answer.

    Each database is weighted equally; each part averages the difference times the stratum indicator,
    so `present` and `absent` add up to `total`. Each part gives the estimate (pp) with 95% database
    and question bootstrap intervals (`db_lo`, `db_hi`, `q_lo`, `q_hi`) and `detected` (both intervals
    exclude 0). The database interval resamples databases (10,000 draws, seed 20261105); the
    question interval resamples questions with weights 1 / database size (10,000 draws, seed
    20261109); each estimate draws from fresh generators.
    """
    est = _strata_estimate
    out = {"total": est(dict(difference), dict(clusters))}
    for s, name in ((1, "present"), (0, "absent")):
        out[name] = est({q: difference[q] * (available[q] == s) for q in difference}, dict(clusters))
    return out


def _pool_rows(path=None):
    # one dict per (generator, pool), keyed by question; generation is the share of correct runs
    data = collections.defaultdict(dict)
    rows = readers.read_csv(path or paths.inputs(STRATA_OBJECT) / "pool_strata_questions.csv")
    for r in rows:
        data[(r["recipient"], r["pool"])][int(r["question_id"])] = dict(
            db=r["db_id"], ref=int(r["has_reference"]), oracle=int(r["correct_candidate"]),
            ne=int(r["nonempty"]), gen=int(r["generation_correct"]) / int(r["generation_runs"]),
            runs=int(r["generation_runs"]), sel=int(r["selector"]) if r["selector"] != "" else None)
    return data


def pool_comparisons() -> List[Tuple[str, str]]:
    """The nine BIRD (generator, candidate pool) comparisons, in table order."""
    return list(POOL_COMPARISONS)


def generation_gain_by_availability(generator: str = "deepseek-v4-pro", pool: str = "four_model",
                                    against: str = "nonempty", with_reference_only: bool = False,
                                    path=None) -> Dict[str, Any]:
    """Generation's gain on BIRD, split by whether the candidate pool holds a correct query.

    `against` is `nonempty` (the Nonempty execution vote) or `selector` (the LLM selector given the same
    candidates); `selector` applies only to the pools with a selector (DeepSeek-V4-Pro and Qwen3-32B
    generators). `with_reference_only` drops the questions whose reference query has no completed
    result. Returns the three parts of `split_by_availability` and the question counts
    (`questions`, `absent_questions`).
    """
    X = _pool_rows(path)[(generator, pool)]
    if against == "selector" and any(v["sel"] is None for v in X.values()):
        raise ValueError(f"the pool of {generator} ({pool}) has no selector; use against='nonempty'")
    qs = [q for q in X if X[q]["ref"]] if with_reference_only else list(X)
    ref = "ne" if against == "nonempty" else "sel"
    diff = {q: X[q]["gen"] - X[q][ref] for q in qs}
    out: Dict[str, Any] = split_by_availability(diff, {q: X[q]["oracle"] for q in qs}, {q: X[q]["db"] for q in qs})
    out["questions"] = len(qs)
    out["absent_questions"] = sum(1 for q in qs if X[q]["oracle"] == 0)
    return out


def generation_gain_range(path=None) -> Dict[str, float]:
    """Smallest and largest present- and absent-candidate parts of generation minus Nonempty over the
    nine comparisons (pp)."""
    parts = [generation_gain_by_availability(g, p, path=path) for g, p in pool_comparisons()]
    pres = [x["present"]["est"] for x in parts]
    absn = [x["absent"]["est"] for x in parts]
    return {"present_min": min(pres), "present_max": max(pres), "absent_min": min(absn), "absent_max": max(absn)}


def method_gain_by_availability(method: str = "DS_G", reference: str = "Nonempty", benchmark: str = "bird",
                                path=None) -> Dict[str, Any]:
    """Gain of one method over another on the four-model pool, split by whether a candidate is correct.

    Methods are the columns of the pool table: `Valid`, `Nonempty` (execution votes), `Q3_S_al`,
    `DS_S_al` (Qwen3-32B and DeepSeek-V4-Pro selectors), `Q3_G`, `DS_G` (their generation). Each
    database is weighted equally; intervals are 95% database bootstrap intervals (10,000 draws, seed
    20261105, a fresh generator per part). Returns, for `total`,
    `present` and `absent`: `est`, `lo`, `hi`, `detected`; for the two strata also `questions`,
    `gained` and `lost` (questions the method answers and the reference does not, and the reverse) and
    `method_correct` (questions the method answers).
    """
    boot = _db_equal_bootstrap
    rows = [{k: (v if k in ("benchmark", "db_id") else int(v)) for k, v in r.items()}
            for r in readers.read_csv(path or paths.inputs(POOL_OBJECT) / "pool_questions.csv")]
    R = [r for r in rows if r["benchmark"] == benchmark]
    cl = {r["question_id"]: r["db_id"] for r in R}

    def part(vals):
        est, lo, hi = boot(vals, cl)
        return {"est": est, "lo": lo, "hi": hi, "detected": bool(lo > 0 or hi < 0)}

    out: Dict[str, Any] = {"total": part({r["question_id"]: r[method] - r[reference] for r in R})}
    for s, name in ((1, "present"), (0, "absent")):
        sub = [r for r in R if r["Oracle"] == s]
        out[name] = part({r["question_id"]: (r[method] - r[reference]) * (r["Oracle"] == s) for r in R})
        out[name].update(questions=len(sub), gained=sum(bool(r[method] and not r[reference]) for r in sub),
                         lost=sum(bool(r[reference] and not r[method]) for r in sub),
                         method_correct=sum(r[method] for r in sub))
    return out
