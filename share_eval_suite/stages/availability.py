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
POOL_COMPARISONS = (
    ("deepseek-v4-pro", "four_model"), ("qwen3-32b", "four_model"), ("qwen3-32b", "qwen3_glm"),
    ("qwen2.5-7b", "four_model"), ("qwen2.5-32b", "four_model"), ("qwen2.5-7b", "six_candidates"),
    ("qwen2.5-32b", "six_candidates"), ("gemma-4-26b", "four_model"), ("gpt-oss-120b", "four_model"))
DRAWS = 10000
SEED_DB = 20261105
SEED_QUESTION = 20261109


def configurations() -> List[Tuple[str, str]]:
    return [(p, b) for p in PROTOCOLS for b in BACKBONES]


def run_flags(row: Mapping[str, Any]) -> Dict[str, int]:
    r = {k: str(v) for k, v in row.items()}
    c0, ct, cd = int(r["C0"]), int(r["CT"]), int(r["CD"])
    y, yx = int(r[BASELINE_READING]), int(r["lenient_execution_first"])
    return dict(C0=c0, CT=ct, CD=cd, Y=y, Yx=yx, loss=int(c0 == 1 and ct == 0),
                recovery=int(c0 == 0 and ct == 1), returned=int(r["final"]),
                ordered=int(ct >= cd >= y))


def _by_configuration(runs):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    db = {}
    for r in runs:
        by[(r["protocol"], r["backbone"])][r["question"]].append(r)
        db[r["question"]] = r["db_id"]
    return by, db


def candidate_flow(runs: Optional[Iterable[Mapping[str, Any]]] = None,
                   configs: Optional[Sequence[Tuple[str, str]]] = None) -> Dict[str, Any]:
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


def _db_equal_bootstrap(values, clusters, seed=SEED_DB):
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
    qs = sorted(values)
    v = np.array([values[q] for q in qs], float)
    n = collections.Counter(clusters[q] for q in qs)
    w = np.array([1.0 / n[clusters[q]] for q in qs])
    idx = np.random.default_rng(seed).integers(0, len(v), size=(DRAWS, len(v)))
    b = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return 100 * float(lo), 100 * float(hi)


def _strata_estimate(values, clusters):
    pt, dlo, dhi = _db_equal_bootstrap(values, clusters)
    qlo, qhi = _question_bootstrap(values, clusters)
    det = int((dlo > 0 or dhi < 0) and (qlo > 0 or qhi < 0))
    return dict(est=pt, db_lo=dlo, db_hi=dhi, q_lo=qlo, q_hi=qhi, detected=det)


def split_by_availability(difference: Mapping[Any, float], available: Mapping[Any, int],
                          clusters: Mapping[Any, str]) -> Dict[str, Dict[str, float]]:
    est = _strata_estimate
    out = {"total": est(dict(difference), dict(clusters))}
    for s, name in ((1, "present"), (0, "absent")):
        out[name] = est({q: difference[q] * (available[q] == s) for q in difference}, dict(clusters))
    return out


def _pool_rows(path=None):
    data = collections.defaultdict(dict)
    rows = readers.read_csv(path or paths.inputs(STRATA_OBJECT) / "pool_strata_questions.csv")
    for r in rows:
        data[(r["recipient"], r["pool"])][int(r["question_id"])] = dict(
            db=r["db_id"], ref=int(r["has_reference"]), oracle=int(r["correct_candidate"]),
            ne=int(r["nonempty"]), gen=int(r["generation_correct"]) / int(r["generation_runs"]),
            runs=int(r["generation_runs"]), sel=int(r["selector"]) if r["selector"] != "" else None)
    return data


def pool_comparisons() -> List[Tuple[str, str]]:
    return list(POOL_COMPARISONS)


def generation_gain_by_availability(generator: str = "deepseek-v4-pro", pool: str = "four_model",
                                    against: str = "nonempty", with_reference_only: bool = False,
                                    path=None) -> Dict[str, Any]:
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
    parts = [generation_gain_by_availability(g, p, path=path) for g, p in pool_comparisons()]
    pres = [x["present"]["est"] for x in parts]
    absn = [x["absent"]["est"] for x in parts]
    return {"present_min": min(pres), "present_max": max(pres), "absent_min": min(absn), "absent_max": max(absn)}


def method_gain_by_availability(method: str = "DS_G", reference: str = "Nonempty", benchmark: str = "bird",
                                path=None) -> Dict[str, Any]:
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
