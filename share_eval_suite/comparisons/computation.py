from __future__ import annotations

import collections
import functools
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from share_eval_suite import stats
from share_eval_suite.comparisons.replacement import _inp, _read
from share_eval_suite.reports.intervals import Interval, TwoIntervals, question_and_database_bootstrap

PROTOCOLS = {"du-mad": "dumad", "mad-m2": "madm2"}
BACKBONES = {"llama3-8b": "llama", "qwen2.5-7b": "qwen"}
BUDGETS = {"3x2": 6, "3x3": 9, "5x2": 10, "3x4": 12}


def _two(V: dict, key: str) -> TwoIntervals:
    e = V[key]
    return TwoIntervals(e, Interval(e, V[key + "_q_lo"], V[key + "_q_hi"]), Interval(e, V[key + "_db_lo"], V[key + "_db_hi"]))


def _pp(x) -> float:
    return 100 * float(x)


_BUDGET_SEED, _BUDGET_DRAWS, _MAX_DRAFTS = 20261109, 10000, 20
_PROTOS = ("du_mad", "madm2_objective")
_BBS = ("llama3-8b", "qwen2.5-7b")
_PN = {"du_mad": "dumad", "madm2_objective": "madm2"}
_DRAFT_SETS = {"du_mad": ("protocol", "t0.7"), "madm2_objective": ("protocol",)}


def _vote(labels) -> Optional[int]:
    live = [x for x in labels if x is not None]
    if not live:
        return None
    cnt = collections.Counter(live)
    top = max(cnt.values())
    return next(i for i, x in enumerate(labels) if x is not None and cnt[x] == top)


@functools.lru_cache(maxsize=4)
def _fixed_values(inp: str) -> dict:
    drafts = collections.defaultdict(list)
    lab = lambda x: None if x == "" else int(x)
    for r in _read("fig_debate-budget", "fixed_response_drafts.csv", inp):
        drafts[(r["protocol"], r["backbone"], r["question"])].append(
            (int(r["draft"]), lab(r["lenient_exec_label"]), int(r["lenient_correct"]),
             lab(r["strict_string_label"]), int(r["strict_correct"])))
    runs = collections.defaultdict(list)
    dbmap = {}
    for r in _read("fig_debate-budget", "fixed_response_runs.csv", inp):
        runs[(r["protocol"], r["backbone"], r["question"])].append((int(r["final"]), int(r["last_lenient_exec"])))
        dbmap[r["question"]] = r["db_id"]
    units = sorted(dbmap)
    per = collections.defaultdict(list)
    for p in _PROTOS:
        for b in _BBS:
            for u in units:
                d = sorted(drafts[(p, b, u)])
                i = _vote([x[1] for x in d])
                v_exec = float(i is not None and d[i][2])
                i = _vote([x[3] for x in d])
                v_str = float(i is not None and d[i][4])
                rr = runs[(p, b, u)]
                fin = np.mean([x[0] for x in rr])
                yx = np.mean([x[1] for x in rr])
                per["exec_minus_final"].append(v_exec - fin)
                per["exec_minus_last"].append(v_exec - yx)
                per["string_minus_final"].append(v_str - fin)
    n = len(units)
    q = stats.resample_index(n, _BUDGET_DRAWS, _BUDGET_SEED)
    dbs = sorted(set(dbmap.values()))
    mem = [np.array([j for j, u in enumerate(units) if dbmap[u] == d]) for d in dbs]
    rng = np.random.default_rng(_BUDGET_SEED)
    dd = [np.concatenate([mem[j] for j in rng.integers(0, len(dbs), size=len(dbs))]) for _ in range(_BUDGET_DRAWS)]
    V = {}
    for name, vals in per.items():
        v = np.array(vals).reshape(4, n).mean(0)
        lo, hi = np.percentile(v[q].mean(1), [2.5, 97.5])
        dlo, dhi = np.percentile([v[ix].mean() for ix in dd], [2.5, 97.5])
        k = f"fixed_{name}"
        V[k] = _pp(v.mean())
        V[k + "_q_lo"], V[k + "_q_hi"] = _pp(lo), _pp(hi)
        V[k + "_db_lo"], V[k + "_db_hi"] = _pp(dlo), _pp(dhi)
    return V


def fixed_response_votes(inputs=None) -> Dict[str, TwoIntervals]:
    V = _fixed_values(str(_inp("fig_debate-budget", inputs)))
    return {"execution_vote_minus_output": _two(V, "fixed_exec_minus_final"),
            "execution_vote_minus_last_round": _two(V, "fixed_exec_minus_last"),
            "exact_match_vote_minus_output": _two(V, "fixed_string_minus_final")}


class _Budget:

    def __init__(self, inp):
        self.runs = collections.defaultdict(lambda: collections.defaultdict(list))
        self.failed = collections.Counter()
        db_of = {}
        for r in _read("fig_debate-budget", "debate_runs.csv", inp):
            key = (r["protocol"], r["backbone"], r["config"])
            db_of[r["question"]] = r["db_id"]
            if r["status"] == "ok":
                self.runs[key][r["question"]].append(r)
            else:
                self.failed[key + (r["question"],)] += 1
        self.units = sorted(db_of)
        self.uix = {u: i for i, u in enumerate(self.units)}
        dbs = sorted(set(db_of.values()))
        n = len(self.units)
        rng = np.random.default_rng(_BUDGET_SEED)
        self.WQ = rng.multinomial(n, [1 / n] * n, size=_BUDGET_DRAWS).astype(float)
        self.WD = rng.multinomial(len(dbs), [1 / len(dbs)] * len(dbs), size=_BUDGET_DRAWS).astype(float)[
            :, [dbs.index(db_of[u]) for u in self.units]]
        self.drafts = collections.defaultdict(list)
        for r in _read("fig_debate-budget", "drafts.csv", inp):
            self.drafts[(r["protocol"], r["backbone"], r["draft_set"], r["question"])].append(
                (int(r["draft"]), None if r["result"] == "" else int(r["result"]), int(r["correct"]),
                 int(r["prompt_tokens"]) + int(r["completion_tokens"])))

    def boot(self, x, mask) -> dict:
        x = np.where(mask, x, 0.0)
        m = mask.astype(float)
        out = dict(est=_pp(x.sum() / max(m.sum(), 1)), n=int(m.sum()))
        for tag, W in (("q", self.WQ), ("db", self.WD)):
            e = (W @ x) / np.maximum(W @ m, 1e-12)
            lo, hi = np.percentile(e, [2.5, 97.5])
            out[tag + "_lo"], out[tag + "_hi"] = _pp(lo), _pp(hi)
        return out

    def dvec(self, p, b, c, metric, failed_as_wrong=False):
        x = np.full(len(self.units), np.nan)
        for u in self.units:
            vals = [float(r[metric]) for r in self.runs[(p, b, c)].get(u, [])]
            if failed_as_wrong:
                vals += [0.0] * self.failed[(p, b, c, u)]
            if vals:
                x[self.uix[u]] = np.mean(vals)
        return x

    def vote_curves(self, p, b, s):
        n = len(self.units)
        first, tok = (np.full((_MAX_DRAFTS, n), np.nan) for _ in range(2))
        for u in self.units:
            d = sorted(self.drafts[(p, b, s, u)])
            if len(d) < _MAX_DRAFTS:
                continue
            j = self.uix[u]
            lab = [x[1] for x in d]
            cor = [x[2] for x in d]
            ct = np.cumsum([x[3] for x in d])
            for k in range(1, _MAX_DRAFTS + 1):
                i = _vote(lab[:k])
                first[k - 1, j] = float(i is not None and cor[i])
                tok[k - 1, j] = ct[k - 1]
        return first, tok


@functools.lru_cache(maxsize=4)
def _budget_values(inp: str) -> dict:
    bd = _Budget(inp)
    V: dict = {}

    def put(key, r):
        V[key] = r["est"]
        for f in ("q_lo", "q_hi", "db_lo", "db_hi"):
            V[f"{key}_{f}"] = r[f]

    for p in _PROTOS:
        for b in _BBS:
            for c in BUDGETS:
                rs = [r for u in bd.units for r in bd.runs[(p, b, c)].get(u, [])]
                k = f"{_PN[p]}_{BACKBONES[b]}_{c}"
                V[f"{k}_calls"] = float(np.mean([int(r["calls"]) for r in rs]))
                V[f"{k}_tokens"] = float(np.mean([int(r["prompt_tokens"]) + int(r["completion_tokens"]) for r in rs]))
                V[f"{k}_failed"] = sum(v for kk, v in bd.failed.items() if kk[:3] == (p, b, c))
                V[f"{k}_last_acc"] = _pp(np.nanmean(bd.dvec(p, b, c, "last_vote_correct")))

    for c in ("3x3", "5x2", "3x4"):
        for faw in (False, True):
            tag = "faw_" if faw else ""
            per = {(p, b): bd.dvec(p, b, c, "last_vote_correct", faw) - bd.dvec(p, b, "3x2", "last_vote_correct")
                   for p in _PROTOS for b in _BBS}
            for p in _PROTOS:
                D = np.vstack([per[(p, b)] for b in _BBS])
                put(f"{tag}{_PN[p]}_{c}_minus_base", bd.boot(np.nan_to_num(D.mean(0)), np.isfinite(D).all(0)))
            D = np.vstack(list(per.values()))
            put(f"{tag}all4_{c}_minus_base", bd.boot(np.nan_to_num(D.mean(0)), np.isfinite(D).all(0)))

    curves = {}
    for p in _PROTOS:
        for b in _BBS:
            for s in _DRAFT_SETS[p]:
                first, tok = bd.vote_curves(p, b, s)
                curves[(p, b, s)] = (first, tok)
                k = f"{_PN[p]}_{BACKBONES[b]}_{s.replace('.', '')}"
                for kk in range(1, _MAX_DRAFTS + 1):
                    V[f"{k}_vote{kk}_acc"] = _pp(np.nanmean(first[kk - 1]))

    for c, n in BUDGETS.items():
        for s in ("protocol", "t0.7"):
            cm, tm = {}, {}
            for p in _PROTOS:
                if s not in _DRAFT_SETS[p]:
                    continue
                for b in _BBS:
                    first, tok = curves[(p, b, s)]
                    x = bd.dvec(p, b, c, "last_vote_correct")
                    cm[(p, b)] = x - first[n - 1]
                    k = f"{_PN[p]}_{BACKBONES[b]}_{c}_{s.replace('.', '')}"
                    put(f"call_{k}", bd.boot(np.nan_to_num(cm[(p, b)]), np.isfinite(cm[(p, b)])))
                    mt = np.nanmean(tok, axis=1)
                    ks = [kk for kk in range(1, _MAX_DRAFTS + 1) if mt[kk - 1] >= V[f"{_PN[p]}_{BACKBONES[b]}_{c}_tokens"]]
                    kstar = ks[0] if ks else _MAX_DRAFTS
                    V[f"kstar_{k}"] = kstar
                    tm[(p, b)] = x - first[kstar - 1]
                    put(f"token_{k}", bd.boot(np.nan_to_num(tm[(p, b)]), np.isfinite(tm[(p, b)])))
            for tag, dd in (("call", cm), ("token", tm)):
                for p in _PROTOS:
                    if all((p, b) in dd for b in _BBS):
                        D = np.vstack([dd[(p, b)] for b in _BBS])
                        put(f"{tag}_{_PN[p]}_{c}_{s.replace('.', '')}",
                            bd.boot(np.nan_to_num(D.mean(0)), np.isfinite(D).all(0)))
    return V


def _budget_key(protocol, backbone, config, drafts):
    p = PROTOCOLS[protocol]
    if config not in BUDGETS:
        raise ValueError(f"config must be one of {tuple(BUDGETS)}")
    s = {"protocol": "protocol", "sampled": "t07"}[drafts]
    return f"{p}_{BACKBONES[backbone]}_{config}_{s}" if backbone else f"{p}_{config}_{s}"


def debate_minus_vote(protocol: str, config: str, matched: str = "calls", drafts: str = "protocol",
                      backbone: Optional[str] = None, inputs=None) -> TwoIntervals:
    V = _budget_values(str(_inp("fig_debate-budget", inputs)))
    tag = {"calls": "call", "tokens": "token"}[matched]
    return _two(V, f"{tag}_" + _budget_key(protocol, backbone, config, drafts))


def matched_draft_count(protocol: str, backbone: str, config: str, drafts: str = "protocol", inputs=None) -> int:
    V = _budget_values(str(_inp("fig_debate-budget", inputs)))
    return int(V["kstar_" + _budget_key(protocol, backbone, config, drafts)])


def larger_debate_minus_base(protocol: Optional[str], config: str, failed_as_wrong: bool = False,
                             inputs=None) -> TwoIntervals:
    V = _budget_values(str(_inp("fig_debate-budget", inputs)))
    head = PROTOCOLS[protocol] if protocol else "all4"
    return _two(V, ("faw_" if failed_as_wrong else "") + f"{head}_{config}_minus_base")


def budget_accuracy(protocol: str, backbone: str, inputs=None) -> Dict[int, Dict[str, float]]:
    V = _budget_values(str(_inp("fig_debate-budget", inputs)))
    k = f"{PROTOCOLS[protocol]}_{BACKBONES[backbone]}"
    return {n: dict(debate=V[f"{k}_{c}_last_acc"], vote=V[f"{k}_protocol_vote{n}_acc"]) for c, n in BUDGETS.items()}


SCALE_PROTOCOLS = {"du-mad": "du_mad", "mad-m2": "madm2_objective", "bccs": "bccs", "mean": "mean"}
_SCALE_EXCLUDED = {"bird|518", "bird|701"}
_SCALE_DRAWS, _SCALE_SEED = 10000, stats.SEED_REPLAY
_SCALE_MODELS = {"7B": "qwen2.5-7b", "14B": "qwen25-14b-awq", "32B": "qwen25-32b-awq", "72B": "qwen25-72b-awq"}
_SCALE_PAIRS = (("72b_7b", "qwen25-72b-awq", "qwen2.5-7b"), ("14b_7b", "qwen25-14b-awq", "qwen2.5-7b"),
                ("32b_14b", "qwen25-32b-awq", "qwen25-14b-awq"), ("72b_32b", "qwen25-72b-awq", "qwen25-32b-awq"),
                ("72b_14b", "qwen25-72b-awq", "qwen25-14b-awq"),
                ("gptoss_120b_20b", "gpt-oss-120b", "gpt-oss-20b"), ("llama_70b_8b", "llama33-70b", "llama3-8b"))
_SCALE_HOLM = ("72b_7b", "14b_7b", "32b_14b", "72b_32b", "72b_14b")


def _pp2(v) -> float:
    return round(100 * float(v), 2)


def _round_change(rows) -> Dict[str, float]:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["question"]].append(r["last"] - r["initial"])
    return {u: float(np.mean(v)) for u, v in by.items()}


@functools.lru_cache(maxsize=4)
def _scale(inp: str):
    cells, db = collections.defaultdict(list), {}
    for r in _read("tab_scale", "debate_runs.csv", inp):
        if r["question"] in _SCALE_EXCLUDED:
            continue
        db[r["question"]] = r["database"]
        cells[(r["model"], r["protocol"], r["condition"])].append(
            dict(question=r["question"], repeat=int(r["repeat"]),
                 initial=float(r["initial_correct"]), last=float(r["last_correct"])))
    dp = {}
    for m in {k[0] for k in cells}:
        for p in ("du_mad", "madm2_objective", "bccs"):
            F, N = cells.get((m, p, "with_peers")), cells.get((m, p, "no_peers"))
            if F and N:
                ef, en = _round_change(F), _round_change(N)
                dp[(m, p)] = {u: ef[u] - en[u] for u in set(ef) & set(en)}
        if (m, "du_mad") in dp and (m, "madm2_objective") in dp:
            a, b = dp[(m, "du_mad")], dp[(m, "madm2_objective")]
            dp[(m, "mean")] = {u: float(np.mean([a[u], b[u]])) for u in set(a) & set(b)}
        elif (m, "du_mad") in dp:
            dp[(m, "mean")] = dp[(m, "du_mad")]
    return cells, db, dp


def interaction_by_size(protocol: str, size: str, inputs=None) -> TwoIntervals:
    _, db, dp = _scale(str(_inp("tab_scale", inputs)))
    return question_and_database_bootstrap(dp[(_SCALE_MODELS[size], SCALE_PROTOCOLS[protocol])], db)[0]


def scale_contrasts(inputs=None) -> Dict[str, dict]:
    _, db, dp = _scale(str(_inp("tab_scale", inputs)))
    con = {}
    for name, a, b in _SCALE_PAIRS:
        da, dbb = dp[(a, "mean")], dp[(b, "mean")]
        est, p = question_and_database_bootstrap({u: da[u] - dbb[u] for u in set(da) & set(dbb)}, db)
        con[name] = dict(contrast=est, p=p)
    order = sorted(_SCALE_HOLM, key=lambda n: con[n]["p"])
    run = 0.0
    for rank, n in enumerate(order):
        run = max(run, min(1.0, (len(_SCALE_HOLM) - rank) * con[n]["p"]))
        con[n]["p_holm"] = round(run, 5)
    return con


def scale_equivalence(inputs=None) -> Dict[str, float]:
    cells, _, _ = _scale(str(_inp("tab_scale", inputs)))
    F = cells[("qwen25-72b-awq", "madm2_objective", "with_peers")]
    N = cells[("qwen25-72b-awq", "madm2_objective", "no_peers")]
    ef, en = _round_change(F), _round_change(N)
    d = np.array([ef[u] - en[u] for u in sorted(set(ef) & set(en))])
    b = d[stats.resample_index(len(d), _SCALE_DRAWS, _SCALE_SEED)].mean(axis=1)
    lo, hi = np.percentile(b, [5, 95])
    out = dict(estimate=_pp2(d.mean()), ci90_lo=_pp2(lo), ci90_hi=_pp2(hi),
               inside_margin=int(100 * lo > -2 and 100 * hi < 2))
    for rep in (1, 2):
        a = _round_change([r for r in F if r["repeat"] == rep])
        c = _round_change([r for r in N if r["repeat"] == rep])
        out[f"repeat{rep}"] = _pp2(np.mean([a[u] - c[u] for u in sorted(set(a) & set(c))]))
    return out


STRENGTH_MODELS = {"deepseek-v4-pro": "DS", "qwen3-32b": "Q3"}
_OPS = {"selector": "S_al", "generation": "Generation", "valid": "Valid", "nonempty": "Nonempty", "oracle": "Oracle"}


@functools.lru_cache(maxsize=4)
def _strength_rows(inp: str, benchmark: str):
    ops = {int(r["question_id"]): r for r in _read("tab_strength", "pool_operations_per_question.csv", inp)
           if r["pool"] == "original" and r["benchmark"] == benchmark}
    ds = {int(r["question_id"]): r for r in _read("tab_strength", "deepseek_per_question.csv", inp)
          if r["benchmark"] == benchmark}
    qs = sorted(ops)
    cl = {q: ops[q]["db_id"] for q in qs}
    x = {}
    for q in qs:
        d, o = ds.get(q, {}), ops[q]
        x[q] = dict(DS_S_al=int(d.get("DS_S_al") or 0), DS_Generation=int(d.get("DS_Generation") or 0),
                    Q3_S_al=int(o["S_al"]), Q3_Generation=int(o["Generation"]), Valid=int(o["Valid"]),
                    Nonempty=int(o["Nonempty"]), Oracle=int(o["Oracle"]))
    return qs, cl, x


def _col(model, op):
    o = _OPS[op]
    return f"{STRENGTH_MODELS[model]}_{o}" if op in ("selector", "generation") else o


def _database_equal(values: dict, clusters: dict, seed: int) -> dict:
    r = stats.cluster_bootstrap(values, clusters, "cluster_equal", stats.DRAWS, seed)
    return dict(pt=_pp(r["point"]), lo=_pp(r["lo"]), hi=_pp(r["hi"]), p=r["p_boot"])


def _holm_rounded(results: List[dict], alpha: float = 0.05) -> None:
    srt = sorted(results, key=lambda r: round(r["p"], 4))
    stop = False
    for j, r in enumerate(srt):
        rej = (not stop) and round(r["p"], 4) <= alpha / (len(srt) - j)
        stop = stop or not rej
        r["holm"] = int(rej)


def _iv(r) -> Interval:
    return Interval(r["pt"], r["lo"], r["hi"])


def stronger_model_accuracy(model: str, operation: str, benchmark: str = "bird", inputs=None) -> Interval:
    qs, cl, x = _strength_rows(str(_inp("tab_strength", inputs)), benchmark)
    return _iv(_database_equal({q: x[q][_col(model, operation)] for q in qs}, cl, 20261105))


def stronger_model_contrasts(model: str, benchmark: str = "bird", inputs=None) -> Dict[str, dict]:
    qs, cl, x = _strength_rows(str(_inp("tab_strength", inputs)), benchmark)
    fam = []
    for a_, z_ in (("selector", "valid"), ("selector", "nonempty"), ("generation", "valid"),
                   ("generation", "nonempty"), ("generation", "selector")):
        A, Z = _col(model, a_), _col(model, z_)
        r = _database_equal({q: x[q][A] - x[q][Z] for q in qs}, cl, 20261105)
        r["key"] = f"{a_}_minus_{z_}"
        fam.append(r)
    _holm_rounded(fam)
    return {r["key"]: dict(contrast=_iv(r), holm=r["holm"]) for r in fam}


def stronger_model_gain_difference(operation: str, baseline: str, benchmark: str = "bird", inputs=None) -> Interval:
    qs, cl, x = _strength_rows(str(_inp("tab_strength", inputs)), benchmark)

    def gain(xx, m):
        return xx[_col(m, operation)] - xx[_col(m, baseline)]
    return _iv(_database_equal({q: gain(x[q], "deepseek-v4-pro") - gain(x[q], "qwen3-32b") for q in qs}, cl, 20261106))


FAMILY_RECIPIENTS = {"qwen3-32b": "qwen3", "gpt-oss-120b": "gptoss", "gemma-4-26b": "gemma"}


_FAMILY_DRAWS, _FAMILY_SEED = 10000, 20261028
_FAMILY_RULES = (("released", "unfiltered"), ("valid_majority", "valid"), ("nonempty_majority", "nonempty"))
_FEEDBACK_DRAWS, _FEEDBACK_SEED = 4000, 20261013


def _database_mean_boot(by_db: Dict[str, list]):
    keys = sorted(by_db)
    per = np.array([float(np.mean(by_db[k])) for k in keys])
    draws = per[np.random.default_rng(_FAMILY_SEED).integers(0, len(per), size=(_FAMILY_DRAWS, len(per)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return float(per.mean()), float(lo), float(hi), float(min(1.0, max(p, 1.0 / _FAMILY_DRAWS)))


def _feedback_boot(by_db: Dict[str, list]):
    keys = list(by_db)
    flat = [d for k in keys for d in by_db[k]]
    point = sum(flat) / len(flat)
    rnd, sims = random.Random(_FEEDBACK_SEED), []
    for _ in range(_FEEDBACK_DRAWS):
        pool = []
        for _ in range(len(keys)):
            pool.extend(by_db[keys[rnd.randrange(len(keys))]])
        sims.append(sum(pool) / len(pool))
    sims.sort()
    return point, sims[int(0.025 * _FEEDBACK_DRAWS)], sims[min(_FEEDBACK_DRAWS - 1, int(0.975 * _FEEDBACK_DRAWS))], len(flat)


@functools.lru_cache(maxsize=4)
def _family_values(inp: str) -> dict:
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in _read("fig_sql-family", "crossfamily_generation_selection_per_question.csv", inp):
        data[(r["benchmark"], r["recipient"], r["selector"])][r["db_id"]].append(float(r["difference"]))
    V: dict = {}
    cells = []
    for b in ("bird", "spider"):
        for rec in FAMILY_RECIPIENTS:
            for sel, lab in _FAMILY_RULES:
                est, lo, hi, p = _database_mean_boot(data[(b, rec, sel)])
                k = f"{b}_{FAMILY_RECIPIENTS[rec]}_gen_minus_{lab}"
                V[k], V[k + "_lo"], V[k + "_hi"], V[k + "_p"] = 100 * est, 100 * lo, 100 * hi, p
                cells.append(k)
    for k, flag in zip(cells, stats.holm([V[k + "_p"] for k in cells])):
        V[k + "_holm_joint"] = int(flag)

    rows = [r for r in _read("fig_sql-family", "bird_qwen25_7b_feedback_per_question.csv", inp)
            if r["all_four_conditions_present"] == "1"]
    rows.sort(key=lambda r: int(r["question_id"]))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["db_id"]].append(float(r["candidate_feedback_correct"]) - float(r["candidate_only_correct"]))
    point, lo, hi, n = _feedback_boot(by)
    k = "bird_qwen25_7b_feedback_minus_queries_only"
    V[k], V[k + "_lo"], V[k + "_hi"] = 100 * point, 100 * lo, 100 * hi
    V["bird_feedback_paired_questions"] = n
    return V


def generation_minus_vote(benchmark: str, recipient: str, rule: str, inputs=None) -> dict:
    V = _family_values(str(_inp("fig_sql-family", inputs)))
    k = f"{benchmark}_{FAMILY_RECIPIENTS[recipient]}_gen_minus_{rule}"
    return dict(contrast=Interval(V[k], V[k + "_lo"], V[k + "_hi"]), p=V[k + "_p"], holm=V[k + "_holm_joint"])


def execution_feedback(inputs=None) -> Tuple[Interval, int]:
    V = _family_values(str(_inp("fig_sql-family", inputs)))
    k = "bird_qwen25_7b_feedback_minus_queries_only"
    return Interval(V[k], V[k + "_lo"], V[k + "_hi"]), int(V["bird_feedback_paired_questions"])
