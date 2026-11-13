"""Dependency checks: is a replayed stage change valid for the deployed protocol?

A replay re-reads logged answers under another extractor, vote, round or tie rule. The replay equals the
deployed outcome when no later step of the protocol reads the changed labels (Du-MAD, MAD-M2). BCCS reads
answer verdicts to stop, to choose a fallback and to select peers, so each BCCS run carries the first
point where its deployed path would diverge; the deployed outcome then lies in an interval [lower,
upper] around the replay.

The checks here are
- replay bounds: per contrast and group of configurations, the replay estimate with its question and
  database intervals, and the deployment bound with its interval (Lemma 1);
- divergence shares: share of BCCS runs whose deployed path would diverge under a reading;
- read checks: BCCS verdicts, stop flags and peer choices recomputed from the logged prefix;
- the online check: BCCS run with the execution vote inside the protocol, compared with the replay;
- the sensitivity census: how often alternative output rules change a conclusion.

This module also holds the question-level arrays of the BIRD debates (`_load_runs`, `_build`,
`_pooled`) that the peer and interaction comparisons share.

Estimates are in percentage points, except the divergence shares (fractions) and the read checks
(counts); intervals are 95% percentile bootstrap intervals. Readings are named 'ref' (protocol
extractor, exact-match vote), 'E' (lenient extractor), 'I' (execution vote), 'EI' (both) and 'ref_tl'
(the reference reading with last-agent ties, keyed 'ref_tie_last' in the divergence records).
"""
from __future__ import annotations

import collections
import csv
import itertools
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from share_eval_suite import paths, stats

FACTORIAL = "tab_factorial"
ONLINE = "tab_s-online"
CENSUS = "tab_sensitivity-census"

DRAWS, SEED = stats.DRAWS, stats.SEED_REPLAY
INTER = ("du_mad", "madm2_objective", "bccs")          # configurations whose debates are replayed
EXTS, IDS, RDS, ARMS = ("strict", "lenient"), ("string", "exec"), ("r0", "last"), ("NOPEER", "FULL")

# a reading's per-run score keys at the last round and at the initial answers, and its divergence record
_READING = {"ref": ("strict|string|last|first", "strict|string|r0|first", "ref"),
            "I": ("strict|exec|last|first", "strict|exec|r0|first", "I"),
            "E": ("lenient|string|last|first", "lenient|string|r0|first", "E"),
            "EI": ("lenient|exec|last|first", "lenient|exec|r0|first", "EI"),
            "ref_tl": ("strict|string|last|last", "strict|string|r0|last", "ref_tie_last")}

# replay contrasts: terms (sign, reading, round, peers condition) summed per question
_TERMS = {
    "E": [(1, "E", "last", "FULL"), (-1, "ref", "last", "FULL")],
    "I": [(1, "I", "last", "FULL"), (-1, "ref", "last", "FULL")],
    "EI": [(1, "EI", "last", "FULL"), (-1, "ref", "last", "FULL")],
    "nonadd_EI": [(1, "EI", "last", "FULL"), (-1, "E", "last", "FULL"), (-1, "I", "last", "FULL"),
                  (1, "ref", "last", "FULL")],
    "nonadd_EIR": [(1, "EI", "r0", "FULL"), (-1, "E", "last", "FULL"), (-1, "I", "last", "FULL"),
                   (-1, "ref", "r0", "FULL"), (1, "ref", "last", "FULL"), (1, "ref", "last", "FULL")],
    "ref_minus_final": [(1, "ref", "last", "FULL"), (-1, "ref", "final", "FULL")],
    "final_minus_EIr0": [(1, "ref", "final", "FULL"), (-1, "EI", "r0", "FULL")],
    "dpeer_EI": [(1, "EI", "last", "FULL"), (-1, "EI", "r0", "FULL"), (-1, "EI", "last", "NOPEER"),
                 (1, "EI", "r0", "NOPEER")],
    "later_EI": [(1, "EI", "last", "FULL"), (-1, "EI", "r0", "FULL")],
    "dpeer_ref": [(1, "ref", "last", "FULL"), (-1, "ref", "r0", "FULL"), (-1, "ref", "last", "NOPEER"),
                  (1, "ref", "r0", "NOPEER")],
    "dpeer_ref_tl": [(1, "ref_tl", "last", "FULL"), (-1, "ref_tl", "r0", "FULL"),
                     (-1, "ref_tl", "last", "NOPEER"), (1, "ref_tl", "r0", "NOPEER")],
}
CONTRASTS = tuple(_TERMS)


def _dir(obj: str, inputs) -> pathlib.Path:
    return pathlib.Path(inputs) if inputs is not None else paths.inputs(obj)


# ------------------------------------------------------------------ question-level arrays (BIRD debates)
def _load_runs(d: pathlib.Path):
    """Per-run scores, and the runs grouped by (protocol, backbone) and question."""
    runs = [json.loads(line) for line in (d / "per_run_scores.jsonl").read_text().splitlines()]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        by[(r["protocol"], r["backbone"])][r["unit"]].append(r)
    return runs, by


def _build(by_unit, tie: str = "first"):
    """Questions with runs in both peer conditions, Y[q, extractor, vote, round, peers] and F[q, peers]
    (the protocol's returned answer); repeats are averaged within a question and condition."""
    units = sorted(u for u, rs in by_unit.items() if {x["arm"] for x in rs} >= set(ARMS))
    Y = np.zeros((len(units), 2, 2, 2, 2))
    F = np.zeros((len(units), 2))
    for ui, u in enumerate(units):
        for a, arm in enumerate(ARMS):
            rs = [x for x in by_unit[u] if x["arm"] == arm]
            for e, i, rd in itertools.product((0, 1), repeat=3):
                Y[ui, e, i, rd, a] = np.mean([x[f"{EXTS[e]}|{IDS[i]}|{RDS[rd]}|{tie}"] for x in rs])
            F[ui, a] = np.mean([x["final"] for x in rs])
    return units, Y, F


def _pooled(by, cells, tie: str = "first"):
    """Questions common to `cells` and the configuration mean of Y and F on them (cells in given order)."""
    built = {c: _build(by[c], tie) for c in cells}
    common = sorted(set.intersection(*[set(built[c][0]) for c in cells]))
    Ys, Fs = [], []
    for c in cells:
        pos = {u: k for k, u in enumerate(built[c][0])}
        sel = [pos[u] for u in common]
        Ys.append(built[c][1][sel])
        Fs.append(built[c][2][sel])
    return common, np.mean(Ys, axis=0), np.mean(Fs, axis=0)


def _est_ci(d: np.ndarray, idx: np.ndarray) -> Tuple[float, float, float]:
    """Mean (pp) of per-question values and its percentile interval under the index matrix `idx`."""
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return 100 * float(d.mean()), 100 * float(lo), 100 * float(hi)


def _db_ci(units: List[str], est: np.ndarray, dbmap: Dict[str, str]) -> Tuple[float, float]:
    """95% percentile interval (pp) of the question mean when the databases are resampled with
    replacement: 10,000 draws of 11 databases, drawn one draw at a time from one generator (seed
    20261106); a draw averages the questions of its drawn databases."""
    dbs = sorted({dbmap[u] for u in units})
    members = {d: np.array([i for i, u in enumerate(units) if dbmap[u] == d]) for d in dbs}
    rng = np.random.default_rng(SEED)
    out = np.empty(DRAWS)
    for b in range(DRAWS):
        pick = rng.integers(0, len(dbs), size=len(dbs))
        out[b] = est[np.concatenate([members[dbs[j]] for j in pick])].mean()
    return 100 * float(np.percentile(out, 2.5)), 100 * float(np.percentile(out, 97.5))


# ------------------------------------------------------------------ replay bounds
_BOUNDS: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {}


def _load_div(d: pathlib.Path) -> Dict[Tuple[str, str, str, int], Dict[str, Any]]:
    div = {}
    for line in (d / "bccs_divergence_runs.jsonl").read_text().splitlines():
        x = json.loads(line)
        div[(x["unit"], x["arm"], x["backbone"], int(x["repeat"]))] = x
    return div


def load_scored_runs(inputs=None):
    """Per-run scores of the BIRD debates and the BCCS first-divergence records.

    Returns (runs, divergence, database of each question)."""
    d = _dir(FACTORIAL, inputs)
    runs, _ = _load_runs(d)
    div = _load_div(d)
    with open(d / "bird_units_db.csv") as fh:
        dbmap = {r["unit"]: r["db_id"] for r in csv.DictReader(fh)}
    return runs, div, dbmap


def _task_arrays(runs, div, terms):
    """Per question: replay value of a contrast and its deployment bound (lower, upper), each term
    averaged over the repeats of its peers condition."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        by[r["unit"]][r["arm"]].append(r)
    units = sorted(u for u, a in by.items() if {"FULL", "NOPEER"} <= set(a))
    est, lo, hi = np.zeros(len(units)), np.zeros(len(units)), np.zeros(len(units))
    for k, u in enumerate(units):
        for sgn, name, rnd, arm in terms:
            vals = [run_interval(r, div, name, rnd) for r in by[u][arm]]
            l_, h_, R_ = (np.mean([v[j] for v in vals]) for j in range(3))
            est[k] += sgn * R_
            if sgn > 0:
                lo[k] += l_
                hi[k] += h_
            else:
                lo[k] -= h_
                hi[k] -= l_
    return units, est, lo, hi


def _bounds(runs, div, dbmap):
    runs = [r for r in runs if r["protocol"] in INTER]
    cells = sorted({(r["protocol"], r["backbone"]) for r in runs})
    res = {}
    idx = None
    for cname, terms in _TERMS.items():
        arrays = {c: _task_arrays([r for r in runs if (r["protocol"], r["backbone"]) == c], div, terms)
                  for c in cells}
        u0 = arrays[cells[0]][0]
        assert all(arrays[c][0] == u0 for c in cells)
        if idx is None:                       # one question index matrix shared by every contrast
            idx = stats.resample_index(len(u0), DRAWS, SEED)
        groups = {f"{c[0]}|{c[1]}": [c] for c in cells}
        groups["mean6"] = cells
        groups["mean4"] = [c for c in cells if c[0] != "bccs"]
        for g, sub in groups.items():
            E = np.mean([arrays[c][1] for c in sub], axis=0)
            L = np.mean([arrays[c][2] for c in sub], axis=0)
            H = np.mean([arrays[c][3] for c in sub], axis=0)
            be, bl, bh = E[idx].mean(axis=1), L[idx].mean(axis=1), H[idx].mean(axis=1)
            rec = dict(est=100 * float(E.mean()),
                       lo=100 * float(np.percentile(be, 2.5)), hi=100 * float(np.percentile(be, 97.5)),
                       bound_lo=100 * float(L.mean()), bound_hi=100 * float(H.mean()),
                       bound_ci_lo=100 * float(np.percentile(bl, 2.5)),
                       bound_ci_hi=100 * float(np.percentile(bh, 97.5)))
            if g in ("mean4", "mean6"):
                rec["db_lo"], rec["db_hi"] = _db_ci(u0, E, dbmap)
            res[(cname, g)] = rec
    return res


def replay_bounds(inputs=None) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Every replay contrast for every group of configurations.

    Keys are (contrast, group); groups are 'protocol|backbone' for one configuration, 'mean4' for the
    four Du-MAD and MAD-M2 configurations and 'mean6' for the six including BCCS. Each value holds
    `est`, `lo`, `hi` (replay estimate and question interval), `bound_lo`, `bound_hi` (deployment
    bound), `bound_ci_lo`, `bound_ci_hi` (its interval) and, for the means, `db_lo`, `db_hi`
    (database interval).

    Per question, each term is averaged over repeats and the terms are summed; configurations are
    averaged per question. The question interval resamples the 198 questions (10,000 draws, seed
    20261106, one index matrix shared by every contrast and by the replay and bound arrays); the
    bound interval takes the 2.5th percentile of the lower and the 97.5th of the upper bound. The
    database interval resamples the 11 databases (10,000 sequential draws, seed 20261106)."""
    key = str(_dir(FACTORIAL, inputs))
    if key not in _BOUNDS:
        runs, div, dbmap = load_scored_runs(inputs)
        _BOUNDS[key] = _bounds(runs, div, dbmap)
    return _BOUNDS[key]


def replay_bound(contrast: str, group: str = "mean4", inputs=None) -> Dict[str, float]:
    """One replay contrast for one group of configurations (see `replay_bounds`)."""
    if contrast not in CONTRASTS:
        raise KeyError(contrast)
    return dict(replay_bounds(inputs)[(contrast, group)])


def run_interval(run: Dict[str, Any], divergence: Dict, reading: str, round_: str) -> Tuple[float, float, float]:
    """(lower, upper, replay) of a run's deployed outcome under a reading at a round.

    `reading` is 'ref' (protocol extractor, exact-match vote), 'E', 'I', 'EI' or 'ref_tl' (last-agent
    ties); `round_` is 'last', 'r0' (initial answers) or 'final' (the protocol's returned answer). The
    initial answers precede every read, and Du-MAD and MAD-M2 never read the labels, so there the
    interval is the replay itself; a BCCS run at the last round takes the bound of its divergence record."""
    if round_ == "final":
        v = run["final"]
        return v, v, v
    R = run[_READING[reading][0 if round_ == "last" else 1]]
    if round_ == "r0" or run["protocol"] != "bccs":
        return R, R, R
    d = divergence[(run["unit"], run["arm"], run["backbone"], int(run["repeat"]))][_READING[reading][2]]
    assert d["R_last"] == R
    return d["O_lo"], d["O_hi"], R


def divergence_shares(inputs=None) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """Per (backbone, peers condition, reading): share of BCCS runs that diverge (`P_D`), that diverge
    unrecoverably (`P_U`), and the exact (Clopper-Pearson) 95% interval of the share of questions with
    any diverging run (`cp_lo`, `cp_hi`), all as fractions in [0, 1]. Readings are 'ref', 'EI' and
    'ref_tie_last' (the 'ref_tl' reading of `run_interval`); peers conditions are 'FULL' (with peers)
    and 'NOPEER' (without)."""
    _, div, _ = load_scored_runs(inputs)
    out = {}
    rows = collections.defaultdict(list)
    for d in div.values():
        rows[(d["backbone"], d["arm"])].append(d)
    for (bb, arm), rs in rows.items():
        for reading in ("ref", "EI", "ref_tie_last"):
            kinds = collections.Counter(x[reading]["D"] for x in rs)
            byu = collections.defaultdict(list)
            for x in rs:
                byu[x["unit"]].append(x[reading]["D"])
            any_d = sum(1 for v in byu.values() if any(k != "none" for k in v))
            n = len(rs)
            lo, hi = stats.clopper_pearson(any_d, len(byu))
            out[(bb, arm, reading)] = dict(P_D=(n - kinds["none"]) / n, P_U=kinds["unrecoverable"] / n,
                                           cp_lo=lo, cp_hi=hi)
    return out


def read_checks(inputs=None) -> Dict[str, int]:
    """BCCS reads recomputed from the logged prefix: checked and reproduced counts of answer verdicts,
    stop flags and peer choices."""
    d = _dir(FACTORIAL, inputs)
    with open(d / "bccs_read_checks.csv") as fh:
        reads = list(csv.DictReader(fh))
    out = {"runs": len(reads)}
    for k in ("verdicts", "stop_flags", "peer_choices"):
        out[f"{k}_checked"] = sum(int(r[f"{k}_checked"]) for r in reads)
        out[f"{k}_reproduced"] = sum(int(r[f"{k}_reproduced"]) for r in reads)
    out["not_reproduced"] = sum(out[f"{k}_checked"] - out[f"{k}_reproduced"]
                                for k in ("verdicts", "stop_flags", "peer_choices"))
    return out


# ------------------------------------------------------------------ online check
_EXCLUDED = {"bird|518", "bird|701"}
# tie rule -> columns of the initial answers and the last round
_ONLINE_TIES = {"first": ("initial_first", "last_first"), "last": ("initial_last_agent", "last_last_agent"),
                "random": ("initial_random", "last_random")}


def _r2(v) -> float:
    return round(100 * float(v), 2)


def _r4(v) -> float:
    return round(100 * float(v), 4)


def _online_rows(inputs):
    d = _dir(ONLINE, inputs)
    with open(d / "online_bccs_runs.csv", newline="") as fh:
        online = [r for r in csv.DictReader(fh) if r["question"] not in _EXCLUDED]
    with open(d / "replay_bccs_qwen7b.csv", newline="") as fh:
        replay = [r for r in csv.DictReader(fh) if r["question"] not in _EXCLUDED]
    return online, replay


def _per_question(rows, f) -> Dict[str, float]:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["question"]].append(f(r))
    return {u: float(np.mean(v)) for u, v in by.items()}


def _online_dpeer(rows, ini: str, last: str) -> Dict[str, float]:
    """Per question: (last round - initial answers) with peers minus without peers, repeats averaged."""
    change = lambda r: float(r[last]) - float(r[ini])          # noqa: E731
    a = _per_question([r for r in rows if r["condition"] == "with_peers"], change)
    b = _per_question([r for r in rows if r["condition"] == "no_peers"], change)
    return {u: a[u] - b[u] for u in set(a) & set(b)}


def _online_stats(d: Dict[str, float], db: Dict[str, str]) -> Dict[str, float]:
    """Mean (pp) with a question interval (10,000 draws, seed 20261106), a database interval (10,000
    draws of the databases, one draw at a time from its own generator with seed 20261106; a draw
    averages the questions of its drawn databases) and the two-sided bootstrap p (floor 1/10,000)."""
    us = sorted(d)
    x = np.array([d[u] for u in us])
    t = x[stats.resample_index(len(x), DRAWS, SEED)].mean(axis=1)
    g = collections.defaultdict(list)
    for i, u in enumerate(us):
        g[db[u]].append(i)
    keys = sorted(g)
    rng = np.random.default_rng(SEED)
    c = np.array([x[[i for k in rng.integers(0, len(keys), size=len(keys)) for i in g[keys[k]]]].mean()
                  for _ in range(DRAWS)])
    p = max(1 / DRAWS, min(1.0, 2 * min((t <= 0).mean(), (t >= 0).mean())))
    return dict(n=len(us), est=_r2(x.mean()), q_lo=_r2(np.percentile(t, 2.5)), q_hi=_r2(np.percentile(t, 97.5)),
                db_lo=_r2(np.percentile(c, 2.5)), db_hi=_r2(np.percentile(c, 97.5)), p=round(float(p), 5))


def _summ(x: np.ndarray) -> Dict[str, float]:
    """Mean (pp) of a per-question difference and its question interval (10,000 draws, seed 20261106)."""
    b = x[stats.resample_index(len(x), DRAWS, SEED)].mean(axis=1)
    return dict(n=int(len(x)), est=_r2(x.mean()), q_lo=_r2(np.percentile(b, 2.5)), q_hi=_r2(np.percentile(b, 97.5)))


def online_peer_effect(tie: str = "first", exclude_single_answer_stops: bool = False,
                       inputs=None) -> Dict[str, float]:
    """With-minus-without-peers change (last round minus initial answers) of online BCCS with the
    execution vote inside the protocol (Qwen2.5-7B): `est`, question and database intervals, `p`, `n`.

    `tie` is 'first', 'last' or 'random'; single-answer stops can be excluded (ties to the first agent).
    Questions bird|518 and bird|701 are excluded."""
    online, _ = _online_rows(inputs)
    db = {r["question"]: r["database"] for r in online}
    if exclude_single_answer_stops:
        if tie != "first":
            raise ValueError("single-answer stops are excluded with ties to the first agent")
        online = [r for r in online if r["single_answer_stop"] == "0"]
    return _online_stats(_online_dpeer(online, *_ONLINE_TIES[tie]), db)


def online_minus_replay(condition: str = "with_peers", inputs=None) -> Dict[str, Any]:
    """Online minus replayed last-round accuracy per question (ties to the first agent), overall and by
    the logged divergence class of the question, with the deployment bounds and run disagreement.

    `condition` is 'with_peers' or 'no_peers'. A question's class is 'none' when no logged run diverges,
    'unrecoverable' when any does unrecoverably, else 'recoverable_only'."""
    online, replay = _online_rows(inputs)
    on = _per_question([r for r in online if r["condition"] == condition], lambda r: float(r["last_first"]))
    lg = collections.defaultdict(list)
    for r in replay:
        if r["condition"] == condition:
            lg[r["question"]].append(r)
    strata = collections.defaultdict(list)
    diff, lo, hi, onm = [], [], [], []
    for u in sorted(u for u in on if u in lg):
        kinds = [r["divergence"] for r in lg[u]]
        st = ("none" if all(k == "none" for k in kinds) else
              "unrecoverable" if any(k == "unrecoverable" for k in kinds) else "recoverable_only")
        rp = np.mean([float(r["replay_last_correct"]) for r in lg[u]])
        lo.append(np.mean([float(r["bound_low"]) for r in lg[u]]))
        hi.append(np.mean([float(r["bound_high"]) for r in lg[u]]))
        diff.append(on[u] - rp)
        onm.append(on[u])
        strata[st].append(on[u] - rp)
    lg_runs = [r for r in replay if r["condition"] == condition]
    on_run = {(r["question"], r["repeat"]): float(r["last_first"]) for r in online if r["condition"] == condition}
    lg_run = {(r["question"], r["repeat"]): float(r["replay_last_correct"]) for r in lg_runs}
    keys = sorted(set(on_run) & set(lg_run))
    return dict(overall=_summ(np.array(diff)),
                by_divergence={st: _summ(np.array(x)) for st, x in strata.items()},
                online_accuracy=_r2(np.mean(onm)),
                bound_low=_r2(np.mean(lo)), bound_high=_r2(np.mean(hi)),
                online_inside_bounds=bool(np.mean(lo) <= np.mean(onm) <= np.mean(hi)),
                diverging_runs_pct=_r4(np.mean([r["divergence"] != "none" for r in lg_runs])),
                run_disagreement_pct=_r4(np.mean([on_run[k] != lg_run[k] for k in keys])))


# ------------------------------------------------------------------ sensitivity census
# (family, tie rule, tag) of the case families
_FAMILIES = [("S1", None, "bird_sharing"), ("S2", None, "bird_dpeer"), ("S3", None, "bird_final_vs_vote"),
             ("S4", None, "em_qa"), ("S5", "first", "return_first"), ("S5", "last", "return_last"),
             ("S5", "keep", "return_kept")]


def _excludes_zero(ci: Optional[List[float]]) -> Optional[bool]:
    return None if ci is None else (ci[0] > 0 or ci[1] < 0)


def _classify(r) -> Tuple[bool, bool, bool]:
    """(sign flip, detection flip, stable) of a case: the sign flips when both estimates are nonzero
    with opposite signs; detection flips when exactly one of the two intervals excludes zero."""
    n, c = r["naive"], r["controlled"]
    sign = n != 0 and c != 0 and (n > 0) != (c > 0)
    sn, sc = _excludes_zero(r["naive_ci"]), _excludes_zero(r["controlled_ci"])
    detect = sn is not None and sc is not None and sn != sc
    return sign, detect, (not sign) and not detect


def _sgn(x: float, tol: float = 1e-9) -> int:
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)


def sensitivity_census(inputs=None) -> Dict[str, Any]:
    """How often an alternative output rule changes a conclusion.

    Per case family: `cases`, `sign` (sign flips), `detect` (exactly one interval excludes zero),
    `stable`; also the negative-control cases, the grouping combinations (pair F1 versus cluster F1;
    a reversal is a strictly positive change on one and a strictly negative change on the other) and
    the measurement totals (`changed` of `eligible`). Cases with an undefined baseline are not eligible."""
    d = _dir(CENSUS, inputs)
    rows = json.loads((d / "census_cases.json").read_text())
    fams, changed, eligible = {}, 0, 0
    for fam, tie, tag in _FAMILIES:
        rs = [r for r in rows if r["family"] == fam and (tie is None or r.get("tie") == tie) and not r["naive_undefined"]]
        cls = [_classify(r) for r in rs]
        fams[tag] = dict(cases=len(rs), sign=sum(c[0] for c in cls), detect=sum(c[1] for c in cls),
                         stable=sum(c[2] for c in cls))
        if fam in ("S1", "S2", "S3", "S4"):
            eligible += len(rs)
            changed += sum(not c[2] for c in cls)
    ctrl = [_classify(r) for r in rows if r["family"] == "S4" and r.get("negative_control")]
    with open(d / "grouping_sharing_effects.csv") as fh:
        g = list(csv.DictReader(fh))
    pairs = [(_sgn(float(r["d_pair_pp"])), _sgn(float(r["d_cluster_pp"]))) for r in g]
    return dict(families=fams,
                control=dict(cases=len(ctrl), sign=sum(c[0] for c in ctrl), detect=sum(c[1] for c in ctrl),
                             stable=sum(c[2] for c in ctrl)),
                grouping=dict(cases=len(pairs), reversals=sum(1 for p, c in pairs if p * c == -1),
                              stable=sum(1 for p, c in pairs if p * c == 1)),
                measurement_changed=changed, measurement_eligible=eligible)
