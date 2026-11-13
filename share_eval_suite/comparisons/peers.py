"""Peer inputs: debate with peers versus without peers under a fixed extractor, vote, round and tie rule.

The peer effect of a configuration is the change from the initial answers to the last round with peers
minus the same change without peers, per question (repeats averaged within a question and condition,
configurations averaged per question). Extractor: 'protocol' (the protocol's own extractor) or
'lenient'; vote: 'exact' (exact-match vote over query strings) or 'execution' (queries grouped by
execution result, failing queries cast no vote); tie: 'first', 'last' or 'random' agent. Results
keyed by reading name the protocol extractor 'strict', the exact-match vote 'string' and the
execution vote 'exec'.

Configurations on BIRD: Du-MAD and MAD-M2 with Llama-3-8B and Qwen2.5-7B agents ('both' = the four,
'du_mad', 'madm2'; one configuration is 'cell_protocol|backbone', or 'protocol|backbone' in
`answer_only_contrast`, with MAD-M2 named 'madm2_objective'). The replay groups name the same four
'mean4' and the six with BCCS 'mean6'. BCCS also enters the per-configuration figure and the
per-configuration levels (with MoA). Populations: the 198 sampled questions and the 1,531 scored
development questions (198 sampled + 1,333 remaining).

Controls: peers' extracted answers only (answer-only peers), peers from other questions of the same
database (placebo), Du-MAD at temperature 0.7, and a unified SQL answer format.

Estimates are in percentage points (levels in %); intervals are 95% percentile bootstrap intervals over
questions (`q_lo`, `q_hi`) and over the 11 BIRD databases (`db_lo`, `db_hi`).
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
from typing import Any, Dict, Tuple

import numpy as np

from share_eval_suite import stats
from share_eval_suite.comparisons import dependency

FACTORIAL = "tab_factorial"
READOUT = "fig_readout-cells"
PEER_CONTROLS = "tab_peer-controls"
ANSWER_CONTENT = "tab_answer-content"
CONTROLS = "tab_s-controls"
MATRIX = "tab_matrix"

EXTRACTORS = {"protocol": "strict", "strict": "strict", "lenient": "lenient"}
VOTES = {"exact": "string", "string": "string", "execution": "exec", "exec": "exec"}
CONFIGURATIONS = {"both": "4cell", "du_mad": "DuMAD_2cell", "madm2": "MADM2_2cell"}
QUESTIONS = {"all": ("new", "198"), "remaining": ("new",), "sampled": ("198",)}
# replay contrast of a reading (see `dependency.CONTRASTS`)
REPLAY = {("strict", "string", "first"): "dpeer_ref", ("lenient", "exec", "first"): "dpeer_EI",
          ("strict", "string", "last"): "dpeer_ref_tl"}

DRAWS, SEED = stats.DRAWS, stats.SEED_REPLAY
SEED_CONTENT = 20261109                   # answer-only peers: question weights, then database weights
TIES = ("first", "last", "random")
DU = [("du_mad", "llama3-8b"), ("du_mad", "qwen2.5-7b")]
M2 = [("madm2_objective", "llama3-8b"), ("madm2_objective", "qwen2.5-7b")]
C4 = DU + M2
GROUPS = {"4cell": C4, "DuMAD_2cell": DU, "MADM2_2cell": M2, **{f"cell_{c[0]}|{c[1]}": [c] for c in C4}}
# configurations of the per-configuration figure, in plotted order, and its readings (extractor, vote)
FIGURE_CELLS = ["du_mad|qwen2.5-7b", "du_mad|llama3-8b", "madm2_objective|qwen2.5-7b",
                "madm2_objective|llama3-8b", "bccs|qwen2.5-7b", "bccs|llama3-8b", "mean"]
FIGURE_READINGS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _reading(extractor: str, vote: str) -> Tuple[str, str]:
    return EXTRACTORS[extractor], VOTES[vote]


def _flat(b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(est=b["est"], q_lo=b["q_ci"][0], q_hi=b["q_ci"][1], db_lo=b["db_ci"][0], db_hi=b["db_ci"][1])
    for k in ("n_questions", "detected", "detected_both"):
        if k in b:
            out[k] = b[k]
    return out


def _cells(group: str):
    return GROUPS[CONFIGURATIONS.get(group, group)]


def _r2(x) -> float:
    return round(100 * float(x), 2)


def _excl0(ci) -> bool:
    return bool(ci[0] > 0 or ci[1] < 0)


# ------------------------------------------------------------------ estimators on the BIRD debate runs
class _Sample198:
    """Per-run scores of the four Du-MAD and MAD-M2 configurations on the 198 sampled questions (the
    questions with two runs in each peer condition in every configuration).

    Question interval: 10,000 draws of the questions (seed 20261106). Database interval: 10,000 draws
    of the 11 databases, drawn one draw at a time from one generator (seed 20261106); a draw is the sum
    of the values of its drawn databases over the sum of their question counts."""

    def __init__(self, d: pathlib.Path):
        self.by = collections.defaultdict(lambda: collections.defaultdict(list))
        self.db = {}
        with open(d / "runs_198.csv") as f:
            for r in csv.DictReader(f):
                self.by[(r["protocol"], r["backbone"])][r["unit"]].append(r)
                self.db[r["unit"]] = r["db"]
        self.units = sorted(set.intersection(*[{u for u, rs in self.by[c].items() if len(rs) == 4} for c in C4]))
        n = len(self.units)
        self.qidx = stats.resample_index(n, DRAWS, SEED)
        dbs = sorted({self.db[u] for u in self.units})
        self.members = [np.array([i for i, u in enumerate(self.units) if self.db[u] == d]) for d in dbs]
        rng = np.random.default_rng(SEED)
        self.pick = np.array([rng.integers(0, len(dbs), size=len(dbs)) for _ in range(DRAWS)])

    def ci(self, d):
        bq = d[self.qidx].mean(axis=1)
        gs = np.array([d[m].sum() for m in self.members])
        gc = np.array([len(m) for m in self.members])
        bd = gs[self.pick].sum(1) / gc[self.pick].sum(1)
        q = [_r2(np.percentile(bq, 2.5)), _r2(np.percentile(bq, 97.5))]
        b = [_r2(np.percentile(bd, 2.5)), _r2(np.percentile(bd, 97.5))]
        return dict(est=_r2(d.mean()), q_ci=q, db_ci=b, detected=_excl0(q) and _excl0(b))

    def arr(self, c, key, cond):
        return np.array([np.mean([float(x[key]) for x in self.by[c][u] if x["condition"] == cond])
                         for u in self.units])

    def dpeer(self, cells, reading, tie):
        e, v = reading.split("|")
        kl, k0 = f"{e}|{v}|last|{tie}", f"{e}|{v}|r0|{tie}"
        wf = np.mean([self.arr(c, kl, "FULL") - self.arr(c, k0, "FULL") for c in cells], axis=0)
        wn = np.mean([self.arr(c, kl, "NOPEER") - self.arr(c, k0, "NOPEER") for c in cells], axis=0)
        return self.ci(wf - wn)


class _Runs:
    """Runs indexed by (source, protocol, backbone, condition) -> question.

    The with-peers runs carry the condition label FULL. The without-peers runs are labelled NOPEERC for
    Du-MAD (later-round prompts contain no peer text) and NOPEER for MAD-M2. One collection (source
    label "orig") holds the 198 sampled questions; the other ("rx") holds the 1,333 remaining
    questions, the controls and the Du-MAD runs without peers. Subclasses load `rows`, set `U198`,
    `NEWU`, `DB` and `METRIC` and call `_index`."""

    detected_key = "detected"

    def _index(self, cond_key: str):
        self.IDX = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in self.rows:
            self.IDX[(r["source"], r["protocol"], r["backbone"], r[cond_key])][r["unit"]].append(r)

    def per_unit(self, src, cell, cond, metric, units):
        """Per question value of one condition, repeats averaged."""
        d = self.IDX[(src, cell[0], cell[1], cond)]
        f = self.METRIC[metric]
        return {u: float(np.mean([f(r) for r in d[u]])) for u in units if d.get(u)}

    def boot(self, items):
        """Mean of (question, value) items with question and database intervals.

        Items of a question (configurations) are pooled as sums over counts. Question interval: 10,000
        draws of the questions; database interval: 10,000 draws of the databases; each from its own
        generator with seed 20261106."""
        by = collections.defaultdict(list)
        for u, v in items:
            by[u].append(v)
        us = sorted(by)
        sums = np.array([sum(by[u]) for u in us])
        cnts = np.array([len(by[u]) for u in us])
        est = sums.sum() / cnts.sum()
        idx = stats.resample_index(len(us), DRAWS, SEED)
        tq = sums[idx].sum(1) / cnts[idx].sum(1)
        g = collections.defaultdict(list)
        for i, u in enumerate(us):
            g[self.DB[u]].append(i)
        keys = sorted(g)
        gs = np.array([sums[g[k]].sum() for k in keys])
        gc = np.array([cnts[g[k]].sum() for k in keys])
        pick = stats.resample_index(len(keys), DRAWS, SEED)
        td = gs[pick].sum(1) / gc[pick].sum(1)
        out = dict(est=_r2(est), q_ci=[_r2(np.percentile(tq, 2.5)), _r2(np.percentile(tq, 97.5))],
                   db_ci=[_r2(np.percentile(td, 2.5)), _r2(np.percentile(td, 97.5))], n_questions=len(us))
        out[self.detected_key] = _excl0(out["q_ci"]) and _excl0(out["db_ci"])
        return out

    @staticmethod
    def nop_new(c):
        return ("rx", "NOPEERC") if c[0] == "du_mad" else ("rx", "NOPEER")

    @staticmethod
    def nop_198(c):
        return ("rx", "NOPEERC") if c[0] == "du_mad" else ("orig", "NOPEER")

    def diff_items(self, spec_a, spec_b, cells, metric, units):
        """(question, condition a - condition b) per configuration, on questions present in every one."""
        per = {}
        for c in cells:
            sa = spec_a(c) if callable(spec_a) else spec_a
            sb = spec_b(c) if callable(spec_b) else spec_b
            a = self.per_unit(sa[0], c, sa[1], metric, units)
            b = self.per_unit(sb[0], c, sb[1], metric, units)
            per[c] = {u: a[u] - b[u] for u in set(a) & set(b)}
        common = set.intersection(*[set(v) for v in per.values()])
        return [(u, v) for c in cells for u, v in per[c].items() if u in common]

    def did_items(self, full_a, nop_a, full_b, nop_b, cells, units):
        """(change full_a - change nop_a) - (change full_b - change nop_b) per question and configuration."""
        per = {}
        for c in cells:
            def g(s, c=c):
                sp = s(c) if callable(s) else s
                return self.per_unit(sp[0], c, sp[1], "E1", units)
            fa, na, fb, nb = g(full_a), g(nop_a), g(full_b), g(nop_b)
            per[c] = {u: (fa[u] - na[u]) - (fb[u] - nb[u]) for u in set(fa) & set(na) & set(fb) & set(nb)}
        common = set.intersection(*[set(v) for v in per.values()])
        return [(u, per[c][u]) for c in cells for u in common]

    def fulldev_items(self, cells, metric, sets=("new", "198")):
        """With peers minus without peers, per configuration, on the remaining ('new') and the sampled
        ('198') questions."""
        items = []
        if "new" in sets:
            items += self.diff_items(("rx", "FULL"), self.nop_new, cells, metric, self.NEWU)
        if "198" in sets:
            items += self.diff_items(("orig", "FULL"), self.nop_198, cells, metric, self.U198)
        return items

    def fulldev_level(self, cells, side, metric, keep_metric):
        """Accuracy of one condition on the questions of the paired 1,531-question contrast."""
        keep = {u for u, _ in self.fulldev_items(cells, keep_metric)}
        items = []
        for c in cells:
            if side == "with":
                specs = [(("rx", "FULL"), self.NEWU), (("orig", "FULL"), self.U198)]
            else:
                specs = [(self.nop_new(c), self.NEWU), (self.nop_198(c), self.U198)]
            for (src, cond), us in specs:
                items += list(self.per_unit(src, c, cond, metric, [u for u in us if u in keep]).items())
        return self.boot(items)


class _Rows1531(_Runs):
    """The 1,531 development questions: per-run correctness of the execution vote under both
    extractors, both rounds and the three tie rules."""

    def __init__(self, d: pathlib.Path):
        qs = json.loads((d / "question_sets.json").read_text())
        self.U198, self.NEWU = qs["sampled"], qs["remaining"]
        self.rows = []
        with open(d / "runs_1531.csv") as f:
            for r in csv.DictReader(f):
                for k in list(r):
                    if k.count("|") == 2:
                        r[k] = float(r[k])
                self.rows.append(r)
        self.DB = {r["unit"]: r["db"] for r in self.rows}
        self._index("condition")
        self.METRIC = {}
        for e in ("strict", "lenient"):
            for t in TIES:
                self.METRIC[f"E1_{e}_{t}"] = (lambda r, e=e, t=t: r[f"{e}|last|{t}"] - r[f"{e}|r0|{t}"])
                self.METRIC[f"last_{e}_{t}"] = (lambda r, e=e, t=t: r[f"{e}|last|{t}"])
                self.METRIC[f"r0_{e}_{t}"] = (lambda r, e=e, t=t: r[f"{e}|r0|{t}"])


class _ControlRows(_Runs):
    """Runs of the controls (lenient extractor, execution vote, first-agent ties): with peers, without
    peers, placebo peers from other questions of the same database (label PLACEBO), Du-MAD at
    temperature 0.7 (labels T07_*) and a unified SQL answer format (labels FMT_*). Questions whose
    gold query fails are dropped. Both intervals excluding zero is reported as `detected_both`."""

    detected_key = "detected_both"

    def __init__(self, d: pathlib.Path):
        self.rows = [json.loads(line) for line in (d / "rows.jsonl").open()]
        ql = json.loads((d / "question_lists.json").read_text())
        qlc = json.loads((d / "question_lists_e4c.json").read_text())
        gold_failed = set(json.loads((d / "eval_meta.json").read_text())["gold_failed_units"])
        blocks = {k: [u for u in v if u not in gold_failed] for k, v in
                  {"u198": ql["units198"], "E4a": ql["E4a"], "E4b": ql["E4b"], "E4c": qlc["E4c"]}.items()}
        self.U198 = blocks["u198"]
        self.NEWU = blocks["E4a"] + blocks["E4b"] + blocks["E4c"]
        self.DB = {r["unit"]: r["db"] for r in self.rows}
        self._index("arm")
        self.METRIC = {"E1": lambda r: float(r["last"]) - float(r["r0"])}


# ------------------------------------------------------------------ 198 sampled questions
def peer_effect(extractor: str = "protocol", vote: str = "exact", tie: str = "first",
                configurations: str = "both", inputs=None) -> Dict[str, Any]:
    """Peer effect on the 198 sampled questions under one extractor, vote and tie rule.

    Returns `est`, `q_lo`, `q_hi`, `db_lo`, `db_hi` (pp, two decimals) and `detected` (both intervals
    exclude zero)."""
    e, v = _reading(extractor, vote)
    S = _Sample198(dependency._dir(PEER_CONTROLS, inputs))
    return _flat(S.dpeer(_cells(configurations), f"{e}|{v}", tie))


def peer_effect_replay(extractor: str = "protocol", vote: str = "exact", tie: str = "first",
                       group: str = "mean4", inputs=None) -> Dict[str, float]:
    """Peer effect on the 198 sampled questions as a replay contrast with its deployment bound.

    Readings: protocol extractor with the exact-match vote (first- or last-agent ties) and the lenient
    extractor with the execution vote (first-agent ties). `group` is 'mean4' (Du-MAD and MAD-M2),
    'mean6' (with BCCS) or 'protocol|backbone'. See `dependency.replay_bounds` for the fields."""
    e, v = _reading(extractor, vote)
    return dependency.replay_bound(REPLAY[(e, v, tie)], group, inputs)


def peer_effect_by_configuration(tie: str = "first", inputs=None) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Peer effect per BIRD configuration (Du-MAD, MAD-M2, BCCS x Llama-3-8B, Qwen2.5-7B) and their
    equal-weight mean ('mean'), under each extractor and vote.

    Returns {configuration: {'extractor|vote': (est, lo, hi)}} with the question interval, extractor in
    ('strict', 'lenient') and vote in ('string', 'exec')."""
    _, runs = dependency._load_runs(dependency._dir(READOUT, inputs))
    by = {f"{p}|{b}": units for (p, b), units in runs.items()}
    inter = sorted(c for c in by if c.split("|")[0] in dependency.INTER)
    built = {c: dependency._build(by[c], tie)[:2] for c in inter}
    common = sorted(set.intersection(*[set(built[c][0]) for c in inter]))
    Ys = []
    for c in inter:
        pos = {u: k for k, u in enumerate(built[c][0])}
        Ys.append(built[c][1][[pos[u] for u in common]])
    built["mean"] = (common, np.mean(Ys, axis=0))
    out = {}
    for c in FIGURE_CELLS:
        units, Y = built[c]
        idx = stats.resample_index(len(units), DRAWS, SEED)       # shared by the four readings
        res = {}
        for e, i in FIGURE_READINGS:
            dd = (Y[:, e, i, 1, 1] - Y[:, e, i, 0, 1]) - (Y[:, e, i, 1, 0] - Y[:, e, i, 0, 0])
            b = dd[idx].mean(axis=1)
            lo, hi = 100 * np.percentile(b, [2.5, 97.5])
            res[f"{dependency.EXTS[e]}|{dependency.IDS[i]}"] = (100 * float(dd.mean()), float(lo), float(hi))
        out[c] = res
    return out


def levels_by_configuration(tie: str = "first", inputs=None) -> Dict[str, Dict[str, float]]:
    """Accuracy (%) with peers per BIRD configuration (and MoA) on the 198 sampled questions.

    Keys per configuration: 'protocol_exact_last', 'lenient_exact_last', 'lenient_execution_initial',
    'lenient_execution_last' and 'final' (the protocol's returned answer)."""
    _, by = dependency._load_runs(dependency._dir(FACTORIAL, inputs))
    cells = sorted(c for c in by if c[0] in dependency.INTER) + [("moa", "roster")]
    out = {}
    for c in cells:
        _, Y, F = dependency._build(by[c], tie)
        out[f"{c[0]}|{c[1]}"] = dict(protocol_exact_last=100 * Y[:, 0, 0, 1, 1].mean(),
                                     lenient_exact_last=100 * Y[:, 1, 0, 1, 1].mean(),
                                     lenient_execution_initial=100 * Y[:, 1, 1, 0, 1].mean(),
                                     lenient_execution_last=100 * Y[:, 1, 1, 1, 1].mean(),
                                     final=100 * F[:, 1].mean())
    return {c: {k: float(v) for k, v in d.items()} for c, d in out.items()}


# ------------------------------------------------------------------ 1,531 development questions
def _rows_1531(inputs) -> _Rows1531:
    return _Rows1531(dependency._dir(PEER_CONTROLS, inputs))


def peer_effect_full_dev(configurations: str = "both", tie: str = "first", questions: str = "all",
                         inputs=None) -> Dict[str, Any]:
    """Peer effect on the BIRD development questions, lenient extractor and execution vote.

    `configurations`: 'both', 'du_mad', 'madm2' or 'cell_protocol|backbone'; `questions`: 'all' (1,531),
    'remaining' (1,333) or 'sampled' (198)."""
    R = _rows_1531(inputs)
    return _flat(R.boot(R.fulldev_items(_cells(configurations), f"E1_lenient_{tie}", sets=QUESTIONS[questions])))


def difference_full_dev(configurations: str = "both", round_: str = "last", inputs=None) -> Dict[str, Any]:
    """With minus without peers accuracy on the 1,531 questions at one round ('last' or 'initial'),
    lenient extractor, execution vote, first-agent ties."""
    R = _rows_1531(inputs)
    met = {"last": "last_lenient_first", "initial": "r0_lenient_first"}[round_]
    return _flat(R.boot(R.fulldev_items(_cells(configurations), met)))


def levels_full_dev(configurations: str = "both", inputs=None) -> Dict[str, float]:
    """Accuracy (%) on the paired 1,531-question contrast: 'initial_with', 'initial_without',
    'last_with', 'last_without' (lenient extractor, execution vote, first-agent ties)."""
    R = _rows_1531(inputs)
    cells = _cells(configurations)
    return {f"{tag}_{side}": R.fulldev_level(cells, side, f"{rd}_lenient_first", "E1_lenient_first")["est"]
            for tag, rd in (("initial", "r0"), ("last", "last")) for side in ("with", "without")}


# ------------------------------------------------------------------ peer content: answer-only peers
def _content_runs(inputs):
    """Runs of the matched collection: correctness of the initial answers, last round and final answer."""
    rows = []
    with open(dependency._dir(ANSWER_CONTENT, inputs) / "runs.csv") as f:
        for r in csv.DictReader(f):
            for k in ("r0", "last", "final"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def answer_only_contrast(first: str = "ANSWER_ONLY", second: str = "FULL", metric: str = "change",
                         configurations: str = "both", inputs=None) -> Dict[str, Any]:
    """Paired contrast of two peer conditions on the 198 sampled questions (matched collection).

    Conditions: 'FULL' (full peer responses), 'ANSWER_ONLY' (each peer's extracted SQL only), 'NOPEER'.
    `metric`: 'change' (last round minus initial answers), 'initial', 'last' or 'final'.
    `configurations`: 'both', 'du_mad', 'madm2' or 'protocol|backbone'. Intervals: 10,000 draws, seed
    20261109 (question weights first, then database weights)."""
    rows = _content_runs(inputs)
    met = {"change": "E1", "initial": "r0", "last": "last", "final": "final"}[metric]
    # one generator: multinomial question weights, then multinomial database weights
    units = sorted({r["unit"] for r in rows})
    db_of = {r["unit"]: r["db"] for r in rows}
    dbs = sorted({db_of[u] for u in units})
    uix = {u: i for i, u in enumerate(units)}
    rng = np.random.default_rng(SEED_CONTENT)
    WQ = rng.multinomial(len(units), [1 / len(units)] * len(units), size=DRAWS).astype(float)
    cnt_db = rng.multinomial(len(dbs), [1 / len(dbs)] * len(dbs), size=DRAWS).astype(float)
    WD = cnt_db[:, np.array([dbs.index(db_of[u]) for u in units])]
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        per[(r["protocol"], r["backbone"], r["condition"])][r["unit"]].append(r)

    def vec(cell, cond):
        x = np.full(len(units), np.nan)
        for u, rs in per[(cell[0], cell[1], cond)].items():
            x[uix[u]] = np.mean([r["last"] - r["r0"] for r in rs]) if met == "E1" else np.mean([r[met] for r in rs])
        return x

    scopes = {"both": C4, "du_mad": DU, "madm2": M2, **{f"{c[0]}|{c[1]}": [c] for c in C4}}
    D = np.vstack([vec(c, first) - vec(c, second) for c in scopes[configurations]])
    mask = np.isfinite(D).all(0)
    x = np.where(mask, np.nan_to_num(D.mean(0)), 0.0)
    w = mask.astype(float)
    out = dict(est=round(100 * float(x.sum() / w.sum()), 2), n_questions=int(w.sum()))
    for tag, W in (("q", WQ), ("db", WD)):
        lo, hi = np.percentile((W @ x) / np.maximum(W @ w, 1e-12), [2.5, 97.5])
        out[f"{tag}_lo"], out[f"{tag}_hi"] = round(100 * float(lo), 2), round(100 * float(hi), 2)
    out["detected"] = all(not (out[f"{t}_lo"] <= 0 <= out[f"{t}_hi"]) for t in ("q", "db"))
    return out


def agent_loss_recovery(protocol: str, backbone: str, condition: str = "FULL", inputs=None) -> Dict[str, float]:
    """Per-agent conditional loss (% of initially correct answers wrong in the last round) and recovery
    (% of initially wrong answers correct) in one configuration and peer condition, pooled over repeats
    and agents, on the matched collection of the 198 sampled questions."""
    rows = _content_runs(inputs)
    cnt = collections.Counter()
    for r in rows:
        if (r["protocol"], r["backbone"], r["condition"]) != (protocol, backbone, condition):
            continue
        for x, y in zip(r["agents_r0"], r["agents_last"]):
            cnt[("retained" if y == "1" else "lost") if x == "1" else ("recovered" if y == "1" else "stayed_wrong")] += 1
    return dict(loss=round(100 * cnt["lost"] / (cnt["retained"] + cnt["lost"]), 4),
                recovery=round(100 * cnt["recovered"] / (cnt["recovered"] + cnt["stayed_wrong"]), 4))


# ------------------------------------------------------------------ no-peer and other controls
def peer_controls(inputs=None) -> Dict[str, Dict[str, Any]]:
    """Controls of the peer effect (lenient extractor, execution vote, first-agent ties).

    'remaining_1333': peer effect on the 1,333 remaining questions (four configurations);
    'placebo_real_minus_placebo', 'placebo_minus_no_peers': change from initial answers to the last
    round, real peers versus peers from other questions of the same database, and placebo versus no
    peers (198 questions, four configurations); 'temperature_0.7': Du-MAD peer effect at temperature
    0.7 (198 questions); 'format_change': change of the peer effect under a unified SQL answer format. Each value has the
    fields of `peer_effect`, with `detected_both` in place of `detected`, and `n_questions`."""
    R = _ControlRows(dependency._dir(CONTROLS, inputs))
    U = R.U198
    out = {
        "remaining_1333": R.boot(R.fulldev_items(C4, "E1", sets=("new",))),
        "placebo_real_minus_placebo": R.boot(R.diff_items(("orig", "FULL"), ("rx", "PLACEBO"), C4, "E1", U)),
        "placebo_minus_no_peers": R.boot(R.diff_items(("rx", "PLACEBO"), R.nop_198, C4, "E1", U)),
        "temperature_0.7": R.boot(R.diff_items(("rx", "T07_FULL"), ("rx", "T07_NOPEERC"), DU, "E1", U)),
        "format_change": R.boot(R.did_items(
            ("rx", "FMT_FULL"), lambda c: ("rx", "FMT_NOPEERC") if c[0] == "du_mad" else ("rx", "FMT_NOPEER"),
            ("orig", "FULL"), lambda c: ("rx", "NOPEERC") if c[0] == "du_mad" else ("orig", "NOPEER"), C4, U)),
    }
    return {k: _flat(b) for k, b in out.items()}


def bird_populations(inputs=None) -> Dict[str, int]:
    """Question counts of the BIRD debates: sampled and scored questions, their databases and
    difficulty mix, and the scored development questions."""
    d = dependency._dir(MATRIX, inputs)
    # sampled questions: those with runs in both peer conditions, the same in all four configurations
    by = collections.defaultdict(lambda: collections.defaultdict(set))
    for line in (d / "per_run_scores.jsonl").open():
        r = json.loads(line)
        if (r["protocol"], r["backbone"]) in C4:
            by[(r["protocol"], r["backbone"])][r["unit"]].add(r["arm"])
    per_cell = [sorted(u for u, a in by[c].items() if {"FULL", "NOPEER"} <= a) for c in C4]
    assert all(x == per_cell[0] for x in per_cell)
    units = per_cell[0]
    with open(d / "bird_dev_questions.csv") as fh:
        dev = list(csv.DictReader(fh))
    diff_of = {r["unit"]: r["difficulty"] for r in dev}
    lists = json.loads((d / "question_lists.json").read_text())
    R = _ControlRows(d)
    out = dict(n_questions_sample=len(units), n_sample_drawn=len(units) + len(lists["excluded_from_200"]),
               n_sample_databases=len({r["db_id"] for r in dev if r["unit"] in set(units)}),
               n_bird_dev=len(dev), n_questions_fulldev=len({u for u, _ in R.fulldev_items(C4, "E1")}))
    for lev in ("simple", "moderate", "challenging"):
        out[f"sample_{lev}"] = sum(diff_of[u] == lev for u in units)
    return out
