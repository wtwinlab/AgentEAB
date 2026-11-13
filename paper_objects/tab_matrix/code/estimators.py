r"""Estimators for the BIRD debate controls (Du-MAD and MAD-M2, Llama-3-8B and Qwen2.5-7B agents).

Shared by the paper objects tab_matrix, tab_s-controls and tab_s-reader-sensitivity; each object
ships its own copy next to its compute.py. Nothing here reads outside the object's inputs/ directory.

Reading everywhere: lenient extractor, execution vote (queries that fail to execute cast no vote), ties to the
first agent. Per-run correctness of that vote is stored in inputs/rows.jsonl (`r0` = initial answers,
`last` = last round; `r0_strict` / `last_strict` = the protocol's own extractor with the same vote).

Two bootstrap implementations are used, each exactly as in the analysis that produced the printed number:

* `boot` (1,531 questions and the 198-question controls): items are (question, value) pairs; repeats are
  averaged within a question first; a multi-cell mean keeps only questions present in every cell. The
  question interval resamples questions, the database interval resamples the 11 BIRD databases (ratio of
  summed values to summed counts); 10,000 draws, seed 20261106 for each of the two.
* `Sample198` (the 198 sampled questions as in Table IV): per question, repeats averaged within a
  condition, cells averaged; the question interval resamples questions (10,000 draws, seed 20261106), the
  database interval resamples databases and averages the questions of the drawn databases.

The minimum detectable effect at 80% power is (1.959964 + 0.841621) times the bootstrap standard error
(the standard deviation of the 10,000 bootstrap means).
"""
from __future__ import annotations

import collections
import csv
import itertools
import json
import pathlib

import numpy as np

BOOT, SEED = 10000, 20261106
Z = 1.959964 + 0.841621
DU = [("du_mad", "llama3-8b"), ("du_mad", "qwen2.5-7b")]
M2 = [("madm2_objective", "llama3-8b"), ("madm2_objective", "qwen2.5-7b")]
C4 = DU + M2
GROUPS = {"4cell": C4, "DuMAD_2cell": DU, "MADM2_2cell": M2, **{f"cell_{c[0]}|{c[1]}": [c] for c in C4}}


def r2(x):
    return round(100 * float(x), 2)


def excl0(ci):
    return bool(ci[0] > 0 or ci[1] < 0)


# --------------------------------------------------------------------------------------------------
# 1,531-question data and the per-question bootstrap
# --------------------------------------------------------------------------------------------------
class Rows:
    """inputs/rows.jsonl indexed by (source, protocol, backbone, condition) -> question -> runs.

    Conditions: FULL (with peers), NOPEER / NOPEERC (without peers; Du-MAD uses NOPEERC, the no-peer
    records whose later-round prompts contain no peer text; MAD-M2 uses NOPEER), PLACEBO (peers taken
    from other questions of the same database), T07_* (Du-MAD at temperature 0.7), FMT_* (unified SQL
    answer format). Source "orig" holds the 198 sampled questions, "rx" the controls and the 1,333
    remaining questions.
    """

    def __init__(self, inputs: pathlib.Path):
        self.rows = [json.loads(l) for l in (inputs / "rows.jsonl").open()]
        ql = json.loads((inputs / "question_lists.json").read_text())
        qlc = json.loads((inputs / "question_lists_e4c.json").read_text())
        gold_failed = set(json.loads((inputs / "eval_meta.json").read_text())["gold_failed_units"])
        self.blocks = {k: [u for u in v if u not in gold_failed] for k, v in
                       {"u198": ql["units198"], "E4a": ql["E4a"], "E4b": ql["E4b"], "E4c": qlc["E4c"]}.items()}
        self.U198 = self.blocks["u198"]
        self.NEWU = self.blocks["E4a"] + self.blocks["E4b"] + self.blocks["E4c"]
        self.DB = {r["unit"]: r["db"] for r in self.rows}
        self.DIFF = {r["unit"]: r["difficulty"] for r in self.rows}
        self.IDX = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in self.rows:
            self.IDX[(r["source"], r["protocol"], r["backbone"], r["arm"])][r["unit"]].append(r)
        self.METRIC = {"E1": lambda r: float(r["last"]) - float(r["r0"]),
                       "final": lambda r: float(r["final"]),
                       "r0": lambda r: float(r["r0"]), "last": lambda r: float(r["last"]),
                       "diverse": lambda r: float(r["r0_diverse"]),
                       "r0_strict_share": lambda r: float(r["r0_strict_share"]),
                       "r1_prompt_tokens": lambda r: float(r["r1_prompt_tokens"]),
                       "a_last": lambda r: float(r["last_strict"]),
                       "a_E1": lambda r: float(r["last_strict"]) - float(r["r0_strict"])}

    # ---- per question value of one condition (repeats averaged)
    def per_unit(self, src, cell, arm, metric, units):
        d = self.IDX[(src, cell[0], cell[1], arm)]
        f = self.METRIC[metric]
        return {u: float(np.mean([f(r) for r in d[u]])) for u in units if d.get(u)}

    def boot(self, items):
        if not items:
            return None
        by = collections.defaultdict(list)
        for u, v in items:
            by[u].append(v)
        us = sorted(by)
        sums = np.array([sum(by[u]) for u in us])
        cnts = np.array([len(by[u]) for u in us])
        est = sums.sum() / cnts.sum()
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(us), size=(BOOT, len(us)))
        tq = sums[idx].sum(1) / cnts[idx].sum(1)
        g = collections.defaultdict(list)
        for i, u in enumerate(us):
            g[self.DB[u]].append(i)
        keys = sorted(g)
        gs = np.array([sums[g[k]].sum() for k in keys])
        gc = np.array([cnts[g[k]].sum() for k in keys])
        rng = np.random.default_rng(SEED)
        pick = rng.integers(0, len(keys), size=(BOOT, len(keys)))
        td = gs[pick].sum(1) / gc[pick].sum(1)
        out = dict(est=r2(est), q_ci=[r2(np.percentile(tq, 2.5)), r2(np.percentile(tq, 97.5))],
                   db_ci=[r2(np.percentile(td, 2.5)), r2(np.percentile(td, 97.5))],
                   n_questions=len(us), n_items=int(cnts.sum()), n_dbs=len(keys),
                   mde80_q=r2(Z * tq.std(ddof=1)), mde80_db=r2(Z * td.std(ddof=1)))
        out["detected_both"] = excl0(out["q_ci"]) and excl0(out["db_ci"])
        return out

    def diff_items(self, spec_a, spec_b, cells, metric, units, require_all=True):
        per = {}
        for c in cells:
            sa = spec_a(c) if callable(spec_a) else spec_a
            sb = spec_b(c) if callable(spec_b) else spec_b
            a = self.per_unit(sa[0], c, sa[1], metric, units)
            b = self.per_unit(sb[0], c, sb[1], metric, units)
            per[c] = {u: a[u] - b[u] for u in set(a) & set(b)}
        common = set.intersection(*[set(v) for v in per.values()]) if require_all else None
        items = []
        for c in cells:
            for u, v in per[c].items():
                if common is None or u in common:
                    items.append((u, v))
        return items

    def did_items(self, full_a, nop_a, full_b, nop_b, cells, units):
        """(E1 full_a - E1 nop_a) - (E1 full_b - E1 nop_b) per question, averaged over cells."""
        per = {}
        for c in cells:
            def g(s, c=c):
                sp = s(c) if callable(s) else s
                return self.per_unit(sp[0], c, sp[1], "E1", units)
            fa, na, fb, nb = g(full_a), g(nop_a), g(full_b), g(nop_b)
            per[c] = {u: (fa[u] - na[u]) - (fb[u] - nb[u]) for u in set(fa) & set(na) & set(fb) & set(nb)}
        common = set.intersection(*[set(v) for v in per.values()])
        return [(u, per[c][u]) for c in cells for u in common]

    # ---- the 1,531-question contrast: with peers vs the no-peer records of each cell
    @staticmethod
    def nop_new(c):
        return ("rx", "NOPEERC") if c[0] == "du_mad" else ("rx", "NOPEER")

    @staticmethod
    def nop_198(c):
        return ("rx", "NOPEERC") if c[0] == "du_mad" else ("orig", "NOPEER")

    def fulldev_items(self, cells, metric, sets=("new", "198"), new_units=None, u198_units=None):
        newu = self.NEWU if new_units is None else new_units
        u198 = self.U198 if u198_units is None else u198_units
        items = []
        if "new" in sets:
            items += self.diff_items(("rx", "FULL"), self.nop_new, cells, metric, newu)
        if "198" in sets:
            items += self.diff_items(("orig", "FULL"), self.nop_198, cells, metric, u198)
        return items

    def fulldev_level(self, cells, side, metric):
        """accuracy of one condition on exactly the questions of the paired 1,531-question contrast."""
        keep = {u for u, _ in self.fulldev_items(cells, "E1")}
        items = []
        for c in cells:
            if side == "with":
                a = self.per_unit("rx", c, "FULL", metric, [u for u in self.NEWU if u in keep])
                b = self.per_unit("orig", c, "FULL", metric, [u for u in self.U198 if u in keep])
            else:
                sn, s1 = self.nop_new(c), self.nop_198(c)
                a = self.per_unit(sn[0], c, sn[1], metric, [u for u in self.NEWU if u in keep])
                b = self.per_unit(s1[0], c, s1[1], metric, [u for u in self.U198 if u in keep])
            items += list(a.items()) + list(b.items())
        return self.boot(items)


# --------------------------------------------------------------------------------------------------
# 198 sampled questions: per-run scores of Table IV and its bootstrap
# --------------------------------------------------------------------------------------------------
LAST, R0 = "lenient|exec|last|first", "lenient|exec|r0|first"


class Sample198:
    """inputs/per_run_scores.jsonl (the 198 sampled questions, two repeats per condition). The Du-MAD
    no-peer runs in this table are the no-peer records whose later-round prompts contain no peer text."""

    TERMS = {"last_diff": [(1, LAST, "FULL"), (-1, LAST, "NOPEER")],
             "r0_diff": [(1, R0, "FULL"), (-1, R0, "NOPEER")],
             "dpeer": [(1, LAST, "FULL"), (-1, R0, "FULL"), (-1, LAST, "NOPEER"), (1, R0, "NOPEER")],
             "later_rounds_with": [(1, LAST, "FULL"), (-1, R0, "FULL")]}

    def __init__(self, inputs: pathlib.Path):
        runs = [json.loads(l) for l in (inputs / "per_run_scores.jsonl").open()]
        self.runs = [r for r in runs if (r["protocol"], r["backbone"]) in C4]
        self.dbmap = {}
        with open(inputs / "bird_dev_questions.csv") as f:
            for row in csv.DictReader(f):
                self.dbmap[row["unit"]] = row["db_id"]

    def task_arrays(self, cell, terms):
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in self.runs:
            if (r["protocol"], r["backbone"]) == cell:
                by[r["unit"]][r["arm"]].append(r)
        units = sorted(u for u, a in by.items() if {"FULL", "NOPEER"} <= set(a))
        est = np.zeros(len(units))
        for k, u in enumerate(units):
            for sgn, key, arm in terms:
                est[k] += sgn * np.mean([r[key] for r in by[u][arm]])
        return units, est

    def db_cluster_ci(self, units, est):
        dbs = sorted({self.dbmap[u] for u in units})
        members = {d: np.array([i for i, u in enumerate(units) if self.dbmap[u] == d]) for d in dbs}
        rng = np.random.default_rng(SEED)
        out = np.empty(BOOT)
        for b in range(BOOT):
            pick = rng.integers(0, len(dbs), size=len(dbs))
            ix = np.concatenate([members[dbs[j]] for j in pick])
            out[b] = est[ix].mean()
        return out

    def evaluate(self, term, cells):
        arrays = {c: self.task_arrays(c, self.TERMS[term]) for c in C4}
        u0 = arrays[C4[0]][0]
        assert all(arrays[c][0] == u0 for c in C4)
        idx = np.random.default_rng(SEED).integers(0, len(u0), size=(BOOT, len(u0)))
        E = np.mean([arrays[c][1] for c in cells], axis=0)
        bq = E[idx].mean(axis=1)
        bd = self.db_cluster_ci(u0, E)
        q = [r2(np.percentile(bq, 2.5)), r2(np.percentile(bq, 97.5))]
        d = [r2(np.percentile(bd, 2.5)), r2(np.percentile(bd, 97.5))]
        return dict(est=r2(E.mean()), q_ci=q, db_ci=d, n_questions=len(u0),
                    mde80_q=r2(Z * bq.std(ddof=1)), mde80_db=r2(Z * bd.std(ddof=1)),
                    detected_both=excl0(q) and excl0(d), units=u0, per_question=E)

    def lodo(self):
        """Leave one database out, four-cell interaction effect (common units of the four cells)."""
        full = self.evaluate("dpeer", C4)
        units, E = full["units"], full["per_question"]
        out = {}
        for d in sorted({self.dbmap[u] for u in units}):
            keep = [i for i, u in enumerate(units) if self.dbmap[u] != d]
            x = E[keep]
            ku = [units[i] for i in keep]
            idx = np.random.default_rng(SEED).integers(0, len(x), size=(BOOT, len(x)))
            bq = x[idx].mean(axis=1)
            bd = self.db_cluster_ci(ku, x)
            out[d] = dict(est=r2(x.mean()), q_ci=[r2(np.percentile(bq, 2.5)), r2(np.percentile(bq, 97.5))],
                          db_ci=[r2(np.percentile(bd, 2.5)), r2(np.percentile(bd, 97.5))])
        return out


# --------------------------------------------------------------------------------------------------
# Extractor re-reading on the 1,531 questions (per-agent table: inputs/candidates.jsonl)
# --------------------------------------------------------------------------------------------------
def vote(ag, drop=None, strict=False):
    """execution vote over one round's agents, first tie; returns winner correctness."""
    lab = []
    for i, a in enumerate(ag):
        ok = a["l"] and (a["s"] or not strict) and i != drop
        lab.append(a["sig"] if ok else None)
    live = [x for x in lab if x is not None]
    if not live:
        return False
    cnt = collections.Counter(live)
    top = max(cnt.values())
    first = next(x for x in live if cnt[x] == top)
    return bool(ag[lab.index(first)]["c"])


def vote_lab(labs):
    live = [s for s, _ in labs if s is not None]
    if not live:
        return False
    cnt = collections.Counter(live)
    top = max(cnt.values())
    first = next(x for x in live if cnt[x] == top)
    return bool(labs[[s for s, _ in labs].index(first)][1])


def options(a, var):
    """distinct (signature, correct) outcomes over a reply's candidate queries. P: the candidates the
    lenient extractor's own detection can return; B: every fenced block and every bare SELECT/WITH line."""
    cs = a["bcands"] if var == "B" else a["cands"]
    return sorted({(x[0], x[1]) for x in cs}, key=lambda t: (t[0] is None, str(t[0]), t[1]))


def joint_range(base, idxs, opts):
    if not idxs:
        v = vote_lab(base)
        return v, v
    lists = [opts[i] for i in idxs]
    lo, hi = True, False
    lab = list(base)
    for combo in itertools.product(*lists):
        for i, o in zip(idxs, combo):
            lab[i] = o
        v = vote_lab(lab)
        lo, hi = lo and v, hi or v
        if hi and not lo:
            break
    return lo, hi


class Reader:
    """Re-reads the 1,531-question Du-MAD / MAD-M2 runs from the per-agent candidate table."""

    ROUNDS = ("r0", "last")
    VARIANTS = ("P", "B")

    def __init__(self, R: Rows, inputs: pathlib.Path):
        self.R = R
        cand = {}
        for l in (inputs / "candidates.jsonl").open():
            d = json.loads(l)
            cand[tuple(d["key"])] = d
        n = 0
        for r in R.rows:
            k = (r["source"], r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))
            if k not in cand:
                continue
            d = cand[k]
            n += 1
            for rd in self.ROUNDS:
                g = d[rd]
                ag = [dict(s=a["s"], l=a["l"], sig=a["cur"]["sig"] if a["l"] else None,
                           c=a["cur"]["c"] if a["l"] else 0) for a in g]
                v = vote(ag)
                assert v == r[rd] and vote(ag, strict=True) == r[rd + "_strict"], (k, rd)
                rec = [i for i, a in enumerate(ag) if a["l"] and not a["s"]]
                dep = any(vote(ag, drop=i) != v for i in rec)
                r[f"h_{rd}"] = float(v)
                r[f"dep_{rd}"] = float(dep)
                r[f"x_{rd}"] = float(v)
                base = [((a["cur"]["sig"], a["cur"]["c"]) if a["l"] else (None, 0)) for a in g]
                r[f"_g_{rd}"] = g
                r[f"_base_{rd}"] = base
        self.n_runs = n
        R.METRIC.update({"h_last": lambda r: r["h_last"], "h_E1": lambda r: r["h_last"] - r["h_r0"],
                         "x_last": lambda r: r["x_last"], "x_E1": lambda r: r["x_last"] - r["x_r0"]})
        self._annotate_amb()

    def _classify(self, r, rd, var):
        g, base = r[f"_g_{rd}"], r[f"_base_{rd}"]
        rec = [i for i, a in enumerate(g) if a["l"] and not a["s"]]
        opts = {i: options(g[i], var) for i in rec}
        for i in rec:
            if base[i] not in opts[i]:
                opts[i] = opts[i] + [base[i]]
        amb = [i for i in rec if len(opts[i]) > 1]
        return rec, opts, amb

    def _annotate_amb(self):
        for r in self.R.rows:
            if "_g_r0" not in r:
                continue
            for rd in self.ROUNDS:
                base = r[f"_base_{rd}"]
                for var in self.VARIANTS:
                    rec, opts, amb = self._classify(r, rd, var)
                    lo, hi = joint_range(base, amb, opts)
                    r[f"amin{var}_{rd}"], r[f"amax{var}_{rd}"] = float(lo), float(hi)
                    if var == "P":
                        r[f"_rec_{rd}"], r[f"_amb_{rd}"] = rec, amb
                v = vote_lab(base)
                piv1 = []
                for i in r[f"_rec_{rd}"]:
                    lab = list(base)
                    lab[i] = (None, 0)
                    if vote_lab(lab) != v:
                        piv1.append(i)
                r[f"_piv1_{rd}"] = piv1

    def side_rows(self, cells):
        R = self.R
        keep = {u for u, _ in R.fulldev_items(cells, "h_E1")}
        out = {"with": [], "without": []}
        for c in cells:
            for src_arm, units, side in ((("rx", "FULL"), R.NEWU, "with"), (("orig", "FULL"), R.U198, "with"),
                                         (None, R.NEWU, "without"), (None, R.U198, "without")):
                sa = src_arm or (R.nop_new(c) if units is R.NEWU else R.nop_198(c))
                d = R.IDX[(sa[0], c[0], c[1], sa[1])]
                for u in units:
                    if u in keep:
                        out[side] += d.get(u, [])
        return out

    @staticmethod
    def reset(rows):
        for r in rows:
            r["x_r0"], r["x_last"] = r["h_r0"], r["h_last"]

    def worst_amb(self, cells, metric, direction, var):
        """every ambiguous recovered reply reassigned jointly to the outcome least favourable to the
        estimate; direction -1 shrinks a negative estimate, +1 a positive one."""
        sr = self.side_rows(cells)
        self.reset(sr["with"] + sr["without"])
        nflip = 0
        rounds = ("last",) if metric == "last" else ("last", "r0")
        for side, rows in sr.items():
            for rd in rounds:
                up = (side == "with") == (rd == "last")
                if direction > 0:
                    up = not up
                for r in rows:
                    new = r[f"amax{var}_{rd}"] if up else r[f"amin{var}_{rd}"]
                    nflip += new != r[f"x_{rd}"]
                    r[f"x_{rd}"] = new
        b = self.R.boot(self.R.fulldev_items(cells, f"x_{metric}"))
        self.reset(sr["with"] + sr["without"])
        b["n_votes_changed"] = int(nflip)
        return b

    def perturb_means(self, cells, metric, p, rng, directed, ndraw):
        """flip each single-removal dependent vote with probability p (directed: only toward shrinking a
        negative estimate); returns the mean estimate over the draws. The random stream is consumed in the
        source's order (one uniform number per dependent vote, with peers before without, last round before
        initial answers); the estimate of a draw is updated from the flipped votes only."""
        sr = self.side_rows(cells)
        rows_all = sr["with"] + sr["without"]
        self.reset(rows_all)
        items = self.R.fulldev_items(cells, f"x_{metric}")
        n_items = len(items)
        base = sum(v for _, v in items)
        nrep = collections.Counter((r["unit"], r["protocol"], r["backbone"], r["source"], r["arm"]) for r in rows_all)
        rounds = ("last",) if metric == "last" else ("last", "r0")
        entries = []                                  # (current value, raise if directed, weight), source order
        for side, rows in sr.items():
            for rd in rounds:
                up = (side == "with") == (rd == "last")
                sgn = (1 if side == "with" else -1) * (1 if rd == "last" else -1)
                for r in rows:
                    if r[f"dep_{rd}"]:
                        w = sgn / nrep[(r["unit"], r["protocol"], r["backbone"], r["source"], r["arm"])]
                        entries.append((r[f"x_{rd}"], up, w))
        x0 = np.array([e[0] for e in entries])
        upv = np.array([e[1] for e in entries])
        wv = np.array([e[2] for e in entries])
        est = []
        for _ in range(ndraw):
            hit = rng.random(len(entries)) < p
            if directed:
                hit &= x0 == np.where(upv, 0.0, 1.0)
            delta = float(np.sum((1.0 - 2.0 * x0[hit]) * wv[hit]))
            est.append(r2((base + delta) / n_items))
        return float(np.mean(est))

    @staticmethod
    def _est(items):
        return r2(sum(v for _, v in items) / len(items))
