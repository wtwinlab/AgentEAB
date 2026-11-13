"""Peer effects of Du-MAD and MAD-M2 on BIRD under fixed output rules, tie rules and per-agent transitions.

    python3 code/compute.py --out DIR     # -> DIR/values.json, DIR/effects.json

Inputs (inputs/):
  runs_198.csv        the 198 sampled questions: one row per debate run, correctness of the vote under
                      the protocol extractor with the exact-match vote, the protocol extractor with the
                      execution vote and the lenient extractor with the execution vote, for the initial
                      answers and the last round, under first-, last- and random-agent ties
  runs_1531.csv       the 1,531 development questions: one row per debate run, correctness of the
                      execution vote under both extractors, both rounds and the three tie rules, and the
                      correctness of each agent's initial and last-round answer (lenient extractor)
  question_sets.json  the 198 sampled questions and the 1,333 remaining questions

Peer interaction effect per question = (last round - initial answers) with peers minus the same change
without peers; repeats are averaged within a question and condition, configurations are averaged per
question. A contrast is detected when its question-level and database-cluster 95% intervals both
exclude 0.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
BOOT, SEED = 10000, 20261106
DU = [("du_mad", "llama3-8b"), ("du_mad", "qwen2.5-7b")]
M2 = [("madm2_objective", "llama3-8b"), ("madm2_objective", "qwen2.5-7b")]
C4 = DU + M2
GROUPS = {"4cell": C4, "DuMAD_2cell": DU, "MADM2_2cell": M2, **{f"cell_{c[0]}|{c[1]}": [c] for c in C4}}
TIES = ("first", "last", "random")


def r2(x):
    return round(100 * float(x), 2)


def excl0(ci):
    return bool(ci[0] > 0 or ci[1] < 0)


# ================================================================== 198 sampled questions
class Sample198:
    """Per-run scores of the four Du-MAD and MAD-M2 configurations on the 198 sampled questions.
    Question interval: 10,000 draws of questions; database interval: 10,000 draws of the 11 databases
    (summed values over summed question counts); seed 20261106 for each."""

    def __init__(self):
        self.by = collections.defaultdict(lambda: collections.defaultdict(list))
        self.db = {}
        with open(INP / "runs_198.csv") as f:
            for r in csv.DictReader(f):
                self.by[(r["protocol"], r["backbone"])][r["unit"]].append(r)
                self.db[r["unit"]] = r["db"]
        self.units = sorted(set.intersection(*[{u for u, rs in self.by[c].items() if len(rs) == 4} for c in C4]))
        n = len(self.units)
        self.qidx = np.random.default_rng(SEED).integers(0, n, size=(BOOT, n))
        dbs = sorted({self.db[u] for u in self.units})
        self.members = [np.array([i for i, u in enumerate(self.units) if self.db[u] == d]) for d in dbs]
        rng = np.random.default_rng(SEED)
        self.pick = np.array([rng.integers(0, len(dbs), size=len(dbs)) for _ in range(BOOT)])

    def ci(self, d):
        bq = d[self.qidx].mean(axis=1)
        gs = np.array([d[m].sum() for m in self.members])
        gc = np.array([len(m) for m in self.members])
        bd = gs[self.pick].sum(1) / gc[self.pick].sum(1)
        q = [r2(np.percentile(bq, 2.5)), r2(np.percentile(bq, 97.5))]
        b = [r2(np.percentile(bd, 2.5)), r2(np.percentile(bd, 97.5))]
        return dict(est=r2(d.mean()), q_ci=q, db_ci=b, detected=excl0(q) and excl0(b))

    def arr(self, c, key, cond):
        return np.array([np.mean([float(x[key]) for x in self.by[c][u] if x["condition"] == cond])
                         for u in self.units])

    def dpeer(self, cells, reading, tie):
        e, v = reading.split("|")
        kl, k0 = f"{e}|{v}|last|{tie}", f"{e}|{v}|r0|{tie}"
        wf = np.mean([self.arr(c, kl, "FULL") - self.arr(c, k0, "FULL") for c in cells], axis=0)
        wn = np.mean([self.arr(c, kl, "NOPEER") - self.arr(c, k0, "NOPEER") for c in cells], axis=0)
        return self.ci(wf - wn)


# ================================================================== 1,531 questions
class Rows:
    """Runs on the 1,531 questions indexed by (source, protocol, backbone, condition) -> question.

    With peers: FULL. Without peers: NOPEERC for Du-MAD (later-round prompts contain no peer text),
    NOPEER for MAD-M2. Source "orig" holds the 198 sampled questions, "rx" the 1,333 remaining
    questions and the Du-MAD no-peer runs on all 1,531."""

    def __init__(self):
        qs = json.loads((INP / "question_sets.json").read_text())
        self.U198, self.NEWU = qs["sampled"], qs["remaining"]
        self.rows = []
        with open(INP / "runs_1531.csv") as f:
            for r in csv.DictReader(f):
                for k in list(r):
                    if k.count("|") == 2:
                        r[k] = float(r[k])
                a0 = [int(x) for x in r["agents_r0"]]
                a1 = [int(x) for x in r["agents_last"]]
                n = len(a0)
                cnt = collections.Counter(("RET" if y else "LOSS") if x else ("REC" if y else "STAY")
                                          for x, y in zip(a0, a1))
                for k in ("RET", "LOSS", "REC", "STAY"):
                    r[f"n_{k}"] = cnt[k]
                    r[k] = cnt[k] / n
                r["net"] = r["REC"] - r["LOSS"]
                self.rows.append(r)
        self.DB = {r["unit"]: r["db"] for r in self.rows}
        self.IDX = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in self.rows:
            self.IDX[(r["source"], r["protocol"], r["backbone"], r["condition"])][r["unit"]].append(r)
        self.METRIC = {}
        for e in ("strict", "lenient"):
            for t in TIES:
                self.METRIC[f"E1_{e}_{t}"] = (lambda r, e=e, t=t: r[f"{e}|last|{t}"] - r[f"{e}|r0|{t}"])
                self.METRIC[f"last_{e}_{t}"] = (lambda r, e=e, t=t: r[f"{e}|last|{t}"])
                self.METRIC[f"r0_{e}_{t}"] = (lambda r, e=e, t=t: r[f"{e}|r0|{t}"])
        for k in ("LOSS", "REC", "net"):
            self.METRIC[k] = (lambda r, k=k: r[k])

    def per_unit(self, src, cell, cond, metric, units):
        d = self.IDX[(src, cell[0], cell[1], cond)]
        f = self.METRIC[metric]
        return {u: float(np.mean([f(r) for r in d[u]])) for u in units if d.get(u)}

    def boot(self, items):
        """Question-level and database-cluster percentile intervals (10,000 draws each, seed 20261106);
        repeats and configurations enter as items of their question."""
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
                   db_ci=[r2(np.percentile(td, 2.5)), r2(np.percentile(td, 97.5))], n_questions=len(us))
        out["detected"] = excl0(out["q_ci"]) and excl0(out["db_ci"])
        return out

    @staticmethod
    def nop_new(c):
        return ("rx", "NOPEERC") if c[0] == "du_mad" else ("rx", "NOPEER")

    @staticmethod
    def nop_198(c):
        return ("rx", "NOPEERC") if c[0] == "du_mad" else ("orig", "NOPEER")

    def diff_items(self, spec_a, spec_b, cells, metric, units):
        per = {}
        for c in cells:
            sb = spec_b(c)
            a = self.per_unit(spec_a[0], c, spec_a[1], metric, units)
            b = self.per_unit(sb[0], c, sb[1], metric, units)
            per[c] = {u: a[u] - b[u] for u in set(a) & set(b)}
        common = set.intersection(*[set(v) for v in per.values()])
        return [(u, v) for c in cells for u, v in per[c].items() if u in common]

    def fulldev_items(self, cells, metric, sets=("new", "198")):
        items = []
        if "new" in sets:
            items += self.diff_items(("rx", "FULL"), self.nop_new, cells, metric, self.NEWU)
        if "198" in sets:
            items += self.diff_items(("orig", "FULL"), self.nop_198, cells, metric, self.U198)
        return items

    def fulldev_level(self, cells, side, metric):
        """Accuracy of one condition on the questions of the paired 1,531-question contrast."""
        keep = {u for u, _ in self.fulldev_items(cells, "E1_lenient_first")}
        items = []
        for c in cells:
            if side == "with":
                specs = [(("rx", "FULL"), self.NEWU), (("orig", "FULL"), self.U198)]
            else:
                specs = [(self.nop_new(c), self.NEWU), (self.nop_198(c), self.U198)]
            for (src, cond), us in specs:
                items += list(self.per_unit(src, c, cond, metric, [u for u in us if u in keep]).items())
        return self.boot(items)

    # ---- conditional loss and recovery: ratios of summed agent counts per configuration
    def cond_rates(self, cells):
        pairs = [(("rx", "FULL"), self.nop_new, self.NEWU), (("orig", "FULL"), self.nop_198, self.U198)]
        per_cell = {}
        for c in cells:
            sides = {"with": {}, "without": {}}
            for fa, fb, us in pairs:
                for side, spec in (("with", fa), ("without", fb(c))):
                    d = self.IDX[(spec[0], c[0], c[1], spec[1])]
                    for u in us:
                        if d.get(u):
                            sides[side][u] = d[u]
            per_cell[c] = sides
        units = sorted(set.intersection(*[set(s["with"]) & set(s["without"]) for s in per_cell.values()]))
        B = CountBoot(units, self.DB)
        out = {}
        for nm, num, den in (("cond_loss", "n_LOSS", ("n_RET", "n_LOSS")),
                             ("cond_rec", "n_REC", ("n_REC", "n_STAY"))):
            arrs = {}
            for c, sides in per_cell.items():
                for side, d in sides.items():
                    nu = np.array([sum(x[num] for x in d[u]) for u in units], float)
                    de = np.array([sum(x[den[0]] + x[den[1]] for x in d[u]) for u in units], float)
                    arrs[(c, side)] = (nu, de)
            out[nm] = dict(with_=B.ratio([[arrs[(c, "with")]] for c in cells]),
                           without=B.ratio([[arrs[(c, "without")]] for c in cells]),
                           diff=B.ratio([[arrs[(c, "with")], arrs[(c, "without")]] for c in cells]),
                           n_questions=len(units))
        return out


class CountBoot:
    """Question and database-cluster draws as resampling counts over a fixed list of questions
    (10,000 draws each, seed 20261106)."""

    def __init__(self, units, dbmap):
        n = len(units)
        qidx = np.random.default_rng(SEED).integers(0, n, size=(BOOT, n))
        dbs = sorted({dbmap[u] for u in units})
        members = {d: np.array([i for i, u in enumerate(units) if dbmap[u] == d]) for d in dbs}
        rng = np.random.default_rng(SEED)
        self.dcount = np.zeros((BOOT, n))
        for b in range(BOOT):
            for j in rng.integers(0, len(dbs), size=len(dbs)):
                self.dcount[b, members[dbs[j]]] += 1
        self.qcount = np.zeros((BOOT, n))
        for b in range(BOOT):
            np.add.at(self.qcount[b], qidx[b], 1)
        self.n = n

    def ratio(self, cells_num_den):
        """Mean over configurations of num/den (one part) or of num_a/den_a - num_b/den_b (two parts)."""
        def stat(W):
            tot = 0.0
            for parts in cells_num_den:
                vals = [(W @ nu) / np.maximum(W @ de, 1e-12) for nu, de in parts]
                tot = tot + (vals[0] - vals[1] if len(vals) == 2 else vals[0])
            return tot / len(cells_num_den)
        est = float(stat(np.ones((1, self.n)))[0])
        bq, bd = stat(self.qcount), stat(self.dcount)
        q = [r2(np.percentile(bq, 2.5)), r2(np.percentile(bq, 97.5))]
        d = [r2(np.percentile(bd, 2.5)), r2(np.percentile(bd, 97.5))]
        return dict(est=r2(est), q_ci=q, db_ci=d, detected=excl0(q) and excl0(d))


def put(V, key, b):
    V[key] = b["est"]
    V[key + "_q_lo"], V[key + "_q_hi"] = b["q_ci"]
    V[key + "_db_lo"], V[key + "_db_hi"] = b["db_ci"]
    V[key + "_detected"] = b["detected"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    V, E = {}, {}

    # ---- 198 sampled questions: peer interaction effect by output rule and tie rule
    S = Sample198()
    V["s198_n_questions"] = len(S.units)
    g198 = {"4cell": C4, "DuMAD_2cell": DU, "MADM2_2cell": M2}
    for g, cells in g198.items():
        for reading in ("strict|string", "lenient|exec"):
            for t in TIES:
                b = S.dpeer(cells, reading, t)
                put(V, f"s198|{g}|{reading}|{t}|dpeer", b)
                E[f"s198|{g}|{reading}|{t}|dpeer"] = b

    # ---- 1,531 questions
    R = Rows()
    V["n_questions_fulldev"] = len(R.U198) + len(R.NEWU)
    V["n_questions_remaining"] = len(R.NEWU)
    for g, cells in GROUPS.items():
        for term, met in (("last_diff", "last_lenient_first"), ("r0_diff", "r0_lenient_first")):
            b = R.boot(R.fulldev_items(cells, met))
            put(V, f"{g}|{term}", b)
            E[f"{g}|{term}"] = b
        for side in ("with", "without"):
            for m in ("r0", "last"):
                V[f"{g}|{m}_{side}"] = R.fulldev_level(cells, side, f"{m}_lenient_first")["est"]
        for t in TIES:
            b = R.boot(R.fulldev_items(cells, f"E1_lenient_{t}"))
            put(V, f"{g}|dpeer|{t}", b)
            E[f"{g}|dpeer|{t}"] = b
        for block, sets in (("1333", ("new",)), ("198", ("198",))):
            b = R.boot(R.fulldev_items(cells, "E1_lenient_first", sets=sets))
            put(V, f"{g}|dpeer|first|{block}", b)
            E[f"{g}|dpeer|first|{block}"] = b
        # per-agent transitions: shares of all initial-to-last agent trajectories
        for m in ("LOSS", "REC", "net"):
            b = R.boot(R.fulldev_items(cells, m))
            put(V, f"{g}|agent_{m}", b)
            E[f"{g}|agent_{m}"] = b
        if g in ("4cell", "DuMAD_2cell", "MADM2_2cell"):
            cr = R.cond_rates(cells)
            for nm, d in cr.items():
                put(V, f"{g}|{nm}_diff", d["diff"])
                V[f"{g}|{nm}_with"] = d["with_"]["est"]
                V[f"{g}|{nm}_without"] = d["without"]["est"]
                E[f"{g}|{nm}"] = d

    # ---- the two Du-MAD no-peer collections on the 198 sampled questions
    readings = [f"{e}|exec|{rd}|{t}" for e in ("strict", "lenient") for rd in ("r0", "last") for t in TIES]
    s198 = {}
    for c in DU:
        for u, rs in S.by[c].items():
            for x in rs:
                if x["condition"] == "NOPEER":
                    s198[(u, c[1], int(x["repeat"]))] = x
    n_runs = n_diff = 0
    for c in DU:
        for u in R.U198:
            for r in R.IDX[("rx", c[0], c[1], "NOPEERC")].get(u, []):
                x = s198.get((u, c[1], int(r["repeat"])))
                if x is None:
                    continue
                n_runs += 1
                n_diff += any(float(x[k]) != r[k.replace("|exec", "")] for k in readings)
    V["dumad_nopeer_runs_compared"] = n_runs
    V["dumad_nopeer_runs_differing"] = n_diff

    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n")
    (out / "effects.json").write_text(json.dumps(E, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
