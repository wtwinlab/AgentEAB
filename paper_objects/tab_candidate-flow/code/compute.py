"""Candidate flow on the BIRD debates with peers: Section VI-A (sec:candidate-flow), Eq. eq:candidate-flow,
Supplementary Table tab:s-candidate-flow, and the extractor-vote interaction under the three tie
rules with its corrections and regressions (supplementary sec:s-output-interactions).

    python3 code/compute.py --out output     # -> output/values.json, output/candidate_flow.csv

Reads only ../inputs/candidate_flow_runs.csv: one row per run with peers (198 BIRD questions, two
repeats, Du-MAD, MAD-M2 and BCCS with Llama-3-8B or Qwen2.5-7B). Per run:

  C0  some agent's initial answer is correct           (lenient extractor)
  CT  some agent's last-round answer is correct        (lenient extractor)
  CD  some last-round answer read by the protocol extractor is correct
  Y   the baseline vote is correct                     (protocol extractor, exact-match vote, first-agent ties)
  Yx  lenient extractor and execution vote, first-agent ties

  1 - Y = (1 - C0) + (C0 - CT) + (CT - CD) + (CD - Y)

Loss is a run whose initial pool holds a correct answer and whose last-round pool does not;
recovery is the reverse. Repeats are averaged within a question; multi-configuration values are
equal-weight means over configurations per question. Question intervals are percentile bootstrap
intervals over the 198 questions (10,000 draws, seed 20261106); database intervals resample the 11
BIRD databases with replacement (10,000 sequential draws, seed 20261106). An effect is detected when
both 95% intervals exclude 0.
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
PROTOCOLS = ("Du-MAD", "MAD-M2", "BCCS")
BACKBONES = ("llama3-8b", "qwen2.5-7b")
CELLS6 = [(p, b) for p in PROTOCOLS for b in BACKBONES]
CELLS4 = CELLS6[:4]
TIES = ("first", "last", "random")
BASE = "protocol_exact_first"
TAG = {"Du-MAD": "dumad", "MAD-M2": "madm2", "BCCS": "bccs", "llama3-8b": "llama", "qwen2.5-7b": "qwen"}
FLOW = ("C0", "CT", "CD", "Y", "loss", "recovery", "Yx")


def flags(r):
    c0, ct, cd = int(r["C0"]), int(r["CT"]), int(r["CD"])
    y, yx = int(r[BASE]), int(r["lenient_execution_first"])
    return dict(C0=c0, CT=ct, CD=cd, Y=y, Yx=yx, loss=int(c0 == 1 and ct == 0),
                recovery=int(c0 == 0 and ct == 1), returned=int(r["final"]),
                ordered=int(ct >= cd >= y))


class Boot:
    """Question-level and database-cluster percentile bootstrap over the same question order."""

    def __init__(self, units, dbmap):
        self.qidx = np.random.default_rng(SEED).integers(0, len(units), size=(BOOT, len(units)))
        dbs = sorted({dbmap[u] for u in units})
        self.members = [np.array([i for i, u in enumerate(units) if dbmap[u] == d]) for d in dbs]
        rng = np.random.default_rng(SEED)
        self.pick = np.array([rng.integers(0, len(dbs), size=len(dbs)) for _ in range(BOOT)])

    def ci(self, d):
        bq = d[self.qidx].mean(axis=1)
        gs = np.array([d[m].sum() for m in self.members])
        gc = np.array([len(m) for m in self.members])
        bd = gs[self.pick].sum(1) / gc[self.pick].sum(1)
        q = [100 * float(np.percentile(bq, x)) for x in (2.5, 97.5)]
        b = [100 * float(np.percentile(bd, x)) for x in (2.5, 97.5)]
        det = (q[0] > 0 or q[1] < 0) and (b[0] > 0 or b[1] < 0)
        return 100 * float(d.mean()), q, b, int(det)


def r4(x):
    return round(float(x), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    runs = list(csv.DictReader(open(INP / "candidate_flow_runs.csv", encoding="utf-8")))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    dbmap = {}
    for r in runs:
        by[(r["protocol"], r["backbone"])][r["question"]].append(r)
        dbmap[r["question"]] = r["db_id"]
    units = sorted(set.intersection(*[set(by[c]) for c in CELLS6]))
    V = {"questions": len(units), "runs": len(runs),
         "runs_four_configurations": sum(len(by[c][u]) for c in CELLS4 for u in units),
         "repeats_per_question": min(len(by[c][u]) for c in CELLS6 for u in units)}

    def arr(c, f):
        return np.array([np.mean([f(x) for x in by[c][u]]) for u in units], dtype=float)

    # ---- candidate flow per configuration and mean of six
    table, stack = [], {t: [] for t in FLOW + ("returned",)}
    ordered = identity = 1
    for c in CELLS6:
        fl = {t: arr(c, lambda x, t=t: flags(x)[t]) for t in FLOW + ("returned",)}
        ordered &= int(all(flags(x)["ordered"] for u in units for x in by[c][u]))
        lhs = 1 - fl["Y"]
        rhs = (1 - fl["C0"]) + (fl["C0"] - fl["CT"]) + (fl["CT"] - fl["CD"]) + (fl["CD"] - fl["Y"])
        identity &= int(np.allclose(lhs, rhs, atol=1e-12))
        tag = f"{TAG[c[0]]}_{TAG[c[1]]}"
        row = {"protocol": c[0], "backbone": c[1]}
        for t in FLOW + ("returned",):
            stack[t].append(fl[t])
            V[f"{t}_{tag}"] = r4(100 * fl[t].mean())
            row[t] = f"{100 * fl[t].mean():.2f}"
        table.append(row)
    mean = {t: np.mean(stack[t], axis=0) for t in stack}
    row = {"protocol": "Mean", "backbone": ""}
    for t in FLOW:
        V[f"{t}_mean6"] = r4(100 * mean[t].mean())
        row[t] = f"{100 * mean[t].mean():.2f}"
    table.append(row)
    m = {t: 100 * mean[t].mean() for t in FLOW}
    V["revision_net_mean6"] = r4(m["C0"] - m["CT"])
    V["extraction_gap_mean6"] = r4(m["CT"] - m["CD"])
    V["vote_gap_mean6"] = r4(m["CD"] - m["Y"])
    V["Yx_minus_Y_mean6"] = r4(m["Yx"] - m["Y"])
    V["flow_identity_holds"] = identity
    V["CT_ge_CD_ge_Y_every_run"] = ordered
    V["initial_pool_correct_below_half"] = int(m["C0"] < 50)
    V["extraction_gap_exceeds_revision_net"] = int(m["CT"] - m["CD"] > m["C0"] - m["CT"])
    V["vote_gap_exceeds_revision_net"] = int(m["CD"] - m["Y"] > m["C0"] - m["CT"])

    # ---- extractor-vote interaction under each tie rule (relative to the protocol extractor and
    #      exact-match vote at the same tie rule)
    B = Boot(units, dbmap)
    groups = {"mean4": CELLS4, "mean6": CELLS6}

    def garr(g, f):
        return np.mean([arr(c, f) for c in groups[g]], axis=0)

    for g in groups:
        for t in TIES:
            y = {k: garr(g, lambda x, k=k: int(x[f"{k}_{t}"]))
                 for k in ("protocol_exact", "lenient_exact", "protocol_execution", "lenient_execution")}
            d_e = y["lenient_exact"] - y["protocol_exact"]
            d_i = y["protocol_execution"] - y["protocol_exact"]
            d_ei = y["lenient_execution"] - y["protocol_exact"]
            est, q, b, det = B.ci(d_ei - d_e - d_i)
            V[f"interaction_{g}_{t}"] = r4(est)
            V[f"interaction_{g}_{t}_q_lo"], V[f"interaction_{g}_{t}_q_hi"] = r4(q[0]), r4(q[1])
            V[f"interaction_{g}_{t}_db_lo"], V[f"interaction_{g}_{t}_db_hi"] = r4(b[0]), r4(b[1])
            V[f"interaction_{g}_{t}_detected"] = det

    # ---- corrections (baseline wrong, changed rule right) and regressions over the four
    #      Du-MAD and MAD-M2 configurations, counted in runs
    for name, k in (("joint", "lenient_execution_first"), ("tie_last", "protocol_exact_last")):
        rs = [x for c in CELLS4 for u in units for x in by[c][u]]
        V[f"{name}_corrections_runs_mean4"] = sum(int(x[BASE]) == 0 and int(x[k]) == 1 for x in rs)
        V[f"{name}_regressions_runs_mean4"] = sum(int(x[BASE]) == 1 and int(x[k]) == 0 for x in rs)

    with open(out / "candidate_flow.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
