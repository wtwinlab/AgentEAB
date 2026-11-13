"""SQL adapter: per-question outcomes from the existing graders, state from the request files.

Outcomes are the execution-match scores already produced by the existing scorers
(`score_repeat_generation.py`, `score_sal.parse` with candidate labels) and stored in
`repeat/output/per_question_decodes.csv`; this adapter does not re-execute SQL. Usage comes from
the raw call records.
"""
from __future__ import annotations

import collections
import csv
import json

import core as C

SRC = C.REPO / "data/derived/sql_pool_operation_transfer"
OPMAP = {"G": "Generation", "S_al": "S_al"}


def _last(path):
    d = {}
    for r in map(json.loads, open(path)):
        if r["unit_id"] not in d or r["status"] == "ok":
            d[r["unit_id"]] = r
    return d


def load(spec):
    rows = list(csv.DictReader(open(SRC / "repeat/output/per_question_decodes.csv")))
    state = {}
    for s, pool in (("P0", "original"), ("P1", "qwen3_glm")):
        for r in map(json.loads, open(SRC / f"data/requests_Sal_{s}.jsonl")):
            c = r["messages"][-1]["content"]
            state[(pool, r["benchmark"], r["question_id"])] = dict(
                candidate_set=len(r["id_to_model"]) if r["id_to_model"] else None,
                execution_feedback=("execution" in c.lower()) or None)
    inp = [dict(output="candidates", produced_at_step="candidate_acquisition", producer="candidate_pool"),
           dict(output="execution_feedback", produced_at_step="execution_feedback", producer="read_only_executor")]
    recs, pops = [], {"full": collections.defaultdict(set), "common": collections.defaultdict(set)}
    by_pool = collections.defaultdict(set)
    for r in rows:
        pool, u = r["pool"], (r["benchmark"], int(r["question_id"]))
        st = state.get((pool,) + u, {})
        by_pool[pool].add(u)
        pops["full"][pool].add(u)
        for op in ("G", "S_al"):
            for d in (1, 2):
                recs.append(C.Record(op=op, pool=pool, unit=u, cluster=r["db_id"], repeat=d,
                                     value=float(r[f"{OPMAP[op]}_d{d}"]), state=st, inputs=inp))
                if op == "G":
                    recs.append(C.Record(op="G_identity", pool=pool, unit=u, cluster=r["db_id"], repeat=d,
                                         value=float(r[f"Generation_d{d}"]), state=st, inputs=inp))
        recs.append(C.Record(op="Nonempty", pool=pool, unit=u, cluster=r["db_id"], repeat=None,
                             value=float(r["Nonempty"]), state=st, inputs=inp))
    excl = {("bird", 518), ("bird", 701)}
    common = (by_pool["original"] & by_pool["qwen3_glm"]) - excl
    for p in ("original", "qwen3_glm"):
        pops["common"][p] = set(common)
    return recs, {k: dict(v) for k, v in pops.items()}


def costs(spec, recs, pops):
    """Database-equal mean tokens per (arm, decode), Qwen3/GLM pool, common population."""
    cl = C.cluster_of(recs)
    files = {("G", 1): (SRC / "raw/main_G.jsonl", None), ("S_al", 1): (SRC / "raw/main_Sal_P1.jsonl", None),
             ("G", 2): (SRC / "repeat/raw/main_repeat.jsonl", "G"), ("S_al", 2): (SRC / "repeat/raw/main_repeat.jsonl", "Sal")}
    out = {}
    cache = {}
    for (op, d), (path, arm) in files.items():
        if path not in cache:
            cache[path] = _last(path)
        tok = {}
        for r in cache[path].values():
            if arm is not None and r.get("arm") != arm:
                continue
            u = r.get("usage") or {}
            tok[(r["benchmark"], r["question_id"])] = (u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0,
                                                      (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        for b in ("bird", "spider"):
            units = {x for x in pops["common"]["qwen3_glm"] if x[0] == b}
            row = {}
            for i, name in enumerate(("mean_input", "mean_output", "mean_reasoning")):
                row[name] = C.cluster_boot({x: tok.get(x, (0, 0, 0))[i] for x in units}, cl, "cluster_equal", 1, 0)["point"]
            row["calls"] = sum(x in tok for x in units)
            out[(op, d, b)] = row
    return out


def analyze(spec, recs, pops):
    est = C.run_estimates(spec, recs, pops)
    cost = costs(spec, recs, pops)
    got = {k: v for k, v in est.items()}
    got["S6a"] = dict(value=cost[("S_al", 2, "bird")]["mean_input"])
    got["S6b"] = dict(value=cost[("S_al", 2, "bird")]["mean_output"])
    got["S6c"] = dict(value=cost[("G", 2, "bird")]["mean_output"])
    cl = C.cluster_of(recs)
    sizes = [len({u for u in pops["full"]["qwen3_glm"] if u[0] == b}) for b in ("bird", "spider")] + \
            [len({u for u in pops["common"]["qwen3_glm"] if u[0] == b}) for b in ("bird", "spider")] + \
            [len({cl[u] for u in pops["full"]["qwen3_glm"] if u[0] == b}) for b in ("bird", "spider")]
    got["S7"] = dict(value=sizes)
    cost_rows = [dict(arm=op, decode=d, benchmark=b, population="common", pool="qwen3_glm",
                      weighting="database_equal", **{k: round(v, 2) for k, v in row.items()})
                 for (op, d, b), row in sorted(cost.items())]
    cost_rows.append(dict(arm="candidate_acquisition", decode=None, benchmark="both", population="common",
                          pool="qwen3_glm", weighting="shared prefix, reported once",
                          note="acquisition tokens are in data/derived/sql_pool_operation_transfer/output/acquisition.csv; not added to any arm"))
    return got, cost_rows, {}
