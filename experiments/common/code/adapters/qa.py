"""QA adapter: regrade the recorded recipient texts with the existing grader (`w3_score.grade`)."""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

import core as C

SP = C.REPO / "experiments/15_stronger_peers"
W3 = C.REPO / "experiments/15_stronger_peers"
sys.path.insert(0, str(W3 / "code"))
import w3_score as S                                                        # noqa: E402

POOL = "qwen3_peers"


def _last(path):
    d = {}
    for r in map(json.loads, open(path)):
        k = r.get("unit_id") or (r["task"], r["seat"])
        if k not in d or r.get("status") == "ok":
            d[k] = r
    return d


def load(spec):
    peers = {}
    for r in _last(W3 / "raw/w3_peer_qwen3-32b.jsonl").values():
        peers[(r["task"], r["seat"])] = r
    recs = []
    units = set()
    pin = dict(output="peer_outputs", produced_at_step="peer_generation", producer="qwen3_32b_peers")
    for r in map(json.loads, open(SP / "data/no_peers_e1.jsonl")):
        u = (r["task"], r["seat"])
        units.add(u)
        st = dict(problem=True, peer_outputs=all((r["task"], j) in peers for j in (0, 1, 2) if j != r["seat"]) or None)
        recs.append(C.Record(op="no_peers", pool=POOL, unit=u, cluster=r["task"], repeat=None,
                             value=float(S.grade(r["benchmark"], r["answer_kind"], r["raw_reply"], r["task"])),
                             state=st, cost=dict(input=r["prompt_tokens"], output=r["completion_tokens"])))
    for r in _last(SP / "raw/recipient_qwen25-7b.jsonl").values():
        u = (r["task"], r["seat"])
        ok = r["status"] == "ok"
        v = float(S.grade(r["benchmark"], r["answer_kind"], r["final_text"] if ok else "", r["task"]))
        st = dict(problem=True, peer_outputs=all(p in peers for p in [(r["task"], j) for j in r["peer_seats"]]) or None)
        us = r.get("usage") or {}
        cost = dict(input=us.get("prompt_tokens") or 0, output=us.get("completion_tokens") or 0)
        for op in ([r["condition"]] + (["answer_only_identity"] if r["condition"] == "answer_only" else [])):
            recs.append(C.Record(op=op, pool=POOL, unit=u, cluster=r["task"], repeat=None, value=v,
                                 status="ok" if ok else "failed", state=st, inputs=[pin], cost=cost))
    return recs, {"full": {POOL: units}}


def check(spec, recs, pops):
    """Regraded outcomes must equal the stored per-unit outcomes."""
    stored = {(r["task"], int(r["seat"])): r for r in csv.DictReader(open(SP / "output/per_unit.csv"))}
    bad = sum(1 for r in recs if r.op in ("no_peers", "answer_only", "full")
              and int(stored[r.unit][f"y_{r.op}"]) != int(r.value))
    return [] if not bad else [C.Finding("REGRADE_MISMATCH", C.ERROR, "records", f"{bad} regraded outcomes differ from per_unit.csv")]


def analyze(spec, recs, pops):
    got = C.run_estimates(spec, recs, pops)
    y = {op: C.unit_values(recs, op, POOL) for op in ("no_peers", "answer_only", "full")}
    cl = C.cluster_of(recs)
    tr = C.transitions(y["no_peers"], y["answer_only"], cl)
    got["Q3a"], got["Q3b"], got["Q3c"] = dict(value=tr["recovered_pp"]), dict(value=tr["lost_pp"]), dict(value=tr["net_pp"])
    cost_rows = []
    for op, tid in (("no_peers", "Q4a"), ("answer_only", "Q4b"), ("full", "Q4c")):
        rr = [r for r in recs if r.op == op]
        mi, mo = float(np.mean([r.cost["input"] for r in rr])), float(np.mean([r.cost["output"] for r in rr]))
        got[tid] = dict(value=[round(mi, 2), round(mo, 2)])
        cost_rows.append(dict(component="recipient_call", operation=op, units=len(rr), mean_input=round(mi, 2), mean_output=round(mo, 2)))
    pr = list(_last(W3 / "raw/w3_peer_qwen3-32b.jsonl").values())
    mi = float(np.mean([(r.get("usage") or {}).get("prompt_tokens") or 0 for r in pr]))
    mo = float(np.mean([(r.get("usage") or {}).get("completion_tokens") or 0 for r in pr]))
    got["Q4d"] = dict(value=[round(mi, 2), round(mo, 2)])
    cost_rows.append(dict(component="peer_acquisition (shared prefix, reported once; not added to answers or full)",
                          operation="qwen3_32b_peers", units=len(pr), mean_input=round(mi, 2), mean_output=round(mo, 2)))
    return got, cost_rows, dict(transitions_answer_only_vs_no_peers=tr)
