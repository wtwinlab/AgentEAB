"""Adapter for the existing share_eval CLI: real CHESS pilot logs, after `extract_b1.py`."""
from __future__ import annotations

import json

import core as C

OPS = ("UT_full", "SC_official", "Valid_vote", "Nonempty_vote", "Unfiltered_vote",
       "First_executable_diag", "First_nonempty_diag", "UT_identity", "Any_correct")
_FACTS = {}


def load(spec):
    perq = json.loads((C.REPO / spec["analysis"]["per_question"]).read_text())
    recs, units = [], set()
    for q in perq:
        u = (q["question_id"],)
        units.add(u)
        _FACTS[q["question_id"]] = q
        b1 = (q.get("b1") or {}).get("sqls") or None
        st = dict(boundary_candidates=b1, question=True, hint=True, db_id=q["db_id"])
        inp = [dict(output="boundary_candidates", produced_at_step="revision", producer="candidate_generator")]
        for op in OPS:
            sc = (q.get("scores") or {}).get(op)
            pick = (q.get("picks") or {}).get(op) if op != "Any_correct" else True
            status = "ok" if sc is not None and pick else ("failed" if q.get("status") == "TASK_FAILED" else "unresolved")
            recs.append(C.Record(op=op, pool="chess", unit=u, cluster=q["db_id"], repeat=None,
                                 value=float(sc) if sc is not None else None, status=status, state=st, inputs=inp))
    return recs, {"full": {"chess": units}}


def check(spec, recs, pops):
    out = []
    for qid, q in _FACTS.items():
        for fl in q.get("flags", []):
            code = fl.split(":")[0]
            sev = C.ERROR if code in ("BOUNDARY_STATE_MISMATCH", "CANDIDATES_CHANGED_AFTER_BOUNDARY") else (
                C.INFO if code == "WRITE_STATEMENT" else C.UNSUPPORTED)
            out.append(C.Finding(code, sev, f"question {qid}", fl))
        if q.get("picks") and q["picks"].get("UT_full") and q.get("ut_selected_in_b1") is False:
            out.append(C.Finding("UT_PICK_NOT_IN_B1", C.ERROR, f"question {qid}", "UT selection is not a B1 candidate"))
    return out


def analyze(spec, recs, pops):
    got = {}
    for op in OPS:
        y = C.unit_values(recs, op, "chess")
        got[op] = dict(correct=int(sum(y.values())), of=len(y))
    ut, idn = C.unit_values(recs, "UT_full", "chess"), C.unit_values(recs, "UT_identity", "chess")
    got["no_change_identity_equal"] = dict(value=[sum(ut[u] == idn[u] for u in ut), len(ut)])
    return got, [], {}
