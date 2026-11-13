"""CHESS adapter (commit 3d6e835f): boundary branches from per-question execution logs.

Input: the `<question_id>_<db_id>.json` execution-history files CHESS's Logger writes
(`runner/logger.py:108-118`), a BIRD-format question file with gold SQL (scoring only), and the
databases. The official repository ships no such logs; until a real run exists this adapter is
exercised only on fixtures marked synthetic.

Boundary B1 (after the Candidate Generator, before the Unit Tester): the candidate list the Unit
Tester reads is `list(state.SQL_meta_infos.keys())[-1]` (generate_unit_test.py:28). In the log it
is the `candidates` of the `generate_unit_test` record (generate_unit_test.py:125-129); it must
equal the `candidates` of the last `revise` / `generate_candidate` record before it, else the
boundary state is flagged. Earlier candidate lists are historical and are not merged.

Branches on the same list: CHESS's own self-consistency (evaluate.py:106-114, clustering as
evaluate.py:185-202), Valid and Nonempty in list order, and the logged UT selection
(`selected_candidate`, evaluate.py:204-212). Candidates are executed read-only (`mode=ro`);
CHESS itself opens the database writable (execution.py:29), so a non-SELECT candidate is flagged.
Scoring follows CHESS `compare_sqls`: set equality of execution results, predicted SQL cleaned as
`_clean_sql` (execution.py:56-66, 166-214).
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import threading

import core as C

UT_TOOLS = ("generate_unit_test", "evaluate")
CG_TOOLS = ("generate_candidate", "revise")


def _clean_sql(sql):
    return sql.replace("\n", " ").replace('"', "'").strip("`.")


def execute_ro(db_path, sql, timeout=60):
    """(status, rows): status ok | error | timeout. Read-only connection."""
    box = {}

    def run():
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
            try:
                box["rows"] = con.execute(sql).fetchall()
            finally:
                con.close()
        except Exception as e:                                         # noqa: BLE001
            box["error"] = str(e)
    t = threading.Thread(target=run, daemon=True)
    t.start(); t.join(timeout)
    if t.is_alive():
        return "timeout", []
    if "error" in box:
        return "error", None
    return "ok", box["rows"]


def _cands(rec):
    out = []
    for c in rec.get("candidates") or []:
        out.append(c if isinstance(c, str) else (c.get("SQL") or c.get("refined_query")))
    return out


def parse_history(hist):
    """Boundary state and control-flow facts from one execution_history list."""
    names = [h.get("tool_name") for h in hist if isinstance(h, dict) and "tool_name" in h]
    first_ut = next((i for i, h in enumerate(hist) if h.get("tool_name") in UT_TOOLS), None)
    flags = []
    pre = [h for h in hist[:first_ut] if h.get("tool_name") in CG_TOOLS] if first_ut is not None else []
    last_cg = _cands(pre[-1]) if pre else None
    gut = next((h for h in hist if h.get("tool_name") == "generate_unit_test"), None)
    ev = [h for h in hist if h.get("tool_name") == "evaluate"]
    boundary = None
    if gut is not None and _cands(gut):
        boundary = _cands(gut)
        if last_cg is not None and last_cg != boundary:
            flags.append("BOUNDARY_STATE_MISMATCH")
    elif last_cg is not None:
        boundary = last_cg                    # generate_unit_test returned early (<= 1 candidate or 1 cluster)
    if first_ut is None:
        flags.append("NO_UNIT_TESTER_RECORD")
    ut_seq = [n for n in names if n in UT_TOOLS]
    if ut_seq != ["generate_unit_test", "evaluate"]:
        flags.append(f"CONTROL_FLOW_DEVIATION:{'>'.join(ut_seq)}")
    if any(n in CG_TOOLS for n in names[first_ut or 0:]) and first_ut is not None:
        flags.append("CANDIDATES_CHANGED_AFTER_BOUNDARY")
    sel = ev[-1].get("selected_candidate") if ev else None
    return dict(boundary=boundary, ut_selected=sel, unit_tests=(gut or {}).get("unit_tests"),
                n_evaluate_records=len(ev), flags=flags, tool_sequence=names)


def self_consistency(cands, results):
    """CHESS Evaluate.self_consistency over Evaluate.execution_based_clustering."""
    clusters = {}
    for sql, (st, rows) in zip(cands, results):
        if st == "error":
            continue                          # execution_result raises -> skipped (evaluate.py:194-197)
        key = repr(rows if st == "ok" else [])
        clusters.setdefault(key, []).append(sql)
    if not clusters:
        return None
    largest = max(clusters, key=lambda k: len(clusters[k]))
    return clusters[largest][0]


def first_valid(cands, results, nonempty):
    for sql, (st, rows) in zip(cands, results):
        if st == "ok" and (rows or not nonempty):
            return sql
    return None


def score(db, sql, gold):
    if not sql:
        return 0
    st, pr = execute_ro(db, _clean_sql(sql))
    sg, gr = execute_ro(db, gold)
    if st != "ok" or sg != "ok":
        return 0
    return int(set(pr) == set(gr))


def load(spec):
    a = spec.get("analysis", {})
    if not a.get("log_dir"):
        raise C.SpecError([C.Finding("NO_RECORDS", C.UNSUPPORTED, "analysis.log_dir",
                                     "no CHESS execution logs are available; the official repository ships none")])
    root = C.REPO if not pathlib.Path(a["log_dir"]).is_absolute() else pathlib.Path("/")
    logs = pathlib.Path(a["log_dir"]) if pathlib.Path(a["log_dir"]).is_absolute() else root / a["log_dir"]
    tasks = {int(t["question_id"]): t for t in json.loads(pathlib.Path(a["dataset"]).read_text())}
    db_root = pathlib.Path(a["db_root"])
    recs, units = [], set()
    _CACHE["facts"] = {}
    for p in sorted(logs.glob("*.json")):
        if p.name.startswith("-"):
            continue
        qid = int(p.stem.split("_", 1)[0])
        t = tasks[qid]
        hist = json.loads(p.read_text())
        f = parse_history(hist)
        db = db_root / t["db_id"] / f"{t['db_id']}.sqlite"
        cands = f["boundary"] or []
        res = [execute_ro(db, c) for c in cands]
        writes = sum(1 for c in cands if not c.lstrip().lower().startswith(("select", "with")))
        picks = dict(UT_full=f["ut_selected"], UT_identity=f["ut_selected"], self_consistency=self_consistency(cands, res),
                     Valid=first_valid(cands, res, False), Nonempty=first_valid(cands, res, True))
        u = (qid,)
        units.add(u)
        st = dict(boundary_candidates=cands or None, question=t.get("question"), hint=t.get("evidence", ""), db_id=t["db_id"])
        for op, sql in picks.items():
            recs.append(C.Record(op=op, pool="chess", unit=u, cluster=t["db_id"], repeat=None,
                                 value=float(score(db, sql, t["SQL"])), status="ok" if sql else "unresolved", state=st,
                                 inputs=[dict(output="boundary_candidates", produced_at_step="revision", producer="candidate_generator")]))
        recs.append(C.Record(op="test_best", pool="chess", unit=u, cluster=t["db_id"], repeat=None,
                             value=float(max([score(db, c, t["SQL"]) for c in cands] or [0])), state=st))
        _CACHE["facts"][qid] = dict(flags=f["flags"], tool_sequence=f["tool_sequence"], n_candidates=len(cands),
                                    non_select_candidates=writes, picks=picks)
    return recs, {"full": {"chess": units}}


_CACHE = {}


def check(spec, recs, pops):
    out = []
    for qid, f in _CACHE.get("facts", {}).items():
        for fl in f["flags"]:
            sev = C.ERROR if fl in ("BOUNDARY_STATE_MISMATCH", "CANDIDATES_CHANGED_AFTER_BOUNDARY") else C.UNSUPPORTED
            out.append(C.Finding(fl.split(":")[0], sev, f"question {qid}", fl))
        if f["non_select_candidates"]:
            out.append(C.Finding("WRITE_STATEMENT", C.INFO, f"question {qid}",
                                 f"{f['non_select_candidates']} non-SELECT candidate(s): CHESS executes them on a writable connection, this adapter read-only"))
    return out


def analyze(spec, recs, pops):
    got = {}
    for op in ("UT_full", "self_consistency", "Valid", "Nonempty", "UT_identity", "test_best"):
        y = C.unit_values(recs, op, "chess")
        got[f"level_{op}"] = dict(value=100 * sum(y.values()) / max(1, len(y)), n=len(y))
    return got, [], dict(facts=_CACHE.get("facts"))
