r"""Offline B1 extraction (formal copy: --calls for per-shard proxy logs, --results-prefix f_), branches, scoring and cost accounting. No model call.

    python3 extract_b1.py --base ${CHESS_WORKDIR} --out out/

For every frozen question (all 10 stay in the denominator):
  * reads CHESS's per-question execution history, its stdout, and the proxy call log;
  * candidate lists with producer and dependency; B1 = the last generate_candidate/revise list
    before the first Unit Tester record; UT-visible list = generate_unit_test.candidates (fallback
    to B1 when that record is empty, flagged); named flags;
  * executes each B1 candidate once, read-only, 60 s timeout; every rule reads those results;
  * branches: UT_full, SC_official, Valid_vote, Nonempty_vote, Unfiltered_vote (diag),
    First_executable_diag, First_nonempty_diag, Any_correct (oracle reference);
  * scores with CHESS compare_sqls semantics (set equality, predicted SQL cleaned); gold is read
    only here, after every branch has picked.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import pathlib
import sqlite3
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b1_selectors as B1                                                   # noqa: E402  (copied read-only, hash logged)

UT_TOOLS = ("generate_unit_test", "evaluate")
CG_TOOLS = ("generate_candidate", "revise")


def execute_ro(db, sql, timeout=60):
    box = {}

    def run():
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
            try:
                box["rows"] = con.execute(sql).fetchall()
            finally:
                con.close()
        except Exception as e:                                         # noqa: BLE001
            box["error"] = f"{type(e).__name__}: {e}"
    t0 = time.time()
    t = threading.Thread(target=run, daemon=True)
    t.start(); t.join(timeout)
    dt = time.time() - t0
    if t.is_alive():
        return dict(status="timeout", rows=[], error="timeout", seconds=dt)
    if "error" in box:
        return dict(status="error", rows=None, error=box["error"], seconds=dt)
    return dict(status="ok", rows=box["rows"], error=None, seconds=dt)


def clean_sql(sql):
    return sql.replace("\n", " ").replace('"', "'").strip("`.")


def cands_of(rec):
    out = []
    for c in rec.get("candidates") or []:
        out.append(c if isinstance(c, str) else (c.get("SQL") or c.get("refined_query")))
    return out


def sha(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False).encode()).hexdigest()


def sc_official(cands, ex):
    """CHESS Evaluate.self_consistency over Evaluate.execution_based_clustering (evaluate.py:106-114, 185-202).
    repr of the result list; an execution that raises is skipped; a timeout yields [] (sql_meta_info.py:30-32)."""
    clusters = {}
    for i, (sql, e) in enumerate(zip(cands, ex)):
        if e["status"] == "error":
            continue
        key = repr(e["rows"] if e["status"] == "ok" else [])
        clusters.setdefault(key, []).append(i)
    if not clusters:
        return None
    largest = max(clusters, key=lambda k: len(clusters[k]))
    return clusters[largest][0]


def vote_pick(cands, ex, which):
    live = [dict(sql=s, model=f"c{i}", idx=i,
                 exec=dict(status=("ok" if e["status"] == "ok" else e["status"]), rows=e["rows"] or [],
                           n_rows_returned=len(e["rows"] or []) if e["status"] == "ok" else 0))
            for i, (s, e) in enumerate(zip(cands, ex)) if s]
    if not live:
        return None
    fn = dict(Unfiltered_vote=B1.sel_released, Valid_vote=B1.sel_valid_majority,
              Nonempty_vote=B1.sel_nonempty_majority)[which]
    return fn(live)["idx"]


def first(cands, ex, nonempty):
    for i, e in enumerate(ex):
        if e["status"] == "ok" and (e["rows"] or not nonempty):
            return i
    return None


def score(db, sql, gold_rows):
    if not sql or gold_rows is None:
        return 0
    e = execute_ro(db, clean_sql(sql))
    return int(e["status"] == "ok" and set(e["rows"]) == set(gold_rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="R1:")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--calls", nargs="+", default=None, help="proxy logs (formal: one per shard)")
    ap.add_argument("--results-prefix", default="q_")
    a = ap.parse_args()
    B = pathlib.Path(a.base)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(pathlib.Path(a.tasks or (B / "run/pilot_tasks.json")).read_text())
    calls = [json.loads(l) for f in (a.calls or [B / "raw/proxy_calls.jsonl"]) for l in open(f)]
    perq, brows, cost_rows, exrows = [], [], [], []
    for t in tasks:
        qid, dbid = t["question_id"], t["db_id"]
        db = B / f"data/dev_databases/{dbid}/{dbid}.sqlite"
        rec = dict(question_id=qid, db_id=dbid, flags=[])
        hist_files = sorted(glob.glob(str(B / f"chess/results/dev/CHESS_IR_CG_UT/{a.results_prefix}{qid}/*/{qid}_{dbid}.json")))
        qcalls = [c for c in calls if c.get("question") == f"{a.tag}q{qid}"]
        rec["calls"] = sum(not c["refused"] for c in qcalls)
        rec["refused_calls"] = sum(c["refused"] for c in qcalls)
        stdout = B / f"logs/chess_q{qid}.out"
        rec["stdout_tail"] = stdout.read_text(errors="replace")[-1500:] if stdout.exists() else None
        if not hist_files:
            rec["status"] = "TASK_FAILED"; rec["flags"].append("TASK_FAILED:no execution history")
            perq.append(rec); continue
        hist = json.loads(pathlib.Path(hist_files[-1]).read_text())
        rec["history_file"] = hist_files[-1].replace(str(B) + "/", "")
        tools = [h.get("tool_name") for h in hist if isinstance(h, dict) and "tool_name" in h]
        rec["tool_sequence"] = tools
        rec["tool_errors"] = [dict(tool=h.get("tool_name"), error=h.get("error")) for h in hist
                              if isinstance(h, dict) and h.get("status") == "error"]
        lists = []
        for i, h in enumerate(hist):
            if isinstance(h, dict) and h.get("tool_name") in CG_TOOLS:
                c = cands_of(h)
                lists.append(dict(record=i, tool=h["tool_name"], n=len(c), sha256=sha(c),
                                  original_SQL_id=h.get("original_SQL_id"), refined_SQL_id=h.get("refined_SQL_id"), sqls=c))
        rec["candidate_lists"] = lists
        first_ut = next((i for i, h in enumerate(hist) if isinstance(h, dict) and h.get("tool_name") in UT_TOOLS), None)
        pre = [l for l in lists if first_ut is None or l["record"] < first_ut]
        post = [l for l in lists if first_ut is not None and l["record"] > first_ut]
        b1 = pre[-1]["sqls"] if pre else []
        rec["b1"] = dict(source_record=pre[-1]["record"] if pre else None, source_tool=pre[-1]["tool"] if pre else None,
                         n=len(b1), sha256=sha(b1), sqls=b1)
        gut = next((h for h in hist if isinstance(h, dict) and h.get("tool_name") == "generate_unit_test"), None)
        evs = [h for h in hist if isinstance(h, dict) and h.get("tool_name") == "evaluate"]
        vis = cands_of(gut) if gut else []
        if gut is not None and vis:
            rec["ut_visible"] = dict(source="generate_unit_test.candidates", n=len(vis), sha256=sha(vis))
            if vis != b1:
                rec["flags"].append("BOUNDARY_STATE_MISMATCH")
        else:
            rec["ut_visible"] = dict(source="fallback: last CG list (generate_unit_test record empty or absent)",
                                     n=len(b1), sha256=sha(b1))
            if gut is not None:
                rec["flags"].append("UT_VISIBLE_FROM_FALLBACK")
        ut_seq = [n for n in tools if n in UT_TOOLS]
        if not ut_seq:
            rec["flags"].append("UT_NOT_RUN")
        elif ut_seq != ["generate_unit_test", "evaluate"]:
            rec["flags"].append("CONTROL_FLOW_DEVIATION:" + ">".join(ut_seq))
        if post:
            rec["flags"].append("CANDIDATES_CHANGED_AFTER_BOUNDARY")
        if any(e["tool"] in UT_TOOLS for e in rec["tool_errors"]):
            rec["flags"].append("UT_FAILED")
        rec["unit_tests"] = (gut or {}).get("unit_tests")
        rec["evaluate"] = [dict(scores=e.get("scores"), n_candidates=len(cands_of(e)), selected=e.get("selected_candidate"))
                           for e in evs]
        ut_sel = evs[-1].get("selected_candidate") if evs else None
        ev = next((h for h in reversed(hist) if isinstance(h, dict) and h.get("tool_name") in ("execution_accuracy", "evaluation")), None)  # Tool.tool_name = camel_to_snake("ExecutionAccuracy")
        fin = (ev or {}).get("final_SQL") if ev else None
        keys = [k for k in (ev or {}) if isinstance((ev or {}).get(k), dict) and k != "final_SQL"]
        rec["evaluation_keys"] = keys
        rec["final_key"] = keys[-1] if keys else None
        rec["final_sql"] = (fin or {}).get("PREDICTED_SQL")
        if rec["final_key"] and not rec["final_key"].startswith("evaluate"):
            rec["flags"].append("FINAL_NOT_UT")
        writes = sum(1 for c in b1 if c and not c.lstrip().lower().startswith(("select", "with")))
        if writes:
            rec["flags"].append(f"WRITE_STATEMENT:{writes}")
        # executions (shared by every rule)
        t0 = time.time()
        ex = [execute_ro(db, c) if c else dict(status="error", rows=None, error="empty", seconds=0.0) for c in b1]
        rec["rule_execution_seconds"] = round(time.time() - t0, 3)
        for i, (c, e) in enumerate(zip(b1, ex)):
            exrows.append(dict(question_id=qid, idx=i, status=e["status"], n_rows=len(e["rows"] or []) if e["status"] == "ok" else None,
                               result_sha=sha(sorted(map(str, e["rows"]))) if e["status"] == "ok" else None,
                               seconds=round(e["seconds"], 4), error=e["error"], sql=c))
        picks = dict(UT_full=ut_sel, UT_identity=ut_sel)
        for name, idx in (("SC_official", sc_official(b1, ex)), ("Valid_vote", vote_pick(b1, ex, "Valid_vote")),
                          ("Nonempty_vote", vote_pick(b1, ex, "Nonempty_vote")),
                          ("Unfiltered_vote", vote_pick(b1, ex, "Unfiltered_vote")),
                          ("First_executable_diag", first(b1, ex, False)), ("First_nonempty_diag", first(b1, ex, True))):
            picks[name] = b1[idx] if idx is not None else None
        rec["picks"] = picks
        rec["ut_selected_in_b1"] = (ut_sel in b1) if ut_sel else None
        # scoring: gold read only now
        g = execute_ro(db, t["SQL"])
        rec["gold_status"] = g["status"]
        gold_rows = g["rows"] if g["status"] == "ok" else None
        sc = {k: score(db, v, gold_rows) for k, v in picks.items()}
        sc["Any_correct"] = int(any(score(db, c, gold_rows) for c in b1)) if b1 else 0
        sc["CHESS_final_SQL"] = score(db, rec["final_sql"], gold_rows)
        rec["scores"] = sc
        rec["status"] = "COMPLETED" if ("UT_NOT_RUN" not in rec["flags"] and ut_sel) else "UT_NOT_RETURNED"
        for k, v in sc.items():
            brows.append(dict(question_id=qid, db_id=dbid, branch=k, correct=v, sql=picks.get(k)))
        perq.append(rec)
    # costs from the proxy log
    ut_start = {}
    for c in calls:
        if c.get("tool") in ("agent_tool_choice:unit_tester",) + UT_TOOLS and not c["refused"]:
            ut_start.setdefault(c["question"], c["ts_start"])
    agg = collections.defaultdict(lambda: collections.Counter())
    for c in calls:
        q = c.get("question") or ""
        if c["refused"]:
            agg[(q, "refused")]["calls"] += 1
            continue
        part = "probe" if q.endswith("probe") else ("ut_increment" if q in ut_start and c["ts_start"] >= ut_start[q] else "prefix")
        u = c.get("usage") or {}
        for key in ((q, part), (q, "tool:" + c["tool"])):
            a_ = agg[key]
            a_["calls"] += 1; a_["ok"] += c.get("status") == 200
            a_["prompt_tokens"] += u.get("prompt_tokens") or 0; a_["completion_tokens"] += u.get("completion_tokens") or 0
            a_["latency_s"] += c.get("latency_s") or 0
    for (q, part), v in sorted(agg.items()):
        cost_rows.append(dict(question=q, part=part, **{k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()}))
    (out / "per_question.json").write_text(json.dumps(perq, indent=1, default=str) + "\n")
    for name, rows in (("branches.csv", brows), ("executions.csv", exrows), ("costs.csv", cost_rows)):
        if rows:
            with open(out / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list({k: None for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    print(json.dumps([dict(q=r["question_id"], status=r["status"], flags=r["flags"], calls=r["calls"],
                           b1=r.get("b1", {}).get("n"), scores=r.get("scores")) for r in perq], indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
