r"""Shared pieces for the text-to-SQL workload: schema rendering, read-only execution, scoring.

The hard boundary in this file is between what an agent may see and what only the evaluator may see.

A policy may see: the question, the database schema, the benchmark's own `evidence` hint, and the
result of executing **its own** candidate SQL against the database. That last item is genuine
execution feedback and is what makes this workload different from QA.

A policy may never see: the gold SQL, the gold result set, or any evaluator verdict. Those live in
`gold_sql()` and `execution_match()`, which the collectors do not call. Feedback is produced by
`execute()` from the candidate alone, so there is no path by which a correctness signal reaches a
prompt.

Execution is read-only and bounded. The connection is opened with `mode=ro` and `query_only=ON`, a
statement timeout is enforced through a progress handler, and only a bounded number of rows is ever
materialised. A query that errors, times out or returns nothing still produces feedback: the error
text or the empty-result fact is informative and is what a developer would see.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
BIRD = os.path.join(ROOT_RECORDS, "dataeng_raw", "sql", "dev_20240627")
DBDIR = os.path.join(BIRD, "dev_databases")

MAX_ROWS = 20              # bounded materialisation
TIMEOUT_S = 5.0            # per-statement wall clock
SAMPLE_ROWS_IN_FEEDBACK = 3


def load_questions() -> List[Dict[str, Any]]:
    with open(os.path.join(BIRD, "dev.json"), encoding="utf-8") as f:
        return json.load(f)


def db_path(db_id: str) -> str:
    return os.path.join(DBDIR, db_id, f"{db_id}.sqlite")


_SCHEMA_CACHE: Dict[str, str] = {}


def schema_text(db_id: str, max_cols: int = 40) -> str:
    """CREATE statements as the database itself reports them, lightly trimmed."""
    if db_id in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db_id]
    con = _connect(db_id)
    try:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    parts = []
    for name, sql in rows:
        s = re.sub(r"\s+", " ", sql or "").strip()
        if len(s) > 1200:
            s = s[:1200] + " ...)"
        parts.append(s)
    out = "\n".join(parts)
    _SCHEMA_CACHE[db_id] = out
    return out


def _connect(db_id: str) -> sqlite3.Connection:
    uri = f"file:{db_path(db_id)}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=TIMEOUT_S)
    con.execute("PRAGMA query_only=ON")
    return con


def execute(db_id: str, sql: str, max_rows: int = MAX_ROWS,
            timeout_s: float = TIMEOUT_S) -> Dict[str, Any]:
    """Run a candidate query read-only. Never consults gold.

    Returns a structured outcome: ok with columns, a bounded row sample and a row count, or an error
    with its message, or a timeout. All three are legitimate feedback.
    """
    con = _connect(db_id)
    deadline = time.time() + timeout_s
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 1000)
    t0 = time.time()
    try:
        cur = con.execute(sql)
        cols = [c[0] for c in (cur.description or [])]
        rows = cur.fetchmany(max_rows)
        extra = cur.fetchone() is not None
        return {"status": "ok", "columns": cols,
                "rows": [list(map(_coerce, r)) for r in rows],
                "n_rows_returned": len(rows), "more_rows_exist": extra,
                "elapsed_s": round(time.time() - t0, 3)}
    except Exception as exc:                                              # noqa: BLE001
        msg = str(exc)
        kind = "timeout" if "interrupted" in msg.lower() else "error"
        return {"status": kind, "error": msg[:300],
                "elapsed_s": round(time.time() - t0, 3)}
    finally:
        try:
            con.close()
        except Exception:
            pass


def _coerce(v: Any) -> Any:
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, float):
        return round(v, 6)
    return v


def feedback_text(res: Dict[str, Any]) -> str:
    """Execution feedback as a developer would see it. Contains no gold information."""
    if res["status"] == "ok":
        head = (f"query ran in {res['elapsed_s']}s; returned "
                f"{res['n_rows_returned']}{'+' if res['more_rows_exist'] else ''} row(s); "
                f"columns: {', '.join(map(str, res['columns'])) or '(none)'}")
        if res["n_rows_returned"] == 0:
            return head + "\n  result set is empty"
        sample = res["rows"][:SAMPLE_ROWS_IN_FEEDBACK]
        body = "\n".join("  " + ", ".join(str(x) for x in r) for r in sample)
        return head + "\n  first rows:\n" + body
    if res["status"] == "timeout":
        return f"query did not finish within {TIMEOUT_S}s and was cancelled"
    return f"query failed: {res['error']}"


# --------------------------------------------------------------------------- evaluator only
def gold_sql(q: Dict[str, Any]) -> str:
    """Evaluator-only. No collector or prompt in this study calls this."""
    return q["SQL"]


def _norm_result(rows: Sequence[Sequence[Any]]) -> set:
    return {tuple(_coerce(x) for x in r) for r in rows}


def execution_match(db_id: str, pred_sql: str, gold: str,
                    cap: int = 10000) -> Dict[str, Any]:
    """BIRD-style execution comparison, as a set of result tuples. Evaluator-only.

    Set comparison is the benchmark's own notion and it is not order-sensitive. Its limits are
    stated rather than assumed away: two different queries can coincide on one database instance,
    so a match is evidence on this instance, not proof of semantic equivalence.
    """
    pr = execute(db_id, pred_sql, max_rows=cap)
    if pr["status"] != "ok":
        return {"match": False, "reason": pr["status"], "pred_status": pr["status"]}
    gr = execute(db_id, gold, max_rows=cap)
    if gr["status"] != "ok":
        return {"match": False, "reason": "gold_failed", "pred_status": pr["status"]}
    same = _norm_result(pr["rows"]) == _norm_result(gr["rows"])
    return {"match": bool(same), "reason": "compared", "pred_status": "ok",
            "n_pred_rows": pr["n_rows_returned"], "n_gold_rows": gr["n_rows_returned"],
            "caveat": "single-instance set equality, not proven semantic equivalence"}


# --------------------------------------------------------------------------- prompting
PROMPT = (
    "You are writing one SQLite query.\n\n"
    "Database schema:\n{schema}\n\n"
    "Question: {question}\n"
    "{evidence}"
    "\nReturn only the SQL query, with no explanation and no code fence."
)


def render(q: Dict[str, Any]) -> str:
    ev = q.get("evidence") or ""
    return PROMPT.format(schema=schema_text(q["db_id"]), question=q["question"],
                         evidence=(f"Hint: {ev}\n" if ev.strip() else ""))


_FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.S | re.I)


def parse_sql(text: str) -> Optional[str]:
    """Extract one statement. Returns None when nothing SQL-like is present."""
    if not text:
        return None
    m = _FENCE.search(text)
    s = (m.group(1) if m else text).strip()
    s = re.sub(r"^(?:sql|sqlite)\s*[:\n]", "", s, flags=re.I).strip()
    i = s.lower().find("select")
    if i < 0:
        i = s.lower().find("with")
    if i < 0:
        return None
    s = s[i:].strip().rstrip(";").strip()
    return s or None
