"""Shared pieces of the replay steps: question databases, stored execution results, read-only
execution of BIRD queries, and the checksum file of an output directory.

Execution results of logged queries are stored in JSON-lines caches keyed by
`query_key(database, query)`:

- a signature row holds the execution status and, for a query that ran, the first 20 hex digits of
  the sha256 of its sorted, normalised result rows (`status`, `sig`, `s` seconds);
- a verdict row holds whether the query returns the reference result (`correct`, `outcome`,
  `queries`, `seconds`): `match`, `mismatch`, an execution failure class, or `gold_failed` when the
  reference query itself fails.

A query without a stored result is executed read-only against
`<data root>/data/records/dataeng_raw/sql/dev_20240627/dev_databases/<db>/<db>.sqlite` (data root:
`ARTIFACT_ROOT`, else the repository root), with a 5-second statement limit and at most 10,000 rows.
When the database file is absent the connection fails and the query is recorded as an execution
error (signature status `error`; verdict `other_exec_error`).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import time
from typing import Any, Dict, Sequence, Tuple

from share_eval_suite import paths
from share_eval_suite.records.readers import query_key

CACHE = paths.REPO / "data" / "caches" / "sql_execution"
OUTPUT = paths.REPO / "output" / "share_eval_suite"
BIRD_DEV = paths.REPO / "data" / "records" / "dataeng_raw" / "sql" / "dev_20240627" / "dev.json"

ROW_CAP = 10000                 # rows materialised per query
TIMEOUT_S = 5.0                 # statement time limit

_MISSING = re.compile(r"no such (table|column|function)", re.I)
_SYNTAX = re.compile(r"(syntax error|incomplete input|unrecognized token)", re.I)
_TYPE = re.compile(r"(datatype mismatch|wrong number of arguments|misuse of aggregate|"
                   r"ambiguous column name|too many terms)", re.I)


# ------------------------------------------------------------------ questions and outputs
def question_databases(path) -> Dict[str, str]:
    """Database of each BIRD question (`bird|<id>` -> database), from the BIRD dev.json or from a CSV
    with columns unit, db_id."""
    p = pathlib.Path(path)
    if p.suffix == ".csv":
        with open(p, encoding="utf-8") as fh:
            return {r["unit"]: r["db_id"] for r in csv.DictReader(fh)}
    return {f"bird|{q['question_id']}": q["db_id"] for q in json.loads(p.read_text())}


def write_checksums(out: pathlib.Path) -> None:
    """`<out>/SHA256SUMS`: the sha256 of every other file in `out`, in name order."""
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS"}
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in hashes.items()))


def load_jsonl_cache(cache_paths: Sequence[pathlib.Path]) -> Dict[str, Dict[str, Any]]:
    """Rows of the given JSON-lines caches keyed by `k`; a later row of the same key replaces an earlier."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in cache_paths:
        if p and p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    out[d["k"]] = d
    return out


# ------------------------------------------------------------------ read-only execution
def data_root() -> str:
    return os.environ.get("ARTIFACT_ROOT") or str(paths.REPO)


def bird_dir() -> str:
    return os.path.join(data_root(), "data", "records", "dataeng_raw", "sql", "dev_20240627")


def reference_queries() -> Dict[str, str]:
    """Reference query of every BIRD development question, keyed `bird|<question_id>`."""
    p = pathlib.Path(bird_dir()) / "dev.json"
    return {f"bird|{q['question_id']}": q["SQL"] for q in json.loads(p.read_text())}


def _connect(db_id: str) -> sqlite3.Connection:
    db = os.path.join(bird_dir(), "dev_databases", db_id, f"{db_id}.sqlite")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=TIMEOUT_S)
    con.execute("PRAGMA query_only=ON")
    return con


def _coerce(v: Any) -> Any:
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, float):
        return round(v, 6)
    return v


def _norm_result(rows) -> set:
    """A result as a set of tuples (row order and duplicates ignored)."""
    return {tuple(_coerce(x) for x in r) for r in rows}


def execute(db_id: str, sql: str, max_rows: int = ROW_CAP, timeout_s: float = TIMEOUT_S) -> Dict[str, Any]:
    """Run a query read-only: `ok` with its first `max_rows` rows, `error` with the message, or
    `timeout`. A database that cannot be opened raises."""
    con = _connect(db_id)
    deadline = time.time() + timeout_s
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 1000)
    t0 = time.time()
    try:
        cur = con.execute(sql)
        cols = [c[0] for c in (cur.description or [])]
        rows = cur.fetchmany(max_rows)
        extra = cur.fetchone() is not None
        return {"status": "ok", "columns": cols, "rows": [list(map(_coerce, r)) for r in rows],
                "n_rows_returned": len(rows), "more_rows_exist": extra,
                "elapsed_s": round(time.time() - t0, 3)}
    except Exception as exc:                                              # noqa: BLE001
        msg = str(exc)
        kind = "timeout" if "interrupted" in msg.lower() else "error"
        return {"status": kind, "error": msg[:300], "elapsed_s": round(time.time() - t0, 3)}
    finally:
        try:
            con.close()
        except Exception:                                                 # noqa: BLE001
            pass


def classify_error(msg: str) -> str:
    """Failure class of an execution error message."""
    m = msg or ""
    if _SYNTAX.search(m):
        return "syntax_error"
    if _MISSING.search(m):
        return "missing_object"
    if _TYPE.search(m):
        return "type_or_arg_error"
    return "other_exec_error"


def signature(args: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
    """(key, signature row) of one query."""
    db, sql = args
    t0 = time.time()
    try:
        r = execute(db, sql, max_rows=ROW_CAP)
    except Exception as exc:                                              # noqa: BLE001
        return query_key(db, sql), {"status": "error", "error": str(exc)[:120], "s": 0.0}
    if r["status"] != "ok":
        return query_key(db, sql), {"status": r["status"], "s": round(time.time() - t0, 3)}
    s = sorted(repr(t) for t in _norm_result(r["rows"]))
    return query_key(db, sql), {"status": "ok",
                                "sig": hashlib.sha256("\n".join(s).encode()).hexdigest()[:20],
                                "s": round(time.time() - t0, 3)}


def verdict(args: Tuple[str, str, str]) -> Tuple[str, Dict[str, Any]]:
    """(key, verdict row) of one query against the reference query of its question."""
    db, pred, gold = args
    try:
        t0 = time.time()
        pr = execute(db, pred, max_rows=ROW_CAP)
        if pr["status"] != "ok":
            return query_key(db, pred), {"correct": False,
                                         "outcome": ("exec_timeout" if pr["status"] == "timeout"
                                                     else classify_error(pr.get("error") or "")),
                                         "queries": 1, "seconds": round(time.time() - t0, 3)}
        gr = execute(db, gold, max_rows=ROW_CAP)
        if gr["status"] != "ok":
            return query_key(db, pred), {"correct": None, "outcome": "gold_failed", "queries": 2,
                                         "seconds": round(time.time() - t0, 3)}
        same = _norm_result(pr["rows"]) == _norm_result(gr["rows"])
        return query_key(db, pred), {"correct": bool(same), "outcome": "match" if same else "mismatch",
                                     "queries": 2, "seconds": round(time.time() - t0, 3)}
    except Exception as exc:                                              # noqa: BLE001
        return query_key(db, pred), {"correct": False, "outcome": "other_exec_error",
                                     "error": f"{type(exc).__name__}: {exc}"[:200], "queries": 0,
                                     "seconds": 0.0}
