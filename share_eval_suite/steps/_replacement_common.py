from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import sqlite3
import threading
import time
from typing import List, Optional, Sequence

from share_eval_suite import paths

REPO = paths.REPO
OUT = REPO / "output/share_eval_suite/stage_replacement"
EXT = REPO / "data/records/macsql_comem_stages"
CHESS = REPO / "data/records/chess_unit_tester"
BIRD = REPO / "data/records/dataeng_raw/sql/dev_20240627"
SPIDER = REPO / "data/records/dataeng_raw/spider"
SPIDER_DB = REPO / "data/external/spider_data/database"


def parse_args(description: str, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    bird_db = BIRD / "dev_databases"
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", default=str(OUT), help="output directory (default %(default)s)")
    ap.add_argument("--external-run", default=str(EXT),
                    help="MAC-SQL and ComEM run directory with output/, data/llm4em/, raw/ and logs/ (default %(default)s)")
    ap.add_argument("--chess-run", default=str(CHESS),
                    help="CHESS run directory with output/, data/, raw/ and logs/ (default %(default)s)")
    ap.add_argument("--bird", default=str(BIRD), help="BIRD development release with dev.json (default %(default)s)")
    ap.add_argument("--bird-db", default=os.environ.get("BIRD_DB_DIR") or str(bird_db),
                    help="BIRD dev_databases directory (default $BIRD_DB_DIR or <bird>/dev_databases)")
    ap.add_argument("--spider", default=str(SPIDER), help="Spider directory with dev.json (default %(default)s)")
    ap.add_argument("--spider-db", default=os.environ.get("SPIDER_DB_DIR") or str(SPIDER_DB),
                    help="Spider database directory (default $SPIDER_DB_DIR or %(default)s)")
    return ap.parse_args(argv)


def write_table(path: pathlib.Path, rows: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list({k: None for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)


def read_table(path: pathlib.Path) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def write_json(path: pathlib.Path, obj, **kw) -> None:
    path.write_text(json.dumps(obj, indent=1, **kw) + "\n")


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def coerce(v):
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, float):
        return round(v, 6)
    return v


NO_OUTPUT = dict(status="none", rows=None)


def macsql_execute(dbroot, db, sql, cap=10000, limit=30.0):
    if not sql or not isinstance(sql, str):
        return dict(status="none", rows=None, seconds=0.0)
    con = sqlite3.connect(f"file:{dbroot}/{db}/{db}.sqlite?mode=ro", uri=True, timeout=limit)
    con.execute("PRAGMA query_only=ON")
    con.text_factory = lambda b: b.decode(errors="ignore")
    t0 = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - t0 > limit else 0, 1000)
    try:
        rows = con.execute(sql).fetchmany(cap)
        return dict(status="ok", rows=rows, seconds=round(time.time() - t0, 4))
    except Exception as e:
        return dict(status="timeout" if "interrupted" in str(e).lower() else "error", rows=None,
                    error=str(e)[:200], seconds=round(time.time() - t0, 4))
    finally:
        con.close()


def macsql_class(ex):
    return ("OK", frozenset(tuple(coerce(x) for x in r) for r in ex["rows"])) if ex["status"] == "ok" else ("FAIL",)


def chess_clean(sql: str) -> str:
    return sql.replace("\n", " ").replace('"', "'").strip("`.")


def chess_execute(db, sql, timeout=60):
    box = {}

    def run():
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
            try:
                box["rows"] = con.execute(sql).fetchall()
            finally:
                con.close()
        except Exception as e:
            box["error"] = f"{type(e).__name__}: {e}"
    t0 = time.time()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return dict(status="timeout", rows=None, seconds=time.time() - t0)
    if "error" in box:
        return dict(status="error", rows=None, error=box["error"][:200], seconds=time.time() - t0)
    return dict(status="ok", rows=box["rows"], seconds=time.time() - t0)


def chess_class(ex):
    return ("OK", frozenset(ex["rows"])) if ex["status"] == "ok" else ("FAIL",)


def sqlcommon_class(ex, cap=10000):
    return ("OK", frozenset(tuple(coerce(x) for x in r) for r in ex["rows"][:cap])) if ex["status"] == "ok" else ("FAIL",)


def class_hash(c) -> str:
    return "FAIL" if c[0] == "FAIL" else hashlib.sha256(json.dumps(sorted(map(repr, c[1]))).encode()).hexdigest()[:16]
