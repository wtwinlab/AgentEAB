r"""Spider 1.0 adapter for P4, mirroring `sql_common.py` so BIRD and Spider are comparable.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
# Delivered layout: the question file ships with the artifact; the databases are a separate
# third-party download, see data/external/README.md.
SRC = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data/records/dataeng_raw/spider")
DB_EXTERNAL = os.path.join(os.environ.get("SPIDER_DB_DIR") or SRC, "database")
DBDIR = DB_EXTERNAL

sys.path.insert(0, os.path.join(ROOT, "experiments/06_cost_resources/code"))
import sql_common as S                   # noqa: E402


def db_path(db_id: str) -> str:
    return os.path.join(DBDIR, db_id, f"{db_id}.sqlite")


# `sql_common.execute` resolves its file through `sql_common.db_path`, which points at BIRD.
_BIRD_DB_PATH = S.db_path
S.db_path = db_path
S._SCHEMA_CACHE.clear()                  # BIRD's cached schemas must not enter Spider prompts
REDIRECTED = {"sql_common.db_path": "spider_common.db_path",
              "bird_resolver_kept_as": "_BIRD_DB_PATH",
              "reason": "share one execution implementation without editing sql_common"}

# one execution regime for both benchmarks
execute = S.execute
feedback_text = S.feedback_text
parse_sql = S.parse_sql
MAX_ROWS = S.MAX_ROWS
TIMEOUT_S = S.TIMEOUT_S
SAMPLE_ROWS_IN_FEEDBACK = S.SAMPLE_ROWS_IN_FEEDBACK

PROVENANCE = {
    "benchmark": "Spider 1.0",
    "paper": "Yu et al., Spider: A Large-Scale Human-Labeled Dataset for Complex and "
             "Cross-Domain Semantic Parsing and Text-to-SQL Task, EMNLP 2018",
    "release_file": "spider_data.zip",
    "release_sha256": "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b",
    "release_bytes": 205800266,
    "source_url": "https://drive.usercontent.google.com/download?id="
                  "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t",
    "linked_from": "https://yale-lily.github.io/spider",
    "licence": "CC BY-SA 4.0",
    "downloaded_utc": "2026-10-15",
    "readme_notes": "release README records 2020 corrections: ~40 dev.json annotation fixes and a "
                    "column-name fix in the scholar and formula_1 schemas",
    "split_used": "dev (1,034 questions, 20 databases), public gold SQL",
    "split_not_used": "test.json / test_gold.sql are present in this release and are public, but "
                      "are not used: dev is the comparable split and mixing both would mix two "
                      "protocols in one number",
    "evaluator": "execution accuracy by result-set equality, via sql_common.execution_match — the "
                 "identical rule applied to BIRD",
    "no_evidence_field": "Spider questions carry no hint/evidence string, unlike BIRD. The prompt "
                         "renders without a hint rather than inventing one.",
    "excluded_alternative": "Spider 2.0 — interactive/agentic, excluded by the plan as not a "
                            "drop-in replacement for fixed-query feedback",
}


def load_questions() -> List[Dict[str, Any]]:
    """Dev questions, normalised to the key names `collect_sql_arms.py` already uses."""
    with open(os.path.join(SRC, "dev.json"), encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for i, q in enumerate(raw):
        out.append({"question_id": i, "db_id": q["db_id"], "question": q["question"],
                    "SQL": q["query"], "evidence": "", "difficulty": "unlabelled"})
    return out


_SCHEMA_CACHE: Dict[str, str] = {}


def schema_text(db_id: str, max_cols: int = 40) -> str:
    """CREATE-statement schema read from the live database, the same way BIRD's is.
    """
    if db_id in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db_id]
    con = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True, timeout=TIMEOUT_S)
    try:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE "
            "'sqlite_%' ORDER BY name").fetchall()
        parts = []
        for name, ddl in rows:
            cols = con.execute(f'PRAGMA table_info("{name}")').fetchall()
            keep = cols[:max_cols]
            cl = ", ".join(f"{c[1]} {c[2]}" for c in keep)
            if len(cols) > max_cols:
                cl += f", ... ({len(cols) - max_cols} more columns)"
            parts.append(f"TABLE {name} ({cl})")
        txt = "\n".join(parts)
    finally:
        con.close()
    _SCHEMA_CACHE[db_id] = txt
    return txt


def render(q: Dict[str, Any]) -> str:
    """Identical prompt structure to BIRD's, minus the hint line Spider has no data for."""
    return S.PROMPT.format(schema=schema_text(q["db_id"]), question=q["question"], evidence="")


def execution_match(db_id: str, pred_sql: str, gold: str, cap: int = 10000) -> Dict[str, Any]:
    return S.execution_match(db_id, pred_sql, gold, cap=cap)


def database_hashes() -> Dict[str, Dict[str, Any]]:
    """Per-database size and hash, so a later run can prove it used the same files."""
    out = {}
    for q in load_questions():
        d = q["db_id"]
        if d in out:
            continue
        p = db_path(d)
        out[d] = {"bytes": os.path.getsize(p),
                  "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    return out


# --------------------------------------------------------------------------- acceptance
def _acceptance() -> int:
    """Protocol acceptance BEFORE any large collection, as the plan requires."""
    qs = load_questions()
    dbs = sorted({q["db_id"] for q in qs})
    print(f"=== Spider dev: {len(qs)} questions, {len(dbs)} databases ===")
    fail = 0

    missing = [d for d in dbs if not os.path.exists(db_path(d))]
    print(f"  databases present: {len(dbs) - len(missing)}/{len(dbs)}"
          + (f"  MISSING {missing}" if missing else ""))
    fail += bool(missing)

    # every gold query must run, or the question cannot be scored at all
    bad_gold, empty_gold = [], 0
    for q in qs:
        r = execute(q["db_id"], q["SQL"])
        if r["status"] != "ok":
            bad_gold.append((q["question_id"], q["db_id"], r.get("error", r["status"])[:80]))
        elif r["n_rows_returned"] == 0:
            empty_gold += 1
    print(f"  gold queries that fail to execute: {len(bad_gold)} of {len(qs)}")
    for b in bad_gold[:5]:
        print(f"    q{b[0]} [{b[1]}] {b[2]}")
    print(f"  gold queries returning zero rows: {empty_gold} "
          f"({100*empty_gold/len(qs):.1f}%) — legitimate, but they make a wrong query easy to "
          f"match, so they are reported as a stratum rather than dropped")

    # the evaluator must score gold against itself as correct
    self_match = sum(1 for q in qs[:200]
                     if execution_match(q["db_id"], q["SQL"], q["SQL"])["match"])
    print(f"  evaluator sanity, gold vs gold on the first 200: {self_match}/200 match")
    fail += self_match != 200

    # a deliberately wrong query must not match
    wrong = execution_match(qs[0]["db_id"], "SELECT 1", qs[0]["SQL"])
    print(f"  evaluator sanity, 'SELECT 1' vs gold: match={wrong['match']} (must be False)")
    fail += bool(wrong["match"])

    # schema rendering and prompt
    print(f"\n  schema sample [{qs[0]['db_id']}]:\n    "
          + schema_text(qs[0]["db_id"])[:300].replace("\n", "\n    "))
    pr = render(qs[0])
    print(f"\n  prompt chars {len(pr)}; contains no hint line: {'Hint:' not in pr}")
    print(f"  prompt tail: ...{pr[-90:]!r}")
    fail += "Hint:" in pr

    # the gold SQL must never be reachable from a prompt
    n_exposed = sum(1 for q in qs[:300] if q["SQL"].strip().lower() in render(q).lower())
    print(f"  gold SQL appearing in its own prompt, first 300: {n_exposed} (must be 0)")
    fail += bool(n_exposed)

    print(f"\n  ACCEPTANCE {'PASSED' if not fail else 'FAILED'} ({fail} problem(s))")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_acceptance())
