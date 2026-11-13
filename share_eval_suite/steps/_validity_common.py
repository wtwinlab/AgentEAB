"""Shared inputs of the replay-validity steps: stored execution results, verdicts and reference queries.

Execution results (signatures) and correctness verdicts of logged queries are stored in JSON-lines
caches keyed by `query_key(database, query)`. The steps read them only; a query without a stored
result would have to be executed against the BIRD databases, which the steps do not do: they report
the number of such queries and stop.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, Mapping, Sequence

from share_eval_suite import paths

CACHE = paths.REPO / "data" / "caches" / "sql_execution"
OUTPUT = paths.REPO / "output" / "share_eval_suite"


class ExecutionNeeded(RuntimeError):
    """Some logged queries have no stored execution result or verdict."""

    def __init__(self, signatures: int, verdicts: int):
        super().__init__(
            f"{signatures} queries have no stored execution result and {verdicts} no stored verdict; "
            "they need read-only execution against the BIRD databases, which this step does not run")
        self.signatures = signatures
        self.verdicts = verdicts


def data_root() -> pathlib.Path:
    """Directory that holds `data/records/` (`ARTIFACT_ROOT`, else the repository root)."""
    return pathlib.Path(os.environ.get("ARTIFACT_ROOT") or paths.REPO)


def bird_reference_queries() -> Dict[str, str]:
    """Reference query of every BIRD development question, keyed `bird|<question_id>`."""
    p = data_root() / "data" / "records" / "dataeng_raw" / "sql" / "dev_20240627" / "dev.json"
    return {f"bird|{q['question_id']}": q["SQL"] for q in json.loads(p.read_text())}


def read_cache_lines(path: pathlib.Path):
    """Rows of one JSON-lines cache (nothing when the file is absent)."""
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def load_cache(paths_: Sequence[pathlib.Path]) -> Dict[str, Dict[str, Any]]:
    """Stored rows keyed by query key; a row in a later file replaces an earlier one."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in paths_:
        if p:
            for d in read_cache_lines(p):
                out[d["k"]] = d
    return out


def cache_paths(arg: str) -> list:
    """Comma-separated paths of a command-line option, empty entries dropped."""
    return [pathlib.Path(p) for p in arg.split(",") if p]


def require_cached(todo: Mapping, gtodo: Mapping) -> None:
    """Stop with `ExecutionNeeded` when some query has no stored signature or verdict."""
    if todo or gtodo:
        raise ExecutionNeeded(len(todo), len(gtodo))
