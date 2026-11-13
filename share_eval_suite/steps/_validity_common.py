from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, Mapping, Sequence

from share_eval_suite import paths

CACHE = paths.REPO / "data" / "caches" / "sql_execution"
OUTPUT = paths.REPO / "output" / "share_eval_suite"


class ExecutionNeeded(RuntimeError):

    def __init__(self, signatures: int, verdicts: int):
        super().__init__(
            f"{signatures} queries have no stored execution result and {verdicts} no stored verdict; "
            "they need read-only execution against the BIRD databases, which this step does not run")
        self.signatures = signatures
        self.verdicts = verdicts


def data_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("ARTIFACT_ROOT") or paths.REPO)


def bird_reference_queries() -> Dict[str, str]:
    p = data_root() / "data" / "records" / "dataeng_raw" / "sql" / "dev_20240627" / "dev.json"
    return {f"bird|{q['question_id']}": q["SQL"] for q in json.loads(p.read_text())}


def read_cache_lines(path: pathlib.Path):
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def load_cache(paths_: Sequence[pathlib.Path]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in paths_:
        if p:
            for d in read_cache_lines(p):
                out[d["k"]] = d
    return out


def cache_paths(arg: str) -> list:
    return [pathlib.Path(p) for p in arg.split(",") if p]


def require_cached(todo: Mapping, gtodo: Mapping) -> None:
    if todo or gtodo:
        raise ExecutionNeeded(len(todo), len(gtodo))
