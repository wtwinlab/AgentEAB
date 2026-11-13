"""Database of each BIRD question, read from the BIRD dev.json or from a CSV with columns unit, db_id."""
from __future__ import annotations

import csv
import json
import pathlib


def db_of(path) -> dict:
    p = pathlib.Path(path)
    if p.suffix == ".csv":
        return {r["unit"]: r["db_id"] for r in csv.DictReader(open(p, encoding="utf-8"))}
    return {f"bird|{q['question_id']}": q["db_id"] for q in json.loads(p.read_text())}
