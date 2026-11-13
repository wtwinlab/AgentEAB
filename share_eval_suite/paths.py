from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
PAPER_OBJECTS = REPO / "paper_objects"


def inputs(obj: str) -> pathlib.Path:
    return PAPER_OBJECTS / obj / "inputs"
