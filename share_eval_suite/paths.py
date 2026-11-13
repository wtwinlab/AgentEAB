"""Locations of the shipped scored tables."""
from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
PAPER_OBJECTS = REPO / "paper_objects"


def inputs(obj: str) -> pathlib.Path:
    """The scored input tables of a paper object, `paper_objects/<obj>/inputs/`."""
    return PAPER_OBJECTS / obj / "inputs"
