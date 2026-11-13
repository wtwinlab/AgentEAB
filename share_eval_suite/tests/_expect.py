"""Comparison of a suite result with the value a paper object ships in `output/values.json`."""
from __future__ import annotations

import math

from share_eval_suite import _repo

_VALUES = {}


def expected(obj: str, key: str):
    if obj not in _VALUES:
        _VALUES[obj] = _repo.expected(obj)
    return _VALUES[obj][key]


def assert_reproduces(obj: str, key: str, got, rel: float = 1e-9, abs_: float = 1e-9):
    """`got` equals the shipped value of `key` (numbers within a float tolerance, others exactly)."""
    want = expected(obj, key)
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        assert math.isclose(float(got), float(want), rel_tol=rel, abs_tol=abs_), f"{obj}:{key}: {got} != {want}"
    else:
        assert got == want, f"{obj}:{key}: {got!r} != {want!r}"
