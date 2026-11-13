#!/usr/bin/env python3
r"""An exact reference splitter for MoA, replacing a prompt-level regex that is not decodable.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import arm_transform as AT

#: reference replies observed in the current unit-run, in call order.
REF_TEXTS: List[str] = []
#: per-call record of which reading was used, drained into the unit-run's trace
SPLIT_MODE: List[Dict[str, Any]] = []

_RELEASED_MOA_REFS = AT.moa_refs
_NUMBERED = re.compile(r"\n(\d+)\. ")


def reset() -> None:
    REF_TEXTS.clear()
    SPLIT_MODE.clear()


def observe_reply(text: str) -> None:
    REF_TEXTS.append(text if isinstance(text, str) else str(text or ""))


def exact_refs(text: str, refs: Sequence[str]) -> Optional[List[Tuple[int, int, str, str]]]:
    """`[(start, end, number, body)]` when `text` provably ends with the rendered `refs`, else None.
    """
    i = text.find(AT.MOA_REFS_HEADER)
    if i < 0 or not refs:
        return None
    base = i + len(AT.MOA_REFS_HEADER)
    rendered = "".join("\n%d. %s" % (n + 1, o) for n, o in enumerate(refs))
    if text[base:] != rendered:
        return None
    out: List[Tuple[int, int, str, str]] = []
    pos = base
    for n, o in enumerate(refs):
        delim = "\n%d. " % (n + 1)
        start = pos
        body_start = pos + len(delim)
        out.append((start, body_start + len(o), str(n + 1), o))
        pos = body_start + len(o)
    return out


def moa_refs(text: str) -> List[Tuple[int, int, str, str]]:
    """Exact when the reference texts explain the prompt; else released."""
    got = exact_refs(text, REF_TEXTS)
    if got is not None:
        SPLIT_MODE.append({"mode": "exact", "n_refs": len(got)})
        return got
    fallback = _RELEASED_MOA_REFS(text)
    SPLIT_MODE.append({"mode": "released_regex_fallback", "n_refs": len(fallback),
                       "n_observed_replies": len(REF_TEXTS),
                       "why": "the aggregator prompt is not the concatenation of the replies this "
                              "process observed, so the boundaries are not known"})
    return fallback


_INSTALLED = False


def install() -> Dict[str, Any]:
    """Rebind `AT.moa_refs`. Idempotent; safe for every protocol, since only MoA calls it."""
    global _INSTALLED
    if not _INSTALLED:
        AT.moa_refs = moa_refs
        _INSTALLED = True
    return {"moa_refs": "ext_arms.moa_refs (exact split by observed reference replies, released "
                        "regex as a recorded fallback)",
            "replaces": "arm_transform.moa_refs",
            "reason": "MoA's reference delimiter '\\n<n>. ' is byte-identical to a model's own "
                      "numbered step, so the released regex counted 735 references where 3 were "
                      "rendered on 100 real GPQA aggregator prompts"}


def split_summary() -> Dict[str, Any]:
    modes: Dict[str, int] = {}
    for m in SPLIT_MODE:
        modes[m["mode"]] = modes.get(m["mode"], 0) + 1
    return {"calls": len(SPLIT_MODE), "modes": modes}


# ---------------------------------------------------------------- offline measurement
def measure_inflation(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Released regex vs exact split, on stored MoA records. Used by the acceptance test."""
    reg = ex = calls = agg = 0
    mismatch = 0
    for r in records:
        refs = [c.get("raw_reply") or "" for c in (r.get("calls") or [])
                if c.get("role") == "reference"]
        for c in r.get("calls") or []:
            if c.get("role") != "aggregator":
                continue
            agg += 1
            for m in c.get("messages_sent") or []:
                t = m.get("content") or ""
                if AT.MOA_REFS_HEADER not in t:
                    continue
                calls += 1
                reg += len(_RELEASED_MOA_REFS(t))
                got = exact_refs(t, refs)
                if got is None:
                    mismatch += 1
                else:
                    ex += len(got)
    return {"aggregator_calls": agg, "prompts_with_header": calls,
            "released_regex_references": reg, "exact_references": ex,
            "prompts_the_exact_split_could_not_explain": mismatch,
            "inflation_factor": round(reg / ex, 3) if ex else None}
