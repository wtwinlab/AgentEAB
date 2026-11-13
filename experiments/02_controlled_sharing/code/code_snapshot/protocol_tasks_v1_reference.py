#!/usr/bin/env python3
"""Task-family extension: run the frozen BCCS / DMAD / MAD-M2 adapters on matching and SQL.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

BENCH_MATCHING = "matching_pair"
BENCH_SQL = "sql_bird"
NEW_KINDS = {BENCH_MATCHING: "binary_match", BENCH_SQL: "sql"}

_FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.S)
_MATCH_TOKENS = re.compile(r"\b(non-?match|not\s+a?\s*match|different|match|same)\b", re.I)


def unwrap_sql(raw: str) -> str:
    """Protocol-agnostic unwrapping only: code fence, trailing semicolon, whitespace."""
    t = (raw or "").strip()
    if not t:
        return ""
    m = _FENCE.findall(t)
    if m:
        t = m[-1]
    t = t.strip()
    # a bare leading language tag some models emit without a fence
    t = re.sub(r"^(?:sql|SQL)\s*[\n:]", "", t).strip()
    t = t.rstrip(";").strip()
    t = " ".join(t.split())
    return t


def extract_binary_match(raw: str) -> str:
    """`match` / `non-match`, or '' when the text states neither."""
    t = (raw or "").strip()
    if not t:
        return ""
    m = _MATCH_TOKENS.search(t)
    if not m:
        return ""
    tok = m.group(1).lower().replace(" ", "").replace("-", "")
    if tok in ("nonmatch", "notamatch", "notmatch", "different"):
        return "non-match"
    if tok in ("match", "same"):
        return "match"
    return ""


def _dispatch(kind: str, raw: str) -> Optional[str]:
    if kind == "binary_match":
        return extract_binary_match(raw)
    if kind == "sql":
        return unwrap_sql(raw)
    return None


def _boxed(text: str) -> str:
    """BCCS's own protocol-level extraction, reproduced so the new kinds respect it."""
    hits = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    return hits[-1].strip() if hits else ""


def _after_answer(text: str) -> str:
    """MAD-M2's own protocol-level extraction: the tail after `Answer:`."""
    t = text or ""
    return t.split("Answer:")[-1] if "Answer:" in t else t


# Each adapter's PROTOCOL-level extraction must run first; only the KIND normalisation is new.
PRE = {"extract_boxed": _boxed, "extract_answer": _after_answer, "normalise": lambda t: t or ""}


def extend(mod, extractor_name: str, none_on_empty: bool) -> None:
    """Register the new kinds on one frozen adapter module without editing its file."""
    if getattr(mod, "_cross_task_extended", False):
        return
    orig = getattr(mod, extractor_name)
    pre = PRE[extractor_name]

    def wrapped(text, kind, *a, **kw):
        if kind in ("binary_match", "sql"):
            v = _dispatch(kind, pre(text))
            if none_on_empty:
                return v or None
            return v or ""
        return orig(text, kind, *a, **kw)

    wrapped.__name__ = extractor_name
    wrapped.__doc__ = (orig.__doc__ or "") + "\n\nExtended for binary_match and sql kinds."
    setattr(mod, extractor_name, wrapped)
    mod.ANSWER_KIND.update(NEW_KINDS)
    try:
        mod.ADAPTED.update(NEW_KINDS.keys())
    except AttributeError:
        mod.ADAPTED = set(mod.ADAPTED) | set(NEW_KINDS.keys())
    mod._cross_task_extended = True


def install() -> Dict[str, Any]:
    """Extend all three adapters. Returns what was changed, for the manifest."""
    from consensus_delivery.adapters import bccs, dmad, mad_m2
    # dmad normalises an already-tag-extracted answer; bccs reads a \boxed{}; mad_m2 reads
    # after 'Answer:'. Each keeps its own protocol-level extraction; only the KIND layer is new.
    extend(dmad, "normalise", none_on_empty=False)
    extend(bccs, "extract_boxed", none_on_empty=False)
    extend(mad_m2, "extract_answer", none_on_empty=True)
    return {"adapters_extended": ["bccs", "dmad", "mad_m2"],
            "new_answer_kinds": NEW_KINDS,
            "adapter_files_modified": False,
            "marked_task_adaptation": sorted(NEW_KINDS.keys())}


# ------------------------------------------------------------------ question construction
def matching_question(a: Dict[str, str], b: Dict[str, str], fields) -> str:
    """The two records as the protocol's agents see them. No gold, no peer, no execution."""
    def block(tag, r):
        lines = [f"  {f}: {r.get(f, '')}" for f in fields]
        return f"Record {tag}:\n" + "\n".join(lines)
    return ("Decide whether the two records below describe the same real-world entity.\n\n"
            + block("A", a) + "\n\n" + block("B", b)
            + "\n\nAnswer exactly one of: match | non-match")


def sql_question(question: str, schema: str, evidence: str = "") -> str:
    """The NL question with its schema. No gold SQL, no execution result."""
    ev = f"\n\nAdditional context: {evidence}" if evidence else ""
    return ("Write a single SQL query that answers the question for the given database schema.\n\n"
            f"Schema:\n{schema}\n\nQuestion: {question}{ev}\n\n"
            "Return only the SQL query.")


if __name__ == "__main__":
    # deterministic self-tests; no model, no network
    assert unwrap_sql("```sql\nSELECT a FROM t;\n```") == "SELECT a FROM t"
    assert unwrap_sql("SELECT   a\n  FROM t ;") == "SELECT a FROM t"
    assert unwrap_sql("```\nSELECT 1\n```") == "SELECT 1"
    assert unwrap_sql("") == ""
    assert unwrap_sql("sql\nSELECT 2") == "SELECT 2"
    assert extract_binary_match("These are a match.") == "match"
    assert extract_binary_match("non-match") == "non-match"
    assert extract_binary_match("They are different products") == "non-match"
    assert extract_binary_match("I cannot tell") == ""
    assert extract_binary_match("") == ""
    # case preserved, whitespace normalised -> exact-string identity is meaningful
    assert unwrap_sql("SELECT A FROM T") != unwrap_sql("select a from t")
    # protocol-level extraction must run before the kind layer
    assert _boxed("x \\boxed{SELECT a FROM t} y") == "SELECT a FROM t"
    assert _boxed("no box here") == ""
    assert _after_answer("blah Answer: match").strip() == "match"
    print("protocol_tasks self-tests: OK")
