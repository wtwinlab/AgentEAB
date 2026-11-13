#!/usr/bin/env python3
"""BIRD SQL as a task family, with a question that leaves the output format to each protocol's own
output contract (benchmark id `sql_bird_v2`).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict

BENCH = "sql_bird_v2"
KIND = "sql"

_FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.S)


def unwrap_sql(raw: Any) -> str:
    """Protocol-agnostic unwrapping only: code fence, bare language tag, trailing semicolon,
whitespace. Case is preserved, so exact string identity stays meaningful."""
    t = str(raw or "").strip()
    if not t:
        return ""
    m = _FENCE.findall(t)
    if m:
        t = m[-1]
    t = t.strip()
    t = re.sub(r"^(?:sql|SQL)\s*[\n:]", "", t).strip()
    t = t.rstrip(";").strip()
    return " ".join(t.split())


def _boxed(text: Any) -> str:
    hits = re.findall(r"\\boxed\{([^{}]*)\}", str(text or ""))
    return hits[-1].strip() if hits else ""


def _after_answer(text: Any) -> str:
    t = str(text or "")
    return t.split("Answer:")[-1] if "Answer:" in t else t


def _identity(text: Any) -> str:
    return str(text or "")


def _wrap_kind(mod, fname: str, pre: Callable[[Any], str], none_on_empty: bool) -> None:
    orig = getattr(mod, fname)

    def wrapped(text, kind, *a, **kw):
        if kind == KIND:
            v = unwrap_sql(pre(text))
            return (v or None) if none_on_empty else v
        return orig(text, kind, *a, **kw)

    wrapped.__name__ = fname
    setattr(mod, fname, wrapped)


def install() -> Dict[str, Any]:
    """Register `sql_bird_v2` on the four workflows the cross-task goal names."""
    from consensus_delivery.adapters import bccs, dmad, mad_m2
    done = {}
    for mod, fname, pre, non in ((bccs, "extract_boxed", _boxed, False),
                                 (dmad, "normalise", _identity, False),
                                 (mad_m2, "extract_answer", _after_answer, True)):
        name = mod.__name__.rsplit(".", 1)[-1]
        if not getattr(mod, "_sqlv2_installed", False):
            _wrap_kind(mod, fname, pre, non)
            mod.ANSWER_KIND[BENCH] = KIND
            try:
                mod.ADAPTED.add(BENCH)
            except AttributeError:
                mod.ADAPTED = set(mod.ADAPTED) | {BENCH}
            mod._sqlv2_installed = True
        done[name] = {"extractor": fname, "pre": pre.__name__}
    return {"benchmark": BENCH, "answer_kind": KIND, "adapters_extended": done,
            "adapter_files_modified": False,
            "prompt_rule": ("the question states what the answer must be; each protocol's released "
                            "contract governs how it is emitted"),
            "supersedes": ("results/protocol_cross_task_2026-10-18 sql_bird, which kept the "
                           "format-dictating line. That run's records are preserved and are "
                           "never pooled with these.")}


def verify_signature_order() -> Dict[str, str]:
    """Each wrapper must match its adapter's real argument order. A swap is silent: the call falls
through to the released branch and every SQL string normalises to `""`."""
    import inspect
    from consensus_delivery.adapters import bccs, dmad, mad_m2
    got = {
        "bccs.extract_boxed": list(inspect.signature(bccs.extract_boxed).parameters)[:2],
        "dmad.normalise": list(inspect.signature(dmad.normalise).parameters)[:2],
        "mad_m2.extract_answer": list(inspect.signature(mad_m2.extract_answer).parameters)[:2],
    }
    for k, v in got.items():
        if len(v) < 2:
            raise AssertionError(f"{k} takes {v}, expected (text, kind)")
    return {k: ", ".join(v) for k, v in got.items()}


def sql_question(question: str, schema: str, evidence: str = "") -> str:
    """The NL question with its schema. No gold SQL, no execution result, and **no instruction
about where to put the answer** -- that is each protocol's own contract."""
    ev = f"\n\nAdditional context: {evidence}" if evidence else ""
    return ("Write a single SQL query that answers the question for the given database schema.\n\n"
            f"Schema:\n{schema}\n\nQuestion: {question}{ev}\n\n"
            "The answer is one SQL query for this schema.")


if __name__ == "__main__":
    assert unwrap_sql("```sql\nSELECT a FROM t;\n```") == "SELECT a FROM t"
    assert unwrap_sql("SELECT   a\n  FROM t ;") == "SELECT a FROM t"
    assert unwrap_sql("sql\nSELECT 2") == "SELECT 2"
    assert unwrap_sql("") == "" and unwrap_sql(None) == ""
    assert unwrap_sql("SELECT A FROM T") != unwrap_sql("select a from t")   # case preserved
    assert _boxed(r"x \boxed{SELECT a FROM t} y") == "SELECT a FROM t"
    assert _after_answer("blah Answer: SELECT 1").strip() == "SELECT 1"
    q = sql_question("How many?", "CREATE TABLE t(a int)")
    assert "Return only the SQL query." not in q      # no format-dictating line
    assert q.endswith("The answer is one SQL query for this schema.")
    print("sql_tasks self-tests: OK")
