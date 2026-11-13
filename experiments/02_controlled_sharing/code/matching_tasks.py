#!/usr/bin/env python3
"""Register entity matching as a task family on all nine protocol adapters, without editing them.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

BENCH = "matching_pair"
KIND = "binary_match"
LABELS = ("match", "non-match")

#: Mirrors of each protocol's own strategyqa template, label set substituted. Kept together so the
#: mirroring rule is auditable in one place.
TEMPLATES = {
    "du_mad.FRAMING":
        "Can you answer the following match/non-match question? {q} Explain your reasoning. ",
    "du_mad.ANSWER_CLAUSE":
        "Your final answer should be a single word, match or non-match, in the form "
        "\\boxed{{answer}}, at the end of your response.",
    "du_mad.SUFFIX_FRAMING":
        "\n\n Using the solutions from other agents as additional information, can you provide "
        "your answer to the question? \n The original question is {q}. ",
    "debate_or_vote.SUFFIX":
        ' Make sure to state your final answer in curly brackets at the very end of your '
        'response, just like: "{final answer: match}".',
    "moa.ANSWER_INSTRUCTION":
        " State exactly 'match' or 'non-match' at the end.",
    "ford.INSTRUCTION":
        'Read the question and our opinions carefully to get the key points of the question then '
        'consider the context to determine if the two records really refer to the same '
        'real-world entity. The question is hypothetical, do not introduce any new situation '
        'into the question. Use your general knowledge and understanding to provide your '
        'response with "Answer: (match|non-match) is more plausible." and then briefly explain '
        'your reasoning, starting with "Explanation: ". Remember, your answer should be either '
        '"match" or "non-match".',
}

# ----------------------------------------------------------------- label normalisation
#: One ordered alternation, scanned left to right.
_TOKEN = re.compile(
    r"\b(?:non-?match|not\s+(?:a\s+)?match(?:es)?|no\s+match|do(?:es)?\s+not\s+match|"
    r"different|distinct|match(?:es)?|same)\b", re.I)
_NEGATIVE = ("non", "not", "no ", "doe", "different", "distinct")


def to_label(raw: Any) -> str:
    """`match` / `non-match`, or `""` when the text commits to neither.
    """
    t = str(raw if raw is not None else "").strip()
    if not t:
        return ""
    hits = _TOKEN.findall(t) or [m.group(0) for m in _TOKEN.finditer(t)]
    if not hits:
        return ""
    last = hits[-1].strip().lower()
    return "non-match" if last.startswith(_NEGATIVE) else "match"


# ------------------------------------------------- each protocol's own extraction, first
def _boxed(text: Any) -> str:
    hits = re.findall(r"\\boxed\{([^{}]*)\}", str(text or ""))
    return hits[-1].strip() if hits else ""


def _after_answer(text: Any) -> str:
    t = str(text or "")
    return t.split("Answer:")[-1] if "Answer:" in t else t


def _identity(text: Any) -> str:
    return str(text or "")


def _wrap_kind(mod, fname: str, pre: Callable[[Any], str], none_on_empty: bool) -> None:
    """Wrap `f(text, kind)` so `binary_match` runs `pre` then the label normalisation."""
    orig = getattr(mod, fname)

    def wrapped(text, kind, *a, **kw):
        if kind == KIND:
            v = to_label(pre(text))
            return (v or None) if none_on_empty else v
        return orig(text, kind, *a, **kw)

    wrapped.__name__ = fname
    wrapped.__wrapped_for__ = KIND
    setattr(mod, fname, wrapped)


def _wrap_benchmark(mod, fname: str, pre: Callable[[Any], str], none_on_empty: bool) -> None:
    """Wrap `f(benchmark, text, ...)` -- du_mad's shape, which keys on benchmark not kind."""
    orig = getattr(mod, fname)

    def wrapped(benchmark, text, *a, **kw):
        if benchmark == BENCH:
            v = to_label(pre(text))
            return (v or None) if none_on_empty else v
        return orig(benchmark, text, *a, **kw)

    wrapped.__name__ = fname
    wrapped.__wrapped_for__ = KIND
    setattr(mod, fname, wrapped)


def _wrap_answer_benchmark(mod, fname: str, pre: Callable[[Any], str],
                           none_on_empty: bool) -> None:
    """Wrap `f(answer, benchmark)` -- ReConcile's shape, whose argument order is the reverse of
du_mad's. Getting this backwards silently routes every call to the released branch, where the
numeric fallback returns `""` for every label."""
    orig = getattr(mod, fname)

    def wrapped(answer, benchmark, *a, **kw):
        if benchmark == BENCH:
            v = to_label(pre(answer))
            return (v or None) if none_on_empty else v
        return orig(answer, benchmark, *a, **kw)

    wrapped.__name__ = fname
    wrapped.__wrapped_for__ = KIND
    setattr(mod, fname, wrapped)


def _add(mod, attr: str, value: Any) -> None:
    cur = getattr(mod, attr, None)
    if isinstance(cur, dict):
        cur[BENCH] = value
    elif isinstance(cur, (set, frozenset)):
        setattr(mod, attr, set(cur) | {BENCH})
    elif cur is None:
        setattr(mod, attr, {BENCH: value} if value is not None else {BENCH})


# ------------------------------------------------------------------------------ install
def install() -> Dict[str, Any]:
    from consensus_delivery.adapters import (bccs, debate_or_vote, dmad, du_mad, ford, liang_mad,
                                             mad_m2, moa, reconcile)
    done: Dict[str, Any] = {}

    # --- protocols whose extraction keys on `kind`
    for mod, fname, pre, non in (
            (bccs, "extract_boxed", _boxed, False),          # BCCS reads its own \boxed{}
            (dmad, "normalise", _identity, False),           # DMAD normalises a tag-extracted answer
            (mad_m2, "extract_answer", _after_answer, True),  # MAD-M2 reads after "Answer:"
            (liang_mad, "normalise", _identity, False),      # Liang MAD parses a JSON field first
            (moa, "extract", _identity, False),              # MoA reads its aggregator's reply
    ):
        name = mod.__name__.rsplit(".", 1)[-1]
        if getattr(mod, "_matching_installed", False):
            continue
        _wrap_kind(mod, fname, pre, non)
        _add(mod, "ANSWER_KIND", KIND)
        _add(mod, "ADAPTED", None)
        mod._matching_installed = True
        done[name] = {"extractor": fname, "keyed_on": "kind"}

    # --- Debate-or-Vote: two parsers, and its own curly-bracket contract in SUFFIX
    if not getattr(debate_or_vote, "_matching_installed", False):
        for fname in ("parse_released", "parse_strict"):
            _wrap_kind(debate_or_vote, fname, debate_or_vote.bracketed, False)
        _add(debate_or_vote, "ANSWER_KIND", KIND)
        _add(debate_or_vote, "ADAPTED", None)
        _add(debate_or_vote, "SUFFIX", TEMPLATES["debate_or_vote.SUFFIX"])
        debate_or_vote._matching_installed = True
        done["debate_or_vote"] = {"extractor": "parse_released + parse_strict",
                                  "keyed_on": "kind", "pre": "bracketed"}

    # --- Du-MAD: extraction keys on benchmark; three prompt templates to mirror
    if not getattr(du_mad, "_matching_installed", False):
        _wrap_benchmark(du_mad, "extract_for", du_mad.boxed_group, True)
        _add(du_mad, "FRAMING", TEMPLATES["du_mad.FRAMING"])
        _add(du_mad, "ANSWER_CLAUSE", TEMPLATES["du_mad.ANSWER_CLAUSE"])
        _add(du_mad, "SUFFIX_FRAMING", TEMPLATES["du_mad.SUFFIX_FRAMING"])
        _add(du_mad, "ADAPTED", None)
        du_mad._matching_installed = True
        done["du_mad"] = {"extractor": "extract_for", "keyed_on": "benchmark",
                          "pre": "boxed_group"}

    # --- MoA: its answer instruction is appended to the reference prompt
    _add(moa, "ANSWER_INSTRUCTION", TEMPLATES["moa.ANSWER_INSTRUCTION"])

    # --- FORD: `instruction(benchmark)` is a function, and parse_prediction keys on benchmark
    if not getattr(ford, "_matching_installed", False):
        _orig_instruction = ford.instruction

        def instruction(benchmark, *a, **kw):
            return TEMPLATES["ford.INSTRUCTION"] if benchmark == BENCH else \
                _orig_instruction(benchmark, *a, **kw)
        ford.instruction = instruction

        _orig_parse = ford.parse_prediction

        def parse_prediction(text, benchmark, *a, **kw):
            if benchmark != BENCH:
                return _orig_parse(text, benchmark, *a, **kw)
            # upstream's shape kept: the answer lives after "Answer:", the rest is the explanation
            t = str(text or "").strip()
            try:
                after = t.split("Answer:")[1]
                prediction, explanation = after.split("Explanation: ")
                return (to_label(prediction) or "None"), explanation.strip()
            except Exception:                                 # noqa: BLE001 — upstream's bare except
                return "None", t
        ford.parse_prediction = parse_prediction
        _add(ford, "ADAPTED", None)
        ford._matching_installed = True
        done["ford"] = {"extractor": "parse_prediction + instruction",
                        "keyed_on": "benchmark"}

    # --- ReConcile: JSON answer field, and the convincing-sample directory
    if not getattr(reconcile, "_matching_installed", False):
        _wrap_answer_benchmark(reconcile, "normalise_answer", _identity, False)
        _add(reconcile, "ANSWER_KIND", KIND)
        # Its release ships convincing samples for Aqua, ECQA, GSM8k and SQA only.
        _add(reconcile, "DATASET_FOR", "SQA")
        reconcile._matching_installed = True
        done["reconcile"] = {"extractor": "normalise_answer", "keyed_on": "benchmark",
                             "convincing_samples": "SQA, reused as a labelled adaptation"}

    return {"benchmark": BENCH, "answer_kind": KIND, "labels": list(LABELS),
            "adapters_extended": done, "adapter_files_modified": False,
            "template_rule": ("each protocol's matching_pair template mirrors its own strategyqa "
                              "template with match/non-match for yes/no"),
            "prompt_rule": ("the question states the label set and not the output format; each "
                            "protocol's own contract governs emission"),
            "templates": TEMPLATES}


# ------------------------------------------------------------------- question construction
def matching_question(task: str, a: Dict[str, Any], b: Dict[str, Any], fields=None) -> str:
    """The pair as the agents see it. No gold, no entity id, no peer, no execution result.
    """
    def block(tag: str, r: Dict[str, Any]) -> str:
        keys = fields if fields else [k for k in r if r.get(k) not in (None, "")]
        lines = [f"  {k}: {r.get(k, '')}" for k in keys if r.get(k) not in (None, "")]
        return f"Record {tag}:\n" + "\n".join(lines)

    return (f"{task}\n\n{block('A', a)}\n\n{block('B', b)}\n\n"
            "The answer is either match or non-match.")


def verify_signature_order() -> Dict[str, str]:
    """Each wrapper must match its adapter's real argument order. A swap is silent, not loud: the
call simply falls through to the released branch and every label normalises to `""`."""
    import inspect
    from consensus_delivery.adapters import du_mad, ford, reconcile
    got = {
        "du_mad.extract_for": list(inspect.signature(du_mad.extract_for).parameters)[:2],
        "ford.parse_prediction": list(inspect.signature(ford.parse_prediction).parameters)[:2],
        "reconcile.normalise_answer":
            list(inspect.signature(reconcile.normalise_answer).parameters)[:2],
    }
    want = {"du_mad.extract_for": ["benchmark", "text"],
            "ford.parse_prediction": ["text", "benchmark"],
            "reconcile.normalise_answer": ["answer", "benchmark"]}
    for k, w in want.items():
        if got[k] != w:
            raise AssertionError(f"{k} argument order is {got[k]}, wrapper assumes {w}")
    return {k: ", ".join(v) for k, v in got.items()}


if __name__ == "__main__":
    # deterministic self-tests; no model, no network, no adapter import
    assert to_label("These are a match.") == "match"
    assert to_label("non-match") == "non-match"
    assert to_label("They are different products") == "non-match"
    assert to_label("not a match") == "non-match"
    assert to_label("no match") == "non-match"
    assert to_label("I cannot tell") == ""
    assert to_label("") == "" and to_label(None) == ""
    # a text that mentions both commits to whichever it states last
    assert to_label("a match? no, non-match") == "non-match"
    assert to_label("non-match at first, but actually a match") == "match"
    assert _boxed(r"x \boxed{non-match} y") == "non-match"
    assert _after_answer("blah Answer: match").strip() == "match"
    q = matching_question("Decide whether.", {"name": "a", "x": ""}, {"name": "b"})
    assert q.endswith("The answer is either match or non-match.")
    assert "Answer exactly one of" not in q          # no format-dictating line
    assert "  x: " not in q                          # empty fields are not rendered
    print("matching_tasks self-tests: OK")
