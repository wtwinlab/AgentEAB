"""Official-reference scorers for the frozen benchmark blocks.

Separate from the mechanisms and from the study's own metrics: these answer only "does the delivered
text match the benchmark's own reference, by the benchmark's own rule". A scorer that is assumed
rather than tested is how a parser error becomes a quality finding — a first pass at GSM8K reported
4 of 6 because the extractor did `"370".rstrip(".0")`, which is `"37"`.

`tests/test_official_scorers.py` holds the counterexamples: trailing zeros, a decimal that is really
an integer, zero itself, negatives, thousands separators, currency, bold markup, and GSM8K's own
`#### N` reference format.
"""
from __future__ import annotations

import re
from typing import Optional

#: A number with optional sign, optional thousands separators and an optional fractional part.
_NUM = re.compile(r"[-+]?\d(?:[\d,]*\d)?(?:\.\d+)?")

#: GSM8K's reference puts the rationale first and the answer after a `####` marker.
_GSM8K_GOLD = re.compile(r"####\s*([-+]?\d(?:[\d,]*\d)?(?:\.\d+)?)")


def _to_number(tok: str) -> Optional[float]:
    """Parse one numeric token. Never strips characters from the token."""
    try:
        return float(tok.replace(",", "").replace("+", ""))
    except ValueError:
        return None


def last_number(text: object) -> Optional[float]:
    """The final numeric token in the text, as a number.

    The benchmark's convention is that the answer comes last, so this takes the last token and parses
    it. It does **not** strip trailing characters: `rstrip(".0")` turns 370 into 37, 300 into 3, 0 into
    nothing and 10.50 into 10.5 — four different wrong answers from one line.
    """
    if text is None:
        return None
    s = str(text).replace("$", "").replace("*", "").replace("−", "-")
    toks = _NUM.findall(s)
    for tok in reversed(toks):
        n = _to_number(tok)
        if n is not None:
            return n
    return None


def gsm8k_gold(reference: object) -> Optional[float]:
    """The gold answer from a GSM8K reference string, by its own `#### N` convention.

    Falls back to the last number only when the marker is absent, and says so by returning `None`
    rather than guessing when the reference is empty — an empty reference is a data issue
    (`SQuAD2.py`'s no-answer items produce exactly that) and must not score as 0.
    """
    if reference is None:
        return None
    s = str(reference)
    m = _GSM8K_GOLD.search(s)
    if m:
        return _to_number(m.group(1))
    return None


def gsm8k_correct(delivered: object, reference: object, *, tol: float = 1e-6) -> Optional[bool]:
    """`True`/`False`, or **`None` when the comparison is not defined**.

    `None` is not a failure: an absent reference or an answer with no number in it is an unscorable
    cell, and folding either into `False` would charge a data issue or an empty output to the
    method's accuracy. The caller keeps them in the denominator as unscorable.
    """
    gold = gsm8k_gold(reference)
    pred = last_number(delivered)
    if gold is None or pred is None:
        return None
    return abs(gold - pred) <= tol


# --------------------------------------------------------------------- multiple choice
#: A standalone **uppercase** option letter, optionally wrapped. `A` in "A. 90954 m" counts; the `A`
#: in "Avogadro" does not, which is why a bare `[A-J]` search is wrong.
#:
#: Two further exclusions, both found by an independent recomputation disagreeing with this function
#: on a real GPQA answer. That answer was truncated mid-formula and stated no option letter at all:
#:
#:   * **LaTeX and subscript context.** The text contained `\rangle_{F=1}`, and a
#:     letter-only pattern matches that `F`; with ten options it would score the answer as `F`. A letter preceded by
#:     `\ { _ ^` or followed by `=` is part of a formula, not an answer.
#:   * **Case.** The independent implementation folded case and matched the English article "a" in
#:     "In a standard …", picking `A`. An article is not an option letter. Only uppercase counts.
_LETTER = re.compile(r"(?<![A-Za-z\\{_^])\(?([A-J])\)?(?![A-Za-z]|\s*=)")


def letter_answer(text: object, n_options: int = 10) -> Optional[str]:
    """The final standalone option letter within range, or `None`.

    Takes the **last** such letter: a reply that reasons through options and then states one ends with
    its answer. Letters beyond `n_options` are ignored — MMLU-Pro is *not* uniformly ten options
    (the frozen block holds 5, 8, 9 and 10), so `J` is a valid answer on one task and out of range on
    another.
    """
    if text is None:
        return None
    allowed = {chr(ord("A") + i) for i in range(max(1, min(10, n_options)))}
    found = [m.group(1) for m in _LETTER.finditer(str(text)) if m.group(1) in allowed]
    return found[-1] if found else None


def choice_correct(delivered: object, gold_letter: object, n_options: int = 10) -> Optional[bool]:
    pred = letter_answer(delivered, n_options)
    gold = str(gold_letter).strip().upper() if gold_letter is not None else None
    if pred is None or not gold or gold not in {chr(ord("A") + i) for i in range(10)}:
        return None
    return pred == gold


# --------------------------------------------------------------------- boolean
_YES = ("yes", "true", "correct")
_NO = ("no", "false", "incorrect")


def boolean_answer(text: object) -> Optional[bool]:
    """The last standalone yes/no token. `None` when the reply commits to neither."""
    if text is None:
        return None
    toks = re.findall(r"[A-Za-z]+", str(text).lower())
    for tok in reversed(toks):
        if tok in _YES:
            return True
        if tok in _NO:
            return False
    return None


def strategyqa_correct(delivered: object, reference: object) -> Optional[bool]:
    """BIG-bench StrategyQA scores a boolean; the gold is the target with score 1."""
    pred = boolean_answer(delivered)
    g = str(reference).strip().lower()
    gold = True if g in _YES else False if g in _NO else None
    if pred is None or gold is None:
        return None
    return pred is gold


# --------------------------------------------------------------------- extractive span
def _squad_norm(s: str) -> str:
    """SQuAD's own normalisation: lowercase, strip articles, punctuation and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def squad_v2_scores(delivered: object, references: object, *, unanswerable: bool
                    ) -> Optional[dict]:
    """Exact match and token F1 under SQuAD 2.0's rules, with no-answer handled explicitly.

    An unanswerable question is answered correctly by **abstaining**, so the prediction is scored
    against "no answer" rather than against a span. Folding an unanswerable item into a span
    comparison is the issue `SQuAD2.py:33` produces upstream, where `references` becomes `[]` and is
    indistinguishable from a load failure.
    """
    if delivered is None:
        return None
    pred = str(delivered).strip()
    said_none = _squad_norm(pred) in ("", "no answer", "unanswerable", "none",
                                      "not answerable", "cannot be answered")
    if unanswerable:
        return {"exact": float(said_none), "f1": float(said_none), "mode": "no-answer"}
    golds = [str(x) for x in (references or []) if str(x) != "<unanswerable>"]
    if not golds:
        return None
    pn = _squad_norm(pred)
    best = {"exact": 0.0, "f1": 0.0, "mode": "span"}
    for g in golds:
        gn = _squad_norm(g)
        exact = float(pn == gn)
        pt, gt = pn.split(), gn.split()
        common = 0
        gt_pool = list(gt)
        for t in pt:
            if t in gt_pool:
                gt_pool.remove(t)
                common += 1
        f1 = 0.0 if not common else (2 * (common / len(pt)) * (common / len(gt))
                                     / ((common / len(pt)) + (common / len(gt))))
        if (exact, f1) > (best["exact"], best["f1"]):
            best = {"exact": exact, "f1": f1, "mode": "span"}
    return best
