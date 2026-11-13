#!/usr/bin/env python3
r"""Tiered answer extraction for the ANSONLY (content) condition, with the tier recorded per block.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

#: the kind string the matching task adaptation actually installs.
try:
    from matching_tasks import KIND as MATCHING_KIND
except Exception:                                      # offline tooling without the task module
    MATCHING_KIND = "binary_match"

#: answer kind per benchmark, central rather than per adapter.
KIND_BY_BENCHMARK = {
    "wdc_products": MATCHING_KIND,
    "geographic_settlements": MATCHING_KIND,
    "musicbrainz20k": MATCHING_KIND,
    "gsm8k": "numeric",
    "strategyqa": "boolean",
    "mmlu_pro": "mcq",
    "gpqa": "mcq",
}

# ---------------------------------------------------------------- matching
#: polarity-bearing surface forms, longest first so "non-match" wins over "match"
_POLAR = re.compile(
    r"\b(non-?match(?:es)?|match(?:es|ed|ing)?|same|identical|equivalent|duplicates?|"
    r"different|distinct|unrelated)\b", re.I)
_NEG_CUE = re.compile(
    r"\b(?:not|n't|no|never|cannot|can't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"nor|neither|without|fails? to|unlikely to|rather than)\b", re.I)
_NEGATIVE_WORDS = {"different", "distinct", "unrelated"}
_LOOKBACK_WORDS = 5


def to_label_v2(raw) -> str:
    """Last polarity word wins, with a negation window in front of it.
    """
    t = str(raw if raw is not None else "").strip()
    if not t:
        return ""
    hits = list(_POLAR.finditer(t))
    if not hits:
        return ""
    m = hits[-1]
    word = m.group(1).lower()
    positive = not (word.startswith("non") or word in _NEGATIVE_WORDS)
    before = t[max(0, m.start() - 120):m.start()]
    tail = " ".join(before.split()[-_LOOKBACK_WORDS:])
    if _NEG_CUE.search(tail):
        positive = not positive
    return "match" if positive else "non-match"


# ---------------------------------------------------------------- QA kinds
_ANSWER_LEAD = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|answer\s*:|answer\s+is|conclusion\s*:)\s*", re.I)
_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_MCQ_LEAD = re.compile(r"\b(?:option|choice|answer)\s*\(?([A-J])\)?\b", re.I)
_MCQ_PAREN = re.compile(r"\(([A-J])\)")
#: A bare option letter must not be preceded by a digit.
_MCQ_BARE = re.compile(r"(?:^|[\s:])(?<![0-9])([A-J])(?:[\.\)\,:]|\s|$)")
_YES = re.compile(r"\b(yes|true|correct)\b", re.I)
_NO = re.compile(r"\b(no|false|incorrect)\b", re.I)


def _numeric(t: str) -> Optional[str]:
    """Prefer a number introduced by an answer lead; otherwise the last number in the text."""
    seg = t
    leads = list(_ANSWER_LEAD.finditer(t))
    if leads:
        seg = t[leads[-1].end():]
        m = _NUM.search(seg)
        if m:
            return m.group(0).replace("$", "").replace(",", "")
    hits = _NUM.findall(t)
    if not hits:
        return None
    return hits[-1].replace("$", "").replace(",", "")


def _mcq(t: str) -> Optional[str]:
    """An option letter. Prefer one introduced by a lead-in or wrapped in parentheses."""
    leads = list(_ANSWER_LEAD.finditer(t))
    seg = t[leads[-1].end():] if leads else ""
    for pat in (_MCQ_LEAD, _MCQ_PAREN, _MCQ_BARE):
        if seg:
            m = pat.search(seg)
            if m:
                return m.group(1).upper()
    for pat in (_MCQ_LEAD, _MCQ_PAREN):
        hits = pat.findall(t)
        if hits:
            return hits[-1].upper()
    return None


def _boolean(t: str) -> Optional[str]:
    """Yes/no, decided by whichever polarity word appears last after any answer lead."""
    leads = list(_ANSWER_LEAD.finditer(t))
    seg = t[leads[-1].end():] if leads else t
    y = list(_YES.finditer(seg))
    n = list(_NO.finditer(seg))
    if not y and not n:
        return None
    if y and n:
        return "yes" if y[-1].start() > n[-1].start() else "no"
    return "yes" if y else "no"


_NL = {"numeric": _numeric, "mcq": _mcq, "boolean": _boolean}
for _k in {MATCHING_KIND, "binary_match", "matching_pair"}:
    _NL[_k] = lambda t: to_label_v2(t) or None


def nl_answer(text: str, kind: str) -> Optional[str]:
    """The validated natural-language parser for `kind`, or None when no answer is stated."""
    fn = _NL.get(kind)
    if fn is None or not (text or "").strip():
        return None
    try:
        out = fn(text)
    except Exception:
        return None
    out = (str(out).strip() if out is not None else "")
    return out or None


#: what a well-formed answer of each kind looks like.
_WELL_FORMED = {
    "numeric": lambda s: bool(re.fullmatch(r"-?\d+(?:\.\d+)?", s.strip().replace(",", ""))),
    "mcq": lambda s: bool(re.fullmatch(r"[A-Ja-j]", s.strip())),
    "boolean": lambda s: s.strip().lower() in ("yes", "no", "true", "false"),
}
for _k in {MATCHING_KIND, "binary_match", "matching_pair"}:
    _WELL_FORMED[_k] = lambda s: s.strip().lower().replace("_", "-") in ("match", "non-match",
                                                                        "nonmatch")


def is_well_formed(ans: str, kind: str) -> bool:
    """True when `ans` is an answer of `kind` rather than prose that happens to contain one."""
    fn = _WELL_FORMED.get(kind)
    if fn is None:
        return bool((ans or "").strip())
    try:
        return fn(ans or "")
    except Exception:
        return False


def extract(text: str, kind: str, contract_fn) -> Tuple[Optional[str], str]:
    """Return `(answer, tier)` with tier in {"contract", "nl", "failed"}.
    """
    t = text or ""
    if not t.strip():
        return None, "failed"
    try:
        lab = contract_fn(t)
    except Exception:
        lab = None
    if lab not in (None, "", "None"):
        s = str(lab).strip()
        if is_well_formed(s, kind):
            return s, "contract"
    lab = nl_answer(t, kind)
    if lab is not None:
        return lab, "nl"
    return None, "failed"


def self_test() -> int:
    bad = []

    def ck(got, want, label):
        if got != want:
            bad.append(f"{label}: want {want!r} got {got!r}")

    # the real Du-MAD block that the contract-only extractor gave up on
    real = ("The two product records are non-match because they describe completely different "
            "products. Record A is a Fujifilm camera lens, while Record B is a Ubiquiti video "
            "camera. They have different titles, descriptions, and prices.")
    ck(nl_answer(real, "matching_pair"), "non-match", "real du_mad block")
    ck(extract(real, "matching_pair", lambda t: None), ("non-match", "nl"), "tier ladder -> nl")
    ck(extract("\\boxed{match}", "matching_pair", lambda t: "match"), ("match", "contract"),
       "tier ladder -> contract")
    ck(extract("I have no idea.", "matching_pair", lambda t: None), (None, "failed"),
       "tier ladder -> failed")

    # negation handling must survive the copy
    ck(nl_answer("These two products are not the same.", "matching_pair"), "non-match", "neg 1")
    ck(nl_answer("They are not different products.", "matching_pair"), "match", "neg 2")
    ck(nl_answer("The records do not refer to the same entity.", "matching_pair"), "non-match",
       "neg 3")

    ck(nl_answer("So 3 apples plus 4 is 7. The answer is 7.", "numeric"), "7", "numeric lead")
    ck(nl_answer("He paid $1,250 total.", "numeric"), "1250", "numeric last + commas")
    ck(nl_answer("Step 1: 18/2 = 9. Step 2: 9+3 = 12.", "numeric"), "12", "numeric no lead")
    ck(nl_answer("No numbers here at all.", "numeric"), None, "numeric absent")

    ck(nl_answer("Therefore the answer is (C).", "mcq"), "C", "mcq paren")
    ck(nl_answer("I would choose option B over the others.", "mcq"), "B", "mcq lead word")
    ck(nl_answer("Final answer: D", "mcq"), "D", "mcq bare after lead")
    ck(nl_answer("None of these seem right.", "mcq"), None, "mcq absent")

    ck(nl_answer("The answer is yes, they did.", "boolean"), "yes", "bool yes")
    ck(nl_answer("Answer: no", "boolean"), "no", "bool no")
    ck(nl_answer("It is true that the claim holds.", "boolean"), "yes", "bool true")
    ck(nl_answer("I cannot determine this.", "boolean"), None, "bool absent")

    # an unknown kind must fail closed rather than guess
    ck(nl_answer("whatever", "span"), None, "unknown kind fails closed")

    # a contract extractor that hands back prose must NOT be accepted as an answer. This is the
    # real MoA block and the real passthrough its patched extractor produces.
    prose = "Record A is a video camera, while Record B is a solid state drive (SSD).\n\nnon-match"
    ck(extract(prose, "matching_pair", lambda t: prose.replace("\n\n", " ")),
       ("non-match", "nl"), "prose-returning contract falls through to nl")
    ck(is_well_formed("non-match", "matching_pair"), True, "well-formed label")
    ck(is_well_formed(prose, "matching_pair"), False, "prose is not well-formed")
    ck(is_well_formed("7", "numeric"), True, "well-formed number")
    ck(is_well_formed("about 7 apples", "numeric"), False, "prose is not a number")
    ck(is_well_formed("C", "mcq"), True, "well-formed option")
    ck(is_well_formed("option C", "mcq"), False, "prose is not an option")
    ck(is_well_formed("yes", "boolean"), True, "well-formed boolean")
    ck(is_well_formed("I think yes", "boolean"), False, "prose is not a boolean")
    # a bare label needs no reduction and must still be read through the contract tier
    ck(extract("non-match", "matching_pair", lambda t: "non-match"), ("non-match", "contract"),
       "already-bare label stays contract")

    # the kind the collection actually passes must resolve. Production threads `matching_tasks.KIND`, which is
    # `"binary_match"`; a table without that key would report no answer for every matching block.
    ck(MATCHING_KIND, "binary_match", "matching_tasks.KIND is not what this module assumes")
    ck(nl_answer(real, MATCHING_KIND), "non-match", "installed kind reaches the parser")
    ck(nl_answer(real, "binary_match"), "non-match", "literal binary_match reaches the parser")
    ck(is_well_formed("non-match", MATCHING_KIND), True, "installed kind validates a label")
    ck(is_well_formed(prose, MATCHING_KIND), False, "installed kind rejects prose")
    for b, k in KIND_BY_BENCHMARK.items():
        ck(k in _NL, True, f"{b} -> {k!r} has a natural-language parser")

    for b in bad:
        print("  FAIL " + b)
    print("answer_extract self-test: %s (%d checks)" % ("PASS" if not bad else "FAIL", 31))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(self_test())
