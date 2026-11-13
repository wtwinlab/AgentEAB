from __future__ import annotations

import re
from typing import Optional

_NUM = re.compile(r"[-+]?\d(?:[\d,]*\d)?(?:\.\d+)?")

_GSM8K_GOLD = re.compile(r"####\s*([-+]?\d(?:[\d,]*\d)?(?:\.\d+)?)")


def _to_number(tok: str) -> Optional[float]:
    try:
        return float(tok.replace(",", "").replace("+", ""))
    except ValueError:
        return None


def last_number(text: object) -> Optional[float]:
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
    if reference is None:
        return None
    s = str(reference)
    m = _GSM8K_GOLD.search(s)
    if m:
        return _to_number(m.group(1))
    return None


def gsm8k_correct(delivered: object, reference: object, *, tol: float = 1e-6) -> Optional[bool]:
    gold = gsm8k_gold(reference)
    pred = last_number(delivered)
    if gold is None or pred is None:
        return None
    return abs(gold - pred) <= tol


_LETTER = re.compile(r"(?<![A-Za-z\\{_^])\(?([A-J])\)?(?![A-Za-z]|\s*=)")


def letter_answer(text: object, n_options: int = 10) -> Optional[str]:
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


_YES = ("yes", "true", "correct")
_NO = ("no", "false", "incorrect")


def boolean_answer(text: object) -> Optional[bool]:
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
    pred = boolean_answer(delivered)
    g = str(reference).strip().lower()
    gold = True if g in _YES else False if g in _NO else None
    if pred is None or gold is None:
        return None
    return pred is gold


def _squad_norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def squad_v2_scores(delivered: object, references: object, *, unanswerable: bool
                    ) -> Optional[dict]:
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
