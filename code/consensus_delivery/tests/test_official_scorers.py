"""Counterexamples for the official-reference scorers.

Written because the first GSM8K scoring pass reported 4 of 6 on a 6-of-6 run: the extractor did
`"370".rstrip(".0")`, which yields `"37"`. Every case below is one the incorrect version got wrong, or
a boundary of GSM8K's own reference format.
"""
from __future__ import annotations

import sys

from ..adapters.official_scorers import gsm8k_correct, gsm8k_gold, last_number

CHECKS = 0
FAILED: list = []


def check(name: str, cond: bool, detail: object = "") -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILED.append(name)
        print(f"  FAIL  {name}   [{detail}]")
    else:
        print(f"  ok    {name}")


def _naive(ans: str):
    """A naive extractor, used as the counterexample generator."""
    import re
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(ans).replace("$", "").replace("**", ""))
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".0") or None


def test_trailing_zeros_are_not_stripped():
    for text, want in (("The answer is 370", 370.0), ("300", 300.0), ("Answer: 2800", 2800.0),
                       ("10", 10.0), ("100.00", 100.0)):
        check(f"last_number({text!r}) == {want}", last_number(text) == want,
              f"got {last_number(text)}")
    # the exact failures the incorrect version produced
    for text, naive_to in (("The answer is 370", "37"), ("300", "3")):
        check(f"the naive extractor turned {text!r} into {naive_to!r}", _naive(text) == naive_to,
              f"naive gave {_naive(text)!r}")


def test_zero_survives():
    check("zero parses as 0.0, not None", last_number("the result is 0") == 0.0,
          last_number("the result is 0"))
    check("the naive extractor annihilated zero", _naive("the result is 0") is None,
          _naive("the result is 0"))
    check("0.0 parses as 0.0", last_number("0.0") == 0.0, last_number("0.0"))


def test_negatives_and_decimals():
    for text, want in (("-27", -27.0), ("the total is -1,250", -1250.0), ("12.5", 12.5),
                       ("-0.75", -0.75), ("+48", 48.0), ("minus sign U+2212: −9", -9.0)):
        check(f"last_number({text!r}) == {want}", last_number(text) == want,
              f"got {last_number(text)}")


def test_thousands_separators_and_markup():
    check("2,880,000 parses", last_number("Answer: 2,880,000") == 2880000.0,
          last_number("Answer: 2,880,000"))
    check("currency is ignored", last_number("$9,360") == 9360.0, last_number("$9,360"))
    check("bold markup is ignored", last_number("**91**") == 91.0, last_number("**91**"))
    check("a trailing period does not become part of the number",
          last_number("The answer is 18.") == 18.0, last_number("The answer is 18."))


def test_the_answer_is_the_last_number_not_the_first():
    text = ("She buys 3 boxes at 12 each, so 3*12 = 36, then spends 4 more. "
            "The final answer is 40")
    check("the last number wins", last_number(text) == 40.0, last_number(text))


def test_gsm8k_gold_uses_the_hash_marker():
    ref = ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
           "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72")
    check("the gold is the #### value, not the last inline number",
          gsm8k_gold(ref) == 72.0, gsm8k_gold(ref))
    check("a #### value with separators parses", gsm8k_gold("#### 2,880,000") == 2880000.0,
          gsm8k_gold("#### 2,880,000"))
    check("a #### value with trailing zeros parses", gsm8k_gold("#### 370") == 370.0,
          gsm8k_gold("#### 370"))
    check("a negative #### value parses", gsm8k_gold("#### -5") == -5.0, gsm8k_gold("#### -5"))
    check("#### 0 parses as 0.0", gsm8k_gold("#### 0") == 0.0, gsm8k_gold("#### 0"))


def test_an_unscorable_cell_is_none_not_false():
    check("an absent reference is unscorable, not wrong",
          gsm8k_correct("the answer is 5", "") is None, gsm8k_correct("the answer is 5", ""))
    check("an empty reference list element is unscorable",
          gsm8k_correct("5", None) is None, gsm8k_correct("5", None))
    check("an answer with no number is unscorable, not wrong",
          gsm8k_correct("I cannot determine the answer", "#### 5") is None,
          gsm8k_correct("I cannot determine the answer", "#### 5"))
    check("a real mismatch is False, not None",
          gsm8k_correct("the answer is 6", "#### 5") is False,
          gsm8k_correct("the answer is 6", "#### 5"))
    check("a real match is True", gsm8k_correct("the answer is 5", "#### 5") is True,
          gsm8k_correct("the answer is 5", "#### 5"))
    check("370 against #### 370 matches (the original failure)",
          gsm8k_correct("The answer is 370", "rationale\n#### 370") is True,
          gsm8k_correct("The answer is 370", "rationale\n#### 370"))
    check("370.0 against #### 370 matches",
          gsm8k_correct("The answer is 370.0", "#### 370") is True,
          gsm8k_correct("The answer is 370.0", "#### 370"))
    check("37 against #### 370 does NOT match",
          gsm8k_correct("The answer is 37", "#### 370") is False,
          gsm8k_correct("The answer is 37", "#### 370"))





def test_letters_in_formulas_and_articles_are_not_answers():
    """Counterexamples an independent recomputation produced by disagreeing on a real GPQA answer.

    `gpqa-17ee204f8a3ef890`'s reply was truncated mid-formula and stated no option letter at all. One
    implementation matched the `F` in `\\rangle_{F=1}` and returned `None` only because `F` is out of
    range at four options; the other folded case and matched the article "a" in "In a standard",
    scoring the answer as option A. Both were wrong, in opposite directions, on the same text, and
    the shared right answer is that an answer stating no letter is unscorable.
    """
    from ..adapters.official_scorers import choice_correct, letter_answer
    latex = r"\langle S \rangle_{F=1} = -1.25"
    check("a LaTeX subscript is not an answer letter, even when in range",
          letter_answer(latex, 10) is None, letter_answer(latex, 10))
    check("a letter followed by = is not an answer letter",
          letter_answer("where B = 7 tesla", 10) is None, letter_answer("where B = 7 tesla", 10))
    check("a lowercase article is not an answer letter",
          letter_answer("In a standard hydrogen atom", 4) is None,
          letter_answer("In a standard hydrogen atom", 4))
    check("a lowercase option mention is not an answer letter",
          letter_answer("the a and b terms cancel", 4) is None,
          letter_answer("the a and b terms cancel", 4))
    check("an explicit answer still parses", letter_answer("Therefore the answer is C.", 4) == "C",
          letter_answer("Therefore the answer is C.", 4))
    check("a parenthesised answer still parses", letter_answer("Answer: (B)", 4) == "B",
          letter_answer("Answer: (B)", 4))
    check("an option line still parses", letter_answer("D. 64218 m", 4) == "D",
          letter_answer("D. 64218 m", 4))
    check("an answer with no letter at all is unscorable, not wrong",
          choice_correct(r"the result is \frac{1}{2}", "C", 4) is None,
          choice_correct(r"the result is \frac{1}{2}", "C", 4))


def main() -> int:
    for fn in (test_trailing_zeros_are_not_stripped, test_zero_survives,
               test_negatives_and_decimals, test_thousands_separators_and_markup,
               test_the_answer_is_the_last_number_not_the_first,
               test_gsm8k_gold_uses_the_hash_marker,
               test_an_unscorable_cell_is_none_not_false,
               test_letters_in_formulas_and_articles_are_not_answers):
        fn()
    print(f"\n{CHECKS} checks, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  - {f}")
    return 1 if FAILED else 0


def test_official_scorer_counterexamples():
    """pytest entrypoint, so `conftest` enforces the checks above."""
    rc = main()
    assert rc == 0, f"recorded check failures: {FAILED}"
    assert CHECKS > 0


if __name__ == "__main__":
    sys.exit(main())
