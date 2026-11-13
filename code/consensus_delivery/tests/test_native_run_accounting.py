"""Counterexamples for what a native run claims about its own completeness.

Two real pilot cells motivated every check here. One was killed by a signal and wrote 14 of 40
records while reporting no failures; the other reported three "failed example ids" that were
fragments of a model's prose, because the failure regex spanned newlines and read apostrophes as
quotes. Both passed the accounting check that existed at the time, because that check computed the
absent count as a residual and then asserted the residual balanced.
"""
from __future__ import annotations

import sys

from ..adapters.native_run import failed_samples

CHECKS = 0
FAILED = []


def check(name: str, cond: bool, detail: object = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  ok    {name}")
    else:
        FAILED.append(f"{name}: {detail}")
        print(f"  FAIL  {name}: {detail}")


#: A log with prose containing apostrophes and no failure phrase: an unbounded pattern would match
#: across all of it.
PROSE_LOG = """
[21:12:01] DEBUG Agent 0a1c is improving the solution.
He couldn't have done something good because he is incapable of doing good.
*   **Premise:** The person is incapable of doing good.
He can't do it because he can't reach it, and "can't do it" is identical to "didn't do it".
[21:12:40] DEBUG Agent 96a6 agreed with the solution.
"""

#: The shape upstream actually writes: the phrase on one line, the list on the next.
REAL_LOG = """
INFO:mallm:Starting 40 samples
WARNING:mallm:2 samples failed. Here is a list of their example_ids:
['gsm8k-0019309e83e7cf0e', 'gsm8k-003c5304d9aeda0f']
ERROR:mallm:No more samples in the datasets to substitute failed samples.
INFO:mallm:division by zero
"""

#: The same information on a single line, which some versions emit.
ONE_LINE_LOG = ("WARNING:mallm:1 samples failed: ['gsm8k-0019309e83e7cf0e']\n"
                "INFO:mallm:division by zero\n")

BLOCK = {"gsm8k-0019309e83e7cf0e", "gsm8k-003c5304d9aeda0f", "gsm8k-00654999ca85285f"}


def test_prose_is_never_a_failed_task_id():
    r = failed_samples(PROSE_LOG, valid_ids=BLOCK)
    check("a log with no failure line yields no failures",
          r["n_failed"] == 0 and r["failed_example_ids"] == [], r)
    r2 = failed_samples(PROSE_LOG)
    check("and still none without a block to validate against",
          r2["n_failed"] == 0, r2)


def test_a_real_failure_line_is_read_and_validated():
    r = failed_samples(REAL_LOG, valid_ids=BLOCK)
    check("the list on the line AFTER the phrase is found, as upstream writes it",
          r["n_failed"] == 2, r["failed_example_ids"])
    one = failed_samples(ONE_LINE_LOG, valid_ids=BLOCK)
    check("and the single-line form still works", one["n_failed"] == 1,
          one["failed_example_ids"])
    check("they are the block's own ids",
          set(r["failed_example_ids"]) <= BLOCK, r["failed_example_ids"])
    check("the reason is counted", r["reason_counts"].get("division by zero") == 1,
          r["reason_counts"])
    check("nothing was rejected", r["rejected_as_not_task_ids"] == [],
          r["rejected_as_not_task_ids"])


def test_an_id_outside_the_block_is_rejected_not_counted():
    log = "[21:13:02] WARNING 1 samples failed: ['mmlu_pro-deadbeefdeadbeef']\n"
    r = failed_samples(log, valid_ids=BLOCK)
    check("an id from another benchmark is not a failed task here", r["n_failed"] == 0, r)
    check("and it is named rather than dropped silently",
          r["rejected_as_not_task_ids"] == ["mmlu_pro-deadbeefdeadbeef"],
          r["rejected_as_not_task_ids"])


def test_the_search_reaches_at_most_the_following_line():
    """An unbounded `.*?` under `re.S` would reach a bracket arbitrarily far below.

    The bound is narrow rather than absent. A strictly single-line pattern would find
    nothing in a real log, because upstream puts the list on the next line; it would have rewritten
    three genuine main failures to zero.
    """
    far = "samples failed\nsome prose\nand then ['gsm8k-0019309e83e7cf0e']\n"
    check("a bracket two lines below is not this phrase's list",
          failed_samples(far, valid_ids=BLOCK)["n_failed"] == 0, failed_samples(far))
    near = "samples failed. Here is a list of their example_ids:\n['gsm8k-0019309e83e7cf0e']\n"
    check("a bracket on the next line is",
          failed_samples(near, valid_ids=BLOCK)["n_failed"] == 1,
          failed_samples(near, valid_ids=BLOCK))


def main() -> int:
    for fn in (test_prose_is_never_a_failed_task_id,
               test_a_real_failure_line_is_read_and_validated,
               test_an_id_outside_the_block_is_rejected_not_counted,
               test_the_search_reaches_at_most_the_following_line):
        fn()
    print(f"\n{CHECKS} checks, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  - {f}")
    return 1 if FAILED else 0


def test_native_run_accounting_counterexamples():
    """pytest entrypoint, so `conftest` enforces the checks above."""
    rc = main()
    assert rc == 0, f"recorded check failures: {FAILED}"
    assert CHECKS > 0


if __name__ == "__main__":
    sys.exit(main())
