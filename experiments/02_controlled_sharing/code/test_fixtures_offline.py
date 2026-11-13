#!/usr/bin/env python3
r"""Fixed fixtures for the condition transforms: the awkward peer shapes, with reasoned expectations.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "code")]

import matching_tasks as MT                                            # noqa: E402
MT.install()
import answer_extract as AX                                            # noqa: E402
import arm_transform as AT                                             # noqa: E402

KIND, TASK = MT.KIND, MT.BENCH
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "", on_pass: str = ""):
    note = on_pass if cond else detail
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + note) if note else ''}")
    if not cond:
        FAILS.append(f"{name}: {detail}")


def du_mad_prompt(*bodies: str) -> list:
    blocks = "".join("\n\n One agent solution: ```%s```" % b for b in bodies)
    return [{"role": "user", "content":
             "These are the solutions to the problem from other agents:" + blocks +
             "\n\n Using the solutions from other agents as additional information, can you "
             "provide your answer to the question? "}]


#: (label, peer text, expected answer or None, why)
CASES = [
    ("no boxed, plain claim",
     "Comparing the two listings, they are the same item sold by two retailers.",
     "match", "the peer says the listings are the same item, which is a match"),
    ("no boxed, negated",
     "After comparing titles and specs, these are not the same product.",
     "non-match", "'not the same' negates the positive polarity word"),
    ("double negation",
     "I disagree with the other agent: these are not different products at all.",
     "match", "'not different' is a match; the negation flips a negative word"),
    ("boxed contract",
     "Long chain of reasoning about model numbers. \\boxed{non-match}",
     "non-match", "the released contract is present and wins"),
    ("prose before the label, MoA shape",
     "Record A is a video camera, while Record B is a solid state drive.\n\nnon-match",
     "non-match", "the stated label is non-match; the sentence before it is explanation"),
    ("multiple answers, last one is the conclusion",
     "At first I thought match. On closer reading of the capacity field, non-match.",
     "non-match", "the peer revises its view; the final polarity word is the conclusion"),
    ("confidence field alongside the answer",
     "Answer: match\nConfidence: 0.62\nThe model numbers agree.",
     "match", "a confidence field is metadata, not an answer; the answer is still match"),
    ("truncated mid-sentence, answer already stated",
     "The two records are a match because the GTIN and the manufacturer part number both ag",
     "match", "the answer was stated before the truncation point"),
    ("truncated before any answer",
     "Let me compare the two records field by field. First, the title of record A is",
     None, "nothing was concluded, so there is no answer to extract"),
    ("no answer at all",
     "I do not have enough information to decide.",
     None, "an explicit abstention states no answer"),
    ("error response text",
     "[error] upstream returned HTTP 500 after 3 retries",
     None, "an infrastructure message is not a peer answer"),
    ("empty peer",
     "",
     None, "an empty block carries no answer"),
]


def main() -> int:
    print("fixed fixtures, offline (kind=%r, task=%r)\n" % (KIND, TASK))

    print("1. answer extraction from awkward peer text")
    for label, text, want, why in CASES:
        got, tier = AT._extract_answer(text, "du_mad", KIND, TASK)
        # the rendered form is du_mad's \boxed{...}; compare the label inside it
        inner = None
        if got is not None:
            inner = got.replace("\\boxed{", "").rstrip("}")
        ok = (inner == want) if want else (got is None and tier == "failed")
        check(f"{label}", ok,
              f"expected {want!r} ({why}); got {inner!r} via tier {tier!r}",
              f"{want!r} via {tier!r}" if want else f"recorded as failed ({why})")

    print("\n2. an unreadable peer is never widened, blanked, or dropped")
    msgs = du_mad_prompt("Let me compare the two records field by field. First, the title")
    new, tr = AT.transform("du_mad", "ANSONLY", msgs, kind=KIND, benchmark=TASK, call_seq=0)
    body_kept = "compare the two records field by field" in new[0]["content"]
    check("the block stays byte-for-byte when no answer can be read",
          body_kept and tr["ansonly_extract_failed"] == 1 and not tr["changed"],
          f"kept={body_kept} failed={tr['ansonly_extract_failed']} changed={tr['changed']}",
          "block untouched and the failure recorded")
    check("a failed extraction is not counted as an applied tier",
          tr["ansonly_tier"] == {"contract": 0, "nl": 0},
          f"tiers were {tr['ansonly_tier']}", "no tier credited")
    check("the peer is still present, so ANSONLY did not become NOPEER",
          len(AT.peer_blocks("du_mad", new[0]["content"])) == 1,
          "the unreadable peer was dropped, which would turn content into access",
          "1 peer block still rendered")

    print("\n3. several peers, including duplicates and an empty one")
    msgs = du_mad_prompt("They are the same product. \\boxed{match}",
                         "They are the same product. \\boxed{match}",
                         "These are not the same product.",
                         "")
    new, tr = AT.transform("du_mad", "ANSONLY", msgs, kind=KIND, benchmark=TASK, call_seq=0)
    check("every non-empty peer is found", tr["peer_blocks_found"] == 4,
          f"found {tr['peer_blocks_found']} of 4")
    txt = new[0]["content"]
    check("duplicate peers are both reduced, not deduplicated",
          txt.count("\\boxed{match}") == 2,
          f"found {txt.count('boxed{match}')} match labels; peers must not be silently merged",
          "both duplicates kept and reduced")
    check("the dissenting peer keeps its own answer", "\\boxed{non-match}" in txt,
          "the negated peer lost or flipped its answer")
    check("the empty peer is recorded as a failure, not as an answer",
          tr["ansonly_extract_failed"] == 1,
          f"failures were {tr['ansonly_extract_failed']}, expected exactly the empty one")
    check("no explanation text survives the reduced blocks",
          "same product" not in txt and "not the same" not in txt,
          f"explanation still present: {txt[:200]!r}", "all four explanations removed")

    print("\n4. NOPEER on the same prompt")
    new, tr = AT.transform("du_mad", "NOPEER", msgs, kind=KIND, benchmark=TASK, call_seq=0)
    txt = new[0]["content"]
    check("no peer block survives", not AT.peer_blocks("du_mad", txt),
          f"{len(AT.peer_blocks('du_mad', txt))} blocks left")
    check("the clause that referred to peers is rewritten",
          "Using the solutions from other agents" not in txt and
          "Can you provide your answer to the question?" in txt,
          f"clause handling wrong: {txt[-120:]!r}", "clause replaced with the peerless form")
    check("the question itself is untouched",
          "These are the solutions to the problem from other agents:" not in txt,
          "the peer preamble was left behind with nothing after it")

    print("\n5. a round-0 prompt with no peers is a no-op on every arm")
    bare = [{"role": "user", "content": "Can you provide your answer to the question? "}]
    for arm in AT.ARMS:
        new, tr = AT.transform("du_mad", arm, bare, kind=KIND, benchmark=TASK, call_seq=0)
        check(f"{arm} leaves a peerless prompt alone",
              new == [dict(m) for m in bare] and not tr["changed"],
              f"{arm} modified a prompt that had no peers")

    print("\n6. the QA answer kinds, on peer text of the shape those benchmarks produce")
    qa = [("numeric", "She bakes 3 trays of 12. The answer is 36.", "36",
           "the stated final answer is 36"),
          ("numeric", "18 / 2 = 9, then 9 + 3 = 12", "12",
           "the last computed value is the result"),
          ("boolean", "Elephants are mammals, so yes.", "yes", "the peer concludes yes"),
          ("boolean", "That is not true for penguins. Answer: no", "no",
           "the peer's stated answer is no"),
          ("mcq", "Eliminating A and B, the best choice is (D).", "D",
           "the peer selects option D"),
          ("mcq", "Final answer: C", "C", "the peer states option C")]
    for kind, text, want, why in qa:
        got = AX.nl_answer(text, kind)
        check(f"{kind}: {text[:38]!r}", got == want,
              f"expected {want!r} ({why}), got {got!r}", f"{want!r}")

    print("\n6b. FORD's view: peers are every message after the question, and the "
          "answer-format instruction is the protocol's own")
    INSTR = ("\nRead the question and our opinions carefully to get the key points of the question "
             'then give your answer starting with "Answer: (match|non-match) is more plausible." '
             'and explain very shortly starting with "Explanation: ".')
    # turn 2 of a FORD debate, in the first speaker's frame: question, agent-0's turn, agent-1's
    # turn, with the instruction glued to the last message exactly as `run_one` appends it.
    view = [{"role": "user", "content": "Do these two records refer to the same product?"},
            {"role": "assistant", "content":
             "I reason at length over the titles and specifications, and they are the same "
             "product."},
            {"role": "user", "content":
             "I reason at length over the prices and brands, and they are not the same product."
             + INSTR}]

    new, tr = AT.transform("ford", "NOPEER", view, kind=KIND, benchmark=TASK, call_seq=2)
    joined = " ".join(m["content"] for m in new)
    check("NOPEER finds BOTH prior turns, including the role-flipped one",
          tr["peer_blocks_found"] == 2,
          f"found {tr['peer_blocks_found']} of 2; a role-based rule misses the flipped turn",
          "2 peer turns found")
    check("NOPEER removes every prior turn",
          "I reason at length" not in joined,
          f"peer text survived: {joined[:160]!r}", "no peer turn text left")
    check("NOPEER KEEPS the answer-format instruction",
          'Answer: (match|non-match) is more plausible.' in joined
          and "Explanation" in joined,
          "the instruction went with the peer message, so the agent stops answering in the form "
          "FORD parses, the stop rule can never fire, and the run burns its whole budget",
          "format instruction preserved on the question")
    check("NOPEER rewrites 'our opinions' since there are none",
          "our opinions" not in joined and "Read the question carefully" in joined,
          f"instruction wording wrong: {joined[-200:]!r}", "instruction reworded, format intact")
    check("the question survives", "same product?" in joined, "the question was dropped")

    new, tr = AT.transform("ford", "ANSONLY", view, kind=KIND, benchmark=TASK, call_seq=2)
    joined = " ".join(m["content"] for m in new)
    check("ANSONLY keeps the instruction too",
          'Answer: (match|non-match) is more plausible.' in joined
          and "Read the question and our opinions" in joined,
          f"instruction lost or reworded: {joined[-200:]!r}",
          "instruction intact and unreworded")
    check("ANSONLY removes the prior turns' reasoning and keeps both answers",
          "reason at length" not in joined
          and "Answer: match is more plausible." in joined
          and "Answer: non-match is more plausible." in joined,
          f"reasoning survived: {joined[:200]!r}", "both turns reduced")
    check("ANSONLY still has as many peer messages as it started with",
          len(new) == len(view), f"message count changed {len(view)} -> {len(new)}; "
          "dropping a message would make this an access arm", "message count preserved")

    print("\n7. a wrong task string must not produce a plausible-looking wrong label")
    # ford.parse_prediction dispatches on the TASK name; handing it the answer kind takes the
    # StrategyQA branch and turns a matching answer into "no"
    ford_txt = "Answer: non-match Explanation: the two records describe different products."
    good = AT._label_via_adapter(ford_txt, "ford", KIND, TASK)
    check("FORD with the correct task string reads its own contract", good == "non-match",
          f"got {good!r}", "'non-match'")
    check("the rendered FORD answer is a legal matching label, never a yes/no",
          AX.is_well_formed(good or "", KIND),
          f"{good!r} is not a well-formed {KIND} answer")

    print(f"\n{'ALL FIXTURE CHECKS PASS' if not FAILS else str(len(FAILS)) + ' FAILURES'}")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
