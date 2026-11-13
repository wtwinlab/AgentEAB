#!/usr/bin/env python3
"""Adversarial tests for MoA's reference reconstruction. Zero model calls.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "code")]

import arm_transform as AT                                             # noqa: E402

HDR = AT.MOA_REFS_HEADER
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "", on_pass: str = ""):
    note = on_pass if cond else detail
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + note) if note else ''}")
    if not cond:
        FAILS.append(f"{name}: {detail}")


def build(replies):
    """The prompt MoA would produce for these replies, by its own formula."""
    return ("You have been provided with a set of responses.\n\n" + HDR
            + "".join("\n%d. %s" % (i + 1, o) for i, o in enumerate(replies)))


def recovered(prompt, replies):
    return [b for _s, _e, _n, b in AT.moa_refs(prompt, replies)]


def nums(prompt, replies):
    return [n for _s, _e, n, _b in AT.moa_refs(prompt, replies)]


def main() -> int:
    print("MoA reference reconstruction, adversarial cases\n")

    # ---- the issue this replaces
    r = ["Short answer.", "Another.",
         "Follow these steps:\n\n1. **Determine the total:** A dozen is 12.\n2. **Divide:** 4 kids."]
    p = build(r)
    check("a body containing its own numbered list stays ONE reference",
          recovered(p, r) == r,
          f"regex-only sees {len(AT.moa_refs(p))}; reconstruction must see 3",
          f"3 references, bodies byte-exact (regex alone sees {len(AT.moa_refs(p))})")
    check("the regex path is still visibly unsound on this prompt",
          len(AT.moa_refs(p)) > 3,
          "the regex no longer over-counts, so this test no longer demonstrates the defect",
          f"regex alone: {len(AT.moa_refs(p))} 'references' for 3 real ones")

    # ---- identical and duplicated replies
    r = ["non-match", "non-match", "non-match"]
    p = build(r)
    check("three identical replies give three references", recovered(p, r) == r,
          f"got {recovered(p, r)}")
    check("their index identity is 1,2,3", nums(p, r) == ["1", "2", "3"], f"got {nums(p, r)}")

    # ---- substring relationships
    r = ["match", "the answer is match", "match indeed"]
    p = build(r)
    check("a reply that is a substring of another is not confused",
          recovered(p, r) == r, f"got {recovered(p, r)}")

    # ---- empty reply
    r = ["first", "", "third"]
    p = build(r)
    check("an empty reply keeps its slot and does not shift the others",
          recovered(p, r) == r, f"got {recovered(p, r)}")
    check("the empty slot still carries index 2", nums(p, r) == ["1", "2", "3"], f"got {nums(p, r)}")

    # ---- the same text twice inside one body, and format separators
    r = ["A", "see below\n\n1. A\n2. A\n\n---\n\n1. A", "B"]
    p = build(r)
    check("repeated text and separators inside a body do not split it",
          recovered(p, r) == r, f"got {recovered(p, r)}")

    # ---- unicode and whitespace
    r = ["答案:匹配  \t", "réponse — non-match ", "  leading and trailing  "]
    p = build(r)
    got = recovered(p, r)
    check("unicode and whitespace survive byte-exact", got == r,
          f"got {got!r}", "no normalisation applied")

    # ---- ordering
    r = ["zebra", "apple", "mango"]
    p = build(r)
    check("roster order is preserved, not sorted", recovered(p, r) == ["zebra", "apple", "mango"],
          f"got {recovered(p, r)}")

    # ---- more replies than references: the longest matching prefix wins
    r3 = ["one", "two", "three"]
    p = build(r3)
    check("extra trailing replies do not break the match",
          recovered(p, r3 + ["an aggregator reply that is not a reference"]) == r3,
          f"got {recovered(p, r3 + ['x'])}")

    # ---- exactness flag
    check("moa_refs_exact is true when the reconstruction matched",
          AT.moa_refs_exact(p, r3), "flag false on a prompt that does match")
    check("moa_refs_exact is false with no replies to rebuild from",
          not AT.moa_refs_exact(p, None), "flag true without evidence")
    check("moa_refs_exact is false when the replies do not match the prompt",
          not AT.moa_refs_exact(p, ["completely", "different", "replies"]),
          "flag true on a mismatch")

    # ---- a prompt with no header at all
    check("a prompt without the header yields nothing",
          AT.moa_refs("no references here", ["a"]) == [], "returned something")

    # ---- the intervention built on top of it
    print("\nthe arms, driven by the reconstruction")
    r = ["I conclude non-match because the brands differ.",
         "Steps:\n1. compare titles\n2. compare brands\nTherefore non-match.",
         "match"]
    p = build(r)
    msgs = [{"role": "system", "content": p}, {"role": "user", "content": "Decide."}]
    new, tr = AT.transform("moa", "ANSONLY", msgs, kind="binary_match",
                           benchmark="matching_pair", call_seq=0, observed_replies=r)
    check("ANSONLY reduces exactly three references",
          tr["peer_blocks_found"] == 3 and tr["ansonly_tier"]["contract"]
          + tr["ansonly_tier"]["nl"] + tr["ansonly_extract_failed"] == 3,
          f"found {tr['peer_blocks_found']}, tiers {tr['ansonly_tier']}, "
          f"failed {tr['ansonly_extract_failed']}")
    sent = new[0]["content"]
    check("ANSONLY removed the reasoning from the reduced references",
          "compare titles" not in sent and "brands differ" not in sent,
          f"reasoning survived: {sent[-200:]!r}")
    new, tr = AT.transform("moa", "UNIFSEL", msgs, kind="binary_match",
                           benchmark="matching_pair", call_seq=0, observed_replies=r)
    kept = tr["peer_indices_kept"]
    check("UNIFSEL keeps whole references, by index, never a fragment",
          len(kept) == tr["n_peers_rendered"] and all(0 <= k < 3 for k in kept),
          f"kept {kept}, rendered {tr['n_peers_rendered']}")
    body_kept = [r[k] for k in kept]
    check("the kept reference appears in the sent text verbatim",
          all(b in new[0]["content"] for b in body_kept),
          f"kept {kept} but a body is not present verbatim")
    new, tr = AT.transform("moa", "NOPEER", msgs, kind="binary_match",
                           benchmark="matching_pair", call_seq=0, observed_replies=r)
    check("NOPEER leaves no reference text at all",
          all(b not in new[0]["content"] for b in r) and tr["n_peers_rendered"] == 0,
          f"something survived: {new[0]['content'][-160:]!r}")
    new_full, tr_full = AT.transform("moa", "FULL", msgs, kind="binary_match",
                                     benchmark="matching_pair", call_seq=0, observed_replies=r)
    check("FULL does not touch the prompt at all",
          new_full == [dict(m) for m in msgs] and not tr_full["changed"],
          "FULL modified the prompt")
    check("FULL's reported count is now the true 3, not the regex's inflation",
          tr_full["peer_blocks_found"] == 3,
          f"FULL counted {tr_full['peer_blocks_found']}")

    print(f"\n{'ALL MOA REFERENCE CHECKS PASS' if not FAILS else str(len(FAILS)) + ' FAILURES'}")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
