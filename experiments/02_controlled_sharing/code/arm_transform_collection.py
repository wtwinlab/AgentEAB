#!/usr/bin/env python3
"""Prompt transform as imported by the collection runner for the records it logged.

`verify_sent_input.py` uses this module to recompute the request hashes stored in those records.
`arm_transform.py` builds the no-peer prompts of the no-peer collection whose results the paper
reports; its peer-block delimiting differs from this module's (see its `peer_blocks`).

The three interventions, applied as surgery on the outgoing prompt rather than as adapter forks.

Why at the message level. Each intervention is, in substance, a statement about what the recipient
*sees*: whether peer text is present (access), what form it takes (content), and which peers are
included (selection). All three are therefore transformations of the request a protocol is about to
send, and doing them there has three properties that patching five different `run_one` bodies does
not:

* **No protocol logic is touched.** Rounds, agent counts, terminal rules and stop conditions run
  exactly as released; only the rendered peer block differs. A protocol that legitimately stops
  earlier because its peers changed is showing a *consequence* of the intervention, which is the
  measurement, not a violation of it.
* **The diff is auditable.** Every transform returns the before/after text and the peer indices it
  kept, so the acceptance check reads the actual prompt rather than trusting the patch.
* **It covers protocols whose peer block is assembled inline.** MoA builds its aggregator system
  prompt inside `run_one`, and MAD-M² and FORD assemble their peer turns there too. At the message
  level that distinction disappears.

Peer-block grammars, transcribed from the released adapters:

| protocol | marker | per-peer block |
|---|---|---|
| BCCS | `These are the solutions to the problem from other agents: ` | ` One agent solution: ```…``` ` / ` One leader solution: ```…``` ` / ` One supporting agent solution: ```…``` ` |
| Du-MAD | same prefix | `\\n\\n One agent solution: ```…``` ` |
| MAD-M² | `These are the solutions to the problem from other agents:\\n` | `` One agent solution: ```…``` `` |
| MoA | `You have been provided with a set of responses…` (system) | numbered `\\n\\n{i}. {text}` |
| FORD | opponent turn arrives as its own chat message | the whole message |

`ANSONLY` extracts through the tier ladder in `answer_extract`: the protocol's released answer
contract first, then a validated natural-language parser for that answer kind, and only then a
recorded failure. The middle tier is not a convenience -- without it the contract-only ladder gave up
on half of all real Du-MAD blocks, blocks that stated "the two product records are non-match" in
plain prose, and shipped them whole while logging a failure, which made ANSONLY silently identical to
FULL on that half. A `failed` block is still left untouched and still recorded, and a call containing
one is excluded from the content dimension's valid sample rather than counted as an intervention;
ANSONLY is never faked by blanking a block or dropping a peer, either of which would turn the
content arm into an access arm.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import answer_extract as AX

ARMS = ("FULL", "NOPEER", "ANSONLY", "UNIFSEL")

#: the sentence each protocol uses to introduce peer text, and the clause that refers back to it
PEER_PREFIX = {
    "bccs": "These are the solutions to the problem from other agents:",
    "du_mad": "These are the solutions to the problem from other agents:",
    "madm2_objective": "These are the solutions to the problem from other agents:",
    "madm2_naive": "These are the solutions to the problem from other agents:",
}
#: clauses that only make sense when peers are present; dropped by NOPEER, verbatim from the sources
PEER_CLAUSES = [
    ("Selecting and using the trustable solutions from current collaboration as additional "
     "information, can you provide your answer to the problem?", "Can you provide your answer to the problem?"),
    ("Selecting and using the leading solutions from current collaboration as additional "
     "information, can you provide your answer to the problem?", "Can you provide your answer to the problem?"),
    ("Using the solutions from other agents as additional information, can you provide your answer "
     "to the question?", "Can you provide your answer to the question?"),
    ("Using the solutions from other agents as additional information, can you provide your answer "
     "to the math problem?", "Can you provide your answer to the math problem?"),
    ("Using the solutions from other agents as additional information, give your",
     "Give your"),
    ("Read the question and our opinions carefully", "Read the question carefully"),
]
#: BCCS and Du-MAD fence each peer in triple backticks. Verified against sent prompts.
BLOCK_RE = re.compile(r"\n\n ?One (?:agent|leader|supporting agent) solution: ```(.*?)```", re.S)
#: MAD-M^2 does NOT fence its peers: `\n\n\nOne agent's solution: <text>` and the block runs to
#: the next such marker or to the suffix. Verified against sent prompts; the backtick shape
#: of the other two protocols does not apply here.
MADM2_BLOCK_RE = re.compile(
    r"\n\n\nOne agent's solution: (.*?)(?=\n\n\nOne agent's solution: |\n\nUsing the solutions"
    r" from other agents|\Z)", re.S)
#: MoA appends `Responses from models:` then SINGLE-newline numbered bodies, which may themselves
#: contain blank lines. Verified against sent prompts.
MOA_REFS_HEADER = "Responses from models:"
MOA_REF_RE = re.compile(r"\n(\d+)\. (.*?)(?=\n\d+\. |\Z)", re.S)
MOA_MARKER = "You have been provided with a set of responses"
#: FORD appends its answer-format instruction to the last message of every view. All of its
#: per-benchmark variants open with this sentence, so it is the marker that separates the
#: protocol's own instruction from the peer content it happens to be glued to.
FORD_INSTR_MARKER = "Read the question and our opinions carefully"
#: FORD's instruction differs per benchmark, and only some variants open with the marker above:
#: StrategyQA and the matching adaptation do, while GSM8K and MMLU-Pro open with "Do you think I am
#: more reasonable? ...". Splitting on one marker therefore failed to detach the instruction on
#: those two, so NOPEER deleted it along with the peer turn it was glued to, the agents stopped
#: answering in the form `parse_prediction` reads, and FORD delivered nothing. The instruction is
#: now taken from the adapter itself and stripped as an exact suffix, so no variant can be missed.
_FORD_INSTR_CACHE: Dict[str, List[str]] = {}


def _ford_instructions(benchmark: str) -> List[str]:
    """Every instruction string FORD might have appended for this task, longest first."""
    key = benchmark or ""
    if key in _FORD_INSTR_CACHE:
        return _FORD_INSTR_CACHE[key]
    out: List[str] = []
    try:
        from consensus_delivery.adapters import ford as _f
        for n in (2, 3, 4, 5, 6, 10):
            try:
                t = _f.instruction(key, n)
            except Exception:
                continue
            if t and t not in out:
                out.append(t)
    except Exception:
        pass
    out.sort(key=len, reverse=True)
    _FORD_INSTR_CACHE[key] = out
    return out


def _ford_split_instruction(text: str, benchmark: str) -> Tuple[str, str]:
    """`(instruction including its leading newline, the text without it)`.

    Falls back to the marker only when the adapter cannot be consulted, so a missing import
    keeps the marker-based split rather than dropping the instruction.
    """
    t = text or ""
    for instr in _ford_instructions(benchmark):
        if t.endswith("\n" + instr):
            return "\n" + instr, t[: -(len(instr) + 1)]
        if t.endswith(instr):
            return instr, t[: -len(instr)]
    j = t.rfind(FORD_INSTR_MARKER)
    if j > 0:
        cut = j - 1 if t[j - 1] == "\n" else j
        return t[cut:], t[:cut]
    return "", t


def peer_blocks(protocol: str, text: str):
    """[(start, end, body)] of the peer blocks this protocol renders in `text`."""
    rx = MADM2_BLOCK_RE if protocol.startswith("madm2") else BLOCK_RE
    return [(m.start(), m.end(), m.group(1)) for m in rx.finditer(text or "")]


def moa_refs(text: str, observed_replies: Optional[Sequence[str]] = None):
    """[(start, end, number, body)] of MoA's references, after its header.

    **Reconstructed, not parsed.** `moa.run_one` builds the aggregator prompt as

        AGGREGATOR_SYSTEM_PROMPT + "".join(f"\n{i+1}. {o}" for i, o in enumerate(outputs))

    so the only sound way to recover the boundaries is to rebuild that exact suffix from the
    reference replies this execution produced and require it to match the prompt **byte for byte**.
    When it matches, the boundaries are known by construction, the index identity is the roster
    order, and the body is the original reply verbatim -- nothing is inferred from delimiters.

    Parsing by delimiter is unsound: the `\n<n>. ` delimiter is byte-identical to a numbered
    step a model writes inside its own answer, so one reference containing "1. Determine… 2.
    Divide…" would split into three (2.71-2.85x over-count on GSM8K prompts, 1.03-1.11x on WDC).

    The reconstruction is also immune to the cases a search-based version gets wrong: identical or
    duplicated replies, a reply that is a substring of another, an empty reply, numbered lists or
    format separators inside a body, the same text appearing more than once, and ordering -- none of
    them involve searching, so none of them can mis-assign a body.

    `observed_replies` may hold more entries than there are references (a protocol can have made
    earlier calls), so the longest matching **prefix** is used. With no replies, or no match, the
    regex is the only thing left; it is returned with `approximate=True` on the trace by the caller
    and must not be trusted for an intervention.
    """
    t = text or ""
    i = t.find(MOA_REFS_HEADER)
    if i < 0:
        return []
    base = i + len(MOA_REFS_HEADER)
    if observed_replies:
        for k in range(len(observed_replies), 0, -1):
            cand = list(observed_replies[:k])
            suffix = "".join("\n%d. %s" % (n + 1, o or "") for n, o in enumerate(cand))
            if t[base:base + len(suffix)] == suffix:
                out, pos = [], base
                for n, o in enumerate(cand):
                    head = "\n%d. " % (n + 1)
                    body_at = pos + len(head)
                    out.append((pos, body_at + len(o or ""), str(n + 1), o or ""))
                    pos = body_at + len(o or "")
                return out
    out = []
    for m in MOA_REF_RE.finditer(t[base:]):
        out.append((base + m.start(), base + m.end(), m.group(1), m.group(2)))
    return out


def moa_refs_exact(text: str, observed_replies: Optional[Sequence[str]] = None) -> bool:
    """True when the reference boundaries were reconstructed rather than guessed by the regex."""
    t = text or ""
    i = t.find(MOA_REFS_HEADER)
    if i < 0 or not observed_replies:
        return False
    base = i + len(MOA_REFS_HEADER)
    for k in range(len(observed_replies), 0, -1):
        suffix = "".join("\n%d. %s" % (n + 1, o or "")
                         for n, o in enumerate(observed_replies[:k]))
        if t[base:base + len(suffix)] == suffix:
            return True
    return False


#: how each protocol RENDERS an answer, so ANSONLY hands the recipient something its own parser
#: reads. The label itself comes from the adapter's own extractor, never from a regex invented here.
_RENDER = {
    "bccs": lambda lab: "\\boxed{%s}" % lab,
    "du_mad": lambda lab: "\\boxed{%s}" % lab,
    "madm2_objective": lambda lab: "Answer: %s" % lab,
    "madm2_naive": lambda lab: "Answer: %s" % lab,
    "ford": lambda lab: 'Answer: %s is more plausible.' % lab,
    # MoA's released instruction is "State exactly 'match' or 'non-match' at the end", so a bare
    # label is the form its aggregator expects -- there is no marker to reproduce.
    "moa": lambda lab: lab,
}


def _label_via_adapter(text: str, protocol: str, kind: str,
                       benchmark: str = "") -> Optional[str]:
    """The protocol's OWN extractor, i.e. the released answer contract. None when it reads nothing.

    Using each adapter's extractor rather than a pattern written here is the difference between
    ANSONLY being faithful and being a guess: MoA, for one, emits a bare `non-match` because its
    released instruction asks for exactly that, so a `\\boxed{}`-or-`Answer:` regex finds nothing and
    the intervention silently does not engage.

    **The five extractors do not take the same second argument**, and the collection's two task
    strings are not interchangeable: the matching adaptation registers `BENCH = "matching_pair"` as
    the task name and `KIND = "binary_match"` as the answer kind. Checked against the sources:

        bccs.extract_boxed(text, kind)          -> kind
        du_mad.extract_for(benchmark, text)     -> benchmark, and the text is the SECOND argument
        mad_m2.extract_answer(text, kind)       -> kind
        moa.extract(text, kind)                 -> kind
        ford.parse_prediction(text, benchmark)  -> benchmark

    Passing the wrong one does not fail loudly. `mad_m2.extract_answer` and `moa.extract` return
    their whole input, which the well-formedness gate in `answer_extract` catches; but
    `ford.parse_prediction(text, "binary_match")` takes the non-matching branch and returns
    `"no"` for a `non-match` answer, because that path lowercases for StrategyQA. A wrong label that
    *looks* well formed is worse than a failure, so the argument each adapter wants is passed
    explicitly here rather than inferred from one shared `kind`.
    """
    t = text or ""
    if not t.strip():
        return None
    k = kind or "binary_match"
    b = benchmark or "matching_pair"
    try:
        from consensus_delivery.adapters import bccs, du_mad, ford, mad_m2, moa
        if protocol == "bccs":
            lab = bccs.extract_boxed(t, k)
        elif protocol == "du_mad":
            lab = du_mad.extract_for(b, t)
        elif protocol.startswith("madm2"):
            lab = mad_m2.extract_answer(t, k)
        elif protocol == "ford":
            lab = ford.parse_prediction(t, b)[0]
            if lab in (None, "", "None", "none"):
                lab = None
        elif protocol == "moa":
            lab = moa.extract(t, k)
        else:
            return None
    except Exception:
        return None
    if lab in (None, "", "None"):
        return None
    return str(lab).strip() or None


def _extract_answer(text: str, protocol: str, kind: str,
                    benchmark: str = "") -> Tuple[Optional[str], str]:
    """`(answer re-rendered in the protocol's own form, tier)`; tier in contract / nl / failed.

    The tier is carried out of here because the content dimension's valid sample is defined by it.
    A `failed` block is left as it was, and a call containing one is NOT a successful ANSONLY
    intervention: on real Du-MAD traffic the contract-only ladder gave up on half the blocks --
    blocks that stated "the two product records are non-match" in prose -- and shipping them whole
    while logging a failure made ANSONLY silently identical to FULL on that half.
    """
    lab, tier = AX.extract(text, kind or "binary_match",
                           lambda t: _label_via_adapter(t, protocol, kind, benchmark))
    if lab is None:
        return None, tier
    render = _RENDER.get(protocol)
    return (render(lab) if render else lab), tier


def transform(protocol: str, arm: str, messages: Sequence[Dict[str, str]], *,
              kind: str = "", seed: int = 20261019, call_seq: int = 0, benchmark: str = "",
              observed_replies: Optional[Sequence[str]] = None,
              ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Return (new_messages, trace). `trace` always records what was found and what was done."""
    msgs = [dict(m) for m in messages]
    tr: Dict[str, Any] = {"arm": arm, "protocol": protocol, "peer_blocks_found": 0,
                          "peer_indices_kept": [], "n_peers_rendered": 0,
                          "ansonly_extract_failed": 0, "ansonly_tier": {"contract": 0, "nl": 0},
                          "clauses_rewritten": [],
                          "changed": False}
    if arm == "FULL":
        # still report what the released prompt rendered, so FULL has the same trace fields
        for m in msgs:
            c = m.get("content") or ""
            tr["peer_blocks_found"] += len(peer_blocks(protocol, c))
            if MOA_MARKER in c:
                tr["peer_blocks_found"] += len(moa_refs(c, observed_replies))
        if protocol == "ford":
            # same rule the intervention arms use: everything after the question is a prior turn
            tr["peer_blocks_found"] = max(len(msgs) - 1, 0)
        tr["n_peers_rendered"] = tr["peer_blocks_found"]
        tr["peer_indices_kept"] = list(range(tr["peer_blocks_found"]))
        return msgs, tr

    rng = random.Random((seed, protocol, arm, call_seq).__hash__() & 0xFFFFFFFF)

    # ---------- FORD: the opponent's turn is its own message -------------------------------
    if protocol == "ford":
        # `run_one` keeps one shared transcript whose first entry is the question and whose later
        # entries are the prior turns, then flips the roles for the second speaker. So **every
        # message after index 0 is peer content, whatever its role**; a rule on
        # `role == "user" and i > 0` would miss the flipped turns.
        peer_msgs = list(range(1, len(msgs)))
        tr["peer_blocks_found"] = len(peer_msgs)
        # The released code appends the answer-format instruction to the LAST message of the view,
        # which from turn 1 on *is* a peer turn. Deleting peer messages would delete the
        # instruction with them, and without the `Answer: … Explanation: …` form the stop rule
        # could not fire. The instruction is therefore detached first and re-attached after.
        instr = ""
        if msgs:
            instr, body = _ford_split_instruction(msgs[-1].get("content") or "", benchmark)
            if instr:
                msgs[-1]["content"] = body
        if arm == "NOPEER":
            msgs = [msgs[0]] if msgs else msgs
            tr["changed"] = len(peer_msgs) > 0
        elif arm == "ANSONLY":
            kept = []
            for i in peer_msgs:
                got, tier = _extract_answer(msgs[i].get("content") or "", "ford", kind, benchmark)
                if got is None:
                    tr["ansonly_extract_failed"] += 1
                else:
                    msgs[i]["content"] = got
                    tr["ansonly_tier"][tier] += 1
                    kept.append(i - 1)
                    tr["changed"] = True
            tr["peer_indices_kept"] = kept
        else:  # UNIFSEL is a USE contrast for FORD and is not a prompt edit
            tr["note"] = ("FORD has one peer, so there is no peer to choose between; its "
                          "selection/use factor is the final-answer rule and is applied outside "
                          "the prompt (see use_arm)")
        if instr and msgs:
            msgs[-1]["content"] = (msgs[-1].get("content") or "") + instr
        for a, b in PEER_CLAUSES:
            for m in msgs:
                if a in (m.get("content") or "") and arm == "NOPEER":
                    m["content"] = m["content"].replace(a, b)
                    tr["clauses_rewritten"].append(a[:40])
                    tr["changed"] = True
        tr["n_peers_rendered"] = 0 if arm == "NOPEER" else (
            len(tr["peer_indices_kept"]) if arm == "ANSONLY" else len(peer_msgs))
        return msgs, tr

    # ---------- MoA: numbered references inside the aggregator system prompt ----------------
    if protocol == "moa":
        for m in msgs:
            c = m.get("content") or ""
            if MOA_MARKER not in c:
                continue
            refs = moa_refs(c, observed_replies)
            tr["peer_blocks_found"] = len(refs)
            if not refs:
                continue
            hdr = c.find(MOA_REFS_HEADER)
            if arm == "NOPEER":
                head = c[:hdr].rstrip()
                m["content"] = (head.replace(
                    "You have been provided with a set of responses from various open-source "
                    "models to the latest user query. Your task is to synthesize these responses "
                    "into a single, high-quality response.",
                    "Answer the latest user query directly with a single, high-quality response.")
                    .replace("these responses", "your own reasoning"))
                tr["clauses_rewritten"].append("aggregator synthesis framing -> direct answer")
                tr["n_peers_rendered"] = 0
                tr["changed"] = True
            elif arm == "ANSONLY":
                new = c[:hdr] + MOA_REFS_HEADER
                for _s0, _e0, n, body in refs:
                    got, tier = _extract_answer(body, "moa", kind, benchmark)
                    if got is None:
                        tr["ansonly_extract_failed"] += 1
                        new += f"\n{n}. {body}"
                    else:
                        new += f"\n{n}. {got}"
                        tr["ansonly_tier"][tier] += 1
                        tr["changed"] = True
                m["content"] = new
                tr["peer_indices_kept"] = list(range(len(refs)))
                tr["n_peers_rendered"] = len(refs)
            elif arm == "UNIFSEL":
                keep = sorted(rng.sample(range(len(refs)), 1))
                new = c[:hdr] + MOA_REFS_HEADER
                for j, (_s0, _e0, _n, body) in enumerate(refs):
                    if j in keep:
                        new += f"\n{len(tr['peer_indices_kept']) + 1}. {body}"
                        tr["peer_indices_kept"].append(j)
                m["content"] = new
                tr["n_peers_rendered"] = len(keep)
                tr["changed"] = True
        return msgs, tr

    # ---------- BCCS / Du-MAD / MAD-M^2: text blocks in a user message ----------------------
    for m in msgs:
        c = m.get("content") or ""
        blocks = peer_blocks(protocol, c)
        if not blocks:
            continue
        tr["peer_blocks_found"] += len(blocks)
        if arm == "NOPEER":
            new = c
            for st, en, _body in reversed(blocks):
                new = new[:st] + new[en:]
            pref = PEER_PREFIX.get(protocol)
            if pref:
                new = new.replace(pref, "").lstrip()
            m["content"] = new
            tr["n_peers_rendered"] = 0
            tr["changed"] = True
        elif arm == "ANSONLY":
            new, kept = c, []
            for j, (st, en, body) in reversed(list(enumerate(blocks))):
                got, tier = _extract_answer(body, protocol, kind, benchmark)
                if got is None:
                    tr["ansonly_extract_failed"] += 1
                    continue
                bs = c.index(body, st, en)
                new = new[:bs] + got + new[bs + len(body):]
                kept.append(j)
                tr["ansonly_tier"][tier] += 1
                tr["changed"] = True
            m["content"] = new
            tr["peer_indices_kept"] = sorted(kept)
            tr["n_peers_rendered"] = len(blocks)
        elif arm == "UNIFSEL":
            k = max(1, len(blocks) // 2) if len(blocks) > 1 else len(blocks)
            keep = set(rng.sample(range(len(blocks)), k))
            new = c
            for j, (st, en, _body) in reversed(list(enumerate(blocks))):
                if j not in keep:
                    new = new[:st] + new[en:]
            m["content"] = new
            tr["peer_indices_kept"] = sorted(keep)
            tr["n_peers_rendered"] = len(keep)
            tr["changed"] = len(keep) != len(blocks)
    if arm == "NOPEER":
        for a, b in PEER_CLAUSES:
            for m in msgs:
                if a in (m.get("content") or ""):
                    m["content"] = m["content"].replace(a, b)
                    tr["clauses_rewritten"].append(a[:40])
                    tr["changed"] = True
    return msgs, tr


def use_arm_final_answer(protocol: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """FORD's selection/use contrast, computed from the record -- not a prompt edit.

    Released rule: stop on two consecutive agreeing predictions and take that prediction.
    Contrast: the majority over every round's predictions. Both read the same calls, so this needs
    no extra model call and is verified separately from the peer-selection arms.
    """
    preds = []
    for c in record.get("calls") or []:
        p = c.get("prediction")
        if p not in (None, "", "None"):
            preds.append(p)
    released = record.get("majority_answer")
    maj = None
    if preds:
        counts: Dict[str, int] = {}
        for p in preds:
            counts[p] = counts.get(p, 0) + 1
        maj = max(counts.items(), key=lambda kv: kv[1])[0]
    return {"released_rule_answer": released, "all_round_majority": maj,
            "n_predictions": len(preds), "rules_agree": released == maj}
