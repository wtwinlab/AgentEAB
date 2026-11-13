r"""Mixture-of-Agents — a **boundary comparison**, not a consensus mechanism.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from ..providers import TransportFailure, chat
from ..registry_live import REGISTRY

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UPSTREAM = "vendor/consensus_literature/moa"

#: `moa.py:17-20`, verbatim including the author's spelling of the variable name.
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

ANSWER_KIND = {"gsm8k": "numeric", "mmlu_pro": "mcq", "gpqa": "mcq", "strategyqa": "boolean"}
#: The released demo answers an open-ended question with no scorer. Every benchmark here needs an
#: answer the official rule can read, so a one-line answer instruction is appended and labelled.
ANSWER_INSTRUCTION = {
    "gsm8k": " State the final numeric answer at the end.",
    "mmlu_pro": " State the single letter of your choice at the end.",
    "gpqa": " State the single letter of your choice at the end.",
    "strategyqa": " State exactly 'yes' or 'no' at the end.",
}


#: A letter inside `\boxed{...}`, which `official_scorers` deliberately rejects — its `{` lookbehind
#: exists to stop a LaTeX subscript being read as an option.
_BOXED = re.compile(r"\\boxed\{\s*\(?([A-J])\)?\s*\}")


def extract(text: str, kind: str) -> str:
    r"""The study's own extraction. **Not a second implementation of it.**
    """
    from .official_scorers import last_number, letter_answer  # noqa: PLC0415
    if kind == "numeric":
        n = last_number(text)
        return "" if n is None else str(round(n, 4))
    if kind == "mcq":
        return letter_answer(text, 10) or ""
    if kind == "boolean":
        toks = re.findall(r"[a-z]+", (text or "").lower())
        for t in reversed(toks):
            if t in ("yes", "true"):
                return "yes"
            if t in ("no", "false"):
                return "no"
        return ""
    return " ".join((text or "").split())


def run_one(question: str, *, task_id: str, benchmark: str, references: Sequence[str],
            aggregator: str, max_tokens: int, timeout_s: float, retries: int,
            temperature: float) -> Dict[str, Any]:
    kind = ANSWER_KIND[benchmark]
    prompt = question + ANSWER_INSTRUCTION[benchmark]
    calls: List[Dict[str, Any]] = []
    order: List[Dict[str, Any]] = []
    t0 = time.time()
    outputs: List[str] = []

    for i, alias in enumerate(references):
        messages = [{"role": "user", "content": prompt}]
        text, rec = chat(registry=REGISTRY, alias=alias, messages=messages,
                         temperature=temperature, top_p=1.0, max_tokens=max_tokens,
                         retries=retries, timeout_s=timeout_s)
        outputs.append(text)
        calls.append({
            "seq": i, "role": "reference", "agent_index": i, "alias": alias,
            "attempt": rec.attempt, "messages_sent": [dict(m) for m in messages],
            "raw_reply": text, "extracted_answer": extract(text, kind),
            "request_sha": rec.request_sha, "served_model": rec.served_model,
            "fingerprint": rec.fingerprint, "finish_reason": rec.finish_reason,
            "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
            "latency_s": rec.latency_s, "usd": rec.usd,
        })
        order.append({"seq": i, "role": "reference", "alias": alias})

    # The aggregator sees every reference output once. This is the ONLY interaction in the protocol.
    system = AGGREGATOR_SYSTEM_PROMPT + "".join(
        f"\n{i + 1}. {o}" for i, o in enumerate(outputs))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text, rec = chat(registry=REGISTRY, alias=aggregator, messages=messages,
                     temperature=temperature, top_p=1.0, max_tokens=max_tokens,
                     retries=retries, timeout_s=timeout_s)
    calls.append({
        "seq": len(references), "role": "aggregator", "agent_index": None, "alias": aggregator,
        "attempt": rec.attempt, "messages_sent": [dict(m) for m in messages],
        "raw_reply": text, "extracted_answer": extract(text, kind),
        "request_sha": rec.request_sha, "served_model": rec.served_model,
        "fingerprint": rec.fingerprint, "finish_reason": rec.finish_reason,
        "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
        "latency_s": rec.latency_s, "usd": rec.usd,
    })
    order.append({"seq": len(references), "role": "aggregator", "alias": aggregator})

    final = extract(text, kind)
    ref_answers = [extract(o, kind) for o in outputs]
    delivered = final != ""
    # The second reading, recorded and never substituted. `debate_or_vote` carries its two parses
    # the same way: both are reported and neither is silently preferred.
    boxed = (_BOXED.findall(text or "")[-1] if kind == "mcq" and _BOXED.search(text or "")
             else None)
    return {
        "run_id": f"moa-{aggregator}-{task_id}-r0",
        "example_id": task_id, "mechanism": "moa", "repeat": 0,
        "model_alias": "+".join(references) + f">{aggregator}",
        "roster": list(references), "aggregator": aggregator,
        "arm_role": "boundary",
        "arm_role_note": ("Mixture-of-Agents is a boundary comparison, not a consensus mechanism: "
                          "no reference model ever sees another's output, there is no round, no "
                          "stance and no stop rule. It is never counted among the mechanisms"),
        "question": question, "benchmark": benchmark, "answer_kind": kind,
        "is_task_adaptation": True,
        "adaptation_note": ("the released demo answers one open-ended question with no scorer; a "
                            "one-line answer instruction is appended so the official rule can read "
                            "the output, and that is labelled rather than glossed"),
        "n_references": len(references),
        "n_references_released": 4,
        "reference_count_note": ("the release lists four reference models; three families are what "
                                 "this study serves, and the count is recorded"),
        "per_agent_final_answer": [a if a else None for a in ref_answers],
        "n_distinct_final_answers": len({a for a in ref_answers if a}),
        "references_unanimous": len({a for a in ref_answers if a}) == 1 and all(ref_answers),
        "majority_answer": final if delivered else None,
        # A second, more permissive reading of the SAME text, reported as a sensitivity.
        "majority_answer_boxed": boxed,
        "parses_agree": (boxed is None) or (boxed == final),
        "boxed_note": ("official_scorers rejects a letter inside `\\boxed{}` on purpose, so this "
                       "field records where the aggregator put its answer there. Reported, not "
                       "substituted"),
        "aggregator_matches_a_reference": final in [a for a in ref_answers if a],
        # Nothing here has a rule to satisfy or an acceptance to express.
        "rule_satisfied": None,
        "rule_note": ("there is no stop rule: every reference answers once and the aggregator "
                      "synthesises once"),
        "accepted_delivery": "not-measured",
        "accepted_delivery_note": ("no reference model is asked whether it accepts anything, and "
                                   "the aggregator is a writer rather than a participant expressing "
                                   "acceptance. Absence of a verdict is not a refusal"),
        "absence_cause": (None if delivered else
                          "the aggregator's output contained no answer the rule can read"),
        "native_stop_reason": "single pass: references then one aggregation",
        "calls": calls, "order": order,
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0 for c in calls), 6),
        "wall_s": round(time.time() - t0, 2),
    }


def _assert_transcription() -> None:
    src = os.path.join(REPO, "vendor", "consensus_literature", "moa", "moa.py")
    # **Not conditional on the file existing.** Guarding this with `if os.path.exists` made the whole
    # verbatim check pass silently in any tree without `vendor/` — including a packaged artifact,
    # which is the one place a reader most needs the transcription to be checked.
    text = open(src, encoding="utf-8").read()
    assert AGGREGATOR_SYSTEM_PROMPT in text, (
        "the aggregator system prompt does not appear verbatim in moa.py")
    assert "temperature=0.7" in text, "the released temperature is 0.7"
    assert "max_tokens=512" in text, "the released max_tokens is 512"
    assert AGGREGATOR_SYSTEM_PROMPT.rstrip().endswith("Responses from models:")
    assert extract("so the answer is (C)", "mcq") == "C"
    assert extract("Avogadro's number", "mcq") == "", "a letter inside a word is not an answer"
    # Three cases a hand-rolled pattern gets wrong; each one can fail.
    assert extract("The answer is (B). Note that F = ma.", "mcq") == "B", "a formula is not an answer"
    assert extract("The answer is B, which is a correct choice.", "mcq") == "B", \
        "the article 'a' is not an answer; this is why the text must not be case-folded"
    assert extract("so the answer is C = 5", "mcq") == "", "a letter followed by = is not an answer"
    assert extract("the total is 42 apples", "numeric") == "42.0"
    assert extract("the change is −5 apples", "numeric") == "-5.0", \
        "U+2212 is a minus sign, and official_scorers.last_number reads it"
    assert extract("I would say no.", "boolean") == "no"
    assert extract("", "numeric") == ""
    assert _BOXED.findall(r"therefore \boxed{G}") == ["G"]


_assert_transcription()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--alias", required=True,
                    help="reference aliases, comma separated")
    ap.add_argument("--aggregator", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)    # moa.py:30
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="the release uses 512; 1024 matches this study's frozen budget")
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    refs = tuple(x.strip() for x in a.alias.split(",") if x.strip())
    if len(refs) < 2:
        print(f"  REFUSING: Mixture-of-Agents needs at least two reference models, got {refs}")
        return 2

    bman = a.block.replace(".json", "") + ".manifest.json"
    bench = (json.load(open(bman, encoding="utf-8")).get("benchmark")
             if os.path.exists(bman) else os.path.basename(a.block).split("_block")[0])
    if bench not in ANSWER_KIND:
        print(f"  REFUSING: no declared answer kind for benchmark {bench!r}")
        return 1

    from ._runner import run_block  # noqa: PLC0415
    return run_block(
        block_path=a.block, out_path=a.out, adapter_file=os.path.abspath(__file__),
        # Deduplicated: the release reuses one reference as the aggregator, so passing both put
        # `qwen25-7b` in the manifest's roster twice and made `model_name` a four-way join against a
        # recorded `n_references: 3`.
        mechanism="moa", aliases=list(dict.fromkeys(list(refs) + [a.aggregator])),
        limit=a.limit, upstream=UPSTREAM,
        per_task=lambda q, tid, task: run_one(
            q, task_id=tid, benchmark=bench, references=refs, aggregator=a.aggregator,
            max_tokens=a.max_tokens, timeout_s=a.timeout_s, retries=a.retries,
            temperature=a.temperature),
        manifest_extra={
            "arm_role": "boundary",
            "arm_role_note": ("a boundary comparison, not a consensus mechanism: the references "
                              "never see each other and nothing is accepted. Never counted among "
                              "the eight mechanisms"),
            "references": list(refs), "aggregator": a.aggregator,
            "n_references": len(refs), "n_references_released": 4,
            "temperature": a.temperature, "max_tokens": a.max_tokens,
            "released_config": "moa.py: 4 reference models, temperature 0.7, max_tokens 512, one of "
                               "them reused as the aggregator",
            "stop_rule": "none; a single pass",
            "is_task_adaptation": True,
        })


if __name__ == "__main__":
    sys.exit(main())
