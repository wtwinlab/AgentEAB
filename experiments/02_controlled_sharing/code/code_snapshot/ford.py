r"""FORD — "Examining Inter-Consistency of LLMs Collaboration via Debate" (Xiong et al., EMNLP 2023).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ..providers import TransportFailure, chat
from ..registry_live import REGISTRY

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UPSTREAM = "_external/mechanism_screening_2026-10-08/repos/ford @ 14988c9"

#: `debate_chatgpt.py:120-127`, verbatim. `strategyqa` is the authors' own revised string.
INSTRUCTION_STRATEGYQA = (
    "Read the question and our opinions carefully to get the key points of the question then "
    "consider the context to determine if the statement is really true or false. The question is "
    "hypothetical, do not introduce any new situation into the question. Use your general knowledge "
    "and understanding to provide your response with \"Answer: (yes|no) is more plausible.\" and "
    "then briefly explain your reasoning, starting with \"Explanation: \". Remember, your answer "
    "should be either \"yes\" or \"no\".")
INSTRUCTION_CSQA = (
    "Do you think I am more reasonable or you have a different answer? Please give your final "
    "answer starting with \"Answer: (A|B|C|D|E) is more plausible.\" and explain very shortly "
    "starting with \"Explanation: \". You should choose only one answer.")
INSTRUCTION_TWO_CHOICE = (
    "Do you think I am more reasonable? Please give your final answer starting with \"Answer: (A|B) "
    "is more plausible.\" and explain very shortly starting with \"Explanation: \". You should "
    "choose only one answer.")
#: GSM8K is not a FORD dataset; the sentence keeps the released shape and is labelled.
INSTRUCTION_NUMERIC = (
    "Do you think I am more reasonable? Please give your final answer starting with \"Answer: \" "
    "followed by the number, and explain very shortly starting with \"Explanation: \". You should "
    "give only one number.")

#: `debate_chatgpt.py:129`, verbatim.
PATTERN = re.compile(r"\(?(A|B|C|D|E|F|G|yes|no|Yes|No)\)?")
ADAPTED = {"gsm8k", "mmlu_pro", "gpqa", "squad_v2"}


def instruction(benchmark: str, n_options: int = 4) -> str:
    if benchmark == "strategyqa":
        return INSTRUCTION_STRATEGYQA
    if benchmark == "gsm8k":
        return INSTRUCTION_NUMERIC
    if benchmark in ("mmlu_pro", "gpqa"):
        if n_options <= 2:
            return INSTRUCTION_TWO_CHOICE
        letters = "|".join(chr(65 + i) for i in range(min(n_options, 10)))
        return (f"Do you think I am more reasonable? Please give your final answer starting with "
                f"\"Answer: ({letters}) is more plausible.\" and explain very shortly starting with "
                f"\"Explanation: \". You should choose only one answer.")
    return INSTRUCTION_CSQA


def _lower_for_strategyqa(pred: str, benchmark: str) -> str:
    """`get_prediction_explanation` ends with `if task == 'strategyqa': prediction = pred.lower()`.
    """
    return pred.lower() if benchmark == "strategyqa" else pred


def parse_prediction(text: str, benchmark: str) -> Tuple[str, str]:
    """`get_prediction_explanation`, transcribed. `"None"` on failure, as upstream uses."""
    t = (text or "").strip()
    try:
        after = t.split("Answer:")[1]
        prediction, explanation = after.split("Explanation: ")
        if benchmark == "gsm8k":
            nums = re.findall(r"[-+]?\d*\.?\d+", prediction.replace(",", "").replace("$", ""))
            pred = str(round(float(nums[0]), 4)) if nums else "None"
        else:
            hits = PATTERN.findall(prediction.replace("Answer", "answer"))
            pred = hits[0] if hits else "None"
        return _lower_for_strategyqa(pred, benchmark), explanation.strip()
    except Exception:                             # noqa: BLE001 — upstream's bare except, kept
        # Upstream lowers **after** the try/except, so its failure sentinel is `"none"`, not
        # `"None"`. Lowering only inside the try would leave the two paths inconsistent.
        return _lower_for_strategyqa("None", benchmark), t


def flip_roles(transcript: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """`debate_chatgpt.py:144-146`: the answering model sees its own turns as `assistant`."""
    return [{"role": ("assistant" if m["role"] == "user" else "user"), "content": m["content"]}
            for m in transcript]


def run_one(question: str, *, task_id: str, benchmark: str, aliases: Tuple[str, str],
            max_rounds: int, max_tokens: int, timeout_s: float, retries: int,
            n_options: int) -> Dict[str, Any]:
    """One task. Two models alternate until their predictions agree or the budget runs out."""
    instr = instruction(benchmark, n_options)
    calls: List[Dict[str, Any]] = []
    order: List[Dict[str, Any]] = []
    predictions: List[str] = []
    # The shared transcript, in the first speaker's frame. It is flipped per call.
    transcript: List[Dict[str, str]] = [{"role": "user", "content": question}]
    t0 = time.time()
    stop_reason = f"budget-exhausted after {max_rounds} turns"
    stopped_at: Optional[int] = None

    for turn in range(max_rounds):
        speaker = turn % 2
        alias = aliases[speaker]
        # The instruction is appended for this call only and is not stored, exactly as upstream
        # appends it and then strips it back out of the saved transcript.
        view = flip_roles(transcript) if speaker == 1 else [dict(m) for m in transcript]
        view[-1] = dict(view[-1], content=view[-1]["content"] + f"\n{instr}")
        text, rec = chat(registry=REGISTRY, alias=alias, messages=view,
                         temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                         retries=retries, timeout_s=timeout_s)
        pred, expl = parse_prediction(text, benchmark)
        predictions.append(pred)
        transcript.append({"role": "assistant" if speaker == 0 else "user",
                           "content": expl.strip() or text.strip()})
        calls.append({
            "seq": turn, "turn": turn, "agent_index": speaker, "alias": alias,
            "attempt": rec.attempt, "messages_sent": [dict(m) for m in view],
            "raw_reply": text, "prediction": pred, "explanation": expl,
            "request_sha": rec.request_sha, "served_model": rec.served_model,
            "fingerprint": rec.fingerprint, "finish_reason": rec.finish_reason,
            "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
            "latency_s": rec.latency_s, "usd": rec.usd,
        })
        order.append({"seq": turn, "turn": turn, "agent_index": speaker, "alias": alias})
        # The released stop rule: this prediction equals the previous speaker's.
        if len(predictions) >= 2 and pred != "None" and pred == predictions[-2]:
            stop_reason = "two consecutive agreeing predictions"
            stopped_at = turn
            break

    agreed = stop_reason == "two consecutive agreeing predictions"
    # **The released comparison is case-sensitive, and that matters.** Upstream's pattern captures
    # `yes|no|Yes|No` as four distinct strings and then compares `prediction == c['P'][-2]`
    # verbatim, so two agents that both say "no" — one capitalised — never register as agreeing and
    # the stop rule does not fire.
    fold = [p.casefold() for p in predictions]
    fold_turn: Optional[int] = None
    for i in range(1, len(fold)):
        if fold[i] != "none" and fold[i] == fold[i - 1]:
            fold_turn = i
            break
    final = predictions[-1] if predictions else "None"
    delivered = final != "None"
    return {
        "run_id": f"ford-{aliases[0]}+{aliases[1]}-{task_id}-r0",
        "example_id": task_id, "mechanism": "ford", "repeat": 0,
        "model_alias": f"{aliases[0]}+{aliases[1]}",
        "roster": list(aliases),
        "roster_is_heterogeneous": aliases[0] != aliases[1],
        "question": question, "benchmark": benchmark,
        "is_task_adaptation": benchmark in ADAPTED,
        "adaptation_note": ("FORD's released instructions are choice-style; StrategyQA uses the "
                            "authors' own string, and this benchmark uses a labelled variant"
                            if benchmark in ADAPTED else None),
        "max_rounds": max_rounds, "turns_taken": len(predictions),
        "per_turn_prediction": predictions,
        "per_agent_final_answer": [predictions[-2] if len(predictions) >= 2 else None,
                                   final if delivered else None],
        "n_distinct_final_answers": len({p for p in predictions[-2:] if p != "None"}),
        "majority_answer": final if delivered else None,
        # The stop rule IS agreement, so rule satisfaction is measured and is not a round budget.
        "rule_satisfied": agreed,
        "rule_note": ("the released stop rule: two consecutive agreeing predictions, compared "
                      "case-sensitively exactly as upstream does"),
        "stopped_at_turn": stopped_at,
        "rule_satisfied_casefold": fold_turn is not None,
        "stopped_at_turn_casefold": fold_turn,
        "casefold_note": ("the same rule with the predictions case-folded. Upstream compares raw "
                          "strings, so 'no' and 'No' are different predictions; this is the "
                          "recorded alternative, not a correction applied to the primary"),
        "native_stop_reason": stop_reason,
        # The agents assert which answer is "more plausible"; they are not asked to accept a
        # delivered artifact. Agreement is not acceptance and is reported separately.
        "accepted_delivery": "not-measured",
        "accepted_delivery_note": ("FORD asks each model whether the other is more reasonable and "
                                   "stops when two consecutive predictions agree. It solicits no "
                                   "acceptance of a delivered artifact; agreement between two "
                                   "predictions is a different dimension and is reported as "
                                   "`rule_satisfied`"),
        "absence_cause": (None if delivered else
                          "no turn produced a parseable 'Answer: ... Explanation: ...' reply"),
        "calls": calls, "order": order,
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0 for c in calls), 6),
        "wall_s": round(time.time() - t0, 2),
    }


def _assert_transcription() -> None:
    assert INSTRUCTION_TWO_CHOICE == (
        "Do you think I am more reasonable? Please give your final answer starting with \"Answer: "
        "(A|B) is more plausible.\" and explain very shortly starting with \"Explanation: \". You "
        "should choose only one answer."), "two-choice instruction drifted"
    assert INSTRUCTION_STRATEGYQA.startswith(
        "Read the question and our opinions carefully to get the key points of the question then "
        "consider the context"), "strategyqa instruction drifted"
    assert "Remember, your answer should be either \"yes\" or \"no\"." in INSTRUCTION_STRATEGYQA
    assert PATTERN.findall("(B) is more plausible") == ["B"]
    assert PATTERN.findall("yes is more plausible") == ["yes"]
    p, e = parse_prediction("Answer: (C) is more plausible.\nExplanation: because X.", "gpqa")
    assert (p, e) == ("C", "because X."), (p, e)
    p2, _ = parse_prediction("I have no idea.", "gpqa")
    assert p2 == "None", "an unparseable reply is None, as upstream uses"
    p3, _ = parse_prediction("Answer: 42\nExplanation: arithmetic.", "gsm8k")
    assert p3 == "42.0", p3
    flipped = flip_roles([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    assert [m["role"] for m in flipped] == ["assistant", "user"], "role flip drifted"
    assert instruction("strategyqa") is INSTRUCTION_STRATEGYQA
    assert "(A|B|C|D|E|F|G|H|I|J)" in instruction("mmlu_pro", 10), instruction("mmlu_pro", 10)


_assert_transcription()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--alias", required=True,
                    help="the pair, comma separated; FORD needs two DIFFERENT models")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    pair = tuple(x.strip() for x in a.alias.split(",") if x.strip())
    if len(pair) != 2:
        print(f"  REFUSING: FORD needs exactly two aliases, got {pair}")
        return 2
    if pair[0] == pair[1]:
        print("  REFUSING: FORD measures inter-consistency between two DIFFERENT models; "
              "one backbone against itself is a different question and is not this method")
        return 2

    from ._runner import run_block  # noqa: PLC0415
    return run_block(
        block_path=a.block, out_path=a.out, adapter_file=os.path.abspath(__file__),
        mechanism="ford", aliases=pair, limit=a.limit, upstream=UPSTREAM,
        per_task=lambda q, tid, task: run_one(
            q, task_id=tid,
            benchmark=(json.load(open(a.block.replace(".json", "") + ".manifest.json",
                                      encoding="utf-8")).get("benchmark")
                       if os.path.exists(a.block.replace(".json", "") + ".manifest.json")
                       else os.path.basename(a.block).split("_block")[0]),
            aliases=pair, max_rounds=a.max_rounds, max_tokens=a.max_tokens,
            timeout_s=a.timeout_s, retries=a.retries,
            n_options=(task.get("metadata") or {}).get("n_options", 4)),
        manifest_extra={
            "roster_is_heterogeneous": True,
            "roster_note": ("FORD measures inter-consistency between two different LLMs; the pair "
                            "is part of this cell's identity and is never homogenised"),
            "max_rounds": a.max_rounds, "temperature": 0.0, "max_tokens": a.max_tokens,
            "stop_rule": "two consecutive agreeing predictions (debate_chatgpt.py:168)",
            "is_task_adaptation": None,
            "note": ("FORD as released: dataset instruction strings verbatim, roles flipped per "
                     "call, agreement-based stop. StrategyQA uses the authors' own string."),
        })


if __name__ == "__main__":
    sys.exit(main())
