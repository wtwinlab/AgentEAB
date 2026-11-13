r"""ReConcile — round-table conference among heterogeneous agents, as released.

Source `vendor/reconcile`. Prompts, the confidence binning and the weighted vote are transcribed from
`generation.py` and `utils.py`; the assertions at the bottom check them.

## The roster stays heterogeneous, and is frozen by name

ReConcile's point is a **round table of different model families** — the released code runs
ChatGPT, Claude 2 and Bard/PaLM, and each agent is shown the *other two's* convincing samples.
Homogenising it and still calling it ReConcile is exactly what the authorisation forbids, so the
roster is part of the cell identity:

    llama3-8b-instruct  +  qwen2.5-7b-instruct  +  gemma4-26b-a4b-it

Three distinct families on local endpoints. This is an **open-weight backend adaptation**, labelled
as one, and it is never described as a reproduction of the paper's numbers.

## Coverage is limited by data the mechanism needs, not by preference

Each agent is primed with **convincing samples** — human-authored convincing explanations that ship
per dataset in `convincing/{Aqua,ECQA,GSM8k,SQA}/{chatgpt,claude,bard}.json`. Those four are the only
sets that exist. Two of them are ours: **GSM8k** and **SQA (StrategyQA)**. MMLU-Pro and GPQA have
none, and writing them would be **fabricating the mechanism's own input**, so those arms are
excluded with evidence.

The three released sample sets are assigned to the three roster members by a fixed declared mapping
(`SAMPLE_SOURCE_FOR`), recorded in every manifest. The samples are convincing explanations about
*training* examples, so which model's run produced them does not change what the receiving agent is
shown; the mapping is fixed so two cells cannot differ on it silently.

## It has a real stop rule

`utils.py:174-183` builds `debate_prompt` **only when the agents disagree** (`len(vote) > 1`), and
the debate functions are guarded by `len(s['debate_prompt_r'])`. So **unanimity of answers ends the
debate for that sample** — a genuine stop rule, not a round budget.

## Two issues recorded rather than inherited

* **the confidence scale is never stated.** The required output is
  `{"reasoning": "", "answer": "", "confidence_level": ""}` with no range given anywhere in the
  prompt, yet `trans_confidence` bins on `x <= 0.6 / 0.6 < x < 0.8 / 0.8 <= x < 0.9 /
  0.9 <= x < 1 / x == 1`. Any value **above 1** — a model answering "8" on a 0-10 scale, which the
  prompt invites — falls through every branch and returns `None`, and the next line does
  `None + 1e-5`: a **TypeError** in the weighted vote. The adapter keeps the prompt verbatim,
  records every out-of-range or unparseable confidence as `unusable`, and reports the rate instead
  of crashing or silently clamping.
* **the weighted and unweighted winners can differ**, and the released code computes both
  (`weighted_max` and `majority_ans`). Both are recorded per task so neither has to be re-run.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..providers import TransportFailure, chat
from ..registry_live import REGISTRY

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UPSTREAM = "vendor/reconcile"
CONVINCING = os.path.join(REPO, "vendor", "reconcile", "convincing")

#: `generation.py:95-98`, verbatim.
INSTRUC_REVIEW = ("\n\nCarefully review the following solutions from other agents as additional "
                  "information, and provide your own answer and step-by-step reasoning to the "
                  "question.")
INSTRUC_STANCE = "Clearly states that which pointview do you agree or disagree and why.\n\n"
INSTRUC_JSON = ("Output your answer in json format, with the format as follows: "
                "{\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. "
                "Please strictly output in JSON format.")

#: `utils.py:164`, verbatim.
EXP_TEMPLATE = ("I think the answer is {answer} because {reasoning} My confidence level is "
                "{confidence_level}.")

#: Our benchmarks to ReConcile's own dataset names, which decide which convincing samples exist.
DATASET_FOR = {"gsm8k": "GSM8k", "strategyqa": "SQA"}
#: The fixed mapping from roster position to released sample set. Declared so it cannot drift.
SAMPLE_SOURCE_FOR = {0: "chatgpt", 1: "claude", 2: "bard"}
ANSWER_KIND = {"gsm8k": "numeric", "strategyqa": "boolean"}


def load_convincing(benchmark: str) -> Dict[str, List[Dict[str, Any]]]:
    """The released convincing samples for this dataset, or a refusal if there are none."""
    name = DATASET_FOR.get(benchmark)
    if name is None:
        raise ValueError(
            f"ReConcile has no convincing samples for {benchmark!r}; the released sets are "
            f"{sorted(DATASET_FOR.values())} and authoring new ones would fabricate the "
            f"mechanism's input")
    out: Dict[str, List[Dict[str, Any]]] = {}
    for src in SAMPLE_SOURCE_FOR.values():
        path = os.path.join(CONVINCING, name, f"{src}.json")
        with open(path, encoding="utf-8") as f:
            out[src] = json.load(f)
    return out


def convincing_text(samples: Sequence[Dict[str, Any]]) -> str:
    """One convincing sample rendered as the released context does: question, then explanation."""
    parts: List[str] = []
    for s in samples:
        ts = s.get("train_sample") or {}
        q = ts.get("question", "")
        opts = ts.get("options") or []
        ans = ts.get("answer", "")
        expl = ts.get("gold_explanation", "")
        body = f"Question: {q}"
        if opts:
            body += "\nOptions: " + " ".join(str(o) for o in opts)
        body += f"\nAnswer: {ans}\nExplanation: {expl}"
        parts.append(body)
    return "\n\n".join(parts)


def trans_confidence(x: Any) -> Optional[float]:
    """`utils.py:145-151`, transcribed. **Returns `None` above 1**, which upstream then adds to."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v <= 0.6:
        return 0.1
    if 0.8 > v > 0.6:
        return 0.3
    if 0.9 > v >= 0.8:
        return 0.5
    if 1 > v >= 0.9:
        return 0.8
    if v == 1:
        return 1.0
    # No branch covers v > 1. The released function falls off the end and returns None.
    return None


def parse_json_reply(text: str) -> Optional[Dict[str, Any]]:
    """The released contract: the reply must contain a JSON object with the three keys."""
    t = text or ""
    if "{" not in t or "}" not in t:
        return None
    start, end = t.find("{"), t.rfind("}")
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def normalise_answer(answer: Any, benchmark: str) -> str:
    """`generation.py:136-141`: SQA lowercases, Aqua uppercases, GSM8k/ECQA keep the number."""
    a = str(answer if answer is not None else "").strip()
    if benchmark == "strategyqa":
        low = a.lower()
        return "yes" if low.startswith("y") else "no" if low.startswith("n") else ""
    m = re.findall(r"[-+]?\d*\.?\d+", a.replace(",", "").replace("$", ""))
    if not m:
        return ""
    try:
        return str(round(float(m[-1]), 4))
    except ValueError:
        return ""


def weighted_vote(answers: Sequence[str], confidences: Sequence[Any]
                  ) -> Tuple[Dict[str, float], Optional[str], List[Dict[str, Any]]]:
    """`utils.py:165-172`: per-answer sum of binned confidence, `+1e-5` on first occurrence."""
    tally: Dict[str, float] = {}
    unusable: List[Dict[str, Any]] = []
    for ans, conf in zip(answers, confidences):
        if ans == "":
            continue
        w = trans_confidence(conf)
        if w is None:
            # The released code would do `None + 1e-5` here and raise. Recorded instead, with the
            # value that caused it, so the rate is measurable.
            unusable.append({"answer": ans, "confidence_level": conf,
                             "reason": "trans_confidence returns None for values above 1 or "
                                       "unparseable text; upstream then adds it and raises"})
            continue
        tally[ans] = tally.get(ans, 0.0) + w + (1e-5 if ans not in tally else 0.0)
    winner = max(tally, key=tally.get) if tally else None
    return tally, winner, unusable


def debate_prompt(answers: Sequence[str], explanations: Sequence[str]) -> str:
    """`utils.py:174-183`: built **only** when the agents disagree. Empty means the debate ends."""
    present = [a for a in answers if a != ""]
    vote = collections.Counter(present).most_common(2)
    if len(vote) <= 1:
        return ""
    out = ""
    for value, count in vote:
        out += f"There are {count} agents think the answer is {value}. "
        group = [e for a, e in zip(answers, explanations) if a == value]
        out += "\n".join("One agent solution: " + g for g in group) + "\n\n"
    return out


def run_one(question: str, *, task_id: str, benchmark: str, roster: Sequence[str],
            convincing: Dict[str, List[Dict[str, Any]]], rounds: int, max_tokens: int,
            timeout_s: float, retries: int) -> Dict[str, Any]:
    kind = ANSWER_KIND[benchmark]
    calls: List[Dict[str, Any]] = []
    order: List[Dict[str, Any]] = []
    per_round: List[Dict[str, Any]] = []
    t0 = time.time()
    seq = 0
    prompt_from_last_round = ""
    stop_reason = f"round-budget-exhausted after {rounds} rounds"
    stopped_at: Optional[int] = None
    unusable_confidences: List[Dict[str, Any]] = []

    for rnd in range(rounds):
        answers: List[str] = []
        confidences: List[Any] = []
        explanations: List[str] = []
        for i, alias in enumerate(roster):
            # Each agent is primed with the OTHER TWO sources' convincing samples.
            others = [SAMPLE_SOURCE_FOR[j] for j in SAMPLE_SOURCE_FOR if j != i]
            primer = convincing_text([s for o in others for s in convincing[o]])
            content = f"{primer}\n\n{question}" if primer else question
            if rnd > 0 and prompt_from_last_round:
                content += (INSTRUC_REVIEW + " " + INSTRUC_STANCE + " "
                            + prompt_from_last_round + " " + INSTRUC_JSON)
            else:
                content += " " + INSTRUC_JSON
            messages = [{"role": "user", "content": content}]
            text, rec = chat(registry=REGISTRY, alias=alias, messages=messages,
                             temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                             retries=retries, timeout_s=timeout_s)
            obj = parse_json_reply(text)
            ans = normalise_answer((obj or {}).get("answer"), benchmark)
            conf = (obj or {}).get("confidence_level")
            reasoning = str((obj or {}).get("reasoning") or "")
            answers.append(ans)
            confidences.append(conf)
            explanations.append(EXP_TEMPLATE.format(answer=(obj or {}).get("answer"),
                                                    reasoning=reasoning,
                                                    confidence_level=conf))
            calls.append({
                "seq": seq, "round": rnd, "agent_index": i, "alias": alias,
                "attempt": rec.attempt, "messages_sent": [dict(m) for m in messages],
                "raw_reply": text, "parsed_json": obj,
                "answer": ans, "confidence_level": conf,
                "confidence_weight": trans_confidence(conf),
                "json_contract_met": obj is not None and set(obj) >= {
                    "reasoning", "answer", "confidence_level"},
                "request_sha": rec.request_sha, "served_model": rec.served_model,
                "fingerprint": rec.fingerprint, "finish_reason": rec.finish_reason,
                "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
                "latency_s": rec.latency_s, "usd": rec.usd,
            })
            order.append({"seq": seq, "round": rnd, "agent_index": i, "alias": alias})
            seq += 1
        tally, winner, unusable = weighted_vote(answers, confidences)
        unusable_confidences.extend([dict(u, round=rnd) for u in unusable])
        present = [a for a in answers if a != ""]
        majority = collections.Counter(present).most_common(1)[0][0] if present else None
        nxt = debate_prompt(answers, explanations)
        per_round.append({
            "round": rnd, "answers": answers, "confidences": confidences,
            "weighted_vote": tally, "weighted_max": winner, "majority_answer": majority,
            "unusable_confidences": unusable,
            "agents_unanimous": len(set(present)) == 1 and len(present) == len(roster),
            "debate_prompt_was_built": bool(nxt),
        })
        if not nxt:
            # The released stop rule: no debate prompt means the agents agreed, and the guard
            # `len(debate_prompt)` stops asking them.
            stop_reason = ("agents unanimous, so no debate prompt was built"
                           if present else "no agent produced a readable answer")
            stopped_at = rnd
            break
        prompt_from_last_round = nxt

    final = per_round[-1]
    delivered = final["weighted_max"] is not None
    return {
        "run_id": f"reconcile-{'+'.join(roster)}-{task_id}-r0",
        "example_id": task_id, "mechanism": "reconcile", "repeat": 0,
        "model_alias": "+".join(roster), "roster": list(roster),
        "roster_is_heterogeneous": len(set(roster)) == len(roster),
        "sample_source_mapping": {str(k): v for k, v in SAMPLE_SOURCE_FOR.items()},
        "question": question, "benchmark": benchmark, "answer_kind": kind,
        "is_task_adaptation": False,
        "adaptation_note": ("the backend is open-weight rather than ChatGPT/Claude/Bard; the "
                            "dataset and its convincing samples are the released ones"),
        "rounds_configured": rounds, "rounds_run": len(per_round),
        "per_agent_final_answer": [a if a else None for a in final["answers"]],
        "per_agent_final_confidence": final["confidences"],
        "n_distinct_final_answers": len({a for a in final["answers"] if a}),
        # Both winners, because the released code computes both and they can differ.
        "majority_answer": final["weighted_max"] if delivered else None,
        "unweighted_majority_answer": final["majority_answer"],
        "weighted_and_unweighted_agree": final["weighted_max"] == final["majority_answer"],
        "weighted_vote": final["weighted_vote"],
        # Unanimity of answers IS the stop rule here.
        "rule_satisfied": stop_reason.startswith("agents unanimous"),
        "rule_note": ("utils.py builds a debate prompt only when the agents disagree, and the "
                      "debate functions are guarded by its length, so unanimity ends the debate"),
        "stopped_at_round": stopped_at,
        "native_stop_reason": stop_reason,
        # Each agent states a confidence; none is asked to accept a delivered artifact.
        "accepted_delivery": "not-measured",
        "accepted_delivery_note": ("ReConcile elicits an answer, a reasoning and a confidence, and "
                                   "aggregates by confidence-weighted vote. It solicits no "
                                   "acceptance of a delivered artifact; a confidence is not a "
                                   "stance and absence of a verdict is not a refusal"),
        "absence_cause": (None if delivered else
                          "no agent produced a usable answer with a bin-able confidence"),
        "unusable_confidences": unusable_confidences,
        "n_unusable_confidences": len(unusable_confidences),
        "rounds_detail": per_round,
        "calls": calls, "order": order,
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0 for c in calls), 6),
        "wall_s": round(time.time() - t0, 2),
    }


def _assert_transcription() -> None:
    assert INSTRUC_JSON == ("Output your answer in json format, with the format as follows: "
                            "{\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. "
                            "Please strictly output in JSON format."), "json instruction drifted"
    assert INSTRUC_REVIEW.startswith("\n\nCarefully review the following solutions from other "
                                     "agents as additional information"), "review drifted"
    assert INSTRUC_STANCE == ("Clearly states that which pointview do you agree or disagree and "
                              "why.\n\n"), "stance instruction drifted"
    # The released confidence bins, including the hole above 1.
    assert trans_confidence(0.5) == 0.1 and trans_confidence(0.6) == 0.1
    assert trans_confidence(0.7) == 0.3
    assert trans_confidence(0.8) == 0.5 and trans_confidence(0.85) == 0.5
    assert trans_confidence(0.9) == 0.8 and trans_confidence(0.95) == 0.8
    assert trans_confidence(1) == 1.0
    assert trans_confidence(8) is None, (
        "values above 1 fall through every branch; upstream then does None + 1e-5 and raises. "
        "The prompt never states the scale, so an 8 is exactly what a model may answer")
    assert trans_confidence("high") is None
    # The weighted vote keeps going where upstream would crash, and records why.
    tally, winner, unusable = weighted_vote(["yes", "no", "yes"], [1, 8, 0.95])
    assert winner == "yes" and len(unusable) == 1, (tally, winner, unusable)
    assert abs(tally["yes"] - (1.0 + 1e-5 + 0.8)) < 1e-9, tally
    # The stop rule: a debate prompt exists only on disagreement.
    assert debate_prompt(["yes", "yes", "yes"], ["a", "b", "c"]) == "", "unanimity must stop it"
    assert "There are 2 agents think the answer is yes." in debate_prompt(
        ["yes", "yes", "no"], ["a", "b", "c"])
    assert normalise_answer("Yes, definitely", "strategyqa") == "yes"
    assert normalise_answer("the answer is 12 apples", "gsm8k") == "12.0"
    assert parse_json_reply('x {"answer": "yes", "reasoning": "r", "confidence_level": 1} y')[
        "answer"] == "yes"
    assert parse_json_reply("no braces") is None


_assert_transcription()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--alias", required=True, help="the roster, comma separated; three families")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    roster = tuple(x.strip() for x in a.alias.split(",") if x.strip())
    if len(roster) != 3:
        print(f"  REFUSING: ReConcile's round table is three agents, got {roster}")
        return 2
    families = {REGISTRY.get(x).checkpoint.split("/")[0].lower() for x in roster}
    if len(families) < 3:
        print(f"  REFUSING: ReConcile is a heterogeneous round table and these aliases cover only "
              f"{sorted(families)}; homogenising it and calling it ReConcile is not this method")
        return 2

    bman = a.block.replace(".json", "") + ".manifest.json"
    bench = (json.load(open(bman, encoding="utf-8")).get("benchmark")
             if os.path.exists(bman) else os.path.basename(a.block).split("_block")[0])
    try:
        convincing = load_convincing(bench)
    except ValueError as e:
        print(f"  REFUSING: {e}")
        return 1

    from ._runner import run_block  # noqa: PLC0415
    return run_block(
        block_path=a.block, out_path=a.out, adapter_file=os.path.abspath(__file__),
        mechanism="reconcile", aliases=roster, limit=a.limit, upstream=UPSTREAM,
        per_task=lambda q, tid, task: run_one(
            q, task_id=tid, benchmark=bench, roster=roster, convincing=convincing,
            rounds=a.rounds, max_tokens=a.max_tokens, timeout_s=a.timeout_s, retries=a.retries),
        manifest_extra={
            "roster_is_heterogeneous": True,
            "roster_note": ("ReConcile is a round table of different model families; the roster is "
                            "part of this cell's identity and is never homogenised. This is an "
                            "open-weight backend adaptation of ChatGPT/Claude/Bard, labelled as "
                            "one, and is not a reproduction of the paper's numbers"),
            "convincing_dataset": DATASET_FOR[bench],
            "convincing_counts": {k: len(v) for k, v in convincing.items()},
            "sample_source_mapping": {str(k): v for k, v in SAMPLE_SOURCE_FOR.items()},
            "rounds": a.rounds, "temperature": 0.0, "max_tokens": a.max_tokens,
            "stop_rule": ("a debate prompt is built only on disagreement (utils.py:176), so "
                          "unanimity of answers ends the debate"),
            "confidence_audit": ("the prompt never states the confidence scale, and "
                                 "trans_confidence returns None above 1, which upstream then adds "
                                 "to and raises on. Out-of-range values are recorded as unusable "
                                 "per task instead"),
            "aggregation": "confidence-weighted vote; the unweighted majority is recorded beside it",
        })


if __name__ == "__main__":
    sys.exit(main())
