r"""DMAD — "Demystifying Multi-Agent Debate: The Role of Confidence and Diversity", as released.

Pin `b64418a`. Every template is transcribed from `dmad/inference/prompts.py` and checked by the
assertions at the bottom against the released literals.

## The released configuration, not a guess

`configs/debate_qwen.yaml` and `configs/debate_llama.yaml`: **5 agents, 5 rounds,
`use_confidence: true`, temperature 1.0, top_p 0.9, `max_tokens` 1024** — and they name
`Qwen/Qwen2.5-7B-Instruct` and a Llama variant, so both of this study's backbones are configurations
the authors themselves specified rather than substitutions.

Temperature 1.0 is **not** the temperature the original 30 cells ran at. It is the released value, it
is recorded per cell, and no contrast is drawn across two temperatures.

## What the confidence is, and what it is not

Each agent emits `<confidence>0-10</confidence>`, and every other agent **sees it** in the next
round's prompt (`AGENT_RESPONSE_WITH_CONFIDENCE`). So confidence is an input to the debate.

It is **not** part of the aggregation: `vllm_debate.py`'s own docstring says "the final answer is
determined by majority vote over the last round", and `get_majority_answer` takes a plain majority
with dataset-specific normalisation. So confidence is reported here as **its own dimension** — a
stated certainty — and never as acceptance, and never as a vote weight. Treating it as either would
describe a mechanism the released code does not implement.

## Delivery and extraction kept apart

A reply must carry `<answer>...</answer>`. One that does not yields no vote, exactly as upstream's
`parse_response` returns an empty answer. When no agent produces a parseable answer the mechanism
delivers nothing, and that is recorded as an **extraction failure**, distinct from a protocol
declining to publish.
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
UPSTREAM = "_external/mechanism_screening_2026-10-08/repos/dmad @ b64418a"

#: `prompts.py:20-23`. MMLU-Pro and GPQA have more than four options, so the sentence is widened to
#: the block's own option count — a labelled adaptation of one clause, not a new instruction.
MCQA_INSTRUCTION = ", where YOUR FINAL ANSWER is one of (A), (B), (C) or (D)."

INITIAL_PROMPT_WITH_CONFIDENCE = """\
Answer the question. Think step by step first, at the end of your reasoning, \
provide your final answer and confidence level, ranging from 0 to 10, where \
0 means no confidence at all and 10 means complete confidence.

Output format:
<reasoning>YOUR DETAILED REASONING HERE</reasoning>
<answer>YOUR FINAL ANSWER</answer>{extra_info}
<confidence>INTEGER</confidence>

{question}"""

DEBATE_PROMPT_WITH_CONFIDENCE = """\
You are revising your answer after reviewing other agents' reasoning.

Question: {question}

Other agents' responses:
{other_agents}

Your previous reasoning and answer:
{reasoning}

Instructions:
- Reflect on how others reasoned.
- You may revise your answer if someone's reasoning provides stronger evidence.
- However, if you believe all of them missed something important, propose a \
better or alternative answer — clearly explain why.
- Be concise and clear.
- Update your confidence level to reflect how certain you are now. Confidence \
level is ranging from 0 to 10, where 0 means no confidence at all and 10 \
means complete confidence

Think step by step first, at the end of your reasoning, provide your final \
answer and confidence level in the following format:

<reasoning>YOUR DETAILED REASONING HERE</reasoning>
<answer>YOUR FINAL ANSWER</answer>{extra_info}
<confidence>INTEGER</confidence>"""

AGENT_RESPONSE_WITH_CONFIDENCE = """\
Agent {agent_id} has provided the following reasoning and final answer \
with confidence level:
<reasoning>{reasoning}</reasoning>
<answer>{final_answer}</answer>
<confidence>{confidence_level}</confidence>

"""

ANSWER_KIND = {"gsm8k": "numeric", "mmlu_pro": "mcq", "gpqa": "mcq",
               "strategyqa": "boolean", "squad_v2": "span"}
#: The released code implements GSM8K, MMLU-style MCQ and CSQA. The other two keep the tag format
#: and are labelled adaptations.
ADAPTED = {"strategyqa", "squad_v2"}


def extra_info(benchmark: str, n_options: int = 4) -> str:
    """The option clause. Widened for blocks with more than four choices, and labelled."""
    if ANSWER_KIND[benchmark] != "mcq":
        return ""
    if n_options <= 4:
        return MCQA_INSTRUCTION
    letters = ", ".join(f"({chr(65 + i)})" for i in range(n_options - 1))
    return f", where YOUR FINAL ANSWER is one of {letters} or ({chr(65 + n_options - 1)})."


def parse_response(text: str) -> Tuple[str, str, str]:
    """`inference/vllm_debate.py:53-78`: reasoning, answer, confidence, with `N/A` when absent."""
    t = text or ""
    reasoning = (t.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
                 if "<reasoning>" in t and "</reasoning>" in t else t.strip())
    answer = (t.split("<answer>")[-1].split("</answer>")[0].strip()
              if "<answer>" in t and "</answer>" in t else "")
    confidence = (t.split("<confidence>")[-1].split("</confidence>")[0].strip()
                  if "<confidence>" in t and "</confidence>" in t else "N/A")
    return reasoning, answer, confidence


def normalise(answer: str, kind: str) -> str:
    """`get_majority_answer`'s dataset-specific normalisation, made explicit."""
    a = (answer or "").strip()
    if not a:
        return ""
    if kind == "numeric":
        m = re.findall(r"[-+]?\d*\.?\d+", a.replace(",", "").replace("$", ""))
        if not m:
            return ""
        try:
            return str(round(float(m[-1]), 4))
        except ValueError:
            return ""
    if kind == "mcq":
        # A standalone letter, so a letter inside a word is not an answer.
        m = re.search(r"(?<![A-Za-z])\(?([A-J])\)?(?![A-Za-z])", a.upper())
        return m.group(1) if m else ""
    if kind == "boolean":
        low = a.lower()
        return "yes" if low.startswith("y") else "no" if low.startswith("n") else ""
    return " ".join(a.split())


def confidence_value(raw: str) -> Optional[int]:
    """`0-10` as an integer, or `None` where the agent gave none or something unreadable."""
    m = re.search(r"-?\d+", raw or "")
    if not m:
        return None
    v = int(m.group(0))
    return v if 0 <= v <= 10 else None


def majority(answers: List[str], task_id: str) -> Dict[str, Any]:
    """A plain majority over the last round. Confidence is deliberately not a weight."""
    valid = [a for a in answers if a != ""]
    if not valid:
        return {"answer": "", "valid_votes": 0, "winner_votes": 0, "tied": [],
                "absence_cause": "no agent emitted a parseable <answer> tag"}
    counter = collections.Counter(valid)
    top = max(counter.values())
    tied = [k for k, v in counter.items() if v == top]
    return {"answer": tied[0], "valid_votes": len(valid), "winner_votes": top,
            "tied": tied if len(tied) > 1 else [],
            "tie_break": "first appearance among the tied, as Counter orders them",
            "support_share_of_valid": round(top / len(valid), 4),
            "support_share_of_agents": round(top / len(answers), 4),
            "absence_cause": None}


def run_one(question: str, *, task_id: str, benchmark: str, alias: str, agents: int, rounds: int,
            max_tokens: int, timeout_s: float, retries: int, temperature: float, top_p: float,
            n_options: int) -> Dict[str, Any]:
    kind = ANSWER_KIND[benchmark]
    xi = extra_info(benchmark, n_options)
    calls: List[Dict[str, Any]] = []
    order: List[Dict[str, Any]] = []
    per_round: List[Dict[str, Any]] = []
    history: List[List[Dict[str, Any]]] = [[] for _ in range(agents)]
    t0 = time.time()
    seq = 0

    for rnd in range(rounds):
        for i in range(agents):
            if rnd == 0:
                prompt = INITIAL_PROMPT_WITH_CONFIDENCE.format(extra_info=xi, question=question)
            else:
                other_text = ""
                for j in range(agents):
                    if j == i:
                        continue
                    prev = history[j][-1]
                    other_text += AGENT_RESPONSE_WITH_CONFIDENCE.format(
                        agent_id=j, reasoning=prev["reasoning"],
                        final_answer=prev["answer"], confidence_level=prev["confidence"])
                own = history[i][-1]
                own_text = (f"{own['reasoning']}\n<answer>{own['answer']}</answer>"
                            f"\n<confidence>{own['confidence']}</confidence>")
                prompt = DEBATE_PROMPT_WITH_CONFIDENCE.format(
                    question=question, other_agents=other_text.strip(),
                    reasoning=own_text, extra_info=xi)
            messages = [{"role": "user", "content": prompt}]
            text, rec = chat(registry=REGISTRY, alias=alias, messages=messages,
                             temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                             retries=retries, timeout_s=timeout_s)
            reasoning, answer, conf = parse_response(text)
            history[i].append({"reasoning": reasoning, "answer": answer, "confidence": conf})
            calls.append({
                "seq": seq, "round": rnd, "agent_index": i, "attempt": rec.attempt,
                "messages_sent": [dict(m) for m in messages],
                "raw_reply": text,
                "parsed_answer": answer, "parsed_confidence": conf,
                "confidence_value": confidence_value(conf),
                "normalised_answer": normalise(answer, kind),
                "request_sha": rec.request_sha, "served_model": rec.served_model,
                "fingerprint": rec.fingerprint, "finish_reason": rec.finish_reason,
                "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
                "latency_s": rec.latency_s, "usd": rec.usd,
            })
            order.append({"seq": seq, "round": rnd, "agent_index": i})
            seq += 1
        votes = [normalise(history[i][-1]["answer"], kind) for i in range(agents)]
        confs = [confidence_value(history[i][-1]["confidence"]) for i in range(agents)]
        per_round.append({"round": rnd, "votes": votes, "confidences": confs,
                          "majority": majority(votes, f"{task_id}:r{rnd}")})

    final = [normalise(history[i][-1]["answer"], kind) for i in range(agents)]
    final_conf = [confidence_value(history[i][-1]["confidence"]) for i in range(agents)]
    res = majority(final, task_id)
    stated = [c for c in final_conf if c is not None]
    return {
        "run_id": f"dmad-{alias}-{task_id}-r0",
        "example_id": task_id, "mechanism": "dmad", "model_alias": alias, "repeat": 0,
        "question": question, "benchmark": benchmark, "answer_kind": kind,
        "is_task_adaptation": benchmark in ADAPTED,
        "adaptation_note": ("the released code implements GSM8K, MMLU-style MCQ and CSQA; this "
                            "benchmark keeps the released tag format and is a labelled adaptation"
                            if benchmark in ADAPTED else None),
        "agents": agents, "rounds": rounds,
        "temperature": temperature, "top_p": top_p,
        "per_agent_final_answer": [v if v != "" else None for v in final],
        "per_agent_final_confidence": final_conf,
        "n_distinct_final_answers": len({v for v in final if v != ""}),
        "majority_answer": res["answer"] or None,
        "tally": res,
        "rounds_detail": per_round,
        # Confidence is a stated certainty, not acceptance and not a vote weight. The released
        # aggregation is a plain majority; recording it as acceptance would describe a mechanism
        # this code does not implement.
        "accepted_delivery": "not-measured",
        "accepted_delivery_note": ("DMAD elicits a 0-10 confidence that other agents see, and "
                                   "aggregates by plain majority over the last round. It solicits "
                                   "no acceptance of a delivered artifact; absence of a verdict is "
                                   "not a refusal, and the confidence is not a substitute for one"),
        "stated_confidence": {
            "n_stated": len(stated), "n_agents": agents,
            "mean": round(sum(stated) / len(stated), 3) if stated else None,
            "min": min(stated) if stated else None, "max": max(stated) if stated else None,
            "all_agents_stated": len(stated) == agents,
        },
        "native_stop_reason": f"fixed-round-budget-exhausted after {rounds} rounds",
        "calls": calls, "order": order,
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0 for c in calls), 6),
        "wall_s": round(time.time() - t0, 2),
    }


def _assert_transcription() -> None:
    assert MCQA_INSTRUCTION == (
        ", where YOUR FINAL ANSWER is one of (A), (B), (C) or (D)."), "mcqa clause drifted"
    assert INITIAL_PROMPT_WITH_CONFIDENCE.startswith(
        "Answer the question. Think step by step first, at the end of your reasoning, provide your "
        "final answer and confidence level, ranging from 0 to 10,"), "initial prompt drifted"
    assert "<confidence>INTEGER</confidence>" in INITIAL_PROMPT_WITH_CONFIDENCE
    assert DEBATE_PROMPT_WITH_CONFIDENCE.startswith(
        "You are revising your answer after reviewing other agents' reasoning."), "debate drifted"
    assert "- Reflect on how others reasoned." in DEBATE_PROMPT_WITH_CONFIDENCE
    assert AGENT_RESPONSE_WITH_CONFIDENCE.startswith(
        "Agent {agent_id} has provided the following reasoning and final answer with confidence "
        "level:"), "agent-response template drifted"
    r, a, c = parse_response("<reasoning>why</reasoning><answer>(B)</answer><confidence>7</confidence>")
    assert (r, a, c) == ("why", "(B)", "7"), (r, a, c)
    assert parse_response("no tags at all")[1] == "", "a reply with no answer tag has no vote"
    assert parse_response("x<answer>4</answer>")[2] == "N/A", "absent confidence must be N/A"
    assert confidence_value("7") == 7 and confidence_value("N/A") is None
    assert confidence_value("42") is None, "a value outside 0-10 is unreadable, not clipped"
    assert normalise("(B)", "mcq") == "B" and normalise("Avogadro", "mcq") == ""
    assert normalise("the answer is 12 apples", "numeric") == "12.0"
    assert extra_info("gpqa", 10).endswith("or (J)."), extra_info("gpqa", 10)
    assert extra_info("gsm8k", 4) == "", "a numeric task gets no option clause"


_assert_transcription()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agents", type=int, default=5)        # configs/debate_qwen.yaml:13
    ap.add_argument("--rounds", type=int, default=5)        # configs/debate_qwen.yaml:14
    ap.add_argument("--temperature", type=float, default=1.0)   # sampling.temperature
    ap.add_argument("--top-p", type=float, default=0.9)         # sampling.top_p
    ap.add_argument("--max-tokens", type=int, default=1024)     # sampling.max_tokens
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"  REFUSING to overwrite {a.out}; a collected run is evidence")
        return 1

    block = json.load(open(a.block, encoding="utf-8"))
    bman_path = a.block.replace(".json", "") + ".manifest.json"
    bman = json.load(open(bman_path, encoding="utf-8")) if os.path.exists(bman_path) else {}
    bench = bman.get("benchmark") or os.path.basename(a.block).split("_block")[0]
    if bench not in ANSWER_KIND:
        print(f"  REFUSING: no declared answer kind for benchmark {bench!r}")
        return 1
    ep = REGISTRY.get(a.alias)
    tasks = block[: a.limit] if a.limit else block

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    t0 = time.time()
    for k, t in enumerate(tasks, 1):
        tid = t["example_id"]
        meta = t.get("metadata") or {}
        question = t["inputs"][0]
        if bench == "squad_v2":
            ctx = (t.get("context") or [""])[0] if isinstance(t.get("context"), list) else ""
            question = f"{ctx}\n\n{question}" if ctx else question
        elif bench in ("mmlu_pro", "gpqa"):
            opts = meta.get("options") or []
            if opts:
                question = f"{question}\n\n" + "\n".join(
                    f"({chr(65 + i)}) {o}" for i, o in enumerate(opts))
        try:
            r = run_one(question, task_id=tid, benchmark=bench, alias=a.alias, agents=a.agents,
                        rounds=a.rounds, max_tokens=a.max_tokens, timeout_s=a.timeout_s,
                        retries=a.retries, temperature=a.temperature, top_p=a.top_p,
                        n_options=meta.get("n_options", 4))
        except (TransportFailure, Exception) as e:   # noqa: BLE001 — a lost task is named
            failures.append({"example_id": tid, "error": f"{type(e).__name__}: {e}"})
            print(f"  [{k}/{len(tasks)}] {tid} FAILED {type(e).__name__}", flush=True)
            continue
        r["dataset_id"] = t["dataset_id"]
        r["references"] = t["references"]
        records.append(r)
        if k % 10 == 0 or k == len(tasks):
            print(f"  [{k}/{len(tasks)}] {len(records)} ok, {len(failures)} failed, "
                  f"{time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)

    src = os.path.join(os.path.dirname(__file__), "dmad.py")
    manifest = {
        "kind": "native-run", "mechanism": "dmad", "upstream_source": UPSTREAM,
        "adapter_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
        "benchmark": bench, "block_path": a.block, "block_sha256": bman.get("block_sha256"),
        "endpoint": {"alias": ep.alias, "provider": ep.provider, "served_name": ep.served_name,
                     "checkpoint": ep.checkpoint, "revision": ep.revision,
                     "chat_template_sha": ep.chat_template_sha,
                     "serving_image": ep.serving_image},
        "endpoint_url": ep.base_url, "model_name": ep.served_name,
        "agents": a.agents, "rounds": a.rounds, "use_confidence": True,
        "temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_tokens,
        "released_config": "configs/debate_qwen.yaml and debate_llama.yaml: 5 agents, 5 rounds, "
                           "use_confidence true, temperature 1.0, top_p 0.9, max_tokens 1024",
        "aggregation": "plain majority over the last round; the elicited confidence is an input to "
                       "the debate and is NOT a vote weight",
        "retries": a.retries, "num_samples": len(tasks), "records_written": len(records),
        "failed_example_ids": [f["example_id"] for f in failures], "n_failed": len(failures),
        "reason_counts": dict(collections.Counter(f["error"].split(":")[0] for f in failures)),
        "unexplained_absent": len(tasks) - len(records) - len(failures),
        "is_task_adaptation": bench in ADAPTED,
        "endpoint_tokens": {"prompt": sum(r["tokens_in"] for r in records),
                            "generation": sum(r["tokens_out"] for r in records)},
        "usd": round(sum(r["usd"] for r in records), 6),
        "wall_s": round(time.time() - t0, 1),
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": ("DMAD's inference protocol as released. The GRPO-trained backbone is out of scope "
                 "and is not substituted for. No writer, gate or stop rule was added."),
    }
    with open(a.out.replace(".json", "") + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"  {os.path.basename(a.block)}: {len(records)} records, {len(failures)} failed, "
          f"{manifest['wall_s']:.0f}s, tokens={manifest['endpoint_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
