r"""MAD-M² — "Multi-Agent Debate with Memory Masking", as released.

Pin `f02069a`. Protocol read from `src/reasoning_models.py` and `src/utils.py`; the assertions at the
bottom check the transcription.

## The released defaults

`src/args.py`: **3 agents, `max_round` 2, `prune_strategy` naive**. `configs.yaml`: temperature 1,
top_p 1, and `Qwen/Qwen2.5-7B-Instruct` among its models — one of this study's backbones.

Two arms are collected, because one of them is the method and the other is its own default:

* **`naive`** — the released default, which does **no masking at all**:
  `new_contexts = history[-1]`. Plain debate with perplexity recorded.
* **`objective`** — the perplexity-driven masking that is the paper's contribution.

## What `objective` masking actually does

`_objective_prune` (`reasoning_models.py:196-212`) sorts the peers' perplexities **descending**,
takes the median as the threshold, and keeps the contexts with `ppl > threshold`. So it retains the
**most uncertain** peers and masks the **most confident** ones — and with three agents the median is
the middle value, so exactly **one** peer context survives: the single highest-perplexity response.
That is what the released code does; whether it is the intent is not something this adapter decides.
Both the mask and the perplexities are recorded per round, so any other masking rule is a
recomputation rather than a re-run.

## Consensus is measured here, not a stop rule

`if_reach_consensus` (`utils.py:79-100`) returns true when the most common answer appears
`count >= len(answers) / 2` times — with three agents a 2-1 split is already consensus, and with an
even roster a tie would be. **The debate loop does not break on it**: `reasoning_models.py:130-147`
runs every round regardless and records `consensus_flags`. So MAD-M²'s consensus is a measured
property of each round, and the stop reason is the round budget.

## One issue recorded rather than inherited

The `naive` branch hard-codes `masks = [[True, True, True]] * len(questions)` — three entries,
whatever `num_agents` is. At the released default of 3 it is right; at any other roster size the
recorded mask would not match the roster. The adapter builds the naive mask from the actual roster
size and records that it differs from the literal.

## Perplexity over our serving path

The released code gets perplexity from its own engine. Here it is computed from the per-token
logprobs the OpenAI-compatible endpoint returns: `exp(-mean(logprob))` over the generated tokens,
which is the definition. The token count it is averaged over is recorded so the figure can be
checked.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ..providers import TransportFailure, chat
from ..registry_live import REGISTRY

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UPSTREAM = "_external/mechanism_screening_2026-10-08/repos/mad-m2 @ f02069a"

#: The released CoT framing for the initial round, and the debate framing for later rounds.
COT_SUFFIX = ("\n\nLet's think step by step, and put your final answer after 'Answer:'.")
DEBATE_PREFIX = "These are the solutions to the problem from other agents:\n"
DEBATE_SUFFIX = ("\n\nUsing the solutions from other agents as additional information, give your "
                 "updated answer. Let's think step by step, and put your final answer after "
                 "'Answer:'.")
ANSWER_KIND = {"gsm8k": "numeric", "mmlu_pro": "mcq", "gpqa": "mcq",
               "strategyqa": "boolean", "squad_v2": "span"}
#: `args.py:16` lists gsm8k, math, mmlu_pro, aime24, aime25. The other two are labelled adaptations.
ADAPTED = {"strategyqa", "squad_v2", "gpqa"}


def extract_answer(text: str, kind: str) -> Optional[str]:
    """After `Answer:`, normalised per task kind. `None` where there is nothing to read."""
    t = text or ""
    tail = t.split("Answer:")[-1] if "Answer:" in t else t
    if kind == "numeric":
        m = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", tail)
        if not m:
            return None
        try:
            return str(round(float(m[-1].replace(",", "")), 4))
        except ValueError:
            return None
    if kind == "mcq":
        m = re.search(r"(?<![A-Za-z])\(?([A-J])\)?(?![A-Za-z])", tail.upper())
        return m.group(1) if m else None
    if kind == "boolean":
        low = tail.strip().lower()
        for tok in re.findall(r"[a-z]+", low):
            if tok in ("yes", "true"):
                return "yes"
            if tok in ("no", "false"):
                return "no"
        return None
    s = " ".join(tail.split())
    return s or None


def if_reach_consensus(answers: List[Optional[str]]) -> Tuple[bool, Optional[str]]:
    """`utils.py:79-100`, transcribed. `count >= len(answers)/2`, over **all** answers."""
    if not answers:
        return False, None
    counts: Dict[Any, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    most_common_answer, count = max(counts.items(), key=lambda x: x[1])
    return (count >= len(answers) / 2), most_common_answer


def perplexity(logprobs: List[float]) -> Optional[float]:
    """`exp(-mean(logprob))` over the generated tokens. `None` where there are none."""
    if not logprobs:
        return None
    return round(math.exp(-sum(logprobs) / len(logprobs)), 6)


def objective_prune(contexts: List[str], ppls: List[Optional[float]]) -> Tuple[List[str], List[bool]]:
    """`_objective_prune`: keep the peers **above** the median perplexity, mask the rest."""
    usable = [p if p is not None else 0.0 for p in ppls]
    if not usable:
        return [], []
    desc = sorted(usable, reverse=True)
    threshold = desc[len(usable) // 2]
    mask = [p > threshold for p in usable]
    return [c for c, m in zip(contexts, mask) if m], mask


def run_one(question: str, *, task_id: str, benchmark: str, alias: str, agents: int, rounds: int,
            prune: str, max_tokens: int, timeout_s: float, retries: int, temperature: float,
            top_p: float) -> Dict[str, Any]:
    kind = ANSWER_KIND[benchmark]
    calls: List[Dict[str, Any]] = []
    order: List[Dict[str, Any]] = []
    per_round: List[Dict[str, Any]] = []
    responses: List[str] = [""] * agents
    t0 = time.time()
    seq = 0

    for rnd in range(rounds):
        if rnd == 0:
            prompts = [question + COT_SUFFIX] * agents
            masks: List[List[bool]] = [[] for _ in range(agents)]
        else:
            prompts = []
            masks = []
            for i in range(agents):
                peers = [r for j, r in enumerate(responses) if j != i]
                peer_ppls = [per_round[-1]["perplexities"][j]
                             for j in range(agents) if j != i]
                if prune == "objective":
                    kept, mask = objective_prune(peers, peer_ppls)
                else:
                    # The released `naive` branch does no masking. Its literal mask is
                    # `[True, True, True]` regardless of the roster; this builds it from the
                    # actual roster size and records the difference.
                    kept, mask = peers, [True] * len(peers)
                masks.append(mask)
                body = "".join(f"\n\nOne agent's solution: {k}" for k in kept)
                prompts.append(DEBATE_PREFIX + body + DEBATE_SUFFIX
                               + f"\n\nThe problem:\n{question}")
        round_answers: List[Optional[str]] = []
        round_ppls: List[Optional[float]] = []
        for i in range(agents):
            messages = [{"role": "user", "content": prompts[i]}]
            text, rec = chat(registry=REGISTRY, alias=alias, messages=messages,
                             temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                             retries=retries, timeout_s=timeout_s, logprobs=True)
            responses[i] = text
            lps = [e["logprob"] for e in (rec.token_logprobs or [])]
            ppl = perplexity(lps)
            ans = extract_answer(text, kind)
            round_answers.append(ans)
            round_ppls.append(ppl)
            calls.append({
                "seq": seq, "round": rnd, "agent_index": i, "attempt": rec.attempt,
                "messages_sent": [dict(m) for m in messages],
                "raw_reply": text, "extracted_answer": ans,
                "perplexity": ppl, "n_logprob_tokens": len(lps),
                "mask_over_peers": masks[i] if rnd > 0 else [],
                "request_sha": rec.request_sha, "served_model": rec.served_model,
                "fingerprint": rec.fingerprint, "finish_reason": rec.finish_reason,
                "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
                "latency_s": rec.latency_s, "usd": rec.usd,
            })
            order.append({"seq": seq, "round": rnd, "agent_index": i})
            seq += 1
        flag, common = if_reach_consensus(round_answers)
        per_round.append({"round": rnd, "answers": round_answers, "perplexities": round_ppls,
                          "masks": masks if rnd > 0 else [],
                          "consensus": flag, "common_answer": common})

    final_answers = per_round[-1]["answers"]
    flag, common = if_reach_consensus(final_answers)
    valid = [a for a in final_answers if a is not None]
    delivered = common is not None
    return {
        "run_id": f"madm2-{prune}-{alias}-{task_id}-r0",
        "example_id": task_id, "mechanism": f"madm2_{prune}", "model_alias": alias, "repeat": 0,
        "question": question, "benchmark": benchmark, "answer_kind": kind,
        "is_task_adaptation": benchmark in ADAPTED,
        "adaptation_note": ("the released datasets are gsm8k, math, mmlu_pro and aime; this "
                            "benchmark is a labelled adaptation"
                            if benchmark in ADAPTED else None),
        "agents": agents, "rounds": rounds, "prune_strategy": prune,
        "temperature": temperature, "top_p": top_p,
        "per_agent_final_answer": final_answers,
        "per_agent_final_perplexity": per_round[-1]["perplexities"],
        "n_distinct_final_answers": len(set(valid)),
        "majority_answer": common if delivered else None,
        # Consensus is measured, not a stop rule: the released loop runs every round regardless.
        "rule_satisfied": flag,
        "rule_note": ("if_reach_consensus: the most common answer appears at least len/2 times. "
                      "With three agents a 2-1 split already counts. The released loop does NOT "
                      "break on it, so this is a measured property and not a stop rule"),
        "consensus_per_round": [r["consensus"] for r in per_round],
        "accepted_delivery": "not-measured",
        "accepted_delivery_note": ("MAD-M2 solicits no acceptance of a delivered artifact; it "
                                   "measures whether answers coincide. Absence of a verdict is not "
                                   "a refusal, and coincidence is not acceptance"),
        "absence_cause": (None if delivered else
                          "no agent's reply yielded an answer after 'Answer:'"),
        "native_stop_reason": f"fixed-round-budget-exhausted after {rounds} rounds",
        "rounds_detail": per_round,
        "naive_mask_literal_differs_from_roster": (prune == "naive" and agents != 3),
        "calls": calls, "order": order,
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0 for c in calls), 6),
        "wall_s": round(time.time() - t0, 2),
    }


def _assert_transcription() -> None:
    # The consensus threshold, including the cases that make `>=` and `/2` matter.
    assert if_reach_consensus(["A", "A", "B"]) == (True, "A"), "2 of 3 must be consensus"
    assert if_reach_consensus(["A", "B", "C"]) == (False, "A"), "1 of 3 must not be"
    assert if_reach_consensus(["A", "A", "B", "B"])[0] is True, (
        "with an even roster a 2-2 tie passes `count >= len/2`; that is the released behaviour")
    assert if_reach_consensus([]) == (False, None)
    assert if_reach_consensus([None, None, "A"]) == (True, None), (
        "upstream counts `None` as an answer value, so three agents with two unreadable replies "
        "reach 'consensus' on nothing; the record keeps `common_answer` None so delivery is absent")
    # The objective prune keeps the peers ABOVE the median, i.e. the most uncertain.
    kept, mask = objective_prune(["a", "b"], [2.0, 1.0])
    assert mask == [True, False] and kept == ["a"], (kept, mask)
    kept3, mask3 = objective_prune(["a", "b", "c"], [3.0, 2.0, 1.0])
    assert mask3 == [True, False, False] and kept3 == ["a"], (
        "with three peers the median threshold leaves exactly the highest-perplexity one")
    assert perplexity([0.0, 0.0]) == 1.0
    assert perplexity([]) is None
    assert abs(perplexity([-1.0, -1.0]) - math.e) < 1e-6
    assert extract_answer("blah Answer: (C) done", "mcq") == "C"
    assert extract_answer("Answer: Avogadro", "mcq") is None, "a letter in a word is not an answer"
    assert extract_answer("Answer: 1,234.5", "numeric") == "1234.5"
    assert extract_answer("nothing here", "boolean") is None


_assert_transcription()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prune", default="naive", choices=("naive", "objective"))
    ap.add_argument("--agents", type=int, default=3)        # args.py:20
    ap.add_argument("--rounds", type=int, default=2)        # args.py:21 (max_round)
    ap.add_argument("--temperature", type=float, default=1.0)   # configs.yaml
    ap.add_argument("--top-p", type=float, default=1.0)         # configs.yaml
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    bman = a.block.replace(".json", "") + ".manifest.json"
    bench = (json.load(open(bman, encoding="utf-8")).get("benchmark")
             if os.path.exists(bman) else os.path.basename(a.block).split("_block")[0])
    if bench not in ANSWER_KIND:
        print(f"  REFUSING: no declared answer kind for benchmark {bench!r}")
        return 1

    from ._runner import run_block  # noqa: PLC0415
    return run_block(
        block_path=a.block, out_path=a.out, adapter_file=os.path.abspath(__file__),
        mechanism=f"madm2_{a.prune}", aliases=[a.alias], limit=a.limit, upstream=UPSTREAM,
        per_task=lambda q, tid, task: run_one(
            q, task_id=tid, benchmark=bench, alias=a.alias, agents=a.agents, rounds=a.rounds,
            prune=a.prune, max_tokens=a.max_tokens, timeout_s=a.timeout_s, retries=a.retries,
            temperature=a.temperature, top_p=a.top_p),
        manifest_extra={
            "prune_strategy": a.prune, "agents": a.agents, "rounds": a.rounds,
            "temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_tokens,
            "released_defaults": "args.py: 3 agents, max_round 2, prune naive; configs.yaml: "
                                 "temperature 1, top_p 1",
            "consensus_rule": ("if_reach_consensus: count >= len/2 over all answers, measured per "
                               "round. The released loop does NOT break on it"),
            "prune_note": ("objective keeps the peers ABOVE the median perplexity, i.e. the most "
                           "uncertain, masking the most confident; with three agents exactly one "
                           "peer context survives"),
            "logprobs_requested": True,
            "is_task_adaptation": bench in ADAPTED,
            "note": ("MAD-M2 as released. Perplexity is computed as exp(-mean(logprob)) over the "
                     "generated tokens from the endpoint's own per-token logprobs."),
        })


if __name__ == "__main__":
    sys.exit(main())
