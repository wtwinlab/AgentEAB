r"""MAD-M² with a **per-agent model**: the second heterogeneous arm of the backbone extension.

Phase B of `paper/evidence/BACKBONE_EXTENSION_FREEZE_2026-10-09.md`, and the companion to
`du_mad_roster`. Same reason for being a separate module: `mad_m2.py` was collecting Phase A when
this was written, so editing it would have orphaned those manifests' adapter hashes.

Everything that defines the protocol is **imported** from `mad_m2` — the prompts, the prune
strategies, the perplexity, the answer extraction, the consensus measure. Only which model answers
which seat is new.

## The thing this arm has to get right, and a reason it is interesting here

MAD-M²'s contribution is `objective_prune`: it keeps the peers **above** the median perplexity and
masks the most confident ones. Perplexity is computed from the answering model's own logprobs, so
with a heterogeneous roster the prune is comparing **perplexities across different tokenizers and
different model families**.

That is not an error to paper over — it is what the released rule does when the agents differ, and the
design says a necessary adaptation must be listed rather than silently applied. **Nothing is
normalised here.** The per-agent perplexity, the model that produced it and the resulting mask are
all recorded per round, so the analysis can ask whether the prune is tracking confidence or merely
tracking tokenizer. A reader who wants a normalised variant can compute it from these records; this
module runs the released rule.

    PYTHONPATH=. python3 -m consensus_delivery.adapters.mad_m2_roster \
        --block ../data/frozen/gsm8k_block02.json --roster llama3-8b,qwen25-7b,mistral-7b \
        --prune objective --rotation 0 --out ... --limit 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from ..providers import chat
from ..registry_live import REGISTRY
from .mad_m2 import (ADAPTED, ANSWER_KIND, COT_SUFFIX, DEBATE_PREFIX, DEBATE_SUFFIX,
                     extract_answer, if_reach_consensus, objective_prune, perplexity)

UPSTREAM = "vendor/mad_m2"


def run_one(question: str, *, task_id: str, benchmark: str, seat_models: Sequence[str],
            rounds: int, prune: str, max_tokens: int, timeout_s: float, retries: int,
            temperature: float, top_p: float) -> Dict[str, Any]:
    """One MAD-M² run in which seat `i` is answered by `seat_models[i]`."""
    agents = len(seat_models)
    kind = ANSWER_KIND[benchmark]
    calls: List[Dict[str, Any]] = []
    order: List[Dict[str, Any]] = []
    per_round: List[Dict[str, Any]] = []
    responses: List[str] = [""] * agents
    t0 = time.time()

    for rnd in range(rounds):
        if rnd == 0:
            prompts = [question + COT_SUFFIX] * agents
            masks: List[List[bool]] = [[] for _ in range(agents)]
        else:
            prompts, masks = [], []
            for i in range(agents):
                peers = [r for j, r in enumerate(responses) if j != i]
                peer_ppls = [per_round[-1]["perplexities"][j]
                             for j in range(agents) if j != i]
                if prune == "objective":
                    kept, mask = objective_prune(peers, peer_ppls)
                else:
                    kept, mask = peers, [True] * len(peers)
                masks.append(mask)
                body = "".join(f"\n\nOne agent's solution: {k}" for k in kept)
                prompts.append(DEBATE_PREFIX + body + DEBATE_SUFFIX
                               + f"\n\nThe problem:\n{question}")
        round_answers: List[Optional[str]] = []
        round_ppls: List[Optional[float]] = []
        for i in range(agents):
            messages = [{"role": "user", "content": prompts[i]}]
            c0 = time.time()
            text, rec = chat(registry=REGISTRY, alias=seat_models[i], messages=messages,
                             temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                             retries=retries, timeout_s=timeout_s, logprobs=True)
            responses[i] = text
            lps = [e["logprob"] for e in (rec.token_logprobs or [])]
            ppl = perplexity(lps)
            ans = extract_answer(text, kind)
            round_answers.append(ans)
            round_ppls.append(ppl)
            calls.append({
                "round": rnd, "agent_index": i, "seq": len(calls),
                "agent_alias": seat_models[i], "served_model": rec.served_model,
                "messages_sent": [dict(m) for m in messages], "raw_reply": text,
                "extracted_answer": ans,
                # Kept per call and NOT normalised: with a heterogeneous roster this perplexity is
                # computed over a different tokenizer per seat, which is what the released prune
                # rule does when the agents differ.
                "perplexity": ppl, "n_logprobs": len(lps), "n_logprob_tokens": len(lps),
                "mask_over_peers": masks[i] if rnd > 0 else [],
                "latency_s": round(time.time() - c0, 3),
                "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
                "finish_reason": rec.finish_reason, "attempt": rec.attempt,
                "usd": rec.usd, "request_sha": rec.request_sha, "fingerprint": rec.fingerprint,
            })
            order.append({"seq": len(order), "round": rnd, "agent_index": i,
                          "agent_alias": seat_models[i]})
        # `if_reach_consensus` is imported, not reimplemented: the released rule reads the final
        # round, and a unanimity or `any round` rule differs from it on five of six answer patterns
        # with three agents.
        flag, common = if_reach_consensus(round_answers)
        per_round.append({
            "round": rnd, "answers": round_answers, "perplexities": round_ppls,
            "masks": masks if rnd > 0 else [],
            # By model, so a prune that systematically masks one family is visible.
            "perplexity_by_model": {m: p for m, p in zip(seat_models, round_ppls)},
            "masked_by_model": ({m: msk for m, msk in zip(seat_models, masks)} if rnd > 0 else {}),
            "consensus": flag, "common_answer": common,
        })

    finals = per_round[-1]["answers"]
    flag, common = if_reach_consensus(finals)
    valid = [a for a in finals if a is not None]
    delivered = common is not None
    return {
        "example_id": task_id,
        "run_id": f"madm2-{prune}-roster-{'+'.join(seat_models)}-{task_id}",
        "mechanism": f"madm2_{prune}_roster", "repeat": 0,
        "roster_is_heterogeneous": len(set(seat_models)) > 1,
        "seat_models": list(seat_models), "model_alias": "+".join(seat_models),
        "question": question, "benchmark": benchmark, "answer_kind": kind,
        "agents": agents, "rounds": rounds, "prune_strategy": prune,
        "temperature": temperature, "top_p": top_p,
        "per_agent_final_answer": finals,
        "per_agent_final_by_model": {m: a for m, a in zip(seat_models, finals)},
        "per_agent_final_perplexity": per_round[-1]["perplexities"],
        "n_distinct_final_answers": len(set(valid)),
        "majority_answer": common if delivered else None,
        "rule_satisfied": flag,
        "rule_note": ("if_reach_consensus: the most common answer appears at least len/2 times. "
                      "With three agents a 2-1 split already counts. The released loop does NOT "
                      "break on it, so this is a measured property and not a stop rule"),
        "consensus_per_round": [r["consensus"] for r in per_round],
        "absence_cause": (None if delivered else
                          "no agent's reply yielded an answer after 'Answer:'"),
        "is_task_adaptation": benchmark in ADAPTED,
        "adaptation_note": ("the released datasets are gsm8k, math, mmlu_pro and aime; this "
                            "benchmark is a labelled adaptation"
                            if benchmark in ADAPTED else None),
        "naive_mask_literal_differs_from_roster": (prune == "naive" and agents != 3),
        "accepted_delivery": "not-measured",
        "accepted_delivery_note": ("no agent is asked whether it accepts the delivered artifact; "
                                   "the confidence signal here is perplexity, not a stance"),
        "cross_tokenizer_perplexity_note": (
            "with a heterogeneous roster the prune compares perplexities across different "
            "tokenizers and families. Nothing is normalised: the released rule is run as released, "
            "and the per-seat perplexity, its model and the resulting mask are recorded so the "
            "question of whether the prune tracks confidence or tokenizer can be asked offline"),
        "native_stop_reason": f"fixed-round-budget-exhausted after {rounds} rounds",
        "rounds_detail": per_round, "calls": calls, "order": order,
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0 for c in calls), 6),
        "wall_s": round(time.time() - t0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--roster", required=True, help="comma-separated aliases, one per seat")
    ap.add_argument("--prune", default="objective", choices=("naive", "objective"))
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    seats = [x.strip() for x in a.roster.split(",") if x.strip()]
    if len(set(seats)) < 2:
        print(f"  REFUSING: {seats} is homogeneous. Use `mad_m2.py` for a homogeneous arm")
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
        mechanism=f"madm2_{a.prune}_roster", aliases=seats, limit=a.limit, upstream=UPSTREAM,
        per_task=lambda q, tid, task: {
            **run_one(q, task_id=tid, benchmark=bench, seat_models=seats, rounds=a.rounds,
                      prune=a.prune, max_tokens=a.max_tokens, timeout_s=a.timeout_s,
                      retries=a.retries, temperature=a.temperature, top_p=a.top_p),
            "rotation": a.rotation},
        manifest_extra={
            "roster": seats, "seat_models": seats, "rotation": a.rotation,
            "roster_is_heterogeneous": True, "prune_strategy": a.prune,
            "agents": len(seats), "rounds": a.rounds,
            "temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_tokens,
            "protocol_source": ("prompts, prune strategies, perplexity and extraction imported "
                                "from adapters/mad_m2.py so the two arms cannot diverge"),
            "cross_tokenizer_perplexity": ("not normalised; the released prune rule is run as "
                                           "released and the per-seat perplexity is recorded"),
        })


if __name__ == "__main__":
    sys.exit(main())
