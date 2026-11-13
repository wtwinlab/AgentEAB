r"""Du MAD with a **per-agent model**: the heterogeneous arm of the new-backbone extension.

Phase B of `paper/evidence/BACKBONE_EXTENSION_FREEZE_2026-10-09.md`. Du et al.'s debate runs three
agents; here each agent is a **different model**, so the roster `(L, Q, M)` produces one real debate
between three families rather than three homogeneous debates stitched together. The design is
explicit that a heterogeneous team must generate its own interaction, and that is the whole reason
this module exists.

## Why a separate module rather than a flag on `du_mad.py`

Two reasons, and the second is the binding one.

**Provenance.** `du_mad.py` was collecting Phase A's cells when this was written. Editing it would
have made every Phase A manifest record a hash belonging to code that did not produce those records
— the orphaning this project has already had once, when 13 pilot cells lost their adapter.

**Identity.** A per-agent roster is a different experimental arm, not a different setting of the same
one, and its records carry fields the homogeneous arm has no use for. Keeping them apart means
neither cell type has to carry the other's columns.

## What is NOT duplicated

Every prompt, every parse and the aggregation are **imported** from `du_mad`, not copied:
`opening_for`, `others_for`, `task_question`, `extract_for`, `most_frequent`. So the two arms cannot
drift on the protocol — only on which model answers which seat. Copying forty lines of prompts here
would have created exactly the divergence `_runner` was factored out to prevent.

## Seat rotation

The roster is a sequence and `--roster` fixes the seat order for a run. The freeze's three rotations
— `(L,Q,M)`, `(Q,M,L)`, `(M,L,Q)` — are three separate runs over the same task ids, each recording
its own seat assignment, averaged within a task afterwards. Three rotations of one task are **not**
three tasks, and nothing here pretends otherwise: the records carry `rotation` and `seat_models` so
the analysis can group by task.

    PYTHONPATH=. python3 -m consensus_delivery.adapters.du_mad_roster \
        --block ../data/frozen/gsm8k_block02.json --roster llama3-8b,qwen25-7b,mistral-7b \
        --rotation 0 --out ../results/backbone_matrix/gsm8k100_dumad_LQM_rot0.json --limit 100
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
# The protocol itself, imported so the two arms cannot diverge on it.
from .du_mad import (extract_for, most_frequent, opening_for, others_for, task_question)

UPSTREAM = "vendor/mad"


def run_one(question: str, *, benchmark: str, seat_models: Sequence[str], rounds: int,
            max_tokens: int, timeout_s: float, retries: int,
            n_options: int = 10) -> Dict[str, Any]:
    """One debate in which seat `i` is answered by `seat_models[i]`.

    Identical to `du_mad.run_one` except that the model is per seat. The context construction, the
    prompts and the aggregation come from that module.
    """
    agents = len(seat_models)
    ctxs: List[List[Dict[str, str]]] = [
        [{"role": "user", "content": opening_for(benchmark, question)}] for _ in range(agents)]
    calls: List[Dict[str, Any]] = []
    order: List[Dict[str, Any]] = []
    t0 = time.time()
    for rnd in range(rounds):
        for i, ctx in enumerate(ctxs):
            if rnd != 0:
                others = ctxs[:i] + ctxs[i + 1:]
                ctx.append(others_for(benchmark, others, question, 2 * rnd - 1))
            sent = [dict(m) for m in ctx]
            c0 = time.time()
            # The one difference from the homogeneous arm: the seat decides the model.
            text, rec = chat(registry=REGISTRY, alias=seat_models[i], messages=sent,
                             temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                             retries=retries, timeout_s=timeout_s)
            ctx.append({"role": "assistant", "content": text})
            calls.append({
                "round": rnd, "agent_index": i, "seq": len(calls),
                # Recorded per call, because "which model said this" is the experiment.
                "agent_alias": seat_models[i], "served_model": rec.served_model,
                "messages_sent": sent, "raw_reply": text,
                "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
                "finish_reason": rec.finish_reason, "attempt": rec.attempt,
                "latency_s": round(time.time() - c0, 3), "usd": rec.usd,
                "request_sha": rec.request_sha, "fingerprint": rec.fingerprint,
                "extracted_answer": extract_for(benchmark, text, n_options),
            })
            order.append({"seq": len(order), "round": rnd, "agent_index": i,
                          "agent_alias": seat_models[i]})
    finals = [ctx[-1]["content"] for ctx in ctxs]
    per_agent = [extract_for(benchmark, t, n_options) for t in finals]
    winner = most_frequent(per_agent)
    return {
        "mechanism": "du-mad-roster", "repeat": 0,
        "roster_is_heterogeneous": len(set(seat_models)) > 1,
        "seat_models": list(seat_models),
        "model_alias": "+".join(seat_models),
        "question": question, "benchmark": benchmark,
        "agents": agents, "rounds": rounds,
        "per_agent_final_answer": per_agent,
        # Which model produced which final answer — the column the homogeneous arm cannot have.
        "per_agent_final_by_model": {m: a for m, a in zip(seat_models, per_agent)},
        "n_distinct_final_answers": len({a for a in per_agent if a}),
        "majority_answer": winner,
        "rule_satisfied": None,
        "rule_note": ("Du et al.'s debate has no stop rule; it runs a fixed round budget. The "
                      "decision is a post-hoc plurality over the agents' final answers"),
        "accepted_delivery": "not-measured",
        "accepted_delivery_note": ("no agent is asked whether it accepts the delivered artifact. "
                                   "Absence of a verdict is not a refusal"),
        "native_stop_reason": "fixed round budget",
        "calls": calls, "order": order,
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0 for c in calls), 6),
        "wall_s": round(time.time() - t0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--roster", required=True,
                    help="comma-separated aliases, one per seat, in seat order")
    ap.add_argument("--rotation", type=int, default=0,
                    help="which of the freeze's cyclic rotations this run is; recorded, not applied")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    seats = [x.strip() for x in a.roster.split(",") if x.strip()]
    if len(seats) < 2:
        print(f"  REFUSING: a roster needs at least two seats, got {seats}")
        return 2
    if len(set(seats)) < 2:
        print(f"  REFUSING: {seats} is homogeneous. Use `du_mad.py` for a homogeneous arm; "
              "claiming a heterogeneous result from one model in three seats would be the "
              "homogenisation the design forbids")
        return 2

    bman = a.block.replace(".json", "") + ".manifest.json"
    bench = (json.load(open(bman, encoding="utf-8")).get("benchmark")
             if os.path.exists(bman) else os.path.basename(a.block).split("_block")[0])

    from ._runner import run_block  # noqa: PLC0415

    def per_task(q: str, tid: str, task: Dict[str, Any]) -> Dict[str, Any]:
        n_opt = (task.get("metadata") or {}).get("n_options", 10)
        r = run_one(task_question(bench, task), benchmark=bench, seat_models=seats,
                    rounds=a.rounds, max_tokens=a.max_tokens, timeout_s=a.timeout_s,
                    retries=a.retries, n_options=n_opt)
        r["example_id"] = tid
        r["run_id"] = f"dumad-roster-{'+'.join(seats)}-rot{a.rotation}-{tid}"
        r["rotation"] = a.rotation
        return r

    return run_block(
        block_path=a.block, out_path=a.out, adapter_file=os.path.abspath(__file__),
        mechanism="du-mad-roster", aliases=seats, limit=a.limit, upstream=UPSTREAM,
        per_task=per_task,
        manifest_extra={
            "roster": seats, "seat_models": seats, "rotation": a.rotation,
            "roster_is_heterogeneous": True,
            "rounds": a.rounds, "agents": len(seats), "max_tokens": a.max_tokens,
            "protocol_source": ("prompts, parsing and aggregation imported from adapters/du_mad.py "
                                "so the heterogeneous and homogeneous arms cannot diverge on the "
                                "protocol"),
            "rotation_note": ("one of the freeze's three cyclic seat assignments. Three rotations "
                              "of a task are averaged within the task and are not three tasks"),
        })


if __name__ == "__main__":
    sys.exit(main())
