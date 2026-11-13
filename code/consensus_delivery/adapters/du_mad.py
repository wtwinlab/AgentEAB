"""Du et al.'s multi-agent debate, run on a frozen block through a served endpoint.

**What is preserved, verbatim from `vendor/mad/gsm/gen_gsm.py`:** the opening prompt, the
other-agents prompt and the no-other-agents prompt (character for character), the roles — `user` and
`assistant` only, no system message — the agent and round counts Du uses for GSM8K (`agents = 3`,
`rounds = 2`), the index `2*round - 1` at which an agent reads the others, the round-robin order, and
the post-hoc majority over per-agent extracted answers that `gsm/eval_gsm.py` performs.

**What is replaced, and only this:**

* the data source. `gen_gsm.py:38` points at the author's own machine
  (`/data/vision/billf/scratch/yilundu/.../test.jsonl`), so it cannot run anywhere else. Our frozen
  block supplies the same field, `question`, and the reference comes from the block.
* the model client. `openai.ChatCompletion.create` is the pre-1.0 API and is gone from the installed
  library. Requests go through our own transport instead, which is also what makes tokens, latency,
  attempts and cost recordable — Du's script records none of them.
* `random.seed(0); random.shuffle(questions)[:100]` is not used. The block is already a pinned,
  hash-identified task list, and re-shuffling it would discard that identity.

**What is *not* added:** no writer, no acceptance solicitation, no stop rule. Du MAD has none. It runs
a fixed round budget and the answer is extracted afterwards, so its native record establishes neither
an acceptance verdict nor a stopping condition. That is a contract fact about the method and it is
reported as `not-measured`, never as a refusal.

A latent issue in the upstream evaluator, recorded rather than inherited: `eval_gsm.py:69-90`
declares `compute_accuracy(gt, pred_solution)` and its list branch iterates `pred_solutions` — a name
that is neither a parameter nor assigned in the function. It resolves to the module-level
`pred_solutions` defined at `:131` under `__main__`, so the function works by global scope in its own
script and raises `NameError` for any other caller. This module implements the same majority with the
name bound properly, and the aggregation rule itself is unchanged.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ..providers import TransportFailure, chat
from ..registry_live import REGISTRY

#: Verbatim from `gen_gsm.py:45`.
OPENING = ("Can you solve the following math problem? {q} Explain your reasoning. Your final answer "
           "should be a single numerical number, in the form \\boxed{{answer}}, at the end of your "
           "response. ")

#: Verbatim from `gen_gsm.py:8`.
SELF_CHECK = ("Can you double check that your answer is correct. Please reiterate your answer, with "
              "your final answer a single numerical number, in the form \\boxed{{answer}}.")

#: Verbatim from `gen_gsm.py:10-18`, including its trailing-space and newline layout.
OTHERS_PREFIX = "These are the solutions to the problem from other agents: "
OTHERS_ITEM = "\n\n One agent solution: ```{a}```"
OTHERS_SUFFIX = ("\n\n Using the solutions from other agents as additional information, can you "
                 "provide your answer to the math problem? \n The original math problem is {q}. "
                 "Your final answer should be a single numerical number, in the form "
                 "\\boxed{{answer}}, at the end of your response.")


#: Du's runner exists only for GSM8K (`gsm/gen_gsm.py`) and arithmetic (`math/gen_math.py`). Taking
#: it to the other four benchmarks means changing **one clause**: how the final answer must be
#: stated. Everything else — the debate structure, the roles, the read index, the round count, the
#: aggregation — is Du's. §2.1A calls these cells `A10+D`, `AMC+D`, `AB+D`, `AS+D`, i.e. explicitly
#: labelled cross-task adaptations, and that is what they are recorded as. GSM8K alone is `R+D`:
#: Du's own runner, nothing adapted but the data source and the client.
ANSWER_CLAUSE = {
    # Verbatim from gen_gsm.py; not an adaptation.
    "gsm8k": "Your final answer should be a single numerical number, in the form "
             "\\boxed{{answer}}, at the end of your response.",
    # Adaptations. Each keeps Du's sentence shape and changes only the required answer form.
    "mmlu_pro": "Your final answer should be a single option letter, in the form "
                "\\boxed{{answer}}, at the end of your response.",
    "gpqa": "Your final answer should be a single option letter, in the form "
            "\\boxed{{answer}}, at the end of your response.",
    "strategyqa": "Your final answer should be a single word, yes or no, in the form "
                  "\\boxed{{answer}}, at the end of your response.",
    "squad_v2": "Your final answer should be the exact answer span copied from the passage, or the "
                "single word unanswerable if the passage does not contain it, in the form "
                "\\boxed{{answer}}, at the end of your response.",
}

#: The task framing. Du's own wording for GSM8K; the others change the noun and nothing else.
FRAMING = {
    "gsm8k": "Can you solve the following math problem? {q} Explain your reasoning. ",
    "mmlu_pro": "Can you answer the following multiple-choice question? {q} Explain your reasoning. ",
    "gpqa": "Can you answer the following multiple-choice question? {q} Explain your reasoning. ",
    "strategyqa": "Can you answer the following yes/no question? {q} Explain your reasoning. ",
    "squad_v2": "Can you answer the following reading-comprehension question? {q} Explain your "
                "reasoning. ",
}

#: Du's other-agents suffix, with the task noun and the answer clause substituted.
SUFFIX_FRAMING = {
    "gsm8k": "\n\n Using the solutions from other agents as additional information, can you provide "
             "your answer to the math problem? \n The original math problem is {q}. ",
    "mmlu_pro": "\n\n Using the solutions from other agents as additional information, can you "
                "provide your answer to the question? \n The original question is {q}. ",
    "gpqa": "\n\n Using the solutions from other agents as additional information, can you provide "
            "your answer to the question? \n The original question is {q}. ",
    "strategyqa": "\n\n Using the solutions from other agents as additional information, can you "
                  "provide your answer to the question? \n The original question is {q}. ",
    "squad_v2": "\n\n Using the solutions from other agents as additional information, can you "
                "provide your answer to the question? \n The original question is {q}. ",
}


def opening_for(benchmark: str, question: str) -> str:
    """The first message. For `gsm8k` this must equal Du's own literal, and that is asserted."""
    # Du's opening ends with a trailing space (`gen_gsm.py:45`). Dropping it is a one-character
    # drift from the prompt the published method sends, which is exactly what the assertion below
    # exists to catch — it caught this.
    text = (FRAMING[benchmark].format(q=question)
            + ANSWER_CLAUSE[benchmark].format(answer="answer") + " ")
    if benchmark == "gsm8k":
        assert text == OPENING.format(q=question), "the gsm8k opening drifted from Du's literal"
    return text


def others_for(benchmark: str, others: List[List[Dict[str, str]]], question: str,
               idx: int) -> Dict[str, str]:
    """The other-agents message. For `gsm8k` this must equal Du's own literal, and that is asserted."""
    if not others:
        return {"role": "user", "content": SELF_CHECK}
    body = OTHERS_PREFIX + "".join(OTHERS_ITEM.format(a=c[idx]["content"]) for c in others)
    text = body + SUFFIX_FRAMING[benchmark].format(q=question) \
        + ANSWER_CLAUSE[benchmark].format(answer="answer")
    if benchmark == "gsm8k":
        assert text == construct_message(others, question, idx)["content"], \
            "the gsm8k others message drifted from Du's literal"
    return {"role": "user", "content": text}


#: The question the agents see. For SQuAD 2.0 the passage is part of the question, because the task
#: is not answerable without it; that too is part of the labelled adaptation.
def task_question(benchmark: str, rec: Dict[str, Any]) -> str:
    q = rec["inputs"][0]
    if benchmark == "squad_v2" and rec.get("context"):
        return f"{rec['context'][0]}\n\nQuestion: {q}"
    return q


def construct_message(others: List[List[Dict[str, str]]], question: str, idx: int) -> Dict[str, str]:
    """`gen_gsm.py:6-19`, unchanged in content."""
    if not others:
        return {"role": "user", "content": SELF_CHECK}
    s = OTHERS_PREFIX
    for ctx in others:
        s += OTHERS_ITEM.format(a=ctx[idx]["content"])
    s += OTHERS_SUFFIX.format(q=question)
    return {"role": "user", "content": s}


def parse_answer(text: str) -> Optional[str]:
    """`eval_gsm.py:55-66`: the last `{...}` group reduced to digits and dots."""
    for m in re.findall(r"\{([0-9.,$]*)\}", text or "")[::-1]:
        sol = re.sub(r"[^0-9.]", "", m)
        if sol:
            return sol
    return None


def solve_math_problems(text: str) -> Optional[str]:
    """`eval_gsm.py:46-53`: the last bare number, Du's fallback when no `\\boxed{}` is present."""
    m = re.findall(r"\d+\.?\d*", text or "")
    return m[-1] if m else None


#: The `\boxed{...}` group, whatever is inside it. Du's own extractor keeps only digits and dots
#: (`eval_gsm.py:62`, `re.sub(r"[^0-9.]", "", ...)`), which is right for a numeric task and empties
#: the box on every other one: `\boxed{yes}` reduced to `""`, fell through to the bare-number
#: fallback, and returned the last digit in the prose. Measured on a StrategyQA smoke, that produced
#: `['3.', '4.', '4.']` for a yes/no question. So the extractor is part of the labelled adaptation,
#: and Du's numeric path is kept untouched for the cell where it belongs.
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def boxed_group(text: str) -> Optional[str]:
    r"""The last `\boxed{...}` content, unreduced."""
    m = _BOXED.findall(text or "")
    for g in m[::-1]:
        if g.strip():
            return g.strip()
    return None


def extract_for(benchmark: str, text: str, n_options: int = 10) -> Optional[str]:
    """Du's extraction for `gsm8k`; the task's own answer form for the adapted cells."""
    if benchmark == "gsm8k":
        return parse_answer(text) or solve_math_problems(text)
    g = boxed_group(text)
    if benchmark in ("mmlu_pro", "gpqa"):
        allowed = {chr(ord("A") + i) for i in range(max(1, min(10, n_options)))}
        if g:
            up = g.strip().strip(".").upper()
            if up in allowed:
                return up
        from .official_scorers import letter_answer  # noqa: PLC0415
        return letter_answer(text, n_options)
    if benchmark == "strategyqa":
        if g:
            low = g.strip().lower().strip(".")
            if low in ("yes", "true"):
                return "yes"
            if low in ("no", "false"):
                return "no"
        from .official_scorers import boolean_answer  # noqa: PLC0415
        b = boolean_answer(text)
        return None if b is None else ("yes" if b else "no")
    if benchmark == "squad_v2":
        # The span as written, or the explicit abstention. Not normalised here: the scorer applies
        # SQuAD's own normalisation, and normalising twice would hide what the agent actually said.
        return g
    return g


def most_frequent(xs: List[Optional[str]]) -> Optional[str]:
    """Du's post-hoc aggregation. Ties resolve to the first-seen, as `Counter` does."""
    vals = [x for x in xs if x is not None]
    return collections.Counter(vals).most_common(1)[0][0] if vals else None


def run_one(question: str, *, benchmark: str, alias: str, agents: int, rounds: int, max_tokens: int,
            timeout_s: float, retries: int, n_options: int = 10) -> Dict[str, Any]:
    """One debate. Returns the full trajectory, every raw reply, and the per-call records."""
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
            sent = [dict(m) for m in ctx]          # exactly what the agent received
            c0 = time.time()
            text, rec = chat(registry=REGISTRY, alias=alias, messages=sent,
                             temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                             retries=retries, timeout_s=timeout_s)
            ctx.append({"role": "assistant", "content": text})
            calls.append({
                "round": rnd, "agent_index": i, "seq": len(calls),
                "messages_sent": sent, "raw_reply": text,
                "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
                "finish_reason": rec.finish_reason, "attempt": rec.attempt,
                "latency_s": round(time.time() - c0, 3), "usd": rec.usd,
                "request_sha": rec.request_sha, "served_model": rec.served_model,
                "fingerprint": rec.fingerprint,
                "extracted_answer": extract_for(benchmark, text, n_options),
            })
            order.append({"seq": len(order), "round": rnd, "agent_index": i})
    finals = [ctx[-1]["content"] for ctx in ctxs]
    per_agent = [extract_for(benchmark, f, n_options) for f in finals]
    return {
        "agent_contexts": ctxs,
        "calls": calls, "order": order,
        "per_agent_final_answer": per_agent,
        "majority_answer": most_frequent(per_agent),
        "n_distinct_final_answers": len({x for x in per_agent if x is not None}),
        # Du MAD has no stop rule and solicits no acceptance. Both are contract facts.
        "native_stop_reason": "fixed-round-budget-exhausted",
        "accepted_delivery": "not-measured",
        "accepted_delivery_note": "Du MAD solicits no acceptance of a delivered artifact and defines "
                                  "no stopping rule; absence of a verdict is not a refusal",
        "wall_s": round(time.time() - t0, 2),
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0.0 for c in calls), 8),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agents", type=int, default=3)     # gen_gsm.py:32
    ap.add_argument("--rounds", type=int, default=2)     # gen_gsm.py:33
    ap.add_argument("--max-tokens", type=int, default=1024)
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
    ep = REGISTRY.get(a.alias)
    bench = bman.get("benchmark") or os.path.basename(a.block).split("_block")[0]
    if bench not in FRAMING:
        print(f"  REFUSING: no declared framing for benchmark {bench!r}")
        return 1
    tasks = block[: a.limit] if a.limit else block

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    t0 = time.time()
    for k, t in enumerate(tasks, 1):
        tid = t["example_id"]
        try:
            r = run_one(task_question(bench, t), benchmark=bench, alias=a.alias,
                        agents=a.agents, rounds=a.rounds, max_tokens=a.max_tokens,
                        timeout_s=a.timeout_s, retries=a.retries,
                        n_options=(t.get("metadata") or {}).get("n_options", 10))
        except (TransportFailure, Exception) as e:  # noqa: BLE001 — a lost task is named, not dropped
            failures.append({"example_id": tid, "error": f"{type(e).__name__}: {e}"})
            print(f"  [{k}/{len(tasks)}] {tid} FAILED {type(e).__name__}")
            continue
        r.update({
            # Stable unique identity for task x mechanism x model x repeat.
            "run_id": f"dumad-{a.alias}-{tid}-r0",
            "example_id": tid, "dataset_id": t["dataset_id"],
            "mechanism": "du-mad", "model_alias": a.alias, "repeat": 0,
            "question": t["inputs"][0], "references": t["references"],
        })
        records.append(r)
        if k % 10 == 0 or k == len(tasks):
            print(f"  [{k}/{len(tasks)}] {len(records)} ok, {len(failures)} failed, "
                  f"{time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)                       # atomic: a reader never sees a half file

    src = os.path.join(os.path.dirname(__file__), "du_mad.py")
    manifest = {
        "kind": "native-run", "mechanism": "du-mad",
        "upstream_source": "vendor/mad/gsm/gen_gsm.py",
        "upstream_sha256": hashlib.sha256(
            open(os.path.join(os.path.dirname(__file__), "..", "..", "..", "vendor", "mad", "gsm",
                              "gen_gsm.py"), "rb").read()).hexdigest(),
        "adapter_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
        "cell_label": "R+D (Du's own runner)" if bench == "gsm8k"
                      else "cross-task adaptation: answer clause and task noun only",
        "answer_clause": ANSWER_CLAUSE[bench],
        "preserved": ["opening prompt", "others prompt", "self-check prompt", "roles user/assistant",
                      "agents=3", "rounds=2", "read index 2*round-1", "round-robin order",
                      "post-hoc majority over per-agent extracted answers"],
        "replaced": ["data source (author's absolute path -> frozen block)",
                     "model client (openai<1.0 -> our transport, which records tokens and cost)",
                     "no reshuffle (the block is already a pinned task list)"],
        "not_added": ["writer", "acceptance solicitation", "stop rule"],
        "block_path": a.block, "block_sha256": bman.get("block_sha256"),
        "benchmark": bman.get("benchmark"),
        "endpoint": ep.identity(), "endpoint_url": ep.base_url,
        "agents": a.agents, "rounds": a.rounds, "max_tokens": a.max_tokens,
        "retries": a.retries, "num_samples": len(tasks),
        "records_written": len(records),
        "failed_example_ids": [f["example_id"] for f in failures],
        "n_failed": len(failures),
        "reason_counts": dict(collections.Counter(
            f["error"].split(":")[0] for f in failures)),
        "endpoint_tokens": {"prompt": sum(r["tokens_in"] for r in records),
                            "generation": sum(r["tokens_out"] for r in records)},
        "usd": round(sum(r["usd"] for r in records), 8),
        "wall_s": round(time.time() - t0, 1),
        "collected_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": "Pilot. The mechanism's own output is the primary object; no writer was added and no "
                "prompt, role, round count or aggregation rule was altered.",
    }
    with open(a.out.replace(".json", "") + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"  wrote {len(records)} records, {len(failures)} failures, "
          f"{manifest['endpoint_tokens']}, ${manifest['usd']:.6f}, {manifest['wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
