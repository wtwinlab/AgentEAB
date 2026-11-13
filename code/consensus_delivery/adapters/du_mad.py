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

OPENING = ("Can you solve the following math problem? {q} Explain your reasoning. Your final answer "
           "should be a single numerical number, in the form \\boxed{{answer}}, at the end of your "
           "response. ")

SELF_CHECK = ("Can you double check that your answer is correct. Please reiterate your answer, with "
              "your final answer a single numerical number, in the form \\boxed{{answer}}.")

OTHERS_PREFIX = "These are the solutions to the problem from other agents: "
OTHERS_ITEM = "\n\n One agent solution: ```{a}```"
OTHERS_SUFFIX = ("\n\n Using the solutions from other agents as additional information, can you "
                 "provide your answer to the math problem? \n The original math problem is {q}. "
                 "Your final answer should be a single numerical number, in the form "
                 "\\boxed{{answer}}, at the end of your response.")


ANSWER_CLAUSE = {
    "gsm8k": "Your final answer should be a single numerical number, in the form "
             "\\boxed{{answer}}, at the end of your response.",
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

FRAMING = {
    "gsm8k": "Can you solve the following math problem? {q} Explain your reasoning. ",
    "mmlu_pro": "Can you answer the following multiple-choice question? {q} Explain your reasoning. ",
    "gpqa": "Can you answer the following multiple-choice question? {q} Explain your reasoning. ",
    "strategyqa": "Can you answer the following yes/no question? {q} Explain your reasoning. ",
    "squad_v2": "Can you answer the following reading-comprehension question? {q} Explain your "
                "reasoning. ",
}

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
    text = (FRAMING[benchmark].format(q=question)
            + ANSWER_CLAUSE[benchmark].format(answer="answer") + " ")
    if benchmark == "gsm8k":
        assert text == OPENING.format(q=question), "the gsm8k opening drifted from Du's literal"
    return text


def others_for(benchmark: str, others: List[List[Dict[str, str]]], question: str,
               idx: int) -> Dict[str, str]:
    if not others:
        return {"role": "user", "content": SELF_CHECK}
    body = OTHERS_PREFIX + "".join(OTHERS_ITEM.format(a=c[idx]["content"]) for c in others)
    text = body + SUFFIX_FRAMING[benchmark].format(q=question) \
        + ANSWER_CLAUSE[benchmark].format(answer="answer")
    if benchmark == "gsm8k":
        assert text == construct_message(others, question, idx)["content"], \
            "the gsm8k others message drifted from Du's literal"
    return {"role": "user", "content": text}


def task_question(benchmark: str, rec: Dict[str, Any]) -> str:
    q = rec["inputs"][0]
    if benchmark == "squad_v2" and rec.get("context"):
        return f"{rec['context'][0]}\n\nQuestion: {q}"
    return q


def construct_message(others: List[List[Dict[str, str]]], question: str, idx: int) -> Dict[str, str]:
    if not others:
        return {"role": "user", "content": SELF_CHECK}
    s = OTHERS_PREFIX
    for ctx in others:
        s += OTHERS_ITEM.format(a=ctx[idx]["content"])
    s += OTHERS_SUFFIX.format(q=question)
    return {"role": "user", "content": s}


def parse_answer(text: str) -> Optional[str]:
    for m in re.findall(r"\{([0-9.,$]*)\}", text or "")[::-1]:
        sol = re.sub(r"[^0-9.]", "", m)
        if sol:
            return sol
    return None


def solve_math_problems(text: str) -> Optional[str]:
    m = re.findall(r"\d+\.?\d*", text or "")
    return m[-1] if m else None


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def boxed_group(text: str) -> Optional[str]:
    m = _BOXED.findall(text or "")
    for g in m[::-1]:
        if g.strip():
            return g.strip()
    return None


def extract_for(benchmark: str, text: str, n_options: int = 10) -> Optional[str]:
    if benchmark == "gsm8k":
        return parse_answer(text) or solve_math_problems(text)
    g = boxed_group(text)
    if benchmark in ("mmlu_pro", "gpqa"):
        allowed = {chr(ord("A") + i) for i in range(max(1, min(10, n_options)))}
        if g:
            up = g.strip().strip(".").upper()
            if up in allowed:
                return up
        from .official_scorers import letter_answer
        return letter_answer(text, n_options)
    if benchmark == "strategyqa":
        if g:
            low = g.strip().lower().strip(".")
            if low in ("yes", "true"):
                return "yes"
            if low in ("no", "false"):
                return "no"
        from .official_scorers import boolean_answer
        b = boolean_answer(text)
        return None if b is None else ("yes" if b else "no")
    if benchmark == "squad_v2":
        return g
    return g


def most_frequent(xs: List[Optional[str]]) -> Optional[str]:
    vals = [x for x in xs if x is not None]
    return collections.Counter(vals).most_common(1)[0][0] if vals else None


def run_one(question: str, *, benchmark: str, alias: str, agents: int, rounds: int, max_tokens: int,
            timeout_s: float, retries: int, n_options: int = 10) -> Dict[str, Any]:
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
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
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
        except (TransportFailure, Exception) as e:
            failures.append({"example_id": tid, "error": f"{type(e).__name__}: {e}"})
            print(f"  [{k}/{len(tasks)}] {tid} FAILED {type(e).__name__}")
            continue
        r.update({
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
    os.replace(tmp, a.out)

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
