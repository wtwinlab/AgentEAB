r"""Debate-or-Vote (Choi, Zhu, Li; NeurIPS 2025), as released, over our serving path.

**Ties are resolved at random**, not by first appearance: `random.choice(most_common)`
(`evaluator.py:64`). The released code seeds `random` once globally from `--seed 42`, so its draw
depends on how many ties preceded it in the run — reproducible only for an identical task order.
Here the draw is seeded **per task** from the task id, which keeps "random among the tied" while
making a single cell reproducible on its own. Each record stores the tied set, so any other
tie-break can be recomputed offline.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ..providers import TransportFailure, chat
from ..registry_live import REGISTRY

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UPSTREAM = "_external/mechanism_screening_2026-10-08/repos/debate-or-vote @ 82c929e"

#: `evaluator.py:17-38`, the default branch: `--bae` off, `--cot` off.
SUFFIX_NUMERIC = (' Make sure to state your final answer in curly brackets at the very end of your '
                  'response, just like: "{final answer: 123}".')
SUFFIX_MCQ = (' Make sure to state your final answer choice in curly brackets at the very end of '
              'your response, just like: "{final answer: (A)}".')
#: StrategyQA and SQuAD 2.0 are **task adaptations**: the released code implements arithmetic, MCQ
#: and summarisation only (`main.py:160-173` raises `NotImplementedError` otherwise).
SUFFIX_BOOLEAN = (' Make sure to state your final answer in curly brackets at the very end of your '
                  'response, just like: "{final answer: yes}".')
SUFFIX_SPAN = (' Make sure to state your final answer in curly brackets at the very end of your '
               'response, just like: "{final answer: the answer span}". If the passage does not '
               'contain the answer, state "{final answer: unanswerable}".')

SUFFIX = {"gsm8k": SUFFIX_NUMERIC, "mmlu_pro": SUFFIX_MCQ, "gpqa": SUFFIX_MCQ,
          "strategyqa": SUFFIX_BOOLEAN, "squad_v2": SUFFIX_SPAN}
ADAPTED = {"strategyqa", "squad_v2"}
ANSWER_KIND = {"gsm8k": "numeric", "mmlu_pro": "mcq", "gpqa": "mcq",
               "strategyqa": "boolean", "squad_v2": "span"}


def peer_message(question_with_suffix: str, responses: List[str], i: int) -> str:
    """`get_new_message`, decentralised and all-to-all, transcribed from `main.py:79-96`."""
    msg = "These are the recent opinions from other agents: "
    peers = [r for j, r in enumerate(responses) if j != i]
    for other in peers:
        msg += f"\n\nOne of the agents' response: \n{other}\n"
    msg += f"\n\nThis was your most recent opinion:\n{responses[i]}\n"
    msg += ("\n\nUse these opinions carefully as additional advice to revise your recent opinion to "
            f"give your final answer to the question:\n{question_with_suffix}")
    return msg


def bracketed(response: str) -> Optional[str]:
    """The last `{...}` span with `final answer:` stripped. `None` where there is none."""
    hits = re.findall(r"\{(.*?)\}", response or "", re.S)
    if not hits:
        return None
    return hits[-1].replace("final answer:", "").strip()


def parse_released(response: str, kind: str) -> str:
    """The released parser. `""` means no vote, which is what the upstream code uses."""
    pred = bracketed(response)
    if pred is None:
        return ""
    if kind == "numeric":
        try:
            return str(round(float(pred.replace(",", "").replace("$", "")), 1))
        except ValueError:
            return ""
    if kind == "mcq":
        # `evaluator.py:80-86`, verbatim including the second-character branch.
        if len(pred) == 0:
            return ""
        ch = pred[0] if len(pred) < 3 else pred[1]
        return f"({ch.upper()})"
    if kind == "boolean":
        low = pred.strip().lower()
        return "yes" if low.startswith("y") else "no" if low.startswith("n") else ""
    return " ".join(pred.split())


def parse_strict(response: str, kind: str) -> str:
    """A recorded alternative for `mcq`, anchored on `(X)`, so the released issue is measurable."""
    pred = bracketed(response)
    if pred is None:
        return ""
    if kind != "mcq":
        return parse_released(response, kind)
    # A **standalone** letter, not one inside a word.
    m = re.search(r"(?<![A-Za-z])\(?([A-J])\)?(?![A-Za-z])", pred.upper())
    return f"({m.group(1)})" if m else ""


def tally(votes: List[str], task_id: str) -> Dict[str, Any]:
    """Plurality with a random tie-break, determinised per task. Every input is kept."""
    valid = [v for v in votes if v != ""]
    if not valid:
        # `evaluator.py:57-60`: all unparseable means the debate answer is the empty string.
        return {"answer": "", "valid_votes": 0, "winner_votes": 0, "tied": [],
                "tie_broken_at_random": False,
                "absence_cause": "no agent's answer could be parsed from a curly-bracket span"}
    counter = collections.Counter(valid)
    top = max(counter.values())
    # Counter preserves first-appearance order, which is the order the released `random.choice`
    # draws from. Sorting here would change which tied answer a given draw selects.
    most_common = [k for k, v in counter.items() if v == top]
    rng = random.Random(int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16))
    answer = rng.choice(most_common) if len(most_common) > 1 else most_common[0]
    return {"answer": answer, "valid_votes": len(valid), "winner_votes": top,
            "tied": most_common if len(most_common) > 1 else [],
            "tie_broken_at_random": len(most_common) > 1,
            "support_share_of_valid": round(top / len(valid), 4),
            "support_share_of_agents": round(top / len(votes), 4),
            "absence_cause": None}


def run_one(question: str, *, task_id: str, benchmark: str, alias: str, agents: int, rounds: int,
            solver: str, max_tokens: int, timeout_s: float, retries: int) -> Dict[str, Any]:
    """One task, with the whole trajectory, every attempt, and both parses recorded."""
    kind = ANSWER_KIND[benchmark]
    q = question + SUFFIX[benchmark]
    contexts: List[List[Dict[str, str]]] = [[{"role": "user", "content": q}] for _ in range(agents)]
    calls: List[Dict[str, Any]] = []
    order: List[Dict[str, Any]] = []
    per_round: List[Dict[str, Any]] = []
    t0 = time.time()
    seq = 0
    latest: List[str] = [""] * agents

    for rnd in range(rounds):
        if rnd > 0:
            for i in range(agents):
                contexts[i].append({"role": "user",
                                    "content": peer_message(q, latest, i)})
        for i in range(agents):
            text, rec = chat(registry=REGISTRY, alias=alias, messages=contexts[i],
                             temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                             retries=retries, timeout_s=timeout_s)
            contexts[i].append({"role": "assistant", "content": text})
            latest[i] = text
            calls.append({
                "seq": seq, "round": rnd, "agent_index": i, "attempt": rec.attempt,
                "messages_sent": [dict(m) for m in contexts[i][:-1]],
                "raw_reply": text,
                "vote_released": parse_released(text, kind),
                "vote_strict": parse_strict(text, kind),
                "request_sha": rec.request_sha, "served_model": rec.served_model,
                "fingerprint": rec.fingerprint, "finish_reason": rec.finish_reason,
                "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
                "latency_s": rec.latency_s, "usd": rec.usd,
            })
            order.append({"seq": seq, "round": rnd, "agent_index": i})
            seq += 1
        votes = [parse_released(t, kind) for t in latest]
        per_round.append({"round": rnd, "votes": votes,
                          "tally": tally(votes, f"{task_id}:r{rnd}")})

    # `--solver vote` tallies the final round's ballots; `--solver debate` takes the same final
    # responses but is scored as a debate outcome. Both are recorded so neither needs a re-run.
    final_votes = [parse_released(t, kind) for t in latest]
    final_votes_strict = [parse_strict(t, kind) for t in latest]
    vote_result = tally(final_votes, task_id)
    strict_result = tally(final_votes_strict, task_id)
    return {
        "run_id": f"dov-{solver}-{alias}-{task_id}-r0",
        "example_id": task_id, "mechanism": f"dov_{solver}", "model_alias": alias, "repeat": 0,
        "question": question, "question_with_suffix": q,
        "benchmark": benchmark, "answer_kind": kind,
        "is_task_adaptation": benchmark in ADAPTED,
        "adaptation_note": ("the released code implements arithmetic, MCQ and summarisation only; "
                            "this benchmark is a labelled task adaptation that keeps the "
                            "curly-bracket answer convention"
                            if benchmark in ADAPTED else None),
        "agents": agents, "rounds": rounds, "solver": solver,
        "topology": "decentralised all-to-all (released default; --sparse and --centralized off)",
        "per_agent_final_answer": [v if v != "" else None for v in final_votes],
        "per_agent_final_answer_strict": [v if v != "" else None for v in final_votes_strict],
        "n_distinct_final_answers": len({v for v in final_votes if v != ""}),
        "majority_answer": vote_result["answer"] or None,
        "majority_answer_strict": strict_result["answer"] or None,
        "parses_agree": vote_result["answer"] == strict_result["answer"],
        "tally": vote_result, "tally_strict": strict_result,
        "rounds_detail": per_round,
        # `vote` tallies ballots, so a ballot is a stance and acceptance is measured. `debate` takes
        # the final text with no ballot, so acceptance is NOT measured — never a refusal.
        "accepted_delivery": (bool(vote_result["valid_votes"]) if solver == "vote"
                              else "not-measured"),
        "accepted_delivery_note": (
            "a ballot cast in the final tally is this protocol's expression of a position"
            if solver == "vote" else
            "the debate solver takes the final response with no ballot; it solicits no acceptance "
            "of a delivered artifact, and absence of a verdict is not a refusal"),
        "native_stop_reason": f"fixed-round-budget-exhausted after {rounds} rounds",
        "calls": calls, "order": order,
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0 for c in calls), 6),
        "wall_s": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------- transcription assertions
def _assert_transcription() -> None:
    """The literals, checked against the released text rather than trusted.
    """
    assert SUFFIX_NUMERIC == (
        ' Make sure to state your final answer in curly brackets at the very end of your response, '
        'just like: "{final answer: 123}".'), "numeric suffix drifted"
    assert SUFFIX_MCQ == (
        ' Make sure to state your final answer choice in curly brackets at the very end of your '
        'response, just like: "{final answer: (A)}".'), "mcq suffix drifted"
    m = peer_message("Q?", ["r0", "r1", "r2"], 1)
    assert m.startswith("These are the recent opinions from other agents: "), "peer prefix drifted"
    assert "One of the agents' response: \nr0\n" in m, "peer body drifted"
    assert "This was your most recent opinion:\nr1\n" in m, "own-opinion line drifted"
    assert m.rstrip().endswith("give your final answer to the question:\nQ?"), "peer tail drifted"
    assert "r1\n" in m and m.count("One of the agents' response") == 2, "peer count wrong"
    # The released MCQ parser's second-character branch, reproduced exactly.
    assert parse_released("x {final answer: (B)} y", "mcq") == "(B)"
    assert parse_released("x {final answer: B} y", "mcq") == "(B)"
    assert parse_released("x {final answer: answer B} y", "mcq") == "(N)", (
        "the released parser takes the second character of a long bracketed string; that is the "
        "defect being preserved, so this assertion must keep failing loudly if it changes")
    assert parse_strict("x {final answer: answer B} y", "mcq") == "(B)", (
        "a standalone-letter rule is required: a bare [A-J] search matches the A in ANSWER")
    assert parse_strict("x {final answer: (C)} y", "mcq") == "(C)"
    assert parse_strict("x {final answer: Avogadro} y", "mcq") == "", (
        "a letter inside a word is not an answer")
    assert parse_released("no brackets here", "numeric") == ""


_assert_transcription()


# --------------------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--solver", default="vote", choices=("vote", "debate"))
    ap.add_argument("--agents", type=int, default=5)        # main.py:43
    ap.add_argument("--rounds", type=int, default=5)        # main.py:58
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
    bench = bman.get("benchmark") or os.path.basename(a.block).split("_block")[0]
    if bench not in SUFFIX:
        print(f"  REFUSING: no declared suffix for benchmark {bench!r}")
        return 1
    ep = REGISTRY.get(a.alias)
    tasks = block[: a.limit] if a.limit else block

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    t0 = time.time()
    for k, t in enumerate(tasks, 1):
        tid = t["example_id"]
        question = t["inputs"][0]
        if bench == "squad_v2":
            ctx = (t.get("context") or [""])[0] if isinstance(t.get("context"), list) else ""
            question = (f"{ctx}\n\n{question}" if ctx else question)
        elif bench in ("mmlu_pro", "gpqa"):
            opts = (t.get("metadata") or {}).get("options") or []
            if opts:
                lettered = "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(opts))
                question = f"{question}\n\n{lettered}"
        try:
            r = run_one(question, task_id=tid, benchmark=bench, alias=a.alias, agents=a.agents,
                        rounds=a.rounds, solver=a.solver, max_tokens=a.max_tokens,
                        timeout_s=a.timeout_s, retries=a.retries)
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
    os.replace(tmp, a.out)                       # atomic: a reader never sees a half file

    src = os.path.join(os.path.dirname(__file__), "debate_or_vote.py")
    manifest = {
        "kind": "native-run", "mechanism": f"dov_{a.solver}",
        "upstream_source": UPSTREAM,
        "adapter_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
        "benchmark": bench, "block_path": a.block,
        "block_sha256": bman.get("block_sha256"),
        "endpoint": {"alias": ep.alias, "provider": ep.provider, "served_name": ep.served_name,
                     "checkpoint": ep.checkpoint, "revision": ep.revision,
                     "chat_template_sha": ep.chat_template_sha,
                     "serving_image": ep.serving_image},
        "endpoint_url": ep.base_url, "model_name": ep.served_name,
        "solver": a.solver, "agents": a.agents, "rounds": a.rounds,
        "topology": "decentralised all-to-all (released default)",
        "temperature": 0.0, "max_tokens": a.max_tokens, "retries": a.retries,
        "num_samples": len(tasks), "records_written": len(records),
        "failed_example_ids": [f["example_id"] for f in failures],
        "n_failed": len(failures),
        "reason_counts": dict(collections.Counter(
            f["error"].split(":")[0] for f in failures)),
        "unexplained_absent": len(tasks) - len(records) - len(failures),
        "is_task_adaptation": bench in ADAPTED,
        "tie_break": ("random among the tied, seeded per task id from sha256 so a cell is "
                      "reproducible on its own; the released code seeds `random` once globally"),
        "endpoint_tokens": {"prompt": sum(r["tokens_in"] for r in records),
                            "generation": sum(r["tokens_out"] for r in records)},
        "usd": round(sum(r["usd"] for r in records), 6),
        "wall_s": round(time.time() - t0, 1),
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": ("Debate-or-Vote as released, both parses recorded. The mechanism's own output is "
                 "the primary object; no writer, gate or stop rule was added."),
    }
    with open(a.out.replace(".json", "") + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"  {os.path.basename(a.block)}: {len(records)} records, {len(failures)} failed, "
          f"{manifest['wall_s']:.0f}s, tokens={manifest['endpoint_tokens']}")
    return 0 if not failures else 0


if __name__ == "__main__":
    sys.exit(main())
