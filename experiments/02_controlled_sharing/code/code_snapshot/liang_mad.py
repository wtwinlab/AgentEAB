r"""Liang MAD — "Encouraging Divergent Thinking in LLMs through Multi-Agent Debate", as released.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ..providers import TransportFailure, chat
from ..registry_live import REGISTRY

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UPSTREAM = "_external/mechanism_screening_2026-10-08/repos/liang-mad @ e58d146"

#: `code/utils/config4all.json`, verbatim.
PLAYER_META = ("You are a debater. Hello and welcome to the debate. It's not necessary to fully "
               "agree with each other's perspectives, as our objective is to find the correct "
               "answer.\nThe debate topic is stated as follows:\n##debate_topic##")
MODERATOR_META = ("You are a moderator. There will be two debaters involved in a debate. They will "
                  "present their answers and discuss their perspectives on the following topic: "
                  "\"##debate_topic##\"\nAt the end of each round, you will evaluate answers and "
                  "decide which is correct.")
AFFIRMATIVE_PROMPT = "##debate_topic##"
NEGATIVE_PROMPT = "##aff_ans##\n\nYou disagree with my answer. Provide your answer and reasons."
MODERATOR_PROMPT = (
    "Now the ##round## round of debate for both sides has ended.\n\nAffirmative side arguing:\n"
    "##aff_ans##\n\nNegative side arguing: ##neg_ans##\n\nYou, as the moderator, will evaluate both "
    "sides' answers and determine if there is a clear preference for an answer candidate. If so, "
    "please summarize your reasons for supporting affirmative/negative side and give the final "
    "answer that you think is correct, and the debate will conclude. If not, the debate will "
    "continue to the next round. Now please output your answer in json format, with the format as "
    "follows: {\"Whether there is a preference\": \"Yes or No\", \"Supported Side\": \"Affirmative "
    "or Negative\", \"Reason\": \"\", \"debate_answer\": \"\"}. Please strictly output in JSON "
    "format, do not output irrelevant content.")
JUDGE_LAST1 = ("Affirmative side arguing: ##aff_ans##\n\nNegative side arguing: ##neg_ans##\n\nNow, "
               "what answer candidates do we have? Present them without reasons.")
JUDGE_LAST2 = ("Therefore, ##debate_topic##\nPlease summarize your reasons and give the final "
               "answer that you think is correct. Now please output your answer in json format, "
               "with the format as follows: {\"Reason\": \"\", \"debate_answer\": \"\"}. Please "
               "strictly output in JSON format, do not output irrelevant content.")
DEBATE_PROMPT = "##oppo_ans##\n\nDo you agree with my perspective? Please provide your reasons and answer."

ANSWER_KIND = {"gsm8k": "numeric", "mmlu_pro": "mcq", "gpqa": "mcq", "strategyqa": "boolean"}
#: Its released tasks are commonsense machine translation and CounterintuitiveQA. Everything here is
#: a labelled adaptation of the general debate configuration.
ADAPTED = {"gsm8k", "mmlu_pro", "gpqa", "strategyqa"}


def parse_json_reply(text: str) -> Optional[Dict[str, Any]]:
    t = text or ""
    if "{" not in t or "}" not in t:
        return None
    try:
        obj = json.loads(t[t.find("{"):t.rfind("}") + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def normalise(answer: Any, kind: str) -> str:
    a = str(answer if answer is not None else "").strip()
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
        m = re.search(r"(?<![A-Za-z])\(?([A-J])\)?(?![A-Za-z])", a.upper())
        return m.group(1) if m else ""
    if kind == "boolean":
        low = a.lower()
        return "yes" if low.startswith("y") else "no" if low.startswith("n") else ""
    return " ".join(a.split())


def has_preference(obj: Optional[Dict[str, Any]]) -> bool:
    """The moderator's own field, read case-insensitively on the value only."""
    if not obj:
        return False
    v = obj.get("Whether there is a preference")
    return isinstance(v, str) and v.strip().lower().startswith("y")


def run_one(question: str, *, task_id: str, benchmark: str, alias: str, max_round: int,
            max_tokens: int, timeout_s: float, retries: int) -> Dict[str, Any]:
    kind = ANSWER_KIND[benchmark]
    calls: List[Dict[str, Any]] = []
    order: List[Dict[str, Any]] = []
    rounds_detail: List[Dict[str, Any]] = []
    t0 = time.time()
    seq = 0

    def ask(role: str, messages: List[Dict[str, str]], rnd: int) -> str:
        nonlocal seq
        text, rec = chat(registry=REGISTRY, alias=alias, messages=messages,
                         temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                         retries=retries, timeout_s=timeout_s)
        calls.append({
            "seq": seq, "round": rnd, "role": role, "attempt": rec.attempt,
            "messages_sent": [dict(m) for m in messages], "raw_reply": text,
            "request_sha": rec.request_sha, "served_model": rec.served_model,
            "fingerprint": rec.fingerprint, "finish_reason": rec.finish_reason,
            "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
            "latency_s": rec.latency_s, "usd": rec.usd,
        })
        order.append({"seq": seq, "round": rnd, "role": role})
        seq += 1
        return text

    player_sys = PLAYER_META.replace("##debate_topic##", question)
    mod_sys = MODERATOR_META.replace("##debate_topic##", question)
    aff_hist: List[Dict[str, str]] = [{"role": "system", "content": player_sys}]
    neg_hist: List[Dict[str, str]] = [{"role": "system", "content": player_sys}]
    mod_hist: List[Dict[str, str]] = [{"role": "system", "content": mod_sys}]

    aff_ans = neg_ans = ""
    final_answer = ""
    settled_by = "unsettled"
    stopped_at: Optional[int] = None
    rule_satisfied = False
    supported_side: Optional[str] = None

    for rnd in range(max_round):
        if rnd == 0:
            aff_hist.append({"role": "user",
                             "content": AFFIRMATIVE_PROMPT.replace("##debate_topic##", question)})
        else:
            aff_hist.append({"role": "user", "content": DEBATE_PROMPT.replace("##oppo_ans##",
                                                                             neg_ans)})
        aff_ans = ask("affirmative", aff_hist, rnd)
        aff_hist.append({"role": "assistant", "content": aff_ans})

        neg_hist.append({"role": "user", "content": (
            NEGATIVE_PROMPT.replace("##aff_ans##", aff_ans) if rnd == 0
            else DEBATE_PROMPT.replace("##oppo_ans##", aff_ans))})
        neg_ans = ask("negative", neg_hist, rnd)
        neg_hist.append({"role": "assistant", "content": neg_ans})

        mod_hist.append({"role": "user", "content": (
            MODERATOR_PROMPT.replace("##round##", str(rnd + 1))
            .replace("##aff_ans##", aff_ans).replace("##neg_ans##", neg_ans))})
        mod_text = ask("moderator", mod_hist, rnd)
        mod_hist.append({"role": "assistant", "content": mod_text})
        mod_obj = parse_json_reply(mod_text)
        pref = has_preference(mod_obj)
        rounds_detail.append({
            "round": rnd, "moderator_json": mod_obj,
            "moderator_preference": pref,
            "supported_side": (mod_obj or {}).get("Supported Side"),
            "moderator_answer": normalise((mod_obj or {}).get("debate_answer"), kind),
            "moderator_json_contract_met": mod_obj is not None and
            "Whether there is a preference" in mod_obj,
        })
        if pref:
            # The released stop rule: the moderator says there is a clear preference.
            final_answer = normalise((mod_obj or {}).get("debate_answer"), kind)
            supported_side = (mod_obj or {}).get("Supported Side")
            settled_by = "moderator preference"
            stopped_at = rnd
            rule_satisfied = True
            break

    judge_steps: List[Dict[str, Any]] = []
    if not rule_satisfied:
        # The released fallback: two judge turns, candidates then the final answer.
        j_hist: List[Dict[str, str]] = [{"role": "system", "content": mod_sys}]
        j_hist.append({"role": "user", "content": JUDGE_LAST1.replace("##aff_ans##", aff_ans)
                       .replace("##neg_ans##", neg_ans)})
        cand = ask("judge_candidates", j_hist, max_round)
        j_hist.append({"role": "assistant", "content": cand})
        j_hist.append({"role": "user",
                       "content": JUDGE_LAST2.replace("##debate_topic##", question)})
        verdict = ask("judge_final", j_hist, max_round)
        obj = parse_json_reply(verdict)
        final_answer = normalise((obj or {}).get("debate_answer"), kind)
        settled_by = "judge fallback after the round budget"
        judge_steps = [{"step": "candidates", "text": cand},
                       {"step": "final", "json": obj,
                        "json_contract_met": obj is not None and "debate_answer" in obj}]

    delivered = final_answer != ""
    return {
        "run_id": f"liangmad-{alias}-{task_id}-r0",
        "example_id": task_id, "mechanism": "liang_mad", "model_alias": alias, "repeat": 0,
        "question": question, "benchmark": benchmark, "answer_kind": kind,
        "is_task_adaptation": benchmark in ADAPTED,
        "adaptation_note": ("the released tasks are commonsense machine translation and "
                            "CounterintuitiveQA; this uses the repo's general debate configuration "
                            "`config4all.json` and is a labelled adaptation"),
        "roles": ["affirmative", "negative", "moderator", "judge (fallback)"],
        "is_adversarial": True,
        "adversarial_note": ("the debaters are assigned opposing sides and told agreement is not "
                             "required; every other mechanism here asks agents to converge"),
        "max_round": max_round, "rounds_run": len(rounds_detail),
        "per_agent_final_answer": [normalise(aff_ans, kind) or None,
                                   normalise(neg_ans, kind) or None],
        "n_distinct_final_answers": len({x for x in (normalise(aff_ans, kind),
                                                     normalise(neg_ans, kind)) if x}),
        "majority_answer": final_answer if delivered else None,
        "settled_by": settled_by, "supported_side": supported_side,
        # The moderator's preference is the condition the released loop breaks on. A debate settled
        # by the judge fallback still delivers, but the stop rule was not met.
        "rule_satisfied": rule_satisfied,
        "rule_note": ("the moderator answering \"Yes\" to \"Whether there is a preference\", which "
                      "is what the released loop breaks on. The moderator and the judge are parts "
                      "of this mechanism and are not this study's post-hoc judge"),
        "stopped_at_round": stopped_at,
        "native_stop_reason": settled_by,
        "accepted_delivery": "not-measured",
        "accepted_delivery_note": ("the debaters are assigned sides and are never asked to accept a "
                                   "delivered artifact; a moderator's preference is a third party's "
                                   "verdict, not the participants' acceptance, and absence of a "
                                   "verdict is not a refusal"),
        "absence_cause": (None if delivered else
                          "neither the moderator nor the judge returned a readable debate_answer"),
        "rounds_detail": rounds_detail, "judge_steps": judge_steps,
        "calls": calls, "order": order,
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0 for c in calls), 6),
        "wall_s": round(time.time() - t0, 2),
    }


def _assert_transcription() -> None:
    cfg = os.path.join(REPO, "_external", "mechanism_screening_2026-10-08", "repos", "liang-mad",
                       "code", "utils", "config4all.json")
    if os.path.exists(cfg):
        # The strongest check available: compare against the file itself rather than a memory of it.
        c = json.load(open(cfg, encoding="utf-8"))
        assert PLAYER_META == c["player_meta_prompt"], "player meta drifted from config4all.json"
        assert MODERATOR_META == c["moderator_meta_prompt"], "moderator meta drifted"
        assert AFFIRMATIVE_PROMPT == c["affirmative_prompt"], "affirmative prompt drifted"
        assert NEGATIVE_PROMPT == c["negative_prompt"], "negative prompt drifted"
        assert MODERATOR_PROMPT == c["moderator_prompt"], "moderator prompt drifted"
        assert JUDGE_LAST1 == c["judge_prompt_last1"], "judge last1 drifted"
        assert JUDGE_LAST2 == c["judge_prompt_last2"], "judge last2 drifted"
        assert DEBATE_PROMPT == c["debate_prompt"], "debate prompt drifted"
    assert has_preference({"Whether there is a preference": "Yes"}) is True
    assert has_preference({"Whether there is a preference": "no"}) is False
    assert has_preference({}) is False and has_preference(None) is False
    assert normalise("(C)", "mcq") == "C" and normalise("Avogadro", "mcq") == ""
    assert normalise("the answer is 7 apples", "numeric") == "7.0"
    assert parse_json_reply('{"debate_answer": "yes"}')["debate_answer"] == "yes"
    assert parse_json_reply("no json") is None


_assert_transcription()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-round", type=int, default=3)     # code: max_round=3
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
        mechanism="liang_mad", aliases=[a.alias], limit=a.limit, upstream=UPSTREAM,
        per_task=lambda q, tid, task: run_one(
            q, task_id=tid, benchmark=bench, alias=a.alias, max_round=a.max_round,
            max_tokens=a.max_tokens, timeout_s=a.timeout_s, retries=a.retries),
        manifest_extra={
            "max_round": a.max_round, "temperature": 0.0, "max_tokens": a.max_tokens,
            "config": "code/utils/config4all.json (the general configuration, not config4tran)",
            "roles": ["affirmative", "negative", "moderator", "judge (fallback)"],
            "is_adversarial": True,
            "stop_rule": ("the moderator answering Yes to \"Whether there is a preference\"; on "
                          "exhaustion a two-step judge pass settles it and rule_satisfied is False"),
            "mechanism_internal_judge": ("the moderator and judge belong to this mechanism and are "
                                         "NOT workstream B's post-hoc judge, which reads finished "
                                         "records and gates nothing"),
            "is_task_adaptation": bench in ADAPTED,
        })


if __name__ == "__main__":
    sys.exit(main())
