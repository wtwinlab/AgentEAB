r"""Debate budgets and independent drafts on BIRD: a runtime override of the extension collector.

Imported before ext_collect. Points the collector at this collection's run identifier and output
directory, the 198 BIRD questions of `questions.json`, Du-MAD and MAD-M2, and the Llama-3-8B and
Qwen2.5-7B servers, and adds the budget conditions. Prompts, adapters, task text, the 1,024-token
completion budget, stopping and return rules are the collector's own; a condition changes only the
number of agents and rounds and, for one draft condition, the decoding temperature.

Conditions:
  FULL           3 agents, 2 rounds (6 calls), the paper's full-response debate
  R3_FULL        3 agents, 3 rounds (9 calls)
  R4_FULL        3 agents, 4 rounds (12 calls)
  A5_FULL        5 agents, 2 rounds (10 calls)
  DRAFTS20       initial round only, 20 agents: 20 independent initial drafts with the protocol's
                 own initial prompt and decoding (Du-MAD greedy, MAD-M2 temperature 1)
  T07_DRAFTS20   Du-MAD only: as DRAFTS20 at temperature 0.7, each call with its own server seed

Later rounds use the protocol's own templates: Du-MAD shows the peers' previous-round replies,
MAD-M2 its debate prompt over the previous round with objective pruning. Du-MAD returns the
majority of the last round, MAD-M2 the answer of its consensus rule.

Environment: BM_ARMS, BM_PROTOS, BM_BACKBONES, BM_REPEATS (default 2), BM_PORT_L8, BM_PORT_Q7,
BM_LANE_TAG, BM_UNIT_SHARD=i/n, BM_RUN_ID (the run identifier; item seeds derive from it), BM_OUT.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time

BM_HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("REPO_ROOT", BM_HERE.parents[2]))
A = pathlib.Path(os.environ.get("PAM_ART", ROOT / "data/records/protocol_all_matching"))
for _p in (ROOT / "code", A, ROOT / "experiments/02_controlled_sharing/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RUN_ID = os.environ.get("BM_RUN_ID", "budget_matched_debates")
OUT = pathlib.Path(os.environ.get("BM_OUT", ROOT / "experiments/17_budget_matched_debates/data"))
LANE = os.environ.get("BM_LANE_TAG") or "lane"
UNITS = json.loads((BM_HERE / "questions.json").read_text())["questions"]

#: condition -> (collector transform, mode)
ARM_SPEC = {
    "FULL": ("FULL", {}),
    "R3_FULL": ("FULL", {"rounds": 3}),
    "R4_FULL": ("FULL", {"rounds": 4}),
    "A5_FULL": ("FULL", {"agents": 5}),
    "DRAFTS20": ("FULL", {"agents": 20, "rounds": 1}),
    "T07_DRAFTS20": ("FULL", {"agents": 20, "rounds": 1, "temperature": 0.7}),
}
DU_ONLY = {"T07_DRAFTS20"}
PROTOS = [p for p in os.environ.get("BM_PROTOS", "du_mad,madm2_objective").split(",") if p]
ARMS = [a for a in os.environ.get("BM_ARMS", "FULL").split(",") if a]
BACKBONES = [b for b in os.environ.get("BM_BACKBONES", "llama3-8b,qwen2.5-7b").split(",") if b]
ALIAS = {"llama3-8b": "llama3-8b-rx", "qwen2.5-7b": "qwen25-7b-rx"}
PORTS = {"llama3-8b-rx": int(os.environ["BM_PORT_L8"]) if os.environ.get("BM_PORT_L8") else None,
         "qwen25-7b-rx": int(os.environ["BM_PORT_Q7"]) if os.environ.get("BM_PORT_Q7") else None}
IMAGE = "vllm/vllm-openai:latest@d076d77d8dad (vLLM 0.30.0)"

import ext_design as D                                                   # noqa: E402

D.RUN_ID = RUN_ID
D.OUT = OUT
D.BENCH_ORDER = ["sql_bird_v2"]
D.PROTOS = list(PROTOS)
D.ARMS = list(ARMS)
D.PANELS = {p: list(BACKBONES) for p in PROTOS}
D.ALIAS = dict(ALIAS)
D.REPEATS = int(os.environ.get("BM_REPEATS", "2"))
_ids = list(UNITS)
if os.environ.get("BM_UNIT_SHARD"):
    _i, _n = (int(x) for x in os.environ["BM_UNIT_SHARD"].split("/"))
    _ids = [u for k, u in enumerate(_ids) if k % _n == _i]
_POP_SHA = hashlib.sha256(json.dumps(sorted(UNITS), sort_keys=True).encode()).hexdigest()[:16]


def _frozen(benchmark):
    assert benchmark == "sql_bird_v2"
    return {"benchmark": benchmark, "unit_ids": list(_ids), "sha256_16": _POP_SHA,
            "n_units": len(_ids), "units_name": "u198"}


def _supported(benchmark, protocol, arm):
    if arm not in ARM_SPEC:
        return f"unknown condition {arm}"
    if protocol not in ("du_mad", "madm2_objective"):
        return "this collection runs Du-MAD and MAD-M2"
    if arm in DU_ONLY and protocol != "du_mad":
        return f"{arm} is a Du-MAD condition"
    return None


D.frozen = _frozen
D.supported = _supported

import ext_stop_control as SC                                            # noqa: E402

SC.RUN_ID = RUN_ID
SC.OUT = OUT
SC.CONTROL = OUT / "CONTROL.json"
SC.GATE = OUT / "GATE.json"


def _gate_state():
    ok, g = False, "absent"
    if SC.GATE.exists():
        g = json.loads(SC.GATE.read_text())
        ok = str(g.get("state")).lower() == "open"
    return ok, [{"condition": f"{SC.GATE} says state=open", "satisfied": ok, "detail": g}]


def _write_atomic(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


SC.gate_state = _gate_state
SC._write_atomic = _write_atomic
SC.STATUS = OUT / f"STATUS_{LANE}.json"
SC.MANIFEST = OUT / f"manifest_{LANE}.jsonl"

import endpoints_all_matching as EP                                      # noqa: E402

_orig_endpoints = EP.endpoints


def _endpoints(ports):
    from consensus_delivery.providers import ModelEndpoint
    new = (ModelEndpoint(alias="llama3-8b-rx", base_url=f"http://127.0.0.1:{PORTS['llama3-8b-rx']}/v1",
                         served_name="llama3-8b-instruct", checkpoint="meta-llama/Meta-Llama-3-8B-Instruct",
                         provider="local", serving_image=IMAGE, max_context=8192, max_completion=2048),
           ModelEndpoint(alias="qwen25-7b-rx", base_url=f"http://127.0.0.1:{PORTS['qwen25-7b-rx']}/v1",
                         served_name="qwen2.5-7b-instruct", checkpoint="Qwen/Qwen2.5-7B-Instruct",
                         provider="local", serving_image=IMAGE, max_context=8192, max_completion=2048))
    return tuple(_orig_endpoints(ports)) + new


EP.endpoints = _endpoints

# ------------------------------------------------------------------ per-item mode
MODE: dict = {}

# innermost chat wrapper: sets the temperature and a per-call seed when the condition asks for it
from consensus_delivery import providers as _PV                          # noqa: E402

_real_chat = _PV.chat


def _chat_inner(**kw):
    n = MODE.get("n_calls", 0)
    MODE["n_calls"] = n + 1
    if MODE.get("temperature") is not None:
        kw["temperature"] = MODE["temperature"]
        s = int(hashlib.sha256(f"{MODE['item_seed']}|{n}".encode()).hexdigest()[:8], 16) % (2 ** 31)
        kw["extra"] = dict(kw.get("extra") or {}, seed=s)
        MODE.setdefault("call_seeds", []).append(s)
    MODE.setdefault("sent", []).append([dict(m) for m in kw.get("messages") or []])
    return _real_chat(**kw)


_PV.chat = _chat_inner

# ------------------------------------------------------------------ adapters: agents and rounds
from consensus_delivery.adapters import du_mad as DU                     # noqa: E402
from consensus_delivery.adapters import mad_m2 as M2                     # noqa: E402


def du_run_one(question, *, benchmark, alias, agents, rounds, max_tokens, timeout_s, retries,
               n_options=10):
    """du_mad.run_one with the condition's number of agents and rounds."""
    agents = int(MODE.get("agents") or agents)
    rounds = int(MODE.get("rounds") or rounds)
    clause = DU.ANSWER_CLAUSE[benchmark].format(answer="answer")
    extract = (lambda t: DU.extract_for(benchmark, t, n_options))
    opening = DU.opening_for(benchmark, question)
    ctxs = [[{"role": "user", "content": opening}] for _ in range(agents)]
    calls, order = [], []
    t0 = time.time()
    for rnd in range(rounds):
        for i, ctx in enumerate(ctxs):
            if rnd != 0:
                idx = 2 * rnd - 1
                others = ctxs[:i] + ctxs[i + 1:]
                body = DU.OTHERS_PREFIX + "".join(DU.OTHERS_ITEM.format(a=c[idx]["content"]) for c in others)
                msg = {"role": "user", "content": body + DU.SUFFIX_FRAMING[benchmark].format(q=question) + clause}
                assert msg == DU.others_for(benchmark, others, question, idx)
                ctx.append(msg)
            sent = [dict(m) for m in ctx]
            c0 = time.time()
            text, rec = DU.chat(registry=DU.REGISTRY, alias=alias, messages=sent,
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
                "extracted_answer": extract(text),
            })
            order.append({"seq": len(order), "round": rnd, "agent_index": i})
    finals = [ctx[-1]["content"] for ctx in ctxs]
    per_agent = [extract(f) for f in finals]
    return {
        "agent_contexts": ctxs, "calls": calls, "order": order,
        "per_agent_final_answer": per_agent,
        "majority_answer": DU.most_frequent(per_agent),
        "n_distinct_final_answers": len({x for x in per_agent if x is not None}),
        "native_stop_reason": "fixed-round-budget-exhausted",
        "accepted_delivery": "not-measured",
        "wall_s": round(time.time() - t0, 2),
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0.0 for c in calls), 8),
    }


def m2_run_one(question, *, task_id, benchmark, alias, agents, rounds, prune, max_tokens,
               timeout_s, retries, temperature, top_p):
    """mad_m2.run_one with the condition's number of agents and rounds."""
    agents = int(MODE.get("agents") or agents)
    rounds = int(MODE.get("rounds") or rounds)
    kind = M2.ANSWER_KIND[benchmark]
    extract = (lambda t: M2.extract_answer(t, kind))
    calls, order, per_round = [], [], []
    responses = [""] * agents
    t0 = time.time()
    seq = 0
    for rnd in range(rounds):
        if rnd == 0:
            prompts = [question + M2.COT_SUFFIX] * agents
            masks = [[] for _ in range(agents)]
        else:
            prompts, masks = [], []
            for i in range(agents):
                peers = [r for j, r in enumerate(responses) if j != i]
                peer_ppls = [per_round[-1]["perplexities"][j] for j in range(agents) if j != i]
                if prune == "objective":
                    kept, mask = M2.objective_prune(peers, peer_ppls)
                else:
                    kept, mask = peers, [True] * len(peers)
                masks.append(mask)
                body = "".join(f"\n\nOne agent's solution: {k}" for k in kept)
                prompts.append(M2.DEBATE_PREFIX + body + M2.DEBATE_SUFFIX + f"\n\nThe problem:\n{question}")
        round_answers, round_ppls = [], []
        for i in range(agents):
            messages = [{"role": "user", "content": prompts[i]}]
            text, rec = M2.chat(registry=M2.REGISTRY, alias=alias, messages=messages,
                                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                retries=retries, timeout_s=timeout_s, logprobs=True)
            responses[i] = text
            lps = [e["logprob"] for e in (rec.token_logprobs or [])]
            ppl = M2.perplexity(lps)
            ans = extract(text)
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
        flag, common = M2.if_reach_consensus(round_answers)
        per_round.append({"round": rnd, "answers": round_answers, "perplexities": round_ppls,
                          "masks": masks if rnd > 0 else [], "consensus": flag, "common_answer": common})
    final_answers = per_round[-1]["answers"]
    flag, common = M2.if_reach_consensus(final_answers)
    valid = [a for a in final_answers if a is not None]
    return {
        "run_id": f"madm2-{prune}-{alias}-{task_id}-r0", "example_id": task_id,
        "mechanism": f"madm2_{prune}", "model_alias": alias, "question": question,
        "benchmark": benchmark, "answer_kind": kind, "agents": agents, "rounds": rounds,
        "prune_strategy": prune, "temperature": temperature, "top_p": top_p,
        "per_agent_final_answer": final_answers,
        "per_agent_final_perplexity": per_round[-1]["perplexities"],
        "n_distinct_final_answers": len(set(valid)),
        "majority_answer": common if common is not None else None,
        "rule_satisfied": flag, "consensus_per_round": [r["consensus"] for r in per_round],
        "accepted_delivery": "not-measured",
        "native_stop_reason": f"fixed-round-budget-exhausted after {rounds} rounds",
        "rounds_detail": per_round, "calls": calls, "order": order,
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0 for c in calls), 6),
        "wall_s": round(time.time() - t0, 2),
    }


DU.run_one = du_run_one
M2.run_one = m2_run_one

# ------------------------------------------------------------------ item wrapper
import ext_unit_driver as EUD                                            # noqa: E402

_orig_run_unit = EUD.run_unit


def _run_unit(benchmark, protocol, unit, alias, *, arm="FULL", **kw):
    return _orig_run_unit(benchmark, protocol, unit, alias, arm=ARM_SPEC[arm][0], **kw)


EUD.run_unit = _run_unit

import ext_collect as XC                                                 # noqa: E402

_orig_run_item = XC._run_item


def override_digest():
    return hashlib.sha256((BM_HERE / "bm_override.py").read_bytes()).hexdigest()[:16]


def _run_item(item):
    MODE.clear()
    MODE.update(ARM_SPEC[item["arm"]][1])
    MODE.update(unit=item["unit"], backbone=item["backbone"], repeat=int(item["repeat"]),
                item_seed=item["seed"])
    rec = _orig_run_item(item)
    rec["bm"] = dict(arm_spec=ARM_SPEC[item["arm"]][0], mode=ARM_SPEC[item["arm"]][1],
                     override_sha16=override_digest(), run_id=RUN_ID, call_seeds=MODE.get("call_seeds"))
    if rec.get("status") == "ok":
        sent = MODE.get("sent") or []
        calls = rec["record"]["calls"]
        if len(sent) == len(calls):
            for c, s in zip(calls, sent):
                if int(c.get("round") or 0) >= 1:
                    c["messages_after_arm"] = s
    return rec


XC._run_item = _run_item
XC.OUT = OUT
XC.RAW = OUT / "raw" / LANE          # one shard per lane
XC.MANIFEST = OUT / f"manifest_{LANE}.jsonl"


def stamp():
    return {"run_id": RUN_ID, "out": str(OUT), "protos": PROTOS, "arms": ARMS, "backbones": BACKBONES,
            "n_units_this_lane": len(_ids), "repeats": D.REPEATS, "lane": LANE,
            "override_sha16": override_digest(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
