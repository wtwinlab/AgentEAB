r"""Answer-only peers for Du-MAD and MAD-M2 on BIRD: an override of the extension collector, not an edit of it.

Imported before the collector's `main()` runs. It points the design (`ext_design`) and stop control
(`ext_stop_control`) at this collection's run id and output directory, restricts the queue to the 198
sampled BIRD questions, Du-MAD and MAD-M2, Llama-3-8B and Qwen2.5-7B agents and two repeats, registers
the two endpoint aliases, and adds the conditions below. Prompts, adapters, SQL task text, rounds,
agents, decoding, the 1,024-token output limit, stopping and return rules are those of the extension
collector.

Conditions (all collected together, so every comparison stays within one collection):
  FULL         with peers, unchanged
  ANSWER_ONLY  every peer block shown in an exchange round carries only that peer's SQL, rendered in the
               protocol's own answer form (Du-MAD "\boxed{q}" inside its usual block, MAD-M2
               "Answer: q"); no peer reasoning. The peer's SQL is the protocol's own extraction, else
               the lenient reading (last fenced block, else last line-initial SELECT/WITH), else empty.
               MAD-M2 still prunes peers on their real perplexities, so the same peers are shown as
               with full responses; Du-MAD keeps each agent's own history.
  NOPEERC      Du-MAD without peers: each peer reply is replaced at prompt construction by a
               placeholder and the no-peer transform removes every peer block with its preamble; every
               sent request is checked to contain no placeholder
  NOPEER       MAD-M2 without peers (its no-peer prompt holds no peer text)

Environment: AO_ARMS, AO_PROTOS, AO_BACKBONES, AO_REPEATS (default 2), AO_LANE_TAG, AO_UNIT_SHARD=i/n,
AO_PORT_L8 / AO_PORT_Q7 (local ports of the two servers), AO_OUT (output directory), AO_RUN_ID (run
identifier; it enters each item's seed and is kept as recorded), PAM_ART (task-definition directory of
the protocol study).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sys
import time

AO_HERE = pathlib.Path(__file__).resolve().parent
_REPO_GUESS = AO_HERE.parents[2]
A = pathlib.Path(os.environ.get("PAM_ART", _REPO_GUESS / "data/records/protocol_all_matching"))
ROOT = A.parents[2]
for _p in (ROOT / "code", A, ROOT / "experiments/02_controlled_sharing/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RUN_ID = os.environ.get("AO_RUN_ID", "share_eval_new_runs_2026-11-09")
OUT = pathlib.Path(os.environ.get("AO_OUT", ROOT / "experiments/16_answer_only_peers/data"))
LANE = os.environ.get("AO_LANE_TAG") or "lane"
UNITS = json.loads((AO_HERE / "units198.json").read_text())["units198"]

#: condition -> (transform of the extension collector, mode)
ARM_SPEC = {
    "FULL": ("FULL", {}),
    "NOPEER": ("NOPEER", {}),
    "NOPEERC": ("NOPEER", {"clean": True}),
    "ANSWER_ONLY": ("FULL", {"answer_only": True}),
}
PROTOS = [p for p in os.environ.get("AO_PROTOS", "du_mad,madm2_objective").split(",") if p]
ARMS = [a for a in os.environ.get("AO_ARMS", "FULL,NOPEERC,NOPEER,ANSWER_ONLY").split(",") if a]
BACKBONES = [b for b in os.environ.get("AO_BACKBONES", "llama3-8b,qwen2.5-7b").split(",") if b]
ALIAS = {"llama3-8b": "llama3-8b-ao", "qwen2.5-7b": "qwen25-7b-ao"}
PORT_ENV = {"llama3-8b-ao": "AO_PORT_L8", "qwen25-7b-ao": "AO_PORT_Q7"}
IMAGE = "vllm/vllm-openai:latest@d076d77d8dad (vLLM 0.30.0)"
PLACEHOLDER = "PEER_REPLY_REMOVED"

import ext_design as D                                                   # noqa: E402

D.RUN_ID = RUN_ID
D.OUT = OUT
D.BENCH_ORDER = ["sql_bird_v2"]
D.PROTOS = list(PROTOS)
D.ARMS = list(ARMS)
D.PANELS = {p: list(BACKBONES) for p in PROTOS}
D.ALIAS = dict(ALIAS)
D.REPEATS = int(os.environ.get("AO_REPEATS", "2"))
_ids = list(UNITS)
if os.environ.get("AO_UNIT_SHARD"):
    _i, _n = (int(x) for x in os.environ["AO_UNIT_SHARD"].split("/"))
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
        return "this collection runs Du-MAD and MAD-M2 only"
    if protocol == "du_mad" and arm == "NOPEER":
        return "Du-MAD's no-peer condition is NOPEERC"
    if protocol != "du_mad" and arm == "NOPEERC":
        return "NOPEERC is a Du-MAD condition"
    return None


D.frozen = _frozen
D.supported = _supported

import ext_stop_control as SC                                            # noqa: E402

SC.RUN_ID, SC.OUT = RUN_ID, OUT
SC.CONTROL, SC.GATE = OUT / "CONTROL.json", OUT / "GATE.json"
SC.STATUS = OUT / f"STATUS_{LANE}.json"
SC.MANIFEST = OUT / f"manifest_{LANE}.jsonl"


def _gate_state():
    ok, g = False, "absent"
    if SC.GATE.exists():
        g = json.loads(SC.GATE.read_text())
        ok = str(g.get("state")).lower() == "open"
    return ok, [{"condition": f"{SC.GATE} says state=open", "satisfied": ok, "detail": g}]


SC.gate_state = _gate_state

import endpoints_all_matching as EP                                      # noqa: E402

_prev_endpoints = EP.endpoints


def _port(alias):
    v = os.environ.get(PORT_ENV[alias])
    if not v:
        raise RuntimeError(f"set {PORT_ENV[alias]} to the local port of the {alias} server")
    return int(v)


def _endpoints(ports):
    from consensus_delivery.providers import ModelEndpoint
    new = (ModelEndpoint(alias="llama3-8b-ao", base_url=f"http://127.0.0.1:{_port('llama3-8b-ao')}/v1",
                         served_name="llama3-8b-instruct", checkpoint="meta-llama/Meta-Llama-3-8B-Instruct",
                         provider="local", serving_image=IMAGE, max_context=8192, max_completion=2048),
           ModelEndpoint(alias="qwen25-7b-ao", base_url=f"http://127.0.0.1:{_port('qwen25-7b-ao')}/v1",
                         served_name="qwen2.5-7b-instruct", checkpoint="Qwen/Qwen2.5-7B-Instruct",
                         provider="local", serving_image=IMAGE, max_context=8192, max_completion=2048))
    return tuple(_prev_endpoints(ports)) + new


EP.endpoints = _endpoints

# ------------------------------------------------------------------ per-item mode and sent requests
MODE: dict = {}
from consensus_delivery import providers as _PV                          # noqa: E402

_real_chat = _PV.chat


def _chat_inner(**kw):
    """Innermost transport wrapper: it sees each request after the condition transform."""
    msgs = [dict(m) for m in kw.get("messages") or []]
    if any(PLACEHOLDER in (m.get("content") or "") for m in msgs):
        raise AssertionError("no-peer placeholder survived the condition transform")
    MODE.setdefault("sent", []).append(msgs)
    return _real_chat(**kw)


_PV.chat = _chat_inner

# ------------------------------------------------------------------ peer answers
import sql_tasks as ST                                                   # noqa: E402
from consensus_delivery.adapters import du_mad as DU                     # noqa: E402
from consensus_delivery.adapters import mad_m2 as M2                     # noqa: E402

_FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.S)
_BARE = re.compile(r"(?im)^\s*((?:SELECT|WITH)\b[^;`]*)")


def lenient(raw):
    """Last fenced block, else the last line-initial SELECT/WITH query (the lenient extractor)."""
    t = str(raw or "")
    m = _FENCE.findall(t)
    if m:
        return ST.unwrap_sql(m[-1])
    b = _BARE.findall(t)
    return ST.unwrap_sql(b[-1]) if b else ""


def answer_only_text(protocol, reply, extracted):
    """(rendered peer answer, source of the SQL: protocol / lenient / empty)."""
    q = ST.unwrap_sql(extracted) if extracted not in (None, "", "None", "none") else ""
    tier = "protocol" if q else None
    if not q:
        q = lenient(reply)
        tier = "lenient" if q else "empty"
    q = (q or "").strip()
    return (("\\boxed{%s}" % q) if protocol == "du_mad" else ("Answer: %s" % q)), tier


def _tier(t):
    MODE.setdefault("answer_sources", {"protocol": 0, "lenient": 0, "empty": 0})
    MODE["answer_sources"][t] += 1


def _call_record(rnd, i, seq, sent, text, rec, c0, extracted):
    return {"round": rnd, "agent_index": i, "seq": seq, "messages_sent": sent, "raw_reply": text,
            "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
            "finish_reason": rec.finish_reason, "attempt": rec.attempt,
            "latency_s": round(time.time() - c0, 3), "usd": rec.usd, "request_sha": rec.request_sha,
            "served_model": rec.served_model, "fingerprint": rec.fingerprint, "extracted_answer": extracted}


# ------------------------------------------------------------------ Du-MAD
_du_orig = DU.run_one


def du_run_one(question, *, benchmark, alias, agents, rounds, max_tokens, timeout_s, retries,
               n_options=10):
    """du_mad.run_one with the peer blocks of later rounds replaced (ANSWER_ONLY) or emptied (NOPEERC).
    With neither condition it is du_mad.run_one unchanged."""
    answer_only, clean = MODE.get("answer_only"), MODE.get("clean")
    if not (answer_only or clean):
        return _du_orig(question, benchmark=benchmark, alias=alias, agents=agents, rounds=rounds,
                        max_tokens=max_tokens, timeout_s=timeout_s, retries=retries, n_options=n_options)
    clause = DU.ANSWER_CLAUSE[benchmark].format(answer="answer")
    extract = (lambda t: DU.extract_for(benchmark, t, n_options))
    ctxs = [[{"role": "user", "content": DU.opening_for(benchmark, question)}] for _ in range(agents)]
    calls, order = [], []
    t0 = time.time()
    for rnd in range(rounds):
        for i, ctx in enumerate(ctxs):
            if rnd != 0:
                idx = 2 * rnd - 1
                items = []
                for j, c in enumerate(ctxs):
                    if j == i:
                        continue
                    if clean:
                        items.append(DU.OTHERS_ITEM.format(a=PLACEHOLDER))
                    else:
                        txt, tier = answer_only_text("du_mad", c[idx]["content"], extract(c[idx]["content"]))
                        _tier(tier)
                        items.append(DU.OTHERS_ITEM.format(a=txt))
                body = DU.OTHERS_PREFIX + "".join(items)
                ctx.append({"role": "user", "content": body + DU.SUFFIX_FRAMING[benchmark].format(q=question) + clause})
            sent = [dict(m) for m in ctx]
            c0 = time.time()
            text, rec = DU.chat(registry=DU.REGISTRY, alias=alias, messages=sent,
                                temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                                retries=retries, timeout_s=timeout_s)
            ctx.append({"role": "assistant", "content": text})
            calls.append(_call_record(rnd, i, len(calls), sent, text, rec, c0, extract(text)))
            order.append({"seq": len(order), "round": rnd, "agent_index": i})
    per_agent = [extract(ctx[-1]["content"]) for ctx in ctxs]
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


# ------------------------------------------------------------------ MAD-M2
_m2_orig = M2.run_one


def m2_run_one(question, *, task_id, benchmark, alias, agents, rounds, prune, max_tokens,
               timeout_s, retries, temperature, top_p):
    """mad_m2.run_one with each kept peer reply replaced by its rendered SQL (ANSWER_ONLY).
    Otherwise mad_m2.run_one unchanged."""
    if not MODE.get("answer_only"):
        return _m2_orig(question, task_id=task_id, benchmark=benchmark, alias=alias, agents=agents,
                        rounds=rounds, prune=prune, max_tokens=max_tokens, timeout_s=timeout_s,
                        retries=retries, temperature=temperature, top_p=top_p)
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
                peer_ans = [per_round[-1]["answers"][j] for j in range(agents) if j != i]
                if prune == "objective":
                    _kept, mask = M2.objective_prune(peers, peer_ppls)
                else:
                    mask = [True] * len(peers)
                masks.append(mask)
                kept = []
                for r_, a_, m_ in zip(peers, peer_ans, mask):
                    if m_:
                        txt, tier = answer_only_text("madm2_objective", r_, a_)
                        _tier(tier)
                        kept.append(txt)
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

# ------------------------------------------------------------------ condition plumbing
import ext_unit_driver as EUD                                            # noqa: E402

_prev_run_unit = EUD.run_unit


def _run_unit(benchmark, protocol, unit, alias, *, arm="FULL", **kw):
    return _prev_run_unit(benchmark, protocol, unit, alias, arm=ARM_SPEC[arm][0], **kw)


EUD.run_unit = _run_unit

import ext_collect as XC                                                 # noqa: E402

XC.OUT = OUT
XC.RAW = OUT / "raw" / LANE           # one shard directory per lane
XC.MANIFEST = OUT / f"manifest_{LANE}.jsonl"
_prev_run_item = XC._run_item


def override_digest():
    return hashlib.sha256((AO_HERE / "ao_override.py").read_bytes()).hexdigest()[:16]


def _run_item(item):
    MODE.clear()
    MODE.update(ARM_SPEC[item["arm"]][1])
    rec = _prev_run_item(item)
    rec["ao"] = dict(override_sha16=override_digest(), run_id=RUN_ID, transform=ARM_SPEC[item["arm"]][0],
                     mode=ARM_SPEC[item["arm"]][1], answer_sources=MODE.get("answer_sources"))
    if rec.get("status") == "ok":
        sent = MODE.get("sent") or []
        calls = rec["record"]["calls"]
        if len(sent) == len(calls):
            for c, s in zip(calls, sent):
                if int(c.get("round") or 0) >= 1:
                    c["messages_after_arm"] = s
        else:
            rec["ao"]["sent_count_mismatch"] = [len(sent), len(calls)]
    return rec


XC._run_item = _run_item


def stamp():
    return {"run_id": RUN_ID, "out": str(OUT), "protos": PROTOS, "arms": ARMS, "backbones": BACKBONES,
            "repeats": D.REPEATS, "n_units_this_lane": len(_ids), "lane": LANE,
            "override_sha16": override_digest(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
