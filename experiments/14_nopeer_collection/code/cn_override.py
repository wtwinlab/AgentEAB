r"""The no-peer collection for Du-MAD and BCCS used by the paper. Layered on the backbone override.

Imports `sb_override` (backbone design: aliases, ports, serving extras, GLM think strip, BCCS
execid), then points the run id / output directory here and adds one condition, NOPEERC: the
NOPEER transform applied to a prompt whose peer replies are replaced at construction time by a
fence-free placeholder, so the transform removes every block completely.
  * Du-MAD : du_mad.run_one verbatim, peer contents -> PLACEHOLDER.
  * BCCS   : bccs.construct_message wrapped; responses -> PLACEHOLDER; the 'One conflicting agent
             solution:' label is rewritten to 'One agent solution:' on the placeholder block only,
             so the transform removes it with the preamble.
Peer *selection* (BCCS clusters, beliefs, which agents are shown) is unchanged; only the rendered
text is removed. Every sent request is asserted placeholder-free and logged (`messages_after_arm`).
The item seed is computed with arm 'NOPEER' and the ORIGINAL run id, i.e. the original no-peer seed.

Env (besides sb_override's): CN_OUT (output directory), CN_RUN_ID (run identifier), CN_ORIG_RUN_ID (run id whose
NOPEER seeds are reproduced), CN_PORT_L8 / CN_PORT_Q7 for the 7-8B aliases.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time

CN_HERE = pathlib.Path(__file__).resolve().parent
ROOT = CN_HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/13_backbone_scale_cross_family/code"))
import sb_override as SB                                                 # noqa: E402
import ext_design as D                                                   # noqa: E402
import ext_stop_control as SC                                            # noqa: E402

RUN_ID = os.environ.get("CN_RUN_ID", "clean_nopeer_2026-11-06")
ORIG_RUN_ID = os.environ.get("CN_ORIG_RUN_ID", "strong_backbone_bird_2026-11-06")
OUT = pathlib.Path(os.environ.get("CN_OUT", ROOT / "experiments/14_nopeer_collection/data/sql"))
PLACEHOLDER = "PEER_REPLY_REMOVED"
LANE = os.environ.get("SB_LANE_TAG") or "lane"

D.RUN_ID = RUN_ID
D.OUT = OUT
D.ARMS = ["NOPEERC"]
SC.RUN_ID, SC.OUT = RUN_ID, OUT
SC.CONTROL, SC.GATE = OUT / "CONTROL.json", OUT / "GATE.json"
SC.STATUS = OUT / f"STATUS_{LANE}.json"
SC.MANIFEST = OUT / f"manifest_{LANE}.jsonl"
SB.OUT = OUT
SB.RUN_ID = RUN_ID

# 7-8B aliases (same served names / checkpoints as the original 7-8B servers, vLLM 0.30.0 image)
SB.ALIAS.update({"llama3-8b": "llama3-8b-rx", "qwen2.5-7b": "qwen25-7b-rx"})
D.ALIAS = dict(SB.ALIAS)
D.PANELS = {p: list(SB.BACKBONES) for p in SB.PROTOS}
_P78 = {"llama3-8b-rx": int(os.environ.get("CN_PORT_L8", "9311")),
        "qwen25-7b-rx": int(os.environ.get("CN_PORT_Q7", "9312"))}
import endpoints_all_matching as EP                                      # noqa: E402

_prev_endpoints = EP.endpoints


def _endpoints(ports):
    from consensus_delivery.providers import ModelEndpoint
    img = "vllm/vllm-openai:latest@d076d77d8dad (vLLM 0.30.0)"
    new = (ModelEndpoint(alias="llama3-8b-rx", base_url=f"http://127.0.0.1:{_P78['llama3-8b-rx']}/v1",
                         served_name="llama3-8b-instruct", checkpoint="meta-llama/Meta-Llama-3-8B-Instruct",
                         provider="local", serving_image=img, max_context=8192, max_completion=2048),
           ModelEndpoint(alias="qwen25-7b-rx", base_url=f"http://127.0.0.1:{_P78['qwen25-7b-rx']}/v1",
                         served_name="qwen2.5-7b-instruct", checkpoint="Qwen/Qwen2.5-7B-Instruct",
                         provider="local", serving_image=img, max_context=8192, max_completion=2048))
    return tuple(_prev_endpoints(ports)) + new


EP.endpoints = _endpoints

MODE: dict = {}
from consensus_delivery import providers as _PV                          # noqa: E402

_below = _PV.chat           # real transport, or sb's GLM think-strip wrapper around it


def _chat_inner(**kw):
    msgs = [dict(m) for m in kw.get("messages") or []]
    if any(PLACEHOLDER in (m.get("content") or "") for m in msgs):
        raise AssertionError("no-peer placeholder survived the arm transform")
    MODE.setdefault("sent", []).append(msgs)
    return _below(**kw)


_PV.chat = _chat_inner

# ------------------------------------------------------------------ Du-MAD
from consensus_delivery.adapters import du_mad as DU                     # noqa: E402

_du_orig = DU.run_one


def du_run_one(question, *, benchmark, alias, agents, rounds, max_tokens, timeout_s, retries,
               n_options=10):
    if not MODE.get("clean"):
        return _du_orig(question, benchmark=benchmark, alias=alias, agents=agents, rounds=rounds,
                        max_tokens=max_tokens, timeout_s=timeout_s, retries=retries, n_options=n_options)
    ctxs = [[{"role": "user", "content": DU.opening_for(benchmark, question)}] for _ in range(agents)]
    calls, order = [], []
    t0 = time.time()
    for rnd in range(rounds):
        for i, ctx in enumerate(ctxs):
            if rnd != 0:
                idx = 2 * rnd - 1
                others = [[{"content": PLACEHOLDER}] * (idx + 1) for j in range(agents) if j != i]
                ctx.append(DU.others_for(benchmark, others, question, idx))
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
                "extracted_answer": DU.extract_for(benchmark, text, n_options),
            })
            order.append({"seq": len(order), "round": rnd, "agent_index": i})
    finals = [ctx[-1]["content"] for ctx in ctxs]
    per_agent = [DU.extract_for(benchmark, f, n_options) for f in finals]
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


DU.run_one = du_run_one

# ------------------------------------------------------------------ BCCS
from consensus_delivery.adapters import bccs as BC                       # noqa: E402

_bc_construct = BC.construct_message
_CONF = "One conflicting agent solution: ```%s```" % PLACEHOLDER


def construct_message(current_agent, responses, question, probs, clusters, conflict, uncertainties,
                      T, top_k):
    if not MODE.get("clean"):
        return _bc_construct(current_agent, responses, question, probs, clusters, conflict,
                             uncertainties, T, top_k)
    msg, shown = _bc_construct(current_agent, [PLACEHOLDER] * len(responses), question, probs,
                               clusters, conflict, uncertainties, T, top_k)
    n_conf = msg.count(_CONF)
    msg = msg.replace(_CONF, "One agent solution: ```%s```" % PLACEHOLDER)
    MODE.setdefault("bccs_conflicting_relabelled", 0)
    MODE["bccs_conflicting_relabelled"] += n_conf
    return msg, shown


BC.construct_message = construct_message

# ------------------------------------------------------------------ arm plumbing
import ext_unit_driver as EUD                                            # noqa: E402

_prev_run_unit = EUD.run_unit


def _run_unit(benchmark, protocol, unit, alias, *, arm="FULL", **kw):
    if arm == "NOPEERC":
        if protocol not in ("du_mad", "bccs"):
            raise AssertionError("NOPEERC is defined for Du-MAD and BCCS only")
        arm = "NOPEER"
    return _prev_run_unit(benchmark, protocol, unit, alias, arm=arm, **kw)


EUD.run_unit = _run_unit

import ext_collect as XC                                                 # noqa: E402

XC.OUT = OUT
XC.RAW = OUT / "raw" / LANE
XC.MANIFEST = OUT / f"manifest_{LANE}.jsonl"
_prev_run_item = XC._run_item


def orig_seed(item):
    raw = "|".join([ORIG_RUN_ID] + [str(item[k]) if k != "arm" else "NOPEER"
                                    for k in ("benchmark", "unit", "protocol", "arm", "backbone", "repeat")])
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def override_digest():
    return hashlib.sha256((CN_HERE / "cn_override.py").read_bytes()).hexdigest()[:16]


def _run_item(item):
    MODE.clear()
    MODE["clean"] = item["arm"] == "NOPEERC"
    item = dict(item, seed=orig_seed(item))
    rec = _prev_run_item(item)
    rec["cn"] = dict(override_sha16=override_digest(), orig_run_id=ORIG_RUN_ID,
                     bccs_conflicting_relabelled=MODE.get("bccs_conflicting_relabelled", 0))
    if rec.get("status") == "ok":
        sent = MODE.get("sent") or []
        calls = rec["record"]["calls"]
        if len(sent) == len(calls):
            for c, s in zip(calls, sent):
                if int(c.get("round") or 0) >= 1:
                    c["messages_after_arm"] = s
        else:
            rec["cn"]["sent_count_mismatch"] = [len(sent), len(calls)]
    return rec


XC._run_item = _run_item


def stamp():
    return dict(SB.stamp(), run_id=RUN_ID, out=str(OUT), cn_override_sha16=override_digest(),
                orig_run_id=ORIG_RUN_ID)
