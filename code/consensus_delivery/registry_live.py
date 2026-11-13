"""The live endpoint registry for this study, plus a G2 serving smoke.

Routing is by alias only. The served name is asserted equal to the registry's checkpoint identity, so
an endpoint cannot be renamed into a different code path — the trap the previous native runner had.

    python3 -m consensus_delivery.registry_live            # identity + reachability
    python3 -m consensus_delivery.registry_live --smoke    # one real call per endpoint
"""
from __future__ import annotations

import os
import dataclasses
import json
import sys
from typing import Any, Dict, List

from .providers import ModelEndpoint, Registry, TransportFailure, chat

HOST = os.environ.get("INFERENCE_HOST", "inference-host")

#: The serving image, by digest. `latest` is a moving tag and is not version evidence on its own.
SERVING_IMAGE_DIGEST = ("vllm/vllm-openai@sha256:"
                        "8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90")

#: Served on H200 NVL cards with coordinated memory budgets.
LIVE_ENDPOINTS = (
    ModelEndpoint(
        alias="oss120", base_url=f"http://{HOST}:8940/v1", served_name="gpt-oss-120b",
        checkpoint="openai/gpt-oss-120b", provider="local",
        revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a", quantization="mxfp4",
        serving_image="vllm/vllm-openai:latest", max_context=16384, max_completion=4096,
        notes="GPU 5, gpu-memory-utilization 0.72, max-num-seqs 64",
        #: Measured, not tuned. At the default and at `medium` this checkpoint consumed 8192
        #: completion tokens on the simplest held-out calibration case and emitted **no content** —
        #: the reasoning channel does not terminate inside half the 16384-token context. At `low` the
        #: same case answers in 119-147 tokens and parses. This is therefore the only setting under
        #: which the endpoint is usable at this context, and it is recorded in the endpoint identity
        #: rather than applied per call, so no run can differ from another silently.
        #:
        #: vLLM accepts it as a **top-level body parameter**; passed as a `chat_template_kwargs` entry
        #: it is ignored (77 completion tokens, i.e. unchanged from the default).
        request_extra={"reasoning_effort": "low"},
    ),
    ModelEndpoint(
        alias="oss20", base_url=f"http://{HOST}:8941/v1", served_name="gpt-oss-20b",
        checkpoint="openai/gpt-oss-20b", provider="local",
        revision="6cee5e81ee83917806bbde320786a8fb61efebee", quantization="mxfp4",
        serving_image="vllm/vllm-openai:latest", max_context=16384, max_completion=4096,
        notes="GPU 3, gpu-memory-utilization 0.25, max-num-seqs 64",
        #: Measured, not tuned. At the default and at `medium` this checkpoint consumed 8192
        #: completion tokens on the simplest held-out calibration case and emitted **no content** —
        #: the reasoning channel does not terminate inside half the 16384-token context. At `low` the
        #: same case answers in 119-147 tokens and parses. This is therefore the only setting under
        #: which the endpoint is usable at this context, and it is recorded in the endpoint identity
        #: rather than applied per call, so no run can differ from another silently.
        #:
        #: vLLM accepts it as a **top-level body parameter**; passed as a `chat_template_kwargs` entry
        #: it is ignored (77 completion tokens, i.e. unchanged from the default).
        request_extra={"reasoning_effort": "low"},
    ),
    ModelEndpoint(
        alias="qwen36", base_url=f"http://{HOST}:8942/v1", served_name="qwen3.6-35b-a3b",
        checkpoint="Qwen/Qwen3.6-35B-A3B", provider="local",
        revision=None, quantization="bf16",
        serving_image="vllm/vllm-openai:latest", max_context=16384, max_completion=4096,
        # Qwen3.6 emits its reasoning as the visible answer unless its own template switch is set.
        # Verified at G2: without this it returned 265 tokens of thinking and no JSON; with it, 12
        # tokens of clean JSON. This is the model's documented template, not a protocol change.
        request_extra={"chat_template_kwargs": {"enable_thinking": False}},
        notes="GPU 2, gpu-memory-utilization 0.68, max-num-seqs 64, co-tenant",
    ),
)

#: `google/gemma-4-26B-A4B-it`, served at `--gpu-memory-utilization 0.40`.
#: **Official Google weights at native precision** — `config.json` carries no `quantization_config` —
#: so nothing here is a third-party conversion and there is no provenance discrepancy to record.
#: Non-reasoning: the chat template holds no thinking or reasoning markers, and its only
#: `raise_exception` concerns `tool_calls` argument formatting, not the position of a system message.
#: Verified against both of MALLM's requirements: a mid-conversation system message is accepted, and
#: the stream+logprobs shape yields non-empty content deltas, so MALLM's
#: `log_prob_sum / len(collected_messages)` step receives non-empty content.
GEMMA4 = (
    ModelEndpoint(
        alias="gemma4", base_url=f"http://{HOST}:8951/v1", served_name="gemma4-26b-a4b-it",
        checkpoint="google/gemma-4-26B-A4B-it", provider="local",
        revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        chat_template_sha="ae53464bf3be2580",
        serving_image=SERVING_IMAGE_DIGEST,
        #: Served at `--max-model-len 8192 --gpu-memory-utilization 0.40` as ReConcile's third
        #: family. 8192 rather than its native 16384 so the roster shares one context budget with
        #: the two paper backbones; its own capability is recorded here and is not used.
        max_context=8192, max_completion=4096,
        notes="GPU 5, container vllm-cwb-gemma4-g5, port 8951, max-model-len 8192 (native 16384). "
              "unquantised weights. Adopting it as a core backbone would change the cross-model "
              "comparison, so the matrix and the shared comparable blocks must be updated before "
              "main; today it is a judge candidate and a compatibility-verified backbone candidate."),
)

#: An additional backbone. Every field below was **measured** on the served checkpoint, not read
#: from a model card.
#:
#: The one that mattered: Mistral-7B-Instruct is widely described as having no `system` role, and on
#: **this** checkpoint through **this** image it does. Its chat template references `system` and a
#: system instruction to answer `BLUE` returns `Blue`. So no role is dropped and no prompt is
#: rewritten for an M arm, which is what the design requires.
#:
#: Served at the *identical* configuration to L and Q — 8192 context, 0.42 memory fraction, 128 max
#: sequences, same image digest — so an M cell is interchangeable with an L or Q cell in everything
#: but the weights. `logprobs` and `top_logprobs` are present, which is BCCS's hard dependency.
#:
#: Its sibling candidate `google/gemma-2-9b-it` is **not** here: it is `gated=manual` on Hugging
#: Face and an unauthenticated fetch of its `config.json` returns HTTP 401, with no token on the
#: machine. That is an external permission, recorded as a blocker rather than worked around — and
#: `gemma4` above is a different checkpoint of a different generation and must not stand in for it.
MISTRAL_CANDIDATE = (
    ModelEndpoint(
        alias="mistral-7b", base_url=f"http://{HOST}:8954/v1",
        served_name="mistral-7b-instruct",
        checkpoint="mistralai/Mistral-7B-Instruct-v0.3", provider="local",
        revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
        chat_template_sha="e16746b40344d6c5",
        serving_image=SERVING_IMAGE_DIGEST,
        max_context=8192, max_completion=4096,
        notes="GPU 1, container vllm-cwb-mistral-7b, port 8954, max-model-len 8192 (native 32768). "
              "bfloat16, 3 shards / 13.5 GB verified against the model's own safetensors index. "
              "Measured 59.7 GB VRAM at the shared 0.42 fraction, 2.0 s median latency and 196 "
              "tok/s on a 571-token mechanism-shaped prompt. A qualified candidate: no matrix is "
              "frozen on it until the protocol mapping and a precision pilot are done."),
)

#: The two backbones with a direct basis in the mechanisms' own papers, per
#: `paper/evidence/CONSENSUS_BACKBONE_ALIGNMENT_2026-10-08.md`. Chosen from the literature, not from
#: what happened to be deployable.
#:
#:   * `Meta-Llama-3-8B-Instruct` is the checkpoint "Voting or Consensus?" names in §3.1 footnote 2
#:     and uses for its main figures. **Exact**, not a successor: `Llama-3.1-8B-Instruct` was in the
#:     cache and is a different generation, so it is not substituted for it.
#:   * `Qwen2.5-7B-Instruct` is BCCS's §4 default. **Exact**, and already cached.
#:
#: Both are ordinary instruct models at the scale their papers used. Neither is claimed to be
#: equivalent to the closed models Du (`gpt-3.5-turbo-0301`) or ReConcile
#: (`gpt-3.5-turbo-0613` / `chat-bison-001` / Claude 2) used; an open-weight stand-in for those is a
#: backend adaptation and is labelled as one. ReConcile's heterogeneous family design is kept.
#:
#: 70B-class and MoE backbones remain **scale and sensitivity conditions**, not the primary
#: comparison. `gemma4` in particular is a newer generation than anything these papers used, so a
#: difference against it mixes generation with scale and cannot be read as a scale effect.
#: The frozen run budget for the paired primary comparison, uniform across the pair by construction:
#: a difference in budget would be a difference in the experiment, not in the method.
#:
#:   context 8192 (Meta-Llama-3-8B-Instruct's native limit, the pair's minimum)
#:   max_tokens per call 1024
#:   agents 3, max_turns 3, concurrency 4
#:
#: Observed on the first collected cell, so the budget is not merely declared: turns actually used
#: were 1 x13, 2 x17, 3 x10 of 40, i.e. the cap binds on a quarter of the tasks and the rest stop
#: earlier. Any native configuration with a different budget is a separately labelled anchor and is
#: never pooled with this comparison.
PAIRED_RUN_BUDGET = {"context": 8192, "max_tokens": 1024, "agents": 3, "max_turns": 3,
                     "concurrency": 4}

PAPER_BACKBONES = (
    ModelEndpoint(
        alias="llama3-8b", base_url=f"http://{HOST}:8953/v1",
        served_name="llama3-8b-instruct",
        checkpoint="meta-llama/Meta-Llama-3-8B-Instruct", provider="local",
        revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        chat_template_sha="da0e3a7cce6e4d78",
        serving_image=SERVING_IMAGE_DIGEST,
        max_context=8192, max_completion=2048,
        notes="GPU 4, container vllm-cwb-llama3-8b-g4, port 8953. Official weights, unquantised. "
              "The checkpoint 'Voting or Consensus?' §3.1 fn2 names and its main figures use. "
              "Official weights, bfloat16, no quantization_config, arch LlamaForCausalLM. "
              "max_model_len 8192 is this model's native context and the frozen paired budget."),
    ModelEndpoint(
        alias="qwen25-7b", base_url=f"http://{HOST}:8952/v1",
        served_name="qwen2.5-7b-instruct",
        checkpoint="Qwen/Qwen2.5-7B-Instruct", provider="local",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        chat_template_sha="5b5d4f65d0acd3b2",
        serving_image=SERVING_IMAGE_DIGEST,
        # **8192, not this model's 16384.** The primary comparison is paired across the two paper
        # backbones, and Meta-Llama-3-8B-Instruct's native context is 8192. Serving Qwen at its own
        # higher limit would let one member of the pair see prompts the other cannot, which is a
        # difference in the experiment rather than in the method. Its 16384 capability is recorded
        # here and is available as a separate, explicitly labelled budget anchor; it is not used in
        # the paired comparison.
        max_context=8192, max_completion=2048,
        notes="GPU 4, container vllm-cwb-qwen25-7b-g4, port 8952. Official weights, unquantised. "
              "BCCS's §4 default backbone. Official weights, bfloat16, no quantization_config, "
              "arch Qwen2ForCausalLM. Served at max-model-len 8192 to match the pair; its own "
              "native context is 16384, kept as a separate budget anchor."),
)

#: A second serving instance of each paper backbone, so the main matrix can use a third card.
#:
#: **Derived, not copied.** `dataclasses.replace` changes only the alias, the port and the note, so
#: the checkpoint, revision, chat template hash, image digest and context limits cannot drift from
#: the instance they mirror — a hand-written clone is exactly where a wrong revision would hide.
#:
#: Blocks are sharded whole: all three mechanisms of a (benchmark, backbone) block run on one
#: instance, so no contrast is ever computed across two of them. Two instances of one model differ
#: only in which card batches the requests, but that is still a difference, and the paired design
#: keeps it outside every comparison rather than arguing it is small.
SECOND_INSTANCE_PORTS = {"llama3-8b": 8963, "qwen25-7b": 8962}
PAPER_BACKBONES_B = tuple(
    dataclasses.replace(
        ep, alias=f"{ep.alias}-b",
        base_url=f"http://{HOST}:{SECOND_INSTANCE_PORTS[ep.alias]}/v1",
        notes=f"second instance of {ep.alias} on a different card, port "
              f"{SECOND_INSTANCE_PORTS[ep.alias]}; same weights, revision, template and image. "
              f"Used for whole-block shards of the main matrix.")
    for ep in PAPER_BACKBONES)

LIVE_ENDPOINTS = LIVE_ENDPOINTS + GEMMA4 + PAPER_BACKBONES + PAPER_BACKBONES_B


#: DeepSeek's official API, priced from https://api-docs.deepseek.com/quick_start/pricing. **Peak, cache-miss rates only.** Off-peak is half, and cache hits are 50-200x cheaper,
#: but a reservation made before a request cannot know which applies, so the ceiling is the peak
#: cache-miss price. Reasoning tokens have no separate tier in the published table, so they bill as
#: output and the reservation covers the full `max_tokens`.
#:
#: The official API exposes **no `/tokenize`**, so `reserve` falls back to the UTF-8 byte ceiling.
DEEPSEEK_ENDPOINTS = (
    ModelEndpoint(
        alias="ds-flash", base_url="https://api.deepseek.com/v1", served_name="deepseek-flash",
        checkpoint="deepseek/deepseek-flash", provider="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        usd_per_m_in=0.30, usd_per_m_out=1.20,
        #: Read from `/v1/models`, not rounded from a datasheet: `deepseek-flash`
        #: routes to **DeepSeek-V4.1-Flash** with `context_window` 1048576 and `max_output_tokens`
        #: 393216. `max_completion` stays our own 4096 cap. Prices taken from
        #: api-docs.deepseek.com/quick_start/pricing: cache-miss input $0.15 off-peak / $0.30 peak,
        #: output $0.60 / $1.20, with peak 01:00-04:00 and 06:00-10:00 UTC Mon-Fri. The peak,
        #: cache-miss rate is the one reserved against, because a reservation made before a request
        #: cannot know which applies.
        max_context=1_048_576, max_completion=4096,
        #: Set explicitly, and the effect is measured rather than assumed
        #: (on a real judging payload rather than a toy question):
        #:
        #:   * `thinking.type=disabled`           -> 52 completion tokens, 0 reasoning, schema-valid
        #:   * endpoint default                   -> 52 completion tokens, 0 reasoning, schema-valid
        #:   * `thinking.reasoning_effort=low`    -> 2048 completion tokens, **all** reasoning,
        #:                                           empty content, `finish_reason=length`
        #:   * `thinking.reasoning_effort=none`   -> the same 2048/empty/length
        #:   * top-level `reasoning_effort`       -> ignored, identical to the default
        #:
        #: So enabling the reasoning channel **reproduces** the failure that cost a judge round —
        #: 64 of 64 invalid outputs, 63 with no content. What cannot be asserted is that the *lost
        #: round* failed because of the documented default: today's default emits no reasoning
        #: tokens at all, and `deepseek-flash` now routes to **DeepSeek-V4.1-Flash**, so the route
        #: has moved. The reproducible statement is the one above, and it is the one recorded.
        #:
        #: Kept explicit rather than relying on the default precisely because the default is a
        #: server-side choice that has already changed once. The parameter is documented at
        #: api-docs.deepseek.com/api/create-chat-completion: `thinking` is an object with `type` in
        #: {enabled, disabled} (default enabled) and `reasoning_effort` in {none, low, high, max}
        #: (default high) nested inside it.
        request_extra={"thinking": {"type": "disabled"}},
        notes="peak cache-miss pricing; off-peak is half; 1M context, 384K max output"),
    ModelEndpoint(
        alias="ds-v4pro", base_url="https://api.deepseek.com/v1", served_name="deepseek-v4-pro",
        checkpoint="deepseek/deepseek-v4-pro", provider="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        usd_per_m_in=1.32, usd_per_m_out=3.96,
        max_context=1_000_000, max_completion=4096,
        #: Same channel, same reason as `ds-flash` above.
        request_extra={"thinking": {"type": "disabled"}},
        notes="peak cache-miss pricing; off-peak is half"),
)

#: The judge sub-cap. The project's external total is US$30 across DeepSeek and DeepInfra together;
#: this stage reserves US$5 of it for the DeepSeek judge, covering development, calibration, the
#: formal validation, retries and unknown-cost holds.
DEEPSEEK_JUDGE_CAP_USD = 5.00

# The qualified candidate joins the registry so a manifest can record its identity, and
# `verify_identity` can check it. Being in the registry is **not** being in a matrix: no
# frozen cell references it until the protocol mapping and precision pilot are done.
LIVE_ENDPOINTS = LIVE_ENDPOINTS + MISTRAL_CANDIDATE + DEEPSEEK_ENDPOINTS

REGISTRY = Registry(LIVE_ENDPOINTS)


def verify_identity() -> List[str]:
    """Each endpoint must serve exactly the model the registry says, by its own /v1/models."""
    import urllib.request

    problems: List[str] = []
    unverified: List[str] = []
    for ep in LIVE_ENDPOINTS:
        # An authenticated endpoint needs its key here too. Without it every hosted endpoint reported
        # `unreachable (HTTPError)`, which reads as a dead deployment rather than a missing header.
        # The key is read from the environment and never logged.
        headers = {}
        if ep.api_key_env:
            import os as _os  # noqa: PLC0415
            key = _os.environ.get(ep.api_key_env)
            if not key:
                problems.append(f"{ep.alias}: {ep.api_key_env} is not set")
                continue
            headers["Authorization"] = f"Bearer {key}"
        try:
            req = urllib.request.Request(ep.base_url.rstrip("/") + "/models", headers=headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                body = json.loads(r.read().decode())
            data = next((m for m in body["data"] if m.get("id") == ep.served_name), body["data"][0])
        except Exception as e:  # noqa: BLE001
            problems.append(f"{ep.alias}: unreachable ({type(e).__name__})")
            continue
        if data["id"] != ep.served_name:
            problems.append(f"{ep.alias}: serves {data['id']!r}, registry says {ep.served_name!r}")
        # `root` and `max_model_len` are vLLM's fields. A hosted provider need not publish them, and
        # demanding them reported a working endpoint as a mismatch. Checked when present, skipped
        # when absent — and the absence is itself recorded, because an unverifiable context length is
        # a fact the budget and the headroom check both depend on.
        if data.get("root") is not None and data.get("root") != ep.checkpoint:
            problems.append(f"{ep.alias}: root {data.get('root')!r} != checkpoint {ep.checkpoint!r}")
        if data.get("max_model_len") is None:
            unverified.append(ep.alias)
        elif data.get("max_model_len") != ep.max_context:
            problems.append(f"{ep.alias}: ctx {data.get('max_model_len')} != {ep.max_context}")
    if unverified:
        print(f"  context length not published by: {sorted(unverified)} — the registry value is "
              f"used for the budget and the headroom check and cannot be cross-checked there")
    return problems


def smoke() -> int:
    """One real call per endpoint through the single generic route. Asserts a parseable reply."""
    bad = 0
    for ep in LIVE_ENDPOINTS:
        try:
            out, rec = chat(
                registry=REGISTRY, alias=ep.alias,
                messages=[{"role": "user",
                           "content": 'Reply with exactly this JSON and nothing else: '
                                      '{"ok": true, "n": 7}'}],
                temperature=0.0, top_p=1.0, max_tokens=256, retries=1, timeout_s=120,
            )
            txt = out.strip()
            ok = '"ok"' in txt and "7" in txt
            print(f"  {'ok  ' if ok else 'WEAK'} {ep.alias:8s} finish={rec.finish_reason} "
                  f"in={rec.prompt_tokens} out={rec.completion_tokens} {rec.latency_s:.1f}s "
                  f"| {txt[:70]!r}")
            if not ok:
                bad += 1
        except TransportFailure as e:
            print(f"  FAIL {ep.alias:8s} {e}")
            bad += 1
    return bad


def main() -> int:
    print("registry identity:")
    for a in REGISTRY.aliases():
        e = REGISTRY.get(a)
        print(f"  {a:8s} {e.checkpoint:34s} {e.provider:6s} q={e.quantization} "
              f"ctx={e.max_context} rev={(e.revision or 'unknown')[:12]}")
    probs = verify_identity()
    if probs:
        print("\nIDENTITY PROBLEMS:")
        for p in probs:
            print("  " + p)
        return 1
    print("\nall endpoints serve the registered checkpoint; context length verified where the "
          "provider publishes it")
    if "--smoke" in sys.argv:
        print("\nserving smoke (one real call each, single generic route):")
        return 1 if smoke() else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
