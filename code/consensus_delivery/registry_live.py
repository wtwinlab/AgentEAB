from __future__ import annotations

import os
import dataclasses
import json
import sys
from typing import Any, Dict, List

from .providers import ModelEndpoint, Registry, TransportFailure, chat

HOST = os.environ.get("INFERENCE_HOST", "inference-host")

SERVING_IMAGE_DIGEST = ("vllm/vllm-openai@sha256:"
                        "8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90")

LIVE_ENDPOINTS = (
    ModelEndpoint(
        alias="oss120", base_url=f"http://{HOST}:8940/v1", served_name="gpt-oss-120b",
        checkpoint="openai/gpt-oss-120b", provider="local",
        revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a", quantization="mxfp4",
        serving_image="vllm/vllm-openai:latest", max_context=16384, max_completion=4096,
        notes="GPU 5, gpu-memory-utilization 0.72, max-num-seqs 64",
        request_extra={"reasoning_effort": "low"},
    ),
    ModelEndpoint(
        alias="oss20", base_url=f"http://{HOST}:8941/v1", served_name="gpt-oss-20b",
        checkpoint="openai/gpt-oss-20b", provider="local",
        revision="6cee5e81ee83917806bbde320786a8fb61efebee", quantization="mxfp4",
        serving_image="vllm/vllm-openai:latest", max_context=16384, max_completion=4096,
        notes="GPU 3, gpu-memory-utilization 0.25, max-num-seqs 64",
        request_extra={"reasoning_effort": "low"},
    ),
    ModelEndpoint(
        alias="qwen36", base_url=f"http://{HOST}:8942/v1", served_name="qwen3.6-35b-a3b",
        checkpoint="Qwen/Qwen3.6-35B-A3B", provider="local",
        revision=None, quantization="bf16",
        serving_image="vllm/vllm-openai:latest", max_context=16384, max_completion=4096,
        request_extra={"chat_template_kwargs": {"enable_thinking": False}},
        notes="GPU 2, gpu-memory-utilization 0.68, max-num-seqs 64",
    ),
)

GEMMA4 = (
    ModelEndpoint(
        alias="gemma4", base_url=f"http://{HOST}:8951/v1", served_name="gemma4-26b-a4b-it",
        checkpoint="google/gemma-4-26B-A4B-it", provider="local",
        revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        chat_template_sha="ae53464bf3be2580",
        serving_image=SERVING_IMAGE_DIGEST,
        max_context=8192, max_completion=4096,
        notes="GPU 5, container vllm-cwb-gemma4-g5, port 8951, max-model-len 8192 (native 16384). "
              "unquantised weights. Adopting it as a core backbone would change the cross-model "
              "comparison, so the matrix and the shared comparable blocks must be updated before "
              "main; today it is a judge candidate and a compatibility-verified backbone candidate."),
)

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
        max_context=8192, max_completion=2048,
        notes="GPU 4, container vllm-cwb-qwen25-7b-g4, port 8952. Official weights, unquantised. "
              "BCCS's §4 default backbone. Official weights, bfloat16, no quantization_config, "
              "arch Qwen2ForCausalLM. Served at max-model-len 8192 to match the pair; its own "
              "native context is 16384, kept as a separate budget anchor."),
)

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


DEEPSEEK_ENDPOINTS = (
    ModelEndpoint(
        alias="ds-flash", base_url="https://api.deepseek.com/v1", served_name="deepseek-flash",
        checkpoint="deepseek/deepseek-flash", provider="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        usd_per_m_in=0.30, usd_per_m_out=1.20,
        max_context=1_048_576, max_completion=4096,
        request_extra={"thinking": {"type": "disabled"}},
        notes="peak cache-miss pricing; off-peak is half; 1M context, 384K max output"),
    ModelEndpoint(
        alias="ds-v4pro", base_url="https://api.deepseek.com/v1", served_name="deepseek-v4-pro",
        checkpoint="deepseek/deepseek-v4-pro", provider="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        usd_per_m_in=1.32, usd_per_m_out=3.96,
        max_context=1_000_000, max_completion=4096,
        request_extra={"thinking": {"type": "disabled"}},
        notes="peak cache-miss pricing; off-peak is half"),
)

DEEPSEEK_JUDGE_CAP_USD = 5.00

LIVE_ENDPOINTS = LIVE_ENDPOINTS + MISTRAL_CANDIDATE + DEEPSEEK_ENDPOINTS

REGISTRY = Registry(LIVE_ENDPOINTS)


def verify_identity() -> List[str]:
    import urllib.request

    problems: List[str] = []
    unverified: List[str] = []
    for ep in LIVE_ENDPOINTS:
        headers = {}
        if ep.api_key_env:
            import os as _os
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
        except Exception as e:
            problems.append(f"{ep.alias}: unreachable ({type(e).__name__})")
            continue
        if data["id"] != ep.served_name:
            problems.append(f"{ep.alias}: serves {data['id']!r}, registry says {ep.served_name!r}")
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
