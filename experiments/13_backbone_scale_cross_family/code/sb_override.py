from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time

SB_HERE = pathlib.Path(__file__).resolve().parent
_REPO_GUESS = SB_HERE.parents[2]
A = pathlib.Path(os.environ.get(
    "PAM_ART", _REPO_GUESS / "data/records/protocol_all_matching"))
ROOT = A.parents[2]
for _p in (ROOT / "code", A, ROOT / "experiments/02_controlled_sharing/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RUN_ID = "strong_backbone_bird_2026-11-06"
OUT = pathlib.Path(os.environ.get("SB_OUT", ROOT / "data/records/backbone_scale_cross_family"))
ALL_PROTOS = ["du_mad", "madm2_objective", "bccs"]
PROTOS = [p for p in os.environ.get("SB_PROTOS", ",".join(ALL_PROTOS)).split(",") if p]
ARMS = ["FULL", "NOPEER"]
ALL_BACKBONES = ["qwen25-72b-awq", "gpt-oss-120b", "qwen25-14b-awq", "qwen25-32b-awq",
                 "gpt-oss-20b", "qwen38-27b", "glm47-flash", "mistral-small-31", "mistral-small-32", "nemotron35-bolt", "glm47-flash-think", "qwen2.5-7b-execid",
                 "gemma4-26b-a4b", "minimax-m25-awq", "agentar-sql-32b",
                 "qwen36-35b-a3b", "qwen3-32b", "llama33-70b"]
BACKBONES = [b for b in os.environ.get("SB_BACKBONES", ",".join(ALL_BACKBONES)).split(",") if b]
ALIAS = {"qwen3-32b": "qwen3-32b-sb", "llama33-70b": "llama33-70b-sb",
         "gpt-oss-120b": "gptoss-120b-sb", "qwen36-35b-a3b": "qwen36-35b-sb",
         "qwen25-72b-awq": "qwen25-72b-awq-sb", "qwen25-32b-awq": "qwen25-32b-awq-sb",
         "qwen25-14b-awq": "qwen25-14b-awq-sb", "gpt-oss-20b": "gptoss-20b-sb",
         "qwen38-27b": "qwen38-27b-sb", "glm47-flash": "glm47-flash-sb",
         "mistral-small-31": "mistral-small-31-sb", "nemotron35-bolt": "nemotron35-bolt-sb",
         "mistral-small-32": "mistral-small-32-sb",
         "glm47-flash-think": "glm47-flash-think-sb", "qwen2.5-7b-execid": "qwen25-7b-execid-sb",
         "gemma4-26b-a4b": "gemma4-26b-sb", "minimax-m25-awq": "minimax-m25-sb", "agentar-sql-32b": "agentar-sql-32b-sb"}
MAX_TOKENS = int(os.environ.get("SB_MAX_TOKENS", "1024"))
PORTS = {"qwen3-32b-sb": int(os.environ.get("SB_PORT_Q3", "8231")),
         "llama33-70b-sb": int(os.environ.get("SB_PORT_L70", "8232")),
         "gptoss-120b-sb": int(os.environ.get("SB_PORT_GO", "8233")),
         "qwen36-35b-sb": int(os.environ.get("SB_PORT_Q36", "8234")),
         "qwen25-72b-awq-sb": 8235, "qwen25-32b-awq-sb": 8236, "qwen25-14b-awq-sb": 8237,
         "qwen38-27b-sb": 8238, "nemotron35-bolt-sb": 8239, "glm47-flash-sb": 8240,
         "mistral-small-31-sb": 8241, "gptoss-20b-sb": 8242, "mistral-small-32-sb": 8243,
         "glm47-flash-think-sb": 8244, "qwen25-7b-execid-sb": 8245,
         "gemma4-26b-sb": 8250, "agentar-sql-32b-sb": 8251, "minimax-m25-sb": 8252}
for _kv in [x for x in os.environ.get("SB_PORT_OVERRIDE", "").split(",") if "=" in x]:
    PORTS[_kv.split("=")[0]] = int(_kv.split("=")[1])
SERVED = {"qwen3-32b-sb": ("qwen3-32b", "Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137",
                           None, "GPU0+1 data-parallel; enable_thinking=false server default", None),
          "llama33-70b-sb": ("llama3.3-70b-awq",
                             "casperhansen/llama-3.3-70b-instruct-awq@64d255621f40b42adaf6d1f32a47e1d4534c0f14",
                             "awq", "GPU2+3 data-parallel", None),
          "gptoss-120b-sb": ("gpt-oss-120b", "openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
                             "mxfp4", "GPU0+1 data-parallel; reasoning_effort=low (always reasons)",
                             {"reasoning_effort": "low"}),
          "qwen36-35b-sb": ("qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0",
                            None, "GPU2+3 data-parallel; enable_thinking=false server default", None),
          "qwen25-72b-awq-sb": ("qwen2.5-72b-awq", "Qwen/Qwen2.5-72B-Instruct-AWQ@698703eae6604af048a3d2f509995dc302088217",
                                "awq", "data-parallel over 2 cards", None),
          "qwen25-32b-awq-sb": ("qwen2.5-32b-awq", "Qwen/Qwen2.5-32B-Instruct-AWQ@5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c",
                                "awq", "data-parallel over 2 cards", None),
          "qwen25-14b-awq-sb": ("qwen2.5-14b-awq", "Qwen/Qwen2.5-14B-Instruct-AWQ@539535859b135b0244c91f3e59816150c8056698",
                                "awq", "data-parallel over 2 cards", None),
          "gptoss-20b-sb": ("gpt-oss-20b", "openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee",
                            "mxfp4", "one card; reasoning_effort=low (always reasons)", {"reasoning_effort": "low"}),
          "qwen38-27b-sb": ("qwen3.8-27b", "Qwen/Qwen3.8-27B@8c0b3d30f3ae96f5c7f774de704bf0ada83e422a",
                            None, "one card; enable_thinking=false server default", None),
          "glm47-flash-sb": ("glm-4.7-flash", "zai-org/GLM-4.7-Flash@7dd20894a642a0aa287e9827cb1a1f7f91386b67",
                             None, "one card; enable_thinking=false server default", None),
          "mistral-small-31-sb": ("mistral-small-3.1-24b",
                                  "mistralai/Mistral-Small-3.1-24B-Instruct-2503@68faf511d618ef198fef186659617cfd2eb8e33a",
                                  None, "one card; non-reasoning", None),
          "mistral-small-32-sb": ("mistral-small-3.2-24b",
                                  "mistralai/Mistral-Small-3.2-24B-Instruct-2506@" + os.environ.get("SB_MIS32_REV", "see serve.log"),
                                  None, "one card; non-reasoning", None),
          "glm47-flash-think-sb": ("glm-4.7-flash-think",
                                   "zai-org/GLM-4.7-Flash@7dd20894a642a0aa287e9827cb1a1f7f91386b67",
                                   None, "one card; thinking ON (model default), glm47 reasoning parser; Du-MAD only", None),
          "qwen25-7b-execid-sb": ("qwen2.5-7b-instruct",
                                  "Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28",
                                  None, "one card; bccs_execid online replay-validity run (variant A9)", None),
          "gemma4-26b-sb": ("gemma-4-26b-a4b-it", "google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
                            None, "one card; thinking default off (vLLM 0.32 Gemma4 template)", None),
          "agentar-sql-32b-sb": ("agentar-sql-32b", "antgroup/Agentar-Scale-SQL-Generation-32B", None,
                                 "one card; SQL specialist", None),
          "minimax-m25-sb": ("minimax-m2.5-awq", "QuantTrio/MiniMax-M2.5-AWQ@bbe738792c897da97ae0dfe5a47bd28a3a1ea10e",
                             "awq", "TP=2 over two cards; reasoning model", None),
          "nemotron35-bolt-sb": ("nemotron-3.5-bolt",
                                 "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16@e09a4cedb1787db1760375ee5539f222cae0c672",
                                 None, "TP=2 over two cards; enable_thinking=false server default", None)}
IMAGE = "vllm/vllm-openai:v0.32.0"

import ext_design as D

D.RUN_ID = RUN_ID
D.OUT = OUT
D.BENCH_ORDER = ["sql_bird_v2"]
D.PROTOS = list(PROTOS)
D.ARMS = list(ARMS)
D.PANELS = {p: list(BACKBONES) for p in PROTOS}
D.ALIAS = dict(ALIAS)
D.PROD_MAX_TOKENS = MAX_TOKENS

if os.environ.get("SB_UNIT_SHARD"):
    _i, _n = (int(x) for x in os.environ["SB_UNIT_SHARD"].split("/"))
    _frozen = D.frozen

    def _frozen_shard(benchmark):
        f = dict(_frozen(benchmark))
        f["unit_ids"] = [u for k, u in enumerate(f["unit_ids"]) if k % _n == _i]
        return f

    D.frozen = _frozen_shard

import ext_stop_control as SC

SC.RUN_ID = RUN_ID
SC.OUT = OUT
SC.CONTROL = OUT / "CONTROL.json"
SC.STATUS = OUT / "STATUS.json"
SC.GATE = OUT / "GATE.json"
SC.MANIFEST = OUT / "manifest.jsonl"


def _gate_state():
    ok = False
    g = "absent"
    if SC.GATE.exists():
        try:
            g = json.loads(SC.GATE.read_text(encoding="utf-8"))
            ok = str(g.get("state")).lower() == "open"
        except Exception as exc:
            g = f"unreadable: {type(exc).__name__}"
    return ok, [{"condition": f"{SC.GATE} says state=open", "satisfied": ok, "detail": g}]


SC.gate_state = _gate_state

_LANE = os.environ.get("SB_LANE_TAG") or "_".join(BACKBONES)


def _write_atomic(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


SC._write_atomic = _write_atomic
SC.STATUS = OUT / f"STATUS_{_LANE}.json"

import endpoints_all_matching as EP

_orig_endpoints = EP.endpoints


def _endpoints(ports):
    from consensus_delivery.providers import ModelEndpoint
    new = tuple(
        ModelEndpoint(alias=a, base_url=f"http://127.0.0.1:{PORTS[a]}/v1", served_name=s,
                      checkpoint=c, provider="local", quantization=q, serving_image=IMAGE,
                      max_context=16384, max_completion=4096, notes=n, request_extra=x)
        for a, (s, c, q, n, x) in SERVED.items())
    return tuple(_orig_endpoints(ports)) + new


EP.endpoints = _endpoints

if os.environ.get("SB_STRIP_THINK"):
    from consensus_delivery import providers as _PV
    _real_chat = _PV.chat

    def _chat_strip(*a, **kw):
        text, rec = _real_chat(*a, **kw)
        if str(kw.get("alias", "")).startswith("glm47-flash") and "</think>" in (text or ""):
            head, tail = text.rsplit("</think>", 1)
            d = OUT / "logs" / "glm_think_strip"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{os.getpid()}.jsonl", "a") as fh:
                fh.write(json.dumps(dict(request_sha=getattr(rec, "request_sha", None),
                                         stripped_prefix=head + "</think>",
                                         n_logprob_tokens=len(getattr(rec, "token_logprobs", None) or []),
                                         utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")
            text = tail.lstrip()
        return text, rec

    _PV.chat = _chat_strip


def override_digest() -> str:
    return hashlib.sha256((SB_HERE / "sb_override.py").read_bytes()).hexdigest()[:16]


def stamp() -> dict:
    return {"run_id": RUN_ID, "out": str(OUT), "protos": PROTOS, "arms": ARMS,
            "backbones": BACKBONES, "alias": ALIAS, "ports": PORTS, "max_tokens": MAX_TOKENS,
            "override_sha16": override_digest(), "frozen_dir": str(D.FROZEN_DIR),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

if os.environ.get("SB_BCCS_EXECID"):
    sys.path.insert(0, str(SB_HERE))
    import sb_bccs_execid as _BX
    _BX.install()

if os.environ.get("SB_HARMONY_SLICE"):
    sys.path.insert(0, str(SB_HERE))
    import sb_harmony as _HS
    _HS.install()

if os.environ.get("SB_THINK_SLICE"):
    sys.path.insert(0, str(SB_HERE))
    import sb_harmony as _HS2
    _HS2.install_think()
