#!/usr/bin/env python3
"""Serving endpoints of the controlled sharing collection, registered beside the model registry.
"""
from __future__ import annotations

import os
import json
import pathlib

from consensus_delivery.providers import ModelEndpoint, Registry
from consensus_delivery import registry_live

HOST = os.environ.get("INFERENCE_HOST", "inference-host")
ROUTES_FILE = pathlib.Path(__file__).resolve().parent / "ROUTES.json"
IMG = ("vllm/vllm-openai@sha256:"
       "8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90")

#: alias -> (port, served_name, checkpoint, note)
PLAN = {
    "llama3-8b-g1": (9211, "llama3-8b-instruct", "meta-llama/Meta-Llama-3-8B-Instruct",
                     "GPU1 dedicated"),
    "llama3-8b-g3": (9213, "llama3-8b-instruct", "meta-llama/Meta-Llama-3-8B-Instruct",
                     "GPU3 dedicated"),
    "qwen25-7b-g2": (9212, "qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct",
                     "GPU2 dedicated"),
    "qwen25-7b-g0": (9210, "qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct",
                     "GPU0 shared with another user's live but idle process, never signalled"),
    "llama3-8b-g5": (9215, "llama3-8b-instruct", "meta-llama/Meta-Llama-3-8B-Instruct",
                     "GPU5, added when that card freed"),
    "qwen25-7b-g5": (9216, "qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct",
                     "GPU5, added when that card freed"),
    "qwen25-7b-g1b": (9221, "qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct",
                      "GPU1, second server: one card serves both backbones"),
    "llama3-8b-g2b": (9222, "llama3-8b-instruct", "meta-llama/Meta-Llama-3-8B-Instruct",
                      "GPU2, second server: one card serves both backbones"),
    "qwen25-7b-g3b": (9223, "qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct",
                      "GPU3, second server: one card serves both backbones"),
    "gemma4-g5": (9224, "gemma4-26b-a4b-it", "google/gemma-4-26B-A4B-it",
                  "third roster family for MoA and ReConcile, on GPU5"),
}
BY_BACKBONE = {"llama3-8b": ["llama3-8b-g1", "llama3-8b-g3", "llama3-8b-g5",
                             "llama3-8b-g2b"],
               "qwen2.5-7b": ["qwen25-7b-g2", "qwen25-7b-g0", "qwen25-7b-g5",
                              "qwen25-7b-g1b", "qwen25-7b-g3b"],
               "gemma4": ["gemma4-g5"]}
DEFAULT_PORTS = {a: p for a, (p, _s, _c, _n) in PLAN.items()}


def resolve_routes():
    ports, moved = dict(DEFAULT_PORTS), {}
    if ROUTES_FILE.exists():
        for alias, port in json.loads(ROUTES_FILE.read_text(encoding="utf-8")).items():
            if alias not in ports:
                raise SystemExit(f"ROUTES.json names unknown alias {alias!r}")
            if int(port) != ports[alias]:
                moved[alias] = {"from_port": ports[alias], "to_port": int(port)}
                ports[alias] = int(port)
    ckpt_by_port = {p: c for p, (_s, c) in
                    {pp: (ss, cc) for pp, ss, cc in
                     ((p, s, c) for p, s, c, _n in PLAN.values())}.items()}
    for alias, mv in moved.items():
        want = PLAN[alias][2]
        got = ckpt_by_port.get(mv["to_port"])
        if got != want:
            raise SystemExit(f"refusing to route {alias} ({want}) onto port {mv['to_port']} "
                             f"serving {got!r}: different checkpoint")
    return ports, moved


def endpoints(ports):
    return tuple(
        ModelEndpoint(alias=a, base_url=f"http://{HOST}:{ports[a]}/v1", served_name=s,
                      checkpoint=c, provider="local", serving_image=IMG,
                      max_context=8192, max_completion=2048, notes=n)
        for a, (_p, s, c, n) in PLAN.items())


def install():
    ports, moved = resolve_routes()
    new = endpoints(ports)
    merged = Registry(tuple(registry_live.REGISTRY._by_alias.values()) + new)
    registry_live.REGISTRY = merged
    from consensus_delivery.adapters import (bccs, debate_or_vote, dmad, du_mad, ford, liang_mad,
                                             mad_m2, moa, reconcile)
    for m in (bccs, debate_or_vote, dmad, du_mad, ford, liang_mad, mad_m2, moa, reconcile):
        if hasattr(m, "REGISTRY"):
            m.REGISTRY = merged
    return {"added": [e.alias for e in new], "total_aliases": len(merged._by_alias),
            "routes": {a: f"http://{HOST}:{p}/v1" for a, p in ports.items()},
            "re_pointed": moved,
            "re_point_note": ("serving location only; checkpoint verified identical. No protocol "
                              "parameter, backbone, roster or agent count is affected."
                              if moved else None)}


if __name__ == "__main__":
    print(json.dumps(install(), indent=1))
