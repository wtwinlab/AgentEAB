r"""Recollect the conditions whose payload the revised price/coordinate contracts actually change.

    PYTHONPATH=code python3 code/collect_fixed_arms.py --dataset wdc_products --arm E5f --split dev
    PYTHONPATH=code python3 code/collect_fixed_arms.py --dataset geographic_settlements --arm E5f --split test
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
sys.path.insert(0, os.path.join(STAGE, "code"))

import collect_matrix as CM              # noqa: E402
import evidence_fixed as F               # noqa: E402
import evidence_templates as V1          # noqa: E402
import price_geo_contract as C           # noqa: E402


def comma_only_price_line(pa: Any, pb: Any) -> str:
    """v1's price line with only the separator error fixed: unmarked values are still subtracted."""
    va, sa, _ = C.parse_price(pa)
    vb, sb, _ = C.parse_price(pb)
    if va is not None and vb is not None:
        rel = abs(va - vb) / max(va, vb) if max(va, vb) else 0.0
        return (f"  - price: A {va:.2f}, B {vb:.2f}, absolute difference {abs(va - vb):.2f}, "
                f"relative {rel:.2f}")
    return (f"  - price: A {pa or 'absent'}, B {pb or 'absent'} "
            f"(difference not computable)")


def block_for(dataset: str, arm: str, A: Dict[str, Any], B: Dict[str, Any]) -> str:
    if dataset == "geographic_settlements":
        return F.evidence_settlements_fixed(A, B)
    if arm == "E5f":
        return F.evidence_products_fixed(A, B)
    # E5c: the fixed block with the currency rule reverted to v1's behaviour
    full = F.evidence_products_fixed(A, B).splitlines()
    out = [ln if not ln.strip().startswith("- price:")
           else comma_only_price_line(A.get("price"), B.get("price")) for ln in full]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["wdc_products", "geographic_settlements"])
    ap.add_argument("--arm", required=True, choices=["E5f", "E5c"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    CM.WORKERS = a.workers

    dom = CM.Domain(a.dataset)
    data = os.path.join(STAGE, "data", a.dataset)
    out_path = os.path.join(data, f"fixed_{a.split}.jsonl")
    have = CM.done_units(out_path)
    blob = json.load(open(os.path.join(STAGE, "data/protocol", f"{a.dataset}.json"),
                          encoding="utf-8"))
    ent, sp = blob["entity_of"], blob["split"]
    cands = [(str(x), str(y)) for x, y in blob["candidates"]
             if sp.get(ent.get(str(x))) == a.split and sp.get(ent.get(str(y))) == a.split]

    import metrics as M                                               # noqa: PLC0415
    pool = M.load_jsonl(os.path.join(data, f"pool_{a.split}.jsonl"))
    cell: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in pool:
        cell[(r["a"], r["b"], r["model"])].append(r)
    pv: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for k, rs in cell.items():
        v = [r["decision"] for r in rs if r["decision"] is not None]
        d = None
        if v:
            c = collections.Counter(v)
            top = max(c.values())
            w = [kk for kk, n in c.items() if n == top]
            d = w[0] if len(w) == 1 else None
        pv[k] = {"model": k[2], "decision": d,
                 "raw_reply": next((r["raw_reply"] for r in rs if r["decision"] == d), None)}

    v1_fn = (V1.evidence_settlements if a.dataset == "geographic_settlements"
             else V1.evidence_products)
    jobs: List[Dict[str, Any]] = []
    skipped = collections.Counter()
    for (x, y) in cands:
        A, B = dom.fields(x), dom.fields(y)
        new = block_for(a.dataset, a.arm, A, B)
        if new == v1_fn(A, B):
            skipped["block unchanged, stored E5 row reused"] += 1
            continue
        for rc in CM.RECIPIENTS:
            others = [m for m in CM.POOL if m != rc]
            peers = [pv[(x, y, m)] for m in sorted(others) if (x, y, m) in pv]
            if len(peers) < 2:
                skipped["fewer than two peers"] += 1
                continue
            for rep in range(CM.REPEATS):
                u = CM.uid("fixed", a.dataset, a.split, x, y, rc, a.arm, rep)
                if u in have:
                    skipped["already collected"] += 1
                    continue
                jobs.append({"unit_id": u, "a": x, "b": y, "recipient": rc, "arm": a.arm,
                             "repeat": rep, "peers": peers, "block": new})
    print(f"  {a.arm}/{a.dataset}/{a.split}: {len(jobs)} calls over "
          f"{len(cands) - skipped['block unchanged, stored E5 row reused']} changed pairs; "
          + ", ".join(f"{v} {k}" for k, v in skipped.items()), flush=True)
    if a.limit:
        jobs = jobs[:a.limit]
    if not jobs:
        return 0

    rng = random.Random(CM.DRAW_SEED)
    blocks = [jobs[i:i + CM.BLOCK] for i in range(0, len(jobs), CM.BLOCK)]
    for b in blocks:
        rng.shuffle(b)
    jobs = [j for b in blocks for j in b]

    def do(j: Dict[str, Any]) -> None:
        pr = (dom.base_prompt(j["a"], j["b"])
              + CM.FRAME.format(block=CM.peer_decisions_block(j["peers"]))
              + "\n\n" + j["block"])
        mt = CM.MAX_TOKENS["E5"]
        g = CM.post(j["recipient"], pr, mt)
        CM.append(out_path, {
            "unit_id": j["unit_id"], "dataset": a.dataset, "split": a.split,
            "a": j["a"], "b": j["b"], "recipient": j["recipient"], "arm": j["arm"],
            "repeat": j["repeat"], "draw": None, "role": "recipient", "status": "ok",
            "decision": dom.parse(g["text"]), "raw_reply": g["text"],
            "finish_reason": g["finish_reason"], "truncated": g["finish_reason"] == "length",
            "prompt_tokens": g["prompt_tokens"], "completion_tokens": g["completion_tokens"],
            "peer_models": [p["model"] for p in j["peers"]],
            "peer_decisions": [p["decision"] for p in j["peers"]],
            "prompt_sha": hashlib.sha256(pr.encode()).hexdigest()[:16], "max_tokens": mt,
            "template_version": F.VERSION if j["arm"] == "E5f" else "E5c/comma-fix-only"})
    st = CM.run(jobs, do, f"{a.arm}/{a.dataset}/{a.split}")
    print(f"  {st}")
    json.dump({"stats": st, "skipped": dict(skipped), "arm": a.arm, "split": a.split,
               "template_version": F.VERSION},
              open(os.path.join(STAGE, "logs",
                                f"collect_fixed_{a.dataset}_{a.arm}_{a.split}.json"), "w",
                   encoding="utf-8"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
