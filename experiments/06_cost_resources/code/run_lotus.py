r"""LOTUS direct and cascade semantic filtering over the same blocked pair relation.

    results/dataeng_2026-10-13/external/lotus_venv/bin/python \
      experiments/06_cost_resources/code/run_lotus.py --dataset wdc_products --mode direct

Both modes execute LOTUS's own `sem_filter` on one relation with one predicate, so the comparison is
inside LOTUS rather than between LOTUS and a reimplementation. Direct runs the reference model over
every row. Cascade runs its proxy-plus-reference plan with its own calibration sampling, and we log
the calls each one actually made.

Two boundaries the plan insists on, enforced here
-------------------------------------------------
**Reference fidelity is not gold accuracy.** The cascade's targets are stated against *the reference
model's own labels*, not the benchmark's. We therefore record both: agreement with the direct run
(reference fidelity) and F1 against the benchmark (gold accuracy). They are reported side by side
and never substituted for one another.

**All costs count.** Proxy calls, reference calls and calibration-sample calls are logged from the
LM's own usage statistics, so the cascade is not given free reference calls. A batch-level target is
also not a per-query guarantee, and nothing here is described as a preserved guarantee once the
relation or the predicate differs from the paper's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")
sys.path.insert(0, HERE)

import em_common as E                                                     # noqa: E402

PREDICATE = "{left_record} and {right_record} refer to the same real-world product"


def build_relation(dname: str, split: str, limit: int) -> "pd.DataFrame":
    import pandas as pd
    ds = E.Matching(dname)
    rows = ds.splits[split][:limit] if limit else ds.splits[split]
    recs = []
    for row in rows:
        L, R = ds.pair_record(row)
        recs.append({
            "pair_key": ds.pair_key(row),
            "block": ds.entity_block(row),
            "left_record": "; ".join(f"{k}: {v}" for k, v in L.items()),
            "right_record": "; ".join(f"{k}: {v}" for k, v in R.items()),
            "_gold": ds.gold(row),          # evaluator-only; dropped before sem_filter sees the frame
        })
    return pd.DataFrame(recs)


def usage(lm) -> Dict[str, Any]:
    st = lm.stats
    out: Dict[str, Any] = {}
    for side in ("physical_usage", "virtual_usage"):
        u = getattr(st, side, None)
        if u is None:
            continue
        out[side] = {k: getattr(u, k, None) for k in
                     ("total_tokens", "prompt_tokens", "completion_tokens", "total_cost")}
    out["cache_hits"] = getattr(st, "cache_hits", None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wdc_products")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default="direct", choices=("direct", "cascade"))
    ap.add_argument("--reference", default="qwen2.5-72b")
    ap.add_argument("--proxy", default="qwen2.5-7b")
    ap.add_argument("--recall-target", type=float, default=0.9)
    ap.add_argument("--precision-target", type=float, default=0.9)
    ap.add_argument("--sampling-percentage", type=float, default=0.1)
    ap.add_argument("--failure-probability", type=float, default=0.2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import lotus
    from lotus.models import LM
    from lotus.types import CascadeArgs

    ref_port = E.POOL[a.reference]["port"]
    lm = LM(model=f"openai/{a.reference}", api_base=f"http://{E.HOST}:{ref_port}/v1",
            api_key="EMPTY", max_ctx_len=4096, max_tokens=384)
    kw: Dict[str, Any] = {}
    if a.mode == "cascade":
        prox_port = E.POOL[a.proxy]["port"]
        helper = LM(model=f"openai/{a.proxy}", api_base=f"http://{E.HOST}:{prox_port}/v1",
                    api_key="EMPTY", max_ctx_len=4096, max_tokens=384)
        lotus.settings.configure(lm=lm, helper_lm=helper)
        kw["cascade_args"] = CascadeArgs(recall_target=a.recall_target,
                                         precision_target=a.precision_target,
                                         sampling_percentage=a.sampling_percentage,
                                         failure_probability=a.failure_probability)
    else:
        helper = None
        lotus.settings.configure(lm=lm)

    df = build_relation(a.dataset, a.split, a.limit)
    gold = dict(zip(df["pair_key"], df["_gold"]))
    blocks = dict(zip(df["pair_key"], df["block"]))
    frame = df.drop(columns=["_gold"])          # the predicate never sees the label

    t0 = time.time()
    kept = frame.sem_filter(PREDICATE, **kw)
    wall = time.time() - t0

    keys = set(kept["pair_key"].tolist())
    rows = [{"pair_key": k, "block": blocks[k], "kept": k in keys, "gold": gold[k]}
            for k in frame["pair_key"].tolist()]
    out = {
        "dataset": a.dataset, "split": a.split, "mode": a.mode,
        "reference_model": a.reference, "proxy_model": (a.proxy if a.mode == "cascade" else None),
        "predicate": PREDICATE, "n_rows": len(frame), "n_kept": len(keys),
        "wall_s": round(wall, 1),
        "cascade_args": (dict(recall_target=a.recall_target, precision_target=a.precision_target,
                              sampling_percentage=a.sampling_percentage,
                              failure_probability=a.failure_probability)
                         if a.mode == "cascade" else None),
        "usage_reference": usage(lm),
        "usage_proxy": (usage(helper) if helper is not None else None),
        "note": ("cascade targets are stated against the reference model's own labels; agreement "
                 "with the direct run is reference fidelity and is not gold accuracy"),
        "rows": rows,
    }
    p = os.path.join(RAW, f"lotus_{a.dataset}_{a.mode}{a.tag}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  {a.dataset} {a.mode}: {len(frame)} rows, kept {len(keys)}, {wall:.0f}s")
    print(f"    reference usage: {out['usage_reference'].get('physical_usage')}")
    if out["usage_proxy"]:
        print(f"    proxy usage:     {out['usage_proxy'].get('physical_usage')}")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
