r"""Run ThriftLLM's own method and capture its per-query predictions and realised cost.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/run_thrift.py --budgets 0.00002,0.00005,0.0001,0.0002

Why a runner rather than `main.py`: `main.py` prints a summary and discards the predictions, and its
summary uses ThriftLLM's own `type=1` metric, which defines "F1-score" as the harmonic mean of
**accuracy and recall**. That is a different quantity from the precision/recall F1 used for every
other method in this study, so comparing the printed numbers across methods would be an
apples-to-oranges error.

This runner imports their `ThriftLLM` class and calls `answer(query, embedding)` unchanged, one
query at a time. Selection, adaptive stopping and likelihood-based aggregation are entirely theirs;
we only record what came out. Their native summary is computed too and reported beside ours, clearly
labelled, so the method is never misrepresented by our choice of metric.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
THRIFT = os.path.join(BASE, "external", "thriftLLM")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")

DS_NAME = {"amazon_google": "AMAZON", "abt_buy": "ABTBUY", "wdc_products": "WDC"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="0.00002,0.00005,0.0001,0.0002,0.0005")
    ap.add_argument("--ncluster", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(THRIFT, "src"))
    os.chdir(os.path.join(THRIFT, "src"))          # their paths are relative to src/
    import ThriftLLM as T                                                 # noqa: PLC0415
    import random
    random.seed(42)

    out_rows: List[Dict[str, Any]] = []
    for dname, short in DS_NAME.items():
        for B in [float(x) for x in a.budgets.split(",")]:
            tr = f"../data/{short}/Queried_{short}_all_models_clean_train.csv"
            te = f"../data/{short}/Queried_{short}_all_models_clean_test.csv"
            emb = f"../data/{short}/{short}_embeddings.json"
            t0 = time.time()
            m = T.ThriftLLM(tr, te, emb, short, B, a.eps, a.delta, "ThriftLLM",
                            "kmeans", 0.5, 10, a.ncluster, 1.0, False)
            ds = m.thrift_ds_test
            idx2class = {v: k for k, v in ds.class2idx.items()}
            n_sel = collections.Counter()
            for q in ds.queries:
                qk = m.make_query_key(q)
                pred, real_cost, exp_cost = m.answer(q, m.ds_embeddings[qk])
                qi = ds.query2idx[q]
                ref = ds.data[qi][0][0]
                out_rows.append({
                    "dataset": dname, "budget": B, "query_raw": q,
                    "pred_class": idx2class.get(int(pred)),
                    "ref_class": idx2class.get(int(ref)),
                    "realised_cost": float(real_cost), "expected_cost": float(exp_cost),
                })
            el = time.time() - t0
            print(f"  {dname:15s} B={B:<9g} queries={len(ds.queries):5d} wall={el:5.1f}s",
                  flush=True)
    os.chdir(BASE)
    p = os.path.join(RAW, f"thrift_preds{a.tag}.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {p}  ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
