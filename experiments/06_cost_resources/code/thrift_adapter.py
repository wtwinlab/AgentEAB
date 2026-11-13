r"""Adapter: turn our collected predictions into the tables ThriftLLM's own code consumes.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/thrift_adapter.py --build

The point of the adapter is that ThriftLLM keeps **its own** selection, adaptive stopping and
likelihood-based aggregation. We supply the prediction table and the embeddings its `main.py`
expects, and we do not reimplement any part of the method. Ordinary majority voting is reported
separately and is never called ThriftLLM.

Three deviations from the shipped artifact, each deliberate and each testable
-----------------------------------------------------------------------------
1. **Invalid outputs must not consult the label.** `ThriftDataset.fill_the_data` maps a prediction
   outside the class set to `(gold + 1) % K`, which reads the reference answer and, for a binary
   task, forces "always wrong". We cannot simply write an out-of-vocabulary token, because their
   loader would then apply exactly that rule. So the abstention is resolved *before* the table is
   written, by a rule fixed in advance and independent of the label: **an abstention becomes the
   negative class `no`**, the conservative default for entity matching, where the cost of a wrong
   merge is the thing being avoided. `--invalid-rule original` reproduces the shipped behaviour for
   the side-by-side comparison, and the per-cell abstention mask is saved either way so the affected
   slice is measurable.

2. **The class vocabulary is declared, not inferred.** Their loader takes
   `sorted(df['ref_answer'].unique())`, i.e. it reads the held-out reference column. We declare
   `['no', 'yes']` from the task schema and assert that the inferred set equals it, so the deviation
   is checked rather than assumed harmless.

3. **Cost.** Their `price.py` fixes output length at one token. That is kept for the
   original-artifact replay. The common-workload comparison uses our recorded token counts, and the
   two are never added together.

`query_raw` must be shaped so their `entityembedding_emb_key` works: it strips `"Product A: "` from
line 0 and `"Product B: "` from line 1 and concatenates, so those exact prefixes are required.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import em_common as E

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")
THRIFT = os.path.join(BASE, "external", "thriftLLM")

#: declared from the task schema, never inferred from a reference column
CLASSES = ["no", "yes"]
DS_NAME = {"amazon_google": "AMAZON", "abt_buy": "ABTBUY", "wdc_products": "WDC"}


def record_text(d: Dict[str, str]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in d.items())


def query_raw(left: Dict[str, str], right: Dict[str, str]) -> str:
    """Their `entityembedding_emb_key` strips 'Product A: ' / 'Product B: ' from the first two
    lines, so the shape is load-bearing rather than cosmetic."""
    return (f"Product A: {record_text(left)}\n"
            f"Product B: {record_text(right)}\n"
            "Do these two records refer to the same real-world product? Answer yes or no.")


def load(tag: str) -> List[Dict[str, Any]]:
    out = []
    p = os.path.join(RAW, f"em_preds{tag}.jsonl")
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                out.append(r)
    return out


def build(invalid_rule: str) -> Dict[str, Any]:
    models = list(E.POOL)
    report: Dict[str, Any] = {"invalid_rule": invalid_rule, "classes_declared": CLASSES,
                              "datasets": {}}
    for split, tag in (("train", "_train"), ("test", "_test")):
        rows = load(tag)
        by: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in rows:
            by[(r["dataset"], r["pair_key"], r["model"])].append(r)
        for dname in sorted({k[0] for k in by}):
            ds = E.Matching(dname)
            out_dir = os.path.join(THRIFT, "data", DS_NAME[dname])
            os.makedirs(out_dir, exist_ok=True)
            pairs = {ds.pair_key(row): row for row in ds.splits[split]}
            recs, n_abstain, n_cells = [], 0, 0
            mask: Dict[str, Dict[str, bool]] = {}
            for pk, row in pairs.items():
                if not any((dname, pk, m) in by for m in models):
                    continue
                L, R = ds.pair_record(row)
                qr = query_raw(L, R)
                rec = {"query_raw": qr, "query": qr,
                       "ref_answer": "yes" if ds.gold(row) == 1 else "no"}
                mrow = {}
                for m in models:
                    cell = by.get((dname, pk, m)) or []
                    # modal decision across seeds; a tie is an abstention
                    v = [c["decision"] for c in cell if c["decision"] is not None]
                    dec = None
                    if v:
                        c = collections.Counter(v)
                        top = max(c.values())
                        w = [k for k, n in c.items() if n == top]
                        dec = w[0] if len(w) == 1 else None
                    n_cells += 1
                    if dec is None:
                        n_abstain += 1
                        mrow[m] = True
                        if invalid_rule == "original":
                            # write an out-of-vocabulary token and let their loader apply its own
                            # gold-reading rule, reproducing the shipped artifact exactly
                            rec[m] = "__INVALID__"
                        else:
                            # label-independent, declared in advance: abstain -> negative class
                            rec[m] = "no"
                    else:
                        mrow[m] = False
                        rec[m] = "yes" if dec == "match" else "no"
                mask[pk] = mrow
                recs.append(rec)
            # their loader infers the class set from this column; check it equals the declared one
            inferred = sorted({r["ref_answer"] for r in recs})
            assert inferred == CLASSES, f"{dname}/{split}: inferred {inferred} != declared {CLASSES}"
            fn = os.path.join(out_dir,
                              f"Queried_{DS_NAME[dname]}_all_models_clean_{split}.csv")
            with open(fn, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["query_raw", "query", "ref_answer"] + models)
                w.writeheader()
                for r in recs:
                    w.writerow(r)
            with open(os.path.join(BASE, "raw",
                                   f"thrift_abstain_mask_{dname}_{split}.json"), "w") as f:
                json.dump(mask, f)
            report["datasets"].setdefault(dname, {})[split] = {
                "rows": len(recs), "cells": n_cells, "abstentions": n_abstain,
                "abstention_rate": n_abstain / max(1, n_cells),
                "inferred_classes_equal_declared": inferred == CLASSES,
                "file": os.path.relpath(fn, BASE)}
            print(f"  {dname:15s} {split:5s} rows={len(recs):5d} cells={n_cells:6d} "
                  f"abstain={n_abstain:5d} ({n_abstain/max(1,n_cells):.2%}) -> {os.path.basename(fn)}")
    return report


def embed() -> Dict[str, Any]:
    """Embeddings keyed exactly as their `entityembedding_emb_key` computes them."""
    sys.path.insert(0, os.path.join(THRIFT, "src"))
    import utils as T                                                     # noqa: PLC0415
    from sentence_transformers import SentenceTransformer                 # noqa: PLC0415
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    out = {}
    for dname, short in DS_NAME.items():
        keys = set()
        for split in ("train", "test"):
            fn = os.path.join(THRIFT, "data", short,
                              f"Queried_{short}_all_models_clean_{split}.csv")
            if not os.path.exists(fn):
                continue
            with open(fn, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    keys.add(T.entityembedding_emb_key(row["query_raw"]))
        keys = sorted(keys)
        vecs = model.encode(keys, batch_size=256, show_progress_bar=False,
                            normalize_embeddings=True)
        emb = {k: v.tolist() for k, v in zip(keys, vecs)}
        p = os.path.join(THRIFT, "data", short, f"{short}_embeddings.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(emb, f)
        out[dname] = {"keys": len(keys), "dim": len(vecs[0]), "file": os.path.relpath(p, BASE),
                      "encoder": "BAAI/bge-small-en-v1.5", "normalized": True}
        print(f"  {dname:15s} {len(keys):6d} embedding keys, dim {len(vecs[0])}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--invalid-rule", default="abstain_to_negative",
                    choices=("abstain_to_negative", "original"))
    a = ap.parse_args()
    rep: Dict[str, Any] = {}
    if a.build:
        print(f"=== building ThriftLLM tables (invalid rule: {a.invalid_rule}) ===")
        rep["build"] = build(a.invalid_rule)
    if a.embed:
        print("=== embeddings ===")
        rep["embed"] = embed()
    if rep:
        p = os.path.join(BASE, "analysis", f"thrift_adapter_{a.invalid_rule}.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, sort_keys=True, default=str)
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
