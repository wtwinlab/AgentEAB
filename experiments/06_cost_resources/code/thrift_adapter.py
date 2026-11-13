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

CLASSES = ["no", "yes"]
DS_NAME = {"amazon_google": "AMAZON", "abt_buy": "ABTBUY", "wdc_products": "WDC"}


def record_text(d: Dict[str, str]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in d.items())


def query_raw(left: Dict[str, str], right: Dict[str, str]) -> str:
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
                            rec[m] = "__INVALID__"
                        else:
                            rec[m] = "no"
                    else:
                        mrow[m] = False
                        rec[m] = "yes" if dec == "match" else "no"
                mask[pk] = mrow
                recs.append(rec)
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
    sys.path.insert(0, os.path.join(THRIFT, "src"))
    import utils as T
    from sentence_transformers import SentenceTransformer
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
