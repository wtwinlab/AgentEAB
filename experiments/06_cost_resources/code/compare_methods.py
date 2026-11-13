r"""Put every matching method on one cohort and one metric, then report the quality-cost frontier.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/compare_methods.py

Two things this file exists to prevent.

**Different cohorts.** ThriftLLM's dataset object keys queries by text
(`sorted(df['query_raw'].unique())`), so two record pairs whose rendered text is identical collapse
into one query. On these benchmarks that removes 5 of 1,916 Abt-Buy pairs and 27 of 2,293
Amazon-Google pairs. It is a property of the artifact, not a issue we introduce, but it means a
naive comparison would score ThriftLLM on 1,911 pairs and the baselines on 1,916. Everything here is
restricted to the **intersection** of pairs every method actually decided, and the dropped count is
reported.

**Different metrics.** ThriftLLM's own `type=1` summary reports "F1-score" as the harmonic mean of
accuracy and recall. Every number below is the harmonic mean of precision and recall, computed by
one scorer for all methods. Their native figure is reported separately and labelled, never mixed in.

Cost is the declared relative unit of `FREEZE_MATCHING.md` for both sides, and raw call counts are
carried alongside.
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import em_common as E
from analyze_em import boot_f1, majority, scores
from thrift_adapter import query_raw

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")
OUT = os.path.join(BASE, "analysis")
DRAWS = 4000
SEED = 20261013


def jl(name: str) -> List[Dict[str, Any]]:
    out = []
    with open(os.path.join(RAW, name), encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def boot_diff_f1(a_items, b_items, draws=DRAWS, seed=SEED, alpha=0.05) -> Dict[str, Any]:
    """Paired cluster bootstrap of F1(a) - F1(b) over entity-connected blocks."""
    by = collections.defaultdict(lambda: {"a": [], "b": []})
    for (blk, d, g) in a_items:
        by[blk]["a"].append((blk, d, g))
    for (blk, d, g) in b_items:
        by[blk]["b"].append((blk, d, g))
    keys = [k for k, v in by.items() if v["a"] and v["b"]]
    point = scores([x for k in keys for x in by[k]["a"]])["f1"] - \
        scores([x for k in keys for x in by[k]["b"]])["f1"]
    rnd = random.Random(seed)
    sims = []
    for _ in range(draws):
        ks = [keys[rnd.randrange(len(keys))] for _ in keys]
        fa = scores([x for k in ks for x in by[k]["a"]])["f1"]
        fb = scores([x for k in ks for x in by[k]["b"]])["f1"]
        sims.append(fa - fb)
    sims.sort()
    c = [s - point for s in sims]
    p = 2.0 * min(sum(1 for z in c if z >= -point), sum(1 for z in c if z <= -point)) / draws
    return {"diff": point, "lo": sims[int(alpha / 2 * draws)],
            "hi": sims[min(draws - 1, int((1 - alpha / 2) * draws))],
            "p": max(1.0 / draws, min(1.0, p)), "n_blocks": len(keys)}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    models = list(E.POOL)
    preds = jl("em_preds_test.jsonl")
    thrift = jl("thrift_preds.jsonl")

    res: Dict[str, Any] = {"metric": "F1 = harmonic mean of precision and recall, one scorer for all",
                           "cohort": "intersection of pairs every method decided", "datasets": {}}

    for dname in ("abt_buy", "amazon_google", "wdc_products"):
        ds = E.Matching(dname)
        # map thrift's query text back to the pair it came from
        qr2pair, pair_meta = {}, {}
        for row in ds.splits["test"]:
            L, R = ds.pair_record(row)
            pk = ds.pair_key(row)
            qr2pair.setdefault(query_raw(L, R), []).append(pk)
            pair_meta[pk] = (ds.entity_block(row), ds.gold(row))
        collapsed = sum(len(v) - 1 for v in qr2pair.values() if len(v) > 1)

        # our per-model modal decisions
        cell = collections.defaultdict(list)
        for r in preds:
            if r.get("status") == "ok" and r["dataset"] == dname:
                cell[(r["pair_key"], r["model"])].append(r)

        def modal(rs):
            v = [x["decision"] for x in rs if x["decision"] is not None]
            if not v:
                return None
            c = collections.Counter(v)
            top = max(c.values())
            w = [k for k, n in c.items() if n == top]
            return w[0] if len(w) == 1 else None

        ours = {k: modal(v) for k, v in cell.items()}
        cost_ours = {k: sum(x["cost_units"] for x in v) / len(v) for k, v in cell.items()}

        # thrift decisions per budget, mapped onto pairs (a collapsed query decides all its pairs)
        tb: Dict[float, Dict[str, str]] = collections.defaultdict(dict)
        tcost: Dict[float, Dict[str, float]] = collections.defaultdict(dict)
        for r in thrift:
            if r["dataset"] != dname:
                continue
            for pk in qr2pair.get(r["query_raw"], []):
                tb[r["budget"]][pk] = "match" if r["pred_class"] == "yes" else "non-match"
                tcost[r["budget"]][pk] = float(r["realised_cost"]) / max(1, len(qr2pair[r["query_raw"]]))

        budgets = sorted(tb)
        common = set(pair_meta)
        for b in budgets:
            common &= set(tb[b])
        for m in models:
            common &= {pk for (pk, mm) in ours if mm == m}
        common = sorted(common)

        def items_for(fn) -> List[Tuple[str, Optional[str], int]]:
            return [(pair_meta[pk][0], fn(pk), pair_meta[pk][1]) for pk in common]

        entry: Dict[str, Any] = {"n_pairs_common": len(common),
                                 "n_pairs_full_split": len(pair_meta),
                                 "pairs_collapsed_by_identical_text": collapsed,
                                 "methods": {}}
        print(f"\n=== {dname} === cohort {len(common)} pairs "
              f"(split {len(pair_meta)}; {collapsed} collapsed by identical rendered text)")
        print(f"  {'method':30s} {'F1':>6s} {'95% CI':>15s} {'P':>6s} {'R':>6s} {'cost':>9s} "
              f"{'calls':>7s}")

        rows: Dict[str, Any] = {}
        for m in models:
            it = items_for(lambda pk, m=m: ours.get((pk, m)))
            s, ci = scores(it), boot_f1(it)
            c = sum(cost_ours.get((pk, m), 0.0) for pk in common)
            rows[f"single:{m}"] = {**s, **ci, "cost_units": c, "calls": len(common), "items": it}
        it = items_for(lambda pk: majority([ours.get((pk, m)) for m in models]))
        s, ci = scores(it), boot_f1(it)
        rows["vote:all6"] = {**s, **ci,
                             "cost_units": sum(cost_ours.get((pk, m), 0.0)
                                               for pk in common for m in models),
                             "calls": len(common) * len(models), "items": it}
        for b in budgets:
            it = items_for(lambda pk, b=b: tb[b].get(pk))
            s, ci = scores(it), boot_f1(it)
            rows[f"thriftllm:B={b:g}"] = {**s, **ci,
                                          "cost_units": sum(tcost[b].get(pk, 0.0) for pk in common),
                                          "calls": None, "items": it}
        for k, v in rows.items():
            print(f"  {k:30s} {v['f1']:6.3f} [{v['lo']:.3f},{v['hi']:.3f}] "
                  f"{v['precision']:6.3f} {v['recall']:6.3f} {v['cost_units']:9.5f} "
                  f"{str(v['calls'] or '-'):>7s}")
            entry["methods"][k] = {kk: vv for kk, vv in v.items() if kk != "items"}

        # the two contrasts that matter, paired on blocks
        best_single = max((k for k in rows if k.startswith("single:")),
                          key=lambda k: rows[k]["f1"])
        best_thrift = max((k for k in rows if k.startswith("thriftllm:")),
                          key=lambda k: rows[k]["f1"])
        for a_, b_ in ((best_thrift, "vote:all6"), (best_thrift, best_single)):
            d = boot_diff_f1(rows[a_]["items"], rows[b_]["items"])
            entry.setdefault("contrasts", {})[f"{a_} - {b_}"] = d
            print(f"  {a_} − {b_}: {d['diff']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] p={d['p']:.4f}"
                  f"   cost {rows[a_]['cost_units']:.4f} vs {rows[b_]['cost_units']:.4f}")
        res["datasets"][dname] = entry

    with open(os.path.join(OUT, "em_method_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {OUT}/em_method_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
