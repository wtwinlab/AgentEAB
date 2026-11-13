r"""M.A2 remainder: same-count selection and final aggregation, replayed offline.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/analyze_select_agg.py

Both dimensions are replays over the frozen per-model predictions, so they cost no model calls and
every rule sees exactly the same candidate set.

**Selection** asks whether *which* k models you consult matters beyond *how many*. Every rule picks
k of the six at a fixed k, so reply count, call count and token cost are matched by construction and
only the criterion differs. The rules are fixed in advance and none of them reads a label:
  random            a seeded draw, averaged over 200 draws so the baseline is not one lucky sample
  cheapest          the k lowest-cost models
  dev_best          the k models with the highest validation F1, chosen on the validation split
  dev_best_diverse  highest validation F1 subject to not taking two models from the same family
An oracle that picks the best k per pair using the label is reported as a labelled diagnostic only.

**Aggregation** holds the candidate set fixed and varies the final rule: majority with a frozen tie
rule, unanimity, any-match (recall-oriented), and validation-weighted voting whose weights come from
the validation split. ThriftLLM's own likelihood aggregation is not reimplemented here; its numbers
come from running its code and live in `em_method_comparison.json`.

Abstentions never silently become non-match: they do not vote, and a rule that cannot decide returns
an abstention, which is scored as its own outcome.
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import em_common as E
from analyze_em import boot_f1, scores

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")
OUT = os.path.join(BASE, "analysis")
DRAWS = 4000
SEED = 20261013
RANDOM_DRAWS = 200
FAMILY = {"qwen2.5-7b": "qwen", "qwen2.5-14b": "qwen", "qwen2.5-32b": "qwen",
          "qwen2.5-72b": "qwen", "llama-3-8b": "llama", "mistral-7b": "mistral"}


def load(tag: str) -> Dict[Tuple[str, str, str], Optional[str]]:
    cell = collections.defaultdict(list)
    with open(os.path.join(RAW, f"em_preds{tag}.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                cell[(r["dataset"], r["pair_key"], r["model"])].append(r)
    out = {}
    cost = {}
    for k, v in cell.items():
        d = [x["decision"] for x in v if x["decision"] is not None]
        if d:
            c = collections.Counter(d)
            top = max(c.values())
            w = [kk for kk, n in c.items() if n == top]
            out[k] = w[0] if len(w) == 1 else None
        else:
            out[k] = None
        cost[k] = sum(x["cost_units"] for x in v) / len(v)
    return out, cost


def vote(ds: List[Optional[str]], rule: str) -> Optional[str]:
    v = [d for d in ds if d is not None]
    if not v:
        return None
    if rule == "majority":
        c = collections.Counter(v)
        top = max(c.values())
        w = [k for k, n in c.items() if n == top]
        return w[0] if len(w) == 1 else None
    if rule == "unanimity":
        return v[0] if len(set(v)) == 1 else None
    if rule == "any_match":
        return "match" if "match" in v else "non-match"
    raise AssertionError(rule)


def weighted_vote(ds: Dict[str, Optional[str]], w: Dict[str, float]) -> Optional[str]:
    tot = collections.Counter()
    for m, d in ds.items():
        if d is not None:
            tot[d] += w.get(m, 0.0)
    if not tot:
        return None
    top = max(tot.values())
    winners = [k for k, v in tot.items() if abs(v - top) < 1e-12]
    return winners[0] if len(winners) == 1 else None


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    test, cost = load("_test")
    valid_pred, _ = load("_pilot")           # development data only, for choosing rules
    models = list(E.POOL)
    res: Dict[str, Any] = {"selection": {}, "aggregation": {}, "random_draws": RANDOM_DRAWS,
                           "note": ("selection rules are fixed in advance and read no label; the "
                                    "oracle row is a labelled diagnostic, not a policy")}

    for dn in ("abt_buy", "amazon_google", "wdc_products"):
        ds = E.Matching(dn)
        pairs = [ds.pair_key(r) for r in ds.splits["test"]]
        blk = {ds.pair_key(r): ds.entity_block(r) for r in ds.splits["test"]}
        gold = {ds.pair_key(r): ds.gold(r) for r in ds.splits["test"]}
        pairs = [p for p in pairs if all((dn, p, m) in test for m in models)]

        # validation-split quality per model, from the pilot (development) collection only
        vp = [ds.pair_key(r) for r in ds.splits["valid"]]
        vg = {ds.pair_key(r): ds.gold(r) for r in ds.splits["valid"]}
        vblk = {ds.pair_key(r): ds.entity_block(r) for r in ds.splits["valid"]}
        devf1 = {}
        for m in models:
            it = [(vblk[p], valid_pred.get((dn, p, m)), vg[p]) for p in vp
                  if (dn, p, m) in valid_pred]
            devf1[m] = scores(it)["f1"] if it else 0.0
        mcost = {m: sum(cost[(dn, p, m)] for p in pairs) for m in models}

        print(f"\n=== {dn} === {len(pairs)} pairs; dev F1 "
              f"{ {m: round(devf1[m],3) for m in models} }")

        # ---------------- selection at fixed k
        print(f"  {'k':>2s} {'rule':20s} {'F1':>6s} {'95% CI':>15s} {'cost':>9s}  models")
        for k in (1, 2, 3):
            subsets = list(itertools.combinations(models, k))
            rules: Dict[str, Any] = {
                "cheapest": tuple(sorted(models, key=lambda m: mcost[m])[:k]),
                "dev_best": tuple(sorted(models, key=lambda m: -devf1[m])[:k]),
            }
            div, seen = [], set()
            for m in sorted(models, key=lambda m: -devf1[m]):
                if FAMILY[m] in seen:
                    continue
                seen.add(FAMILY[m])
                div.append(m)
            rules["dev_best_diverse"] = tuple(div[:k]) if len(div) >= k else rules["dev_best"]

            for name, sub in rules.items():
                it = [(blk[p], vote([test[(dn, p, m)] for m in sub], "majority"), gold[p])
                      for p in pairs]
                s, ci = scores(it), boot_f1(it)
                c = sum(mcost[m] for m in sub)
                res["selection"].setdefault(dn, {}).setdefault(str(k), {})[name] = {
                    **s, **ci, "cost_units": c, "models": list(sub)}
                print(f"  {k:2d} {name:20s} {s['f1']:6.3f} [{ci['lo']:.3f},{ci['hi']:.3f}] "
                      f"{c:9.5f}  {','.join(sub)}")
            # random at the same k, averaged over draws
            rnd = random.Random(SEED)
            f1s, costs = [], []
            for _ in range(RANDOM_DRAWS):
                sub = tuple(rnd.sample(models, k))
                it = [(blk[p], vote([test[(dn, p, m)] for m in sub], "majority"), gold[p])
                      for p in pairs]
                f1s.append(scores(it)["f1"])
                costs.append(sum(mcost[m] for m in sub))
            res["selection"][dn][str(k)]["random_mean"] = {
                "f1": sum(f1s) / len(f1s), "f1_min": min(f1s), "f1_max": max(f1s),
                "cost_units": sum(costs) / len(costs), "draws": RANDOM_DRAWS}
            print(f"  {k:2d} {'random (mean of 200)':20s} {sum(f1s)/len(f1s):6.3f} "
                  f"[{min(f1s):.3f},{max(f1s):.3f}]* {sum(costs)/len(costs):9.5f}  "
                  f"*min/max over draws, not a CI")
            # labelled diagnostic: best subset chosen with the labels
            best = max(subsets, key=lambda sub: scores(
                [(blk[p], vote([test[(dn, p, m)] for m in sub], "majority"), gold[p])
                 for p in pairs])["f1"])
            it = [(blk[p], vote([test[(dn, p, m)] for m in best], "majority"), gold[p])
                  for p in pairs]
            res["selection"][dn][str(k)]["ORACLE_subset"] = {
                **scores(it), "cost_units": sum(mcost[m] for m in best),
                "models": list(best), "diagnostic": True}
            print(f"  {k:2d} {'ORACLE subset (diag)':20s} {scores(it)['f1']:6.3f} "
                  f"{'':17s}{sum(mcost[m] for m in best):9.5f}  {','.join(best)}")

        # ---------------- aggregation over the full pool
        print(f"  {'aggregation rule':24s} {'F1':>6s} {'95% CI':>15s} {'abst':>6s}")
        w = {m: max(0.0, devf1[m]) for m in models}
        for rule in ("majority", "unanimity", "any_match", "dev_weighted"):
            if rule == "dev_weighted":
                it = [(blk[p], weighted_vote({m: test[(dn, p, m)] for m in models}, w), gold[p])
                      for p in pairs]
            else:
                it = [(blk[p], vote([test[(dn, p, m)] for m in models], rule), gold[p])
                      for p in pairs]
            s, ci = scores(it), boot_f1(it)
            res["aggregation"].setdefault(dn, {})[rule] = {**s, **ci}
            print(f"  {rule:24s} {s['f1']:6.3f} [{ci['lo']:.3f},{ci['hi']:.3f}] "
                  f"{s['abstention_rate']:6.1%}")

    with open(os.path.join(OUT, "em_select_agg.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {OUT}/em_select_agg.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
