r"""Score the matching predictions and the deployable baselines that need no extra model calls.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/analyze_em.py --tag _test

Scoring is done once, here, and applied identically to every method, so a method is never credited
for a metric definition of its own. In particular the standard F1 used throughout is the harmonic
mean of **precision and recall**. ThriftLLM's own `type=1` report defines its "F1-score" as the
harmonic mean of **accuracy and recall**, which is a different quantity; that number is reported only
inside the original-artifact replay and is labelled there.

Repeats are averaged within a pair before anything else: the three seeds of one (pair, model) are one
observation, never three. Intervals resample **entity-connected blocks**, because pairs that share a
record are not independent; for WDC the block is a true `cluster_id`, for the other two it is the
left record.

Abstentions stay their own outcome. A method that abstains is not silently credited with a
non-match; `decision_rule` says what each baseline does with one, and the rate is reported.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics as st
from typing import Any, Dict, List, Optional, Tuple

import em_common as E

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")
OUT = os.path.join(BASE, "analysis")
DRAWS = 4000
SEED = 20261013


def load(tag: str) -> List[Dict[str, Any]]:
    out = []
    with open(os.path.join(RAW, f"em_preds{tag}.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                out.append(r)
    return out


def scores(items: List[Tuple[str, Optional[str], int]]) -> Dict[str, float]:
    """items = [(block, decision, gold)]; decision None is an abstention, never a non-match."""
    tp = sum(1 for _, d, g in items if d == "match" and g == 1)
    fp = sum(1 for _, d, g in items if d == "match" and g == 0)
    fn = sum(1 for _, d, g in items if d != "match" and g == 1)
    tn = sum(1 for _, d, g in items if d == "non-match" and g == 0)
    ab = sum(1 for _, d, _ in items if d is None)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    n = len(items)
    return {"n": n, "precision": P, "recall": R, "f1": F,
            "accuracy": (tp + tn) / n if n else 0.0,
            "abstention_rate": ab / n if n else 0.0,
            "false_merges": fp, "missed_matches": fn, "tp": tp, "tn": tn}


def boot_f1(items, draws=DRAWS, seed=SEED, alpha=0.05) -> Dict[str, float]:
    """Cluster bootstrap over entity-connected blocks."""
    by = collections.defaultdict(list)
    for b, d, g in items:
        by[b].append((b, d, g))
    keys = list(by)
    rnd = random.Random(seed)
    sims = []
    for _ in range(draws):
        pool = []
        for _ in range(len(keys)):
            pool.extend(by[keys[rnd.randrange(len(keys))]])
        sims.append(scores(pool)["f1"])
    sims.sort()
    return {"lo": sims[int(alpha / 2 * draws)], "hi": sims[min(draws - 1, int((1 - alpha / 2) * draws))],
            "n_blocks": len(keys)}


def majority(decisions: List[Optional[str]]) -> Optional[str]:
    """Independent voting with a frozen tie and abstention rule.

    Abstentions do not vote; they are not converted into non-match. A tie, or an all-abstain set,
    returns None, which is then scored as an abstention rather than as a silent negative.
    """
    votes = [d for d in decisions if d is not None]
    if not votes:
        return None
    c = collections.Counter(votes)
    top = max(c.values())
    winners = [k for k, v in c.items() if v == top]
    return winners[0] if len(winners) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_test")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = load(a.tag)
    print(f"  loaded {len(rows)} successful calls")

    # gold and blocks, evaluator-side
    gold: Dict[Tuple[str, str], int] = {}
    block: Dict[Tuple[str, str], str] = {}
    for dn in sorted({r["dataset"] for r in rows}):
        ds = E.Matching(dn)
        for row in ds.splits[a.split]:
            k = (dn, ds.pair_key(row))
            gold[k] = ds.gold(row)
            block[k] = ds.entity_block(row)

    # average the seeds within (dataset, pair, model): one observation per pair-model
    cell: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        cell[(r["dataset"], r["pair_key"], r["model"])].append(r)

    def modal(rs) -> Optional[str]:
        ds = [x["decision"] for x in rs]
        v = [d for d in ds if d is not None]
        if not v:
            return None
        c = collections.Counter(v)
        top = max(c.values())
        w = [k for k, n in c.items() if n == top]
        return w[0] if len(w) == 1 else None

    pred: Dict[Tuple[str, str, str], Optional[str]] = {k: modal(v) for k, v in cell.items()}
    cost: Dict[Tuple[str, str, str], float] = {
        k: sum(x["cost_units"] for x in v) / len(v) for k, v in cell.items()}
    toks: Dict[Tuple[str, str, str], Tuple[float, float]] = {
        k: (sum(x["prompt_tokens"] for x in v) / len(v),
            sum(x["completion_tokens"] for x in v) / len(v)) for k, v in cell.items()}

    datasets = sorted({k[0] for k in pred})
    models = list(E.POOL)
    res: Dict[str, Any] = {"draws": DRAWS, "seed": SEED, "split": a.split,
                           "f1_definition": "harmonic mean of precision and recall",
                           "per_model": {}, "voting": {}, "pool": E.POOL}

    print(f"\n{'dataset':15s} {'method':22s} {'n':>5s} {'F1':>6s} {'95% CI':>15s} "
          f"{'P':>6s} {'R':>6s} {'abst':>6s} {'cost':>9s}")
    for dn in datasets:
        pairs = sorted({k[1] for k in pred if k[0] == dn})
        # --- single models
        for m in models:
            items = [(block[(dn, p)], pred.get((dn, p, m)), gold[(dn, p)]) for p in pairs]
            s = scores(items)
            ci = boot_f1(items)
            c = sum(cost.get((dn, p, m), 0.0) for p in pairs)
            pt = sum(toks.get((dn, p, m), (0, 0))[0] for p in pairs)
            ct = sum(toks.get((dn, p, m), (0, 0))[1] for p in pairs)
            res["per_model"].setdefault(dn, {})[m] = {
                **s, **ci, "cost_units": c, "prompt_tokens": pt, "completion_tokens": ct,
                "calls": len(pairs)}
            print(f"{dn:15s} {m:22s} {s['n']:5d} {s['f1']:6.3f} "
                  f"[{ci['lo']:.3f},{ci['hi']:.3f}] {s['precision']:6.3f} {s['recall']:6.3f} "
                  f"{s['abstention_rate']:6.1%} {c:9.5f}")
        # --- independent voting over the whole pool
        items = [(block[(dn, p)], majority([pred.get((dn, p, m)) for m in models]), gold[(dn, p)])
                 for p in pairs]
        s = scores(items)
        ci = boot_f1(items)
        c = sum(cost.get((dn, p, m), 0.0) for p in pairs for m in models)
        res["voting"].setdefault(dn, {})["all6"] = {
            **s, **ci, "cost_units": c, "calls": len(pairs) * len(models),
            "decision_rule": "majority over non-abstaining models; tie or all-abstain -> abstention"}
        print(f"{dn:15s} {'vote(all 6 models)':22s} {s['n']:5d} {s['f1']:6.3f} "
              f"[{ci['lo']:.3f},{ci['hi']:.3f}] {s['precision']:6.3f} {s['recall']:6.3f} "
              f"{s['abstention_rate']:6.1%} {c:9.5f}")

    with open(os.path.join(OUT, f"em_baselines{a.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {OUT}/em_baselines{a.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
