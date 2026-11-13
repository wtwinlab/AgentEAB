r"""M.A2 analysis: what the shared peer block contributes on a data-processing task.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/analyze_share.py

Each contrast pairs two conditions on the **same recipient and the same pair**, so the comparison is
within-recipient and the only thing that differs is the rendered peer block. Intervals resample
entity-connected blocks, never pairs.

The question this answers is whether the QA finding transfers. In the QA study, reducing a peer
reply to its answer value beat sending the full reply. Here the analogous contrast is
`candidate_only` against `full`, with `candidate_evidence` added because a data-processing task
offers something QA did not: field-level evidence that can be checked against the records
themselves. `no_peers` is the independent-revision floor.

Transitions are reported the same way as in the QA study, because a change in aggregate F1 can hide
offsetting corrections and regressions: for each condition we report how often a recipient that was
initially wrong becomes right, and how often a recipient that was initially right becomes wrong.

One stimulus issue, and what is done about it
----------------------------------------------
The frozen parser dropped the surface form "no match", recording 712 readable negatives from
`llama-3-8b` as abstentions (`audit/parser_repair.json`). Because `candidate_only` and
`candidate_evidence` render `decision or 'no answer'`, 84 pairs per condition were shown a peer that
had in fact answered. Re-parsing fixes the measurement but cannot unsend a prompt, so every contrast
is computed twice: once on all pairs and once with those pairs dropped. Both are reported. `full`
shows the verbatim reply and `no_peers` shows no peers, so neither is affected.
"""
from __future__ import annotations

import collections
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import em_common as E
from analyze_em import scores

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")
OUT = os.path.join(BASE, "analysis")
CONDS = ("no_peers", "candidate_only", "candidate_evidence", "full")
DRAWS = 4000
SEED = 20261013


def boot_diff_f1(a_items, b_items, draws=DRAWS, seed=SEED, alpha=0.05) -> Dict[str, Any]:
    by = collections.defaultdict(lambda: {"a": [], "b": []})
    for blk, d, g in a_items:
        by[blk]["a"].append((blk, d, g))
    for blk, d, g in b_items:
        by[blk]["b"].append((blk, d, g))
    keys = [k for k, v in by.items() if v["a"] and v["b"]]
    point = (scores([x for k in keys for x in by[k]["a"]])["f1"]
             - scores([x for k in keys for x in by[k]["b"]])["f1"])
    rnd = random.Random(seed)
    sims = []
    for _ in range(draws):
        ks = [keys[rnd.randrange(len(keys))] for _ in keys]
        sims.append(scores([x for k in ks for x in by[k]["a"]])["f1"]
                    - scores([x for k in ks for x in by[k]["b"]])["f1"])
    sims.sort()
    c = [s - point for s in sims]
    p = 2.0 * min(sum(1 for z in c if z >= -point), sum(1 for z in c if z <= -point)) / draws
    return {"diff": point, "lo": sims[int(alpha / 2 * draws)],
            "hi": sims[min(draws - 1, int((1 - alpha / 2) * draws))],
            "p": max(1.0 / draws, min(1.0, p)), "n_blocks": len(keys)}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with open(os.path.join(RAW, "em_share_test.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                rows.append(r)
    print(f"  loaded {len(rows)} sharing calls")

    gold: Dict[Tuple[str, str], int] = {}
    for dn in sorted({r["dataset"] for r in rows}):
        ds = E.Matching(dn)
        for row in ds.splits["test"]:
            gold[(dn, ds.pair_key(row))] = ds.gold(row)

    idx = {(r["dataset"], r["pair_key"], r["recipient"], r["condition"]): r for r in rows}
    # pairs whose sent prompt described a readable peer as having given no answer
    stale = {(r["dataset"], r["pair_key"], r["recipient"]) for r in rows
             if r.get("peer_block_stale")}
    print(f"  {len(stale)} (dataset, pair, recipient) triples carry a stale peer block; "
          f"every contrast is also recomputed without them")
    res: Dict[str, Any] = {"draws": DRAWS, "seed": SEED, "conditions": list(CONDS),
                           "levels": {}, "contrasts": {}, "contrasts_stale_excluded": {},
                           "transitions": {}, "cost": {},
                           "stale_peer_block": {
                               "triples": len(stale),
                               "cause": ("the frozen parser dropped the surface form 'no match', so "
                                         "candidate_only and candidate_evidence rendered a peer that "
                                         "had answered as 'no answer'; see audit/parser_repair.json"),
                               "handling": ("re-parsing repairs the measurement but cannot unsend a "
                                            "prompt, so contrasts are reported with and without "
                                            "these pairs"),
                               "unaffected": ["full", "no_peers"]}}

    print(f"\n{'dataset':15s} {'recipient':13s} {'condition':20s} {'n':>5s} {'F1':>6s} "
          f"{'P':>6s} {'R':>6s} {'abst':>6s} {'in-tok':>8s}")
    for dn in sorted({r["dataset"] for r in rows}):
        for rec in sorted({r["recipient"] for r in rows if r["dataset"] == dn}):
            pairs = sorted({k[1] for k in idx if k[0] == dn and k[2] == rec
                            and all((dn, k[1], rec, c) in idx for c in CONDS)})
            if not pairs:
                continue
            ds = E.Matching(dn)
            blk = {ds.pair_key(row): ds.entity_block(row) for row in ds.splits["test"]}
            items: Dict[str, List] = {}
            for c in CONDS:
                it = [(blk[p], idx[(dn, p, rec, c)]["decision"], gold[(dn, p)]) for p in pairs]
                items[c] = it
                s = scores(it)
                tok = sum(idx[(dn, p, rec, c)]["prompt_tokens"] for p in pairs) / len(pairs)
                res["levels"].setdefault(dn, {}).setdefault(rec, {})[c] = {**s, "mean_input_tokens": tok}
                res["cost"].setdefault(dn, {}).setdefault(rec, {})[c] = {
                    "mean_input_tokens": tok,
                    "total_cost_units": sum(idx[(dn, p, rec, c)]["cost_units"] for p in pairs)}
                print(f"{dn:15s} {rec:13s} {c:20s} {s['n']:5d} {s['f1']:6.3f} "
                      f"{s['precision']:6.3f} {s['recall']:6.3f} {s['abstention_rate']:6.1%} "
                      f"{tok:8.0f}")
            clean = [p for p in pairs if (dn, p, rec) not in stale]
            items_clean: Dict[str, List] = {
                c: [(blk[p], idx[(dn, p, rec, c)]["decision"], gold[(dn, p)]) for p in clean]
                for c in CONDS}
            for a_, b_ in (("candidate_only", "full"), ("candidate_evidence", "full"),
                           ("full", "no_peers"), ("candidate_only", "no_peers"),
                           ("candidate_evidence", "candidate_only")):
                d = boot_diff_f1(items[a_], items[b_])
                res["contrasts"].setdefault(dn, {}).setdefault(rec, {})[f"{a_} - {b_}"] = d
                d2 = boot_diff_f1(items_clean[a_], items_clean[b_])
                d2["n_pairs"] = len(clean)
                d2["n_pairs_dropped"] = len(pairs) - len(clean)
                res["contrasts_stale_excluded"].setdefault(dn, {}).setdefault(
                    rec, {})[f"{a_} - {b_}"] = d2
            # transitions against the recipient's own frozen initial answer
            for c in CONDS:
                t = collections.Counter()
                for p in pairs:
                    r = idx[(dn, p, rec, c)]
                    init = r.get("own_initial")
                    g = gold[(dn, p)]
                    was = (init == "match") == (g == 1) if init is not None else None
                    now = (r["decision"] == "match") == (g == 1) if r["decision"] is not None else None
                    if was is None or now is None:
                        t["unreadable"] += 1
                    else:
                        t[f"{'right' if was else 'wrong'}_to_{'right' if now else 'wrong'}"] += 1
                res["transitions"].setdefault(dn, {}).setdefault(rec, {})[c] = dict(t)

    print(f"\n=== contrasts (paired on recipient and pair; blocks bootstrapped) ===")
    print(f"  the last two columns drop the pairs whose sent prompt carried a stale peer block")
    print(f"{'dataset':15s} {'recip':13s} {'contrast':38s} {'dF1':>8s} {'95% CI':>18s} {'p':>7s}"
          f" | {'dF1*':>8s} {'p*':>7s} {'drop':>5s}")
    for dn in res["contrasts"]:
        for rec in res["contrasts"][dn]:
            for k, d in res["contrasts"][dn][rec].items():
                e = res["contrasts_stale_excluded"][dn][rec][k]
                print(f"{dn:15s} {rec:13s} {k:38s} {d['diff']:+8.4f} "
                      f"[{d['lo']:+.4f},{d['hi']:+.4f}] {d['p']:7.4f}"
                      f" | {e['diff']:+8.4f} {e['p']:7.4f} {e['n_pairs_dropped']:5d}")

    print(f"\n=== transitions vs the recipient's own initial answer ===")
    for dn in res["transitions"]:
        for rec in res["transitions"][dn]:
            print(f"  {dn} / {rec}")
            for c in CONDS:
                t = res["transitions"][dn][rec][c]
                print(f"    {c:20s} wrong→right {t.get('wrong_to_right',0):4d}  "
                      f"right→wrong {t.get('right_to_wrong',0):4d}  "
                      f"net {t.get('wrong_to_right',0)-t.get('right_to_wrong',0):+4d}")

    with open(os.path.join(OUT, "em_sharing.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {OUT}/em_sharing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
