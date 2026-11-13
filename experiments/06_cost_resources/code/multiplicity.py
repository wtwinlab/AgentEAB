r"""The five pre-specified contrast families, tested and Holm-corrected.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/multiplicity.py

`FREEZE_MATCHING.md` names five contrast families and says Holm correction applies within each. Only
family 4 had actually been computed as a family; families 1, 3 and 5 existed as descriptive tables
with no paired test, and family 2's partner was chosen by looking at test F1. This file is the
confirmatory authority for all five. The descriptive tables keep their own numbers and are not
confirmatory.

Two selection rules, kept apart
-------------------------------
Where the freeze says "selected on validation", the choice is made on the **training** split and the
chosen method is then scored on test. Choosing the best single model by its test F1, as the
descriptive table does when it labels a maximum, is a selection that reads the thing being measured;
it inflates whatever it picks and it is not used here.

Family 2 asks for ThriftLLM "at matched budget". The budget is matched on realised cost, by taking
the B whose realised cost is nearest voting's, which involves no quality comparison at all.

Family 5 compares a score-based subset against random subsets of the **same size**, so call count
and token cost are matched by construction. A random baseline has no paired structure to bootstrap,
so the test is the fraction of the 200 same-size random subsets that reach the score-based subset's
F1. That is a permutation statement about the selection rule, not a bootstrap over records.
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import em_common as E
from analyze_em import boot_f1, majority, scores
from compare_methods import boot_diff_f1, jl
from thrift_adapter import query_raw

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")
OUT = os.path.join(BASE, "analysis")
DATASETS = ("abt_buy", "amazon_google", "wdc_products")
RANDOM_DRAWS = 200
SEED = 20261013
FAMILY_NAMES = {
    1: "best single model (chosen on validation) vs independent sampling and voting",
    2: "ThriftLLM at matched realised budget vs voting",
    3: "LOTUS direct semantic filter vs LOTUS cascade",
    4: "frozen-candidate sharing conditions",
    5: "same-count selection: score-based vs random",
}


def holm(tests: Dict[str, float], alpha: float = 0.05) -> Dict[str, Any]:
    """Holm-Bonferroni, step-down. Returns the per-test threshold and what survives."""
    order = sorted(tests, key=lambda k: tests[k])
    m = len(order)
    out, still = {}, True
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        if still and tests[k] <= thr:
            out[k] = {"p": tests[k], "threshold": thr, "survives": True}
        else:
            still = False
            out[k] = {"p": tests[k], "threshold": thr, "survives": False}
    return {"alpha": alpha, "family_size": m, "tests": out,
            "survivors": [k for k in order if out[k]["survives"]]}


def modal(rs: List[Dict[str, Any]]) -> Optional[str]:
    v = [x["decision"] for x in rs if x["decision"] is not None]
    if not v:
        return None
    c = collections.Counter(v)
    t = max(c.values())
    w = [k for k, n in c.items() if n == t]
    return w[0] if len(w) == 1 else None


def per_model(tag: str):
    cell = collections.defaultdict(list)
    for r in jl(f"em_preds{tag}.jsonl"):
        if r.get("status") == "ok":
            cell[(r["dataset"], r["pair_key"], r["model"])].append(r)
    dec = {k: modal(v) for k, v in cell.items()}
    cost = {k: sum(x["cost_units"] for x in v) / len(v) for k, v in cell.items()}
    return dec, cost


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    models = list(E.POOL)
    test_dec, test_cost = per_model("_test")
    train_dec, _ = per_model("_train")
    thrift = jl("thrift_preds.jsonl")
    sharing = json.load(open(os.path.join(OUT, "em_sharing.json"), encoding="utf-8"))

    res: Dict[str, Any] = {
        "families": FAMILY_NAMES,
        "correction": "Holm-Bonferroni within each family, alpha 0.05, one family per dataset",
        "selection_rule": ("where the freeze says 'selected on validation', the choice is made on "
                           "the training split and scored on test; no method is chosen by its test "
                           "F1"),
        "datasets": {}}

    for dn in DATASETS:
        ds = E.Matching(dn)
        meta = {ds.pair_key(r): (ds.entity_block(r), ds.gold(r)) for r in ds.splits["test"]}
        qr2pair: Dict[str, List[str]] = {}
        for r in ds.splits["test"]:
            L, R = ds.pair_record(r)
            qr2pair.setdefault(query_raw(L, R), []).append(ds.pair_key(r))

        # ---- validation F1 per model, on the training split, never on test
        dev: Dict[str, float] = {}
        for m in models:
            it = [(ds.entity_block(r), train_dec.get((dn, ds.pair_key(r), m)), ds.gold(r))
                  for r in ds.splits["train"]]
            dev[m] = scores(it)["f1"]
        dev_order = sorted(models, key=lambda m: -dev[m])
        chosen_single = dev_order[0]

        tb: Dict[float, Dict[str, str]] = collections.defaultdict(dict)
        tc: Dict[float, Dict[str, float]] = collections.defaultdict(dict)
        for r in thrift:
            if r["dataset"] != dn:
                continue
            for pk in qr2pair.get(r["query_raw"], []):
                tb[r["budget"]][pk] = "match" if r["pred_class"] == "yes" else "non-match"
                tc[r["budget"]][pk] = float(r["realised_cost"]) / len(qr2pair[r["query_raw"]])

        common = set(meta)
        for b in tb:
            common &= set(tb[b])
        for m in models:
            common &= {pk for (d0, pk, mm) in test_dec if d0 == dn and mm == m}
        common = sorted(common)

        def items(fn):
            return [(meta[pk][0], fn(pk), meta[pk][1]) for pk in common]

        it_single = items(lambda pk: test_dec.get((dn, pk, chosen_single)))
        it_vote = items(lambda pk: majority([test_dec.get((dn, pk, m)) for m in models]))
        cost_vote = sum(test_cost.get((dn, pk, m), 0.0) for pk in common for m in models)

        entry: Dict[str, Any] = {
            "n_pairs": len(common), "validation_f1": dev,
            "model_chosen_on_validation": chosen_single,
            "model_best_on_test": max(models, key=lambda m: scores(
                items(lambda pk, m=m: test_dec.get((dn, pk, m))))["f1"]),
            "families": {}}
        print(f"\n################ {dn}  ({len(common)} pairs)")
        print(f"  validation pick: {chosen_single} (dev F1 {dev[chosen_single]:.3f}); "
              f"test-maximal model would be {entry['model_best_on_test']}")

        # ---------------- family 1
        d1 = boot_diff_f1(it_single, it_vote)
        entry["families"]["1"] = {"name": FAMILY_NAMES[1],
                                  "tests": {f"single:{chosen_single} - vote:all6": d1},
                                  "holm": holm({f"single:{chosen_single} - vote:all6": d1["p"]})}

        # ---------------- family 2, budget matched on realised cost
        if tb:
            b_match = min(tb, key=lambda b: abs(sum(tc[b].get(pk, 0.0) for pk in common)
                                                - cost_vote))
            it_t = items(lambda pk, b=b_match: tb[b].get(pk))
            d2 = boot_diff_f1(it_t, it_vote)
            d2["budget"] = b_match
            d2["cost_thrift"] = sum(tc[b_match].get(pk, 0.0) for pk in common)
            d2["cost_vote"] = cost_vote
            key2 = f"thriftllm:B={b_match:g} - vote:all6"
            entry["families"]["2"] = {"name": FAMILY_NAMES[2], "tests": {key2: d2},
                                      "budget_matched_on": "realised cost, not quality",
                                      "holm": holm({key2: d2["p"]})}

        # ---------------- family 3, LOTUS direct vs cascade on the same pairs
        try:
            lo = {m: json.load(open(os.path.join(RAW, f"lotus_{dn}_{m}.json"), encoding="utf-8"))
                  for m in ("direct", "cascade")}
            dd = {m: {r["pair_key"]: r for r in lo[m]["rows"]} for m in lo}
            shared = sorted(set(dd["direct"]) & set(dd["cascade"]))
            def lit(m):
                return [(dd[m][k]["block"], "match" if dd[m][k]["kept"] else "non-match",
                         int(dd[m][k]["gold"])) for k in shared]
            d3 = boot_diff_f1(lit("direct"), lit("cascade"))
            d3["n_pairs"] = len(shared)
            entry["families"]["3"] = {"name": FAMILY_NAMES[3],
                                      "tests": {"lotus:direct - lotus:cascade": d3},
                                      "holm": holm({"lotus:direct - lotus:cascade": d3["p"]}),
                                      "note": ("gold accuracy. Fidelity to the reference model is a "
                                               "separate quantity and is reported in "
                                               "lotus_results.json")}
        except Exception as exc:                                          # noqa: BLE001
            entry["families"]["3"] = {"name": FAMILY_NAMES[3],
                                      "unavailable": f"{type(exc).__name__}: {exc}"}

        # ---------------- family 4, sharing, from the paired analysis
        f4 = {}
        for rec, byc in (sharing["contrasts"].get(dn) or {}).items():
            for c, v in byc.items():
                f4[f"{rec}: {c}"] = v["p"]
        if f4:
            entry["families"]["4"] = {
                "name": FAMILY_NAMES[4], "tests": {
                    k: sharing["contrasts"][dn][k.split(":")[0].strip()][k.split(":", 1)[1].strip()]
                    for k in f4},
                "holm": holm(f4),
                "stale_prompt_check": ("every contrast was recomputed with the stale-peer-block "
                                       "pairs dropped; see contrasts_stale_excluded in "
                                       "em_sharing.json")}

        # ---------------- family 5, score-based vs random at the same k
        f5: Dict[str, float] = {}
        det5: Dict[str, Any] = {}
        for k in (1, 2, 3):
            pick = dev_order[:k]
            f1_pick = scores(items(
                lambda pk, s=pick: majority([test_dec.get((dn, pk, m)) for m in s])))["f1"]
            rnd = random.Random(SEED)
            subsets = list(itertools.combinations(models, k))
            draws = [rnd.choice(subsets) for _ in range(RANDOM_DRAWS)]
            f1s = [scores(items(lambda pk, s=s: majority(
                [test_dec.get((dn, pk, m)) for m in s])))["f1"] for s in draws]
            ge = sum(1 for x in f1s if x >= f1_pick)
            p = max(1.0 / RANDOM_DRAWS, ge / RANDOM_DRAWS)
            f5[f"dev_best@k={k} vs random@k={k}"] = p
            det5[f"dev_best@k={k} vs random@k={k}"] = {
                "f1_score_based": f1_pick, "models": pick,
                "random_mean": sum(f1s) / len(f1s), "random_min": min(f1s), "random_max": max(f1s),
                "draws_reaching_it": ge, "draws": RANDOM_DRAWS, "p": p,
                "test": "one-sided permutation over same-size random subsets"}
        entry["families"]["5"] = {"name": FAMILY_NAMES[5], "tests": det5, "holm": holm(f5)}

        for fid in ("1", "2", "3", "4", "5"):
            fam = entry["families"].get(fid) or {}
            if "holm" not in fam:
                print(f"  family {fid}: {fam.get('unavailable', 'not computed')}")
                continue
            h = fam["holm"]
            print(f"  family {fid} ({h['family_size']} tests): "
                  f"{len(h['survivors'])} survive Holm")
            for k, v in h["tests"].items():
                t = fam["tests"].get(k) or {}
                eff = t.get("diff", t.get("f1_score_based"))
                mark = "SURVIVES" if v["survives"] else "        "
                print(f"    {mark} {k:52s} p={v['p']:.4f} <= {v['threshold']:.4f}? "
                      f"effect {eff:+.4f}" if eff is not None else
                      f"    {mark} {k:52s} p={v['p']:.4f}")
        res["datasets"][dn] = entry

    # ---------------- the SQL sharing family, corrected the same way
    sp = os.path.join(OUT, "sql_results.json")
    if os.path.exists(sp):
        sql = json.load(open(sp, encoding="utf-8"))
        tests = {f"{rec}: {c}": v["p"]
                 for rec, byc in (sql.get("contrasts") or {}).items()
                 for c, v in byc.items()}
        h = holm(tests)
        res["sql"] = {
            "name": "text-to-SQL sharing conditions, including execution feedback",
            "clustering": "databases, not questions",
            "tests": {f"{rec}: {c}": v for rec, byc in sql["contrasts"].items()
                      for c, v in byc.items()},
            "holm": h}
        print(f"\n################ text-to-SQL sharing "
              f"({h['family_size']} tests, clustered on 11 databases)")
        for k, v in h["tests"].items():
            e = res["sql"]["tests"][k]
            print(f"    {'SURVIVES' if v['survives'] else '        '} {k:52s} "
                  f"p={v['p']:.4f} <= {v['threshold']:.4f}? effect {e['mean']:+.4f} "
                  f"[{e['lo']:+.4f},{e['hi']:+.4f}]")
        print(f"  {len(h['survivors'])} survive Holm")

    with open(os.path.join(OUT, "multiplicity.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {OUT}/multiplicity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
