from __future__ import annotations

import collections
import json
import os
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(BASE, "analysis")


def J(name: str) -> Dict[str, Any]:
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    comp = J("em_method_comparison.json")
    lot = J("lotus_results.json")
    sweep = J("lotus_target_sweep.json")
    sel = J("em_select_agg.json")

    res: Dict[str, Any] = {
        "normalisation": ("multiples of the cheapest single model's realised cost on the same "
                          "pairs; absolute relative units and raw counts retained"),
        "budget_controls": {
            "single model": "none (one call per pair)",
            "ensemble + vote": "model count k",
            "ThriftLLM": "per-query cost cap B, with adaptive stopping",
            "LOTUS direct": "none",
            "LOTUS cascade": "batch recall and precision targets",
        },
        "note": "these controls are not interchangeable and are not mapped onto one knob",
        "datasets": {}}

    for dn in ("abt_buy", "amazon_google", "wdc_products"):
        M = comp["datasets"][dn]["methods"]
        singles = {k: v for k, v in M.items() if k.startswith("single:")}
        ref = min(v["cost_units"] for v in singles.values())
        pts: List[Dict[str, Any]] = []
        for k, v in M.items():
            pts.append({"method": k, "f1": v["f1"], "lo": v["lo"], "hi": v["hi"],
                        "cost_units": v["cost_units"], "x_ref": v["cost_units"] / ref,
                        "calls": v.get("calls"),
                        "control": ("per-query cost cap" if k.startswith("thriftllm")
                                    else "model count" if k.startswith("vote")
                                    else "none")})
        for mode in ("direct", "cascade"):
            d = lot[dn][mode]
            pts.append({"method": f"lotus:{mode}", "f1": d["f1"], "lo": d["lo"], "hi": d["hi"],
                        "cost_tokens": d["total_tokens"], "cost_units": None,
                        "x_ref": None, "calls": None,
                        "control": ("batch targets" if mode == "cascade" else "none"),
                        "note": "token-denominated; not on the per-call unit axis"})
        for t, d in sweep[dn].items():
            pts.append({"method": f"lotus:cascade@{t}", "f1": d["f1"],
                        "cost_tokens": d["total"], "cost_units": None, "x_ref": None,
                        "control": "batch targets", "target": t,
                        "ratio_vs_direct": d["ratio_vs_direct"],
                        "note": "token-denominated; not on the per-call unit axis"})
        for k, rules in (sel["selection"].get(dn) or {}).items():
            for rule, v in rules.items():
                if rule in ("random_mean", "ORACLE_subset"):
                    continue
                pts.append({"method": f"select:{rule}@k={k}", "f1": v["f1"],
                            "lo": v.get("lo"), "hi": v.get("hi"),
                            "cost_units": v["cost_units"], "x_ref": v["cost_units"] / ref,
                            "control": "model count", "models": v.get("models")})
        res["datasets"][dn] = {"reference_cost_units": ref, "points": pts}

        pr = [p for p in pts if p.get("cost_units") is not None]
        front = []
        for p in sorted(pr, key=lambda z: z["cost_units"]):
            if not front or p["f1"] > front[-1]["f1"]:
                front.append(p)
        res["datasets"][dn]["pareto_frontier"] = [
            {"method": p["method"], "f1": p["f1"], "x_ref": round(p["x_ref"], 2)} for p in front]
        print(f"\n=== {dn} === reference cost {ref:.5f} units (cheapest single model)")
        print(f"  {'method':30s} {'F1':>6s} {'xref':>6s}  control")
        for p in front:
            print(f"  {p['method']:30s} {p['f1']:6.3f} {p['x_ref']:6.2f}x  {p['control']}")
        best = max(pr, key=lambda z: z["f1"])
        cheap_good = min((p for p in pr if p["f1"] >= best["f1"] - 0.02),
                         key=lambda z: z["cost_units"])
        res["datasets"][dn]["best"] = {"method": best["method"], "f1": best["f1"],
                                       "x_ref": best["x_ref"]}
        res["datasets"][dn]["cheapest_within_2pp_of_best"] = {
            "method": cheap_good["method"], "f1": cheap_good["f1"], "x_ref": cheap_good["x_ref"]}
        print(f"  best: {best['method']} F1 {best['f1']:.3f} at {best['x_ref']:.2f}x")
        print(f"  cheapest within 2pp of best: {cheap_good['method']} F1 {cheap_good['f1']:.3f} "
              f"at {cheap_good['x_ref']:.2f}x  "
              f"({best['x_ref']/cheap_good['x_ref']:.1f}x cheaper than best)")

    with open(os.path.join(OUT, "budget_frontier.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {OUT}/budget_frontier.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
