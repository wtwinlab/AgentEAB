#!/usr/bin/env python3
"""The four research questions, measured with the paper's own estimators. Zero model calls.

    python3 analyze_research_questions.py --benchmarks gsm8k strategyqa mmlu_pro
    python3 analyze_research_questions.py --all
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "analysis/controlled_sharing"
RAW = ROOT / "data/records/controlled_sharing"
PROTOS = ["ford", "moa", "du_mad", "madm2_objective", "bccs"]
ARMS = ["FULL", "NOPEER", "ANSONLY", "UNIFSEL"]
MATCHING = ("wdc_products", "geographic_settlements", "musicbrainz20k")
BOOT = 5000    # 1/B = 2e-4, below the 0.05/161 = 3.11e-4 the strictest Holm step of the combined
               # all-cells family needs (and below the 0.05/115 the main matrix alone needed).
               # 20000 was tried first and was pure waste: 4x the work for resolution nobody uses,
               # and at the nice level that keeps the collection ahead of it the job got no CPU.
SEED = 20261019


# ---------------------------------------------------------------- outcome helpers
def norm(x: Any, kind: str) -> Optional[str]:
    s = str(x if x is not None else "").strip().lower()
    if s in ("", "none", "null"):
        return None
    if kind == "numeric":
        s = s.replace(",", "").replace("$", "").rstrip(".")
        try:
            return str(round(float(s), 4))
        except Exception:
            return s
    if kind == "boolean":
        return {"true": "yes", "false": "no"}.get(s, s)
    if kind == "mcq":
        return s.strip("().")
    return s.replace("_", "-")


def gold_for(benchmark: str) -> Dict[str, str]:
    if benchmark in MATCHING:
        import collect_all_matching as C
        C._ensure_installed()
        rows, _ = C.load_population(benchmark)
        out = {}
        for row in rows:
            k = row.get("unit_id") if isinstance(row, dict) else row[0]
            g = row.get("gold") if isinstance(row, dict) else (row[2] if len(row) > 2 else None)
            if k is None or g is None:
                continue
            out[k] = "match" if str(g).strip() in ("1", "true", "True", "match") else "non-match"
        return out
    blob = json.loads((ROOT / "data" / "frozen" / f"{benchmark}_block02.json")
                      .read_text(encoding="utf-8"))
    recs = blob if isinstance(blob, list) else (blob.get("records") or [])
    out = {}
    for r in recs:
        refs = r.get("references") or []
        if not refs:
            continue
        g = str(refs[0]).strip()
        if benchmark == "gsm8k" and "####" in g:
            g = g.split("####")[-1].strip()
        out[r.get("example_id")] = g
    return out


def _boot_p(vals: List[float]) -> float:
    """Two-sided bootstrap p-value: twice the smaller tail mass on the far side of zero.
    """
    if not vals:
        return float("nan")
    n = len(vals)
    below = sum(1 for v in vals if v < 0)
    above = sum(1 for v in vals if v > 0)
    if below == 0 and above == 0:
        return 1.0                    # no variation at all: no evidence of a difference
    return max(min(1.0, 2.0 * min(below, above) / n), 1.0 / n)


def holm(pvals: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down: returns, per input position, whether it is rejected at `alpha`."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    out = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break                      # step-down: once one fails, all larger p-values fail
    return out


# ---------------------------------------------------------------- the paper's estimators
def paired_effect(by_unit: Dict[str, Dict[str, List[float]]],
                  reps: int = BOOT) -> Dict[str, Any]:
    """Eq. (paired-effect): average repeats within a unit, then average the per-unit differences."""
    diffs = []
    for u, cond in by_unit.items():
        if "1" not in cond or "0" not in cond or not cond["1"] or not cond["0"]:
            continue
        diffs.append((u, sum(cond["1"]) / len(cond["1"]) - sum(cond["0"]) / len(cond["0"])))
    if not diffs:
        return {"n_units": 0, "effect": float("nan"), "lo": float("nan"), "hi": float("nan")}
    point = sum(d for _u, d in diffs) / len(diffs)
    rng = random.Random(SEED)
    means = []
    for _ in range(reps):
        s = [diffs[rng.randrange(len(diffs))][1] for _ in diffs]
        means.append(sum(s) / len(s))
    means.sort()
    return {"n_units": len(diffs), "effect": point,
            "lo": means[int(0.025 * len(means))], "hi": means[int(0.975 * len(means))],
            "p_boot": _boot_p(means)}


def paired_f1(rows: List[Tuple[str, Optional[str], Optional[str], str]],
              charge: bool, reps: int = BOOT) -> Dict[str, Any]:
    """ΔF1 on the match class, recomputing both conditions inside each unit resample."""
    by: Dict[str, List[Tuple[Optional[str], Optional[str], str]]] = collections.defaultdict(list)
    for u, a, b, g in rows:
        by[u].append((a, b, g))
    units = list(by)
    if not units:
        return {"n_units": 0, "f1_1": float("nan"), "f1_0": float("nan"),
                "effect": float("nan"), "lo": float("nan"), "hi": float("nan")}

    def f1(sel, which: int) -> float:
        tp = fp = fn = 0
        for u in sel:
            for trip in by[u]:
                ans, g = trip[which], trip[2]
                if ans is None:
                    if not charge:
                        continue
                    ans = "non-match"      # charged: abstention predicts the negative class
                pred, gold = (1 if ans == "match" else 0), (1 if g == "match" else 0)
                if pred and gold:
                    tp += 1
                elif pred:
                    fp += 1
                elif gold:
                    fn += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    f1_1, f1_0 = f1(units, 0), f1(units, 1)
    rng = random.Random(SEED)
    ds = []
    for _ in range(reps):
        sel = [units[rng.randrange(len(units))] for _ in units]
        ds.append(f1(sel, 0) - f1(sel, 1))
    ds.sort()
    return {"n_units": len(units), "f1_1": f1_1, "f1_0": f1_0, "effect": f1_1 - f1_0,
            "lo": ds[int(0.025 * len(ds))], "hi": ds[int(0.975 * len(ds))],
            "p_boot": _boot_p(ds)}


def rate_ci(num: int, den: int, reps: int = BOOT) -> Tuple[float, float, float]:
    if not den:
        return float("nan"), float("nan"), float("nan")
    p = num / den
    rng = random.Random(SEED)
    xs = []
    for _ in range(reps):
        xs.append(sum(1 for _ in range(den) if rng.random() < p) / den)
    xs.sort()
    return p, xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs))]


# ---------------------------------------------------------------- record reading
def admissible(r: Dict[str, Any]) -> bool:
    """Is this record valid under the code version that produced it?
    """
    import collect_controlled as CC
    cv = r.get("code_version")
    if cv == CC.CODE_VERSION:
        return True
    cell = (r.get("protocol"), r.get("benchmark"), r.get("arm"))
    # the mechanism table is the single source of truth for what a version got wrong
    if cell in CC.contaminated_cells(cv):
        return False
    info = CC.CODE_EQUIVALENCE.get(cv)
    if info is None:
        return False                      # an unknown version is not silently trusted
    cell = (r.get("protocol"), r.get("benchmark"), r.get("arm"))
    return cell not in {tuple(x) for x in info.get("invalid_for", ())}


def read(benchmark: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows, drop = [], collections.Counter()
    for p in sorted(RAW.glob(f"{benchmark}__*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                drop["unparseable"] += 1
                continue
            if r.get("status") != "ok":
                drop["not ok"] += 1
                continue
            if not admissible(r):
                drop["superseded by a repaired version: %s/%s/%s"
                     % (r.get("protocol"), r.get("benchmark"), r.get("arm"))] += 1
                continue
            rows.append(r)
    return rows, dict(drop)


def features(r: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """Everything the diagnostics need, pulled from one record."""
    rec = r.get("record") or {}
    calls = rec.get("calls") or []
    final = norm(rec.get("majority_answer"), kind)
    cands = [norm(a, kind) for a in (rec.get("per_agent_final_answer") or [])]
    # initial answers: round 0 for the round-based protocols; FORD/MoA have no round field, so the
    # first call of each agent is its initial answer
    init: Dict[Any, Optional[str]] = {}
    post: Dict[Any, Optional[str]] = {}
    for c in calls:
        ai = c.get("agent_index")
        if ai is None:
            continue  # downstream aggregator is not an initial agent answer
        rd = c.get("round")
        if rd is None:
            rd = c.get("turn")
        a = norm(c.get("extracted_answer") if c.get("extracted_answer") is not None
                 else c.get("prediction"), kind)
        if ai not in init or (rd is not None and rd == 0):
            init.setdefault(ai, a)
        if rd == 0:
            init[ai] = a
        post[ai] = a                      # last call of this agent wins
    ops = collections.Counter()
    for c in calls:
        role = c.get("role") or "generation"
        ops[("aggregation" if "agg" in str(role).lower() else "generation")] += 1
    return {"final": final, "candidates": cands, "initial": init, "post": post,
            "n_calls": r.get("n_calls") or len(calls),
            "in_tok": r.get("prompt_tokens") or 0, "out_tok": r.get("completion_tokens") or 0,
            "ops": dict(ops),
            "stop_reason": r.get("stop_reason"),
            "rule_satisfied": rec.get("rule_satisfied"),
            "absence_cause": rec.get("absence_cause"),
            "accepted_delivery": rec.get("accepted_delivery")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    import inputs as IN
    names = a.benchmarks
    if a.all or not names:
        names = list(IN.BENCHMARKS)
    OUT.mkdir(parents=True, exist_ok=True)
    print("input registry: %s\n" % IN.ext_gate(require=True))

    effects, diagnostics, costs = [], [], []
    for bench in names:
        bsrc = IN.source_for(bench)
        kind = IN.kind_for(bench)
        is_match = IN.is_matching(bench)
        gold = IN.gold_for(bench)
        rows, dropped = IN.read(bench)
        if not rows:
            continue
        if dropped:
            print("  excluded from this benchmark: %s" % dropped)
        F: Dict[Tuple[str, str, str, str, int], Dict[str, Any]] = {}
        for r in rows:
            F[(r["protocol"], r["arm"], r["backbone"], r["unit"], r["repeat"])] = \
                features(r, kind)
        print(f"\n########## {bench}  ({len(rows):,} ok records, kind={kind}) ##########")

        for proto in PROTOS:
            backbones = sorted({k[2] for k in F if k[0] == proto})
            for bb in backbones:
                base = {(k[3], k[4]): v for k, v in F.items()
                        if k[0] == proto and k[1] == "FULL" and k[2] == bb}
                if not base:
                    continue
                # ---- cost profile, per condition (whole-workflow: every call is charged)
                for arm in ARMS:
                    cur = {(k[3], k[4]): v for k, v in F.items()
                           if k[0] == proto and k[1] == arm and k[2] == bb}
                    if not cur:
                        continue
                    n = len(cur)
                    ops = collections.Counter()
                    for v in cur.values():
                        for o, c in v["ops"].items():
                            ops[o] += c
                    costs.append({"source": bsrc.tag, "benchmark": bench, "protocol": proto,
                                  "backbone": bb, "arm": arm, "runs": n,
                                  "calls_per_unit": sum(v["n_calls"] for v in cur.values()) / n,
                                  "in_tok_per_unit": sum(v["in_tok"] for v in cur.values()) / n,
                                  "out_tok_per_unit": sum(v["out_tok"] for v in cur.values()) / n,
                                  "generation_calls_per_unit": ops["generation"] / n,
                                  "aggregation_calls_per_unit": ops["aggregation"] / n})

                for arm in ("NOPEER", "ANSONLY", "UNIFSEL"):
                    cur = {(k[3], k[4]): v for k, v in F.items()
                           if k[0] == proto and k[1] == arm and k[2] == bb}
                    if not cur:
                        continue
                    shared = sorted(set(cur) & set(base))
                    if not shared:
                        continue
                    # ---- effect, both conventions
                    for charge in (True, False):
                        if is_match:
                            rr = [(u, cur[(u, rp)]["final"], base[(u, rp)]["final"], gold[u])
                                  for (u, rp) in shared if u in gold]
                            est = paired_f1(rr, charge=charge)
                            metric = IN.METRIC[bench]
                        else:
                            bu: Dict[str, Dict[str, List[float]]] = collections.defaultdict(
                                lambda: {"0": [], "1": []})
                            for (u, rp) in shared:
                                g = gold.get(u)
                                if g is None:
                                    continue
                                gn = norm(g, kind)
                                for tag, src in (("1", cur), ("0", base)):
                                    f = src[(u, rp)]["final"]
                                    if f is None and not charge:
                                        continue
                                    bu[u][tag].append(1.0 if (f is not None and f == gn) else 0.0)
                            est = paired_effect(bu)
                            est["f1_1"] = est["f1_0"] = None
                            metric = IN.METRIC[bench]
                        # Delivery denominators, kept beside every effect.
                        d_arm = sum(1 for k in shared if cur[k]["final"] is not None)
                        d_full = sum(1 for k in shared if base[k]["final"] is not None)
                        effects.append({
                            "source": bsrc.tag,
                            "benchmark": bench, "protocol": proto, "backbone": bb,
                            "contrast": f"{arm} vs FULL", "dimension":
                                {"NOPEER": "access", "ANSONLY": "content",
                                 "UNIFSEL": "selection(volume)"}[arm],
                            "metric": metric,
                            "convention": "abstention charged as failure" if charge
                                          else "delivered only (optimistic bound)",
                            "n_units": est["n_units"], "level_arm": est.get("f1_1"),
                            "level_full": est.get("f1_0"),
                            "effect": est["effect"], "lo": est["lo"], "hi": est["hi"],
                            "p_boot": est.get("p_boot"),
                            "n_shared_unit_runs": len(shared),
                            "n_delivered_arm": d_arm, "n_delivered_full": d_full,
                            "degenerate_neither_delivered": (d_arm == 0 and d_full == 0)})

                    # Diagnostics are reduced separately from raw outputs; see diagnostics_clustered.py.

        # a compact console view of the access answer, which is the paper's first question
        print("  access (NOPEER vs FULL), abstention charged:")
        for e in effects:
            if (e["benchmark"] == bench and e["dimension"] == "access"
                    and e["convention"].startswith("abstention")):
                lvl = (f" [{e['level_arm']:.4f} vs {e['level_full']:.4f}]"
                       if e["level_arm"] is not None else "")
                print("     %-18s %-12s n=%-5d %-10s %+.4f [%+.4f, %+.4f]%s"
                      % (e["protocol"], e["backbone"], e["n_units"], e["metric"],
                         e["effect"], e["lo"], e["hi"], lvl))

    # ---- multiple comparisons, in two layers that must not be confused for each other.
    #**Prior families.** The controlled matrix declared its families before its results were seen:
    #one per (dimension, convention), plus a stricter all-cells family.
    for scope in ("prior", "combined"):
        for fam_key in ("dimension", "all"):
            groups: Dict[Any, List[int]] = collections.defaultdict(list)
            for i, e in enumerate(effects):
                k: Tuple[Any, ...] = (e["convention"],)
                if fam_key == "dimension":
                    k = k + (e["dimension"],)
                if scope == "prior":
                    k = k + (e["source"],)
                groups[k].append(i)
            for k, idx in groups.items():
                ps = [effects[i].get("p_boot") or 1.0 for i in idx]
                rej = holm(ps)
                for j, i in enumerate(idx):
                    effects[i]["holm_%s_%s_family_size" % (scope, fam_key)] = len(idx)
                    effects[i]["holm_%s_%s_reject" % (scope, fam_key)] = rej[j]
    for e in effects:
        e["prior_family"] = ("main matrix, pre-declared" if e["source"] == "main"
                             else "benchmark extension, pre-declared")
        # keep the historical column names working for anything that still reads them, pointing at
        # the pre-declared decision rather than silently at the new combined one
        e["holm_dimension_family_size"] = e["holm_prior_dimension_family_size"]
        e["holm_dimension_reject"] = e["holm_prior_dimension_reject"]
        e["holm_all_family_size"] = e["holm_prior_all_family_size"]
        e["holm_all_reject"] = e["holm_prior_all_reject"]

    # ---- what the families actually came out to, printed so the counts are never inferred
    print("\n=== comparison families (charged convention; the delivered-only bound is its own "
          "family) ===")
    ch = [e for e in effects if e["convention"].startswith("abstention")]
    print("%-28s %-20s %6s %10s %12s %12s" % ("family", "dimension", "tests", "CI excl 0",
                                              "Holm reject", "family size"))
    for scope in ("prior", "combined"):
        for dim in ("access", "content", "selection(volume)"):
            for srctag in (("main", "ext") if scope == "prior" else (None,)):
                sel = [e for e in ch if e["dimension"] == dim
                       and (srctag is None or e["source"] == srctag)]
                if not sel:
                    continue
                ci = sum(1 for e in sel if e["lo"] > 0 or e["hi"] < 0)
                rj = sum(1 for e in sel if e["holm_%s_dimension_reject" % scope])
                print("%-28s %-20s %6d %10d %12d %12d"
                      % ("%s%s" % (scope, "/" + srctag if srctag else ""), dim, len(sel), ci, rj,
                         sel[0]["holm_%s_dimension_family_size" % scope]))
    for scope in ("prior", "combined"):
        for srctag in (("main", "ext") if scope == "prior" else (None,)):
            sel = [e for e in ch if srctag is None or e["source"] == srctag]
            rj = sum(1 for e in sel if e["holm_%s_all_reject" % scope])
            print("%-28s %-20s %6d %10d %12d %12d"
                  % ("%s%s" % (scope, "/" + srctag if srctag else ""), "ALL (strictest)",
                     len(sel), sum(1 for e in sel if e["lo"] > 0 or e["hi"] < 0), rj,
                     sel[0]["holm_%s_all_family_size" % scope]))

    deg = [e for e in effects if e.get("degenerate_neither_delivered")
           and e["convention"].startswith("abstention")]
    if deg:
        print("\n=== degenerate contrasts: NEITHER condition ever delivered ===")
        print("a zero-width interval here is two total failures, not equality")
        for e in deg:
            print("   %-18s %-14s %-16s %-11s delivered %d vs %d of %d unit-runs"
                  % (e["dimension"], e["benchmark"], e["protocol"], e["backbone"],
                     e["n_delivered_arm"], e["n_delivered_full"], e["n_shared_unit_runs"]))

    for name, data in (("rq_effects", effects), 
                       ("rq_cost", costs)):
        if not data:
            continue
        keys: List[str] = []
        for d in data:
            for k in d:
                if k not in keys:
                    keys.append(k)
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
    blob = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "estimators": {"effect": "paper Eq. paired-effect, repeats averaged within unit",
                           "f1": "recomputed per condition inside a unit bootstrap",
                           "primary_convention": "non-delivery counts as failure",
                           "bootstrap": BOOT, "seed": SEED},
            "sign_convention": "`effect` is mean(arm - FULL); the report and cost tables present "
                               "FULL - arm, converted once in analyze_cost_tradeoff.py",
            "metrics": {"note": "QA accuracy, pair F1(match) and SQL execution accuracy are "
                                "different quantities on different populations; effects are "
                                "compared within a metric, never pooled across metrics",
                        "by_benchmark": dict(IN.METRIC)},
            "families": {"prior": "pre-declared per collection: main matrix 40/40/35 = 115, "
                                  "benchmark extension 16/16/14 = 46; preserved unchanged",
                         "combined": "post-hoc supplementary Holm over all 161 comparisons, "
                                     "decided after both result sets had been seen and labelled "
                                     "as such throughout"},
            "sql_answer_labelling": getattr(IN._sqlify, "last_stats", None),
            "sql_labelling_note": "every SQL answer is replaced at read time by its execution "
                                  "verdict; 'not_delivered' had no query to run, "
                                  "'gold_failed_or_unscorable' is a unit whose own reference query "
                                  "failed on this instance and is scored neither way, 'no_verdict' "
                                  "would mean the verdict table is stale",
            "degenerate_contrasts": [
                {k: e[k] for k in ("source", "benchmark", "protocol", "backbone", "dimension",
                                   "contrast", "convention", "n_delivered_arm",
                                   "n_delivered_full", "effect", "lo", "hi")}
                for e in effects if e.get("degenerate_neither_delivered")],
            "benchmarks": names, "effects": effects, "diagnostics_source": "analysis/candidate_diagnostics/rates.csv", "cost": costs}
    (OUT / "rq_results.json").write_text(json.dumps(blob, indent=1, default=str) + "\n",
                                         encoding="utf-8")
    print("\n-> %s  (%d effects, %d diagnostics, %d cost rows)"
          % (OUT / "rq_results.json", len(effects), len(diagnostics), len(costs)))
    print("   sha256 %s" % hashlib.sha256((OUT / "rq_results.json").read_bytes()
                                          ).hexdigest()[:32])
    import diagnostics_clustered
    diagnostics_clustered.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
