#!/usr/bin/env python3
"""Per-benchmark deliverable: coverage, the four dimensions, paired CIs, cost, open problems.

    python3 analyze_benchmark.py gsm8k            # one benchmark
    python3 analyze_benchmark.py --all            # every benchmark with records
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "workspace/controlled_sharing"
RAW = OUT / "raw"
ARMS = ["FULL", "NOPEER", "ANSONLY", "UNIFSEL"]
PROTOS = ["ford", "moa", "du_mad", "madm2_objective", "bccs"]
BOOT = 2000
SEED = 20261019


# ---------------------------------------------------------------- loading
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
        return False
    cell = (r.get("protocol"), r.get("benchmark"), r.get("arm"))
    return cell not in {tuple(x) for x in info.get("invalid_for", ())}


def load(benchmark: str) -> List[Dict[str, Any]]:
    rows, skipped = [], 0
    for p in sorted(RAW.glob(f"{benchmark}__*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not admissible(r):
                skipped += 1
                continue
            rows.append(r)
    if skipped:
        print("   (%d superseded records excluded for %s)" % (skipped, benchmark))
    return rows


def gold_for(benchmark: str) -> Dict[str, Any]:
    """Reference answers, read ONLY here -- never anywhere a prompt is built.
    """
    if benchmark in ("wdc_products", "geographic_settlements", "musicbrainz20k"):
        import collect_all_matching as C
        C._ensure_installed()
        pop, _sha = C.load_population(benchmark)
        out = {}
        for row in pop:
            if isinstance(row, dict):
                k, g = row.get("unit_id"), row.get("gold")
            else:
                k, g = row[0], (row[2] if len(row) > 2 else None)
            if k is None or g is None:
                continue
            # the population's gold is 0/1, while the protocols answer `match`/`non-match`.
            out[k] = "match" if str(g).strip() in ("1", "true", "True", "match") else "non-match"
        return out
    p = ROOT / "data" / "frozen" / f"{benchmark}_block02.json"
    blob = json.loads(p.read_text(encoding="utf-8"))
    recs = blob if isinstance(blob, list) else (blob.get("records") or [])
    out = {}
    for r in recs:
        refs = r.get("references") or []
        g = refs[0] if refs else None
        if g is None:
            continue
        g = str(g).strip()
        if benchmark == "gsm8k" and "####" in g:
            g = g.split("####")[-1].strip()
        out[r.get("example_id")] = g
    return out


def norm(x: Any, kind: str) -> str:
    s = str(x if x is not None else "").strip().lower()
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


# ---------------------------------------------------------------- statistics
def cluster_bootstrap(pairs: List[Tuple[str, float]], reps: int = BOOT,
                      seed: int = SEED) -> Tuple[float, float, float]:
    """Mean paired difference with a unit-clustered bootstrap CI. `pairs` is [(unit, diff)]."""
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    by: Dict[str, List[float]] = collections.defaultdict(list)
    for u, d in pairs:
        by[u].append(d)
    units = list(by)
    point = sum(d for ds in by.values() for d in ds) / sum(len(ds) for ds in by.values())
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        draw = [by[units[rng.randrange(len(units))]] for _ in units]
        flat = [d for ds in draw for d in ds]
        if flat:
            means.append(sum(flat) / len(flat))
    means.sort()
    if len(means) < 20:
        return point, float("nan"), float("nan")
    return point, means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def paired_f1_bootstrap(rows: List[Tuple[str, Optional[str], Optional[str], str]],
                        reps: int = BOOT, seed: int = SEED,
                        ) -> Tuple[float, float, float, float, float]:
    """`(f1_arm, f1_full, diff, lo, hi)` for a paired ΔF1, resampling units.
    """
    by: Dict[str, List[Tuple[Optional[str], Optional[str], str]]] = collections.defaultdict(list)
    for u, a, b, g in rows:
        by[u].append((a, b, g))
    units = list(by)
    if not units:
        return (float("nan"),) * 5

    def f1_of(sel, which: int) -> float:
        tp = fp = fn = 0
        for u in sel:
            for trip in by[u]:
                ans, g = trip[which], trip[2]
                if ans is None:
                    continue
                pred = 1 if ans == "match" else 0
                gold = 1 if g == "match" else 0
                if pred and gold:
                    tp += 1
                elif pred:
                    fp += 1
                elif gold:
                    fn += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    f_arm, f_full = f1_of(units, 0), f1_of(units, 1)
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        sel = [units[rng.randrange(len(units))] for _ in units]
        diffs.append(f1_of(sel, 0) - f1_of(sel, 1))
    diffs.sort()
    if len(diffs) < 20:
        return f_arm, f_full, f_arm - f_full, float("nan"), float("nan")
    return (f_arm, f_full, f_arm - f_full,
            diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))])


# ---------------------------------------------------------------- the report
def analyse(benchmark: str) -> Dict[str, Any]:
    import answer_extract as AX
    rows = load(benchmark)
    if not rows:
        return {"benchmark": benchmark, "records": 0}
    kind = AX.KIND_BY_BENCHMARK.get(benchmark, "binary_match")
    is_matching = benchmark in ("wdc_products", "geographic_settlements", "musicbrainz20k")
    gold = gold_for(benchmark)
    ok = [r for r in rows if r.get("status") == "ok"]
    errs = [r for r in rows if r.get("status") != "ok"]

    # ---- coverage per cell
    cov = collections.Counter((r["protocol"], r["arm"], r["backbone"], r["repeat"])
                              for r in ok)
    err_cov = collections.Counter((r["protocol"], r["arm"]) for r in errs)
    err_class = collections.Counter(r.get("error_class") or "?" for r in errs)
    reused = sum(1 for r in ok if r.get("reused"))

    # ---- delivery and accuracy, per (protocol, condition, backbone)
    perf: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    answers: Dict[Tuple[str, str, str, str, int], Optional[str]] = {}
    for r in ok:
        key = (r["protocol"], r["arm"], r["backbone"])
        d = perf.setdefault(key, {"runs": 0, "delivered": 0, "correct": 0, "scored": 0,
                                  "calls": 0, "out_tok": 0, "in_tok": 0,
                                  "stop_reasons": collections.Counter()})
        d["runs"] += 1
        d["calls"] += r.get("n_calls") or 0
        d["out_tok"] += r.get("completion_tokens") or 0
        d["in_tok"] += r.get("prompt_tokens") or 0
        d["stop_reasons"][(r.get("stop_reason") or "?")[:34]] += 1
        ans = (r.get("record") or {}).get("majority_answer")
        delivered = ans not in (None, "", "None", "none")
        if delivered:
            d["delivered"] += 1
        g = gold.get(r["unit"])
        if delivered and g is not None:
            d["scored"] += 1
            if norm(ans, kind) == norm(g, kind):
                d["correct"] += 1
        answers[(r["protocol"], r["arm"], r["backbone"], r["unit"], r["repeat"])] = (
            norm(ans, kind) if delivered else None)

    # ---- paired contrasts against FULL, on matched (unit, backbone, repeat)
    contrasts = []
    for proto in PROTOS:
        for arm in ("NOPEER", "ANSONLY", "UNIFSEL"):
            for scope in ("pooled",) + tuple(sorted(
                    {bb for (p, a, bb, _u, _r) in answers if p == proto})):
                acc_pairs, dlv_pairs, f1_rows = [], [], []
                for (p, a, bb, u, rep), got in answers.items():
                    if p != proto or a != arm:
                        continue
                    if scope != "pooled" and bb != scope:
                        continue
                    if (proto, "FULL", bb, u, rep) not in answers:
                        continue
                    base = answers.get((proto, "FULL", bb, u, rep))
                    g = gold.get(u)
                    dlv_pairs.append((u, (1.0 if got is not None else 0.0)
                                      - (1.0 if base is not None else 0.0)))
                    if g is None:
                        continue
                    gn = norm(g, kind)
                    if is_matching:
                        f1_rows.append((u, got, base, gn))
                    elif got is not None and base is not None:
                        acc_pairs.append((u, (1.0 if got == gn else 0.0)
                                          - (1.0 if base == gn else 0.0)))
                if not dlv_pairs:
                    continue
                dp, dlo, dhi = cluster_bootstrap(dlv_pairs)
                if is_matching:
                    # A per-unit difference does not exist for a set-level metric, so both arms' F1
                    # are recomputed inside each unit resample.
                    f_arm, f_full, ap, alo, ahi = paired_f1_bootstrap(f1_rows)
                    n_scored = sum(1 for _u, a_, b_, _g in f1_rows
                                   if a_ is not None and b_ is not None)
                else:
                    f_arm = f_full = None
                    ap, alo, ahi = cluster_bootstrap(acc_pairs)
                    n_scored = len(acc_pairs)
                contrasts.append({"protocol": proto, "arm": arm, "scope": scope,
                                  "metric": "F1(match)" if is_matching else "accuracy",
                                  "arm_metric": f_arm, "full_metric": f_full,
                                  "n_pairs_scored": n_scored,
                                  "n_pairs_delivery": len(dlv_pairs),
                                  "acc_diff": ap, "acc_lo": alo, "acc_hi": ahi,
                                  "dlv_diff": dp, "dlv_lo": dlo, "dlv_hi": dhi})

    # ---- the four dimensions, from the traces
    dims = {}
    for proto in PROTOS:
        row = {}
        for arm in ARMS:
            ts = [t for r in ok if r["protocol"] == proto and r["arm"] == arm
                  for t in (r.get("arm_traces") or [])]
            tier = collections.Counter()
            for t in ts:
                for k, v in (t.get("ansonly_tier") or {}).items():
                    tier[k] += v
                tier["failed"] += t.get("ansonly_extract_failed", 0)
            ratios = [t["chars_after"] / t["chars_before"] for t in ts
                      if t.get("chars_before") and t.get("changed")]
            row[arm] = {
                "calls": len(ts),
                "peer_blocks_found": sum(t.get("peer_blocks_found", 0) for t in ts),
                "n_peers_rendered": sum(t.get("n_peers_rendered", 0) for t in ts),
                "ansonly_applied": tier["contract"] + tier["nl"],
                "ansonly_contract": tier["contract"], "ansonly_nl": tier["nl"],
                "ansonly_failed": tier["failed"],
                "mean_size_ratio": (sum(ratios) / len(ratios)) if ratios else None,
                "traces_with_peer_fields": sum(
                    1 for t in ts if "peer_indices_kept" in t and "n_peers_rendered" in t),
                # a UNIFSEL call on a prompt that rendered a single peer has nothing to choose
                # between: the condition keeps that peer and changes nothing.
                "degenerate_calls": sum(
                    1 for t in ts if arm == "UNIFSEL" and not t.get("changed")
                    and t.get("peer_blocks_found", 0) <= 1),
                "calls_with_peers": sum(1 for t in ts if t.get("peer_blocks_found", 0) > 0),
            }
        dims[proto] = row

    # ---- FORD's use contrast, recomputed from the records
    ford_use = [r.get("ford_use_contrast") for r in ok
                if r["protocol"] == "ford" and r.get("ford_use_contrast")]
    use_agree = sum(1 for x in ford_use if x.get("rules_agree"))

    return {"benchmark": benchmark, "kind": kind, "records": len(rows), "ok": len(ok),
            "errors": len(errs), "reused_full": reused,
            "error_classes": dict(err_class), "error_cells": {f"{k[0]}/{k[1]}": v
                                                              for k, v in err_cov.items()},
            "cells_present": len({(p, a, bb, rp) for (p, a, bb, rp) in cov}),
            "coverage": {f"{k[0]}|{k[1]}|{k[2]}|r{k[3]}": v for k, v in sorted(cov.items())},
            "performance": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(perf.items())},
            "contrasts": contrasts, "dimensions": dims,
            "ford_use": {"records": len(ford_use), "rules_agree": use_agree},
            "gold_units_available": len(gold)}


def render(a: Dict[str, Any]) -> str:
    b = a["benchmark"]
    L = [f"# {b}: controlled 5x5x4 deliverable", ""]
    if not a.get("records"):
        return "\n".join(L + ["No records yet."])
    import collect_controlled as CC
    okc, have, need = CC.design_complete(b)
    L += [f"Records {a['records']:,} (ok {a['ok']:,}, errors {a['errors']:,}); "
          f"design {have:,}/{need:,} unit-runs"
          f"{'  **COMPLETE**' if okc else '  (incomplete)'}.",
          f"Answer kind `{a['kind']}`; reference answers available for "
          f"{a['gold_units_available']:,} units. "
          f"{a['reused_full']:,} FULL records reused from the protocol-matrix run.", ""]

    L += ["## Coverage per cell (protocol x arm x backbone x repeat)", "",
          "| protocol | arm | backbone | r1 | r2 |", "|---|---|---|---:|---:|"]
    seen = set()
    for kk in a["coverage"]:
        p, arm, bb, _r = kk.split("|")
        seen.add((p, arm, bb))
    for p in PROTOS:
        for arm in ARMS:
            for bb in sorted({x[2] for x in seen if x[0] == p and x[1] == arm}):
                r1 = a["coverage"].get(f"{p}|{arm}|{bb}|r1", 0)
                r2 = a["coverage"].get(f"{p}|{arm}|{bb}|r2", 0)
                L.append(f"| {p} | {arm} | {bb} | {r1:,} | {r2:,} |")
    if a["errors"]:
        L += ["", f"**Errors: {a['errors']:,}**, kept with their true classification "
                  f"(never recounted as not-executed): "
                  + ", ".join(f"`{k}` {v:,}" for k, v in a["error_classes"].items()),
              "", "Affected cells: " + ", ".join(f"{k} ({v})"
                                                 for k, v in a["error_cells"].items())]

    L += ["", "## Quality and delivery (separate denominators)", "",
          "| protocol | arm | backbone | runs | delivered | accuracy on delivered |",
          "|---|---|---|---:|---:|---:|"]
    for kk, d in a["performance"].items():
        p, arm, bb = kk.split("|")
        acc = (d["correct"] / d["scored"]) if d["scored"] else float("nan")
        L.append(f"| {p} | {arm} | {bb} | {d['runs']:,} | {d['delivered']:,}/{d['runs']:,} "
                 f"({100*d['delivered']/max(d['runs'],1):.1f}%) | "
                 + (f"{acc:.4f} (n={d['scored']:,})" if d["scored"] else "n/a") + " |")

    L += ["", "## Paired contrasts against FULL, unit-clustered 95% CI", "",
          "Pairing is on (unit, backbone, repeat). A CI containing zero means *not distinguished "
          "at this sample size* -- not equivalence: no margin was pre-specified.", "",
          "| protocol | arm | scope | n pairs | metric | arm vs FULL | difference [95% CI] | "
          "delivery diff [95% CI] |", "|---|---|---|---:|---|---|---|---|"]
    for c in a["contrasts"]:
        ad = ("%+.4f [%+.4f, %+.4f]" % (c["acc_diff"], c["acc_lo"], c["acc_hi"])
              if c["n_pairs_scored"] and not math.isnan(c["acc_diff"]) else "n/a")
        dd = ("%+.4f [%+.4f, %+.4f]" % (c["dlv_diff"], c["dlv_lo"], c["dlv_hi"])
              if not math.isnan(c["dlv_diff"]) else "n/a")
        lvl = (f"{c['arm_metric']:.4f} vs {c['full_metric']:.4f}"
               if c.get("arm_metric") is not None else "—")
        L.append(f"| {c['protocol']} | {c['arm']} | {c['scope']} | "
                 f"{c['n_pairs_scored']:,}/{c['n_pairs_delivery']:,} | "
                 f"{c.get('metric', 'accuracy')} | {lvl} | {ad} | {dd} |")

    L += ["", "## The four dimensions, from the recorded traces", "",
          "| protocol | arm | calls | peer blocks | peers rendered | ANSONLY applied "
          "(contract/nl) | ANSONLY failed | mean size ratio |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for p, row in a["dimensions"].items():
        for arm in ARMS:
            d = row.get(arm) or {}
            if not d.get("calls"):
                continue
            sr = d.get("mean_size_ratio")
            L.append(f"| {p} | {arm} | {d['calls']:,} | {d['peer_blocks_found']:,} | "
                     f"{d['n_peers_rendered']:,} | {d['ansonly_applied']:,} "
                     f"({d['ansonly_contract']:,}/{d['ansonly_nl']:,}) | "
                     f"{d['ansonly_failed']:,} | " + (f"{sr:.3f}" if sr else "-") + " |")
    L += ["", "Access is read as *peers rendered*: NOPEER must be 0 where FULL is positive. "
              "Content is read as *ANSONLY applied* against *failed*: a failed extraction shipped "
              "the peer's full text, so those calls are ANSONLY in name and FULL in substance and "
              "are excluded from the content contrast.", "",
          "**What UNIFSEL is, and is not.** It keeps a uniform random half of the peers the "
          "released prompt rendered (at least one). That changes *how many* peers are present as "
          "well as *which*, so it is the middle point of an access-volume series -- FULL, half, "
          "none -- and **not** a which-peers selection contrast. A which-peers contrast cannot be "
          "built by message surgery at all: the prompt contains only the peers the protocol "
          "already chose, so the rejected ones are not available to substitute in. The "
          "which-peers question is instead answered by each protocol's own released selection "
          "rule, and for MAD-M^2 that evidence already exists as the objective-versus-naive "
          "pruning pair in the protocol-matrix run (see `REUSE_MANIFEST.md`); FORD's counterpart "
          "is its final-answer rule, reported below.", ""]
    degen = [(p, row["UNIFSEL"]) for p, row in a["dimensions"].items()
             if (row.get("UNIFSEL") or {}).get("calls")
             and row["UNIFSEL"]["n_peers_rendered"] >= row["UNIFSEL"]["peer_blocks_found"]]
    if degen:
        L += ["**UNIFSEL is degenerate for " + ", ".join(p for p, _ in degen) + "** on this "
              "benchmark: the released prompt renders a single peer per call, so a uniform half is "
              "that same peer and the arm is a no-op. Those cells are reported as degenerate, not "
              "as an intervention that made no difference.", ""]

    L += ["", "## Cost", "", "| protocol | arm | calls/unit | output tok/unit | input tok/unit |",
          "|---|---|---:|---:|---:|"]
    for kk, d in a["performance"].items():
        p, arm, bb = kk.split("|")
        if bb != sorted({x.split("|")[2] for x in a["performance"]})[0] and p not in ("ford", "moa"):
            pass
        n = max(d["runs"], 1)
        L.append(f"| {p} | {arm} ({bb}) | {d['calls']/n:.1f} | {d['out_tok']/n:.0f} | "
                 f"{d['in_tok']/n:.0f} |")

    fu = a["ford_use"]
    if fu["records"]:
        L += ["", "## FORD's selection/use contrast", "",
              f"FORD renders one opponent, so there is no peer to choose between; its "
              f"selection/use factor is the final-answer rule. On {fu['records']:,} records the "
              f"released two-consecutive-agreement rule and the all-round majority agree on "
              f"{fu['rules_agree']:,} ({100*fu['rules_agree']/max(fu['records'],1):.1f}%)."]

    L += ["", "## Open problems", ""]
    opens = []
    for p, row in a["dimensions"].items():
        d = row.get("ANSONLY") or {}
        tot = d.get("ansonly_applied", 0) + d.get("ansonly_failed", 0)
        if tot and d.get("ansonly_failed", 0) / tot > 0.05:
            opens.append(f"- {p}/ANSONLY: {d['ansonly_failed']:,} of {tot:,} peer blocks "
                         f"({100*d['ansonly_failed']/tot:.1f}%) carried no readable answer, so "
                         f"those calls were not reduced and are excluded from the content contrast.")
    for kk, d in a["performance"].items():
        if d["runs"] and d["delivered"] / d["runs"] < 0.8:
            opens.append(f"- {kk}: delivery {d['delivered']:,}/{d['runs']:,} "
                         f"({100*d['delivered']/d['runs']:.0f}%); top stop reasons "
                         + ", ".join(f"`{k}` {v}" for k, v in d["stop_reasons"].most_common(2)))
    if a["errors"]:
        opens.append(f"- {a['errors']:,} unit-runs failed and are counted as errors, not as "
                     f"missing coverage.")
    L += opens or ["- none recorded."]
    L += ["", "## Provenance", "",
          f"- raw records: `data/records/controlled_sharing/{b}__*.jsonl` "
          f"(append-only, full input and output per call)",
          f"- frozen sample: `results/{'protocol_all_matching_20261018T192619Z'}"
          f"/analysis/FROZEN_SAMPLE_5X5.json`",
          "- reused FULL records carry `reused: true` with their source run, shard and unit id; "
          "their FULL-arm traces were reconstructed offline from the preserved prompts",
          "- STOP entry: `python3 stop_control.py stop`"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmarks", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    names = a.benchmarks
    if a.all or not names:
        names = [p.name.split("__")[0] for p in RAW.glob("*.jsonl")]
        names = sorted(set(names))
    for b in names:
        res = analyse(b)
        d = OUT / b
        d.mkdir(parents=True, exist_ok=True)
        (d / "REPORT.md").write_text(render(res), encoding="utf-8")
        (d / "report.json").write_text(json.dumps(res, indent=1, default=str) + "\n",
                                       encoding="utf-8")
        print("%-24s records %7s ok %7s errors %5s -> %s"
              % (b, f"{res.get('records', 0):,}", f"{res.get('ok', 0):,}",
                 f"{res.get('errors', 0):,}", (d / 'REPORT.md')))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
