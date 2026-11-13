#!/usr/bin/env python3
"""Per-benchmark deliverable for the extension, computed with the paper's own estimators.

    python3 ext_analyze.py gpqa
    python3 ext_analyze.py --all

The estimators are imported, not re-implemented
-----------------------------------------------
`analyze_research_questions.py` is where the paper's estimators live -- the paired effect with
repeats averaged *inside* the unit, F1 recomputed per condition inside a unit-clustered bootstrap,
the charged-abstention convention as the headline with the delivered-only figure beside it as the
optimistic bound, and the four diagnostics (correction, regression, availability loss, selection
miss) with their stated denominators. `analyze_benchmark.py` is where the per-cell coverage,
delivery/accuracy split and the four-dimension trace table live. Both are imported and pointed at
this run's tree. Re-deriving them here would be a second implementation of the paper's statistics,
which is how two numbers for one quantity appear in one paper.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE, ROOT / "experiments/06_cost_resources/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ext_collect as EC                                                 # noqa: E402
import ext_design as D                                                   # noqa: E402

OUT = D.OUT
RAW = OUT / "raw"
ANALYSIS = OUT / "analysis"


def _ext_gold(benchmark: str) -> Dict[str, str]:
    """Reference answers, read ONLY here.
    """
    if benchmark in D.SQL_BENCH:
        return {u: "correct" for u in D.frozen(benchmark)["unit_ids"]}
    return D.gpqa_gold()


_VERDICTS: Dict[str, Dict[str, Any]] = {}


def _load_verdicts() -> Dict[str, Dict[str, Any]]:
    global _VERDICTS
    if _VERDICTS:
        return _VERDICTS
    p = ANALYSIS / "sql_verdicts.json"
    if not p.exists():
        raise SystemExit(
            f"{p} is missing. The SQL analysis needs an execution verdict per query; run "
            f"`python3 ext_sql_verdicts.py` first. Without it the only computable 'accuracy' is "
            f"string equality with the reference query, which is not the benchmark's notion of "
            f"correctness and must not be reported as if it were.")
    _VERDICTS = json.loads(p.read_text(encoding="utf-8"))["verdicts"]
    return _VERDICTS


def _sqlify(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rewrite every SQL answer in memory as its execution verdict, in place of the query text.
    """
    import sql_tasks as ST
    import ext_sql_verdicts as V
    verdicts = _load_verdicts()
    hit = miss = 0

    def label(db: str, a: Any) -> Optional[str]:
        nonlocal hit, miss
        s = ST.unwrap_sql(a) if a not in (None, "", "None", "none") else ""
        if not s:
            return None
        v = verdicts.get(V.qkey(db, s))
        if v is None:
            miss += 1
            return None
        hit += 1
        c = v.get("correct")
        return None if c is None else ("correct" if c else "incorrect")

    out = []
    for r in rows:
        r = dict(r)
        db = ((r.get("config") or {}).get("unit_meta") or {}).get("db_id")
        rec = dict(r.get("record") or {})
        if db:
            rec["majority_answer"] = label(db, rec.get("majority_answer"))
            if rec.get("per_agent_final_answer") is not None:
                rec["per_agent_final_answer"] = [label(db, a)
                                                 for a in rec["per_agent_final_answer"]]
            calls = []
            for c in rec.get("calls") or []:
                c = dict(c)
                for f in ("extracted_answer", "prediction", "normalised_answer"):
                    if c.get(f) is not None:
                        c[f] = label(db, c[f])
                        break
                calls.append(c)
            rec["calls"] = calls
        r["record"] = rec
        out.append(r)
    if miss:
        print("   (%d SQL answers had no execution verdict and are treated as non-delivered; "
              "re-run ext_sql_verdicts.py if the collection grew since)" % miss)
    return out


def _patch() -> None:
    """Point the two imported analyses at this run's tree, design and admissibility rule."""
    import analyze_benchmark as AB
    import analyze_research_questions as ARQ
    # registers `KIND_BY_BENCHMARK["sql_bird_v2"] = "sql"`.
    import ext_sql_tasks as EST
    EST.install()

    AB.RAW = RAW
    AB.OUT = OUT
    AB.gold_for = _ext_gold
    ARQ.RAW = RAW
    ARQ.OUT = ANALYSIS
    ARQ.gold_for = _ext_gold
    ARQ.MATCHING = ()                       # neither extension benchmark is a matching population

    def admissible(r: Dict[str, Any]) -> bool:
        """Valid under the code version that produced it, using THIS run's equivalence table.
        """
        cv = r.get("code_version")
        if cv == EC.CODE_VERSION:
            return True
        info = EC.CODE_EQUIVALENCE.get(cv)
        if info is None:
            return False
        cell = (r.get("protocol"), r.get("benchmark"), r.get("arm"))
        return cell not in {tuple(x) for x in info.get("invalid_for", ())}

    ARQ.admissible = admissible
    # `analyze_benchmark` applies the same version filter with the main collection's digest
    # (`collect_controlled.CODE_VERSION`); the extension's admissibility rule is passed to it here.
    if hasattr(AB, "admissible"):
        AB.admissible = admissible

    # For SQL, both loaders hand back records whose answers are execution verdicts rather than
    # query text, so every estimator and diagnostic downstream measures correctness on the
    # database instead of string equality with one spelling of the reference.
    _ab_load, _arq_read = AB.load, ARQ.read

    def ab_load(benchmark: str):
        rows = _ab_load(benchmark)
        return _sqlify(rows) if benchmark in D.SQL_BENCH else rows

    def arq_read(benchmark: str):
        rows, drop = _arq_read(benchmark)
        return (_sqlify(rows) if benchmark in D.SQL_BENCH else rows), drop

    AB.load = ab_load
    ARQ.read = arq_read
    return AB, ARQ


# ---------------------------------------------------------------- the extension's own additions
def read_ok(benchmark: str) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted(RAW.glob(f"{benchmark}__*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def excluded_records(benchmark: str) -> Dict[str, Any]:
    """Records on disk that the analysis does not read, counted and named rather than vanishing.
    """
    by: Dict[str, int] = collections.Counter()
    versions: Dict[Any, int] = collections.Counter()
    for r in read_ok(benchmark):
        cv = r.get("code_version")
        versions[cv] += 1
        if r.get("status") != "ok":
            by["failed unit-run (kept as evidence, re-collected under its own key)"] += 1
        elif cv != EC.CODE_VERSION and cv not in EC.CODE_EQUIVALENCE:
            by[f"produced by intervention code {cv!r}, which is not the current version and has "
               f"no recorded equivalence"] += 1
    return {"excluded": dict(by), "code_versions_on_disk": {str(k): v
                                                            for k, v in versions.items()},
            "current_code_version": EC.CODE_VERSION}


def truncation_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per cell: calls, truncated calls, and the share of unit-runs with any truncation.
    """
    agg: Dict[Any, Dict[str, int]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        k = (r["protocol"], r["arm"], r["backbone"])
        d = agg.setdefault(k, {"runs": 0, "calls": 0, "truncated": 0, "runs_with_trunc": 0,
                               "delivered": 0})
        t = r.get("truncated_calls") or 0
        d["runs"] += 1
        d["calls"] += r.get("n_calls") or 0
        d["truncated"] += t
        d["runs_with_trunc"] += 1 if t else 0
        ans = (r.get("record") or {}).get("majority_answer")
        d["delivered"] += 1 if ans not in (None, "", "None", "none") else 0
    out = []
    for (p, a, bb), d in sorted(agg.items()):
        out.append({"protocol": p, "arm": a, "backbone": bb, **d,
                    "truncated_call_rate": round(d["truncated"] / max(d["calls"], 1), 5),
                    "unit_runs_with_any_truncation": round(
                        d["runs_with_trunc"] / max(d["runs"], 1), 5),
                    "delivery_rate": round(d["delivered"] / max(d["runs"], 1), 5)})
    return out


def operation_cost_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cost by operation type, per cell, from each record's own `ops` / `ops_output_tokens`."""
    agg: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        k = (r["protocol"], r["arm"], r["backbone"])
        d = agg.setdefault(k, {"runs": 0, "calls": collections.Counter(),
                               "out": collections.Counter(),
                               "in_tok": 0, "out_tok": 0, "seconds": 0.0})
        d["runs"] += 1
        for op, n in (r.get("ops") or {}).items():
            d["calls"][op] += n
        for op, n in (r.get("ops_output_tokens") or {}).items():
            d["out"][op] += n
        d["in_tok"] += r.get("prompt_tokens") or 0
        d["out_tok"] += r.get("completion_tokens") or 0
        d["seconds"] += r.get("seconds") or 0.0
    out = []
    for (p, a, bb), d in sorted(agg.items()):
        n = max(d["runs"], 1)
        row = {"protocol": p, "arm": a, "backbone": bb, "runs": d["runs"],
               "in_tok_per_unit": round(d["in_tok"] / n, 1),
               "out_tok_per_unit": round(d["out_tok"] / n, 1),
               "wall_seconds_per_unit": round(d["seconds"] / n, 2)}
        for op in ("initial_generation", "revision_with_peers", "aggregation"):
            row[f"calls_{op}_per_unit"] = round(d["calls"][op] / n, 3)
            row[f"out_tok_{op}_per_unit"] = round(d["out"][op] / n, 1)
        out.append(row)
    return out


def moa_split_audit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How often MoA's exact reference split applied, vs the released regex fallback."""
    modes = collections.Counter()
    calls = 0
    for r in rows:
        if r.get("protocol") != "moa" or r.get("status") != "ok":
            continue
        for t in r.get("arm_traces") or []:
            s = t.get("moa_split")
            if not s:
                continue
            calls = max(calls, 0)
            for m, n in (s.get("modes") or {}).items():
                modes[m] += n
    total = sum(modes.values())
    return {"split_decisions": total, "modes": dict(modes),
            "exact_share": round(modes["exact"] / total, 5) if total else None,
            "note": "the released regex counts a model's own numbered steps as references "
                    "(measured 2.45x inflation on 100 real GPQA aggregator prompts); the exact "
                    "split uses the reference replies this process observed and falls back only "
                    "where the prompt is not their concatenation"}


def sql_execution_cost() -> Optional[Dict[str, Any]]:
    """Both execution passes' costs, and the per-unit error taxonomy, kept apart from model cost."""
    out: Dict[str, Any] = {}
    p = ANALYSIS / "sql_execution.json"
    if p.exists():
        blob = json.loads(p.read_text(encoding="utf-8"))
        out["delivered_answer_pass"] = blob.get("execution_cost")
        out["read_only"] = blob.get("read_only")
        out["identity"] = blob.get("identity")
        tax: Dict[str, int] = collections.Counter()
        per_proto: Dict[str, Dict[str, int]] = {}
        for k, c in (blob.get("per_cell") or {}).items():
            proto = k.split("|")[0]
            for kk, vv in c["outcomes"].items():
                tax[kk] += vv
                per_proto.setdefault(proto, collections.Counter())[kk] += vv
        out["error_taxonomy"] = dict(tax)
        out["error_taxonomy_by_protocol"] = {k: dict(v) for k, v in per_proto.items()}
    q = ANALYSIS / "sql_verdicts.json"
    if q.exists():
        blob = json.loads(q.read_text(encoding="utf-8"))
        out["diagnostics_pass"] = blob.get("execution_cost")
    return out or None


# ---------------------------------------------------------------- rendering
def render(benchmark: str, a: Dict[str, Any], extra: Dict[str, Any]) -> str:
    okc, have, need = EC.design_complete(benchmark)
    f = D.frozen(benchmark)
    L = [f"# {benchmark}: benchmark-extension deliverable", "",
         f"Run `{D.RUN_ID}`, code version `{EC.CODE_VERSION}`. "
         f"Design {have:,}/{need:,} unit-runs"
         f"{'  **COMPLETE**' if okc else '  — **INCOMPLETE, do not read the tables below as a '
                                        'finished benchmark**'}.", ""]
    if not okc:
        L += ["> This benchmark is unfinished. Coverage is reported per cell so the gaps are "
              "visible, and no contrast is presented as a result for the benchmark.", ""]
    L += [f"Records {a.get('records', 0):,} (ok {a.get('ok', 0):,}, "
          f"errors {a.get('errors', 0):,}). Answer kind `{a.get('kind')}`; reference answers for "
          f"{a.get('gold_units_available', 0):,} units.", "",
          "## Frozen sample", "",
          f"- {f['n_units']} units, sha `{f['sha256_16']}`, frozen {f['frozen_utc']}",
          f"- selection rule: {f['selection_rule']}", ""]
    if benchmark in D.SQL_BENCH:
        L += [f"- population: {f['population']['source']}, {f['population']['n_questions']:,} "
              f"questions, sha `{f['population']['sha256_16']}`",
              f"- {f['n_databases']} databases, difficulty {f['per_difficulty']}",
              f"- mean schema on every prompt: {f['mean_schema_chars']:.0f} chars", ""]
    else:
        L += [f"- block: {f['block']['repo']} `{f['block']['config']}` / {f['block']['split']} "
              f"(split size {f['block']['split_size']}), block sha256 "
              f"`{f['block']['block_sha256'][:16]}…`",
              f"- option counts {f['option_counts']}, gold letters {f['gold_letters']}, "
              f"domains {f['domains']}", ""]

    L += ["## Coverage per cell (protocol x arm x backbone x repeat)", "",
          "| protocol | arm | backbone | r1 | r2 |", "|---|---|---|---:|---:|"]
    seen = {tuple(k.split("|")[:3]) for k in a.get("coverage", {})}
    for p in D.PROTOS:
        for arm in D.ARMS:
            for bb in sorted({x[2] for x in seen if x[0] == p and x[1] == arm}):
                L.append("| %s | %s | %s | %s | %s |"
                         % (p, arm, bb, f"{a['coverage'].get(f'{p}|{arm}|{bb}|r1', 0):,}",
                            f"{a['coverage'].get(f'{p}|{arm}|{bb}|r2', 0):,}"))
    missing = [c for c in D.cells(benchmark)
               if a.get("coverage", {}).get(f"{c[0]}|{c[1]}|{c[2]}|r{c[3]}", 0) < f["n_units"]]
    if missing:
        L += ["", f"**{len(missing)} of {len(D.cells(benchmark))} cells are short of "
                  f"{f['n_units']} units.**"]
    if a.get("errors"):
        L += ["", f"Errors {a['errors']:,}, kept with their true classification: "
              + ", ".join(f"`{k}` {v:,}" for k, v in (a.get("error_classes") or {}).items())]

    L += ["", "## Quality and delivery (separate denominators)", "",
          "A unit-run that produced no parseable answer is counted in delivery and excluded from "
          "accuracy-on-delivered. The charged convention — non-delivery is a failure — is the "
          "headline in the effects table below.", "",
          "| protocol | arm | backbone | runs | delivered | accuracy on delivered |",
          "|---|---|---|---:|---:|---:|"]
    for kk, d in (a.get("performance") or {}).items():
        p, arm, bb = kk.split("|")
        acc = (d["correct"] / d["scored"]) if d["scored"] else float("nan")
        L.append("| %s | %s | %s | %s | %s/%s (%.1f%%) | %s |"
                 % (p, arm, bb, f"{d['runs']:,}", f"{d['delivered']:,}", f"{d['runs']:,}",
                    100 * d["delivered"] / max(d["runs"], 1),
                    (f"{acc:.4f} (n={d['scored']:,})" if d["scored"] else "n/a")))

    eff = extra.get("effects") or []
    if eff:
        L += ["", "## Paired effects against FULL, unit-clustered 95% CI", "",
              "Pairing is on (unit, backbone, repeat); repeats are averaged inside the unit before "
              "differencing, and the bootstrap resamples units. Backbone is a block: per-backbone "
              "estimates are reported, never pooled away. **A CI containing zero means not "
              "distinguished at this sample size** — no margin was pre-specified, so this is not "
              "equivalence and not an upper bound on the effect.", "",
              "| protocol | backbone | contrast | dimension | convention | n units | metric | "
              "effect [95% CI] |", "|---|---|---|---|---|---:|---|---|"]
        for e in eff:
            L.append("| %s | %s | %s | %s | %s | %s | %s | %+.4f [%+.4f, %+.4f]%s |"
                     % (e["protocol"], e["backbone"], e["contrast"], e["dimension"],
                        "charged" if e["convention"].startswith("abstention") else "delivered only",
                        f"{e['n_units']:,}", e["metric"], e["effect"], e["lo"], e["hi"],
                        "" if not (e["lo"] <= 0 <= e["hi"]) else " *(not distinguished)*"))

    L += ["", "## The four dimensions, from the recorded traces", "",
          "| protocol | arm | calls | peer blocks | peers rendered | ANSONLY applied "
          "(contract/nl) | ANSONLY failed | mean size ratio |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for p, row in (a.get("dimensions") or {}).items():
        for arm in D.ARMS:
            d = row.get(arm) or {}
            if not d.get("calls"):
                continue
            sr = d.get("mean_size_ratio")
            L.append("| %s | %s | %s | %s | %s | %s (%s/%s) | %s | %s |"
                     % (p, arm, f"{d['calls']:,}", f"{d['peer_blocks_found']:,}",
                        f"{d['n_peers_rendered']:,}", f"{d['ansonly_applied']:,}",
                        f"{d['ansonly_contract']:,}", f"{d['ansonly_nl']:,}",
                        f"{d['ansonly_failed']:,}", (f"{sr:.3f}" if sr else "-")))
    L += ["", "Access is read as *peers rendered*: NOPEER must be 0 where FULL is positive. "
              "Content is read as *ANSONLY applied* against *failed* — a failed extraction shipped "
              "the peer's full text, so that call is ANSONLY in name and FULL in substance and is "
              "excluded from the content contrast.", "",
          "**What UNIFSEL is, and is not.** It keeps a uniform random half of the peers the "
          "released prompt rendered (at least one), so it changes *how many* peers are present as "
          "well as *which*: it is the middle of an access-volume series — FULL, half, none — and "
          "not a which-peers contrast. A which-peers contrast cannot be built by message surgery "
          "at all, because the prompt contains only the peers the protocol already chose. FORD has "
          "no UNIFSEL arm for the same reason: it renders one peer, and its selection/use factor "
          "is its final-answer rule, recomputed from the record.", ""]

    dg = extra.get("diagnostics") or []
    if dg:
        L += ["## Correction, regression, availability loss, selection miss", "",
              "Denominators are the paper's: correction over the initially wrong recipients, "
              "regression over the initially correct ones, availability loss conditioned on having "
              "had a correct candidate (so loss is separated from failure to generate), and "
              "selection miss over the units where a correct candidate survived to selection.", "",
              "| protocol | backbone | condition | wrong n | correction | correct n | regression | "
              "had-correct n | availability loss | at-selection n | selection miss |",
              "|---|---|---|---:|---|---:|---|---:|---|---:|---|"]
        for d in dg:
            L.append("| %s | %s | %s | %s | %.4f [%.4f, %.4f] | %s | %.4f [%.4f, %.4f] | %s | "
                     "%.4f [%.4f, %.4f] | %s | %.4f [%.4f, %.4f] |"
                     % (d["protocol"], d["backbone"], d["condition"],
                        f"{d['n_recipients_wrong']:,}", d["correction_rate"], d["correction_lo"],
                        d["correction_hi"], f"{d['n_recipients_correct']:,}",
                        d["regression_rate"], d["regression_lo"], d["regression_hi"],
                        f"{d['n_with_initial_correct']:,}", d["availability_loss"], d["loss_lo"],
                        d["loss_hi"], f"{d['n_correct_at_selection']:,}", d["selection_miss"],
                        d["miss_lo"], d["miss_hi"]))
        L.append("")

    tr = extra.get("truncation") or []
    if tr:
        L += ["## Availability loss at the output limit", "",
              f"`max_tokens` is {D.PROD_MAX_TOKENS}, the value the three delivered QA benchmarks "
              f"used; it is not raised, because raising it would make this benchmark incomparable "
              f"with them. On this benchmark the limit **binds**, so a truncated reply is reported "
              f"as its own loss mechanism rather than merged into abstention.", "",
              "| protocol | arm | backbone | calls | truncated | truncated call rate | "
              "unit-runs with any truncation | delivery |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
        for t in tr:
            L.append("| %s | %s | %s | %s | %s | %.4f | %.4f | %.4f |"
                     % (t["protocol"], t["arm"], t["backbone"], f"{t['calls']:,}",
                        f"{t['truncated']:,}", t["truncated_call_rate"],
                        t["unit_runs_with_any_truncation"], t["delivery_rate"]))
        L.append("")

    oc = extra.get("operation_cost") or []
    if oc:
        L += ["## Cost by operation type", "",
              "Generating a response and revising in the presence of peers are different "
              "operations; aggregation is a third. Charged per unit-run from each record's own "
              "call list, so an arm that shortens prompts shows up as fewer input tokens rather "
              "than as a changed call count.", "",
              "| protocol | arm | backbone | calls init / revise / agg | out tok init / revise / "
              "agg | in tok/unit | out tok/unit | wall s/unit |",
              "|---|---|---|---|---|---:|---:|---:|"]
        for c in oc:
            L.append("| %s | %s | %s | %.2f / %.2f / %.2f | %.0f / %.0f / %.0f | %.0f | %.0f | "
                     "%.2f |" % (c["protocol"], c["arm"], c["backbone"],
                                 c["calls_initial_generation_per_unit"],
                                 c["calls_revision_with_peers_per_unit"],
                                 c["calls_aggregation_per_unit"],
                                 c["out_tok_initial_generation_per_unit"],
                                 c["out_tok_revision_with_peers_per_unit"],
                                 c["out_tok_aggregation_per_unit"],
                                 c["in_tok_per_unit"], c["out_tok_per_unit"],
                                 c["wall_seconds_per_unit"]))
        L.append("")

    ex = extra.get("sql_execution_cost")
    if ex and ex.get("error_taxonomy"):
        tax = ex["error_taxonomy"]
        tot = sum(tax.values())
        L += ["## SQL outcome taxonomy", "",
              "Every unit-run lands in exactly one class, and abstention is never folded into "
              "error: a protocol that produced no query (`declined`) is not the same as one whose "
              "query returned the wrong rows (`mismatch`), and neither is the same as a query that "
              "would not parse. `truncated_only` is a decline whose run also hit the output limit "
              "— a distinct cause. `gold_failed` means the *reference* query failed on this "
              "instance, so the unit cannot be scored at all and is not charged to the protocol.",
              "", "| outcome | unit-runs | share |", "|---|---:|---:|"]
        for k in ("match", "mismatch", "syntax_error", "missing_object", "type_or_arg_error",
                  "other_exec_error", "exec_timeout", "gold_failed", "declined",
                  "truncated_only", "collection_error", "transport_failed"):
            if tax.get(k):
                L.append("| `%s` | %s | %.3f |" % (k, f"{tax[k]:,}", tax[k] / tot))
        L += ["", "| protocol | " + " | ".join(
            k for k in ("match", "mismatch", "syntax_error", "missing_object", "declined")
            if tax.get(k)) + " |",
            "|---|" + "---:|" * len([k for k in ("match", "mismatch", "syntax_error",
                                                 "missing_object", "declined") if tax.get(k)])]
        for proto, pt in (ex.get("error_taxonomy_by_protocol") or {}).items():
            L.append("| %s | " % proto + " | ".join(
                f"{pt.get(k, 0):,}" for k in ("match", "mismatch", "syntax_error",
                                              "missing_object", "declined") if tax.get(k)) + " |")
        L += ["", "**Quality on this benchmark is execution match, not string equality with the "
                  "reference query.** Two different queries can agree on one database instance, so "
                  "a match is evidence on this instance and not proven semantic equivalence; and "
                  "two different spellings of the same correct query are both correct, which is "
                  "why every answer above -- final, per-agent and per-call -- was decided by "
                  "executing it rather than by comparing its text.", ""]
    if ex:
        L += ["## Execution cost, accounted separately from model cost", "",
              "Queries are executed **once**, at evaluation, read-only, in an isolated subprocess "
              "with a statement timeout and a row cap. No arm received an execution result; the "
              "collection process has `sql_common.execute`, `gold_sql` and `execution_match` "
              "replaced by functions that raise, so an accidental call fails loudly instead of "
              "becoming an unrecorded intervention.", "",
              "```json", json.dumps(ex, indent=1), "```", ""]

    ms = extra.get("moa_split")
    if ms and ms.get("split_decisions"):
        L += ["## MoA's reference split", "",
              f"{ms['split_decisions']:,} split decisions, modes {ms['modes']} "
              f"(exact share {ms['exact_share']}). {ms['note']}", ""]

    fu = a.get("ford_use") or {}
    if fu.get("records"):
        L += ["## FORD's selection/use contrast", "",
              f"FORD renders one opponent, so there is no peer to choose between; its "
              f"selection/use factor is its final-answer rule. On {fu['records']:,} records the "
              f"released two-consecutive-agreement rule and the all-round majority agree on "
              f"{fu['rules_agree']:,} "
              f"({100 * fu['rules_agree'] / max(fu['records'], 1):.1f}%). FORD also runs at "
              f"temperature 0, so its two repeats differ only by server-side nondeterminism, not "
              f"by sampling — its repeat spread is not a sampling estimate.", ""]

    exc = extra.get("excluded") or {}
    if exc.get("excluded"):
        L += ["## Records on disk that this analysis does not read", "",
              "`raw/` is append-only: a failed run leaves its failures behind and a cell "
              "re-collected after a repair holds both versions. Nothing is deleted, and nothing is "
              "silently dropped either.", "",
              "| reason | records |", "|---|---:|"]
        for k, v in exc["excluded"].items():
            L.append("| %s | %s |" % (k, f"{v:,}"))
        L += ["", "Code versions present on disk: %s; current `%s`."
              % (exc.get("code_versions_on_disk"), exc.get("current_code_version")), ""]

    L += ["## Open problems", ""]
    opens = []
    for p, row in (a.get("dimensions") or {}).items():
        d = row.get("ANSONLY") or {}
        tot = d.get("ansonly_applied", 0) + d.get("ansonly_failed", 0)
        if tot and d.get("ansonly_failed", 0) / tot > 0.05:
            opens.append(f"- {p}/ANSONLY: {d['ansonly_failed']:,} of {tot:,} peer blocks "
                         f"({100 * d['ansonly_failed'] / tot:.1f}%) carried no readable answer, so "
                         f"those calls were not reduced and are excluded from the content "
                         f"contrast.")
    for kk, d in (a.get("performance") or {}).items():
        if d["runs"] and d["delivered"] / d["runs"] < 0.8:
            opens.append(f"- {kk}: delivery {d['delivered']:,}/{d['runs']:,} "
                         f"({100 * d['delivered'] / d['runs']:.0f}%); top stop reasons "
                         + ", ".join(f"`{k}` {v}" for k, v
                                     in collections.Counter(d["stop_reasons"]).most_common(2)))
    # A FULL cell that delivered nothing is not a quality result and its paired contrasts are
    # uninformative: the baseline is pinned at the floor, so every difference against it measures
    # whether the condition restored delivery rather than whether sharing helped.
    for kk, d in (a.get("performance") or {}).items():
        p_, arm_, bb_ = kk.split("|")
        if arm_ == "FULL" and d["runs"] and d["delivered"] == 0:
            others = {k2.split("|")[1]: v2 for k2, v2 in a["performance"].items()
                      if k2.split("|")[0] == p_ and k2.split("|")[2] == bb_
                      and k2.split("|")[1] != "FULL"}
            opens.append(
                f"- **{p_}/{bb_}: the FULL arm delivered 0 of {d['runs']:,} runs.** Its released "
                f"answer contract was satisfied by no run at all on this backbone, so every paired "
                f"contrast against this FULL baseline is pinned at the floor and measures whether "
                f"the arm restored delivery, not whether sharing helped. Delivery in the other "
                f"arms: "
                + ", ".join(f"{k2} {v2['delivered']:,}/{v2['runs']:,}"
                            for k2, v2 in sorted(others.items()))
                + ". Reported as a contract-compliance failure, not as an intervention effect.")
    if not okc:
        opens.append(f"- the benchmark is incomplete ({have:,}/{need:,}); every table above is "
                     f"partial coverage, not a result.")
    L += opens or ["- none recorded."]

    L += ["", "## Provenance", "",
          f"- raw records: `results/{D.RUN_ID}/raw/{benchmark}__*.jsonl` (append-only, full input "
          f"and output per call, fsynced before counting)",
          f"- frozen sample: `results/{D.RUN_ID}/frozen/{benchmark}.json`",
          f"- offline acceptance: `results/{D.RUN_ID}/ACCEPTANCE_*.json`",
          f"- reuse audit and decision: `results/{D.RUN_ID}/REUSE_DECISION.json`",
          f"- manifest: `results/{D.RUN_ID}/manifest.jsonl`",
          "- STOP entry: `python3 ext_stop_control.py stop`"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmarks", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    AB, ARQ = _patch()
    names = a.benchmarks
    if a.all or not names:
        names = sorted({p.name.split("__")[0] for p in RAW.glob("*.jsonl")})
    if not names:
        print("no records under %s yet" % RAW)
        return 0
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    # the paper's estimators, over the benchmarks named, into this run's analysis directory
    argv = sys.argv
    sys.argv = ["analyze_research_questions.py", "--benchmarks", *names]
    try:
        ARQ.main()
    finally:
        sys.argv = argv
    rq = json.loads((ANALYSIS / "rq_results.json").read_text(encoding="utf-8")) \
        if (ANALYSIS / "rq_results.json").exists() else {"effects": [], "diagnostics": []}

    for b in names:
        res = AB.analyse(b)
        rows = read_ok(b)
        extra = {
            "effects": [e for e in rq.get("effects", []) if e["benchmark"] == b],
            "diagnostics": [d for d in rq.get("diagnostics", []) if d["benchmark"] == b],
            "truncation": truncation_table(rows),
            "operation_cost": operation_cost_table(rows),
            "moa_split": moa_split_audit(rows),
            "excluded": excluded_records(b),
            "sql_execution_cost": sql_execution_cost() if b in D.SQL_BENCH else None,
        }
        okc, have, need = EC.design_complete(b)
        res["design_complete"] = okc
        res["design_have"] = have
        res["design_need"] = need
        res["extension"] = extra
        res["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        d = OUT / b
        d.mkdir(parents=True, exist_ok=True)
        (d / "REPORT.md").write_text(render(b, res, extra), encoding="utf-8")
        (d / "report.json").write_text(json.dumps(res, indent=1, default=str) + "\n",
                                       encoding="utf-8")
        print("%-14s records %7s ok %7s errors %5s design %s/%s %s -> %s"
              % (b, f"{res.get('records', 0):,}", f"{res.get('ok', 0):,}",
                 f"{res.get('errors', 0):,}", f"{have:,}", f"{need:,}",
                 "COMPLETE" if okc else "incomplete", (d / "REPORT.md").relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
