#!/usr/bin/env python3
"""Endpoint evaluation for the BIRD SQL leg. Reference SQL and the database enter only here.

    python3 eval_sql.py                 # complete configs only
    python3 eval_sql.py --allow-partial
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "experiments/06_cost_resources/code")]

RAW_V2 = ROOT / "results/protocol_all_matching_20261018T192619Z/raw_sql"
RAW_OLD = ROOT / "results/protocol_cross_task_2026-10-18/raw"
OUT = ROOT / "experiments/01_protocol_comparison/analysis"
N_QUESTIONS = 1534
ORDER = ["madm2_naive", "madm2_objective", "bccs", "dmad"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--design", choices=("v2", "old"), default="v2",
                    help="v2 = the corrected prompt in raw_sql/; old = the earlier run, kept for "
                         "audit. The two are never pooled: their questions differ.")
    ap.add_argument("--limit-per-config", type=int, default=0,
                    help="SMOKE TEST ONLY. Takes the FIRST N units of a config, which is a prefix "
                         "of the question_id order and therefore a biased subpopulation: on "
                         "MAD-M2 naive the first 300 gave 0.2344 against 0.3350 on all 1,534. The "
                         "output is written to a _smoke suffix so it can never be mistaken for a "
                         "report figure.")
    a = ap.parse_args()

    import sql_common as SQ
    qs = {f"bird|{q['question_id']}": q
          for q in json.load(open(pathlib.Path(SQ.BIRD) / "dev.json", encoding="utf-8"))}

    cells = collections.defaultdict(dict)
    base = RAW_V2 if a.design == "v2" else RAW_OLD
    pat = "sql_bird_v2__*.jsonl" if a.design == "v2" else "sql_bird__*.jsonl"
    for f in sorted(base.glob(pat)):
        parts = f.stem.split("__")
        if len(parts) < 4:
            continue
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = (parts[1], parts[2], parts[3])
            u = r["unit_id"].split("|", 3)[3]
            cur = cells[key].get(u)
            if cur is None or (cur.get("status") != "ok" and r.get("status") == "ok"):
                cells[key][u] = r

    report = {"per_config": {}, "by_protocol": {}, "configs_excluded": [],
              "caveat": ("execution match is single-instance set equality, the benchmark's own "
                         "notion, not proven semantic equivalence; execution is read-only and "
                         "happens only at evaluation")}

    for key in sorted(cells):
        units = cells[key]
        n_ok = sum(1 for v in units.values() if v.get("status") == "ok")
        if n_ok < N_QUESTIONS and not a.allow_partial:
            report["configs_excluded"].append(
                {"config": "|".join(key), "ok_records": n_ok, "expected": N_QUESTIONS})
            continue
        av = collections.Counter()
        reasons = collections.Counter()
        ok = scored = 0
        cost = collections.defaultdict(list)
        distinct = []
        items = list(units.items())
        if a.limit_per_config:
            items = items[: a.limit_per_config]
        for u, env in items:
            st = env.get("status")
            if st != "ok":
                e = str(env.get("error") or env.get("reason") or st)
                av["transport_failed" if "TransportFailure" in e else "collection_error"] += 1
                continue
            rec = env.get("record") or {}
            ans = rec.get("majority_answer")
            cost["calls"].append(env["n_calls"])
            cost["out"].append(env["completion_tokens_total"])
            if rec.get("n_distinct_final_answers") is not None:
                distinct.append(rec["n_distinct_final_answers"])
            if ans in (None, ""):
                av["declined"] += 1
                continue
            av["delivered"] += 1
            q = qs.get(u)
            if q is None:
                av["no_gold"] += 1
                continue
            m = SQ.execution_match(q["db_id"], ans, SQ.gold_sql(q))
            reasons[m.get("reason", "?")] += 1
            scored += 1
            ok += 1 if m["match"] else 0
        n = sum(av.values())
        charged = av["delivered"] + av["declined"]
        report["per_config"]["|".join(key)] = {
            "n_units": n, "availability": dict(av),
            "execution_accuracy_delivered_only": round(ok / scored, 6) if scored else None,
            "execution_accuracy_declines_charged": round(ok / charged, 6) if charged else None,
            "n_scored": scored, "n_matched": ok,
            "declined_share": round(av["declined"] / n, 6) if n else None,
            "transport_failed_share": round(av["transport_failed"] / n, 6) if n else None,
            "pred_status": dict(reasons),
            "string_agreement": {
                "mean_distinct_final_answers": round(statistics.fmean(distinct), 4)
                if distinct else None,
                "share_all_agents_identical": round(
                    sum(1 for x in distinct if x == 1) / len(distinct), 4) if distinct else None},
            "cost": {k: round(statistics.fmean(v), 2) for k, v in cost.items() if v},
        }

    for pr in ORDER:
        ks = [k for k in report["per_config"] if k.split("|")[0] == pr]
        if not ks:
            continue
        acc = [report["per_config"][k]["execution_accuracy_delivered_only"] for k in ks
               if report["per_config"][k]["execution_accuracy_delivered_only"] is not None]
        ch = [report["per_config"][k]["execution_accuracy_declines_charged"] for k in ks
              if report["per_config"][k]["execution_accuracy_declines_charged"] is not None]
        dec = [report["per_config"][k]["declined_share"] for k in ks]
        ag = [report["per_config"][k]["string_agreement"]["share_all_agents_identical"]
              for k in ks
              if report["per_config"][k]["string_agreement"]["share_all_agents_identical"]
              is not None]
        report["by_protocol"][pr] = {
            "configs_complete": len(ks), "configs_expected": 6,
            "execution_accuracy_delivered_only": {
                "mean": round(statistics.fmean(acc), 6) if acc else None,
                "sd": round(statistics.stdev(acc), 6) if len(acc) > 1 else None},
            "execution_accuracy_declines_charged": {
                "mean": round(statistics.fmean(ch), 6) if ch else None},
            "declined_share": round(statistics.fmean(dec), 6) if dec else None,
            "share_all_agents_identical_sql": round(statistics.fmean(ag), 4) if ag else None,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    report["design"] = a.design
    report["source"] = str(base)
    # a prefix-limited run must not land on the path the report reads
    suffix = "_smoke" if a.limit_per_config else ""
    if a.limit_per_config:
        report["WARNING"] = (
            f"prefix of {a.limit_per_config} units per config, NOT a random sample: the question_id "
            f"order is not difficulty-uniform, so this is a biased subpopulation and is a smoke "
            f"test only")
    p = OUT / f"sql_evaluation_{a.design}{suffix}.json"
    p.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"complete configs scored: {len(report['per_config'])}   "
          f"excluded: {len(report['configs_excluded'])}")
    if report["by_protocol"]:
        print(f"\n{'workflow':18s}{'cfgs':>6s}{'exec acc':>10s}{'sd':>9s}{'charged':>9s}"
              f"{'declined':>10s}{'all agents same SQL':>21s}")
        for pr, s in report["by_protocol"].items():
            m = s["execution_accuracy_delivered_only"]
            sd = m.get("sd")
            print(f"  {pr:16s}{s['configs_complete']:3d}/6"
                  f"{(m['mean'] if m['mean'] is not None else 0):10.4f}"
                  f"{(f'{sd:.4f}' if sd is not None else '-'):>9s}"
                  f"{(s['execution_accuracy_declines_charged']['mean'] or 0):9.4f}"
                  f"{(s['declined_share'] or 0):10.4f}"
                  f"{(s['share_all_agents_identical_sql'] if s['share_all_agents_identical_sql'] is not None else 0):21.4f}")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
