#!/usr/bin/env python3
"""The 25x4 preflight: one frozen unit per (benchmark, protocol, condition), under one persistent ledger.

    python3 preflight2.py --plan     # offline: the order and the worst-case budget, no calls
    python3 preflight2.py --live     # one bounded pass
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from preflight_budget import Budget, BudgetExceeded                      # noqa: E402
import arm_transform as AT                                               # noqa: E402
import unit_driver as UD                                                 # noqa: E402

PROTOS = ["ford", "moa", "du_mad", "madm2_objective", "bccs"]
BENCH = ["wdc_products", "geographic_settlements", "gsm8k", "strategyqa", "mmlu_pro"]
ARMS = ["FULL", "NOPEER", "ANSONLY", "UNIFSEL"]
#: measured calls per unit from the collection, used only to order the sweep
COST = {"ford": 3.2, "moa": 4.0, "du_mad": 6.0, "madm2_objective": 6.0, "bccs": 11.7}
ALIAS = {"ford": "llama3-8b-g1", "moa": "llama3-8b-g1", "du_mad": "llama3-8b-g1",
         "madm2_objective": "llama3-8b-g1", "bccs": "llama3-8b-g1"}

CAPS = dict(max_calls=9000, max_output_tokens=3_000_000, max_input_tokens=11_000_000,
            max_gpu_hours=0.5, max_wall_seconds=30 * 60)
CARDS = 2
MAX_UNIT_RUNS = 100          # the authorised main pass
MAX_ATTEMPTS_TOTAL = 150     # including any targeted retry
FROZEN = ROOT / "data/frozen/FROZEN_SAMPLE_5X5.json"
OUT = ROOT / "results/controlled_5x5_preflight2"
LEDGER = OUT / "PERSISTENT_LEDGER.json"


def frozen_unit(bench: str) -> str:
    blob = json.loads(FROZEN.read_text(encoding="utf-8"))
    if bench in blob["matching"]:
        return blob["matching"][bench]["unit_ids"][0]
    return blob["qa"][bench]["shared_task_ids"][0]


def sweep():
    """[(benchmark, protocol, condition, unit)] breadth-first, cheap protocols first."""
    cells = []
    for b in BENCH:
        u = frozen_unit(b)
        for p in PROTOS:
            for a in ARMS:
                why = UD.supported(b, p, a)
                cells.append({"benchmark": b, "protocol": p, "arm": a, "unit": u,
                              "runnable": why is None, "reason": why,
                              "cost": COST[p]})
    cells.sort(key=lambda c: (c["cost"], c["protocol"], c["benchmark"], ARMS.index(c["arm"])))
    return cells


def load_prior() -> dict:
    if LEDGER.exists():
        try:
            return json.loads(LEDGER.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"attempts": 0, "calls": 0, "output_tokens": 0, "input_tokens": 0,
            "gpu_seconds": 0.0, "passes": []}


def save_prior(prior: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(prior, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--only-protocol", action="append")
    ap.add_argument("--only-arm", action="append")
    a = ap.parse_args()
    cells = sweep()
    runnable = [c for c in cells if c["runnable"]]
    if a.only_protocol:
        runnable = [c for c in runnable if c["protocol"] in a.only_protocol]
    if a.only_arm:
        runnable = [c for c in runnable if c["arm"] in a.only_arm]
    prior = load_prior()

    print(f"cells: {len(cells)}  runnable: {len([c for c in cells if c['runnable']])}  "
          f"selected this pass: {len(runnable)}")
    worst_calls = sum(int(c["cost"] * 1.6) + 2 for c in runnable)
    print(f"worst-case calls {worst_calls}, worst-case output "
          f"{worst_calls * UD.PROD_MAX_TOKENS:,} tokens at max_tokens={UD.PROD_MAX_TOKENS}")
    print(f"persistent ledger so far: attempts {prior['attempts']}, calls {prior['calls']}, "
          f"output {prior['output_tokens']:,}, gpu_seconds {prior['gpu_seconds']:.0f}")
    for c in cells:
        if not c["runnable"]:
            print(f"  NOT RUNNABLE  {c['benchmark']}/{c['protocol']}/{c['arm']}: {c['reason'][:90]}")
    if a.plan:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = OUT / f"live_{stamp}"
    run_dir.mkdir()
    # the remaining allowance is the cap minus what this goal already spent
    budget = Budget(max_calls=CAPS["max_calls"] - prior["calls"],
                    max_output_tokens=CAPS["max_output_tokens"] - prior["output_tokens"],
                    max_input_tokens=CAPS["max_input_tokens"] - prior["input_tokens"],
                    max_gpu_hours=CAPS["max_gpu_hours"] - prior["gpu_seconds"] * CARDS / 3600.0,
                    max_wall_seconds=CAPS["max_wall_seconds"], cards=CARDS,
                    state_path=run_dir / "BUDGET.json")
    print(f"\nremaining allowance: calls {budget.caps['calls']}, output "
          f"{budget.caps['output_tokens']:,}, wall {budget.effective_wall_seconds:.0f}s")

    already = set()
    for pdir in sorted(OUT.glob("live_*")):
        rp = pdir / "RESULTS.jsonl"
        if not rp.exists():
            continue
        for line in rp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                already.add((r["benchmark"], r["protocol"], r["arm"]))
    if already:
        before = len(runnable)
        runnable = [c for c in runnable
                    if (c["benchmark"], c["protocol"], c["arm"]) not in already]
        print(f"  {before - len(runnable)} cells already verified in an earlier pass, skipped")

    results, stop = [], None
    done = prior["attempts"]
    for c in runnable:
        if done >= MAX_ATTEMPTS_TOTAL:
            stop = f"attempt ceiling {MAX_ATTEMPTS_TOTAL} reached"
            break
        if len(results) >= MAX_UNIT_RUNS:
            stop = f"main-pass ceiling {MAX_UNIT_RUNS} unit-runs reached"
            break
        try:
            budget.check()
        except BudgetExceeded as e:
            stop = e.reason
            break
        rec = {k: c[k] for k in ("benchmark", "protocol", "arm", "unit")}
        t0 = time.time()
        try:
            out = UD.run_unit(c["benchmark"], c["protocol"], c["unit"], ALIAS[c["protocol"]],
                              arm=c["arm"], max_tokens=UD.PROD_MAX_TOKENS, timeout=90.0,
                              budget=budget)
            r = out["record"]
            import collect_all_matching as C
            rec.update(status="ok", calls=len(r.get("calls") or []),
                       majority_answer=r.get("majority_answer"),
                       raw_io=C.check_raw_preserved(r) or "preserved",
                       stop_reason=r.get("stop_reason") or r.get("native_stop_reason"),
                       rounds_run=r.get("rounds_run"),
                       config=out["config"], arm_traces=out["arm_traces"],
                       completion_tokens=sum(x.get("completion_tokens") or 0
                                             for x in r.get("calls") or []),
                       prompt_tokens=sum(x.get("prompt_tokens") or 0
                                         for x in r.get("calls") or []))
            if c["protocol"] == "ford":
                rec["ford_use_contrast"] = AT.use_arm_final_answer("ford", r)
        except BudgetExceeded as e:
            stop = e.reason
            rec.update(status="stopped")
            results.append(rec)
            done += 1
            break
        except Exception as exc:
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}"[:400])
        rec["seconds"] = round(time.time() - t0, 2)
        results.append(rec)
        done += 1
        budget.persist()
        (run_dir / "RESULTS.jsonl").write_text(
            "\n".join(json.dumps(x) for x in results) + "\n", encoding="utf-8")

    snap = budget.snapshot()
    snap["stop_reason"] = stop or snap.get("stop_reason")
    (run_dir / "FINAL.json").write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8")
    prior["attempts"] = done
    prior["calls"] += snap["used"]["calls"]
    prior["output_tokens"] += snap["used"]["output_tokens"]
    prior["input_tokens"] += snap["used"]["input_tokens"]
    prior["gpu_seconds"] += snap["elapsed_seconds"]
    prior["passes"].append({"dir": run_dir.name, "used": snap["used"],
                            "elapsed": snap["elapsed_seconds"], "stop": snap["stop_reason"]})
    save_prior(prior)

    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nunit-runs {len(results)}, ok {ok}, errors {len(results) - ok}")
    print(f"calls {snap['used']['calls']}, output {snap['used']['output_tokens']:,}, "
          f"input {snap['used']['input_tokens']:,}, elapsed {snap['elapsed_seconds']}s, "
          f"gpu_hours {snap['gpu_hours_used']}")
    print(f"stop: {snap['stop_reason'] or 'sweep complete'}")
    print(f"-> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
