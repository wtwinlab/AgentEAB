#!/usr/bin/env python3
"""Controlled-5x5 preflight: function check for each intervention condition, under hard caps.

    python3 preflight_runner.py --dry-run     # offline: canned replies, no model calls
    python3 preflight_runner.py --live        # one bounded live pass
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import random
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "code"), str(HERE),
                str(ROOT / "experiments/04_sql_selection_generation/code"),
                str(ROOT / "experiments/06_cost_resources/code"),
                str(ROOT / "experiments/05_matching_downstream/code")]

from preflight_budget import Budget, BudgetExceeded            # noqa: E402

CAPS = dict(max_calls=9000, max_output_tokens=3_000_000, max_input_tokens=11_000_000,
            max_gpu_hours=0.5, max_wall_seconds=30 * 60)
CARDS = 2                      # GPU1 + GPU2; GPU0 released, GPU3 not used
SEED = 20261019
N_PER_ARM = 3
MAX_TOKENS = 256               # clamped: worst case per call, so the output cap cannot be crossed

FROZEN = ROOT / "data/frozen/FROZEN_SAMPLE_5X5.json"
OUT = ROOT / "workspace/controlled_sharing_preflight"

#: protocol -> condition -> (installer, note). An installer returns a context manager that patches the
#: released adapter for the duration of one unit-run.
PROTOCOLS = ["bccs", "du_mad", "madm2_objective", "ford", "moa"]
ARMS = ["FULL", "NOPEER", "ANSONLY", "UNIFSEL"]
#: the five frozen benchmarks of the design.
BENCHMARKS = ["wdc_products", "geographic_settlements", "gsm8k", "strategyqa", "mmlu_pro"]
LIVE_BENCHMARKS = ["wdc_products", "geographic_settlements"]
ALIAS_FOR = {"bccs": "llama3-8b-g1", "du_mad": "llama3-8b-g1",
             "madm2_objective": "llama3-8b-g1", "ford": "llama3-8b-g1", "moa": "llama3-8b-g1"}


def arm_installer(proto: str, arm: str):
    """Return (ctx_factory, reason_if_unavailable)."""
    from consensus_delivery.adapters import bccs, du_mad, mad_m2, ford
    import matching_tasks as MT

    if arm == "FULL":
        return contextlib.nullcontext, None

    if proto == "bccs":
        if arm == "UNIFSEL":
            @contextlib.contextmanager
            def ctx():
                orig = bccs.find_top_indices
                rng = random.Random(SEED)
                def patched(lst, top_k):
                    idx = list(range(len(lst)))
                    rng.shuffle(idx)
                    return idx[:top_k]            # same count, uniform choice
                bccs.find_top_indices = patched
                try: yield {"hook": "bccs.find_top_indices", "rule": "uniform k of n"}
                finally: bccs.find_top_indices = orig
            return ctx, None
        if arm == "ANSONLY":
            @contextlib.contextmanager
            def ctx():
                orig = bccs.construct_message
                def patched(cur, responses, question, *a, **kw):
                    stripped = [("\\boxed{%s}" % MT._boxed(r)) if MT._boxed(r) else r
                                for r in responses]
                    return orig(cur, stripped, question, *a, **kw)
                bccs.construct_message = patched
                try: yield {"hook": "bccs.construct_message", "rule": "peer text -> its boxed answer"}
                finally: bccs.construct_message = orig
            return ctx, None
        if arm == "NOPEER":
            @contextlib.contextmanager
            def ctx():
                orig = bccs.construct_message
                def patched(cur, responses, question, *a, **kw):
                    prompt, _shown = orig(cur, responses, question, *a, **kw)
                    # drop the peer block and the clause that refers to it; keep the ask + question
                    tail = "\n\n Can you provide your answer to the problem? \n {}. ".format(question)
                    return tail.lstrip(), []
                bccs.construct_message = patched
                try: yield {"hook": "bccs.construct_message", "rule": "no peer block; ask clause only"}
                finally: bccs.construct_message = orig
            return ctx, None

    if proto == "du_mad":
        if arm == "NOPEER":
            @contextlib.contextmanager
            def ctx():
                orig = du_mad.others_for
                def patched(benchmark, others, question, idx):
                    return orig(benchmark, [], question, idx)      # released SELF_CHECK path
                du_mad.others_for = patched
                try: yield {"hook": "du_mad.others_for", "rule": "others=[] -> released SELF_CHECK"}
                finally: du_mad.others_for = orig
            return ctx, None
        if arm == "ANSONLY":
            @contextlib.contextmanager
            def ctx():
                orig = du_mad.others_for
                def patched(benchmark, others, question, idx):
                    red = []
                    for ctxs in others:
                        c = list(ctxs)
                        b = du_mad.boxed_group(c[idx].get("content", "")) if idx < len(c) else None
                        if b:
                            c[idx] = dict(c[idx], content="\\boxed{%s}" % b)
                        red.append(c)
                    return orig(benchmark, red, question, idx)
                du_mad.others_for = patched
                try: yield {"hook": "du_mad.others_for", "rule": "peer text -> its boxed answer"}
                finally: du_mad.others_for = orig
            return ctx, None
        if arm == "UNIFSEL":
            @contextlib.contextmanager
            def ctx():
                orig = du_mad.others_for
                rng = random.Random(SEED)
                def patched(benchmark, others, question, idx):
                    keep = list(others)
                    if len(keep) > 1:
                        rng.shuffle(keep)
                        keep = keep[:1]            # one uniformly chosen peer instead of all
                    return orig(benchmark, keep, question, idx)
                du_mad.others_for = patched
                try: yield {"hook": "du_mad.others_for", "rule": "1 uniform peer instead of all"}
                finally: du_mad.others_for = orig
            return ctx, None

    if proto == "madm2_objective":
        if arm == "UNIFSEL":
            @contextlib.contextmanager
            def ctx():
                orig = mad_m2.objective_prune
                rng = random.Random(SEED)
                def patched(contexts, ppls):
                    kept, mask = orig(contexts, ppls)
                    k = len(kept)
                    idx = list(range(len(contexts))); rng.shuffle(idx)
                    chosen = set(idx[:k])          # same number kept, uniform choice
                    m = [i in chosen for i in range(len(contexts))]
                    return [c for c, keep in zip(contexts, m) if keep], m
                mad_m2.objective_prune = patched
                try: yield {"hook": "mad_m2.objective_prune",
                            "rule": "same count as the perplexity rule, uniform choice"}
                finally: mad_m2.objective_prune = orig
            return ctx, None
        return None, ("MAD-M^2 assembles its round-1 peer block inline in run_one "
                      "(DEBATE_PREFIX + peers + DEBATE_SUFFIX); NOPEER/ANSONLY would require "
                      "rewriting run_one, which is outside the preflight's change surface")

    if proto == "ford":
        if arm == "NOPEER":
            @contextlib.contextmanager
            def ctx():
                orig = ford.instruction
                def patched(benchmark, *a, **kw):
                    t = orig(benchmark, *a, **kw)
                    return t.replace("Read the question and our opinions carefully",
                                     "Read the question carefully")
                ford.instruction = patched
                try: yield {"hook": "ford.instruction", "rule": "drop 'and our opinions'"}
                finally: ford.instruction = orig
            return ctx, None
        return None, ("FORD assembles the opponent's turn inline in run_one as chat history; "
                      "ANSONLY needs that assembly rewritten, and its selection factor is a "
                      "final-answer USE contrast (released two-consecutive-agreement stop vs "
                      "all-round majority) which is verified separately, not as peer selection")

    if proto == "moa":
        return None, ("MoA builds the aggregator system prompt inline in run_one; all three arms "
                      "require rewriting it, so none is installed here")
    return None, "no installer defined"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if not (a.dry_run or a.live):
        print("pick --dry-run or --live"); return 2
    mode = "dry-run" if a.dry_run else "live"

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = OUT / f"{mode}_{stamp}"
    run_dir.mkdir()
    budget = Budget(**CAPS, cards=CARDS, state_path=run_dir / "BUDGET.json")
    print(f"preflight {mode}: caps {CAPS}, cards={CARDS}, "
          f"effective wall-clock {budget.effective_wall_seconds:.0f}s")
    print(f"  output -> {run_dir}")

    # ---- which conditions are installable, reported before anything runs
    plan, skipped = [], []
    for bench in BENCHMARKS:
        for proto in PROTOCOLS:
            for arm in ARMS:
                ctx, reason = arm_installer(proto, arm)
                (plan if ctx else skipped).append(
                    {"benchmark": bench, "protocol": proto, "arm": arm, "reason": reason})
    (run_dir / "PLAN.json").write_text(json.dumps(
        {"installable": plan, "not_executed": skipped,
         "n_installable_cells": len(plan), "n_not_executed": len(skipped)},
        indent=1) + "\n", encoding="utf-8")
    print(f"  installable (benchmark,protocol,arm): {len(plan)}   not executable: {len(skipped)}")
    by_reason = {}
    for s in skipped:
        by_reason.setdefault(s["reason"][:60], 0)
        by_reason[s["reason"][:60]] += 1
    for r, n in by_reason.items():
        print(f"    {n:3d} cells: {r}...")
    budget.persist()
    if a.dry_run:
        print("\n  dry-run complete: caps verified, installable arms enumerated, no model calls")
        return 0

    # ---------------------------------------------------------------- live pass
    import collect_all_matching as C
    import matching_tasks as MT
    from consensus_delivery import providers

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    units_by_bench = {}
    for b in BENCHMARKS:
        if b in frozen["matching"]:
            units_by_bench[b] = frozen["matching"][b]["unit_ids"][:N_PER_ARM]
        elif b in frozen["qa"]:
            units_by_bench[b] = frozen["qa"][b]["shared_task_ids"][:N_PER_ARM]

    # per-call budget enforcement at the single generic route
    orig_chat = providers.chat
    def guarded_chat(**kw):
        mt = min(int(kw.get("max_tokens") or MAX_TOKENS), MAX_TOKENS)
        kw["max_tokens"] = mt
        est_in = sum(len(m.get("content", "")) for m in kw.get("messages") or []) // 3
        timeout = budget.reserve(mt, est_prompt_tokens=est_in)
        kw["timeout_s"] = min(float(kw.get("timeout_s") or timeout), timeout)
        kw["retries"] = 0                      # a failure is evidence; retries burn the cap
        try:
            text, rec = orig_chat(**kw)
        except Exception:
            budget.settle(mt, est_in, 0)
            raise
        budget.settle(mt, rec.prompt_tokens or 0, rec.completion_tokens or 0)
        return text, rec
    providers.chat = guarded_chat
    for modname in ("bccs", "du_mad", "mad_m2", "ford", "moa"):
        m = __import__(f"consensus_delivery.adapters.{modname}", fromlist=["chat"])
        if hasattr(m, "chat"):
            m.chat = guarded_chat

    results, stop_reason = [], None
    # breadth first: round r takes the r-th unit of every installable cell
    for r in range(N_PER_ARM):
        if stop_reason: break
        for cell in plan:
            bench, proto, arm = cell["benchmark"], cell["protocol"], cell["arm"]
            if bench not in LIVE_BENCHMARKS:
                continue
            ids = units_by_bench.get(bench) or []
            if r >= len(ids): continue
            uid = ids[r]
            try:
                budget.check()
            except BudgetExceeded as e:
                stop_reason = e.reason; break
            rec = {"benchmark": bench, "protocol": proto, "arm": arm, "unit": uid, "round": r}
            t0 = time.time()
            try:
                ctxf, _ = arm_installer(proto, arm)
                with ctxf() as hook:
                    rec["hook"] = hook
                    alias = ALIAS_FOR[proto]
                    out = C.run_single(bench, proto, uid, alias,
                                       max_tokens=MAX_TOKENS, timeout=30.0, retries=0)
                    rec["status"] = "ok"
                    rec["calls"] = len(out.get("calls") or [])
                    rec["raw_io_ok"] = C.check_raw_preserved(out) or "preserved"
                    rec["majority_answer"] = out.get("majority_answer")
                    rec["peer_indices_present"] = sum(
                        1 for c in (out.get("calls") or [])
                        if ("peer_indices" in c) or ("shown" in c) or ("mask_over_peers" in c))
                    rec["prompt_chars"] = [len(" ".join(m.get("content","") for m in (c.get("messages_sent") or [])))
                                           for c in (out.get("calls") or [])]
                    rec["record"] = out
            except BudgetExceeded as e:
                stop_reason = e.reason; rec["status"] = "stopped"; results.append(rec); break
            except Exception as exc:
                rec["status"] = "error"; rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            rec["seconds"] = round(time.time() - t0, 2)
            rec["budget"] = budget.snapshot()["used"]
            results.append(rec)
            budget.persist()

    (run_dir / "RESULTS.jsonl").write_text(
        "\n".join(json.dumps(x) for x in results) + "\n", encoding="utf-8")
    final = budget.snapshot(); final["stop_reason"] = stop_reason or final.get("stop_reason")
    (run_dir / "STOPPED.json" if final["stop_reason"] else run_dir / "FINISHED.json").write_text(
        json.dumps(final, indent=1) + "\n", encoding="utf-8")
    providers.chat = orig_chat
    ok = sum(1 for x in results if x.get("status") == "ok")
    print(f"\n  unit-runs attempted {len(results)}, ok {ok}")
    print(f"  calls {final['used']['calls']}, output {final['used']['output_tokens']}, "
          f"input {final['used']['input_tokens']}, elapsed {final['elapsed_seconds']}s, "
          f"gpu_hours {final['gpu_hours_used']}")
    print(f"  stop reason: {final['stop_reason'] or 'none (plan exhausted)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
