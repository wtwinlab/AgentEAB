"""Compute per-benchmark and leave-one-benchmark-out effects from saved cross-family QA responses. No model calls."""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]
OUT = WORK / "output"
PREV = WORK
for _p in (str(REPO / "code"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from consensus_delivery.adapters import mad_m2 as M2                       # noqa: E402
from consensus_delivery.adapters import official_scorers as OS             # noqa: E402

CONDITIONS = ("no_peers", "full", "answer_only")
PRIMARY = (("answer_only", "full"), ("full", "no_peers"), ("answer_only", "no_peers"))
BOOT_DRAWS = 10000
BOOT_SEED = 20261028

#: how the frozen initial-state strata relate to what the recipient was shown
PEER_GROUP = {
    "correct_unanimous": "peers all correct",
    "correct_majority": "correct answer in the majority",
    "correct_tied": "correct answer tied",
    "correct_minority": "correct answer in the minority",
    "wrong_unanimous": "peers all wrong",
    "wrong_dispersed": "peers wrong and split",
}


def grade(bench: str, ans: Optional[str], ref: Dict[str, Any]) -> Optional[bool]:
    if ans is None or not str(ans).strip():
        return None
    refs = [str(x) for x in (ref.get("references") or [])]
    md = ref.get("metadata") or {}
    r0 = refs[0] if refs else None
    if bench == "gsm8k":
        return bool(OS.gsm8k_correct(ans, r0))
    if bench == "strategyqa":
        return bool(OS.strategyqa_correct(ans, r0))
    if bench in ("gpqa", "mmlu_pro"):
        return bool(OS.choice_correct(ans, r0, int(md.get("n_options", 10) or 10)))
    raise AssertionError(bench)


def boot(by_task: Dict[str, List[float]], rng) -> Dict[str, float]:
    ids = sorted(by_task)
    per = np.array([float(np.mean(by_task[t])) for t in ids])
    draws = per[rng.integers(0, len(per), size=(BOOT_DRAWS, len(per)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return dict(mean=float(per.mean()), lo=float(lo), hi=float(hi), n_tasks=len(ids),
                p=float(min(1.0, max(p, 1.0 / BOOT_DRAWS))))


def load() -> Tuple[Dict[str, Dict[Tuple, Dict[str, Any]]], Dict[str, Any]]:
    inputs = {json.loads(l)["unit"]: json.loads(l)
              for l in open(PREV / "data/qa_replay_inputs.jsonl")}
    refs = json.loads((PREV / "data/qa_task_refs.json").read_text(encoding="utf-8"))
    per_model: Dict[str, Dict[Tuple, Dict[str, Any]]] = {}
    for path in sorted((PREV / "output/raw").glob("qa_*.jsonl")):
        if path.name.startswith("pilot"):
            continue
        model = path.name[len("qa_"):-len(".jsonl")]
        cells: Dict[Tuple, Dict[str, Any]] = {}
        for line in open(path):
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            u = inputs.get(r["unit_id"])
            if u is None:
                continue
            content = r["response"]["choices"][0]["message"].get("content") or ""
            g = grade(u["benchmark"], M2.extract_answer(content, u["answer_kind"]),
                      refs[u["task"]])
            key = (u["panel"], u["prefix_id"], u["agent"], u["decode"])
            cells.setdefault(key, {})[u["condition"]] = dict(
                correct=bool(g is True), gradeable=(g is not None),
                benchmark=u["benchmark"], task=u["task"], init_state=u["init_state"])
        per_model[model] = cells
        print(f"{model}: {len(cells)} cells, "
              f"{sum(len(v) for v in cells.values())} graded conditions", flush=True)
    return per_model, refs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    per_model, _ = load()

    levels, contrasts, loo_rows, mech = [], [], [], []
    for model, cells in sorted(per_model.items()):
        benches = sorted({c[next(iter(c))]["benchmark"] for c in cells.values() if c})

        # ---------------- absolute level per benchmark x condition (exploratory stratification)
        for b in benches:
            for cond in CONDITIONS:
                by_task = collections.defaultdict(list)
                for v in cells.values():
                    if cond in v and v[cond]["benchmark"] == b:
                        by_task[v[cond]["task"]].append(float(v[cond]["correct"]))
                if not by_task:
                    continue
                s = boot(by_task, np.random.default_rng(BOOT_SEED))
                levels.append(dict(recipient=model, benchmark=b, condition=cond,
                                   independent_tasks=s["n_tasks"],
                                   accuracy=round(100 * s["mean"], 3),
                                   lo=round(100 * s["lo"], 3), hi=round(100 * s["hi"], 3),
                                   stratification="exploratory (declared as such)"))

        # ---------------- paired contrast per benchmark, and leave-one-benchmark-out on the pool
        for a, bb in PRIMARY:
            pooled = collections.defaultdict(list)
            per_bench: Dict[str, Dict[str, List[float]]] = {
                b: collections.defaultdict(list) for b in benches}
            for v in cells.values():
                if a not in v or bb not in v:
                    continue
                d = float(v[a]["correct"]) - float(v[bb]["correct"])
                pooled[v[a]["task"]].append(d)
                per_bench[v[a]["benchmark"]][v[a]["task"]].append(d)
            if not pooled:
                continue
            tot = boot(pooled, np.random.default_rng(BOOT_SEED))
            for b in benches:
                if not per_bench[b]:
                    continue
                s = boot(per_bench[b], np.random.default_rng(BOOT_SEED))
                contrasts.append(dict(
                    recipient=model, benchmark=b, contrast=f"{a} - {bb}",
                    independent_tasks=s["n_tasks"], effect_pp=round(100 * s["mean"], 3),
                    lo_pp=round(100 * s["lo"], 3), hi_pp=round(100 * s["hi"], 3),
                    ci_excludes_zero=bool(s["lo"] > 0 or s["hi"] < 0),
                    pooled_effect_pp=round(100 * tot["mean"], 3),
                    stratification="exploratory (declared as such)"))
            for drop in benches:
                kept = {t: d for b in benches if b != drop
                        for t, d in per_bench[b].items()}
                if not kept:
                    continue
                s = boot(kept, np.random.default_rng(BOOT_SEED))
                loo_rows.append(dict(
                    recipient=model, contrast=f"{a} - {bb}", dropped_benchmark=drop,
                    pooled_effect_pp=round(100 * tot["mean"], 3),
                    effect_without_pp=round(100 * s["mean"], 3),
                    shift_pp=round(100 * (s["mean"] - tot["mean"]), 3),
                    lo_pp=round(100 * s["lo"], 3), hi_pp=round(100 * s["hi"], 3),
                    pooled_inside_this_interval=bool(s["lo"] <= tot["mean"] <= s["hi"])))

        # ---------------- correction / regression conditioned on the delivered peers' state
        for cond in ("full", "answer_only"):
            by_state: Dict[str, Dict[str, int]] = collections.defaultdict(
                lambda: dict(n=0, corrected=0, regressed=0, both_right=0, both_wrong=0))
            for v in cells.values():
                if cond not in v or "no_peers" not in v:
                    continue
                st = v[cond]["init_state"]
                g = by_state[st]
                a_ok, b_ok = v[cond]["correct"], v["no_peers"]["correct"]
                g["n"] += 1
                if a_ok and not b_ok:
                    g["corrected"] += 1
                elif b_ok and not a_ok:
                    g["regressed"] += 1
                elif a_ok:
                    g["both_right"] += 1
                else:
                    g["both_wrong"] += 1
            for st, g in sorted(by_state.items()):
                if g["n"] < 30:
                    continue
                mech.append(dict(
                    recipient=model, condition=cond, peer_state=st,
                    peer_state_meaning=PEER_GROUP.get(st, "unreadable or unscorable panel"),
                    cells=g["n"],
                    correction_rate=round(100 * g["corrected"] / g["n"], 3),
                    regression_rate=round(100 * g["regressed"] / g["n"], 3),
                    net_pp=round(100 * (g["corrected"] - g["regressed"]) / g["n"], 3),
                    both_right=g["both_right"], both_wrong=g["both_wrong"]))

    summary = dict(
        source=dict(raw=str((PREV / "output/raw").relative_to(REPO)),
                    inputs=str((PREV / "data/qa_replay_inputs.jsonl").relative_to(REPO)),
                    note="read-only; no re-collection, no extractor or grader change"),
        design=dict(independent_unit="base task", boot_draws=BOOT_DRAWS, boot_seed=BOOT_SEED,
                    pooled_family="declared confirmatory",
                    per_benchmark="declared exploratory",
                    correction_regression=("within a cell, the recipient's own unaided answer is "
                                           "its no_peers outcome; diagnostic only")),
        levels=levels, contrasts_per_benchmark=contrasts,
        leave_one_benchmark_out=loo_rows, correction_regression=mech)
    (OUT / "e4_qa_bydomain.json").write_text(json.dumps(summary, indent=1, sort_keys=True),
                                             encoding="utf-8")
    for name, rows in (("e4_levels", levels), ("e4_contrasts_per_benchmark", contrasts),
                       ("e4_leave_one_benchmark_out", loo_rows),
                       ("e4_correction_regression", mech)):
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    print("\nper-benchmark paired contrasts (exploratory)")
    for r in contrasts:
        mark = "*" if r["ci_excludes_zero"] else " "
        print(f"  {r['recipient']:14s} {r['benchmark']:11s} {r['contrast']:24s} "
              f"{r['effect_pp']:+7.2f} [{r['lo_pp']:+.2f},{r['hi_pp']:+.2f}]{mark} "
              f"(pooled {r['pooled_effect_pp']:+.2f}, n={r['independent_tasks']})")
    print("\nleave-one-benchmark-out on the pooled effect")
    for r in loo_rows:
        print(f"  {r['recipient']:14s} {r['contrast']:24s} drop {r['dropped_benchmark']:11s} "
              f"{r['pooled_effect_pp']:+6.2f} -> {r['effect_without_pp']:+6.2f} "
              f"(shift {r['shift_pp']:+.2f})")
    print("\ncorrection / regression by delivered peer state")
    for r in mech:
        print(f"  {r['recipient']:14s} {r['condition']:12s} {r['peer_state']:26s} "
              f"n={r['cells']:5d} corr={r['correction_rate']:5.2f}% "
              f"regr={r['regression_rate']:5.2f}% net={r['net_pp']:+6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
