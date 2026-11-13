"""Score saved cross-family QA responses on fixed peer inputs. No model calls."""
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
RAW = OUT / "raw"
DATA = WORK / "data"
for _p in (str(REPO / "code"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from consensus_delivery.adapters import mad_m2 as M2                       # noqa: E402
from consensus_delivery.adapters import official_scorers as OS             # noqa: E402

CONDITIONS = ("no_peers", "full", "answer_only")
#: the declared primary comparisons, per recipient
PRIMARY = (("answer_only", "full"), ("full", "no_peers"), ("answer_only", "no_peers"))
BOOT_DRAWS = 10000
BOOT_SEED = 20261028


def grade(bench: str, ans: Optional[str], ref: Dict[str, Any]) -> Optional[bool]:
    """`True` / `False` / `None`. `None` is unscorable and never folded into wrong."""
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


def holm(ps: List[float], alpha: float = 0.05) -> List[bool]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out = [False] * m
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break
    return out


def cluster_boot(by_task: Dict[str, List[float]], rng) -> Dict[str, float]:
    """Resample base tasks, not rows. A repeated decode or a second seat is not new evidence."""
    ids = sorted(by_task)
    per = np.array([np.mean(by_task[t]) for t in ids], dtype=float)
    draws = per[rng.integers(0, len(per), size=(BOOT_DRAWS, len(per)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # two-sided bootstrap p for H0: mean = 0
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return dict(mean=float(per.mean()), lo=float(lo), hi=float(hi),
                p=float(min(1.0, max(p, 1.0 / BOOT_DRAWS))), n_tasks=len(ids))


def main() -> int:
    inputs = {}
    for line in open(DATA / "qa_replay_inputs.jsonl"):
        r = json.loads(line)
        inputs[r["unit"]] = r
    refs = json.loads((DATA / "qa_task_refs.json").read_text(encoding="utf-8"))

    per_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    resources: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for path in sorted(RAW.glob("qa_*.jsonl")):
        if path.name.startswith("pilot"):
            continue
        model = path.name[len("qa_"):-len(".jsonl")]
        cells: Dict[str, Dict[str, Any]] = {}
        attempts = collections.Counter()
        for line in open(path):
            r = json.loads(line)
            attempts[r["status"]] += 1
            if r.get("status") != "ok":
                continue
            u = inputs.get(r["unit_id"])
            if u is None:
                continue
            ch = r["response"]["choices"][0]
            content = ch["message"].get("content") or ""
            ans = M2.extract_answer(content, u["answer_kind"])
            g = grade(u["benchmark"], ans, refs[u["task"]])
            usage = r["response"].get("usage") or {}
            det = usage.get("completion_tokens_details") or {}
            cells[r["unit_id"]] = dict(
                u, answer=ans, correct=bool(g is True), gradeable=(g is not None),
                truncated=(ch.get("finish_reason") == "length"),
                empty_final=not content.strip(),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                reasoning_tokens=det.get("reasoning_tokens"),
                latency_s=r.get("latency_s"))
            resources[model].append(dict(
                condition=u["condition"], benchmark=u["benchmark"],
                prompt_tokens=usage.get("prompt_tokens") or 0,
                completion_tokens=usage.get("completion_tokens") or 0,
                reasoning_tokens=det.get("reasoning_tokens") or 0,
                latency_s=r.get("latency_s") or 0.0))
        per_model[model] = cells
        print(f"{model}: ok={len(cells)} statuses={dict(attempts)}", flush=True)

    # ---------------------------------------------------------------- descriptive levels
    levels = []
    for model, cells in sorted(per_model.items()):
        for cond in CONDITIONS:
            rs = [c for c in cells.values() if c["condition"] == cond]
            if not rs:
                continue
            by_task = collections.defaultdict(list)
            for c in rs:
                by_task[c["task"]].append(float(c["correct"]))
            rng = np.random.default_rng(BOOT_SEED)
            v = cluster_boot(by_task, rng)
            levels.append(dict(
                recipient=model, condition=cond, requests=len(rs),
                independent_tasks=v["n_tasks"],
                accuracy=round(100 * v["mean"], 3),
                accuracy_lo=round(100 * v["lo"], 3), accuracy_hi=round(100 * v["hi"], 3),
                unreadable_share=round(sum(1 for c in rs if not c["gradeable"]) / len(rs), 5),
                empty_final_share=round(sum(1 for c in rs if c["empty_final"]) / len(rs), 5),
                truncated_share=round(sum(1 for c in rs if c["truncated"]) / len(rs), 5),
                mean_prompt_tokens=round(float(np.mean([c["prompt_tokens"] or 0 for c in rs])), 1),
                mean_completion_tokens=round(float(np.mean([c["completion_tokens"] or 0 for c in rs])), 1),
                mean_reasoning_tokens=round(float(np.mean([c["reasoning_tokens"] or 0 for c in rs])), 1),
                mean_latency_s=round(float(np.mean([c["latency_s"] or 0 for c in rs])), 3)))

    # ---------------------------------------------------------------- paired primary family
    contrasts = []
    for model, cells in sorted(per_model.items()):
        # a cell is (panel, prefix, seat, decode); conditions are paired inside it
        key = lambda c: (c["panel"], c["prefix_id"], c["agent"], c["decode"])          # noqa: E731
        by_cell: Dict[Tuple, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
        for c in cells.values():
            by_cell[key(c)][c["condition"]] = c
        for a, b in PRIMARY:
            pairs = [(v[a], v[b]) for v in by_cell.values() if a in v and b in v]
            if not pairs:
                continue
            by_task = collections.defaultdict(list)
            for ca, cb in pairs:
                by_task[ca["task"]].append(float(ca["correct"]) - float(cb["correct"]))
            rng = np.random.default_rng(BOOT_SEED)
            v = cluster_boot(by_task, rng)
            ben = sum(1 for ca, cb in pairs if ca["correct"] and not cb["correct"])
            harm = sum(1 for ca, cb in pairs if cb["correct"] and not ca["correct"])
            same = len(pairs) - ben - harm
            resid = sum(1 for ca, _ in pairs if not ca["correct"])
            contrasts.append(dict(
                recipient=model, contrast=f"{a} - {b}", dimension=
                {"answer_only - full": "content", "full - no_peers": "access",
                 "answer_only - no_peers": "access+content"}[f"{a} - {b}"],
                paired_cells=len(pairs), independent_tasks=v["n_tasks"],
                effect_pp=round(100 * v["mean"], 3),
                lo_pp=round(100 * v["lo"], 3), hi_pp=round(100 * v["hi"], 3),
                ci_half_width_pp=round(100 * (v["hi"] - v["lo"]) / 2, 3),
                ci_excludes_zero=bool(v["lo"] > 0 or v["hi"] < 0),
                p_boot=v["p"],
                benefited_cells=ben, harmed_cells=harm, unchanged_cells=same,
                residual_errors=resid,
                residual_error_share=round(resid / len(pairs), 5)))
    if contrasts:
        flags = holm([c["p_boot"] for c in contrasts])
        for c, f in zip(contrasts, flags):
            c["holm_significant_in_family_of_%d" % len(contrasts)] = bool(f)

    # ---------------------------------------------------------------- decode replication
    decode = []
    for model, cells in sorted(per_model.items()):
        base = {}
        for c in cells.values():
            base.setdefault((c["panel"], c["prefix_id"], c["agent"], c["condition"]), {})[c["decode"]] = c
        both = [v for v in base.values() if 0 in v and 1 in v]
        if both:
            agree = sum(1 for v in both if v[0]["correct"] == v[1]["correct"])
            decode.append(dict(recipient=model, cells_with_two_decodes=len(both),
                               outcome_agreement=round(agree / len(both), 5),
                               implied_decode_sd_pp=round(
                                   100 * float(np.sqrt((1 - agree / len(both)) / 2)), 3)))

    summary = dict(
        levels=levels, primary_contrasts=contrasts, decode_replication=decode,
        design=dict(conditions=list(CONDITIONS), primary=[f"{a} - {b}" for a, b in PRIMARY],
                    independent_unit="base task", boot_draws=BOOT_DRAWS, boot_seed=BOOT_SEED,
                    multiplicity="Holm within the primary family only",
                    answer_source="final channel (message.content) only; reasoning never parsed",
                    abstention="charged as failure and also reported separately"),
        replay_ceiling=dict(
            what=("the published QA replay gain came from excluding unreadable candidates before "
                  "voting. That gain is bounded above by the share of deliveries that are "
                  "unreadable, so the per-recipient unreadable share below is the ceiling on it."),
            by_recipient={r["recipient"]: max(
                x["unreadable_share"] for x in levels if x["recipient"] == r["recipient"])
                for r in levels}))
    (OUT / "qa2_scored.json").write_text(json.dumps(summary, indent=1, sort_keys=True),
                                         encoding="utf-8")
    for name, rows in (("qa2_levels", levels), ("qa2_contrasts", contrasts)):
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    print("\nlevels")
    for r in levels:
        print(f"  {r['recipient']:14s} {r['condition']:12s} n={r['requests']:5d} "
              f"acc={r['accuracy']:6.2f} [{r['accuracy_lo']:.2f},{r['accuracy_hi']:.2f}] "
              f"unreadable={r['unreadable_share']:.4f} trunc={r['truncated_share']:.4f} "
              f"in={r['mean_prompt_tokens']:.0f} out={r['mean_completion_tokens']:.0f}")
    print("\nprimary contrasts (FULL family, Holm within it)")
    for r in contrasts:
        print(f"  {r['recipient']:14s} {r['contrast']:26s} {r['effect_pp']:+7.2f} pp "
              f"[{r['lo_pp']:+.2f},{r['hi_pp']:+.2f}] half={r['ci_half_width_pp']:.2f} "
              f"holm={r[[k for k in r if k.startswith('holm_')][0]]} "
              f"benefit={r['benefited_cells']} harm={r['harmed_cells']} "
              f"residual_err={r['residual_error_share']:.3f}")
    for r in decode:
        print(f"\ndecode replication {r['recipient']}: {r['cells_with_two_decodes']} cells, "
              f"outcome agreement {r['outcome_agreement']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
