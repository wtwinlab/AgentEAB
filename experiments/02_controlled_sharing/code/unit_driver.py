#!/usr/bin/env python3
"""One unit, one protocol, one condition -- across both task families, with the condition applied at the prompt.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Any, Callable, Dict, List, Optional

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE,
           ROOT / "experiments/04_sql_selection_generation/code",
           ROOT / "experiments/06_cost_resources/code",
           ROOT / "experiments/05_matching_downstream/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import answer_extract as AX                                            # noqa: E402
import arm_transform as AT                                              # noqa: E402

PROD_MAX_TOKENS = 1024          # the collectors' value; never reduced to fit a cap
MATCHING = ("wdc_products", "geographic_settlements", "musicbrainz20k")
QA = ("gsm8k", "strategyqa", "mmlu_pro", "gpqa", "squad_v2")
#: QA shapes, from collect_backbone.PROTOCOLS; the matching shapes come from
#: collect_all_matching.SPECS. Kept separate because the released configurations differ per family.
QA_KW = {
    "du_mad": dict(agents=3, rounds=2),
    "madm2_objective": dict(agents=3, rounds=2, prune="objective", temperature=1.0, top_p=1.0),
    "madm2_naive": dict(agents=3, rounds=2, prune="naive", temperature=1.0, top_p=1.0),
    "bccs": dict(agents=7, rounds=3, n_clusters=3, top_k=2, temperature=0.7, belief="intended"),
    "ford": dict(max_rounds=6, n_options=2),
    "moa": dict(temperature=0.7),
}
_READY = False

#: letters that can introduce an option in an mcq question body
_OPTION_RE = re.compile(r"(?:^|\n)\s*([A-J])[.)]\s")


def mcq_option_count(question: str) -> Optional[int]:
    """How many options this task actually offers, or None when it is not an mcq body.
    """
    letters = sorted({m.group(1) for m in _OPTION_RE.finditer(question or "")})
    if len(letters) < 2:
        return None
    # the options are a prefix of A..J; anything else means this is not an option list
    expect = [chr(ord("A") + i) for i in range(len(letters))]
    return len(letters) if letters == expect else None



def _ensure_installed():
    global _READY
    if _READY:
        return
    import collect_all_matching as C
    C._ensure_installed()
    _READY = True


def _qa_block(bench: str) -> List[Dict[str, Any]]:
    p = ROOT / "data" / "frozen" / f"{bench}_block02.json"
    blob = json.loads(p.read_text(encoding="utf-8"))
    return blob if isinstance(blob, list) else (blob.get("records") or [])


def install_arm(protocol: str, arm: str, trace_sink: List[Dict[str, Any]], kind: str,
                budget=None, seed: int = 20261019, task_name: str = ""):
    """Wrap `providers.chat` so every request passes the condition transform AND the budget guard.
    """
    from consensus_delivery import providers
    orig = providers.chat
    seq = {"n": 0}
    # every reply this execution produced. MoA assembles its aggregator prompt from exactly these
    # strings, and the `\n<n>.
    replies: List[str] = []

    def wrapped(**kw):
        msgs = list(kw.get("messages") or [])
        before = [dict(m) for m in msgs]
        new, tr = AT.transform(protocol, arm, msgs, kind=kind, call_seq=seq["n"], seed=seed,
                               benchmark=task_name, observed_replies=list(replies))
        kw["messages"] = new
        tr["call_seq"] = seq["n"]
        tr["chars_before"] = sum(len(m.get("content") or "") for m in before)
        tr["chars_after"] = sum(len(m.get("content") or "") for m in new)
        seq["n"] += 1
        trace_sink.append(tr)
        if budget is None:
            text, rec = orig(**kw)
            replies.append(text or "")
            return text, rec
        mt = int(kw.get("max_tokens") or PROD_MAX_TOKENS)
        est_in = tr["chars_after"] // 3
        timeout = budget.reserve(mt, est_prompt_tokens=est_in)
        kw["timeout_s"] = min(float(kw.get("timeout_s") or timeout), timeout)
        try:
            text, rec = orig(**kw)
        except Exception:
            budget.settle(mt, est_in, 0)
            raise
        budget.settle(mt, rec.prompt_tokens or 0, rec.completion_tokens or 0)
        replies.append(text or "")
        return text, rec

    providers.chat = wrapped
    for mod in ("bccs", "du_mad", "mad_m2", "ford", "moa", "reconcile", "dmad",
                "debate_or_vote", "liang_mad"):
        try:
            m = __import__(f"consensus_delivery.adapters.{mod}", fromlist=["chat"])
            if hasattr(m, "chat"):
                m.chat = wrapped
        except Exception:
            pass
    return orig


def restore_chat(orig):
    from consensus_delivery import providers
    providers.chat = orig
    for mod in ("bccs", "du_mad", "mad_m2", "ford", "moa", "reconcile", "dmad",
                "debate_or_vote", "liang_mad"):
        try:
            m = __import__(f"consensus_delivery.adapters.{mod}", fromlist=["chat"])
            if hasattr(m, "chat"):
                m.chat = orig
        except Exception:
            pass


def run_unit(benchmark: str, protocol: str, unit: str, alias: str, *,
             arm: str = "FULL", max_tokens: int = PROD_MAX_TOKENS,
             timeout: float = 120.0, budget=None,
             seed: int = 20261019) -> Dict[str, Any]:
    """Run one unit and return {record, arm_traces, config}. Raises on a real failure.
    """
    _ensure_installed()
    import collect_all_matching as C
    import matching_tasks as MT

    traces: List[Dict[str, Any]] = []
    if benchmark in MATCHING:
        kind = AX.KIND_BY_BENCHMARK.get(benchmark, MT.KIND)
        orig = install_arm(protocol, arm, traces, kind, budget=budget, seed=seed,
                           task_name=MT.BENCH)
        try:
            rec = C.run_single(benchmark, protocol, unit, alias,
                               max_tokens=max_tokens, timeout=timeout, retries=0)
        finally:
            restore_chat(orig)
        cfg = dict(C.SPECS[protocol]["kw"])
        cfg.update(benchmark=benchmark, family="matching", alias=alias, max_tokens=max_tokens,
                   answer_kind=kind, task_name=MT.BENCH, seed=seed)
        return {"record": rec, "arm_traces": traces, "config": cfg}

    if benchmark in QA:
        spec = QA_KW.get(protocol)
        if spec is None:
            raise KeyError(f"no QA configuration frozen for {protocol}")
        mod_name = {"bccs": "bccs", "du_mad": "du_mad", "madm2_objective": "mad_m2",
                    "madm2_naive": "mad_m2", "ford": "ford", "moa": "moa"}[protocol]
        mod = __import__(f"consensus_delivery.adapters.{mod_name}", fromlist=["run_one"])
        tasks = _qa_block(benchmark)
        t = next((x for x in tasks if x.get("example_id") == unit), None)
        if t is None:
            raise KeyError(f"{unit} is not in the frozen {benchmark} block")
        qfun = getattr(mod, "task_question", None)
        question = qfun(benchmark, t) if qfun else t["inputs"][0]
        kw = dict(spec)
        kw.update(max_tokens=max_tokens, timeout_s=timeout, retries=0)
        if protocol in ("ford", "moa"):
            roster = ("llama3-8b-g1", "qwen25-7b-g2") if protocol == "ford" else \
                ("llama3-8b-g1", "qwen25-7b-g2", "gemma4-g5")
            kw["aliases" if protocol == "ford" else "references"] = \
                tuple(roster) if protocol == "ford" else list(roster)
            if protocol == "moa":
                kw["aggregator"] = "qwen25-7b-g2"
        else:
            kw["alias"] = alias
        # the adapter's own ANSWER_KIND is empty for Du-MAD and FORD, which would hand extraction
        # an empty kind on QA and make it fall back to the matching contract; take the kind from
        # the central map and only consult the adapter as a cross-check
        kind = AX.KIND_BY_BENCHMARK.get(benchmark) or getattr(mod, "ANSWER_KIND", {}).get(
            benchmark, "")
        adapter_kind = getattr(mod, "ANSWER_KIND", {}).get(benchmark)
        if adapter_kind and adapter_kind != kind:
            raise AssertionError(f"answer-kind disagreement for {protocol}/{benchmark}: "
                                 f"central {kind!r} vs adapter {adapter_kind!r}")
        # FORD renders its answer contract from `n_options`; the frozen 2 is the matching
        # family's value and silently truncates an mcq task's option list.
        if protocol == "ford" and kind == "mcq":
            n_opt = mcq_option_count(question)
            if n_opt:
                kw["n_options"] = n_opt
                cfg_extra = {"n_options_from_task": n_opt}
            else:
                cfg_extra = {"n_options_from_task": None}
        else:
            cfg_extra = {}
        orig = install_arm(protocol, arm, traces, kind, budget=budget, seed=seed,
                           task_name=benchmark)
        try:
            rec = mod.run_one(question, task_id=unit, benchmark=benchmark, **kw) \
                if "task_id" in mod.run_one.__code__.co_varnames else \
                mod.run_one(question, benchmark=benchmark, **kw)
        finally:
            restore_chat(orig)
        cfg = dict(spec)
        cfg.update(benchmark=benchmark, family="qa", alias=alias, max_tokens=max_tokens,
                   answer_kind=kind, seed=seed, **cfg_extra)
        return {"record": rec, "arm_traces": traces, "config": cfg}

    raise KeyError(f"unknown benchmark {benchmark}")


def supported(benchmark: str, protocol: str, arm: str) -> Optional[str]:
    """None when the (benchmark, protocol, condition) can run; otherwise the reason it cannot."""
    if benchmark in MATCHING:
        import collect_all_matching as C
        if protocol not in C.SPECS:
            return f"{protocol} has no frozen matching configuration"
    elif benchmark in QA:
        if protocol not in QA_KW:
            return f"{protocol} has no frozen QA configuration"
        p = ROOT / "data" / "frozen" / f"{benchmark}_block02.json"
        if not p.exists():
            return f"frozen block missing: {p}"
    else:
        return f"unknown benchmark {benchmark}"
    if arm not in AT.ARMS:
        return f"unknown arm {arm}"
    if protocol == "ford" and arm == "UNIFSEL":
        return ("FORD renders one peer, so there is nothing to choose between; its selection/use "
                "factor is the final-answer rule, verified from the record by "
                "arm_transform.use_arm_final_answer rather than by a prompt edit")
    return None
