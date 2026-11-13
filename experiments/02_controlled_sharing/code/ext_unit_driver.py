#!/usr/bin/env python3
"""One unit, one protocol, one condition, for the two extension benchmarks.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE, ROOT / "experiments/06_cost_resources/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import answer_extract as AX                                              # noqa: E402
import ext_arms as EA                                                    # noqa: E402
import ext_design as D                                                   # noqa: E402
import unit_driver as UD                                                 # noqa: E402

PROD_MAX_TOKENS = D.PROD_MAX_TOKENS

#: module name per protocol, same mapping the main driver uses
MODULE = {"bccs": "bccs", "du_mad": "du_mad", "madm2_objective": "mad_m2",
          "madm2_naive": "mad_m2", "ford": "ford", "moa": "moa"}
#: the frozen rosters, identical to the main design's: the roster IS the method for these two
ROSTER = {"ford": ("llama3-8b-g1", "qwen25-7b-g2"),
          "moa": ("llama3-8b-g1", "qwen25-7b-g2", "gemma4-g5")}
MOA_AGGREGATOR = "qwen25-7b-g2"

_READY: Dict[str, Any] = {}


def _poison_execution() -> None:
    """Make an execution call inside a collection process impossible rather than merely unlikely."""
    import sql_common as SQ
    if getattr(SQ, "_ext_poisoned", False):
        return

    def _refuse(*_a, **_kw):
        raise RuntimeError(
            "execution is forbidden inside a collection process: no arm may receive a database "
            "result or an evaluator verdict. Run evaluation in ext_eval_sql.py instead.")

    SQ._ext_execute_real = SQ.execute
    SQ._ext_gold_sql_real = SQ.gold_sql
    SQ._ext_execution_match_real = SQ.execution_match
    SQ.execute = _refuse
    SQ.gold_sql = _refuse
    SQ.execution_match = _refuse
    SQ._ext_poisoned = True


def _ensure_endpoints() -> str:
    """Install the endpoint registry once per process, tolerating an inherited one.
    """
    import collect_all_matching as C
    if getattr(C, "_INSTALLED", False):
        return "already installed in this process"
    try:
        UD._ensure_installed()
        return "installed"
    except ValueError as exc:
        if "duplicate alias" not in str(exc):
            raise
        import matching_tasks as MT
        MT.install()                     # the task adaptation is separately idempotent
        C._INSTALLED = True
        return "registry inherited from the parent process; install skipped"


def ensure_installed(family: str) -> Dict[str, Any]:
    """Per-process, idempotent setup for one task family."""
    if family in _READY:
        return _READY[family]
    endpoints = _ensure_endpoints()              # endpoint registry + matching task adaptation
    EA.install()                                 # MoA's exact reference split
    if family == "sql":
        import ext_sql_tasks as EST
        info = EST.install()
        _poison_execution()
    else:
        info = {"family": "qa", "note": "no task-level registration needed: every adapter already "
                                        "declares gpqa"}
    info = dict(info)
    info["endpoints"] = endpoints
    _READY[family] = info
    return info


# ---------------------------------------------------------------- the two populations
def _qa_block(bench: str) -> List[Dict[str, Any]]:
    p = ROOT / "data" / "frozen" / f"{bench}_block02.json"
    blob = json.loads(p.read_text(encoding="utf-8"))
    return blob if isinstance(blob, list) else (blob.get("records") or [])


_QA_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}
_SQL_CACHE: Dict[str, Dict[str, Any]] = {}


def qa_task(benchmark: str, unit: str) -> Dict[str, Any]:
    if benchmark not in _QA_CACHE:
        _QA_CACHE[benchmark] = {r["example_id"]: r for r in _qa_block(benchmark)}
    t = _QA_CACHE[benchmark].get(unit)
    if t is None:
        raise KeyError(f"{unit} is not in the frozen {benchmark} block")
    return t


def sql_task(unit: str) -> Dict[str, Any]:
    """The question text for one BIRD unit. Reads the schema; never the gold or a result set."""
    if not _SQL_CACHE:
        import sql_common as SQ
        qs = json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text(encoding="utf-8"))
        for q in qs:
            _SQL_CACHE[f"bird|{q['question_id']}"] = q
    q = _SQL_CACHE.get(unit)
    if q is None:
        raise KeyError(f"{unit} is not in the BIRD dev population")
    import sql_common as SQ
    import sql_tasks as ST
    text = ST.sql_question(q["question"], SQ.schema_text(q["db_id"]), q.get("evidence") or "")
    return {"question": text, "db_id": q["db_id"], "difficulty": q.get("difficulty"),
            "question_id": int(q["question_id"])}


# ---------------------------------------------------------------- the reply recorder
_RECORDER_MODS = ("bccs", "du_mad", "mad_m2", "ford", "moa")


def _install_reply_recorder():
    """Wrap `providers.chat` so every reply text is observed, and return the original.
    """
    from consensus_delivery import providers
    orig = providers.chat

    def wrapped(**kw):
        text, rec = orig(**kw)
        EA.observe_reply(text)
        return text, rec

    providers.chat = wrapped
    for name in _RECORDER_MODS:
        try:
            m = __import__(f"consensus_delivery.adapters.{name}", fromlist=["chat"])
            if hasattr(m, "chat"):
                m.chat = wrapped
        except Exception:
            pass
    return orig


def _restore_reply_recorder(orig) -> None:
    from consensus_delivery import providers
    providers.chat = orig
    for name in _RECORDER_MODS:
        try:
            m = __import__(f"consensus_delivery.adapters.{name}", fromlist=["chat"])
            if hasattr(m, "chat"):
                m.chat = orig
        except Exception:
            pass


# ---------------------------------------------------------------- the call
def run_unit(benchmark: str, protocol: str, unit: str, alias: str, *,
             arm: str = "FULL", max_tokens: int = PROD_MAX_TOKENS,
             timeout: float = 180.0, budget=None, seed: int = 20261020) -> Dict[str, Any]:
    """Run one unit-run and return `{record, arm_traces, config}`. Raises on a real failure."""
    family = "sql" if benchmark in D.SQL_BENCH else "qa"
    install_info = ensure_installed(family)
    traces: List[Dict[str, Any]] = []

    mod = __import__(f"consensus_delivery.adapters.{MODULE[protocol]}", fromlist=["run_one"])

    if family == "qa":
        kw = D.qa_kw(benchmark, protocol)
        task = qa_task(benchmark, unit)
        kind = AX.KIND_BY_BENCHMARK.get(benchmark)
        meta = {"n_options": (task.get("metadata") or {}).get("n_options"),
                "domain": (task.get("metadata") or {}).get("high_level_domain"),
                "subdomain": (task.get("metadata") or {}).get("subdomain")}
        if meta["n_options"] and kw.get("n_options") not in (None, meta["n_options"]):
            raise AssertionError(
                f"{protocol}/{benchmark}/{unit}: configured n_options={kw['n_options']} but the "
                f"task has {meta['n_options']} options. The instruction would name option letters "
                f"the task does not offer, or omit letters it does.")
        qfun = getattr(mod, "task_question", None)
        question = qfun(benchmark, task) if qfun else task["inputs"][0]
    else:
        # the SQL configuration is the one already frozen for this family by `collect_sql_v2`
        # (the same values `collect_all_matching` carries, but this is the SQL run's own table);
        # only the three missing adaptations are new, and they live in ext_sql_tasks
        import collect_sql_v2 as CS
        if protocol not in CS.SPECS:
            raise KeyError(f"no frozen SQL configuration for {protocol}")
        kw = dict(CS.SPECS[protocol]["kw"])
        task = sql_task(unit)
        kind = "sql"
        meta = {"db_id": task["db_id"], "difficulty": task["difficulty"],
                "question_id": task["question_id"]}
        question = task["question"]

    kw = dict(kw)
    kw.update(max_tokens=max_tokens, timeout_s=timeout, retries=0)
    if protocol in ROSTER:
        if protocol == "ford":
            kw["aliases"] = tuple(ROSTER["ford"])
        else:
            kw["references"] = list(ROSTER["moa"])
            kw["aggregator"] = MOA_AGGREGATOR
    else:
        kw["alias"] = alias

    # the adapter's own answer kind must agree with the central map, or extraction silently falls
    # back to the matching contract
    adapter_kind = getattr(mod, "ANSWER_KIND", {}).get(benchmark)
    if adapter_kind and adapter_kind != kind:
        raise AssertionError(f"answer-kind disagreement for {protocol}/{benchmark}: central "
                             f"{kind!r} vs adapter {adapter_kind!r}")

    # MoA's reference replies are therefore observed before its aggregator prompt is built,
    # which is what makes the exact split exact.
    EA.reset()
    recorder_orig = _install_reply_recorder()
    orig = UD.install_arm(protocol, arm, traces, kind, budget=budget, seed=seed,
                          task_name=benchmark)
    try:
        if "task_id" in mod.run_one.__code__.co_varnames:
            rec = mod.run_one(question, task_id=unit, benchmark=benchmark, **kw)
        else:
            rec = mod.run_one(question, benchmark=benchmark, **kw)
    finally:
        UD.restore_chat(orig)
        _restore_reply_recorder(recorder_orig)
    for t in traces:
        t.setdefault("moa_split", None)
    if protocol == "moa":
        summary = EA.split_summary()
        for t in traces:
            t["moa_split"] = summary

    cfg = dict(kw)
    cfg.update(benchmark=benchmark, family=family, alias=alias, max_tokens=max_tokens,
               answer_kind=kind, seed=seed, unit_meta=meta,
               run_id=D.RUN_ID, task_install=install_info,
               config_source=("ext_design.QA_KW_GPQA" if family == "qa"
                              else "collect_all_matching.SPECS + ext_sql_tasks"),
               config_deviations=(D.QA_KW_GPQA_DEVIATIONS if family == "qa" else {}))
    return {"record": rec, "arm_traces": traces, "config": cfg}


def supported(benchmark: str, protocol: str, arm: str) -> Optional[str]:
    """None when the cell can run; otherwise the reason. Adds the SQL-side adapter check."""
    why = D.supported(benchmark, protocol, arm)
    if why:
        return why
    if benchmark in D.QA_BENCH:
        if protocol not in D.QA_KW_GPQA:
            return f"{protocol} has no frozen GPQA configuration"
        if not (ROOT / "data" / "frozen" / f"{benchmark}_block02.json").exists():
            return f"frozen block missing for {benchmark}"
        return None
    if benchmark in D.SQL_BENCH:
        import collect_sql_v2 as CS
        if protocol not in CS.SPECS:
            return f"{protocol} has no frozen configuration in collect_sql_v2.SPECS"
        return None
    return f"unknown benchmark {benchmark}"
