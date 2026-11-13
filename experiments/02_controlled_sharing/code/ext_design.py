#!/usr/bin/env python3
"""The frozen design of the two extension benchmarks: samples, per-protocol configuration, panels.

    python3 ext_design.py freeze      # write data/frozen/extension/*
    python3 ext_design.py show        # the design and its scale, offline
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE, ROOT / "experiments/06_cost_resources/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RUN_ID = "benchmark_extension_2026-10-20"
# Delivered layout: the extension's records, analysis and frozen sample live under their
# research-question locations rather than under a run-id directory.
OUT = ROOT / "experiments/02_controlled_sharing/extension_analysis"
FROZEN_DIR = ROOT / "data/frozen/extension"

#: short tasks first, as ordered.
BENCH_ORDER = ["gpqa", "sql_bird_v2"]
QA_BENCH = ("gpqa",)
SQL_BENCH = ("sql_bird_v2",)

PROTOS = ["ford", "moa", "du_mad", "madm2_objective", "bccs"]
ARMS = ["FULL", "NOPEER", "ANSONLY", "UNIFSEL"]
#: identical to the main design: homogeneous protocols block on backbone, heterogeneous ones run
#: their frozen roster, which must not vary or the selection condition confounds with the roster
PANELS = {"du_mad": ["llama3-8b", "qwen2.5-7b"], "madm2_objective": ["llama3-8b", "qwen2.5-7b"],
          "bccs": ["llama3-8b", "qwen2.5-7b"], "ford": ["roster"], "moa": ["roster"]}
REPEATS = 2
ALIAS = {"llama3-8b": "llama3-8b-g1", "qwen2.5-7b": "qwen25-7b-g1b", "roster": "llama3-8b-g1"}
PROD_MAX_TOKENS = 1024
#: cheap protocols first inside a benchmark, so a stop leaves the most finished cells.
COST = {"ford": 1.0, "moa": 4.0, "du_mad": 6.0, "madm2_objective": 8.0, "bccs": 18.0}

#: GPQA sample: the rule is the main design's rule, applied to GPQA
GPQA_N_UNITS = 100
#: SQL sample: see the module docstring
SQL_N_UNITS = 200
SQL_SEED = 20261020

# --------------------------------------------------------------------------- per-protocol config
#: The frozen QA configuration **per benchmark**, because one of its values is not
#: benchmark-independent.
QA_KW_GPQA: Dict[str, Dict[str, Any]] = {
    "du_mad": dict(agents=3, rounds=2, n_options=4),
    "madm2_objective": dict(agents=3, rounds=2, prune="objective", temperature=1.0, top_p=1.0),
    "bccs": dict(agents=7, rounds=3, n_clusters=3, top_k=2, temperature=0.7, belief="intended"),
    "ford": dict(max_rounds=6, n_options=4),
    "moa": dict(temperature=0.7),
}
#: why each deviation from `unit_driver.QA_KW` exists, carried into the record's config
QA_KW_GPQA_DEVIATIONS = {
    "ford.n_options": {
        "unit_driver_value": 2, "used": 4,
        "reason": "n_options=2 is the matching family's binary value; on a 4-option MCQ it makes "
                  "ford.instruction return the two-choice string, so the 100/200 GPQA tasks whose "
                  "gold is C or D cannot be answered at all",
        "evidence": "results/batch2_matrix/gpqa100_ford_roster.json stores prompts ending "
                    "'Answer: (A|B|C|D) is more plausible.', i.e. the released cell used 4",
    },
    "du_mad.n_options": {
        "unit_driver_value": "10 (adapter default, not set in QA_KW)", "used": 4,
        "reason": "du_mad.extract_for builds its allowed letter set from n_options. GPQA has 4 "
                  "options, so A-J would accept an out-of-range letter as an answer instead of "
                  "recording that no valid option was stated",
        "evidence": "data/frozen/gpqa_block02.manifest + every task's metadata.n_options == 4",
    },
    "moa.extract": {
        "unit_driver_value": "n/a", "used": "unchanged",
        "reason": "moa.extract hardcodes letter_answer(text, 10) and takes no n_options. It is "
                  "left as released; a peer letter in E-J is rendered as the peer stated it and "
                  "scores as wrong, which is honest. Recorded here so the asymmetry with du_mad "
                  "is visible rather than discovered later",
        "evidence": "code/consensus_delivery/adapters/moa.py extract()",
    },
}

#: FORD runs at `temperature=0.0`, hardcoded in its released `run_one`. Its two repeats therefore
#: differ only by server-side nondeterminism (batch composition), not by sampling.
FORD_IS_GREEDY = True


def qa_kw(benchmark: str, protocol: str) -> Dict[str, Any]:
    if benchmark == "gpqa":
        return dict(QA_KW_GPQA[protocol])
    raise KeyError(f"no frozen QA configuration for {benchmark}")


# --------------------------------------------------------------------------- GPQA sample
def _qa_block(bench: str) -> List[Dict[str, Any]]:
    p = ROOT / "data" / "frozen" / f"{bench}_block02.json"
    blob = json.loads(p.read_text(encoding="utf-8"))
    return blob if isinstance(blob, list) else (blob.get("records") or [])


#: the released cells whose task coverage defines the shared-100 rule for GPQA
GPQA_RELEASED_CELLS = {
    "bccs|llama3-8b": "results/batch2_matrix/gpqa100_bccs_llama3-8b.json",
    "bccs|qwen2.5-7b": "results/batch2_matrix/gpqa100_bccs_qwen25-7b.json",
    "madm2_objective|llama3-8b": "results/batch2_matrix/gpqa100_madm2_objective_llama3-8b.json",
    "madm2_objective|qwen2.5-7b": "results/batch2_matrix/gpqa100_madm2_objective_qwen25-7b.json",
    "ford|roster": "results/batch2_matrix/gpqa100_ford_roster.json",
    "moa|roster": "results/batch2_matrix/gpqa100_moa_roster.json",
    "du_mad|llama3-8b": "results/main_matrix/gpqa200_dumad_llama3-8b.json",
    "du_mad|qwen2.5-7b": "results/main_matrix/gpqa200_dumad_qwen25-7b.json",
}


def _cell_ids(rel: str) -> List[str]:
    blob = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    recs = blob if isinstance(blob, list) else (blob.get("records") or [])
    return [r.get("example_id") for r in recs]


def freeze_gpqa() -> Dict[str, Any]:
    """The 100 GPQA tasks shared by all five protocols' released coverage, and the audit of that.
    """
    block = _qa_block("gpqa")
    order = [r["example_id"] for r in block]
    per_cell: Dict[str, int] = {}
    inter: Optional[set] = None
    for name, rel in GPQA_RELEASED_CELLS.items():
        ids = set(_cell_ids(rel))
        per_cell[name] = len(ids)
        inter = ids if inter is None else (inter & ids)
    inter = inter or set()
    shared = [u for u in order if u in inter]
    if len(shared) != GPQA_N_UNITS:
        raise SystemExit(f"GPQA shared coverage is {len(shared)} tasks, the design expects "
                         f"{GPQA_N_UNITS}; the rule, not the number, is what is frozen -- "
                         f"re-examine the released cells before changing this")
    man = json.loads((ROOT / "data/frozen/gpqa_block02.manifest.json").read_text(encoding="utf-8"))
    meta = {r["example_id"]: (r.get("metadata") or {}) for r in block}
    nopt = collections.Counter(meta[u].get("n_options") for u in shared)
    if set(nopt) != {4}:
        raise SystemExit(f"GPQA option counts in the sample are {dict(nopt)}; the FORD and Du-MAD "
                         f"n_options=4 choice assumes a single option count")
    return {
        "benchmark": "gpqa", "family": "qa", "n_units": len(shared),
        "unit_ids": shared,
        "sha256_16": hashlib.sha256(json.dumps(shared, sort_keys=True).encode()
                                    ).hexdigest()[:16],
        "selection_rule": ("the tasks covered by ALL FIVE protocols' released GPQA cells -- the "
                           "same rule FROZEN_SAMPLE_5X5.json used for gsm8k / strategyqa / "
                           "mmlu_pro. No RNG. On GPQA this intersection is the first 100 ids of "
                           "the block, because the released 100-task cells were collected with "
                           "--limit 100 over the block order"),
        "per_released_cell_task_count": per_cell,
        "block": {"path": "data/frozen/gpqa_block02.json",
                  "block_sha256": man["block_sha256"], "repo": man["repo"],
                  "config": man["config"], "split": man["split"],
                  "split_size": man["split_size"], "offset": man["offset"], "n": man["n"],
                  "upstream_note": man["upstream_note"]},
        "option_counts": dict(nopt),
        "gold_letters": dict(collections.Counter(
            (next(r for r in block if r["example_id"] == u)["references"] or [None])[0]
            for u in shared)),
        "domains": dict(collections.Counter(meta[u].get("high_level_domain") for u in shared)),
        "answer_kind": "mcq",
        "analysis_unit": "recipient x task", "cluster_on": "task",
    }


def gpqa_gold() -> Dict[str, str]:
    """Reference option letters. Read only by the evaluator; never on a prompt-building path."""
    return {r["example_id"]: str((r.get("references") or [None])[0]).strip()
            for r in _qa_block("gpqa") if r.get("references")}


# --------------------------------------------------------------------------- SQL sample
def freeze_sql() -> Dict[str, Any]:
    """`SQL_N_UNITS` BIRD dev questions, stratified over database and difficulty, no RNG.
    """
    import sql_common as SQ
    qs = json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text(encoding="utf-8"))
    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for q in qs:
        strata[(q["db_id"], str(q.get("difficulty") or "?"))].append(q)
    for key in strata:
        strata[key].sort(key=lambda q: hashlib.sha256(
            f"{SQL_SEED}|{q['db_id']}|{q['question_id']}".encode()).hexdigest())
    keys = sorted(strata)
    picked: List[Dict[str, Any]] = []
    i = 0
    while len(picked) < SQL_N_UNITS:
        progressed = False
        for key in keys:
            if i < len(strata[key]) and len(picked) < SQL_N_UNITS:
                picked.append(strata[key][i])
                progressed = True
        if not progressed:
            break
        i += 1
    picked.sort(key=lambda q: int(q["question_id"]))
    ids = [f"bird|{q['question_id']}" for q in picked]
    dbs = sorted({q["db_id"] for q in picked})
    schema_chars = {db: len(SQ.schema_text(db)) for db in dbs}
    return {
        "benchmark": "sql_bird_v2", "family": "sql", "n_units": len(ids), "unit_ids": ids,
        "sha256_16": hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()[:16],
        "selection_rule": ("round-robin over (db_id, difficulty) strata, each stratum ordered by "
                           f"sha256('{SQL_SEED}|db_id|question_id'); deterministic, no RNG, "
                           "balanced across the 11 databases. A question_id prefix was rejected: "
                           "eval_sql.py measured it as a biased subpopulation"),
        "seed": SQL_SEED,
        "population": {"source": "BIRD dev_20240627 dev.json", "n_questions": len(qs),
                       "path": str(pathlib.Path(SQ.BIRD) / "dev.json"),
                       "sha256_16": hashlib.sha256(
                           json.dumps(sorted(int(q["question_id"]) for q in qs)).encode()
                       ).hexdigest()[:16]},
        "databases": dbs, "n_databases": len(dbs),
        "per_db": dict(collections.Counter(q["db_id"] for q in picked)),
        "per_difficulty": dict(collections.Counter(str(q.get("difficulty")) for q in picked)),
        "schema_chars_per_db": schema_chars,
        "mean_schema_chars": round(sum(schema_chars[q["db_id"]] for q in picked) / len(picked), 1),
        "answer_kind": "sql",
        "evaluation": {"mode": "read-only, at evaluation only, in an isolated subprocess",
                       "identity": "single-instance execution match (set equality of result "
                                   "tuples), the benchmark's own notion -- not proven semantic "
                                   "equivalence",
                       "feedback_to_arms": "none; no arm receives an execution result"},
        "analysis_unit": "recipient x question", "cluster_on": "question",
    }


# --------------------------------------------------------------------------- the design queue
def frozen(benchmark: str) -> Dict[str, Any]:
    p = FROZEN_DIR / f"{benchmark}.json"
    if not p.exists():
        raise SystemExit(f"{p} is missing: run `python3 ext_design.py freeze` first")
    return json.loads(p.read_text(encoding="utf-8"))


def supported(benchmark: str, protocol: str, arm: str) -> Optional[str]:
    """None when the cell can run; otherwise the reason it cannot. Same exclusions as the main design,
for the same reasons -- FORD renders one peer, so UNIFSEL is not a prompt condition for it."""
    if arm not in ARMS:
        return f"unknown arm {arm}"
    if protocol not in PROTOS:
        return f"{protocol} is not in the five protocol families"
    if protocol == "ford" and arm == "UNIFSEL":
        return ("FORD renders one peer, so there is nothing to choose between; its selection/use "
                "factor is the final-answer rule, recomputed from the record by "
                "arm_transform.use_arm_final_answer rather than by a prompt edit")
    return None


def cells(benchmark: str) -> List[Tuple[str, str, str, int]]:
    out = []
    for proto in PROTOS:
        for arm in ARMS:
            if supported(benchmark, proto, arm) is not None:
                continue
            for bb in PANELS[proto]:
                for rep in range(1, REPEATS + 1):
                    out.append((proto, arm, bb, rep))
    return out


def design_size(benchmark: str) -> int:
    return len(cells(benchmark)) * frozen(benchmark)["n_units"]


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "freeze":
        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        wrote = []
        for name, fn in (("gpqa", freeze_gpqa), ("sql_bird_v2", freeze_sql)):
            try:
                blob = fn()
            except Exception as exc:                                      # noqa: BLE001
                print("  %-14s NOT frozen: %s: %s" % (name, type(exc).__name__, str(exc)[:200]))
                continue
            blob["frozen_utc"] = __import__("time").strftime(
                "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())
            p = FROZEN_DIR / f"{name}.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
            tmp.replace(p)
            wrote.append(name)
            print("  %-14s %4d units, sha %s -> %s"
                  % (name, blob["n_units"], blob["sha256_16"], p.relative_to(ROOT)))
        return 0 if wrote else 1
    print("benchmark order: %s" % " -> ".join(BENCH_ORDER))
    for b in BENCH_ORDER:
        try:
            f = frozen(b)
        except SystemExit as exc:
            print("  %-14s %s" % (b, exc))
            continue
        cs = cells(b)
        print("  %-14s %4d units x %d cells = %s unit-runs   (sha %s)"
              % (b, f["n_units"], len(cs), f"{len(cs) * f['n_units']:,}", f["sha256_16"]))
        by = collections.Counter(p for p, _a, _bb, _r in cs)
        print("                 cells per protocol: %s" % dict(by))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
