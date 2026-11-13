#!/usr/bin/env python3
"""The one input layer for B-group analysis: 5 protocol families x 7 benchmarks, two raw roots.

    python3 inputs.py                 # print the registry and what each source contributes

What settles it is the same witness the main collection's input verification uses: `providers.chat`
hashes the request body it is about to send, **after** the condition wrapper has replaced the messages. If
today's transform code reproduces that hash for every extension call, then today's code is
behaviourally identical to the collection code *on this data*, which is the equivalence claim the
digest is standing in for. `verify_sent_input.py --root ext` produces that evidence and this module
**refuses to admit the extension unless the evidence file says every call reproduced**. The gate is
mechanical, not a comment.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE, ROOT / "experiments/06_cost_resources/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "analysis/controlled_sharing"
MAIN_RAW = ROOT / "data/records/controlled_sharing"
EXT_RAW = ROOT / "data/records/controlled_sharing_gpqa_bird"
EXT_ANALYSIS = ROOT / "experiments/02_controlled_sharing/extension_analysis"

PROTOS = ["ford", "moa", "du_mad", "madm2_objective", "bccs"]
ARMS = ["FULL", "NOPEER", "ANSONLY", "UNIFSEL"]

MAIN_QA = ("gsm8k", "strategyqa", "mmlu_pro")
MAIN_MATCHING = ("wdc_products", "geographic_settlements")
EXT_QA = ("gpqa",)
EXT_SQL = ("sql_bird_v2",)
BENCHMARKS = list(MAIN_QA) + list(MAIN_MATCHING) + list(EXT_QA) + list(EXT_SQL)

#: The three outcome variables are **not interchangeable** and are never averaged together.
METRIC = {
    **{b: "QA accuracy" for b in MAIN_QA + EXT_QA},
    **{b: "pair F1(match)" for b in MAIN_MATCHING},
    **{b: "SQL execution accuracy" for b in EXT_SQL},
}
#: The digest every extension record carries, admitted only against the evidence described above.
EXT_COLLECTION_DIGEST = "f53d22b76fd2bc6b"
_GATE_CHECKED: Optional[str] = None


def _ext_equivalence_evidence() -> Tuple[bool, str]:
    """Did today's code reproduce every request the extension actually sent?"""
    p = OUT / "sent_input_verification.json"
    if not p.exists():
        return False, ("%s is missing. Run `PYTHONHASHSEED=0 python3 verify_sent_input.py "
                       "--verify` first: the extension's records carry digest %s while today's "
                       "recomputed digest differs, so they are admitted only against proof that "
                       "today's transform reproduces the requests that were sent."
                       % (p, EXT_COLLECTION_DIGEST))
    blob = json.loads(p.read_text(encoding="utf-8"))
    roots = blob.get("by_root") or {}
    ev = roots.get("ext")
    if not ev:
        return False, ("%s carries no 'ext' root. Re-run `verify_sent_input.py --verify` with the "
                       "extension root included." % p)
    if ev.get("calls") and ev.get("reproduced") == ev.get("calls"):
        return True, ("%s of %s extension calls reproduce the hash of the request that was sent "
                      "(verify_sent_input.py, trace-replay mode)"
                      % (f"{ev['reproduced']:,}", f"{ev['calls']:,}"))
    return False, ("only %s of %s extension calls reproduce the sent request; the digest cannot be "
                   "treated as equivalent and the extension must not enter the effects table."
                   % (f"{ev.get('reproduced', 0):,}", f"{ev.get('calls', 0):,}"))


def ext_gate(require: bool = True) -> str:
    global _GATE_CHECKED
    if _GATE_CHECKED is not None:
        return _GATE_CHECKED
    ok, why = _ext_equivalence_evidence()
    if not ok:
        if require:
            raise SystemExit("REFUSING to admit the extension benchmarks: " + why)
        return why
    _GATE_CHECKED = why
    return why


# ---------------------------------------------------------------- per-source machinery
def _install_main() -> None:
    import matching_tasks as MT
    MT.install()


def _install_ext() -> None:
    # registers the SQL answer kind, the three protocols' SQL adaptations and FORD's SQL re-render.
    import ext_sql_tasks as EST
    EST.install()


def _gold_main(benchmark: str) -> Dict[str, str]:
    import analyze_research_questions as ARQ
    return ARQ.gold_for(benchmark)


def _gold_ext(benchmark: str) -> Dict[str, str]:
    """GPQA: the option letter. SQL: `"correct"` for every unit.
    """
    import ext_design as D
    if benchmark in EXT_SQL:
        return {u: "correct" for u in D.frozen(benchmark)["unit_ids"]}
    return D.gpqa_gold()


def _admissible_main(r: Dict[str, Any]) -> bool:
    import analyze_research_questions as ARQ
    return ARQ.admissible(r)


def _admissible_ext(r: Dict[str, Any]) -> bool:
    """Single digest, admitted against the request-hash evidence; anything else is not trusted."""
    ext_gate(require=True)
    return r.get("code_version") == EXT_COLLECTION_DIGEST


def _sqlify(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Read-time rewrite of every SQL answer as its execution verdict. Raw shards are untouched.
    """
    import ext_sql_verdicts as V
    import sql_tasks as ST
    p = EXT_ANALYSIS / "sql_verdicts.json"
    if not p.exists():
        raise SystemExit("%s is missing; run `python3 ext_sql_verdicts.py`. Without execution "
                         "verdicts the only computable 'accuracy' is string equality with the "
                         "reference query, which is not this benchmark's notion of correctness."
                         % p)
    verdicts = json.loads(p.read_text(encoding="utf-8"))["verdicts"]
    stats = collections.Counter()

    def label(db: str, a: Any) -> Optional[str]:
        s = ST.unwrap_sql(a) if a not in (None, "", "None", "none") else ""
        if not s:
            stats["not_delivered"] += 1
            return None
        v = verdicts.get(V.qkey(db, s))
        if v is None:
            stats["no_verdict"] += 1
            return None
        c = v.get("correct")
        if c is None:
            stats["gold_failed_or_unscorable"] += 1
            return None
        stats["correct" if c else "incorrect"] += 1
        return "correct" if c else "incorrect"

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
    _sqlify.last_stats = dict(stats)                                      # type: ignore[attr-defined]
    return out


_SQL_VERDICTS: Optional[Dict[str, Any]] = None


def sql_label(db_id: str, answer: Any) -> Optional[str]:
    """`correct` / `incorrect` for one SQL answer, or None when it cannot be scored.
    """
    global _SQL_VERDICTS
    import ext_sql_verdicts as V
    import sql_tasks as ST
    if _SQL_VERDICTS is None:
        p = EXT_ANALYSIS / "sql_verdicts.json"
        if not p.exists():
            raise SystemExit("%s is missing; run `python3 ext_sql_verdicts.py`." % p)
        _SQL_VERDICTS = json.loads(p.read_text(encoding="utf-8"))["verdicts"]
    s = ST.unwrap_sql(answer) if answer not in (None, "", "None", "none") else ""
    if not s:
        return None
    v = _SQL_VERDICTS.get(V.qkey(db_id, s))
    if v is None or v.get("correct") is None:
        return None
    return "correct" if v["correct"] else "incorrect"


def score_answer(benchmark: str, record: Dict[str, Any], answer: Any,
                 gold: Dict[str, str], unit: str, kind: str) -> Optional[bool]:
    """Is this answer right? One place, so the three metrics cannot drift apart.
    """
    import analyze_research_questions as ARQ
    if benchmark in EXT_SQL:
        db = ((record.get("config") or {}).get("unit_meta") or {}).get("db_id")
        lab = sql_label(db, answer) if db else None
        return None if lab is None else (lab == "correct")
    g = gold.get(unit)
    if g is None:
        return None
    a = ARQ.norm(answer, kind)
    return None if a is None else (a == ARQ.norm(g, kind))


class Source:
    def __init__(self, tag: str, root: pathlib.Path, benchmarks: Tuple[str, ...],
                 install: Callable[[], None], gold: Callable[[str], Dict[str, str]],
                 admissible: Callable[[Dict[str, Any]], bool],
                 rows_transform: Optional[Callable[[List[Dict[str, Any]]],
                                                   List[Dict[str, Any]]]] = None,
                 collection_digest: Optional[str] = None, note: str = "") -> None:
        self.tag, self.root, self.benchmarks = tag, root, benchmarks
        self.install, self.gold, self.admissible = install, gold, admissible
        self.rows_transform = rows_transform
        self.collection_digest, self.note = collection_digest, note


SOURCES: Dict[str, Source] = {
    "main": Source(
        "main", MAIN_RAW, MAIN_QA + MAIN_MATCHING, _install_main, _gold_main, _admissible_main,
        note="the controlled 5x5x4 matrix; append-only, so a cell re-collected after the "
             "2026-10-20 repair holds both versions and admissibility is decided per (digest, "
             "cell) by collect_controlled.CONTAMINATION / CODE_EQUIVALENCE"),
    "ext": Source(
        "ext", EXT_RAW, EXT_QA + EXT_SQL, _install_ext, _gold_ext, _admissible_ext,
        rows_transform=lambda rows: _sqlify(rows),
        collection_digest=EXT_COLLECTION_DIGEST,
        note="the 2026-10-20 benchmark extension; one digest, no superseded records, admitted "
             "against request-hash evidence that today's transform reproduces what was sent"),
}
_BY_BENCH = {b: s for s in SOURCES.values() for b in s.benchmarks}


def source_for(benchmark: str) -> Source:
    s = _BY_BENCH.get(benchmark)
    if s is None:
        raise SystemExit("%r is not a registered benchmark; known: %s"
                         % (benchmark, ", ".join(BENCHMARKS)))
    return s


def installed_for(benchmark: str) -> Source:
    s = source_for(benchmark)
    s.install()
    return s


def gold_for(benchmark: str) -> Dict[str, str]:
    return installed_for(benchmark).gold(benchmark)


def kind_for(benchmark: str) -> str:
    installed_for(benchmark)
    import answer_extract as AX
    return AX.KIND_BY_BENCHMARK.get(benchmark, "binary_match")


def is_matching(benchmark: str) -> bool:
    return benchmark in MAIN_MATCHING


def read(benchmark: str, gate: bool = True,
         apply_transform: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Every admissible ok record for one benchmark, from whichever root owns it.
    """
    s = installed_for(benchmark)
    if s.tag == "ext" and gate:
        ext_gate(require=True)
    rows: List[Dict[str, Any]] = []
    drop: Dict[str, int] = collections.Counter()
    for p in sorted(s.root.glob(f"{benchmark}__*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                drop["unparseable"] += 1
                continue
            if r.get("status") != "ok":
                drop["not ok (%s)" % r.get("status")] += 1
                continue
            if gate and not s.admissible(r):
                drop["superseded or inadmissible digest: %s/%s/%s @ %s"
                     % (r.get("protocol"), r.get("benchmark"), r.get("arm"),
                        r.get("code_version"))] += 1
                continue
            r["_source"] = s.tag
            rows.append(r)
    if apply_transform and s.rows_transform is not None and benchmark in EXT_SQL:
        rows = s.rows_transform(rows)
        for r in rows:
            r["_source"] = s.tag
    return rows, dict(drop)


def main() -> int:
    print("B-group input registry: %d protocol families x %d benchmarks\n"
          % (len(PROTOS), len(BENCHMARKS)))
    print("%-18s %-7s %-24s %s" % ("benchmark", "source", "metric", "root"))
    for b in BENCHMARKS:
        s = source_for(b)
        print("%-18s %-7s %-24s %s" % (b, s.tag, METRIC[b], s.root.relative_to(ROOT)))
    ok, why = _ext_equivalence_evidence()
    print("\nextension digest %s: %s\n   %s"
          % (EXT_COLLECTION_DIGEST, "ADMITTED" if ok else "NOT ADMITTED", why))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
