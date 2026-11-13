"""Readers of logged runs and of the scored per-run tables.

Two sources reach the stage evaluation:

- the scored per-run tables shipped with the paper objects: one row per run with the correctness of
  every reading (extractor, vote, round, tie rule) and of the returned answer
  (`paper_objects/tab_factorial/inputs/per_run_scores.jsonl`), and the candidate-flow flags of the
  BIRD debates with peers (`paper_objects/tab_candidate-flow/inputs/candidate_flow_runs.csv`);
- the logged BIRD debates (`data/records/bird_debates/raw/sql_bird_v2__*.jsonl`): every call of every
  agent in every round, with the raw reply. The readers below select the completed runs, extract the
  candidates of every round with the protocol extractor and the lenient extractor, and score the vote
  under every reading.

The extraction and the vote (`_unwrap_sql`, `_sql_of`, `_lenient`, `_round_of`, `_vote`, `_load_runs`,
`_build_rows`, `_score_run`) are defined here; the replay steps (`steps/`) use them.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import pathlib
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from share_eval_suite import paths
from share_eval_suite.records.schema import Call, Decision, ExecutionRecord

FACTORIAL = "tab_factorial"
CANDIDATE_FLOW = "tab_candidate-flow"
DEBATES = paths.REPO / "data" / "records" / "bird_debates"

# Fields of a per-run score row that describe the run; every other field is a score.
RUN_FIELDS = ("unit", "protocol", "arm", "backbone", "repeat", "last_round")
FLOW_FIELDS = ("question", "db_id", "protocol", "backbone", "repeat")


# ------------------------------------------------------------------ plain tables
def read_jsonl(path) -> List[Dict[str, Any]]:
    """Rows of a JSON-lines file."""
    return [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_csv(path) -> List[Dict[str, str]]:
    """Rows of a CSV file, values as strings."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_per_run_scores(path=None) -> List[Dict[str, Any]]:
    """Scored BIRD debate runs, one row per run (question, protocol, backbone, with or without peers,
    repeat): the correctness of the returned answer (`final`) and of the vote under every reading,
    keyed `extractor|vote|round|tie` (extractor strict or lenient, vote string, exec or exec_ident,
    round r0 or last, tie first, last or random)."""
    return read_jsonl(path or paths.inputs(FACTORIAL) / "per_run_scores.jsonl")


def read_question_databases(path=None) -> Dict[str, str]:
    """The BIRD database of each development question (`bird|<id>` -> database)."""
    return {r["unit"]: r["db_id"] for r in read_csv(path or paths.inputs(FACTORIAL) / "bird_units_db.csv")}


def read_candidate_flow_runs(path=None) -> List[Dict[str, Any]]:
    """BIRD debate runs with peers, one row per run, with the candidate-flow flags as integers:
    `C0` (some initial answer correct, lenient extractor), `CT` (some last-round answer correct,
    lenient extractor), `CD` (some last-round answer correct, protocol extractor), `final` (returned
    answer correct) and the last-round vote under each extractor, vote and tie rule
    (`{protocol,lenient}_{exact,execution}_{first,last,random}`)."""
    rows = read_csv(path or paths.inputs(CANDIDATE_FLOW) / "candidate_flow_runs.csv")
    return [{k: (v if k in FLOW_FIELDS and k != "repeat" else int(v)) for k, v in r.items()} for r in rows]


# ------------------------------------------------------------------ scored rows as records
def per_run_scores_to_records(rows: Optional[Iterable[Mapping[str, Any]]] = None,
                              databases: Optional[Mapping[str, str]] = None) -> List[ExecutionRecord]:
    """Execution records of scored runs. Calls are not part of the table, so a record holds the task
    scores (every reading and `final`), the run's last round, and the question's database as the
    resampling cluster."""
    rows = read_per_run_scores() if rows is None else rows
    databases = read_question_databases() if databases is None else databases
    out = []
    for r in rows:
        out.append(ExecutionRecord(
            task=r["unit"], workflow=r["protocol"], condition=r["arm"], backbone=r["backbone"],
            repeat=int(r["repeat"]), cluster=databases.get(r["unit"], ""),
            decision=Decision(stopped_at_round=r.get("last_round")),
            scores={k: float(v) for k, v in r.items() if k not in RUN_FIELDS},
            meta={"last_round": r.get("last_round")}))
    return out


def candidate_flow_to_records(rows: Optional[Iterable[Mapping[str, Any]]] = None) -> List[ExecutionRecord]:
    """Execution records of the candidate-flow runs (with peers); the flags are the task scores."""
    rows = read_candidate_flow_runs() if rows is None else rows
    return [ExecutionRecord(task=r["question"], workflow=r["protocol"], condition="FULL",
                            backbone=r["backbone"], repeat=int(r["repeat"]), cluster=r["db_id"],
                            scores={k: float(v) for k, v in r.items() if k not in FLOW_FIELDS})
            for r in rows]


# ------------------------------------------------------------------ extraction and vote
SEED_TIE = 20261106                                 # seed of the random tie rule
INTERACTIVE = ("du_mad", "madm2_objective", "bccs")  # debates with rounds of interaction
AGG_ONLY = ("moa",)                                  # aggregation: the last round is the aggregator's answer
ARMS = ("NOPEER", "FULL")                            # without and with peers
EXTRACTORS = ("strict", "lenient")
VOTES = ("string", "exec", "exec_ident")
ROUNDS = ("r0", "last")
TIES = ("first", "last", "random")

_FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.S)
_BARE = re.compile(r"(?im)^\s*((?:SELECT|WITH)\b[^;`]*)")


def _unwrap_sql(raw: Any) -> str:
    """A query with its last code fence, a leading language tag, a trailing semicolon and repeated
    whitespace removed. Case is kept, so the exact-string vote compares queries as written."""
    t = str(raw or "").strip()
    if not t:
        return ""
    m = _FENCE.findall(t)
    if m:
        t = m[-1]
    t = t.strip()
    t = re.sub(r"^(?:sql|SQL)\s*[\n:]", "", t).strip()
    t = t.rstrip(";").strip()
    return " ".join(t.split())


def _sql_of(raw: Any) -> str:
    """Protocol extractor: the answer the protocol itself extracted, unwrapped; empty when it has none."""
    return _unwrap_sql(raw) if raw not in (None, "", "None", "none") else ""


def _lenient(raw_reply: Any) -> str:
    """Lenient extractor on a raw reply: the last fenced SQL block, else the last line that starts
    with SELECT or WITH; empty when neither is present."""
    t = str(raw_reply or "")
    m = _FENCE.findall(t)
    if m:
        return _unwrap_sql(m[-1])
    b = _BARE.findall(t)
    return _unwrap_sql(b[-1]) if b else ""


def _round_of(proto: str, c: Mapping[str, Any]) -> int:
    """Round of a call; for MoA the reference answers are round 0 and the aggregator's answer round 1."""
    if proto == "moa":
        return 0 if str(c.get("role")) == "reference" else 1
    return int(c.get("round") or 0)


def _vote(labels: Sequence[Optional[Any]], tie: str = "first", key: str = "") -> Tuple[Optional[Any], bool]:
    """Most frequent label among the candidates that are not None. Ties go to the tied label that
    appears first (`first`) or last (`last`) in agent order, or (`random`) to a tied label chosen
    uniformly by the sha256 hash of the seed and a per-run key, so the choice is reproducible.
    Returns (winner, whether the top count was tied)."""
    live = [x for x in labels if x is not None]
    if not live:
        return None, False
    cnt = collections.Counter(live)
    top = max(cnt.values())
    win = {v for v in cnt if cnt[v] == top}
    order = list(dict.fromkeys(x for x in live if x in win))       # tied labels in first-appearance order
    if tie == "first" or len(order) == 1:
        w = order[0]
    elif tie == "last":
        w = next(x for x in reversed(live) if x in win)
    else:
        h = int(hashlib.sha256(f"{SEED_TIE}|{key}".encode()).hexdigest()[:12], 16)
        w = order[h % len(order)]
    return w, len(win) > 1


def _load_runs(run_dir: pathlib.Path, per_unit_path: Optional[pathlib.Path], protocols: Sequence[str]):
    """Completed logged runs keyed (question, protocol, condition, backbone, repeat), the evaluated set
    (or None) and the counts of the records set aside, by reason. Files are read in name order; a
    later completed record of the same run replaces an earlier one."""
    per_unit = None
    if per_unit_path and per_unit_path.exists():
        per_unit = {}
        for line in per_unit_path.read_text().splitlines():
            r = json.loads(line)
            per_unit[(r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))] = r
    last: Dict[Tuple, Dict[str, Any]] = {}
    seen: collections.Counter = collections.Counter()
    for p in sorted((run_dir / "raw").glob("sql_bird_v2__*.jsonl")):
        if p.name.split("__")[1] not in protocols or p.name.split("__")[2] not in ARMS:
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                seen["unparseable_line"] += 1
                continue
            if r.get("protocol") not in protocols or r.get("arm") not in ARMS:
                continue
            k = (r.get("unit"), r.get("protocol"), r.get("arm"), r.get("backbone"),
                 int(r.get("repeat") or 0))
            if r.get("status") != "ok":
                seen["non_ok_record"] += 1
                continue
            if per_unit is not None and k not in per_unit:
                seen["ok_record_not_in_evaluated_set"] += 1
                continue
            if k in last:
                seen["duplicate_ok_record_later_kept"] += 1
            last[k] = r
    return last, per_unit, dict(seen)


def _build_rows(runs: Mapping[Tuple, Mapping[str, Any]], per_unit, protocols: Sequence[str]) -> List[Dict[str, Any]]:
    """One row per run with an initial round: the candidates of round 0 and of the last round, in
    agent order, read by the protocol extractor and by the lenient extractor (the protocol's own
    answer when it has one, else the lenient reading of the raw reply)."""
    rows = []
    for k, r in runs.items():
        unit, proto, arm, bb, rep = k
        db = ((r.get("config") or {}).get("unit_meta") or {}).get("db_id")
        calls = (r.get("record") or {}).get("calls") or []
        by_round = collections.defaultdict(list)
        for c in calls:
            by_round[_round_of(proto, c)].append(c)
        if 0 not in by_round:
            continue
        lr = max(by_round)
        sets = {}
        for name, cs in (("r0", by_round[0]), ("last", by_round[lr])):
            cs = sorted(cs, key=lambda c: int(c.get("agent_index") or 0))
            strict = [_sql_of(c.get("extracted_answer", c.get("prediction"))) for c in cs]
            lens = [s if s else _lenient(c.get("raw_reply")) for s, c in zip(strict, cs)]
            sets[name] = {"strict": strict, "lenient": lens}
        fin = per_unit[k]["outcome"] if per_unit is not None else None
        rows.append(dict(unit=unit, protocol=proto, arm=arm, backbone=bb, repeat=rep, db=db,
                         sets=sets, last_round=lr, final_outcome=fin))
    return rows


def _score_run(row: Mapping[str, Any], sigs: Mapping[str, Mapping[str, Any]],
               verdicts: Mapping[str, Mapping[str, Any]]):
    """Correctness of the vote for every (extractor, vote, round, tie rule), and per (extractor, vote,
    round) whether the top count was tied and whether some candidate was readable.

    Votes: `string` groups identical query strings; `exec` groups queries by execution result and
    drops failed executions; `exec_ident` puts failed executions in one FAIL group that can win and
    then scores 0. The winner is scored by the correctness verdict of its first query."""
    db = row["db"]
    out, ties, live = {}, {}, {}
    for rd in ROUNDS:
        for e in EXTRACTORS:
            qs = row["sets"][rd][e]
            corr = [bool(q and (verdicts.get(query_key(db, q)) or {}).get("correct")) for q in qs]
            for i in VOTES:
                lab = []
                for q in qs:
                    if not q:
                        lab.append(None)
                    elif i == "string":
                        lab.append(q)
                    else:
                        s = sigs.get(query_key(db, q)) or {}
                        if s.get("status") == "ok":
                            lab.append(s["sig"])
                        else:
                            lab.append("FAIL" if i == "exec_ident" else None)
                rk = f"{row['unit']}|{row['protocol']}|{row['arm']}|{row['backbone']}|{row['repeat']}|{rd}|{e}|{i}"
                for tie in TIES:
                    w, t = _vote(lab, tie, rk)
                    ok = w is not None and w != "FAIL" and corr[lab.index(w)]
                    out[(e, i, rd, tie)] = float(ok)
                ties[(e, i, rd)] = float(t)
                live[(e, i, rd)] = float(any(x is not None for x in lab))
    return out, ties, live


# ------------------------------------------------------------------ logged debates

def logged_debates_available(run_dir=None) -> bool:
    """True when the logged BIRD debates are present (`<run_dir>/raw/sql_bird_v2__*.jsonl`)."""
    raw = pathlib.Path(run_dir or DEBATES) / "raw"
    return raw.is_dir() and any(raw.glob("sql_bird_v2__*.jsonl"))


def load_logged_debates(run_dir=None, per_unit=None,
                        protocols: Optional[Sequence[str]] = None) -> Tuple[Dict[Tuple, Dict[str, Any]],
                                                                            Optional[Dict[Tuple, Dict[str, Any]]],
                                                                            Dict[str, int]]:
    """The logged runs, keyed (question, protocol, condition, backbone, repeat).

    Only completed runs count; a later completed record of the same run replaces an earlier one. With
    an evaluated set (`per_unit`, default `<run_dir>/analysis/sql_per_unit.jsonl` when present) only
    its runs are kept and their stored returned-answer outcomes are returned. The third value counts
    the records that were set aside, by reason.
    """
    run_dir = pathlib.Path(run_dir or DEBATES)
    if per_unit is None:
        p = run_dir / "analysis" / "sql_per_unit.jsonl"
        per_unit = p if p.exists() else None
    protocols = tuple(protocols or (INTERACTIVE + AGG_ONLY))
    return _load_runs(run_dir, pathlib.Path(per_unit) if per_unit else None, protocols)


def logged_debate_to_record(key: Tuple, raw: Mapping[str, Any]) -> ExecutionRecord:
    """One logged run as an execution record.

    Each call keeps its agent, round, raw reply, the candidate read by the protocol extractor
    (`candidate`), the candidate read by the lenient extractor (`meta["lenient"]`) and the logged
    prompt and completion tokens. The decision holds the run's returned answer and stopping round as
    logged.
    """
    unit, proto, arm, backbone, repeat = key
    rec = raw.get("record") or {}
    calls = []
    for c in rec.get("calls") or []:
        strict = _sql_of(c.get("extracted_answer", c.get("prediction")))
        calls.append(Call(
            agent=int(c.get("agent_index") or 0), round=_round_of(proto, c),
            response=str(c.get("raw_reply") or ""), candidate=strict or None,
            prompt_tokens=int(c.get("prompt_tokens") or 0),
            completion_tokens=int(c.get("completion_tokens") or 0),
            stage="generation" if _round_of(proto, c) == 0 else "revision",
            meta={"lenient": strict or _lenient(c.get("raw_reply")) or None}))
    db = ((raw.get("config") or {}).get("unit_meta") or {}).get("db_id") or ""
    decision = Decision(answer=rec.get("majority_answer", rec.get("final_answer")),
                        rule=proto, stopped_at_round=rec.get("stopped_at_round"),
                        stop_reason=str(rec.get("native_stop_reason") or ""))
    return ExecutionRecord(task=unit, workflow=proto, condition=arm, backbone=backbone,
                           repeat=int(repeat), cluster=db, calls=calls, decision=decision)


def read_logged_debates(run_dir=None, per_unit=None,
                        protocols: Optional[Sequence[str]] = None) -> List[ExecutionRecord]:
    """Every selected logged run as an execution record (see `load_logged_debates`)."""
    runs, _, _ = load_logged_debates(run_dir, per_unit, protocols)
    return [logged_debate_to_record(k, r) for k, r in runs.items()]


def candidate_sets(runs: Mapping[Tuple, Mapping[str, Any]], per_unit=None,
                   protocols: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Per run, the candidates of the initial round (`r0`) and of the last round (`last`), each read by
    the protocol extractor (`strict`) and the lenient extractor (`lenient`), in agent order. Runs
    without an initial round are left out."""
    return _build_rows(runs, per_unit, tuple(protocols or (INTERACTIVE + AGG_ONLY)))


def score_candidate_set(row: Mapping[str, Any], signatures: Mapping[str, Mapping[str, Any]],
                        verdicts: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """Correctness of the vote under every reading of one run, keyed `extractor|vote|round|tie` as in
    the scored per-run table. `signatures` and `verdicts` are the stored execution results and
    correctness verdicts of the run's queries, keyed by `query_key(database, query)`."""
    scores, _, _ = _score_run(row, signatures, verdicts)
    return {"|".join(k): v for k, v in scores.items()}


def query_key(database: str, query: str) -> str:
    """Key of a query's stored execution result and verdict."""
    return hashlib.sha256(f"{database}|{query}".encode()).hexdigest()[:24]
