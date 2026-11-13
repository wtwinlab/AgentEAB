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

RUN_FIELDS = ("unit", "protocol", "arm", "backbone", "repeat", "last_round")
FLOW_FIELDS = ("question", "db_id", "protocol", "backbone", "repeat")


def read_jsonl(path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_csv(path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_per_run_scores(path=None) -> List[Dict[str, Any]]:
    return read_jsonl(path or paths.inputs(FACTORIAL) / "per_run_scores.jsonl")


def read_question_databases(path=None) -> Dict[str, str]:
    return {r["unit"]: r["db_id"] for r in read_csv(path or paths.inputs(FACTORIAL) / "bird_units_db.csv")}


def read_candidate_flow_runs(path=None) -> List[Dict[str, Any]]:
    rows = read_csv(path or paths.inputs(CANDIDATE_FLOW) / "candidate_flow_runs.csv")
    return [{k: (v if k in FLOW_FIELDS and k != "repeat" else int(v)) for k, v in r.items()} for r in rows]


def per_run_scores_to_records(rows: Optional[Iterable[Mapping[str, Any]]] = None,
                              databases: Optional[Mapping[str, str]] = None) -> List[ExecutionRecord]:
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
    rows = read_candidate_flow_runs() if rows is None else rows
    return [ExecutionRecord(task=r["question"], workflow=r["protocol"], condition="FULL",
                            backbone=r["backbone"], repeat=int(r["repeat"]), cluster=r["db_id"],
                            scores={k: float(v) for k, v in r.items() if k not in FLOW_FIELDS})
            for r in rows]


SEED_TIE = 20261106
INTERACTIVE = ("du_mad", "madm2_objective", "bccs")
AGG_ONLY = ("moa",)
ARMS = ("NOPEER", "FULL")
EXTRACTORS = ("strict", "lenient")
VOTES = ("string", "exec", "exec_ident")
ROUNDS = ("r0", "last")
TIES = ("first", "last", "random")

_FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.S)
_BARE = re.compile(r"(?im)^\s*((?:SELECT|WITH)\b[^;`]*)")


def _unwrap_sql(raw: Any) -> str:
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
    return _unwrap_sql(raw) if raw not in (None, "", "None", "none") else ""


def _lenient(raw_reply: Any) -> str:
    t = str(raw_reply or "")
    m = _FENCE.findall(t)
    if m:
        return _unwrap_sql(m[-1])
    b = _BARE.findall(t)
    return _unwrap_sql(b[-1]) if b else ""


def _round_of(proto: str, c: Mapping[str, Any]) -> int:
    if proto == "moa":
        return 0 if str(c.get("role")) == "reference" else 1
    return int(c.get("round") or 0)


def _vote(labels: Sequence[Optional[Any]], tie: str = "first", key: str = "") -> Tuple[Optional[Any], bool]:
    live = [x for x in labels if x is not None]
    if not live:
        return None, False
    cnt = collections.Counter(live)
    top = max(cnt.values())
    win = {v for v in cnt if cnt[v] == top}
    order = list(dict.fromkeys(x for x in live if x in win))
    if tie == "first" or len(order) == 1:
        w = order[0]
    elif tie == "last":
        w = next(x for x in reversed(live) if x in win)
    else:
        h = int(hashlib.sha256(f"{SEED_TIE}|{key}".encode()).hexdigest()[:12], 16)
        w = order[h % len(order)]
    return w, len(win) > 1


def _load_runs(run_dir: pathlib.Path, per_unit_path: Optional[pathlib.Path], protocols: Sequence[str]):
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


def logged_debates_available(run_dir=None) -> bool:
    raw = pathlib.Path(run_dir or DEBATES) / "raw"
    return raw.is_dir() and any(raw.glob("sql_bird_v2__*.jsonl"))


def load_logged_debates(run_dir=None, per_unit=None,
                        protocols: Optional[Sequence[str]] = None) -> Tuple[Dict[Tuple, Dict[str, Any]],
                                                                            Optional[Dict[Tuple, Dict[str, Any]]],
                                                                            Dict[str, int]]:
    run_dir = pathlib.Path(run_dir or DEBATES)
    if per_unit is None:
        p = run_dir / "analysis" / "sql_per_unit.jsonl"
        per_unit = p if p.exists() else None
    protocols = tuple(protocols or (INTERACTIVE + AGG_ONLY))
    return _load_runs(run_dir, pathlib.Path(per_unit) if per_unit else None, protocols)


def logged_debate_to_record(key: Tuple, raw: Mapping[str, Any]) -> ExecutionRecord:
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
    runs, _, _ = load_logged_debates(run_dir, per_unit, protocols)
    return [logged_debate_to_record(k, r) for k, r in runs.items()]


def candidate_sets(runs: Mapping[Tuple, Mapping[str, Any]], per_unit=None,
                   protocols: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    return _build_rows(runs, per_unit, tuple(protocols or (INTERACTIVE + AGG_ONLY)))


def score_candidate_set(row: Mapping[str, Any], signatures: Mapping[str, Mapping[str, Any]],
                        verdicts: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    scores, _, _ = _score_run(row, signatures, verdicts)
    return {"|".join(k): v for k, v in scores.items()}


def query_key(database: str, query: str) -> str:
    return hashlib.sha256(f"{database}|{query}".encode()).hexdigest()[:24]
