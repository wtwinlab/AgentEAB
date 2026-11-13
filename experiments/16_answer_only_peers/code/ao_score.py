r"""Score the answer-only collection into the per-run table of paper object tab_answer-content.

    python3 -I experiments/16_answer_only_peers/code/ao_score.py [--workers 16]     # from the repository root

No model call; read-only SQL execution (5 s statement limit, 10,000-row cap). For every run the last
record with status ok is read. Each agent's answer in each round is read with the lenient extractor
(protocol extraction, else the last fenced block, else the last line-initial SELECT/WITH query); a query
is correct when its result equals the reference query's result. The vote is the execution vote: the
largest cluster of identical execution results among queries that execute, ties to the first agent.
`final` scores the protocol's own final answer.

Reads   experiments/16_answer_only_peers/data/raw/<lane>/sql_bird_v2__*.jsonl and BIRD dev.json.
Writes  experiments/16_answer_only_peers/output/runs.csv (the columns of
        paper_objects/tab_answer-content/inputs/runs.csv), output/exclusions.json, and the execution
        caches output/eval_cache/{sigs,verdicts}.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "share_eval/replay"))
import factorial_replay as FR                                             # noqa: E402
import sql_common as SQ                                                   # noqa: E402

EXP = HERE.parent
RAW = EXP / "data/raw"
OUT = EXP / "output"
CACHE = OUT / "eval_cache"
UNITS = set(json.loads((HERE / "units198.json").read_text())["units198"])
KEEP = {("du_mad", "FULL"), ("du_mad", "ANSWER_ONLY"), ("du_mad", "NOPEERC"),
        ("madm2_objective", "FULL"), ("madm2_objective", "ANSWER_ONLY"), ("madm2_objective", "NOPEER")}


def read(p):
    for line in p.open(encoding="utf-8", errors="replace"):
        try:
            yield json.loads(line)
        except ValueError:
            continue


def load():
    last, errs = {}, collections.defaultdict(list)
    for p in sorted(RAW.glob("*/sql_bird_v2__*.jsonl")):
        if p.parent.name == "smoke":
            continue
        for r in read(p):
            if r.get("unit") not in UNITS or (r.get("protocol"), r.get("arm")) not in KEEP:
                continue
            k = (r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))
            if r.get("status") == "ok":
                last[k] = r
            else:
                errs[k].append(dict(error_type=r.get("error_type"), error=(r.get("error") or "")[:160]))
    return last, {k: v for k, v in errs.items() if k not in last}


def cache(name):
    p = CACHE / name
    d = {}
    if p.exists():
        for line in p.open():
            x = json.loads(line)
            d[x["k"]] = x
    return p, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    gold = {f"bird|{q['question_id']}": q for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}
    last, excl = load()
    sp, sigs = cache("sigs.jsonl")
    vp, verd = cache("verdicts.jsonl")
    runs, todo_s, todo_v = [], {}, {}
    for k, r in sorted(last.items()):
        u = k[0]
        db = gold[u]["db_id"]
        calls = (r.get("record") or {}).get("calls") or []
        byr = collections.defaultdict(list)
        for c in calls:
            byr[int(c.get("round") or 0)].append(c)
        rounds = {}
        for rn, cs in byr.items():
            cs = sorted(cs, key=lambda c: int(c.get("agent_index") or 0))
            own = [FR.sql_of(c.get("extracted_answer")) for c in cs]
            rounds[rn] = [s if s else FR.lenient(c.get("raw_reply")) for s, c in zip(own, cs)]
        fin = FR.sql_of((r.get("record") or {}).get("majority_answer"))
        for q in [fin] + [q for qs in rounds.values() for q in qs]:
            if q:
                kk = FR.qkey(db, q)
                if kk not in sigs:
                    todo_s[kk] = (db, q)
                if kk not in verd:
                    todo_v[kk] = (db, q, gold[u]["SQL"])
        runs.append((k, db, rounds, fin))
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex, open(sp, "a") as fs, open(vp, "a") as fv:
        for kk, v in ex.map(FR._sig, list(todo_s.values()), chunksize=8):
            v["k"] = kk
            sigs[kk] = v
            fs.write(json.dumps(v) + "\n")
        for kk, v in ex.map(FR._verdict, list(todo_v.values()), chunksize=8):
            v["k"] = kk
            verd[kk] = v
            fv.write(json.dumps(v) + "\n")

    def corr(db, q):
        return bool(q and (verd.get(FR.qkey(db, q)) or {}).get("correct"))

    def sig(db, q):
        s = sigs.get(FR.qkey(db, q)) if q else None
        return s["sig"] if s and s.get("status") == "ok" else None

    def vote(db, qs):
        lab = [sig(db, q) for q in qs]
        w, _tied = FR.vote(lab)
        return bool(w) and corr(db, qs[lab.index(w)])

    gold_failed = collections.Counter()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "db", "protocol", "backbone", "condition", "repeat", "r0", "last", "final",
                    "agents_r0", "agents_last"])
        for (u, proto, arm, bb, rep), db, rounds, fin in runs:
            fv_ = verd.get(FR.qkey(db, fin)) if fin else None
            if fv_ and fv_.get("outcome") == "gold_failed":
                gold_failed[u] += 1
            r0, lst = rounds[0], rounds[max(rounds)]
            cond = "NOPEER" if arm in ("NOPEER", "NOPEERC") else arm
            w.writerow([u, db, proto, bb, cond, rep, int(vote(db, r0)), int(vote(db, lst)),
                        int(bool(fv_ and fv_.get("correct"))),
                        "".join(str(int(corr(db, q))) for q in r0), "".join(str(int(corr(db, q))) for q in lst)])
    (OUT / "exclusions.json").write_text(json.dumps(dict(
        failed_runs=[dict(unit=k[0], protocol=k[1], condition=k[2], backbone=k[3], repeat=k[4], attempts=v)
                     for k, v in sorted(excl.items())],
        gold_failed_units=dict(gold_failed), runs_scored=len(runs)), indent=1) + "\n")
    print(f"{len(runs)} runs scored; {len(excl)} without an ok record; gold failures {dict(gold_failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
