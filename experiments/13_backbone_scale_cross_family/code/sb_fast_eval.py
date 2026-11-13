r"""Fast per-unit evaluation of the delivered final SQL (same outcome classes as ext_eval_sql).

    python sb_run.py-free:  PAM_ART=... python sb_fast_eval.py [--workers 48]

`ext_eval_sql` spawns one process per query with a 20 s parent kill; at ~30k runs it took > 40 min
per pass. This evaluator uses the same read-only `sql_common.execute` (statement timeout 5 s, 10,000
row cap) and `ext_sql_verdicts._one` comparison inside a process pool, de-duplicated by (db, query).
The only difference is that no parent-side hard kill exists beyond the statement timeout. Writes
<OUT>/analysis/sql_per_unit.jsonl (one row per evaluated run key, the last ok record) and
sql_fast_eval_cost.json.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sb_override as SB                                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    import sql_common as SQ
    import sql_tasks as ST
    import ext_sql_verdicts as XV
    gold = {f"bird|{q['question_id']}": q["SQL"]
            for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}
    last = {}
    for p in sorted((SB.OUT / "raw").glob("sql_bird_v2__*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") != "ok":
                continue
            last[(r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))] = r
    rows, todo = [], {}
    for k, r in last.items():
        db = ((r.get("config") or {}).get("unit_meta") or {}).get("db_id")
        ans = (r.get("record") or {}).get("majority_answer")
        pred = ST.unwrap_sql(ans) if ans not in (None, "", "None", "none") else ""
        rows.append((k, r, db, pred))
        if pred and db and k[0] in gold:
            todo[XV.qkey(db, pred)] = (db, pred, gold[k[0]])
    t0 = time.time()
    res = {}
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for kk, v in ex.map(XV._one, list(todo.values()), chunksize=8):
            res[kk] = v
    out = SB.OUT / "analysis"
    out.mkdir(exist_ok=True)
    cnt = collections.Counter()
    with open(out / "sql_per_unit.jsonl.tmp", "w") as fh:
        for (u, proto, arm, bb, rep), r, db, pred in rows:
            if not pred:
                oc = "truncated_only" if (r.get("truncated_calls") or 0) else "declined"
                rec = dict(unit=u, protocol=proto, arm=arm, backbone=bb, repeat=rep, outcome="declined",
                           decline_kind=oc, truncated_calls=r.get("truncated_calls") or 0)
            else:
                v = res.get(XV.qkey(db, pred)) or {"outcome": "other_exec_error"}
                rec = dict(unit=u, protocol=proto, arm=arm, backbone=bb, repeat=rep, outcome=v["outcome"],
                           truncated_calls=r.get("truncated_calls") or 0, db_id=db)
            cnt[rec["outcome"]] += 1
            fh.write(json.dumps(rec) + "\n")
    (out / "sql_per_unit.jsonl.tmp").replace(out / "sql_per_unit.jsonl")
    (out / "sql_fast_eval_cost.json").write_text(json.dumps(dict(
        runs=len(rows), distinct_queries=len(todo), wall_s=round(time.time() - t0, 1),
        workers=a.workers, outcomes=dict(cnt), evaluator="sb_fast_eval (ext_sql_verdicts._one semantics)",
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))
    print(len(rows), len(todo), dict(cnt), round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
