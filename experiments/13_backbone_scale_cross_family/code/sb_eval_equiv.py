r"""Equivalence of sb_fast_eval vs ext_eval_sql on stored outcomes. Zero model calls.

    python sb_eval_equiv.py <ext_dir> <per_unit_jsonl_from_ext_eval_sql> <out_json> [--workers 48]

Re-scores every evaluated run's delivered SQL with the sb_fast_eval rule (ext_sql_verdicts._one in a
process pool; sql_common statement timeout, 10,000-row cap) and compares outcome classes and the
match/non-match bit with the stored ext_eval_sql per-unit outcome (process-per-query, 20 s parent
kill, same statement timeout and row cap).
"""
import collections
import concurrent.futures as cf
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sb_override as SB                                                 # noqa: E402,F401


def main():
    ext, pu_path, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 48
    import sql_common as SQ
    import sql_tasks as ST
    import ext_sql_verdicts as XV
    gold = {f"bird|{q['question_id']}": q["SQL"]
            for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}
    stored = {}
    for l in pu_path.read_text().splitlines():
        r = json.loads(l)
        stored[(r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))] = r["outcome"]
    last = {}
    for p in sorted((ext / "raw").glob("sql_bird_v2__*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok":
                k = (r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))
                if k in stored:
                    last[k] = r
    todo, rows = {}, []
    for k, r in last.items():
        db = ((r.get("config") or {}).get("unit_meta") or {}).get("db_id")
        ans = (r.get("record") or {}).get("majority_answer")
        pred = ST.unwrap_sql(ans) if ans not in (None, "", "None", "none") else ""
        rows.append((k, db, pred))
        if pred and db:
            todo[XV.qkey(db, pred)] = (db, pred, gold[k[0]])
    res = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for kk, v in ex.map(XV._one, list(todo.values()), chunksize=8):
            res[kk] = v
    per_bb = collections.defaultdict(collections.Counter)
    ex_dis = collections.defaultdict(list)
    for k, db, pred in rows:
        new = "declined" if not pred else (res.get(XV.qkey(db, pred)) or {}).get("outcome", "other_exec_error")
        old = stored[k]
        bb = k[3]
        per_bb[bb]["units"] += 1
        same_class = new == old
        same_bit = (new == "match") == (old == "match")
        per_bb[bb]["class_agree"] += same_class
        per_bb[bb]["match_bit_agree"] += same_bit
        if not same_bit or not same_class:
            per_bb[bb][f"{old}->{new}"] += 1
            if len(ex_dis[bb]) < 8:
                ex_dis[bb].append(dict(key=list(k), old=old, new=new, sql=pred[:300]))
    blob = dict(ext=str(ext), per_unit=str(pu_path), per_backbone={b: dict(c) for b, c in per_bb.items()},
                examples=ex_dis)
    out.write_text(json.dumps(blob, indent=1))
    for b, c in per_bb.items():
        print(b, dict(c))


if __name__ == "__main__":
    main()
