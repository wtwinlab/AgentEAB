r"""A15 equivalence on the strong-backbone runs: for a seeded sample of evaluated runs per backbone,
score the delivered SQL with the frozen ext_eval_sql.execute_isolated (process per query, 20 s parent
kill) and compare with sb_fast_eval's stored outcome. Threads only drive the isolated processes.
    python sb_eval_equiv_sample.py [per_backbone=250] -> <OUT>/eval_equiv_strong_sample.json"""
import collections
import concurrent.futures as cf
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sb_override as SB                                                 # noqa: E402


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    import sql_common as SQ
    import sql_tasks as ST
    import ext_eval_sql as EV
    gold = {f"bird|{q['question_id']}": q["SQL"] for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}
    fast = {}
    for l in (SB.OUT / "analysis/sql_per_unit.jsonl").read_text().splitlines():
        r = json.loads(l)
        fast[(r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))] = r["outcome"]
    last = {}
    for p in sorted((SB.OUT / "raw").glob("sql_bird_v2__*.jsonl")):
        for line in p.read_text(errors="replace").splitlines():
            r = json.loads(line)
            if r.get("status") == "ok":
                last[(r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))] = r
    by = collections.defaultdict(list)
    for k, r in last.items():
        ans = (r.get("record") or {}).get("majority_answer")
        pred = ST.unwrap_sql(ans) if ans not in (None, "", "None", "none") else ""
        if pred and k in fast:
            by[k[3]].append((k, ((r.get("config") or {}).get("unit_meta") or {}).get("db_id"), pred))
    rng = random.Random(20261106)
    sample = []
    for bb, xs in sorted(by.items()):
        rng.shuffle(xs)
        sample += xs[:per]
    res = {}
    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        futs = {ex.submit(EV.execute_isolated, db, pred, gold[k[0]]): k for k, db, pred in sample}
        for f in cf.as_completed(futs):
            res[futs[f]] = f.result()["outcome"]
    per_bb = collections.defaultdict(collections.Counter)
    exs = collections.defaultdict(list)
    for k, db, pred in sample:
        o, n = res[k], fast[k]
        c = per_bb[k[3]]
        c["n"] += 1
        c["class_agree"] += o == n
        c["match_bit_agree"] += (o == "match") == (n == "match")
        if o != n:
            c[f"{o}->{n}"] += 1
            if len(exs[k[3]]) < 5:
                exs[k[3]].append(dict(key=list(k), old=o, fast=n, sql=pred[:240]))
    out = dict(per_backbone={b: dict(c) for b, c in per_bb.items()}, examples=exs, per_backbone_sample=per,
               old="ext_eval_sql.execute_isolated (process per query, 20 s parent kill, 5 s statement timeout, 10k rows)",
               fast="sb_fast_eval (ext_sql_verdicts._one in a pool, 5 s statement timeout, 10k rows)")
    (SB.OUT / "eval_equiv_strong_sample.json").write_text(json.dumps(out, indent=1))
    for b, c in per_bb.items():
        print(b, dict(c))


if __name__ == "__main__":
    main()
