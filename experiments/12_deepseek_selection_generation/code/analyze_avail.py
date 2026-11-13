r"""Part A scoring and analysis. Offline; read-only DBs via the existing executors.

    python3 code/analyze_avail.py   # -> output/avail_per_question.csv, avail_levels.csv, avail_contrasts.csv, avail_changes.csv

Generation per (question, k): `final_channel` + `parse_sql` + `execution_match` (gold only here);
not run (fewer than two live candidates) or unparsed/failed = 0. Rules and Oracle from build_avail.
Database-equal primary, question-equal beside; DB-cluster bootstrap 10,000 draws seed 20261105.
"""
import collections
import csv
import json
import pathlib
import sys

sys.dont_write_bytecode = True
W = pathlib.Path(__file__).resolve().parents[1]
REPO = W.parents[1]
sys.path.insert(0, str(REPO / "experiments/common/code"))
sys.path.insert(0, str(REPO / "experiments/common/code"))
import build_inputs as BI                                                  # noqa: E402
import core as C                                                           # noqa: E402

DRAWS, SEED = 10000, 20261105
O = W / "output"; O.mkdir(exist_ok=True)
per, levels, cons, chg, health = [], [], [], [], collections.Counter()
for b in ("bird", "spider"):
    rp = W / f"raw/avail_G_{b}.jsonl"
    if not rp.exists():
        continue
    S = BI.sqlmod(b)
    gold = {int(q["question_id"]): (q["db_id"], q["SQL"]) for q in S.load_questions()}
    G = {}
    for r in map(json.loads, open(rp)):
        if r["unit_id"] not in G or r.get("status") == "ok":
            G[r["unit_id"]] = r
    rows = list(csv.DictReader(open(W / f"data/avail_rules_{b}.csv")))
    y = collections.defaultdict(dict)
    for r in rows:
        q, k = int(r["question_id"]), int(r["k"])
        u = f"avail|G|{b}|{q}|k{k}"
        g = G.get(u)
        gen = 0
        if k >= 2 and g and g.get("status") == "ok":
            sql = S.parse_sql(BI.final_channel(g.get("final_text") or ""))
            health[f"{b}_G_length"] += g.get("finish_reason") == "length"
            health[f"{b}_G_unparsed"] += not sql
            db, gs = gold[q]
            gen = int(bool(sql) and S.execution_match(db, sql, gs).get("match") is True)
        elif k >= 2:
            health[f"{b}_G_missing_k{k}"] += 1
        rec = dict(benchmark=b, question_id=q, db_id=r["db_id"], k=k, n_live=int(r["n_live"]), Oracle=int(r["Oracle"]),
                   Original=int(r["Original"]), Valid=int(r["Valid"]), Nonempty=int(r["Nonempty"]), Generation=gen if k >= 2 else None)
        per.append(rec)
        y[k][(q,)] = rec
    cl = {(int(r["question_id"]),): r["db_id"] for r in rows}
    for k in sorted(y):
        for wt in ("cluster_equal", "unit"):
            wn = "database_equal" if wt == "cluster_equal" else "question_equal"
            for m in ("Oracle", "Original", "Valid", "Nonempty") + (("Generation",) if k >= 2 else ()):
                r = C.cluster_boot({u: v[m] for u, v in y[k].items()}, cl, wt, DRAWS, SEED)
                levels.append(dict(benchmark=b, k=k, weighting=wn, method=m, pct=round(100 * r["point"], 2), lo=round(100 * r["lo"], 2), hi=round(100 * r["hi"], 2)))
            if k >= 2:
                for base in ("Valid", "Nonempty"):
                    r = C.cluster_boot({u: v["Generation"] - v[base] for u, v in y[k].items()}, cl, wt, DRAWS, SEED)
                    cons.append(dict(benchmark=b, k=k, weighting=wn, contrast=f"Generation - {base}", effect_pp=round(100 * r["point"], 2),
                                     lo_pp=round(100 * r["lo"], 2), hi_pp=round(100 * r["hi"], 2), p_boot=round(r["p_boot"], 4)))
    if 2 in y and 6 in y:
        for wt in ("cluster_equal", "unit"):
            wn = "database_equal" if wt == "cluster_equal" else "question_equal"
            for name, f in (("Oracle", lambda v: v["Oracle"]), ("Valid", lambda v: v["Valid"]), ("Nonempty", lambda v: v["Nonempty"]),
                            ("Generation", lambda v: v["Generation"]), ("Generation - Valid", lambda v: v["Generation"] - v["Valid"]),
                            ("Generation - Nonempty", lambda v: v["Generation"] - v["Nonempty"])):
                r = C.cluster_boot({u: f(y[6][u]) - f(y[2][u]) for u in y[2]}, cl, wt, DRAWS, SEED)
                chg.append(dict(benchmark=b, weighting=wn, quantity=f"Δ(k6 - k2) {name}", effect_pp=round(100 * r["point"], 2),
                                lo_pp=round(100 * r["lo"], 2), hi_pp=round(100 * r["hi"], 2), p_boot=round(r["p_boot"], 4)))
for name, rows in (("avail_per_question.csv", per), ("avail_levels.csv", levels), ("avail_contrasts.csv", cons), ("avail_changes.csv", chg)):
    if rows:
        with open(O / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(O / "avail_health.json").write_text(json.dumps(dict(health), indent=1) + "\n")
print(json.dumps(dict(health)), len(per))
for r in chg:
    if r["weighting"] == "database_equal":
        print(r)
