r"""Part D analysis: DeepSeek-V4-Pro S_al / Generation vs the stored rules, and the
Qwen3-32B values of the same operations on the same questions (side by side, not pooled).

    python3 code/analyze_ds.py   # reads output/ds_per_question.csv and the pool-transfer per_question_all.csv
                                 # -> output/ds_contrasts.csv, ds_levels.csv
"""
import csv
import json
import pathlib
import sys

sys.dont_write_bytecode = True
W = pathlib.Path(__file__).resolve().parents[1]
REPO = W.parents[1]
sys.path.insert(0, str(REPO / "experiments/common/code"))
import core as C                                                           # noqa: E402

DRAWS, SEED = 10000, 20261105
ds = {(r["benchmark"], int(r["question_id"])): r for r in csv.DictReader(open(W / "output/ds_per_question.csv"))}
q3 = {(r["benchmark"], int(r["question_id"])): r for r in csv.DictReader(open(REPO / "data/derived/sql_pool_operation_transfer/per_question_all.csv"))
      if r["pool"] == "original"}
cons, lev = [], []
for b in ("bird", "spider"):
    keys = sorted(k for k in q3 if k[0] == b)
    v = {}
    for k in keys:
        d, q = ds.get(k, {}), q3[k]
        v[(k[1],)] = dict(DS_S_al=int(d.get("DS_S_al") or 0), DS_Generation=int(d.get("DS_Generation") or 0),
                          Q3_S_al=int(q["S_al"]), Q3_Generation=int(q["Generation"]), Valid=int(q["Valid"]), Nonempty=int(q["Nonempty"]),
                          Oracle=int(q["Oracle"]), missing_ds=int(not d or d.get("DS_S_al") in (None, "") or d.get("DS_Generation") in (None, "")))
    cl = {(k[1],): q3[k]["db_id"] for k in keys}
    for wt in ("cluster_equal", "unit"):
        wn = "database_equal" if wt == "cluster_equal" else "question_equal"
        for m in ("Oracle", "Valid", "Nonempty", "Q3_S_al", "Q3_Generation", "DS_S_al", "DS_Generation"):
            r = C.cluster_boot({u: x[m] for u, x in v.items()}, cl, wt, DRAWS, SEED)
            lev.append(dict(benchmark=b, weighting=wn, method=m, questions=r["units"], pct=round(100 * r["point"], 2), lo=round(100 * r["lo"], 2), hi=round(100 * r["hi"], 2)))
        for model in ("DS", "Q3"):
            fam = []
            for a_, b_ in (("S_al", "Valid"), ("S_al", "Nonempty"), ("Generation", "Valid"), ("Generation", "Nonempty"), ("Generation", "S_al")):
                A = f"{model}_{a_}"; B = b_ if b_ in ("Valid", "Nonempty") else f"{model}_{b_}"
                r = C.cluster_boot({u: x[A] - x[B] for u, x in v.items()}, cl, wt, DRAWS, SEED)
                row = dict(benchmark=b, weighting=wn, model=model, contrast=f"{A} - {B}", effect_pp=round(100 * r["point"], 2),
                           lo_pp=round(100 * r["lo"], 2), hi_pp=round(100 * r["hi"], 2), p_boot=round(r["p_boot"], 4))
                fam.append(row)
            o = sorted(range(len(fam)), key=lambda i: fam[i]["p_boot"]); stop = False
            for j, i in enumerate(o):
                rej = (not stop) and fam[i]["p_boot"] <= 0.05 / (len(fam) - j); stop = stop or not rej
                fam[i]["holm_reject"] = rej
            cons += fam
    lev.append(dict(benchmark=b, weighting="count", method="missing_DS", questions=sum(x["missing_ds"] for x in v.values())))
for name, rows in (("ds_levels.csv", lev), ("ds_contrasts.csv", cons)):
    with open(W / "output" / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list({k: None for r in rows for k in r})); w.writeheader(); w.writerows(rows)
for r in cons:
    if r["weighting"] == "database_equal":
        print(r)
