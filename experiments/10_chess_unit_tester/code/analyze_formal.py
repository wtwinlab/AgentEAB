r"""Formal analysis. Offline; reads the extracted per-question records and proxy logs.

    python3 code/analyze_formal.py   # -> output/levels.csv, contrasts.csv, per_db.csv, agreement.csv,
                                     #    costs_per_question.csv, flags.json, summary.json

Primary UT branch = CHESS's returned final_SQL (UT's selection wherever UT ran). Sensitivities:
UT failures scored 0; UT-ran subset. Contrasts UT - {SC_official, Valid_vote, Nonempty_vote}, Holm
0.05 over the 3; question-equal primary, database-equal sensitivity; paired cluster bootstrap over
databases (share_eval core, 10,000 draws, seed 20261105); restricted wild cluster bootstrap-t
(`experiments/common/code/a_sql_small_cluster.py`, exact for 11 clusters) as a small-cluster sensitivity.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True
W = pathlib.Path(__file__).resolve().parents[1]
REPO = W.parents[1]
sys.path.insert(0, str(REPO / "experiments/common/code"))
sys.path.insert(0, str(REPO / "experiments/common/code"))
import core as C                                                            # noqa: E402
import a_sql_small_cluster as WC                                           # noqa: E402

PRIMARY = ("SC_official", "Valid_vote", "Nonempty_vote")
BRANCHES = ("UT", "UT_fail0", "SC_official", "Valid_vote", "Nonempty_vote", "Unfiltered_vote",
            "First_executable_diag", "First_nonempty_diag", "Any_correct")
DRAWS, SEED = 10000, 20261105


def holm(rows):
    o = sorted(range(len(rows)), key=lambda i: rows[i]["p_boot"])
    stop = False
    for j, i in enumerate(o):
        rej = (not stop) and rows[i]["p_boot"] <= 0.05 / (len(rows) - j)
        stop = stop or not rej
        rows[i]["holm_reject"] = rej
        rows[i]["holm_family_size"] = len(rows)


def main() -> int:
    perq = json.load(open(W / "output/offline/per_question.json"))
    assert len(perq) == 147, len(perq)
    y, db = {}, {}
    for q in perq:
        k = (q["question_id"],)
        db[k] = q["db_id"]
        sc = q.get("scores") or {}
        ut_ran = q.get("status") == "COMPLETED"
        y[k] = dict(UT=sc.get("CHESS_final_SQL", 0), UT_fail0=sc.get("UT_full", 0) if ut_ran else 0,
                    **{b: sc.get(b, 0) for b in BRANCHES if b not in ("UT", "UT_fail0")}, ut_ran=ut_ran)
    keys = sorted(y)
    ran = [k for k in keys if y[k]["ut_ran"]]
    levels, contrasts = [], []
    for wt in ("unit", "cluster_equal"):
        for b in BRANCHES:
            r = C.cluster_boot({k: y[k][b] for k in keys}, db, wt, DRAWS, SEED)
            levels.append(dict(weighting="question_equal" if wt == "unit" else "database_equal", branch=b,
                               questions=r["units"], databases=r["clusters"], pct=round(100 * r["point"], 2),
                               lo=round(100 * r["lo"], 2), hi=round(100 * r["hi"], 2)))
        for ut, pop, kk in (("UT", "all_147", keys), ("UT_fail0", "all_147", keys), ("UT", "ut_ran", ran)):
            fam = []
            for b in PRIMARY:
                vals = {k: y[k][ut] - y[k][b] for k in kk}
                r = C.cluster_boot(vals, db, wt, DRAWS, SEED)
                by = collections.defaultdict(list)
                for k in kk:
                    by[db[k]].append(vals[k])
                dbs = sorted(by)
                arrs = [np.array(by[d], float) for d in dbs]
                means = np.array([a.mean() for a in arrs])
                w = np.ones(len(dbs)) if wt == "cluster_equal" else np.array([len(a) for a in arrs], float)
                wild = WC.wild_cluster_t(arrs, w, means, WC.DRAWS, np.random.default_rng(WC.SEED))
                row = dict(weighting="question_equal" if wt == "unit" else "database_equal", ut_branch=ut, population=pop,
                           contrast=f"{ut} - {b}", questions=len(kk), effect_pp=round(100 * r["point"], 2),
                           lo_pp=round(100 * r["lo"], 2), hi_pp=round(100 * r["hi"], 2), p_boot=round(r["p_boot"], 4),
                           ut_only_correct=sum(1 for k in kk if y[k][ut] and not y[k][b]),
                           rule_only_correct=sum(1 for k in kk if y[k][b] and not y[k][ut]),
                           wild_p=wild.get("p_value"), wild_ci_lo_pp=wild.get("ci_lo_pp"), wild_ci_hi_pp=wild.get("ci_hi_pp"))
                fam.append(row)
            holm(fam)
            for r in fam:
                r["family"] = "primary" if (ut == "UT" and pop == "all_147" and wt == "unit") else "sensitivity"
            contrasts += fam
    per_db = []
    for d in sorted(set(db.values())):
        kk = [k for k in keys if db[k] == d]
        per_db.append(dict(db_id=d, questions=len(kk), ut_ran=sum(y[k]["ut_ran"] for k in kk),
                           **{b: sum(y[k][b] for k in kk) for b in BRANCHES}))
    agree = []
    picks = {q["question_id"]: q.get("picks") or {} for q in perq}
    for b in PRIMARY:
        same = sum(1 for q in perq if q.get("picks") and q["picks"].get(b) and q.get("final_sql")
                   and q["picks"][b].strip() == q["final_sql"].strip())
        agree.append(dict(branch=b, same_sql_as_UT=same, of=len(perq)))
    flags = collections.Counter(f.split(":")[0] for q in perq for f in q.get("flags", []))
    status = collections.Counter(q.get("status") for q in perq)
    calls = [json.loads(l) for p in sorted((W / "raw").glob("formal_s*.jsonl")) for l in open(p)]
    cq = collections.defaultdict(lambda: collections.Counter())
    first_ut = {}
    for c in sorted(calls, key=lambda c: c.get("ts_start") or 0):
        if c["refused"]:
            continue
        if c["tool"] in ("agent_tool_choice:unit_tester", "generate_unit_test", "evaluate"):
            first_ut.setdefault(c["question"], c["ts_start"])
    for c in calls:
        if c["refused"]:
            continue
        part = "ut_increment" if c["question"] in first_ut and c["ts_start"] >= first_ut[c["question"]] else "prefix"
        u = c.get("usage") or {}
        a = cq[(c["question"], part)]
        a["calls"] += 1; a["prompt_tokens"] += u.get("prompt_tokens") or 0
        a["completion_tokens"] += u.get("completion_tokens") or 0; a["latency_s"] += c.get("latency_s") or 0
        a["non200"] += c.get("status") != 200
    cost_rows = [dict(question=q, part=p, **{k: round(v, 3) for k, v in a.items()}) for (q, p), a in sorted(cq.items())]
    exec_s = {q["question_id"]: q.get("rule_execution_seconds") for q in perq}

    def agg(part):
        rs = [a for (q, p), a in cq.items() if p == part and q.startswith("F:q")]
        return dict(questions=len(rs), calls=sum(a["calls"] for a in rs), prompt_tokens=sum(a["prompt_tokens"] for a in rs),
                    completion_tokens=sum(a["completion_tokens"] for a in rs),
                    mean_calls=round(np.mean([a["calls"] for a in rs]), 2) if rs else None)
    out = W / "output"
    for name, rs in (("levels.csv", levels), ("contrasts.csv", contrasts), ("per_db.csv", per_db),
                     ("agreement.csv", agree), ("costs_per_question.csv", cost_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list({k: None for r in rs for k in r})); w.writeheader(); w.writerows(rs)
    summ = dict(questions=len(perq), status=dict(status), flags=dict(flags), ut_ran=len(ran),
                calls_total=sum(1 for c in calls if not c["refused"]), calls_non200=sum(1 for c in calls if not c["refused"] and c.get("status") != 200),
                cost_prefix=agg("prefix"), cost_ut_increment=agg("ut_increment"),
                rule_execution_seconds=dict(total=round(sum(v for v in exec_s.values() if v), 2),
                                            max=max(v for v in exec_s.values() if v is not None)),
                primary=[r for r in contrasts if r["family"] == "primary"])
    (out / "flags.json").write_text(json.dumps(dict(status=dict(status), flags=dict(flags),
                                                    per_question={q["question_id"]: q.get("flags") for q in perq if q.get("flags")}), indent=1) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
    print(json.dumps(summ, indent=1, default=str))
    for r in levels:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
