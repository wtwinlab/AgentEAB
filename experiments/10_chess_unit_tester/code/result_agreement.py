r"""Descriptive: agreement of the picks by execution result.

    python3 code/result_agreement.py   # -> output/result_agreement.csv

The frozen agreement.csv compares SQL text; tie rules pick different members of the same result
cluster, so text agreement understates choosing the same answer. Here a pick is mapped to the
result_sha of its B1 execution (failures map to their status). UT = CHESS final_SQL.
"""
import collections
import csv
import json
import pathlib

W = pathlib.Path(__file__).resolve().parents[1]
P = json.load(open(W / "output/offline/per_question.json"))
ex = collections.defaultdict(dict)
for r in csv.DictReader(open(W / "output/offline/executions.csv")):
    ex[int(r["question_id"])][r["sql"].strip()] = r["result_sha"] if r["status"] == "ok" else "ERR:" + r["status"]
rows = []
for b in ("SC_official", "Valid_vote", "Nonempty_vote", "Unfiltered_vote"):
    same = differ = unmapped = 0
    for q in P:
        m = ex[q["question_id"]]
        u, v = m.get((q.get("final_sql") or "").strip()), m.get(((q.get("picks") or {}).get(b) or "").strip())
        if u is None or v is None:
            unmapped += 1
        elif u == v:
            same += 1
        else:
            differ += 1
    rows.append(dict(branch=b, same_result_as_UT=same, different_result=differ, unmapped=unmapped, of=len(P)))
with open(W / "output/result_agreement.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
