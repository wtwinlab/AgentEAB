r"""Part A inputs: parse, execute (read-only) and label the six Qwen3-32B samples, draw
the nested subsets, apply the rules, and write the generation requests for k = 2..6.

    python3 code/build_avail.py --bench bird     # -> data/avail_{rules,labels}_<bench>.*, data/avail_requests_G_<bench>.jsonl

Builders are imported from the pool-transfer round (`build_inputs.sqlmod/final_channel/generation_prompt`,
`collect_sql_arms`, `b1_selectors`), not rewritten. Gold is used only for the label file.
"""
import argparse
import collections
import csv
import json
import pathlib
import random
import sys

sys.dont_write_bytecode = True
W = pathlib.Path(__file__).resolve().parents[1]
REPO = W.parents[1]
sys.path.insert(0, str(REPO / "experiments/common/code"))
import build_inputs as BI                                                  # noqa: E402

SLOTS = tuple(f"qwen3-32b|s{i}" for i in range(6))
RULES = ("released", "valid_majority", "nonempty_majority")
NAME = {"released": "Original", "valid_majority": "Valid", "nonempty_majority": "Nonempty"}

ap = argparse.ArgumentParser(); ap.add_argument("--bench", required=True); bench = ap.parse_args().bench
S = BI.sqlmod(bench)
import collect_sql_arms as A                                              # noqa: E402
import b1_selectors as B1                                                 # noqa: E402
qs = {int(q["question_id"]): q for q in S.load_questions()}
gold = {qid: q["SQL"] for qid, q in qs.items()}
recs = {}
for r in map(json.loads, open(W / f"raw/avail_cand_{bench}.jsonl")):
    if r.get("status") == "ok":
        recs[(int(r["question_id"]), f"{r['source_family']}|s{r['sample_index']}")] = r
gen_seed = {}
for r in map(json.loads, open(BI.GEN_RAW / f"sql_{bench}_qwen3-32b.jsonl")):
    if r["unit_id"].endswith("|r0"):                 # the matched generation unit, as build_inputs
        gen_seed[int(r["unit_id"].split("|")[2])] = r["request_body"]["seed"]
cands, labels, health = {}, [], collections.Counter()
for qid in sorted(qs):
    db = qs[qid]["db_id"]
    for slot in SLOTS:
        r = recs.get((qid, slot))
        sql = S.parse_sql(BI.final_channel(r["final_content"])) if r else None
        health["missing_record"] += r is None
        health["length"] += bool(r) and r.get("finish_reason") == "length"
        if not sql:
            c = dict(model=slot, sql="", exec={"status": "unparsed", "n_rows_returned": 0}, exec_status="unparsed"); m = False
            health["unparsed"] += 1
        else:
            ex = S.execute(db, sql)
            c = dict(model=slot, sql=sql, exec=ex, exec_status=ex.get("status"))
            m = bool(S.execution_match(db, sql, gold[qid]).get("match") is True)
        cands[(qid, slot)] = c
        labels.append(dict(benchmark=bench, question_id=qid, db_id=db, model=slot, match=m))
lab = {(x["question_id"], x["model"]): x["match"] for x in labels}
rows, reqs = [], []
for qid in sorted(qs):
    q = qs[qid]
    perm = list(SLOTS); random.Random(f"avail|{bench}|{qid}").shuffle(perm)
    for k in range(1, 7):
        sub = sorted(perm[:k])                      # base (sorted slot) order, as the stored pool
        cl = [cands[(qid, s)] for s in sub]
        live = [c for c in cl if c["sql"]]
        row = dict(benchmark=bench, question_id=qid, db_id=q["db_id"], k=k, slots="|".join(sub), n_live=len(live),
                   Oracle=int(any(lab[(qid, s)] for s in sub)))
        for s in RULES:
            pick = B1.SELECTORS[s](live, None) if live else None
            row[NAME[s]] = int(bool(pick) and lab[(qid, pick["model"])])
            row[NAME[s] + "_pick"] = pick["model"] if pick else ""
        rows.append(row)
        if k >= 2:
            gp, npl = BI.generation_prompt(S, A, q, cl)
            if gp is None:
                health[f"generation_fewer_than_two_k{k}"] += 1
            else:
                reqs.append(dict(benchmark=bench, question_id=qid, db_id=q["db_id"], k=k, slots="|".join(sub),
                                 unit_id=f"avail|G|{bench}|{qid}|k{k}", n_peers_shown=npl,
                                 messages=[{"role": "user", "content": gp}], seed=gen_seed[qid]))
with open(W / f"data/avail_rules_{bench}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(W / f"data/avail_labels_{bench}.jsonl", "w") as f:
    for x in labels:
        f.write(json.dumps(x) + "\n")
with open(W / f"data/avail_requests_G_{bench}.jsonl", "w") as f:
    for x in reqs:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(bench, len(qs), "requests", len(reqs), dict(health))
