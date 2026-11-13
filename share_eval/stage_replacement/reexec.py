r"""Read-only re-execution for the change indicator A_u by result equality.

    python3 share_eval/stage_replacement/reexec.py [--out DIR] [--external-run DIR] [--chess-run DIR] [--bird-db DIR] [--spider-db DIR]
    # -> <out>/macsql_units_<model>_<bench>.csv, <out>/chess_units.csv, <out>/reexec_log.json

No model, API or GPU call. Every SQLite connection is mode=ro (MAC-SQL also query_only). The local DB files
are checked against the sha256 recorded with the runs. Gold is executed only to re-score the
re-executed outputs and compare with the stored scores (a reproduction check); A_u never reads gold.

MAC-SQL, per unit (stage = Refiner `pred`, alternative = Keep = the record's `final_sql`, the Decomposer's
last ```sql``` block, which is the Refiner's first input; for REFINER_FAILED units Keep is re-parsed from the
Decomposer reply by the formal extraction):
  A_trigger = Refiner LLM fired >= 1 time (refiner_llm_calls >= 1);
  A_text    = Refiner output text != Keep text (strip), a missing Refiner output counts as different;
  A_result  = execution classes differ under the MAC-SQL scorer (sql_common semantics).
  Units whose Refiner record exists and whose text equals Keep are not executed: same text, same class.
CHESS, per unit and alternative (stage = UT = CHESS final_SQL; alternatives SC_official / Valid_vote /
Nonempty_vote): each pick is executed after chess_clean, exactly as the EX scorer executes it.
  A_old       = the post hoc result_agreement mapping (multiset sha of the raw-SQL B1 execution);
  A_result    = CHESS EX classes differ (set of raw tuples) -- primary;
  A_sqlcommon = sql_common classes differ (first 10,000 rows, _coerce'd set) -- sensitivity.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time

import pathlib as _pl, sys as _sys
_sys.dont_write_bytecode = True
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import common as K  # noqa: E402

K.configure(__doc__.splitlines()[0])

LOG = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"))


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def class_hash(c):
    return "FAIL" if c[0] == "FAIL" else hashlib.sha256(json.dumps(sorted(map(repr, c[1]))).encode()).hexdigest()[:16]


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list({k: None for r in rows for k in r})); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------- DB integrity
db_check = {}
for line in open(K.EXT / "logs/host/db_before.sha256"):
    h, p = line.split()
    f = K.BIRD_DB / p if (K.BIRD_DB / p).exists() else K.SPIDER_DB / p
    db_check[p] = (sha_file(f) == h)
for line in open(K.CHESS / "logs/db_formal_after.sha256"):
    h, p = line.split()
    rel = p.split("dev_databases/")[-1]
    db_check["chess:" + rel] = (sha_file(K.BIRD_DB / rel) == h)
LOG["db_sha256_match"] = dict(checked=len(db_check), matched=sum(db_check.values()),
                              mismatched=sorted(k for k, v in db_check.items() if not v))
if not all(db_check.values()):
    sys.exit(f"DB hash mismatch: {LOG['db_sha256_match']['mismatched']}")

# ---------------------------------------------------------------- MAC-SQL
bird_gold = {q["question_id"]: q["SQL"] for q in json.load(open(K.BIRD / "dev.json"))}
spider_gold = {i: q["query"] for i, q in enumerate(json.load(open(K.SPIDER / "dev.json")))}
LOG["macsql"] = {}
for mt in ("q3", "q72"):
    for b in ("bird", "spider"):
        root = K.BIRD_DB if b == "bird" else K.SPIDER_DB
        gold = bird_gold if b == "bird" else spider_gold
        U = json.load(open(K.EXT / f"output/macsql_{mt}_{b}.json"))["units"]
        rows, nexec, repro_bad, t0 = [], 0, [], time.time()
        for u in U:
            keep, pred = u["boundary_sql"], u["refiner_sql"]
            failed = "REFINER_FAILED" in u["flags"]
            same_text = (not failed) and pred is not None and keep is not None and pred.strip() == keep.strip()
            yr, yk = u["scores"]["Refiner"], u["scores"]["Keep"]
            row = dict(question_id=u["question_id"], db_id=u["db_id"], flags="|".join(u["flags"]),
                       refiner_llm_calls=u["refiner_llm_calls"], trigger_reason=u["trigger_reason"],
                       A_trigger=int(u["refiner_llm_calls"] >= 1), A_text=int(not same_text), executed=0,
                       keep_class="", refiner_class="", A_result=0, Y_stage=yr, Y_alt=yk, D=yr - yk,
                       repro_stage="", repro_alt="")
            if not same_text:
                nexec += 1
                ek = K.macsql_execute(root, u["db_id"], keep) if keep else dict(status="none", rows=None)
                er = dict(status="none", rows=None) if (failed or not pred) else K.macsql_execute(root, u["db_id"], pred)
                ck, cr = K.macsql_class(ek), K.macsql_class(er)
                g = K.macsql_execute(root, u["db_id"], gold[u["question_id"]])
                cg = K.macsql_class(g)
                sc = lambda c: int(cg[0] == "OK" and c == cg)                       # noqa: E731
                rr, rk = (0 if failed else sc(cr)), sc(ck)
                row.update(executed=1, keep_class=f"{ek['status']}:{class_hash(ck)}",
                           refiner_class=f"{er['status']}:{class_hash(cr)}", A_result=int(ck != cr),
                           repro_stage=int(rr == yr), repro_alt=int(rk == yk))
                if rr != yr or rk != yk:
                    repro_bad.append(u["question_id"])
            rows.append(row)
        write(K.OUT / f"macsql_units_{mt}_{b}.csv", rows)
        LOG["macsql"][f"{mt}_{b}"] = dict(units=len(rows), executed_units=nexec, seconds=round(time.time() - t0, 1),
                                          score_reproduction_mismatches=repro_bad)
        print(mt, b, LOG["macsql"][f"{mt}_{b}"], flush=True)

# ---------------------------------------------------------------- CHESS
P = json.load(open(K.CHESS / "output/offline/per_question.json"))
tasks = {t["question_id"]: t for t in json.load(open(K.CHESS / "data/tasks_147.json"))}
b1ex = {}
for r in csv.DictReader(open(K.CHESS / "output/offline/executions.csv")):     # as code/result_agreement.py
    b1ex.setdefault(int(r["question_id"]), {})[r["sql"].strip()] = r["result_sha"] if r["status"] == "ok" else "ERR:" + r["status"]
ALTS = ("SC_official", "Valid_vote", "Nonempty_vote")
rows, repro_bad, t0 = [], [], time.time()
for q in P:
    qid, db = q["question_id"], K.BIRD_DB / q["db_id"] / f"{q['db_id']}.sqlite"
    picks = dict(UT=q["final_sql"], **{a: q["picks"].get(a) for a in ALTS})
    ex = {k: (K.chess_execute(db, K.chess_clean(v)) if v else dict(status="none", rows=None)) for k, v in picks.items()}
    g = K.chess_execute(db, tasks[qid]["SQL"])
    gold_rows = set(g["rows"]) if g["status"] == "ok" else None
    sc = {k: int(gold_rows is not None and e["status"] == "ok" and set(e["rows"]) == gold_rows) for k, e in ex.items()}
    stored = dict(UT=q["scores"]["CHESS_final_SQL"], **{a: q["scores"][a] for a in ALTS})
    if sc != stored:
        repro_bad.append(qid)
    cc = {k: K.chess_class(e) for k, e in ex.items()}
    cs = {k: K.sqlcommon_class(e) for k, e in ex.items()}
    m = b1ex.get(qid, {})
    old_u = m.get((q["final_sql"] or "").strip())
    for a in ALTS:
        old_v = m.get((picks[a] or "").strip())
        rows.append(dict(question_id=qid, db_id=q["db_id"], alternative=a, flags="|".join(q["flags"]),
                         text_differs=int((picks["UT"] or "").strip() != (picks[a] or "").strip()),
                         A_old=("" if old_u is None or old_v is None else int(old_u != old_v)),
                         A_result=int(cc["UT"] != cc[a]), A_sqlcommon=int(cs["UT"] != cs[a]),
                         stage_class=f"{ex['UT']['status']}:{class_hash(cc['UT'])}", alt_class=f"{ex[a]['status']}:{class_hash(cc[a])}",
                         Y_stage=stored["UT"], Y_alt=stored[a], D=stored["UT"] - stored[a],
                         repro_stage=int(sc["UT"] == stored["UT"]), repro_alt=int(sc[a] == stored[a])))
write(K.OUT / "chess_units.csv", rows)
LOG["chess"] = dict(questions=len(P), rows=len(rows), seconds=round(time.time() - t0, 1), score_reproduction_mismatches=repro_bad)
print("chess", LOG["chess"], flush=True)
LOG["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
(K.OUT / "reexec_log.json").write_text(json.dumps(LOG, indent=1) + "\n")
