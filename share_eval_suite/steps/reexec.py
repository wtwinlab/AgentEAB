"""Read-only re-execution for the change indicator A_u by result equality.

Same outputs as `share_eval/stage_replacement/reexec.py`.

    python3 -m share_eval_suite.steps.reexec [--out DIR] [--external-run DIR] [--chess-run DIR] [--bird DIR]
        [--bird-db DIR] [--spider DIR] [--spider-db DIR]
    # -> <out>/macsql_units_<model>_<bench>.csv, <out>/chess_units.csv, <out>/reexec_log.json

No model, API or GPU call. Every SQLite connection is read-only (MAC-SQL also query_only). Each local
database file is checked against the SHA-256 recorded with the runs before any query runs; without the
BIRD and Spider databases the step stops and reports the missing files (their number and the first
five). The reference query is executed only to re-score the re-executed outputs against the stored
scores (a reproduction check); A_u never reads the reference.

MAC-SQL, per question (stage = the Refiner's query; rule = Keep, the Decomposer's last SQL block, which
is the Refiner's first input):
  A_trigger = the Refiner's LLM fired at least once;
  A_text    = the Refiner's text differs from Keep after stripping; a missing Refiner output differs;
  A_result  = the two execution classes differ under the MAC-SQL scorer.
  Questions whose Refiner output exists and equals Keep in text are not executed (same text, same class).
CHESS, per question and rule (stage = the Unit Tester's final query; rules = the self-consistency, Valid
and Nonempty votes): each pick is cleaned and executed as the CHESS scorer executes it.
  A_old       = the agreement of the logged executions of the raw queries (result digests);
  A_result    = the CHESS execution classes differ (set of raw tuples), primary;
  A_sqlcommon = the shared-scorer classes differ (first 10,000 rows, normalised), sensitivity.
D_u = Y_u(stage) - Y_u(rule) with the stored scores.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from share_eval_suite.steps import _replacement_common as C

MACSQL_CELLS = tuple((mt, b) for mt in ("q3", "q72") for b in ("bird", "spider"))
ALTS = ("SC_official", "Valid_vote", "Nonempty_vote")


class DatabasesUnavailable(RuntimeError):
    """The BIRD or Spider database files needed for execution are not present."""


class DatabaseMismatch(RuntimeError):
    """A local database file differs from the SHA-256 recorded with the runs."""


# ---------------------------------------------------------------- database integrity
def database_files(ext: pathlib.Path, chess: pathlib.Path, bird_db: pathlib.Path,
                   spider_db: pathlib.Path) -> List[Tuple[str, str, pathlib.Path]]:
    """(check key, recorded SHA-256, local file) for every database the runs recorded. MAC-SQL paths are
    relative to the BIRD directory, else to the Spider directory; CHESS paths end in dev_databases/<rel>."""
    out = []
    for line in open(ext / "logs/host/db_before.sha256"):
        h, p = line.split()
        out.append((p, h, bird_db / p if (bird_db / p).exists() else spider_db / p))
    for line in open(chess / "logs/db_formal_after.sha256"):
        h, p = line.split()
        rel = p.split("dev_databases/")[-1]
        out.append(("chess:" + rel, h, bird_db / rel))
    return out


def check_databases(ext, chess, bird_db, spider_db) -> dict:
    files = database_files(ext, chess, bird_db, spider_db)
    missing = sorted({str(f) for _, _, f in files if not f.is_file()})
    if missing:
        raise DatabasesUnavailable(f"{len(missing)} database files are missing (set --bird-db/--spider-db or "
                                   f"$BIRD_DB_DIR/$SPIDER_DB_DIR): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    ok = {}
    for key, h, f in files:
        ok[key] = (C.sha_file(f) == h)
    r = dict(checked=len(ok), matched=sum(ok.values()), mismatched=sorted(k for k, v in ok.items() if not v))
    if not all(ok.values()):
        raise DatabaseMismatch(f"DB hash mismatch: {r['mismatched']}")
    return r


# ---------------------------------------------------------------- MAC-SQL
def macsql_base(u: dict) -> Tuple[dict, bool]:
    """The unit row's fields that need no execution, and whether the unit is executed (A_text = 1).
    Execution fields hold their defaults (empty classes, A_result 0)."""
    keep, pred = u["boundary_sql"], u["refiner_sql"]
    failed = "REFINER_FAILED" in u["flags"]
    same_text = (not failed) and pred is not None and keep is not None and pred.strip() == keep.strip()
    yr, yk = u["scores"]["Refiner"], u["scores"]["Keep"]
    row = dict(question_id=u["question_id"], db_id=u["db_id"], flags="|".join(u["flags"]),
               refiner_llm_calls=u["refiner_llm_calls"], trigger_reason=u["trigger_reason"],
               A_trigger=int(u["refiner_llm_calls"] >= 1), A_text=int(not same_text), executed=0,
               keep_class="", refiner_class="", A_result=0, Y_stage=yr, Y_alt=yk, D=yr - yk,
               repro_stage="", repro_alt="")
    return row, not same_text


def macsql_row(u: dict, root, gold: Dict, execute: Callable = C.macsql_execute) -> Tuple[dict, bool]:
    """One MAC-SQL unit row (`gold`: reference queries by question id); the second value is True when a
    re-score differs from the stored score."""
    row, run = macsql_base(u)
    if not run:
        return row, False
    keep, pred = u["boundary_sql"], u["refiner_sql"]
    failed = "REFINER_FAILED" in u["flags"]
    yr, yk = row["Y_stage"], row["Y_alt"]
    ek = execute(root, u["db_id"], keep) if keep else dict(C.NO_OUTPUT)
    er = dict(C.NO_OUTPUT) if (failed or not pred) else execute(root, u["db_id"], pred)
    ck, cr = C.macsql_class(ek), C.macsql_class(er)
    cg = C.macsql_class(execute(root, u["db_id"], gold[u["question_id"]]))

    def sc(c):
        return int(cg[0] == "OK" and c == cg)
    rr, rk = (0 if failed else sc(cr)), sc(ck)
    row.update(executed=1, keep_class=f"{ek['status']}:{C.class_hash(ck)}",
               refiner_class=f"{er['status']}:{C.class_hash(cr)}", A_result=int(ck != cr),
               repro_stage=int(rr == yr), repro_alt=int(rk == yk))
    return row, (rr != yr or rk != yk)


def macsql_gold(bird: pathlib.Path, spider: pathlib.Path) -> Dict[str, dict]:
    """Reference queries by question id: BIRD by `question_id`, Spider by position in dev.json."""
    return dict(bird={q["question_id"]: q["SQL"] for q in json.load(open(bird / "dev.json"))},
                spider={i: q["query"] for i, q in enumerate(json.load(open(spider / "dev.json")))})


# ---------------------------------------------------------------- CHESS
def chess_logged_executions(chess: pathlib.Path) -> Dict[int, Dict[str, str]]:
    """Result digest (or ERR:<status>) of every logged execution of a raw query, by question and stripped text."""
    b1ex: Dict[int, Dict[str, str]] = {}
    with open(chess / "output/offline/executions.csv") as f:
        for r in csv.DictReader(f):
            b1ex.setdefault(int(r["question_id"]), {})[r["sql"].strip()] = \
                r["result_sha"] if r["status"] == "ok" else "ERR:" + r["status"]
    return b1ex


def chess_picks(q: dict) -> Dict[str, Optional[str]]:
    """The stage's query (UT, the CHESS final query) and each rule's pick."""
    return dict(UT=q["final_sql"], **{a: q["picks"].get(a) for a in ALTS})


def chess_stored(q: dict) -> Dict[str, int]:
    return dict(UT=q["scores"]["CHESS_final_SQL"], **{a: q["scores"][a] for a in ALTS})


def chess_base(q: dict, b1ex: Dict[int, Dict[str, str]]) -> List[dict]:
    """The unit rows of one question (one per rule) with the fields that need no execution; execution
    fields are None."""
    qid = q["question_id"]
    picks, stored = chess_picks(q), chess_stored(q)
    m = b1ex.get(qid, {})
    old_u = m.get((q["final_sql"] or "").strip())
    rows = []
    for a in ALTS:
        old_v = m.get((picks[a] or "").strip())
        rows.append(dict(question_id=qid, db_id=q["db_id"], alternative=a, flags="|".join(q["flags"]),
                         text_differs=int((picks["UT"] or "").strip() != (picks[a] or "").strip()),
                         A_old=("" if old_u is None or old_v is None else int(old_u != old_v)),
                         A_result=None, A_sqlcommon=None, stage_class=None, alt_class=None,
                         Y_stage=stored["UT"], Y_alt=stored[a], D=stored["UT"] - stored[a],
                         repro_stage=None, repro_alt=None))
    return rows


def chess_rows(q: dict, b1ex, bird_db: pathlib.Path, gold_sql: str,
               execute: Callable = C.chess_execute) -> Tuple[List[dict], bool]:
    """The unit rows of one question; the second value is True when a re-score differs from the stored score."""
    db = bird_db / q["db_id"] / f"{q['db_id']}.sqlite"
    picks, stored = chess_picks(q), chess_stored(q)
    ex = {k: (execute(db, C.chess_clean(v)) if v else dict(C.NO_OUTPUT)) for k, v in picks.items()}
    g = execute(db, gold_sql)
    gold_rows = set(g["rows"]) if g["status"] == "ok" else None
    sc = {k: int(gold_rows is not None and e["status"] == "ok" and set(e["rows"]) == gold_rows) for k, e in ex.items()}
    cc = {k: C.chess_class(e) for k, e in ex.items()}
    cs = {k: C.sqlcommon_class(e) for k, e in ex.items()}
    rows = chess_base(q, b1ex)
    for r in rows:
        a = r["alternative"]
        r.update(A_result=int(cc["UT"] != cc[a]), A_sqlcommon=int(cs["UT"] != cs[a]),
                 stage_class=f"{ex['UT']['status']}:{C.class_hash(cc['UT'])}",
                 alt_class=f"{ex[a]['status']}:{C.class_hash(cc[a])}",
                 repro_stage=int(sc["UT"] == stored["UT"]), repro_alt=int(sc[a] == stored[a]))
    return rows, sc != stored


# ---------------------------------------------------------------- the step
def run(out=C.OUT, external_run=C.EXT, chess_run=C.CHESS, bird=C.BIRD, bird_db=C.BIRD / "dev_databases",
        spider=C.SPIDER, spider_db=C.SPIDER_DB, verbose: bool = True) -> dict:
    """Write the MAC-SQL and CHESS unit files and the log; returns the log."""
    out, ext, chess = pathlib.Path(out), pathlib.Path(external_run), pathlib.Path(chess_run)
    bird, bird_db, spider, spider_db = (pathlib.Path(p) for p in (bird, bird_db, spider, spider_db))
    out.mkdir(parents=True, exist_ok=True)
    log = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    log["db_sha256_match"] = check_databases(ext, chess, bird_db, spider_db)

    gold = macsql_gold(bird, spider)
    log["macsql"] = {}
    for mt, b in MACSQL_CELLS:
        root = bird_db if b == "bird" else spider_db
        units = json.load(open(ext / f"output/macsql_{mt}_{b}.json"))["units"]
        rows, nexec, repro_bad, t0 = [], 0, [], time.time()
        for u in units:
            row, bad = macsql_row(u, root, gold[b])
            nexec += row["executed"]
            if bad:
                repro_bad.append(u["question_id"])
            rows.append(row)
        C.write_table(out / f"macsql_units_{mt}_{b}.csv", rows)
        log["macsql"][f"{mt}_{b}"] = dict(units=len(rows), executed_units=nexec, seconds=round(time.time() - t0, 1),
                                          score_reproduction_mismatches=repro_bad)
        if verbose:
            print(mt, b, log["macsql"][f"{mt}_{b}"], flush=True)

    P = json.load(open(chess / "output/offline/per_question.json"))
    tasks = {t["question_id"]: t for t in json.load(open(chess / "data/tasks_147.json"))}
    b1ex = chess_logged_executions(chess)
    rows, repro_bad, t0 = [], [], time.time()
    for q in P:
        rs, bad = chess_rows(q, b1ex, bird_db, tasks[q["question_id"]]["SQL"])
        if bad:
            repro_bad.append(q["question_id"])
        rows += rs
    C.write_table(out / "chess_units.csv", rows)
    log["chess"] = dict(questions=len(P), rows=len(rows), seconds=round(time.time() - t0, 1),
                        score_reproduction_mismatches=repro_bad)
    if verbose:
        print("chess", log["chess"], flush=True)
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    C.write_json(out / "reexec_log.json", log)
    return log


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = C.parse_args(__doc__.splitlines()[0], argv)
    try:
        run(a.out, a.external_run, a.chess_run, a.bird, a.bird_db, a.spider, a.spider_db)
    except DatabasesUnavailable as e:
        print(f"reexec needs to execute queries on the BIRD and Spider databases: {e}", file=sys.stderr)
        return 2
    except DatabaseMismatch as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
