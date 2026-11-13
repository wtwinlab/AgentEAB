"""The stage-replacement steps of the suite write the same files as the `share_eval/stage_replacement/`
scripts on the same inputs.

- throughput, decompose, costs_comparable: run from the unit files shipped in
  `paper_objects/fig_stage-act/inputs/` and the MAC-SQL, ComEM and CHESS runs (hosted data); every
  output is compared byte for byte, and with the outputs of the hosted `stage_decomposition/` folder up
  to 1e-12 relative in each number.
- reexec: on small SQLite databases built here (always), both write the same unit files and log; on
  the BIRD and Spider databases (only with `BIRD_DB_DIR` and `SPIDER_DB_DIR`) the same, and the unit
  files equal the shipped ones except the result-class digests of queries that read the current time.
  Without databases, the unit fields that need no execution are checked against the shipped unit
  files, the execution classes against the script's shared functions, and the step reports the missing
  databases.

Fields removed before comparing `reexec_log.json` (all other content must match exactly):
- `started`, `finished`: wall-clock time of the run;
- `seconds` of each MAC-SQL cell and of CHESS: wall-clock duration of the executions.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys

import pytest

from share_eval_suite.steps import _replacement_common as C
from share_eval_suite.steps import costs_comparable, decompose, reexec, throughput

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "share_eval/stage_replacement"
DATA = pathlib.Path(os.environ.get("SHARE_EVAL_DATA_ROOT") or REPO)
EXT = DATA / "data/records/macsql_comem_stages"
CHESS = DATA / "data/records/chess_unit_tester"
HOSTED_UNITS = EXT / "stage_decomposition"
SHIPPED = REPO / "paper_objects/fig_stage-act/inputs"
UNIT_FILES = [f"macsql_units_{mt}_{b}.csv" for mt, b in reexec.MACSQL_CELLS] + ["chess_units.csv"]
STEP_MODULES = dict(reexec=reexec, throughput=throughput, decompose=decompose, costs_comparable=costs_comparable)
OUTPUTS = dict(
    throughput=["throughput_by_server.csv", "stage_gpu_seconds.csv", "throughput.json"],
    decompose=["decomposition.csv", "decomposition.json", "movers.csv", "costs.csv",
               "comem_per_dataset.csv", "comem_pair_level.csv"],
    costs_comparable=["costs_comparable.csv"],
)
# wall-clock fields of reexec_log.json (see the module docstring)
LOG_TIME_FIELDS = ("started", "finished")


def _need_runs():
    missing = [str(p) for p in (EXT / "output", EXT / "raw", EXT / "data/llm4em", CHESS / "output", CHESS / "raw")
               if not p.exists()]
    if missing:
        pytest.skip(f"MAC-SQL, ComEM and CHESS runs missing: {missing}")


def _script(name, args):
    env = dict(os.environ, ARTIFACT_ROOT=str(DATA))
    r = subprocess.run([sys.executable, "-I", "-B", str(SCRIPTS / f"{name}.py"), *map(str, args)],
                       capture_output=True, text=True, env=env, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-2000:]
    return r


def _step(name, args):
    with contextlib.redirect_stdout(io.StringIO()):
        rc = STEP_MODULES[name].main([str(a) for a in args])
    assert rc == 0


def _same_bytes(a: pathlib.Path, b: pathlib.Path, names):
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n


def _strip_log(path: pathlib.Path) -> dict:
    log = json.loads(path.read_text())
    for k in LOG_TIME_FIELDS:
        log.pop(k)
    for cell in log["macsql"].values():
        cell.pop("seconds")
    log["chess"].pop("seconds")
    return log


def _load_common():
    spec = importlib.util.spec_from_file_location("_stage_replacement_common", SCRIPTS / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- command line
@pytest.mark.parametrize("name", list(STEP_MODULES))
def test_same_options(name, capsys):
    ref = _script(name, ["--help"]).stdout
    with pytest.raises(SystemExit):
        STEP_MODULES[name].main(["--help"])
    ours = capsys.readouterr().out
    opts = lambda s: sorted(set(re.findall(r"--[a-z][a-z-]*", s)))  # noqa: E731
    assert opts(ours) == opts(ref)


# ---------------------------------------------------------------- throughput, decompose, costs_comparable
def test_throughput_decompose_costs_identical(tmp_path):
    _need_runs()
    ref, ours = tmp_path / "ref", tmp_path / "suite"
    for d in (ref, ours):
        d.mkdir()
        for f in UNIT_FILES:
            shutil.copy(SHIPPED / f, d / f)
    for name in ("throughput", "decompose", "costs_comparable"):
        args = ["--external-run", EXT, "--chess-run", CHESS]
        _script(name, ["--out", ref, *args])
        _step(name, ["--out", ours, *args])
        _same_bytes(ref, ours, OUTPUTS[name])
    if HOSTED_UNITS.is_dir():           # the outputs the hosted data ships with
        _same_as_recorded(HOSTED_UNITS, ours, [n for v in OUTPUTS.values() for n in v if (HOSTED_UNITS / n).exists()])


def _numbers_equal(a: str, b: str) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
    except ValueError:
        return False


def _same_as_recorded(recorded: pathlib.Path, ours: pathlib.Path, names):
    """Outputs recorded on another machine: identical text, except that a number may differ in its last
    digits (1e-12 relative), as floating-point sums do across Python and numpy builds. The script and the
    suite on the same machine are compared byte for byte."""
    tok = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\D+")
    for n in names:
        a, b = (recorded / n).read_text(), (ours / n).read_text()
        if a == b:
            continue
        ta, tb = tok.findall(a), tok.findall(b)
        assert len(ta) == len(tb), n
        for x, y in zip(ta, tb):
            assert x == y or _numbers_equal(x, y), (n, x, y)


# ---------------------------------------------------------------- reexec without databases
def test_shipped_unit_files_equal_hosted():
    if not HOSTED_UNITS.is_dir():
        pytest.skip(f"hosted unit files missing: {HOSTED_UNITS}")
    for f in UNIT_FILES:
        assert (HOSTED_UNITS / f).read_bytes() == (SHIPPED / f).read_bytes(), f
        other = REPO / "paper_objects/tab_replacement-cost/inputs" / f
        if other.exists():
            assert other.read_bytes() == (SHIPPED / f).read_bytes(), f


MACSQL_NO_EXEC = ("question_id", "db_id", "flags", "refiner_llm_calls", "trigger_reason", "A_trigger", "A_text",
                  "Y_stage", "Y_alt", "D")
CHESS_NO_EXEC = ("question_id", "db_id", "alternative", "flags", "text_differs", "A_old", "Y_stage", "Y_alt", "D")


def _as_csv(tmp_path, rows):
    p = tmp_path / "rows.csv"
    C.write_table(p, rows)
    with open(p, newline="") as f:
        header = f.readline()
    return header, C.read_table(p)


def test_reexec_fields_without_execution(tmp_path):
    """A_trigger, A_text, the executed flag, Keep, the picks, A_old and the stored scores, from the records."""
    _need_runs()
    log = json.loads((HOSTED_UNITS / "reexec_log.json").read_text()) if (HOSTED_UNITS / "reexec_log.json").exists() else None
    for mt, b in reexec.MACSQL_CELLS:
        units = json.load(open(EXT / f"output/macsql_{mt}_{b}.json"))["units"]
        rows = [reexec.macsql_base(u)[0] for u in units]
        header, got = _as_csv(tmp_path, rows)
        with open(SHIPPED / f"macsql_units_{mt}_{b}.csv", newline="") as f:
            assert f.readline() == header
        want = C.read_table(SHIPPED / f"macsql_units_{mt}_{b}.csv")
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert {k: g[k] for k in MACSQL_NO_EXEC} == {k: w[k] for k in MACSQL_NO_EXEC}
            assert w["executed"] == g["A_text"]   # executed exactly when the texts differ
            if w["executed"] == "0":
                assert g == w                     # an unexecuted unit row needs no execution at all
        if log:
            assert log["macsql"][f"{mt}_{b}"]["units"] == len(rows)
            assert log["macsql"][f"{mt}_{b}"]["executed_units"] == sum(r["A_text"] == "1" for r in got)
    P = json.load(open(CHESS / "output/offline/per_question.json"))
    b1ex = reexec.chess_logged_executions(CHESS)
    rows = [r for q in P for r in reexec.chess_base(q, b1ex)]
    header, got = _as_csv(tmp_path, rows)
    with open(SHIPPED / "chess_units.csv", newline="") as f:
        assert f.readline() == header
    want = C.read_table(SHIPPED / "chess_units.csv")
    assert [{k: g[k] for k in CHESS_NO_EXEC} for g in got] == [{k: w[k] for k in CHESS_NO_EXEC} for w in want]
    if log:
        assert (log["chess"]["questions"], log["chess"]["rows"]) == (len(P), len(rows))


def test_execution_classes_match_script_functions():
    K = _load_common()
    execs = [dict(status="none", rows=None), dict(status="error", rows=None), dict(status="timeout", rows=None),
             dict(status="ok", rows=[]), dict(status="ok", rows=[(1, 0.30000000000000004, b"\x01\x02", "a")]),
             dict(status="ok", rows=[(1, 0.3, b"\x0a\x0b", "a"), (1, 0.3, b"\x0a\x0b", "a")]),
             dict(status="ok", rows=[(i,) for i in range(10005)])]
    for e in execs:
        assert C.macsql_class(e) == K.macsql_class(e)
        assert C.chess_class(e) == K.chess_class(e)
        assert C.sqlcommon_class(e) == K.sqlcommon_class(e)
    for v in (b"ab", bytearray(b"abc"), 1.23456789, 3, "x", None):
        assert C.coerce(v) == K.coerce(v)
    sqls = ['SELECT "a" FROM t\n', "`SELECT 1.`", " SELECT 2 "]
    if (CHESS / "output/offline/per_question.json").exists():
        for q in json.load(open(CHESS / "output/offline/per_question.json")):
            sqls += [v for v in reexec.chess_picks(q).values() if v]
    for s in sqls:
        assert C.chess_clean(s) == K.chess_clean(s)


def test_reexec_reports_missing_databases(tmp_path, capsys):
    _need_runs()
    out = tmp_path / "out"
    rc = reexec.main(["--out", str(out), "--external-run", str(EXT), "--chess-run", str(CHESS),
                      "--bird-db", str(tmp_path / "none"), "--spider-db", str(tmp_path / "none")])
    assert rc == 2
    assert "database files are missing" in capsys.readouterr().err
    assert not any(out.glob("*.csv")) and not (out / "reexec_log.json").exists()


# ---------------------------------------------------------------- reexec on small databases built here
def _sqlite(path: pathlib.Path, script: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(script)
    con.commit()
    con.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: pathlib.Path) -> dict:
    """Two databases, MAC-SQL records for the four cells, CHESS records; covers equal and different text,
    equal and different results, a failed Refiner, missing outputs, errors, writes on a read-only
    connection, float rounding, blobs, undecodable text, duplicate rows and wrong stored scores."""
    bird_db, spider_db = root / "bird/dev_databases", root / "spider_db"
    ha = _sqlite(bird_db / "dbA/dbA.sqlite",
                 "CREATE TABLE t(id INTEGER, x REAL, s TEXT, b BLOB);"
                 "INSERT INTO t VALUES (1, 0.1 + 0.2, 'a', X'0102'), (2, 0.3, 'b', NULL),"
                 " (3, 0.3, CAST(X'FF61' AS TEXT), NULL);")
    hs = _sqlite(spider_db / "dbS/dbS.sqlite", "CREATE TABLE u(k INTEGER, v TEXT); INSERT INTO u VALUES (1, 'p'), (2, 'q');")
    ext, chess = root / "ext", root / "chess"
    (ext / "logs/host").mkdir(parents=True)
    (ext / "logs/host/db_before.sha256").write_text(f"{ha}  dbA/dbA.sqlite\n{hs}  dbS/dbS.sqlite\n")
    (chess / "logs").mkdir(parents=True)
    (chess / "logs/db_formal_after.sha256").write_text(f"{ha}  /run/data/dev_databases/dbA/dbA.sqlite\n")

    bird_cases = [  # (keep, refiner, flags, gold, stored Refiner, stored Keep)
        ("SELECT id FROM t WHERE id=1", "  SELECT id FROM t WHERE id=1\n", [], "SELECT id FROM t WHERE id=1", 1, 1),
        ("SELECT id FROM t WHERE id=1", "SELECT id FROM t WHERE id = 1", [], "SELECT id FROM t WHERE id=1", 1, 1),
        ("SELECT id FROM t", "SELECT id FROM t WHERE id=2", [], "SELECT id FROM t WHERE id=2", 1, 0),
        ("SELECT id FROM t", None, ["REFINER_FAILED"], "SELECT id FROM t", 0, 1),
        ("SELECT id FROM t", "SELEC id FROM", [], "SELECT id FROM t", 0, 1),
        ("SELECT x FROM t WHERE id=1", "SELECT 0.3", [], "SELECT 0.3", 1, 1),
        ("SELECT b FROM t WHERE id=1", "SELECT X'0A0B'", [], "SELECT b FROM t WHERE id=1", 1, 1),
        ("SELECT s FROM t WHERE id=3", "SELECT 'a'", [], "SELECT 'a'", 1, 1),
        (None, "SELECT id FROM t WHERE id=2", [], "SELECT id FROM t WHERE id=2", 0, 0),
        ("SELECT id FROM t WHERE id=2", "", [], "SELECT id FROM t WHERE id=2", 0, 1),
        ("SELECT id FROM t WHERE id=2", "DELETE FROM t", [], "SELECT id FROM t WHERE id=2", 0, 1),
        ("SELECT id FROM t WHERE id=2", "SELECT 1", ["GOLD_FAILED"], "SELECT nope FROM t", 0, 0),
    ]
    spider_cases = [
        ("SELECT v FROM u", "SELECT v FROM u ORDER BY k DESC", [], "SELECT v FROM u", 1, 1),
        ("SELECT v FROM u WHERE k=1", "SELECT v FROM u WHERE k=1", [], "SELECT v FROM u WHERE k=1", 1, 1),
        ("SELECT v FROM u WHERE k=1", "SELECT v FROM u WHERE k=2", [], "SELECT v FROM u WHERE k=2", 1, 1),
    ]
    root_bird = root / "bird"
    (root_bird / "dev.json").write_text(json.dumps([dict(question_id=100 + i, SQL=c[3]) for i, c in enumerate(bird_cases)]))
    (root / "spider").mkdir()
    (root / "spider/dev.json").write_text(json.dumps([dict(query=c[3]) for c in spider_cases]))
    (ext / "output").mkdir()
    for mt in ("q3", "q72"):
        for b, cases, db, off in (("bird", bird_cases, "dbA", 100), ("spider", spider_cases, "dbS", 0)):
            units = [dict(question_id=off + i, db_id=db, flags=fl, boundary_sql=k, refiner_sql=p,
                          refiner_llm_calls=(i + (mt == "q72")) % 3, trigger_reason="error" if i % 2 else "ok",
                          scores=dict(Refiner=yr, Keep=yk, Either_correct=max(yr, yk)))
                     for i, (k, p, fl, _, yr, yk) in enumerate(cases)]
            (ext / f"output/macsql_{mt}_{b}.json").write_text(json.dumps(dict(units=units)))

    chess_cases = [  # (final, SC, Valid, Nonempty, gold, stored UT, SC, Valid, Nonempty)
        ("SELECT id FROM t WHERE id=1\n", "SELECT id FROM t WHERE id=1", 'SELECT "a"', None, "SELECT id FROM t WHERE id=1", 1, 1, 0, 0),
        ("SELECT x FROM t WHERE id=1", "SELECT 0.3", "SELECT x FROM t WHERE id=2", "SELECT 0.3.", "SELECT x FROM t WHERE id=2", 0, 1, 1, 1),
        ("SELECT s FROM t WHERE id=3", "SELECT 'a'", "SELECT s FROM t WHERE id=1", "SELECT id FROM t", "SELECT 'a'", 0, 1, 1, 0),
        ("SELECT 1 UNION ALL SELECT 1", "SELECT 1", "`SELECT 2`", "SELECT nope", "SELECT 1", 1, 1, 0, 0),
        ("SELECT id FROM t", "SELECT id FROM t ORDER BY id DESC", "DELETE FROM t", "SELECT id FROM t", "SELEC bad", 0, 0, 0, 0),
    ]
    per_q, tasks, ex_rows = [], [], []
    for i, (f, sc, va, ne, g, *y) in enumerate(chess_cases):
        qid = 10 + i
        per_q.append(dict(question_id=qid, db_id="dbA", flags=["F"] if i == 2 else [], final_sql=f,
                          picks=dict(UT_full=f, SC_official=sc, Valid_vote=va, Nonempty_vote=ne),
                          scores=dict(CHESS_final_SQL=y[0], SC_official=y[1], Valid_vote=y[2], Nonempty_vote=y[3])))
        tasks.append(dict(question_id=qid, SQL=g))
        if i != 4:
            ex_rows.append(dict(question_id=qid, idx=0, status="ok", result_sha=f"h{i}", sql=" " + f))
            ex_rows.append(dict(question_id=qid, idx=1, status="ok" if i % 2 else "error",
                                result_sha=f"h{i}" if i == 0 else f"k{i}", sql=sc))
    (chess / "output/offline").mkdir(parents=True)
    (chess / "data").mkdir()
    (chess / "output/offline/per_question.json").write_text(json.dumps(per_q))
    (chess / "data/tasks_147.json").write_text(json.dumps(tasks))
    with open(chess / "output/offline/executions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["question_id", "idx", "status", "result_sha", "sql"])
        w.writeheader()
        w.writerows(ex_rows)
    return dict(ext=ext, chess=chess, bird=root_bird, bird_db=bird_db, spider=root / "spider", spider_db=spider_db)


def _reexec_args(paths, out):
    return ["--out", out, "--external-run", paths["ext"], "--chess-run", paths["chess"], "--bird", paths["bird"],
            "--bird-db", paths["bird_db"], "--spider", paths["spider"], "--spider-db", paths["spider_db"]]


def _compare_reexec(ref: pathlib.Path, ours: pathlib.Path):
    _same_bytes(ref, ours, UNIT_FILES)
    assert _strip_log(ref / "reexec_log.json") == _strip_log(ours / "reexec_log.json")


def test_reexec_identical_on_small_databases(tmp_path):
    paths = _fixture(tmp_path / "data")
    ref, ours = tmp_path / "ref", tmp_path / "suite"
    _script("reexec", _reexec_args(paths, ref))
    _step("reexec", _reexec_args(paths, ours))
    _compare_reexec(ref, ours)
    log = json.loads((ours / "reexec_log.json").read_text())
    assert log["db_sha256_match"] == dict(checked=3, matched=3, mismatched=[])
    assert log["macsql"]["q3_bird"]["executed_units"] == 11
    rows = C.read_table(ours / "macsql_units_q3_bird.csv")
    assert [r["A_result"] for r in rows] == ["0", "0", "1", "1", "1", "0", "0", "0", "1", "1", "1", "1"]


def test_reexec_stops_on_database_mismatch(tmp_path, capsys):
    paths = _fixture(tmp_path / "data")
    log = paths["ext"] / "logs/host/db_before.sha256"
    log.write_text(log.read_text().replace(log.read_text()[:4], "0000", 1))
    assert reexec.main([str(a) for a in _reexec_args(paths, tmp_path / "out")]) == 1
    assert "DB hash mismatch: ['dbA/dbA.sqlite']" in capsys.readouterr().err


# ---------------------------------------------------------------- reexec on the BIRD and Spider databases
def test_reexec_identical_on_benchmark_databases(tmp_path):
    bird_db, spider_db = os.environ.get("BIRD_DB_DIR"), os.environ.get("SPIDER_DB_DIR")
    if not (bird_db and spider_db):
        pytest.skip("BIRD_DB_DIR and SPIDER_DB_DIR are not set (the BIRD and Spider databases)")
    _need_runs()
    bird, spider = DATA / "data/records/dataeng_raw/sql/dev_20240627", DATA / "data/records/dataeng_raw/spider"
    for p in (bird / "dev.json", spider / "dev.json"):
        if not p.exists():
            pytest.skip(f"benchmark questions missing: {p}")
    paths = dict(ext=EXT, chess=CHESS, bird=bird, bird_db=bird_db, spider=spider, spider_db=spider_db)
    ref, ours = tmp_path / "ref", tmp_path / "suite"
    _script("reexec", _reexec_args(paths, ref))
    _step("reexec", _reexec_args(paths, ours))
    _compare_reexec(ref, ours)
    _same_units_as_shipped(ours)


# A query that reads the current time (SQLite 'now', e.g. an age computed from a birth date) returns a
# result that depends on when it runs, so its result-class hash differs from the shipped unit file; its
# change indicator, scores and difference do not. Only these two hash columns of such rows are not compared.
CURRENT_DATE = re.compile(r"""['"]now['"]""", re.I)
CLASS_COLUMNS = ("keep_class", "refiner_class")


def _same_units_as_shipped(ours: pathlib.Path):
    for name in UNIT_FILES:
        if not name.startswith("macsql_units_"):
            _same_bytes(SHIPPED, ours, [name])
            continue
        cell = name.removeprefix("macsql_units_").removesuffix(".csv")
        units = json.loads((EXT / f"output/macsql_{cell}.json").read_text())["units"]
        dated = {str(u["question_id"]) for u in units
                 if any(CURRENT_DATE.search(u.get(k) or "") for k in ("boundary_sql", "refiner_sql"))}
        a = list(csv.DictReader(io.StringIO((SHIPPED / name).read_text(), newline="")))
        b = list(csv.DictReader(io.StringIO((ours / name).read_text(), newline="")))
        assert [r["question_id"] for r in a] == [r["question_id"] for r in b], name
        for ra, rb in zip(a, b):
            skip = CLASS_COLUMNS if ra["question_id"] in dated else ()
            assert {k: v for k, v in ra.items() if k not in skip} == \
                   {k: v for k, v in rb.items() if k not in skip}, (name, ra["question_id"])
