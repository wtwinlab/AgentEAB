"""The replay steps of the suite, and the bounds steps on the shipped inputs, write the same files as
the scripts of the same name in `share_eval/`.

Each test runs the `share_eval/` script (`python3 -I -B`, every path explicit) and the suite step
(`python3 -B -m share_eval_suite.steps.<name>`) on the same inputs into separate temporary
directories and compares every output file byte for byte, except the fields listed in `VOLATILE`.

The shipped inputs (`paper_objects/tab_factorial/inputs/`) are always tested. The logged debates and
execution caches are read from the directory named by `SHARE_EVAL_DATA_ROOT` (default: the repository
root), which holds `data/records/...` and `data/caches/sql_execution/`; a test whose data is absent
skips. A query without a stored result is executed on the BIRD databases under that directory; where a
database is absent, both implementations record the query as an execution error, and the per-run scores
then reproduce the shipped ones only when every query has a stored result.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Optional

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.environ.get("SHARE_EVAL_DATA_ROOT") or REPO)
CACHE = DATA_ROOT / "data" / "caches" / "sql_execution"
DEV = DATA_ROOT / "data" / "records" / "dataeng_raw" / "sql" / "dev_20240627" / "dev.json"
SHIPPED = REPO / "paper_objects" / "tab_factorial" / "inputs"

# Fields removed before comparing, per file name, each with the reason it differs between two runs.
VOLATILE = {
    "factorial_replay.json": {
        ("generated_utc",): "generation timestamp",
        ("wall_s",): "wall-clock seconds of the run",
        ("execution_cost", "wall_s"): "wall-clock seconds of the query executions",
    },
    "exec_signature_cache.jsonl": {("s",): "wall-clock seconds of one query execution"},
    "verdict_cache.jsonl": {("seconds",): "wall-clock seconds of one query execution"},
}
# SHA256SUMS: the digest of factorial_replay.json differs because that file holds the timestamp; the
# line is checked against the digest of the file in its own directory instead.
DIGEST_OF_TIMESTAMPED = ("factorial_replay.json",)

# Digests of the outputs the paper's numbers come from.
PAPER_DIGESTS = {
    "db_cluster_bootstrap.json": "8afeb1bce714f6ff7074d9d6704af42b47c00c6133879c09f476db9a473ac651",
    "bounds.json": "15ac967e9fc98b840d0f6b76ea385863d2c9c174d750ab0c9baab037f26425dc",
}

FACTORIAL_SCRIPT = "share_eval/replay/factorial_replay.py"


# ------------------------------------------------------------------ running
def _env():
    env = dict(os.environ, ARTIFACT_ROOT=str(DATA_ROOT), PYTHONDONTWRITEBYTECODE="1")
    return env


def _check(r, what):
    assert r.returncode == 0, f"{what} failed:\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}"


def run_script(rel: str, *args):
    r = subprocess.run([sys.executable, "-I", "-B", str(REPO / rel), *map(str, args)], cwd=REPO,
                       env=_env(), capture_output=True, text=True)
    _check(r, rel)
    return r.stdout


def run_step(name: str, *args):
    r = subprocess.run([sys.executable, "-B", "-m", f"share_eval_suite.steps.{name}", *map(str, args)],
                       cwd=REPO, env=_env(), capture_output=True, text=True)
    _check(r, f"share_eval_suite.steps.{name}")
    return r.stdout


# ------------------------------------------------------------------ comparison
def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _drop(d, path):
    for k in path[:-1]:
        d = d[k]
    d.pop(path[-1])


def _normalised(p: pathlib.Path) -> bytes:
    rules = VOLATILE.get(p.name)
    if not rules:
        return p.read_bytes()
    if p.suffix == ".jsonl":
        lines = []
        for line in p.read_text().splitlines():
            d = json.loads(line)
            for path in rules:
                _drop(d, path)
            lines.append(json.dumps(d))
        return "\n".join(lines).encode()
    d = json.loads(p.read_text())
    for path in rules:
        _drop(d, path)
    return json.dumps(d, indent=1, default=str).encode()


def _sums(p: pathlib.Path):
    rows = [line.split("  ", 1) for line in p.read_text().splitlines()]
    return [(n, h) for h, n in rows]


def compare_dirs(a: pathlib.Path, b: pathlib.Path):
    """Every file of `a` exists in `b` with the same content (after removing the VOLATILE fields)."""
    fa = sorted(x.name for x in a.iterdir() if x.is_file())
    fb = sorted(x.name for x in b.iterdir() if x.is_file())
    assert fa == fb, (fa, fb)
    for name in fa:
        pa, pb = a / name, b / name
        if name == "SHA256SUMS":
            sa, sb = _sums(pa), _sums(pb)
            assert [n for n, _ in sa] == [n for n, _ in sb], name
            for (n, ha), (_, hb) in zip(sa, sb):
                if n in DIGEST_OF_TIMESTAMPED:
                    assert ha == _sha(a / n) and hb == _sha(b / n), n
                else:
                    assert ha == hb, n
            continue
        if name in VOLATILE and pa.read_bytes() == pb.read_bytes():
            continue
        assert _normalised(pa) == _normalised(pb), f"{name} differs"


def _need(*paths):
    missing = [str(p) for p in paths if not pathlib.Path(p).exists()]
    if missing:
        pytest.skip("hosted data missing: " + ", ".join(missing))


# ------------------------------------------------------------------ shipped inputs (always)
def test_shipped_inputs_db_cluster_bootstrap_and_bounds(tmp_path):
    a, b = tmp_path / "share_eval", tmp_path / "suite"
    for d in (a, b):
        d.mkdir()
        shutil.copy2(SHIPPED / "per_run_scores.jsonl", d)
    q = SHIPPED / "bird_units_db.csv"
    div = SHIPPED / "bccs_divergence_runs.jsonl"
    scores = SHIPPED / "per_run_scores.jsonl"
    run_script("share_eval/replay/db_cluster_bootstrap.py", "--out", a, "--questions", q)
    run_step("db_cluster_bootstrap", "--out", b, "--questions", q)
    run_script("share_eval/replay_validity/bounds.py", "--div", div, "--scores", scores, "--questions", q,
               "--out", a / "bounds.json")
    run_step("bounds", "--div", div, "--scores", scores, "--questions", q, "--out", b / "bounds.json")
    run_script("share_eval/replay_validity/bounds_bccs_only.py", "--div", div, "--out", a / "bounds_bccs.json")
    run_step("bounds_bccs_only", "--div", div, "--out", b / "bounds_bccs.json")
    compare_dirs(a, b)
    for name, digest in PAPER_DIGESTS.items():
        assert _sha(b / name) == digest, name


# ------------------------------------------------------------------ logged debates (hosted data)
def _factorial_args(ext: pathlib.Path, out: pathlib.Path, recorded: Optional[pathlib.Path] = None):
    """Options of factorial_replay. With `recorded`, the execution results that run wrote are added to the
    stored ones, so no query is executed again."""
    sig = [CACHE / "r0_exec_signatures.jsonl", CACHE / "replay_exec_signature_cache.jsonl"]
    ver = [CACHE / "replay_verdict_cache.jsonl"]
    if recorded is not None:
        sig.append(recorded / "exec_signature_cache.jsonl")
        ver.append(recorded / "verdict_cache.jsonl")
    return ["--ext", ext, "--out", out,
            "--seed-sig-cache", ",".join(str(x) for x in sig),
            "--verdict-caches", ",".join(str(x) for x in ver),
            "--audit-verdicts", CACHE / "verdict_audit.json"]


# A query that reads the current time (SQLite 'now', e.g. julianday('now') - julianday(birth_date)) returns a
# result that depends on when it runs, and a query near the 5-second limit may finish in one execution and
# time out in another. Their execution results are the only ones not compared between the two executions.
CURRENT_TIME = re.compile(r"""['"]now['"]""", re.I)


def _queries_by_key(ext: pathlib.Path):
    from share_eval_suite.records import readers
    runs, per_unit, _ = readers.load_logged_debates(ext)
    out = {}
    for r in readers.candidate_sets(runs, per_unit):
        for rd in ("r0", "last"):
            for e in ("strict", "lenient"):
                for q in r["sets"][rd][e]:
                    if q and r["db"]:
                        out[readers.query_key(r["db"], q)] = q
    return out


def _jsonl(p: pathlib.Path):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def compare_executions(a: pathlib.Path, b: pathlib.Path, ext: pathlib.Path):
    """The queries both implementations executed, in the same order, with the same results, except the
    results of queries that read the current time or that timed out in either execution."""
    queries = None
    for name, seconds in (("exec_signature_cache.jsonl", "s"), ("verdict_cache.jsonl", "seconds")):
        ra, rb = _jsonl(a / name), _jsonl(b / name)
        assert [x["k"] for x in ra] == [x["k"] for x in rb], name
        for x, y in zip(ra, rb):
            x, y = ({k: v for k, v in d.items() if k != seconds} for d in (x, y))
            if x == y:
                continue
            if "timeout" in json.dumps(x) + json.dumps(y):
                continue
            queries = queries if queries is not None else _queries_by_key(ext)
            assert CURRENT_TIME.search(queries.get(x["k"], "")), (name, x, y)


def _replay_chain(run_dir: str, tmp: pathlib.Path):
    """factorial_replay, db_cluster_bootstrap and report_tables of both implementations; returns the two
    output directories after comparing them.

    Both implementations first execute, each on its own, the queries without a stored result
    (`compare_executions`). The replay is then compared on identical execution results: both run again
    with the results the share_eval/ execution wrote, so that no query without a stored result is
    executed, and every output file is compared. In every run the recheck of the suspect queries executes
    them again."""
    ext = DATA_ROOT / "data" / "records" / run_dir
    _need(ext / "raw", ext / "analysis" / "sql_per_unit.jsonl", CACHE, DEV)
    a, b = tmp / "share_eval", tmp / "suite"
    run_script(FACTORIAL_SCRIPT, *_factorial_args(ext, a))
    run_step("factorial_replay", *_factorial_args(ext, b))
    if not _jsonl(a / "exec_signature_cache.jsonl") and not _jsonl(a / "verdict_cache.jsonl"):
        compare_dirs(a, b)                       # all results stored: only the recheck executed queries
    else:
        compare_executions(a, b, ext)
        recorded = tmp / "executed"
        a.rename(recorded)
        shutil.rmtree(b)
        run_script(FACTORIAL_SCRIPT, *_factorial_args(ext, a, recorded))
        run_step("factorial_replay", *_factorial_args(ext, b, recorded))
        compare_dirs(a, b)
    run_script("share_eval/replay/db_cluster_bootstrap.py", "--out", a, "--questions", DEV)
    run_step("db_cluster_bootstrap", "--out", b, "--questions", DEV)
    compare_dirs(a, b)
    run_script("share_eval/replay/report_tables.py", "--out", a)
    run_step("report_tables", "--out", b)
    compare_dirs(a, b)
    return a, b


def test_bird_debates_replay_chain(tmp_path):
    a, b = _replay_chain("bird_debates", tmp_path)
    # the per-run scores are those the paper objects ship
    assert (b / "per_run_scores.jsonl").read_bytes() == (SHIPPED / "per_run_scores.jsonl").read_bytes()


def test_bird_debates_backbones_replay_chain(tmp_path):
    _replay_chain("bird_debates_backbones", tmp_path)


def test_bounds_bccs_only_qwen25_14b(tmp_path):
    """Divergence records of the Qwen2.5-14B BCCS debates, produced by the share_eval script chain
    (`bccs_divergence.py --scores ""`), are the common input of both implementations."""
    ext = DATA_ROOT / "data" / "records" / "bird_debates_backbones"
    _need(ext / "raw", CACHE, DEV)
    div_dir = tmp_path / "divergence"
    sig = ",".join(str(CACHE / n) for n in ("r0_exec_signatures.jsonl", "replay_exec_signature_cache.jsonl",
                                            "exec_fresh.jsonl", "divergence_exec_signature_cache.jsonl"))
    ver = ",".join(str(CACHE / n) for n in ("replay_verdict_cache.jsonl", "divergence_verdict_cache.jsonl"))
    run_script("share_eval/replay_validity/bccs_divergence.py", "--ext", ext, "--scores", "",
               "--backbones", "qwen25-14b-awq", "--out", div_dir, "--sig-caches", sig, "--verdict-caches", ver)
    div = div_dir / "bccs_divergence_runs.jsonl"
    a, b = tmp_path / "share_eval", tmp_path / "suite"
    a.mkdir()
    b.mkdir()
    run_script("share_eval/replay_validity/bounds_bccs_only.py", "--div", div, "--out", a / "bounds_bccs.json")
    run_step("bounds_bccs_only", "--div", div, "--out", b / "bounds_bccs.json")
    compare_dirs(a, b)
    assert list(json.loads((b / "bounds_bccs.json").read_text())) == ["qwen25-14b-awq"]
