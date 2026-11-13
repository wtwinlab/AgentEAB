"""The replay-validity steps of the suite write the same files as the scripts of `share_eval/replay_validity/`.

Each test runs the script of `share_eval/replay_validity/` and the suite step on the same inputs into
separate temporary directories and compares every output file byte for byte. The one field removed
before a comparison is listed where it applies, with its reason.

The steps on shipped inputs (`grouping_bound`, `sensitivity_census`, the no-record branch of
`online_validation`) always run. The steps on logged runs read the hosted data folder named by
`SHARE_EVAL_DATA_ROOT` (a directory holding `data/records/` and `data/caches/`, default the repository
root) and skip when it lacks the records they need. No suite step executes SQL: every query result
comes from the stored caches, and a step stops with exit status 2 when one is missing.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from share_eval_suite.steps import bccs_divergence, grouping_bound, online_validation, sensitivity_census

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "share_eval" / "replay_validity"
DATA = SCRIPTS / "data"
FACTORIAL = REPO / "paper_objects" / "tab_factorial" / "inputs"
ROOT = pathlib.Path(os.environ.get("SHARE_EVAL_DATA_ROOT") or REPO)
RECORDS = ROOT / "data" / "records"
CACHE = ROOT / "data" / "caches" / "sql_execution"
DEV_JSON = RECORDS / "dataeng_raw" / "sql" / "dev_20240627" / "dev.json"
SIG_DIV = ("r0_exec_signatures.jsonl", "replay_exec_signature_cache.jsonl", "exec_fresh.jsonl",
           "divergence_exec_signature_cache.jsonl")
SIG_ONLINE = ("r0_exec_signatures.jsonl", "replay_exec_signature_cache.jsonl", "divergence_exec_signature_cache.jsonl")
VERDICTS = ("replay_verdict_cache.jsonl", "divergence_verdict_cache.jsonl")

# Fields removed before comparing a JSON output, with the reason.
VOLATILE = {
    "bccs_divergence.json": [("exec_cost", "wall_s")],   # wall-clock seconds spent on query execution
}


def _script(name, *args):
    """Run `share_eval/replay_validity/<name>.py` with every path given explicitly."""
    env = dict(os.environ, ARTIFACT_ROOT=str(ROOT))
    r = subprocess.run([sys.executable, "-I", "-B", str(SCRIPTS / f"{name}.py"), *map(str, args)],
                       cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def _step(module, *args):
    """Run the suite step in-process through its command line."""
    assert module.main([str(a) for a in args]) == 0


def _strip(name, raw: bytes) -> bytes:
    if name not in VOLATILE:
        return raw
    d = json.loads(raw)
    for path in VOLATILE[name]:
        node = d
        for k in path[:-1]:
            node = node[k]
        node.pop(path[-1], None)
    return json.dumps(d, indent=1).encode()


def _same(a: pathlib.Path, b: pathlib.Path, names):
    for n in names:
        assert (a / n).is_file() and (b / n).is_file(), n
        assert _strip(n, (a / n).read_bytes()) == _strip(n, (b / n).read_bytes()), f"{n} differs"
    assert sorted(p.name for p in a.iterdir()) == sorted(p.name for p in b.iterdir())


def _caches(names):
    return ",".join(str(CACHE / n) for n in names)


def _need(*paths_):
    missing = [str(p) for p in paths_ if not pathlib.Path(p).exists()]
    if missing:
        pytest.skip("missing hosted data: " + ", ".join(missing))


# ------------------------------------------------------------------ shipped inputs
def test_grouping_bound(tmp_path):
    ref, got = tmp_path / "ref", tmp_path / "suite"
    _script("grouping_bound", "--sharing", DATA / "sharing_long.csv", "--out", ref)
    _step(grouping_bound, "--sharing", DATA / "sharing_long.csv", "--out", got)
    _same(ref, got, ["grouping_bound.json"])


def _replay_dir(d: pathlib.Path) -> pathlib.Path:
    """A replay directory from shipped inputs: the per-run scores, and a replay summary holding only
    the tie shares, which are the census's only reading of it (`vote_ties.csv` is the replay's tie table)."""
    d.mkdir()
    shutil.copy2(FACTORIAL / "per_run_scores.jsonl", d / "per_run_scores.jsonl")
    cells = {}
    with open(FACTORIAL / "vote_ties.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            cells.setdefault(r["cell"], {"ties": {}})["ties"][r["ext|ident|round|arm"]] = {
                "tie_share": float(r["tie_share"]), "no_live_label_share": float(r["no_live_label_share"])}
    (d / "factorial_replay.json").write_text(json.dumps({"cells": cells}, indent=1))
    return d


def test_sensitivity_census(tmp_path):
    replay = _replay_dir(tmp_path / "replay")
    ref, got = tmp_path / "ref", tmp_path / "suite"
    common = ("--replay", replay, "--match-qa", DATA / "match_qa_audit.json",
              "--return-rule", DATA / "return_rule_cells.csv")
    _script("sensitivity_census", *common, "--out", ref)
    _step(sensitivity_census, *common, "--out", got)
    _same(ref, got, ["sensitivity_census.json", "sensitivity_census.csv"])


def test_online_validation_without_online_runs(tmp_path):
    empty = tmp_path / "empty"
    (empty / "raw").mkdir(parents=True)
    ref, got = tmp_path / "ref", tmp_path / "suite"
    ref.mkdir()
    got.mkdir()
    common = ("--ext", empty, "--div", FACTORIAL / "bccs_divergence_runs.jsonl")
    _script("online_validation", *common, "--out", ref / "online_validation.json")
    _step(online_validation, *common, "--out", got / "online_validation.json")
    _same(ref, got, ["online_validation.json"])


def test_steps_load_no_other_implementation(tmp_path):
    """Running the steps on shipped inputs loads nothing from share_eval/, paper_objects/, experiments/ or
    code/consensus_delivery/."""
    replay = _replay_dir(tmp_path / "replay")
    code = (
        "import sys\n"
        "sys.dont_write_bytecode = True\n"
        "from share_eval_suite.steps import grouping_bound as G, sensitivity_census as S\n"
        f"G.run(out={str(tmp_path / 'g')!r}, log=lambda *a: None)\n"
        f"S.run(replay={str(replay)!r}, out={str(tmp_path / 's')!r}, log=lambda *a: None)\n"
        "bad = [getattr(m, '__file__', '') or '' for m in list(sys.modules.values())]\n"
        "print('\\n'.join(f for f in bad if any(s in f for s in "
        "('share_eval/', 'paper_objects', 'experiments/', 'consensus_delivery'))))\n")
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "", r.stdout


# ------------------------------------------------------------------ logged runs (hosted data)
def _div_args(out):
    return ("--ext", RECORDS / "bird_debates", "--scores", FACTORIAL / "per_run_scores.jsonl", "--out", out,
            "--sig-caches", _caches(SIG_DIV), "--verdict-caches", _caches(VERDICTS))


def test_bccs_divergence(tmp_path, monkeypatch):
    _need(RECORDS / "bird_debates" / "raw", RECORDS / "bird_debates" / "analysis" / "sql_per_unit.jsonl",
          CACHE / "replay_verdict_cache.jsonl", DEV_JSON)
    monkeypatch.setenv("ARTIFACT_ROOT", str(ROOT))
    ref, got = tmp_path / "ref", tmp_path / "suite"
    _script("bccs_divergence", *_div_args(ref))
    _step(bccs_divergence, *_div_args(got))
    _same(ref, got, ["bccs_divergence.json", "bccs_divergence_runs.jsonl"])
    # the run table is the one the paper objects ship
    shipped = FACTORIAL / "bccs_divergence_runs.jsonl"
    assert (got / "bccs_divergence_runs.jsonl").read_bytes() == shipped.read_bytes(), \
        "bccs_divergence_runs.jsonl differs from paper_objects/tab_factorial/inputs/"
    # BCCS's own functions reproduce every logged read
    summary = json.loads((got / "bccs_divergence.json").read_text())
    assert not any(k.endswith("_bad") for k in summary["sanity"]), summary["sanity"]
    assert summary["factorial_score_mismatches"] == {}


def _online_args(ext, out, div, *extra):
    return ("--ext", ext, "--logged-ext", RECORDS / "bird_debates", "--div", div,
            "--sig-caches", _caches(SIG_ONLINE), "--verdict-caches", _caches(VERDICTS), "--out", out, *extra)


@pytest.mark.parametrize("selftest", [False, True], ids=["online", "selftest"])
def test_online_validation(tmp_path, monkeypatch, selftest):
    ext = RECORDS / ("bird_debates" if selftest else "bird_debates_backbones")
    _need(*dict.fromkeys((ext / "raw", RECORDS / "bird_debates" / "raw", CACHE / "replay_verdict_cache.jsonl",
                          DEV_JSON)))
    monkeypatch.setenv("ARTIFACT_ROOT", str(ROOT))
    div = FACTORIAL / "bccs_divergence_runs.jsonl"
    extra = ("--selftest",) if selftest else ()
    ref, got = tmp_path / "ref", tmp_path / "suite"
    ref.mkdir()
    got.mkdir()
    _script("online_validation", *_online_args(ext, ref / "online_validation.json", div, *extra))
    _step(online_validation, *_online_args(ext, got / "online_validation.json", div, *extra))
    _same(ref, got, ["online_validation.json"])
    res = json.loads((got / "online_validation.json").read_text())
    assert res["status"] == "computed"
    if selftest:
        # the logged runs read as online runs: every paired difference is zero (conditions without a
        # divergence table, answer-only and uniform selection, have no pairs)
        paired = {a: v for a, v in res["by_arm"].items() if v["n_tasks_paired"]}
        assert set(paired) == {"FULL", "NOPEER"}
        for arm in paired.values():
            assert arm["online_minus_replay_all"]["est_pp"] == 0
            assert arm["online_minus_replay_all"]["ci95_pp"] == [0.0, 0.0]


def test_bccs_divergence_reports_queries_without_stored_result(tmp_path, monkeypatch, capsys):
    """Without the stored execution results the step names the queries it would have to execute and
    stops; it never executes them."""
    _need(RECORDS / "bird_debates" / "raw", DEV_JSON)
    monkeypatch.setenv("ARTIFACT_ROOT", str(ROOT))
    code = bccs_divergence.main(["--ext", str(RECORDS / "bird_debates"), "--scores", "", "--out", str(tmp_path),
                                 "--sig-caches", str(tmp_path / "none.jsonl"), "--verdict-caches", ""])
    assert code == 2
    assert "no stored execution result" in capsys.readouterr().err
    assert not (tmp_path / "bccs_divergence_runs.jsonl").exists()


def test_sensitivity_census_on_replay_output(tmp_path):
    """The census on the replay's own output: `factorial_replay.py` on the hosted BIRD debates writes
    the per-run scores and the replay summary, then the script and the step read that directory.
    The replay executes again the few queries whose stored results disagree with their verdict class,
    only for its own recheck block, which the census does not read; without the databases those
    executions fail and are recorded as errors."""
    debates = RECORDS / "bird_debates"
    _need(debates / "raw", debates / "analysis" / "sql_per_unit.jsonl", CACHE / "r0_exec_signatures.jsonl",
          CACHE / "replay_exec_signature_cache.jsonl", CACHE / "replay_verdict_cache.jsonl", DEV_JSON)
    replay = tmp_path / "replay"
    env = dict(os.environ, ARTIFACT_ROOT=str(ROOT))
    r = subprocess.run([sys.executable, "-I", "-B", str(REPO / "share_eval" / "replay" / "factorial_replay.py"),
                        "--ext", str(debates),
                        "--seed-sig-cache", _caches(("r0_exec_signatures.jsonl", "replay_exec_signature_cache.jsonl")),
                        "--verdict-caches", _caches(("replay_verdict_cache.jsonl",)),
                        "--audit-verdicts", str(CACHE / "verdict_audit.json"), "--out", str(replay)],
                       cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (replay / "per_run_scores.jsonl").read_bytes() == (FACTORIAL / "per_run_scores.jsonl").read_bytes()
    ref, got = tmp_path / "ref", tmp_path / "suite"
    common = ("--replay", replay, "--match-qa", DATA / "match_qa_audit.json",
              "--return-rule", DATA / "return_rule_cells.csv")
    _script("sensitivity_census", *common, "--out", ref)
    _step(sensitivity_census, *common, "--out", got)
    _same(ref, got, ["sensitivity_census.json", "sensitivity_census.csv"])
