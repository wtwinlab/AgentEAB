#!/usr/bin/env python3
"""Rerun the SHARE-Eval steps whose inputs ship with this repository and compare with the expected outputs.

    python3 -I share_eval/check_consistency.py

1. share_eval/replay/db_cluster_bootstrap.py on paper_objects/tab_factorial/inputs/per_run_scores.jsonl
   (the per-run scores written by share_eval/replay/factorial_replay.py for the 198 BIRD questions);
2. share_eval/replay_validity/bounds.py on the same scores and
   paper_objects/tab_factorial/inputs/bccs_divergence_runs.jsonl (written by bccs_divergence.py);
3. share_eval/replay_validity/grouping_bound.py on share_eval/replay_validity/data/sharing_long.csv.

Each output must have the SHA-256 recorded below, which is that of the output the paper's numbers come
from. The steps that read the logged runs (factorial_replay.py, bccs_divergence.py,
online_validation.py, the stage-replacement scripts) need the run directories described in
share_eval/README.md and are not part of this check. No model call, no SQL execution.
"""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
INP = REPO / "paper_objects/tab_factorial/inputs"
EXPECTED = {
    "db_cluster_bootstrap.json": "8afeb1bce714f6ff7074d9d6704af42b47c00c6133879c09f476db9a473ac651",
    "bounds.json": "15ac967e9fc98b840d0f6b76ea385863d2c9c174d750ab0c9baab037f26425dc",
    "grouping_bound.json": "ba7c7d5901167881914b37e29301b9478b2eaff09eb28101fdbea901e8c22709",
}


def run(*args):
    r = subprocess.run([sys.executable, "-I", *map(str, args)], cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{args[0]} failed:\n{r.stderr[-800:]}")


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        t = pathlib.Path(td)
        (t / "replay").mkdir()
        shutil.copy2(INP / "per_run_scores.jsonl", t / "replay")
        run("share_eval/replay/db_cluster_bootstrap.py", "--out", t / "replay", "--questions", INP / "bird_units_db.csv")
        run("share_eval/replay_validity/bounds.py", "--div", INP / "bccs_divergence_runs.jsonl",
            "--scores", INP / "per_run_scores.jsonl", "--questions", INP / "bird_units_db.csv", "--out", t / "bounds.json")
        run("share_eval/replay_validity/grouping_bound.py", "--out", t)
        got = {"db_cluster_bootstrap.json": sha(t / "replay/db_cluster_bootstrap.json"),
               "bounds.json": sha(t / "bounds.json"), "grouping_bound.json": sha(t / "grouping_bound.json")}
    ok = True
    for k, v in EXPECTED.items():
        same = got[k] == v
        ok &= same
        print(f"{'PASS' if same else 'FAIL'}  {k}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
