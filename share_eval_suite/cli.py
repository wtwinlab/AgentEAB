"""Command line of the suite.

    python3 -m share_eval_suite list-protocols [--kind consensus|pipeline] [--paper]
    python3 -m share_eval_suite list-tasks
    python3 -m share_eval_suite replay [--suite] [STEP] [ARGS...]
    python3 -m share_eval_suite stage-replacement [--suite] [STEP] [ARGS...]
    python3 -m share_eval_suite replay-validity [--suite] [STEP] [ARGS...]
    python3 -m share_eval_suite check

`replay`, `stage-replacement` and `replay-validity` run the `share_eval/` scripts, which wrote the inputs
of the paper objects, with the given arguments (default outputs under `output/share_eval/`). `--suite`
runs the suite's step of the same name (`share_eval_suite/steps/`) instead; it writes the same files, by
default under `output/share_eval_suite/`. Run without arguments, a step whose default inputs (the logged runs,
see `share_eval/README.md`) are absent prints its `--help` and the missing paths instead of running and
exits with status 2.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from share_eval_suite import paths

REPO = paths.REPO

_BIRD_RUNS = "data/records/bird_debates"
_BIRD_DEV = "data/records/dataeng_raw/sql/dev_20240627/dev.json"
_SCORES = "output/{out}/replay/per_run_scores.jsonl"
_DIVERGENCE = "output/{out}/replay_validity/bccs_divergence_runs.jsonl"
_STAGE_RUNS = ("data/records/macsql_comem_stages", "data/records/chess_unit_tester")

#: command -> step -> (share_eval/ script, default inputs the step reads); the first step is the default.
#: `{out}` is `share_eval` for the `share_eval/` scripts and `share_eval_suite` for the suite's steps.
STEPS: Dict[str, Dict[str, Tuple[str, Tuple[str, ...]]]] = {
    "replay": {
        "factorial_replay": ("share_eval/replay/factorial_replay.py", (_BIRD_RUNS,)),
        "db_cluster_bootstrap": ("share_eval/replay/db_cluster_bootstrap.py", (_SCORES, _BIRD_DEV)),
        "report_tables": ("share_eval/replay/report_tables.py",
                          ("output/{out}/replay/factorial_replay.json",)),
    },
    "stage-replacement": {
        "reexec": ("share_eval/stage_replacement/reexec.py", _STAGE_RUNS),
        "throughput": ("share_eval/stage_replacement/throughput.py", _STAGE_RUNS),
        "decompose": ("share_eval/stage_replacement/decompose.py", _STAGE_RUNS),
        "costs_comparable": ("share_eval/stage_replacement/costs_comparable.py", _STAGE_RUNS),
    },
    "replay-validity": {
        "bccs_divergence": ("share_eval/replay_validity/bccs_divergence.py", (_BIRD_RUNS, _SCORES)),
        "bounds": ("share_eval/replay_validity/bounds.py", (_DIVERGENCE, _SCORES, _BIRD_DEV)),
        "bounds_bccs_only": ("share_eval/replay_validity/bounds_bccs_only.py", ()),
        "grouping_bound": ("share_eval/replay_validity/grouping_bound.py",
                           ("share_eval/replay_validity/data/sharing_long.csv",)),
        "online_validation": ("share_eval/replay_validity/online_validation.py",
                              ("data/records/bird_debates_backbones", _DIVERGENCE)),
        "sensitivity_census": ("share_eval/replay_validity/sensitivity_census.py",
                               ("output/{out}/replay",)),
    },
}

#: steps with required arguments: run without arguments they print their --help
_NEEDS_ARGS = {"bounds_bccs_only"}


def _command(step: str, script: str, args: Sequence[str], suite: bool) -> List[str]:
    if suite:
        return [sys.executable, "-B", "-m", f"share_eval_suite.steps.{step}", *args]
    return [sys.executable, "-B", "-I", str(REPO / script), *args]


def run_step(cmd: str, step: Optional[str], args: Sequence[str], suite: bool = False) -> int:
    """Run the `share_eval/` script of one step (or, with `suite`, the suite's step); print its `--help`
    when its default inputs are missing."""
    steps = STEPS[cmd]
    step = step or next(iter(steps))
    if step not in steps:
        print(f"{cmd}: unknown step {step!r}; steps: {', '.join(steps)}", file=sys.stderr)
        return 2
    script, needs = steps[step]
    needs = [p.format(out="share_eval_suite" if suite else "share_eval") for p in needs]
    if not args:
        missing = [p for p in needs if not (REPO / p).exists()]
        if missing or step in _NEEDS_ARGS:
            if missing:
                print(f"{cmd} {step}: default inputs not found: {', '.join(missing)}\n"
                      f"(see share_eval/README.md, 'Inputs'); pass the paths as arguments.\n", flush=True)
            subprocess.call(_command(step, script, ["--help"], suite), cwd=REPO)
            return 2 if missing else 0
    return subprocess.call(_command(step, script, args, suite), cwd=REPO)


def _list_protocols(kind: Optional[str], paper: bool) -> int:
    from share_eval_suite.adapters import protocols as P
    k = {"consensus": P.CONSENSUS, "pipeline": P.PIPELINE}.get(kind) if kind else None
    rows = P.list_protocols(kind=k, in_paper=True if paper else None)
    w = max(len(p.name) for p in rows)
    for p in rows:
        print(f"{p.name:<{w}}  {p.kind:<18}  {'paper' if p.in_paper else '     '}  "
              f"{' -> '.join(p.stages)}")
        print(f"{'':<{w}}  final decision: {p.final_decision}")
        print(f"{'':<{w}}  implementation: {', '.join(p.files)}")
    return 0


def _list_tasks() -> int:
    from share_eval_suite.adapters import tasks as T
    for t in T.list_tasks():
        print(f"{t.name}: {', '.join(t.benchmarks)}")
        print(f"  input: {getattr(t.build_input, '__name__', '-')}, answer: {t.read_answer.__name__}, "
              f"score: {t.score.__name__}")
        print(f"  sources: {', '.join(t.sources)}")
    return 0


def check() -> int:
    """Run the suite's tests."""
    return subprocess.call([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "share_eval_suite/tests"], cwd=REPO)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in STEPS:
        return _run_command(argv[0], argv[1:])
    ap = argparse.ArgumentParser(prog="python3 -m share_eval_suite",
                                 description="SHARE-Eval benchmarking suite")
    sub = ap.add_subparsers(dest="command", required=True)
    lp = sub.add_parser("list-protocols", help="debate and consensus methods and multi-stage pipelines")
    lp.add_argument("--kind", choices=("consensus", "pipeline"))
    lp.add_argument("--paper", action="store_true", help="only the workflows the paper reports")
    sub.add_parser("list-tasks", help="question answering, text-to-SQL and entity matching")
    for cmd in STEPS:
        sp = sub.add_parser(cmd, add_help=False,
                            help=f"steps of share_eval/{cmd.replace('-', '_')}/ ({', '.join(STEPS[cmd])})")
        sp.add_argument("rest", nargs=argparse.REMAINDER)
    sub.add_parser("check", help="run the suite's tests")
    a = ap.parse_args(argv)
    if a.command == "list-protocols":
        return _list_protocols(a.kind, a.paper)
    if a.command == "list-tasks":
        return _list_tasks()
    if a.command == "check":
        return check()
    return _run_command(a.command, a.rest)


def _run_command(cmd: str, rest: Sequence[str]) -> int:
    rest = list(rest)
    suite = "--suite" in rest[:2]
    if suite:
        rest.remove("--suite")
    step = rest.pop(0) if rest and not rest[0].startswith("-") else None
    if step is None and rest[:1] in (["-h"], ["--help"]):
        print(f"usage: python3 -m share_eval_suite {cmd} [--suite] [STEP] [ARGS...]\n"
              f"steps: {', '.join(STEPS[cmd])} (default {next(iter(STEPS[cmd]))}); each runs the script "
              f"of the same name in share_eval/{cmd.replace('-', '_')}/\n"
              f"--suite runs the suite's step of the same name (share_eval_suite/steps/), same outputs")
        return 0
    return run_step(cmd, step, rest, suite)
