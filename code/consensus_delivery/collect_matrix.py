from __future__ import annotations

import argparse
import concurrent.futures
import errno
import json
import os
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

from .adapters.mallm_upstream import INTENDED_PROTOCOL_VARIANT, VARIANTS
from .adapters.native_run import run as mallm_run
from .registry_live import PAIRED_RUN_BUDGET, REGISTRY

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASES = {
    "pilot": {"dir": os.path.join(REPO, "paper", "evidence", "repair_validation", "cells",
                                  "primary_matrix"),
              "block": "block01", "num_samples": 40},
    "main": {"dir": os.path.join(REPO, "results", "main_matrix"),
             "block": "block02", "num_samples": 200},
}

BENCHMARKS = ["gsm8k", "strategyqa", "mmlu_pro", "gpqa", "squad_v2"]
BACKBONES = ["llama3-8b", "qwen25-7b"]
MECHANISMS = ["unanimity_consensus", "simple_voting", "du_mad"]
PROMPTS = {
    "gsm8k": "Solve the grade-school math problem. State the final numeric answer at the end.",
    "mmlu_pro": "Answer the multiple-choice question. State the single letter of your choice at the end.",
    "gpqa": "Answer the multiple-choice question. State the single letter of your choice at the end.",
    "strategyqa": "Answer the yes/no question. State exactly 'yes' or 'no' at the end.",
    "squad_v2": ("Answer the question using only the given passage. If the passage does not contain "
                 "the answer, reply exactly 'unanswerable'. State the answer span at the end."),
}


def tag_for(benchmark: str, mechanism: str, alias: str, phase: str = "pilot") -> str:
    mech = "dumad" if mechanism == "du_mad" else mechanism
    return f"{benchmark}{PHASES[phase]['num_samples']}_{mech}_{alias}"


def cell_paths(benchmark: str, mechanism: str, alias: str,
               phase: str = "pilot") -> Tuple[str, str, str]:
    tag = tag_for(benchmark, mechanism, alias, phase)
    d = PHASES[phase]["dir"]
    return (os.path.join(d, tag + ".json"),
            os.path.join(d, tag + ".manifest.json"),
            os.path.join(d, tag + ".lock"))


def block_for(benchmark: str, phase: str) -> str:
    return os.path.join(REPO, "data", "frozen", f"{benchmark}_{PHASES[phase]['block']}.json")


def endpoint_alias(backbone: str, instance: str) -> str:
    return backbone if instance == "a" else f"{backbone}-{instance}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim(lock_path: str) -> bool:
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            try:
                held = json.load(open(lock_path, encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                held = {}
            pid = held.get("pid")
            if isinstance(pid, int) and _pid_alive(pid) and pid != os.getpid():
                return False
            try:
                os.remove(lock_path)
            except OSError:
                return False
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "argv": sys.argv}, f)
        return True
    return False


def release(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except OSError:
        pass


def completion(manifest_path: str, records_path: str, assigned: int) -> Tuple[bool, str]:
    try:
        man = json.load(open(manifest_path, encoding="utf-8"))
        n = len(json.load(open(records_path, encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return False, "manifest or records unreadable"
    failed = man.get("n_failed") or 0
    ec = man.get("exit_code")
    if n + failed < assigned:
        return False, f"{n} records + {failed} failed < {assigned} assigned, exit {ec}"
    if n + failed > assigned:
        return False, f"{n} records + {failed} failed > {assigned} assigned, exit {ec}"
    if ec not in (0, None) and n == assigned:
        return True, ""
    return True, ""


def status(benchmark: str, mechanism: str, alias: str, phase: str = "pilot") -> str:
    out, man, lock = cell_paths(benchmark, mechanism, alias, phase)
    if os.path.exists(man):
        ok, why = completion(man, out, PHASES[phase]["num_samples"])
        return "collected" if ok else f"incomplete({why})"
    if os.path.exists(lock):
        try:
            pid = json.load(open(lock, encoding="utf-8")).get("pid")
        except (OSError, json.JSONDecodeError):
            pid = None
        if isinstance(pid, int) and _pid_alive(pid):
            return f"in-flight(pid {pid})"
        return "stale-lock"
    if os.path.exists(out):
        return "records-without-manifest"
    return "missing"


def collect_mallm(benchmark: str, protocol: str, alias: str, phase: str = "pilot",
                  instance: str = "a") -> int:
    ep = REGISTRY.get(endpoint_alias(alias, instance))
    out, man, _ = cell_paths(benchmark, protocol, alias, phase)
    return mallm_run(
        variant_root=VARIANTS[INTENDED_PROTOCOL_VARIANT],
        block_path=block_for(benchmark, phase),
        out_path=out, manifest_path=man,
        endpoint_url=ep.base_url, model_name=ep.served_name,
        instruction=PROMPTS[benchmark], protocol=protocol,
        num_agents=PAIRED_RUN_BUDGET["agents"], max_turns=PAIRED_RUN_BUDGET["max_turns"],
        max_tokens=min(PAIRED_RUN_BUDGET["max_tokens"], ep.max_completion),
        concurrency=PAIRED_RUN_BUDGET["concurrency"],
        num_samples=PHASES[phase]["num_samples"])


def collect_du_mad(benchmark: str, alias: str, phase: str = "pilot",
                   instance: str = "a") -> int:
    out, _, _ = cell_paths(benchmark, "du_mad", alias, phase)
    cmd = [sys.executable, "-m", "consensus_delivery.adapters.du_mad",
           "--block", block_for(benchmark, phase),
           "--alias", endpoint_alias(alias, instance), "--out", out,
           "--agents", str(PAIRED_RUN_BUDGET["agents"]),
           "--max-tokens", str(PAIRED_RUN_BUDGET["max_tokens"]),
           "--limit", str(PHASES[phase]["num_samples"])]
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "code"))
    return subprocess.run(cmd, cwd=os.path.join(REPO, "code"), env=env).returncode


def plan(mechanisms: List[str], backbones: List[str], phase: str = "pilot",
         benchmarks: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    benchmarks = benchmarks or BENCHMARKS
    todo: List[Tuple[str, str, str]] = []
    print(f"  {'mechanism':20s} {'backbone':14s} " +
          " ".join(f"{b:>12s}" for b in benchmarks))
    for mech in mechanisms:
        for alias in backbones:
            row = []
            for b in benchmarks:
                st = status(b, mech, alias, phase)
                row.append("ok" if st == "collected" else st.split("(")[0])
                if st == "missing" or st.startswith(("stale-lock", "records-without-manifest",
                                                     "incomplete")):
                    todo.append((b, mech, alias))
            print(f"  {mech:20s} {alias:14s} " + " ".join(f"{c:>12s}" for c in row))
    sts = [status(b, m, a, phase) for m in mechanisms for a in backbones for b in benchmarks]
    done = sum(1 for s in sts if s == "collected")
    flight = sum(1 for s in sts if s.startswith("in-flight"))
    total = len(mechanisms) * len(backbones) * len(benchmarks)
    print(f"\n  collected {done}/{total}; {flight} locked in flight; {len(todo)} unclaimed")
    return todo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print matrix status and exit")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--mechanism", action="append", choices=MECHANISMS,
                    help="restrict to one mechanism; repeatable")
    ap.add_argument("--backbone", action="append", choices=BACKBONES)
    ap.add_argument("--phase", default="pilot", choices=sorted(PHASES),
                    help="pilot: block01 at M=40. main: the disjoint block02 at M=200.")
    ap.add_argument("--benchmark", action="append", choices=BENCHMARKS,
                    help="restrict to these benchmarks. Whole blocks, so a shard never splits a "
                         "contrast across two serving instances.")
    ap.add_argument("--instance", default="a", choices=("a", "b"),
                    help="which serving instance of each backbone to use. Does not change the "
                         "cell name; it is recorded in the manifest as the endpoint URL.")
    ap.add_argument("--claim", default="cell", choices=("cell", "block"),
                    help="block: take a whole (benchmark, backbone) block and run its mechanisms, "
                         "so several collectors on different serving instances balance themselves "
                         "without splitting a contrast across instances.")
    ap.add_argument("--streams", type=int, default=1,
                    help="cells collected concurrently. Each run gets its own working directory, "
                         "so upstream's `log.txt` and the generated driver cannot collide.")
    a = ap.parse_args()

    os.makedirs(PHASES[a.phase]["dir"], exist_ok=True)
    mechs = a.mechanism or MECHANISMS
    backs = a.backbone or BACKBONES
    benches = a.benchmark or BENCHMARKS
    print(f"  phase {a.phase}: {PHASES[a.phase]['block']}, M={PHASES[a.phase]['num_samples']}, "
          f"{os.path.relpath(PHASES[a.phase]['dir'], REPO)}, instance {a.instance}, "
          f"{a.streams} stream(s)")
    todo = plan(mechs, backs, a.phase, benches)
    if not a.run:
        return 0

    say = threading.Lock()

    def block_lock(benchmark: str, backbone: str) -> str:
        return os.path.join(PHASES[a.phase]["dir"], f"_{benchmark}_{backbone}.blocklock")

    def one_cell(item: Tuple[str, str, str]) -> None:
        benchmark, mech, alias = item
        out, man, lock = cell_paths(benchmark, mech, alias, a.phase)
        tag = tag_for(benchmark, mech, alias, a.phase)
        if os.path.exists(man) and completion(man, out, PHASES[a.phase]["num_samples"])[0]:
            return
        if not claim(lock):
            with say:
                print(f"  skip {tag}: held elsewhere", flush=True)
            return
        if os.path.exists(out):
            stamp = int(os.path.getmtime(out))
            for src in (out, man):
                if os.path.exists(src):
                    kept = src.replace(".json", f".interrupted-{stamp}.json")
                    os.replace(src, kept)
                    with say:
                        print(f"  preserved {os.path.basename(src)} as "
                              f"{os.path.basename(kept)}", flush=True)
        t0 = time.time()
        with say:
            print(f"  collecting {tag} ...", flush=True)
        try:
            rc = (collect_du_mad(benchmark, alias, a.phase, a.instance) if mech == "du_mad"
                  else collect_mallm(benchmark, mech, alias, a.phase, a.instance))
        except Exception as e:
            with say:
                print(f"  {tag} RAISED {type(e).__name__}: {e}", flush=True)
            rc = -1
        finally:
            release(lock)
        with say:
            print(f"  {tag}: rc={rc} in {time.time()-t0:.0f}s", flush=True)

    def one_block(key: Tuple[str, str]) -> None:
        benchmark, backbone = key
        lk = block_lock(benchmark, backbone)
        if not claim(lk):
            with say:
                print(f"  skip block {benchmark}/{backbone}: held by another collector", flush=True)
            return
        try:
            for mech in mechs:
                one_cell((benchmark, mech, backbone))
        finally:
            release(lk)

    if a.claim == "cell":
        if a.streams <= 1:
            for item in todo:
                one_cell(item)
        else:
            order = sorted(todo, key=lambda t: (benches.index(t[0]), t[2], t[1]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.streams) as pool:
                list(pool.map(one_cell, order))
    else:
        def remaining() -> List[Tuple[str, str]]:
            return [(b, k) for b in benches for k in backs
                    if any(status(b, m, k, a.phase) != "collected" for m in mechs)]

        rounds = 0
        while True:
            left = remaining()
            if not left:
                break
            rounds += 1
            if a.streams <= 1:
                for key in left:
                    one_block(key)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=a.streams) as pool:
                    list(pool.map(one_block, left))
            after = remaining()
            if len(after) >= len(left):
                print(f"  round {rounds} collected nothing of {len(left)} remaining "
                      f"({[f'{b}/{k}' for b, k in after]}); another collector holds them or they "
                      f"fail. Stopping rather than spinning.", flush=True)
                break
    plan(mechs, backs, a.phase, benches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
