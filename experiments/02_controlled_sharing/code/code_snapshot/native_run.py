"""Run a native mechanism once, and preserve everything a later re-evaluation will need.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional


def _block_ids(block_path: str) -> Optional[set]:
    """The frozen block's task ids, so a log-extracted failure can be checked against reality."""
    try:
        return {r["example_id"] for r in json.load(open(block_path, encoding="utf-8"))}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _sha(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def endpoint_counters(base_url: str) -> Dict[str, float]:
    """vLLM's Prometheus counters, so a run's token cost is recorded even when the method exposes none."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    out: Dict[str, float] = {}
    try:
        with urllib.request.urlopen(root.rstrip("/") + "/metrics", timeout=20) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001
        return out
    for name in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
                 "vllm:request_success_total"):
        tot = 0.0
        found = False
        for line in text.splitlines():
            if line.startswith(name):
                try:
                    tot += float(line.rsplit(" ", 1)[1])
                    found = True
                except (ValueError, IndexError):
                    pass
        if found:
            out[name] = tot
    return out


def failed_samples(log_text: str, valid_ids: Optional[set] = None) -> Dict[str, Any]:
    r"""The failures upstream records only in its log, extracted before the log is overwritten.
    """
    ids: List[str] = []
    # The phrase, then optionally one newline, then the bracketed list. The last occurrence wins.
    hits = list(re.finditer(r"samples failed[^\n\[]*\n?[^\n\[]*\[([^\]]*)\]", log_text))
    if hits:
        ids = re.findall(r"'([^'\n]+)'", hits[-1].group(1))
    rejected = []
    if valid_ids is not None:
        rejected = [i for i in ids if i not in valid_ids]
        ids = [i for i in ids if i in valid_ids]
    reasons: Dict[str, int] = {}
    for pat in ("division by zero", "Exceeded maximum API retries",
                "System message must be at the beginning", "context length",
                "Expecting value"):
        n = log_text.count(pat)
        if n:
            reasons[pat] = n
    return {"failed_example_ids": ids, "n_failed": len(ids), "reason_counts": reasons,
            "rejected_as_not_task_ids": rejected}


def run(*, variant_root: str, block_path: str, out_path: str, endpoint_url: str, model_name: str,
        instruction: str, protocol: str, num_agents: int, max_turns: int, max_tokens: int,
        concurrency: int, num_samples: int, manifest_path: str,
        python_bin: str = "",
        workdir: Optional[str] = None) -> int:
    """One native run, with its manifest. Returns the subprocess exit code.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tag = os.path.basename(out_path)[: -len(".json")] if out_path.endswith(".json") \
        else os.path.basename(out_path)
    workdir = workdir or os.path.join(os.path.dirname(os.path.abspath(out_path)), f"_run_{tag}")
    os.makedirs(workdir, exist_ok=True)
    driver = os.path.join(workdir, "_native_driver.py")
    with open(driver, "w", encoding="utf-8") as f:
        f.write(
            "import json, sys, time, warnings\n"
            "warnings.filterwarnings('ignore')\n"
            f"sys.path.insert(0, {variant_root!r})\n"
            "from mallm.scheduler import Scheduler\n"
            "from mallm.utils.config import Config\n"
            "cfg = Config(\n"
            f"    input_json_file_path={block_path!r},\n"
            f"    output_json_file_path={out_path!r},\n"
            f"    task_instruction_prompt={instruction!r},\n"
            f"    endpoint_url={endpoint_url!r}, model_name={model_name!r}, api_key='-',\n"
            f"    decision_protocol={protocol!r}, num_agents={num_agents}, max_turns={max_turns},\n"
            f"    max_tokens={max_tokens}, concurrent_api_requests={concurrency},\n"
            # Frozen: the all-endorsement window is undetermined in the source and is reachable only
            # through `skip_decision_making=True` or a judge regenerate.
            "    skip_decision_making=False,\n"
            f"    num_samples={num_samples},\n"
            ")\n"
            "t0 = time.time()\n"
            "Scheduler(cfg).run()\n"
            "print(f'WALL {time.time()-t0:.1f}')\n")

    before = endpoint_counters(endpoint_url)
    t0 = time.time()
    proc = subprocess.run([python_bin, driver], cwd=workdir, capture_output=True, text=True)
    wall = time.time() - t0
    after = endpoint_counters(endpoint_url)

    # Written into this run's own working directory, so a concurrent cell cannot overwrite it.
    log_src = os.path.join(workdir, "log.txt")
    if not os.path.isfile(log_src):
        # Older runs put it in the upstream tree; still read it if that is where it landed.
        log_src = os.path.join(variant_root, "log.txt")
    log_dst = out_path.replace(".json", "") + ".log.txt"
    log_text = ""
    if os.path.isfile(log_src):
        # Copied before the next run's `filemode="w"` destroys it.
        shutil.copy2(log_src, log_dst)
        log_text = open(log_dst, encoding="utf-8", errors="replace").read()

    try:
        commit = subprocess.run(["git", "-C", variant_root, "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""

    n_records = 0
    if os.path.isfile(out_path):
        try:
            n_records = len(json.load(open(out_path, encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass

    block_manifest = block_path.replace(".json", "") + ".manifest.json"
    manifest = {
        "kind": "native-run",
        "variant_root": variant_root, "variant_base_commit": commit,
        "variant_consensus_sha256": _sha(os.path.join(
            variant_root, "mallm", "decision_protocols", "consensus.py")),
        "block_path": block_path,
        "block_sha256": (json.load(open(block_manifest)).get("block_sha256")
                         if os.path.isfile(block_manifest) else None),
        "endpoint_url": endpoint_url, "model_name": model_name,
        "workdir": os.path.relpath(workdir, os.path.dirname(os.path.abspath(manifest_path))),
        "protocol": protocol, "num_agents": num_agents, "max_turns": max_turns,
        "max_tokens": max_tokens, "concurrency": concurrency, "num_samples": num_samples,
        "skip_decision_making": False, "judge_regenerate": False,
        "instruction": instruction,
        "records_written": n_records,
        "wall_s": round(wall, 1),
        "exit_code": proc.returncode,
        # Token cost of the whole run, from the endpoint's own counters. The method exposes none
        # per call, so this is the only cost record a native run can have.
        "endpoint_tokens": {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)},
        "stdout_tail": proc.stdout.strip().splitlines()[-3:],
        **failed_samples(log_text, valid_ids=_block_ids(block_path)),
        # The log is this run's own: `workdir` is private, so no concurrent run could overwrite it.
        "log_reliable": True,
        "log_copy": os.path.basename(log_dst) if log_text else None,
        "note": "Pilot. The mechanism's own output is the primary object; no writer was added and no "
                "prompt, role or stop rule was altered.",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.remove(driver)
    try:
        if not os.listdir(workdir):
            os.rmdir(workdir)                    # nothing left to keep; the log was copied aside
    except OSError:
        pass
    print(f"  {os.path.basename(block_path)}: {n_records} records, {wall:.0f}s, "
          f"{manifest['n_failed']} failed, tokens={manifest['endpoint_tokens']}")
    return proc.returncode


if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
