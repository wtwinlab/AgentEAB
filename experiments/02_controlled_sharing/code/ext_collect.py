#!/usr/bin/env python3
"""Collector for the two extension benchmarks: GPQA, then BIRD SQL.

    python3 ext_collect.py --plan                  # offline: queue, scale, what is already done
    python3 ext_collect.py --run                   # collect; refuses while the gate is closed
    python3 ext_collect.py --run --wait-gate       # recoverable wait, then collect when it opens
    python3 ext_collect.py --run --smoke 2         # 2 units per (protocol, condition) cell, stratified
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE, ROOT / "experiments/06_cost_resources/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ext_design as D                                                   # noqa: E402
import ext_stop_control as SC                                            # noqa: E402

OUT = D.OUT
RAW = OUT / "raw"
MANIFEST = OUT / "manifest.jsonl"

SPILLOVER_FRAC = 0.0            # one benchmark at a time: a complete benchmark is the deliverable
POLL_SECONDS = 2.0
MANIFEST_EVERY = 60.0


# ---------------------------------------------------------------- identity and versioning
def code_digest() -> str:
    """Digest of every file that decides what gets sent and what counts as an answer.
    """
    h = hashlib.sha256()
    for name in ("arm_transform.py", "answer_extract.py", "unit_driver.py",
                 "ext_sql_tasks.py", "ext_design.py", "ext_unit_driver.py"):
        h.update((HERE / name).read_bytes())
    return h.hexdigest()[:16]


CODE_VERSION = code_digest()
#: Empty until a revision happens; a differing digest is never trusted implicitly and never
#: discarded without a recorded reason established by replaying both versions over the stored
#: prompts.
CODE_EQUIVALENCE: Dict[str, Dict[str, Any]] = {}


def cache_key(item: Dict[str, Any], digest: Optional[str] = None) -> str:
    payload = {k: item[k] for k in ("benchmark", "population_sha", "unit", "protocol", "arm",
                                    "backbone", "repeat", "alias", "max_tokens")}
    payload["code"] = digest or CODE_VERSION
    payload["run_id"] = D.RUN_ID
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def accepted_keys(item: Dict[str, Any]) -> List[str]:
    keys = [cache_key(item, CODE_VERSION)]
    cell = (item["protocol"], item["benchmark"], item["arm"])
    for digest, info in CODE_EQUIVALENCE.items():
        if cell in {tuple(x) for x in info.get("invalid_for", ())}:
            continue
        keys.append(cache_key(item, digest))
    return keys


def item_seed(item: Dict[str, Any]) -> int:
    raw = "|".join([D.RUN_ID] + [str(item[k]) for k in ("benchmark", "unit", "protocol", "arm",
                                                        "backbone", "repeat")])
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- the queue
def items_for(benchmark: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    f = D.frozen(benchmark)
    units = list(f["unit_ids"])[: limit] if limit else list(f["unit_ids"])
    sha = f["sha256_16"]
    out: List[Dict[str, Any]] = []
    for proto in D.PROTOS:
        for arm in D.ARMS:
            if D.supported(benchmark, proto, arm) is not None:
                continue
            for backbone in D.PANELS[proto]:
                for rep in range(1, D.REPEATS + 1):
                    for u in units:
                        it = {"benchmark": benchmark, "population_sha": sha, "unit": u,
                              "protocol": proto, "arm": arm, "backbone": backbone,
                              "repeat": rep, "alias": D.ALIAS[backbone],
                              "max_tokens": D.PROD_MAX_TOKENS, "cost": D.COST[proto]}
                        it["key"] = cache_key(it)
                        it["accepted_keys"] = accepted_keys(it)
                        it["seed"] = item_seed(it)
                        out.append(it)
    # cheap protocols first so a stop leaves the most finished cells; unit before backbone so both
    # endpoints stay busy instead of one card idling through a whole panel
    out.sort(key=lambda d: (d["cost"], d["protocol"], D.ARMS.index(d["arm"]), d["unit"],
                            d["backbone"], d["repeat"]))
    return out


def shard_path(item: Dict[str, Any]) -> pathlib.Path:
    return RAW / ("%s__%s__%s__%s__r%d.jsonl" % (item["benchmark"], item["protocol"],
                                                 item["arm"], item["backbone"].replace(".", ""),
                                                 item["repeat"]))


def done_keys() -> set:
    """Rebuilt from the shards themselves, so the done-set cannot disagree with what is on disk."""
    keys = set()
    if not RAW.exists():
        return keys
    for p in RAW.glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            # only a record that actually carries a result counts; an error line is evidence of a
            # failure and is re-attempted, not treated as coverage
            if r.get("status") == "ok":
                keys.add(r["key"])
    return keys


def design_complete(benchmark: str, done: Optional[set] = None) -> Tuple[bool, int, int]:
    """`(complete, have, need)` against the FULL frozen design, never against this run's queue."""
    done = done_keys() if done is None else done
    total = items_for(benchmark)
    have = sum(1 for it in total if any(k in done for k in it["accepted_keys"]))
    return have == len(total), have, len(total)


# ---------------------------------------------------------------- the worker
def _worker_init():
    for _p in (ROOT / "code", HERE, ROOT / "experiments/06_cost_resources/code"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    if os.environ.get("PAM_FAKE_RUN"):
        return                    # the STOP test exercises the scheduler, not the adapters
    import ext_unit_driver as EUD
    # register up front, so a worker never pays the cost mid-item and (for SQL) the execution
    # poison is in place before the first prompt is built
    EUD.ensure_installed("qa")


def _fake_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Synthetic unit-run for the STOP test (`PAM_FAKE_RUN=<seconds>`).
    """
    time.sleep(float(os.environ.get("PAM_FAKE_RUN") or 0.2))
    rec = {k: item[k] for k in ("benchmark", "protocol", "arm", "backbone", "repeat", "unit",
                                "key", "seed", "population_sha", "alias", "max_tokens")}
    rec.update(status="ok", synthetic=True, n_calls=1, completion_tokens=1, prompt_tokens=1,
               raw_io="preserved", code_version=CODE_VERSION, pid=os.getpid(), seconds=0.0,
               run_id=D.RUN_ID)
    return rec


def _classify(exc: BaseException) -> str:
    """Four-valued availability, so an error is never silently folded into coverage."""
    name = type(exc).__name__
    text = str(exc).lower()
    if "transport" in name.lower() or "timeout" in text or "connection" in text:
        return "transport_failed"
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return "collection_error"
    if "refus" in text or "declin" in text:
        return "declined"
    return "collection_error"


def _run_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if os.environ.get("PAM_FAKE_RUN"):
        return _fake_item(item)
    import arm_transform as AT
    import collect_all_matching as C
    import ext_unit_driver as EUD
    t0 = time.time()
    rec: Dict[str, Any] = {k: item[k] for k in ("benchmark", "protocol", "arm", "backbone",
                                                "repeat", "unit", "key", "seed",
                                                "population_sha", "alias", "max_tokens")}
    rec["code_version"] = CODE_VERSION
    rec["run_id"] = D.RUN_ID
    rec["pid"] = os.getpid()
    # UNIFSEL seeds its subset from a tuple hash, and Python salts string hashing per process, so
    # the drawn subset is not re-derivable from the recorded seed unless the salt is fixed.
    rec["hash_seed"] = os.environ.get("PYTHONHASHSEED")
    try:
        out = EUD.run_unit(item["benchmark"], item["protocol"], item["unit"], item["alias"],
                           arm=item["arm"], max_tokens=item["max_tokens"], timeout=300.0,
                           seed=item["seed"])
        r = out["record"]
        calls = r.get("calls") or []
        rec.update(status="ok", record=r, arm_traces=out["arm_traces"], config=out["config"],
                   n_calls=len(calls),
                   completion_tokens=sum(x.get("completion_tokens") or 0 for x in calls),
                   prompt_tokens=sum(x.get("prompt_tokens") or 0 for x in calls),
                   raw_io=C.check_raw_preserved(r) or "preserved",
                   truncated_calls=sum(1 for x in calls if x.get("finish_reason") == "length"),
                   stop_reason=r.get("stop_reason") or r.get("native_stop_reason"))
        # cost by operation type, from the record itself: generating a response and using other
        # agents' responses are different operations and are charged separately
        ops = collections.Counter()
        otok = collections.Counter()
        for c in calls:
            seq = c.get("seq")
            rnd = c.get("round") if c.get("round") is not None else c.get("turn")
            op = "aggregation" if str(c.get("role") or "").lower().startswith("agg") else (
                "initial_generation" if (rnd in (0, None) and (seq in (0, None) or rnd == 0))
                else "revision_with_peers")
            ops[op] += 1
            otok[op] += c.get("completion_tokens") or 0
        rec["ops"] = dict(ops)
        rec["ops_output_tokens"] = dict(otok)
        if item["protocol"] == "ford":
            rec["ford_use_contrast"] = AT.use_arm_final_answer("ford", r)
    except Exception as exc:                                              # noqa: BLE001
        rec.update(status="error", error_type=type(exc).__name__, error=str(exc)[:600],
                   error_class=_classify(exc))
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


# ---------------------------------------------------------------- writer and manifest
class Writer:
    """Single-writer, append-only, fsynced. The parent is the only process that touches a shard."""

    def __init__(self):
        RAW.mkdir(parents=True, exist_ok=True)
        self._fh: Dict[pathlib.Path, Any] = {}

    def append(self, item: Dict[str, Any], rec: Dict[str, Any]) -> None:
        p = shard_path(item)
        fh = self._fh.get(p)
        if fh is None:
            fh = self._fh[p] = open(p, "a", encoding="utf-8")
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def close(self):
        for fh in self._fh.values():
            try:
                fh.close()
            except Exception:
                pass


def _record_index() -> Dict[str, Dict[str, Any]]:
    """key -> the best record on disk for it (an ok record wins over an error record)."""
    idx: Dict[str, Dict[str, Any]] = {}
    if not RAW.exists():
        return idx
    for p in sorted(RAW.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = r.get("key")
            if k is None:
                continue
            cur = idx.get(k)
            if cur is None or (cur.get("status") != "ok" and r.get("status") == "ok"):
                idx[k] = r
    return idx


def write_manifest(inflight_keys: Optional[set] = None, rate: Optional[float] = None) -> int:
    """Regenerate `manifest.jsonl` from the frozen design plus the records on disk, atomically.
    """
    idx = _record_index()
    inflight_keys = inflight_keys or set()
    rows: List[Dict[str, Any]] = []
    for b in D.BENCH_ORDER:
        try:
            items = items_for(b)
        except SystemExit:
            continue
        for it in items:
            rec = next((idx[k] for k in it["accepted_keys"] if k in idx), None)
            if rec is not None and rec.get("status") == "ok":
                status = "done"
            elif it["key"] in inflight_keys:
                status = "running"
            elif rec is not None:
                status = "failed"
            else:
                status = "to-collect"
            row = {"benchmark": b, "protocol": it["protocol"], "arm": it["arm"],
                   "backbone": it["backbone"], "model": it["alias"], "repeat": it["repeat"],
                   "task_id": it["unit"], "key": it["key"], "status": status,
                   "n_calls": (rec or {}).get("n_calls"),
                   "completion_tokens": (rec or {}).get("completion_tokens"),
                   "prompt_tokens": (rec or {}).get("prompt_tokens"),
                   "seconds": (rec or {}).get("seconds"),
                   "error_class": (rec or {}).get("error_class"),
                   "code_version": (rec or {}).get("code_version")}
            rows.append(row)
    by = collections.Counter(r["status"] for r in rows)
    pending = by["to-collect"] + by["running"]
    st = SC.read_status()
    gate_open, gate_conds = SC.gate_state()
    head = {"_header": True, "run_id": D.RUN_ID,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_version": CODE_VERSION,
            "phase": st.get("phase", "not started"),
            "gate_open": gate_open,
            "gate_unmet": [c["condition"] for c in gate_conds if not c["satisfied"]],
            "waiting_for_gate": st.get("phase") == "waiting_for_gate",
            "waiting_for_endpoints": st.get("phase") == "waiting_for_endpoints",
            "endpoints": st.get("endpoints"),
            "benchmark_order": D.BENCH_ORDER,
            "benchmark_priority_reason": {
                "gpqa": "ordered first: ~15.6 M output tokens at this design, estimated from the "
                        "released GPQA cells' own token counts, and every adapter already declares "
                        "gpqa so only verification is needed. A complete benchmark is the "
                        "deliverable, so the cheaper one ships first",
                "sql_bird_v2": "second: input-heavy (mean schema 2,284 chars on every call) and "
                               "three of the five protocols had no SQL adaptation, so it carries "
                               "both the larger cost and the new code"},
            "design": {b: {"units": D.frozen(b)["n_units"], "cells": len(D.cells(b)),
                           "unit_runs": len(D.cells(b)) * D.frozen(b)["n_units"]}
                       for b in D.BENCH_ORDER if (D.FROZEN_DIR / f"{b}.json").exists()},
            "by_status": dict(by),
            "eta_seconds": (round(pending / rate, 1) if rate and pending else None),
            "eta_utc": (time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime(time.time() + pending / rate))
                        if rate and pending else None),
            "eta_basis": (f"measured {rate:.3f} unit-runs/s in the current run" if rate
                          else "no throughput measured yet"),
            "reuse": json.loads((OUT / "REUSE_DECISION.json").read_text(encoding="utf-8"))
            if (OUT / "REUSE_DECISION.json").exists() else None}
    tmp = MANIFEST.with_suffix(".jsonl.tmp")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(head, ensure_ascii=False) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, MANIFEST)
    return len(rows)


# ---------------------------------------------------------------- endpoint preflight
def required_aliases(benchmark: str) -> List[str]:
    """Every alias the benchmark's queue will address, panels and rosters together."""
    import ext_unit_driver as EUD
    out = set()
    for proto in D.PROTOS:
        if proto in EUD.ROSTER:
            out |= set(EUD.ROSTER[proto])
        else:
            out |= {D.ALIAS[bb] for bb in D.PANELS[proto]}
    return sorted(out)


def endpoints_ready(benchmark: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """`(ready, per-alias detail)` by asking each endpoint what it serves, right now.
    """
    import urllib.request
    import ext_unit_driver as EUD
    # `install()` rebuilds the registry as `Registry(existing + new)` and `Registry.__init__`
    # raises `duplicate alias` if an alias is already there, so it is install-once, not idempotent.
    EUD.ensure_installed("qa")
    from consensus_delivery.registry_live import REGISTRY
    detail: List[Dict[str, Any]] = []
    for alias in required_aliases(benchmark):
        row: Dict[str, Any] = {"alias": alias}
        try:
            ep = REGISTRY.get(alias) if hasattr(REGISTRY, "get") else REGISTRY[alias]
            row["base_url"] = getattr(ep, "base_url", None)
            row["expected_served"] = getattr(ep, "served_name", None)
        except Exception as exc:                                          # noqa: BLE001
            row.update(ok=False, why=f"alias not registered: {type(exc).__name__}")
            detail.append(row)
            continue
        try:
            with urllib.request.urlopen(row["base_url"].rstrip("/") + "/models", timeout=8) as fh:
                served = [m.get("id") for m in (json.loads(fh.read().decode()).get("data") or [])]
            row["served"] = served
            row["ok"] = row["expected_served"] in served
            if not row["ok"]:
                row["why"] = (f"the endpoint answers but serves {served}, not "
                              f"{row['expected_served']!r}")
        except Exception as exc:                                          # noqa: BLE001
            row.update(ok=False, served=None,
                       why=f"no answer: {type(exc).__name__}: {str(exc)[:80]}")
        detail.append(row)
    return all(r.get("ok") for r in detail), detail


# ---------------------------------------------------------------- planning
def stratify(items: List[Dict[str, Any]], per_cell: int) -> List[Dict[str, Any]]:
    """Round-robin across (protocol, condition) cells, so a bounded run touches every cell.
    """
    by: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.OrderedDict()
    for it in items:
        by.setdefault((it["protocol"], it["arm"]), []).append(it)
    out: List[Dict[str, Any]] = []
    for i in range(per_cell):
        for cell in by.values():
            if i < len(cell):
                out.append(cell[i])
    return out


def plan(limit: Optional[int], smoke_per_cell: Optional[int] = None,
         only: Optional[List[str]] = None) -> List[Tuple[str, List[Dict[str, Any]]]]:
    done = done_keys()
    queues = []
    for b in D.BENCH_ORDER:
        if only and b not in only:
            continue
        if not (D.FROZEN_DIR / f"{b}.json").exists():
            continue
        items = [it for it in items_for(b, limit)
                 if not any(k in done for k in it["accepted_keys"])]
        if smoke_per_cell:
            items = stratify(items, smoke_per_cell)
        queues.append((b, items))
    return queues


# ---------------------------------------------------------------- the driver
def run(workers: int, limit: Optional[int], smoke: Optional[int], only: Optional[List[str]],
        wait_gate: bool, autoserve: bool = False) -> int:
    # ---- the gate, re-evaluated from disk right now
    while True:
        open_, conds = SC.require_gate()
        if open_:
            break
        unmet = [c["condition"] for c in conds if not c["satisfied"]]
        print("[GATE] closed -- %s" % "; ".join(unmet))
        if not wait_gate:
            print("GPU collection not started. Use --wait-gate to hold a recoverable wait.")
            return 4
        SC.write_status({"phase": "waiting_for_gate", "code_version": CODE_VERSION,
                         "gate_open": False, "gate_unmet": unmet,
                         "note": "no dispatch; re-evaluated from disk every 60 s. A process that "
                                 "exited and a file that exists are not the gate."})
        if SC.stop_requested():
            print("stop flag set while waiting for the gate -- exiting without dispatching")
            return 2
        time.sleep(60.0)

    # ---- the endpoints, asked rather than assumed
    first = next((b for b in D.BENCH_ORDER
                  if (not only or b in only) and (D.FROZEN_DIR / f"{b}.json").exists()), None)
    served_attempts = 0
    while first and not os.environ.get("PAM_FAKE_RUN"):
        ready, detail = endpoints_ready(first)
        if ready:
            print("[ENDPOINTS] %s" % ", ".join("%s=%s" % (r["alias"], r.get("served", ["?"])[0])
                                               for r in detail))
            break
        bad = [f"{r['alias']}: {r.get('why')}" for r in detail if not r.get("ok")]
        print("[ENDPOINTS] not ready -- %s" % "; ".join(bad))
        # Start this run's own llama3 server, but only now -- after the gate opened.
        if autoserve and served_attempts < 3 and any(
                not r.get("ok") and r["alias"].startswith("llama3") for r in detail):
            served_attempts += 1
            print("[ENDPOINTS] attempt %d: sh ext_serve.sh up  (ext-llama3-8b, GPU 0 only, "
                  "util 0.55)" % served_attempts)
            import subprocess
            p = subprocess.run(["sh", str(HERE / "ext_serve.sh"), "up"], cwd=str(HERE),
                               capture_output=True, text=True, timeout=1200)
            (OUT / "logs").mkdir(parents=True, exist_ok=True)
            (OUT / "logs" / f"ext_serve_up_{served_attempts}.log").write_text(
                p.stdout + "\n" + p.stderr, encoding="utf-8")
            print("[ENDPOINTS] ext_serve.sh up exited %d" % p.returncode)
            continue
        SC.write_status({"phase": "waiting_for_endpoints", "code_version": CODE_VERSION,
                         "benchmark": first, "endpoints": detail, "gate_open": True,
                         "note": "dispatch withheld on purpose: running against a missing "
                                 "endpoint would turn work-not-yet-done into transport failures. "
                                 "`sh ext_serve.sh status` shows the same state."})
        if not wait_gate:
            print("Start the missing server (`sh ext_serve.sh up`) or pass --wait-gate to hold.")
            return 5
        if SC.stop_requested():
            print("stop flag set while waiting for endpoints -- exiting without dispatching")
            return 2
        time.sleep(30.0)

    queues = plan(limit, smoke_per_cell=smoke, only=only)
    done_before = len(done_keys())
    total_pending = sum(len(q) for _b, q in queues)
    print("code version %s | run_id %s | workers %d | pending %s unit-runs | %s already on disk"
          % (CODE_VERSION, D.RUN_ID, workers, f"{total_pending:,}", f"{done_before:,}"))
    for b, q in queues:
        print("  %-14s %s pending" % (b, f"{len(q):,}"))
    if SC.stop_requested():
        print("\nCONTROL.json says stop -- not dispatching. "
              "`ext_stop_control.py resume` to clear.")
        return 2

    writer = Writer()
    t0 = time.time()
    agg = collections.Counter()
    per_bench = collections.Counter()
    bench_done: List[str] = []
    cursor = 0
    inflight: Dict[cf.Future, Dict[str, Any]] = {}
    stopping = False
    stop_at = None
    last_manifest = 0.0

    def rate() -> Optional[float]:
        n = agg["ok"] + agg["error"]
        el = time.time() - t0
        return (n / el) if (n >= 5 and el > 1) else None

    def status(phase: str):
        SC.write_status({
            "phase": phase, "code_version": CODE_VERSION,
            "benchmark": queues[cursor][0] if cursor < len(queues) else "-",
            "benchmark_index": "%d/%d" % (cursor + 1, len(queues)),
            "benchmarks_complete": bench_done,
            "workers_busy": len(inflight), "inflight": len(inflight),
            "completed_unit_runs": agg["ok"] + agg["error"],
            "ok": agg["ok"], "errors": agg["error"],
            "remaining_in_benchmark": len(queues[cursor][1]) if cursor < len(queues) else 0,
            "calls": agg["calls"], "output_tokens": agg["out"], "input_tokens": agg["inp"],
            "truncated_calls": agg["trunc"],
            "elapsed_seconds": round(time.time() - t0, 1),
            "unit_runs_per_second": round(rate(), 4) if rate() else None,
            "stop_requested": stopping, "gate_open": True,
        })

    def next_item() -> Optional[Dict[str, Any]]:
        nonlocal cursor
        while cursor < len(queues) and not queues[cursor][1]:
            cursor += 1
        if cursor >= len(queues):
            return None
        return queues[cursor][1].pop(0) if queues[cursor][1] else None

    def spill_item() -> Optional[Dict[str, Any]]:
        if SPILLOVER_FRAC <= 0:
            return None
        for j in range(cursor + 1, len(queues)):
            if queues[j][1]:
                return queues[j][1].pop(0)
        return None

    write_manifest({it["key"] for it in inflight.values()}, rate())
    ex = cf.ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
    try:
        while True:
            if not stopping and SC.stop_requested():
                stopping = True
                stop_at = time.time()
                print("\n[STOP] requested -- no new dispatch; draining up to %.0fs"
                      % SC.DRAIN_SECONDS)
                status("stopping")

            if not stopping:
                spill_budget = max(0, int(workers * SPILLOVER_FRAC))
                spilled = sum(1 for it in inflight.values()
                              if cursor < len(queues) and it["benchmark"] != queues[cursor][0])
                while len(inflight) < workers:
                    it = next_item()
                    if it is None:
                        if spilled >= spill_budget:
                            break
                        it = spill_item()
                        if it is None:
                            break
                        spilled += 1
                    inflight[ex.submit(_run_item, it)] = it

            if not inflight:
                if stopping or all(not q for _b, q in queues):
                    break
                continue

            timeout = POLL_SECONDS
            if stopping:
                left = SC.DRAIN_SECONDS - (time.time() - stop_at)
                if left <= 0:
                    print("[STOP] drain window elapsed with %d request(s) in flight; "
                          "cancelling them" % len(inflight))
                    for f in inflight:
                        f.cancel()
                    break
                timeout = min(POLL_SECONDS, left)
            done_now, _ = cf.wait(list(inflight), timeout=timeout,
                                  return_when=cf.FIRST_COMPLETED)
            for f in done_now:
                it = inflight.pop(f)
                try:
                    rec = f.result()
                except Exception as exc:
                    rec = dict(it)
                    rec.update(status="error", error_type=type(exc).__name__,
                               error=str(exc)[:600], error_class="collection_error")
                writer.append(it, rec)
                agg[rec["status"]] += 1
                per_bench[(it["benchmark"], rec["status"])] += 1
                agg["calls"] += rec.get("n_calls") or 0
                agg["out"] += rec.get("completion_tokens") or 0
                agg["inp"] += rec.get("prompt_tokens") or 0
                agg["trunc"] += rec.get("truncated_calls") or 0
            if done_now:
                status("stopping" if stopping else "running")
                n = agg["ok"] + agg["error"]
                if n % 25 == 0 or stopping:
                    el = time.time() - t0
                    print("  %5d runs | ok %d err %d | %s calls | %.2f Mtok out | %.0f tok/s | %s"
                          % (n, agg["ok"], agg["error"], f"{agg['calls']:,}",
                             agg["out"] / 1e6, agg["out"] / max(el, 1),
                             queues[cursor][0] if cursor < len(queues) else "-"))
                if time.time() - last_manifest > MANIFEST_EVERY:
                    write_manifest({x["key"] for x in inflight.values()}, rate())
                    last_manifest = time.time()
    finally:
        ex.shutdown(wait=not stopping, cancel_futures=stopping)
        writer.close()
        status("stopped" if stopping else "finished")

    el = time.time() - t0
    print("\n%s: %d ok, %d errors, %s calls, %.2f Mtok out, %.2f Mtok in, %.1f min, %.0f tok/s"
          % ("STOPPED" if stopping else "finished", agg["ok"], agg["error"],
             f"{agg['calls']:,}", agg["out"] / 1e6, agg["inp"] / 1e6, el / 60,
             agg["out"] / max(el, 1)))
    final_done = done_keys()
    for b in D.BENCH_ORDER:
        if not (D.FROZEN_DIR / f"{b}.json").exists():
            continue
        okc, have, need = design_complete(b, final_done)
        if okc:
            bench_done.append(b)
        else:
            print("  %-14s incomplete: %s/%s unit-runs on disk" % (b, f"{have:,}", f"{need:,}"))
    print("benchmarks COMPLETE per the frozen design: %s" % (", ".join(bench_done) or "none"))
    write_manifest(set(), rate())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--manifest", action="store_true", help="regenerate manifest.jsonl and exit")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="units per cell; the design when off")
    ap.add_argument("--smoke", type=int, default=None,
                    help="N units per (protocol, arm) cell, stratified -- small-batch validation")
    ap.add_argument("--benchmark", action="append", default=None)
    ap.add_argument("--wait-gate", action="store_true")
    ap.add_argument("--autoserve", action="store_true",
                    help="after the gate opens, start THIS run's own llama3 server (GPU 0, "
                         "util 0.55, container ext-llama3-8b) if that endpoint is missing. Never "
                         "before the gate: holding a shared card idle while waiting is the "
                         "opposite of releasing it. Bounded to 3 attempts.")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.manifest:
        n = write_manifest()
        print("manifest.jsonl: %s rows -> %s" % (f"{n:,}", MANIFEST))
        return 0
    if a.plan:
        qs = plan(a.limit, smoke_per_cell=a.smoke, only=a.benchmark)
        tot = sum(len(q) for _b, q in qs)
        print("code version %s | run_id %s | pending %s unit-runs"
              % (CODE_VERSION, D.RUN_ID, f"{tot:,}"))
        for b, q in qs:
            by = collections.Counter((it["protocol"], it["arm"]) for it in q)
            okc, have, need = design_complete(b)
            print("  %-14s %7s pending  (%d cells)  design %s/%s%s"
                  % (b, f"{len(q):,}", len(by), f"{have:,}", f"{need:,}",
                     "  COMPLETE" if okc else ""))
        open_, conds = SC.gate_state()
        print("\ngate: %s" % ("OPEN" if open_ else "CLOSED"))
        for c in conds:
            print("  [%s] %s" % ("x" if c["satisfied"] else " ", c["condition"]))
        first = next((b for b, _q in qs), None)
        if first:
            ready, detail = endpoints_ready(first)
            print("endpoints for %s: %s" % (first, "READY" if ready else "NOT READY"))
            for r in detail:
                print("  [%s] %-16s %s  %s" % ("x" if r.get("ok") else " ", r["alias"],
                                               r.get("base_url"), r.get("why", "")))
        print("STOP entry: python3 ext_stop_control.py stop")
        return 0
    if a.run:
        return run(a.workers, a.limit, a.smoke, a.benchmark, a.wait_gate, a.autoserve)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
