#!/usr/bin/env python3
"""Benchmark-major controlled collection: 5 protocol families x 5 benchmarks x 4 conditions.

    python3 collect_controlled.py --plan              # offline: queue, scale, cache hits
    python3 collect_controlled.py --run               # collect, honouring CONTROL.json
    python3 collect_controlled.py --run --smoke 12    # small batch first, then stop
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
from typing import Any, Dict, Iterator, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import stop_control as SC                                              # noqa: E402

OUT = ROOT / "workspace/controlled_sharing"
RAW = OUT / "raw"
FROZEN = ROOT / "data/frozen/FROZEN_SAMPLE_5X5.json"

#: benchmark order, and why. A QA benchmark is ~11.8 M output tokens; a matching benchmark is ~106
#: M.
BENCH_ORDER = ["gsm8k", "strategyqa", "mmlu_pro", "geographic_settlements", "wdc_products"]
MATCHING = {"wdc_products", "geographic_settlements", "musicbrainz20k"}

PROTOS = ["ford", "moa", "du_mad", "madm2_objective", "bccs"]
ARMS = ["FULL", "NOPEER", "ANSONLY", "UNIFSEL"]
#: frozen panels. Homogeneous protocols run both backbones as a blocking factor; the heterogeneous
#: ones run their frozen roster, which must not vary or the selection condition confounds with the roster.
PANELS = {"du_mad": ["llama3-8b", "qwen2.5-7b"], "madm2_objective": ["llama3-8b", "qwen2.5-7b"],
          "bccs": ["llama3-8b", "qwen2.5-7b"], "ford": ["roster"], "moa": ["roster"]}
REPEATS = 2
#: one primary alias per backbone; the registry's replicas are load-balanced behind it
ALIAS = {"llama3-8b": "llama3-8b-g1", "qwen2.5-7b": "qwen25-7b-g1b", "roster": "llama3-8b-g1"}
PROD_MAX_TOKENS = 1024

#: cheap protocols first inside a benchmark, so a stop leaves the most finished cells
COST = {"ford": 3.3, "moa": 4.0, "du_mad": 6.0, "madm2_objective": 6.0, "bccs": 15.0}
SPILLOVER_FRAC = 0.25
POLL_SECONDS = 2.0


# ---------------------------------------------------------------- identity and versioning
def code_digest() -> str:
    """Digest of the files that decide what gets sent and what counts as an answer.
    """
    h = hashlib.sha256()
    for name in ("arm_transform.py", "answer_extract.py", "unit_driver.py"):
        h.update((HERE / name).read_bytes())
    return h.hexdigest()[:16]


CODE_VERSION = code_digest()

#: A differing digest is not by itself a reason to discard a record -- what matters is whether the code
#: behaves the same for that cell -- and a matching digest is not a substitute for checking.
CONTAMINATION: List[Dict[str, Any]] = [
    {
        "defect": "moa-reference-split",
        "mechanism": "MoA joins its references with `\\n<n>. `, which is byte-identical to a "
                     "numbered step a model writes inside its own answer, so one reference was "
                     "parsed as several (2.71-2.85x over-count on GSM8K, 1.03-1.11x on WDC). "
                     "ANSONLY then rewrote fragments instead of references and UNIFSEL could keep "
                     "a fragment in place of a reference.",
        "corrupts_sent_input_for": [("moa", b, a)
                                    for b in ("gsm8k", "strategyqa", "mmlu_pro",
                                              "geographic_settlements", "wdc_products")
                                    for a in ("ANSONLY", "UNIFSEL")],
        "metadata_only_for": [("moa", b, a) for b in ("gsm8k", "strategyqa", "mmlu_pro",
                                                      "geographic_settlements", "wdc_products")
                              for a in ("FULL", "NOPEER")],
        "metadata_note": "FULL does not modify the prompt and NOPEER rebuilds it with no "
                         "references at all, so their *data* is valid; only the recorded peer "
                         "counts were inflated, and those are recomputed offline from the stored "
                         "prompts plus the stored replies.",
        "digests": ["1d9c1de0c3e0ea86", "0474f3a32b6d8687", "68ed04479c53495f"],
        "repaired_in": "the boundaries are rebuilt from the reply list by MoA's own formula and "
                       "required to match byte for byte; any digest NOT listed in `digests` above "
                       "carries the repair (a digest is named here only when it is contaminated, "
                       "so the list cannot go stale as the code version moves)",
    },
    {
        "defect": "ford-instruction-dropped-by-peer-removal",
        "mechanism": "`run_one` appends FORD's answer-format instruction to the LAST message of "
                     "the view, which from turn 1 on is a peer turn, and the instruction text "
                     "differs per benchmark. Splitting on one marker sentence failed on the "
                     "variants that open with \"Do you think I am more reasonable?\" (GSM8K, "
                     "MMLU-Pro), so removing peer turns removed the instruction with them: the "
                     "agents stopped answering in the form `parse_prediction` reads, the "
                     "two-consecutive-agreement rule could never fire, and FORD delivered nothing "
                     "(0 of 200 on GSM8K).",
        "corrupts_sent_input_for": [("ford", b, a) for b in ("gsm8k", "mmlu_pro")
                                    for a in ("NOPEER", "ANSONLY")],
        "metadata_only_for": [],
        "metadata_note": "",
        "digests": ["1d9c1de0c3e0ea86", "0474f3a32b6d8687"],
        "repaired_in": "the instruction is taken from the adapter and stripped as an exact suffix, "
                       "so no per-benchmark variant can be missed",
    },
    {
        "defect": "ford-mcq-option-count",
        "mechanism": "`n_options` was frozen at 2 for FORD -- the matching family's binary value -- "
                     "and `ford.instruction` renders the answer contract from it, so every MMLU-Pro "
                     "prompt said \"Answer: (A|B) is more plausible.\" while 79 of the 100 frozen "
                     "tasks offer ten options and 78 have a gold letter outside {A,B}. Measurable: "
                     "FORD's answers span A-G and never H/I/J, while BCCS on the same tasks reaches "
                     "all ten; A and B are 42% of delivered answers against 22% of gold.",
        "corrupts_sent_input_for": [("ford", "mmlu_pro", a)
                                    for a in ("FULL", "NOPEER", "ANSONLY")],
        "metadata_only_for": [],
        "metadata_note": "",
        "digests": ["1d9c1de0c3e0ea86", "0474f3a32b6d8687", "68ed04479c53495f"],
        "repaired_in": "the count comes from the option letters each task actually lists "
                       "(MMLU-Pro resolves to 4/7/8/9/10, GPQA to 4); as above, only contaminated "
                       "digests are listed, so this does not go stale",
    },
]


def contaminated_cells(digest: str) -> set:
    """Cells whose **sent input** is corrupt under `digest`, from the mechanism table."""
    out = set()
    for d in CONTAMINATION:
        if digest in d["digests"]:
            out |= {tuple(c) for c in d["corrupts_sent_input_for"]}
    return out


CODE_EQUIVALENCE: Dict[str, Dict[str, Any]] = {
    "03eaf1cd1f67d933": {
        "superseded_by": "the mcq bare-letter parser now refuses a letter preceded by a digit",
        "why_it_mattered": "without the guard the parser read the letter H out of '6H' and "
                           "the letter E out of '4E' -- NMR multiplicities and energy levels "
                           "-- as option letters. "
                           "It is a real defect, but it changed **nothing already sent**: measured "
                           "over 4,032 controlled MMLU-Pro and 4,018 extension GPQA ANSONLY "
                           "prompts, the rendering is identical under both parsers in every single "
                           "case, so this digest's records stay valid for every cell.",
        "invalid_for": [],
        "verified_by": "zero-difference measurement over 8,050 real prompts (mcq_impact)",
    },
    "5efa521db60f4dc4": {
        "superseded_by": "the FORD per-task `n_options` override was moved to after `kind` is "
                         "assigned",
        "why_it_mattered": "placed before it, the override raised `UnboundLocalError: kind` and "
                           "failed every FORD mcq item. The change touches only the "
                           "`protocol == 'ford' and kind == 'mcq'` path, so it cannot alter any "
                           "other protocol or any non-mcq benchmark -- which is why the MoA "
                           "records collected under this digest stay valid.",
        "invalid_for": [("ford", b, a) for b in ("mmlu_pro", "gpqa")
                        for a in ("FULL", "NOPEER", "ANSONLY", "UNIFSEL")],
        "verified_by": "the only diff is inside the ford/mcq branch; MoA and every non-mcq cell "
                       "take an unchanged path",
    },
    "68ed04479c53495f": {
        "superseded_by": "MoA's reference split now uses the replies the execution actually "
                         "produced instead of the `\\n<n>. ` regex",
        "why_it_mattered": "MoA joins its references with a delimiter byte-identical to a numbered "
                           "step a model writes inside its own answer, so one reference was split "
                           "into several: 2.71-2.85x over-count on GSM8K, 1.03-1.11x on WDC. "
                           "ANSONLY then rewrote fragments instead of references (18-90% of "
                           "prompts affected, by benchmark) and UNIFSEL could keep a fragment in "
                           "place of a reference (20-94%). NOPEER rebuilds the prompt with no "
                           "references at all, so it is untouched -- and neither is any other "
                           "protocol, since the regex is MoA-only.",
        "invalid_for": "see CONTAMINATION; resolved by contaminated_cells()",
        "verified_by": "verify_equiv.py plus ext_test_moa_refs.py (23 adversarial cases)",
    },
    "0474f3a32b6d8687": {
        "superseded_by": "the FORD instruction split now comes from the adapter instead of one "
                         "hard-coded marker sentence",
        "why_it_mattered": "FORD's instruction differs per benchmark. GSM8K and MMLU-Pro open "
                           "with \"Do you think I am more reasonable? ...\" rather than the "
                           "marker the old code split on, so on those two the instruction stayed "
                           "glued to the peer turn and NOPEER deleted it. The agents then stopped "
                           "answering in the form `parse_prediction` reads, the stop rule could "
                           "never fire, and FORD delivered nothing (0/200 on GSM8K).",
        # the FORD instruction issue, plus the MoA reference-split issue: the latter predates this
        # digest as well, so records made under it are equally unusable for those cells.
        "invalid_for": [("ford", "gsm8k", "NOPEER"), ("ford", "gsm8k", "ANSONLY"),
                        ("ford", "mmlu_pro", "NOPEER"), ("ford", "mmlu_pro", "ANSONLY")],
        "plus": "see CONTAMINATION; resolved by contaminated_cells()",
        "verified_by": "verify_equiv.py: both versions applied to the stored prompts; 91 cells "
                       "byte-identical, exactly these 4 changed",
    },
}


def cache_key(item: Dict[str, Any], digest: Optional[str] = None) -> str:
    payload = {k: item[k] for k in ("benchmark", "population_sha", "unit", "protocol", "arm",
                                    "backbone", "repeat", "alias", "max_tokens")}
    payload["code"] = digest or CODE_VERSION
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def accepted_keys(item: Dict[str, Any]) -> List[str]:
    """Every cache key under which a VALID record for this item may already exist.
    """
    keys = [cache_key(item, CODE_VERSION)]
    cell = (item["protocol"], item["benchmark"], item["arm"])
    for digest, info in CODE_EQUIVALENCE.items():
        iv = info.get("invalid_for")
        listed = {tuple(x) for x in iv} if isinstance(iv, (list, tuple)) else set()
        if cell in listed or cell in contaminated_cells(digest):
            continue
        keys.append(cache_key(item, digest))
    return keys


def item_seed(item: Dict[str, Any]) -> int:
    """Content-derived seed: identical regardless of when the item ran."""
    raw = "|".join(str(item[k]) for k in ("benchmark", "unit", "protocol", "arm", "backbone",
                                          "repeat"))
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- the queue
def frozen_units(benchmark: str) -> Tuple[List[str], str]:
    blob = json.loads(FROZEN.read_text(encoding="utf-8"))
    if benchmark in blob.get("matching", {}):
        e = blob["matching"][benchmark]
        return list(e["unit_ids"]), e["sha256_16"]
    e = blob["qa"][benchmark]
    return list(e["shared_task_ids"]), e["shared_sha256_16"]


def items_for(benchmark: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Every unit-run the frozen design asks for in this benchmark, cheap protocols first."""
    import unit_driver as UD
    units, sha = frozen_units(benchmark)
    if limit:
        units = units[:limit]
    out: List[Dict[str, Any]] = []
    for proto in PROTOS:
        for arm in ARMS:
            if UD.supported(benchmark, proto, arm) is not None:
                continue                      # e.g. FORD/UNIFSEL is a use contrast, not a prompt
            for backbone in PANELS[proto]:
                for rep in range(1, REPEATS + 1):
                    for u in units:
                        it = {"benchmark": benchmark, "population_sha": sha, "unit": u,
                              "protocol": proto, "arm": arm, "backbone": backbone,
                              "repeat": rep, "alias": ALIAS[backbone],
                              "max_tokens": PROD_MAX_TOKENS, "cost": COST[proto]}
                        it["key"] = cache_key(it)
                        it["accepted_keys"] = accepted_keys(it)
                        it["seed"] = item_seed(it)
                        out.append(it)
    # Unit before backbone, deliberately.
    out.sort(key=lambda d: (d["cost"], d["protocol"], ARMS.index(d["arm"]), d["unit"],
                            d["backbone"], d["repeat"]))
    return out


def design_complete(benchmark: str, done: Optional[set] = None) -> Tuple[bool, int, int]:
    """`(complete, done, total)` against the FULL frozen design, never against this run's queue.
    """
    done = done_keys() if done is None else done
    total = items_for(benchmark)
    have = sum(1 for it in total if any(k in done for k in it["accepted_keys"]))
    return have == len(total), have, len(total)


def shard_path(item: Dict[str, Any]) -> pathlib.Path:
    return RAW / ("%s__%s__%s__%s__r%d.jsonl" % (item["benchmark"], item["protocol"],
                                                 item["arm"], item["backbone"].replace(".", ""),
                                                 item["repeat"]))


def done_keys() -> set:
    """Rebuilt from the shards, so the done-set cannot disagree with what is on disk."""
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
            # Only a successful record completes an item.
            if r.get("status") == "ok":
                keys.add(r.get("key"))
    return keys


# ---------------------------------------------------------------- the worker
_WORKER_READY = False


def _worker_init():
    """Per-process one-time setup. Each process gets its own module globals, which is the point."""
    global _WORKER_READY
    for _p in (ROOT / "code", HERE):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import unit_driver as UD
    UD._ensure_installed()
    _WORKER_READY = True


def _fake_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """A synthetic unit-run for the STOP test, enabled by `PAM_FAKE_RUN=<seconds>`.
    """
    time.sleep(float(os.environ.get("PAM_FAKE_RUN") or 0.2))
    rec = {k: item[k] for k in ("benchmark", "protocol", "arm", "backbone", "repeat", "unit",
                                "key", "seed", "population_sha", "alias", "max_tokens")}
    rec.update(status="ok", synthetic=True, n_calls=1, completion_tokens=1, prompt_tokens=1,
               raw_io="preserved", code_version=CODE_VERSION, pid=os.getpid(), seconds=0.0)
    return rec


def _run_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """One unit-run, in this process, with this item's condition installed and then removed."""
    if os.environ.get("PAM_FAKE_RUN"):
        return _fake_item(item)
    import unit_driver as UD
    t0 = time.time()
    rec: Dict[str, Any] = {k: item[k] for k in ("benchmark", "protocol", "arm", "backbone",
                                                "repeat", "unit", "key", "seed",
                                                "population_sha", "alias", "max_tokens")}
    rec["code_version"] = CODE_VERSION
    rec["pid"] = os.getpid()
    # UNIFSEL seeds its subset from `(seed, protocol, condition, call_seq).__hash__()`, and Python salts
    # string hashing per process, so that value differs in every worker and the drawn subset cannot
    # be recomputed from the recorded seed.
    rec["hash_seed"] = os.environ.get("PYTHONHASHSEED")
    try:
        out = UD.run_unit(item["benchmark"], item["protocol"], item["unit"], item["alias"],
                          arm=item["arm"], max_tokens=item["max_tokens"], timeout=180.0,
                          seed=item["seed"])
        r = out["record"]
        import collect_all_matching as C
        import arm_transform as AT
        calls = r.get("calls") or []
        rec.update(status="ok", record=r, arm_traces=out["arm_traces"], config=out["config"],
                   n_calls=len(calls),
                   completion_tokens=sum(x.get("completion_tokens") or 0 for x in calls),
                   prompt_tokens=sum(x.get("prompt_tokens") or 0 for x in calls),
                   raw_io=C.check_raw_preserved(r) or "preserved",
                   stop_reason=r.get("stop_reason") or r.get("native_stop_reason"))
        if item["protocol"] == "ford":
            rec["ford_use_contrast"] = AT.use_arm_final_answer("ford", r)
    except Exception as exc:
        # keep the true classification: an infrastructure failure is an error, never "not executed"
        rec.update(status="error", error_type=type(exc).__name__, error=str(exc)[:600],
                   error_class=_classify(exc))
    rec["seconds"] = round(time.time() - t0, 2)
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


# ---------------------------------------------------------------- the driver
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


def stratify(items: List[Dict[str, Any]], per_cell: int) -> List[Dict[str, Any]]:
    """Round-robin across cells, so a bounded run touches every (protocol, condition) rather than one.
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
         ) -> List[Tuple[str, List[Dict[str, Any]]]]:
    done = done_keys()
    queues = []
    for b in BENCH_ORDER:
        items = [it for it in items_for(b, limit)
                 if not any(k in done for k in it["accepted_keys"])]
        if smoke_per_cell:
            items = stratify(items, smoke_per_cell)
        queues.append((b, items))
    return queues


def run(workers: int, limit: Optional[int], smoke: Optional[int]) -> int:
    queues = plan(limit, smoke_per_cell=smoke)
    done_before = len(done_keys())
    total_pending = sum(len(q) for _b, q in queues)
    print("code version %s | workers %d | pending %s unit-runs | %s already on disk"
          % (CODE_VERSION, workers, f"{total_pending:,}", f"{done_before:,}"))
    for b, q in queues:
        print("  %-24s %s pending" % (b, f"{len(q):,}"))
    if SC.stop_requested():
        print("\nCONTROL.json says stop -- not dispatching. `stop_control.py resume` to clear.")
        return 2

    writer = Writer()
    t0 = time.time()
    agg = collections.Counter()
    per_bench = collections.Counter()
    bench_done: List[str] = []
    queue_drained: List[str] = []
    cursor = 0                                  # index of the benchmark being completed
    inflight: Dict[cf.Future, Dict[str, Any]] = {}
    stopping = False
    stop_at = None
    cap = None

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
            "elapsed_seconds": round(time.time() - t0, 1),
            "stop_requested": stopping,
        })

    def next_item() -> Optional[Dict[str, Any]]:
        """Current benchmark first; spill into the next one only to fill idle capacity."""
        nonlocal cursor
        while cursor < len(queues) and not queues[cursor][1]:
            b = queues[cursor][0]
            # only a benchmark whose whole frozen design is on disk counts as complete
            if b not in queue_drained:
                queue_drained.append(b)   # judged at report time, not now: more may still land
            cursor += 1
        if cursor >= len(queues):
            return None
        if queues[cursor][1]:
            return queues[cursor][1].pop(0)
        return None

    def spill_item() -> Optional[Dict[str, Any]]:
        for j in range(cursor + 1, len(queues)):
            if queues[j][1]:
                return queues[j][1].pop(0)
        return None

    ex = cf.ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
    try:
        while True:
            if not stopping and SC.stop_requested():
                stopping = True
                stop_at = time.time()
                print("\n[STOP] requested -- no new dispatch; draining up to %.0fs"
                      % SC.DRAIN_SECONDS)
                status("stopping")

            # ---- dispatch
            if not stopping and not (cap and agg["ok"] + agg["error"] + len(inflight) >= cap):
                spill_budget = max(1, int(workers * SPILLOVER_FRAC))
                spilled = sum(1 for it in inflight.values()
                              if cursor < len(queues) and it["benchmark"] != queues[cursor][0])
                while len(inflight) < workers:
                    if cap and agg["ok"] + agg["error"] + len(inflight) >= cap:
                        break
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
                if stopping or cap and agg["ok"] + agg["error"] >= cap:
                    break
                if all(not q for _b, q in queues):
                    break
                continue

            # ---- collect
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
            if done_now:
                status("stopping" if stopping else "running")
                n = agg["ok"] + agg["error"]
                if n % 25 == 0 or stopping:
                    el = time.time() - t0
                    print("  %5d runs | ok %d err %d | %s calls | %.1f Mtok out | %.0f tok/s "
                          "| %s" % (n, agg["ok"], agg["error"], f"{agg['calls']:,}",
                                    agg["out"] / 1e6, agg["out"] / max(el, 1),
                                    queues[cursor][0] if cursor < len(queues) else "-"))
    finally:
        ex.shutdown(wait=not stopping, cancel_futures=stopping)
        writer.close()
        status("stopped" if stopping else "finished")

    el = time.time() - t0
    print("\n%s: %d ok, %d errors, %s calls, %.2f Mtok out, %.2f Mtok in, %.1f min, %.0f tok/s"
          % ("STOPPED" if stopping else "finished", agg["ok"], agg["error"],
             f"{agg['calls']:,}", agg["out"] / 1e6, agg["inp"] / 1e6, el / 60,
             agg["out"] / max(el, 1)))
    for b in BENCH_ORDER:
        o, e = per_bench[(b, "ok")], per_bench[(b, "error")]
        if o or e:
            print("  %-24s ok %s, errors %s" % (b, f"{o:,}", f"{e:,}"))
    # completeness is judged once, at the end, against the frozen design and the final done-set
    final_done = done_keys()
    for b in BENCH_ORDER:
        okc, have, total = design_complete(b, final_done)
        if okc:
            bench_done.append(b)
    print("benchmarks COMPLETE per the frozen design: %s" % (", ".join(bench_done) or "none"))
    for b in BENCH_ORDER:
        okc, have, total = design_complete(b, final_done)
        if not okc and have:
            print("  %-24s incomplete: %s/%s unit-runs on disk"
                  % (b, f"{have:,}", f"{total:,}"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None,
                    help="units per cell; the frozen design when omitted")
    ap.add_argument("--smoke", type=int, default=None,
                    help="N units per (protocol, arm) cell, stratified -- small-batch validation")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.plan:
        qs = plan(a.limit, smoke_per_cell=a.smoke)
        tot = sum(len(q) for _b, q in qs)
        print("code version %s | pending %s unit-runs" % (CODE_VERSION, f"{tot:,}"))
        for b, q in qs:
            by = collections.Counter((it["protocol"], it["arm"]) for it in q)
            print("  %-24s %8s pending  (%d cells)" % (b, f"{len(q):,}", len(by)))
        print("\nbenchmark order: %s" % " -> ".join(BENCH_ORDER))
        print("STOP entry: python3 stop_control.py stop")
        return 0
    if a.run:
        return run(a.workers, a.limit, a.smoke)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
