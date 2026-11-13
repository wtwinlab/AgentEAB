#!/usr/bin/env python3
"""Run every interaction workflow on the three common matching test populations.

    python3 collect_all_matching.py --population musicbrainz20k --protocol dmad \
        --alias llama3-8b-g1 --repeat 0 --out <path>
    python3 collect_all_matching.py --population wdc_products --protocol ford --repeat 0 --out <path>
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import pathlib
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "code"),
                str(ROOT / "experiments/04_sql_selection_generation/code"),
                str(ROOT / "experiments/06_cost_resources/code"),
                str(ROOT / "experiments/05_matching_downstream/code"),
                str(HERE)]

import matching_tasks as MT                                            # noqa: E402
import endpoints_all_matching as EP                                    # noqa: E402

SEED = 20261018
LOCK = threading.Lock()
POPULATIONS = ("geographic_settlements", "wdc_products", "musicbrainz20k")
EXPECTED = {"geographic_settlements": 4719, "wdc_products": 2615, "musicbrainz20k": 32199}

#: Fields shown per population. Settlements renders through its own record_text, so it is unset.
FIELDS = {"musicbrainz20k": ["title", "artist", "album", "year", "length", "number", "language"]}


#: Per-call field holding each protocol's OWN extracted answer.
CONTRACT_KEYS = {
    "bccs": ("extracted_answer",), "madm2_naive": ("extracted_answer",),
    "madm2_objective": ("extracted_answer",), "du_mad": ("extracted_answer",),
    "dov_vote": ("vote_released", "vote_strict"), "ford": ("prediction",),
    "moa": ("extracted_answer", "answer"), "reconcile": ("answer",),
    "dmad": ("normalised_answer", "parsed_answer"), "liang_mad": (),
}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ------------------------------------------------------------------- the three populations
#: Set to 24 after 8 proved too tight.
MAX_UNIT_ATTEMPTS = 24


def load_population(name: str):
    """[(unit_key, question, gold, meta)] over the test candidate pairs, in a frozen order.
    """
    import collect_matrix as CM
    proto = (ROOT / "data/records/matching_protocol" /
             f"{name}.json")
    blob = json.loads(proto.read_text(encoding="utf-8"))
    ent, split = blob["entity_of"], blob["split"]
    dom = CM.Domain(name)
    pairs = sorted({(str(a), str(b)) for a, b in blob["candidates"]
                    if split.get(ent.get(str(a))) == "test"
                    and split.get(ent.get(str(b))) == "test"})
    if len(pairs) != EXPECTED[name]:
        raise SystemExit(f"{name}: {len(pairs)} test pairs, expected {EXPECTED[name]}")
    flds = FIELDS.get(name)
    out = []
    for a, b in pairs:
        q = MT.matching_question(dom.task, dom.fields(a), dom.fields(b), flds)
        out.append((f"{name}|{a}|{b}", q,
                    1 if ent[a] == ent[b] else 0, {"a": a, "b": b, "population": name}))
    return out, sha(proto.read_text(encoding="utf-8"))


# -------------------------------------------------------- one invocation spec per workflow
#: `team` is "homog" (one alias, so backbone x repeat) or "hetero" (the roster IS the method).
#: Every config value is the paper's own, from collect_batch2.MECHANISMS / collect_backbone.
SPECS = {
    "bccs": dict(team="homog", module="bccs", kw=dict(
        agents=7, rounds=3, n_clusters=3, top_k=2, temperature=0.7, belief="intended")),
    # `n_options` is required by dmad and ford but is inert for a binary task: it only sizes the
    # MCQ letter list, and `extra_info("matching_pair", 2)` returns "". 2 = the two labels.
    "dmad": dict(team="homog", module="dmad", kw=dict(
        agents=5, rounds=5, temperature=1.0, top_p=0.9, n_options=2)),
    "madm2_naive": dict(team="homog", module="mad_m2", kw=dict(
        agents=3, rounds=2, prune="naive", temperature=1.0, top_p=1.0)),
    "madm2_objective": dict(team="homog", module="mad_m2", kw=dict(
        agents=3, rounds=2, prune="objective", temperature=1.0, top_p=1.0)),
    "dov_vote": dict(team="homog", module="debate_or_vote", kw=dict(
        agents=5, rounds=5, solver="vote")),
    "du_mad": dict(team="homog", module="du_mad", no_task_id=True, kw=dict(
        agents=3, rounds=2)),
    "liang_mad": dict(team="homog", module="liang_mad", kw=dict(max_round=3)),
    "ford": dict(team="hetero", module="ford", kw=dict(max_rounds=6, n_options=2),
                 roster=("llama3-8b-g1", "qwen25-7b-g2"), arg="aliases"),
    "moa": dict(team="hetero", module="moa", kw=dict(temperature=0.7),
                roster=("llama3-8b-g1", "qwen25-7b-g2", "gemma4-g5"), arg="references",
                extra=dict(aggregator="qwen25-7b-g2")),
    "reconcile": dict(team="hetero", module="reconcile", kw=dict(rounds=3),
                      roster=("llama3-8b-g1", "qwen25-7b-g2", "gemma4-g5"), arg="roster"),
}


def runner(protocol: str):
    import importlib
    mod = importlib.import_module(
        f"consensus_delivery.adapters.{SPECS[protocol]['module']}")
    return mod.run_one, mod


def build_kwargs(protocol: str, spec, alias, max_tokens, timeout, retries):
    kw = dict(spec["kw"])
    kw.update(max_tokens=max_tokens, timeout_s=timeout, retries=retries)
    if spec["team"] == "homog":
        kw["alias"] = alias
    else:
        kw[spec["arg"]] = list(spec["roster"]) if spec["arg"] != "aliases" \
            else tuple(spec["roster"])
        kw.update(spec.get("extra") or {})
        if protocol == "reconcile":
            from consensus_delivery.adapters import reconcile as RC
            kw["convincing"] = RC.load_convincing(MT.BENCH)
    return kw


_INSTALLED = False


def _ensure_installed():
    """Idempotently perform the installs `main()` does before its first call."""
    global _INSTALLED
    if _INSTALLED:
        return
    MT.install()
    import endpoints_all_matching as _EP
    _EP.install()
    _INSTALLED = True


def run_single(population: str, protocol: str, unit_key: str, alias: str, *,
               max_tokens: int = 256, timeout: float = 60.0, retries: int = 0):
    """One unit through one protocol, with no shard file and no resume.
    """
    # `main()` performs two installs before any call: the task adaptation that teaches every
    # adapter the `matching_pair` benchmark, and the endpoint registry that maps an alias to a
    # served checkpoint.
    _ensure_installed()
    units, _sha = load_population(population)
    spec = SPECS[protocol]
    run_one, _mod = runner(protocol)
    hit = next((u for u in units if u[0] == unit_key), None)
    if hit is None:
        raise KeyError(f"{unit_key} is not in {population}'s frozen population")
    uid, question, _gold, meta = hit
    exposed = [x for x in (str(meta["a"]), str(meta["b"])) if len(x) > 6 and x in question]
    if exposed:
        raise RuntimeError(f"gold id present in the prompt: {exposed}")
    call = dict(build_kwargs(protocol, spec, alias, max_tokens, timeout, retries))
    if not spec.get("no_task_id"):
        call["task_id"] = uid
    return run_one(question, benchmark=MT.BENCH, **call)


# --------------------------------------------------------------------------- raw preservation
def check_raw_preserved(rec) -> str:
    """`""` when every call kept its full input and output, else the reason it did not."""
    calls = rec.get("calls") or []
    if not calls:
        return "no calls recorded"
    for i, c in enumerate(calls):
        if not c.get("messages_sent"):
            return f"call {i} lost messages_sent (input not re-parseable offline)"
        if c.get("raw_reply") is None:
            return f"call {i} lost raw_reply (output not re-parseable offline)"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True, choices=POPULATIONS)
    ap.add_argument("--protocol", required=True, choices=sorted(SPECS))
    ap.add_argument("--alias", default="")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="pilot only; 0 means the full population")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--attempt", type=int, default=0, help="a retry writes its own attempt file")
    ap.add_argument("--retry-from", default="",
                    help="path to a base shard file; collect ONLY the unit_ids whose record there "
                         "is not ok. The base file is read, never written: the failure stays in "
                         "the evidence and the retry lands in this run's own --out attempt file.")
    ap.add_argument("--roster", default="",
                    help="comma-separated replica aliases overriding a hetero team's default. "
                         "Serving location only: each entry must be a replica of the same "
                         "checkpoint the default names, which endpoints_all_matching verifies.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = dict(SPECS[a.protocol])
    if a.roster:
        want = tuple(x.strip() for x in a.roster.split(",") if x.strip())
        import endpoints_all_matching as _EP
        base = [_EP.PLAN[x][2] for x in spec["roster"]]
        got = [_EP.PLAN[x][2] for x in want]
        if got != base:
            raise SystemExit(f"--roster {want} serves {got}, not the method's {base}: a roster "
                             f"override may move serving location, never the model family")
        spec["roster"] = want
    if spec["team"] == "homog" and not a.alias:
        raise SystemExit(f"{a.protocol} is homogeneous: --alias is required")
    install = MT.install()
    install["signature_order"] = MT.verify_signature_order()
    install["endpoints"] = EP.install()
    run_one, mod = runner(a.protocol)

    units, pop_sha = load_population(a.population)
    if a.limit:
        units = units[:a.limit]
    team_id = a.alias if spec["team"] == "homog" else "+".join(spec["roster"])

    out_path = pathlib.Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Only an OK record counts as collected.
    done = set()
    attempts = collections.Counter()
    last_error = {}
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                done.add(r["unit_id"])
            else:
                attempts[r["unit_id"]] += 1
                last_error[r["unit_id"]] = str(r.get("error") or r.get("reason") or r.get("status"))
    # Measured on Debate-or-Vote: five WDC units overflow the frozen 8192-token context in every
    # configuration, and one shard spent 399 calls retrying its thirteen never-landed units up to 32
    # times each.
    retired = {u: {"attempts": attempts[u], "last_error": last_error.get(u, "")}
               for u in attempts if u not in done and attempts[u] >= MAX_UNIT_ATTEMPTS}
    if a.retry_from:
        base = pathlib.Path(a.retry_from)
        failed = set()
        for line in base.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok":
                failed.add(r["unit_id"])
        ok_ids = set()
        for line in base.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                ok_ids.add(r["unit_id"])
        failed -= ok_ids
        todo = [u for u in units
                if f"{a.protocol}|{team_id}|{a.repeat}|{u[0]}" in failed
                and f"{a.protocol}|{team_id}|{a.repeat}|{u[0]}" not in done]
        print(f"  retry mode: {len(failed)} failed units in {base.name}, "
              f"{len(todo)} still to re-collect", flush=True)
    else:
        todo = [u for u in units
                if f"{a.protocol}|{team_id}|{a.repeat}|{u[0]}" not in done
                and f"{a.protocol}|{team_id}|{a.repeat}|{u[0]}" not in retired]
    print(f"  {a.population}/{a.protocol}/{team_id}/r{a.repeat}: "
          f"{len(todo)} to collect, {len(done)} present"
          + (f", {len(retired)} retired after {MAX_UNIT_ATTEMPTS} failed attempts"
             if retired else ""), flush=True)
    if retired:
        # named, with its cause, next to the shard it belongs to
        exh = out_path.with_suffix(out_path.suffix + ".retired.json")
        exh.write_text(json.dumps({
            "population": a.population, "protocol": a.protocol, "team": team_id,
            "repeat": a.repeat, "max_unit_attempts": MAX_UNIT_ATTEMPTS,
            "note": ("these units were attempted at least MAX_UNIT_ATTEMPTS times and never "
                     "produced an ok record; they are excluded from collection, NOT from the "
                     "population count, so the cell still reports short of its population"),
            "retired": retired}, indent=1) + "\n", encoding="utf-8")
    if not todo:
        return 0

    stats = collections.Counter()
    t0 = time.time()
    kwargs = build_kwargs(a.protocol, spec, a.alias, a.max_tokens, a.timeout, a.retries)

    def work(item):
        uid, question, gold, meta = item
        full = f"{a.protocol}|{team_id}|{a.repeat}|{uid}"
        # gold isolation: neither the label nor either entity id may appear in the prompt
        exposed = [s for s in (str(meta["a"]), str(meta["b"])) if len(s) > 6 and s in question]
        if exposed:
            with LOCK:
                stats["id_leak_blocked"] += 1
            return
        t1 = time.time()
        call = dict(kwargs)
        if not spec.get("no_task_id"):
            call["task_id"] = uid
        try:
            rec = run_one(question, benchmark=MT.BENCH, **call)
        except Exception as exc:                                        # noqa: BLE001
            with LOCK:
                stats["error"] += 1
                stats[f"err:{type(exc).__name__}"] += 1
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "unit_id": full, "population": a.population, "protocol": a.protocol,
                        "team": team_id, "repeat": a.repeat, "attempt": a.attempt,
                        "status": "error", "error": f"{type(exc).__name__}: {exc}"[:600],
                        "meta": meta}) + "\n")
            return
        bad = check_raw_preserved(rec)
        if bad:
            with LOCK:
                stats["raw_not_preserved"] += 1
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "unit_id": full, "population": a.population, "protocol": a.protocol,
                        "team": team_id, "repeat": a.repeat, "attempt": a.attempt,
                        "status": "raw_not_preserved", "reason": bad, "meta": meta,
                        "record": rec}) + "\n")
            return
        calls = rec.get("calls") or []
        env = {
            "unit_id": full, "population": a.population, "protocol": a.protocol,
            "team": team_id, "team_kind": spec["team"], "repeat": a.repeat,
            "attempt": a.attempt, "status": "ok", "benchmark": MT.BENCH,
            "answer_kind": MT.KIND, "question_sha": sha(question),
            "question_chars": len(question), "population_sha256_16": pop_sha,
            "meta": meta, "gold_withheld_from_prompt": True,
            "is_task_adaptation": True,
            "adapter_stamped_adaptation": bool(rec.get("is_task_adaptation")),
            "raw_io_preserved": True,
            "wall_s": round(time.time() - t1, 3), "n_calls": len(calls),
            "prompt_tokens_total": sum(int(c.get("prompt_tokens") or 0) for c in calls),
            "completion_tokens_total": sum(int(c.get("completion_tokens") or 0) for c in calls),
            "truncated_calls": sum(1 for c in calls if c.get("finish_reason") == "length"),
            "retry_calls": sum(1 for c in calls if int(c.get("attempt") or 0) > 0),
            # contract satisfaction: how many calls produced an answer under the protocol's OWN
            # extraction.
            "calls_with_extracted_answer": (
                sum(1 for c in calls
                    if next((c.get(k) for k in CONTRACT_KEYS.get(a.protocol, ())
                             if c.get(k) not in (None, "", "None")), None) is not None)
                if CONTRACT_KEYS.get(a.protocol) else None),
            "record": rec,
        }
        with LOCK:
            stats["ok"] += 1
            stats["calls"] += len(calls)
            stats["ctok"] += env["completion_tokens_total"]
            stats["ptok"] += env["prompt_tokens_total"]
            stats["trunc"] += env["truncated_calls"]
            # None means "this workflow has no per-call answer field", which is Liang MAD.
            if env["calls_with_extracted_answer"] is not None:
                stats["contract_ok"] += env["calls_with_extracted_answer"]
            if rec.get("majority_answer") in (None, ""):
                stats["declined"] += 1
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(env) + "\n")
            n = stats["ok"] + stats["error"]
            if n % 25 == 0:
                el = max(time.time() - t0, 1e-9)
                print(f"    {n}/{len(todo)} {el:.0f}s {n/el:.2f} exec/s "
                      f"{stats['calls']/el:.1f} calls/s out {stats['ctok']/el:.0f} tok/s "
                      f"contract {stats['contract_ok']/max(stats['calls'],1):.3f} "
                      f"declined {stats['declined']/max(stats['ok'],1):.3f} "
                      f"trunc={stats['trunc']}", flush=True)

    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))

    el = max(time.time() - t0, 1e-9)
    summary = {"population": a.population, "protocol": a.protocol, "team": team_id,
               "team_kind": spec["team"], "repeat": a.repeat, "attempt": a.attempt,
               "units_attempted": len(todo), "elapsed_s": round(el, 1),
               "stats": dict(stats), "seed": SEED,
               "exec_per_s": round(stats["ok"] / el, 4),
               "calls_per_s": round(stats["calls"] / el, 2),
               "output_tok_per_s": round(stats["ctok"] / el, 1),
               "contract_satisfaction": round(stats["contract_ok"] / max(stats["calls"], 1), 4),
               "decline_rate": round(stats["declined"] / max(stats["ok"], 1), 4),
               "config": {k: v for k, v in kwargs.items() if k != "convincing"},
               "extension": install, "population_sha256_16": pop_sha}
    log = (ROOT / "results/protocol_all_matching_20261018T192619Z/logs" /
           (out_path.stem + f".a{a.attempt}.summary.json"))
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"  done: {dict(stats)} in {el:.0f}s -> {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
