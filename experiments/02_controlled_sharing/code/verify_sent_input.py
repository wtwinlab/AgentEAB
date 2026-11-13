#!/usr/bin/env python3
"""Verify the reconstructed prompt against a hash of the request that was actually sent.

    PYTHONHASHSEED=0 python3 verify_sent_input.py --calibrate      # is the payload recipe right?
    PYTHONHASHSEED=0 python3 verify_sent_input.py --verify         # all arms, full population

Writes `analysis/controlled_sharing/sent_input_verification.{csv,json}`. Zero model calls.

Why a request hash
------------------

`chars_after` is a character count written by the same transform code at collection time, so
matching it checks self-consistency only; a hash of the sent payload checks the bytes.

There is a real witness in the records. `providers.chat` computes

    request_sha = sha256(json.dumps(payload, sort_keys=True))[:32]                  # providers.py:194

over the payload it is about to send, **after** the arm wrapper has already replaced
`kw["messages"]`. So `request_sha` fingerprints the real request body, transformed messages
included. Reproducing it proves the reconstruction equals what was sent.

Reproducing it also needs every other payload field, and getting one wrong would fail the hash for
reasons having nothing to do with the prompt. The recipe is read off the adapters rather than
guessed -- `top_p=1.0` everywhere, `temperature` 0.7 for BCCS/MoA and 1.0 for MAD-M^2 from the
driver's config but hard-coded 0.0 inside FORD and Du-MAD, `logprobs=True, top_logprobs=1` for the
three protocols that consume token probabilities, `max_tokens` 1024, and no `request_extra` on these
local endpoints -- and then **calibrated on the FULL arm**, where the transform is the identity and
the messages are therefore certainly right. If the recipe did not reproduce FULL's hashes, nothing
below would be evidence about prompts, and `--calibrate` says so rather than reporting a percentage.

Two reconstruction modes, because they prove different things
-------------------------------------------------------------

**seed-only** re-derives the arm edit from `(seed, protocol, arm, call_seq)` exactly as the
collection code did. This is the property the report was really claiming, and it is the one that
fails: UNIFSEL draws its subset from `random.Random((seed, protocol, arm, call_seq).__hash__())`,
and Python salts string hashing per process, so the draw is not recomputable in a later process
unless `PYTHONHASHSEED` was pinned.

**trace-replay** substitutes the subset the collection run actually drew, which is recorded per call
in the trace's `peer_indices_kept`. The transform consumes that draw only through membership tests,
so replaying the recorded indices reproduces the text byte for byte. The distinction matters: the selection is **not recomputable from the seed**, but the sent
prompt is still **verifiable**, because the draw was recorded and the request hash can check it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "analysis/controlled_sharing"

#: The decoding parameters each adapter passes to `providers.chat`, read off the call sites rather
#: than inferred from the record: FORD (`ford.py:155`) and Du-MAD (`du_mad.py:244`) pin
#: `temperature=0.0` inside the adapter, so the driver's configured temperature is NOT what was
#: sent for those two; the others pass the configured value through.
TEMPERATURE: Dict[str, Optional[float]] = {
    "bccs": None, "moa": None, "madm2_objective": None, "madm2_naive": None,
    "ford": 0.0, "du_mad": 0.0,
}
#: `logprobs=True, top_logprobs=1` is requested by the protocols whose mechanism consumes token
#: probabilities (BCCS multiplies them into a belief, MAD-M^2 masks peers by perplexity).
LOGPROB_PROTOCOLS = {"bccs", "madm2_objective", "madm2_naive"}
TOP_P = 1.0


def request_sha(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def payload_for(messages: Sequence[Dict[str, str]], served: str, protocol: str,
                temperature: float, max_tokens: int) -> Dict[str, Any]:
    """Exactly `providers.chat`'s assembly order, for the endpoints used in this collection."""
    payload: Dict[str, Any] = {
        "model": served,
        "messages": [dict(m) for m in messages],
        "temperature": temperature,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
    }
    if protocol in LOGPROB_PROTOCOLS:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1
    # these local vLLM endpoints carry no `request_extra`, and no caller passes `extra`
    return payload


class _ReplayRandom:
    """Stands in for `random.Random` and returns the draw the collection run actually made.

    UNIFSEL is the only arm that draws, it draws once per block-bearing message through
    `rng.sample`, and the transform uses the result only through membership tests. So handing back
    the recorded `peer_indices_kept` reproduces the rendered text exactly, even though the sorted
    record has lost the draw order.
    """

    def __init__(self, kept: Sequence[int]) -> None:
        self.kept = list(kept)
        self.calls = 0
        self.size_mismatch = False

    def sample(self, population: Iterable[int], k: int) -> List[int]:
        self.calls += 1
        if len(self.kept) != k:
            # the recorded draw cannot be the one this call site asked for
            self.size_mismatch = True
        return list(self.kept)


class _ReplayFactory:
    def __init__(self, kept: Sequence[int]) -> None:
        self.rng = _ReplayRandom(kept)

    def Random(self, _seed: Any = None) -> _ReplayRandom:  # noqa: N802  (mirrors random.Random)
        return self.rng


def check_record(r: Dict[str, Any], mode: str, AT, kind: str, task: str) -> List[Dict[str, Any]]:
    """One record -> one row per call, saying whether the rebuilt payload hashes to the sent one."""
    import random as _real_random

    rows: List[Dict[str, Any]] = []
    protocol = r["protocol"]
    rec = r.get("record") or {}
    calls = rec.get("calls") or []
    traces = r.get("arm_traces") or []
    cfg = r.get("config") or {}
    max_tokens = r.get("max_tokens") or cfg.get("max_tokens")
    temp = TEMPERATURE.get(protocol)
    if temp is None:
        temp = cfg.get("temperature")
    replies: List[str] = []
    for ci, c in enumerate(calls):
        stored = c.get("messages_sent") or []
        rsha = c.get("request_sha")
        if not stored or not rsha:
            replies.append(c.get("raw_reply") or "")
            continue
        tr_rec = traces[ci] if ci < len(traces) else {}
        kept = tr_rec.get("peer_indices_kept") or []
        factory: Any = None
        if mode == "replay":
            factory = _ReplayFactory(kept)
            AT.random = factory
        try:
            new, tr = AT.transform(protocol, r["arm"], stored, kind=kind, benchmark=task,
                                   call_seq=ci, seed=r.get("seed", 0),
                                   observed_replies=list(replies))
        finally:
            AT.random = _real_random
        got = request_sha(payload_for(new, c.get("served_model"), protocol, temp, max_tokens))
        rebuilt_chars = sum(len(m.get("content") or "") for m in new)
        rows.append({
            "source": r.get("_source"),
            "benchmark": r["benchmark"], "protocol": protocol, "arm": r["arm"],
            "backbone": r["backbone"], "unit": r["unit"], "repeat": r["repeat"], "call_seq": ci,
            "mode": mode,
            "sha_match": got == rsha,
            "recorded_request_sha": rsha, "rebuilt_request_sha": got,
            "recorded_chars_after": tr_rec.get("chars_after"),
            "rebuilt_chars": rebuilt_chars,
            "chars_match": tr_rec.get("chars_after") == rebuilt_chars,
            "recorded_kept": json.dumps(kept),
            "rebuilt_kept": json.dumps(tr.get("peer_indices_kept") or []),
            "kept_match": list(kept) == list(tr.get("peer_indices_kept") or []),
            "draw_was_replayed": bool(factory and factory.rng.calls),
            "replay_size_mismatch": bool(factory and factory.rng.size_mismatch),
            "trace_changed": tr_rec.get("changed"),
            "code_version": r.get("code_version"),
        })
        replies.append(c.get("raw_reply") or "")
    return rows


def _one_benchmark(spec: Tuple[str, str, int]) -> List[Dict[str, Any]]:
    bench, mode, limit = spec
    # Request hashes are recomputed with the transform the collection runner imported for these
    # records (`arm_transform_collection.py`). Set VERIFY_ARM_TRANSFORM=current for records whose
    # prompts were built by `arm_transform.py`.
    if os.environ.get("VERIFY_ARM_TRANSFORM", "collection") == "current":
        import arm_transform as AT
    else:
        import arm_transform_collection as AT
    import inputs as IN
    src = IN.installed_for(bench)
    kind = IN.kind_for(bench)
    if src.tag == "main" and IN.is_matching(bench):
        import matching_tasks as MT
        task = MT.BENCH                    # the three matching populations share one task adapter
    else:
        task = bench                       # QA and SQL pass the benchmark name straight through
    # `gate=False`: this script produces the evidence the extension's admissibility gate consults.
    # `apply_transform=False`: the SQL verdict rewrite touches answers, not the prompts measured here.
    recs, _drop = IN.read(bench, gate=(src.tag == "main"), apply_transform=False)
    out: List[Dict[str, Any]] = []
    seen: Dict[Any, int] = collections.Counter()
    for r in recs:
        cell = (r["protocol"], r["arm"], r["backbone"])
        if limit and seen[cell] >= limit:
            continue
        seen[cell] += 1
        try:
            out.extend(check_record(r, mode, AT, kind, task))
        except Exception as exc:  # noqa: BLE001
            out.append({"source": r.get("_source"),
                        "benchmark": bench, "protocol": r["protocol"], "arm": r["arm"],
                        "backbone": r["backbone"], "unit": r["unit"], "repeat": r["repeat"],
                        "call_seq": -1, "mode": mode, "sha_match": False,
                        "recorded_request_sha": None, "rebuilt_request_sha": None,
                        "recorded_chars_after": None, "rebuilt_chars": None,
                        "chars_match": False, "recorded_kept": None, "rebuilt_kept": None,
                        "kept_match": False, "draw_was_replayed": False,
                        "replay_size_mismatch": False, "trace_changed": None,
                        "code_version": r.get("code_version"),
                        "error": "%s: %s" % (type(exc).__name__, exc)})
    return out


def rate(rows: Sequence[Dict[str, Any]], key: str = "sha_match") -> Tuple[int, int]:
    return sum(1 for r in rows if r.get(key)), len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="FULL arm only: is the recipe right?")
    ap.add_argument("--verify", action="store_true", help="all arms, both reconstruction modes")
    ap.add_argument("--limit", type=int, default=0, help="records per cell; 0 = the whole population")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("NOTE: PYTHONHASHSEED is %r, not '0'. The seed-only mode is then being given a "
              "different salt than the pinned part of the collection had, which understates it. "
              "Re-run with PYTHONHASHSEED=0." % os.environ.get("PYTHONHASHSEED"))
    import inputs as IN
    benches = list(IN.BENCHMARKS)
    modes = ["seed"] if a.calibrate else ["seed", "replay"]
    specs = [(b, m, a.limit) for m in modes for b in benches]
    t0 = time.time()
    with mp.Pool(a.procs) as pool:
        got = pool.map(_one_benchmark, specs)
    rows = [r for g in got for r in g]
    if not rows:
        print("no records read")
        return 1
    print("read %s call-rows across %d benchmarks in %.0f s\n" % (f"{len(rows):,}", len(benches),
                                                                  time.time() - t0))

    full = [r for r in rows if r["arm"] == "FULL" and r["mode"] == "seed"]
    fm, fn = rate(full)
    print("=== CALIBRATION: the FULL arm, where the transform is the identity ===")
    print("the reconstructed messages are certainly right here, so this tests the payload recipe")
    by_p = collections.defaultdict(list)
    for r in full:
        by_p[r["protocol"]].append(r)
    for p in sorted(by_p):
        m, n = rate(by_p[p])
        print("   %-18s %9s calls %9s reproduce the sent hash %7.2f%%"
              % (p, f"{n:,}", f"{m:,}", 100 * m / max(n, 1)))
    print("   %-18s %9s calls %9s %28.2f%%" % ("ALL FULL", f"{fn:,}", f"{fm:,}",
                                               100 * fm / max(fn, 1)))
    conclusive = bool(fn and fm == fn)
    if not conclusive:
        print("\n   -> the recipe does NOT reproduce the hash even where the messages are certain,")
        print("      so a mismatch elsewhere is not attributable to the prompt and nothing below")
        print("      may be read as evidence about what was sent.")
    else:
        print("   -> every FULL call reproduces exactly. The recipe is right, so any mismatch on an")
        print("      intervention arm is attributable to the reconstructed messages.")

    if a.calibrate:
        print("\n--calibrate only; no table written.")
        return 0 if conclusive else 1

    print("\n=== the intervention arms, by reconstruction mode ===")
    print("%-9s %-18s %12s %22s %22s" % ("arm", "protocol", "calls",
                                         "seed-only reproduces", "trace-replay reproduces"))
    inter = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        inter[(r["arm"], r["protocol"])][r["mode"]].append(r)
    for k in sorted(inter):
        s_m, s_n = rate(inter[k]["seed"])
        r_m, r_n = rate(inter[k]["replay"])
        print("%-9s %-18s %12s %13s %7.2f%% %13s %7.2f%%"
              % (k[0], k[1], f"{s_n:,}", f"{s_m:,}", 100 * s_m / max(s_n, 1),
                 f"{r_m:,}", 100 * r_m / max(r_n, 1)))

    print("\n=== by raw root, trace-replay mode (this is what the admissibility gate reads) ===")
    by_root: Dict[str, Dict[str, Any]] = {}
    for tag in sorted({r["source"] for r in rows if r.get("source")}):
        sel = [r for r in rows if r["source"] == tag and r["mode"] == "replay"]
        m, n = rate(sel)
        digs = sorted({r["code_version"] for r in sel})
        by_root[tag] = {"calls": n, "reproduced": m, "digests": digs,
                        "benchmarks": sorted({r["benchmark"] for r in sel})}
        print("   %-5s %12s calls %12s reproduce the sent request %8.4f%%   digests: %s"
              % (tag, f"{n:,}", f"{m:,}", 100 * m / max(n, 1), ", ".join(digs)))
    if "ext" in by_root:
        e = by_root["ext"]
        print("   -> the extension's digest %s is %s: today's transform %s the requests that"
              % (", ".join(e["digests"]),
                 "ADMISSIBLE" if e["reproduced"] == e["calls"] else "NOT admissible",
                 "reproduces" if e["reproduced"] == e["calls"] else "does NOT reproduce"))
        print("      were actually sent, which is the equivalence the moved digest stands in for.")

    seed_rows = [r for r in rows if r["mode"] == "seed"]
    rep_rows = [r for r in rows if r["mode"] == "replay"]
    s_m, s_n = rate(seed_rows)
    r_m, r_n = rate(rep_rows)
    print("\nDENOMINATOR: %s calls, every one in the valid analysis population (`read()` applies the"
          % f"{s_n:,}")
    print("same admissibility rule as the effects tables, so superseded records are not counted).")
    print("   seed-only   : %s of %s reproduce the sent request  (%.4f%%; %s do not)"
          % (f"{s_m:,}", f"{s_n:,}", 100 * s_m / max(s_n, 1), f"{s_n - s_m:,}"))
    print("   trace-replay: %s of %s reproduce the sent request  (%.4f%%; %s do not)"
          % (f"{r_m:,}", f"{r_n:,}", 100 * r_m / max(r_n, 1), f"{r_n - r_m:,}"))

    print("\n=== what the non-reproducing calls are, under trace-replay ===")
    bad = [r for r in rep_rows if not r["sha_match"]]
    if not bad:
        print("   none: every call in the valid population is byte-exactly accounted for.")
    else:
        dist = collections.Counter((r["arm"], r["protocol"], r["benchmark"]) for r in bad)
        for k, n in dist.most_common(20):
            print("   %-9s %-18s %-24s %s" % (k[0], k[1], k[2], f"{n:,}"))
        print("\n   difference types:")
        kinds = collections.Counter()
        for r in bad:
            if r.get("error"):
                kinds["raised while transforming: %s" % r["error"][:60]] += 1
            elif r["replay_size_mismatch"]:
                kinds["recorded draw size does not match the call site's k"] += 1
            elif not r["kept_match"]:
                kinds["selected peer set differs from the recorded one"] += 1
            elif not r["chars_match"]:
                kinds["same selection, different length: the text itself differs"] += 1
            else:
                kinds["same selection and same length, different bytes"] += 1
        for k, n in kinds.most_common():
            print("      %-62s %s" % (k, f"{n:,}"))

    # the arms whose effect estimates would be touched if a call cannot be accounted for
    print("\n=== does any of this touch the valid data the effects are computed from? ===")
    aff = collections.Counter()
    for r in bad:
        aff[(r["benchmark"], r["protocol"], r["arm"], r["backbone"])] += 1
    units = {(r["benchmark"], r["protocol"], r["arm"], r["backbone"], r["unit"], r["repeat"])
             for r in bad}
    print("   calls not accounted for: %s" % f"{len(bad):,}")
    print("   unit-runs containing at least one: %s" % f"{len(units):,}")
    print("   cells containing at least one: %s" % f"{len(aff):,}")

    # The per-call table is 1.9 M rows / 425 MB, which is not a reasonable repository artifact when
    # the script regenerates it in under a minute. Persist the per-cell aggregate (every call is
    # still counted in it) plus every row that did NOT reproduce -- the rows an auditor needs to
    # look at individually.
    agg: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        k = (r["mode"], r["source"], r["arm"], r["protocol"], r["benchmark"], r["backbone"], r["code_version"])
        a = agg.setdefault(k, {"mode": k[0], "arm": k[1], "protocol": k[2], "benchmark": k[3],
                               "backbone": k[4], "code_version": k[5], "calls": 0,
                               "reproduced_sent_request": 0, "matched_recorded_chars_after": 0,
                               "selected_subset_differs": 0,
                               "wrong_but_chars_after_would_pass": 0})
        a["calls"] += 1
        a["reproduced_sent_request"] += 1 if r["sha_match"] else 0
        a["matched_recorded_chars_after"] += 1 if r["chars_match"] else 0
        if not r["sha_match"]:
            a["selected_subset_differs"] += 0 if r["kept_match"] else 1
            a["wrong_but_chars_after_would_pass"] += 1 if r["chars_match"] else 0
    with open(OUT / "sent_input_verification.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(agg.values()))))
        w.writeheader()
        w.writerows([agg[k] for k in sorted(agg)])
    fields = ["source", "benchmark", "protocol", "arm", "backbone", "unit", "repeat", "call_seq",
              "mode",
              "sha_match", "recorded_request_sha", "rebuilt_request_sha",
              "recorded_chars_after", "rebuilt_chars", "chars_match", "recorded_kept",
              "rebuilt_kept", "kept_match", "draw_was_replayed", "replay_size_mismatch",
              "trace_changed", "code_version", "error"]
    with open(OUT / "sent_input_mismatches.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows([r for r in rows if not r["sha_match"]])

    blob = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "witness": "providers.py:194 computes request_sha = sha256(json.dumps(payload, "
                   "sort_keys=True))[:32] AFTER the arm wrapper replaced kw['messages'], so it "
                   "fingerprints the request body that was actually sent",
        "supersedes": "the report's '97.96% of 18,193 calls match chars_after', which compared "
                      "against a character count written by the same transform at collection time "
                      "(self-consistency, and a count cannot see a changed byte) on an unstated "
                      "sample",
        "calibration": {"arm": "FULL", "calls": fn, "reproduced": fm, "conclusive": conclusive,
                        "meaning": "the transform is the identity on FULL, so this tests only the "
                                   "payload recipe"},
        "denominator": "every call of every record in the valid analysis population, selected by "
                       "the same admissibility rule as the effects tables",
        "modes": {
            "seed": {"calls": s_n, "reproduced": s_m,
                     "meaning": "the arm edit re-derived from (seed, protocol, arm, call_seq) as "
                                "the collection code did; this is what the report was claiming"},
            "replay": {"calls": r_n, "reproduced": r_m,
                       "meaning": "the UNIFSEL draw substituted from the per-call recorded "
                                  "peer_indices_kept; the transform consumes the draw only through "
                                  "membership tests, so this reproduces the text exactly"},
        },
        "by_root": by_root,
        "ext_digest_admissibility": (
            "the extension's records carry a digest that no longer recomputes (three of the six "
            "files it hashes were edited by the 2026-10-20 repair) and its CODE_EQUIVALENCE table "
            "is empty, so inputs.py admits them only if every extension call here reproduces the "
            "hash of the request that was sent"),
        "not_accounted_for": {"calls": len(bad), "unit_runs": len(units), "cells": len(aff)},
        "wrong_reconstructions_that_chars_after_would_have_passed": sum(
            1 for r in seed_rows if not r["sha_match"] and r["chars_match"]),
        "persisted": "sent_input_verification.csv holds the per-cell aggregate (every call "
                     "counted); sent_input_mismatches.csv holds every non-reproducing row. The "
                     "1.9M-row per-call table is regenerated by re-running this script.",
        "by_arm_protocol": {"%s|%s|%s" % (k[0], k[1], m):
                            {"calls": len(v[m]), "reproduced": rate(v[m])[0]}
                            for k, v in inter.items() for m in v},
    }
    (OUT / "sent_input_verification.json").write_text(
        json.dumps(blob, indent=1, default=str) + "\n", encoding="utf-8")
    print("\n-> %s (%s per-cell rows aggregating %s calls)"
          % (OUT / "sent_input_verification.csv", f"{len(agg):,}", f"{len(rows):,}"))
    print("-> %s (%s non-reproducing calls)"
          % (OUT / "sent_input_mismatches.csv", f"{sum(1 for r in rows if not r['sha_match']):,}"))
    print("   sha256 %s" % hashlib.sha256(
        (OUT / "sent_input_verification.json").read_bytes()).hexdigest()[:32])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
