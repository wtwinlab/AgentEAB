#!/usr/bin/env python3
"""Import the already-collected FULL condition from the protocol matrix run. Zero model calls.

    python3 import_full_cache.py --check     # audit only: what qualifies and what does not
    python3 import_full_cache.py --import    # write the qualifying records into the collection
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SOURCE_RUN = "protocol_all_matching_20261018T192619Z"
SRC = ROOT / "workspace" / SOURCE_RUN / "raw"
OKMARK = '"status": "ok"'
#: served model name -> the frozen panel it is a replica of
CKPT = {"llama3-8b-instruct": "llama3-8b", "qwen2.5-7b-instruct": "qwen2.5-7b"}
#: the frozen rosters, which must match exactly
ROSTERS = {"ford": "llama3-8b-g1+qwen25-7b-g2",
           "moa": "llama3-8b-g1+qwen25-7b-g2+gemma4-g5"}
MATCHING = ("wdc_products", "geographic_settlements")


def qualify(rec: Dict[str, Any], protocol: str, frozen: set,
            ) -> Tuple[Optional[Dict[str, Any]], str]:
    """`(descriptor, reason)`. The descriptor is None when the record must not be reused."""
    parts = (rec.get("unit_id") or "").split("|")
    if len(parts) < 6:
        return None, "unit_id not parseable"
    unit = "|".join(parts[3:])
    if unit not in frozen:
        return None, "unit outside the frozen sample"
    if not rec.get("raw_io_preserved"):
        return None, "raw I/O not preserved"
    if rec.get("answer_kind") != "binary_match" or rec.get("benchmark") != "matching_pair":
        return None, "different task adaptation"
    if not rec.get("gold_withheld_from_prompt", False):
        return None, "gold-withholding not asserted"
    calls = (rec.get("record") or {}).get("calls") or []
    if not calls:
        return None, "no calls"
    if any(c.get("finish_reason") != "stop" for c in calls):
        return None, "a call was truncated, so the token cap was binding"
    served = {c.get("served_model") for c in calls}
    team = rec.get("team") or ""
    if protocol in ROSTERS:
        if team != ROSTERS[protocol]:
            return None, f"roster {team!r} is not the frozen roster"
        backbone = "roster"
    else:
        if len(served) != 1:
            return None, f"homogeneous protocol served {sorted(served)}"
        backbone = CKPT.get(next(iter(served)), "")
        if not backbone:
            return None, f"unknown checkpoint {sorted(served)}"
    rep = rec.get("repeat")
    if rep not in (0, 1):            # the design takes two repeats; the source has three
        return None, "repeat outside the design's two"
    return {"unit": unit, "backbone": backbone, "repeat": rep + 1,
            "source_unit_id": rec.get("unit_id"), "source_repeat": rep}, "ok"


def rebuild_trace(protocol: str, rec: Dict[str, Any], kind: str, task: str) -> List[Dict[str, Any]]:
    """The FULL-arm trace, recomputed from the preserved prompts with the live transform code."""
    import arm_transform as AT
    out = []
    for i, c in enumerate(rec.get("calls") or []):
        msgs = c.get("messages_sent") or []
        _new, tr = AT.transform(protocol, "FULL", msgs, kind=kind, benchmark=task, call_seq=i)
        tr["call_seq"] = i
        tr["chars_before"] = tr["chars_after"] = sum(len(m.get("content") or "") for m in msgs)
        tr["reconstructed_offline"] = True
        out.append(tr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--import", dest="do_import", action="store_true")
    a = ap.parse_args()

    import collect_controlled as CC
    import matching_tasks as MT
    MT.install()
    kind, task = MT.KIND, MT.BENCH

    frozen_by_bench = {b: set(CC.frozen_units(b)[0]) for b in MATCHING}
    sha_by_bench = {b: CC.frozen_units(b)[1] for b in MATCHING}
    done = CC.done_keys()
    stats = collections.Counter()
    rejects = collections.Counter()
    staged: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for bench in MATCHING:
        frozen = frozen_by_bench[bench]
        for protocol in CC.PROTOS:
            seen = set()
            for shard in sorted(glob.glob(str(SRC / ("%s__%s__*.jsonl" % (bench, protocol))))):
                for line in open(shard, encoding="utf-8"):
                    if OKMARK not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        rejects["unparseable line"] += 1
                        continue
                    desc, why = qualify(rec, protocol, frozen)
                    if desc is None:
                        rejects[why] += 1
                        continue
                    item = {"benchmark": bench, "population_sha": sha_by_bench[bench],
                            "unit": desc["unit"], "protocol": protocol, "arm": "FULL",
                            "backbone": desc["backbone"], "repeat": desc["repeat"],
                            "alias": CC.ALIAS[desc["backbone"]],
                            "max_tokens": CC.PROD_MAX_TOKENS, "cost": CC.COST[protocol]}
                    item["key"] = CC.cache_key(item)
                    item["seed"] = CC.item_seed(item)
                    if item["key"] in done or item["key"] in seen:
                        rejects["already present"] += 1
                        continue
                    seen.add(item["key"])
                    stats[(bench, protocol, desc["backbone"])] += 1
                    if a.do_import:
                        r = rec.get("record") or {}
                        calls = r.get("calls") or []
                        out = {k: item[k] for k in ("benchmark", "protocol", "arm", "backbone",
                                                    "repeat", "unit", "key", "seed",
                                                    "population_sha", "alias", "max_tokens")}
                        out.update(status="ok", record=r, config=dict(
                            __import__("collect_all_matching").SPECS[protocol]["kw"],
                            benchmark=bench, family="matching", alias=item["alias"],
                            max_tokens=CC.PROD_MAX_TOKENS, answer_kind=kind, task_name=task,
                            seed=item["seed"]),
                            arm_traces=rebuild_trace(protocol, r, kind, task),
                            n_calls=len(calls),
                            completion_tokens=sum(c.get("completion_tokens") or 0 for c in calls),
                            prompt_tokens=sum(c.get("prompt_tokens") or 0 for c in calls),
                            raw_io="preserved",
                            stop_reason=r.get("native_stop_reason"),
                            code_version=CC.CODE_VERSION,
                            reused=True, source_run=SOURCE_RUN,
                            source_shard=pathlib.Path(shard).name,
                            source_unit_id=desc["source_unit_id"],
                            source_repeat=desc["source_repeat"])
                        staged.append((item, out))

    total = sum(stats.values())
    print("source run: %s" % SOURCE_RUN)
    print("qualifying FULL records not already on disk: %s\n" % f"{total:,}")
    print("%-24s %-18s %-12s %8s" % ("benchmark", "protocol", "backbone", "records"))
    for (b, p, bb), n in sorted(stats.items()):
        print("%-24s %-18s %-12s %8s" % (b, p, bb, f"{n:,}"))
    print("\nrejected, by reason:")
    for why, n in rejects.most_common():
        print("   %-52s %s" % (why, f"{n:,}"))

    if not a.do_import:
        print("\n--check only; nothing written. Re-run with --import to write them.")
        return 0

    writer = CC.Writer()
    tok = 0
    try:
        for item, out in staged:
            writer.append(item, out)
            tok += out["completion_tokens"]
    finally:
        writer.close()
    print("\nimported %s records, %.1f M output tokens that do not need re-collection"
          % (f"{len(staged):,}", tok / 1e6))
    after = CC.done_keys()
    print("done-set: %s -> %s keys" % (f"{len(done):,}", f"{len(after):,}"))
    for b in MATCHING:
        okc, have, need = CC.design_complete(b, after)
        print("  %-24s %s/%s unit-runs on disk%s"
              % (b, f"{have:,}", f"{need:,}", "  COMPLETE" if okc else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
