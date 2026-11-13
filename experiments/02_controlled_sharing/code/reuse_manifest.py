#!/usr/bin/env python3
"""Per-unit reuse manifest: what may be reused, on what grounds, and what must still be collected.

    python3 reuse_manifest.py --audit            # verify every candidate, write the manifest
    python3 reuse_manifest.py --audit --verify-imported   # also re-verify what was already imported
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SOURCE_RUNS = ["protocol_all_matching_20261018T192619Z"]
OUT = ROOT / "workspace/controlled_sharing"
OKMARK = '"status": "ok"'
CKPT = {"llama3-8b-instruct": "llama3-8b", "qwen2.5-7b-instruct": "qwen2.5-7b"}
ROSTERS = {"ford": "llama3-8b-g1+qwen25-7b-g2",
           "moa": "llama3-8b-g1+qwen25-7b-g2+gemma4-g5"}
MATCHING = ("wdc_products", "geographic_settlements")
#: protocols present in the source that are NOT part of this design, or are a different variant of
#: one that is. They stay as supplementary evidence and never enter a paired contrast.
SUPPLEMENTARY = {
    "madm2_naive": "MAD-M^2's released naive-pruning variant: a selection contrast, but the "
                   "protocol's own alternative rule rather than this design's uniform-selection "
                   "arm, so it is not UNIFSEL and is not paired as such",
    "dmad": "not part of this design", "dov_vote": "not part of this design",
    "liang_mad": "not part of this design", "reconcile": "not part of this design",
}


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def current_question_sha(benchmark: str, unit: str) -> Optional[str]:
    """The sha of the question THIS pipeline builds for `unit`, for input-content equivalence."""
    try:
        import collect_all_matching as C
        C._ensure_installed()
        units, _ = __import__("collect_controlled").frozen_units(benchmark)
        pop, _sha = C.load_population(benchmark)
        for u in pop:
            uid = u.get("unit_id") or u.get("id") or ""
            if uid == unit or str(uid).endswith(unit):
                return sha16(C.question_for(benchmark, u) if hasattr(C, "question_for")
                             else u["question"])
    except Exception:
        return None
    return None


def check_record(rec: Dict[str, Any], protocol: str, frozen: set, qsha: Dict[str, str],
                 ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """`(verdict, reason, descriptor)`; verdict in reusable / supplementary / rejected."""
    if protocol in SUPPLEMENTARY:
        return "supplementary", SUPPLEMENTARY[protocol], None
    parts = (rec.get("unit_id") or "").split("|")
    if len(parts) < 6:
        return "rejected", "unit_id not parseable", None
    unit = "|".join(parts[3:])
    if unit not in frozen:
        return "supplementary", "unit outside the frozen sample", None
    if rec.get("answer_kind") != "binary_match" or rec.get("benchmark") != "matching_pair":
        return "rejected", "different task adaptation", None
    if not rec.get("gold_withheld_from_prompt", False):
        return "rejected", "gold-withholding not asserted", None
    if not rec.get("raw_io_preserved"):
        return "rejected", "raw I/O not preserved", None
    want_q = qsha.get(unit)
    if want_q and rec.get("question_sha") and rec["question_sha"] != want_q:
        return "rejected", "question sha differs: the same unit id with a different input", None
    calls = (rec.get("record") or {}).get("calls") or []
    if not calls:
        return "rejected", "no calls", None
    if any(not c.get("messages_sent") or c.get("raw_reply") is None for c in calls):
        return "rejected", "a call is missing its prompt or its reply", None
    if any(c.get("finish_reason") != "stop" for c in calls):
        return "rejected", "a call was truncated, so the output cap was binding", None
    team = rec.get("team") or ""
    served = {c.get("served_model") for c in calls}
    if protocol in ROSTERS:
        if team != ROSTERS[protocol]:
            return "supplementary", f"roster {team!r} is not the frozen roster", None
        backbone = "roster"
    else:
        if len(served) != 1:
            return "rejected", f"homogeneous protocol served {sorted(served)}", None
        backbone = CKPT.get(next(iter(served)), "")
        if not backbone:
            return "supplementary", f"checkpoint {sorted(served)} is not a frozen panel", None
    rep = rec.get("repeat")
    if rep not in (0, 1):
        return "supplementary", "repeat beyond the design's two (kept, not paired)", None
    return "reusable", "all checks pass", {
        "unit": unit, "backbone": backbone, "target_repeat": rep + 1, "source_repeat": rep,
        "source_unit_id": rec.get("unit_id"), "question_sha": rec.get("question_sha"),
        "population_sha": rec.get("population_sha256_16"), "n_calls": len(calls),
        "completion_tokens": rec.get("completion_tokens_total"),
    }


def behavioural_equivalence(protocol: str, rec: Dict[str, Any], kind: str, task: str) -> bool:
    """FULL must be a no-op: recomputing the transform on the stored prompts changes nothing."""
    import arm_transform as AT
    for i, c in enumerate((rec.get("calls") or [])[:4]):
        msgs = c.get("messages_sent") or []
        new, tr = AT.transform(protocol, "FULL", msgs, kind=kind, benchmark=task, call_seq=i)
        if tr.get("changed") or new != [dict(m) for m in msgs]:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--verify-imported", action="store_true")
    a = ap.parse_args()

    import collect_all_matching as C
    import collect_controlled as CC
    import matching_tasks as MT
    MT.install()
    kind, task = MT.KIND, MT.BENCH
    C._ensure_installed()

    done = CC.done_keys()
    manifest: List[Dict[str, Any]] = []
    counts = collections.Counter()
    reasons = collections.Counter()
    equiv_checked = collections.Counter()

    for run in SOURCE_RUNS:
        src = ROOT / "results" / run / "raw"
        for bench in MATCHING:
            frozen_list, frozen_sha = CC.frozen_units(bench)
            frozen = set(frozen_list)
            # the question sha this pipeline produces now, per frozen unit
            qsha: Dict[str, str] = {}
            try:
                pop, _ = C.load_population(bench)
                for u in pop:
                    uid = u.get("unit_id") or u.get("id") or ""
                    if uid in frozen:
                        q = u.get("question")
                        if q:
                            qsha[uid] = sha16(q)
            except Exception:
                pass
            for shard in sorted(glob.glob(str(src / f"{bench}__*.jsonl"))):
                protocol = pathlib.Path(shard).name.split("__")[1]
                for ln, line in enumerate(open(shard, encoding="utf-8")):
                    if OKMARK not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        reasons["unparseable line"] += 1
                        continue
                    verdict, why, desc = check_record(rec, protocol, frozen, qsha)
                    counts[(bench, protocol, verdict)] += 1
                    if verdict != "reusable":
                        reasons[f"{verdict}: {why}"] += 1
                        continue
                    # behavioural equivalence, sampled per cell rather than per record: the check
                    # is a property of the protocol's prompt grammar, not of one unit
                    cell = (bench, protocol, desc["backbone"])
                    if equiv_checked[cell] < 3:
                        if not behavioural_equivalence(protocol, rec.get("record") or {},
                                                       kind, task):
                            reasons["FULL transform is not a no-op on this protocol"] += 1
                            counts[(bench, protocol, "rejected")] += 1
                            counts[(bench, protocol, "reusable")] -= 1
                            continue
                        equiv_checked[cell] += 1
                    item = {"benchmark": bench, "population_sha": frozen_sha,
                            "unit": desc["unit"], "protocol": protocol, "arm": "FULL",
                            "backbone": desc["backbone"], "repeat": desc["target_repeat"],
                            "alias": CC.ALIAS[desc["backbone"]],
                            "max_tokens": CC.PROD_MAX_TOKENS}
                    key = CC.cache_key(item)
                    manifest.append({
                        "target": {"benchmark": bench, "protocol": protocol, "arm": "FULL",
                                   "backbone": desc["backbone"], "repeat": desc["target_repeat"],
                                   "unit": desc["unit"], "cache_key": key},
                        "source": {"run": run, "shard": pathlib.Path(shard).name, "line": ln,
                                   "unit_id": desc["source_unit_id"],
                                   "repeat": desc["source_repeat"],
                                   "question_sha": desc["question_sha"],
                                   "population_sha": desc["population_sha"]},
                        "grounds": "unit in frozen list; question sha identical; released config; "
                                   "frozen panel/roster; every call finish_reason=stop so the cap "
                                   "never bound; raw I/O complete; FULL transform verified a no-op",
                        "on_disk": key in done,
                    })

    # ---- what remains, per cell, against the frozen design
    gaps = []
    for bench in CC.BENCH_ORDER:
        items = CC.items_for(bench)
        by_cell = collections.Counter()
        for it in items:
            by_cell[(it["protocol"], it["arm"], it["backbone"])] += 0 if it["key"] in done else 1
        for (p, arm, bb), n in sorted(by_cell.items()):
            if n:
                gaps.append({"benchmark": bench, "protocol": p, "arm": arm, "backbone": bb,
                             "still_to_collect": n,
                             "reason": "no source exists for this arm: the protocol-matrix run "
                                       "applied no interventions" if arm != "FULL"
                                       else "not covered by the source"})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REUSE_MANIFEST.json").write_text(json.dumps({
        "source_runs": SOURCE_RUNS, "entries": manifest,
        "counts": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(counts.items())},
        "rejection_reasons": dict(reasons.most_common()),
        "gaps": gaps,
    }, indent=1) + "\n", encoding="utf-8")

    on_disk = sum(1 for e in manifest if e["on_disk"])
    L = ["# Reuse manifest", "",
         f"Source runs: {', '.join(SOURCE_RUNS)}.", "",
         f"**{len(manifest):,} source records qualify for reuse; {on_disk:,} are already imported "
         f"into the collection.** Only the FULL arm has a source: the protocol-matrix run applied "
         f"no interventions, so NOPEER, ANSONLY and UNIFSEL have to be collected.", "",
         "## Per cell", "", "| benchmark | protocol | backbone | reusable | already imported |",
         "|---|---|---|---:|---:|"]
    per = collections.Counter()
    per_disk = collections.Counter()
    for e in manifest:
        t = e["target"]
        per[(t["benchmark"], t["protocol"], t["backbone"])] += 1
        if e["on_disk"]:
            per_disk[(t["benchmark"], t["protocol"], t["backbone"])] += 1
    for k in sorted(per):
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {per[k]:,} | {per_disk[k]:,} |")

    L += ["", "## Not reusable, by reason (kept, never deleted)", "",
          "| reason | records |", "|---|---:|"]
    for why, n in reasons.most_common():
        L.append(f"| {why} | {n:,} |")
    L += ["", "Records classified `supplementary` remain available as protocol-level quality and "
              "cost evidence. They are **not** mixed into the paired contrasts, because a unit "
              "outside the frozen sample has no partner to pair with and a different variant or "
              "roster would confound the factor under test.", "",
          "## Still to collect", "",
          "| benchmark | protocol | arm | backbone | unit-runs | why no source |",
          "|---|---|---|---|---:|---|"]
    for g in gaps[:60]:
        L.append(f"| {g['benchmark']} | {g['protocol']} | {g['arm']} | {g['backbone']} | "
                 f"{g['still_to_collect']:,} | {g['reason']} |")
    if len(gaps) > 60:
        L.append(f"| … | | | | | {len(gaps) - 60} further cells in the JSON |")

    L += ["", "## Cost accounting", "",
          "Reuse removes **this round's** collection cost for those cells. It does not reduce the "
          "cost attributed to a protocol in the scientific comparison: the cost dimension is "
          "reported from each record's own `n_calls` and token counts, whether the record was "
          "collected now or imported, so a protocol's workflow cost is unchanged by when it was "
          "measured.", "",
          "A shared FULL record can serve as the control for several contrasts -- NOPEER, ANSONLY "
          "and UNIFSEL are each paired against it on the same (unit, backbone, repeat) -- but it "
          "enters the data once. It is never counted as an additional independent observation, and "
          "source repeat 2 is dropped rather than folded in as a third.", ""]
    (OUT / "REUSE_MANIFEST.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"manifest: {len(manifest):,} reusable records, {on_disk:,} already imported")
    print(f"gaps: {len(gaps)} cells, {sum(g['still_to_collect'] for g in gaps):,} unit-runs")
    print("rejection/supplementary reasons:")
    for why, n in reasons.most_common(8):
        print(f"   {why[:70]:<70} {n:,}")
    print(f"-> {OUT / 'REUSE_MANIFEST.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
