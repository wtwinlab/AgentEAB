#!/usr/bin/env python3
"""The B-group input inventory: what enters the analysis, from which root, and why it is valid.

    python3 build_input_manifest.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import inputs as IN                                                      # noqa: E402

OUT = IN.OUT
#: the designed panel per protocol, identical in both designs: homogeneous protocols block on
#: backbone, heterogeneous ones run a frozen roster that must not vary or the condition confounds with it
PANELS = {"du_mad": ["llama3-8b", "qwen2.5-7b"], "madm2_objective": ["llama3-8b", "qwen2.5-7b"],
          "bccs": ["llama3-8b", "qwen2.5-7b"], "ford": ["roster"], "moa": ["roster"]}
REPEATS = (1, 2)


def frozen_units(benchmark: str) -> Tuple[set, str]:
    """The frozen unit set for a benchmark, with the sample hash each design already recorded.
    """
    src = IN.source_for(benchmark)
    if src.tag == "ext":
        import ext_design as D
        f = D.frozen(benchmark)
        return set(f["unit_ids"]), f["sha256_16"]
    import collect_controlled as CC
    ids, sha = CC.frozen_units(benchmark)
    return set(ids), sha


def arm_behaviour(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Did the condition do what its name says? Read from the traces, not assumed from the label."""
    peers = extract_fail = kept_strict = kept_total = 0
    traces = 0
    for r in rows:
        for t in r.get("arm_traces") or []:
            traces += 1
            peers += t.get("n_peers_rendered") or 0
            extract_fail += t.get("ansonly_extract_failed") or 0
            found = t.get("peer_blocks_found") or 0
            kept = len(t.get("peer_indices_kept") or [])
            if found:
                kept_total += 1
                kept_strict += 1 if kept < found else 0
    return {"traces": traces, "peers_rendered_total": peers,
            "ansonly_extract_failed": extract_fail,
            "traces_with_peers": kept_total,
            "traces_keeping_strict_subset": kept_strict}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(IN.ext_gate(require=True), "\n")
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    coord_seen: Dict[Tuple, int] = collections.Counter()
    bench_roots: Dict[str, set] = collections.defaultdict(set)

    for bench in IN.BENCHMARKS:
        src = IN.source_for(bench)
        units, uhash = frozen_units(bench)
        recs, dropped = IN.read(bench, apply_transform=False)
        bench_roots[bench].add(src.tag)
        by_cell: Dict[Tuple, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in recs:
            by_cell[(r["protocol"], r["arm"], r["backbone"])].append(r)
            coord_seen[(src.tag, bench, r["protocol"], r["arm"], r["backbone"], r["repeat"],
                        r["unit"])] += 1
        print("%-22s %-5s %6s records  %3d cells  frozen units %-5d (sha %s)"
              % (bench, src.tag, f"{len(recs):,}", len(by_cell), len(units), uhash))
        if dropped:
            tot = sum(dropped.values())
            print("    excluded: %s line(s) -- %s" % (f"{tot:,}", "; ".join(
                "%s x%d" % (k[:72], v) for k, v in sorted(dropped.items())[:4])))

        for (proto, arm, bb), rs in sorted(by_cell.items()):
            uset = {r["unit"] for r in rs}
            outside = uset - units
            reps = sorted({r["repeat"] for r in rs})
            io_bad = sum(1 for r in rs if (r.get("raw_io") or "preserved") != "preserved")
            beh = arm_behaviour(rs)
            if bb not in PANELS.get(proto, []):
                problems.append("%s/%s/%s: backbone %r is not in the designed panel %s"
                                % (bench, proto, arm, bb, PANELS.get(proto)))
            if outside:
                problems.append("%s/%s/%s/%s: %d unit(s) outside the frozen sample"
                                % (bench, proto, arm, bb, len(outside)))
            if reps != list(REPEATS):
                problems.append("%s/%s/%s/%s: repeats %s, designed %s"
                                % (bench, proto, arm, bb, reps, list(REPEATS)))
            if io_bad:
                problems.append("%s/%s/%s/%s: %d record(s) without preserved raw I/O"
                                % (bench, proto, arm, bb, io_bad))
            if arm == "NOPEER" and beh["peers_rendered_total"]:
                problems.append("%s/%s/%s: NOPEER rendered %d peers"
                                % (bench, proto, bb, beh["peers_rendered_total"]))
            rows.append({
                "source": src.tag, "benchmark": bench, "metric": IN.METRIC[bench],
                "protocol": proto, "arm": arm, "backbone": bb,
                "records": len(rs), "distinct_units": len(uset),
                "frozen_units": len(units), "frozen_unit_sha16": uhash,
                "repeats": ",".join(map(str, reps)),
                "units_outside_frozen": len(outside),
                "raw_io_not_preserved": io_bad,
                "code_versions": ",".join(sorted({str(r.get("code_version")) for r in rs})),
                "reused_from_group_a": sum(1 for r in rs if r.get("reused")),
                "traces": beh["traces"], "peers_rendered_total": beh["peers_rendered_total"],
                "ansonly_extract_failed": beh["ansonly_extract_failed"],
                "traces_with_peers": beh["traces_with_peers"],
                "traces_keeping_strict_subset": beh["traces_keeping_strict_subset"],
            })

    dupes = {k: v for k, v in coord_seen.items() if v > 1}
    if dupes:
        problems.append("%d coordinate(s) hold more than one admissible record -- that is double "
                        "counting; first: %s" % (len(dupes), next(iter(dupes))))
    cross = {b: r for b, r in bench_roots.items() if len(r) > 1}
    if cross:
        problems.append("benchmark(s) served by more than one root, which risks pooling two "
                        "collections: %s" % cross)

    print("\n=== coverage by source ===")
    print("%-5s %8s %8s %8s %10s" % ("source", "benchs", "cells", "records", "coordinates"))
    for tag in sorted({r["source"] for r in rows}):
        sel = [r for r in rows if r["source"] == tag]
        print("%-5s %8d %8d %8s %10s"
              % (tag, len({r["benchmark"] for r in sel}), len(sel),
                 f"{sum(r['records'] for r in sel):,}",
                 f"{sum(1 for k in coord_seen if k[0] == tag):,}"))
    print("%-5s %8d %8d %8s %10s"
          % ("ALL", len({r["benchmark"] for r in rows}), len(rows),
             f"{sum(r['records'] for r in rows):,}", f"{len(coord_seen):,}"))

    print("\n=== the three outcome metrics, which are never pooled ===")
    for m in sorted({r["metric"] for r in rows}):
        bs = sorted({r["benchmark"] for r in rows if r["metric"] == m})
        sizes = {b: next(r["frozen_units"] for r in rows if r["benchmark"] == b) for b in bs}
        print("   %-24s %s" % (m, ", ".join("%s (%d frozen)" % (b, sizes[b]) for b in bs)))

    print("\n=== integrity ===")
    if problems:
        print("   %d problem(s):" % len(problems))
        for p in problems[:30]:
            print("     - %s" % p)
    else:
        print("   no problems: one admissible record per coordinate, every unit inside its frozen")
        print("   sample, designed panels and repeats present, raw I/O preserved on every record,")
        print("   NOPEER rendering zero peers everywhere, and no benchmark served by two roots.")

    with open(OUT / "input_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    blob = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol_families": IN.PROTOS, "benchmarks": IN.BENCHMARKS,
            "metric_by_benchmark": IN.METRIC,
            "sources": {t: {"root": str(s.root.relative_to(ROOT)),
                            "benchmarks": list(s.benchmarks),
                            "collection_digest": s.collection_digest, "note": s.note}
                        for t, s in IN.SOURCES.items()},
            "ext_digest_admitted_because": IN.ext_gate(require=True),
            "coordinates": len(coord_seen),
            "records": sum(r["records"] for r in rows),
            "cells": len(rows), "problems": problems, "rows": rows}
    (OUT / "input_manifest.json").write_text(json.dumps(blob, indent=1, default=str) + "\n",
                                             encoding="utf-8")
    print("\n-> %s" % (OUT / "input_manifest.csv"))
    print("   sha256 %s" % hashlib.sha256((OUT / "input_manifest.csv").read_bytes()
                                          ).hexdigest()[:32])
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
