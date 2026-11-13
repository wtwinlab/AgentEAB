#!/usr/bin/env python3
"""Snapshot the reused source files into this run, with hashes, and publish a READY marker.

    python3 snapshot_reused.py --plan      # what would be copied, and why
    python3 snapshot_reused.py --snapshot  # copy, verify, then publish
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SOURCE_RUN = "protocol_all_matching_20261018T192619Z"
SRC_ROOT = ROOT / "results" / SOURCE_RUN
OUT = ROOT / "workspace/controlled_sharing"
SNAP = OUT / "reused_sources"
READY = SNAP / "READY.json"
FILE_MANIFEST = SNAP / "FILE_MANIFEST.jsonl"
#: the five families whose FULL condition is reused, plus `madm2_naive`.
PROTOS = ["ford", "moa", "du_mad", "madm2_objective", "madm2_naive", "bccs"]
MATCHING = ("wdc_products", "geographic_settlements")
#: files without which the copied shards cannot be interpreted
SUPPORT = [
    "analysis/FROZEN_SAMPLE_5X5.json",
    "GATE.json",
    "TERMINAL_STATE.md",
    "FINAL_REPORT_ZH.md",
]


def sha256_file(p: pathlib.Path, chunk: int = 1 << 22) -> Tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def writers_holding(paths: List[pathlib.Path]) -> List[str]:
    """Any process with these files open for writing. A file still being appended is not a snapshot.
    """
    held = []
    for p in paths:
        try:
            out = subprocess.run(["lsof", "-F", "an", "--", str(p)], capture_output=True,
                                 text=True, timeout=30).stdout
        except Exception:
            return ["lsof unavailable: cannot prove the source is closed"]
        for line in out.splitlines():
            if line.startswith("a") and any(m in line for m in ("w", "u")):
                held.append(p.name)
                break
    return held


def wanted_files() -> List[Tuple[pathlib.Path, str]]:
    """[(absolute source path, relative path inside the snapshot)] -- shards plus support files."""
    out: List[Tuple[pathlib.Path, str]] = []
    for bench in MATCHING:
        for proto in PROTOS:
            for f in sorted(glob.glob(str(SRC_ROOT / "raw" / f"{bench}__{proto}__*.jsonl"))):
                p = pathlib.Path(f)
                out.append((p, f"raw/{p.name}"))
    for rel in SUPPORT:
        p = SRC_ROOT / rel
        if p.exists():
            out.append((p, rel))
    # the population definitions the unit ids refer to.
    for bench in MATCHING:
        cand = (ROOT / "data/records/matching_protocol" /
                f"{bench}.json")
        if cand.exists():
            out.append((cand, f"population/{cand.name}"))
    return out


def copy_verified(src: pathlib.Path, dst: pathlib.Path) -> Dict[str, Any]:
    """Copy, then re-hash the copy from disk. Never overwrite a differing file in place."""
    src_sha, src_bytes = sha256_file(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    status = "copied"
    if dst.exists():
        dst_sha, _ = sha256_file(dst)
        if dst_sha == src_sha:
            return {"status": "already identical", "sha256": src_sha, "bytes": src_bytes,
                    "snapshot_path": str(dst.relative_to(OUT))}
        # a different file is already there: keep both, distinguish by the source hash
        dst = dst.with_name(f"{dst.stem}.src-{src_sha[:12]}{dst.suffix}")
        status = "copied under a distinguishing name (existing file differs, left untouched)"
        if dst.exists():
            dst_sha, _ = sha256_file(dst)
            if dst_sha == src_sha:
                return {"status": "already identical", "sha256": src_sha, "bytes": src_bytes,
                        "snapshot_path": str(dst.relative_to(OUT))}
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copyfile(src, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    got_sha, got_bytes = sha256_file(tmp)
    if got_sha != src_sha or got_bytes != src_bytes:
        tmp.unlink(missing_ok=True)
        raise IOError(f"copy of {src.name} does not verify: {got_sha} != {src_sha}")
    os.replace(tmp, dst)
    return {"status": status, "sha256": src_sha, "bytes": src_bytes,
            "snapshot_path": str(dst.relative_to(OUT))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--snapshot", action="store_true")
    a = ap.parse_args()

    files = wanted_files()
    total = sum(p.stat().st_size for p, _ in files)
    print(f"source run: {SOURCE_RUN}")
    print(f"files to snapshot: {len(files)}  ({total / 2**30:.2f} GiB)")
    by = collections.Counter()
    for p, rel in files:
        by[rel.split('/')[0]] += 1
    for k, v in sorted(by.items()):
        print(f"   {k:<12} {v}")

    # the source must be closed before anything is called a snapshot
    held = writers_holding([p for p, _ in files])
    newest = max(p.stat().st_mtime for p, _ in files)
    age_h = (time.time() - newest) / 3600.0
    print(f"\nnewest source mtime: {age_h:.1f} h ago")
    if held:
        print("REFUSING: these source files are open for writing: " + ", ".join(held[:5]))
        return 2
    print("no process holds any source file for writing -- the source is closed")

    if a.plan:
        print("\n--plan only; nothing copied.")
        return 0
    if not a.snapshot:
        ap.print_help()
        return 1

    SNAP.mkdir(parents=True, exist_ok=True)
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    # a stale READY must not survive a re-snapshot
    READY.unlink(missing_ok=True)

    lines, copied_bytes, t0 = [], 0, time.time()
    host = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
    for p, rel in files:
        dst = SNAP / SOURCE_RUN / rel
        info = copy_verified(p, dst)
        lines.append(json.dumps({
            "source_host": host, "source_path": str(p), "source_run": SOURCE_RUN,
            "source_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime(p.stat().st_mtime)),
            "snapshot_path": info["snapshot_path"], "bytes": info["bytes"],
            "sha256": info["sha256"], "status": info["status"],
            "copied_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))
        copied_bytes += info["bytes"]
        print(f"   {info['status']:<22} {rel}  {info['bytes'] / 2**20:8.1f} MiB  "
              f"{info['sha256'][:16]}")
    FILE_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # re-verify every copy from disk before publishing READY
    bad = []
    for line in FILE_MANIFEST.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        q = OUT / e["snapshot_path"]
        if not q.exists():
            bad.append(e["snapshot_path"] + " missing")
            continue
        sha, n = sha256_file(q)
        if sha != e["sha256"] or n != e["bytes"]:
            bad.append(e["snapshot_path"] + " hash mismatch")
    if bad:
        print("\nVERIFICATION FAILED, READY not published:")
        for b in bad[:10]:
            print("   " + b)
        return 3

    tmp = READY.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({
            "state": "ready",
            "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_run": SOURCE_RUN, "source_host": host,
            "files": len(lines), "bytes": copied_bytes,
            "file_manifest": str(FILE_MANIFEST.relative_to(OUT)),
            "record_manifest": "REUSE_MANIFEST.json",
            "verified": "every copy re-hashed from disk after writing",
            "source_closed": "no process held any source file for writing at snapshot time",
            "note": "byte copies, not links: ext4 here has no reflink support. The historical tree "
                    "was opened read-only and is unchanged.",
        }, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, READY)
    print(f"\nsnapshot verified and published: {len(lines)} files, "
          f"{copied_bytes / 2**30:.2f} GiB in {time.time() - t0:.0f}s")
    print(f"-> {SNAP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
