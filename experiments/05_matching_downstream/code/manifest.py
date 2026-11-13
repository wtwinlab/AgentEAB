r"""Manifest of generated outputs, plus an independent backup verified by re-reading it.

    python3 experiments/05_matching_downstream/code/manifest.py \
        --backup ~/downstream_replication_backup_2026-10-13

Re-runnable: run it again after writing more files and the manifest is regenerated to match.

Two classes. **Generated** is this round's model outputs, analyses, figures, code and documents;
losing them means re-running 113,256 model calls. **Sourced** is the Leipzig archive and its
extracted files, which are recoverable from a pinned URL and a recorded archive hash; they are hashed
for drift detection and not duplicated.

The backup is verified by re-hashing what landed in it, not by trusting the copy. It is written to
another directory on the **same host**, which protects against an accidental deletion inside the
repository and does **not** protect against loss of this machine or its disk. That distinction is
stated in the output rather than left for a reader to assume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))

SOURCED_PREFIXES = ("data/",)


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", default="")
    a = ap.parse_args()

    man: Dict[str, Any] = {
        "round": "downstream_replication_2026-10-13",
        "workload": "Geographic Settlements (Leipzig DBS), real multi-source, CC BY 4.0",
        "source_archive": {
            "url": "https://dbs.uni-leipzig.de/files/datasets/geographicalSettelments.zip",
            "sha256": "76a5b8d261607f6c77a9cd79eaf4b0bb60e8b4ebfad3802b4bd01a02d97cc02e"},
        "retention": {
            "prompt": "stored verbatim in `request.messages[0].content`, with its sha and length",
            "reply": "complete raw reply text",
            "decision": "parsed with parser v3; the ambiguity record and `v2_would_say` kept too",
            "logprobs": ("top-5 alternatives at each generated position, as returned by the "
                         "serving interface. NOT a distribution over the vocabulary; "
                         "`is_raw_logits` is false and no original decode is recoverable"),
            "tokens": "prompt and completion counts, finish reason, truncation flag, latency",
            "failures": "retained with their error text and never counted as abstentions"},
        "classes": {
            "generated": "this round's outputs; re-creating them means 113,256 model calls",
            "sourced": "the Leipzig archive and its extracted files, re-fetchable from the URL"},
        "files": {}}

    for dirpath, dirs, names in os.walk(BASE):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for nm in sorted(names):
            if nm.startswith("."):
                continue
            fp = os.path.join(dirpath, nm)
            rel = os.path.relpath(fp, BASE)
            if rel in ("MANIFEST.json",):
                continue
            d: Dict[str, Any] = {"bytes": os.path.getsize(fp), "sha256": sha256_file(fp)}
            d["klass"] = "sourced" if rel.startswith(SOURCED_PREFIXES) else "generated"
            if nm.endswith(".jsonl"):
                with open(fp, "rb") as f:
                    d["records"] = sum(1 for _ in f)
            if nm.endswith(".csv"):
                with open(fp, encoding="utf-8", errors="replace") as f:
                    d["rows"] = max(0, sum(1 for _ in f) - 1)
            man["files"][rel] = d

    gen = {k: v for k, v in man["files"].items() if v["klass"] == "generated"}
    src = {k: v for k, v in man["files"].items() if v["klass"] == "sourced"}
    man["totals"] = {
        "generated": {"files": len(gen), "bytes": sum(v["bytes"] for v in gen.values()),
                      "jsonl_records": sum(v.get("records", 0) for v in gen.values())},
        "sourced": {"files": len(src), "bytes": sum(v["bytes"] for v in src.values())}}
    for nm, t in man["totals"].items():
        print(f"  {nm:10s} {t['files']:4d} files  {t['bytes']/2**20:9.2f} MiB"
              + (f"  {t.get('jsonl_records', 0):,} JSONL records" if "jsonl_records" in t else ""))
    lp = [k for k, v in gen.items() if k.endswith(".jsonl")]
    print(f"  model-output files: {lp}")

    if a.backup:
        dst = os.path.expanduser(a.backup)
        print(f"\n  backing up the generated class to {dst}")
        os.makedirs(dst, exist_ok=True)
        copied = verified = bad = 0
        badf: List[str] = []
        for rel, meta in gen.items():
            out = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(out) or dst, exist_ok=True)
            shutil.copy2(os.path.join(BASE, rel), out)
            copied += 1
            if sha256_file(out) == meta["sha256"]:
                verified += 1
            else:
                bad += 1
                badf.append(rel)
        man["backup"] = {
            "path": dst, "scope": "generated only", "copied": copied,
            "hash_verified_by_rereading": verified, "mismatched": bad, "mismatched_files": badf,
            "kind": "same-host copy",
            "what_it_protects_against": ("an accidental delete or overwrite inside the repository "
                                         "working tree"),
            "what_it_does_NOT_protect_against": ("loss of this machine, its disk, or its "
                                                 "filesystem. This is not off-host disaster "
                                                 "recovery and must not be described as a backup "
                                                 "in that sense"),
            "sourced_recovery": ("the Leipzig archive is re-fetchable from the recorded URL and "
                                 "its sha256 is in this manifest, so the sourced class needs no "
                                 "copy")}
        print(f"  copied {copied}, hash-verified by re-reading the copy {verified}, "
              f"mismatched {bad}")
        if badf:
            print(f"  MISMATCH: {badf}")
        with open(os.path.join(dst, "MANIFEST.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2, sort_keys=True)

    mp = os.path.join(BASE, "MANIFEST.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    print(f"\n  wrote {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
