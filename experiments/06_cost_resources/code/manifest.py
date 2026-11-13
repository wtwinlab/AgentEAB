r"""V.4: a manifest of every artifact, an independent backup, and a hash check of that backup.

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/manifest.py --backup ~/dataeng_backup_2026-10-13

The manifest records, per file, its size, line count, SHA-256, and what each record holds. The
backup is written outside the repository tree and then verified by re-hashing what landed there, not
by trusting the copy. A backup nobody re-read is a belief, not a backup.

What was and was not captured per call
--------------------------------------
Every record keeps the full prompt hash, the complete raw reply, token counts, the finish reason,
and the cost. Token log-probabilities were **not** requested from the server, so they are absent for
matching, sharing and SQL. The prompts and replies are retained verbatim, which is what teacher-
forced rescoring needs: the same checkpoints can be re-served and the stored reply scored under them
offline. That is a re-measurement, not a recovery of the original decode, and it is labelled as one.
ThriftLLM's own likelihood table is a separate object and it is present, because their aggregation
needs it; it holds class scores over the declared vocabulary, not raw token log-probabilities.

Generated against sourced
-------------------------
Files are classified before anything is copied. **Generated** files are this round's model outputs,
analyses, code and logs: losing them means re-running 266k model calls, so they are copied and
re-hashed. **Sourced** files are the benchmarks and upstream checkouts, which are recoverable from
their pinned origins; they are hashed and their provenance recorded, but not duplicated. Both classes
appear in the manifest with their hashes, so the sourced half can be checked for drift later.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: str) -> Dict[str, Any]:
    d: Dict[str, Any] = {"bytes": os.path.getsize(path), "sha256": sha256_file(path)}
    if path.endswith(".jsonl"):
        n = 0
        keys: set = set()
        st = collections.Counter()
        with open(path, encoding="utf-8") as f:
            for line in f:
                n += 1
                try:
                    r = json.loads(line)
                except Exception:
                    st["unparsable_line"] += 1
                    continue
                st[r.get("status", "no_status")] += 1
                if n <= 200:
                    keys |= set(r)
        d.update({"records": n, "status": dict(st), "fields": sorted(keys)})
        d["has_raw_reply"] = "raw_reply" in keys
        d["has_token_logprobs"] = any(k in keys for k in ("logprobs", "top_logprobs"))
        d["has_token_counts"] = "prompt_tokens" in keys or "completion_tokens" in keys
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", default="")
    a = ap.parse_args()

    man: Dict[str, Any] = {
        "root": BASE,
        "retention": {
            "prompt": "sha-256 of the exact prompt, plus its character length",
            "reply": "complete raw reply text, verbatim",
            "tokens": "prompt and completion token counts, and the finish reason",
            "token_logprobs": ("not requested from the server; absent for matching, sharing and "
                               "SQL"),
            "offline_rescoring": ("possible without new generation: the same checkpoints can be "
                                  "re-served and the stored reply scored teacher-forced. This "
                                  "measures the retained text under the model, and is not the "
                                  "original decode's log-probabilities"),
            "thrift_likelihood_table": ("present as a separate artifact; class scores over the "
                                        "declared vocabulary, which is what their aggregation "
                                        "consumes, not raw token log-probabilities"),
        },
        "files": {}}

    for sub in ("raw", "analysis", "code", "logs", "audit"):
        p = os.path.join(BASE, sub)
        if not os.path.isdir(p):
            continue
        for dirpath, _dirs, names in os.walk(p):
            for nm in sorted(names):
                if nm.startswith("."):
                    continue
                fp = os.path.join(dirpath, nm)
                rel = os.path.relpath(fp, BASE)
                d = describe(fp)
                d["klass"] = "sourced" if rel.startswith(("raw/em/", "raw/sql/")) else "generated"
                man["files"][rel] = d
    for nm in ("EXECUTION_CHECKLIST.md", "FREEZE_MATCHING.md"):
        fp = os.path.join(BASE, nm)
        if os.path.exists(fp):
            man["files"][nm] = {**describe(fp), "klass": "generated"}

    gen = {k: v for k, v in man["files"].items() if v["klass"] == "generated"}
    src_ = {k: v for k, v in man["files"].items() if v["klass"] == "sourced"}
    man["classes"] = {
        "generated": ("this round's model outputs, analyses, code and logs; losing these means "
                      "re-running the collection"),
        "sourced": ("benchmark data and upstream checkouts, recoverable from their pinned origins; "
                    "hashed for drift detection, not duplicated into the backup"),
    }
    for nm, grp in (("generated", gen), ("sourced", src_)):
        b = sum(v["bytes"] for v in grp.values())
        man["totals"] = man.get("totals", {})
        man["totals"][nm] = {"files": len(grp), "bytes": b, "gib": round(b / 2**30, 3),
                             "jsonl_records": sum(v.get("records", 0) for v in grp.values())}
        print(f"  {nm:10s} {len(grp):5d} files  {b/2**30:7.3f} GiB  "
              f"{sum(v.get('records', 0) for v in grp.values()):,} JSONL records")

    lp = [k for k, v in gen.items() if v.get("has_token_logprobs")]
    rr = [k for k, v in gen.items() if v.get("has_raw_reply")]
    man["totals"]["generated_files_with_token_logprobs"] = len(lp)
    man["totals"]["generated_files_with_verbatim_replies"] = len(rr)
    print(f"  files carrying token log-probabilities: {len(lp)}; "
          f"files carrying verbatim replies: {len(rr)}")
    print("  (prompts and replies are retained verbatim, so offline teacher-forced rescoring "
          "needs no new generation)")

    if a.backup:
        dst = os.path.expanduser(a.backup)
        print(f"\n  backing up the generated class to {dst}")
        os.makedirs(dst, exist_ok=True)
        copied = verified = mismatched = 0
        bad: List[str] = []
        for rel, meta in gen.items():
            src = os.path.join(BASE, rel)
            out = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(src, out)
            copied += 1
            # verify by re-hashing what actually landed, not by trusting the copy
            if sha256_file(out) == meta["sha256"]:
                verified += 1
            else:
                mismatched += 1
                bad.append(rel)
        man["backup"] = {"path": dst, "scope": "generated only", "copied": copied,
                         "hash_verified": verified, "mismatched": mismatched,
                         "mismatched_files": bad,
                         "sourced_recovery": ("benchmarks re-fetchable from their pinned origins; "
                                              "hashes in this manifest detect drift")}
        print(f"  copied {copied}, hash-verified by re-reading the copy {verified}, "
              f"mismatched {mismatched}")
        if bad:
            print(f"  MISMATCH: {bad}")
        with open(os.path.join(dst, "MANIFEST.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2, sort_keys=True)

    mp = os.path.join(BASE, "MANIFEST.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    print(f"\n  wrote {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
