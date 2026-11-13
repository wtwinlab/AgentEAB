#!/usr/bin/env python3
"""Check the paper's printed values against this repository.

    python3 verify.py               # shipped outputs against the printed values
    python3 verify.py --recompute   # also run every paper object's computation again (about 1 minute)
    python3 verify.py --list        # what is checked, without running
    python3 verify.py --recompute --json results.json   # also write the per-value results

The check reads only files inside the repository, makes no model call, opens no database and needs
no GPU.

1. Paper objects (`paper_objects/`). Every value listed in an object's `object.json` (the number as
   printed in the paper, with the section or table that prints it) must equal the object's computed
   value at the printed precision; every listed statement (an interval excludes 0 or not) must hold.
   With `--recompute`, each object's computation is rerun from its `inputs/` and must reproduce the
   shipped `output/values.json`.
2. Coverage. `PAPER_MAP.md` points to every paper-object directory and lists every table and
   figure label a paper object names.
3. The supplement (`paper/supplementary.pdf`) is present.
4. Repository invariants. No file names a path outside the repository, no symlink, no file over
   50 MB.

`--json FILE` writes every check and every printed value with its computed value to FILE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

sys.dont_write_bytecode = True
ART = pathlib.Path(__file__).resolve().parent
PO = ART / "paper_objects"
PAPER = ART / "paper"
sys.path.insert(0, str(PO))
import check_objects as CO                                              # noqa: E402

checks: list[dict] = []


def ck(name: str, passed: bool, detail: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def sha_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default="", help="also write every check and printed value to this file")
    a = ap.parse_args()

    if a.list:
        print("paper objects:")
        for od in CO.objects():
            meta = json.loads((od / "object.json").read_text(encoding="utf-8"))
            print("  %-34s %3d printed values  %s" % (od.name, len(meta["values"]), ", ".join(meta["labels"])))
        print("coverage of PAPER_MAP.md; supplement present; repository invariants")
        return 0

    # ------------------------------------------------------------------ 1. paper objects
    results: list = []
    for od in CO.objects():
        k, n = CO.run_object(od, a.recompute, results)
        meta = json.loads((od / "object.json").read_text(encoding="utf-8"))
        bad = [r for r in results if r["object"] == od.name and not r["pass"]]
        ck("%s (%s): %d of %d checks pass%s"
           % (od.name, ", ".join(meta["labels"]), k, n, ", recomputation included" if a.recompute else ""),
           k == n, "; ".join("%s %s: %s" % (r.get("key", r.get("check")), r.get("printed", ""), r["detail"])
                             for r in bad[:4]))

    # ------------------------------------------------------------------ 2. coverage
    pm = (ART / "PAPER_MAP.md").read_text(encoding="utf-8")
    in_map = set(re.findall(r"`((?:tab|fig):[^`]+)`", pm))
    obj_labels = set()
    for od in CO.objects():
        meta = json.loads((od / "object.json").read_text(encoding="utf-8"))
        obj_labels |= {l for l in meta["labels"] if l.startswith(("tab:", "fig:"))}
    unknown = sorted(obj_labels - in_map)
    ck("Every table or figure label a paper object names is listed in PAPER_MAP.md", not unknown, str(unknown[:8]))
    not_mapped = sorted(od.name for od in CO.objects() if od.name not in pm)
    ck("PAPER_MAP.md points to every paper-object directory", not not_mapped, str(not_mapped))

    # ------------------------------------------------------------------ 3. supplement
    ck("The supplement is present (paper/supplementary.pdf)", (PAPER / "supplementary.pdf").is_file())

    # ------------------------------------------------------------------ 4. invariants
    OUTSIDE = (re.compile(r"/Users/"), re.compile(r"/mnt/data"), re.compile(r"/home/[a-z]"),
               re.compile(r"(?<![A-Za-z0-9_.])/tmp/"))
    KEEP_SUFFIX = {".py", ".sh", ".md", ".json", ".txt", ".csv", ".tex", ".cfg", ".toml", ".yaml"}
    bad_abs = []
    for p in ART.rglob("*"):
        if p.is_dir() or p.suffix not in KEEP_SUFFIX:
            continue
        rel = p.relative_to(ART)
        # this checker holds the literals as search patterns
        if ".git" in rel.parts or rel.name == "verify.py":
            continue
        if p.stat().st_size > 50_000_000:
            continue
        try:
            body = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pat in OUTSIDE:
            m = pat.search(body)
            if m:
                bad_abs.append("%s: %s" % (rel, body[m.start():m.start() + 40].split()[0]))
                break
    ck("Nothing in the package names a path outside it", not bad_abs, "; ".join(bad_abs[:4]))
    ck("No symlink in the delivered tree", not [p for p in ART.rglob("*") if p.is_symlink() and ".git" not in p.parts])
    big = [str(p.relative_to(ART)) for p in ART.rglob("*") if p.is_file() and ".git" not in p.parts
           and p.stat().st_size > 50_000_000]
    ck("No file over 50 MB", not big, str(big[:6]))

    width = max(len(c["name"]) for c in checks)
    for c in checks:
        print("%-*s %s%s" % (min(width, 110), c["name"][:110], "PASS" if c["passed"] else "FAIL",
                             ("  (%s)" % c["detail"]) if c["detail"] and not c["passed"] else ""))
    n_ok = sum(1 for c in checks if c["passed"])
    n_vals = sum(1 for r in results if "key" in r)
    n_vals_ok = sum(1 for r in results if "key" in r and r["pass"])
    print("\n%d of %d checks passed; %d of %d printed values reproduce" % (n_ok, len(checks), n_vals_ok, n_vals))
    if a.json:
        pathlib.Path(a.json).write_text(
            json.dumps({"checks": checks, "passed": n_ok, "total": len(checks),
                        "printed_values": {"passed": n_vals_ok, "total": n_vals},
                        "recompute": a.recompute,
                        "values": results}, indent=1) + "\n", encoding="utf-8")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
