#!/usr/bin/env python3
"""Check every paper object in this directory against the values printed in the paper.

    python3 check_objects.py                 # compare shipped outputs with the printed values
    python3 check_objects.py --recompute     # also run each object's computation again and compare
    python3 check_objects.py --only tab_factorial,fig_readout-cells
    python3 check_objects.py --json out.json # write the per-value results

Each subdirectory holds one paper object (or one table together with its full supplementary
version): `object.json`, `inputs/`, `code/`, `output/`. `object.json` lists the printed values;
`output/values.json` holds the computed values under the same keys. A printed value passes when

1. (when the paper's LaTeX source is placed under `paper/`) its printed form occurs in the named
   source file, after the paper's number macros are expanded, and
2. the computed value rounds to it at the printed number of decimals (or, for a value marked
   `"relation": "le"`, does not exceed the bound the paper states).

`statements` in `object.json` list qualitative claims (an interval excludes 0 or not, a count is zero);
each must equal its expected value in `values.json`.

With `--recompute`, each object's command is run into a temporary directory and its
`values.json` must equal the shipped one. The check reads only files inside the artifact, makes
no model call and opens no database unless an object's README says otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE.parent
PAPER = ART / "paper"
MACRO_FILES = ["sections/pending_numbers.tex", "sections/q4_numbers.tex", "main.tex", "supplementary.tex"]
WRAPPERS = ("fusionrev", "latestrev")


def _macros() -> dict:
    out = {}
    pat = re.compile(r"\\(?:new|renew|provide)command\{?\\([A-Za-z]+)\}?(?:\[\d\])?\{")
    for rel in MACRO_FILES:
        p = PAPER / rel
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8")
        for m in pat.finditer(s):
            i, depth = m.end(), 1
            j = i
            while j < len(s) and depth:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            body = s[i:j - 1]
            if "#" not in body:          # macros with arguments are markup, not numbers
                out.setdefault(m.group(1), body)
    return out


MACROS = _macros()
_RENDER_CACHE: dict = {}


def _strip_comments(s: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in s.splitlines())


def render(rel: str) -> str:
    """Paper source with comments removed, number macros expanded and math markup flattened."""
    if rel in _RENDER_CACHE:
        return _RENDER_CACHE[rel]
    s = _strip_comments((PAPER / rel).read_text(encoding="utf-8"))
    for _ in range(3):
        s = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])",
                   lambda m: MACROS.get(m.group(1), m.group(0)) if m.group(1) in MACROS
                   and re.search(r"\d", MACROS.get(m.group(1), "")) else m.group(0), s)
    for w in WRAPPERS:
        s = s.replace("\\" + w + "{", "{")
    for tok in ("\\,", "\\;", "\\!", "\\textminus", "\\cis"):
        s = s.replace(tok, " " if tok in ("\\,", "\\;") else ("-" if tok == "\\textminus" else ""))
    s = s.replace("$", "").replace("{", "").replace("}", "").replace("\u2212", "-")
    s = s.replace("\\%", "%")
    _RENDER_CACHE[rel] = s
    return s


def printed_in_source(printed: str, rel: str) -> bool:
    txt = render(rel)
    pat = r"(?<![\d.])" + re.escape(printed) + r"(?![\d])"
    if printed[:1] not in "+-":
        # an unsigned value may follow a range dash ("13--59") but not a minus sign
        pat = r"(?<![\d.+])(?<![^-]-)" + re.escape(printed) + r"(?![\d])"
    return re.search(pat, txt) is not None


def parse_printed(printed: str) -> tuple[float, int]:
    t = printed.replace(",", "").replace("%", "").replace("+", "").strip()
    d = len(t.split(".")[1]) if "." in t else 0
    return float(t), d


def compare(entry: dict, values: dict) -> tuple[bool, str]:
    key = entry["key"]
    if key not in values:
        return False, f"key {key!r} missing from output/values.json"
    v = values[key]
    if v is None:
        return False, f"{key} is null"
    v = float(v)
    tr = entry.get("transform")
    if tr == "abs":
        v = abs(v)
    elif tr == "neg":
        v = -v
    elif tr == "pct":
        v = 100.0 * v
    p, d = parse_printed(entry["printed"])
    if entry.get("relation") == "le":           # the paper states a bound ("no more than 2 pp")
        return v <= p + 1e-9, f"computed {v:.6g}, printed bound {entry['printed']}"
    ok = abs(v - p) <= 0.5 * 10 ** (-d) + 1e-9
    return ok, f"computed {v:.6g}, printed {entry['printed']}"


def run_object(od: pathlib.Path, recompute: bool, results: list) -> tuple[int, int]:
    meta = json.loads((od / "object.json").read_text(encoding="utf-8"))
    vals = json.loads((od / "output" / "values.json").read_text(encoding="utf-8"))
    n_ok = n_all = 0
    if recompute:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix=od.name + "_"))
        cmd = [sys.executable if c == "python3" else c for c in meta["compute"]] + ["--out", str(tmp)]
        r = subprocess.run(cmd, cwd=od, capture_output=True, text=True)
        n_all += 1
        if r.returncode != 0:
            results.append({"object": od.name, "check": "recompute", "pass": False,
                            "detail": (r.stderr or r.stdout)[-800:]})
        else:
            new = json.loads((tmp / "values.json").read_text(encoding="utf-8"))
            diff = [k for k in set(new) | set(vals)
                    if k not in new or k not in vals or not _same(new[k], vals[k])]
            ok = not diff
            n_ok += ok
            results.append({"object": od.name, "check": "recompute", "pass": ok,
                            "detail": "values.json identical" if ok else f"differs at {sorted(diff)[:8]}"})
        shutil.rmtree(tmp, ignore_errors=True)
    for e in meta["values"]:
        n_all += 1
        src = e.get("tex_file")
        # the paper source is not shipped with this repository; when present it is checked too
        in_src = printed_in_source(e["printed"], src) if src and (PAPER / src).is_file() else True
        ok, det = compare(e, vals)
        if not in_src:
            det += f"; printed form not found in paper/{src}"
        ok = ok and in_src
        n_ok += ok
        results.append({"object": od.name, "label": e.get("label"), "key": e["key"],
                        "printed": e["printed"], "where": e.get("where", ""), "pass": ok, "detail": det})
    for st in meta.get("statements", []):       # qualitative claims: detected / not detected, zero counts
        n_all += 1
        v = vals.get(st["key"])
        ok = v is not None and _same(v, st["expected"])
        n_ok += ok
        results.append({"object": od.name, "check": "statement", "statement": st["text"], "pass": ok,
                        "detail": f"{st['key']} = {v}, expected {st['expected']}"})
    return n_ok, n_all


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9)
    return a == b


def objects(only=None):
    for od in sorted(p for p in HERE.iterdir() if (p / "object.json").exists()):
        if only and od.name not in only:
            continue
        yield od


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)
    results: list = []
    tot_ok = tot = 0
    for od in objects(only):
        k, n = run_object(od, a.recompute, results)
        tot_ok += k
        tot += n
        if not a.quiet:
            print(f"{'PASS' if k == n else 'FAIL'}  {od.name}: {k}/{n}")
    for r in results:
        if not r["pass"]:
            print(f"  FAIL {r['object']} {r.get('label', '')} {r.get('key', r.get('check'))}: {r['detail']}")
    print(f"{tot_ok}/{tot} checks pass")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0 if tot_ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
