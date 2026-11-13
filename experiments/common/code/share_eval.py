r"""One entry point for checking and recomputing a comparison on existing records.

    python3 code/share_eval.py validate spec/instances/sql_pool_repeat.json
    python3 code/share_eval.py analyze  spec/instances/sql_pool_repeat.json
    python3 code/share_eval.py validate spec/instances/chess_ut_boundary.json --spec-only

validate: schema + semantic checks of the spec, then record checks through the task adapter.
analyze:  validate, hash inputs, recompute through the adapter (existing graders / estimators),
          compare with the locked targets, write output/<id>/{validation,analysis,targets,
          inputs}.json and tables. Exit 0 only if no error finding and every target reproduces.
No model is called.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as C                                                            # noqa: E402

OUT = HERE.parent / "output"


def _adapter(spec):
    return importlib.import_module(f"adapters.{spec['metric']['adapter']}")


def validate(spec, spec_only=False):
    findings = C.check_spec(spec)
    recs = pops = None
    if not spec_only and not any(f.severity == C.ERROR for f in findings):
        A = _adapter(spec)
        try:
            recs, pops = A.load(spec)
        except C.SpecError as e:              # e.g. no records exist: reported, not faked
            return findings + e.findings, None, None
        findings += C.check_records(spec, recs, pops)
        if hasattr(A, "check"):
            findings += A.check(spec, recs, pops)
    return findings, recs, pops


def _write(d, name, obj):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("validate", "analyze"))
    ap.add_argument("spec")
    ap.add_argument("--spec-only", action="store_true", help="check the spec without loading records")
    a = ap.parse_args()
    spec = json.loads(pathlib.Path(a.spec).read_text())
    d = OUT / spec["id"]
    findings, recs, pops = validate(spec, a.spec_only)
    errs = [f for f in findings if f.severity == C.ERROR]
    val = dict(spec=a.spec, findings=[f.as_dict() for f in findings], errors=len(errs),
               unsupported=sum(f.severity == C.UNSUPPORTED for f in findings),
               records=None if recs is None else len(recs),
               populations=None if pops is None else {n: {p: len(u) for p, u in v.items()} for n, v in pops.items()})
    _write(d, "validation.json", val)
    for f in findings:
        print(f"[{f.severity}] {f.code} @ {f.where}: {f.message}")
    print(f"validate {spec['id']}: {len(errs)} error(s), {val['unsupported']} unsupported, records {val['records']}")
    if a.command == "validate":
        return 1 if errs else 0
    if errs:
        print("analyze refused: validation errors")
        return 1
    if recs is None:
        print("analyze not run: no records (see unsupported findings)")
        return 2
    _write(d, "inputs.json", C.hash_inputs(spec))
    A = _adapter(spec)
    got, cost_rows, extra = A.analyze(spec, recs, pops)
    tg = [C.compare_target(t, got.get(t["id"], {})) for t in spec.get("targets", [])]
    _write(d, "targets.json", tg)
    _write(d, "analysis.json", dict(estimates={k: v for k, v in got.items()}, extra=extra))
    if cost_rows:
        with open(d / "costs.csv", "w", newline="") as fh:
            keys = list({k: None for r in cost_rows for k in r})
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(cost_rows)
    for t in tg:
        print(("OK  " if t["reproduced"] else "FAIL"), t["id"], t["quantity"],
              {k: t[k] for k in t if k.startswith("got_")})
    bad = [t for t in tg if not t["reproduced"]]
    print(f"analyze {spec['id']}: {len(tg) - len(bad)}/{len(tg)} targets reproduced")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
