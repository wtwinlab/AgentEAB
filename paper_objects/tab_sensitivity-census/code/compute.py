"""Table tab:sensitivity-census: how often alternative output rules change the conclusion.

    python3 code/compute.py --out output

Reads only ../inputs.
- census_cases.json: one entry per case, with the baseline estimate (field `naive`) and the
  alternative estimate (field `controlled`) and their 95% intervals. A case flips its sign when
  both estimates are nonzero with opposite signs; it flips detection when exactly one of the two intervals excludes zero; it is stable when neither happens.
  Cases whose baseline estimate is undefined (Du-MAD/Llama-3-8B on BIRD, whose extractor delivers
  nothing) are not eligible.
- grouping_sharing_effects.csv: one row per sharing-grouping combination on the test split, with the
  sharing effect on pair F1 and on cluster F1 (pp); a reversal is a strictly positive change on one
  and a strictly negative change on the other, and a combination is stable when both move in the
  same direction.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
FAMILIES = [("S1", None, "bird_sharing"), ("S2", None, "bird_dpeer"), ("S3", None, "bird_final_vs_vote"),
            ("S4", None, "em_qa"), ("S5", "first", "return_first"), ("S5", "last", "return_last"),
            ("S5", "keep", "return_kept")]


def sig(ci):
    return None if ci is None else (ci[0] > 0 or ci[1] < 0)


def classify(r):
    n, c = r["naive"], r["controlled"]
    sign = n != 0 and c != 0 and (n > 0) != (c > 0)
    sn, sc = sig(r["naive_ci"]), sig(r["controlled_ci"])
    detect = sn is not None and sc is not None and sn != sc
    return sign, detect, (not sign) and not detect


def sgn(x, tol=1e-9):
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((INP / "census_cases.json").read_text())
    V, changed_total, eligible_total = {}, 0, 0
    for fam, tie, tag in FAMILIES:
        rs = [r for r in rows if r["family"] == fam and (tie is None or r.get("tie") == tie) and not r["naive_undefined"]]
        cls = [classify(r) for r in rs]
        V[f"{tag}_cases"] = len(rs)
        V[f"{tag}_sign"] = sum(c[0] for c in cls)
        V[f"{tag}_detect"] = sum(c[1] for c in cls)
        V[f"{tag}_stable"] = sum(c[2] for c in cls)
        if fam in ("S1", "S2", "S3", "S4"):
            eligible_total += len(rs)
            changed_total += sum(not c[2] for c in cls)
    ctrl = [r for r in rows if r["family"] == "S4" and r.get("negative_control")]
    cc = [classify(r) for r in ctrl]
    V["control_cases"] = len(ctrl)
    V["control_sign"] = sum(c[0] for c in cc)
    V["control_detect"] = sum(c[1] for c in cc)
    V["control_stable"] = sum(c[2] for c in cc)
    V["measurement_changed"] = changed_total
    V["measurement_eligible"] = eligible_total
    g = list(csv.DictReader(open(INP / "grouping_sharing_effects.csv")))
    pairs = [(sgn(float(r["d_pair_pp"])), sgn(float(r["d_cluster_pp"]))) for r in g]
    V["grouping_cases"] = len(pairs)
    V["grouping_reversals"] = sum(1 for p, c in pairs if p * c == -1)
    V["grouping_stable"] = sum(1 for p, c in pairs if p * c == 1)
    (out / "values.json").write_text(json.dumps(dict(sorted(V.items())), indent=1) + "\n")
    print(len(V), "values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
