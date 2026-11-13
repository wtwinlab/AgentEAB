r"""Materialise WDC / GS datasets whose `E5` condition uses the revised price and coordinate contracts.

    PYTHONPATH=code python3 code/materialise_fixed_dataset.py --dataset wdc_products --arm E5f
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
from typing import Any, Dict

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["wdc_products", "geographic_settlements"])
    ap.add_argument("--arm", required=True, choices=["E5f", "E5c"])
    a = ap.parse_args()
    src = os.path.join(STAGE, "data", a.dataset)
    dst = os.path.join(STAGE, "data", f"{a.dataset}_{a.arm}")
    os.makedirs(dst, exist_ok=True)

    wrote_any = False
    for split in ("dev", "test"):
        src_arms = os.path.join(src, f"arms_{split}.jsonl")
        if not os.path.exists(src_arms):
            continue
        fixed_path = os.path.join(src, f"fixed_{split}.jsonl")
        fixed: Dict[tuple, Dict[str, Any]] = {}
        if os.path.exists(fixed_path):
            for ln in open(fixed_path, encoding="utf-8"):
                o = json.loads(ln)
                if o.get("arm") != a.arm:
                    continue
                fixed[(o["a"], o["b"], o["recipient"], o["repeat"])] = o

        out = os.path.join(dst, f"arms_{split}.jsonl")
        tmp = out + ".partial"
        kept = collections.Counter()
        used = 0
        missing = []
        with open(tmp, "w", encoding="utf-8") as w:
            for ln in open(src_arms, encoding="utf-8"):
                o = json.loads(ln)
                if o.get("arm") != "E5":
                    w.write(ln if ln.endswith("\n") else ln + "\n")
                    kept[o.get("arm")] += 1
                    continue
                if o.get("role") != "recipient":
                    continue
                key = (o["a"], o["b"], o["recipient"], o["repeat"])
                new = fixed.get(key)
                if new is None:
                    # the revision did not change this pair's block: the stored row IS the revised row
                    w.write(ln if ln.endswith("\n") else ln + "\n")
                    kept["E5"] += 1
                    continue
                row = dict(new)
                row["arm"] = "E5"
                row["substituted_from"] = a.arm
                row["replaces_unit_id"] = o["unit_id"]
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept["E5"] += 1
                used += 1
        if used != len(fixed):
            os.remove(tmp)
            print(f"  FAILED {split}: used {used} of {len(fixed)} collected cells; the "
                  f"substitution is not one-to-one. Nothing written.")
            return 1
        os.replace(tmp, out)
        wrote_any = True
        print(f"  {split}: rows by arm {dict(sorted(kept.items()))}; "
              f"{used} cells substituted from {a.arm}, "
              f"{kept['E5'] - used} stored E5 rows reused unchanged")
        print(f"     sha256 {hashlib.sha256(open(out, 'rb').read()).hexdigest()}")
        for stem in ("pool", "verify"):
            s = os.path.join(src, f"{stem}_{split}.jsonl")
            d = os.path.join(dst, f"{stem}_{split}.jsonl")
            if os.path.exists(s) and not os.path.exists(d):
                os.symlink(s, d)

    proto_s = os.path.join(STAGE, "data/protocol", f"{a.dataset}.json")
    proto_d = os.path.join(STAGE, "data/protocol", f"{a.dataset}_{a.arm}.json")
    if not os.path.exists(proto_d):
        shutil.copyfile(proto_s, proto_d)
    if wrote_any:
        print(f"\n  now runnable unmodified, e.g.:")
        print(f"    PYTHONPATH=code python3 code/analyse_matrix.py "
              f"--dataset {a.dataset}_{a.arm} --splits dev,test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
