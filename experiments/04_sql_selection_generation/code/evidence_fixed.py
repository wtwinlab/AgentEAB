r"""Revised WDC and Geographic Settlements evidence blocks (`E5f`), version-isolated.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
sys.path.insert(0, os.path.join(STAGE, "code"))

import evidence_templates as V1          # noqa: E402  helpers reused verbatim
import price_geo_contract as C           # noqa: E402

VERSION = "E5f/v2-price-geo-contract"


# --------------------------------------------------------------------------- WDC
def evidence_products_fixed(A: Dict[str, Any], B: Dict[str, Any]) -> str:
    """v1's product block with only the price line replaced."""
    lines: List[str] = []
    for f in ("title", "name", "brand", "manufacturer"):
        if A.get(f) or B.get(f):
            lines.append(V1._fmt_pair(f, A.get(f), B.get(f)))
            ov = V1._overlap_line(f, A.get(f), B.get(f))
            if ov:
                lines.append(ov)

    def models(d: Dict[str, Any]) -> List[str]:
        blob = " ".join(str(v) for k, v in d.items() if v and k not in ("price",))
        return sorted({t for t in re.findall(r"[a-z]*\d[a-z0-9\-]*", blob.lower())
                       if any(c.isdigit() for c in t) and len(t) >= 3})

    ma, mb = models(A), models(B)
    lines.append(f"  - alphanumeric model-like tokens: shared {sorted(set(ma) & set(mb)) or 'none'}; "
                 f"only in A {sorted(set(ma) - set(mb))[:6] or 'none'}; "
                 f"only in B {sorted(set(mb) - set(ma))[:6] or 'none'}")
    lines.append(price_line(A.get("price"), B.get("price")))
    return V1.HEADING + "\n" + "\n".join(lines)


def price_line(pa: Any, pb: Any) -> str:
    va, sa, _ = C.parse_price(pa)
    vb, sb, _ = C.parse_price(pb)
    show_a = "absent" if sa == C.MISSING else repr(str(pa))
    show_b = "absent" if sb == C.MISSING else repr(str(pb))
    if va is None or vb is None:
        why = {C.MISSING: "absent", C.AMBIGUOUS: "separator cannot be decided",
               C.UNPARSABLE: "not a price", C.OK: None}

        def show(v: Any, st: str, raw: str) -> str:
            # only the side that failed carries a reason; a side that parsed shows its value
            if st == C.MISSING:
                return "absent"
            if why.get(st) is None:
                return raw
            return f"{raw} ({why[st]})"

        return (f"  - price: A {show(pa, sa, show_a)}, B {show(pb, sb, show_b)} "
                f"(difference not computable)")
    can, reason = C.price_comparable(pa, pb)
    if not can:
        return (f"  - price: A {show_a}, B {show_b} (difference not computable: {reason})")
    rel = abs(va - vb) / max(va, vb) if max(va, vb) else 0.0
    return (f"  - price: A {va:.2f}, B {vb:.2f}, absolute difference {abs(va - vb):.2f}, "
            f"relative {rel:.2f}")


# --------------------------------------------------------------------------- GS
def evidence_settlements_fixed(A: Dict[str, Any], B: Dict[str, Any]) -> str:
    """v1's settlement block, rebuilt line for line, differing **only** where a coordinate is invalid.
    """
    lines: List[str] = []
    lines.append(V1._fmt_pair("name", A.get("name"), B.get("name")))
    ov = V1._overlap_line("name", A.get("name"), B.get("name"))
    if ov:
        lines.append(ov)
    qa = re.findall(r"\(([^)]*)\)", str(A.get("name") or ""))
    qb = re.findall(r"\(([^)]*)\)", str(B.get("name") or ""))
    if qa or qb:
        lines.append(f"  - parenthesised qualifier: A {qa or 'none'}, B {qb or 'none'}")
    pa, sa, _ = C.parse_coords(A.get("coordinates"))
    pb, sb, _ = C.parse_coords(B.get("coordinates"))
    if pa and pb:
        d = V1._haversine_km(pa, pb)
        lines.append(f"  - great-circle distance between the stated coordinates: {d:.2f} km")
        if d == 0.0:
            lines.append("  - coordinates are identical to the stated precision")
    else:
        # an out-of-range coordinate lands in this branch instead of producing a distance.
        extra = ""
        if "out_of_range" in (sa, sb):
            side = "A" if sa == "out_of_range" else "B"
            extra = (f" [{side} is outside the valid latitude/longitude range, so no distance is "
                     f"computed]")
        lines.append(f"  - coordinates: A {A.get('coordinates') or 'absent'}, "
                     f"B {B.get('coordinates') or 'absent'} (distance not computable){extra}")
    for f in ("type", "type detail"):
        if A.get(f) or B.get(f):
            lines.append(V1._fmt_pair(f, A.get(f), B.get(f)))
    lines.append(V1._fmt_pair("catalogue", A.get("catalogue"), B.get("catalogue")))
    return V1.HEADING + "\n" + "\n".join(lines)


# --------------------------------------------------------------------------- enumeration
def enumerate_changes() -> Dict[str, Any]:
    """Which pairs' blocks actually change, and why. Run before any collection."""
    sys.path.insert(0, os.path.join(ROOT, "experiments/06_cost_resources/code"))
    sys.path.insert(0, os.path.join(ROOT, "experiments/05_matching_downstream/code"))
    import collect_matrix as CM                                       # noqa: PLC0415
    rep: Dict[str, Any] = {"version": VERSION, "datasets": {}}

    # ---- WDC
    dom = CM.Domain("wdc_products")
    blob = json.load(open(os.path.join(STAGE, "data/protocol/wdc_products.json"),
                          encoding="utf-8"))
    ent, sp = blob["entity_of"], blob["split"]
    per_split: Dict[str, Any] = {}
    for split in ("dev", "test"):
        cands = [(str(x), str(y)) for x, y in blob["candidates"]
                 if sp.get(ent.get(str(x))) == split and sp.get(ent.get(str(y))) == split]
        cause = collections.Counter()
        changed = 0
        for (x, y) in cands:
            A, B = dom.fields(x), dom.fields(y)
            old = V1.evidence_products(A, B)
            new = evidence_products_fixed(A, B)
            if old == new:
                cause["unchanged"] += 1
                continue
            changed += 1
            va_o, vb_o = V1._price(A.get("price")), V1._price(B.get("price"))
            va_n, _, _ = C.parse_price(A.get("price"))
            vb_n, _, _ = C.parse_price(B.get("price"))
            comma = ((va_o is not None and va_n is not None and abs(va_o - va_n) > 1e-9)
                     or (vb_o is not None and vb_n is not None and abs(vb_o - vb_n) > 1e-9))
            can, _ = C.price_comparable(A.get("price"), B.get("price"))
            if comma and not can:
                cause["both: comma value fixed AND currency now unstated"] += 1
            elif comma:
                cause["comma value fixed (100x correction)"] += 1
            elif not can:
                cause["currency rule: difference withdrawn"] += 1
            else:
                cause["other (ambiguous separator or formatting)"] += 1
        per_split[split] = {"n_pairs": len(cands), "n_changed": changed,
                            "frac_changed": round(changed / max(len(cands), 1), 4),
                            "by_cause": dict(cause)}
    rep["datasets"]["wdc_products"] = per_split

    # ---- GS
    sys.path.insert(0, os.path.join(ROOT, "experiments/05_matching_downstream/code"))
    domg = CM.Domain("geographic_settlements")
    blobg = json.load(open(os.path.join(STAGE, "data/protocol/geographic_settlements.json"),
                           encoding="utf-8"))
    entg, spg = blobg["entity_of"], blobg["split"]
    gs: Dict[str, Any] = {}
    for split in ("dev", "test"):
        cands = [(str(x), str(y)) for x, y in blobg["candidates"]
                 if spg.get(entg.get(str(x))) == split and spg.get(entg.get(str(y))) == split]
        changed = [k for k in cands
                   if V1.evidence_settlements(domg.fields(k[0]), domg.fields(k[1]))
                   != evidence_settlements_fixed(domg.fields(k[0]), domg.fields(k[1]))]
        gs[split] = {"n_pairs": len(cands), "n_changed": len(changed),
                     "frac_changed": round(len(changed) / max(len(cands), 1), 4),
                     "changed_pairs": changed[:20]}
    rep["datasets"]["geographic_settlements"] = gs
    return rep


def _main() -> int:
    rep = enumerate_changes()
    print("=== payloads that actually change under the repaired contracts ===")
    for ds, splits in rep["datasets"].items():
        print(f"\n  {ds}")
        for split, v in splits.items():
            print(f"    {split}: {v['n_changed']} of {v['n_pairs']} pairs "
                  f"({100*v['frac_changed']:.1f}%)")
            for k, n in (v.get("by_cause") or {}).items():
                print(f"       {n:>6}  {k}")
            if v.get("changed_pairs"):
                print(f"       changed pairs: {v['changed_pairs']}")
    p = os.path.join(STAGE, "output", "price_geo_change_enumeration.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
