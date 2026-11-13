r"""Deterministic record-field evidence templates (condition E5), one per domain.

    python3 experiments/04_sql_selection_generation/code/evidence_templates.py   # self-test
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

HEADING = "Field comparison computed from the two records:"

#: Values that mean "no data" in these corpora.
MISSING_TOKENS = {"", "null", "none", "nan", "n/a", "na", "unk.", "unk", "unknown", "-", "?"}


def _is_missing(v: Any) -> bool:
    return v is None or str(v).strip().lower() in MISSING_TOKENS


# --------------------------------------------------------------------- shared helpers
def _norm(s: Any) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def _tokens(s: Any) -> set:
    return set(_norm(s).split())


def _jaccard(a: Any, b: Any) -> Optional[float]:
    if _is_missing(a) or _is_missing(b):
        return None
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return None
    return len(ta & tb) / len(ta | tb)


def _fmt_pair(label: str, va: Any, vb: Any) -> str:
    """Report a field's two values, naming absence explicitly."""
    sa = "absent" if _is_missing(va) else f"{va!r}"
    sb = "absent" if _is_missing(vb) else f"{vb!r}"
    if _is_missing(va) or _is_missing(vb):
        return f"  - {label}: A {sa}, B {sb} (not comparable)"
    if _norm(va) == _norm(vb):
        return f"  - {label}: identical after normalisation ({sa})"
    return f"  - {label}: A {sa} vs B {sb} (differ)"


def _overlap_line(label: str, va: Any, vb: Any) -> Optional[str]:
    j = _jaccard(va, vb)
    if j is None:
        return None
    ta, tb = _tokens(va), _tokens(vb)
    shared = sorted(ta & tb)
    only_a = sorted(ta - tb)
    only_b = sorted(tb - ta)
    return (f"  - {label} token overlap: {j:.2f}; shared {shared or 'none'}; "
            f"only in A {only_a or 'none'}; only in B {only_b or 'none'}")


# --------------------------------------------------------------------- geographic settlements
def _parse_coords(s: Any) -> Optional[Tuple[float, float]]:
    if _is_missing(s):
        return None
    m = re.findall(r"-?\d+(?:\.\d+)?", str(s))
    if len(m) < 2:
        return None
    try:
        return float(m[0]), float(m[1])
    except ValueError:
        return None


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0088
    p1, l1 = math.radians(a[0]), math.radians(a[1])
    p2, l2 = math.radians(b[0]), math.radians(b[1])
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2)
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))


def evidence_settlements(A: Dict[str, Any], B: Dict[str, Any]) -> str:
    """Name agreement, a computed great-circle distance, and type/catalogue comparison."""
    lines: List[str] = []
    lines.append(_fmt_pair("name", A.get("name"), B.get("name")))
    ov = _overlap_line("name", A.get("name"), B.get("name"))
    if ov:
        lines.append(ov)
    # a region qualifier in parentheses is the dataset's own convention; report it, do not strip it
    qa = re.findall(r"\(([^)]*)\)", str(A.get("name") or ""))
    qb = re.findall(r"\(([^)]*)\)", str(B.get("name") or ""))
    if qa or qb:
        lines.append(f"  - parenthesised qualifier: A {qa or 'none'}, B {qb or 'none'}")
    ca, cb = _parse_coords(A.get("coordinates")), _parse_coords(B.get("coordinates"))
    if ca and cb:
        d = _haversine_km(ca, cb)
        lines.append(f"  - great-circle distance between the stated coordinates: {d:.2f} km")
        if d == 0.0:
            lines.append("  - coordinates are identical to the stated precision")
    else:
        lines.append(f"  - coordinates: A {A.get('coordinates') or 'absent'}, "
                     f"B {B.get('coordinates') or 'absent'} (distance not computable)")
    for f in ("type", "type detail"):
        if A.get(f) or B.get(f):
            lines.append(_fmt_pair(f, A.get(f), B.get(f)))
    lines.append(_fmt_pair("catalogue", A.get("catalogue"), B.get("catalogue")))
    return HEADING + "\n" + "\n".join(lines)


# --------------------------------------------------------------------- musicbrainz
def _parse_len_seconds(s: Any) -> Optional[float]:
    """Seconds from the several duration formats this corpus actually uses.
    """
    if _is_missing(s):
        return None
    t = str(s).strip().lower()
    m = re.fullmatch(r"(?:(\d+)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*s(?:ec)?)?", t)
    if m and (m.group(1) or m.group(2)):
        return int(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    m = re.fullmatch(r"(\d+):(\d{1,2})(?:\.(\d+))?", t)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    if re.fullmatch(r"\d+(\.\d+)?", t):
        v = float(t)
        return v / 1000.0 if v > 1000 else v          # ms vs s, by magnitude
    return None


def evidence_musicbrainz(A: Dict[str, Any], B: Dict[str, Any]) -> str:
    """Artist/title/album agreement, a computed length difference, and year/number comparison."""
    lines: List[str] = []
    for f in ("title", "artist", "album"):
        lines.append(_fmt_pair(f, A.get(f), B.get(f)))
        ov = _overlap_line(f, A.get(f), B.get(f))
        if ov:
            lines.append(ov)
    la, lb = _parse_len_seconds(A.get("length")), _parse_len_seconds(B.get("length"))
    if la is not None and lb is not None:
        lines.append(f"  - track length: A {la:.0f}s, B {lb:.0f}s, "
                     f"difference {abs(la - lb):.0f}s")
    else:
        lines.append(f"  - track length: A {A.get('length') or 'absent'}, "
                     f"B {B.get('length') or 'absent'} (difference not computable)")
    # year needs care: this corpus mixes two-digit and four-digit years ("75" vs "1975"), and
    # subtracting them would report a 1900-year gap for what is the same year in two formats.
    ya = "" if _is_missing(A.get("year")) else str(A.get("year")).strip()
    yb = "" if _is_missing(B.get("year")) else str(B.get("year")).strip()
    if ya and yb and ya.isdigit() and yb.isdigit():
        if len(ya) == len(yb):
            lines.append(f"  - year: A {ya}, B {yb}, difference {abs(int(ya) - int(yb))}")
        else:
            same_tail = ya[-2:] == yb[-2:]
            lines.append(f"  - year: A {ya}, B {yb} (different digit widths, so no numeric "
                         f"difference is computed; last two digits "
                         f"{'agree' if same_tail else 'differ'})")
    else:
        lines.append(_fmt_pair("year", A.get("year"), B.get("year")))
    for f in ("number", "language"):
        lines.append(_fmt_pair(f, A.get(f), B.get(f)))
    return HEADING + "\n" + "\n".join(lines)


# --------------------------------------------------------------------- products
def _price(s: Any) -> Optional[float]:
    if _is_missing(s):
        return None
    m = re.findall(r"\d+(?:[.,]\d+)?", str(s).replace(",", ""))
    try:
        return float(m[0]) if m else None
    except ValueError:
        return None


def evidence_products(A: Dict[str, Any], B: Dict[str, Any]) -> str:
    """Title/brand agreement, model-number-like tokens, and a computed price difference.
    """
    lines: List[str] = []
    for f in ("title", "name", "brand", "manufacturer"):
        if A.get(f) or B.get(f):
            lines.append(_fmt_pair(f, A.get(f), B.get(f)))
            ov = _overlap_line(f, A.get(f), B.get(f))
            if ov:
                lines.append(ov)
    def models(d: Dict[str, Any]) -> List[str]:
        # price is reported on its own line below; letting it into the model-token set would put
        # "199" beside real model numbers and invite a spurious mismatch signal
        blob = " ".join(str(v) for k, v in d.items() if v and k not in ("price",))
        return sorted({t for t in re.findall(r"[a-z]*\d[a-z0-9\-]*", blob.lower())
                       if any(c.isdigit() for c in t) and len(t) >= 3})
    ma, mb = models(A), models(B)
    lines.append(f"  - alphanumeric model-like tokens: shared {sorted(set(ma) & set(mb)) or 'none'}; "
                 f"only in A {sorted(set(ma) - set(mb))[:6] or 'none'}; "
                 f"only in B {sorted(set(mb) - set(ma))[:6] or 'none'}")
    pa, pb = _price(A.get("price")), _price(B.get("price"))
    if pa is not None and pb is not None:
        rel = abs(pa - pb) / max(pa, pb) if max(pa, pb) else 0.0
        lines.append(f"  - price: A {pa:.2f}, B {pb:.2f}, absolute difference {abs(pa - pb):.2f}, "
                     f"relative {rel:.2f}")
    else:
        lines.append(f"  - price: A {A.get('price') or 'absent'}, B {B.get('price') or 'absent'} "
                     f"(difference not computable)")
    return HEADING + "\n" + "\n".join(lines)


TEMPLATES = {"geographic_settlements": evidence_settlements,
             "musicbrainz20k": evidence_musicbrainz,
             "wdc_products": evidence_products}


def _selftest() -> int:
    print("=== geographic_settlements ===")
    A = {"name": "Moscow (Russia)", "coordinates": "55.7522, 37.6156",
         "catalogue": "New York Times"}
    B = {"name": "Moscow", "coordinates": "55.7522, 37.6156", "type": "city, village,...",
         "type detail": "capital of a political entity", "catalogue": "GeoNames"}
    print(evidence_settlements(A, B))
    print("\n--- a far-apart pair, and a missing-coordinate pair ---")
    print(evidence_settlements({"name": "Springfield", "coordinates": "39.7817, -89.6501",
                                "catalogue": "GeoNames"},
                               {"name": "Springfield (Massachusetts)",
                                "coordinates": "42.1015, -72.5898", "catalogue": "DBpedia"}))
    print(evidence_settlements({"name": "Oslo", "catalogue": "Freebase"},
                               {"name": "Oslo", "coordinates": "59.9139, 10.7522",
                                "catalogue": "GeoNames"}))
    print("\n=== musicbrainz20k ===")
    print(evidence_musicbrainz(
        {"title": "Daniel Balavoine - L'enfant aux yeux d'Italie", "artist": "",
         "album": "De vous a elle en passant par moi", "year": "75", "length": "219",
         "number": "9", "language": "French"},
        {"title": "L'enfant aux yeux d'Italie", "artist": "Daniel Balavoine",
         "album": "De vous a elle en passant par moi", "year": "1975", "length": "220",
         "number": "9", "language": ""}))
    print("\n=== wdc_products ===")
    print(evidence_products(
        {"title": "sony cyber-shot dsc-w120 pink", "brand": "sony", "price": "199.99"},
        {"title": "sony dscw120 cybershot 7.2mp pink digital camera", "price": "188.00"}))
    # determinism
    for name, fn in TEMPLATES.items():
        a = fn({"name": "x", "title": "x"}, {"name": "y", "title": "y"})
        b = fn({"name": "x", "title": "x"}, {"name": "y", "title": "y"})
        assert a == b, name
    print("\ndeterminism check: identical inputs give identical evidence for all three templates")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
