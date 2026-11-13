r"""Frozen parsing contracts for WDC `price` and Geographic Settlements coordinates.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# marks observed in this corpus, longest first so `R$` is tested before `$`
CURRENCY_MARKS = ("R$", "лв", "Lei", "US$", "AUD", "CAD", "EUR", "GBP", "USD",
                  "€", "£", "$", "¥", "₹")
OK = "ok"
AMBIGUOUS = "ambiguous_separator"
MISSING = "missing"
UNPARSABLE = "unparsable"

MISSING_TOKENS = {"", "null", "none", "nan", "n/a", "na", "-", "?", "unknown"}


def currency_of(s: Any) -> Optional[str]:
    """The currency mark present in the value, or None when unmarked."""
    t = str(s or "")
    for m in CURRENCY_MARKS:
        if m in t:
            return m
    return None


def _strip_non_numeric(t: str) -> str:
    return re.sub(r"[^0-9.,]", "", t)


# 1,454 of 17,722 WDC price values are in scientific notation (`4.4735E2`, `7.591E1`), 8.2% of the
# column.
_SCI = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*[eE]\s*([+-]?\d+)\s*$")


def parse_price(s: Any) -> Tuple[Optional[float], str, str]:
    """Return (value, status, basis). The separator comes from the value's shape, not its size."""
    raw = str(s if s is not None else "").strip()
    if raw.lower() in MISSING_TOKENS:
        return None, MISSING, "null-like token"

    # scientific notation, after removing any currency mark but before separator analysis
    bare = raw
    for mk in CURRENCY_MARKS:
        bare = bare.replace(mk, "")
    bare = bare.replace("‎", "").replace("‏", "").strip()
    m = _SCI.match(bare)
    if m:
        mant = m.group(1).replace(",", ".")
        try:
            return float(mant) * (10 ** int(m.group(2))), OK, \
                "scientific notation; mantissa and exponent read from the value"
        except (ValueError, OverflowError):
            return None, UNPARSABLE, "scientific notation that does not evaluate"
    if re.search(r"\d\s*[eE]", bare):
        # an exponent marker next to a digit that does not form a complete scientific number.
        return None, UNPARSABLE, \
            "an exponent marker follows a digit but the value is not a complete scientific number"

    t = _strip_non_numeric(raw)
    if not re.search(r"\d", t):
        return None, UNPARSABLE, "no digits after removing currency and spacing"
    t = t.strip(".,")

    has_c, has_d = "," in t, "." in t
    if has_c and has_d:
        last = max(t.rfind(","), t.rfind("."))
        dec, grp = t[last], ("." if t[last] == "," else ",")
        body = t.replace(grp, "")
        body = body.replace(dec, ".")
        try:
            return float(body), OK, f"both separators present; last ('{dec}') is decimal"
        except ValueError:
            return None, UNPARSABLE, "both separators present but not numeric"
    if has_c:
        parts = t.split(",")
        if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            return float(f"{parts[0]}.{parts[1]}"), OK, \
                "single comma with 1-2 trailing digits: decimal comma"
        if all(len(p) == 3 for p in parts[1:]) and parts[0]:
            return float("".join(parts)), OK, "comma groups of exactly 3: thousands separator"
        return None, AMBIGUOUS, f"comma in an undecidable position ({t!r})"
    if has_d:
        parts = t.split(".")
        if len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3:
            return None, AMBIGUOUS, \
                f"single period with exactly 3 trailing digits ({t!r}): 1.234 could be a thousands " \
                f"separator or a decimal point"
        try:
            return float(t if len(parts) == 2 else "".join(parts[:-1]) + "." + parts[-1]), OK, \
                "period as decimal point"
        except ValueError:
            return None, UNPARSABLE, "period present but not numeric"
    try:
        return float(t), OK, "plain integer"
    except ValueError:
        return None, UNPARSABLE, "not numeric"


def price_comparable(a: Any, b: Any) -> Tuple[bool, str]:
    """Whether a price difference may be computed at all."""
    ca, cb = currency_of(a), currency_of(b)
    if ca is None and cb is None:
        return False, "neither value states a currency, and this corpus mixes currencies, so an " \
                      "unmarked pair cannot be assumed to share one"
    if ca is None or cb is None:
        return False, f"only one side states a currency ({ca or cb})"
    if ca != cb:
        return False, f"different currencies ({ca} vs {cb}); no exchange rate is used"
    return True, f"both values are in {ca}"


def parse_coords(s: Any) -> Tuple[Optional[Tuple[float, float]], str, str]:
    """Latitude/longitude, refusing anything outside the coordinate system's own range."""
    raw = str(s if s is not None else "").strip()
    if raw.lower() in MISSING_TOKENS:
        return None, MISSING, "null-like token"
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    if len(nums) < 2:
        return None, UNPARSABLE, f"fewer than two numbers in {raw!r}"
    lat, lon = float(nums[0]), float(nums[1])
    if not (-90.0 <= lat <= 90.0):
        return None, "out_of_range", f"latitude {lat} is outside -90..90"
    if not (-180.0 <= lon <= 180.0):
        return None, "out_of_range", f"longitude {lon} is outside -180..180"
    return (lat, lon), OK, "two numbers inside the coordinate system's range"


# --------------------------------------------------------------------------- acceptance
def _acceptance() -> int:
    """Known answers, every one derivable from the value's own format or from the stated rule."""
    price_cases = [
        # (raw, expected value, expected status, why the answer is known)
        ("154.10", 154.10, OK, "plain decimal point"),
        ("2322.00", 2322.00, OK, "plain decimal point"),
        ("16371", 16371.0, OK, "plain integer"),
        ("6145.0000", 6145.0, OK, "four decimal places, unambiguous"),
        ("1,234.56", 1234.56, OK, "both separators: period is decimal, comma is grouping"),
        ("8,450.00", 8450.00, OK, "both separators"),
        ("1.234,56", 1234.56, OK, "both separators, European order"),
        ("378,00 лв", 378.00, OK, "single comma, two trailing digits: decimal comma — the v1 parse error "
                                  "returned 37800.0"),
        ("€ 230,35", 230.35, OK, "decimal comma — v1 returned 23035.0"),
        ("26,21 €", 26.21, OK, "decimal comma"),
        ("151,94€", 151.94, OK, "decimal comma"),
        ("$‎234.95", 234.95, OK, "currency mark and an invisible mark stripped"),
        ("1,049.99", 1049.99, OK, "both separators"),
        ("1,234", 1234.0, OK, "comma group of exactly three: thousands"),
        ("1.234", None, AMBIGUOUS, "could be 1234 or 1.234; reported ambiguous, never guessed"),
        ("12,34,56", None, AMBIGUOUS, "comma groups that are not all three digits"),
        ("", None, MISSING, "empty"),
        ("null", None, MISSING, "null-like token"),
        ("Rin cx tal", None, UNPARSABLE, "no digits"),
        # scientific notation: 8.2% of this column. v1 returned the mantissa only.
        ("4.4735E2", 447.35, OK, "exponent read from the value; v1 returned 4.4735"),
        ("7.591E1", 75.91, OK, "v1 returned 7.591"),
        ("5.0E1", 50.0, OK, "exponent 1"),
        ("6.15425E3", 6154.25, OK, "exponent 3"),
        ("7.9E0", 7.9, OK, "exponent 0 leaves the mantissa"),
        ("$5.1499E2", 514.99, OK, "currency mark removed before the exponent is read"),
        ("1.2E", None, UNPARSABLE, "no exponent digits"),
    ]
    comparable_cases = [
        ("154.10", "153.99", True, "both unmarked -> NOT comparable", False),
        ("€ 230,35", "€ 259,00", True, "same mark", True),
        ("€ 230,35", "154.10", False, "one side unmarked", False),
        ("€ 230,35", "$234.95", False, "different marks", False),
        ("378,00 лв", "26,21 €", False, "different marks", False),
    ]
    coord_cases = [
        ("30.3167, 35.4833", (30.3167, 35.4833), OK, "in range"),
        ("51.0509, 13.7383", (51.0509, 13.7383), OK, "in range"),
        ("-33.8688, 151.2093", (-33.8688, 151.2093), OK, "negative latitude in range"),
        ("8990.3300, -5170.8700", None, "out_of_range", "record 3077: latitude 8990 is impossible"),
        ("", None, MISSING, "empty"),
        ("48.8566", None, UNPARSABLE, "only one number"),
    ]

    fails = 0
    print("=== WDC price ===")
    print(f"{'raw':<18}{'expected':>12}{'got':>12}{'status':<22} why")
    for raw, exp, st, why in price_cases:
        v, s, basis = parse_price(raw)
        ok = s == st and ((exp is None and v is None)
                          or (exp is not None and v is not None and abs(v - exp) < 1e-9))
        fails += not ok
        print(f"{raw!r:<18}{(f'{exp:g}' if exp is not None else '-'):>12}"
              f"{(f'{v:g}' if v is not None else '-'):>12}{s:<22}{'' if ok else 'FAIL '}{why}")

    print("\n=== currency comparability (no exchange rate, ever) ===")
    for a, b, _same_mark, why, expected in comparable_cases:
        can, reason = price_comparable(a, b)
        ok = can == expected
        fails += not ok
        print(f"  {a!r:<14} vs {b!r:<12} comparable={can!s:<6}"
              f"{'' if ok else 'FAIL '}{reason}")

    print("\n=== GS coordinates ===")
    for raw, exp, st, why in coord_cases:
        v, s, basis = parse_coords(raw)
        ok = s == st and ((exp is None and v is None) or (exp is not None and v == exp))
        fails += not ok
        print(f"  {raw!r:<26} -> {str(v):<28}{s:<14}{'' if ok else 'FAIL '}{why}")

    n = len(price_cases) + len(comparable_cases) + len(coord_cases)
    print(f"\n{n} known-answer cases, {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_acceptance())
