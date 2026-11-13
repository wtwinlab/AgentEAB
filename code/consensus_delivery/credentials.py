from __future__ import annotations

import os
from typing import List, Optional, Tuple

CANDIDATES = ("../../.env", "../../../.env", "../.env", ".env")


def _paths() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.normpath(os.path.join(here, c)) for c in CANDIDATES]


def _parse(path: str) -> dict:
    out = {}
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("\"'")
    except OSError:
        return {}
    return out


def ensure(name: str) -> Tuple[Optional[str], str]:
    v = os.environ.get(name)
    if v:
        return v, "environ"
    for p in _paths():
        if not os.path.exists(p):
            continue
        env = _parse(p)
        for k, val in env.items():
            if k.lower() == name.lower() and val:
                os.environ[name] = val
                return val, p
    return None, "missing"


def available(name: str) -> bool:
    return ensure(name)[0] is not None


def redact(text: str) -> str:
    out = text
    for p in _paths():
        for val in _parse(p).values():
            if val and len(val) >= 12 and val in out:
                out = out.replace(val, "<redacted>")
    for k, val in os.environ.items():
        if k.lower().endswith(("_api_key", "_token")) and val and len(val) >= 12 and val in out:
            out = out.replace(val, "<redacted>")
    return out
