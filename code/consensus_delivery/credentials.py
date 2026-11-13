r"""Read API credentials from `.env`, programmatically, and never anywhere else.

The standing constraint is that credentials are read by the program from `.env` — not exported in a
shell, not passed on a command line, not written to a log, a document, a tmux pane or git. Before
this module there was no shared loader, so every paid call depended on someone having exported the
variable first; that puts the secret in a shell history and in any process listing of the command
that set it.

`ensure(name)` is called from the one place a credential is used, so no entry point can forget it and
no caller needs to handle a key at all. The value is returned to that caller and to nobody else:

* it is never printed, and the functions here return booleans and file paths, never the secret;
* it is not written back to `.env` or to any other file;
* `redact` exists for the one case where a value might otherwise reach a message.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

#: Searched in order, relative to this file, so the loader works from any working directory.
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
    """The credential and where it came from, loading `.env` if the environment lacks it.

    Returns `(value, source)`. `source` is `"environ"`, the `.env` path it was read from, or
    `"missing"`. **The value is for the caller that makes the request; it is never logged here.**
    Matching is case-insensitive on the key, because `.env` files in this project use both
    `DEEPSEEK_API_KEY` and `deepinfra_api_key`.
    """
    v = os.environ.get(name)
    if v:
        return v, "environ"
    for p in _paths():
        if not os.path.exists(p):
            continue
        env = _parse(p)
        for k, val in env.items():
            if k.lower() == name.lower() and val:
                # Placed in this process's environment so a retry does not re-read the file, and
                # only in this process: nothing is exported to a parent shell.
                os.environ[name] = val
                return val, p
    return None, "missing"


def available(name: str) -> bool:
    """Whether a credential can be obtained, without revealing or returning it."""
    return ensure(name)[0] is not None


def redact(text: str) -> str:
    """Replace any known credential value appearing in `text`. A last line of defence."""
    out = text
    for p in _paths():
        for val in _parse(p).values():
            if val and len(val) >= 12 and val in out:
                out = out.replace(val, "<redacted>")
    for k, val in os.environ.items():
        if k.lower().endswith(("_api_key", "_token")) and val and len(val) >= 12 and val in out:
            out = out.replace(val, "<redacted>")
    return out
