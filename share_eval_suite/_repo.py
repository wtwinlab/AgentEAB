"""Access to the repository's other code.

The protocol registry loads each protocol's collection code from `code/consensus_delivery/` and
`experiments/` on request. The tests use the
paper objects' expected values (`output/values.json`) and cross-check some results against the paper
objects' own computations. The stage evaluation, comparisons and reports do not use this module.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import pathlib
import sys
from types import ModuleType
from typing import Any, Dict

sys.dont_write_bytecode = True

REPO = pathlib.Path(__file__).resolve().parents[1]
PAPER_OBJECTS = REPO / "paper_objects"

_CACHE: Dict[str, ModuleType] = {}


@contextlib.contextmanager
def _on_path(*dirs: pathlib.Path):
    added = [str(d) for d in dirs if str(d) not in sys.path]
    for d in reversed(added):
        sys.path.insert(0, d)
    try:
        yield
    finally:
        for d in added:
            with contextlib.suppress(ValueError):
                sys.path.remove(d)


def load(relpath: str, *extra_paths: str) -> ModuleType:
    """Import the repository file `relpath` (relative to the repository root) as a module.

    The file's own directory and `extra_paths` (also relative to the root) are on `sys.path` while it
    loads, so its sibling imports resolve as they do when the file runs as a script. Modules are cached.
    """
    if relpath in _CACHE:
        return _CACHE[relpath]
    path = REPO / relpath
    if not path.is_file():
        raise FileNotFoundError(relpath)
    name = "share_eval_suite._repo_." + relpath.replace("/", ".").replace("-", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with _on_path(path.parent, *(REPO / p for p in extra_paths)):
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    _CACHE[relpath] = mod
    return mod


def load_package(dotted: str, root: str = "code") -> ModuleType:
    """Import a package module (e.g. `consensus_delivery.adapters.du_mad`) with `root` on the path.

    The protocol adapters use package-relative imports, so they load as package modules.
    """
    with _on_path(REPO / root):
        return importlib.import_module(dotted)


def paper_object(obj: str) -> ModuleType:
    """The computation module of `paper_objects/<obj>/code/compute.py`."""
    return load(f"paper_objects/{obj}/code/compute.py")


def inputs(obj: str) -> pathlib.Path:
    """The scored input tables shipped with a paper object."""
    return PAPER_OBJECTS / obj / "inputs"


def expected(obj: str) -> Dict[str, Any]:
    """The expected values shipped with a paper object (`output/values.json`)."""
    return json.loads((PAPER_OBJECTS / obj / "output" / "values.json").read_text())
