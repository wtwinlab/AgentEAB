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
    with _on_path(REPO / root):
        return importlib.import_module(dotted)


def paper_object(obj: str) -> ModuleType:
    return load(f"paper_objects/{obj}/code/compute.py")


def inputs(obj: str) -> pathlib.Path:
    return PAPER_OBJECTS / obj / "inputs"


def expected(obj: str) -> Dict[str, Any]:
    return json.loads((PAPER_OBJECTS / obj / "output" / "values.json").read_text())
