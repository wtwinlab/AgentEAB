from __future__ import annotations

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    failed = getattr(pyfuncitem.module, "FAILED", None)
    before = len(failed) if failed is not None else None
    outcome = yield
    if before is None:
        return
    new = failed[before:]
    if not new:
        return
    outcome.force_exception(AssertionError(
        f"{len(new)} recorded check failure(s) in {pyfuncitem.module.__name__}."
        f"{pyfuncitem.name}: " + "; ".join(map(str, new[:8])) + (" …" if len(new) > 8 else "")))
