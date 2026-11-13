"""Make pytest enforce what these suites check.

Every suite in this directory predates the pytest entrypoint. They were written to be run as
`python3 -m consensus_delivery.tests.test_x`, where a module-level `check(name, cond)` appends to
`FAILED` and `main()` returns non-zero if anything landed there. `check` deliberately **does not
raise**, so that one failure does not hide the remaining checks.

Under `pytest`, `main()` is never called. Nothing read `FAILED`, so a test function that recorded a
hundred failures still returned `None` and counted as a pass. Measured: a plugin that replaced
`check` in all seven modules with a function that *always* records a failure still produced
`95 passed`. 302 of the suites' 312 checks were unenforced by the command this project publishes as
its verification path, and the clean-source smoke runs that same command.

The fix keeps the collect-everything behaviour and adds the missing assertion: snapshot the module's
`FAILED` around each test and fail the test if it grew. It hooks the **call** phase rather than using
a fixture, so a suite's recorded failure is reported as a test failure and not as a teardown error
alongside a `passed` tally.
"""
from __future__ import annotations

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """Fail a test if its module recorded a `check()` failure while it ran."""
    failed = getattr(pyfuncitem.module, "FAILED", None)
    before = len(failed) if failed is not None else None
    outcome = yield
    if before is None:
        return
    new = failed[before:]
    if not new:
        return
    # Replace a "passed" result with the failure the suite already knows about.
    outcome.force_exception(AssertionError(
        f"{len(new)} recorded check failure(s) in {pyfuncitem.module.__name__}."
        f"{pyfuncitem.name}: " + "; ".join(map(str, new[:8])) + (" …" if len(new) > 8 else "")))
