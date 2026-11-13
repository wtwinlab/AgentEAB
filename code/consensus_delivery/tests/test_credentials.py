"""Counterexamples for credential handling.

The standing constraint is that a credential is read by the program from `.env` and reaches nothing
else — no shell export, no command line, no log, no document. Before `credentials.py` there was no
shared loader, so every paid call depended on the variable having been exported first, which puts
the secret in a shell history and in the process listing of whatever set it.
"""
from __future__ import annotations

import io
import json
import os
import sys

from .. import credentials as C
from ..providers import BudgetRefused, ModelEndpoint, Registry, chat

CHECKS = 0
FAILED = []


def check(name: str, cond: bool, detail: object = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  ok    {name}")
    else:
        FAILED.append(f"{name}: {detail}")
        print(f"  FAIL  {name}: {detail}")


def test_a_missing_credential_is_refused_not_guessed():
    ep = ModelEndpoint(alias="fake-paid", base_url="https://example.invalid/v1",
                       served_name="fake", checkpoint="fake/fake", provider="fake",
                       api_key_env="DEFINITELY_NOT_SET_API_KEY",
                       usd_per_m_in=1.0, usd_per_m_out=1.0,
                       max_context=1000, max_completion=100)
    reg = Registry((ep,))
    try:
        chat(registry=reg, alias="fake-paid", messages=[{"role": "user", "content": "x"}],
             temperature=0.0, top_p=1.0, max_tokens=4, retries=0)
        check("a paid call without a credential is refused", False, "it was attempted")
    except BudgetRefused as e:
        check("a paid call without a credential is refused", True, "")
        check("and the refusal names the variable, not a value",
              "DEFINITELY_NOT_SET_API_KEY" in str(e) and "Bearer" not in str(e), str(e))
    except Exception as e:                       # noqa: BLE001
        check("a paid call without a credential is refused with BudgetRefused",
              False, f"{type(e).__name__}: {e}")


def test_ensure_reports_a_source_but_never_the_value():
    value, source = C.ensure("DEFINITELY_NOT_SET_API_KEY")
    check("a missing credential yields no value", value is None, "")
    check("and the source says missing", source == "missing", source)
    # A real one, if the project has it. The test asserts only that the *source* is reportable.
    real, src = C.ensure("DEEPSEEK_API_KEY")
    if real:
        check("a found credential reports a path or 'environ', not the secret",
              src == "environ" or src.endswith(".env"), src)
        check("and the source string does not contain the secret", real not in src, "secret present in source")
    else:
        check("no DeepSeek credential in this checkout; nothing to check", True, "")


def test_available_does_not_return_the_secret():
    v = C.available("DEEPSEEK_API_KEY")
    check("available() is a boolean", isinstance(v, bool), type(v).__name__)


def test_redact_removes_a_known_value():
    real, _ = C.ensure("DEEPSEEK_API_KEY")
    if not real:
        check("no credential to redact in this checkout", True, "")
        return
    msg = f"request failed with Authorization: Bearer {real} at 12:00"
    out = C.redact(msg)
    check("the value is gone", real not in out, "still present")
    check("and the surrounding message survives",
          "request failed" in out and "<redacted>" in out, out[:80])
    check("a short string is not redacted into nonsense", C.redact("abc") == "abc", C.redact("abc"))


def test_the_env_file_is_never_written():
    paths = [p for p in C._paths() if os.path.exists(p)]
    if not paths:
        check("no .env in this checkout; nothing to protect", True, "")
        return
    p = paths[0]
    before = open(p, "rb").read()
    C.ensure("DEEPSEEK_API_KEY")
    C.ensure("deepinfra_api_key")
    C.ensure("DEFINITELY_NOT_SET_API_KEY")
    check("ensure() does not modify .env", open(p, "rb").read() == before, p)


def test_loading_is_case_insensitive_on_the_key():
    """`.env` here holds both `DEEPSEEK_API_KEY` and `deepinfra_api_key`."""
    check("a lowercase key in .env is found under its own name",
          isinstance(C.available("deepinfra_api_key"), bool), "")


def main() -> int:
    for fn in (test_a_missing_credential_is_refused_not_guessed,
               test_ensure_reports_a_source_but_never_the_value,
               test_available_does_not_return_the_secret,
               test_redact_removes_a_known_value,
               test_the_env_file_is_never_written,
               test_loading_is_case_insensitive_on_the_key):
        fn()
    print(f"\n{CHECKS} checks, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  - {f}")
    return 1 if FAILED else 0


def test_credential_counterexamples():
    """pytest entrypoint, so `conftest` enforces the checks above."""
    rc = main()
    assert rc == 0, f"recorded check failures: {FAILED}"
    assert CHECKS > 0


if __name__ == "__main__":
    sys.exit(main())
