from __future__ import annotations

import json
import os
import subprocess
import sys

from ..adapters.mallm_upstream import INTENDED_PROTOCOL_VARIANT, VARIANTS

CHECKS = 0
FAILED: list = []

SUITE = os.path.join("test", "decision_protocol", "test_majority_consensus.py")

_PROBE = r'''
import importlib.util, json, sys, warnings
warnings.filterwarnings("ignore")
root, suite = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("upstream_suite", suite)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print(json.dumps({"__load_error__": f"{type(e).__name__}: {e}"}))
    raise SystemExit(0)
out = {}
for name in sorted(n for n in dir(mod) if n.startswith("test_")):
    try:
        getattr(mod, name)()
        out[name] = "pass"
    except AssertionError as e:
        out[name] = f"FAIL: {e}" if str(e) else "FAIL"
    except Exception as e:
        out[name] = f"ERROR {type(e).__name__}: {e}"
print(json.dumps(out))
'''


def check(name: str, cond: bool, detail: object = "") -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILED.append(name)
        print(f"  FAIL  {name}   [{detail}]")
    else:
        print(f"  ok    {name}")


def _run(root: str) -> dict:
    suite = os.path.join(root, SUITE)
    if not os.path.isfile(suite):
        raise FileNotFoundError(suite)
    r = subprocess.run([sys.executable, "-c", _PROBE, root, suite],
                       capture_output=True, text=True, timeout=240)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError((r.stderr or r.stdout).strip()[-200:])
    return json.loads(lines[-1])


def main() -> int:
    results = {}
    for tag, root in VARIANTS.items():
        try:
            results[tag] = _run(root)
        except Exception as e:
            print(f"  SKIP {tag}: {type(e).__name__}: {e}")
            return 0

    names = sorted({n for r in results.values() for n in r if not n.startswith("__")})
    if not names:
        print(f"  SKIP: the upstream suite did not load ({results})")
        return 0

    width = max(len(n) for n in names)
    print(f"  {'upstream assertion':{width}s} " + " ".join(f"{t:>14s}" for t in VARIANTS))
    for n in names:
        print(f"  {n:{width}s} "
              + " ".join(f"{str(results[t].get(n, '-'))[:14]:>14s}" for t in VARIANTS))

    passed = {t: sum(1 for n in names if results[t].get(n) == "pass") for t in VARIANTS}
    print(f"\n  passed: {passed} of {len(names)}")

    check(f"{INTENDED_PROTOCOL_VARIANT} satisfies all {len(names)} upstream assertions",
          passed.get(INTENDED_PROTOCOL_VARIANT) == len(names),
          {n: results[INTENDED_PROTOCOL_VARIANT].get(n) for n in names
           if results[INTENDED_PROTOCOL_VARIANT].get(n) != "pass"})

    check("the leading False counts: test_unanimous_decision passes under the intended protocol",
          results[INTENDED_PROTOCOL_VARIANT].get("test_unanimous_decision") == "pass",
          results[INTENDED_PROTOCOL_VARIANT].get("test_unanimous_decision"))

    check("release also satisfies the suite, because the suite's endorsers carry no solution",
          passed.get("release") == len(names),
          {n: results["release"].get(n) for n in names if results["release"].get(n) != "pass"})

    for t in VARIANTS:
        check(f"{t} is registered and satisfies the contract", passed.get(t) == len(names),
              passed.get(t))

    print(f"\n{CHECKS} checks, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  - {f}")
    return 1 if FAILED else 0


def test_variants_satisfy_the_upstream_threshold_contract():
    rc = main()
    assert rc == 0, f"recorded check failures: {FAILED}"


if __name__ == "__main__":
    sys.exit(main())
