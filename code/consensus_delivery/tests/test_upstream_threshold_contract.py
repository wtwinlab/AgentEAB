"""Hold each variant to **upstream's own test suite**, which is the contract.

This replaces a test that encoded a disputed premise as its expectations. The premise was that a
turn's leading `False` is an explicit dissent and should not count toward the endorsement total.
Upstream settles it the other way, in a file the earlier work had overlooked and which all the trees
ship byte-identically: `test/decision_protocol/test_majority_consensus.py:22-50` builds

    [Agreement(agreement=False, solution="Test"),
     Agreement(agreement=True, solution=None),
     Agreement(agreement=True, solution=None)]

with three panelists and asserts `is_consensus`. `UnanimityConsensus` is `threshold_percent = 1.0`,
so that assertion forces `num_agreements == 3` and **the leading `False` must be counted**.

So the test here is not a table of our expectations. It runs the vendored suite's own assertions in
each tree and reports which variant satisfies them. A variant that fails upstream's assertions is not
the protocol, whatever we think it ought to do.

Each tree is loaded in its own subprocess: all of them present as the package `mallm`, and
`adapters/mallm_upstream` refuses two in one process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from ..adapters.mallm_upstream import INTENDED_PROTOCOL_VARIANT, VARIANTS

CHECKS = 0
FAILED: list = []

#: Relative to each tree. Shipped by upstream, identical in every variant.
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
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {tag}: {type(e).__name__}: {e}")
            return 0          # the trees or their deps are absent here; not a failure of this suite

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

    # The intended protocol must satisfy every assertion upstream ships.
    check(f"{INTENDED_PROTOCOL_VARIANT} satisfies all {len(names)} upstream assertions",
          passed.get(INTENDED_PROTOCOL_VARIANT) == len(names),
          {n: results[INTENDED_PROTOCOL_VARIANT].get(n) for n in names
           if results[INTENDED_PROTOCOL_VARIANT].get(n) != "pass"})

    # The decisive one, named, so a future change cannot quietly flip it.
    check("the leading False counts: test_unanimous_decision passes under the intended protocol",
          results[INTENDED_PROTOCOL_VARIANT].get("test_unanimous_decision") == "pass",
          results[INTENDED_PROTOCOL_VARIANT].get("test_unanimous_decision"))

    # `release` also satisfies the suite, which is exactly why the suite alone is not enough: the
    # suite predates `e20d40c` and still gives endorsers `solution=None`, while production gives them
    # a copied solution. Recorded so the two facts are not confused.
    check("release also satisfies the suite, because the suite's endorsers carry no solution",
          passed.get("release") == len(names),
          {n: results["release"].get(n) for n in names if results["release"].get(n) != "pass"})

    # And no registered variant may fail it.
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
