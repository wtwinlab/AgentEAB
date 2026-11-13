r"""Run a frozen collector/evaluator module under the strong-backbone override.

    python sb_run.py ext_collect --plan
    python sb_run.py ext_collect --run --benchmark sql_bird_v2 --smoke 40 --workers 96
    python sb_run.py ext_eval_sql [--allow-partial]
    python sb_run.py ext_sql_verdicts

The module is imported by name (not via runpy), so its functions keep their module identity and
fork-started workers inherit the patched design. Linux (fork) only for the collector.
"""
import importlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sb_override as SB                                                 # noqa: E402

if __name__ == "__main__":
    mod = sys.argv[1]
    sys.argv = [mod + ".py"] + sys.argv[2:]
    SB.OUT.mkdir(parents=True, exist_ok=True)
    (SB.OUT / "logs").mkdir(exist_ok=True)
    with open(SB.OUT / "logs" / "sb_run_invocations.jsonl", "a") as fh:
        fh.write(json.dumps(dict(SB.stamp(), module=mod, argv=sys.argv[1:])) + "\n")
    m = importlib.import_module(mod)
    if hasattr(m, "MANIFEST"):
        # per-lane manifest: the frozen write_manifest uses one fixed temp name (same race as STATUS)
        m.MANIFEST = SB.OUT / f"manifest_{SB._LANE}.jsonl"
    raise SystemExit(m.main())
