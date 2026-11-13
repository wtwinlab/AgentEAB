import importlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sb_override as SB

if __name__ == "__main__":
    mod = sys.argv[1]
    sys.argv = [mod + ".py"] + sys.argv[2:]
    SB.OUT.mkdir(parents=True, exist_ok=True)
    (SB.OUT / "logs").mkdir(exist_ok=True)
    with open(SB.OUT / "logs" / "sb_run_invocations.jsonl", "a") as fh:
        fh.write(json.dumps(dict(SB.stamp(), module=mod, argv=sys.argv[1:])) + "\n")
    m = importlib.import_module(mod)
    if hasattr(m, "MANIFEST"):
        m.MANIFEST = SB.OUT / f"manifest_{SB._LANE}.jsonl"
    raise SystemExit(m.main())
