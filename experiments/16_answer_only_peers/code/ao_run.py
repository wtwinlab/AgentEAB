import importlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ao_override as AO

if __name__ == "__main__":
    mod = sys.argv[1]
    sys.argv = [mod + ".py"] + sys.argv[2:]
    (AO.OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(AO.OUT / "logs" / "ao_run_invocations.jsonl", "a") as fh:
        fh.write(json.dumps(dict(AO.stamp(), module=mod, argv=sys.argv[1:])) + "\n")
    m = importlib.import_module(mod)
    raise SystemExit(m.main())
