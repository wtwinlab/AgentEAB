r"""python cn_run.py ext_collect --run --benchmark sql_bird_v2 --workers N [--smoke K]   (under cn_override)"""
import importlib, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cn_override as CN                                                 # noqa: E402

if __name__ == "__main__":
    mod = sys.argv[1]
    sys.argv = [mod + ".py"] + sys.argv[2:]
    (CN.OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(CN.OUT / "logs" / "cn_run_invocations.jsonl", "a") as fh:
        fh.write(json.dumps(dict(CN.stamp(), module=mod, argv=sys.argv[1:])) + "\n")
    m = importlib.import_module(mod)
    raise SystemExit(m.main())
