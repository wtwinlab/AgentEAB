import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cn_override as CN
CN.D.BENCH_ORDER = ["gpqa"]
import ext_collect as XC

if __name__ == "__main__":
    (CN.OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(CN.OUT / "logs" / "cn_run_invocations.jsonl", "a") as fh:
        fh.write(json.dumps(dict(CN.stamp(), module="cg_run", argv=sys.argv[1:])) + "\n")
    sys.argv = ["ext_collect.py"] + sys.argv[1:]
    raise SystemExit(XC.main())
