r"""GPQA (ext harness) no-peer: cn_override with BENCH_ORDER = [gpqa].
    python cg_run.py --run --benchmark gpqa --workers N [--smoke K]"""
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cn_override as CN                                                 # noqa: E402
CN.D.BENCH_ORDER = ["gpqa"]
import ext_collect as XC                                                 # noqa: E402

if __name__ == "__main__":
    (CN.OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(CN.OUT / "logs" / "cn_run_invocations.jsonl", "a") as fh:
        fh.write(json.dumps(dict(CN.stamp(), module="cg_run", argv=sys.argv[1:])) + "\n")
    sys.argv = ["ext_collect.py"] + sys.argv[1:]
    raise SystemExit(XC.main())
