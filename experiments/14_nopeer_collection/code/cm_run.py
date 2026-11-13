r"""python cm_run.py --run --workers N [--smoke K]   (collect_controlled under cm_override)"""
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cm_override as CM                                                 # noqa: E402

if __name__ == "__main__":
    (CM.OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(CM.OUT / "logs" / "cm_run_invocations.jsonl", "a") as fh:
        fh.write(json.dumps(dict(argv=sys.argv[1:], bench=CM.CC.BENCH_ORDER, protos=CM.CC.PROTOS,
                                 cn_sha16=CM.CN.override_digest(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")
    sys.argv = ["collect_controlled.py"] + sys.argv[1:]
    raise SystemExit(CM.CC.main())
