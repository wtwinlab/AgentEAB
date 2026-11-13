"""Is the FORD instruction fix behaviourally identical everywhere it must be? Zero model calls."""
import collections, importlib.util, json, os, pathlib, sys
R = pathlib.Path(os.environ.get("COLLECTION_ROOT")
                 or pathlib.Path(__file__).resolve().parents[3])
HERE = R/"experiments/02_controlled_sharing/code"
sys.path[:0] = [str(HERE), str(R/"code")]
import matching_tasks as MT; MT.install()
import answer_extract as AX
import arm_transform as OLD

CAND = os.environ.get("CANDIDATE_ARM_TRANSFORM") or str(HERE / "arm_transform.py")
spec = importlib.util.spec_from_file_location("arm_transform_new", CAND)
NEW = importlib.util.module_from_spec(spec); spec.loader.exec_module(NEW)

RAW = R/"data/records/controlled_sharing"
MATCH = ("wdc_products","geographic_settlements")
print("%-18s %-24s %-9s %7s %9s %s" % ("protocol","benchmark","arm","calls","identical","verdict"))
summary = collections.Counter()
for proto in ("bccs","du_mad","madm2_objective","moa","ford"):
    for bench in ("gsm8k","strategyqa","mmlu_pro","wdc_products","geographic_settlements"):
        kind = AX.KIND_BY_BENCHMARK.get(bench)
        task = MT.BENCH if bench in MATCH else bench
        for arm in ("FULL","NOPEER","ANSONLY","UNIFSEL"):
            n=same=0
            for p in sorted(RAW.glob("%s__%s__%s__*.jsonl" % (bench,proto,arm)))[:1]:
                for line in p.read_text(encoding="utf-8",errors="replace").splitlines():
                    if not line.strip() or n>=120: continue
                    try: r=json.loads(line)
                    except Exception: continue
                    if r.get("reused"): continue
                    seed=r.get("seed",20261019)
                    for ci,c in enumerate((r.get("record") or {}).get("calls") or []):
                        msgs=c.get("messages_sent") or []
                        if not msgs: continue
                        a,_=OLD.transform(proto,arm,msgs,kind=kind,benchmark=task,call_seq=ci,seed=seed)
                        b,_=NEW.transform(proto,arm,msgs,kind=kind,benchmark=task,call_seq=ci,seed=seed)
                        n+=1; same += 1 if a==b else 0
            if not n: continue
            expect_change = (proto=="ford" and bench in ("gsm8k","mmlu_pro")
                             and arm in ("NOPEER","ANSONLY"))
            allsame = (same==n)
            if expect_change:
                verdict = "FIXED (differs, as intended)" if not allsame else "PROBLEM: unchanged"
            else:
                verdict = "equivalent" if allsame else "PROBLEM: behaviour changed"
            summary[verdict.split(":")[0].split(" ")[0]] += 1
            print("%-18s %-24s %-9s %7d %8d%% %s" % (proto,bench,arm,n,round(100*same/n),verdict))
print("\nsummary: %s" % dict(summary))
print("the fix must change FORD/NOPEER and FORD/ANSONLY on gsm8k and mmlu_pro, and nothing else.")
