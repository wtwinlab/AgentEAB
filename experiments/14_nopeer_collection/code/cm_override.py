import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("SB_BACKBONES", "llama3-8b,qwen2.5-7b")
os.environ.setdefault("SB_PROTOS", "bccs,du_mad")
import cn_override as CN
import endpoints_all_matching as EP

_ports = {"llama3-8b-g1": int(os.environ.get("CN_PORT_L8", "9311")),
          "qwen25-7b-g1b": int(os.environ.get("CN_PORT_Q7", "9312"))}
EP.HOST = "127.0.0.1"


def _resolve_routes():
    p = dict(EP.DEFAULT_PORTS)
    p.update(_ports)
    return p, {}


EP.resolve_routes = _resolve_routes

import stop_control as SCM
import collect_controlled as CC
import unit_driver as UD

OUT = pathlib.Path(os.environ.get("CM_OUT", CN.ROOT / "experiments/14_nopeer_collection/data/main"))
LANE = os.environ.get("SB_LANE_TAG") or "lane"
SCM.OUT, SCM.CONTROL, SCM.STATUS = OUT, OUT / "CONTROL.json", OUT / f"STATUS_{LANE}.json"
CC.OUT, CC.RAW = OUT, OUT / "raw" / LANE
CC.BENCH_ORDER = [b for b in os.environ.get("CM_BENCH", "gsm8k,strategyqa,mmlu_pro").split(",") if b]
CC.PROTOS = [p for p in os.environ.get("CM_PROTOS", "bccs,du_mad").split(",") if p]
CC.ARMS = ["NOPEERC"]
_bbs = [b for b in os.environ.get("CM_BACKBONES", "llama3-8b,qwen2.5-7b").split(",") if b]
CC.PANELS = {p: list(_bbs) for p in CC.PROTOS}
CC.COST.update({p: CC.COST.get(p, 1.0) for p in CC.PROTOS})

_orig_supported = UD.supported


def _supported(benchmark, protocol, arm):
    return _orig_supported(benchmark, protocol, "NOPEER" if arm == "NOPEERC" else arm)


UD.supported = _supported
_orig_item_seed = CC.item_seed


def _item_seed(item):
    return _orig_item_seed(dict(item, arm="NOPEER" if item["arm"] == "NOPEERC" else item["arm"]))


CC.item_seed = _item_seed
_orig_ud_run_unit = UD.run_unit


def _ud_run_unit(benchmark, protocol, unit, alias, *, arm="FULL", **kw):
    if arm == "NOPEERC":
        assert protocol in ("du_mad", "bccs")
        arm = "NOPEER"
    return _orig_ud_run_unit(benchmark, protocol, unit, alias, arm=arm, **kw)


UD.run_unit = _ud_run_unit
_orig_run_item = CC._run_item


def _run_item(item):
    CN.MODE.clear()
    CN.MODE["clean"] = item["arm"] == "NOPEERC"
    rec = _orig_run_item(item)
    rec["cn"] = dict(override_sha16=CN.override_digest(), harness="collect_controlled",
                     bccs_conflicting_relabelled=CN.MODE.get("bccs_conflicting_relabelled", 0))
    if rec.get("status") == "ok":
        sent = CN.MODE.get("sent") or []
        calls = rec["record"].get("calls") or []
        if len(sent) == len(calls):
            for c, s in zip(calls, sent):
                if int(c.get("round") or 0) >= 1:
                    c["messages_after_arm"] = s
        else:
            rec["cn"]["sent_count_mismatch"] = [len(sent), len(calls)]
    return rec


CC._run_item = _run_item
