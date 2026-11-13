r"""ComEM formal runner. Official ComEM.__call__ (ranking "matching", topK=4) on every
test anchor of the 8 datasets; ranker-only scores on the entity-disjoint dev anchors (no selector
call). Records the boundary and the official single-match control. Labels are not loaded.

    cd <workdir> && OPENAI_BASE_URL=... OPENAI_API_KEY=EMPTY CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=<comem> \
      python -u run_comem_formal.py <llm4em_dir> <comem_dev.json> <selector_model> <out_dir>
"""
import hashlib
import json
import math
import os
import sys
import time

import pandas as pd

import src.utils as U

COUNT = {"forward": 0, "seconds": 0.0}
for name in ("cal_log_probs", "generate"):
    orig = getattr(U.Seq2SeqWrapper, name)

    def counted(self, *a, _o=orig, **k):
        t0 = time.time()
        r = _o(self, *a, **k)
        COUNT["forward"] += 1
        COUNT["seconds"] += time.time() - t0
        return r
    setattr(U.Seq2SeqWrapper, name, counted)

from src.compound import ComEM  # noqa: E402

llm_dir, dev_path, sel_model, out_dir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
dev = json.load(open(dev_path))
t0 = time.time()
compound = ComEM(ranking_model_name="flan-t5-xl", selecting_model_name=sel_model)
load_s = time.time() - t0
cur = {}
orig_score = compound.ranker.score


def rec_score(instance, use_prob=False):
    s = orig_score(instance, use_prob=use_prob)
    cur.setdefault("scores", s)
    return s


compound.ranker.score = rec_score
orig_cc = compound.selector.chat_complete


def rec_cc(*a, **k):
    t1 = time.time()
    try:
        r = orig_cc(*a, **k)
    except Exception as e:                                                 # noqa: BLE001
        cur["select_error"] = repr(e)
        raise
    cur["select"] = {"prompt": k["messages"][0]["content"], "params": {x: v for x, v in k.items() if x != "messages"},
                     "content": r.choices[0].message.content, "usage": r.usage.model_dump() if r.usage else None,
                     "latency_s": round(time.time() - t1, 3)}
    return r


compound.selector.chat_complete = rec_cc
gate_done = False
for dataset in sorted(dev):
    df = pd.read_csv(f"{llm_dir}/{dataset}.csv", dtype={"id_left": str, "id_right": str})
    groups = df.groupby("id_left", sort=False)[["record_left", "record_right", "id_right"]].apply(lambda x: x.to_dict("list")).to_dict()
    with open(f"{out_dir}/{dataset}_test.jsonl", "w") as fo:
        for aid in sorted(groups, key=lambda x: int(x) if x.isdigit() else x):
            g = groups[aid]
            inst = {"anchor": g["record_left"][0], "candidates": g["record_right"]}
            cur = {}
            c0 = dict(COUNT); t1 = time.time(); status, preds = "COMPLETED", None
            try:
                preds = compound(inst, topK=4)
            except Exception as e:                                         # noqa: BLE001
                status = "SELECT_FAILED" if "select_error" in cur else "TASK_FAILED"
                cur["error"] = repr(e)
            fwd = COUNT["forward"] - c0["forward"]; sec = COUNT["seconds"] - c0["seconds"]
            sc = cur.get("scores")
            topk = sorted(range(len(sc)), key=lambda j: (sc[j], j), reverse=True)[:4] if sc else None
            snap = dict(cur)
            try:
                official = compound.ranker(inst, single_match=True)
            except Exception as e:                                         # noqa: BLE001
                official = repr(e)
            cur = snap
            ctrl = [s >= max(sc) and s > 0 for s in sc] if sc else None
            cands = [inst["candidates"][j] for j in topk] if topk else None
            r = dict(dataset=dataset, id_left=aid, status=status, id_right=g["id_right"], scores=sc,
                     scores_finite=bool(sc) and all(math.isfinite(s) for s in sc), topk=topk,
                     boundary_sha256=hashlib.sha256(json.dumps(cands).encode()).hexdigest() if cands else None,
                     candidates_topk=cands, select=cur.get("select"), preds_select=preds, preds_ranker_single=ctrl,
                     preds_ranker_single_official=official, ranker_forward=fwd, ranker_seconds=round(sec, 3),
                     control_extra_forward=COUNT["forward"] - c0["forward"] - fwd, wall_s=round(time.time() - t1, 3), error=cur.get("error"))
            fo.write(json.dumps(r) + "\n"); fo.flush()
            if not gate_done:
                gate_done = True
                ok = status == "COMPLETED" and r["scores_finite"] and r["select"] is not None and ctrl == official
                print("COMEM GATE", "PASS" if ok else "FAIL", flush=True)
                if not ok:
                    sys.exit(3)
    with open(f"{out_dir}/{dataset}_dev.jsonl", "w") as fo:
        for d in dev[dataset]["dev"]:
            inst = {"anchor": d["record_left"], "candidates": d["record_right"]}
            cur = {}; c0 = dict(COUNT)
            s = compound.ranker.score(inst, use_prob=True)
            fo.write(json.dumps(dict(dataset=dataset, id_left=d["id_left"], id_right=d["id_right"], scores=s,
                                     ranker_forward=COUNT["forward"] - c0["forward"])) + "\n")
    print(dataset, "done", json.dumps(COUNT), flush=True)
print(json.dumps({"model_load_s": round(load_s, 1), "totals": COUNT}), flush=True)
