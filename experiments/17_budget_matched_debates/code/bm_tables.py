r"""Per-run and per-draft tables of paper object fig_debate-budget, from the scored collection.

    python bm_tables.py [<scored output dir>] [<tables dir>]

Reads rows.jsonl, drafts.jsonl and exclusions.json written by bm_score.py (default <BM_OUT>/output)
and writes debate_runs.csv (one debate run per configuration, question and repeat; failed runs carry
status "failed") and drafts.csv (one draft: execution result label within its question, empty when
the query does not execute; correctness; tokens).
"""
import csv
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(os.environ.get("BM_OUT", HERE.parents[2] / "experiments/17_budget_matched_debates/data"))
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "output"
DST = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "tables"
CONF = {"FULL": "3x2", "R3_FULL": "3x3", "A5_FULL": "5x2", "R4_FULL": "3x4"}
SET = {"DRAFTS20": "protocol", "T07_DRAFTS20": "t0.7"}


def main():
    DST.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (SRC / "rows.jsonl").open()]
    excl = json.loads((SRC / "exclusions.json").read_text())
    assert not excl["gold_failed_units"]
    db_of = {r["unit"]: r["db"] for r in rows}
    out = []
    for r in rows:
        if r["arm"] in CONF:
            out.append(dict(protocol=r["protocol"], backbone=r["backbone"], config=CONF[r["arm"]], question=r["unit"],
                            db_id=r["db"], repeat=r["repeat"], status="ok", calls=r["n_calls"],
                            prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
                            last_vote_correct=int(bool(r["last"])), final_correct=int(bool(r["final"]))))
    for e in excl["failed_runs"]:
        if e["arm"] in CONF:
            out.append(dict(protocol=e["protocol"], backbone=e["backbone"], config=CONF[e["arm"]], question=e["unit"],
                            db_id=db_of[e["unit"]], repeat=e["repeat"], status="failed", calls="", prompt_tokens="",
                            completion_tokens="", last_vote_correct="", final_correct=""))
    out.sort(key=lambda d: (d["protocol"], d["backbone"], d["config"], d["question"], d["repeat"]))
    with open(DST / "debate_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    dr = []
    for line in (SRC / "drafts.jsonl").open():
        d = json.loads(line)
        ids = {}
        for i, s in enumerate(d["sig"]):
            dr.append(dict(protocol=d["protocol"], backbone=d["backbone"], draft_set=SET[d["arm"]], question=d["unit"],
                           draft=i, result="" if s is None else ids.setdefault(s, len(ids)),
                           correct=int(bool(d["correct"][i])), prompt_tokens=d["prompt_tokens"][i],
                           completion_tokens=d["completion_tokens"][i]))
    dr.sort(key=lambda d: (d["protocol"], d["backbone"], d["draft_set"], d["question"], d["draft"]))
    with open(DST / "drafts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dr[0]))
        w.writeheader()
        w.writerows(dr)
    print(len(out), "debate runs,", len(dr), "drafts")


if __name__ == "__main__":
    main()
