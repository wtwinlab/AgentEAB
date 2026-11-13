"""QA return rules replayed on fixed post-update answers.

    python3 code/compute.py --out DIR     # -> DIR/values.json

Input: inputs/qa_vote_replay_sets.csv, one row per set of three post-update answers
(task, fixed set of initial responses `prefix_id`, update repeat), with the outcome of the
original plurality (unreadable answers count as one answer category; the set abstains if that
category wins) and of the readable-only plurality (unreadable answers removed before voting).
Rows with delivery "native_ppl_high (released)" are the protocol's own peer choice and are the
population the paper reports.

The interval resamples tasks (4,000 draws, Python `random.Random(20261013)`), each task weighted
by its number of sets, in the row order of the input file.
"""
import argparse
import collections
import csv
import json
import pathlib
import random

INP = pathlib.Path(__file__).resolve().parent.parent / "inputs" / "qa_vote_replay_sets.csv"
DRAWS, SEED = 4000, 20261013


def boot_diff(pairs, alpha=0.05):
    byt = collections.defaultdict(list)
    for t, a, b in pairs:
        byt[t].append(a - b)
    keys = list(byt)
    per = {k: sum(v) / len(v) for k, v in byt.items()}
    w = {k: len(v) for k, v in byt.items()}
    point = sum(per[k] * w[k] for k in keys) / sum(w.values())
    rnd = random.Random(SEED)
    sims = []
    for _ in range(DRAWS):
        x = y = 0.0
        for _ in range(len(keys)):
            k = keys[rnd.randrange(len(keys))]
            x += per[k] * w[k]
            y += w[k]
        sims.append(x / y if y else 0.0)
    sims.sort()
    return point, sims[max(0, int(alpha / 2 * DRAWS) - 1)], sims[min(DRAWS - 1, int((1 - alpha / 2) * DRAWS) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    b = lambda x: x == "True"                                                # noqa: E731
    rows = [r for r in csv.DictReader(open(INP)) if r["delivery"] == "native_ppl_high (released)"]
    V = dict(sets=len(rows),
             tasks=len({(r["benchmark"], r["task"]) for r in rows}),
             initial_sets=len({r["prefix_id"] for r in rows}),
             initial_sets_per_task=max(collections.Counter((r["benchmark"], r["task"]) for r in
                                                           {r["prefix_id"]: r for r in rows}.values()).values()),
             repeats=len({r["repeat"] for r in rows}))
    fails = [r for r in rows if not b(r["released_correct"])]
    rec = [r for r in fails if b(r["available"]) and b(r["readable_only_correct"])]
    V["recovered"] = len(rec)
    V["recovered_that_had_abstained"] = sum(b(r["released_abstained"]) for r in rec)
    V["successes_overturned"] = sum(1 for r in rows if b(r["released_correct"]) and not b(r["readable_only_correct"]))
    pt, lo, hi = boot_diff([((r["benchmark"], r["task"]), float(b(r["readable_only_correct"])),
                             float(b(r["released_correct"]))) for r in rows])
    V["gain_pp"], V["gain_lo_pp"], V["gain_hi_pp"] = 100 * pt, 100 * lo, 100 * hi
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
