"""Supplementary Table tab:s-failures: why a final answer on BIRD is wrong, with peers.

    python3 code/compute.py --out output

Reads only ../inputs/failure_classes_runs.csv: one row per run with peers (198 BIRD questions x 2
repeats per cell), with its class:
  correct     the protocol's final answer is correct;
  generation  no reply of the debate holds a correct query under the lenient extractor;
  format      a reply holds a correct query that the protocol's own extractor misses;
  selection   the protocol's own extractor finds a correct query, but the vote, stop rule, aggregator,
              or return step does not return it.
The replies considered are every agent's initial reply and last-round reply. Shares are % of runs.
Also reads ../inputs/failure_classes_candidate_sets.csv (class counts per cell under the initial and
last-round candidate set and under every round) for the largest class move between the two sets.
"""
import argparse
import collections
import csv
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
CELLS = [("du_mad", "llama3-8b"), ("du_mad", "qwen2.5-7b"), ("madm2_objective", "llama3-8b"),
         ("madm2_objective", "qwen2.5-7b"), ("bccs", "llama3-8b"), ("bccs", "qwen2.5-7b"),
         ("moa", "roster"), ("ford", "roster")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    cnt = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(INP / "failure_classes_runs.csv")):
        cnt[(r["protocol"], r["backbone"])][r["failure_class"]] += 1
    V = {}
    for c in CELLS:
        n = sum(cnt[c].values())
        tag = f"{c[0]}_{c[1]}".replace(".", "").replace("-", "")
        V[f"{tag}_runs"] = n
        for k in ("correct", "generation", "format", "selection"):
            V[f"{tag}_{k}_pct"] = round(100 * cnt[c][k] / n, 6)
    gen = [V[k] for k in V if k.endswith("_generation_pct")]
    V["generation_min_pct"], V["generation_max_pct"] = min(gen), max(gen)
    sel = [V["moa_roster_selection_pct"], V["bccs_qwen257b_selection_pct"]]
    V["selection_moa_bccs_qwen_min_pct"], V["selection_moa_bccs_qwen_max_pct"] = min(sel), max(sel)
    # candidate set of every round against the initial and last-round replies: largest class move (pp)
    sets = collections.defaultdict(dict)
    for r in csv.DictReader(open(INP / "failure_classes_candidate_sets.csv")):
        sets[(r["protocol"], r["backbone"])][r["candidate_set"]] = r
    moves = []
    for cell, s in sorted(sets.items()):
        a, b = s["initial_and_last_round"], s["every_round"]
        for k in ("correct", "a_generation", "b_format", "c_selection"):
            moves.append(abs(100 * (int(b[k]) - int(a[k])) / int(a["runs"])))
    V["every_round_max_class_move_pp"] = round(max(moves), 6)
    V["every_round_cells_agree_with_runs"] = int(all(
        int(s["initial_and_last_round"]["a_generation"]) == cnt[c]["generation"]
        and int(s["initial_and_last_round"]["correct"]) == cnt[c]["correct"]
        for c, s in zip(CELLS, [sets[k] for k in (("Du-MAD", "llama3-8b"), ("Du-MAD", "qwen2.5-7b"),
                                                  ("MAD-M2", "llama3-8b"), ("MAD-M2", "qwen2.5-7b"),
                                                  ("BCCS", "llama3-8b"), ("BCCS", "qwen2.5-7b"),
                                                  ("MoA", "roster"), ("FORD", "roster"))])))
    (out / "values.json").write_text(json.dumps(dict(sorted(V.items())), indent=1) + "\n")
    print(len(V), "values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
