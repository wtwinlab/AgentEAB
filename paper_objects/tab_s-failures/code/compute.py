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
