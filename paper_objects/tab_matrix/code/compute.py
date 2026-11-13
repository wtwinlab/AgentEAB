r"""BIRD debate populations of Section IV (sec:setup) and Table II (tab:matrix): the 198 sampled questions,
their databases and difficulty mix, and the 1,531 scored development questions; the four Du-MAD and MAD-M2
configurations on the 1,531 questions are estimated alongside.

    python3 code/compute.py --out output

Reads only ../inputs. Writes <out>/values.json (pp, %, counts as printed) and <out>/fulldev_effects.json
(every estimate with both 95% intervals).
"""
import argparse
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402

INP = HERE.parent / "inputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    R = E.Rows(INP)
    V, full = {}, {}

    def put(prefix, b, ci=True):
        V[prefix] = b["est"]
        if ci:
            V[prefix + "_q_lo"], V[prefix + "_q_hi"] = b["q_ci"]
            V[prefix + "_db_lo"], V[prefix + "_db_hi"] = b["db_ci"]
        full[prefix] = {k: b[k] for k in ("est", "q_ci", "db_ci", "n_questions", "detected_both", "mde80_q", "mde80_db")}

    # ---- 1,531 questions: levels, last-round difference, initial-answer difference, interaction effect
    for g in ("4cell", "DuMAD_2cell", "MADM2_2cell", "cell_du_mad|llama3-8b", "cell_du_mad|qwen2.5-7b",
              "cell_madm2_objective|llama3-8b", "cell_madm2_objective|qwen2.5-7b"):
        cells = E.GROUPS[g]
        for side in ("with", "without"):
            for m in ("r0", "last"):
                put(f"{g}|{m}_{side}", R.fulldev_level(cells, side, m), ci=False)
        put(f"{g}|last_diff", R.boot(R.fulldev_items(cells, "last")))
        put(f"{g}|r0_diff", R.boot(R.fulldev_items(cells, "r0")))
        put(f"{g}|dpeer", R.boot(R.fulldev_items(cells, "E1")))
    b = R.boot(R.fulldev_items(E.C4, "E1"))
    V["n_questions_fulldev"] = b["n_questions"]
    V["fulldev_mde80_q"], V["fulldev_mde80_db"] = b["mde80_q"], b["mde80_db"]
    V["DuMAD_2cell|last_diff_abs_1dp"] = abs(V["DuMAD_2cell|last_diff"])
    V["n_questions_remaining"] = len(R.NEWU)

    # ---- 198 sampled questions (per-run scores of Table IV)
    S = E.Sample198(INP)
    for term in ("last_diff", "r0_diff", "dpeer", "later_rounds_with"):
        for g in ("4cell", "DuMAD_2cell"):
            b = S.evaluate(term, E.GROUPS[g])
            put(f"s198|{g}|{term}", b)
    b = S.evaluate("dpeer", E.C4)
    V["s198_mde80_q"], V["s198_mde80_db"] = b["mde80_q"], b["mde80_db"]
    V["n_questions_sample"] = b["n_questions"]

    # ---- sample composition (Section IV): difficulty counts and over-representation of challenging
    dev = list(csv.DictReader(open(INP / "bird_dev_questions.csv")))
    diff_of = {r["unit"]: r["difficulty"] for r in dev}
    units = S.evaluate("dpeer", E.C4)["units"]
    for lev in ("simple", "moderate", "challenging"):
        V[f"sample_{lev}"] = sum(diff_of[u] == lev for u in units)
    share_s = V["sample_challenging"] / len(units)
    share_d = sum(r["difficulty"] == "challenging" for r in dev) / len(dev)
    V["challenging_overrepresentation"] = round(share_s / share_d, 4)
    V["n_bird_dev"] = len(dev)
    V["n_sample_databases"] = len({r["db_id"] for r in dev if r["unit"] in set(units)})
    lists = json.loads((INP / "question_lists.json").read_text())
    V["n_sample_drawn"] = len(units) + len(lists["excluded_from_200"])
    V["n_sample_reference_timeouts"] = len(lists["excluded_from_200"])
    meta = json.loads((INP / "eval_meta.json").read_text())
    V["n_fulldev_reference_failures"] = len(meta["gold_failed_units"])

    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True))
    (out / "fulldev_effects.json").write_text(json.dumps(full, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
