"""What peers share: field comparisons in matching, and QA debate traces.

    python3 code/compute.py --out DIR     # -> DIR/values.json, DIR/matching_levels.csv, DIR/traces.csv

Inputs (inputs/):
  amazon_google_pairs.csv  one row per recipient x split x input x candidate pair (Amazon-Google):
                           gold label, the recipient's decision ("" = unreadable), the peer
                           decision shown, prompt tokens
  dblp_acm_pairs.csv       the same for DBLP-ACM (no shown-peer column)
  qa_trace_runs.csv        one row per planned QA debate run: protocol variant, whether the trace
                           is missing, whether a correct candidate exists initially, in the final
                           pool and at any stage, final success and the endpoint state

Pair F1 is on the match class; an unreadable decision counts as a non-match. Development pair F1
selects one input per recipient, ties resolved by fewer mean prompt tokens, then input name.
"""
import argparse
import collections
import csv
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
MENU = ("no_peer", "peer_decision", "decision_plus_fields", "full_explanation")
ORDER = {a: i for i, a in enumerate(sorted(MENU))}
NAME = {"gemma-4-26b": "gemma", "gpt-oss-120b": "gptoss", "qwen3-32b": "qwen3"}
VARIANTS = ("du-mad", "bccs", "dmad", "dov_debate", "dov_vote", "madm2_naive", "madm2_objective",
            "du-mad-roster", "madm2_objective_roster")


def prf(rows):
    tp = sum(1 for r in rows if r["pred"] == "match" and r["gold"] == "match")
    fp = sum(1 for r in rows if r["pred"] == "match" and r["gold"] != "match")
    fn = sum(1 for r in rows if r["pred"] != "match" and r["gold"] == "match")
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(f1=100 * (2 * p * rc / (p + rc) if p + rc else 0.0), recall=100 * rc, tp=tp, fp=fp, fn=fn, n=len(rows))


def matching(name, V, levels):
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(INP / f"{name}_pairs.csv")):
        rows[(r["recipient"], r["split"], r["input"])].append(r)
    tag = "ag" if name == "amazon_google" else "da"
    for m in NAME:
        dev = {a: prf(rows[(m, "dev", a)]) for a in MENU}
        tok = {a: sum(int(r["prompt_tokens"]) for r in rows[(m, "dev", a)]) / len(rows[(m, "dev", a)]) for a in MENU}
        best = max(d["f1"] for d in dev.values())
        chosen = sorted([a for a in MENU if dev[a]["f1"] == best], key=lambda a: (tok[a], ORDER[a]))[0]
        test = {a: prf(rows[(m, "test", a)]) for a in MENU}
        for a in MENU:
            levels.append(dict(dataset=name, recipient=m, input=a, dev_f1=dev[a]["f1"], selected=a == chosen, **test[a]))
            k = f"{tag}_{NAME[m]}_{a}"
            V[k + "_f1"] = test[a]["f1"]
            V[k + "_pairs"] = test[a]["n"]
            if tag == "ag":
                for x in ("tp", "fp", "fn", "recall"):
                    V[f"{k}_{x}"] = test[a][x]
        V[f"{tag}_{NAME[m]}_selected_is_no_peer"] = int(chosen == "no_peer")
        V[f"{tag}_{NAME[m]}_selected_is_peer_decision"] = int(chosen == "peer_decision")
        V[f"{tag}_{NAME[m]}_dev_and_test_best_is_no_peer"] = int(
            chosen == "no_peer" and max(test, key=lambda a: test[a]["f1"]) == "no_peer")
        V[f"{tag}_{NAME[m]}_selected_minus_full"] = test[chosen]["f1"] - test["full_explanation"]["f1"]
        V[f"{tag}_{NAME[m]}_best_minus_selected"] = max(t["f1"] for t in test.values()) - test[chosen]["f1"]
        V[f"{tag}_{NAME[m]}_decision_minus_fields"] = test["peer_decision"]["f1"] - test["decision_plus_fields"]["f1"]
    return rows


def amazon_google_qwen3(rows, V):
    """Paired changes of the Qwen3 judgments: decision with field comparisons against no peers."""
    ok = lambda r: (r["pred"] or "non-match") == r["gold"]                   # noqa: E731
    base = {r["pair"]: r for r in rows[("qwen3-32b", "test", "no_peer")]}
    trt = {r["pair"]: r for r in rows[("qwen3-32b", "test", "decision_plus_fields")]}
    pairs = sorted(set(base) & set(trt))
    c2w = [p for p in pairs if ok(base[p]) and not ok(trt[p])]
    w2c = [p for p in pairs if not ok(base[p]) and ok(trt[p])]
    V["ag_fields_test_pairs"] = len(pairs)
    V["ag_fields_correct_to_wrong"] = len(c2w)
    V["ag_fields_wrong_to_correct"] = len(w2c)
    V["ag_fields_c2w_true_match_to_nonmatch"] = sum(
        1 for p in c2w if base[p]["gold"] == "match" and base[p]["pred"] == "match" and trt[p]["pred"] == "non-match")
    V["ag_fields_c2w_to_unreadable"] = sum(1 for p in c2w if trt[p]["pred"] == "")
    V["ag_fields_c2w_follow_peer"] = sum(1 for p in c2w if (trt[p]["pred"] or "non-match") == trt[p]["shown_peer_decision"])
    V["ag_fields_w2c_follow_peer"] = sum(1 for p in w2c if (trt[p]["pred"] or "non-match") == trt[p]["shown_peer_decision"])
    V["ag_true_matches"] = sum(1 for r in trt.values() if r["gold"] == "match")
    V["ag_true_nonmatches"] = sum(1 for r in trt.values() if r["gold"] != "match")
    V["ag_peer_rejects_true_matches"] = sum(1 for r in trt.values() if r["gold"] == "match" and r["shown_peer_decision"] != "match")


def traces(V, out):
    rows = list(csv.DictReader(open(INP / "qa_trace_runs.csv")))
    table = []
    for v in VARIANTS + ("ALL",):
        rs = [r for r in rows if v == "ALL" or r["variant"] == v]
        saved = [r for r in rs if r["missing"] == "0"]
        it = [r for r in saved if r["availability_initial"] == "true"]
        fail = [r for r in it if r["final_success"] != "1"]
        lost = sum(r["availability_terminal_pool"] == "false" for r in fail)
        ret = sum(r["availability_terminal_pool"] == "true" for r in fail)
        unres = sum(r["availability_terminal_pool"] == "unknown" for r in fail)
        tt = [r for r in saved if r["availability_terminal_pool"] == "true"]
        unsc = sum(r["endpoint_state"] == "unscorable" for r in tt)
        miss = sum(r["endpoint_state"] in ("wrong", "recorded_no_output") for r in tt)
        d = dict(variant=v, planned=len(rs), traced=len(saved), missing=len(rs) - len(saved),
                 successes=sum(r["final_success"] == "1" for r in saved), initial_correct=len(it),
                 initial_correct_success=len(it) - len(fail), failures=len(fail), lost=lost,
                 retained=ret, unresolved=unres, loss_den=len(it) - unres, miss_num=miss,
                 miss_den=len(tt) - unsc, miss_incl_unscorable_num=sum(r["final_success"] != "1" for r in tt),
                 miss_incl_unscorable_den=len(tt), unscorable=unsc)
        table.append(d)
        for k, x in d.items():
            if k != "variant":
                V[f"tr_{v}_{k}"] = x
    a = table[-1]
    V["tr_missing_in_dov"] = V["tr_dov_debate_missing"] + V["tr_dov_vote_missing"]
    V["tr_success_initial_correct_pct"] = 100 * a["initial_correct_success"] / a["successes"]
    V["tr_loss_pct"] = 100 * a["lost"] / a["loss_den"]
    V["tr_loss_bound_lo_pct"] = 100 * a["lost"] / a["initial_correct"]
    V["tr_loss_bound_hi_pct"] = 100 * (a["lost"] + a["unresolved"]) / a["initial_correct"]
    V["tr_miss_pct"] = 100 * a["miss_num"] / a["miss_den"]
    V["tr_miss_incl_unscorable_pct"] = 100 * a["miss_incl_unscorable_num"] / a["miss_incl_unscorable_den"]
    wrong0 = [r for r in rows if r["missing"] == "0" and r["availability_initial"] == "false"]
    V["tr_initial_all_wrong"] = len(wrong0)
    V["tr_initial_all_wrong_later_correct"] = sum(r["availability_anystage"] == "true" for r in wrong0)
    V["tr_initial_all_wrong_returned"] = sum(r["final_success"] == "1" for r in wrong0)
    V["tr_initial_all_wrong_later_correct_pct"] = 100 * V["tr_initial_all_wrong_later_correct"] / len(wrong0)
    V["tr_initial_all_wrong_returned_pct"] = 100 * V["tr_initial_all_wrong_returned"] / len(wrong0)
    with open(out / "traces.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    V, levels = {}, []
    ag = matching("amazon_google", V, levels)
    matching("dblp_acm", V, levels)
    amazon_google_qwen3(ag, V)
    traces(V, out)
    V["shared_content_test_pairs"] = V["da_qwen3_no_peer_pairs"] + V["ag_qwen3_no_peer_pairs"]
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n")
    with open(out / "matching_levels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(levels[0])); w.writeheader(); w.writerows(levels)


if __name__ == "__main__":
    main()
