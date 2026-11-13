"""Stronger peers: Qwen3-32B peer responses in place of Llama-3-8B and Qwen2.5-7B ones.

    python3 code/compute.py --out DIR     # -> DIR/values.json, DIR/effects.csv, DIR/changes.csv

Inputs (inputs/):
  peer_source_outcomes.csv   recipient, peer source, condition, task, position, correct (0/1);
                             the Qwen3-32B source has no no-peer rows of its own: its baseline is
                             the no-peer response collected with the Qwen2.5-7B peer set
                             (sensitivity: the one collected with the Llama-3-8B set)
  peer_source_peers.csv      each peer's own answer, graded
  weaker_recipient_units.csv the Qwen2.5-7B recipient with Qwen3-32B peers, one row per position

Intervals resample tasks (480) with replacement, positions of a task weighted by their count.
Each part draws from one generator in a fixed order, as the source analysis does
(seed 20261101 for the three larger recipients, 20261105 for the Qwen2.5-7B recipient).
"""
import argparse
import collections
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
RECIPIENTS = ("qwen3-32b", "gpt-oss-120b", "gemma-4-26b")
NAME = {"qwen3-32b": "qwen3", "gpt-oss-120b": "gptoss", "gemma-4-26b": "gemma"}
BENCH = ("gsm8k", "strategyqa", "mmlu_pro", "gpqa")
OLD = ("llama3-8b", "qwen25-7b")
NEW = "qwen3-32b"
CONTRASTS = (("full", "no_peers"), ("answer_only", "no_peers"), ("answer_only", "full"))
SHORT = {"full - no_peers": "full_none", "answer_only - no_peers": "answers_none",
         "answer_only - full": "answers_full"}
DRAWS = 10000


def boot_diff(a, b, keys, rng):
    """Paired difference a - b over (task, position) keys; tasks resampled, weighted by positions."""
    by_task = collections.defaultdict(list)
    for k in keys:
        by_task[k[0]].append(a[k] - b[k])
    tasks = sorted(by_task)
    per = np.array([np.mean(by_task[t]) for t in tasks])
    w = np.array([len(by_task[t]) for t in tasks], dtype=float)
    pt = float(np.sum(per * w) / np.sum(w))
    d = rng.integers(0, len(tasks), size=(DRAWS, len(tasks)))
    bs = (per[d] * w[d]).sum(1) / w[d].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return pt, float(lo), float(hi), p


def holm(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_boot"])
    m, stop = len(rows), False
    for j, i in enumerate(order):
        rej = (not stop) and rows[i]["p_boot"] <= 0.05 / (m - j)
        stop = stop or not rej
        rows[i]["holm_reject"] = rej


def transitions(base, trt, keys):
    b = np.array([base[k] for k in keys]); t = np.array([trt[k] for k in keys])
    q = b.mean(); c = t[b == 0].mean(); r = (1 - t[b == 1]).mean()
    return dict(q=100 * q, c=100 * c, r=100 * r, recovered=100 * (1 - q) * c, lost=100 * q * r)


def larger_recipients(V):
    out, tb = collections.defaultdict(dict), {}
    for r in csv.DictReader(open(INP / "peer_source_outcomes.csv")):
        out[(r["recipient"], r["source"], r["condition"])][(r["task"], int(r["seat"]))] = int(r["correct"])
        tb[r["task"]] = r["benchmark"]
    for m in RECIPIENTS:
        out[(m, NEW, "no_peers")] = dict(out[(m, "qwen25-7b", "no_peers")])
        out[(m, NEW + "|sens", "no_peers")] = dict(out[(m, "llama3-8b", "no_peers")])
        for c in ("full", "answer_only"):
            out[(m, NEW + "|sens", c)] = out[(m, NEW, c)]
    rng = np.random.default_rng(20261101)
    effects, change, trans, complete = [], [], {}, {}
    for m in RECIPIENTS:
        for src in OLD + (NEW, NEW + "|sens"):
            conds = {c: out.get((m, src, c), {}) for c in ("no_peers", "full", "answer_only")}
            keys = sorted(set(conds["no_peers"]) & set(conds["full"]) & set(conds["answer_only"]))
            complete[(m, src)] = keys
            for b in BENCH + ("all",):
                kb = [k for k in keys if b == "all" or tb.get(k[0]) == b]
                for a, z in CONTRASTS:
                    pt, lo, hi, p = boot_diff(conds[a], conds[z], kb, rng)
                    effects.append(dict(recipient=m, source=src, benchmark=b, contrast=f"{a} - {z}",
                                        effect_pp=100 * pt, lo_pp=100 * lo, hi_pp=100 * hi, p_boot=round(p, 4)))
                if b == "all":
                    for a in ("full", "answer_only"):
                        trans[(m, src, a)] = transitions(conds["no_peers"], conds[a], kb)
        for b in BENCH + ("all",):
            for a, z in CONTRASTS:
                for old in OLD:
                    kk = sorted(set(complete[(m, NEW)]) & set(complete[(m, old)]))
                    kk = [k for k in kk if b == "all" or tb.get(k[0]) == b]
                    dn = {k: out[(m, NEW, a)][k] - out[(m, NEW, z)][k] for k in kk}
                    do = {k: out[(m, old, a)][k] - out[(m, old, z)][k] for k in kk}
                    pt, lo, hi, p = boot_diff(dn, do, kk, rng)
                    change.append(dict(recipient=m, benchmark=b, contrast=f"{a} - {z}", vs_old_source=old,
                                       change_pp=100 * pt, lo_pp=100 * lo, hi_pp=100 * hi, p_boot=round(p, 4)))
    famA = [e for e in effects if e["source"] == NEW and e["benchmark"] == "all"]
    famB = [c for c in change if c["benchmark"] == "all" and c["vs_old_source"] == "qwen25-7b"]
    holm(famA); holm(famB)
    for e in famA:
        k = f"eff_{NAME[e['recipient']]}_{SHORT[e['contrast']]}"
        V[k], V[k + "_lo"], V[k + "_hi"] = e["effect_pp"], e["lo_pp"], e["hi_pp"]
    V["eff_holm9_rejections"] = sum(e["holm_reject"] for e in famA)
    for e in effects:
        if e["source"] == NEW + "|sens" and e["benchmark"] == "all" and e["contrast"] != "answer_only - full":
            V[f"sens_{NAME[e['recipient']]}_{SHORT[e['contrast']]}"] = e["effect_pp"]
    for c in famB:
        k = f"chg_{NAME[c['recipient']]}_{SHORT[c['contrast']]}"
        V[k], V[k + "_lo"], V[k + "_hi"] = c["change_pp"], c["lo_pp"], c["hi_pp"]
        V[k + "_holm"] = int(c["holm_reject"])
    V["chg_holm9_rejections"] = sum(c["holm_reject"] for c in famB)
    falls = 0
    for m in RECIPIENTS:
        for a, s in (("full", "full"), ("answer_only", "answers")):
            n, o = trans[(m, NEW, a)], trans[(m, "qwen25-7b", a)]
            # Unrounded differences, and the differences of the two-decimal components that the
            # supplementary table prints (its recovered and lost columns subtract rounded values).
            V[f"chg_{NAME[m]}_{s}_drec_exact"] = n["recovered"] - o["recovered"]
            V[f"chg_{NAME[m]}_{s}_dlost_exact"] = n["lost"] - o["lost"]
            V[f"chg_{NAME[m]}_{s}_drec"] = round(round(n["recovered"], 2) - round(o["recovered"], 2), 2)
            V[f"chg_{NAME[m]}_{s}_dlost"] = round(round(n["lost"], 2) - round(o["lost"], 2), 2)
            falls += int(n["r"] < o["r"])
    V["regression_rate_falls_combinations"] = falls
    # peers' own accuracy by source and benchmark
    acc = collections.defaultdict(list)
    for r in csv.DictReader(open(INP / "peer_source_peers.csv")):
        acc[(r["source"], r["benchmark"])].append(int(r["correct"]))
    for (s, b), v in acc.items():
        V[f"peer_{s}_{b}"] = 100 * float(np.mean(v))
    V["peer_qwen3_above_both_benchmarks"] = sum(
        int(V[f"peer_{NEW}_{b}"] > max(V[f"peer_llama3-8b_{b}"], V[f"peer_qwen25-7b_{b}"])) for b in BENCH)
    return effects, change


def weaker_recipient(V):
    rows = list(csv.DictReader(open(INP / "weaker_recipient_units.csv")))
    keys = [(r["task"], int(r["seat"])) for r in rows]
    tb = {k: r["benchmark"] for k, r in zip(keys, rows)}
    y = {c: {k: int(r[f"y_{c}"]) for k, r in zip(keys, rows)} for c in ("no_peers", "answer_only", "full")}
    peer = {}
    for r in rows:
        for s in (0, 1, 2):
            peer[(r["task"], s)] = int(r[f"peer_own_seat{s}"])
    old = {c: {k: int(r[f"qwen25_peers_{c}"]) for k, r in zip(keys, rows) if r[f"qwen25_peers_{c}"] != ""}
           for c in ("full", "answer_only")}
    old["no_peers"] = y["no_peers"]
    rng = np.random.default_rng(20261105)
    sel = lambda b: [k for k in keys if b == "all" or tb[k] == b]        # noqa: E731
    con = (("answer_only", "no_peers"), ("full", "no_peers"), ("answer_only", "full"))
    for b in BENCH + ("all",):
        kb = sel(b)
        pt, lo, hi, _ = boot_diff(peer, y["no_peers"], kb, rng)
        V[f"w_peer_{b}"] = 100 * float(np.mean([peer[k] for k in kb]))
        V[f"w_gap_{b}"], V[f"w_gap_{b}_lo"], V[f"w_gap_{b}_hi"] = 100 * pt, 100 * lo, 100 * hi
        for c in ("no_peers", "answer_only", "full"):
            V[f"w_acc_{b}_{c}"] = 100 * float(np.mean([y[c][k] for k in kb]))
        for a, z in con:
            pt, lo, hi, _ = boot_diff(y[a], y[z], kb, rng)
            k = f"w_eff_{b}_{SHORT[f'{a} - {z}']}"
            V[k], V[k + "_lo"], V[k + "_hi"] = 100 * pt, 100 * lo, 100 * hi
            if b == "all" and z == "no_peers":
                t = transitions(y[z], y[a], kb)
                V[f"w_{a}_recovered"], V[f"w_{a}_lost"] = t["recovered"], t["lost"]
    for b in BENCH + ("all",):
        kb = [k for k in sel(b) if k in old["full"] and k in old["answer_only"]]
        for a, z in con:
            pt, lo, hi, _ = boot_diff(old[a], old[z], kb, rng)
            dn = {k: y[a][k] - y[z][k] for k in kb}; do = {k: old[a][k] - old[z][k] for k in kb}
            ct, clo, chi, _ = boot_diff(dn, do, kb, rng)
            if b == "all" and (a, z) == ("answer_only", "full"):
                V["w_qwen25peers_answers_full"], V["w_qwen25peers_answers_full_lo"], V["w_qwen25peers_answers_full_hi"] = 100 * pt, 100 * lo, 100 * hi
                V["w_source_change"], V["w_source_change_lo"], V["w_source_change_hi"] = 100 * ct, 100 * clo, 100 * chi
    for c in ("no_peers", "answer_only", "full"):
        V[f"w_{c}_mean_in"] = float(np.mean([int(r[f"{c}_in"]) for r in rows]))
        V[f"w_{c}_mean_out"] = float(np.mean([int(r[f"{c}_out"]) for r in rows]))
        V[f"w_{c}_trunc_pct"] = 100 * float(np.mean([int(r[f"{c}_length"]) for r in rows]))
    V["w_trunc_min"] = min(V[f"w_{c}_trunc_pct"] for c in ("no_peers", "answer_only", "full"))
    V["w_trunc_max"] = max(V[f"w_{c}_trunc_pct"] for c in ("no_peers", "answer_only", "full"))
    V["w_rescored_agree"] = sum(int(r["e1_agree"]) for r in rows)
    V["w_units"] = len(rows)
    V["w_tasks"] = len({r["task"] for r in rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    V = {}
    effects, change = larger_recipients(V)
    weaker_recipient(V)
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n")
    for name, rs in (("effects.csv", effects), ("changes.csv", change)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list({k: None for r in rs for k in r})); w.writeheader(); w.writerows(rs)


if __name__ == "__main__":
    main()
