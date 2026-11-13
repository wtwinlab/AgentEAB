"""QA with fixed peer responses: sharing effects, first-decode check, transitions and peer quality.

    python3 code/compute.py --out DIR      # -> DIR/values.json, DIR/contrasts.csv, DIR/transitions.csv

Inputs (inputs/):
  qa_recipient_units.csv  one row per scored recipient response
  qa_peer_slots.csv       one row per peer answer shown in an answers-only prompt

Every interval resamples base tasks (480) with replacement; repeated decodes and the three
positions of one task stay together.
"""
import argparse
import collections
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
RECIPIENTS = ("gemma-4-26b", "gpt-oss-120b", "qwen3-32b")
CONTRASTS = (("answer_only", "full"), ("full", "no_peers"), ("answer_only", "no_peers"))
SHORT = {"answer_only - full": "answers_full", "full - no_peers": "full_none",
         "answer_only - no_peers": "answers_none"}
NAME = {"gemma-4-26b": "gemma", "gpt-oss-120b": "gptoss", "qwen3-32b": "qwen3"}


def load():
    cells = {m: {} for m in RECIPIENTS}
    for r in csv.DictReader(open(INP / "qa_recipient_units.csv")):
        for k in ("agent", "decode", "correct", "gradeable"):
            r[k] = int(r[k])
        cells[r["recipient"]][r["unit_id"]] = r
    slots = []
    for r in csv.DictReader(open(INP / "qa_peer_slots.csv")):
        for k in ("seat", "seen_by_seat", "decode", "correct", "gradeable", "correct_as_rendered"):
            r[k] = int(r[k])
        slots.append(r)
    return cells, slots


def complete_cells(cells, conditions):
    """Responses paired inside one (peer set, prefix, position, decode) cell."""
    by = collections.defaultdict(dict)
    for c in cells.values():
        by[(c["panel"], c["prefix_id"], c["agent"], c["decode"])][c["condition"]] = c
    return {k: v for k, v in by.items() if all(cd in v for cd in conditions)}


def task_boot(by_task, rng, draws=10000):
    """Mean of per-task means; percentile interval and two-sided p over task resamples."""
    per = np.array([np.mean(by_task[t]) for t in sorted(by_task)], dtype=float)
    bs = per[rng.integers(0, len(per), size=(draws, len(per)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(per.mean()), float(lo), float(hi), float(min(1.0, max(p, 1.0 / draws))), float(min(1.0, p))


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            out[i] = True
        else:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    cells, slots = load()
    V = {}

    # --- Table tab:family-sharing: no-peer accuracy and the nine paired contrasts (seed 20261028)
    for m in RECIPIENTS:
        by = collections.defaultdict(list)
        for c in cells[m].values():
            if c["condition"] == "no_peers":
                by[c["task"]].append(float(c["correct"]))
        V[f"{NAME[m]}_nopeer_acc"] = 100 * task_boot(by, np.random.default_rng(20261028))[0]
    rows = []
    for m in RECIPIENTS:
        for a, b in CONTRASTS:
            pairs = complete_cells(cells[m], (a, b))
            by = collections.defaultdict(list)
            for v in pairs.values():
                by[v[a]["task"]].append(float(v[a]["correct"]) - float(v[b]["correct"]))
            est, lo, hi, p, _ = task_boot(by, np.random.default_rng(20261028))
            rows.append(dict(recipient=m, contrast=f"{a} - {b}", paired_cells=len(pairs),
                             tasks=len(by), effect_pp=100 * est, lo_pp=100 * lo, hi_pp=100 * hi, p_boot=p))
    for r, f in zip(rows, holm([r["p_boot"] for r in rows])):
        r["holm9"] = f
        k = f"{NAME[r['recipient']]}_{SHORT[r['contrast']]}"
        V[k], V[k + "_lo"], V[k + "_hi"] = r["effect_pp"], r["lo_pp"], r["hi_pp"]
        V[k + "_holm9"] = int(f)
    V["holm9_rejections"] = sum(r["holm9"] for r in rows)

    # --- Supplementary tab:s-first-decode: one generator (seed 20261105) over both scopes
    rng = np.random.default_rng(20261105)
    sens = []
    for m in RECIPIENTS:
        for scope in ("all_matched", "first_decode"):
            sel = {k: u for k, u in cells[m].items() if scope == "all_matched" or u["decode"] == 0}
            for a, b in (("full", "no_peers"), ("answer_only", "no_peers"), ("answer_only", "full")):
                pairs = complete_cells(sel, (a, b))
                by = collections.defaultdict(list)
                for v in pairs.values():
                    by[v[a]["task"]].append(int(v[a]["correct"]) - int(v[b]["correct"]))
                vals = np.array([np.mean(by[t]) for t in sorted(by)])
                bs = vals[rng.integers(0, len(vals), (10000, len(vals)))].mean(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                pv = min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))
                sens.append(dict(recipient=m, scope=scope, contrast=f"{a} - {b}", paired=len(pairs),
                                 effect_pp=100 * float(vals.mean()), lo_pp=100 * lo, hi_pp=100 * hi,
                                 p_boot=float(pv)))
    for scope in ("all_matched", "first_decode"):
        fam = sorted([x for x in sens if x["scope"] == scope], key=lambda x: x["p_boot"])
        stop = False
        for i, x in enumerate(fam):
            x["holm9"] = bool(not stop and x["p_boot"] <= 0.05 / (9 - i))
            stop = stop or not x["holm9"]
    fd = {(x["recipient"], x["contrast"]): x for x in sens if x["scope"] == "first_decode"}
    for (m, c), x in fd.items():
        k = f"fd_{NAME[m]}_{SHORT[c]}"
        V[k], V[k + "_lo"], V[k + "_hi"] = x["effect_pp"], x["lo_pp"], x["hi_pp"]
        V[f"fd_{NAME[m]}_paired"] = x["paired"]
    main_eff = {(r["recipient"], r["contrast"]): r for r in rows}
    V["fd_max_abs_change"] = max(abs(x["effect_pp"] - main_eff[k]["effect_pp"]) for k, x in fd.items())
    V["fd_holm_decisions_changed"] = sum(int(x["holm9"] != main_eff[k]["holm9"]) for k, x in fd.items())

    # --- Supplementary tab:s-qa-transitions: answers-only versus no peers, task-weighted
    shown = {}
    for s in slots:                       # peer strata use the answers-only rendering as graded
        shown[(s["panel"], s["task"], s["seat"])] = s["correct_as_rendered"]
    trows = []
    for m in RECIPIENTS:
        paired = complete_cells(cells[m], ("no_peers", "answer_only"))
        cnt = collections.Counter(v["no_peers"]["task"] for v in paired.values())
        us = []
        for v in paired.values():
            b, a = v["no_peers"], v["answer_only"]
            p = sum(shown[(b["panel"], b["task"], j)] for j in range(3) if j != b["agent"])
            us.append((b["correct"], a["correct"], p, 1 / len(cnt) / cnt[b["task"]]))
        for stratum in ("all", 0):
            sub = [u for u in us if stratum == "all" or u[2] == stratum]
            mass = sum(u[3] for u in sub)
            good = sum(u[3] * u[0] for u in sub)
            fix = sum(u[3] for u in sub if not u[0] and u[1])
            harm = sum(u[3] for u in sub if u[0] and not u[1])
            t = dict(recipient=m, correct_peers=stratum, n=len(sub), q=100 * good / mass,
                     c=100 * fix / (mass - good), r=100 * harm / good,
                     recovered=100 * fix / mass, lost=100 * harm / mass, net=100 * (fix - harm) / mass)
            trows.append(t)
            pre = f"tr_{NAME[m]}" if stratum == "all" else f"tr0_{NAME[m]}"
            for k in ("q", "c", "r", "recovered", "lost", "net"):
                V[f"{pre}_{k}"] = t[k]

    # --- Peer quality: distinct peer answers, the unreadable literal charged as wrong
    dedup = {}
    for s in slots:
        dedup[(s["panel"], s["task"], s["seat"])] = s
    V["peer_distinct_answers"] = len(dedup)
    V["peer_distinct_accuracy"] = 100 * float(np.mean([s["correct"] for s in dedup.values()]))
    below = 0
    for m in RECIPIENTS:
        for bench in sorted({c["benchmark"] for c in cells[m].values()}):
            sel = [c for c in cells[m].values() if c["benchmark"] == bench]
            tasks = {c["task"] for c in sel}
            by = collections.defaultdict(list)
            for s in slots:
                if s["task"] in tasks:
                    by[s["task"]].append(float(s["correct"]))
            peer = float(np.mean([np.mean(v) for v in by.values()]))
            rb = collections.defaultdict(list)
            for c in sel:
                if c["condition"] == "no_peers":
                    rb[c["task"]].append(float(c["correct"]))
            rec = float(np.mean([np.mean(v) for v in rb.values()]))
            below += int(peer < rec)
    V["peer_below_recipient_comparisons"] = below
    pr = collections.defaultdict(list)
    for s in dedup.values():
        pr[(s["panel"], s["task"])].append(float(s["correct"]))
    shares = []
    for m in RECIPIENTS:
        npb = collections.defaultdict(list)
        for c in cells[m].values():
            if c["condition"] == "no_peers":
                npb[(c["panel"], c["task"])].append(float(c["correct"]))
        keys = sorted(set(npb) & set(pr))
        hit = sum(1 for k in keys if np.mean(npb[k]) >= 0.5 and np.mean(pr[k]) < 0.5)
        V[f"overlap_{NAME[m]}_cells"] = len(keys)
        V[f"overlap_{NAME[m]}_pct"] = 100 * hit / len(keys)
        shares.append(100 * hit / len(keys))
    V["overlap_min_pct"], V["overlap_max_pct"] = min(shares), max(shares)
    V["qa_tasks"] = len({c["task"] for m in RECIPIENTS for c in cells[m].values()})

    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n")
    for name, rs in (("contrasts.csv", rows), ("first_decode.csv", sens), ("transitions.csv", trows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)


if __name__ == "__main__":
    main()
