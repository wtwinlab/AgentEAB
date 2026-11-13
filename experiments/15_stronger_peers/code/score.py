r"""Score this round. Offline, no model call. Rules are fixed in advance.

    python3 code/score.py

Inputs: raw/recipient_qwen25-7b.jsonl (new), data/no_peers_e1.jsonl (reused baseline),
W3 peer records, E1 old-panel records (descriptive context only).
Grader, bootstrap, Holm and transitions are W3's (`w3_score`), imported unchanged.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]
W3 = WORK
sys.path.insert(0, str(W3 / "code"))
import w3_score as S                                                        # noqa: E402

OUT = WORK / "output"
RAW = WORK / "raw" / "recipient_qwen25-7b.jsonl"
NOP = WORK / "data" / "no_peers_e1.jsonl"
PEERS = W3 / "raw" / "w3_peer_qwen3-32b.jsonl"
REPLAY = REPO / "experiments/08_cross_family/data/qa_replay_inputs.jsonl"
E1 = REPO / "data/records/framework_raw/e1_units.jsonl"
BENCH = ("gsm8k", "strategyqa", "mmlu_pro", "gpqa")
CONDS = ("no_peers", "answer_only", "full")
CONTRASTS = (("answer_only", "no_peers"), ("full", "no_peers"), ("answer_only", "full"))
S.DRAWS = 10000
SEED = 20261105


def r2(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def write(name, rows):
    if not rows:
        return
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    y = collections.defaultdict(dict)          # y[cond][(task, seat)]
    bench, meta = {}, {}
    cost = collections.defaultdict(lambda: collections.Counter())
    toks = collections.defaultdict(lambda: collections.defaultdict(list))

    for r in map(json.loads, NOP.open()):
        k = (r["task"], r["seat"])
        bench[r["task"]] = r["benchmark"]
        y["no_peers"][k] = S.grade(r["benchmark"], r["answer_kind"], r["raw_reply"], r["task"])
        c = cost["no_peers"]
        c["units"] += 1; c["attempts"] += r["attempts"]; c["failed_attempts"] += r["failed_attempts"]
        c["length"] += r["finish_reason"] == "length"
        c["e1_post_correct_agree"] += int(bool(r["e1_post_correct"])) == y["no_peers"][k]
        toks["no_peers"]["in"].append(r["prompt_tokens"]); toks["no_peers"]["out"].append(r["completion_tokens"])

    last = {}
    for r in map(json.loads, RAW.open()):
        cost[r["condition"]]["attempts"] += 1
        if r["status"] != "ok":
            cost[r["condition"]]["failed_attempts"] += 1
        if r["unit_id"] not in last or r["status"] == "ok":
            last[r["unit_id"]] = r
    for r in last.values():
        k, cnd = (r["task"], r["seat"]), r["condition"]
        txt = r["final_text"] if r["status"] == "ok" else ""
        y[cnd][k] = S.grade(r["benchmark"], r["answer_kind"], txt, r["task"])
        c = cost[cnd]
        c["units"] += 1; c["unit_failed"] += r["status"] != "ok"
        c["length"] += r.get("finish_reason") == "length"
        u = r.get("usage") or {}
        toks[cnd]["in"].append(u.get("prompt_tokens") or 0); toks[cnd]["out"].append(u.get("completion_tokens") or 0)
        meta[(k, cnd)] = r

    keys = sorted(set(y["no_peers"]) & set(y["answer_only"]) & set(y["full"]))
    assert len(keys) == 1440, len(keys)
    tb = {k: bench[k[0]] for k in keys}

    # peer own accuracy (W3 peers, same grader) and what each position was shown
    peer = {}
    ptoks = collections.defaultdict(list)
    for r in map(json.loads, PEERS.open()):
        if r.get("status") == "ok":
            peer[(r["task"], r["seat"])] = S.grade(r["benchmark"], r["answer_kind"],
                                                  r.get("final_text") or "", r["task"])
            u = r.get("usage") or {}
            ptoks["in"].append(u.get("prompt_tokens") or 0); ptoks["out"].append(u.get("completion_tokens") or 0)
    assert len(peer) == 1440
    shown = {k: sum(peer[(k[0], s)] for s in (0, 1, 2) if s != k[1]) for k in keys}
    shown_frac = {k: shown[k] / 2 for k in keys}

    rng = np.random.default_rng(SEED)
    sel = lambda b: [k for k in keys if b == "all" or tb[k] == b]          # noqa: E731

    rel, levels, effects, trans, bypeer = [], [], [], [], []
    for b in BENCH + ("all",):
        kb = sel(b)
        pt, lo, hi, p, nt, nu = S.boot_diff(peer, y["no_peers"], kb, rng)
        rel.append(dict(benchmark=b, tasks=nt, units=nu,
                        peer_own_pct=r2(100 * np.mean([peer[k] for k in kb])),
                        recipient_no_peers_pct=r2(100 * np.mean([y["no_peers"][k] for k in kb])),
                        peer_minus_recipient_pp=r2(100 * pt), lo_pp=r2(100 * lo), hi_pp=r2(100 * hi),
                        shown_peers_mean_pct=r2(100 * np.mean([shown_frac[k] for k in kb])),
                        peer_stronger=bool(lo > 0)))
        for c in CONDS:
            levels.append(dict(benchmark=b, condition=c, tasks=len({k[0] for k in kb}), units=len(kb),
                               accuracy_pct=r2(100 * np.mean([y[c][k] for k in kb]))))
        for a, z in CONTRASTS:
            pt, lo, hi, p, nt, nu = S.boot_diff(y[a], y[z], kb, rng)
            effects.append(dict(benchmark=b, contrast=f"{a} - {z}", tasks=nt, units=nu,
                                effect_pp=r2(100 * pt), lo_pp=r2(100 * lo), hi_pp=r2(100 * hi),
                                p_boot=round(p, 4), ci_excludes_zero=bool(lo > 0 or hi < 0)))
            t = S.transitions(y[z], y[a], kb)
            trans.append(dict(benchmark=b, treatment=a, base=z, **t))
        for nc in (0, 1, 2):
            kk = [k for k in kb if shown[k] == nc]
            if not kk:
                continue
            row = dict(benchmark=b, correct_peers_shown=nc, units=len(kk))
            for c in CONDS:
                row[f"{c}_pct"] = r2(100 * np.mean([y[c][k] for k in kk]))
            for a in ("answer_only", "full"):
                t = S.transitions(y["no_peers"], y[a], kk)
                row[f"{a}_recovered_pp"], row[f"{a}_lost_pp"] = t["recovered_pp"], t["lost_pp"]
            bypeer.append(row)
    fam = [e for e in effects if e["benchmark"] == "all"]
    S.holm(fam)
    for e in effects:
        e.setdefault("holm_reject", None); e.setdefault("holm_family_size", None)

    costrows = []
    for c in CONDS:
        cc = cost[c]
        costrows.append(dict(condition=c, source="E1 reused" if c == "no_peers" else "new",
                             units=cc["units"], attempts=cc["attempts"],
                             failed_attempts=cc["failed_attempts"], units_failed=cc["unit_failed"],
                             truncated_length=cc["length"],
                             truncation_pct=r2(100 * cc["length"] / cc["units"]),
                             mean_input_tokens=r2(np.mean(toks[c]["in"])),
                             mean_output_tokens=r2(np.mean(toks[c]["out"])),
                             total_input_tokens=int(np.sum(toks[c]["in"])),
                             total_output_tokens=int(np.sum(toks[c]["out"]))))
    costrows.append(dict(condition="peer_acquisition (W3, per peer call)", source="W3 reused",
                         units=len(ptoks["in"]), attempts=None, failed_attempts=None, units_failed=None,
                         truncated_length=None, truncation_pct=None,
                         mean_input_tokens=r2(np.mean(ptoks["in"])), mean_output_tokens=r2(np.mean(ptoks["out"])),
                         total_input_tokens=int(np.sum(ptoks["in"])), total_output_tokens=int(np.sum(ptoks["out"]))))

    # descriptive context: same recipient with its old homogeneous Qwen2.5-7B peers (E1, repeat 0)
    rep = {}
    for r in map(json.loads, REPLAY.open()):
        if r["panel"] == "qwen25-7b" and r["decode"] == 0 and r["condition"] in ("full", "answer_only"):
            rep[r["source_unit_id"]] = r
    old = collections.defaultdict(dict)
    for r in map(json.loads, E1.open()):
        u = rep.get(r["unit_id"])
        if u is not None and r.get("status") == "ok":
            old[u["condition"]][(u["task"], u["agent"])] = S.grade(u["benchmark"], u["answer_kind"],
                                                                  r["raw_reply"], u["task"])
    old["no_peers"] = y["no_peers"]
    ctx = []
    for b in BENCH + ("all",):
        kb = [k for k in sel(b) if k in old["full"] and k in old["answer_only"]]
        for a, z in CONTRASTS:
            pt, lo, hi, _, nt, nu = S.boot_diff(old[a], old[z], kb, rng)
            dn = {k: y[a][k] - y[z][k] for k in kb}; do = {k: old[a][k] - old[z][k] for k in kb}
            ct, clo, chi, _, _, _ = S.boot_diff(dn, do, kb, rng)
            ctx.append(dict(benchmark=b, contrast=f"{a} - {z}", units=nu,
                            old_qwen25_peers_effect_pp=r2(100 * pt), old_lo=r2(100 * lo), old_hi=r2(100 * hi),
                            new_qwen3_peers_effect_pp=next(e["effect_pp"] for e in effects
                                                           if e["benchmark"] == b and e["contrast"] == f"{a} - {z}"),
                            change_new_minus_old_pp=r2(100 * ct), change_lo=r2(100 * clo), change_hi=r2(100 * chi)))

    per_unit = []
    for k in keys:
        row = dict(task=k[0], seat=k[1], benchmark=tb[k], correct_peers_shown=shown[k])
        for c in CONDS:
            row[f"y_{c}"] = y[c][k]
        for c in ("answer_only", "full"):
            m = meta[(k, c)]
            row[f"{c}_finish"] = m.get("finish_reason"); row[f"{c}_status"] = m["status"]
            row[f"{c}_out_tokens"] = (m.get("usage") or {}).get("completion_tokens")
        per_unit.append(row)
    per_task = []
    for t in sorted({k[0] for k in keys}):
        kk = [k for k in keys if k[0] == t]
        per_task.append(dict(task=t, benchmark=bench[t], positions=len(kk),
                             **{f"acc_{c}": round(float(np.mean([y[c][k] for k in kk])), 4) for c in CONDS},
                             peer_own=round(float(np.mean([peer[k] for k in kk])), 4)))

    for name, rows in (("peer_vs_recipient.csv", rel), ("levels.csv", levels), ("effects.csv", effects),
                       ("transitions.csv", trans), ("by_correct_peers_shown.csv", bypeer),
                       ("cost.csv", costrows), ("context_old_qwen25_peers.csv", ctx),
                       ("per_unit.csv", per_unit), ("per_task.csv", per_task)):
        write(name, rows)
    summary = dict(units=len(keys), tasks=len({k[0] for k in keys}),
                   e1_post_correct_agreement=f"{cost['no_peers']['e1_post_correct_agree']}/1440",
                   family=[{k: e[k] for k in ("contrast", "effect_pp", "lo_pp", "hi_pp", "p_boot",
                                              "holm_reject", "holm_family_size")} for e in fam],
                   peer_vs_recipient=rel, cost=costrows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    for t in trans:
        if t["benchmark"] == "all":
            print("trans", t)
    for c in ctx:
        if c["benchmark"] == "all":
            print("ctx", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
