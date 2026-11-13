r"""W3 scoring: absolute accuracy, sharing effects, source changes, transitions, gap display.

    python3 code/w3_score.py        # -> output/w3_levels.csv, w3_effects.csv, w3_source_change.csv,
                                    #    w3_transitions.csv, w3_gap_points.csv, w3_peer_quality.csv,
                                    #    w3_input_size.csv, w3_gap.pdf/png, w3_health.json

The scoring rules are fixed before the new-source recipient responses are read. Grader and extractor are the project's existing implementations.
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
OUT = WORK / "output"
sys.path.insert(0, str(REPO / "code"))
from consensus_delivery.adapters import mad_m2 as M2                        # noqa: E402
from consensus_delivery.adapters import official_scorers as OS              # noqa: E402

RECIPIENTS = ("qwen3-32b", "gpt-oss-120b", "gemma-4-26b")
BENCH = ("gsm8k", "strategyqa", "mmlu_pro", "gpqa")
OLD = ("llama3-8b", "qwen25-7b")
NEW = "qwen3-32b-new"
SOURCES = OLD + (NEW,)
PRIMARY_BASE = "qwen25-7b"       # baseline call for the new source, frozen before collection
SENS_BASE = "llama3-8b"
DRAWS, SEED = 10000, 20261101
CONTRASTS = (("full", "no_peers"), ("answer_only", "no_peers"), ("answer_only", "full"))
SHARE_INSTR = "Using the solutions from other agents as additional information"
INPUTS = REPO / "experiments/08_cross_family/data/qa_replay_inputs.jsonl"
REFS = REPO / "data/derived/substantive_completion/data/qa_task_refs.json"
OLDRAW = REPO / "data/derived/substantive_completion/output/raw"

refs = json.loads(REFS.read_text())


def grade(bench, kind, text, task):
    ans = M2.extract_answer(text or "", kind)
    if ans is None or not str(ans).strip():
        return 0
    ref = refs[task]
    r0 = (ref.get("references") or [None])[0]
    md = ref.get("metadata") or {}
    if bench == "gsm8k":
        return int(bool(OS.gsm8k_correct(ans, r0)))
    if bench == "strategyqa":
        return int(bool(OS.strategyqa_correct(ans, r0)))
    return int(bool(OS.choice_correct(ans, r0, int(md.get("n_options", 10) or 10))))


def load():
    units = {}
    for line in open(INPUTS):
        r = json.loads(line)
        if r["decode"] == 0:
            units[r["unit"]] = r
    # outcome[(recipient, source, condition)][(task, seat)] = 0/1
    out = collections.defaultdict(dict)
    health = collections.Counter()
    peer_chars = collections.defaultdict(list)
    for m in RECIPIENTS:
        for line in open(OLDRAW / f"qa_{m}.jsonl"):
            r = json.loads(line)
            u = units.get(r.get("unit_id"))
            if u is None:
                continue
            txt = ""
            if r.get("status") == "ok":
                txt = r["response"]["choices"][0]["message"].get("content") or ""
            else:
                health[f"old_error:{m}"] += 1
            k = (u["task"], u["agent"])
            out[(m, u["panel"], u["condition"])][k] = grade(u["benchmark"], u["answer_kind"],
                                                           txt, u["task"])
            out[("meta", "bench", "")][u["task"]] = u["benchmark"]
            if m == RECIPIENTS[0] and u["condition"] != "no_peers":
                c = u["messages"][-1]["content"]
                peer_chars[(u["panel"], u["condition"])].append(c.find(SHARE_INSTR))
        for line in open(WORK / "raw" / f"w3_recipient_{m}.jsonl") if \
                (WORK / "raw" / f"w3_recipient_{m}.jsonl").is_file() else []:
            r = json.loads(line)
            if r.get("status") != "ok":
                health[f"new_error:{m}"] += 1
            if r.get("finish_reason") == "length":
                health[f"new_truncated:{m}"] += 1
            out[(m, NEW, r["condition"])][(r["task"], r["seat"])] = grade(
                r["benchmark"], r["answer_kind"], r.get("final_text") or "", r["task"])
            if m == RECIPIENTS[0]:
                c = r["messages"][0]["content"]
                peer_chars[(NEW, r["condition"])].append(c.find(SHARE_INSTR))
        # the new source's no_peers is the frozen old baseline call
        for src_base, tag in ((PRIMARY_BASE, NEW), (SENS_BASE, NEW + "|sens")):
            out[(m, tag, "no_peers")] = dict(out[(m, src_base, "no_peers")])
        out[(m, NEW + "|sens", "full")] = out[(m, NEW, "full")]
        out[(m, NEW + "|sens", "answer_only")] = out[(m, NEW, "answer_only")]
    return out, health, peer_chars


def peer_quality(out_bench):
    """Own accuracy of each peer source on the same tasks."""
    rows = []
    acc = collections.defaultdict(list)
    for line in open(INPUTS):
        r = json.loads(line)
        if r["decode"] == 0 and r["condition"] == "no_peers" and r.get("old_pre_correct") is not None:
            acc[(r["panel"], r["benchmark"])].append(int(bool(r["old_pre_correct"])))
    p = WORK / "raw" / "w3_peer_qwen3-32b.jsonl"
    seen = {}
    for line in open(p):
        r = json.loads(line)
        seen[(r["task"], r["seat"])] = r
    for r in seen.values():
        txt = r.get("final_text") if r.get("status") == "ok" else ""
        acc[(NEW, r["benchmark"])].append(grade(r["benchmark"], r["answer_kind"], txt, r["task"]))
    for (src, b), v in sorted(acc.items()):
        rows.append(dict(source=src, benchmark=b, n=len(v), own_accuracy_pct=round(float(100 * np.mean(v)), 2)))
    return rows, {(r["source"], r["benchmark"]): r["own_accuracy_pct"] for r in rows}


def boot_diff(a: dict, b: dict, keys, rng):
    """Paired difference a - b over keys, cluster bootstrap on the task."""
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
    return pt, float(lo), float(hi), p, len(tasks), len(keys)


def holm(rows, key="p_boot"):
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    m, stop = len(rows), False
    for j, i in enumerate(order):
        rej = (not stop) and rows[i][key] <= 0.05 / (m - j)
        stop = stop or not rej
        rows[i]["holm_reject"] = rej
        rows[i]["holm_family_size"] = m


def transitions(base: dict, trt: dict, keys):
    b = np.array([base[k] for k in keys]); t = np.array([trt[k] for k in keys])
    q = b.mean()
    c = t[b == 0].mean() if (b == 0).any() else float("nan")
    r = (1 - t[b == 1]).mean() if (b == 1).any() else float("nan")
    return dict(n=len(keys), q_pct=round(100 * q, 2), c_pct=round(100 * c, 2),
                r_pct=round(100 * r, 2), recovered_pp=round(100 * (1 - q) * c, 2),
                lost_pp=round(100 * q * r, 2), net_pp=round(100 * ((1 - q) * c - q * r), 2))


def main() -> int:
    out, health, pchars = load()
    tb = out[("meta", "bench", "")]
    pq_rows, pq = peer_quality(tb)
    rng = np.random.default_rng(SEED)
    complete = {}
    levels, effects, change, trans, gap = [], [], [], [], []
    for m in RECIPIENTS:
        for src in SOURCES + (NEW + "|sens",):
            conds = {c: out.get((m, src, c), {}) for c in ("no_peers", "full", "answer_only")}
            keys = sorted(set(conds["no_peers"]) & set(conds["full"]) & set(conds["answer_only"]))
            complete[(m, src)] = keys
            for b in BENCH + ("all",):
                kb = [k for k in keys if b == "all" or tb.get(k[0]) == b]
                if not kb:
                    continue
                for c in ("no_peers", "full", "answer_only"):
                    levels.append(dict(recipient=m, source=src, benchmark=b, condition=c,
                                       units=len(kb), accuracy_pct=round(100 * np.mean(
                                           [conds[c][k] for k in kb]), 2)))
                for a, z in CONTRASTS:
                    pt, lo, hi, p, nt, nu = boot_diff(conds[a], conds[z], kb, rng)
                    effects.append(dict(recipient=m, source=src, benchmark=b, contrast=f"{a} - {z}",
                                        tasks=nt, units=nu, effect_pp=round(100 * pt, 2),
                                        lo_pp=round(100 * lo, 2), hi_pp=round(100 * hi, 2),
                                        p_boot=round(p, 4), ci_excludes_zero=bool(lo > 0 or hi < 0)))
                    if a != "answer_only" or z != "full":
                        pass
                for a in ("full", "answer_only"):
                    tr = transitions(conds["no_peers"], conds[a], kb)
                    trans.append(dict(recipient=m, source=src, benchmark=b, condition=a, **tr))
                if b != "all" and src in SOURCES:
                    own = 100 * np.mean([conds["no_peers"][k] for k in kb])
                    peer = pq.get((src, b))
                    for a in ("full", "answer_only"):
                        e = next(x for x in effects if x["recipient"] == m and x["source"] == src
                                 and x["benchmark"] == b and x["contrast"] == f"{a} - no_peers")
                        gap.append(dict(recipient=m, source=src, benchmark=b, condition=a,
                                        recipient_no_peers_pct=round(own, 2), peer_own_pct=peer,
                                        gap_pp=round(own - peer, 2) if peer is not None else None,
                                        effect_pp=e["effect_pp"]))
        # source change, new - qwen25 panel, paired on (task, seat) over units complete in both
        for b in BENCH + ("all",):
            for a, z in CONTRASTS:
                for old in OLD:
                    kk = sorted(set(complete[(m, NEW)]) & set(complete[(m, old)]))
                    kk = [k for k in kk if b == "all" or tb.get(k[0]) == b]
                    if not kk:
                        continue
                    dn = {k: out[(m, NEW, a)][k] - out[(m, NEW, z)][k] for k in kk}
                    do = {k: out[(m, old, a)][k] - out[(m, old, z)][k] for k in kk}
                    pt, lo, hi, p, nt, nu = boot_diff(dn, do, kk, rng)
                    change.append(dict(recipient=m, benchmark=b, contrast=f"{a} - {z}",
                                       vs_old_source=old, tasks=nt, units=nu,
                                       change_pp=round(100 * pt, 2), lo_pp=round(100 * lo, 2),
                                       hi_pp=round(100 * hi, 2), p_boot=round(p, 4),
                                       ci_excludes_zero=bool(lo > 0 or hi < 0)))
    famA = [e for e in effects if e["source"] == NEW and e["benchmark"] == "all"]
    famB = [c for c in change if c["benchmark"] == "all" and c["vs_old_source"] == PRIMARY_BASE]
    if famA:
        holm(famA)
    if famB:
        holm(famB)
    for rows in (effects, change):
        for r in rows:
            r.setdefault("holm_reject", None); r.setdefault("holm_family_size", None)

    size = []
    for (src, c), v in sorted(pchars.items()):
        size.append(dict(source=src, condition=c, prompts=len(v),
                         mean_peer_block_chars=round(float(np.mean(v)), 1),
                         median_peer_block_chars=float(np.median(v))))

    OUT.mkdir(exist_ok=True)
    for name, rows in (("w3_levels.csv", levels), ("w3_effects.csv", effects),
                       ("w3_source_change.csv", change), ("w3_transitions.csv", trans),
                       ("w3_gap_points.csv", gap), ("w3_peer_quality.csv", pq_rows),
                       ("w3_input_size.csv", size)):
        if rows:
            with open(OUT / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    hl = dict(health=dict(health),
              complete_units={f"{m}|{s}": len(v) for (m, s), v in complete.items()})
    (OUT / "w3_health.json").write_text(json.dumps(hl, indent=1) + "\n")
    print(json.dumps(hl, indent=1))
    for r in pq_rows:
        print("  peer", r)
    for e in effects:
        if e["benchmark"] == "all":
            print(f"  {e['recipient']:13} {e['source']:18} {e['contrast']:24} {e['effect_pp']:+6.2f} "
                  f"[{e['lo_pp']:+.2f},{e['hi_pp']:+.2f}] holm={e['holm_reject']} n={e['units']}")
    for c in change:
        if c["benchmark"] == "all":
            print(f"  CHANGE {c['recipient']:13} vs {c['vs_old_source']:10} {c['contrast']:24} "
                  f"{c['change_pp']:+6.2f} [{c['lo_pp']:+.2f},{c['hi_pp']:+.2f}] holm={c['holm_reject']}")
    for s in size:
        print("  size", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
