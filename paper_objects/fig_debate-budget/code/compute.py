"""Debate budgets against voting over independent drafts on BIRD (Du-MAD and MAD-M2).

    python3 code/compute.py --out DIR     # -> DIR/values.json, DIR/plotted_values.csv,
                                          #    DIR/vote_curves.csv, DIR/contrasts.csv

Inputs (inputs/):
  fixed_response_drafts.csv  the six initial answers of the two with-peers repeats of each question
                             (repeat 1 agents, then repeat 2), labelled for the execution vote over the
                             lenient extractor and for the exact-match vote over the protocol's extractor
  fixed_response_runs.csv    per repeat: the protocol's final answer and the last-round execution vote
                             over the lenient extractor
  debate_runs.csv            the budget collection: one run per configuration (agents x rounds), question
                             and repeat, with calls, tokens, last-round execution vote, final answer, or a
                             failed status
  drafts.csv                 20 independent initial drafts per question, backbone and decoding: execution
                             result label (empty when the query does not execute), correctness, tokens

A vote is a plurality vote over execution results (or exact strings); a tie goes to the first tied
draft. Repeats are averaged within a question and cells (protocol x backbone) are averaged per question.

Fixed-response comparison: question bootstrap and database bootstrap (question-weighted), 10,000 draws,
seed 20261109, each from its own generator. Budget collection: one generator (seed 20261109) draws
10,000 multinomial question weights and then 10,000 database weights, shared by every contrast.
Detected means both 95% intervals exclude 0. A failed run is left out (a question stays when one repeat
succeeds); the sensitivity counts it as wrong.
"""
import argparse
import collections
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
SEED, B, K = 20261109, 10000, 20
PROTOS = ("du_mad", "madm2_objective")
BBS = ("llama3-8b", "qwen2.5-7b")
PN = {"du_mad": "dumad", "madm2_objective": "madm2"}
BN = {"llama3-8b": "llama", "qwen2.5-7b": "qwen"}
CONF = {"3x2": 6, "3x3": 9, "5x2": 10, "3x4": 12}
DRAFT_SETS = {"du_mad": ("protocol", "t0.7"), "madm2_objective": ("protocol",)}


def read(name):
    return list(csv.DictReader(open(INP / name, encoding="utf-8")))


def vote(labels):
    """Index of the first draft carrying the most frequent label (None labels do not vote)."""
    live = [x for x in labels if x is not None]
    if not live:
        return None
    cnt = collections.Counter(live)
    top = max(cnt.values())
    return next(i for i, x in enumerate(labels) if x is not None and cnt[x] == top)


# ---------------------------------------------------------------------------- fixed responses
def fixed_response(V):
    drafts = collections.defaultdict(list)
    for r in read("fixed_response_drafts.csv"):
        lab = lambda s: None if s == "" else int(s)          # noqa: E731
        drafts[(r["protocol"], r["backbone"], r["question"])].append(
            (int(r["draft"]), lab(r["lenient_exec_label"]), int(r["lenient_correct"]),
             lab(r["strict_string_label"]), int(r["strict_correct"])))
    runs = collections.defaultdict(list)
    dbmap = {}
    for r in read("fixed_response_runs.csv"):
        runs[(r["protocol"], r["backbone"], r["question"])].append((int(r["final"]), int(r["last_lenient_exec"])))
        dbmap[r["question"]] = r["db_id"]
    units = sorted(dbmap)
    per = collections.defaultdict(list)
    for p in PROTOS:
        for b in BBS:
            for u in units:
                d = sorted(drafts[(p, b, u)])
                i = vote([x[1] for x in d])
                v_exec = float(i is not None and d[i][2])
                i = vote([x[3] for x in d])
                v_str = float(i is not None and d[i][4])
                rr = runs[(p, b, u)]
                fin = np.mean([x[0] for x in rr])
                yx = np.mean([x[1] for x in rr])
                per["exec_minus_final"].append(v_exec - fin)
                per["exec_minus_last"].append(v_exec - yx)
                per["string_minus_final"].append(v_str - fin)
    n = len(units)
    q = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    dbs = sorted(set(dbmap.values()))
    mem = [np.array([j for j, u in enumerate(units) if dbmap[u] == d]) for d in dbs]
    rng = np.random.default_rng(SEED)
    dd = [np.concatenate([mem[j] for j in rng.integers(0, len(dbs), size=len(dbs))]) for _ in range(B)]
    for name, vals in per.items():
        v = np.array(vals).reshape(4, n).mean(0)               # mean of the four cells per question
        lo, hi = np.percentile(v[q].mean(1), [2.5, 97.5])
        dlo, dhi = np.percentile([v[ix].mean() for ix in dd], [2.5, 97.5])
        k = f"fixed_{name}"
        V[k] = 100 * float(v.mean())
        V[k + "_q_lo"], V[k + "_q_hi"] = 100 * float(lo), 100 * float(hi)
        V[k + "_db_lo"], V[k + "_db_hi"] = 100 * float(dlo), 100 * float(dhi)
        V[k + "_detected"] = int((lo > 0 or hi < 0) and (dlo > 0 or dhi < 0) and np.sign(lo) == np.sign(dlo))
    V["fixed_questions"] = n
    V["fixed_drafts_per_question"] = len(drafts[(PROTOS[0], BBS[0], units[0])])


# ---------------------------------------------------------------------------- budget collection
class Budget:
    def __init__(self):
        self.runs = collections.defaultdict(lambda: collections.defaultdict(list))
        self.failed = collections.Counter()
        db_of = {}
        for r in read("debate_runs.csv"):
            key = (r["protocol"], r["backbone"], r["config"])
            db_of[r["question"]] = r["db_id"]
            if r["status"] == "ok":
                self.runs[key][r["question"]].append(r)
            else:
                self.failed[key + (r["question"],)] += 1
        self.units = sorted(db_of)
        self.uix = {u: i for i, u in enumerate(self.units)}
        dbs = sorted(set(db_of.values()))
        n = len(self.units)
        rng = np.random.default_rng(SEED)
        self.WQ = rng.multinomial(n, [1 / n] * n, size=B).astype(float)
        self.WD = rng.multinomial(len(dbs), [1 / len(dbs)] * len(dbs), size=B).astype(float)[
            :, [dbs.index(db_of[u]) for u in self.units]]
        self.drafts = collections.defaultdict(list)
        for r in read("drafts.csv"):
            self.drafts[(r["protocol"], r["backbone"], r["draft_set"], r["question"])].append(
                (int(r["draft"]), None if r["result"] == "" else int(r["result"]), int(r["correct"]),
                 int(r["prompt_tokens"]) + int(r["completion_tokens"])))

    def boot(self, x, mask):
        x = np.where(mask, x, 0.0)
        m = mask.astype(float)
        out = dict(est=100 * float(x.sum() / max(m.sum(), 1)), n=int(m.sum()))
        for tag, W in (("q", self.WQ), ("db", self.WD)):
            e = (W @ x) / np.maximum(W @ m, 1e-12)
            lo, hi = np.percentile(e, [2.5, 97.5])
            out[tag + "_lo"], out[tag + "_hi"] = 100 * float(lo), 100 * float(hi)
        out["detected"] = int(all(not (round(out[t + "_lo"], 3) <= 0 <= round(out[t + "_hi"], 3)) for t in ("q", "db")))
        return out

    def dvec(self, p, b, c, metric, failed_as_wrong=False):
        x = np.full(len(self.units), np.nan)
        for u in self.units:
            vals = [float(r[metric]) for r in self.runs[(p, b, c)].get(u, [])]
            if failed_as_wrong:
                vals += [0.0] * self.failed[(p, b, c, u)]
            if vals:
                x[self.uix[u]] = np.mean(vals)
        return x

    def vote_curves(self, p, b, s):
        """Per question: first-k vote correctness (k = 1..20), its mean over 20 orderings, cumulative tokens."""
        n = len(self.units)
        first, perm, tok, dist = (np.full((K, n), np.nan) for _ in range(4))
        perms = [np.random.default_rng(SEED + i).permutation(K) for i in range(20)]
        distinct, multiple = [], []
        for u in self.units:
            d = sorted(self.drafts[(p, b, s, u)])
            if len(d) < K:
                continue
            j = self.uix[u]
            lab = [x[1] for x in d]
            cor = [x[2] for x in d]
            ct = np.cumsum([x[3] for x in d])

            def acc(order):
                i = vote([lab[o] for o in order])
                return float(i is not None and cor[order[i]])
            for k in range(1, K + 1):
                first[k - 1, j] = acc(list(range(k)))
                perm[k - 1, j] = np.mean([acc(list(pm[:k])) for pm in perms])
                tok[k - 1, j] = ct[k - 1]
            nd = len({x for x in lab if x is not None})
            distinct.append(nd)
            multiple.append(nd >= 2)
        return first, perm, tok, float(np.mean(distinct)), 100 * float(np.mean(multiple)), len(distinct)


def budget(V, out):
    bd = Budget()
    V["budget_questions"] = len(bd.units)
    V["budget_repeats"] = max(int(r["repeat"]) for rs in bd.runs.values() for v in rs.values() for r in v)
    rows = []

    def put(key, r, label):
        V[key] = r["est"]
        for f in ("q_lo", "q_hi", "db_lo", "db_hi"):
            V[f"{key}_{f}"] = r[f]
        V[f"{key}_detected"], V[f"{key}_n"] = r["detected"], r["n"]
        rows.append(dict(contrast=label, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))

    # cost, failures, levels
    total_failed = 0
    for p in PROTOS:
        for b in BBS:
            for c in CONF:
                rs = [r for u in bd.units for r in bd.runs[(p, b, c)].get(u, [])]
                k = f"{PN[p]}_{BN[b]}_{c}"
                V[f"{k}_calls"] = float(np.mean([int(r["calls"]) for r in rs]))
                V[f"{k}_tokens"] = float(np.mean([int(r["prompt_tokens"]) + int(r["completion_tokens"]) for r in rs]))
                V[f"{k}_failed"] = sum(v for kk, v in bd.failed.items() if kk[:3] == (p, b, c))
                total_failed += V[f"{k}_failed"]
                x = bd.dvec(p, b, c, "last_vote_correct")
                V[f"{k}_last_acc"] = 100 * float(np.nanmean(x))
    V["dumad_failed_total"] = sum(V[f"dumad_{BN[b]}_{c}_failed"] for b in BBS for c in CONF)
    V["madm2_failed_total"] = sum(V[f"madm2_{BN[b]}_{c}_failed"] for b in BBS for c in CONF)
    V["dumad_failed_at_larger_budgets"] = sum(V[f"dumad_{BN[b]}_{c}_failed"] for b in BBS for c in CONF if c != "3x2")

    # more budget against the base debate (last-round vote)
    for c in ("3x3", "5x2", "3x4"):
        for faw in (False, True):
            tag = "faw_" if faw else ""
            per = {(p, b): bd.dvec(p, b, c, "last_vote_correct", faw) - bd.dvec(p, b, "3x2", "last_vote_correct")
                   for p in PROTOS for b in BBS}
            for p in PROTOS:
                D = np.vstack([per[(p, b)] for b in BBS])
                put(f"{tag}{PN[p]}_{c}_minus_base", bd.boot(np.nan_to_num(D.mean(0)), np.isfinite(D).all(0)),
                    f"{PN[p]} {c} - 3x2" + (" (failed = wrong)" if faw else ""))
            D = np.vstack(list(per.values()))
            put(f"{tag}all4_{c}_minus_base", bd.boot(np.nan_to_num(D.mean(0)), np.isfinite(D).all(0)),
                f"4 cells {c} - 3x2" + (" (failed = wrong)" if faw else ""))

    # draft curves, diversity
    curves = {}
    crow = []
    for p in PROTOS:
        for b in BBS:
            for s in DRAFT_SETS[p]:
                first, perm, tok, nd, mult, nq = bd.vote_curves(p, b, s)
                curves[(p, b, s)] = (first, tok)
                k = f"{PN[p]}_{BN[b]}_{s.replace('.', '')}"
                V[f"{k}_distinct"], V[f"{k}_multiple_pct"], V[f"{k}_draft_questions"] = nd, mult, nq
                for kk in range(1, K + 1):
                    V[f"{k}_vote{kk}_acc"] = 100 * float(np.nanmean(first[kk - 1]))
                    crow.append(dict(protocol=p, backbone=b, drafts=s, k=kk,
                                     accuracy_first_k=round(100 * float(np.nanmean(first[kk - 1])), 4),
                                     accuracy_mean_20_orderings=round(100 * float(np.nanmean(perm[kk - 1])), 4),
                                     tokens=round(float(np.nanmean(tok[kk - 1])), 1)))
                d = first[K - 1] - first[0]
                put(f"{k}_vote20_minus_vote1", bd.boot(np.nan_to_num(d), np.isfinite(d)), f"{k} vote@20 - vote@1")

    # call- and token-matched: debate (last-round vote) minus vote over the same protocol's drafts
    for c, n in CONF.items():
        for s in ("protocol", "t0.7"):
            cm, tm = {}, {}
            for p in PROTOS:
                if s not in DRAFT_SETS[p]:
                    continue
                for b in BBS:
                    first, tok = curves[(p, b, s)]
                    x = bd.dvec(p, b, c, "last_vote_correct")
                    cm[(p, b)] = x - first[n - 1]
                    k = f"{PN[p]}_{BN[b]}_{c}_{s.replace('.', '')}"
                    put(f"call_{k}", bd.boot(np.nan_to_num(cm[(p, b)]), np.isfinite(cm[(p, b)])), f"call {k}")
                    mt = np.nanmean(tok, axis=1)
                    ks = [kk for kk in range(1, K + 1) if mt[kk - 1] >= V[f"{PN[p]}_{BN[b]}_{c}_tokens"]]
                    kstar = ks[0] if ks else K
                    V[f"kstar_{k}"], V[f"kstar_beyond20_{k}"] = kstar, int(not ks)
                    tm[(p, b)] = x - first[kstar - 1]
                    put(f"token_{k}", bd.boot(np.nan_to_num(tm[(p, b)]), np.isfinite(tm[(p, b)])), f"token {k}")
            for tag, dd in (("call", cm), ("token", tm)):
                for p in PROTOS:
                    if all((p, b) in dd for b in BBS):
                        D = np.vstack([dd[(p, b)] for b in BBS])
                        put(f"{tag}_{PN[p]}_{c}_{s.replace('.', '')}", bd.boot(np.nan_to_num(D.mean(0)), np.isfinite(D).all(0)),
                            f"{tag} {PN[p]} {c} drafts {s}")

    # summaries the text states
    m = [V[f"call_madm2_{c}_protocol"] for c in ("3x3", "5x2", "3x4")]
    V["madm2_call_lead_9to12_min"], V["madm2_call_lead_9to12_max"] = -max(m), -min(m)
    V["madm2_call_9to12_detected"] = sum(V[f"call_madm2_{c}_protocol_detected"] for c in ("3x3", "5x2", "3x4"))
    t = [V[f"token_madm2_{c}_protocol"] for c in CONF]
    V["madm2_token_lead_min"], V["madm2_token_lead_max"] = -max(t), -min(t)
    ks = [V[f"kstar_madm2_{BN[b]}_{c}_protocol"] for b in BBS for c in CONF]
    V["madm2_kstar_min"], V["madm2_kstar_max"] = min(ks), max(ks)
    V["madm2_call_token_detected"] = sum(V[f"{g}_madm2_{c}_protocol_detected"] for g in ("call", "token") for c in CONF)
    V["madm2_3x3_5x2_gain_equal"] = int(round(V["madm2_3x3_minus_base"], 2) == round(V["madm2_5x2_minus_base"], 2))
    V["dumad_beyond20_larger"] = sum(V[f"kstar_beyond20_dumad_{BN[b]}_{c}_protocol"] for b in BBS for c in ("3x3", "5x2", "3x4"))
    V["madm2_call_3x3_llama_debate_ahead"] = int(V["call_madm2_llama_3x3_protocol"] > 0)
    V["madm2_call_3x3_qwen_vote_ahead"] = int(V["call_madm2_qwen_3x3_protocol"] < 0)
    V["dumad_greedy_call_detected"] = sum(V[f"call_dumad_{c}_protocol_detected"] for c in CONF)
    V["dumad_t07_call_detected"] = sum(V[f"call_dumad_{c}_t07_detected"] for c in CONF)
    V["dumad_greedy_vote20_detected"] = V["dumad_llama_protocol_vote20_minus_vote1_detected"] + V["dumad_qwen_protocol_vote20_minus_vote1_detected"]
    V["sampled_vote20_detected"] = sum(V[f"{k}_vote20_minus_vote1_detected"] for k in
                                       ("dumad_llama_t07", "dumad_qwen_t07", "madm2_llama_protocol", "madm2_qwen_protocol"))
    V["draft_sets_complete"] = sum(V[k] for k in V if k.endswith("_draft_questions"))

    with open(out / "contrasts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "vote_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crow[0]))
        w.writeheader()
        w.writerows(crow)


def figure(V, out):
    """Fig. debate-budget: MAD-M2 at 6, 9, 10, 12 calls. The figure draws accuracy at one decimal on the
    scale y = 1.2 x (accuracy - 20) and x = 5 x (calls - 6)."""
    rows = []
    for b in BBS:
        for c, n in CONF.items():
            for series, acc in (("debate", V[f"madm2_{BN[b]}_{c}_last_acc"]),
                                ("independent vote", V[f"madm2_{BN[b]}_protocol_vote{n}_acc"])):
                y = round(1.2 * (round(acc, 1) - 20), 2)
                key = f"fig_{BN[b]}_{'debate' if series == 'debate' else 'vote'}_{n}_y"
                V[key] = y
                rows.append(dict(backbone=b, series=series, calls=n, x=5 * (n - 6), accuracy=round(acc, 4), y=y))
    with open(out / "plotted_values.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    V = {}
    fixed_response(V)
    budget(V, out)
    figure(V, out)
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
