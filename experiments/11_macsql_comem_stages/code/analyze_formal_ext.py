r"""Formal analysis. Offline; reads output/ (copied from the host).

    python3 code/analyze_formal_ext.py
    # inputs : output/macsql_<model>_<bench>.json (host extraction), output/comem_<model>/{<ds>_test,<ds>_dev}.jsonl,
    #          data/llm4em/<ds>.csv (test labels), data/comem_dev_labels.json (dev labels)
    # outputs: output/macsql_estimates.csv, macsql_triggered.csv, macsql_costs.csv,
    #          comem_estimates.csv, comem_per_dataset.csv, comem_stage_changes.csv, comem_dev_tau.csv, summary.json
MAC-SQL: E1 total Refiner - Keep (question-equal primary, database-equal sensitivity; share_eval
core.cluster_boot, 10,000 draws, seed 20261105; Holm over the 2 benchmarks per model); E2 on
triggered questions (descriptive, bootstrap interval), gains/losses, trigger reasons.
ComEM: C1 Selecting - Ranker_single (8 datasets; and the 5 dev-supported), C2 Selecting - Ranker_dev
(5 datasets); macro pair-F1, anchor bootstrap within datasets; Holm over C1(8), C2 per model.
"""
import collections
import csv
import json
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True
W = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W.parents[1] / "experiments/common/code"))
import core as C                                                            # noqa: E402

DRAWS, SEED = 10000, 20261105
O = W / "output"
DEV_OK = ("dblp-scholar", "imdb-tmdb", "imdb-tvdb", "tmdb-tvdb", "walmart-amazon")


def holm(rows, key="p_boot"):
    o = sorted(range(len(rows)), key=lambda i: rows[i][key])
    stop = False
    for j, i in enumerate(o):
        rej = (not stop) and rows[i][key] <= 0.05 / (len(rows) - j)
        stop = stop or not rej
        rows[i]["holm_reject"] = rej
        rows[i]["holm_family_size"] = len(rows)


def write(name, rows):
    if not rows:
        return
    with open(O / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list({k: None for r in rows for k in r})); w.writeheader(); w.writerows(rows)


summ = {}
# ---------------------------------------------------------------- MAC-SQL
est, trig_rows, cost_rows = [], [], []
for mt in ("q3", "q72"):
    fam = []
    for b in ("bird", "spider"):
        p = O / f"macsql_{mt}_{b}.json"
        if not p.exists():
            continue
        X = json.load(open(p)); U = X["units"]
        y = {(u["question_id"],): u["scores"]["Refiner"] - u["scores"]["Keep"] for u in U}
        cl = {(u["question_id"],): u["db_id"] for u in U}
        for wt in ("unit", "cluster_equal"):
            r = C.cluster_boot(y, cl, wt, DRAWS, SEED)
            row = dict(model=mt, bench=b, estimand="E1_total", weighting="question_equal" if wt == "unit" else "database_equal",
                       questions=r["units"], databases=r["clusters"], effect_pp=round(100 * r["point"], 3), lo_pp=round(100 * r["lo"], 3),
                       hi_pp=round(100 * r["hi"], 3), p_boot=round(r["p_boot"], 4),
                       refiner_correct=sum(u["scores"]["Refiner"] for u in U), keep_correct=sum(u["scores"]["Keep"] for u in U),
                       gains=sum(1 for u in U if u["scores"]["Refiner"] > u["scores"]["Keep"]),
                       losses=sum(1 for u in U if u["scores"]["Refiner"] < u["scores"]["Keep"]))
            est.append(row)
            if wt == "unit":
                fam.append(row)
        T = [u for u in U if u["refiner_llm_calls"] > 0]
        yt = {(u["question_id"],): y[(u["question_id"],)] for u in T}
        rt = C.cluster_boot(yt, cl, "unit", DRAWS, SEED) if T else None
        trig_rows.append(dict(model=mt, bench=b, questions=len(U), triggered=len(T), trigger_rate_pct=round(100 * len(T) / len(U), 2),
                              E2_effect_pp=round(100 * rt["point"], 2) if rt else None, E2_lo_pp=round(100 * rt["lo"], 2) if rt else None,
                              E2_hi_pp=round(100 * rt["hi"], 2) if rt else None, gains=sum(1 for u in T if yt[(u["question_id"],)] > 0),
                              losses=sum(1 for u in T if yt[(u["question_id"],)] < 0),
                              both_wrong=sum(1 for u in T if not u["scores"]["Refiner"] and not u["scores"]["Keep"]),
                              both_right=sum(1 for u in T if u["scores"]["Refiner"] and u["scores"]["Keep"]),
                              refiner_changed_sql=sum(1 for u in T if (u["refiner_sql"] or "").strip() != (u["boundary_sql"] or "").strip()),
                              reasons=json.dumps(dict(collections.Counter(u["trigger_reason"] for u in T))),
                              reasons_all=json.dumps(dict(collections.Counter(u["trigger_reason"] for u in U))),
                              flags=json.dumps(dict(collections.Counter(f for u in U for f in u["flags"])))))
        agg = collections.defaultdict(collections.Counter)
        for u in U:
            for s, v in u["calls_by_stage"].items():
                agg[s].update(v)
        for s, v in agg.items():
            cost_rows.append(dict(model=mt, bench=b, stage=s, part="stage" if s == "refiner" else "prefix", **v))
        cost_rows.append(dict(model=mt, bench=b, stage="refiner_tool_time_s", part="stage",
                              seconds=round(sum(u["keep_seconds"] for u in T), 3)))
        cost_rows.append(dict(model=mt, bench=b, stage="proxy_total", part="all", calls=X["proxy_calls"], non200=X["proxy_non200"],
                              refused=X["proxy_refused"], by_stage=json.dumps(X["proxy_by_stage"])))
    if fam:
        holm(fam)
write("macsql_estimates.csv", est); write("macsql_triggered.csv", trig_rows); write("macsql_costs.csv", cost_rows)
summ["macsql"] = dict(estimates=[r for r in est if r["weighting"] == "question_equal"], triggered=trig_rows)


# ---------------------------------------------------------------- ComEM
def counts(p, l):
    return (sum(a and b for a, b in zip(p, l)), sum(a and not b for a, b in zip(p, l)), sum((not a) and b for a, b in zip(p, l)))


def f1(t, fp, fn):
    return 2 * t / (2 * t + fp + fn) if t else 0.0


lab = {}
devlab = json.load(open(W / "data/comem_dev_labels.json")) if (W / "data/comem_dev_labels.json").exists() else {}
comem_est, per_ds, changes, taus = [], [], [], []
for mt in ("q3", "q7"):
    d = O / f"comem_{mt}"
    if not d.exists():
        continue
    data = {}
    for ds in sorted(devlab):
        tp = d / f"{ds}_test.jsonl"
        if not tp.exists():
            continue
        if ds not in lab:
            lab[ds] = {(r["id_left"], r["id_right"]): r["label"] in ("True", "1", "true") for r in csv.DictReader(open(W / f"data/llm4em/{ds}.csv"))}
        R = [json.loads(l) for l in open(tp)]
        # dev tau
        tau = None
        if ds in DEV_OK:
            dl = {x["id_left"]: x["label"] for x in devlab[ds]["dev"]}
            D = [json.loads(l) for l in open(d / f"{ds}_dev.jsonl")]
            tops = []
            for x in D:
                o = sorted(range(len(x["scores"])), key=lambda j: (x["scores"][j], j), reverse=True)
                tops.append((x["scores"][o[0]], o[0], dl[x["id_left"]]))
            grid = sorted({-1.0} | {s for s, _, _ in tops})
            best = None
            for t in grid:
                T = Fp = Fn = 0
                for s, j, L in tops:
                    p = [False] * len(L)
                    if s > t:
                        p[j] = True
                    a, b_, c = counts(p, L); T += a; Fp += b_; Fn += c
                key = (f1(T, Fp, Fn), -abs(t))
                if best is None or key > best[0]:
                    best = (key, t)
            tau = best[1]
            taus.append(dict(model=mt, dataset=ds, dev_anchors=len(D), dev_pos=sum(sum(x) for x in dl.values()), tau=tau, dev_f1=round(100 * best[0][0], 2)))
        units = []
        for r in R:
            L = [lab[ds][(r["id_left"], x)] for x in r["id_right"]]
            n = len(L)
            P = {"Selecting": r["preds_select"] or [False] * n, "Ranker_single": r["preds_ranker_single"] or [False] * n}
            if r["topk"]:
                t1 = [False] * n; t1[r["topk"][0]] = True; P["Top1_always"] = t1
                if tau is not None:
                    pd_ = [False] * n
                    if r["scores"][r["topk"][0]] > tau:
                        pd_[r["topk"][0]] = True
                    P["Ranker_dev"] = pd_
            units.append(dict(L=L, P=P, r=r))
        data[ds] = units
        sel_rank = collections.Counter()
        for u in units:
            ps = u["P"]["Selecting"]
            k = "none" if not any(ps) else ("rank1" if u["r"]["topk"] and ps[u["r"]["topk"][0]] else "rank2-4")
            ok = ps == u["L"]
            sel_rank[(k, ok)] += 1
        changes.append(dict(model=mt, dataset=ds, **{f"{k}_{'correct' if ok else 'wrong'}": v for (k, ok), v in sorted(sel_rank.items())},
                            flags=json.dumps(dict(collections.Counter(u["r"]["status"] for u in units))),
                            unparsed=sum(1 for u in units if (u["r"].get("select") or {}).get("content") is not None and "[" not in u["r"]["select"]["content"]),
                            control_mismatch=sum(1 for u in units if u["r"]["preds_ranker_single"] != u["r"]["preds_ranker_single_official"]),
                            selector_prompt_tokens=sum(((u["r"].get("select") or {}).get("usage") or {}).get("prompt_tokens") or 0 for u in units),
                            selector_completion_tokens=sum(((u["r"].get("select") or {}).get("usage") or {}).get("completion_tokens") or 0 for u in units),
                            selector_calls=sum(1 for u in units if u["r"].get("select")),
                            ranker_forward_test=sum(u["r"]["ranker_forward"] for u in units),
                            ranker_seconds_test=round(sum(u["r"]["ranker_seconds"] for u in units), 2)))
        for b in ("Selecting", "Ranker_single", "Ranker_dev", "Top1_always"):
            if b not in units[0]["P"]:
                continue
            T = Fp = Fn = 0
            for u in units:
                a, b_, c = counts(u["P"][b], u["L"]); T += a; Fp += b_; Fn += c
            per_ds.append(dict(model=mt, dataset=ds, branch=b, tp=T, fp=Fp, fn=Fn, f1=round(100 * f1(T, Fp, Fn), 2),
                               anchors_correct=sum(1 for u in units if u["P"][b] == u["L"]), anchors=len(units)))
    rng = np.random.default_rng(SEED)
    idx = {ds: rng.integers(0, len(u), size=(DRAWS, len(u))) for ds, u in data.items()}

    def macro(dss, a, b, metric):
        pts, bs = [], np.zeros(DRAWS)
        for ds in dss:
            U = data[ds]
            if metric == "f1":
                ca = np.array([counts(u["P"][a], u["L"]) for u in U]); cb = np.array([counts(u["P"][b], u["L"]) for u in U])
                fa = lambda c: np.where(c[..., 0] > 0, 2 * c[..., 0] / np.maximum(2 * c[..., 0] + c[..., 1] + c[..., 2], 1), 0.0)  # noqa: E731
                pts.append(float(fa(ca.sum(0)) - fa(cb.sum(0))))
                bs += fa(ca[idx[ds]].sum(1)) - fa(cb[idx[ds]].sum(1))
            else:
                va = np.array([u["P"][a] == u["L"] for u in U], float); vb = np.array([u["P"][b] == u["L"] for u in U], float)
                pts.append(float(va.mean() - vb.mean())); bs += va[idx[ds]].mean(1) - vb[idx[ds]].mean(1)
        bs /= len(dss); pt = float(np.mean(pts))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(effect_pp=round(100 * pt, 3), lo_pp=round(100 * lo, 3), hi_pp=round(100 * hi, 3),
                    p_boot=round(float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))), 4))
    fam = []
    all8 = sorted(data)
    sup = [x for x in DEV_OK if x in data]
    for cid, b, dss, prim in (("C1", "Ranker_single", all8, True), ("C2", "Ranker_dev", sup, True), ("C1_on5", "Ranker_single", sup, False)):
        if not dss:
            continue
        for metric in ("f1", "anchor_correct"):
            row = dict(model=mt, estimand=cid, contrast=f"Selecting - {b}", metric=metric, datasets=len(dss), **macro(dss, "Selecting", b, metric))
            comem_est.append(row)
            if prim and metric == "f1":
                fam.append(row)
    holm(fam)
write("comem_estimates.csv", comem_est); write("comem_per_dataset.csv", per_ds); write("comem_stage_changes.csv", changes); write("comem_dev_tau.csv", taus)
summ["comem"] = dict(estimates=comem_est, dev_tau=taus)
(O / "summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
print(json.dumps(summ, indent=1, default=str)[:6000])
