r"""Stage decomposition, Eq. (4): change rate times gain when the stage changes the result (offline; reads
the unit files written by reexec.py and the ComEM records; no model call, no SQL execution).

    python3 share_eval/stage_replacement/decompose.py [--out DIR] [--external-run DIR] [--chess-run DIR]
    # -> <out>/decomposition.csv, decomposition.json, movers.csv, costs.csv,
    #    comem_per_dataset.csv, comem_pair_level.csv

Per row: n, n_A, P(A) [CI], E[D|A] [CI; only if n_A >= 30], gains / losses / both-wrong / both-right among acting
units, total = mean D [CI], the identity check total - P(A)*E[D|A] and the count of A=0 units with D != 0
(named violations; 0 by construction for result-equality definitions). D_u = Y(stage) - Y(alternative) in {-1,0,1}.
Inference: percentile bootstrap, 10,000 draws, seed 20261105. SQL: DB-cluster bootstrap, question-equal
(cluster bootstrap in common.py). ComEM: stratified by dataset (dataset fixed; anchors resampled within it),
pooled anchor-equal; pair level: anchors as clusters within datasets.
"""
from __future__ import annotations

import collections
import csv
import json

import pathlib as _pl, sys as _sys
_sys.dont_write_bytecode = True
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import common as K  # noqa: E402

K.configure(__doc__.splitlines()[0])

MIN_N = 30
ROWS, MOVERS, COSTS = [], [], []


def read(p):
    return list(csv.DictReader(open(p)))


def summarise(system, cell, alternative, a_def, U, boot, extra=None):
    """U: list of dict(key, cluster, A, Ys, Ya). boot(values_dict, clusters_dict) -> dict or None."""
    n = len(U)
    D = {u["key"]: u["Ys"] - u["Ya"] for u in U}
    assert set(D.values()) <= {-1, 0, 1}
    cl = {u["key"]: u["cluster"] for u in U}
    act = [u for u in U if u["A"]]
    nA = len(act)
    viol = [u["key"] for u in U if not u["A"] and D[u["key"]] != 0]
    tot = boot(D, cl)
    pa = boot({u["key"]: float(u["A"]) for u in U}, cl)
    cond = boot({u["key"]: D[u["key"]] for u in act}, cl) if nA >= MIN_N else None
    e_cond = (sum(D[u["key"]] for u in act) / nA) if nA else None
    total_pt = sum(D.values()) / n
    ident = (nA / n) * (e_cond or 0.0)
    r = dict(system=system, cell=cell, alternative=alternative, A_definition=a_def, n=n, n_A=nA,
             P_A_pct=round(100 * nA / n, 2), P_A_lo=round(100 * pa["lo"], 2), P_A_hi=round(100 * pa["hi"], 2),
             gains=sum(1 for u in act if D[u["key"]] > 0), losses=sum(1 for u in act if D[u["key"]] < 0),
             both_wrong_A=sum(1 for u in act if not u["Ys"] and not u["Ya"]),
             both_right_A=sum(1 for u in act if u["Ys"] and u["Ya"]),
             E_D_given_A_pp=round(100 * e_cond, 2) if e_cond is not None else None,
             E_D_given_A_lo=round(100 * cond["lo"], 2) if cond else None,
             E_D_given_A_hi=round(100 * cond["hi"], 2) if cond else None,
             conditional_interval="reported" if cond else f"insufficient (n_A={nA} < {MIN_N}): raw counts only",
             total_pp=round(100 * total_pt, 3), total_lo=round(100 * tot["lo"], 3), total_hi=round(100 * tot["hi"], 3),
             stage_correct=sum(u["Ys"] for u in U), alt_correct=sum(u["Ya"] for u in U),
             identity_residual_pp=round(100 * (total_pt - ident), 6), A0_with_D_nonzero=len(viol),
             A0_with_D_nonzero_keys=json.dumps(viol[:20]), clusters=tot["clusters"])
    r.update(extra or {})
    ROWS.append(r)
    return r


def sql_boot(vals, cl):
    return K.cluster_boot(vals, cl)


def crosstab(system, cell, alternative, old, new, U):
    c = collections.Counter((u[old], u[new]) for u in U)
    MOVERS.append(dict(system=system, cell=cell, alternative=alternative, old=old, new=new,
                       old1_new1=c[(1, 1)], old1_new0=c[(1, 0)], old0_new1=c[(0, 1)], old0_new0=c[(0, 0)],
                       old_missing=sum(v for (o, _), v in c.items() if o not in (0, 1))))


# ---------------------------------------------------------------- MAC-SQL
for mt, mname in (("q3", "Qwen3-32B"), ("q72", "Qwen2.5-72B-AWQ")):
    for b in ("bird", "spider"):
        R = read(K.OUT / f"macsql_units_{mt}_{b}.csv")
        X = json.load(open(K.EXT / f"output/macsql_{mt}_{b}.json"))
        cell = f"{mname}, {'BIRD dev' if b == 'bird' else 'Spider dev'}"
        base = [dict(key=(int(r["question_id"]),), cluster=r["db_id"], Ys=int(r["Y_stage"]), Ya=int(r["Y_alt"]),
                     A_trigger=int(r["A_trigger"]), A_text=int(r["A_text"]), A_result=int(r["A_result"]),
                     failed="REFINER_FAILED" in r["flags"], repro=(r["repro_stage"], r["repro_alt"])) for r in R]
        for a in ("A_result", "A_text", "A_trigger"):
            U = [dict(u, A=u[a]) for u in base]
            summarise("MAC-SQL", cell, "Keep (Decomposer final_sql)", a + (" (primary)" if a == "A_result" else ""), U, sql_boot,
                      dict(refiner_failed_units=sum(u["failed"] for u in base),
                           refiner_failed_in_A=sum(u["failed"] and u[a] for u in base)))
        crosstab("MAC-SQL", cell, "Keep", "A_trigger", "A_result", base)
        crosstab("MAC-SQL", cell, "Keep", "A_text", "A_result", base)
        crosstab("MAC-SQL", cell, "Keep", "A_trigger", "A_text", base)
        # costs: denominator = all Selector + Decomposer + Refiner LLM calls of the cell (trace, all questions);
        # the 6-per-model server pings are not question calls and are reported apart.
        agg = collections.defaultdict(collections.Counter)
        for u in X["units"]:
            for s, v in u["calls_by_stage"].items():
                agg[s].update(v)
        allc = sum(v["calls"] for v in agg.values())
        allt = sum(v["prompt_tokens"] + v["completion_tokens"] for v in agg.values())
        st = agg["refiner"]
        COSTS.append(dict(system="MAC-SQL", cell=cell, stage="Refiner LLM", stage_calls=st["calls"], all_calls=allc,
                          stage_call_share_pct=round(100 * st["calls"] / allc, 2),
                          stage_prompt_tokens=st["prompt_tokens"], stage_completion_tokens=st["completion_tokens"],
                          all_tokens=allt, stage_token_share_pct=round(100 * (st["prompt_tokens"] + st["completion_tokens"]) / allt, 2),
                          stage_completion_share_pct=round(100 * st["completion_tokens"] / sum(v["completion_tokens"] for v in agg.values()), 2),
                          alternative_calls=0,
                          denominator="Selector+Decomposer+Refiner LLM calls/tokens over all questions (api_trace)",
                          apart=json.dumps(dict(pings=X["proxy_by_stage"].get("ping", 0),
                                                unattributed=sum(v["calls"] for s, v in agg.items() if s == "unattributed"),
                                                by_stage={s: v["calls"] for s, v in agg.items()}))))

# ---------------------------------------------------------------- CHESS
R = read(K.OUT / "chess_units.csv")
for alt in ("SC_official", "Valid_vote", "Nonempty_vote"):
    base = [dict(key=(int(r["question_id"]),), cluster=r["db_id"], Ys=int(r["Y_stage"]), Ya=int(r["Y_alt"]),
                 A_result=int(r["A_result"]), A_sqlcommon=int(r["A_sqlcommon"]), A_text=int(r["text_differs"]),
                 A_old=(int(r["A_old"]) if r["A_old"] != "" else None)) for r in R if r["alternative"] == alt]
    for a in ("A_result", "A_sqlcommon", "A_old"):
        if any(u[a] is None for u in base):
            continue
        summarise("CHESS B1", "Qwen3-32B, BIRD SDS 147", alt, a + (" (primary)" if a == "A_result" else ""),
                  [dict(u, A=u[a]) for u in base], sql_boot)
    crosstab("CHESS B1", "Qwen3-32B, BIRD SDS 147", alt, "A_old", "A_result", base)
    crosstab("CHESS B1", "Qwen3-32B, BIRD SDS 147", alt, "A_sqlcommon", "A_result", base)
    crosstab("CHESS B1", "Qwen3-32B, BIRD SDS 147", alt, "A_text", "A_result", base)
cq = read(K.CHESS / "output/costs_per_question.csv")
tot = collections.Counter(); ut = collections.Counter()
for r in cq:
    for k in ("calls", "prompt_tokens", "completion_tokens"):
        tot[k] += int(r[k])
        if r["part"] == "ut_increment":
            ut[k] += int(r[k])
COSTS.append(dict(system="CHESS B1", cell="Qwen3-32B, BIRD SDS 147", stage="Unit Tester increment (from UT agent's first call)",
                  stage_calls=ut["calls"], all_calls=tot["calls"], stage_call_share_pct=round(100 * ut["calls"] / tot["calls"], 2),
                  stage_prompt_tokens=ut["prompt_tokens"], stage_completion_tokens=ut["completion_tokens"],
                  all_tokens=tot["prompt_tokens"] + tot["completion_tokens"],
                  stage_token_share_pct=round(100 * (ut["prompt_tokens"] + ut["completion_tokens"]) / (tot["prompt_tokens"] + tot["completion_tokens"]), 2),
                  stage_completion_share_pct=round(100 * ut["completion_tokens"] / tot["completion_tokens"], 2),
                  alternative_calls=0, denominator="all proxy-logged LLM calls/tokens of the 147 questions (prefix + UT increment)",
                  apart=json.dumps(dict(rule_execution_seconds_total=38.03))))

# ---------------------------------------------------------------- ComEM
lab = {}
per_ds_rows, pair_rows = [], []
for mt, mname in (("q3", "Qwen3-32B"), ("q7", "Qwen2.5-7B")):
    d = K.EXT / f"output/comem_{mt}"
    units, pair_groups_D, pair_groups_A, pair_units = [], collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    sel_calls = sel_pt = sel_ct = fwd = 0
    for tp in sorted(d.glob("*_test.jsonl")):
        ds = tp.name[:-len("_test.jsonl")]
        if ds not in lab:
            lab[ds] = {(r["id_left"], r["id_right"]): r["label"] in ("True", "1", "true")
                       for r in csv.DictReader(open(K.EXT / f"data/llm4em/{ds}.csv"))}
        for line in open(tp):
            r = json.loads(line)
            L = [lab[ds][(r["id_left"], x)] for x in r["id_right"]]
            n = len(L)
            ps = r["preds_select"] or [False] * n                   # official fallback: all False (as analyze_formal_ext)
            pr = r["preds_ranker_single"] or [False] * n
            units.append(dict(key=(ds, r["id_left"]), cluster=ds, A=int(ps != pr), Ys=int(ps == L), Ya=int(pr == L)))
            pd_, pa_ = [int(ps[j] == L[j]) - int(pr[j] == L[j]) for j in range(n)], [int(ps[j] != pr[j]) for j in range(n)]
            pair_groups_D[ds].append(pd_); pair_groups_A[ds].append(pa_)
            pair_units[ds].append(dict(D=pd_, A=pa_, Ys=[int(ps[j] == L[j]) for j in range(n)], Ya=[int(pr[j] == L[j]) for j in range(n)]))
            s = r.get("select") or {}
            if s:
                sel_calls += 1
                sel_pt += (s.get("usage") or {}).get("prompt_tokens") or 0
                sel_ct += (s.get("usage") or {}).get("completion_tokens") or 0
            fwd += r.get("ranker_forward") or 0

    def comem_boot(vals, cl):
        g = collections.defaultdict(list)
        for k, v in vals.items():
            g[cl[k]].append(v)
        return K.strat_boot(g)
    summarise("ComEM", f"{mname} selector, 8 datasets x 400", "Ranker_single", "A_vector (primary)", units, comem_boot)
    for ds in sorted({u["cluster"] for u in units}):
        S = [u for u in units if u["cluster"] == ds]
        act = [u for u in S if u["A"]]
        per_ds_rows.append(dict(model=mname, dataset=ds, n=len(S), n_A=len(act), P_A_pct=round(100 * len(act) / len(S), 2),
                                gains=sum(u["Ys"] > u["Ya"] for u in act), losses=sum(u["Ys"] < u["Ya"] for u in act),
                                both_wrong_A=sum(not u["Ys"] and not u["Ya"] for u in act),
                                total_pp=round(100 * sum(u["Ys"] - u["Ya"] for u in S) / len(S), 2)))
    # pair level (secondary): unit = pair, cluster = anchor, stratum = dataset
    allp = [(dv, av, ys, ya) for ds in pair_units for pu in pair_units[ds] for dv, av, ys, ya in zip(pu["D"], pu["A"], pu["Ys"], pu["Ya"])]
    nP = len(allp); nPA = sum(a for _, a, _, _ in allp)
    totb = K.strat_cluster_boot(pair_groups_D)
    condg = {ds: [[dv for dv, av in zip(pu["D"], pu["A"]) if av] for pu in pair_units[ds]] for ds in pair_units}
    condb = K.strat_cluster_boot(condg)
    pab = K.strat_cluster_boot(pair_groups_A)
    sumD = sum(dv for dv, _, _, _ in allp); sumDA = sum(dv for dv, av, _, _ in allp if av)
    pair_rows.append(dict(model=mname, unit="pair (anchor = cluster, dataset = stratum)", n_pairs=nP, n_A=nPA,
                          P_A_pct=round(100 * nPA / nP, 2), P_A_lo=round(100 * pab["lo"], 2), P_A_hi=round(100 * pab["hi"], 2),
                          gains=sum(1 for dv, av, _, _ in allp if av and dv > 0), losses=sum(1 for dv, av, _, _ in allp if av and dv < 0),
                          both_wrong_A=sum(1 for _, av, ys, ya in allp if av and not ys and not ya),
                          E_D_given_A_pp=round(100 * sumDA / nPA, 2), E_D_given_A_lo=round(100 * condb["lo"], 2), E_D_given_A_hi=round(100 * condb["hi"], 2),
                          total_pp=round(100 * sumD / nP, 3), total_lo=round(100 * totb["lo"], 3), total_hi=round(100 * totb["hi"], 3),
                          identity_residual_pp=round(100 * (sumD / nP - (nPA / nP) * (sumDA / nPA)), 6),
                          A0_with_D_nonzero=sum(1 for dv, av, _, _ in allp if not av and dv != 0),
                          stage_pair_acc_pct=round(100 * sum(ys for _, _, ys, _ in allp) / nP, 2),
                          alt_pair_acc_pct=round(100 * sum(ya for _, _, _, ya in allp) / nP, 2)))
    COSTS.append(dict(system="ComEM", cell=f"{mname} selector, 8 datasets x 400", stage="Selecting (selector LLM over top-4)",
                      stage_calls=sel_calls, all_calls=sel_calls, stage_call_share_pct=100.0,
                      stage_prompt_tokens=sel_pt, stage_completion_tokens=sel_ct, all_tokens=sel_pt + sel_ct, stage_token_share_pct=100.0,
                      stage_completion_share_pct=100.0, alternative_calls=0,
                      denominator="LLM calls/tokens of the test run: the selector is the only LLM; the shared Flan-T5-XL ranker is not an LLM call and is reported apart",
                      apart=json.dumps(dict(flan_t5_xl_forward_passes_test=fwd,
                                            stage_share_of_all_model_invocations_pct=round(100 * sel_calls / (sel_calls + fwd), 2)))))


def write(name, rows):
    with open(K.OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list({k: None for r in rows for k in r})); w.writeheader(); w.writerows(rows)


write("decomposition.csv", ROWS); write("movers.csv", MOVERS); write("costs.csv", COSTS)
write("comem_per_dataset.csv", per_ds_rows); write("comem_pair_level.csv", pair_rows)
(K.OUT / "decomposition.json").write_text(json.dumps(dict(rows=ROWS, movers=MOVERS, costs=COSTS, comem_pair_level=pair_rows,
                                                         comem_per_dataset=per_ds_rows), indent=1, default=str) + "\n")
for r in ROWS:
    print({k: r[k] for k in ("system", "cell", "alternative", "A_definition", "n", "n_A", "P_A_pct", "gains", "losses", "both_wrong_A",
                             "both_right_A", "E_D_given_A_pp", "E_D_given_A_lo", "E_D_given_A_hi", "total_pp", "total_lo", "total_hi",
                             "identity_residual_pp", "A0_with_D_nonzero")})
for r in MOVERS:
    print(r)
for r in COSTS:
    print(r)
for r in pair_rows:
    print(r)
