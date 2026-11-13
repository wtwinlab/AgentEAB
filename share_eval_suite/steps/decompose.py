"""Stage decomposition, Eq. 4: the change rate times the gain when the stage changes the result.

Same outputs as `share_eval/stage_replacement/decompose.py`.

    python3 -m share_eval_suite.steps.decompose [--out DIR] [--external-run DIR] [--chess-run DIR]
    # reads <out>/macsql_units_<model>_<bench>.csv and <out>/chess_units.csv (the reexec step's unit files)
    # -> <out>/decomposition.csv, decomposition.json, movers.csv, costs.csv,
    #    comem_per_dataset.csv, comem_pair_level.csv

No model call and no SQL execution. D_u = Y_u(stage) - Y_u(rule) in {-1, 0, 1}. Per row: n, n_A, the
change rate P(A) with its interval, the gain E[D | A] (interval only with n_A >= 30), gains, losses,
both-wrong and both-right among changed units, total = mean D with its interval, the identity residual
total - P(A) * E[D | A], and the units with A = 0 and D != 0. Intervals are 95% percentile bootstraps
with 10,000 draws and seed 20261105: SQL rows resample databases with question weights; ComEM rows
resample query records within each dataset (pooled, record-equal); the ComEM pair level resamples
query records (clusters of pairs) within each dataset.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import sys
from typing import Callable, Dict, List, Optional, Sequence

from share_eval_suite import stats
from share_eval_suite.comparisons import replacement
from share_eval_suite.steps import _replacement_common as C

MIN_N = replacement.MIN_CHANGED
MODELS_SQL = (("q3", "Qwen3-32B"), ("q72", "Qwen2.5-72B-AWQ"))
MODELS_EM = (("q3", "Qwen3-32B"), ("q7", "Qwen2.5-7B"))
CHESS_CELL = "Qwen3-32B, BIRD SDS 147"
CHESS_ALTS = ("SC_official", "Valid_vote", "Nonempty_vote")


def database_boot(values: dict, clusters: dict) -> Optional[dict]:
    """Database-cluster bootstrap with question weights."""
    if not values:
        return None
    return stats.cluster_bootstrap(values, clusters, "unit", stats.DRAWS, stats.SEED_STAGE)


def dataset_boot(values: dict, clusters: dict) -> Optional[dict]:
    """Units resampled within each dataset; units enter their dataset in the order of `values`."""
    g = collections.defaultdict(list)
    for k, v in values.items():
        g[clusters[k]].append(v)
    return stats.stratified_bootstrap(g, stats.DRAWS, stats.SEED_STAGE)


def summarise(system, cell, alternative, a_def, U: List[dict], boot: Callable, extra=None) -> dict:
    """One decomposition row. U: dicts with key, cluster, A, Ys, Ya."""
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
    return r


def crosstab(system, cell, alternative, old, new, U) -> dict:
    """Counts of units by the two change indicators `old` and `new`."""
    c = collections.Counter((u[old], u[new]) for u in U)
    return dict(system=system, cell=cell, alternative=alternative, old=old, new=new,
                old1_new1=c[(1, 1)], old1_new0=c[(1, 0)], old0_new1=c[(0, 1)], old0_new0=c[(0, 0)],
                old_missing=sum(v for (o, _), v in c.items() if o not in (0, 1)))


def macsql_cell(mname: str, b: str) -> str:
    return f"{mname}, {'BIRD dev' if b == 'bird' else 'Spider dev'}"


def stage_aggregate(units: List[dict]) -> Dict[str, collections.Counter]:
    """Calls, tokens and latency per MAC-SQL agent summed over all questions."""
    agg = collections.defaultdict(collections.Counter)
    for u in units:
        for s, v in u["calls_by_stage"].items():
            agg[s].update(v)
    return agg


def run(out=C.OUT, external_run=C.EXT, chess_run=C.CHESS, verbose: bool = True) -> dict:
    """Write the decomposition tables; returns the object written to decomposition.json."""
    out, ext, chess = pathlib.Path(out), pathlib.Path(external_run), pathlib.Path(chess_run)
    out.mkdir(parents=True, exist_ok=True)
    rows, movers, costs = [], [], []

    # ------------------------------------------------------------ MAC-SQL
    for mt, mname in MODELS_SQL:
        for b in ("bird", "spider"):
            R = C.read_table(out / f"macsql_units_{mt}_{b}.csv")
            X = json.load(open(ext / f"output/macsql_{mt}_{b}.json"))
            cell = macsql_cell(mname, b)
            base = [dict(key=(int(r["question_id"]),), cluster=r["db_id"], Ys=int(r["Y_stage"]), Ya=int(r["Y_alt"]),
                         A_trigger=int(r["A_trigger"]), A_text=int(r["A_text"]), A_result=int(r["A_result"]),
                         failed="REFINER_FAILED" in r["flags"]) for r in R]
            for a in ("A_result", "A_text", "A_trigger"):
                U = [dict(u, A=u[a]) for u in base]
                rows.append(summarise("MAC-SQL", cell, "Keep (Decomposer final_sql)",
                                      a + (" (primary)" if a == "A_result" else ""), U, database_boot,
                                      dict(refiner_failed_units=sum(u["failed"] for u in base),
                                           refiner_failed_in_A=sum(u["failed"] and u[a] for u in base))))
            movers.append(crosstab("MAC-SQL", cell, "Keep", "A_trigger", "A_result", base))
            movers.append(crosstab("MAC-SQL", cell, "Keep", "A_text", "A_result", base))
            movers.append(crosstab("MAC-SQL", cell, "Keep", "A_trigger", "A_text", base))
            # denominator: all Selector, Decomposer and Refiner calls of the cell; server pings apart
            agg = stage_aggregate(X["units"])
            allc = sum(v["calls"] for v in agg.values())
            allt = sum(v["prompt_tokens"] + v["completion_tokens"] for v in agg.values())
            st = agg["refiner"]
            costs.append(dict(system="MAC-SQL", cell=cell, stage="Refiner LLM", stage_calls=st["calls"], all_calls=allc,
                              stage_call_share_pct=round(100 * st["calls"] / allc, 2),
                              stage_prompt_tokens=st["prompt_tokens"], stage_completion_tokens=st["completion_tokens"],
                              all_tokens=allt,
                              stage_token_share_pct=round(100 * (st["prompt_tokens"] + st["completion_tokens"]) / allt, 2),
                              stage_completion_share_pct=round(100 * st["completion_tokens"]
                                                               / sum(v["completion_tokens"] for v in agg.values()), 2),
                              alternative_calls=0,
                              denominator="Selector+Decomposer+Refiner LLM calls/tokens over all questions (api_trace)",
                              apart=json.dumps(dict(pings=X["proxy_by_stage"].get("ping", 0),
                                                    unattributed=sum(v["calls"] for s, v in agg.items() if s == "unattributed"),
                                                    by_stage={s: v["calls"] for s, v in agg.items()}))))

    # ------------------------------------------------------------ CHESS
    R = C.read_table(out / "chess_units.csv")
    for alt in CHESS_ALTS:
        base = [dict(key=(int(r["question_id"]),), cluster=r["db_id"], Ys=int(r["Y_stage"]), Ya=int(r["Y_alt"]),
                     A_result=int(r["A_result"]), A_sqlcommon=int(r["A_sqlcommon"]), A_text=int(r["text_differs"]),
                     A_old=(int(r["A_old"]) if r["A_old"] != "" else None)) for r in R if r["alternative"] == alt]
        for a in ("A_result", "A_sqlcommon", "A_old"):
            if any(u[a] is None for u in base):
                continue
            rows.append(summarise("CHESS B1", CHESS_CELL, alt, a + (" (primary)" if a == "A_result" else ""),
                                  [dict(u, A=u[a]) for u in base], database_boot))
        movers.append(crosstab("CHESS B1", CHESS_CELL, alt, "A_old", "A_result", base))
        movers.append(crosstab("CHESS B1", CHESS_CELL, alt, "A_sqlcommon", "A_result", base))
        movers.append(crosstab("CHESS B1", CHESS_CELL, alt, "A_text", "A_result", base))
    cq = C.read_table(chess / "output/costs_per_question.csv")
    tot, ut = collections.Counter(), collections.Counter()
    for r in cq:
        for k in ("calls", "prompt_tokens", "completion_tokens"):
            tot[k] += int(r[k])
            if r["part"] == "ut_increment":
                ut[k] += int(r[k])
    costs.append(dict(system="CHESS B1", cell=CHESS_CELL, stage="Unit Tester increment (from UT agent's first call)",
                      stage_calls=ut["calls"], all_calls=tot["calls"], stage_call_share_pct=round(100 * ut["calls"] / tot["calls"], 2),
                      stage_prompt_tokens=ut["prompt_tokens"], stage_completion_tokens=ut["completion_tokens"],
                      all_tokens=tot["prompt_tokens"] + tot["completion_tokens"],
                      stage_token_share_pct=round(100 * (ut["prompt_tokens"] + ut["completion_tokens"])
                                                  / (tot["prompt_tokens"] + tot["completion_tokens"]), 2),
                      stage_completion_share_pct=round(100 * ut["completion_tokens"] / tot["completion_tokens"], 2),
                      alternative_calls=0,
                      denominator="all proxy-logged LLM calls/tokens of the 147 questions (prefix + UT increment)",
                      apart=json.dumps(dict(rule_execution_seconds_total=38.03))))

    # ------------------------------------------------------------ ComEM
    lab = {}
    per_ds_rows, pair_rows = [], []
    for mt, mname in MODELS_EM:
        d = ext / f"output/comem_{mt}"
        units = []
        pair_groups_D, pair_groups_A = collections.defaultdict(list), collections.defaultdict(list)
        pair_units = collections.defaultdict(list)
        sel_calls = sel_pt = sel_ct = fwd = 0
        for tp in sorted(d.glob("*_test.jsonl")):
            ds = tp.name[:-len("_test.jsonl")]
            if ds not in lab:
                with open(ext / f"data/llm4em/{ds}.csv") as fh:
                    lab[ds] = {(r["id_left"], r["id_right"]): r["label"] in ("True", "1", "true")
                               for r in csv.DictReader(fh)}
            for line in open(tp):
                r = json.loads(line)
                L = [lab[ds][(r["id_left"], x)] for x in r["id_right"]]
                n = len(L)
                ps = r["preds_select"] or [False] * n           # a missing decision counts as all False
                pr = r["preds_ranker_single"] or [False] * n
                units.append(dict(key=(ds, r["id_left"]), cluster=ds, A=int(ps != pr), Ys=int(ps == L), Ya=int(pr == L)))
                pd_ = [int(ps[j] == L[j]) - int(pr[j] == L[j]) for j in range(n)]
                pa_ = [int(ps[j] != pr[j]) for j in range(n)]
                pair_groups_D[ds].append(pd_)
                pair_groups_A[ds].append(pa_)
                pair_units[ds].append(dict(D=pd_, A=pa_, Ys=[int(ps[j] == L[j]) for j in range(n)],
                                           Ya=[int(pr[j] == L[j]) for j in range(n)]))
                s = r.get("select") or {}
                if s:
                    sel_calls += 1
                    sel_pt += (s.get("usage") or {}).get("prompt_tokens") or 0
                    sel_ct += (s.get("usage") or {}).get("completion_tokens") or 0
                fwd += r.get("ranker_forward") or 0
        rows.append(summarise("ComEM", f"{mname} selector, 8 datasets x 400", "Ranker_single", "A_vector (primary)",
                              units, dataset_boot))
        for ds in sorted({u["cluster"] for u in units}):
            S = [u for u in units if u["cluster"] == ds]
            act = [u for u in S if u["A"]]
            per_ds_rows.append(dict(model=mname, dataset=ds, n=len(S), n_A=len(act), P_A_pct=round(100 * len(act) / len(S), 2),
                                    gains=sum(u["Ys"] > u["Ya"] for u in act), losses=sum(u["Ys"] < u["Ya"] for u in act),
                                    both_wrong_A=sum(not u["Ys"] and not u["Ya"] for u in act),
                                    total_pp=round(100 * sum(u["Ys"] - u["Ya"] for u in S) / len(S), 2)))
        # pair level: unit = pair, cluster = query record, stratum = dataset
        allp = [(dv, av, ys, ya) for ds in pair_units for pu in pair_units[ds]
                for dv, av, ys, ya in zip(pu["D"], pu["A"], pu["Ys"], pu["Ya"])]
        nP = len(allp)
        nPA = sum(a for _, a, _, _ in allp)
        totb = stats.stratified_cluster_bootstrap(pair_groups_D, stats.DRAWS, stats.SEED_STAGE)
        condg = {ds: [[dv for dv, av in zip(pu["D"], pu["A"]) if av] for pu in pair_units[ds]] for ds in pair_units}
        condb = stats.stratified_cluster_bootstrap(condg, stats.DRAWS, stats.SEED_STAGE)
        pab = stats.stratified_cluster_bootstrap(pair_groups_A, stats.DRAWS, stats.SEED_STAGE)
        sumD = sum(dv for dv, _, _, _ in allp)
        sumDA = sum(dv for dv, av, _, _ in allp if av)
        pair_rows.append(dict(model=mname, unit="pair (anchor = cluster, dataset = stratum)", n_pairs=nP, n_A=nPA,
                              P_A_pct=round(100 * nPA / nP, 2), P_A_lo=round(100 * pab["lo"], 2), P_A_hi=round(100 * pab["hi"], 2),
                              gains=sum(1 for dv, av, _, _ in allp if av and dv > 0),
                              losses=sum(1 for dv, av, _, _ in allp if av and dv < 0),
                              both_wrong_A=sum(1 for _, av, ys, ya in allp if av and not ys and not ya),
                              E_D_given_A_pp=round(100 * sumDA / nPA, 2), E_D_given_A_lo=round(100 * condb["lo"], 2),
                              E_D_given_A_hi=round(100 * condb["hi"], 2),
                              total_pp=round(100 * sumD / nP, 3), total_lo=round(100 * totb["lo"], 3),
                              total_hi=round(100 * totb["hi"], 3),
                              identity_residual_pp=round(100 * (sumD / nP - (nPA / nP) * (sumDA / nPA)), 6),
                              A0_with_D_nonzero=sum(1 for dv, av, _, _ in allp if not av and dv != 0),
                              stage_pair_acc_pct=round(100 * sum(ys for _, _, ys, _ in allp) / nP, 2),
                              alt_pair_acc_pct=round(100 * sum(ya for _, _, _, ya in allp) / nP, 2)))
        costs.append(dict(system="ComEM", cell=f"{mname} selector, 8 datasets x 400", stage="Selecting (selector LLM over top-4)",
                          stage_calls=sel_calls, all_calls=sel_calls, stage_call_share_pct=100.0,
                          stage_prompt_tokens=sel_pt, stage_completion_tokens=sel_ct, all_tokens=sel_pt + sel_ct,
                          stage_token_share_pct=100.0, stage_completion_share_pct=100.0, alternative_calls=0,
                          denominator="LLM calls/tokens of the test run: the selector is the only LLM; the shared "
                                      "Flan-T5-XL ranker is not an LLM call and is reported apart",
                          apart=json.dumps(dict(flan_t5_xl_forward_passes_test=fwd,
                                                stage_share_of_all_model_invocations_pct=round(100 * sel_calls / (sel_calls + fwd), 2)))))

    C.write_table(out / "decomposition.csv", rows)
    C.write_table(out / "movers.csv", movers)
    C.write_table(out / "costs.csv", costs)
    C.write_table(out / "comem_per_dataset.csv", per_ds_rows)
    C.write_table(out / "comem_pair_level.csv", pair_rows)
    obj = dict(rows=rows, movers=movers, costs=costs, comem_pair_level=pair_rows, comem_per_dataset=per_ds_rows)
    C.write_json(out / "decomposition.json", obj, default=str)
    if verbose:
        for r in rows:
            print({k: r[k] for k in ("system", "cell", "alternative", "A_definition", "n", "n_A", "P_A_pct", "gains",
                                     "losses", "E_D_given_A_pp", "total_pp", "total_lo", "total_hi", "A0_with_D_nonzero")})
    return obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = C.parse_args(__doc__.splitlines()[0], argv)
    run(a.out, a.external_run, a.chess_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
