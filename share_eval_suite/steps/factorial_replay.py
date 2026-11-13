r"""Factorial replay of the logged BIRD debates: every reading of the same responses, no model call.

    python3 -m share_eval_suite.steps.factorial_replay              # BIRD debates, Llama-3-8B and Qwen2.5-7B
    python3 -m share_eval_suite.steps.factorial_replay --ext data/records/bird_debates_backbones \
            --out output/share_eval_suite/replay_backbones         # larger and cross-family backbones

Same outputs as `share_eval/replay/factorial_replay.py`.

Four two-level factors are read from one task set per (protocol, backbone):

- E extractor: the protocol's own extracted answer (strict) | lenient (the protocol's answer, else the
  last fenced SQL block, else the last line-initial SELECT/WITH), applied to every round;
- I vote: exact query string (string) | largest execution-result group, failed executions dropped
  (exec); a secondary vote `exec_ident` puts failed executions in one group that can win and scores 0;
- R round: the independent initial answers (r0) | every agent's answer in the run's last round (last);
- A peers: without (NOPEER) | with (FULL).

Ties go to the earliest tied answer in agent order (sensitivity: the last, or a seeded random one).
The question is the unit: repeats are averaged within a question before any contrast, and intervals
are 95% percentile intervals over 10,000 resamples of the questions (seed 20261106). An effect of a
factor set S is the contrast of the 16 cells with weights prod_{j in S} sign_j * 2^|S| / 16
(difference of differences). The reference reading is (strict, string, last, FULL); the protocol's
returned answer (`final`) is a reference row, never a factor level. MoA is replayed by the same code
and reported separately: its last round is the aggregator's single answer.

Outputs in `--out`: `per_run_scores.jsonl` (one row per run: correctness under every extractor × vote
× round × tie rule), `factorial_replay.json` and `.csv` (levels, effects, ledger and intervals),
`ties.csv`, the execution caches the run adds (`exec_signature_cache.jsonl`, `verdict_cache.jsonl`)
and `SHA256SUMS`.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import itertools
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from share_eval_suite import paths, stats
from share_eval_suite.comparisons import interactions
from share_eval_suite.records import readers as R
from share_eval_suite.steps import _replay_common as C

sys.dont_write_bytecode = True

REPO = paths.REPO
RUN_DIR = REPO / "data" / "records" / "bird_debates"
OUT_DIR = C.OUTPUT / "replay"

BOOT = stats.DRAWS
SEED = stats.SEED_REPLAY
INTERACTIVE, AGG_ONLY, ARMS = R.INTERACTIVE, R.AGG_ONLY, R.ARMS
EXTS, ROUNDS, TIES = R.EXTRACTORS, R.ROUNDS, R.TIES
FACTORS = ("E", "I", "R", "A")
LEVELS = {"E": EXTS, "I": ("string", "exec"), "R": ROUNDS, "A": ARMS}
REF = {"E": 0, "I": 0, "R": 1, "A": 1}          # strict, string, last, FULL
FAILV = {"syntax_error", "missing_object", "other_exec_error", "type_or_arg_error", "exec_timeout", "declined"}

qkey = R.query_key


# ------------------------------------------------------------------ estimators
def boot_idx(n: int) -> np.ndarray:
    return stats.resample_index(n, BOOT, SEED)


def summ(d: np.ndarray, idx: np.ndarray) -> Dict[str, Any]:
    """Mean of per-question values (pp), its percentile interval and two-sided bootstrap p (floor 1/BOOT)."""
    b = d[idx].mean(axis=1)
    lo, hi = stats.percentile_interval(b)
    p = stats.bootstrap_p(b, floor=BOOT)
    return dict(est=round(100 * float(d.mean()), 2), lo=round(100 * lo, 2),
                hi=round(100 * hi, 2), p=round(float(p), 4))


def _cell(c) -> str:
    return f"{EXTS[c[0]]}|{LEVELS['I'][c[1]]}|{ROUNDS[c[2]]}|{ARMS[c[3]]}"


def analyse_cell(units: List[str], Y: np.ndarray, F: Optional[np.ndarray], idx: np.ndarray) -> Dict[str, Any]:
    """Levels, factorial effects, peer effect per (E, I), simple effects, the shared-reference ledger and
    the returned-answer reference row of one question array Y[q, E, I, R, A] (F[q, A]: returned answer)."""
    out: Dict[str, Any] = dict(n_units=len(units))
    out["levels"] = {_cell((e, i, r, a)): round(100 * float(Y[:, e, i, r, a].mean()), 2)
                     for e, i, r, a in itertools.product((0, 1), repeat=4)}
    out["effects"] = interactions.factorial_effects(Y)
    # peer effect: (last - r0) with peers minus (last - r0) without, per (E, I)
    dp = {}
    for e, i in itertools.product((0, 1), repeat=2):
        name = f"{EXTS[e]}|{LEVELS['I'][i]}"
        d = (Y[:, e, i, 1, 1] - Y[:, e, i, 0, 1]) - (Y[:, e, i, 1, 0] - Y[:, e, i, 0, 0])
        dp[name] = summ(d, idx)
        dp[name]["last_minus_r0_FULL"] = summ(Y[:, e, i, 1, 1] - Y[:, e, i, 0, 1], idx)
        dp[name]["last_minus_r0_NOPEER"] = summ(Y[:, e, i, 1, 0] - Y[:, e, i, 0, 0], idx)
    out["delta_peer"] = dp
    # simple effect of every factor at every combination of the other three
    simple = {}
    for f in range(4):
        others = [j for j in range(4) if j != f]
        for combo in itertools.product((0, 1), repeat=3):
            hi_, lo_ = [0] * 4, [0] * 4
            for j, v in zip(others, combo):
                hi_[j] = lo_[j] = v
            hi_[f], lo_[f] = 1, 0
            d = Y[:, hi_[0], hi_[1], hi_[2], hi_[3]] - Y[:, lo_[0], lo_[1], lo_[2], lo_[3]]
            tag = "|".join(f"{FACTORS[j]}={LEVELS[FACTORS[j]][v]}" for j, v in zip(others, combo))
            simple[f"{FACTORS[f]}@{tag}"] = summ(d, idx)
    out["simple_effects"] = simple
    # shared-reference ledger: joint change of switching a factor set, sum of single switches, joint - sum
    ref = [REF[f] for f in FACTORS]
    yref = Y[:, ref[0], ref[1], ref[2], ref[3]]
    single = {}
    for j, f in enumerate(FACTORS):
        c = list(ref)
        c[j] = 1 - c[j]
        single[f] = Y[:, c[0], c[1], c[2], c[3]] - yref
    led: Dict[str, Any] = dict(reference_cell="strict|string|last|FULL", reference_level=round(100 * float(yref.mean()), 2))
    rows = []
    for k in range(1, 5):
        for S in itertools.combinations(range(4), k):
            c = list(ref)
            for j in S:
                c[j] = 1 - c[j]
            joint = Y[:, c[0], c[1], c[2], c[3]] - yref
            ssum = sum(single[FACTORS[j]] for j in S)
            rows.append(dict(switched="+".join(FACTORS[j] for j in S), cell=_cell(c),
                             level=round(100 * float(Y[:, c[0], c[1], c[2], c[3]].mean()), 2),
                             joint=summ(joint, idx), sum_of_singles=round(100 * float(ssum.mean()), 2),
                             nonadditivity=summ(joint - ssum, idx) if k > 1 else None))
    led["rows"] = rows
    out["ledger"] = led
    if F is not None:
        out["final_reference"] = dict(
            level_final_FULL=round(100 * float(F[:, 1].mean()), 2),
            level_final_NOPEER=round(100 * float(F[:, 0].mean()), 2),
            final_minus_refcell_FULL=summ(F[:, 1] - yref, idx),
            final_minus_lenient_exec_r0_FULL=summ(F[:, 1] - Y[:, 1, 1, 0, 1], idx),
            final_minus_lenient_exec_last_FULL=summ(F[:, 1] - Y[:, 1, 1, 1, 1], idx),
            final_FULL_minus_NOPEER=summ(F[:, 1] - F[:, 0], idx))
    return out


def _flat_rows(name: str, res: Dict[str, Any]):
    """Rows of factorial_replay.csv for one cell."""
    keys = ("est", "lo", "hi", "p")
    for k, v in res["effects"].items():
        yield dict(cell=name, table="effect", term=k, **{x: v[x] for x in keys})
    for k, v in res["delta_peer"].items():
        yield dict(cell=name, table="delta_peer", term=k, **{x: v[x] for x in keys})
    for k, v in res["simple_effects"].items():
        yield dict(cell=name, table="simple_effect", term=k, **{x: v[x] for x in keys})
    for r in res["ledger"]["rows"]:
        yield dict(cell=name, table="ledger_joint", term=r["switched"], **{x: r["joint"][x] for x in keys})
        if r["nonadditivity"]:
            yield dict(cell=name, table="ledger_nonadditivity", term=r["switched"],
                       **{x: r["nonadditivity"][x] for x in keys})
    for k, v in res["levels"].items():
        yield dict(cell=name, table="level", term=k, est=v, lo=None, hi=None, p=None)
    for k, v in (res.get("final_reference") or {}).items():
        if isinstance(v, dict):
            yield dict(cell=name, table="final_reference", term=k, **{x: v[x] for x in keys})
        else:
            yield dict(cell=name, table="final_reference", term=k, est=v, lo=None, hi=None, p=None)


# ------------------------------------------------------------------ run
def run(ext=RUN_DIR, out=OUT_DIR, per_unit=None, verdicts_path=None,
        seed_sig_caches: Sequence = (C.CACHE / "r0_exec_signatures.jsonl", C.CACHE / "replay_exec_signature_cache.jsonl"),
        verdict_caches: Sequence = (C.CACHE / "replay_verdict_cache.jsonl",),
        workers: int = 8, protocols: Sequence[str] = INTERACTIVE + AGG_ONLY,
        exclude_units: Sequence[str] = ("bird|518", "bird|701"),
        audit_verdicts=C.CACHE / "verdict_audit.json") -> int:
    """Score every run under every reading and write the outputs to `out`.

    Execution results and verdicts come from the stored caches (`seed_sig_caches`, then
    `verdict_caches` over the run directory's stored verdicts, then `<out>`'s own caches). A query
    without a stored result is executed read-only and appended to `<out>`'s caches; the step prints
    how many. The queries named in the audit list, or whose stored signature and verdict disagree on
    whether the query ran, are executed again on this host and the cell levels they would change are
    reported; the primary analysis keeps the stored results."""
    ext = pathlib.Path(ext).resolve()
    out = pathlib.Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    protos = tuple(protocols)
    t_start = time.time()

    per_unit_p = pathlib.Path(per_unit) if per_unit else ext / "analysis/sql_per_unit.jsonl"
    last, per_unit_rows, load_counts = R._load_runs(ext, per_unit_p if per_unit_p.exists() else None, protos)
    rows = R._build_rows(last, per_unit_rows, protos)
    print(f"{len(rows)} runs loaded from {ext}; per_unit={'yes' if per_unit_rows is not None else 'no'}", flush=True)

    gold = C.reference_queries()
    verd_p = pathlib.Path(verdicts_path) if verdicts_path else ext / "analysis/sql_verdicts.json"
    verdicts = json.loads(verd_p.read_text())["verdicts"] if verd_p.exists() else {}
    own_v = out / "verdict_cache.jsonl"
    for k, v in C.load_jsonl_cache([pathlib.Path(p) for p in verdict_caches if p] + [own_v]).items():
        verdicts[k] = v
    own_s = out / "exec_signature_cache.jsonl"
    sigs = C.load_jsonl_cache([pathlib.Path(p) for p in seed_sig_caches if p] + [own_s])

    # ---- queries without a stored execution result or verdict
    todo, gtodo = {}, {}
    for r in rows:
        for rd in ROUNDS:
            for e in EXTS:
                for q in r["sets"][rd][e]:
                    if q and r["db"]:
                        kk = qkey(r["db"], q)
                        if kk not in sigs:
                            todo[kk] = (r["db"], q)
                        if kk not in verdicts:
                            gtodo[kk] = (r["db"], q, gold[r["unit"]])
    print(f"to sign {len(todo)}; to judge {len(gtodo)}", flush=True)
    if todo or gtodo:
        print(f"executing {len(todo)} queries for their result and {len(gtodo)} for their verdict, read-only, "
              f"against {C.bird_dir()}/dev_databases (a missing database is recorded as an execution error)",
              flush=True)
    t0 = time.time()
    if todo or gtodo:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            with open(own_s, "a") as fh:
                for kk, v in ex.map(C.signature, list(todo.values()), chunksize=16):
                    v["k"] = kk
                    sigs[kk] = v
                    fh.write(json.dumps(v) + "\n")
            with open(own_v, "a") as fh:
                for kk, v in ex.map(C.verdict, list(gtodo.values()), chunksize=16):
                    verdicts[kk] = v
                    fh.write(json.dumps(dict(v, k=kk)) + "\n")
    exec_cost = dict(new_signatures=len(todo), new_verdicts=len(gtodo), wall_s=round(time.time() - t0, 1))

    # ---- questions excluded: no reference result (named a priori, or a stored verdict says so)
    a_priori = {u for u in exclude_units if u}
    gold_failed_units = set(a_priori)
    data_driven = set()
    for r in rows:
        if r["final_outcome"] == "gold_failed":
            data_driven.add(r["unit"])
        for rd in ROUNDS:
            for e in EXTS:
                for q in r["sets"][rd][e]:
                    if q and (verdicts.get(qkey(r["db"], q)) or {}).get("outcome") == "gold_failed":
                        data_driven.add(r["unit"])
    gold_failed_units |= data_driven
    excl = dict(gold_failed_units=sorted(gold_failed_units), a_priori_units=sorted(a_priori),
                data_driven_gold_failed_units=sorted(data_driven),
                data_driven_not_in_a_priori=sorted(data_driven - a_priori),
                gold_failed_runs_in_per_unit=sum(1 for r in rows if r["final_outcome"] == "gold_failed"),
                runs_dropped_by_unit_exclusion=sum(1 for r in rows if r["unit"] in gold_failed_units),
                no_db_id_runs=sum(1 for r in rows if not r["db"]),
                load_counts=load_counts)

    # ---- per-run scores
    scored = collections.defaultdict(lambda: collections.defaultdict(list))   # (proto, bb) -> question -> runs
    tie_acc = collections.defaultdict(lambda: collections.defaultdict(list))
    run_dump = []
    for r in rows:
        if r["unit"] in gold_failed_units or not r["db"]:
            continue
        s, t, lv = R._score_run(r, sigs, verdicts)
        rec = dict(arm=r["arm"], repeat=r["repeat"], s=s,
                   final=None if r["final_outcome"] is None else float(r["final_outcome"] == "match"),
                   last_round=r["last_round"])
        scored[(r["protocol"], r["backbone"])][r["unit"]].append(rec)
        for key in t:
            tie_acc[(r["protocol"], r["backbone"], r["arm"])][key].append((t[key], lv[key]))
        run_dump.append(dict(unit=r["unit"], protocol=r["protocol"], arm=r["arm"], backbone=r["backbone"],
                             repeat=r["repeat"], last_round=r["last_round"], final=rec["final"],
                             **{"|".join(k): v for k, v in s.items()}))

    def build_Y(by_unit, ident_set, keep=lambda x: True, tie="first"):
        """Questions with runs in both peer conditions; Y[q, E, I, R, A] and F[q, A], repeats averaged."""
        units = sorted(u for u, rs in by_unit.items() if {x["arm"] for x in rs if keep(x)} >= set(ARMS))
        Y = np.zeros((len(units), 2, 2, 2, 2))
        F = np.zeros((len(units), 2)) if per_unit_rows is not None else None
        for ui, u in enumerate(units):
            for a_i, arm in enumerate(ARMS):
                rs = [x for x in by_unit[u] if x["arm"] == arm and keep(x)]
                for e, i, rd in itertools.product((0, 1), repeat=3):
                    Y[ui, e, i, rd, a_i] = np.mean([x["s"][(EXTS[e], ident_set[i], ROUNDS[rd], tie)] for x in rs])
                if F is not None:
                    F[ui, a_i] = np.mean([x["final"] for x in rs])
        return units, Y, F

    cells: Dict[str, Any] = {}
    pooled_inputs: Dict = {}
    partial_units = {}
    pooled_tie = collections.defaultdict(dict)
    for (proto, bb), by_unit in sorted(scored.items()):
        units, Y, F = build_Y(by_unit, ("string", "exec"))
        partial_units[f"{proto}|{bb}"] = sorted(set(by_unit) - set(units))
        idx = boot_idx(len(units))
        res: Dict[str, Any] = {}
        res["primary"] = analyse_cell(units, Y, F, idx)
        if proto in INTERACTIVE:
            pooled_inputs[(proto, bb)] = (units, Y, F)
        u2, Y2, F2 = build_Y(by_unit, ("string", "exec_ident"))
        r_ = analyse_cell(u2, Y2, F2, boot_idx(len(u2)))
        res["secondary_exec_ident"] = {k: r_[k] for k in ("n_units", "levels", "effects", "delta_peer")}
        res["secondary_exec_ident"]["note"] = "identity level 1 = exec_ident (failures form one cluster; a FAIL win scores 0)"
        # completeness: questions with all 16 cells, and with both repeats in both peer conditions
        reps = {u: collections.Counter(x["arm"] for x in rs) for u, rs in by_unit.items()}
        res["completeness"] = dict(
            tasks_with_runs=len(by_unit), tasks_all_16_cells=len(units),
            tasks_both_repeats_both_arms=sum(1 for c in reps.values() if all(c[a] >= 2 for a in ARMS)))
        # each repeat on its own
        byrep = {}
        for rep in sorted({x["repeat"] for rs in by_unit.values() for x in rs}):
            ur, Yr, Fr = build_Y(by_unit, ("string", "exec"), keep=lambda x, rep=rep: x["repeat"] == rep)
            rr = analyse_cell(ur, Yr, Fr, boot_idx(len(ur)))
            byrep[f"r{rep}"] = {k: rr[k] for k in ("n_units", "delta_peer", "effects")}
        res["by_repeat"] = byrep
        # early stopping: runs whose last round is the initial round
        early = [x for rs in by_unit.values() for x in rs if x["last_round"] == 0]
        early_units = {u for u, rs in by_unit.items() if any(x["last_round"] == 0 for x in rs)}
        res["early_stop"] = dict(runs_last_eq_r0=len(early),
                                 by_arm={a: sum(1 for x in early if x["arm"] == a) for a in ARMS},
                                 tasks_with_any_early_stop=len(early_units),
                                 treatment="primary: last = r0 for those runs (the deployed protocol stops)")
        if early and proto in INTERACTIVE:
            sub = {u: rs for u, rs in by_unit.items() if u not in early_units}
            us, Ys, Fs = build_Y(sub, ("string", "exec"))
            if len(us) >= 10:
                rs_ = analyse_cell(us, Ys, Fs, boot_idx(len(us)))
                res["early_stop"]["sensitivity_excluding_tasks_with_any_early_stop"] = {
                    k: rs_[k] for k in ("n_units", "effects", "delta_peer", "ledger")}
                res["early_stop"]["sensitivity_note"] = ("drops every task where any run stopped at round 0; "
                                                         "selection on a post-treatment event, so descriptive only")
        # tie rule sensitivity (every vote-based cell)
        tsens = {}
        for tie in TIES[1:]:
            ut, Yt, Ft = build_Y(by_unit, ("string", "exec"), tie=tie)
            rt = analyse_cell(ut, Yt, Ft, idx)
            if proto in INTERACTIVE:
                pooled_tie[tie][(proto, bb)] = (ut, Yt, Ft)
            tsens[tie] = {k: rt[k] for k in ("n_units", "levels", "effects", "delta_peer", "ledger", "final_reference")
                          if k in rt}
        res["tie_sensitivity"] = tsens
        tt = {}
        for arm in ARMS:
            for key, v in tie_acc[(proto, bb, arm)].items():
                tt[f"{'|'.join(key)}|{arm}"] = dict(tie_share=round(float(np.mean([x[0] for x in v])), 4),
                                                    no_live_label_share=round(1 - float(np.mean([x[1] for x in v])), 4))
        res["ties"] = tt
        allr = [x for rs in by_unit.values() for x in rs]
        res["share_runs_last_round_gt0"] = {arm: round(float(np.mean([x["last_round"] > 0 for x in allr if x["arm"] == arm])), 4)
                                            for arm in ARMS}
        res["runs"] = len(allr)
        res["role"] = "aggregation_only (R=last is the aggregator's single answer)" if proto in AGG_ONLY else "interactive"
        cells[f"{proto}|{bb}"] = res
        pr = res["primary"]
        print(f"{proto:16s} {bb:11s} n={pr['n_units']} ref={pr['ledger']['reference_level']:6.2f} "
              f"E={pr['effects']['E']['est']:+6.2f} I={pr['effects']['I']['est']:+6.2f} "
              f"R={pr['effects']['R']['est']:+6.2f} A={pr['effects']['A']['est']:+6.2f} "
              f"RxA={pr['effects']['RxA']['est']:+6.2f}", flush=True)

    # ---- equal-weight mean over the interactive (protocol, backbone) cells on their common questions
    def _pooled_arrays(inp, common):
        Ys, Fs = [], []
        for (units, Y, F) in inp.values():
            pos = {u: i for i, u in enumerate(units)}
            sel = [pos[u] for u in common]
            Ys.append(Y[sel])
            if F is not None:
                Fs.append(F[sel])
        return np.mean(Ys, axis=0), (np.mean(Fs, axis=0) if Fs else None)

    common = sorted(set.intersection(*[set(v[0]) for v in pooled_inputs.values()])) if pooled_inputs else []
    pooled = None
    if common:
        Ym, Fm = _pooled_arrays(pooled_inputs, common)
        pooled = analyse_cell(common, Ym, Fm, boot_idx(len(common)))
        pooled["note"] = ("equal-weight average over the interactive (protocol, backbone) cells, per task; "
                          "tasks resampled; same source version; not independent replications")
        pooled["cells_averaged"] = [f"{p}|{b}" for p, b in pooled_inputs]
        pooled["tie_sensitivity"] = {}
        for tie, inp in pooled_tie.items():
            Ym, Fm = _pooled_arrays(inp, common)
            rt = analyse_cell(common, Ym, Fm, boot_idx(len(common)))
            pooled["tie_sensitivity"][tie] = {k: rt[k] for k in ("levels", "effects", "delta_peer", "ledger", "final_reference") if k in rt}

    # ---- execution-class inconsistencies: stored results against a fresh execution on this host
    cand = {}
    for r in rows:
        for rd in ROUNDS:
            for e in EXTS:
                for q in r["sets"][rd][e]:
                    if q and r["db"]:
                        cand.setdefault(qkey(r["db"], q), (r["db"], q, gold[r["unit"]], r["unit"]))
    suspects = {}
    ap_ = pathlib.Path(audit_verdicts) if audit_verdicts else None
    if ap_ is not None and ap_.exists():
        av = json.loads(ap_.read_text())
        for x in (av.get("fresh_vs_stored") or {}).get("differing", []) + \
                 (av.get("audited_sig_vs_stored") or {}).get("inconsistent", []):
            suspects[x["k"]] = "audit_list"
    for kk in cand:
        sg, vd = sigs.get(kk) or {}, verdicts.get(kk) or {}
        if not sg or not vd:
            continue
        if (sg.get("status") == "ok") != (vd.get("outcome") not in FAILV):
            suspects.setdefault(kk, "signature_vs_verdict_class")
    sus_in = {kk: v for kk, v in suspects.items() if kk in cand}
    alt_s, alt_v = dict(sigs), dict(verdicts)
    if sus_in:
        print(f"re-executing {len(sus_in)} suspect queries read-only on this host", flush=True)
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            items = [cand[kk] for kk in sus_in]
            for kk, v in ex.map(C.signature, [(d, q) for d, q, _g, _u in items]):
                alt_s[kk] = v
            for kk, v in ex.map(C.verdict, [(d, q, g) for d, q, g, _u in items]):
                alt_v[kk] = v
    affected = []
    if sus_in:
        hit_units = {cand[kk][3] for kk in sus_in}
        delta = collections.defaultdict(float)
        for r in rows:
            if r["unit"] in gold_failed_units or not r["db"] or r["unit"] not in hit_units:
                continue
            qs = {qkey(r["db"], q) for rd in ROUNDS for e in EXTS for q in r["sets"][rd][e] if q}
            if not qs & set(sus_in):
                continue
            s0, _, _ = R._score_run(r, sigs, verdicts)
            s1, _, _ = R._score_run(r, alt_s, alt_v)
            nrep = len([x for x in scored[(r["protocol"], r["backbone"])][r["unit"]] if x["arm"] == r["arm"]])
            ntask = cells[f"{r['protocol']}|{r['backbone']}"]["primary"]["n_units"]
            for kx in s0:
                if s0[kx] != s1[kx]:
                    delta[(r["protocol"], r["backbone"], r["arm"]) + kx] += 100 * (s1[kx] - s0[kx]) / nrep / ntask
        affected = [dict(cell=f"{k[0]}|{k[1]}", arm=k[2], ext=k[3], ident=k[4], round=k[5], tie=k[6],
                         level_change_pp=round(v, 3)) for k, v in sorted(delta.items()) if abs(v) > 1e-9]
    inconsistency = dict(
        suspects_total=len(suspects), suspects_in_candidate_sets=len(sus_in),
        suspects=[dict(k=kk, source=v, unit=cand[kk][3],
                       stored_or_cached=dict(sig_status=(sigs.get(kk) or {}).get("status"),
                                             verdict=(verdicts.get(kk) or {}).get("outcome")),
                       this_host=dict(sig_status=(alt_s.get(kk) or {}).get("status"),
                                      verdict=(alt_v.get(kk) or {}).get("outcome"))) for kk, v in sus_in.items()],
        affected_cell_levels=affected,
        max_abs_level_change_pp=round(max([abs(x["level_change_pp"]) for x in affected] or [0.0]), 3),
        note="primary analysis keeps the cached verdicts; this re-executes only the suspect queries on this host")

    blob = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), model_calls=0,
                source=str(ext.relative_to(REPO) if str(ext).startswith(str(REPO)) else ext),
                per_unit_used=per_unit_rows is not None, boot_draws=BOOT, seed=SEED,
                reference_cell="strict|string|last|FULL", effect_convention="difference-of-differences, pp",
                exclusions=excl, verdict_inconsistency_recheck=inconsistency, partial_units_dropped=partial_units,
                execution_cost=exec_cost, pooled_interactive=pooled, cells=cells,
                wall_s=round(time.time() - t_start, 1))
    (out / "factorial_replay.json").write_text(json.dumps(blob, indent=1, default=str))
    with open(out / "per_run_scores.jsonl", "w") as fh:
        for d in run_dump:
            fh.write(json.dumps(d) + "\n")

    flat = []
    for name, res in cells.items():
        flat += list(_flat_rows(name, res["primary"]))
    if pooled:
        flat += list(_flat_rows("POOLED_interactive", pooled))
    with open(out / "factorial_replay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "table", "term", "est", "lo", "hi", "p"])
        w.writeheader()
        w.writerows(flat)
    with open(out / "ties.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "ext|ident|round|arm", "tie_share", "no_live_label_share"])
        for name, res in cells.items():
            for k, v in res["ties"].items():
                w.writerow([name, k, v["tie_share"], v["no_live_label_share"]])

    C.write_checksums(out)
    print(f"exclusions: {excl['gold_failed_units']} ({excl['runs_dropped_by_unit_exclusion']} runs); "
          f"exec {exec_cost}; wall {time.time() - t_start:.0f}s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Factorial replay of the logged BIRD debates")
    ap.add_argument("--ext", default=str(RUN_DIR), help="run directory with raw/sql_bird_v2__*.jsonl")
    ap.add_argument("--per-unit", default=None,
                    help="evaluated set with stored final outcomes (default <ext>/analysis/sql_per_unit.jsonl; "
                         "if absent, all ok records are used and no final reference row is produced)")
    ap.add_argument("--verdicts", default=None,
                    help="stored verdict json (default <ext>/analysis/sql_verdicts.json; optional)")
    ap.add_argument("--seed-sig-cache", default=",".join(str(C.CACHE / n) for n in (
                        "r0_exec_signatures.jsonl", "replay_exec_signature_cache.jsonl")),
                    help="read-only signature cache(s), comma-separated")
    ap.add_argument("--verdict-caches", default=str(C.CACHE / "replay_verdict_cache.jsonl"),
                    help="read-only verdict cache(s), comma-separated, applied before <out>/verdict_cache.jsonl")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--protocols", default=",".join(INTERACTIVE + AGG_ONLY))
    ap.add_argument("--exclude-units", default="bird|518,bird|701",
                    help="tasks dropped a priori: the reference query fails or times out under the 5 s evaluator. "
                         "Data-driven gold_failed tasks are added.")
    ap.add_argument("--audit-verdicts", default=str(C.CACHE / "verdict_audit.json"),
                    help="optional list of stored-vs-fresh execution inconsistencies to re-check")
    a = ap.parse_args(argv)
    return run(ext=a.ext, out=a.out, per_unit=a.per_unit, verdicts_path=a.verdicts,
               seed_sig_caches=[p for p in a.seed_sig_cache.split(",") if p],
               verdict_caches=[p for p in a.verdict_caches.split(",") if p],
               workers=a.workers, protocols=a.protocols.split(","),
               exclude_units=a.exclude_units.split(","), audit_verdicts=a.audit_verdicts)


if __name__ == "__main__":
    sys.exit(main())
