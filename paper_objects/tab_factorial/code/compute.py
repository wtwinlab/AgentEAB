"""Extraction and voting on fixed BIRD debate responses: Table tab:factorial (Section VI-B) and the
numbers that discuss it in Sections VI-B, VI-C and VII-A (with the debate row of Table
tab:replacement-cost), and Supplementary Tables tab:s-contrasts (rows R1, R3-R11) and tab:s-levels.

    python3 code/compute.py --out output

Reads only ../inputs. Estimates are in pp (levels in %). Intervals are 95% percentile bootstrap
intervals over questions (10,000 draws, seed 20261106, repeats averaged within a question before
resampling; one index matrix shared by every contrast, as in the source analysis). Database
intervals resample the 11 BIRD databases with replacement (10,000 sequential draws, seed 20261106).

Notation of the per-run score keys: extractor|vote|round|tie, with extractor strict (the protocol's own
extractor) or lenient, vote string or exec (queries grouped by execution result, failing queries
dropped), round r0 (initial answers) or last, tie first / last / random.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import pathlib

import numpy as np
from scipy.stats import beta

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
BOOT, SEED = 10000, 20261106
INTER = ("du_mad", "madm2_objective", "bccs")
EXTS, IDS, RDS, ARMS = ("strict", "lenient"), ("string", "exec"), ("r0", "last"), ("NOPEER", "FULL")
FACT = ("E", "I", "R", "A")
REF = (0, 0, 1, 1)                      # strict extractor, exact-match vote, last round, with peers


# ------------------------------------------------------------------ question-level arrays
def load_runs():
    runs = [json.loads(l) for l in (INP / "per_run_scores.jsonl").read_text().splitlines()]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        by[(r["protocol"], r["backbone"])][r["unit"]].append(r)
    return runs, by


def build(by_unit, tie="first"):
    """Y[q, extractor, vote, round, peers] and F[q, peers] (final answer), repeats averaged."""
    units = sorted(u for u, rs in by_unit.items() if {x["arm"] for x in rs} >= set(ARMS))
    Y = np.zeros((len(units), 2, 2, 2, 2))
    F = np.zeros((len(units), 2))
    for ui, u in enumerate(units):
        for a, arm in enumerate(ARMS):
            rs = [x for x in by_unit[u] if x["arm"] == arm]
            for e, i, rd in itertools.product((0, 1), repeat=3):
                Y[ui, e, i, rd, a] = np.mean([x[f"{EXTS[e]}|{IDS[i]}|{RDS[rd]}|{tie}"] for x in rs])
            F[ui, a] = np.mean([x["final"] for x in rs])
    return units, Y, F


def pooled(by, cells, tie="first"):
    built = {c: build(by[c], tie) for c in cells}
    common = sorted(set.intersection(*[set(built[c][0]) for c in cells]))
    Ys, Fs = [], []
    for c in cells:
        pos = {u: k for k, u in enumerate(built[c][0])}
        sel = [pos[u] for u in common]
        Ys.append(built[c][1][sel])
        Fs.append(built[c][2][sel])
    return common, np.mean(Ys, axis=0), np.mean(Fs, axis=0)


def qidx(n):
    return np.random.default_rng(SEED).integers(0, n, size=(BOOT, n))


def est_ci(d, idx):
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return 100 * float(d.mean()), 100 * float(lo), 100 * float(hi)


def dpeer(Y, e, i):
    return (Y[:, e, i, 1, 1] - Y[:, e, i, 0, 1]) - (Y[:, e, i, 1, 0] - Y[:, e, i, 0, 0])


def ledger(Y):
    """Shared-reference ledger: joint change, sum of single changes, joint minus sum."""
    yref = Y[:, REF[0], REF[1], REF[2], REF[3]]
    single = {}
    for j in range(4):
        c = list(REF)
        c[j] = 1 - c[j]
        single[j] = Y[:, c[0], c[1], c[2], c[3]] - yref
    out = {}
    for k in range(1, 5):
        for S in itertools.combinations(range(4), k):
            c = list(REF)
            for j in S:
                c[j] = 1 - c[j]
            joint = Y[:, c[0], c[1], c[2], c[3]] - yref
            out["+".join(FACT[j] for j in S)] = (joint, sum(single[j] for j in S))
    return yref, out


# ------------------------------------------------------------------ replay bounds (BCCS)
RD = {"ref": ("strict|string|last|first", "strict|string|r0|first", "ref"),
      "I": ("strict|exec|last|first", "strict|exec|r0|first", "I"),
      "E": ("lenient|string|last|first", "lenient|string|r0|first", "E"),
      "EI": ("lenient|exec|last|first", "lenient|exec|r0|first", "EI"),
      "ref_tl": ("strict|string|last|last", "strict|string|r0|last", "ref_tie_last")}
CONTRASTS = {
    "E": [(1, "E", "last", "FULL"), (-1, "ref", "last", "FULL")],
    "I": [(1, "I", "last", "FULL"), (-1, "ref", "last", "FULL")],
    "EI": [(1, "EI", "last", "FULL"), (-1, "ref", "last", "FULL")],
    "nonadd_EI": [(1, "EI", "last", "FULL"), (-1, "E", "last", "FULL"), (-1, "I", "last", "FULL"),
                  (1, "ref", "last", "FULL")],
    "nonadd_EIR": [(1, "EI", "r0", "FULL"), (-1, "E", "last", "FULL"), (-1, "I", "last", "FULL"),
                   (-1, "ref", "r0", "FULL"), (1, "ref", "last", "FULL"), (1, "ref", "last", "FULL")],
    "ref_minus_final": [(1, "ref", "last", "FULL"), (-1, "ref", "final", "FULL")],
    "final_minus_EIr0": [(1, "ref", "final", "FULL"), (-1, "EI", "r0", "FULL")],
    "dpeer_EI": [(1, "EI", "last", "FULL"), (-1, "EI", "r0", "FULL"), (-1, "EI", "last", "NOPEER"), (1, "EI", "r0", "NOPEER")],
    "later_EI": [(1, "EI", "last", "FULL"), (-1, "EI", "r0", "FULL")],
    "dpeer_ref": [(1, "ref", "last", "FULL"), (-1, "ref", "r0", "FULL"), (-1, "ref", "last", "NOPEER"), (1, "ref", "r0", "NOPEER")],
    "dpeer_ref_tl": [(1, "ref_tl", "last", "FULL"), (-1, "ref_tl", "r0", "FULL"), (-1, "ref_tl", "last", "NOPEER"), (1, "ref_tl", "r0", "NOPEER")],
}


def load_div():
    div = {}
    for l in (INP / "bccs_divergence_runs.jsonl").read_text().splitlines():
        d = json.loads(l)
        div[(d["unit"], d["arm"], d["backbone"], int(d["repeat"]))] = d
    return div


def interval(run, div, name, rnd):
    """(lo, hi, replay) of the deployed outcome of a reading at round rnd ('last', 'r0', 'final')."""
    if rnd == "final":
        v = run["final"]
        return v, v, v
    R = run[RD[name][0 if rnd == "last" else 1]]
    if rnd == "r0" or run["protocol"] != "bccs":
        return R, R, R            # initial answers precede every read; Du-MAD and MAD-M2 never read them
    d = div[(run["unit"], run["arm"], run["backbone"], int(run["repeat"]))][RD[name][2]]
    assert d["R_last"] == R
    return d["O_lo"], d["O_hi"], R


def task_arrays(runs, div, terms):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        by[r["unit"]][r["arm"]].append(r)
    units = sorted(u for u, a in by.items() if {"FULL", "NOPEER"} <= set(a))
    est, lo, hi = np.zeros(len(units)), np.zeros(len(units)), np.zeros(len(units))
    for k, u in enumerate(units):
        for sgn, name, rnd, arm in terms:
            vals = [interval(r, div, name, rnd) for r in by[u][arm]]
            l_, h_, R_ = (np.mean([v[j] for v in vals]) for j in range(3))
            est[k] += sgn * R_
            if sgn > 0:
                lo[k] += l_
                hi[k] += h_
            else:
                lo[k] -= h_
                hi[k] -= l_
    return units, est, lo, hi


def db_ci(units, est, dbmap):
    dbs = sorted({dbmap[u] for u in units})
    members = {d: np.array([i for i, u in enumerate(units) if dbmap[u] == d]) for d in dbs}
    rng = np.random.default_rng(SEED)
    out = np.empty(BOOT)
    for b in range(BOOT):
        pick = rng.integers(0, len(dbs), size=len(dbs))
        out[b] = est[np.concatenate([members[dbs[j]] for j in pick])].mean()
    return 100 * float(np.percentile(out, 2.5)), 100 * float(np.percentile(out, 97.5))


def bounds(runs, div, dbmap):
    runs = [r for r in runs if r["protocol"] in INTER]
    cells = sorted({(r["protocol"], r["backbone"]) for r in runs})
    res = {}
    for cname, terms in CONTRASTS.items():
        arrays = {c: task_arrays([r for r in runs if (r["protocol"], r["backbone"]) == c], div, terms) for c in cells}
        u0 = arrays[cells[0]][0]
        assert all(arrays[c][0] == u0 for c in cells)
        idx = qidx(len(u0))
        groups = {f"{c[0]}|{c[1]}": [c] for c in cells}
        groups["mean6"] = cells
        groups["mean4"] = [c for c in cells if c[0] != "bccs"]
        for g, sub in groups.items():
            E = np.mean([arrays[c][1] for c in sub], axis=0)
            L = np.mean([arrays[c][2] for c in sub], axis=0)
            H = np.mean([arrays[c][3] for c in sub], axis=0)
            be, bl, bh = E[idx].mean(axis=1), L[idx].mean(axis=1), H[idx].mean(axis=1)
            rec = dict(est=100 * float(E.mean()),
                       lo=100 * float(np.percentile(be, 2.5)), hi=100 * float(np.percentile(be, 97.5)),
                       bound_lo=100 * float(L.mean()), bound_hi=100 * float(H.mean()),
                       bound_ci_lo=100 * float(np.percentile(bl, 2.5)), bound_ci_hi=100 * float(np.percentile(bh, 97.5)))
            if g in ("mean4", "mean6"):
                rec["db_lo"], rec["db_hi"] = db_ci(u0, E, dbmap)
            res[(cname, g)] = rec
    return res


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def divergence_shares(div):
    """Share of BCCS runs whose deployed path would diverge under a reading (and unrecoverably)."""
    out = {}
    rows = collections.defaultdict(list)
    for d in div.values():
        rows[(d["backbone"], d["arm"])].append(d)
    for (bb, arm), rs in rows.items():
        for mp in ("ref", "EI", "ref_tie_last"):
            kinds = collections.Counter(x[mp]["D"] for x in rs)
            byu = collections.defaultdict(list)
            for x in rs:
                byu[x["unit"]].append(x[mp]["D"])
            any_d = sum(1 for v in byu.values() if any(k != "none" for k in v))
            n = len(rs)
            lo, hi = cp(any_d, len(byu))
            out[(bb, arm, mp)] = dict(P_D=(n - kinds["none"]) / n, P_U=kinds["unrecoverable"] / n, cp_lo=lo, cp_hi=hi)
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    V = {}

    def put(key, x):
        V[key] = round(float(x), 6)

    def put_ci(key, t):
        put(key, t[0]); put(key + "_lo", t[1]); put(key + "_hi", t[2])

    runs, by = load_runs()
    cells6 = sorted(c for c in by if c[0] in INTER)
    dbmap = {r["unit"]: r["db_id"] for r in csv.DictReader(open(INP / "bird_units_db.csv"))}

    # ---- six interactive cells (Du-MAD, MAD-M2, BCCS x Llama-3-8B, Qwen2.5-7B), ties to the first agent
    units, Y, F = pooled(by, cells6)
    put("n_questions", len(units))
    idx = qidx(len(units))
    yref, led = ledger(Y)
    put("published_level_mean6", 100 * yref.mean())
    for name, key in (("E", "lenient"), ("I", "exec_vote"), ("R", "initial"), ("A", "nopeer")):
        put_ci(f"single_{key}", est_ci(led[name][0], idx))
    for name, key in (("E+I", "EI"), ("E+I+R", "EIR"), ("E+I+R+A", "EIRA")):
        joint, ssum = led[name]
        put_ci(f"joint_{key}", est_ci(joint, idx))
        put(f"sum_{key}", 100 * ssum.mean())
        put_ci(f"nonadd_{key}", est_ci(joint - ssum, idx))
    put_ci("dpeer6_strict_string", est_ci(dpeer(Y, 0, 0), idx))
    put_ci("dpeer6_lenient_exec", est_ci(dpeer(Y, 1, 1), idx))
    put_ci("later6_strict_string", est_ci(Y[:, 0, 0, 1, 1] - Y[:, 0, 0, 0, 1], idx))
    put_ci("later6_lenient_exec", est_ci(Y[:, 1, 1, 1, 1] - Y[:, 1, 1, 0, 1], idx))
    put_ci("final_minus_string_r0_6", est_ci(F[:, 1] - Y[:, 0, 0, 0, 1], idx))
    put_ci("final_minus_exec_r0_6", est_ci(F[:, 1] - Y[:, 1, 1, 0, 1], idx))

    # ---- per cell: interaction effect under each reading and tie rule (MAD-M2 / Qwen2.5-7B tie dependence)
    for tie in ("first", "last", "random"):
        u_, Yt, _ = build(by[("madm2_objective", "qwen2.5-7b")], tie)
        put_ci(f"dpeer_madm2_qwen_strict_string_{tie}", est_ci(dpeer(Yt, 0, 0), qidx(len(u_))))
    for c in cells6 + [("moa", "roster")]:
        u_, Yc, Fc = build(by[c])
        tag = f"{c[0]}_{c[1]}".replace(".", "").replace("-", "")
        put(f"level_{tag}_strict_string_last_full", 100 * Yc[:, 0, 0, 1, 1].mean())
        put(f"level_{tag}_lenient_string_last_full", 100 * Yc[:, 1, 0, 1, 1].mean())
        put(f"level_{tag}_lenient_exec_r0_full", 100 * Yc[:, 1, 1, 0, 1].mean())
        put(f"level_{tag}_lenient_exec_last_full", 100 * Yc[:, 1, 1, 1, 1].mean())
        put(f"level_{tag}_final_full", 100 * Fc[:, 1].mean())

    # ---- MoA: final answer minus the execution vote over its references, by tie rule (R3)
    for tie in ("first", "last", "random"):
        u_, Ym, Fm = build(by[("moa", "roster")], tie)
        put(f"moa_final_minus_exec_r0_{tie}", 100 * (Fm[:, 1] - Ym[:, 1, 1, 0, 1]).mean())
    # MoA: exact-match vote over the initial answers, ties to the first minus the last agent, with peers
    _, Yf, _ = build(by[("moa", "roster")], "first")
    _, Yl, _ = build(by[("moa", "roster")], "last")
    put("moa_string_r0_first_minus_last", 100 * (Yf[:, 0, 0, 0, 1] - Yl[:, 0, 0, 0, 1]).mean())
    ties = {(r["cell"], r["ext|ident|round|arm"]): float(r["tie_share"]) for r in csv.DictReader(open(INP / "vote_ties.csv"))}
    put("moa_string_r0_tie_share_pct", 100 * ties[("moa|roster", "strict|string|r0|FULL")])

    # ---- matching protocols: largest move of the sharing effect under the all-round majority (Du-MAD, MAD-M2, BCCS)
    moves, tied = [], []
    for r in csv.DictReader(open(INP / "matching_return_rule.csv")):
        if r["protocol"] == "ford" or r["benchmark"] not in ("wdc_products", "geographic_settlements"):
            continue
        tied += [float(r["share_majority_tied_arm"]), float(r["share_majority_tied_full"])]
        e = float(r["sharing_effect_released_pp"])
        for k in ("sharing_effect_all_round_majority_pp", "sharing_effect_all_round_majority_tie_last_pp",
                  "sharing_effect_all_round_majority_tie_keep_pp"):
            moves.append(abs(float(r[k]) - e))
    put("matching_return_rule_max_move", max(moves))
    put("matching_tied_share_min_pct", 100 * min(tied))
    put("matching_tied_share_max_pct", 100 * max(tied))

    # ---- replay bounds and four-cell (Du-MAD, MAD-M2) estimates
    div = load_div()
    B = bounds(runs, div, dbmap)
    for (cname, g), rec in B.items():
        for k, v in rec.items():
            put(f"rb_{cname}_{g.replace('|', '_').replace('.', '').replace('-', '')}_{k}", v)
    for (bb, arm, mp), s in divergence_shares(div).items():
        tag = f"{bb.replace('.', '').replace('-', '')}_{arm}_{mp}"
        put(f"diverge_{tag}", s["P_D"])
        put(f"diverge_unrec_{tag}", s["P_U"])
        put(f"diverge_tasks_cp_lo_{tag}", s["cp_lo"])
        put(f"diverge_tasks_cp_hi_{tag}", s["cp_hi"])
    sh = divergence_shares(div)
    ei = [sh[(bb, arm, "EI")]["P_D"] for bb in ("llama3-8b", "qwen2.5-7b") for arm in ARMS]
    put("diverge_EI_min_pct", 100 * min(ei))
    put("diverge_EI_max_pct", 100 * max(ei))

    # ---- runs that the lenient extractor with the execution vote corrects or regresses (Du-MAD, MAD-M2,
    #      runs with peers, last round, ties to the first agent)
    base, alt = "strict|string|last|first", "lenient|exec|last|first"
    four = [r for r in runs if r["protocol"] in ("du_mad", "madm2_objective") and r["arm"] == "FULL"
            and r["unit"] in set(units)]
    put("runs_EI_mean4", len(four))
    put("runs_EI_corrected_mean4", sum(r[base] == 0 and r[alt] == 1 for r in four))
    put("runs_EI_regressed_mean4", sum(r[base] == 1 and r[alt] == 0 for r in four))

    # ---- debate rounds removed by the initial-answer vote: calls and token shares (runs with peers)
    cost = collections.defaultdict(list)
    for r in csv.DictReader(open(INP / "debate_cost_runs.csv")):
        if r["outcome"] == "gold_failed":
            continue
        cost[(r["protocol"], r["backbone"])].append(r)
    later_calls, kept = [], []
    for c in cells6:
        rs = cost[c]
        g = lambda k: sum(float(x[k]) for x in rs) / len(rs)          # noqa: E731
        tok_r0 = round(round(g("prompt_r0"), 1) + round(g("completion_r0"), 1), 1)
        tot = round(tok_r0 + round(g("prompt_later"), 1) + round(g("completion_later"), 1), 1)
        later_calls.append(round(g("calls_later"), 2))
        kept.append(tok_r0 / tot)
    put("removed_calls_min", min(later_calls))
    put("removed_calls_max", max(later_calls))
    put("kept_token_share_min_pct", 100 * min(kept))
    put("kept_token_share_max_pct", 100 * max(kept))
    put("removed_token_share_min_pct", 100 * (1 - max(kept)))
    put("removed_token_share_max_pct", 100 * (1 - min(kept)))

    # ---- BCCS reads recomputed from the logged prefix: answer verdicts, stop flags, and peer choices
    reads = list(csv.DictReader(open(INP / "bccs_read_checks.csv")))
    put("bccs_read_runs", len(reads))
    for k in ("verdicts", "stop_flags", "peer_choices"):
        put(f"bccs_{k}_checked", sum(int(r[f"{k}_checked"]) for r in reads))
        put(f"bccs_{k}_reproduced", sum(int(r[f"{k}_reproduced"]) for r in reads))
    put("bccs_reads_not_reproduced", sum(int(r[f"{k}_checked"]) - int(r[f"{k}_reproduced"])
                                         for r in reads for k in ("verdicts", "stop_flags", "peer_choices")))

    # ---- classes of the 24 replacement contrasts
    cls = collections.Counter()
    for r in csv.DictReader(open(INP / "replacement_contrasts.csv")):
        c = r["cls"]
        if "bounded" in c and "identified" in c:
            cls["partly"] += 1
        elif c.startswith("bounded"):
            cls["bounded"] += 1
        elif c.startswith("identified in distribution"):
            cls["average"] += 1
        else:
            cls["exact"] += 1
    for k in ("exact", "average", "bounded", "partly"):
        put(f"contrasts_{k}", cls[k])
    put("contrasts_total", sum(cls.values()))

    (out / "values.json").write_text(json.dumps(dict(sorted(V.items())), indent=1) + "\n")
    print(len(V), "values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
