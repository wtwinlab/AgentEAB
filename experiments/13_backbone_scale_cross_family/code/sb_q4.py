r"""Q4 numbers (scale and cross-family interaction), zero model calls.

    python3 -I sb_q4.py        -> output/q4_numbers.json, output/q4_tables.md

Readout: lenient extraction, exec_filt aggregator, first tie (interaction_matched_rows.jsonl).
Per (model, protocol): E1 = last - r0 (FULL), dpeer = E1_FULL - E1_NOPEER, paired per task, repeats
averaged within task. "Identified mean" = per-task mean of Du-MAD and MAD-M² (both have fixed rounds
and no stop rule); BCCS is reported separately as a frozen readout (its stop rule reads its own
string identity, so its last round depends on the identity notion). CIs: task bootstrap and
database-cluster bootstrap (11 BIRD DBs), 10,000 draws, seed 20261106. Tasks bird|518, bird|701
excluded (gold fails/times out). Models never pooled.
"""
import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "output"
EXCL = {"bird|518", "bird|701"}
E = "exec_filt|lenient"
BOOT, SEED = 10000, 20261106
QWEN = ["qwen2.5-7b", "qwen25-14b-awq", "qwen25-32b-awq", "qwen25-72b-awq"]
CROSS = ["gpt-oss-20b", "gpt-oss-120b", "glm47-flash", "mistral-small-32", "qwen36-35b-a3b", "qwen38-27b",
         "qwen3-32b", "gemma4-26b-a4b", "llama33-70b", "minimax-m25-awq", "agentar-sql-32b", "llama3-8b"]
EXCLUDED = {"mistral-small-31", "glm47-flash-think", "nemotron35-bolt"}


def stats(d, db):
    us = sorted(d)
    if not us:
        return None
    x = np.array([d[u] for u in us])
    rng = np.random.default_rng(SEED)
    t = x[rng.integers(0, len(x), size=(BOOT, len(x)))].mean(axis=1)
    g = collections.defaultdict(list)
    for i, u in enumerate(us):
        g[db[u]].append(i)
    keys = sorted(g)
    rng = np.random.default_rng(SEED)
    c = np.array([x[[i for k in rng.integers(0, len(keys), size=len(keys)) for i in g[keys[k]]]].mean()
                  for _ in range(BOOT)])
    p = max(1 / BOOT, min(1.0, 2 * min((t <= 0).mean(), (t >= 0).mean())))
    r = lambda v: round(100 * float(v), 2)
    return dict(n=len(us), est=r(x.mean()), task_ci=[r(v) for v in np.percentile(t, [2.5, 97.5])],
                db_ci=[r(v) for v in np.percentile(c, [2.5, 97.5])], p_task=round(float(p), 5))


def per_unit(rows, f):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["unit"]].append(f(r))
    return {u: float(np.mean(v)) for u, v in by.items()}


def e1(r):
    return float(r[f"last|{E}"]) - float(r[f"r0|{E}"])


def main():
    sys.path.insert(0, str(REPO / "experiments/02_controlled_sharing/code"))
    import sql_common as SQ
    db = {f"bird|{q['question_id']}": q["db_id"] for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}
    rows = []
    for f in (OUT / "interaction_matched_rows.jsonl", OUT / "ref_7_8b/interaction_matched_rows.jsonl"):
        rows += [json.loads(l) for l in f.read_text().splitlines()]
    rows = [r for r in rows if r["unit"] not in EXCL and r["backbone"] not in EXCLUDED
            and r["protocol"] in ("du_mad", "madm2_objective", "bccs")]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["backbone"], r["protocol"], r["arm"])].append(r)
    maps = {}
    out = dict(readout="lenient extraction, exec_filt, first tie; tasks bird|518, bird|701 excluded",
               models={})
    for bb in sorted({k[0] for k in cells}):
        m = {}
        for proto in ("du_mad", "madm2_objective", "bccs"):
            F, N = cells.get((bb, proto, "FULL"), []), cells.get((bb, proto, "NOPEER"), [])
            if not F:
                continue
            ef, en = per_unit(F, e1), per_unit(N, e1)
            dp = {u: ef[u] - en[u] for u in set(ef) & set(en)}
            ff, fn = per_unit(F, lambda r: float(r["final"])), per_unit(N, lambda r: float(r["final"]))
            reps = {}
            for rep in sorted({r["repeat"] for r in F}):
                a = per_unit([r for r in F if r["repeat"] == rep], e1)
                b = per_unit([r for r in N if r["repeat"] == rep], e1)
                reps[str(rep)] = stats({u: a[u] - b[u] for u in set(a) & set(b)}, db)
            m[proto] = dict(runs_full=len(F), runs_nopeer=len(N),
                            complete=len({(r["unit"], r["repeat"]) for r in F}) >= 396 and
                            len({(r["unit"], r["repeat"]) for r in N}) >= 396,
                            E1=stats(ef, db), E1_nopeer=stats(en, db), dpeer=stats(dp, db), dpeer_per_repeat=reps,
                            final_full_minus_nopeer=stats({u: ff[u] - fn[u] for u in set(ff) & set(fn)}, db),
                            level_final_full=round(100 * np.mean(list(ff.values())), 2),
                            level_r0_full=round(100 * np.mean(list(per_unit(F, lambda r: float(r[f"r0|{E}"])).values())), 2),
                            level_last_full=round(100 * np.mean(list(per_unit(F, lambda r: float(r[f"last|{E}"])).values())), 2),
                            frozen_readout=(proto == "bccs"))
            maps[(bb, proto)] = (ef, dp)
        ident = [p for p in ("du_mad", "madm2_objective") if (bb, p) in maps]
        if len(ident) == 2:
            us = set(maps[(bb, "du_mad")][0]) & set(maps[(bb, "madm2_objective")][0])
            ie = {u: np.mean([maps[(bb, p)][0][u] for p in ident]) for u in us}
            us2 = set(maps[(bb, "du_mad")][1]) & set(maps[(bb, "madm2_objective")][1])
            idp = {u: np.mean([maps[(bb, p)][1][u] for p in ident]) for u in us2}
            maps[(bb, "ident")] = (ie, idp)
            m["identified_mean"] = dict(E1=stats(ie, db), dpeer=stats(idp, db),
                                        complete=all(m[p]["complete"] for p in ident))
        elif ident:
            m["identified_mean"] = dict(note=f"only {ident} available (e.g. Du-MAD-only model)",
                                        E1=m[ident[0]]["E1"], dpeer=m[ident[0]]["dpeer"],
                                        complete=m[ident[0]]["complete"])
            maps[(bb, "ident")] = maps[(bb, ident[0])]
        out["models"][bb] = m
    # scale contrasts on the identified mean (Qwen2.5 family; 7B is fp16, others AWQ)
    sc = []
    pairs = [("qwen25-72b-awq", "qwen2.5-7b"), ("qwen25-14b-awq", "qwen2.5-7b"), ("qwen25-32b-awq", "qwen25-14b-awq"),
             ("qwen25-72b-awq", "qwen25-32b-awq"), ("qwen25-72b-awq", "qwen25-14b-awq"),
             ("gpt-oss-120b", "gpt-oss-20b"), ("llama33-70b", "llama3-8b")]
    for a, b in pairs:
        if (a, "ident") in maps and (b, "ident") in maps:
            (ea, da), (eb, dbb) = maps[(a, "ident")], maps[(b, "ident")]
            sc.append(dict(larger=a, smaller=b,
                           dE1=stats({u: ea[u] - eb[u] for u in set(ea) & set(eb)}, db),
                           ddpeer=stats({u: da[u] - dbb[u] for u in set(da) & set(dbb)}, db),
                           note=("7B point is fp16 (other Qwen2.5 points AWQ)" if "7b" in b and "qwen" in b else
                                 "Llama 3 -> 3.3 is also a version change and AWQ" if "llama" in a else
                                 "gpt-oss: Du-MAD + MAD-M² (A11 final-channel logprobs)" if "gpt" in a else None)))
    # Holm over the 5 Qwen2.5 scale contrasts; the declared primary scale test is 72B - 7B
    qsc = [c for c in sc if c["larger"].startswith("qwen") and c["ddpeer"]]
    order = sorted(range(len(qsc)), key=lambda i: qsc[i]["ddpeer"]["p_task"])
    run = 0.0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (len(qsc) - r_) * qsc[i]["ddpeer"]["p_task"]))
        qsc[i]["ddpeer"]["p_holm_5"] = round(run, 5)
    for c in qsc:
        c["role"] = ("PRIMARY declared scale test" if (c["larger"], c["smaller"]) == ("qwen25-72b-awq", "qwen2.5-7b")
                     else "secondary (exploratory unless it survives Holm over the 5)")
    out["scale_contrasts_identified_mean"] = sc
    # A3 primary cell: Qwen2.5-72B-AWQ x MAD-M2, dpeer TOST at +-2 pp (90% CI)
    F = cells.get(("qwen25-72b-awq", "madm2_objective", "FULL"), [])
    N = cells.get(("qwen25-72b-awq", "madm2_objective", "NOPEER"), [])
    if F and N:
        ef, en = per_unit(F, e1), per_unit(N, e1)
        d = np.array([ef[u] - en[u] for u in sorted(set(ef) & set(en))])
        rng = np.random.default_rng(SEED)
        b = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)
        lo, hi = np.percentile(b, [5, 95])
        out["primary_cell_A3"] = dict(cell="qwen25-72b-awq x madm2_objective", margin_pp=2.0,
                                      dpeer=round(100 * d.mean(), 2), ci90=[round(100 * lo, 2), round(100 * hi, 2)],
                                      practically_equivalent=bool(100 * lo > -2 and 100 * hi < 2),
                                      per_repeat=out["models"]["qwen25-72b-awq"]["madm2_objective"]["dpeer_per_repeat"])
    # cross-family Holm on the identified-mean dpeer (final set, complete models only)
    fam = [b for b in ("qwen3-32b", "qwen36-35b-a3b", "qwen38-27b", "glm47-flash", "mistral-small-32",
                       "gemma4-26b-a4b", "llama33-70b", "minimax-m25-awq", "agentar-sql-32b")
           if b in out["models"] and out["models"][b].get("identified_mean", {}).get("complete")]
    ps = [out["models"][b]["identified_mean"]["dpeer"]["p_task"] for b in fam]
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    run, adj = 0.0, [None] * len(ps)
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r_) * ps[i]))
        adj[i] = round(run, 5)
    out["cross_family_holm_identified_dpeer"] = {b: dict(p=ps[i], p_holm=adj[i]) for i, b in enumerate(fam)}
    (OUT / "q4_numbers.json").write_text(json.dumps(out, indent=1))
    # markdown
    L = []
    f = lambda s: "—" if not s else f"{s['est']:+.2f} [{s['task_ci'][0]:+.2f}, {s['task_ci'][1]:+.2f}] (DB {s['db_ci'][0]:+.2f}, {s['db_ci'][1]:+.2f})"
    L.append("| model | protocol | complete | final | r0 vote | last vote | E1 | Δpeer | Δpeer r1 / r2 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    order = [b for b in QWEN + CROSS + ["qwen2.5-7b-execid"] if b in out["models"]]
    for bb in order:
        m = out["models"][bb]
        for p in ("du_mad", "madm2_objective", "bccs"):
            if p not in m:
                continue
            c = m[p]
            reps = " / ".join("—" if not v else f"{v['est']:+.2f}" for v in c["dpeer_per_repeat"].values())
            L.append(f"| {bb} | {p}{' (frozen readout)' if p == 'bccs' else ''} | {c['complete']} | {c['level_final_full']} | "
                     f"{c['level_r0_full']} | {c['level_last_full']} | {f(c['E1'])} | {f(c['dpeer'])} | {reps} |")
        if "identified_mean" in m:
            im = m["identified_mean"]
            L.append(f"| **{bb}** | **identified mean (Du-MAD+MAD-M²)** | {im['complete']} | | | | {f(im['E1'])} | {f(im['dpeer'])} | |")
    L.append("")
    L.append("| larger | smaller | ΔE1 (identified mean) | ΔΔpeer (identified mean) | note |")
    L.append("|---|---|---|---|---|")
    for s in sc:
        L.append(f"| {s['larger']} | {s['smaller']} | {f(s['dE1'])} | {f(s['ddpeer'])} | {s['note'] or ''} |")
    (OUT / "q4_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
