r"""Pre-specified estimands for the strong-backbone BIRD run, zero model calls.

    python3 -I sb_analyze.py

Inputs (from the unchanged analysis code pointed at a run directory; `sb_local_pipeline.sh`):
  output/interaction_matched_rows{,_tie_last,_tie_random}.jsonl     strong-backbone per-run rows
  output/ref_7_8b/interaction_matched_rows{...}.jsonl               7-8B rows (side by side only)
  output/dov_transfer.json                                          final - r0_vote (protocol identity)
  data/records/backbone_scale_cross_family/raw                       configs, stop counts, cost
Writes output/sb_estimands.json.

Rules: task-level gold exclusion (I1 audit; gold status from dov_transfer_audit gold_fresh.json),
tie-break first = primary, last/random = sensitivity; repeats averaged within a task before the
bootstrap (10,000 task resamples, seed 20261106); two-sided p = 2 min(P*(d<=0), P*(d>=0)) floored at
1e-4; Holm within each model over its primary cells; families summarised separately; nothing pooled
across models or with the 7-8B cells.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "output"
RAW = REPO / "data/records/backbone_scale_cross_family/raw"
GOLD = REPO / "data/derived/dov_transfer_audit/gold_fresh.json"
BOOT, SEED = 10000, 20261106
PROTOS = ("du_mad", "madm2_objective", "bccs")
TIES = {"first": "", "last": "_tie_last", "random": "_tie_random"}
E = "exec_filt|lenient"
PRIMARY = ("qwen25-72b-awq", "madm2_objective")
MARGIN = 2.0
FAMILIES = {"qwen2.5-awq": ["qwen25-14b-awq", "qwen25-32b-awq", "qwen25-72b-awq"],
            "cross-family": ["gpt-oss-120b", "qwen36-35b-a3b", "qwen3-32b"],
            "7-8B (reference)": ["qwen2.5-7b", "llama3-8b"]}
DESIGN_RUNS = 396          # 198 scorable tasks x 2 repeats per (protocol, arm, backbone)


def excluded_units():
    g = json.loads(GOLD.read_text())["gold"]
    return sorted(u for u, v in g.items() if v.get("status") != "ok")


def draws(d):
    rng = np.random.default_rng(SEED)
    return d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)


def summ(d, level=95):
    d = np.asarray(list(d), dtype=float)
    if len(d) == 0:
        return None
    b = draws(d)
    a = (100 - level) / 2
    lo, hi = np.percentile(b, [a, 100 - a])
    p = max(1.0 / BOOT, min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))
    return dict(n_units=int(len(d)), est=round(100 * d.mean(), 2), lo=round(100 * lo, 2),
                hi=round(100 * hi, 2), p=round(float(p), 5))


def per_unit(rs, a, b=None, cond=None):
    by = collections.defaultdict(list)
    for x in rs:
        if cond and not cond(x):
            continue
        by[x["unit"]].append(float(x[a]) - (float(x[b]) if b else 0.0))
    return {u: float(np.mean(v)) for u, v in by.items()}


def lvl(rs, a):
    v = per_unit(rs, a)
    return round(100 * float(np.mean(list(v.values()))), 2) if v else None


def paired(a, b):
    us = sorted(set(a) & set(b))
    return [a[u] - b[u] for u in us]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


#: backbones excluded at G1 or dropped: their records are never analysed
EXCLUDED_BACKBONES = {"mistral-small-31", "nemotron35-bolt"}   # GLM v1/v2 shards are moved aside, not in raw/


def load_rows(path, excl):
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    return [r for r in rows if r["unit"] not in excl and r["protocol"] in PROTOS
            and r["backbone"] not in EXCLUDED_BACKBONES]


def verdict(e, s):
    if e["p_holm"] < 0.05 and e["est"] > 0 and s and s["lo"] > 0:
        v = "interaction adds accuracy"
    elif e["p_holm"] < 0.05 and e["est"] > 0:
        v = "round effect, not interaction"
    elif e["hi"] < 3.0:
        v = "no interaction effect larger than 3 pp"
    else:
        v = "not determined"
    if e["p_holm"] < 0.05 and e["est"] < 0:
        v += " (E1 reliably negative)"
    return v


def cell_estimates(F, N):
    e1 = per_unit(F, f"last|{E}", f"r0|{E}")
    e1n = per_unit(N, f"last|{E}", f"r0|{E}")
    rec = dict(
        runs_full=len(F), runs_nopeer=len(N),
        units=len({x["unit"] for x in F}),
        E1=summ(e1.values()),
        dpeer=summ(paired(e1, e1n)) if N else None,
        E1_nopeer=summ(e1n.values()) if N else None,
        E1_strict=summ(per_unit(F, "last|exec_filt|strict", "r0|exec_filt|strict").values()),
        E1_exec_ident=summ(per_unit(F, "last|exec_ident|lenient", "r0|exec_ident|lenient").values()),
        dpeer_strict=summ(paired(per_unit(F, "last|exec_filt|strict", "r0|exec_filt|strict"),
                                 per_unit(N, "last|exec_filt|strict", "r0|exec_filt|strict"))) if N else None,
        last_mile_final_minus_last=summ(per_unit(F, "final", f"last|{E}").values()),
        last_mile_strict=summ(per_unit(F, "final", "last|exec_filt|strict").values()),
        pooled_minus_r0=summ(per_unit(F, f"pooled|{E}", f"r0|{E}").values()),
        share_continued_full=round(float(np.mean([x["last_round"] > 0 for x in F])), 4),
        share_continued_nopeer=round(float(np.mean([x["last_round"] > 0 for x in N])), 4) if N else None,
        early_stopped_runs_full=int(sum(x["last_round"] == 0 for x in F)),
        early_stopped_runs_nopeer=int(sum(x["last_round"] == 0 for x in N)),
        E1_given_continued=summ(per_unit(F, f"last|{E}", f"r0|{E}", cond=lambda x: x["last_round"] > 0).values()),
        final_failed_or_undelivered=round(float(np.mean([x["final_failed"] for x in F])), 4),
        level_final=lvl(F, "final"), level_r0=lvl(F, f"r0|{E}"), level_last=lvl(F, f"last|{E}"),
        level_pooled=lvl(F, f"pooled|{E}"),
        level_final_nopeer=lvl(N, "final") if N else None,
        level_r0_nopeer=lvl(N, f"r0|{E}") if N else None,
        level_last_nopeer=lvl(N, f"last|{E}") if N else None)
    per_rep = {}
    for rep in sorted({x["repeat"] for x in F}):
        Fr = [x for x in F if x["repeat"] == rep]
        Nr = [x for x in N if x["repeat"] == rep]
        a = per_unit(Fr, f"last|{E}", f"r0|{E}")
        b = per_unit(Nr, f"last|{E}", f"r0|{E}")
        per_rep[str(rep)] = dict(E1=summ(a.values()), E1_nopeer=summ(b.values()) if Nr else None,
                                 dpeer=summ(paired(a, b)) if Nr else None,
                                 dpeer_90=summ(paired(a, b), 90) if Nr else None)
    rec["per_repeat"] = per_rep
    rec["dpeer_90"] = summ(paired(e1, e1n), 90) if N else None
    rec["_e1"] = e1
    rec["_dp"] = {u: e1[u] - e1n[u] for u in set(e1) & set(e1n)}
    return rec


def analyse(rows_by_tie, design_runs=DESIGN_RUNS):
    rows = rows_by_tie["first"]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["protocol"], r["arm"], r["backbone"])].append(r)
    models = {}
    for bb in sorted({k[2] for k in cells}):
        recs = []
        for proto in PROTOS:
            F, N = cells.get((proto, "FULL", bb), []), cells.get((proto, "NOPEER", bb), [])
            if not F:
                continue
            rec = dict(protocol=proto, backbone=bb, **cell_estimates(F, N))
            rec["complete"] = (len({(x["unit"], x["repeat"]) for x in F}) >= design_runs and
                               len({(x["unit"], x["repeat"]) for x in N}) >= design_runs)
            # tie-break sensitivity on E1 and dpeer
            sens = {}
            for t in ("last", "random"):
                rt = [x for x in rows_by_tie.get(t, []) if x["backbone"] == bb and x["protocol"] == proto]
                Ft = [x for x in rt if x["arm"] == "FULL"]
                Nt = [x for x in rt if x["arm"] == "NOPEER"]
                if Ft:
                    a = per_unit(Ft, f"last|{E}", f"r0|{E}")
                    b = per_unit(Nt, f"last|{E}", f"r0|{E}")
                    sens[t] = dict(E1=summ(a.values()), dpeer=summ(paired(a, b)) if Nt else None,
                                   level_r0=lvl(Ft, f"r0|{E}"), level_last=lvl(Ft, f"last|{E}"),
                                   last_mile=summ(per_unit(Ft, "final", f"last|{E}").values()))
            rec["tie_sensitivity"] = sens
            recs.append(rec)
        adj = holm([r["E1"]["p"] for r in recs])
        for r, a in zip(recs, adj):
            r["E1"]["p_holm"] = round(a, 5)
            r["holm_family_size"] = len(recs)
            r["verdict"] = verdict(r["E1"], r["dpeer"])
        units = set.intersection(*[set(r["_e1"]) for r in recs]) if recs else set()
        mean_e1 = summ([np.mean([r["_e1"][u] for r in recs]) for u in sorted(units)]) if recs else None
        for r in recs:
            UNITMAPS[(bb, r["protocol"])] = dict(e1=r.pop("_e1"), dp=r.pop("_dp"))
        models[bb] = dict(cells=recs, E1_mean_over_protocols=mean_e1, n_protocols=len(recs),
                          complete=all(r["complete"] for r in recs))
    fam = {}
    for f, bbs in FAMILIES.items():
        have = [b for b in bbs if b in models and models[b]["E1_mean_over_protocols"]]
        if not have:
            continue
        adj = holm([models[b]["E1_mean_over_protocols"]["p"] for b in have])
        fam[f] = {b: dict(models[b]["E1_mean_over_protocols"], p_holm=round(a, 5)) for b, a in zip(have, adj)}
    return dict(models=models, families=fam, n_rows=len(rows))


UNITMAPS = {}
PAIRS = [("gpt-oss-120b", "gpt-oss-20b"), ("qwen25-72b-awq", "qwen25-14b-awq"),
         ("qwen25-72b-awq", "qwen25-32b-awq"), ("qwen25-32b-awq", "qwen25-14b-awq"),
         ("qwen25-14b-awq", "qwen2.5-7b")]


def pair_contrasts():
    """Same-family scale contrasts, paired per task: (E1_a - E1_b) and (dpeer_a - dpeer_b)."""
    out = []
    for a, b in PAIRS:
        for proto in PROTOS:
            A, B = UNITMAPS.get((a, proto)), UNITMAPS.get((b, proto))
            if not A or not B:
                continue
            out.append(dict(larger=a, smaller=b, protocol=proto,
                            note=("secondary: the 7B point is fp16, not AWQ" if "7b" in b else None),
                            dE1=summ(paired(A["e1"], B["e1"])), ddpeer=summ(paired(A["dp"], B["dp"]))))
    return out


def raw_facts():
    """Configs, early stops and cost per (protocol, arm, backbone), straight from the records."""
    if not RAW.exists():
        return {}
    last = {}
    for p in sorted(RAW.glob("sql_bird_v2__*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("backbone") not in EXCLUDED_BACKBONES:
                last[(r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))] = r
    by = collections.defaultdict(list)
    for k, r in last.items():
        by[(k[1], k[2], k[3])].append(r)
    out = {}
    for (proto, arm, bb), rs in sorted(by.items()):
        cfgs = collections.Counter(json.dumps({k: (r.get("config") or {}).get(k) for k in
                                               ("agents", "rounds", "max_tokens", "temperature", "top_p",
                                                "prune", "n_clusters", "top_k", "belief")}, sort_keys=True)
                                   for r in rs)
        calls = [r.get("n_calls") or 0 for r in rs]
        rounds_run = collections.Counter(len({c.get("round") for c in r["record"]["calls"]}) for r in rs)
        out[f"{proto}|{arm}|{bb}"] = dict(
            runs=len(rs), configs=dict(cfgs), rounds_run=dict(rounds_run),
            stop_reasons=dict(collections.Counter(str(r.get("stop_reason")) for r in rs)),
            mean_calls=round(float(np.mean(calls)), 3),
            mean_prompt_tokens=round(float(np.mean([r.get("prompt_tokens") or 0 for r in rs])), 1),
            mean_completion_tokens=round(float(np.mean([r.get("completion_tokens") or 0 for r in rs])), 1),
            total_calls=int(sum(calls)),
            total_prompt_tokens=int(sum(r.get("prompt_tokens") or 0 for r in rs)),
            total_completion_tokens=int(sum(r.get("completion_tokens") or 0 for r in rs)),
            truncated_calls=int(sum(r.get("truncated_calls") or 0 for r in rs)),
            mean_unit_run_seconds=round(float(np.mean([r.get("seconds") or 0 for r in rs])), 2),
            max_tokens=sorted({r.get("max_tokens") for r in rs}),
            code_versions=sorted({r.get("code_version") for r in rs}))
    return out


def exec_cost():
    p = OUT / "r0_exec_signatures.jsonl"
    if not p.exists():
        return None
    st = collections.Counter()
    s = 0.0
    n = 0
    for l in p.read_text().splitlines():
        d = json.loads(l)
        st[d.get("status")] += 1
        s += float(d.get("s") or 0)
        n += 1
    return dict(distinct_queries_signed=n, status=dict(st), summed_exec_seconds=round(s, 1),
                note="read-only SQLite, local; never converted into tokens")


def main():
    excl = excluded_units()
    strong = {t: load_rows(OUT / f"interaction_matched_rows{s}.jsonl", excl) for t, s in TIES.items()}
    ref = {t: load_rows(OUT / f"ref_7_8b/interaction_matched_rows{s}.jsonl", excl) for t, s in TIES.items()}
    out = dict(note=("E1 = last - r0, exec_filt, lenient symmetric, FULL; dpeer = E1_FULL - E1_NOPEER "
                     "paired per task; Holm within each model; never pooled across models."),
               excluded_tasks=excl, boot=BOOT, seed=SEED,
               primary_cell=dict(backbone=PRIMARY[0], protocol=PRIMARY[1], margin_pp=MARGIN),
               strong=analyse(strong), raw=raw_facts(), exec_cost=exec_cost())
    if ref["first"]:
        out["ref_7_8b"] = analyse(ref)
    out["scale_pairs"] = pair_contrasts()
    pc = next((c for c in out["strong"]["models"].get(PRIMARY[0], {}).get("cells", [])
               if c["protocol"] == PRIMARY[1]), None)
    if pc and pc["dpeer_90"]:
        d = pc["dpeer_90"]
        out["primary_cell"].update(
            dpeer=pc["dpeer"], dpeer_90=d, per_repeat={k: v["dpeer"] for k, v in pc["per_repeat"].items()},
            practically_equivalent=bool(d["lo"] > -MARGIN and d["hi"] < MARGIN), complete=pc["complete"])
    dvp = OUT / "dov_transfer.json"
    if dvp.exists():
        dv = json.loads(dvp.read_text())
        out["final_minus_r0_vote_protocol_identity"] = [
            {k: c.get(k) for k in ("protocol", "arm", "backbone", "runs", "r0_extracted", "r0_vote_tie_share",
                                   "level_final", "level_r0_vote", "level_r0_exec", "final_minus_r0_vote_effect_pp",
                                   "final_minus_r0_vote_lo_pp", "final_minus_r0_vote_hi_pp",
                                   "final_minus_r0_vote_tie_last_pp", "final_minus_r0_vote_tie_random_pp",
                                   "final_minus_r0_exec_effect_pp", "final_minus_r0_exec_lo_pp",
                                   "final_minus_r0_exec_hi_pp", "r0_call_share", "r0_token_share")}
            for c in dv["cells"] if c["task"] == "sql_bird" and c["backbone"] not in EXCLUDED_BACKBONES]
        out["dov_exclusions"] = dv.get("exclusions")
    (OUT / "sb_estimands.json").write_text(json.dumps(out, indent=1))
    for pop in ("strong", "ref_7_8b"):
        if pop not in out:
            continue
        print(f"== {pop}  (excluded tasks {excl})")
        for bb, m in out[pop]["models"].items():
            print(f"  -- {bb}  complete={m['complete']}  mean E1 over protocols {m['E1_mean_over_protocols']}")
            for r in m["cells"]:
                e, s = r["E1"], r["dpeer"]
                reps = {k: (v["dpeer"] or {}).get("est") for k, v in r["per_repeat"].items()}
                print(f"    {r['protocol']:16s} runs {r['runs_full']}/{r['runs_nopeer']} final={r['level_final']} "
                      f"r0={r['level_r0']} last={r['level_last']} E1={e['est']:+.2f}[{e['lo']:+.2f},{e['hi']:+.2f}] "
                      f"pH={e['p_holm']:.3f} dpeer={s and s['est']}[{s and s['lo']},{s and s['hi']}] per-rep {reps} "
                      f"cont={r['share_continued_full']}/{r['share_continued_nopeer']} -> {r['verdict']}")
    print("primary", out["primary_cell"])
    for c in out["scale_pairs"]:
        print("pair", c["larger"], "vs", c["smaller"], c["protocol"], "dE1", c["dE1"], "ddpeer", c["ddpeer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
