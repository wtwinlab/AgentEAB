r"""Score W6 arms A (fixed order, thinking off), B (permuted, off), C (same permutation, on).

    python3 code/w6_arms_score.py   # -> output/w6_arms_levels.csv, w6_arms_contrasts.csv,
                                    #    w6_arms_decomposition.csv, w6_arms_cost.json,
                                    #    w6_arms_position.csv, w6_arms_per_question.csv

Family fixed before any B/C call: per benchmark B − A, C − B,
C − Nonempty, Generation − C; Holm 0.05; database-equal primary. Rules, generation and labels are
read exactly as in `w6_score.py`, which is imported.
"""
import collections, csv, json, pathlib, sys
import numpy as np
sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w6_score as S
W, OUT = S.WORK, S.OUT
CON = (("B", "A"), ("C", "B"), ("C", "Nonempty"), ("Generation", "C"))
FILES = {"A": "w6_selector_main.jsonl", "B": "w6_armB_main.jsonl", "C": "w6_armC_main.jsonl"}


def main():
    labels, cf, _ = S.load()
    arms = {}
    for k, f in FILES.items():
        arms[k] = {(r["benchmark"], int(r["question_id"])): r
                   for r in map(json.loads, open(W / "raw" / f))}
    rows = []
    for (b, q), c in sorted(cf.items()):
        row = dict(benchmark=b, question_id=q, db_id=c["db_id"], Original=c["released"],
                   Valid=c["valid_majority"], Nonempty=c["nonempty_majority"],
                   Generation=c["generation"],
                   Oracle=int(any(bool((labels.get((b, q, m)) or {}).get("match")) for m in S.POOL)))
        for k in arms:
            r = arms[k].get((b, q)) or {}
            pm = r.get("picked_model")
            row[k] = int(bool(pm and (labels.get((b, q, pm)) or {}).get("match")))
            row[f"{k}_pick"] = pm
            row[f"{k}_finish"] = r.get("finish_reason")
        rows.append(row)
    missing = {k: sum(1 for (b, q) in cf if (b, q) not in arms[k]) for k in arms}
    with open(OUT / "w6_arms_per_question.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    viol = {k: sum(1 for r in rows if r[k] and not r["Oracle"]) for k in ("A", "B", "C")}
    rng = np.random.default_rng(S.SEED)
    M = ("Oracle", "Original", "Valid", "Nonempty", "A", "B", "C", "Generation")
    levels, cons = [], []
    for b in ("bird", "spider"):
        sub = [r for r in rows if r["benchmark"] == b]
        dbs = sorted({r["db_id"] for r in sub}); by = {d: [r for r in sub if r["db_id"] == d] for d in dbs}
        dr = rng.integers(0, len(dbs), size=(S.DRAWS, len(dbs)))

        def sc(fn, wt):
            per = np.array([np.mean([fn(r) for r in by[d]]) for d in dbs])
            n = np.array([len(by[d]) for d in dbs], float); ww = np.ones(len(dbs)) if wt == "database_equal" else n
            pt = float((ww * per).sum() / ww.sum()); bs = (per[dr] * ww[dr]).sum(1) / ww[dr].sum(1)
            lo, hi = np.percentile(bs, [2.5, 97.5]); return pt, float(lo), float(hi), bs
        for wt in ("database_equal", "question"):
            for m in M:
                pt, lo, hi, _ = sc(lambda r, m=m: r[m], wt)
                levels.append(dict(benchmark=b, weighting=wt, method=m, questions=len(sub), databases=len(dbs),
                                   accuracy_pct=round(100 * pt, 2), lo_pct=round(100 * lo, 2), hi_pct=round(100 * hi, 2)))
            for x, y in CON:
                pt, lo, hi, bs = sc(lambda r, x=x, y=y: r[x] - r[y], wt)
                pv = float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
                cons.append(dict(benchmark=b, weighting=wt, contrast=f"{x} - {y}", effect_pp=round(100 * pt, 2),
                                 lo_pp=round(100 * lo, 2), hi_pp=round(100 * hi, 2), p_boot=round(pv, 4),
                                 ci_excludes_zero=bool(lo > 0 or hi < 0)))
    fam = sorted([c for c in cons if c["weighting"] == "database_equal"], key=lambda c: c["p_boot"])
    stop = False
    for i, c in enumerate(fam):
        rej = (not stop) and c["p_boot"] <= 0.05 / (len(fam) - i); stop = stop or not rej
        c["holm_reject_in_arms_family"] = rej; c["holm_family_size"] = len(fam)
    for c in cons:
        c.setdefault("holm_reject_in_arms_family", None); c.setdefault("holm_family_size", None)
    dec = []
    for b in ("bird", "spider"):
        sub = [r for r in rows if r["benchmark"] == b]; n = len(sub)
        base = sum(1 for r in sub if r["Oracle"] and not r["Original"])
        for m in ("Original", "Valid", "Nonempty", "A", "B", "C"):
            miss = sum(1 for r in sub if r["Oracle"] and not r[m])
            dec.append(dict(benchmark=b, method=m, weighting="question", questions=n,
                            oracle_pct=round(100 * sum(r["Oracle"] for r in sub) / n, 2),
                            returned_correct_pct=round(100 * sum(r[m] for r in sub) / n, 2),
                            available_not_returned_pct=round(100 * miss / n, 2),
                            share_of_gap_recovered_vs_original=round(1 - miss / base, 4)))
    pos = []
    for k in ("A", "B", "C"):
        for b in ("bird", "spider"):
            rs = [r for r in arms[k].values() if r["benchmark"] == b]
            for cid in ("C1", "C2", "C3", "C4"):
                off = [r for r in rs if cid in r["id_to_model"]]
                pk = [r for r in off if r.get("picked_id") == cid]
                pos.append(dict(arm=k, benchmark=b, position=cid, offered=len(off),
                                picked_pct=round(100 * len(pk) / max(1, len(off)), 2)))
    cost = {}
    for k in arms:
        for b in ("bird", "spider"):
            rs = [r for r in arms[k].values() if r["benchmark"] == b]; u = [r.get("usage") or {} for r in rs]
            rt = [((x.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0) for x in u]
            cost[f"{k}|{b}"] = dict(calls=len(rs), prompt_tokens=sum(x.get("prompt_tokens", 0) for x in u),
                                    completion_tokens=sum(x.get("completion_tokens", 0) for x in u),
                                    reasoning_tokens_reported=sum(rt),
                                    mean_completion_tokens=round(float(np.mean([x.get("completion_tokens", 0) for x in u])), 1),
                                    mean_latency_s=round(float(np.mean([r["latency_s"] for r in rs])), 2),
                                    nonselections=sum(1 for r in rs if not r.get("picked_id")),
                                    truncated=sum(1 for r in rs if r.get("finish_reason") == "length"),
                                    errors=sum(1 for r in rs if r.get("status") != "ok"))
    pil = {}
    for k in ("B", "C"):
        rs = [json.loads(l) for l in open(W / "raw" / f"w6_arm{k}_pilot.jsonl")]
        pil[k] = dict(calls=len(rs), stop=sum(r["finish_reason"] == "stop" for r in rs),
                      valid_id=sum(bool(r.get("picked_id")) for r in rs),
                      completion_tokens=sum(r["usage"]["completion_tokens"] for r in rs))
    for name, data in (("w6_arms_levels.csv", levels), ("w6_arms_contrasts.csv", cons),
                       ("w6_arms_decomposition.csv", dec), ("w6_arms_position.csv", pos)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (OUT / "w6_arms_cost.json").write_text(json.dumps(dict(cost=cost, pilot=pil, missing=missing,
                                                            oracle_violations=viol), indent=1) + "\n")
    print("missing", missing, "violations", viol)
    for L in levels:
        if L["weighting"] == "database_equal": print(f"  {L['benchmark']:6} {L['method']:10} {L['accuracy_pct']:6.2f} [{L['lo_pct']}, {L['hi_pct']}]")
    for c in cons:
        print(f"  {c['benchmark']:6} {c['weighting']:14} {c['contrast']:16} {c['effect_pp']:+6.2f} [{c['lo_pp']:+.2f},{c['hi_pp']:+.2f}] p={c['p_boot']} holm={c['holm_reject_in_arms_family']}")
    for d in dec: print("  ", d)
    for p_ in pos: print("  ", p_)
    print(json.dumps(dict(cost=cost, pilot=pil), indent=1))


if __name__ == "__main__":
    main()
