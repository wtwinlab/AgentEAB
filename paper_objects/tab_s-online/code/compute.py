r"""Online BCCS with execution voting inside the protocol, Qwen2.5-7B (Table tab:s-online, sec:s-online-bound).

    python3 code/compute.py --out DIR      -> DIR/values.json, DIR/online_vs_replay.csv

Inputs
  inputs/online_bccs_runs.csv   one row per online BCCS run (condition, question, repeat): correctness of the
                                initial answers and of the last round under the lenient extractor and the execution
                                vote, with ties to the first agent, to the last agent, or at random; and whether
                                the run's stop test fired on a single executing answer.
  inputs/replay_bccs_qwen7b.csv one row per logged Qwen2.5-7B BCCS run: the replayed last-round outcome under the
                                same reading, the divergence class of the run (none / recoverable / unrecoverable),
                                and the lower and upper deployment bounds of Lemma 1 for that run.

Estimands (pp)
  interaction effect  Delta_peer = (last - initial) with peers minus (last - initial) without peers, per question,
                      repeats averaged, paired by question;
  online - replay     per question, online mean over its repeats minus replay mean over the logged repeats,
                      per condition, overall and by the logged divergence class of the question;
  deployment bounds   mean over questions of the per-run lower and upper bounds (Manski interval), per condition;
  allowed gap         share of logged runs whose replay diverges from deployment (P(V=1) of Lemma 1), in pp;
  run disagreement    share of runs (same question and repeat index) whose online and replayed last-round
                      outcomes differ; repeat disagreement: share of questions whose two repeats differ within
                      one collection (online or logged).
Intervals: 10,000 bootstrap draws over questions (and, for Delta_peer, over the 11 BIRD databases), seed 20261106.
Questions bird|518 and bird|701 are excluded.
"""
import argparse
import collections
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INP = HERE.parent / "inputs"
EXCL = {"bird|518", "bird|701"}
BOOT, SEED = 10000, 20261106
TIES = [("first", "initial_first", "last_first"), ("last", "initial_last_agent", "last_last_agent"),
        ("random", "initial_random", "last_random")]


def r2(v):
    return round(100 * float(v), 2)


def stats(d, db):
    """Estimate, question and database intervals, two-sided p (same estimator, seed and draws as the source analysis)."""
    us = sorted(d)
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
    return dict(n=len(us), est=r2(x.mean()), q_lo=r2(np.percentile(t, 2.5)), q_hi=r2(np.percentile(t, 97.5)),
                db_lo=r2(np.percentile(c, 2.5)), db_hi=r2(np.percentile(c, 97.5)), p=round(float(p), 5))


def r4(v):
    return round(100 * float(v), 4)


def summ(x):
    """Question bootstrap of a per-question difference (the replay check's estimator)."""
    b = x[np.random.default_rng(SEED).integers(0, len(x), size=(BOOT, len(x)))].mean(axis=1)
    return dict(n=int(len(x)), est=r2(x.mean()), q_lo=r2(np.percentile(b, 2.5)), q_hi=r2(np.percentile(b, 97.5)))


def per_question(rows, f):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["question"]].append(f(r))
    return {u: float(np.mean(v)) for u, v in by.items()}


def dpeer(rows, ini, last):
    e = lambda r: float(r[last]) - float(r[ini])
    a = per_question([r for r in rows if r["condition"] == "with_peers"], e)
    b = per_question([r for r in rows if r["condition"] == "no_peers"], e)
    return {u: a[u] - b[u] for u in set(a) & set(b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    with open(INP / "online_bccs_runs.csv", newline="") as fh:
        online = [r for r in csv.DictReader(fh) if r["question"] not in EXCL]
    with open(INP / "replay_bccs_qwen7b.csv", newline="") as fh:
        replay = [r for r in csv.DictReader(fh) if r["question"] not in EXCL]
    db = {r["question"]: r["database"] for r in online}
    V = {"n_databases": len(set(db.values()))}
    for tie, ini, last in TIES:
        s = stats(dpeer(online, ini, last), db)
        for f in ("est", "q_lo", "q_hi", "db_lo", "db_hi", "p"):
            V[f"dpeer_tie_{tie}_{f}"] = s[f]
    d = dpeer(online, "initial_first", "last_first")
    per_db = collections.defaultdict(list)
    for u, v in d.items():
        per_db[db[u]].append(v)
    V["n_databases_positive_dpeer"] = int(sum(np.mean(v) > 0 for v in per_db.values()))
    single = [r for r in online if r["single_answer_stop"] == "1"]
    V["n_single_answer_stop_runs"] = len(single)
    s = stats(dpeer([r for r in online if r["single_answer_stop"] == "0"], "initial_first", "last_first"), db)
    for f in ("n", "est", "q_lo", "q_hi", "db_lo", "db_hi", "p"):
        V[f"dpeer_excl_single_{f}"] = s[f]
    # online minus replay, per condition (ties to the first agent)
    rows = []
    for cond, tag in (("with_peers", "peers"), ("no_peers", "nopeers")):
        on = per_question([r for r in online if r["condition"] == cond], lambda r: float(r["last_first"]))
        lg = collections.defaultdict(list)
        for r in replay:
            if r["condition"] == cond:
                lg[r["question"]].append(r)
        strata = collections.defaultdict(list)
        diff, lo, hi, onm = [], [], [], []
        for u in sorted(u for u in on if u in lg):
            kinds = [r["divergence"] for r in lg[u]]
            st = ("none" if all(k == "none" for k in kinds) else
                  "unrecoverable" if any(k == "unrecoverable" for k in kinds) else "recoverable_only")
            rp = np.mean([float(r["replay_last_correct"]) for r in lg[u]])
            lo.append(np.mean([float(r["bound_low"]) for r in lg[u]]))
            hi.append(np.mean([float(r["bound_high"]) for r in lg[u]]))
            diff.append(on[u] - rp)
            onm.append(on[u])
            strata[st].append(on[u] - rp)
            rows.append(dict(condition=cond, question=u, divergence_class=st, online=on[u], replay=rp))
        s = summ(np.array(diff))
        V.update({f"online_minus_replay_{tag}_{f}": s[f] for f in ("n", "est", "q_lo", "q_hi")})
        for st, x in strata.items():
            s = summ(np.array(x))
            V.update({f"online_minus_replay_{tag}_{st}_{f}": s[f] for f in ("n", "est", "q_lo", "q_hi")})
        V[f"online_accuracy_{tag}"] = r2(np.mean(onm))
        V[f"bound_low_{tag}"], V[f"bound_high_{tag}"] = r2(np.mean(lo)), r2(np.mean(hi))
        V[f"online_inside_bounds_{tag}"] = int(np.mean(lo) <= np.mean(onm) <= np.mean(hi))
        lg_runs = [r for r in replay if r["condition"] == cond]
        V[f"diverging_runs_{tag}_pct"] = r4(np.mean([r["divergence"] != "none" for r in lg_runs]))
        on_run = {(r["question"], r["repeat"]): float(r["last_first"]) for r in online if r["condition"] == cond}
        lg_run = {(r["question"], r["repeat"]): float(r["replay_last_correct"]) for r in lg_runs}
        keys = sorted(set(on_run) & set(lg_run))
        V[f"run_disagreement_{tag}_pct"] = r4(np.mean([on_run[k] != lg_run[k] for k in keys]))
        qs = sorted({k[0] for k in keys})
        for side, runs_ in (("online", on_run), ("replay", lg_run)):
            V[f"repeat_disagreement_{side}_{tag}_pct"] = r4(np.mean([runs_[(u, "1")] != runs_[(u, "2")] for u in qs]))
    rep = [V[k] for k in V if k.startswith("repeat_disagreement_")]
    V["repeat_disagreement_min_pct"], V["repeat_disagreement_max_pct"] = min(rep), max(rep)
    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True))
    with open(out / "online_vs_replay.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
