#!/usr/bin/env python3
"""Stronger gold-free selectors on the existing SQL candidate pool.

    python3 code/b1_selectors.py

Writes `output/b1_*` only. Zero model calls, zero new
database executions: every selector reads the stored candidate records the workflow already had.

The released rule is `collect_sql_arms.majority_pick`, transcribed and checked against the stored
`pick_model` on every question-pool pair before anything is compared to it.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import pathlib
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]
DEC = REPO / "data/derived/sql_decomposition"
DATA = REPO / "data/records/three_domain_four_question"
OUT = WORK / "output"

#: published order; `majority_pick` ends in a stable sort so the order decides byte-identical ties
POOL_MODELS = {
    "pool4": ("qwen2.5-7b", "llama-3-8b", "qwen2.5-14b", "qwen2.5-32b"),
    "pool6": ("qwen2.5-7b", "llama-3-8b", "qwen2.5-14b", "qwen2.5-32b",
              "mistral-7b", "qwen2.5-72b"),
}
POOL_ARM = {"pool4": "S2", "pool6": "S3"}
POOL_SIZE = {"pool4": 4, "pool6": 6}
K_FOLDS = 5
DRAWS = 5000
SEED = 20261026

METHODS = ("released", "valid_majority", "nonempty_majority",
           "reliability_vote", "reliability_nonempty")


def result_signature(res: Dict[str, Any]) -> str:
    """Transcribed from `collect_sql_arms.result_signature`; reads only this candidate's result."""
    if res.get("status") != "ok":
        return "__%s__" % res.get("status")
    rows = res.get("rows") or []
    norm = sorted(tuple(str(c) for c in r) for r in rows)
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()[:24]


def tie_break(cands: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The released tie-break: lowest-sorted SQL text. Shared by every method."""
    return sorted(cands, key=lambda x: x["sql"].strip())[0]


def groups_of(live: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for c in live:
        g[result_signature(c["exec"])].append(c)
    return g


def sel_released(live, rel=None):
    g = groups_of(live)
    top = max(len(v) for v in g.values())
    return tie_break([x for v in g.values() if len(v) == top for x in v])


def _ok(live):
    return [c for c in live if (c.get("exec") or {}).get("status") == "ok"]


def _nonempty(cands):
    return [c for c in cands if ((c.get("exec") or {}).get("n_rows_returned") or 0) > 0]


def sel_valid_majority(live, rel=None):
    ok = _ok(live)
    if not ok:
        return sel_released(live)
    g = groups_of(ok)
    top = max(len(v) for v in g.values())
    return tie_break([x for v in g.values() if len(v) == top for x in v])


def sel_nonempty_majority(live, rel=None):
    ok = _ok(live)
    if not ok:
        return sel_released(live)
    ne = _nonempty(ok)
    pool = ne if ne else ok
    g = groups_of(pool)
    top = max(len(v) for v in g.values())
    return tie_break([x for v in g.values() if len(v) == top for x in v])


def _rel_pick(cands, rel: Dict[str, float]):
    g = groups_of(cands)
    best, bestv = None, None
    for v in g.values():
        w = sum(rel.get(c["model"], 0.0) for c in v)
        if bestv is None or w > bestv:
            best, bestv = v, w
    return tie_break(best)


def sel_reliability_vote(live, rel: Dict[str, float]):
    ok = _ok(live)
    if not ok:
        return sel_released(live)
    return _rel_pick(ok, rel)


def sel_reliability_nonempty(live, rel: Dict[str, float]):
    ok = _ok(live)
    if not ok:
        return sel_released(live)
    ne = _nonempty(ok)
    return _rel_pick(ne if ne else ok, rel)


SELECTORS = {
    "released": sel_released,
    "valid_majority": sel_valid_majority,
    "nonempty_majority": sel_nonempty_majority,
    "reliability_vote": sel_reliability_vote,
    "reliability_nonempty": sel_reliability_nonempty,
}


def load_peers(bench: str) -> Dict[int, Dict[str, Dict[str, Any]]]:
    out: Dict[int, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
    with open(DATA / bench / "peers.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("role") in ("peer", "extra"):
                out[int(r["question_id"])][r["model"]] = r
    return out


def load_cand_labels(bench: str) -> Dict[Tuple[int, str], Optional[bool]]:
    """Gold correctness per candidate. Used only for development reliability, scoring and oracle."""
    out: Dict[Tuple[int, str], Optional[bool]] = {}
    with open(DEC / ("cand_%s.jsonl" % bench), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(int(r["question_id"]), r["model"])] = r.get("match")
    return out


def db_boot(by_db: Dict[str, List[float]], draws=DRAWS, seed=SEED) -> Dict[str, float]:
    keys = sorted(by_db)
    tot = sum(len(v) for v in by_db.values())
    if not tot:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    point = sum(sum(v) for v in by_db.values()) / tot
    rnd = random.Random(seed)
    sims = []
    for _ in range(draws):
        s = c = 0.0
        for _ in range(len(keys)):
            k = keys[rnd.randrange(len(keys))]
            s += sum(by_db[k])
            c += len(by_db[k])
        sims.append(s / c if c else 0.0)
    sims.sort()
    return {"mean": point, "lo": sims[int(0.025 * draws)], "hi": sims[int(0.975 * draws) - 1]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    per = list(csv.DictReader(open(DEC / "per_execution.csv", encoding="utf-8")))
    peers = {b: load_peers(b) for b in ("bird", "spider")}
    labels = {b: load_cand_labels(b) for b in ("bird", "spider")}

    # --- transcription check against the stored pick, before any comparison
    agree = dis = 0
    for key in sorted({(r["benchmark"], r["pool"], int(r["question_id"]), r["pick_model"])
                       for r in per}):
        b, pool, q, stored = key
        live = [peers[b][q][m] for m in POOL_MODELS[pool] if m in peers[b][q]
                and (peers[b][q][m].get("sql") or "").strip()]
        got = sel_released(live)["model"] if live else None
        agree += int(got == (stored or None))
        dis += int(got != (stored or None))
    print("released-rule transcription vs stored pick: %d agree, %d disagree" % (agree, dis))
    if dis:
        raise SystemExit("transcription does not reproduce the deployed selector")

    # --- usable executions, same exclusion for every method
    exes = []
    dropped = 0
    for r in per:
        if r["repeat"] == "" or r["selection_correct"] == "" or r["generation_correct"] == "":
            dropped += 1
            continue
        exes.append({"benchmark": r["benchmark"], "pool": r["pool"], "recipient": r["recipient"],
                     "db_id": r["db_id"], "question_id": int(r["question_id"]),
                     "repeat": int(r["repeat"]),
                     "published_selection_correct": r["selection_correct"] == "1",
                     "generate_correct": r["generation_correct"] == "1"})
    print("usable executions %d, excluded %d (missing generation cell, no repeat index)"
          % (len(exes), dropped))

    folds: Dict[Tuple[str, str], int] = {}
    for b in sorted({e["benchmark"] for e in exes}):
        dbs = sorted({e["db_id"] for e in exes if e["benchmark"] == b})
        for i, db in enumerate(dbs):
            folds[(b, db)] = i % min(K_FOLDS, len(dbs))

    # question -> db, and the per-question pool
    qdb: Dict[Tuple[str, int], str] = {}
    for e in exes:
        qdb[(e["benchmark"], e["question_id"])] = e["db_id"]

    # --- per (benchmark, pool, question): each method's chosen candidate and its correctness
    choice: Dict[Tuple[str, str, int, str], Optional[bool]] = {}
    oracle: Dict[Tuple[str, str, int], bool] = {}
    chosen_model: Dict[Tuple[str, str, int, str], Optional[str]] = {}
    keys = sorted({(e["benchmark"], e["pool"], e["question_id"]) for e in exes})

    # reliability per fold, estimated on development databases only
    rel_by_fold: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for b, pool in sorted({(k[0], k[1]) for k in keys}):
        nf = min(K_FOLDS, len({qdb[(b, q)] for _b, _p, q in keys if _b == b}))
        for k in range(nf):
            num: Dict[str, int] = collections.Counter()
            den: Dict[str, int] = collections.Counter()
            for _b, _p, q in keys:
                if _b != b or _p != pool:
                    continue
                if folds[(b, qdb[(b, q)])] == k:
                    continue                      # held out; never used for reliability
                for m in POOL_MODELS[pool]:
                    lab = labels[b].get((q, m))
                    if lab is None:
                        continue
                    den[m] += 1
                    num[m] += int(bool(lab))
            rel_by_fold[(b, pool, k)] = {m: (num[m] / den[m] if den[m] else 0.0)
                                         for m in POOL_MODELS[pool]}

    for b, pool, q in keys:
        live = [peers[b][q][m] for m in POOL_MODELS[pool] if m in peers[b][q]
                and (peers[b][q][m].get("sql") or "").strip()]
        k = folds[(b, qdb[(b, q)])]
        rel = rel_by_fold[(b, pool, k)]
        oracle[(b, pool, q)] = any(bool(labels[b].get((q, m))) for m in POOL_MODELS[pool])
        for name, fn in SELECTORS.items():
            c = fn(live, rel) if live else None
            chosen_model[(b, pool, q, name)] = c["model"] if c else None
            choice[(b, pool, q, name)] = (bool(labels[b].get((q, c["model"])))
                                          if c else False)

    # --- consistency: the released method must reproduce the published selection_correct column
    bad = sum(1 for e in exes
              if choice[(e["benchmark"], e["pool"], e["question_id"], "released")]
              != e["published_selection_correct"])
    print("executions where the transcribed released selector's correctness differs from the "
          "published column: %d" % bad)

    rows: List[Dict[str, Any]] = []
    cells = sorted({(e["benchmark"], e["pool"], e["recipient"]) for e in exes})
    for b, pool, rec in cells:
        R = [e for e in exes if e["benchmark"] == b and e["pool"] == pool
             and e["recipient"] == rec]
        n = len(R)
        acc: Dict[str, Dict[str, List[float]]] = {}
        for name in METHODS + ("pool_oracle", "generation"):
            by: Dict[str, List[float]] = collections.defaultdict(list)
            for e in R:
                k = (b, pool, e["question_id"])
                if name == "pool_oracle":
                    y = float(oracle[k])
                elif name == "generation":
                    y = float(e["generate_correct"])
                else:
                    y = float(choice[(b, pool, e["question_id"], name)])
                by[e["db_id"]].append(y)
            acc[name] = by
        for name in METHODS + ("pool_oracle", "generation"):
            st = db_boot(acc[name])
            d_rel: Dict[str, List[float]] = collections.defaultdict(list)
            d_gen: Dict[str, List[float]] = collections.defaultdict(list)
            rec_win = reg_win = rec_gen = reg_gen = 0
            for e in R:
                k = (b, pool, e["question_id"])
                y = (float(oracle[k]) if name == "pool_oracle"
                     else float(e["generate_correct"]) if name == "generation"
                     else float(choice[(b, pool, e["question_id"], name)]))
                yr = float(choice[(b, pool, e["question_id"], "released")])
                yg = float(e["generate_correct"])
                d_rel[e["db_id"]].append(y - yr)
                d_gen[e["db_id"]].append(y - yg)
                rec_win += int(y > yr)
                reg_win += int(y < yr)
                rec_gen += int(y > yg)
                reg_gen += int(y < yg)
            dr = db_boot(d_rel)
            dg = db_boot(d_gen)
            oracle_mean = db_boot(acc["pool_oracle"])["mean"]
            rows.append({
                "benchmark": b, "pool": pool, "recipient": rec, "method": name,
                "executions": n,
                "accuracy": st["mean"], "accuracy_lo": st["lo"], "accuracy_hi": st["hi"],
                "minus_released_pp": 100 * dr["mean"],
                "minus_released_lo_pp": 100 * dr["lo"], "minus_released_hi_pp": 100 * dr["hi"],
                "minus_generation_pp": 100 * dg["mean"],
                "minus_generation_lo_pp": 100 * dg["lo"],
                "minus_generation_hi_pp": 100 * dg["hi"],
                "recoveries_vs_released": rec_win, "regressions_vs_released": reg_win,
                "recoveries_vs_generation": rec_gen, "regressions_vs_generation": reg_gen,
                "gap_to_pool_oracle_pp": 100 * (oracle_mean - st["mean"]),
                "extra_model_calls": 0, "extra_database_executions": 0,
                "development_gold_labels_required": name.startswith("reliability"),
                "recipient_calls": n if name == "generation" else 0,
                "candidate_pool_calls_per_question": POOL_SIZE[pool],
            })

    print("\n%-7s %-6s %-12s %-22s %7s %22s %22s %7s"
          % ("bench", "pool", "recipient", "method", "acc", "vs released (pp)",
             "vs generation (pp)", "oracleΔ"))
    for r in rows:
        print("%-7s %-6s %-12s %-22s %7.4f %+6.2f [%+5.2f,%+5.2f] %+6.2f [%+5.2f,%+5.2f] %7.2f"
              % (r["benchmark"], r["pool"], r["recipient"], r["method"], r["accuracy"],
                 r["minus_released_pp"], r["minus_released_lo_pp"], r["minus_released_hi_pp"],
                 r["minus_generation_pp"], r["minus_generation_lo_pp"],
                 r["minus_generation_hi_pp"], r["gap_to_pool_oracle_pp"]))

    with open(OUT / "b1_selectors.json", "w", encoding="utf-8") as f:
        json.dump({"methods": list(METHODS), "folds": K_FOLDS, "draws": DRAWS,
                   "transcription_check": {"agree": agree, "disagree": dis},
                   "released_matches_published_column_mismatches": bad,
                   "excluded_executions": dropped,
                   "interval_note": "databases resampled; the method set is fixed in the protocol "
                                    "so the interval conditions on no tuning step",
                   "results": rows}, f, indent=1, sort_keys=True, default=str)
    cols = list(rows[0].keys())
    with open(OUT / "b1_selectors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote output/b1_selectors.csv (%d rows)" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
