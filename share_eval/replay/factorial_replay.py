r"""Shared-reference factorial replay on BIRD protocol logs (zero model calls).

    python3 -I share_eval/replay/factorial_replay.py              # BIRD debates, Llama-3-8B and Qwen2.5-7B
    python3 -I share_eval/replay/factorial_replay.py --ext data/records/bird_debates_backbones \
            --out output/share_eval/replay_backbones                 # larger and cross-family backbones

Factors, all on one task set per (protocol, backbone):
  E extraction  strict (protocol's own `extracted_answer`) | lenient (own, else last fenced sql block,
                else last line-initial SELECT/WITH); applied identically to every round
  I identity    string (exact string after sql_tasks.unwrap_sql; failures are ordinary labels)
                | exec (largest execution-result cluster, failed executions dropped)
                secondary: exec_ident (failures form one FAIL cluster that can win, and then scores 0)
  R round       r0 (the independent initial answers) | last (every agent's answer in the run's last round)
  A access      NOPEER | FULL

Ties go to the earliest answer in agent order. A task is the BIRD question; repeats are averaged within
a task before any contrast; the bootstrap resamples tasks. Gold is used only to score a chosen query.
The protocol's released `final` (stored per-unit outcome) is a reference row, never a factor level.
MoA is replayed by the same code but reported separately: its "last" is the aggregator's single answer
(aggregation, not interaction). FORD is excluded (its trace has no per-agent rounds).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import hashlib
import itertools
import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO / "experiments/02_controlled_sharing/code"), str(REPO / "experiments/06_cost_resources/code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Defaults follow the data layout of the repository README.
RUN_DIR = REPO / "data/records/bird_debates"                 # raw/sql_bird_v2__*.jsonl, analysis/sql_per_unit.jsonl, analysis/sql_verdicts.json
CACHE = REPO / "data/caches/sql_execution"                   # execution signatures and verdicts of every logged query
OUT_DIR = REPO / "output/share_eval/replay"

import sql_tasks as ST                                                     # noqa: E402

BOOT = 10000
SEED = 20261106
INTERACTIVE = ("du_mad", "madm2_objective", "bccs")
AGG_ONLY = ("moa",)
ARMS = ("NOPEER", "FULL")                      # level 0, level 1
EXTS = ("strict", "lenient")
IDENTS = ("string", "exec", "exec_ident")
ROUNDS = ("r0", "last")
FACTORS = ("E", "I", "R", "A")
LEVELS = {"E": EXTS, "I": ("string", "exec"), "R": ROUNDS, "A": ARMS}
REF = {"E": 0, "I": 0, "R": 1, "A": 1}          # strict, string, last, FULL

FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.S)
BARE = re.compile(r"(?im)^\s*((?:SELECT|WITH)\b[^;`]*)")


# ------------------------------------------------------------------ extraction, identity
def qkey(db: str, sql: str) -> str:
    return hashlib.sha256(f"{db}|{sql}".encode()).hexdigest()[:24]


def sql_of(raw: Any) -> str:
    return ST.unwrap_sql(raw) if raw not in (None, "", "None", "none") else ""


def lenient(raw_reply: Any) -> str:
    t = str(raw_reply or "")
    m = FENCE.findall(t)
    if m:
        return ST.unwrap_sql(m[-1])
    b = BARE.findall(t)
    return ST.unwrap_sql(b[-1]) if b else ""


def round_of(proto: str, c: Dict[str, Any]) -> int:
    if proto == "moa":
        return 0 if str(c.get("role")) == "reference" else 1
    return int(c.get("round") or 0)


TIES = ("first", "last", "random")


def vote(labels: Sequence[Optional[Any]], tie: str = "first", key: str = "") -> Tuple[Optional[Any], bool]:
    """Most frequent non-None label. Ties: `first` / `last` tied label in agent order, or `random`
    (uniform over tied labels, seeded by SEED and a per-run/per-cell key, so it is reproducible).
    Returns (winner, tied)."""
    live = [x for x in labels if x is not None]
    if not live:
        return None, False
    cnt = collections.Counter(live)
    top = max(cnt.values())
    win = {v for v in cnt if cnt[v] == top}
    order = list(dict.fromkeys(x for x in live if x in win))       # first-appearance order
    if tie == "first" or len(order) == 1:
        w = order[0]
    elif tie == "last":
        w = next(x for x in reversed(live) if x in win)
    else:
        h = int(hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest()[:12], 16)
        w = order[h % len(order)]
    return w, len(win) > 1


# ------------------------------------------------------------------ read-only execution
def _sig(args: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
    db, sql = args
    import sql_common as SQ
    t0 = time.time()
    try:
        r = SQ.execute(db, sql, max_rows=10000)
    except Exception as exc:                                             # noqa: BLE001
        return qkey(db, sql), {"status": "error", "error": str(exc)[:120], "s": 0.0}
    if r["status"] != "ok":
        return qkey(db, sql), {"status": r["status"], "s": round(time.time() - t0, 3)}
    s = sorted(repr(t) for t in SQ._norm_result(r["rows"]))
    return qkey(db, sql), {"status": "ok",
                           "sig": hashlib.sha256("\n".join(s).encode()).hexdigest()[:20],
                           "s": round(time.time() - t0, 3)}


def _verdict(args: Tuple[str, str, str]) -> Tuple[str, Dict[str, Any]]:
    """Copy of ext_sql_verdicts._one semantics."""
    db, pred, gold = args
    try:
        import sql_common as SQ
        t0 = time.time()
        pr = SQ.execute(db, pred, max_rows=10000)
        if pr["status"] != "ok":
            import ext_eval_sql as EV
            return qkey(db, pred), {"correct": False,
                                    "outcome": ("exec_timeout" if pr["status"] == "timeout"
                                                else EV.classify_error(pr.get("error") or "")),
                                    "queries": 1, "seconds": round(time.time() - t0, 3)}
        gr = SQ.execute(db, gold, max_rows=10000)
        if gr["status"] != "ok":
            return qkey(db, pred), {"correct": None, "outcome": "gold_failed", "queries": 2,
                                    "seconds": round(time.time() - t0, 3)}
        same = SQ._norm_result(pr["rows"]) == SQ._norm_result(gr["rows"])
        return qkey(db, pred), {"correct": bool(same), "outcome": "match" if same else "mismatch",
                                "queries": 2, "seconds": round(time.time() - t0, 3)}
    except Exception as exc:                                              # noqa: BLE001
        return qkey(db, pred), {"correct": False, "outcome": "other_exec_error",
                                "error": f"{type(exc).__name__}: {exc}"[:200], "queries": 0,
                                "seconds": 0.0}


def load_jsonl_cache(paths: Sequence[pathlib.Path]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        if p and p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    d = json.loads(l)
                    out[d["k"]] = d
    return out


# ------------------------------------------------------------------ loading
def load_runs(ext: pathlib.Path, per_unit_p: Optional[pathlib.Path], protos: Sequence[str]):
    per_unit = None
    if per_unit_p and per_unit_p.exists():
        per_unit = {}
        for l in per_unit_p.read_text().splitlines():
            r = json.loads(l)
            per_unit[(r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))] = r
    last: Dict[Tuple, Dict[str, Any]] = {}
    seen = collections.Counter()
    for p in sorted((ext / "raw").glob("sql_bird_v2__*.jsonl")):
        if p.name.split("__")[1] not in protos or p.name.split("__")[2] not in ARMS:
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                seen["unparseable_line"] += 1
                continue
            if r.get("protocol") not in protos or r.get("arm") not in ARMS:
                continue
            k = (r.get("unit"), r.get("protocol"), r.get("arm"), r.get("backbone"),
                 int(r.get("repeat") or 0))
            if r.get("status") != "ok":
                seen["non_ok_record"] += 1
                continue
            if per_unit is not None and k not in per_unit:
                seen["ok_record_not_in_evaluated_set"] += 1
                continue
            if k in last:
                seen["duplicate_ok_record_later_kept"] += 1
            last[k] = r                     # append-only convention: last ok record wins
    return last, per_unit, dict(seen)


def build_rows(last, per_unit, protos):
    """One row per run: candidate queries per (round, extraction)."""
    rows = []
    for k, r in last.items():
        unit, proto, arm, bb, rep = k
        db = ((r.get("config") or {}).get("unit_meta") or {}).get("db_id")
        calls = (r.get("record") or {}).get("calls") or []
        by_round = collections.defaultdict(list)
        for c in calls:
            by_round[round_of(proto, c)].append(c)
        if 0 not in by_round:
            continue
        lr = max(by_round)
        sets = {}
        for name, cs in (("r0", by_round[0]), ("last", by_round[lr])):
            cs = sorted(cs, key=lambda c: int(c.get("agent_index") or 0))
            strict = [sql_of(c.get("extracted_answer", c.get("prediction"))) for c in cs]
            lens = [s if s else lenient(c.get("raw_reply")) for s, c in zip(strict, cs)]
            sets[name] = {"strict": strict, "lenient": lens}
        fin = None
        if per_unit is not None:
            fin = per_unit[k]["outcome"]
        rows.append(dict(unit=unit, protocol=proto, arm=arm, backbone=bb, repeat=rep, db=db,
                         sets=sets, last_round=lr, final_outcome=fin))
    return rows


# ------------------------------------------------------------------ scoring
def score_run(row, sigs, verdicts):
    """Correctness for every (ext, ident, round, tie rule); tie / live flags for every (ext, ident, round)."""
    db = row["db"]
    out, ties, live = {}, {}, {}
    for rd in ROUNDS:
        for e in EXTS:
            qs = row["sets"][rd][e]
            corr = [bool(q and (verdicts.get(qkey(db, q)) or {}).get("correct")) for q in qs]
            for i in IDENTS:
                lab = []
                for q in qs:
                    if not q:
                        lab.append(None)
                    elif i == "string":
                        lab.append(q)
                    else:
                        s = sigs.get(qkey(db, q)) or {}
                        if s.get("status") == "ok":
                            lab.append(s["sig"])
                        else:
                            lab.append("FAIL" if i == "exec_ident" else None)
                rk = f"{row['unit']}|{row['protocol']}|{row['arm']}|{row['backbone']}|{row['repeat']}|{rd}|{e}|{i}"
                for tie in TIES:
                    w, t = vote(lab, tie, rk)
                    ok = w is not None and w != "FAIL" and corr[lab.index(w)]
                    out[(e, i, rd, tie)] = float(ok)
                ties[(e, i, rd)] = float(t)
                live[(e, i, rd)] = float(any(x is not None for x in lab))
    return out, ties, live


def boot_idx(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, size=(BOOT, n))


def summ(d: np.ndarray, idx: np.ndarray) -> Dict[str, Any]:
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    p = max(1.0 / BOOT, min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))
    return dict(est=round(100 * float(d.mean()), 2), lo=round(100 * float(lo), 2),
                hi=round(100 * float(hi), 2), p=round(float(p), 4))


def factorial(Y: np.ndarray, idx: np.ndarray) -> Dict[str, Any]:
    """Y: units x 2x2x2x2 over (E, I, R, A). Effects use the difference-of-differences convention:
    main = mean(level1) - mean(level0); two-way = (effect of X at Y=1) - (effect of X at Y=0),
    each averaged over the remaining factors; likewise higher orders."""
    n = Y.shape[0]
    flat = Y.reshape(n, 16)
    signs = np.array(list(itertools.product((-1, 1), repeat=4)))   # E,I,R,A order matches reshape
    res = {}
    for k in range(1, 5):
        for S in itertools.combinations(range(4), k):
            w = np.prod(signs[:, S], axis=1) * (2 ** k) / 16.0
            res["x".join(FACTORS[j] for j in S)] = summ(flat @ w, idx)
    return res


def cell_index(e, i, r, a):
    return (e, i, r, a)


def analyse_cell(units: List[str], Y: np.ndarray, F: Optional[np.ndarray], idx: np.ndarray):
    out = dict(n_units=len(units))
    out["levels"] = {f"{EXTS[e]}|{LEVELS['I'][i]}|{ROUNDS[r]}|{ARMS[a]}": round(100 * float(Y[:, e, i, r, a].mean()), 2)
                     for e, i, r, a in itertools.product((0, 1), repeat=4)}
    out["effects"] = factorial(Y, idx)
    # Δpeer per (E, I)
    dp = {}
    for e, i in itertools.product((0, 1), repeat=2):
        d = (Y[:, e, i, 1, 1] - Y[:, e, i, 0, 1]) - (Y[:, e, i, 1, 0] - Y[:, e, i, 0, 0])
        dp[f"{EXTS[e]}|{LEVELS['I'][i]}"] = summ(d, idx)
        dp[f"{EXTS[e]}|{LEVELS['I'][i]}"]["last_minus_r0_FULL"] = summ(Y[:, e, i, 1, 1] - Y[:, e, i, 0, 1], idx)
        dp[f"{EXTS[e]}|{LEVELS['I'][i]}"]["last_minus_r0_NOPEER"] = summ(Y[:, e, i, 1, 0] - Y[:, e, i, 0, 0], idx)
    out["delta_peer"] = dp
    # simple effects of every factor at every combination of the other three
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
    # shared-reference ledger
    ref = [REF[f] for f in FACTORS]
    yref = Y[:, ref[0], ref[1], ref[2], ref[3]]
    single = {}
    for j, f in enumerate(FACTORS):
        c = list(ref)
        c[j] = 1 - c[j]
        single[f] = Y[:, c[0], c[1], c[2], c[3]] - yref
    led = dict(reference_cell="strict|string|last|FULL", reference_level=round(100 * float(yref.mean()), 2))
    rows = []
    for k in range(1, 5):
        for S in itertools.combinations(range(4), k):
            c = list(ref)
            for j in S:
                c[j] = 1 - c[j]
            joint = Y[:, c[0], c[1], c[2], c[3]] - yref
            ssum = sum(single[FACTORS[j]] for j in S)
            rows.append(dict(switched="+".join(FACTORS[j] for j in S),
                             cell=f"{EXTS[c[0]]}|{LEVELS['I'][c[1]]}|{ROUNDS[c[2]]}|{ARMS[c[3]]}",
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


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default=str(RUN_DIR),
                    help="run directory with raw/sql_bird_v2__*.jsonl")
    ap.add_argument("--per-unit", default=None,
                    help="evaluated set with stored final outcomes (default <ext>/analysis/sql_per_unit.jsonl; "
                         "if absent, all ok records are used and no final reference row is produced)")
    ap.add_argument("--verdicts", default=None,
                    help="stored verdict json (default <ext>/analysis/sql_verdicts.json; optional)")
    ap.add_argument("--seed-sig-cache", default=",".join(str(CACHE / n) for n in ("r0_exec_signatures.jsonl", "replay_exec_signature_cache.jsonl")),
                    help="read-only signature cache(s), comma-separated")
    ap.add_argument("--verdict-caches", default=str(CACHE / "replay_verdict_cache.jsonl"),
                    help="read-only verdict cache(s), comma-separated, applied before <out>/verdict_cache.jsonl")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--protocols", default=",".join(INTERACTIVE + AGG_ONLY))
    ap.add_argument("--exclude-units", default="bird|518,bird|701",
                    help="tasks dropped a priori: the reference query fails or times out under the 5 s evaluator. "
                         "Data-driven gold_failed tasks are added.")
    ap.add_argument("--audit-verdicts", default=str(CACHE / "verdict_audit.json"),
                    help="optional list of stored-vs-fresh execution inconsistencies to re-check")
    a = ap.parse_args()
    ext = pathlib.Path(a.ext).resolve()
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    protos = tuple(a.protocols.split(","))
    t_start = time.time()

    per_unit_p = pathlib.Path(a.per_unit) if a.per_unit else ext / "analysis/sql_per_unit.jsonl"
    last, per_unit, load_counts = load_runs(ext, per_unit_p if per_unit_p.exists() else None, protos)
    rows = build_rows(last, per_unit, protos)
    print(f"{len(rows)} runs loaded from {ext}; per_unit={'yes' if per_unit is not None else 'no'}", flush=True)

    import sql_common as SQ
    gold = {f"bird|{q['question_id']}": q["SQL"]
            for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}

    verd_p = pathlib.Path(a.verdicts) if a.verdicts else ext / "analysis/sql_verdicts.json"
    verdicts = json.loads(verd_p.read_text())["verdicts"] if verd_p.exists() else {}
    own_v = out / "verdict_cache.jsonl"
    for k, v in load_jsonl_cache([pathlib.Path(p) for p in a.verdict_caches.split(",") if p] + [own_v]).items():
        verdicts[k] = v
    own_s = out / "exec_signature_cache.jsonl"
    sigs = load_jsonl_cache([pathlib.Path(p) for p in a.seed_sig_cache.split(",") if p] + [own_s])

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
    t0 = time.time()
    if todo or gtodo:
        with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
            with open(own_s, "a") as fh:
                for kk, v in ex.map(_sig, list(todo.values()), chunksize=16):
                    v["k"] = kk
                    sigs[kk] = v
                    fh.write(json.dumps(v) + "\n")
            with open(own_v, "a") as fh:
                for kk, v in ex.map(_verdict, list(gtodo.values()), chunksize=16):
                    verdicts[kk] = v
                    fh.write(json.dumps(dict(v, k=kk)) + "\n")
    exec_cost = dict(new_signatures=len(todo), new_verdicts=len(gtodo), wall_s=round(time.time() - t0, 1))

    # ---- task-set exclusions (named and counted)
    a_priori = {u for u in a.exclude_units.split(",") if u}
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
    scored = collections.defaultdict(lambda: collections.defaultdict(list))   # (proto, bb) -> unit -> runs
    tie_acc = collections.defaultdict(lambda: collections.defaultdict(list))
    run_dump = []
    for r in rows:
        if r["unit"] in gold_failed_units or not r["db"]:
            continue
        s, t, lv = score_run(r, sigs, verdicts)
        rec = dict(arm=r["arm"], repeat=r["repeat"], s=s, final=None if r["final_outcome"] is None else float(r["final_outcome"] == "match"),
                   last_round=r["last_round"])
        scored[(r["protocol"], r["backbone"])][r["unit"]].append(rec)
        for key in t:
            tie_acc[(r["protocol"], r["backbone"], r["arm"])][key].append((t[key], lv[key]))
        run_dump.append(dict(unit=r["unit"], protocol=r["protocol"], arm=r["arm"], backbone=r["backbone"],
                             repeat=r["repeat"], last_round=r["last_round"], final=rec["final"],
                             **{"|".join(k): v for k, v in s.items()}))

    def build_Y(by_unit, ident_set, keep=lambda x: True, tie="first"):
        units = sorted(u for u, rs in by_unit.items() if {x["arm"] for x in rs if keep(x)} >= set(ARMS))
        Y = np.zeros((len(units), 2, 2, 2, 2))
        F = np.zeros((len(units), 2)) if per_unit is not None else None
        for ui, u in enumerate(units):
            for a_i, arm in enumerate(ARMS):
                rs = [x for x in by_unit[u] if x["arm"] == arm and keep(x)]
                for e, i, rd in itertools.product((0, 1), repeat=3):
                    Y[ui, e, i, rd, a_i] = np.mean([x["s"][(EXTS[e], ident_set[i], ROUNDS[rd], tie)] for x in rs])
                if F is not None:
                    F[ui, a_i] = np.mean([x["final"] for x in rs])
        return units, Y, F

    cells, pooled_inputs, partial_units = {}, collections.defaultdict(dict), {}
    pooled_tie = collections.defaultdict(dict)
    for (proto, bb), by_unit in sorted(scored.items()):
        units, Y, F = build_Y(by_unit, ("string", "exec"))
        partial_units[f"{proto}|{bb}"] = sorted(set(by_unit) - set(units))
        idx = boot_idx(len(units))
        res = {}
        res["primary"] = analyse_cell(units, Y, F, idx)
        if proto in INTERACTIVE:
            pooled_inputs[(proto, bb)] = (units, Y, F)
        u2, Y2, F2 = build_Y(by_unit, ("string", "exec_ident"))
        r_ = analyse_cell(u2, Y2, F2, boot_idx(len(u2)))
        res["secondary_exec_ident"] = {k: r_[k] for k in ("n_units", "levels", "effects", "delta_peer")}
        res["secondary_exec_ident"]["note"] = "identity level 1 = exec_ident (failures form one cluster; a FAIL win scores 0)"
        # completeness: tasks with all 16 cells observable, and with both repeats in both arms
        reps = {u: collections.Counter(x["arm"] for x in rs) for u, rs in by_unit.items()}
        res["completeness"] = dict(
            tasks_with_runs=len(by_unit), tasks_all_16_cells=len(units),
            tasks_both_repeats_both_arms=sum(1 for c in reps.values() if all(c[a] >= 2 for a in ARMS)))
        # per independent repeat: r1 FULL vs r1 NOPEER, r2 vs r2
        byrep = {}
        for rep in sorted({x["repeat"] for rs in by_unit.values() for x in rs}):
            ur, Yr, Fr = build_Y(by_unit, ("string", "exec"), keep=lambda x, rep=rep: x["repeat"] == rep)
            rr = analyse_cell(ur, Yr, Fr, boot_idx(len(ur)))
            byrep[f"r{rep}"] = {k: rr[k] for k in ("n_units", "delta_peer", "effects")}
        res["by_repeat"] = byrep
        # early stopping: runs whose last round is the initial round (last = r0, as the deployed protocol stops)
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
        # ties / extraction coverage
        # tie-break sensitivity (every vote-based cell)
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

    # ---- average over the interactive (protocol, backbone) cells on their common task set
    common = sorted(set.intersection(*[set(v[0]) for v in pooled_inputs.values()])) if pooled_inputs else []
    pooled = None
    if common:
        Ys, Fs = [], []
        for (units, Y, F) in pooled_inputs.values():
            pos = {u: i for i, u in enumerate(units)}
            sel = [pos[u] for u in common]
            Ys.append(Y[sel])
            if F is not None:
                Fs.append(F[sel])
        pooled = analyse_cell(common, np.mean(Ys, axis=0), np.mean(Fs, axis=0) if Fs else None, boot_idx(len(common)))
        pooled["note"] = ("equal-weight average over the interactive (protocol, backbone) cells, per task; "
                          "tasks resampled; same source version; not independent replications")
        pooled["cells_averaged"] = [f"{p}|{b}" for p, b in pooled_inputs]
        pooled["tie_sensitivity"] = {}
        for tie, inp in pooled_tie.items():
            Ys, Fs = [], []
            for (units, Y, F) in inp.values():
                pos = {u: i for i, u in enumerate(units)}
                Ys.append(Y[[pos[u] for u in common]])
                if F is not None:
                    Fs.append(F[[pos[u] for u in common]])
            rt = analyse_cell(common, np.mean(Ys, axis=0), np.mean(Fs, axis=0) if Fs else None, boot_idx(len(common)))
            pooled["tie_sensitivity"][tie] = {k: rt[k] for k in ("levels", "effects", "delta_peer", "ledger", "final_reference") if k in rt}

    # ---- re-check of execution-class inconsistencies (stored verdict vs this host / signature status)
    cand = {}
    for r in rows:
        for rd in ROUNDS:
            for e in EXTS:
                for q in r["sets"][rd][e]:
                    if q and r["db"]:
                        cand.setdefault(qkey(r["db"], q), (r["db"], q, gold[r["unit"]], r["unit"]))
    suspects = {}
    ap_ = pathlib.Path(a.audit_verdicts)
    if ap_.exists():
        av = json.loads(ap_.read_text())
        for x in (av.get("fresh_vs_stored") or {}).get("differing", []) + \
                 (av.get("audited_sig_vs_stored") or {}).get("inconsistent", []):
            suspects[x["k"]] = "audit_list"
    FAILV = {"syntax_error", "missing_object", "other_exec_error", "type_or_arg_error", "exec_timeout", "declined"}
    for kk in cand:
        sg, vd = sigs.get(kk) or {}, verdicts.get(kk) or {}
        if not sg or not vd:
            continue
        if (sg.get("status") == "ok") != (vd.get("outcome") not in FAILV):
            suspects.setdefault(kk, "signature_vs_verdict_class")
    sus_in = {kk: v for kk, v in suspects.items() if kk in cand}
    alt_s, alt_v = dict(sigs), dict(verdicts)
    if sus_in:
        with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
            items = [cand[kk] for kk in sus_in]
            for kk, v in ex.map(_sig, [(d, q) for d, q, _g, _u in items]):
                alt_s[kk] = v
            for kk, v in ex.map(_verdict, [(d, q, g) for d, q, g, _u in items]):
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
            s0, _, _ = score_run(r, sigs, verdicts)
            s1, _, _ = score_run(r, alt_s, alt_v)
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
                per_unit_used=per_unit is not None, boot_draws=BOOT, seed=SEED,
                reference_cell="strict|string|last|FULL", effect_convention="difference-of-differences, pp",
                exclusions=excl, verdict_inconsistency_recheck=inconsistency, partial_units_dropped=partial_units, execution_cost=exec_cost,
                pooled_interactive=pooled, cells=cells, wall_s=round(time.time() - t_start, 1))
    (out / "factorial_replay.json").write_text(json.dumps(blob, indent=1, default=str))
    with open(out / "per_run_scores.jsonl", "w") as fh:
        for d in run_dump:
            fh.write(json.dumps(d) + "\n")

    # ---- flat csv tables
    def rows_for(name, res):
        for k, v in res["effects"].items():
            yield dict(cell=name, table="effect", term=k, **{x: v[x] for x in ("est", "lo", "hi", "p")})
        for k, v in res["delta_peer"].items():
            yield dict(cell=name, table="delta_peer", term=k, **{x: v[x] for x in ("est", "lo", "hi", "p")})
        for k, v in res["simple_effects"].items():
            yield dict(cell=name, table="simple_effect", term=k, **{x: v[x] for x in ("est", "lo", "hi", "p")})
        for r in res["ledger"]["rows"]:
            yield dict(cell=name, table="ledger_joint", term=r["switched"], **{x: r["joint"][x] for x in ("est", "lo", "hi", "p")})
            if r["nonadditivity"]:
                yield dict(cell=name, table="ledger_nonadditivity", term=r["switched"],
                           **{x: r["nonadditivity"][x] for x in ("est", "lo", "hi", "p")})
        for k, v in res["levels"].items():
            yield dict(cell=name, table="level", term=k, est=v, lo=None, hi=None, p=None)
        for k, v in (res.get("final_reference") or {}).items():
            if isinstance(v, dict):
                yield dict(cell=name, table="final_reference", term=k, **{x: v[x] for x in ("est", "lo", "hi", "p")})
            else:
                yield dict(cell=name, table="final_reference", term=k, est=v, lo=None, hi=None, p=None)
    flat = []
    for name, res in cells.items():
        flat += list(rows_for(name, res["primary"]))
    if pooled:
        flat += list(rows_for("POOLED_interactive", pooled))
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

    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS"}
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in hashes.items()))
    print(f"exclusions: {excl['gold_failed_units']} ({excl['runs_dropped_by_unit_exclusion']} runs); "
          f"exec {exec_cost}; wall {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
