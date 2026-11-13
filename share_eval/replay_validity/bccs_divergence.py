r"""First-divergence events D for BCCS readout replays on BIRD (zero model calls).

    python3 -I share_eval/replay_validity/bccs_divergence.py     # Llama-3-8B and Qwen2.5-7B, the replay's population
    python3 -I share_eval/replay_validity/bccs_divergence.py --ext data/records/bird_debates_backbones \
            --scores "" --out output/share_eval/replay_validity_backbones --backbones qwen25-14b-awq,...

BCCS reads its own answer labels after every non-final round: the stop test, the no-consensus flag
T, and (FULL only) the conflict matrix that chooses which peers each agent sees. A readout replay
under another identity / extraction (o') is therefore exact only on runs where every such read is
unchanged. This script recomputes those reads on the logged prefix under the protocol's labels
(must reproduce the log exactly) and under each o', finds the first divergence, classifies it as
recoverable (o' would stop earlier: output computable from the log) or unrecoverable (o' would
continue past a logged stop, or would show different peers / a different tail), and turns the
factorial's replay estimates into Manski-type bounds (Lemma 1 of the paper).

The BCCS functions are taken verbatim from the adapter source by AST extraction, so the stop rule,
conflict estimator and peer selection are the protocol's own code, not a re-implementation.
"""
from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures as cf
import hashlib
import json
import math
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.dont_write_bytecode = True

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "share_eval/replay"))
import factorial_replay as FR                                              # noqa: E402
REPLAY_OUT = REPO / "output/share_eval/replay"

BCCS_SRC = REPO / "code/consensus_delivery/adapters/bccs.py"
NEEDED = ("list_mean", "calculate_entropy", "estimate_uncertainty", "_strip_period",
          "estimate_conflict", "estimate_consensus", "find_top_indices", "construct_message")
BOOT, SEED = 10000, 20261106
ARMS = ("NOPEER", "FULL")
REASON = collections.defaultdict(collections.Counter)
MAPS = ("proto", "ref", "I", "E", "EI", "EI_ident", "ref_tie_last")


def load_bccs_functions() -> Tuple[Dict[str, Any], str]:
    src = BCCS_SRC.read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NEEDED]
    assert {n.name for n in keep} == set(NEEDED), "BCCS source changed"
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in ("PREFIX", "TAIL_TRUSTABLE", "TAIL_LEADING") for t in n.targets)]
    assert len(consts) == 3, "BCCS prompt constants changed"
    keep = consts + keep
    mod = ast.Module(body=keep, type_ignores=[])
    ns: Dict[str, Any] = {"math": math, "np": np, "Sequence": Sequence, "List": List, "Dict": Dict,
                          "Any": Any, "Tuple": Tuple, "Optional": Optional}
    exec(compile(mod, str(BCCS_SRC), "exec"), ns)                          # noqa: S102
    return ns, hashlib.sha256(src.encode()).hexdigest()


BC, BCCS_SHA = load_bccs_functions()


def consensus_with_tie(support: Sequence[str], probs: Sequence[float], tie: str) -> Dict[str, Any]:
    """`estimate_consensus` with the modal label among ties chosen first (released) or last."""
    if tie == "first":
        return BC["estimate_consensus"](support, probs)
    srs = [BC["_strip_period"](s if s != "" else "None") for s in support]
    if not srs:
        return {"verdict": "no consensus"}
    top = max(srs.count(s) for s in srs)
    modal = [s for s in srs if srs.count(s) == top][-1]
    freq = srs.count(modal)
    score = sum(p for s, p in zip(srs, probs) if s == modal)
    total = sum(probs)
    cs = (score / total) if total else 1.0
    share = freq / len(srs)
    if share > 2 / 3 and cs > 0.8:
        v = "full consensus"
    elif share >= 2 / len(srs) and cs > 0.5:
        v = "partial consensus"
    else:
        v = "no consensus"
    return {"verdict": v, "modal_answer": modal}


# ------------------------------------------------------------------ label maps
def labels_for(mp: str, run_round: Dict[str, Any], db: str, sigs: Dict[str, Any]) -> List[str]:
    """Support labels for one round under map `mp`. Singleton labels cannot match anything."""
    sup, strict, lenient = run_round["support"], run_round["strict"], run_round["lenient"]
    out = []
    for i, s in enumerate(sup):
        if mp == "proto":
            out.append(s)
            continue
        q = lenient[i] if mp in ("E", "EI", "EI_ident") else strict[i]
        if not q:
            out.append(f"\x00none#{i}")
            continue
        if mp in ("ref", "E", "ref_tie_last"):
            out.append(q)
            continue
        g = sigs.get(FR.qkey(db, q)) or {}
        if g.get("status") == "ok":
            out.append("\x00sig:" + g["sig"])
        elif mp == "EI_ident":
            out.append("\x00FAIL")
        else:
            out.append(f"\x00fail#{i}")
    return out


def reads(labels: List[str], rd: Dict[str, Any], responses: List[str], arm: str, tie: str,
          top_k: int) -> Dict[str, Any]:
    """Everything BCCS reads from the labels after this round."""
    probs = rd["probs"]
    cons = consensus_with_tie(labels, probs, tie)
    out = {"stop": cons["verdict"] == "full consensus", "verdict": cons["verdict"]}
    if out["stop"]:
        return out
    T = 1 if cons["verdict"] == "no consensus" else 0
    conflict, _ = BC["estimate_conflict"](rd["clusters"], probs, labels)
    uncs = BC["estimate_uncertainty"](rd["clusters"], probs)
    shown = []
    for i in range(len(responses)):
        _p, sh = BC["construct_message"](i, responses, "", probs, rd["clusters"], conflict, uncs, T, top_k)
        shown.append(tuple((x["agent"], x["role"], x["cluster"]) for x in sh))
    out.update(T=T, conflict=conflict, shown=shown)
    # In NOPEER the arm transform deletes every peer block and maps both tails to one clause, so
    # neither T nor the shown peers reach the next prompt (arm_transform.py, NOPEER branch).
    out["prompt_key"] = None if arm == "NOPEER" else (T, tuple(shown))
    return out


# ------------------------------------------------------------------ loading
def load(ext: pathlib.Path, backbones: Optional[Sequence[str]], per_unit_p: Optional[pathlib.Path]):
    last, per_unit, counts = FR.load_runs(ext, per_unit_p, ("bccs",))
    runs = []
    for k, r in last.items():
        unit, proto, arm, bb, rep = k
        if backbones and bb not in backbones:
            continue
        d = r.get("record") or {}
        db = ((r.get("config") or {}).get("unit_meta") or {}).get("db_id")
        calls = d.get("calls") or []
        byr = collections.defaultdict(list)
        for c in calls:
            byr[int(c.get("round") or 0)].append(c)
        rounds = []
        for rd in d.get("rounds_detail") or []:
            cs = sorted(byr[rd["round"]], key=lambda c: int(c.get("agent_index") or 0))
            strict = [FR.sql_of(c.get("extracted_answer")) for c in cs]
            len_ = [s if s else FR.lenient(c.get("raw_reply")) for s, c in zip(strict, cs)]
            belief = d.get("belief_used", "intended")
            probs = rd["beliefs_intended"] if belief == "intended" else rd["beliefs_released"]
            nxt = sorted(byr.get(rd["round"] + 1, []), key=lambda c: int(c.get("agent_index") or 0))
            rounds.append(dict(round=rd["round"], support=list(rd["support"]), strict=strict, lenient=len_,
                               probs=probs, clusters=rd["clusters"], logged_verdict=rd["consensus"]["verdict"],
                               logged_T=rd.get("T"), logged_conflict=rd.get("conflict"),
                               responses=[c.get("raw_reply") or "" for c in cs],
                               next_shown=[tuple((x["agent"], x["role"], x["cluster"]) for x in (c.get("peers_shown") or []))
                                           for c in nxt] if nxt else None))
        fin = None if per_unit is None else per_unit[k]["outcome"]
        runs.append(dict(unit=unit, arm=arm, backbone=bb, repeat=rep, db=db, rounds=rounds,
                         rounds_cfg=d.get("rounds_configured", 3), top_k=d.get("top_k", 2),
                         stopped_at=d.get("stopped_at_round"), final_outcome=fin))
    return runs, per_unit, counts


# ------------------------------------------------------------------ divergence
def walk(run, mp, sigs, tie="first"):
    """First divergence of map `mp` against the logged protocol reads. Returns
    (kind, round) with kind in none | recoverable | unrecoverable, and a sanity flag."""
    rs = run["rounds"]
    L = len(rs) - 1
    for r in range(len(rs)):
        if r >= run["rounds_cfg"] - 1:
            break                                   # the last configured round has no downstream read
        rd = rs[r]
        logged = reads(labels_for("proto", rd, run["db"], sigs), rd, rd["responses"], run["arm"], "first", run["top_k"])
        alt = reads(labels_for(mp, rd, run["db"], sigs), rd, rd["responses"], run["arm"], tie, run["top_k"])
        logged_stopped = (r == L) and run["stopped_at"] == r
        if logged["stop"] != logged_stopped:
            return "sanity_fail", r
        if logged["stop"] and alt["stop"]:
            return "none", None
        if alt["stop"] and not logged["stop"]:
            REASON[(run["backbone"], run["arm"], mp)]["o_prime_stops_earlier"] += 1
            return "recoverable", r
        if logged["stop"] and not alt["stop"]:
            REASON[(run["backbone"], run["arm"], mp)][
                "logged_stop_on_None" if labels_for("proto", rd, run["db"], sigs).count("") * 3 > 2 * len(rd["support"])
                else "logged_stop_o_prime_continues"] += 1
            return "unrecoverable", r
        if logged["prompt_key"] != alt["prompt_key"]:
            REASON[(run["backbone"], run["arm"], mp)][
                "T" if logged["T"] != alt["T"] else "shown_peers"] += 1
            return "unrecoverable", r
    return "none", None


def sanity(run, sigs) -> Dict[str, int]:
    """The recomputed protocol reads must equal the logged verdict, T, conflict and peers shown."""
    c = collections.Counter()
    for r, rd in enumerate(run["rounds"]):
        cons = BC["estimate_consensus"](rd["support"], rd["probs"])
        c["verdict_ok" if cons["verdict"] == rd["logged_verdict"] else "verdict_bad"] += 1
        if cons["verdict"] == "full consensus" or rd["logged_T"] is None:
            continue
        x = reads(rd["support"], rd, rd["responses"], "FULL", "first", run["top_k"])
        c["T_ok" if x["T"] == rd["logged_T"] else "T_bad"] += 1
        if rd["logged_conflict"] is not None:
            same = np.allclose(np.array(x["conflict"], float), np.array(rd["logged_conflict"], float), atol=1e-9)
            c["conflict_ok" if same else "conflict_bad"] += 1
        if rd["next_shown"] is not None:
            c["shown_ok" if [tuple(s) for s in x["shown"]] == rd["next_shown"] else "shown_bad"] += 1
    return c


# ------------------------------------------------------------------ scoring of readouts
def round_score(run, r, ext_, ident, sigs, verdicts, tie="first"):
    """Factorial readout semantics (FR.score_run) at an arbitrary logged round r."""
    rd = run["rounds"][r]
    qs = rd["lenient"] if ext_ == "lenient" else rd["strict"]
    db = run["db"]
    lab = []
    for q in qs:
        if not q:
            lab.append(None)
        elif ident == "string":
            lab.append(q)
        else:
            s = sigs.get(FR.qkey(db, q)) or {}
            lab.append(s["sig"] if s.get("status") == "ok" else ("FAIL" if ident == "exec_ident" else None))
    key = f"{run['unit']}|bccs|{run['arm']}|{run['backbone']}|{run['repeat']}|r{r}|{ext_}|{ident}"
    w, _t = FR.vote(lab, tie, key)
    if w is None or w == "FAIL":
        return 0.0
    q = qs[lab.index(w)]
    v = verdicts.get(FR.qkey(db, q))
    if v is None:
        raise KeyError(("missing verdict", db, q[:60]))
    return float(bool(v.get("correct")))


READOUT_OF_MAP = {"ref": ("strict", "string", "first"), "I": ("strict", "exec", "first"),
                  "E": ("lenient", "string", "first"), "EI": ("lenient", "exec", "first"),
                  "EI_ident": ("lenient", "exec_ident", "first"), "ref_tie_last": ("strict", "string", "last")}


def cp(k: int, n: int, a: float = 0.05) -> Tuple[float, float]:
    from scipy.stats import beta  # noqa: PLC0415
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default=str(FR.RUN_DIR), help="run directory (default %(default)s)")
    ap.add_argument("--per-unit", default=None)
    ap.add_argument("--verdicts", default=None)
    ap.add_argument("--scores", default=str(REPLAY_OUT / "per_run_scores.jsonl"),
                    help="per-run scores of share_eval/replay (read-only); empty string to recompute every readout here")
    ap.add_argument("--backbones", default="")
    ap.add_argument("--out", default=str(REPO / "output/share_eval/replay_validity"))
    ap.add_argument("--exclude-units", default="bird|518,bird|701")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extra-caches", default="", help="comma-separated read-only signature/verdict jsonl caches")
    ap.add_argument("--sig-caches", default=",".join(str(FR.CACHE / n) for n in (
                        "r0_exec_signatures.jsonl", "replay_exec_signature_cache.jsonl", "exec_fresh.jsonl",
                        "divergence_exec_signature_cache.jsonl")),
                    help="read-only execution-signature caches, comma-separated (first entry of a query wins)")
    ap.add_argument("--verdict-caches", default=",".join(str(FR.CACHE / n) for n in (
                        "replay_verdict_cache.jsonl", "divergence_verdict_cache.jsonl")),
                    help="read-only verdict caches, comma-separated")
    a = ap.parse_args()
    ext = pathlib.Path(a.ext).resolve()
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    bbs = [b for b in a.backbones.split(",") if b] or None
    per_unit_p = pathlib.Path(a.per_unit) if a.per_unit else ext / "analysis/sql_per_unit.jsonl"
    runs, per_unit, load_counts = load(ext, bbs, per_unit_p if per_unit_p.exists() else None)
    excl = {u for u in a.exclude_units.split(",") if u}
    n_excl = sum(1 for r in runs if r["unit"] in excl)
    runs = [r for r in runs if r["unit"] not in excl and r["db"]]
    print(f"{len(runs)} BCCS runs ({n_excl} dropped by the a-priori task exclusion)", flush=True)

    # caches: read-only seeds + our own
    seed_sig = [pathlib.Path(p) for p in a.sig_caches.split(",") if p]
    extra = [pathlib.Path(p) for p in a.extra_caches.split(",") if p]
    seed_sig += [p for p in extra if "signature" in p.name]
    own_s, own_v = out / "exec_signature_cache.jsonl", out / "verdict_cache.jsonl"
    sigs = {}
    for p in seed_sig + [own_s]:
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    d = json.loads(l)
                    if d.get("status") == "ok" and "sig" not in d:
                        continue                    # exec_fresh rows without a signature are not usable as identity
                    sigs.setdefault(d["k"], d)
    verd_p = pathlib.Path(a.verdicts) if a.verdicts else ext / "analysis/sql_verdicts.json"
    verdicts = json.loads(verd_p.read_text())["verdicts"] if verd_p.exists() else {}
    for p in [pathlib.Path(x) for x in a.verdict_caches.split(",") if x] + [p for p in extra if "verdict" in p.name] + [own_v]:
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    d = json.loads(l)
                    verdicts[d["k"]] = d
    # same rescoring of the suspect queries as the replay
    import sql_common as SQ  # noqa: PLC0415
    gold = {f"bird|{q['question_id']}": q["SQL"] for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}
    todo, gtodo = {}, {}
    for r in runs:
        for rd in r["rounds"]:
            for q in set(rd["strict"]) | set(rd["lenient"]):
                if q:
                    k = FR.qkey(r["db"], q)
                    if k not in sigs:
                        todo[k] = (r["db"], q)
                    if k not in verdicts:
                        gtodo[k] = (r["db"], q, gold[r["unit"]])
    print(f"to sign {len(todo)}; to judge {len(gtodo)}", flush=True)
    t0 = time.time()
    if todo or gtodo:
        with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
            with open(own_s, "a") as fh:
                for kk, v in ex.map(FR._sig, list(todo.values()), chunksize=16):
                    v["k"] = kk
                    sigs[kk] = v
                    fh.write(json.dumps(v) + "\n")
            with open(own_v, "a") as fh:
                for kk, v in ex.map(FR._verdict, list(gtodo.values()), chunksize=16):
                    verdicts[kk] = v
                    fh.write(json.dumps(dict(v, k=kk)) + "\n")
    exec_cost = dict(new_signatures=len(todo), new_verdicts=len(gtodo), wall_s=round(time.time() - t0, 1))

    # ---- sanity: the protocol's own reads are reproduced
    san = collections.Counter()
    for r in runs:
        san.update(sanity(r, sigs))

    # ---- replay scores per run (factorial values where available)
    fac = {}
    if a.scores and pathlib.Path(a.scores).exists():
        for l in pathlib.Path(a.scores).read_text().splitlines():
            d = json.loads(l)
            if d["protocol"] == "bccs":
                fac[(d["unit"], d["arm"], d["backbone"], int(d["repeat"]))] = d
    mismatch = collections.Counter()
    per_run = []
    for r in runs:
        k = (r["unit"], r["arm"], r["backbone"], int(r["repeat"]))
        L = len(r["rounds"]) - 1
        row = dict(unit=r["unit"], arm=r["arm"], backbone=r["backbone"], repeat=r["repeat"], last_round=L,
                   final=None if r["final_outcome"] is None else float(r["final_outcome"] == "match"))
        for mp, (e, i, tie) in READOUT_OF_MAP.items():
            R_last = round_score(r, L, e, i, sigs, verdicts, tie)
            R_r0 = round_score(r, 0, e, i, sigs, verdicts, tie)
            if k in fac:
                for nm, val, rd_ in (("last", R_last, "last"), ("r0", R_r0, "r0")):
                    if fac[k][f"{e}|{i}|{rd_}|{tie}"] != val:
                        mismatch[f"{mp}|{nm}"] += 1
            kind, at = walk(r, mp, sigs, "last" if mp == "ref_tie_last" else "first")
            if kind == "none":
                lo = hi = R_last
            elif kind == "recoverable":
                lo = hi = round_score(r, at, e, i, sigs, verdicts, tie)
            elif kind == "unrecoverable":
                lo, hi = 0.0, 1.0
            else:
                lo, hi = R_last, R_last
            row[mp] = dict(D=kind, at=at, R_last=R_last, R_r0=R_r0, O_lo=lo, O_hi=hi)
        per_run.append(row)
    with open(out / "bccs_divergence_runs.jsonl", "w") as fh:
        for x in per_run:
            fh.write(json.dumps(x) + "\n")

    # ---- summaries
    res = dict(bccs_source_sha256=BCCS_SHA, ext=str(ext.relative_to(REPO)) if ext.is_relative_to(REPO) else str(ext),
               runs=len(runs), runs_dropped_a_priori=n_excl, excluded_units=sorted(excl), load_counts=load_counts,
               exec_cost=exec_cost, sanity=dict(san), factorial_score_mismatches=dict(mismatch), cells={})
    for bb in sorted({x["backbone"] for x in per_run}):
        for arm in ARMS:
            rows = [x for x in per_run if x["backbone"] == bb and x["arm"] == arm]
            if not rows:
                continue
            cell = dict(runs=len(rows), tasks=len({x["unit"] for x in rows}),
                        stopped_round0=sum(1 for x in rows if x["last_round"] == 0))
            for mp in READOUT_OF_MAP:
                kinds = collections.Counter(x[mp]["D"] for x in rows)
                byu = collections.defaultdict(list)
                for x in rows:
                    byu[x["unit"]].append(x[mp]["D"])
                any_d = sum(1 for v in byu.values() if any(k != "none" for k in v))
                any_u = sum(1 for v in byu.values() if any(k == "unrecoverable" for k in v))
                n = len(rows)
                d_all = n - kinds["none"]
                cell[mp] = dict(
                    kinds=dict(kinds),
                    P_D=round(d_all / n, 4), P_D_unrecoverable=round(kinds["unrecoverable"] / n, 4),
                    tasks_any_D=any_d, tasks_any_unrecoverable=any_u, n_tasks=len(byu),
                    CP95_task_any_D=[round(v, 4) for v in cp(any_d, len(byu))],
                    CP95_task_any_unrecoverable=[round(v, 4) for v in cp(any_u, len(byu))],
                    by_round=dict(collections.Counter(f"{x[mp]['D']}@r{x[mp]['at']}" for x in rows if x[mp]["D"] != "none")))
            for mp in READOUT_OF_MAP:
                cell[mp]["divergence_reason"] = dict(REASON[(bb, arm, mp)])
            res["cells"][f"{bb}|{arm}"] = cell
    (out / "bccs_divergence.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("runs", "sanity", "factorial_score_mismatches", "exec_cost")}, indent=1))
    for c, v in res["cells"].items():
        print(c, {mp: (v[mp]["P_D"], v[mp]["P_D_unrecoverable"]) for mp in READOUT_OF_MAP})
    print(f"done in {time.time() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
