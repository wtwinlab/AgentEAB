r"""First divergence of BCCS's own reads under another reading of its answers (no model call).

    python3 -B -m share_eval_suite.steps.bccs_divergence      # Llama-3-8B and Qwen2.5-7B
    python3 -B -m share_eval_suite.steps.bccs_divergence --ext data/records/bird_debates_backbones \
            --scores "" --out output/share_eval_suite/replay_validity_backbones --backbones qwen25-14b-awq,...

Same outputs as `share_eval/replay_validity/bccs_divergence.py`. Unlike the script, the step executes
no query: when some query has no stored execution result or verdict, it reports how many and stops
(exit status 2).

After every non-final round BCCS reads its agents' answer labels three times: the stop test (full
consensus), the no-consensus flag T, and, with peers, the conflict matrix that chooses which peers
each agent sees next. A replay that reads the answers differently (another extractor, an execution
vote, another tie rule; a reading o') equals deployment only on runs where every such read is
unchanged. For every BCCS run and every reading o' the step

  * recomputes the reads on the logged rounds with the protocol's labels, which must reproduce the
    logged verdict, T, conflict matrix and shown peers (the read check);
  * recomputes the reads under o' and finds the first round where they differ (the first divergence
    D): none; recoverable (o' stops earlier, so its answer is computable from the logged rounds); or
    unrecoverable (o' continues past a logged stop, or shows another flag or other peers);
  * gives the run's deployment interval [O_lo, O_hi] for o' (Lemma 1): the last-round replay score
    when D is none, the score at the stopping round when D is recoverable, [0, 1] when unrecoverable;
    a run whose recomputed logged stop differs from the log (`sanity_fail`) keeps its last-round score.

Per backbone and condition it reports the share of runs with a divergence and, over questions, the
share with any divergence and with any unrecoverable divergence, with Clopper–Pearson 95% intervals.
BCCS's stop test, conflict estimate and peer choice are the protocol's own functions, loaded through
the protocol registry.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from share_eval_suite import paths, stats
from share_eval_suite.records import readers
from share_eval_suite.steps import _validity_common as C

ARMS = ("NOPEER", "FULL")
# Reading o' of each label map: extractor, vote, tie rule.
READOUT_OF_MAP = {"ref": ("strict", "string", "first"), "I": ("strict", "exec", "first"),
                  "E": ("lenient", "string", "first"), "EI": ("lenient", "exec", "first"),
                  "EI_ident": ("lenient", "exec_ident", "first"), "ref_tie_last": ("strict", "string", "last")}
SIG_CACHES = ("r0_exec_signatures.jsonl", "replay_exec_signature_cache.jsonl", "exec_fresh.jsonl",
              "divergence_exec_signature_cache.jsonl")
VERDICT_CACHES = ("replay_verdict_cache.jsonl", "divergence_verdict_cache.jsonl")
NEEDED = ("list_mean", "calculate_entropy", "estimate_uncertainty", "_strip_period",
          "estimate_conflict", "estimate_consensus", "find_top_indices", "construct_message")


def bccs_functions() -> Tuple[Dict[str, Any], str]:
    """BCCS's decision functions from its adapter, and the SHA-256 of the adapter source."""
    from share_eval_suite.adapters import protocols                      # noqa: PLC0415
    mod = protocols.load("BCCS")
    missing = [n for n in NEEDED if not hasattr(mod, n)]
    if missing:
        raise RuntimeError(f"BCCS adapter lacks {missing}")
    src = pathlib.Path(mod.__file__).read_text()
    return {n: getattr(mod, n) for n in NEEDED}, hashlib.sha256(src.encode()).hexdigest()


class Reads:
    """BCCS's reads of one round's labels, with its own functions `bc`."""

    def __init__(self, bc: Dict[str, Any]):
        self.bc = bc

    def consensus(self, support: Sequence[str], probs: Sequence[float], tie: str) -> Dict[str, Any]:
        """The consensus verdict; among tied modal labels the first (as deployed) or the last."""
        bc = self.bc
        if tie == "first":
            return bc["estimate_consensus"](support, probs)
        srs = [bc["_strip_period"](s if s != "" else "None") for s in support]
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

    def __call__(self, labels: List[str], rd: Dict[str, Any], responses: List[str], arm: str, tie: str,
                 top_k: int) -> Dict[str, Any]:
        """Stop decision, flag T, conflict matrix and shown peers after one round. Without peers
        neither T nor the shown peers reach the next prompt, so the prompt key is None."""
        bc = self.bc
        probs = rd["probs"]
        cons = self.consensus(labels, probs, tie)
        out = {"stop": cons["verdict"] == "full consensus", "verdict": cons["verdict"]}
        if out["stop"]:
            return out
        T = 1 if cons["verdict"] == "no consensus" else 0
        conflict, _ = bc["estimate_conflict"](rd["clusters"], probs, labels)
        uncs = bc["estimate_uncertainty"](rd["clusters"], probs)
        shown = []
        for i in range(len(responses)):
            _p, sh = bc["construct_message"](i, responses, "", probs, rd["clusters"], conflict, uncs, T, top_k)
            shown.append(tuple((x["agent"], x["role"], x["cluster"]) for x in sh))
        out.update(T=T, conflict=conflict, shown=shown)
        out["prompt_key"] = None if arm == "NOPEER" else (T, tuple(shown))
        return out


# ------------------------------------------------------------------ labels under a reading
def labels_for(mp: str, run_round: Dict[str, Any], db: str, sigs: Dict[str, Any]) -> List[str]:
    """The labels BCCS would read in one round under reading `mp`. A missing answer or a failed
    execution gets a label of its own, which matches no other label (except `EI_ident`, where every
    failed execution shares one FAIL label)."""
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
        g = sigs.get(readers.query_key(db, q)) or {}
        if g.get("status") == "ok":
            out.append("\x00sig:" + g["sig"])
        elif mp == "EI_ident":
            out.append("\x00FAIL")
        else:
            out.append(f"\x00fail#{i}")
    return out


# ------------------------------------------------------------------ loading
def load(ext: pathlib.Path, backbones: Optional[Sequence[str]], per_unit_p: Optional[pathlib.Path]):
    """The logged BCCS runs with, per round, the logged labels, both extractors' queries, beliefs,
    clusters, logged reads and the peers shown in the next round."""
    last, per_unit, counts = readers._load_runs(ext, per_unit_p, ("bccs",))
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
            strict = [readers._sql_of(c.get("extracted_answer")) for c in cs]
            len_ = [s if s else readers._lenient(c.get("raw_reply")) for s, c in zip(strict, cs)]
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
def walk(reads: Reads, run, mp, sigs, reason, tie="first"):
    """First divergence of reading `mp` from the logged reads: (kind, round), kind none, recoverable,
    unrecoverable, or sanity_fail when the recomputed logged stop differs from the logged one. The
    reason of each divergence is counted in `reason`."""
    rs = run["rounds"]
    L = len(rs) - 1
    for r in range(len(rs)):
        if r >= run["rounds_cfg"] - 1:
            break                                   # the last configured round has no later read
        rd = rs[r]
        logged = reads(labels_for("proto", rd, run["db"], sigs), rd, rd["responses"], run["arm"], "first", run["top_k"])
        alt = reads(labels_for(mp, rd, run["db"], sigs), rd, rd["responses"], run["arm"], tie, run["top_k"])
        logged_stopped = (r == L) and run["stopped_at"] == r
        if logged["stop"] != logged_stopped:
            return "sanity_fail", r
        if logged["stop"] and alt["stop"]:
            return "none", None
        if alt["stop"] and not logged["stop"]:
            reason[(run["backbone"], run["arm"], mp)]["o_prime_stops_earlier"] += 1
            return "recoverable", r
        if logged["stop"] and not alt["stop"]:
            reason[(run["backbone"], run["arm"], mp)][
                "logged_stop_on_None" if labels_for("proto", rd, run["db"], sigs).count("") * 3 > 2 * len(rd["support"])
                else "logged_stop_o_prime_continues"] += 1
            return "unrecoverable", r
        if logged["prompt_key"] != alt["prompt_key"]:
            reason[(run["backbone"], run["arm"], mp)][
                "T" if logged["T"] != alt["T"] else "shown_peers"] += 1
            return "unrecoverable", r
    return "none", None


def read_check(reads: Reads, run) -> Dict[str, int]:
    """Counts of rounds where the recomputed verdict, T, conflict matrix and shown peers equal the
    logged ones (`*_ok`) or differ (`*_bad`)."""
    c = collections.Counter()
    for r, rd in enumerate(run["rounds"]):
        cons = reads.bc["estimate_consensus"](rd["support"], rd["probs"])
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


# ------------------------------------------------------------------ replay scores
def round_score(run, r, ext_, ident, sigs, verdicts, tie="first"):
    """Correctness of the vote over the answers of logged round r under extractor `ext_`, vote `ident`
    and tie rule `tie`, with the random-tie key of the scored per-run table."""
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
            s = sigs.get(readers.query_key(db, q)) or {}
            lab.append(s["sig"] if s.get("status") == "ok" else ("FAIL" if ident == "exec_ident" else None))
    key = f"{run['unit']}|bccs|{run['arm']}|{run['backbone']}|{run['repeat']}|r{r}|{ext_}|{ident}"
    w, _t = readers._vote(lab, tie, key)
    if w is None or w == "FAIL":
        return 0.0
    q = qs[lab.index(w)]
    v = verdicts.get(readers.query_key(db, q))
    if v is None:
        raise KeyError(("missing verdict", db, q[:60]))
    return float(bool(v.get("correct")))


def _cp(k: int, n: int) -> Tuple[float, float]:
    return stats.clopper_pearson(k, n, 0.05)


def run(ext, out, per_unit=None, verdicts_path=None, scores=None, backbones="",
        exclude_units="bird|518,bird|701", sig_caches=None, verdict_caches=None, extra_caches="",
        workers: int = 8, log=print) -> Dict[str, Any]:
    """Write `bccs_divergence_runs.jsonl` and `bccs_divergence.json` to `out`; return the summary.
    `workers` is accepted for the command line and unused (no query is executed)."""
    ext = pathlib.Path(ext).resolve()
    out = pathlib.Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    bc, bccs_sha = bccs_functions()
    reads = Reads(bc)
    reason = collections.defaultdict(collections.Counter)
    bbs = [b for b in backbones.split(",") if b] or None
    per_unit_p = pathlib.Path(per_unit) if per_unit else ext / "analysis/sql_per_unit.jsonl"
    runs, _, load_counts = load(ext, bbs, per_unit_p if per_unit_p.exists() else None)
    excl = {u for u in exclude_units.split(",") if u}
    n_excl = sum(1 for r in runs if r["unit"] in excl)
    runs = [r for r in runs if r["unit"] not in excl and r["db"]]
    log(f"{len(runs)} BCCS runs ({n_excl} dropped by the a-priori task exclusion)")

    # stored execution signatures (the first stored row of a query counts) and verdicts (later rows win)
    if sig_caches is None:
        sig_caches = ",".join(str(C.CACHE / n) for n in SIG_CACHES)
    if verdict_caches is None:
        verdict_caches = ",".join(str(C.CACHE / n) for n in VERDICT_CACHES)
    seed_sig = C.cache_paths(sig_caches)
    extra = C.cache_paths(extra_caches)
    seed_sig += [p for p in extra if "signature" in p.name]
    own_s, own_v = out / "exec_signature_cache.jsonl", out / "verdict_cache.jsonl"
    sigs: Dict[str, Any] = {}
    for p in seed_sig + [own_s]:
        for d in C.read_cache_lines(p):
            if d.get("status") == "ok" and "sig" not in d:
                continue                            # a result without a signature identifies nothing
            sigs.setdefault(d["k"], d)
    verd_p = pathlib.Path(verdicts_path) if verdicts_path else ext / "analysis/sql_verdicts.json"
    verdicts = json.loads(verd_p.read_text())["verdicts"] if verd_p.exists() else {}
    for p in C.cache_paths(verdict_caches) + [p for p in extra if "verdict" in p.name] + [own_v]:
        for d in C.read_cache_lines(p):
            verdicts[d["k"]] = d
    gold = C.bird_reference_queries()
    todo, gtodo = {}, {}
    for r in runs:
        for rd in r["rounds"]:
            for q in set(rd["strict"]) | set(rd["lenient"]):
                if q:
                    k = readers.query_key(r["db"], q)
                    if k not in sigs:
                        todo[k] = (r["db"], q)
                    if k not in verdicts:
                        gtodo[k] = (r["db"], q, gold[r["unit"]])
    log(f"to sign {len(todo)}; to judge {len(gtodo)}")
    t0 = time.time()
    C.require_cached(todo, gtodo)
    exec_cost = dict(new_signatures=len(todo), new_verdicts=len(gtodo), wall_s=round(time.time() - t0, 1))

    san = collections.Counter()
    for r in runs:
        san.update(read_check(reads, r))

    fac = {}
    if scores and pathlib.Path(scores).exists():
        for line in pathlib.Path(scores).read_text().splitlines():
            d = json.loads(line)
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
            kind, at = walk(reads, r, mp, sigs, reason, "last" if mp == "ref_tie_last" else "first")
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

    repo = paths.REPO
    res = dict(bccs_source_sha256=bccs_sha, ext=str(ext.relative_to(repo)) if ext.is_relative_to(repo) else str(ext),
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
                    CP95_task_any_D=[round(v, 4) for v in _cp(any_d, len(byu))],
                    CP95_task_any_unrecoverable=[round(v, 4) for v in _cp(any_u, len(byu))],
                    by_round=dict(collections.Counter(f"{x[mp]['D']}@r{x[mp]['at']}" for x in rows if x[mp]["D"] != "none")))
            for mp in READOUT_OF_MAP:
                cell[mp]["divergence_reason"] = dict(reason[(bb, arm, mp)])
            res["cells"][f"{bb}|{arm}"] = cell
    (out / "bccs_divergence.json").write_text(json.dumps(res, indent=1))
    log(json.dumps({k: res[k] for k in ("runs", "sanity", "factorial_score_mismatches", "exec_cost")}, indent=1))
    for c, v in res["cells"].items():
        log(c, {mp: (v[mp]["P_D"], v[mp]["P_D_unrecoverable"]) for mp in READOUT_OF_MAP})
    log(f"done in {time.time() - t_start:.0f} s")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="first divergence of BCCS's reads under every alternative reading")
    ap.add_argument("--ext", default=str(readers.DEBATES), help="run directory (default %(default)s)")
    ap.add_argument("--per-unit", default=None)
    ap.add_argument("--verdicts", default=None)
    ap.add_argument("--scores", default=str(C.OUTPUT / "replay" / "per_run_scores.jsonl"),
                    help="per-run scores of the replay (read-only); empty string to recompute every readout here")
    ap.add_argument("--backbones", default="")
    ap.add_argument("--out", default=str(C.OUTPUT / "replay_validity"))
    ap.add_argument("--exclude-units", default="bird|518,bird|701")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extra-caches", default="", help="comma-separated read-only signature/verdict jsonl caches")
    ap.add_argument("--sig-caches", default=",".join(str(C.CACHE / n) for n in SIG_CACHES),
                    help="read-only execution-signature caches, comma-separated (first entry of a query wins)")
    ap.add_argument("--verdict-caches", default=",".join(str(C.CACHE / n) for n in VERDICT_CACHES),
                    help="read-only verdict caches, comma-separated")
    a = ap.parse_args(argv)
    try:
        run(a.ext, a.out, per_unit=a.per_unit, verdicts_path=a.verdicts, scores=a.scores, backbones=a.backbones,
            exclude_units=a.exclude_units, sig_caches=a.sig_caches, verdict_caches=a.verdict_caches,
            extra_caches=a.extra_caches, workers=a.workers,
            log=lambda *x: print(*x, flush=True))
    except C.ExecutionNeeded as e:
        print(f"bccs_divergence: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
