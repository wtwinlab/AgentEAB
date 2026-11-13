r"""Online check of the BCCS replay, in distribution (no model call).

    python3 -B -m share_eval_suite.steps.online_validation [--ext data/records/bird_debates_backbones] [--logged-ext DIR] [--div DIV] [--out FILE]

Same outputs as `share_eval/replay_validity/online_validation.py`. Unlike the script, the step
executes no query: when some query has no stored execution result or verdict, it reports how many and
stops (exit status 2).

The online runs (Qwen2.5-7B, backbone label `qwen2.5-7b-execid`) run BCCS with an execution vote
inside the protocol. They share no per-request seed with the logged runs and use another serving
image, so the comparison is in distribution only, never run by run:

  * reading: lenient extractor and execution vote with failed executions dropped, first tie, on the
    last round of each online run; the logged side is the replay score of the same reading
    (`EI`, last round) in the divergence table;
  * per question: mean over the online repeats minus mean over the logged repeats, with a 95%
    percentile bootstrap over questions (10,000 draws, seed 20261106);
  * strata by the logged question's first divergence: none in every repeat, recoverable only, any
    unrecoverable; they condition on the logged runs and are descriptive;
  * coverage: whether the online mean lies inside the mean of the logged deployment intervals;
  * descriptive: the share of identical initial answers (raw text and protocol-extracted query)
    between online and logged runs of the same question, agent and repeat, and the online divergence
    share read from `execid_trace` where its fields can be identified.

When no online record exists it writes a status file.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np

from share_eval_suite import paths, stats
from share_eval_suite.records import readers
from share_eval_suite.steps import _validity_common as C

LOGGED_BB = "qwen2.5-7b"
EXCLUDE = {"bird|518", "bird|701"}
SIG_CACHES = ("r0_exec_signatures.jsonl", "replay_exec_signature_cache.jsonl", "divergence_exec_signature_cache.jsonl")
VERDICT_CACHES = ("replay_verdict_cache.jsonl", "divergence_verdict_cache.jsonl")


def find_online(ext: pathlib.Path, selftest: bool = False) -> List[Dict[str, Any]]:
    """Completed online runs (protocol or backbone labelled `execid`; with `selftest`, the logged
    Qwen2.5-7B BCCS runs instead), the last completed record of each (question, condition, repeat)."""
    recs = []
    for p in sorted((ext / "raw").glob("sql_bird_v2__*.jsonl")):
        if "execid" not in p.name and "bccs" not in p.name:
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") != "ok":
                continue
            proto, bb = str(r.get("protocol")), str(r.get("backbone"))
            if "execid" in proto or "execid" in bb or (selftest and proto == "bccs" and bb == LOGGED_BB):
                recs.append(r)
    last = {}
    for r in recs:
        last[(r.get("unit"), r.get("arm"), int(r.get("repeat") or 0))] = r
    return list(last.values())


def calls_by_round(rec):
    """Calls of a run per round, in agent order."""
    by = collections.defaultdict(list)
    for c in (rec.get("record") or {}).get("calls") or []:
        by[int(c.get("round") or 0)].append(c)
    return {k: sorted(v, key=lambda c: int(c.get("agent_index") or 0)) for k, v in by.items()}


def trace_divergence(rec):
    """True when some round's verdict under the execution vote differs from the verdict under the
    exact-string vote in `execid_trace`; None when the fields cannot be identified."""
    tr = (rec.get("record") or {}).get("execid_trace") or rec.get("execid_trace")
    if not isinstance(tr, list) or not tr:
        return None
    seen = False
    for row in tr:
        if not isinstance(row, dict):
            return None
        sv = next((row[k] for k in row if "string" in k.lower() and "verdict" in k.lower()), None)
        ev = next((row[k] for k in row if "exec" in k.lower() and "verdict" in k.lower() and "string" not in k.lower()), None)
        if isinstance(sv, dict):
            sv = sv.get("verdict")
        if isinstance(ev, dict):
            ev = ev.get("verdict")
        if sv is None or ev is None:
            continue
        seen = True
        if sv != ev:
            return True
    return False if seen else None


def summ(x: np.ndarray, rng_seed: int = stats.SEED_REPLAY) -> Dict[str, Any]:
    """Mean of per-question differences in pp with its 95% percentile bootstrap interval."""
    if len(x) == 0:
        return {"n_tasks": 0}
    idx = stats.resample_index(len(x), stats.DRAWS, rng_seed)
    b = x[idx].mean(axis=1)
    return {"n_tasks": int(len(x)), "est_pp": round(100 * float(x.mean()), 2),
            "ci95_pp": [round(100 * float(np.percentile(b, 2.5)), 2), round(100 * float(np.percentile(b, 97.5)), 2)]}


def run(ext, div, logged_ext=None, sig_caches=None, verdict_caches=None, out=None, workers: int = 8,
        selftest: bool = False, log=print) -> Dict[str, Any]:
    """Write the online check to `out` and return it. `workers` is accepted for the command line and
    unused (no query is executed)."""
    logged_ext = pathlib.Path(logged_ext or readers.DEBATES)
    out = pathlib.Path(out or C.OUTPUT / "replay_validity" / "online_validation.json")
    if sig_caches is None:
        sig_caches = ",".join(str(C.CACHE / n) for n in SIG_CACHES)
    if verdict_caches is None:
        verdict_caches = ",".join(str(C.CACHE / n) for n in VERDICT_CACHES)
    ext = pathlib.Path(ext)
    online = [r for r in find_online(ext, selftest) if r.get("unit") not in EXCLUDE]
    if not online:
        res = {"status": "no online bccs_execid records found", "ext": str(ext)}
        out.write_text(json.dumps(res, indent=1))
        log("no online records yet; status written")
        return res

    logged = collections.defaultdict(list)
    for line in pathlib.Path(div).read_text().splitlines():
        d = json.loads(line)
        if d["backbone"] == LOGGED_BB:
            logged[(d["unit"], d["arm"])].append(d)

    gold = C.bird_reference_queries()
    odir = out.parent
    odir.mkdir(parents=True, exist_ok=True)
    sig_p, ver_p = odir / "online_exec_signature_cache.jsonl", odir / "online_verdict_cache.jsonl"
    sigs = C.load_cache(C.cache_paths(sig_caches) + [sig_p])
    verdicts = C.load_cache(C.cache_paths(verdict_caches) + [ver_p])
    vstore = logged_ext / "analysis/sql_verdicts.json"
    if vstore.exists():
        for k, v in json.loads(vstore.read_text())["verdicts"].items():
            verdicts.setdefault(k, v)
    rows = []
    todo, gtodo = {}, {}
    for r in online:
        db = ((r.get("config") or {}).get("unit_meta") or {}).get("db_id")
        cbr = calls_by_round(r)
        if not cbr or not db:
            continue
        L = max(cbr)
        strict = [readers._sql_of(c.get("extracted_answer")) for c in cbr[L]]
        qs = [s if s else readers._lenient(c.get("raw_reply")) for s, c in zip(strict, cbr[L])]
        r0 = [(c.get("raw_reply") or "", readers._sql_of(c.get("extracted_answer"))) for c in cbr.get(0, [])]
        rows.append(dict(unit=r["unit"], arm=r["arm"], repeat=int(r.get("repeat") or 0), db=db, qs=qs, last_round=L,
                         r0=r0, trace_div=trace_divergence(r), image=(r.get("record") or {}).get("calls", [{}])[0].get("fingerprint")))
        for q in qs:
            if q:
                k = readers.query_key(db, q)
                if k not in sigs:
                    todo[k] = (db, q)
                if k not in verdicts:
                    gtodo[k] = (db, q, gold[r["unit"]])
    C.require_cached(todo, gtodo)
    for x in rows:
        lab = []
        for q in x["qs"]:
            s = (sigs.get(readers.query_key(x["db"], q)) or {}) if q else {}
            lab.append(s["sig"] if q and s.get("status") == "ok" else None)
        w, _ = readers._vote(lab, "first", "")
        x["O"] = 0.0 if w is None else float(bool((verdicts.get(readers.query_key(x["db"], x["qs"][lab.index(w)])) or {}).get("correct")))

    r0_logged = {}
    for p in sorted((logged_ext / "raw").glob("sql_bird_v2__bccs__*__qwen25-7b__r*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            cs = sorted([c for c in (r.get("record") or {}).get("calls") or [] if int(c.get("round") or 0) == 0],
                        key=lambda c: int(c.get("agent_index") or 0))
            r0_logged[(r["unit"], r["arm"], int(r["repeat"]))] = [(c.get("raw_reply") or "", readers._sql_of(c.get("extracted_answer"))) for c in cs]

    res = {"status": "computed", "online_runs": len(rows), "arms": sorted({x["arm"] for x in rows}),
           "online_fingerprints": dict(collections.Counter(str(x["image"]) for x in rows)),
           "note_image": ("the online run uses vLLM v0.32.0; the logged 7B run used a different image for the same "
                          "Qwen2.5-7B checkpoint @a09a3545; a discrepancy may come from the serving stack"),
           "note_A1": "no per-request seed in either run: in-distribution comparison only, never pathwise",
           "by_arm": {}}
    for arm in res["arms"]:
        on = collections.defaultdict(list)
        for x in rows:
            if x["arm"] == arm:
                on[x["unit"]].append(x)
        units = sorted(u for u in on if (u, arm) in logged)
        strata = collections.defaultdict(list)
        diff_all, lo_all, hi_all, onm, rpm = [], [], [], [], []
        for u in units:
            lg = logged[(u, arm)]
            kinds = [d["EI"]["D"] for d in lg]
            st = ("D=0" if all(k == "none" for k in kinds) else
                  "any unrecoverable" if any(k == "unrecoverable" for k in kinds) else "recoverable only")
            o = np.mean([x["O"] for x in on[u]])
            rp = np.mean([d["EI"]["R_last"] for d in lg])
            lo_all.append(np.mean([d["EI"]["O_lo"] for d in lg]))
            hi_all.append(np.mean([d["EI"]["O_hi"] for d in lg]))
            diff_all.append(o - rp)
            onm.append(o)
            rpm.append(rp)
            strata[st].append(o - rp)
        diff_all = np.array(diff_all)
        on_lv, rp_lv = float(np.mean(onm)) if onm else None, float(np.mean(rpm)) if rpm else None
        m_text = m_sql = n_pair = 0
        for x in rows:
            if x["arm"] != arm:
                continue
            lg = r0_logged.get((x["unit"], arm, x["repeat"]))
            if not lg:
                continue
            for (t1, s1), (t2, s2) in zip(x["r0"], lg):
                n_pair += 1
                m_text += t1 == t2
                m_sql += s1 == s2
        tdiv = [x["trace_div"] for x in rows if x["arm"] == arm and x["trace_div"] is not None]
        lg_all = [d for (u, ar), ds in logged.items() if ar == arm for d in ds]
        res["by_arm"][arm] = {
            "n_tasks_paired": len(units),
            "online_EX_pp": None if on_lv is None else round(100 * on_lv, 2),
            "replay_EX_pp": None if rp_lv is None else round(100 * rp_lv, 2),
            "online_minus_replay_all": summ(diff_all),
            "online_minus_replay_by_logged_D": {k: summ(np.array(v)) for k, v in sorted(strata.items())},
            "logged_manski_interval_pp": [round(100 * float(np.mean(lo_all)), 2), round(100 * float(np.mean(hi_all)), 2)] if lo_all else None,
            "online_inside_interval": (bool(np.mean(lo_all) <= on_lv <= np.mean(hi_all)) if lo_all else None),
            "descriptive_round0_exact_match": {"agent_pairs": n_pair,
                                               "raw_text_identical": round(m_text / n_pair, 4) if n_pair else None,
                                               "strict_sql_identical": round(m_sql / n_pair, 4) if n_pair else None},
            "online_divergence_share_from_trace": (round(float(np.mean(tdiv)), 4) if tdiv else None),
            "online_trace_runs_readable": len(tdiv),
            "replay_P_D_logged": round(float(np.mean([d["EI"]["D"] != "none" for d in lg_all])), 4) if lg_all else None,
        }
    out.write_text(json.dumps(res, indent=1))
    log(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="online BCCS runs with an execution vote against the replay")
    ap.add_argument("--ext", default=str(paths.REPO / "data/records/bird_debates_backbones"),
                    help="run directory holding the online BCCS runs with execution voting (default %(default)s)")
    ap.add_argument("--div", default=str(C.OUTPUT / "replay_validity" / "bccs_divergence_runs.jsonl"))
    ap.add_argument("--logged-ext", default=str(readers.DEBATES), help="run directory of the logged debates (default %(default)s)")
    ap.add_argument("--sig-caches", default=",".join(str(C.CACHE / n) for n in SIG_CACHES),
                    help="read-only execution-signature caches, comma-separated (a later entry wins)")
    ap.add_argument("--verdict-caches", default=",".join(str(C.CACHE / n) for n in VERDICT_CACHES),
                    help="read-only verdict caches, comma-separated (a later entry wins)")
    ap.add_argument("--out", default=str(C.OUTPUT / "replay_validity" / "online_validation.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    try:
        run(a.ext, a.div, logged_ext=a.logged_ext, sig_caches=a.sig_caches, verdict_caches=a.verdict_caches,
            out=a.out, workers=a.workers, selftest=a.selftest)
    except C.ExecutionNeeded as e:
        print(f"online_validation: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
