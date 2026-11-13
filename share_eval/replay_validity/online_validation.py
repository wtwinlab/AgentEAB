r"""Online check of the BCCS replay prediction, in distribution only (zero model calls).

    python3 -I share_eval/replay_validity/online_validation.py [--ext data/records/bird_debates_backbones] [--logged-ext DIR] [--div DIV] [--out FILE]

The online run `bccs_execid` (Qwen2.5-7B, backbone label `qwen2.5-7b-execid`) runs BCCS with
execution identity inside the protocol. The logged 7B BCCS requests carried no per-request seed and
the online run uses a different vLLM image (v0.32.0 vs the logged run's image `sha256:8a69ff…`), so
exogenous randomness is not shared (A1 fails) and no pathwise check is possible. This script asks
only whether the replay-predicted EX distribution and the online EX distribution agree:

  * readout: lenient extraction + execution identity with failures dropped (`exec_filt`; the
    factorial's `lenient|exec`, first tie), applied identically to the logged last round and to the
    online last round;
  * per task: online mean over its repeats minus replay mean over the logged repeats (paired by task);
  * strata by the LOGGED task's divergence: D = 0 in every logged repeat / any recoverable only /
    any unrecoverable. The lemma predicts equality in distribution on the D = 0 stratum and only an
    interval elsewhere. Strata condition on the logged runs' randomness; read them descriptively;
  * coverage: does the online aggregate fall inside the logged Manski interval;
  * descriptive only: exact match rate of the initial answers (raw text, strict extraction) between online and
    logged runs of the same task, agent and repeat; online divergence frequency read from
    `execid_trace` where its fields can be identified.

Outputs `<out>` (default output/share_eval/replay_validity/online_validation.json). If no online record exists yet, it writes a status file and
exits 0.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np

sys.dont_write_bytecode = True

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "share_eval/replay"))
import factorial_replay as FR                                              # noqa: E402

BOOT, SEED = 10000, 20261106
LOGGED_BB = "qwen2.5-7b"
EXCLUDE = {"bird|518", "bird|701"}
SELFTEST = False   # --selftest: treat the logged 7B BCCS runs as "online"; every difference must be exactly 0


def find_online(ext: pathlib.Path) -> List[Dict[str, Any]]:
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
            if "execid" in proto or "execid" in bb or (SELFTEST and proto == "bccs" and bb == LOGGED_BB):
                recs.append(r)
    last = {}
    for r in recs:                                                       # last ok record wins
        last[(r.get("unit"), r.get("arm"), int(r.get("repeat") or 0))] = r
    return list(last.values())


def calls_by_round(rec):
    by = collections.defaultdict(list)
    for c in (rec.get("record") or {}).get("calls") or []:
        by[int(c.get("round") or 0)].append(c)
    return {k: sorted(v, key=lambda c: int(c.get("agent_index") or 0)) for k, v in by.items()}


def trace_divergence(rec):
    """Best-effort read of `execid_trace`: True if any round's exec-identity verdict differs from the
    string-identity verdict. Returns None if the fields cannot be identified."""
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


def summ(x: np.ndarray, rng_seed: int = SEED) -> Dict[str, Any]:
    if len(x) == 0:
        return {"n_tasks": 0}
    idx = np.random.default_rng(rng_seed).integers(0, len(x), size=(BOOT, len(x)))
    b = x[idx].mean(axis=1)
    return {"n_tasks": int(len(x)), "est_pp": round(100 * float(x.mean()), 2),
            "ci95_pp": [round(100 * float(np.percentile(b, 2.5)), 2), round(100 * float(np.percentile(b, 97.5)), 2)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default=str(REPO / "data/records/bird_debates_backbones"),
                    help="run directory holding the online BCCS runs with execution voting (default %(default)s)")
    ap.add_argument("--div", default=str(REPO / "output/share_eval/replay_validity/bccs_divergence_runs.jsonl"))
    ap.add_argument("--logged-ext", default=str(FR.RUN_DIR), help="run directory of the logged debates (default %(default)s)")
    ap.add_argument("--sig-caches", default=",".join(str(FR.CACHE / n) for n in (
                        "r0_exec_signatures.jsonl", "replay_exec_signature_cache.jsonl", "divergence_exec_signature_cache.jsonl")),
                    help="read-only execution-signature caches, comma-separated (a later entry wins)")
    ap.add_argument("--verdict-caches", default=",".join(str(FR.CACHE / n) for n in (
                        "replay_verdict_cache.jsonl", "divergence_verdict_cache.jsonl")),
                    help="read-only verdict caches, comma-separated (a later entry wins)")
    ap.add_argument("--out", default=str(REPO / "output/share_eval/replay_validity/online_validation.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    global SELFTEST
    SELFTEST = a.selftest
    ext = pathlib.Path(a.ext)
    online = [r for r in find_online(ext) if r.get("unit") not in EXCLUDE]
    if not online:
        pathlib.Path(a.out).write_text(json.dumps({"status": "no online bccs_execid records found", "ext": str(ext)}, indent=1))
        print("no online records yet; status written")
        return 0

    # logged replay (7B, both repeats) from the divergence run file
    logged = collections.defaultdict(list)
    for l in pathlib.Path(a.div).read_text().splitlines():
        d = json.loads(l)
        if d["backbone"] == LOGGED_BB:
            logged[(d["unit"], d["arm"])].append(d)

    # online readout: lenient + exec_filt on the last round, factorial semantics
    import sql_common as SQ  # noqa: PLC0415
    gold = {f"bird|{q['question_id']}": q["SQL"] for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}
    odir = pathlib.Path(a.out).parent
    odir.mkdir(parents=True, exist_ok=True)
    sig_p, ver_p = odir / "online_exec_signature_cache.jsonl", odir / "online_verdict_cache.jsonl"
    sigs = FR.load_jsonl_cache([pathlib.Path(p) for p in a.sig_caches.split(",") if p] + [sig_p])
    verdicts = FR.load_jsonl_cache([pathlib.Path(p) for p in a.verdict_caches.split(",") if p] + [ver_p])
    vstore = pathlib.Path(a.logged_ext) / "analysis/sql_verdicts.json"
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
        strict = [FR.sql_of(c.get("extracted_answer")) for c in cbr[L]]
        qs = [s if s else FR.lenient(c.get("raw_reply")) for s, c in zip(strict, cbr[L])]
        r0 = [(c.get("raw_reply") or "", FR.sql_of(c.get("extracted_answer"))) for c in cbr.get(0, [])]
        rows.append(dict(unit=r["unit"], arm=r["arm"], repeat=int(r.get("repeat") or 0), db=db, qs=qs, last_round=L,
                         r0=r0, trace_div=trace_divergence(r), image=(r.get("record") or {}).get("calls", [{}])[0].get("fingerprint")))
        for q in qs:
            if q:
                k = FR.qkey(db, q)
                if k not in sigs:
                    todo[k] = (db, q)
                if k not in verdicts:
                    gtodo[k] = (db, q, gold[r["unit"]])
    if todo or gtodo:
        with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
            with open(sig_p, "a") as fh:
                for kk, v in ex.map(FR._sig, list(todo.values()), chunksize=16):
                    v["k"] = kk
                    sigs[kk] = v
                    fh.write(json.dumps(v) + "\n")
            with open(ver_p, "a") as fh:
                for kk, v in ex.map(FR._verdict, list(gtodo.values()), chunksize=16):
                    verdicts[kk] = v
                    fh.write(json.dumps(dict(v, k=kk)) + "\n")
    for x in rows:
        lab = []
        for q in x["qs"]:
            s = sigs.get(FR.qkey(x["db"], q)) or {} if q else {}
            lab.append(s["sig"] if q and s.get("status") == "ok" else None)
        w, _ = FR.vote(lab, "first", "")
        x["O"] = 0.0 if w is None else float(bool((verdicts.get(FR.qkey(x["db"], x["qs"][lab.index(w)])) or {}).get("correct")))

    # logged raw initial-answer texts for the descriptive match rate
    r0_logged = {}
    for p in sorted((pathlib.Path(a.logged_ext) / "raw").glob("sql_bird_v2__bccs__*__qwen25-7b__r*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            cs = sorted([c for c in (r.get("record") or {}).get("calls") or [] if int(c.get("round") or 0) == 0],
                        key=lambda c: int(c.get("agent_index") or 0))
            r0_logged[(r["unit"], r["arm"], int(r["repeat"]))] = [(c.get("raw_reply") or "", FR.sql_of(c.get("extracted_answer"))) for c in cs]

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
        # descriptive: exact match of the initial answers
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
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
