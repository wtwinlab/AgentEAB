r"""Score the runs of this collection (no model call; read-only SQL execution).

    python bm_score.py [--workers 16]          (PYTHONPATH and BM_OUT as in bm_stage.sh)

For every last successful record per (question, protocol, condition, backbone, repeat):
  * the lenient extractor takes the protocol's answer when it finds one, else the last fenced SQL
    block, else the last line that starts a query;
  * each query is executed read-only on its BIRD database (5 s statement timeout, 10,000-row cap);
    its execution result is the hash of the sorted, normalised result rows, and it is correct when
    the normalised rows equal those of the reference query;
  * the execution vote groups the queries of a round by execution result, drops queries that fail,
    and returns the most frequent result, a tie going to the first tied agent.
Writes <BM_OUT>/output/rows.jsonl (one debate run: last-round execution vote, the protocol's final
answer, calls, tokens), drafts.jsonl (one 20-draft set: per-draft execution result, correctness and
tokens) and exclusions.json (runs without a successful record, with their error class).
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "experiments/06_cost_resources/code", ROOT / "experiments/02_controlled_sharing/code"):
    sys.path.insert(0, str(_p))
import ext_sql_verdicts as XV                                             # noqa: E402
import sql_common as SQ                                                   # noqa: E402
import sql_tasks as ST                                                    # noqa: E402

OUT = pathlib.Path(os.environ.get("BM_OUT", ROOT / "experiments/17_budget_matched_debates/data"))
UNITS = json.loads((HERE / "questions.json").read_text())["questions"]
DRAFT_ARMS = ("DRAFTS20", "T07_DRAFTS20")
FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.S)
BARE = re.compile(r"(?im)^\s*((?:SELECT|WITH)\b[^;`]*)")


def qkey(db, sql):
    return hashlib.sha256(f"{db}|{sql}".encode()).hexdigest()[:24]


def sql_of(raw):
    return ST.unwrap_sql(raw) if raw not in (None, "", "None", "none") else ""


def lenient(raw_reply):
    t = str(raw_reply or "")
    m = FENCE.findall(t)
    if m:
        return ST.unwrap_sql(m[-1])
    b = BARE.findall(t)
    return ST.unwrap_sql(b[-1]) if b else ""


def signature(args):
    db, sql = args
    t0 = time.time()
    try:
        r = SQ.execute(db, sql, max_rows=10000)
    except Exception as exc:                                             # noqa: BLE001
        return qkey(db, sql), {"status": "error", "error": str(exc)[:120], "s": 0.0}
    if r["status"] != "ok":
        return qkey(db, sql), {"status": r["status"], "s": round(time.time() - t0, 3)}
    s = sorted(repr(t) for t in SQ._norm_result(r["rows"]))
    return qkey(db, sql), {"status": "ok", "sig": hashlib.sha256("\n".join(s).encode()).hexdigest()[:20],
                           "s": round(time.time() - t0, 3)}


def vote(labels):
    """Most frequent non-None label; a tie goes to the earliest."""
    live = [x for x in labels if x is not None]
    if not live:
        return None
    cnt = collections.Counter(live)
    top = max(cnt.values())
    return next(x for x in live if cnt[x] == top)


def read(p):
    for line in p.open(encoding="utf-8", errors="replace"):
        try:
            yield json.loads(line)
        except ValueError:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    units = set(UNITS)
    out = OUT / "output"
    cdir = out / "eval_cache"
    cdir.mkdir(parents=True, exist_ok=True)
    gold = {f"bird|{q['question_id']}": q for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}
    last, errs = {}, collections.defaultdict(list)
    for p in sorted((OUT / "raw").glob("*/sql_bird_v2__*.jsonl")):
        if p.parent.name == "smoke":
            continue
        for r in read(p):
            if r.get("unit") not in units:
                continue
            k = (r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))
            if r.get("status") == "ok":
                last[k] = r
            else:
                errs[k].append(dict(error_type=r.get("error_type"), error_class=r.get("error_class"),
                                    error=(r.get("error") or "")[:160]))
    excl = {k: v for k, v in errs.items() if k not in last}

    def cache(name):
        d = {}
        p = cdir / name
        if p.exists():
            for line in p.open():
                x = json.loads(line)
                d[x["k"]] = x
        return p, d
    sp, sigs = cache("sigs.jsonl")
    vp, verd = cache("verdicts.jsonl")
    runs, todo_s, todo_v = [], {}, {}
    for k, r in last.items():
        u = k[0]
        db = gold[u]["db_id"]
        calls = (r.get("record") or {}).get("calls") or []
        byr = collections.defaultdict(list)
        for c in calls:
            byr[int(c.get("round") or 0)].append(c)
        rounds = {}
        for rn, cs in byr.items():
            cs = sorted(cs, key=lambda c: int(c.get("agent_index") or 0))
            strict = [sql_of(c.get("extracted_answer")) for c in cs]
            lens = [s if s else lenient(c.get("raw_reply")) for s, c in zip(strict, cs)]
            rounds[rn] = dict(lenient=lens, calls=cs)
        fin = sql_of((r.get("record") or {}).get("majority_answer"))
        for q in [fin] + [q for rd in rounds.values() for q in rd["lenient"]]:
            if q:
                kk = qkey(db, q)
                if kk not in sigs:
                    todo_s[kk] = (db, q)
                if kk not in verd:
                    todo_v[kk] = (db, q, gold[u]["SQL"])
        runs.append((k, db, rounds, fin, calls))
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex, open(sp, "a") as fs, open(vp, "a") as fv:
        for kk, v in ex.map(signature, list(todo_s.values()), chunksize=8):
            v["k"] = kk
            sigs[kk] = v
            fs.write(json.dumps(v) + "\n")
        for kk, v in ex.map(XV._one, list(todo_v.values()), chunksize=8):
            v["k"] = kk
            verd[kk] = v
            fv.write(json.dumps(v) + "\n")

    def corr(db, q):
        return bool(q and (verd.get(qkey(db, q)) or {}).get("correct"))

    def sig(db, q):
        s = sigs.get(qkey(db, q)) if q else None
        return s["sig"] if s and s.get("status") == "ok" else None

    def agg(db, qs):
        lab = [sig(db, q) for q in qs]
        f = vote(lab)
        return f is not None and corr(db, qs[lab.index(f)])

    gold_failed = collections.Counter()
    with open(out / "rows.jsonl", "w") as fr, open(out / "drafts.jsonl", "w") as fd:
        for (u, proto, arm, bb, rep), db, rounds, fin, calls in runs:
            fv_ = verd.get(qkey(db, fin)) if fin else None
            if fv_ and fv_.get("outcome") == "gold_failed":
                gold_failed[u] += 1
            base = dict(unit=u, protocol=proto, arm=arm, backbone=bb, repeat=rep, db=db)
            if arm in DRAFT_ARMS:
                cs, lens = rounds[0]["calls"], rounds[0]["lenient"]
                fd.write(json.dumps(dict(base, sig=[sig(db, q) for q in lens], correct=[corr(db, q) for q in lens],
                                         parsed=[bool(q) for q in lens],
                                         prompt_tokens=[int(c.get("prompt_tokens") or 0) for c in cs],
                                         completion_tokens=[int(c.get("completion_tokens") or 0) for c in cs],
                                         truncated=sum(1 for c in cs if c.get("finish_reason") == "length"))) + "\n")
                continue
            lr = max(rounds)
            fr.write(json.dumps(dict(base, last_round=lr, n_agents=len(rounds[0]["lenient"]),
                                     final=bool(fv_ and fv_.get("correct")), last=agg(db, rounds[lr]["lenient"]),
                                     prompt_tokens=sum(int(c.get("prompt_tokens") or 0) for c in calls),
                                     completion_tokens=sum(int(c.get("completion_tokens") or 0) for c in calls),
                                     n_calls=len(calls),
                                     truncated=sum(1 for c in calls if c.get("finish_reason") == "length"))) + "\n")
    ex_list = [dict(unit=k[0], protocol=k[1], arm=k[2], backbone=k[3], repeat=k[4], attempts=v)
               for k, v in sorted(excl.items())]
    (out / "exclusions.json").write_text(json.dumps(dict(
        failed_runs=ex_list,
        failed_runs_by_cell=dict(collections.Counter(f"{e['protocol']}|{e['arm']}|{e['backbone']}" for e in ex_list)),
        gold_failed_units=dict(gold_failed)), indent=1, default=str))
    print(len(runs), "runs scored;", len(ex_list), "runs without a successful record; gold failures", dict(gold_failed))


if __name__ == "__main__":
    main()
