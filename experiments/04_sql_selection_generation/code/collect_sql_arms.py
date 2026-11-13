r"""A4 (deferred P2a/P2b): BIRD evidence-versus-generation and the feedback-content decomposition.

    PYTHONPATH=experiments/06_cost_resources/code:experiments/04_sql_selection_generation/code \
      python3 experiments/04_sql_selection_generation/code/collect_sql_arms.py \
        --stage peers|conditions [--arms S0,S1,S2,S3,S4,F1,F2]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
DATA = os.path.join(STAGE, "data/bird")
PORTS = {"qwen2.5-7b": 9101, "qwen2.5-32b": 9103}
RECIPIENTS = ("qwen2.5-7b", "qwen2.5-32b")
PEER_MODELS = ("qwen2.5-7b", "llama-3-8b", "qwen2.5-14b", "qwen2.5-32b")
PEER_PORTS = {"qwen2.5-7b": 9101, "llama-3-8b": 9102, "qwen2.5-14b": 9105,
              "qwen2.5-32b": 9103, "mistral-7b": 9104, "qwen2.5-72b": 9106}
EXTRA_GENERATORS = ("mistral-7b", "qwen2.5-72b")      # S3's two additional independent candidates
ARMS = ("S0", "S1", "S2", "S3", "S4", "F1", "F2")
REPEATS = 3
SEED = 20261013
MAX_TOKENS = 512
WORKERS = 24
SAMPLE_ROWS = 3           # frozen, matches the existing payload's SAMPLE_ROWS_IN_FEEDBACK

_lock = threading.Lock()


def post(model: str, prompt: str, port: Optional[int] = None,
         max_tokens: int = MAX_TOKENS, timeout: float = 300.0) -> Dict[str, Any]:
    p = port or PEER_PORTS[model]
    body = json.dumps({"model": model, "temperature": 0.0, "top_p": 1.0,
                       "max_tokens": max_tokens, "seed": SEED,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://{os.environ.get('INFERENCE_HOST', 'inference-host')}:{p}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    return {"text": ch["message"]["content"], "finish_reason": ch.get("finish_reason"),
            "prompt_tokens": d["usage"]["prompt_tokens"],
            "completion_tokens": d["usage"]["completion_tokens"]}


def append(path: str, obj: Dict[str, Any]) -> None:
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def done_units(path: str) -> set:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["unit_id"])
            except Exception:
                pass
    return out


def uid(*p: Any) -> str:
    return hashlib.sha256("|".join(str(x) for x in p).encode()).hexdigest()[:20]


def run(jobs: List[Dict[str, Any]], fn, label: str) -> Dict[str, Any]:
    t0 = time.time()
    st = {"n": 0, "err": 0, "errs": collections.Counter()}

    def wrap(j):
        try:
            fn(j)
            with _lock:
                st["n"] += 1
                if st["n"] % 2000 == 0:
                    el = time.time() - t0
                    print(f"    {label}: {st['n']}/{len(jobs)} {el:.0f}s "
                          f"({st['n']/el:.1f}/s)", flush=True)
        except Exception as e:
            with _lock:
                st["err"] += 1
                st["errs"][type(e).__name__] += 1

    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(wrap, jobs))
    el = time.time() - t0
    print(f"  {label}: {st['n']} ok, {st['err']} err {dict(st['errs'])}, {el:.0f}s", flush=True)
    return {"n_ok": st["n"], "n_err": st["err"], "errors": dict(st["errs"]), "wall_s": el}


# ------------------------------------------------------------------ feedback payloads
def payload(res: Dict[str, Any], level: str) -> str:
    """F1 status only; F2 status + error text, no rows; F3 the full established payload."""
    stt = res.get("status")
    if level == "F1":
        return {"ok": "query executed successfully",
                "error": "query failed",
                "timeout": "query timed out"}.get(stt, f"query status {stt}")
    if level == "F2":
        if stt == "ok":
            return "query executed successfully"
        if stt == "timeout":
            return "query timed out"
        return f"query failed: {res.get('error')}"
    if level == "F3":
        if stt == "ok":
            head = (f"query ran in {res.get('elapsed_s', 0):.3f}s; returned "
                    f"{res['n_rows_returned']}{'+' if res.get('more_rows_exist') else ''} row(s); "
                    f"columns: {', '.join(map(str, res.get('columns') or []))}")
            if res["n_rows_returned"] == 0:
                return head + "\n  no rows"
            body = "\n".join("  " + " | ".join(map(str, r))
                             for r in (res.get("rows") or [])[:SAMPLE_ROWS])
            return head + "\n  first rows:\n" + body
        if stt == "timeout":
            return "query timed out and was cancelled"
        return f"query failed: {res.get('error')}"
    raise AssertionError(level)


PEER_FRAME = ("\n\nOther systems proposed these queries for the same question:\n{block}\n\n"
              "Using this as additional information, write your own single SQLite query. "
              "Return only the SQL query, with no explanation and no code fence.")


def peer_block(peers: Sequence[Dict[str, Any]], level: Optional[str]) -> str:
    out = []
    for i, p in enumerate(peers):
        s = f"  system {i+1} proposed:\n    {p['sql'].strip()}"
        if level:
            s += f"\n    execution: {payload(p['exec'], level)}"
        out.append(s)
    return "\n\n".join(out)


def result_signature(res: Dict[str, Any]) -> str:
    """Normalised result-set signature for the frozen majority rule. Order-insensitive."""
    if res.get("status") != "ok":
        return f"__{res.get('status')}__"
    rows = res.get("rows") or []
    norm = sorted(tuple(str(c) for c in r) for r in rows)
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()[:24]


def majority_pick(cands: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Largest result-signature group; ties by the lowest-sorted SQL text. No gold, no evaluator."""
    live = [c for c in cands if c.get("sql")]
    if not live:
        return None
    groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for c in live:
        groups[result_signature(c["exec"])].append(c)
    best = max(groups.values(), key=lambda g: (len(g), -ord(sorted(
        x["sql"].strip() for x in g)[0][:1] or " ")))
    # deterministic: largest group, then lowest-sorted SQL
    top = max(len(g) for g in groups.values())
    cand = [g for g in groups.values() if len(g) == top]
    flat = [x for g in cand for x in g]
    return sorted(flat, key=lambda x: x["sql"].strip())[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["peers", "arms"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "experiments/06_cost_resources/code"))
    import sql_common as S                                              # noqa: PLC0415

    qs = S.load_questions()
    if a.limit:
        qs = qs[:a.limit]
    print(f"BIRD: {len(qs)} questions, {len({q['db_id'] for q in qs})} databases, "
          f"stage={a.stage}", flush=True)

    peers_path = os.path.join(DATA, "peers.jsonl")
    arms_path = os.path.join(DATA, "arms.jsonl")
    stats: Dict[str, Any] = {"n_questions": len(qs)}

    # ------------------------------------------------------------- peer + extra candidates
    if a.stage == "peers":
        have = done_units(peers_path)
        jobs = []
        for q in qs:
            pr = S.render(q)
            for m in tuple(PEER_MODELS) + tuple(EXTRA_GENERATORS):
                u = uid("sqlpeer", q["question_id"], m)
                if u not in have:
                    jobs.append({"unit_id": u, "q": q, "model": m, "prompt": pr,
                                 "role": "extra" if m in EXTRA_GENERATORS else "peer"})
        print(f"  peers+extra: {len(jobs)} calls ({len(have)} present)", flush=True)
        random.Random(SEED).shuffle(jobs)

        def do(j):
            g = post(j["model"], j["prompt"])
            sql = S.parse_sql(g["text"])
            ex = S.execute(j["q"]["db_id"], sql) if sql else {"status": "error",
                                                              "error": "no sql parsed"}
            append(peers_path, {
                "unit_id": j["unit_id"], "question_id": j["q"]["question_id"],
                "db_id": j["q"]["db_id"], "difficulty": j["q"]["difficulty"],
                "model": j["model"], "role": j["role"], "status": "ok",
                "sql": sql, "raw_reply": g["text"], "exec": ex,
                "exec_status": ex.get("status"),
                "prompt_tokens": g["prompt_tokens"],
                "completion_tokens": g["completion_tokens"],
                "finish_reason": g["finish_reason"]})
        stats["peers"] = run(jobs, do, "peers+extra") if jobs else {"n_ok": 0}

    # ------------------------------------------------------------- the recipient conditions
    if a.stage == "arms":
        want = [x for x in a.arms.split(",") if x in ARMS]
        rows = [json.loads(l) for l in open(peers_path, encoding="utf-8")]
        by_q: Dict[int, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
        for r in rows:
            by_q[r["question_id"]][r["model"]] = r
        have = done_units(arms_path)
        jobs = []
        qmap = {q["question_id"]: q for q in qs}
        for qid, got in by_q.items():
            q = qmap.get(qid)
            if q is None:
                continue
            base = S.render(q)
            peers = [got[m] for m in PEER_MODELS if m in got and got[m].get("sql")]
            extra = [got[m] for m in EXTRA_GENERATORS if m in got and got[m].get("sql")]
            if len(peers) < 2:
                continue
            pool34 = peers + extra
            pick34 = majority_pick(pool34)
            for rec in RECIPIENTS:
                for arm in want:
                    for rep in range(REPEATS):
                        u = uid("sqlarm", qid, rec, arm, rep)
                        if u in have:
                            continue
                        jobs.append({"unit_id": u, "q": q, "recipient": rec, "arm": arm,
                                     "repeat": rep, "base": base, "peers": peers,
                                     "extra": extra, "pick34": pick34})
        print(f"  arms {want}: {len(jobs)} calls ({len(have)} present)", flush=True)
        rng = random.Random(SEED)
        rng.shuffle(jobs)

        def do(j):
            arm, q = j["arm"], j["q"]
            peers, extra, pick = j["peers"], j["extra"], j["pick34"]
            level = {"S2": "F3", "F3": "F3", "F1": "F1", "F2": "F2"}.get(arm)
            if arm == "S0":
                prompt = j["base"]
            elif arm in ("S1",):
                prompt = j["base"] + PEER_FRAME.format(block=peer_block(peers, None))
            elif arm in ("S2", "F1", "F2"):
                prompt = j["base"] + PEER_FRAME.format(block=peer_block(peers, level))
            elif arm in ("S3", "S4"):
                # the same candidate set for both; S4 withholds the payload
                lvl = "F3" if arm == "S3" else None
                prompt = j["base"] + PEER_FRAME.format(
                    block=peer_block(peers + extra, lvl))
            else:
                raise AssertionError(arm)
            g = post(j["recipient"], prompt, port=PORTS[j["recipient"]])
            sql = S.parse_sql(g["text"])
            ex = S.execute(q["db_id"], sql) if sql else {"status": "error",
                                                         "error": "no sql parsed"}
            ev = S.execution_match(q["db_id"], sql, S.gold_sql(q)) if sql else {
                "match": False, "reason": "no sql parsed"}
            append(arms_path, {
                "unit_id": j["unit_id"], "question_id": q["question_id"],
                "db_id": q["db_id"], "difficulty": q["difficulty"],
                "recipient": j["recipient"], "arm": arm, "repeat": j["repeat"],
                "role": "recipient", "status": "ok", "sql": sql, "raw_reply": g["text"],
                "exec": ex, "exec_status": ex.get("status"),
                "exec_match": bool(ev.get("match")), "eval_reason": ev.get("reason"),
                "n_peers": len(peers), "n_candidates": len(peers) + (
                    len(extra) if arm in ("S3", "S4") else 0),
                "frozen_pick_sql": (pick or {}).get("sql") if arm in ("S3", "S4") else None,
                "source_statuses": [p["exec_status"] for p in peers],
                "prompt_tokens": g["prompt_tokens"],
                "completion_tokens": g["completion_tokens"],
                "truncated": g["finish_reason"] == "length",
                "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16]})
        stats["arms"] = run(jobs, do, f"arms:{'+'.join(want)}") if jobs else {"n_ok": 0}

    sp = os.path.join(STAGE, "logs", f"collect_sql_{a.stage}.json")
    json.dump(stats, open(sp, "w", encoding="utf-8"), indent=1, default=str)
    print(f"  wrote {sp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
