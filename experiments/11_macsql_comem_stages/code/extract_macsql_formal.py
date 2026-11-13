r"""MAC-SQL formal extraction and scoring on the host. Gold is read only here.

    python3 extract_macsql_formal.py <E> <model_tag> <bench> <out.json>

Per question (all of dev; nothing dropped):
- record from the shard output (`idx` = position in the shard file -> question_id via the ids file);
- trace calls by stage (`api_trace.json` entries carry idx), joined to proxy usage by prompt text;
- boundary = Decomposer `final_sql` (sha256); Refiner `pred`;
- official trigger rule replayed on the boundary execution (120 s, MAC-SQL's limit; BIRD: error /
  0 rows / None in first 5 rows; Spider: error only) vs observed Refiner LLM calls;
- scoring with the paper evaluator's semantics (sql_common.execution_match: set of `_coerce`d
  tuples over the first 10,000 rows, read-only, query_only), statement timeout 30 s.
"""
import collections
import glob
import hashlib
import json
import os
import re
import sqlite3
import sys
import time

E, MT, B, OUT = sys.argv[1:5]
F = f"{E}/formal"
DBROOT = f"{E}/data/dev_databases" if B == "bird" else f"{E}/data/spider_database"
SIG = (("refiner", "When executing SQL below, some errors occurred"),
       ("decomposer", "decompose the question into subquestions"),
       ("decomposer", "then generate the corresponding SQL"),
       ("selector", "As an experienced and professional database administrator"),
       ("ping", "Hello world!"))


def stage(t):
    for n, s in SIG:
        if s in t:
            return n
    return "unattributed"


def coerce(v):
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, float):
        return round(v, 6)
    return v


def execute(db, sql, cap=10000, limit=30.0, first_only=None):
    if not sql or not isinstance(sql, str):
        return dict(status="none", rows=None, seconds=0.0)
    con = sqlite3.connect(f"file:{DBROOT}/{db}/{db}.sqlite?mode=ro", uri=True, timeout=limit)
    con.execute("PRAGMA query_only=ON")
    if first_only is None:
        con.text_factory = lambda b: b.decode(errors="ignore")
    t0 = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - t0 > limit else 0, 1000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(first_only or cap)
        return dict(status="ok", rows=rows, seconds=round(time.time() - t0, 4))
    except Exception as e:                                                 # noqa: BLE001
        return dict(status="timeout" if "interrupted" in str(e).lower() else "error", rows=None, error=str(e)[:300],
                    seconds=round(time.time() - t0, 4))
    finally:
        con.close()


def norm(rows):
    return {tuple(coerce(x) for x in r) for r in rows}


def add_prefix(sql):
    return sql if sql.startswith(("SELECT", "select")) else "SELECT" + sql


def trigger(sql, db):
    """Official Refiner._is_need_refine on its own execution (fetchall, data = first 5 rows)."""
    if "error" in sql:
        return False, "passthrough_error_string"
    ex = execute(db, sql, limit=120.0, first_only=5)
    if ex["status"] == "timeout":
        return False, "timeout_no_refine"
    if ex["status"] != "ok":
        return True, "error"
    if B == "spider":
        return False, "ok"
    if len(ex["rows"]) == 0:
        return True, "empty"
    if any(n is None for t in ex["rows"] for n in t):
        return True, "none_value"
    return False, "ok"


if B == "bird":
    items = json.load(open(f"{E}/gold/bird_dev.json"))
    gold = {q["question_id"]: (q["db_id"], q["SQL"]) for q in items}
else:
    items = json.load(open(f"{E}/gold/spider_dev.json"))
    gold = {i: (q["db_id"], q["query"]) for i, q in enumerate(items)}
proxy = []
for p in glob.glob(f"{F}/raw/{MT}_{B}_s*.jsonl") + glob.glob(f"{F}/raw_b/{MT}_{B}_s*.jsonl"):   # raw_b: phase B (R3)
    proxy += [json.loads(l) for l in open(p)]
byprompt = collections.defaultdict(list)
for c in proxy:
    if c.get("refused"):
        continue
    msgs = c["request"].get("messages") or [{}]
    byprompt[(msgs[0].get("content") or "").strip()].append(c)
res = []
for k in (1, 2, 3):
    ids = json.load(open(f"{F}/data/{B}_s{k}_ids.json"))
    O = f"{F}/out/macsql_{MT}_{B}_s{k}"
    recs = collections.defaultdict(list)
    if os.path.exists(f"{O}/output.jsonl"):
        for l in open(f"{O}/output.jsonl"):
            r = json.loads(l); recs[r["idx"]].append(r)
    trace = collections.defaultdict(list)
    if os.path.exists(f"{O}/api_trace.json"):
        for l in open(f"{O}/api_trace.json"):
            t = json.loads(l); trace[t.get("idx")].append(t)
    for i, q in enumerate(ids):
        db, gsql = gold[q]
        flags = []
        rr = recs.get(i, [])
        r = rr[-1] if rr else None
        if len(rr) > 1:
            flags.append("MULTI_RECORD")
        tr = trace.get(i, [])
        st = [stage(t["input_prompt"]) for t in tr]
        calls = collections.defaultdict(collections.Counter)
        unjoined = 0
        for t in tr:
            cands = byprompt.get(t["input_prompt"].strip())
            c = cands.pop(0) if cands else None
            s = stage(t["input_prompt"])
            calls[s]["calls"] += 1
            calls[s]["prompt_tokens"] += t.get("prompt_token") or 0
            calls[s]["completion_tokens"] += t.get("response_token") or 0
            if c:
                calls[s]["latency_ms"] += int(1000 * (c.get("latency_s") or 0))
            else:
                unjoined += 1
        final_sql = (r or {}).get("final_sql")
        if r is None:
            dec = [t for t in tr if stage(t["input_prompt"]) == "decomposer"]
            flags.append("REFINER_FAILED" if dec else "PREFIX_FAILED")
            if dec:   # Keep from the Decomposer reply, parsed as the official Decomposer does
                m = re.findall(r"```sql(.*?)```", dec[-1]["response"], re.S)
                final_sql = m[-1].strip() if m else None
        keep = final_sql
        ref_calls = [t for t in tr if stage(t["input_prompt"]) == "refiner"]
        trig, reason = trigger(keep, db) if keep else (False, "no_boundary")
        if keep and "error" not in keep:
            if bool(ref_calls) != trig:
                flags.append("TRIGGER_MISMATCH")
            if ref_calls:
                m = re.search(r"【old SQL】\n```sql\n(.*?)\n```\n【SQLite error】", ref_calls[0]["input_prompt"], re.S)
                if not m or m.group(1).strip() != add_prefix(keep).strip():
                    flags.append("BOUNDARY_MISMATCH")
        pred = (r or {}).get("pred")
        g = execute(db, gsql)
        gs = norm(g["rows"]) if g["status"] == "ok" else None
        if gs is None:
            flags.append("GOLD_FAILED")
        ek = execute(db, keep) if keep else dict(status="none", rows=None, seconds=0.0)
        er = ek if (pred is not None and keep is not None and pred.strip() == keep.strip()) else (execute(db, pred) if pred else dict(status="none", rows=None, seconds=0.0))
        sc = lambda ex: int(gs is not None and ex["status"] == "ok" and norm(ex["rows"]) == gs)  # noqa: E731
        res.append(dict(question_id=q, db_id=db, shard=k, idx=i, flags=flags, boundary_sha256=hashlib.sha256(keep.encode()).hexdigest() if keep else None,
                        boundary_sql=keep, refiner_sql=pred, fixed=(r or {}).get("fixed"), try_times=(r or {}).get("try_times"),
                        pruned=(r or {}).get("pruned"), trace_stages=st, refiner_llm_calls=len(ref_calls), trigger_replayed=trig,
                        trigger_reason=reason, keep_status=ek["status"], keep_seconds=ek["seconds"], refiner_status=er["status"],
                        gold_status=g["status"], gold_empty=(gs is not None and len(gs) == 0),
                        scores=dict(Refiner=sc(er) if r else 0, Keep=sc(ek), Either_correct=int((sc(er) if r else 0) or sc(ek))),
                        calls_by_stage={s: dict(v) for s, v in calls.items()}, unjoined_trace_calls=unjoined))
json.dump(dict(model=MT, bench=B, n=len(res), proxy_calls=len(proxy), proxy_refused=sum(bool(c.get("refused")) for c in proxy),
               proxy_non200=sum(c.get("status") != 200 for c in proxy if not c.get("refused")),
               proxy_by_stage=dict(collections.Counter(stage((c["request"].get("messages") or [{}])[0].get("content") or "") for c in proxy)),
               units=res), open(OUT, "w"), default=str)
print(MT, B, len(res), collections.Counter(f for x in res for f in x["flags"]), "triggered", sum(x["refiner_llm_calls"] > 0 for x in res),
      "Refiner", sum(x["scores"]["Refiner"] for x in res), "Keep", sum(x["scores"]["Keep"] for x in res))
