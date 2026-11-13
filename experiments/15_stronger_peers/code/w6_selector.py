r"""W6: one LLM candidate selector over the existing four-model pool.

    python3 code/w6_selector.py --pilot        # 24 BIRD + 24 Spider, interface check only
    python3 code/w6_selector.py --main         # 1,534 BIRD + 1,034 Spider

This is an **LLM candidate selector**, not a reproduction of any published system, and no published
score is claimed. It returns a candidate ID and nothing else; rewriting SQL is not permitted.

Frozen before the pilot:
  * pool and order: qwen2.5-7b, llama-3-8b, qwen2.5-14b, qwen2.5-32b -> C1, C2, C3, C4
  * the selector sees the question, the evidence note, each candidate's SQL and that candidate's
    stored execution feedback. It never sees a gold or reference answer.
  * seed 20261101, temperature 0, thinking disabled, 256 output tokens
  * parse: the first standalone `C<k>` with 1 <= k <= 4 in the final channel, after stripping any
    reasoning channel to the last close tag. No other salvage.
  * a parse failure, a truncation or an out-of-pool ID stays in the denominator as a
    non-selection. Nothing is dropped and gold never patches a choice.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import threading

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                          # noqa: E402

MODEL = "qwen3-32b"
SEED = 20261101
MAX_TOKENS = 256
POOL = ("qwen2.5-7b", "llama-3-8b", "qwen2.5-14b", "qwen2.5-32b")
IDS = tuple(f"C{i}" for i in range(1, len(POOL) + 1))
#: thinking is disabled for the selector: the task is a bounded choice among four offered IDs, and
#: a reasoning channel would consume the whole 256-token budget before producing one.
DECODING = {"temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
PILOT_N = 24
CONCURRENCY = 24
PICK = re.compile(r"\bC([1-9][0-9]*)\b")

PROMPT = """You are choosing which one of several candidate SQL queries best answers a question.

Database: {db_id}
Question: {question}{evidence}

Candidates, each with the result of running it:

{candidates}
Reply with the identifier of the single best candidate and nothing else, for example: {first}
Do not write or modify any SQL. Reply with one identifier only."""


def feedback(exec_obj, exec_status) -> str:
    if not isinstance(exec_obj, dict):
        return f"execution: {exec_status or 'unknown'}"
    st = exec_obj.get("status") or exec_status or "unknown"
    if st != "ok":
        return f"execution: failed ({str(exec_obj.get('error') or st)[:160]})"
    n = exec_obj.get("n_rows_returned")
    cols = exec_obj.get("columns") or []
    rows = exec_obj.get("rows") or []
    head = json.dumps(rows[:3], ensure_ascii=False)[:400]
    more = " (more rows exist)" if exec_obj.get("more_rows_exist") else ""
    return (f"execution: ok, {n} row(s){more}, columns {json.dumps(cols)[:200]}\n"
            f"  first rows: {head}")


def load(bench: str) -> list:
    """One item per question that has at least one pool candidate with SQL."""
    qfile = {"bird": C.input_path("bird_dev.json"), "spider": C.input_path("spider_dev.json")}[bench]
    qs = json.loads(qfile.read_text())
    qtext = {}
    for q in qs:
        qid = q.get("question_id")
        if qid is None:
            qid = len(qtext)
        qtext[int(qid)] = q
    peers = collections.defaultdict(dict)
    with C.input_path(f"peers_{bench}.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            peers[int(r["question_id"])][r["model"]] = r
    items, skipped = [], []
    for qid in sorted(peers):
        pm = peers[qid]
        live = [(i, m, pm[m]) for i, m in enumerate(POOL)
                if m in pm and (pm[m].get("sql") or "").strip()]
        if not live:
            skipped.append(dict(benchmark=bench, question_id=qid, reason="no pool candidate with SQL"))
            continue
        q = qtext.get(qid, {})
        blocks, mapping = [], {}
        for i, m, rec in live:
            cid = IDS[i]
            mapping[cid] = m
            blocks.append(f"{cid}:\n  SQL: {(rec.get('sql') or '').strip()}\n"
                          f"  {feedback(rec.get('exec'), rec.get('exec_status'))}\n")
        ev = q.get("evidence")
        items.append(dict(
            unit_id=f"sel|{bench}|{qid}", benchmark=bench, question_id=qid,
            db_id=q.get("db_id") or (live[0][2].get("db_id")),
            n_candidates=len(live), id_to_model=mapping,
            prompt=PROMPT.format(
                db_id=q.get("db_id") or live[0][2].get("db_id"),
                question=(q.get("question") or "").strip(),
                evidence=f"\nHint: {str(ev).strip()}" if ev else "",
                candidates="\n".join(blocks), first=IDS[0])))
    return items, skipped


def parse(text: str, mapping: dict):
    """First standalone C<k> that is an offered ID. Anything else is a non-selection."""
    for m in PICK.finditer(text or ""):
        cid = f"C{m.group(1)}"
        if cid in mapping:
            return cid, None
        return None, f"out_of_pool:{cid}"
    return None, "no_identifier"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--main", action="store_true")
    a = ap.parse_args()
    if not (a.pilot or a.main):
        print("choose --pilot or --main")
        return 2
    tag = "pilot" if a.pilot else "main"
    out = C.RAW / f"w6_selector_{tag}.jsonl"

    items, skipped = [], []
    for b in ("bird", "spider"):
        it, sk = load(b)
        items += (it[:PILOT_N] if a.pilot else it)
        skipped += sk
    (C.OUT / f"w6_skipped_{tag}.json").parent.mkdir(parents=True, exist_ok=True)
    (C.OUT / f"w6_skipped_{tag}.json").write_text(json.dumps(skipped, indent=1) + "\n")

    done = C.load_done(out)
    todo = [i for i in items if i["unit_id"] not in done]
    print(f"[{tag}] planned {len(items)} (bird+spider)  done {len(done)}  to do {len(todo)}  "
          f"skipped-no-pool {len(skipped)}", flush=True)
    print(f"  per benchmark: {dict(collections.Counter(i['benchmark'] for i in items))}", flush=True)
    if not todo:
        return 0
    if not C.wait_ready(C.PORTS["selector"]):
        print("FATAL: selector endpoint never became ready")
        return 1

    lock = threading.Lock()
    n = collections.Counter()

    def run(it):
        msgs = [{"role": "user", "content": it["prompt"]}]
        rec = None
        for attempt in (1, 2):
            rec = C.call(C.PORTS["selector"], MODEL, msgs, MAX_TOKENS, DECODING, SEED, attempt,
                         timeout=600.0)
            if rec["status"] == "ok":
                break
        final = C.final_channel(rec)
        cid, why = parse(final, it["id_to_model"])
        o = dict(it)
        o.update(rec)
        o["messages"] = msgs
        o["model"] = MODEL
        o["snapshot"] = C.SNAPSHOT[MODEL]
        o["final_text"] = final
        o["finish_reason"] = C.finish_reason(rec)
        o["usage"] = C.usage(rec)
        o["picked_id"] = cid
        o["picked_model"] = it["id_to_model"].get(cid) if cid else None
        o["nonselection_reason"] = why
        with lock:
            C.append(out, o)
            n[rec["status"]] += 1
            if cid:
                n["picked"] += 1
            else:
                n[f"nonselection:{why.split(':')[0]}"] += 1
            if o["finish_reason"] == "length":
                n["truncated"] += 1
            t = n["ok"] + n["error"]
            if t % 100 == 0 or t == len(todo):
                print(f"  {t}/{len(todo)} {dict(n)}", flush=True)

    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(run, todo))
    print(f"[{tag}] done {dict(n)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
