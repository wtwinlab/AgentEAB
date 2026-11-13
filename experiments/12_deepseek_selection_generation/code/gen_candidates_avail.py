r"""Generate a SQL candidate pool from strong source models, with a complete per-attempt call log.

    python3 code/gen_candidates.py --benchmark bird --family qwen3-32b --samples 2 \
        --out output/raw/cand_bird_qwen3-32b.jsonl --max-tokens 12288 --limit 6

Prompts are **not** rebuilt here. `render()` from the benchmark's own module produces them, the same
function that produced the existing pool, so question ids, schema rendering and the parse contract
stay aligned and the two pools are comparable.

Two independent samples per family per question make up half of a 4-candidate pool. The seed is
derived from `(benchmark, question id, family, sample index)`, so a repeat is an independent draw
with its own stable request id rather than the same request counted twice.

Every record carries: the stable request id, the full `messages`, the prompt hash, the requested and
served model, decoding and seed, the complete final and reasoning channels, token usage,
`finish_reason`, timings, attempt/retry/error, and the source family and pool id. SQL is parsed but
not executed here; execution and scoring are a separate step against the local databases.

The endpoints are unauthenticated services on the lab host, so no credential is used or stored.
Append-only and resumable: a unit is skipped only when it already has an `ok` attempt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]

#: the benchmarks' own question loaders, prompt renderers and parse contracts
BENCH = {
    "bird": REPO / "experiments" / "02_controlled_sharing" / "code",
    "spider": REPO / "experiments" / "04_sql_selection_generation" / "code",
}

HOST = os.environ.get("GEN_HOST", "127.0.0.1")   # inference host serving the source models
PORT_OVERRIDE = int(os.environ.get("GEN_PORT", "0"))  # optional port override; server arguments as in FAMILIES
#: one server per card, each launched under this task's VRAM reservation
FAMILIES = {
    "qwen3-32b": dict(port=8071, gpu=1,
                      checkpoint="Qwen/Qwen3-32B",
                      snapshot="9216db5781bf21249d130ec9da846c4624c16137",
                      precision="bf16"),
    "glm47-flash": dict(port=8073, gpu=3,
                        checkpoint="zai-org/GLM-4.7-Flash",
                        snapshot=None, precision="as released in the cached snapshot"),
}
#: identical across families on purpose: with the sampling held fixed, a difference between the two
#: pools' halves is the model rather than the decoding. Family A's values are the ones the existing
#: Qwen3-32B rows were collected under.
DECODING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)

RETRY_LIMIT = 2
TIMEOUT_S = 1200.0
_lock = threading.Lock()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_bench(name: str):
    p = str(BENCH[name])
    if p not in sys.path:
        sys.path.insert(0, p)
    mod = __import__("sql_common" if name == "bird" else "spider_common")
    return mod


def request_id(bench: str, qid: Any, family: str, sample: int) -> str:
    return f"{bench}|q{qid}|{family}|s{sample}"


def seed_for(rid: str) -> int:
    return int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=sorted(BENCH))
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=20000, help="hard cap on this invocation")
    a = ap.parse_args()

    mod = load_bench(a.benchmark)
    fam = FAMILIES[a.family]
    base_url = f"http://{HOST}:{PORT_OVERRIDE or fam['port']}/v1"
    qs = mod.load_questions()
    if a.limit:
        qs = qs[:a.limit]

    def qid_of(q):
        for k in ("question_id", "qid", "id"):
            if k in q:
                return q[k]
        return qs.index(q)

    units = [(q, s) for q in qs for s in range(a.samples)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("status") == "ok":
                done.add(r["request_id"])
    todo = [(q, s) for q, s in units
            if request_id(a.benchmark, qid_of(q), a.family, s) not in done]
    if len(todo) > a.max_calls:
        print(f"FATAL: {len(todo)} calls exceeds --max-calls {a.max_calls}; "
              f"reduce the population or raise the cap deliberately")
        return 1
    print(f"benchmark={a.benchmark} family={a.family} questions={len(qs)} samples={a.samples} "
          f"already_ok={len(done)} todo={len(todo)} endpoint={base_url} "
          f"max_tokens={a.max_tokens} decoding={DECODING}", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    out = open(a.out, "a")
    counts = dict(ok=0, failed=0, attempts=0, parsed=0, truncated=0, served_bad=0)
    tok = dict(prompt=0, completion=0)
    t_start = time.time()

    def write(rec: dict) -> None:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())

    def work(item) -> None:
        q, sample = item
        qid = qid_of(q)
        rid = request_id(a.benchmark, qid, a.family, sample)
        prompt = mod.render(q)
        messages = [{"role": "user", "content": prompt}]
        seed = seed_for(rid)
        body = dict(model=a.family, messages=messages, max_tokens=a.max_tokens, seed=seed,
                    **DECODING)
        base = dict(request_id=rid, benchmark=a.benchmark, question_id=qid,
                    db_id=q.get("db_id"), source_family=a.family, sample_index=sample,
                    pool_id=f"{a.benchmark}|q{qid}|strong4",
                    endpoint=dict(host=HOST, port=fam["port"], gpu=fam["gpu"]),
                    config=dict(requested_model=a.family, checkpoint=fam["checkpoint"],
                                snapshot=fam["snapshot"], precision=fam["precision"],
                                decoding=dict(DECODING), max_tokens=a.max_tokens, seed=seed),
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    messages=messages)
        for attempt in range(RETRY_LIMIT + 1):
            t0, ts0 = time.time(), utc()
            try:
                req = urllib.request.Request(
                    f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                    resp = json.loads(r.read())
                lat = time.time() - t0
                ch = resp["choices"][0]
                msg = ch["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or None
                sql = mod.parse_sql(content)
                usage = resp.get("usage") or {}
                served = resp.get("model")
                with _lock:
                    counts["ok"] += 1
                    counts["attempts"] += 1
                    if sql:
                        counts["parsed"] += 1
                    if ch.get("finish_reason") == "length":
                        counts["truncated"] += 1
                    if served != a.family:
                        counts["served_bad"] += 1
                    tok["prompt"] += usage.get("prompt_tokens") or 0
                    tok["completion"] += usage.get("completion_tokens") or 0
                    write(dict(base, status="ok", attempt=attempt, ts_start_utc=ts0,
                               ts_end_utc=utc(), latency_s=round(lat, 3),
                               served=dict(reported=served, requested=a.family,
                                           matches=served == a.family),
                               response=resp, final_content=content,
                               reasoning_content=reasoning, parsed_sql=sql,
                               finish_reason=ch.get("finish_reason"), usage=usage, error=None))
                    n = counts["ok"] + counts["failed"]
                    if n % 50 == 0 or n == len(todo):
                        el = time.time() - t_start
                        print(f"{n}/{len(todo)} ok={counts['ok']} parsed={counts['parsed']} "
                              f"trunc={counts['truncated']} {el/max(n,1):.2f}s/call "
                              f"eta={(len(todo)-n)*el/max(n,1)/60:.1f}min", flush=True)
                return
            except Exception as e:                                      # noqa: BLE001
                detail = f"{type(e).__name__}: {e}"
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        detail += " | " + e.read().decode()[:500]
                    except Exception:                                   # noqa: BLE001
                        pass
                with _lock:
                    counts["attempts"] += 1
                    write(dict(base, status="attempt_failed", attempt=attempt, ts_start_utc=ts0,
                               ts_end_utc=utc(), latency_s=round(time.time() - t0, 3),
                               served=None, response=None, final_content=None,
                               reasoning_content=None, parsed_sql=None, finish_reason=None,
                               usage=None, error=detail))
                time.sleep(2.0 * (attempt + 1))
        with _lock:
            counts["failed"] += 1
            write(dict(base, status="failed", attempt=RETRY_LIMIT, ts_start_utc=utc(),
                       ts_end_utc=utc(), latency_s=None, served=None, response=None,
                       final_content=None, reasoning_content=None, parsed_sql=None,
                       finish_reason=None, usage=None, error="exhausted retries"))

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(work, todo))
    out.close()
    wall = time.time() - t_start
    ledger = dict(benchmark=a.benchmark, family=a.family, checkpoint=fam["checkpoint"],
                  snapshot=fam["snapshot"], precision=fam["precision"], gpu=fam["gpu"],
                  endpoint=base_url, decoding=DECODING, max_tokens=a.max_tokens,
                  samples=a.samples, questions=len(qs), units_attempted=len(todo),
                  units_ok=counts["ok"], units_failed=counts["failed"],
                  calls_billed=counts["attempts"], parsed_sql=counts["parsed"],
                  truncated=counts["truncated"], served_mismatches=counts["served_bad"],
                  prompt_tokens=tok["prompt"], completion_tokens=tok["completion"],
                  wall_s=round(wall, 1), gpu_hours=round(wall / 3600, 3), finished_utc=utc(),
                  note="one card held for this family; gpu_hours is this invocation's wall time on "
                       "that single card")
    with open(a.out + f".ledger.{int(t_start)}.json", "w") as f:
        json.dump(ledger, f, indent=1)
    print(json.dumps(ledger), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
