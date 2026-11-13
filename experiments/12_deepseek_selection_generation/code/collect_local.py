r"""Local-vLLM collector for Part A generation requests, over local ports (INFERENCE_HOST forwarded to 127.0.0.1).

    python3 code/collect_local.py --input data/avail_requests_G_bird.jsonl --out raw/avail_G_bird.jsonl \
        --ports 8305,8315 --concurrency 64 --max-calls 26000

Decoding fixed to the generation arm of the pool-transfer round: temperature 0.6, top_p 0.95,
top_k 20, min_p 0, thinking default, max_tokens 32768, the stored per-question seed, model qwen3-32b.
Append-only and resumable (a unit is skipped once it has an `ok` record); a transport/HTTP error is
retried once; every attempt is written.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import itertools
import json
import os
import threading
import time
import urllib.request

DEC = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ports", required=True); ap.add_argument("--concurrency", type=int, default=64)
ap.add_argument("--max-calls", type=int, required=True)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.input)]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["unit_id"] for l in open(a.out) if json.loads(l).get("status") == "ok"}
todo = [r for r in rows if r["unit_id"] not in done]
if len(todo) > a.max_calls:
    raise SystemExit(f"FATAL {len(todo)} > cap {a.max_calls}")
ports = itertools.cycle([int(p) for p in a.ports.split(",")])
lock = threading.Lock(); out = open(a.out, "a"); n = [0, 0]; t0 = time.time()
print(f"todo={len(todo)} done={len(done)}", flush=True)


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def work(r):
    body = dict(model="qwen3-32b", messages=r["messages"], max_tokens=32768, seed=r["seed"], **DEC)
    for att in range(2):
        with lock:
            port = next(ports)
        ts = time.time()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as h:
                resp = json.loads(h.read())
            ch = resp["choices"][0]
            rec = dict(unit_id=r["unit_id"], benchmark=r["benchmark"], question_id=r["question_id"], k=r["k"], status="ok",
                       attempt=att, port=port, ts_utc=utc(), latency_s=round(time.time() - ts, 3), request_body=body,
                       final_text=ch["message"].get("content"), reasoning=ch["message"].get("reasoning_content") or ch["message"].get("reasoning"),
                       finish_reason=ch.get("finish_reason"), usage=resp.get("usage"), served=resp.get("model"))
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush(); n[0] += 1
                if n[0] % 200 == 0:
                    el = time.time() - t0
                    print(f"{n[0]}/{len(todo)} fail={n[1]} {el / n[0]:.2f}s/call", flush=True)
            return
        except Exception as e:                                             # noqa: BLE001
            with lock:
                out.write(json.dumps(dict(unit_id=r["unit_id"], status="attempt_failed", attempt=att, port=port, ts_utc=utc(),
                                          error=f"{type(e).__name__}: {e}"[:500])) + "\n"); out.flush()
            time.sleep(5)
    with lock:
        n[1] += 1


with cf.ThreadPoolExecutor(a.concurrency) as ex:
    list(ex.map(work, todo))
print(json.dumps(dict(ok=n[0], failed=n[1], wall_s=round(time.time() - t0, 1), finished=utc())), flush=True)
