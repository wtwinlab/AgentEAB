r"""Collector for a hosted OpenAI-compatible recipient, with a complete per-attempt call log.

A sibling of the local-vLLM QA collector, which talks to a local server over a
port. Everything about the record format, the resume rule and the seed derivation is kept identical
so the two sets of raw logs can be scored by the same code. What differs is the transport: a base
URL instead of a port, and a bearer credential.

    export DEEPSEEK_API_KEY=...                  # provider key, read from the environment
    python3 code/collect_hosted.py --input <data>/qa_replay_inputs.jsonl \
        --model deepseek-v4-pro --out output/raw/qa_deepseek-v4-pro.jsonl \
        --max-tokens 32768 --only-units data/units_fold0.txt --max-usd 15

What every raw record holds, so nothing has to be run to recover it:

  request_body   the exact JSON sent, messages included
  config         requested model, decoding values, max_tokens, seed, provider and base URL.
                 `config_sha` is the identity of that whole configuration, so a run under
                 different sampling is a different request rather than a silent overwrite
  response       the raw response object as returned: content, any reasoning field, usage,
                 finish_reason. Nothing is dropped or normalised at this layer
  served         the model id the server reported back, and whether it equals the one requested.
                 Hosted catalogues do substitute a successor for a retired id while still answering,
                 so this is recorded per call rather than assumed once
  timing         start and end timestamps and the measured latency

**The credential is never written.** It is read from the environment, sent in the header, and the
header is not part of any record. `--key-var` names the variable; its value never enters the log.

Append-only and resumable: a unit is skipped only when it already has an `ok` attempt, so a killed
run continues rather than restarting, and failures and retries stay on disk as evidence.

Scoring never happens here. This writes raw only.

**Spend cap.** `--max-usd` is a hard stop, checked against the usage the server reports as the run
proceeds. On breach the run stops launching new calls and writes the ledger. Prices come from
`--usd-per-m-in` / `--usd-per-m-out`, which the caller states explicitly; nothing is inferred.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

#: Decoding per hosted model. Nothing here is tuned on an outcome; a value enters only after a
#: pilot shows the endpoint accepts it, because an unsupported field is a 400 rather than a default.
DECODING = {
    "deepseek-v4-pro": dict(temperature=0.6, top_p=0.95),
    "deepseek-v4-pro-pilot": dict(temperature=0.6, top_p=0.95),
}

PROVIDERS = {
    "deepseek": dict(base_url="https://api.deepseek.com", key_var="DEEPSEEK_API_KEY"),
    "deepinfra": dict(base_url="https://api.deepinfra.com/v1/openai", key_var="deepinfra_api_key"),
}

RETRY_LIMIT = 2
TIMEOUT_S = 1800.0

_lock = threading.Lock()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seed_for(unit: str) -> int:
    """Frozen per-unit seed, identical to the local collector: unit key only."""
    return int(hashlib.sha256(unit.encode()).hexdigest()[:8], 16)


def config_sha(cfg: dict, seed: int) -> str:
    ident = {k: cfg[k] for k in ("model", "provider", "decoding", "max_tokens")}
    ident["seed"] = seed
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]


def usage_of(resp: dict) -> dict:
    u = (resp or {}).get("usage") or {}
    return dict(prompt_tokens=u.get("prompt_tokens") or 0,
                completion_tokens=u.get("completion_tokens") or 0,
                cached_tokens=(u.get("prompt_cache_hit_tokens")
                               or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="deepseek", choices=sorted(PROVIDERS))
    ap.add_argument("--served-model", default="", help="id to request; defaults to --model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--only-units", default="", help="path to a file of unit keys, one per line")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra", default="", help="JSON object merged into the request body")
    ap.add_argument("--usd-per-m-in", type=float, required=True)
    ap.add_argument("--usd-per-m-out", type=float, required=True)
    ap.add_argument("--usd-per-m-cached", type=float, default=0.0)
    ap.add_argument("--max-usd", type=float, required=True)
    a = ap.parse_args()

    prov = PROVIDERS[a.provider]
    key = os.environ.get(prov["key_var"], "")
    if not key:
        print(f"FATAL: {prov['key_var']} is not set in the environment")
        return 1
    requested = a.served_model or a.model
    decoding = dict(DECODING.get(a.model, {}))
    if a.extra:
        decoding.update(json.loads(a.extra))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cfg = dict(model=requested, provider=a.provider, base_url=prov["base_url"],
               decoding=decoding, max_tokens=a.max_tokens)

    rows = [json.loads(l) for l in open(a.input)]
    if a.only_units:
        keep = {l.strip() for l in open(a.only_units) if l.strip()}
        rows = [r for r in rows if r["unit"] in keep]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("status") == "ok":
                done.add(r["unit_id"])
    todo = [r for r in rows if r["unit"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"input={len(rows)} already_ok={len(done)} todo={len(todo)} provider={a.provider} "
          f"model={requested} max_tokens={a.max_tokens} cap=${a.max_usd:.2f} "
          f"decoding={decoding}", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    out = open(a.out, "a")
    counts = dict(ok=0, failed_units=0, attempts=0, mismatched_served=0)
    spend = dict(usd=0.0, prompt=0, completion=0, cached=0)
    stop = threading.Event()
    t_start = time.time()

    def write(rec: dict) -> None:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())

    def charge(u: dict) -> float:
        billed_in = max(0, u["prompt_tokens"] - u["cached_tokens"])
        return (billed_in / 1e6 * a.usd_per_m_in
                + u["cached_tokens"] / 1e6 * a.usd_per_m_cached
                + u["completion_tokens"] / 1e6 * a.usd_per_m_out)

    def work(row: dict) -> None:
        if stop.is_set():
            return
        seed = seed_for(row["unit"])
        body = dict(model=requested, messages=row["messages"], max_tokens=a.max_tokens,
                    seed=seed, **decoding)
        payload = json.dumps(body, sort_keys=True).encode()
        base = dict(unit_id=row["unit"], model=a.model,
                    endpoint=dict(provider=a.provider, base_url=prov["base_url"],
                                  key_var=prov["key_var"]),
                    config=cfg, config_sha=config_sha(cfg, seed), seed=seed,
                    request_body=body, request_sha=hashlib.sha256(payload).hexdigest())
        for attempt in range(RETRY_LIMIT + 1):
            if stop.is_set():
                return
            t0, ts0 = time.time(), utc()
            try:
                req = urllib.request.Request(
                    f"{prov['base_url']}/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                    resp = json.loads(r.read())
                lat = time.time() - t0
                u = usage_of(resp)
                served = resp.get("model")
                ok_served = (served == requested)
                with _lock:
                    spend["usd"] += charge(u)
                    spend["prompt"] += u["prompt_tokens"]
                    spend["completion"] += u["completion_tokens"]
                    spend["cached"] += u["cached_tokens"]
                    write(dict(base, status="ok", attempt=attempt, ts_start_utc=ts0,
                               ts_end_utc=utc(), latency_s=round(lat, 3), response=resp,
                               served=dict(reported=served, requested=requested,
                                           matches=ok_served),
                               usage_billed=u, usd_this_call=round(charge(u), 6), error=None))
                    counts["ok"] += 1
                    counts["attempts"] += 1
                    if not ok_served:
                        counts["mismatched_served"] += 1
                    n = counts["ok"] + counts["failed_units"]
                    if n % 50 == 0 or n == len(todo):
                        el = time.time() - t_start
                        print(f"{n}/{len(todo)} ok={counts['ok']} failed={counts['failed_units']} "
                              f"${spend['usd']:.3f} {el/max(n,1):.2f}s/call "
                              f"eta={(len(todo)-n)*el/max(n,1)/60:.1f}min", flush=True)
                    if spend["usd"] >= a.max_usd and not stop.is_set():
                        stop.set()
                        print(f"STOP: spend ${spend['usd']:.3f} reached the cap "
                              f"${a.max_usd:.2f}; no further calls will be launched", flush=True)
                return
            except Exception as e:                                      # noqa: BLE001
                detail = f"{type(e).__name__}: {e}"
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        detail += " | " + e.read().decode()[:600]
                    except Exception:                                   # noqa: BLE001
                        pass
                with _lock:
                    write(dict(base, status="attempt_failed", attempt=attempt, ts_start_utc=ts0,
                               ts_end_utc=utc(), latency_s=round(time.time() - t0, 3),
                               response=None, served=None, usage_billed=None,
                               usd_this_call=None, error=detail))
                    counts["attempts"] += 1
                time.sleep(2.0 * (attempt + 1))
        with _lock:
            write(dict(base, status="failed", attempt=RETRY_LIMIT, ts_start_utc=utc(),
                       ts_end_utc=utc(), latency_s=None, response=None, served=None,
                       usage_billed=None, usd_this_call=None, error="exhausted retries"))
            counts["failed_units"] += 1

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(work, todo))
    out.close()
    wall = time.time() - t_start
    ledger = dict(
        input=a.input,
        input_sha256=hashlib.sha256(open(a.input, "rb").read()).hexdigest(),
        model=a.model, requested_model=requested, provider=a.provider,
        base_url=prov["base_url"], key_var=prov["key_var"], decoding=decoding,
        out=a.out, max_tokens=a.max_tokens, concurrency=a.concurrency,
        units_attempted=len(todo), units_ok=counts["ok"], units_failed=counts["failed_units"],
        api_calls_billed=counts["attempts"], served_mismatches=counts["mismatched_served"],
        prompt_tokens=spend["prompt"], completion_tokens=spend["completion"],
        cached_prompt_tokens=spend["cached"],
        prices=dict(usd_per_m_in=a.usd_per_m_in, usd_per_m_out=a.usd_per_m_out,
                    usd_per_m_cached=a.usd_per_m_cached),
        usd_estimated=round(spend["usd"], 4), usd_cap=a.max_usd,
        stopped_on_cap=stop.is_set(), wall_s=round(wall, 1), finished_utc=utc(),
        note="usd_estimated applies the stated prices to the usage the server reported; it is not "
             "an invoice")
    with open(a.out + f".ledger.{int(t_start)}.json", "w") as f:
        json.dump(ledger, f, indent=1)
    print(json.dumps(ledger), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
