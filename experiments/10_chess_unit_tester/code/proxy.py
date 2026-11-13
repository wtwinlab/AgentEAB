r"""Logging and hard-cap proxy between CHESS and the local vLLM server.

    python3 proxy.py --listen 8066 --upstream http://127.0.0.1:8065 --cap 750 \
        --log ${CHESS_WORKDIR}/raw/proxy_calls.jsonl --question-file ${CHESS_WORKDIR}/run/current_question

Every POST to a generation endpoint is one counted call, whatever its outcome (ok, upstream error,
timeout). Once `cap` calls have been forwarded, further generation requests are refused with HTTP
429 and logged as refused, never forwarded. GET /v1/models is passed through uncounted.
Each record: seq, question tag, tool attribution from the prompt template, request body, response
body, usage, HTTP status, start/end time, latency.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

SIGS = (
    ("extract_keywords", "Objective: Analyze the given question and hint to identify and extract keywords"),
    ("generate_candidate", "You are an experienced database expert.\nNow you need to generate a SQL query"),
    ("revise", "You are an SQL database expert tasked with correcting a SQL query"),
    ("generate_unit_test", "generate a set of"),
    ("evaluate", "evaluate whether or not the response passes each unit test"),
    ("output_fixing", "Instructions:\n--------------\n"),
)
AGENT = re.compile(r"You are the (.+?) agent, in a team of agents")


def attribute(body):
    txt = "\n".join(m.get("content") or "" for m in body.get("messages", []) if isinstance(m.get("content"), str))
    m = AGENT.search(txt)
    if m:
        return f"agent_tool_choice:{m.group(1)}"
    for name, sig in SIGS:
        if sig in txt:
            return name
    return "unattributed"


class State:
    lock = threading.Lock()
    seq = 0
    refused = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8066)
    ap.add_argument("--upstream", default="http://127.0.0.1:8065")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--start-count", type=int, default=0, help="calls already spent against the cap")
    ap.add_argument("--log", required=True)
    ap.add_argument("--question-file", required=True)
    a = ap.parse_args()
    State.seq = a.start_count
    logf = open(a.log, "a", buffering=1)

    def qtag():
        try:
            return open(a.question_file).read().strip()
        except OSError:
            return ""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, payload, ctype="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            r = requests.get(a.upstream + self.path, timeout=30)
            self._send(r.status_code, r.content, r.headers.get("Content-Type", "application/json"))

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n)
            try:
                body = json.loads(raw)
            except Exception:                                          # noqa: BLE001
                body = {"_unparsed": raw.decode("utf-8", "replace")}
            with State.lock:
                if State.seq >= a.cap:
                    State.refused += 1
                    rec = dict(seq=None, refused=True, ts=time.time(), question=qtag(), tool=attribute(body),
                               path=self.path, request=body)
                    logf.write(json.dumps(rec) + "\n")
                    self._send(429, {"error": {"message": f"pilot call cap {a.cap} reached", "type": "cap"}})
                    return
                State.seq += 1
                seq = State.seq
            t0 = time.time()
            status, resp, err = None, None, None
            try:
                r = requests.post(a.upstream + self.path, data=raw, headers={"Content-Type": "application/json"}, timeout=1200)
                status = r.status_code
                try:
                    resp = r.json()
                except Exception:                                      # noqa: BLE001
                    resp = {"_text": r.text}
                out = r.content
            except Exception as e:                                     # noqa: BLE001
                status, err = 599, repr(e)
                out = json.dumps({"error": {"message": err}}).encode()
            t1 = time.time()
            rec = dict(seq=seq, refused=False, question=qtag(), tool=attribute(body), path=self.path,
                       ts_start=t0, ts_end=t1, latency_s=round(t1 - t0, 3), status=status, error=err,
                       params={k: v for k, v in body.items() if k != "messages"},
                       request=body, response=resp, usage=(resp or {}).get("usage") if isinstance(resp, dict) else None)
            logf.write(json.dumps(rec) + "\n")
            self._send(status if status != 599 else 502, out)

    srv = ThreadingHTTPServer(("127.0.0.1", a.listen), H)
    print(f"proxy :{a.listen} -> {a.upstream}, cap {a.cap}, start {a.start_count}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
