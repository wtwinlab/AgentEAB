r"""Shared client, record schema and parsing for the W3/W6 collection.

Every call is recorded with model and revision, the full messages, the raw response including any
reasoning channel, the parse, usage, finish_reason, latency, attempt index and the
task/condition/seat/seed keys. Truncations and errors stay in the record and in the denominator.
No secret is recorded: these are local vLLM endpoints with no credentials.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]
RAW = WORK / "raw"
OUT = WORK / "output"
LOGS = WORK / "logs"
DATA = WORK / "data"


def input_path(name: str) -> pathlib.Path:
    """Resolve a frozen input: the local copy under `data/` if present, else the repository.

    Collection runs on the GPU host, where the repository tree is not mounted; analysis runs in
    the repository. Both read the same bytes, and the sha256 is recorded in the run ledger.
    """
    local = DATA / name
    if local.is_file():
        return local
    for cand in (REPO / "experiments" / "08_cross_family" / "data" / name,
                 REPO / "data" / "derived" / "substantive_completion" / "data" / name):
        if cand.is_file():
            return cand
    raise FileNotFoundError(name)


def code_root() -> pathlib.Path:
    """Where `consensus_delivery` lives.

    The scorers and the answer extractor are the project's existing trusted implementations and are
    not reimplemented here. Collection runs on the GPU host, which has the package under
    `pam_server_run/code`; analysis runs in the repository, which has it under `code`.
    """
    for cand in (REPO / "code", pathlib.Path("${REPO_ROOT}/code")):
        if (cand / "consensus_delivery" / "adapters").is_dir():
            return cand
    raise FileNotFoundError("consensus_delivery not found")

HOST = "127.0.0.1"
#: one server per role
PORTS = {"qa": 8031, "selector": 8033}

#: the recorded serving configuration of the original cross-family run, reused unchanged so the
#: new-source rows are comparable with the reused old-source rows
DECODING = {
    "qwen3-32b": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "gpt-oss-120b": {"temperature": 1.0, "reasoning_effort": "medium"},
    "gemma-4-26b": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
}
SNAPSHOT = {
    "qwen3-32b": "9216db5781bf21249d130ec9da846c4624c16137",
    "gpt-oss-120b": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    "gemma-4-26b": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
}
QA_MAX_TOKENS = 32768


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def call(port: int, model: str, messages: list, max_tokens: int, decoding: dict,
         seed: int, attempt: int, timeout: float = 1800.0) -> dict:
    """One chat completion. Returns the full record; never raises on a model-side failure."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "seed": seed}
    body.update(decoding)
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://{HOST}:{port}/v1/chat/completions", data=raw,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    rec = {"attempt": attempt, "seed": seed, "request_body": body,
           "request_sha": sha(json.dumps(body, sort_keys=True)),
           "endpoint": {"host": HOST, "port": port},
           "ts_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rec["response"] = json.loads(r.read().decode("utf-8"))
            rec["status"] = "ok"
            rec["error"] = None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
        detail = ""
        if isinstance(e, urllib.error.HTTPError):
            try:
                detail = e.read().decode("utf-8", "replace")[:800]
            except OSError:
                detail = ""
        rec["response"] = None
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e} {detail}".strip()
    rec["latency_s"] = round(time.time() - t0, 3)
    rec["ts_end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return rec


def final_channel(rec: dict) -> str:
    """The answer-bearing text.

    A server with a reasoning parser puts thinking in `reasoning_content`; one without it leaves
    the thinking inside `content`, where it has to be stripped to the last close tag. Both are
    handled, and the raw response keeps everything either way.
    """
    if not rec or rec.get("status") != "ok" or not rec.get("response"):
        return ""
    msg = (rec["response"].get("choices") or [{}])[0].get("message") or {}
    c = msg.get("content") or ""
    for tag in ("</think>", "</thinking>"):
        if tag in c:
            c = c[c.rindex(tag) + len(tag):]
    return c.strip()


def finish_reason(rec: dict):
    if not rec or rec.get("status") != "ok" or not rec.get("response"):
        return None
    return (rec["response"].get("choices") or [{}])[0].get("finish_reason")


def usage(rec: dict) -> dict:
    if not rec or rec.get("status") != "ok" or not rec.get("response"):
        return {}
    return rec["response"].get("usage") or {}


def load_done(path: pathlib.Path, key: str = "unit_id") -> set:
    """Keys already recorded with status ok, so a resume fills only what is missing."""
    done = set()
    if not path.is_file():
        return done
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok" and r.get(key) is not None:
                done.add(r[key])
    return done


def append(path: pathlib.Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def wait_ready(port: int, tries: int = 240, gap: float = 15.0) -> bool:
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://{HOST}:{port}/v1/models", timeout=8) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(gap)
    return False
