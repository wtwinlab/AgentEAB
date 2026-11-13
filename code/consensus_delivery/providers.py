from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class TransportFailure(RuntimeError):
    pass


class BudgetRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelEndpoint:

    alias: str
    base_url: str
    served_name: str
    checkpoint: str
    provider: str
    revision: Optional[str] = None
    quantization: Optional[str] = None
    chat_template_sha: Optional[str] = None
    serving_image: Optional[str] = None
    api_key_env: Optional[str] = None
    usd_per_m_in: Optional[float] = None
    usd_per_m_out: Optional[float] = None
    max_context: int = 16384
    max_completion: int = 4096
    request_extra: Optional[Mapping[str, Any]] = None
    notes: str = ""

    def identity(self) -> Dict[str, Any]:
        return {
            "alias": self.alias, "provider": self.provider, "served_name": self.served_name,
            "checkpoint": self.checkpoint, "revision": self.revision or "unknown",
            "quantization": self.quantization or "unknown",
            "chat_template_sha": self.chat_template_sha or "unknown",
            "serving_image": self.serving_image or "unknown",
            "max_context": self.max_context, "max_completion": self.max_completion,
            "api_key_env": self.api_key_env,
            "request_extra": dict(self.request_extra) if self.request_extra else None,
        }


class Registry:

    def __init__(self, endpoints: Sequence[ModelEndpoint]) -> None:
        self._by_alias: Dict[str, ModelEndpoint] = {}
        for e in endpoints:
            if e.alias in self._by_alias:
                raise ValueError(f"duplicate alias {e.alias!r}")
            self._by_alias[e.alias] = e
        self.validate()

    def validate(self) -> None:
        seen: Dict[Tuple[str, str], str] = {}
        for e in self._by_alias.values():
            k = (e.base_url, e.served_name)
            if k in seen and seen[k] != e.checkpoint:
                raise ValueError(
                    f"{e.base_url} serves {e.served_name!r} as both {seen[k]!r} and {e.checkpoint!r}")
            seen[k] = e.checkpoint

    def get(self, alias: str) -> ModelEndpoint:
        if alias not in self._by_alias:
            raise KeyError(f"alias {alias!r} is not registered; routing is never inferred from a name")
        return self._by_alias[alias]

    def aliases(self) -> Tuple[str, ...]:
        return tuple(sorted(self._by_alias))

    def manifest(self) -> Dict[str, Any]:
        return {a: self._by_alias[a].identity() for a in self.aliases()}


@dataclass
class CallRecord:
    alias: str
    request_sha: str
    status: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    finish_reason: Optional[str]
    usd: float
    attempt: int
    latency_s: float
    raw_ref: Optional[str] = None
    served_model: Optional[str] = None
    fingerprint: Optional[str] = None
    reasoning_tokens: Optional[int] = None
    reasoning_chars: Optional[int] = None
    token_logprobs: Optional[List[Dict[str, Any]]] = None
    token_top_logprobs: Optional[List[List[Dict[str, Any]]]] = None


def _request_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def chat(
    *,
    registry: Registry,
    alias: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    ledger: Optional["SpendLedger"] = None,
    retries: int = 2,
    timeout_s: float = 300.0,
    extra: Optional[Mapping[str, Any]] = None,
    logprobs: bool = False,
    top_logprobs: int = 1,
    _transport=None,
) -> Tuple[str, CallRecord]:
    ep = registry.get(alias)
    payload: Dict[str, Any] = {
        "model": ep.served_name,
        "messages": list(messages),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    if ep.request_extra:
        payload.update(dict(ep.request_extra))
    if extra:
        payload.update(extra)
    rsha = _request_sha(payload)

    _reservation: Optional[str] = None
    if ledger is not None:
        _exact: Optional[int] = None
        try:
            _exact = int(tokenize_request(ep, payload.get("messages", ()),
                                          timeout_s=min(20.0, timeout_s))["count"])
        except Exception:
            _exact = None
        _reservation = ledger.reserve(alias, ep, payload, max_tokens, attempts=retries + 1,
                                      prompt_tokens=_exact)

    settled = False
    sent = False
    last_err: Optional[Exception] = None
    try:
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                sent = True
                body = (_transport or _http_post)(ep, payload, timeout_s)
                dt = time.time() - t0
                usage = body.get("usage") or {}
                pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
                details = usage.get("completion_tokens_details") or {}
                rtok = details.get("reasoning_tokens")
                choice = (body.get("choices") or [{}])[0]
                tlp = None
                ttop = None
                lp = choice.get("logprobs") or {}
                if isinstance(lp, dict) and lp.get("content"):
                    tlp = [{"token": e.get("token"), "logprob": e.get("logprob")}
                           for e in lp["content"]]
                    if top_logprobs > 1:
                        ttop = [[{"token": c.get("token"), "logprob": c.get("logprob"),
                                  "bytes": c.get("bytes")}
                                 for c in (e.get("top_logprobs") or [])]
                                for e in lp["content"]]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                rchars = len(message.get("reasoning_content") or "")
                finish = choice.get("finish_reason")
                usd = _price(ep, pt, ct)
                if ledger is not None:
                    ledger.check_input_bound(alias, _reservation, pt)
                    ledger.settle(alias, usd, _reservation)
                    settled = True
                return content, CallRecord(alias, rsha, "ok", pt, ct, finish, usd, attempt, dt,
                                           served_model=body.get("model"),
                                           fingerprint=body.get("system_fingerprint"),
                                           reasoning_tokens=rtok, reasoning_chars=rchars,
                                           token_logprobs=tlp,
                                           token_top_logprobs=ttop)
            except BudgetRefused:
                raise
            except TransportFailure as e:
                last_err = e
                if ledger is not None:
                    ledger.note_unbilled_attempt(alias, _reservation, attempt, str(e))
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
            except Exception as e:
                last_err = TransportFailure(f"{type(e).__name__}: {e}")
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
        raise TransportFailure(f"{alias}: retries exhausted ({last_err})")
    finally:
        if ledger is not None and not settled:
            if sent:
                ledger.hold_unknown(alias, _reservation,
                                    f"request sent, no settled cost ({last_err})")
            else:
                ledger.release(alias, _reservation)


def _price(ep: ModelEndpoint, pt: Optional[int], ct: Optional[int]) -> float:
    if ep.provider == "local":
        return 0.0
    if ep.usd_per_m_in is None or ep.usd_per_m_out is None:
        raise BudgetRefused(f"{ep.alias}: no verified price; a request must not be billed blind")
    return ((pt or 0) / 1e6) * ep.usd_per_m_in + ((ct or 0) / 1e6) * ep.usd_per_m_out


def _http_post(ep: ModelEndpoint, payload: Mapping[str, Any], timeout_s: float) -> Dict[str, Any]:
    url = ep.base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if ep.api_key_env:
        from .credentials import ensure
        key, _source = ensure(ep.api_key_env)
        if not key:
            raise BudgetRefused(
                f"{ep.alias}: {ep.api_key_env} is not in the environment and not in .env; "
                f"no request is made")
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise TransportFailure(f"HTTP {e.code}") from None
    except Exception as e:
        raise TransportFailure(f"{type(e).__name__}") from None


def tokenize_request(ep: ModelEndpoint, messages: Sequence[Mapping[str, str]],
                     *, request_extra: Optional[Mapping[str, Any]] = None,
                     timeout_s: float = 20.0, _transport: Any = None) -> Dict[str, Any]:
    extra = dict(ep.request_extra or {})
    if request_extra:
        extra.update(request_extra)
    payload: Dict[str, Any] = {"model": ep.served_name, "messages": list(messages),
                               "add_generation_prompt": True}
    if extra.get("chat_template_kwargs"):
        payload["chat_template_kwargs"] = dict(extra["chat_template_kwargs"])
    body = (_transport or _http_post_path)(ep, "/tokenize", payload, timeout_s)
    if not isinstance(body, Mapping) or "count" not in body:
        raise TransportFailure("tokenize response has no count")
    return {"count": int(body["count"]), "max_model_len": body.get("max_model_len"),
            "source": "served-tokenize"}


def _http_post_path(ep: ModelEndpoint, path: str, payload: Mapping[str, Any],
                    timeout_s: float) -> Dict[str, Any]:
    root = ep.base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    url = root.rstrip("/") + path
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if ep.api_key_env:
        from .credentials import ensure
        key, _source = ensure(ep.api_key_env)
        if not key:
            raise BudgetRefused(
                f"{ep.alias}: {ep.api_key_env} is not in the environment and not in .env; "
                f"no request is made")
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise TransportFailure(f"HTTP {e.code} on {path}") from None
    except Exception as e:
        raise TransportFailure(f"{type(e).__name__} on {path}") from None


TEMPLATE_CEILING_TOKENS = 2048

MAX_PAID_INFLIGHT = 4


class SpendLedger:

    def __init__(self, path: str, cap_usd: float) -> None:
        self.path = path
        self.cap_usd = float(cap_usd)
        self._settled = 0.0
        self._reserved: Dict[str, float] = {}
        self._n = 0
        self._lock = threading.Lock()
        self._seq = 0
        self._run = uuid.uuid4().hex[:8]
        self.breached = False
        self._unknown: Dict[str, float] = {}
        self._input_tokens: Dict[str, float] = {}
        self._inflight = threading.BoundedSemaphore(MAX_PAID_INFLIGHT)
        if os.path.exists(path):
            outstanding: Dict[str, float] = {}
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    kind, tok = r.get("type"), r.get("token")
                    if kind == "settle":
                        self._settled += float(r.get("usd") or 0.0)
                        self._n += 1
                        outstanding.pop(tok, None)
                    elif kind == "reserve":
                        outstanding[tok] = float(r.get("usd_max") or 0.0)
                    elif kind == "release":
                        outstanding.pop(tok, None)
                    elif kind == "unknown_hold":
                        outstanding[tok] = float(r.get("usd_max") or 0.0)
                    elif kind == "reconcile":
                        self._settled += float(r.get("usd") or 0.0)
                        self._n += 1
                        outstanding.pop(tok, None)
            self._reserved.update(outstanding)
            if outstanding:
                self._append({"type": "reload_outstanding", "n": len(outstanding),
                              "usd_max": round(sum(outstanding.values()), 6)})

    @property
    def committed(self) -> float:
        with self._lock:
            return self._settled + sum(self._reserved.values()) + sum(self._unknown.values())

    def remaining(self) -> float:
        return self.cap_usd - self.committed

    def reserve(self, alias: str, ep: ModelEndpoint, payload: Mapping[str, Any],
                max_tokens: int, attempts: int = 1,
                prompt_tokens: Optional[int] = None) -> Optional[str]:
        if ep.provider == "local":
            return None
        if ep.usd_per_m_in is None or ep.usd_per_m_out is None:
            raise BudgetRefused(f"{alias}: unverified price; request refused")
        if prompt_tokens is not None:
            n_in, basis = float(prompt_tokens), "counted"
        else:
            msgs = payload.get("messages", []) or []
            content_bytes = sum(len(str(m.get("content", "")).encode("utf-8")) for m in msgs)
            n_in = float(min(ep.max_context, content_bytes + TEMPLATE_CEILING_TOKENS))
            basis = "content-bytes+declared-template-ceiling"
        worst = ((n_in / 1e6) * ep.usd_per_m_in
                 + (max_tokens / 1e6) * ep.usd_per_m_out) * max(1, int(attempts))
        with self._lock:
            committed = (self._settled + sum(self._reserved.values())
                         + sum(self._unknown.values()))
            if committed + worst > self.cap_usd:
                raise BudgetRefused(
                    f"{alias}: reserving ${worst:.4f} would exceed the ${self.cap_usd:.2f} cap "
                    f"(committed ${committed:.4f})")
            self._seq += 1
            token = f"{alias}:{self._run}:{self._seq}"
            self._reserved[token] = worst
            self._input_tokens[token] = n_in
        self._append({"type": "reserve", "alias": alias, "usd_max": worst, "token": token,
                      "input_basis": basis})
        return token

    def check_input_bound(self, alias: str, token: Optional[str], billed_prompt_tokens: Optional[int]
                          ) -> None:
        if token is None or billed_prompt_tokens is None:
            return
        assumed = self._input_tokens.get(token)
        if assumed is not None and billed_prompt_tokens > assumed:
            self._append({"type": "input_bound_exceeded", "alias": alias, "token": token,
                          "assumed": assumed, "billed": billed_prompt_tokens})
            self.breached = True
            raise BudgetRefused(
                f"{alias}: the provider billed {billed_prompt_tokens} prompt tokens against a "
                f"reserved ceiling of {assumed}. The input bound is a declared ceiling, so this is a "
                f"breach of the spend guarantee and the run stops rather than continuing on a bound "
                f"that has been shown wrong.")

    def settle(self, alias: str, usd: float, token: Optional[str] = None) -> None:
        with self._lock:
            if token is not None:
                self._reserved.pop(token, None)
            self._unknown.pop(token, None)
            self._input_tokens.pop(token, None)
            self._settled += usd
            self._n += 1
            settled = self._settled
            committed = settled + sum(self._reserved.values()) + sum(self._unknown.values())
        self._append({"type": "settle", "alias": alias, "usd": usd, "token": token})
        if settled > self.cap_usd:
            self._append({"type": "cap_breached", "alias": alias, "settled": settled,
                          "cap_usd": self.cap_usd})
            self.breached = True
            raise BudgetRefused(
                f"{alias}: settled spend ${settled:.4f} exceeds the ${self.cap_usd:.2f} cap "
                f"(committed ${committed:.4f}). The request was already billed; this stops the run "
                f"rather than continuing past the cap.")

    def note_unbilled_attempt(self, alias: str, token: Optional[str], attempt: int,
                              why: str) -> None:
        self._append({"type": "unbilled_attempt", "alias": alias, "token": token,
                      "attempt": attempt, "why": why[:120]})

    def release(self, alias: str, token: Optional[str] = None) -> None:
        with self._lock:
            if token is not None:
                self._reserved.pop(token, None)
        self._append({"type": "release", "alias": alias, "token": token})

    def hold_unknown(self, alias: str, token: Optional[str], why: str) -> None:
        if token is None:
            return
        with self._lock:
            amount = self._reserved.pop(token, None)
            if amount is None:
                return
            self._unknown[token] = amount
        self._append({"type": "unknown_hold", "alias": alias, "token": token,
                      "usd_max": amount, "why": why[:120]})

    def reconcile_unknown(self, alias: str, token: str, usd: float) -> None:
        with self._lock:
            if self._unknown.pop(token, None) is None:
                return
            self._settled += usd
            self._n += 1
        self._append({"type": "reconcile", "alias": alias, "token": token, "usd": usd})

    def exposure(self) -> Dict[str, Any]:
        with self._lock:
            return {"cap_usd": self.cap_usd, "settled": round(self._settled, 6),
                    "reserved": round(sum(self._reserved.values()), 6),
                    "unknown_holds": len(self._unknown),
                    "unknown_usd_max": round(sum(self._unknown.values()), 6),
                    "committed": round(self._settled + sum(self._reserved.values())
                                       + sum(self._unknown.values()), 6),
                    "breached": self.breached}

    _append_lock = threading.Lock()

    def _append(self, rec: Mapping[str, Any]) -> None:
        line = json.dumps({**rec, "ts": time.time()}) + "\n"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._append_lock, open(self.path, "a") as f:
            f.write(line)
