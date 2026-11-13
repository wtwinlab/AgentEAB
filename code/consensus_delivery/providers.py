"""One generic OpenAI-compatible transport for every model family.

The previous study's native runner selected its client by substring on the model name
(`'gpt' in model` → OpenAI, `'gemini' in model` → vertexai, `'hf' in model` → in-process). Six of the
seven families in the new matrix contain none of those substrings and could not run at all; worse, an
operator could make any checkpoint take the OpenAI branch by naming the served model `...gpt...`.

Here routing is **never** a function of the model string. An endpoint is addressed by an explicit
registry alias, and the alias → (base_url, served name, checkpoint, revision, quantization) mapping is
required to be unique and is asserted before any request. The same `chat()` function serves every model.

No secret is ever written to a log, a report, or a returned record: `ModelEndpoint.api_key_env` names an
environment variable, and only its *name* is recorded.
"""
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
    """A retryable transport-level failure. Never a stance, never a model result."""


class BudgetRefused(RuntimeError):
    """The spend ledger refused to reserve for this request. Never silently downgraded."""


@dataclass(frozen=True)
class ModelEndpoint:
    """An explicitly registered endpoint. Identity is the registry entry, not the name string."""

    alias: str
    base_url: str
    served_name: str
    checkpoint: str
    provider: str                      # "local" | "deepinfra" | ...
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
        """Everything that defines what produced a token. No secret, ever."""
        return {
            "alias": self.alias, "provider": self.provider, "served_name": self.served_name,
            "checkpoint": self.checkpoint, "revision": self.revision or "unknown",
            "quantization": self.quantization or "unknown",
            "chat_template_sha": self.chat_template_sha or "unknown",
            "serving_image": self.serving_image or "unknown",
            "max_context": self.max_context, "max_completion": self.max_completion,
            "api_key_env": self.api_key_env,          # the NAME only
            # Template switches a model requires. Recorded because they are part of the
            # frozen serving configuration (design 5.4.4), not a protocol change.
            "request_extra": dict(self.request_extra) if self.request_extra else None,
        }


class Registry:
    """Alias → endpoint, with uniqueness assertions that run before any request is made."""

    def __init__(self, endpoints: Sequence[ModelEndpoint]) -> None:
        self._by_alias: Dict[str, ModelEndpoint] = {}
        for e in endpoints:
            if e.alias in self._by_alias:
                raise ValueError(f"duplicate alias {e.alias!r}")
            self._by_alias[e.alias] = e
        self.validate()

    def validate(self) -> None:
        """A served name must identify exactly one checkpoint per base_url, so an alias cannot be
        reused to point two different checkpoints at the same endpoint."""
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
    status: str                 # "ok" | "transport_failure" | "budget_refused"
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    finish_reason: Optional[str]
    usd: float
    attempt: int
    latency_s: float
    raw_ref: Optional[str] = None
    #: What the provider says it served, and its own version signal. A hosted alias can move its
    #: weights without notice and `system_fingerprint` is the only notice we get, so it is recorded
    #: per call rather than probed once and assumed stable.
    served_model: Optional[str] = None
    fingerprint: Optional[str] = None
    #: `usage.completion_tokens_details.reasoning_tokens`, where a provider reports it. Reasoning
    #: tokens are billed as output, so recording them is cost accountability; they are also the only
    #: way to tell whether a reasoning parameter was actually honoured. A judge round was lost to a
    #: configuration whose effect could not be observed, and this is what makes it observable.
    reasoning_tokens: Optional[int] = None
    #: `message.reasoning_content` length, when the provider returns the channel separately.
    reasoning_chars: Optional[int] = None
    #: Per-generated-token `{"token", "logprob"}`, when the call asked for them. Two Batch 2
    #: mechanisms need this and neither can be implemented without it: BCCS multiplies token
    #: probabilities into a per-sentence belief, and MAD-M2 masks peers by response perplexity.
    #: Recorded on the call rather than recomputed later, because a token's probability is not
    #: recoverable from its text.
    token_logprobs: Optional[List[Dict[str, Any]]] = None
    #: Per-generated-token top-k alternatives, `[{"token", "logprob", "bytes"}, ...]` per position,
    #: populated only when a caller asks for `top_logprobs > 1`. It keeps the `top_logprobs` array the
    #: provider returns, so a record states what the chosen token was chosen *over*. That is the
    #: difference between "the answer was likely" and "the answer was preferred to the alternative",
    #: and it is not recoverable afterwards. Kept additive so every existing record and every
    #: request hash is unchanged: the request payload is not touched, only what is read back.
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
    """One model call through the single generic route.

    `alias` is looked up in the registry; the model string plays no part in choosing a code path.
    On exhausted retries this raises `TransportFailure`, which the formation and branch layers turn into
    an infrastructure outcome — never into a stance.
    """
    ep = registry.get(alias)
    payload: Dict[str, Any] = {
        "model": ep.served_name,
        "messages": list(messages),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if logprobs:
        # Asked for before the endpoint's own switches, so a registry `request_extra` can still
        # override it if an endpoint ever needs to. Two Batch 2 mechanisms are unimplementable
        # without per-token probabilities, and a token's probability is not recoverable afterwards.
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    # The endpoint's own required template switches come first; a per-call `extra` may add to them
    # but the registry's frozen configuration is what defines the model's serving contract.
    if ep.request_extra:
        payload.update(dict(ep.request_extra))
    if extra:
        payload.update(extra)
    rsha = _request_sha(payload)

    _reservation: Optional[str] = None
    if ledger is not None:
        # Count the prompt through the serving endpoint's own tokenizer when it will answer, so the
        # pre-call reservation is exact on the input side rather than a chars/3 guess. A failure to
        # count is not fatal; it falls back to the estimate and the basis is recorded.
        _exact: Optional[int] = None
        try:
            _exact = int(tokenize_request(ep, payload.get("messages", ()),
                                          timeout_s=min(20.0, timeout_s))["count"])
        except Exception:  # noqa: BLE001
            _exact = None
        # Reserve the maximum this request could cost *before* sending it, across
        # every attempt the retry loop is allowed to make.
        _reservation = ledger.reserve(alias, ep, payload, max_tokens, attempts=retries + 1,
                                      prompt_tokens=_exact)

    # A reservation that is neither settled nor released is budget consumed for ever. Before this
    # `finally`, six interrupted requests costing $0.00036 permanently held 48% of a $0.50 cap, and
    # the release was only reached when the retry loop ran to exhaustion.
    settled = False
    sent = False
    last_err: Optional[Exception] = None
    try:
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                # Once this line is reached the request is on the wire; from here on its cost is
                # unknown rather than zero.
                sent = True
                body = (_transport or _http_post)(ep, payload, timeout_s)
                dt = time.time() - t0
                usage = body.get("usage") or {}
                pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
                details = usage.get("completion_tokens_details") or {}
                rtok = details.get("reasoning_tokens")
                choice = (body.get("choices") or [{}])[0]
                # `choices[0].logprobs.content[]` on the OpenAI-compatible route. The chosen token
                # and its logprob are always kept. The per-position `top_logprobs` array is kept
                # too whenever the caller asked for more than one alternative, because a record of
                # a token's probability without its competitors cannot answer which candidate the
                # model preferred -- the quantity E6 is built on. It stays `None` for the
                # single-alternative default, so records written before this existed are unchanged.
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
                    # Check the declared input ceiling against what was actually billed before
                    # settling, so a wrong ceiling stops the run instead of being paid quietly.
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
                # A refusal is a budget decision, never a transport failure. Downgrading it meant the
                # run retried a refused request three times and then reported it as a network error.
                raise
            except TransportFailure as e:
                last_err = e
                if ledger is not None:
                    # A failed attempt may still have been generated and billed upstream. We cannot
                    # know its cost, so it is recorded rather than assumed free.
                    ledger.note_unbilled_attempt(alias, _reservation, attempt, str(e))
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
            except Exception as e:  # noqa: BLE001 — anything else is also transport from our side
                last_err = TransportFailure(f"{type(e).__name__}: {e}")
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
        raise TransportFailure(f"{alias}: retries exhausted ({last_err})")
    finally:
        if ledger is not None and not settled:
            if sent:
                # Sent, outcome unknown: the worst case stays committed. Releasing it here is what
                # left `committed` at exactly $0.000000 after six interrupted, possibly-billed calls.
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
        # Loaded from `.env` by the program, here, at the one place the value is used. Exporting it
        # in a shell first would put the secret in a shell history and in the process listing of
        # whatever set it; `credentials.ensure` keeps it inside this process.
        from .credentials import ensure  # noqa: PLC0415 — import-time cycle otherwise
        key, _source = ensure(ep.api_key_env)
        if not key:
            raise BudgetRefused(
                f"{ep.alias}: {ep.api_key_env} is not in the environment and not in .env; "
                f"no request is made")
        headers["Authorization"] = f"Bearer {key}"   # never logged
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise TransportFailure(f"HTTP {e.code}") from None
    except Exception as e:  # noqa: BLE001
        raise TransportFailure(f"{type(e).__name__}") from None


def tokenize_request(ep: ModelEndpoint, messages: Sequence[Mapping[str, str]],
                     *, request_extra: Optional[Mapping[str, Any]] = None,
                     timeout_s: float = 20.0, _transport: Any = None) -> Dict[str, Any]:
    """Ask the **serving endpoint** to tokenise this exact message list.

    vLLM's `/tokenize` applies the template the server will actually use, including the generation
    prompt and any `chat_template_kwargs`. A locally loaded tokenizer can disagree with the served
    template — different revision, different `chat_template.jinja`, a thinking-mode switch compiled
    into one and not the other — and the disagreement is silent. The served count is not an estimate
    of the billed prompt: measured on `gpt-oss-20b`, `/tokenize` returned `count: 74` for a message
    list that a completion call then billed as `prompt_tokens: 74`.

    `_transport` is the injection seam, so the offline suite exercises this path without a socket.
    Raises `TransportFailure` when the endpoint cannot answer; the caller decides what that means.
    """
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
    """POST to an endpoint path that is not `/chat/completions`.

    `/tokenize` sits beside `/v1`, not under it, on a vLLM server: the OpenAI-compatible routes are
    mounted at `/v1` while the tokenizer routes are at the root. Appending the path to `base_url`
    (which ends in `/v1`) produced a 404 that read as an unreachable tokenizer.
    """
    root = ep.base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    url = root.rstrip("/") + path
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if ep.api_key_env:
        # Loaded from `.env` by the program, here, at the one place the value is used. Exporting it
        # in a shell first would put the secret in a shell history and in the process listing of
        # whatever set it; `credentials.ensure` keeps it inside this process.
        from .credentials import ensure  # noqa: PLC0415 — import-time cycle otherwise
        key, _source = ensure(ep.api_key_env)
        if not key:
            raise BudgetRefused(
                f"{ep.alias}: {ep.api_key_env} is not in the environment and not in .env; "
                f"no request is made")
        headers["Authorization"] = f"Bearer {key}"   # never logged
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise TransportFailure(f"HTTP {e.code} on {path}") from None
    except Exception as e:  # noqa: BLE001
        raise TransportFailure(f"{type(e).__name__} on {path}") from None


#: Tokens allowed for chat-template scaffolding that appears in no message. **Declared, not
#: measured.** The measured figure on DeepSeek was 28-30 tokens fixed plus about 2 per message, so
#: this is roughly 60x headroom. A declared ceiling plus enforced detection is a guarantee; five
#: agreeing measurements are an estimate, and an estimate is what `chars/3` was.
TEMPLATE_CEILING_TOKENS = 2048

#: Concurrent in-flight paid requests. Every one of them holds a reservation, so the cap is only
#: honoured if the number of simultaneous holders is bounded.
MAX_PAID_INFLIGHT = 4


class SpendLedger:
    """Durable, reservation-first spend ledger with a hard cap.

    A request reserves its maximum possible cost before it is sent, so concurrent in-flight requests
    cannot jointly overshoot. Checking the balance only after the call is explicitly not allowed.

    **Every mutation holds `_lock`, and a reservation is identified by the token `reserve` returns.**
    The previous version kept a single `_last_token` attribute and `settle` released whatever it
    happened to hold, which is correct for one request at a time and wrong for two: with A and B in
    flight, A's settle released **B's** reservation, leaving A's reserved for ever and B's cost
    uncommitted — so the cap could be overshot in exactly the configuration the cap exists for. The
    cap check and the reservation also have to be one critical section, or two callers both pass a
    check that only one of them should.
    """

    def __init__(self, path: str, cap_usd: float) -> None:
        self.path = path
        self.cap_usd = float(cap_usd)
        self._settled = 0.0
        self._reserved: Dict[str, float] = {}
        self._n = 0
        self._lock = threading.Lock()
        self._seq = 0
        #: A per-ledger-instance nonce in every token. Without it the token was `alias:seq` with
        #: `seq` restarting at 0 in each process, so the **second** process's first reservation
        #: carried the same name as the first process's — and its `settle` released the earlier
        #: run's orphaned reservation instead of its own. Measured on this project's own ledger:
        #: 194 of 331 token names had been issued more than once, and an interrupted request's
        #: $0.076 worst-case hold was cleared by an unrelated later settle. The cap itself was
        #: never at risk, because the check at `reserve` is per-process and settled spend is
        #: exact; what the collision defeated is the one thing the unknown-cost hold exists to
        #: guarantee, which is that a possibly-billed request stays committed until it is
        #: reconciled.
        self._run = uuid.uuid4().hex[:8]
        self.breached = False
        #: Requests that were **sent** and whose cost is unknown — interrupted, timed out, or failed
        #: after the provider may already have generated. Their worst case stays committed. Releasing
        #: them treated a possibly-billed call as free: six interrupted requests left `committed` at
        #: exactly $0.000000.
        self._unknown: Dict[str, float] = {}
        #: The input-token ceiling each reservation assumed, so it can be checked against billing.
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
            # Reservations with neither a settle nor a release were in flight when the process died.
            # Dropping them on reload made a crash-and-resume **under**-count committed spend, which
            # is the direction that overshoots a cap.
            self._reserved.update(outstanding)
            if outstanding:
                self._append({"type": "reload_outstanding", "n": len(outstanding),
                              "usd_max": round(sum(outstanding.values()), 6)})

    @property
    def committed(self) -> float:
        """Settled spend, live reservations **and** unknown-cost holds. All three are real exposure."""
        with self._lock:
            return self._settled + sum(self._reserved.values()) + sum(self._unknown.values())

    def remaining(self) -> float:
        return self.cap_usd - self.committed

    def reserve(self, alias: str, ep: ModelEndpoint, payload: Mapping[str, Any],
                max_tokens: int, attempts: int = 1,
                prompt_tokens: Optional[int] = None) -> Optional[str]:
        """Reserve this request's worst-case cost across **every attempt**. Returns its token.

        `attempts` is `retries + 1`: one reservation covers the whole retry loop, because a provider
        that generated and billed and then returned a 502 bills per attempt. Reserving one attempt's
        cost for a loop of three under-reserves by 3×.
        """
        if ep.provider == "local":
            return None
        if ep.usd_per_m_in is None or ep.usd_per_m_out is None:
            raise BudgetRefused(f"{alias}: unverified price; request refused")
        # The input side is **exact** when the caller has counted it through the serving endpoint's
        # own tokenizer. The chars/3 fallback under-counts CJK text by about 3x
        # and is wrong by an unbounded factor for a provider that bills outside `max_tokens`; an
        # alarm raised in `settle` is a post-hoc detection, not a pre-call guarantee, so the
        # guarantee has to come from counting the prompt before the request goes out.
        if prompt_tokens is not None:
            n_in, basis = float(prompt_tokens), "counted"
        else:
            # **A conservative upper bound, not an estimate.** Not every provider exposes a
            # tokenizer: DeepSeek's official API has no `/tokenize` and the vLLM route cannot be
            # assumed of it. `chars/3` is an *estimate* and it under-counts — measured 4.8x low on
            # CJK text, where it would have admitted an over-cap request.
            #
            # Two parts, because the message bytes alone are **not** a bound. A byte-level BPE cannot
            # emit more tokens than a string has UTF-8 bytes, so bytes bound the *content*. But the
            # chat template adds scaffolding that is in no message: measured against DeepSeek, a 2-byte prompt billed **31** tokens, and a 4-message 8-byte prompt billed
            # 38. So a fixed allowance plus a per-message allowance is added on top.
            #
            # Measured (`deepseek-flash`): bytes/msgs -> billed tokens = 2/1 -> 31, 400/1 -> 80,
            # 399(CJK)/1 -> 163, 4/2 -> 32, 8/4 -> 38. The bound below yields 130, 528, 527, 196,
            # 328 and holds on every one with margin. It over-reserves heavily; that is the price of
            # a guarantee where no tokenizer is published.
            # Content bytes bound content tokens: a byte-level BPE token covers at least one byte.
            # The template scaffolding is in no message and cannot be counted, so a **declared**
            # ceiling is added and the whole thing is capped by the endpoint's context limit. The
            # reservation is then checked against what the provider actually bills, and a request
            # that exceeds it is recorded as a breach rather than absorbed.
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
        """A billed prompt larger than the reservation assumed is a breach of the guarantee.

        The input side is a declared ceiling, not a count, so the only thing that makes it a guarantee
        is detecting when it is wrong. Recorded and raised; not absorbed.
        """
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
        """Record what was actually billed, and refuse to be quiet if it exceeds the cap.

        The reservation bounds an **estimate** — input tokens are estimated from characters, and no
        estimate bounds a provider that bills reasoning or cached tokens outside `max_tokens`. So the
        settled total is checked here too. Measured before this check: with an endpoint reporting
        500 000 prompt tokens against a 0.05 USD cap, `committed` reached 2.0002 USD and `remaining()`
        returned -1.95 USD with nothing raised, warned or recorded.
        """
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
        """An attempt that failed after the provider may already have generated and billed.

        The reservation covers it, but the real cost is unknowable from our side, so it is recorded
        instead of being silently treated as zero. Measured: three attempts against a provider that
        billed each one left $0.08 of $0.12 unrecorded.
        """
        self._append({"type": "unbilled_attempt", "alias": alias, "token": token,
                      "attempt": attempt, "why": why[:120]})

    def release(self, alias: str, token: Optional[str] = None) -> None:
        """Release a reservation for a request that was **never sent**. Nothing can have been billed."""
        with self._lock:
            if token is not None:
                self._reserved.pop(token, None)
        self._append({"type": "release", "alias": alias, "token": token})

    def hold_unknown(self, alias: str, token: Optional[str], why: str) -> None:
        """Keep a sent request's worst case committed when its real cost cannot be known.

        `unbilled_attempt` records that an attempt *may* have been billed; it must not then be
        released as free. The hold stays until `reconcile_unknown` is given a real figure from the
        provider's own accounting, so an unreconciled run reports more exposure than it can prove —
        which is the only safe direction for a hard cap.
        """
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
        """Replace an unknown-cost hold with a figure from the provider's own accounting."""
        with self._lock:
            if self._unknown.pop(token, None) is None:
                return
            self._settled += usd
            self._n += 1
        self._append({"type": "reconcile", "alias": alias, "token": token, "usd": usd})

    def exposure(self) -> Dict[str, Any]:
        """What this run can and cannot prove about its own spend."""
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
        # One lock and one write call per line. Two concurrent appends to a text handle can interleave
        # inside a line and produce a record that parses as neither.
        with self._append_lock, open(self.path, "a") as f:
            f.write(line)
