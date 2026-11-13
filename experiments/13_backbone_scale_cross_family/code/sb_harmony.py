r"""Final-channel logprob slicing for gpt-oss.

vLLM returns per-token logprobs over the whole harmony stream:
  <|channel|>analysis<|message|> ...reasoning... <|end|><|start|>assistant<|channel|>final<|message|> ...answer... <|return|>
`final_slice` keeps the tokens after the LAST `<|channel|>` `final` `<|message|>` marker, up to (not
including) the next `<|return|>` / `<|end|>` / `<|call|>`. MAD-M² perplexity and BCCS beliefs are then
computed on the delivered text only. `None` means the boundary is not unambiguous.
"""
from __future__ import annotations

END = {"<|return|>", "<|end|>", "<|call|>"}


def final_slice(entries):
    toks = [e.get("token", "") for e in entries or []]
    start = None
    for i in range(len(toks) - 2):
        if toks[i] == "<|channel|>" and toks[i + 1].strip() == "final" and toks[i + 2] == "<|message|>":
            start = i + 3
    if start is None:
        return None
    out = []
    for e in entries[start:]:
        if e.get("token", "") in END:
            break
        out.append(e)
    return out


def check(entries, content):
    """(a) marker present, (b) decoded slice equals content (exact, or after whitespace normalisation),
    (c) slice non-empty."""
    sl = final_slice(entries)
    a = sl is not None
    dec = "".join(e.get("token", "") for e in (sl or []))
    exact = dec == (content or "")
    norm = " ".join(dec.split()) == " ".join((content or "").split())
    return dict(marker=a, exact=exact, ws_equal=norm, nonempty=bool(sl), n_slice=len(sl or []),
                n_all=len(entries or []))


def install():
    """Wrap providers.chat so gpt-oss calls carry final-channel logprobs only."""
    from consensus_delivery import providers as PV
    real = PV.chat

    def chat(*a, **kw):
        text, rec = real(*a, **kw)
        if str(kw.get("alias", "")).startswith("gptoss") and rec.token_logprobs:
            sl = final_slice(rec.token_logprobs)
            rec.token_logprobs = sl if sl is not None else []      # ambiguous -> no belief signal
        return text, rec

    PV.chat = chat


# ---- A14: MiniMax-M2.5 think-slice (same idea as A11): keep logprob entries after the last token
# that contains `</think>`; the delivered content is what follows the reasoning.
def think_slice(entries):
    toks = [e.get("token", "") for e in entries or []]
    idx = max((i for i, t in enumerate(toks) if "</think>" in t), default=None)
    if idx is None:
        return None
    return list(entries[idx + 1:])


def check_think(entries, content):
    sl = think_slice(entries)
    dec = "".join(e.get("token", "") for e in (sl or []))
    return dict(marker=sl is not None, exact=dec == (content or ""),
                ws_equal=" ".join(dec.split()) == " ".join((content or "").split()),
                nonempty=bool(sl), n_slice=len(sl or []), n_all=len(entries or []))


def install_think():
    from consensus_delivery import providers as PV
    real = PV.chat

    def chat(*a, **kw):
        text, rec = real(*a, **kw)
        if str(kw.get("alias", "")).startswith("minimax") and rec.token_logprobs:
            sl = think_slice(rec.token_logprobs)
            rec.token_logprobs = sl if sl is not None else []
        return text, rec

    PV.chat = chat
