r"""`bccs_execid`: BCCS with execution-result identity in its consensus/stop test and final
aggregation. Installed by sb_override when SB_BCCS_EXECID=1; the frozen adapter is
not edited, its functions are wrapped at import time in the collection process.

* extraction: the lenient extractor of the factorial / interaction_matched (own `\boxed{}` answer,
  else the last fenced sql block, else the last line-initial SELECT/WITH), applied to every reply;
* identity: read-only execution on the task's BIRD database (sql_common.execute, the evaluator's
  semantics); label = sha of the normalised result set; failed / timed-out / empty answers are
  DROPPED (the factorial's `exec_filt`);
* consensus: the released `estimate_consensus` arithmetic (share > 2/3 and belief-weighted score
  > 0.8 -> "full consensus" -> stop) computed over the executing answers and their beliefs;
  no executing answer -> "no consensus";
* final aggregation: the modal signature's first query in agent order.
Gold is never read here. Every round's labels, statuses and stop verdict are attached to the record
as `execid_trace`.
"""
from __future__ import annotations

import hashlib
import re

from consensus_delivery.adapters import bccs as B

_FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.S)
_BARE = re.compile(r"(?im)^\s*((?:SELECT|WITH)\b[^;`]*)")
STATE = {"db": None, "trace": [], "last_round_queries": []}
_orig_extract = B.extract_boxed
_orig_consensus = B.estimate_consensus
_orig_run_one = B.run_one


def _unwrap(s):
    import sql_tasks as ST
    return ST.unwrap_sql(s)


def _make_lenient(contract):
    def lenient(text: str, kind: str) -> str:
        own = contract(text, kind)
        if own:
            return own
        t = str(text or "")
        m = _FENCE.findall(t)
        if m:
            return _unwrap(m[-1])
        b = _BARE.findall(t)
        return _unwrap(b[-1]) if b else ""
    lenient._sb_lenient = True
    return lenient


def lenient(text: str, kind: str) -> str:
    own = _orig_extract(text, kind)
    if own:
        return own
    t = str(text or "")
    m = _FENCE.findall(t)
    if m:
        return _unwrap(m[-1])
    b = _BARE.findall(t)
    return _unwrap(b[-1]) if b else ""


def _execute(db, sql):
    import sql_common as SQ
    ex = getattr(SQ, "_ext_execute_real", SQ.execute)      # the collection process poisons execute
    try:
        r = ex(db, sql, max_rows=10000)
    except Exception as exc:                                  # noqa: BLE001
        return {"status": "error", "error": str(exc)[:80]}
    if r["status"] != "ok":
        return {"status": r["status"]}
    s = sorted(repr(t) for t in SQ._norm_result(r["rows"]))
    return {"status": "ok", "sig": hashlib.sha256("\n".join(s).encode()).hexdigest()[:20]}


def estimate_consensus_exec(support, probs):
    db = STATE["db"]
    qs = [_unwrap(s) if s not in (None, "", "None") else "" for s in support]
    res = [(_execute(db, q) if (q and db) else {"status": "empty"}) for q in qs]
    labels = [r.get("sig") for r in res]
    live = [(l, p) for l, p in zip(labels, probs) if l]
    if not live:
        out = {"verdict": "no consensus", "frequency_share": 0.0, "consensus_score": 0.0,
               "modal_answer": None}
    else:
        ls = [l for l, _p in live]
        modal = max(ls, key=ls.count)          # ties: first in agent order (max keeps the first)
        freq = ls.count(modal)
        score = sum(p for l, p in live if l == modal)
        total = sum(p for _l, p in live)
        cs = (score / total) if total else 1.0
        share = freq / len(ls)
        if share > (2 / 3) and cs > 0.8:
            v = "full consensus"
        elif share >= (2 / len(ls)) and cs > 0.5:
            v = "partial consensus"
        else:
            v = "no consensus"
        out = {"verdict": v, "frequency_share": round(share, 4), "consensus_score": round(cs, 4),
               "modal_answer": qs[labels.index(modal)], "modal_frequency": freq}
    STATE["trace"].append(dict(labels=labels, status=[r["status"] for r in res],
                               verdict=out["verdict"], n_executing=len(live),
                               strict_verdict=_orig_consensus(support, probs)["verdict"]))
    return out


def run_one_exec(question, *, task_id, benchmark, **kw):
    # `sql_tasks.install()` (worker init) re-wraps bccs.extract_boxed with the SQL contract AFTER
    # this module's install, so the lenient layer is (re)applied here, at call time, around
    # whatever contract extractor is current. Error found 02:40Z: the first G2 ran strict.
    if not getattr(B.extract_boxed, "_sb_lenient", False):
        B.extract_boxed = _make_lenient(B.extract_boxed)
    STATE["trace"] = []
    rec = _orig_run_one(question, task_id=task_id, benchmark=benchmark, **kw)
    rec["execid_trace"] = list(STATE["trace"])
    rec["protocol_variant"] = "bccs_execid (variant A9)"
    return rec


def install():
    B.extract_boxed = lenient
    B.estimate_consensus = estimate_consensus_exec
    B.run_one = run_one_exec
    import ext_unit_driver as EUD
    orig = EUD.run_unit

    def run_unit(benchmark, protocol, unit, alias, **kw):
        STATE["db"] = EUD.sql_task(unit)["db_id"] if benchmark == "sql_bird_v2" else None
        return orig(benchmark, protocol, unit, alias, **kw)

    EUD.run_unit = run_unit
