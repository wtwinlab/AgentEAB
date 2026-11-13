#!/usr/bin/env python3
"""Offline correctness tests for the concurrent collector. Zero model calls.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import pathlib
import random
import sys
import threading

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "code")]

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "", on_pass: str = ""):
    note = on_pass if cond else detail
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + note) if note else ''}")
    if not cond:
        FAILS.append(f"{name}: {detail}")


#: a peer-bearing Du-MAD prompt, in the grammar the collection actually sent
PEER_PROMPT = [{"role": "user", "content":
                "These are the solutions to the problem from other agents:"
                "\n\n One agent solution: ```The two records are the same product because both "
                "list the identical model number and manufacturer.```"
                "\n\n Using the solutions from other agents as additional information, can you "
                "provide your answer to the question? "}]


def _init_fake():
    """Worker init: install a fake provider so `install_arm` has something to wrap."""
    sys.path[:0] = [str(HERE), str(ROOT / "code")]
    import matching_tasks as MT
    MT.install()
    from consensus_delivery import providers

    class Rec:
        prompt_tokens = 10
        completion_tokens = 5

    def fake(**kw):
        # the fake returns the prompt it was given, so the caller can see what was sent
        return json.dumps([m.get("content", "") for m in kw.get("messages") or []]), Rec()

    providers.chat = fake
    for mod in ("du_mad", "bccs", "mad_m2", "ford", "moa"):
        try:
            m = __import__(f"consensus_delivery.adapters.{mod}", fromlist=["chat"])
            m.chat = fake
        except Exception:
            pass


def _one_call(arm: str, barrier: "threading.Barrier | None" = None) -> dict:
    """Install `arm`, issue one request through the patched route, report what the trace says.
    """
    import unit_driver as UD
    import matching_tasks as MT
    sink: list = []
    orig = UD.install_arm("du_mad", arm, sink, MT.KIND, seed=7)
    try:
        if barrier is not None:
            barrier.wait(timeout=30)
        from consensus_delivery import providers
        sent, _ = providers.chat(messages=[dict(m) for m in PEER_PROMPT], max_tokens=64)
    finally:
        UD.restore_chat(orig)
    return {"requested_arm": arm,
            "trace_arms": [t["arm"] for t in sink],
            "sent": sent,
            "peers_rendered": [t["n_peers_rendered"] for t in sink]}


#: what each arm's own transform must leave in the sent text
_EXPLANATION = "identical model number"
_PEER_MARKER = "One agent solution"
_LABEL = "boxed{match}"


def served_own_arm(r: dict) -> bool:
    """True when this call was transformed by exactly its own condition.
    """
    if r["trace_arms"] != [r["requested_arm"]]:
        return False
    sent, arm = r["sent"], r["requested_arm"]
    if arm == "NOPEER":
        return _PEER_MARKER not in sent
    if arm == "ANSONLY":
        return _LABEL in sent and _EXPLANATION not in sent
    return _EXPLANATION in sent          # FULL and UNIFSEL keep the single peer verbatim


def main() -> int:
    print("concurrency correctness, offline\n")
    _init_fake()
    arms = ["FULL", "NOPEER", "ANSONLY", "UNIFSEL"] * 4

    # ---- 1a. the failure mode is real: threads in one process DO cross-talk
    print("1a. does in-process threading corrupt? (it must, or the isolation test is vacuous)")
    out: list = []
    lock = threading.Lock()
    barrier = threading.Barrier(len(arms))

    def worker(a):
        r = _one_call(a, barrier)
        with lock:
            out.append(r)

    ths = [threading.Thread(target=worker, args=(a,)) for a in arms]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    crosstalk = sum(1 for r in out if not served_own_arm(r))
    check("threaded execution cross-talks, as expected",
          crosstalk > 0,
          "no cross-talk observed, so this test cannot exercise the isolation it checks; "
          "the isolation check below then has no contrast",
          f"{crosstalk} of {len(out)} threaded calls were not served by their own arm "
          f"(empty or foreign trace, or text transformed by another arm)")

    # ---- 1b. the process pool does not
    print("\n1b. does the process pool isolate?")
    with cf.ProcessPoolExecutor(max_workers=8, initializer=_init_fake) as ex:
        res = list(ex.map(_one_call, arms))
    bad = [r for r in res if not served_own_arm(r)]
    check("every pooled call was served by its own arm", not bad,
          f"{len(bad)} of {len(res)} calls were not: "
          f"{[(r['requested_arm'], r['trace_arms']) for r in bad[:3]]}",
          f"{len(res)}/{len(res)} calls clean across 8 processes, judged on trace and sent text")

    # the conditions must also have had their defined effect, not merely the right label
    by = {}
    for r in res:
        by.setdefault(r["requested_arm"], []).append(r)
    nop = by.get("NOPEER", [])
    check("NOPEER actually removed the peer text under concurrency",
          bool(nop) and all("One agent solution" not in r["sent"] for r in nop),
          "a NOPEER call still carried peer text",
          f"{len(nop)}/{len(nop)} NOPEER calls peer-free")
    ans = by.get("ANSONLY", [])
    check("ANSONLY actually reduced the peer block under concurrency",
          bool(ans) and all("identical model number" not in r["sent"] for r in ans)
          and all("boxed{match}" in r["sent"] for r in ans),
          f"ANSONLY sent: {ans[0]['sent'][:160] if ans else 'nothing'}",
          f"{len(ans)}/{len(ans)} ANSONLY calls carry the label and not the explanation")
    full = by.get("FULL", [])
    check("FULL under concurrency is untouched",
          bool(full) and all("identical model number" in r["sent"] for r in full),
          "a FULL control lost its peer text", f"{len(full)}/{len(full)} intact")

    # ---- 2. no shared mutable state
    print("\n2. caller state")
    import arm_transform as AT
    import matching_tasks as MT
    before = json.dumps(PEER_PROMPT, sort_keys=True)
    for arm in ("NOPEER", "ANSONLY", "UNIFSEL"):
        AT.transform("du_mad", arm, PEER_PROMPT, kind=MT.KIND, call_seq=0)
    check("transform does not mutate the caller's messages",
          json.dumps(PEER_PROMPT, sort_keys=True) == before,
          "the prompt object was edited in place, so a shared prompt would carry over between conditions",
          "prompt object unchanged after three arms")

    # ---- 3. identity is content-derived
    print("\n3. identity and cache keys")
    import collect_controlled as CC
    items = CC.items_for("gsm8k", limit=3)
    shuffled = list(items)
    random.Random(1).shuffle(shuffled)
    seeds_a = {it["key"]: it["seed"] for it in items}
    seeds_b = {it["key"]: CC.item_seed(it) for it in shuffled}
    check("seeds depend on the item, not on queue position", seeds_a == seeds_b,
          "a shuffle changed some seed", f"{len(seeds_a)} items stable under shuffle")
    check("cache keys are unique per item", len({it["key"] for it in items}) == len(items),
          "two items share a cache key", f"{len(items)} distinct keys")
    k = dict(items[0])
    k_other_arm = dict(k, arm="NOPEER" if k["arm"] != "NOPEER" else "FULL")
    check("cache key separates arms", CC.cache_key(k) != CC.cache_key(k_other_arm),
          "two arms map to one key")
    k_other_rep = dict(k, repeat=k["repeat"] + 1)
    check("cache key separates repeats", CC.cache_key(k) != CC.cache_key(k_other_rep),
          "two repeats map to one key")
    k_other_bb = dict(k, backbone="qwen2.5-7b" if k["backbone"] != "qwen2.5-7b" else "llama3-8b")
    check("cache key separates backbones", CC.cache_key(k) != CC.cache_key(k_other_bb),
          "two backbones map to one key")
    # a change to the intervention code must invalidate the key
    real = CC.CODE_VERSION
    fake_digest = hashlib.sha256(b"different intervention code").hexdigest()[:16]
    CC.CODE_VERSION = fake_digest
    moved = CC.cache_key(k)
    CC.CODE_VERSION = real
    check("cache key moves when the intervention code changes", moved != CC.cache_key(k),
          "records of a different ANSONLY transform version would be reused",
          "a code change invalidates prior records")

    # ---- 4. the queue is benchmark-major
    print("\n4. queue shape")
    qs = CC.plan(limit=2)
    check("benchmark order is the declared one", [b for b, _ in qs] == CC.BENCH_ORDER,
          f"got {[b for b, _ in qs]}", " -> ".join(CC.BENCH_ORDER))
    # The ordering and panel properties belong to the **design**, not to whatever is still pending.
    first = CC.items_for(CC.BENCH_ORDER[0], limit=2)
    check("cheap protocols come first inside a benchmark",
          bool(first) and first[0]["protocol"] == "ford",
          f"first item is {first[0]['protocol'] if first else 'none'}",
          "FORD leads, BCCS last")
    check("FORD/UNIFSEL is not in the design as a prompt arm",
          not any(it["protocol"] == "ford" and it["arm"] == "UNIFSEL"
                  for b in CC.BENCH_ORDER for it in CC.items_for(b, limit=1)),
          "FORD/UNIFSEL appears in the design even though it is a use contrast")
    homo = [it for it in first if it["protocol"] == "du_mad"]
    check("homogeneous protocols carry both frozen panels",
          {it["backbone"] for it in homo} == {"llama3-8b", "qwen2.5-7b"},
          f"got {sorted({it['backbone'] for it in homo})}", "llama3-8b + qwen2.5-7b")
    het = [it for it in first if it["protocol"] == "moa"]
    check("heterogeneous protocols keep one frozen roster",
          {it["backbone"] for it in het} == {"roster"},
          f"got {sorted({it['backbone'] for it in het})}", "roster only")
    check("every item carries both repeats",
          {it["repeat"] for it in first} == {1, 2}, "repeats missing", "repeat 1 and 2")

    print(f"\n{'ALL CONCURRENCY CHECKS PASS' if not FAILS else str(len(FAILS)) + ' FAILURES'}")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
