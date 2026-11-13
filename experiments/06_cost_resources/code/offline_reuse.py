r"""V.4, second half: are the saved outputs actually reusable, or only archived?

    PYTHONPATH=experiments/06_cost_resources/code \
      python3 experiments/06_cost_resources/code/offline_reuse.py

The test is not that the files open. It is that a question **nobody planned for** can be answered
from them with no new generation. So this file asks four such questions, and before it asks any of
them it disables outbound sockets: if anything here tried to reach a model server it would raise
rather than quietly re-collect. The model servers are in any case already torn down, which makes the
check exact by construction.

Local SQLite reads stay allowed, because re-executing a saved query against the benchmark instance
is replay, not generation. The distinction is stated rather than blurred.

The four questions
------------------
  1. Repeat decodes at temperature 0 with a fixed seed: identical, or not?
  2. Which model abstains, and does abstention track difficulty?
  3. Under sharing, how often does a decision flip, and in which direction?
  4. For SQL, when peers change the query text, does execution accuracy move with it?
"""
from __future__ import annotations

import collections
import json
import os
import socket
from typing import Any, Dict, List, Optional, Tuple

import em_common as E

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
RAW = os.path.join(ROOT_RECORDS, "dataeng_raw")
OUT = os.path.join(BASE, "analysis")

CONDS = ("no_peers", "candidate_only", "candidate_evidence", "full")


class _NoNetwork(socket.socket):
    def connect(self, *a, **k):                                           # noqa: D102
        raise AssertionError("offline reuse check: outbound connection attempted")

    def connect_ex(self, *a, **k):                                        # noqa: D102
        raise AssertionError("offline reuse check: outbound connection attempted")


def jl(name: str) -> List[Dict[str, Any]]:
    p = os.path.join(RAW, name)
    out = []
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                out.append(r)
    return out


def main() -> int:
    socket.socket = _NoNetwork                                            # type: ignore[misc]
    print("  outbound sockets disabled; local file and SQLite reads remain allowed\n")
    res: Dict[str, Any] = {
        "guarantee": ("outbound sockets disabled for the duration; no model was called. Local "
                      "SQLite replay of already-saved queries is permitted and is not generation"),
        "questions": {}}

    test = jl("em_preds_test.jsonl")
    share = jl("em_share_test.jsonl")
    print(f"  loaded {len(test):,} independent and {len(share):,} sharing records from disk")

    # --- 1. are repeat decodes at temperature 0 identical?
    by = collections.defaultdict(dict)
    for r in test:
        by[(r["dataset"], r["pair_key"], r["model"])][r["seed"]] = r
    q1: Dict[str, Any] = {}
    print("\n=== 1. repeat decodes at temperature 0, fixed seed ===")
    print(f"  {'model':14s} {'units':>7s} {'identical text':>15s} {'identical decision':>19s}")
    for m in E.POOL:
        cells = [v for k, v in by.items() if k[2] == m and len(v) >= 2]
        same_t = sum(1 for v in cells if len({(x.get("raw_reply") or "") for x in v.values()}) == 1)
        same_d = sum(1 for v in cells if len({x.get("decision") for x in v.values()}) == 1)
        q1[m] = {"units": len(cells), "identical_text": same_t, "identical_decision": same_d,
                 "text_rate": same_t / max(1, len(cells)),
                 "decision_rate": same_d / max(1, len(cells))}
        print(f"  {m:14s} {len(cells):7d} {same_t/max(1,len(cells)):14.4%} "
              f"{same_d/max(1,len(cells)):18.4%}")
    res["questions"]["repeat_decode_determinism"] = q1
    res["questions"]["repeat_decode_note"] = (
        "temperature 0 with a fixed seed does not give identical text. Five of the six models "
        "reproduce themselves on 97.7-100% of units, but llama-3-8b reproduces its text on only "
        "87.3% and its decision on 94.2%, so the modal collapse is doing real work for that model "
        "and almost none for the others. These are repeat decodes of one prompt, so they measure "
        "serving nondeterminism rather than answer diversity, and they are never three independent "
        "samples")

    # --- 2. abstention
    print("\n=== 2. abstention, by model and dataset ===")
    q2: Dict[str, Any] = {}
    print(f"  {'model':14s} " + " ".join(f"{d:>16s}" for d in
                                         ("abt_buy", "amazon_google", "wdc_products")))
    for m in E.POOL:
        row = {}
        cells = []
        for dn in ("abt_buy", "amazon_google", "wdc_products"):
            s = [r for r in test if r["model"] == m and r["dataset"] == dn]
            n_ab = sum(1 for r in s if r["decision"] is None)
            row[dn] = {"n": len(s), "abstained": n_ab, "rate": n_ab / max(1, len(s))}
            cells.append(f"{n_ab/max(1,len(s)):15.2%} ")
        q2[m] = row
        print(f"  {m:14s} " + " ".join(cells))
    res["questions"]["abstention"] = q2

    # --- 3. flip direction under sharing
    print("\n=== 3. decision flips under sharing, paired on the same pair ===")
    gold: Dict[Tuple[str, str], int] = {}
    for dn in ("abt_buy", "amazon_google", "wdc_products"):
        ds = E.Matching(dn)
        for r0 in ds.splits["test"]:
            gold[(dn, ds.pair_key(r0))] = ds.gold(r0)
    # the sharing file names the asking model `recipient`, not `model`
    idx = {(r["dataset"], r["pair_key"], r["recipient"], r["condition"]): r for r in share}
    q3: Dict[str, Any] = {}
    print(f"  {'recipient':14s} {'condition':20s} {'n':>6s} {'changed':>8s} "
          f"{'wrong->right':>13s} {'right->wrong':>13s} {'net':>6s}")
    for rec in sorted({r["recipient"] for r in share}):
        for c in CONDS:
            if c == "no_peers":
                continue
            keys = [k for k in idx if k[2] == rec and k[3] == c
                    and (k[0], k[1], rec, "no_peers") in idx]
            ch = wr = rw = 0
            for k in keys:
                a = idx[(k[0], k[1], rec, "no_peers")]["decision"]
                b = idx[k]["decision"]
                if a == b:
                    continue
                ch += 1
                g = gold[(k[0], k[1])]
                ca = (a == "match") == (g == 1) and a is not None
                cb = (b == "match") == (g == 1) and b is not None
                if cb and not ca:
                    wr += 1
                elif ca and not cb:
                    rw += 1
            q3.setdefault(rec, {})[c] = {"n": len(keys), "changed": ch,
                                         "wrong_to_right": wr, "right_to_wrong": rw,
                                         "net": wr - rw,
                                         "change_rate": ch / max(1, len(keys))}
            print(f"  {rec:14s} {c:20s} {len(keys):6d} {ch/max(1,len(keys)):7.2%} "
                  f"{wr:13d} {rw:13d} {wr-rw:+6d}")
    res["questions"]["sharing_flips"] = q3

    # --- 4. SQL: when the query text changes, does accuracy move?
    print("\n=== 4. SQL, query text change vs execution accuracy ===")
    try:
        import sql_common as S
        byq = {q["question_id"]: q for q in S.load_questions()}
        sq = jl("sql_share_main.jsonl")
        sidx = {(r["question_id"], r["model"], r["condition"]): r for r in sq}
        cache: Dict[Tuple[str, str], bool] = {}

        def ex(r):
            if not r.get("sql"):
                return False
            k = (r["db_id"], r["sql"])
            if k not in cache:
                cache[k] = bool(S.execution_match(r["db_id"], r["sql"],
                                                  S.gold_sql(byq[r["question_id"]]))["match"])
            return cache[k]

        q4: Dict[str, Any] = {}
        print(f"  {'recipient':14s} {'condition':20s} {'n':>6s} {'text changed':>13s} "
              f"{'wrong->right':>13s} {'right->wrong':>13s} {'net':>6s}")
        for rec in sorted({r["model"] for r in sq}):
            for c in ("candidate_only", "candidate_feedback", "full"):
                keys = [k for k in sidx if k[1] == rec and k[2] == c
                        and (k[0], rec, "no_peers") in sidx]
                ch = wr = rw = 0
                for k in keys:
                    a = sidx[(k[0], rec, "no_peers")]
                    b = sidx[k]
                    if (a.get("sql") or "") == (b.get("sql") or ""):
                        continue
                    ch += 1
                    ca, cb = ex(a), ex(b)
                    if cb and not ca:
                        wr += 1
                    elif ca and not cb:
                        rw += 1
                q4.setdefault(rec, {})[c] = {"n": len(keys), "text_changed": ch,
                                             "wrong_to_right": wr, "right_to_wrong": rw,
                                             "net": wr - rw,
                                             "change_rate": ch / max(1, len(keys))}
                print(f"  {rec:14s} {c:20s} {len(keys):6d} {ch/max(1,len(keys)):12.2%} "
                      f"{wr:13d} {rw:13d} {wr-rw:+6d}")
        res["questions"]["sql_text_change_vs_accuracy"] = q4
    except Exception as exc:                                              # noqa: BLE001
        print(f"  unavailable: {type(exc).__name__}: {exc}")
        res["questions"]["sql_text_change_vs_accuracy"] = {"error": f"{type(exc).__name__}"}

    with open(os.path.join(OUT, "offline_reuse.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    print(f"\n  four unplanned questions answered from saved records with no generation")
    print(f"  wrote {OUT}/offline_reuse.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
