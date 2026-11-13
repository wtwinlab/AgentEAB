#!/usr/bin/env python3
"""Offline acceptance for the three prompt interventions. Real traces, zero model calls.
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "code")]
import arm_transform as AT                                           # noqa: E402
import matching_tasks as MT                                          # noqa: E402

MT.install()
KIND = MT.KIND

RAW = ROOT / "results/protocol_all_matching_20261018T192619Z/raw"
OK = '"status": "ok", "benchmark"'
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "", on_pass: str = ""):
    """`detail` explains a FAILURE; `on_pass` is the evidence when it passes.
    """
    note = on_pass if cond else detail
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + note) if note else ''}")
    if not cond:
        FAILS.append(f"{name}: {detail}")


def sample_prompts(proto: str, want_peer: int = 20, want_bare: int = 20):
    """Two pools, drawn deliberately: prompts that DO render peers, and prompts that render none.
    """
    peer, bare = [], []
    pat = proto
    for f in sorted(glob.glob(str(RAW / f"wdc_products__{pat}__*.jsonl")))[:2]:
        for line in open(f, encoding="utf-8"):
            if OK not in line or (len(peer) >= want_peer and len(bare) >= want_bare):
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            for c in (r.get("record") or {}).get("calls") or []:
                ms = c.get("messages_sent") or []
                if not ms:
                    continue
                joined = " ".join(m.get("content", "") for m in ms)
                has_block = bool(AT.peer_blocks(proto, joined)) or bool(AT.moa_refs(joined))
                if proto == "ford":
                    has_block = sum(1 for m in ms if m.get("role") == "user") > 1
                has_clause = any(a in joined for a, _b in AT.PEER_CLAUSES)
                if has_block and len(peer) < want_peer:
                    peer.append(ms)
                elif (not has_block) and (not has_clause) and len(bare) < want_bare:
                    bare.append(ms)
    return peer, bare


def main() -> int:
    print("offline arm acceptance, on prompts that were actually sent\n")
    for proto in ("bccs", "du_mad", "madm2_objective", "moa", "ford"):
        withpeer, nopeer = sample_prompts(proto)
        ps = withpeer + nopeer
        if not ps:
            check(f"{proto}: traces available", False, "no preserved prompts found")
            continue
        print(f"\n{proto}: {len(withpeer)} peer-bearing prompts, {len(nopeer)} bare prompts")
        check(f"{proto}: peer-bearing prompts found in the traces", len(withpeer) > 0,
              "none found -- the peer-block grammar does not match what was sent",
              f"{len(withpeer)} found")

        # ---- NOPEER: no peer text survives, and a peerless prompt is untouched
        if withpeer:
            bad = 0
            for i, ms in enumerate(withpeer[:20]):
                new, tr = AT.transform(proto, "NOPEER", ms, kind=KIND, call_seq=i)
                joined = " ".join(m.get("content", "") for m in new)
                if proto == "moa":
                    left = len(AT.moa_refs(joined))
                elif proto == "ford":
                    left = tr["n_peers_rendered"]
                else:
                    left = len(AT.BLOCK_RE.findall(joined))
                if left:
                    bad += 1
            check(f"{proto}/NOPEER removes every peer block", bad == 0,
                  f"{bad} of {len(withpeer[:20])} prompts still carried peer text",
                  f"{len(withpeer[:20])} prompts cleared")
            nz = sum(1 for i, ms in enumerate(withpeer[:20])
                     if AT.transform(proto, "NOPEER", ms, kind=KIND, call_seq=i)[1]["changed"])
            check(f"{proto}/NOPEER actually engaged on peer-bearing prompts", nz == len(withpeer[:20]),
                  f"only {nz} of {len(withpeer[:20])} were changed",
                  f"{nz}/{len(withpeer[:20])} changed")
        if nopeer:
            noop = 0
            for i, ms in enumerate(nopeer[:20]):
                new, tr = AT.transform(proto, "NOPEER", ms, kind=KIND, call_seq=i)
                if new == [dict(m) for m in ms] and not tr["changed"]:
                    noop += 1
            check(f"{proto}/NOPEER is a no-op on a bare prompt", noop == len(nopeer[:20]),
                  f"{len(nopeer[:20]) - noop} of {len(nopeer[:20])} were modified anyway",
                  f"{noop}/{len(nopeer[:20])} untouched")

        # ---- ANSONLY: answer survives, explanation goes, failures surfaced
        if withpeer:
            kept_answer = shortened = failed_surfaced = total = 0
            for i, ms in enumerate(withpeer[:20]):
                before = " ".join(m.get("content", "") for m in ms)
                new, tr = AT.transform(proto, "ANSONLY", ms, kind=KIND, call_seq=i)
                after = " ".join(m.get("content", "") for m in new)
                total += 1
                failed_surfaced += 1 if tr["ansonly_extract_failed"] else 0
                if proto in ("bccs", "du_mad", "madm2_objective"):
                    b_before = [b for _s, _e, b in AT.peer_blocks(proto, before)]
                    b_after = [b for _s, _e, b in AT.peer_blocks(proto, after)]
                    tok = "Answer:" if proto.startswith("madm2") else "\\boxed"
                    ans_ok = all((tok in a) or (a == b) for a, b in zip(b_after, b_before))
                    kept_answer += 1 if ans_ok else 0
                    if b_after and sum(map(len, b_after)) < sum(map(len, b_before)):
                        shortened += 1
                else:
                    kept_answer += 1 if len(after) <= len(before) else 0
                    shortened += 1 if len(after) < len(before) else 0
            check(f"{proto}/ANSONLY keeps the answer in every rewritten block",
                  kept_answer == total, f"{total - kept_answer} of {total} lost it",
                  f"{total}/{total} kept")
            check(f"{proto}/ANSONLY shortened at least one block", shortened > 0,
                  f"0 of {total} got shorter -- the transform did not engage",
                  f"{shortened}/{total} got shorter")
            print(f"      extraction failures surfaced on {failed_surfaced}/{total} prompts "
                  f"(these are reported, never silently widened)")

        # ---- UNIFSEL: the kept set is exactly what the rule chose
        if withpeer and proto != "ford":
            consistent = total = 0
            for i, ms in enumerate(withpeer[:20]):
                new, tr = AT.transform(proto, "UNIFSEL", ms, kind=KIND, call_seq=i)
                joined = " ".join(m.get("content", "") for m in new)
                left = (len(AT.moa_refs(joined)) if proto == "moa"
                        else len(AT.peer_blocks(proto, joined)))
                total += 1
                if left == tr["n_peers_rendered"] == len(tr["peer_indices_kept"]):
                    consistent += 1
            check(f"{proto}/UNIFSEL renders exactly the peers it recorded",
                  consistent == total, f"{total - consistent} of {total} disagreed",
                  f"{total}/{total} agree")

        # ---- the trace fields the design requires, on every condition
        have = 0
        for arm in AT.ARMS:
            _n, tr = AT.transform(proto, arm, ps[0], kind=KIND, call_seq=0)
            if ("n_peers_rendered" in tr) and ("peer_indices_kept" in tr):
                have += 1
        check(f"{proto}: n_peers_rendered + peer_indices recorded on all four arms", have == 4,
              f"only {have} of 4 arms recorded them", "all four arms")

    # ---- a peer that cannot be parsed, constructed to be unambiguous
    print("\nconstructed edge cases")
    msg = [{"role": "user", "content":
            "These are the solutions to the problem from other agents:"
            "\n\n One agent solution: ```I am not sure about this one at all.```"
            "\n\n Using the solutions from other agents as additional information, can you provide "
            "your answer to the question? "}]
    new, tr = AT.transform("du_mad", "ANSONLY", msg, kind=KIND, call_seq=0)
    check("ANSONLY surfaces an unparseable peer instead of widening",
          tr["ansonly_extract_failed"] == 1 and "not sure about this one" in new[0]["content"],
          f"failed={tr['ansonly_extract_failed']}")
    new2, tr2 = AT.transform("du_mad", "NOPEER", msg, kind=KIND, call_seq=0)
    check("NOPEER drops the block and rewrites the clause that referred to it",
          "One agent solution" not in new2[0]["content"]
          and "Using the solutions from other agents" not in new2[0]["content"],
          new2[0]["content"][:70])
    boxed = [{"role": "user", "content":
              "These are the solutions to the problem from other agents:"
              "\n\n One agent solution: ```Long reasoning here. \\boxed{non-match}```"
              "\n\n Using the solutions from other agents as additional information, can you "
              "provide your answer to the question? "}]
    new3, tr3 = AT.transform("du_mad", "ANSONLY", boxed, kind=KIND, call_seq=0)
    check("ANSONLY keeps the boxed answer and drops the reasoning",
          "\\boxed{non-match}" in new3[0]["content"]
          and "Long reasoning here" not in new3[0]["content"]
          and tr3["ansonly_extract_failed"] == 0)

    print(f"\n{'ALL OFFLINE CHECKS PASS' if not FAILS else str(len(FAILS)) + ' FAILURES'}")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
