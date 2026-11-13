#!/usr/bin/env python3
"""Offline acceptance for the GPQA extension. Zero model calls, real stored prompts.

    python3 ext_test_gpqa_offline.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE, ROOT / "experiments/06_cost_resources/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import answer_extract as AX                                              # noqa: E402
import arm_transform as AT                                              # noqa: E402
import ext_design as D                                                   # noqa: E402

FAIL: List[str] = []
CHECKS = {"n": 0}


def ck(cond: Any, label: str) -> bool:
    CHECKS["n"] += 1
    if not cond:
        FAIL.append(label)
    return bool(cond)


def cell(rel: str) -> List[Dict[str, Any]]:
    blob = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    return blob if isinstance(blob, list) else (blob.get("records") or [])


# ---------------------------------------------------------------- A. prompt identity
class _Captured(Exception):
    """Raised by the capture stub so no model is contacted and the first prompt is kept."""

    def __init__(self, messages):
        super().__init__("captured")
        self.messages = messages


def first_prompt_from_pipeline(protocol: str, unit: str) -> List[Dict[str, str]]:
    """The first outgoing message list this pipeline would send, with no model contacted.
    """
    import ext_unit_driver as EUD
    from consensus_delivery import providers
    EUD.ensure_installed("qa")
    real = providers.chat

    def stub(**kw):
        raise _Captured(list(kw.get("messages") or []))

    providers.chat = stub
    for mod in ("bccs", "du_mad", "mad_m2", "ford", "moa"):
        try:
            m = __import__(f"consensus_delivery.adapters.{mod}", fromlist=["chat"])
            m.chat = stub
        except Exception:
            pass
    try:
        EUD.run_unit("gpqa", protocol, unit, D.ALIAS[D.PANELS[protocol][0]], arm="FULL")
        raise AssertionError("the stub did not intercept the first call")
    except _Captured as c:
        return c.messages
    finally:
        providers.chat = real
        for mod in ("bccs", "du_mad", "mad_m2", "ford", "moa"):
            try:
                m = __import__(f"consensus_delivery.adapters.{mod}", fromlist=["chat"])
                m.chat = real
            except Exception:
                pass


def test_prompt_identity() -> Dict[str, Any]:
    """Every protocol's first prompt must equal the released cell's, byte for byte."""
    cells = {
        "bccs": "results/batch2_matrix/gpqa100_bccs_llama3-8b.json",
        "madm2_objective": "results/batch2_matrix/gpqa100_madm2_objective_llama3-8b.json",
        "du_mad": "results/main_matrix/gpqa200_dumad_llama3-8b.json",
        "ford": "results/batch2_matrix/gpqa100_ford_roster.json",
        "moa": "results/batch2_matrix/gpqa100_moa_roster.json",
    }
    units = D.frozen("gpqa")["unit_ids"]
    out = {}
    for proto, rel in cells.items():
        recs = {r["example_id"]: r for r in cell(rel)}
        same = diff = 0
        example = None
        for u in units[:12]:
            rec = recs.get(u)
            if rec is None or not rec.get("calls"):
                continue
            want = rec["calls"][0]["messages_sent"]
            got = first_prompt_from_pipeline(proto, u)
            if [dict(m) for m in want] == [dict(m) for m in got]:
                same += 1
            else:
                diff += 1
                if example is None:
                    wt = "".join(m.get("content", "") for m in want)
                    gt = "".join(m.get("content", "") for m in got)
                    i = next((j for j in range(min(len(wt), len(gt))) if wt[j] != gt[j]),
                             min(len(wt), len(gt)))
                    example = {"unit": u, "first_difference_at": i,
                               "released": wt[max(0, i - 60):i + 120],
                               "built": gt[max(0, i - 60):i + 120]}
        out[proto] = {"identical": same, "different": diff, "example": example}
        ck(diff == 0 and same > 0,
           f"A/{proto}: prompt identity vs the released GPQA cell ({same} same, {diff} different)"
           + (f" -- {json.dumps(example)[:400]}" if example else ""))
    return out


# ---------------------------------------------------------------- B. the second-argument trap
def test_extractor_arguments() -> Dict[str, Any]:
    from consensus_delivery.adapters import bccs, du_mad, ford, mad_m2, moa
    kind = AX.KIND_BY_BENCHMARK["gpqa"]
    ck(kind == "mcq", "B: gpqa answer kind is not mcq")

    # the right argument, on a reply shaped like the ones these models actually produce
    boxed = "Working it out, the parallax is about 334 mas. \\boxed{C}"
    ansfmt = "Step by step... Answer: C"
    fordfmt = "Answer: (C) is more plausible.\nExplanation: the absorption term dominates."
    ck(bccs.extract_boxed(boxed, kind) == "C", "B: bccs.extract_boxed(text, kind)")
    ck(mad_m2.extract_answer(ansfmt, kind) == "C", "B: mad_m2.extract_answer(text, kind)")
    ck(moa.extract("Therefore the answer is (C).", kind) == "C", "B: moa.extract(text, kind)")
    ck(du_mad.extract_for("gpqa", boxed, 4) == "C", "B: du_mad.extract_for(benchmark, text)")
    ck(ford.parse_prediction(fordfmt, "gpqa")[0] == "C",
       "B: ford.parse_prediction(text, benchmark)")

    # Two of the three failure modes are demonstrated here on a reply of the shape these models
    # really produce -- an answer followed by prose -- because that is where the passthrough shows:
    prose = "Answer: C because the 3.1 absorption coefficient dominates the distance modulus."
    passthrough = mad_m2.extract_answer(prose, "gpqa")       # benchmark passed where kind belongs
    ck(passthrough is not None and not AX.is_well_formed(str(passthrough), kind),
       f"B: mad_m2 with the benchmark instead of the kind returned {str(passthrough)[:60]!r}; the "
       f"well-formedness gate must reject it so it falls through to the natural-language tier")
    ck(mad_m2.extract_answer(prose, kind) == "C",
       "B: mad_m2 with the correct kind failed on an answer followed by prose")
    wrong_ford = ford.parse_prediction(fordfmt, kind)[0]
    ck(wrong_ford == "C",
       "B: ford.parse_prediction with the KIND instead of the benchmark changed the label "
       f"(got {wrong_ford!r}) -- on a non-MCQ task this is the silent-mislabel trap")
    ck(ford.parse_prediction("Answer: non-match is more plausible.\nExplanation: x.",
                             "binary_match")[0] == "no",
       "B: the documented FORD mislabel (kind passed as benchmark turns 'non-match' into 'no') no "
       "longer reproduces, so the reason the call sites pass the benchmark explicitly has changed")

    # and the condition transform's own router must get it right for every protocol
    got = {}
    for proto, text in (("bccs", boxed), ("du_mad", boxed), ("madm2_objective", ansfmt),
                        ("moa", "Therefore the answer is (C)."), ("ford", fordfmt)):
        lab = AT._label_via_adapter(text, proto, kind, "gpqa")
        got[proto] = lab
        ck(lab == "C", f"B: arm_transform._label_via_adapter({proto}) -> {lab!r}, want 'C'")
    return {"router": got}


# ---------------------------------------------------------------- C. FORD's instruction split
def test_ford_instruction() -> Dict[str, Any]:
    from consensus_delivery.adapters import ford
    instr4 = ford.instruction("gpqa", 4)
    ck("(A|B|C|D)" in instr4, "C: ford.instruction('gpqa', 4) is not the four-option variant")
    ck(ford.instruction("gpqa", 2) != instr4,
       "C: n_options=2 and 4 produce the same instruction, so the correction would be moot")
    variants = AT._ford_instructions("gpqa")
    ck(instr4 in variants,
       "C: the 4-option instruction is not in _ford_instructions('gpqa'), so NOPEER could not "
       "detach it and would delete it with the peer turn")

    recs = cell("results/batch2_matrix/gpqa100_ford_roster.json")
    units = set(D.frozen("gpqa")["unit_ids"])
    views = [c["messages_sent"] for r in recs if r["example_id"] in units for c in r["calls"]]
    ck(len(views) > 100, f"C: only {len(views)} stored FORD views to test on")
    tails_with_instr = 0
    split_ok = 0
    for v in views:
        last = v[-1].get("content") or ""
        if last.rstrip().endswith(instr4.rstrip()):
            tails_with_instr += 1
        got, body = AT._ford_split_instruction(last, "gpqa")
        if got.strip() == instr4.strip() and instr4 not in body:
            split_ok += 1
    ck(tails_with_instr == len(views),
       f"C: {len(views) - tails_with_instr} of {len(views)} stored FORD views do not end with the "
       f"4-option instruction -- the released cell used a different one")
    ck(split_ok == len(views),
       f"C: _ford_split_instruction detached the instruction on {split_ok}/{len(views)} views")

    # the real NOPEER path: peers gone, instruction kept, parser still able to read the form
    kept = dropped = 0
    for v in views:
        if len(v) < 2:
            continue                                  # turn 0 has no peer to remove
        new, tr = AT.transform("ford", "NOPEER", v, kind="mcq", benchmark="gpqa")
        txt = "".join(m.get("content") or "" for m in new)
        if instr4.strip() in txt:
            kept += 1
        else:
            dropped += 1
        ck(tr["n_peers_rendered"] == 0, "C: NOPEER left a peer rendered for FORD")
        ck(len(new) == 1, "C: NOPEER left more than the question message for FORD")
    ck(dropped == 0,
       f"C: NOPEER dropped FORD's answer-format instruction on {dropped} of {kept + dropped} "
       f"multi-turn views -- the agents would stop answering in the form parse_prediction reads")
    return {"views": len(views), "instruction": instr4, "nopeer_kept_instruction": kept}


# ---------------------------------------------------------------- D. the conditions on real prompts
PEER_CELLS = {
    "bccs": "results/batch2_matrix/gpqa100_bccs_llama3-8b.json",
    "madm2_objective": "results/batch2_matrix/gpqa100_madm2_objective_llama3-8b.json",
    "du_mad": "results/main_matrix/gpqa200_dumad_llama3-8b.json",
    "moa": "results/batch2_matrix/gpqa100_moa_roster.json",
    "ford": "results/batch2_matrix/gpqa100_ford_roster.json",
}


def test_arms_on_real_prompts() -> Dict[str, Any]:
    """Runs with `ext_arms` already installed, so MoA is measured under the exact split."""
    import ext_arms as EA
    EA.install()
    kind = "mcq"
    units = set(D.frozen("gpqa")["unit_ids"])
    out: Dict[str, Any] = {}

    def arm(proto, v, a, refs, **kw):
        """One transform, with MoA's observed reference replies restored first."""
        EA.reset()
        for t in refs:
            EA.observe_reply(t)
        return AT.transform(proto, a, v, kind=kind, benchmark="gpqa", **kw)

    for proto, rel in PEER_CELLS.items():
        msgs: List[Tuple[List[Dict[str, str]], List[str]]] = []
        for r in cell(rel):
            if r["example_id"] not in units:
                continue
            refs = [c.get("raw_reply") or "" for c in (r.get("calls") or [])
                    if c.get("role") == "reference"]
            for c in r.get("calls") or []:
                msgs.append((c["messages_sent"], refs))
        # keep only the calls that actually rendered a peer, because an condition cannot act otherwise
        with_peers = []
        for v, refs in msgs:
            _n, tr = arm(proto, v, "FULL", refs)
            if tr["peer_blocks_found"] > 0:
                with_peers.append((v, tr["peer_blocks_found"], refs))
        res: Dict[str, Any] = {"calls": len(msgs), "calls_with_peers": len(with_peers)}
        ck(len(with_peers) > 0,
           f"D/{proto}: no stored GPQA call rendered a peer block -- the peer grammar does not "
           f"match this benchmark's prompts, so all three arms would be no-ops")

        # FULL must not change anything
        unchanged = 0
        for v, _n, refs in with_peers[:200]:
            new, tr = arm(proto, v, "FULL", refs)
            if [dict(m) for m in new] == [dict(m) for m in v] and not tr["changed"]:
                unchanged += 1
        ck(unchanged == len(with_peers[:200]), f"D/{proto}: FULL changed the prompt")

        # NOPEER
        zero = 0
        shrank = 0
        for v, _n, refs in with_peers[:200]:
            new, tr = arm(proto, v, "NOPEER", refs)
            if tr["n_peers_rendered"] == 0:
                zero += 1
            before = sum(len(m.get("content") or "") for m in v)
            after = sum(len(m.get("content") or "") for m in new)
            if after < before:
                shrank += 1
        ck(zero == len(with_peers[:200]), f"D/{proto}: NOPEER left peers rendered on some calls")
        ck(shrank == len(with_peers[:200]), f"D/{proto}: NOPEER did not shorten some prompts")

        # ANSONLY: every label this condition WRITES must be a well-formed option letter rendered in the
        # protocol's own form.
        tier = collections.Counter()
        ratios = []
        illegal = []
        for v, _n, refs in with_peers[:200]:
            new, tr = arm(proto, v, "ANSONLY", refs)
            for k, n in (tr.get("ansonly_tier") or {}).items():
                tier[k] += n
            tier["failed"] += tr.get("ansonly_extract_failed", 0)
            before = sum(len(m.get("content") or "") for m in v)
            after = sum(len(m.get("content") or "") for m in new)
            if before:
                ratios.append(after / before)
            # re-derive what the condition wrote, from the same bodies it read
            bodies: List[str] = []
            if proto == "moa":
                EA.reset()
                for t in refs:
                    EA.observe_reply(t)
                for m in v:
                    if AT.MOA_MARKER in (m.get("content") or ""):
                        bodies += [b for _s, _e, _n2, b in AT.moa_refs(m["content"])]
            elif proto == "ford":
                bodies += [(m.get("content") or "") for m in v[1:]]
            else:
                for m in v:
                    bodies += [b for _s, _e, b in AT.peer_blocks(proto, m.get("content") or "")]
            for body in bodies:
                lab, t_ = AX.extract(body, kind,
                                     lambda t: AT._label_via_adapter(t, proto, kind, "gpqa"))
                if lab is None:
                    continue
                if not AX.is_well_formed(lab, kind):
                    illegal.append(f"{t_}:{lab[:40]}")
                rendered = AT._RENDER[proto](lab)
                if len(rendered) > max(len(body), 1):
                    illegal.append(f"render-grew:{rendered[:40]}")
        applied = tier["contract"] + tier["nl"]
        res["ansonly"] = {"contract": tier["contract"], "nl": tier["nl"],
                          "failed": tier["failed"],
                          "fail_rate": round(tier["failed"] / max(applied + tier["failed"], 1), 4),
                          "mean_size_ratio": (round(sum(ratios) / len(ratios), 4)
                                              if ratios else None)}
        ck(applied > 0, f"D/{proto}: ANSONLY applied to 0 blocks on real GPQA prompts")
        ck(not illegal, f"D/{proto}: ANSONLY rendered illegal labels {illegal[:4]}")
        ck(res["ansonly"]["mean_size_ratio"] is None
           or res["ansonly"]["mean_size_ratio"] < 1.0,
           f"D/{proto}: ANSONLY did not shrink the prompt "
           f"(ratio {res['ansonly']['mean_size_ratio']})")

        # UNIFSEL: a strict subset, except where only one peer was rendered (then it is degenerate)
        if proto != "ford":
            degen = subset = 0
            for v, n, refs in with_peers[:200]:
                _new, tr = arm(proto, v, "UNIFSEL", refs, seed=7)
                if n <= 1:
                    degen += 1
                elif tr["n_peers_rendered"] < n:
                    subset += 1
                else:
                    ck(False, f"D/{proto}: UNIFSEL kept every peer on a call with {n} peers")
            res["unifsel"] = {"degenerate_single_peer_calls": degen, "strict_subset_calls": subset}
        out[proto] = res
    return out


# ---------------------------------------------------------------- D2. MoA's reference split
def test_moa_split() -> Dict[str, Any]:
    """The released regex counts a model's own numbered steps as references. Measure and fix it.
    """
    import ext_arms as EA
    units = set(D.frozen("gpqa")["unit_ids"])
    recs = [r for r in cell(PEER_CELLS["moa"]) if r["example_id"] in units]
    before = EA.measure_inflation(recs)
    ck(before["prompts_with_header"] == len(recs),
       f"D2: {before['prompts_with_header']} of {len(recs)} MoA records have an aggregator prompt")
    ck(before["prompts_the_exact_split_could_not_explain"] == 0,
       f"D2: the exact split could not explain "
       f"{before['prompts_the_exact_split_could_not_explain']} aggregator prompts")
    ck(before["exact_references"] == 3 * len(recs),
       f"D2: exact split found {before['exact_references']} references, expected "
       f"{3 * len(recs)} (3 roster models x {len(recs)} units)")
    ck(before["released_regex_references"] > before["exact_references"],
       "D2: the released regex no longer over-counts, so this replacement is unnecessary")

    EA.install()
    counts = []
    ansonly_blocks = 0
    for r in recs:
        refs = [c.get("raw_reply") or "" for c in r["calls"] if c.get("role") == "reference"]
        for c in r["calls"]:
            if c.get("role") != "aggregator":
                continue
            EA.reset()
            for t in refs:
                EA.observe_reply(t)
            _new, tr = AT.transform("moa", "FULL", c["messages_sent"], kind="mcq",
                                    benchmark="gpqa")
            counts.append(tr["peer_blocks_found"])
            EA.reset()
            for t in refs:
                EA.observe_reply(t)
            _new2, tr2 = AT.transform("moa", "ANSONLY", c["messages_sent"], kind="mcq",
                                      benchmark="gpqa")
            ansonly_blocks += (tr2["ansonly_tier"]["contract"] + tr2["ansonly_tier"]["nl"]
                               + tr2["ansonly_extract_failed"])
    ck(set(counts) == {3},
       f"D2: after the replacement, FULL reports peer counts {sorted(set(counts))[:6]}, not 3")
    ck(ansonly_blocks == 3 * len(recs),
       f"D2: ANSONLY touched {ansonly_blocks} blocks, expected {3 * len(recs)}")
    return {"released_vs_exact": before, "peer_counts_after_fix": sorted(set(counts)),
            "ansonly_blocks_after_fix": ansonly_blocks}


# ---------------------------------------------------------------- E. delivery and scoring
def test_scoring() -> Dict[str, Any]:
    gold = D.gpqa_gold()
    units = D.frozen("gpqa")["unit_ids"]
    ck(all(gold.get(u) in ("A", "B", "C", "D") for u in units),
       "E: a frozen GPQA unit has a gold letter outside A-D")
    out = {}
    for proto, rel in PEER_CELLS.items():
        recs = [r for r in cell(rel) if r["example_id"] in set(units)]
        delivered = scored = correct = 0
        for r in recs:
            ans = r.get("majority_answer")
            if ans in (None, "", "None", "none"):
                continue
            delivered += 1
            g = gold.get(r["example_id"])
            if g is None:
                continue
            scored += 1
            if str(ans).strip().upper().strip("().") == g.upper():
                correct += 1
        out[proto] = {"records": len(recs), "delivered": delivered, "scored": scored,
                      "correct": correct,
                      "accuracy_on_delivered": round(correct / scored, 4) if scored else None}
        ck(len(recs) == 100, f"E/{proto}: {len(recs)} of the frozen 100 units present")
        ck(scored > 0, f"E/{proto}: nothing scorable -- the gold and predicted vocabularies miss")
        ck(out[proto]["accuracy_on_delivered"] is None
           or 0.0 < out[proto]["accuracy_on_delivered"] < 1.0,
           f"E/{proto}: accuracy is exactly {out[proto]['accuracy_on_delivered']} on the released "
           f"replies, which is what a vocabulary mismatch looks like")
    return out


def main() -> int:
    report: Dict[str, Any] = {"benchmark": "gpqa", "model_calls": 0}
    print("== A. prompt identity against the released GPQA cells")
    report["prompt_identity"] = test_prompt_identity()
    for p, v in report["prompt_identity"].items():
        print("   %-16s identical %3d  different %3d" % (p, v["identical"], v["different"]))
    print("== B. the extractor second-argument trap")
    report["extractors"] = test_extractor_arguments()
    print("   router labels: %s" % report["extractors"]["router"])
    print("== C. FORD's answer-format instruction under NOPEER")
    report["ford_instruction"] = test_ford_instruction()
    print("   %d stored views; instruction kept on %d"
          % (report["ford_instruction"]["views"],
             report["ford_instruction"]["nopeer_kept_instruction"]))
    print("== D1. MoA's reference split: released regex vs the exact split")
    report["moa_split"] = test_moa_split()
    rv = report["moa_split"]["released_vs_exact"]
    print("   released regex %d refs vs exact %d on %d prompts (inflation %sx); "
          "unexplained %d" % (rv["released_regex_references"], rv["exact_references"],
                              rv["prompts_with_header"], rv["inflation_factor"],
                              rv["prompts_the_exact_split_could_not_explain"]))
    print("== D2. the arms, on the real stored prompts (MoA under the exact split)")
    report["arms"] = test_arms_on_real_prompts()
    for p, v in report["arms"].items():
        a = v.get("ansonly") or {}
        print("   %-16s calls %4d  with peers %4d  ANSONLY contract/nl/failed %d/%d/%d  "
              "size ratio %s" % (p, v["calls"], v["calls_with_peers"], a.get("contract", 0),
                                 a.get("nl", 0), a.get("failed", 0), a.get("mean_size_ratio")))
    print("== E. delivery and scoring on the released replies")
    report["scoring"] = test_scoring()
    for p, v in report["scoring"].items():
        print("   %-16s delivered %3d/%3d  accuracy %s"
              % (p, v["delivered"], v["records"], v["accuracy_on_delivered"]))

    report["checks"] = CHECKS["n"]
    report["failures"] = FAIL
    out = D.OUT / "ACCEPTANCE_gpqa_offline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print()
    for f in FAIL:
        print("  FAIL " + f)
    print("ext_test_gpqa_offline: %s (%d checks) -> %s"
          % ("PASS" if not FAIL else "FAIL", CHECKS["n"], out.relative_to(ROOT)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
