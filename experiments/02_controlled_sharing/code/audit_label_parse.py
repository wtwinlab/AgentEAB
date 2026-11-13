#!/usr/bin/env python3
"""Separate two causes of the Liang MAD matching result, from preserved raw output only.
"""
from __future__ import annotations

import collections
import glob
import importlib.util
import json
import os
import pathlib
import re
import sys

NR = pathlib.Path(os.environ.get("RUN_DIR")
                  or (pathlib.Path(__file__).resolve().parents[3]
                      / "data" / "records" / "controlled_sharing"))
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "code"), str(ROOT / "experiments/06_cost_resources/code")]

import matching_tasks as MT  # noqa: E402  the v1 parser, kept for the before/after comparison

_spec = importlib.util.spec_from_file_location("em", HERE / "eval_matching.py")
em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(em)

#: polarity-bearing surface forms, longest first so "non-match" wins over "match"
_POLAR = re.compile(
    r"\b(non-?match(?:es)?|match(?:es|ed|ing)?|same|identical|equivalent|duplicates?|"
    r"different|distinct|unrelated)\b", re.I)
#: a negation cue anywhere in the few tokens before the polarity word flips its sense
_NEG_CUE = re.compile(
    r"\b(?:not|n't|no|never|cannot|can't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"nor|neither|without|fails? to|unlikely to|rather than)\b", re.I)
_NEGATIVE_WORDS = {"different", "distinct", "unrelated"}
_LOOKBACK_WORDS = 5


def to_label_v2(raw) -> str:
    """Last polarity word wins, with a negation window in front of it.
    """
    t = str(raw if raw is not None else "").strip()
    if not t:
        return ""
    hits = list(_POLAR.finditer(t))
    if not hits:
        return ""
    m = hits[-1]
    word = m.group(1).lower()
    positive = not (word.startswith("non") or word in _NEGATIVE_WORDS)
    before = t[max(0, m.start() - 120):m.start()]
    tail = " ".join(before.split()[-_LOOKBACK_WORDS:])
    if _NEG_CUE.search(tail):
        positive = not positive
    return "match" if positive else "non-match"


def self_test() -> None:
    cases = [
        ("These two products are not the same.", "non-match", "match"),
        ("The records do not refer to the same entity.", "non-match", "match"),
        ("They do not describe the same product.", "non-match", "match"),
        ("The Negative side is correct: these are not the same item.", "non-match", "match"),
        ("I conclude they are not identical.", "non-match", ""),
        ("non-match", "non-match", "non-match"),
        ("MATCH", "match", "match"),
        ("They are a match.", "match", "match"),
        ("The two records are the same product.", "match", "match"),
        ("They refer to different entities.", "non-match", "non-match"),
        ("This is not a match.", "non-match", "non-match"),
        ("The two listings are the same.", "match", "match"),
        ("They are distinct records.", "non-match", "non-match"),
        ("They are not different products.", "match", "non-match"),
    ]
    bad = []
    for text, want, v1_was in cases:
        got = to_label_v2(text)
        if got != want:
            bad.append((text, want, got))
        v1 = MT.to_label(text)
        assert v1 == v1_was, f"v1 drifted on {text!r}: {v1!r} != {v1_was!r}"
    if bad:
        for text, want, got in bad:
            print(f"  FAIL want={want!r} got={got!r}  <- {text}")
        raise SystemExit("v2 self-test failed")
    changed = sum(1 for t, w, v1 in cases if v1 != w)
    print(f"  v2 self-test: {len(cases)} cases pass; v1 was wrong on {changed} of them")


def prf(pairs):
    tp = sum(1 for g, p in pairs if g == 1 and p == 1)
    fp = sum(1 for g, p in pairs if g == 0 and p == 1)
    fn = sum(1 for g, p in pairs if g == 1 and p == 0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(pairs), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def binary(lab):
    return 1 if lab == "match" else (0 if lab == "non-match" else None)


def moderator_answer(reply: str):
    """The moderator's own contract: a JSON object carrying `debate_answer`."""
    try:
        blob = json.loads(reply[reply.index("{"):reply.rindex("}") + 1])
    except Exception:
        return None, None
    side = str(blob.get("Supported Side") or blob.get("supported_side") or "").strip()
    return blob.get("debate_answer"), side


def main() -> int:
    self_test()
    out = {"defect_1_parser": {}, "defect_2_capitulation": {}, "reached_the_prompts": {}}

    # ---- how far does the parser issue reach, across every workflow -------------------------
    print("\n  defect 1: labels that change when the parser is corrected, per workflow")
    print(f"  {'workflow':16s}{'population':24s}{'final texts':>12s}{'changed':>9s}{'share':>8s}")
    for f in sorted(glob.glob(str(NR / "raw/*.jsonl"))):
        pass  # the per-file pass happens below, grouped

    groups = collections.defaultdict(lambda: {"n": 0, "changed": 0})
    for f in sorted(glob.glob(str(NR / "raw/*.jsonl"))):
        parts = pathlib.Path(f).stem.split("__")
        if len(parts) < 4:
            continue
        pop, proto = parts[0], parts[1]
        if pop == "musicbrainz20k":
            continue
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok":
                continue
            rec = r.get("record") or {}
            calls = rec.get("calls") or []
            if not calls:
                continue
            last = calls[-1].get("raw_reply") or ""
            g = groups[(proto, pop)]
            g["n"] += 1
            if MT.to_label(last) != to_label_v2(last):
                g["changed"] += 1
    for (proto, pop), g in sorted(groups.items()):
        share = g["changed"] / g["n"] if g["n"] else 0.0
        out["defect_1_parser"][f"{proto}|{pop}"] = {
            "final_texts": g["n"], "labels_changed": g["changed"], "share": round(share, 4)}
        print(f"  {proto:16s}{pop:24s}{g['n']:12,d}{g['changed']:9,d}{share:8.4f}")

    # ---- Liang MAD: was the opening answer better than the debated one? ----------------------
    print("\n  defect 2: Liang MAD, accuracy at each stage of the debate (parser v2 throughout)")
    for pop in ("wdc_products", "geographic_settlements"):
        ent, _split = em.gold_for(pop)
        stage = collections.defaultdict(list)
        flips = collections.Counter()
        for f in sorted(glob.glob(str(NR / f"raw/{pop}__liang_mad__*.jsonl"))):
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") != "ok":
                    continue
                rec = r.get("record") or {}
                u = r["unit_id"].split("|", 3)[3]
                a, b = u.split("|")[1], u.split("|")[2]
                if a not in ent or b not in ent:
                    continue
                g = 1 if ent[a] == ent[b] else 0
                calls = rec.get("calls") or []
                aff = [c for c in calls if c.get("role") == "affirmative"]
                neg = [c for c in calls if c.get("role") == "negative"]
                mod = [c for c in calls if c.get("role") == "moderator"]
                if aff:
                    p = binary(to_label_v2(aff[0].get("raw_reply")))
                    if p is not None:
                        stage["affirmative round 0, before any peer text"].append((g, p))
                    pl = binary(to_label_v2(aff[-1].get("raw_reply")))
                    if pl is not None:
                        stage["affirmative last round"].append((g, pl))
                    if p is not None and pl is not None:
                        flips["flipped" if p != pl else "held"] += 1
                        if p != pl:
                            flips["flip_to_correct" if pl == g else "flip_to_wrong"] += 1
                if neg:
                    p = binary(to_label_v2(neg[0].get("raw_reply")))
                    if p is not None:
                        stage["negative round 0 (its role is to contradict)"].append((g, p))
                if mod:
                    ans, _side = moderator_answer(mod[-1].get("raw_reply") or "")
                    p = binary(to_label_v2(ans))
                    if p is not None:
                        stage["moderator debate_answer (the endpoint)"].append((g, p))
                p = binary(to_label_v2(rec.get("majority_answer")))
                if p is not None:
                    stage["as collected (parser v1)"].append(
                        (g, binary(rec.get("majority_answer"))))
        prev = sum(g for g, _ in stage["moderator debate_answer (the endpoint)"] or [(0, 0)])
        tot = max(len(stage["moderator debate_answer (the endpoint)"]), 1)
        print(f"\n  {pop}  (prevalence {prev / tot:.4f})")
        res = {}
        for k in ("affirmative round 0, before any peer text",
                  "negative round 0 (its role is to contradict)",
                  "affirmative last round",
                  "moderator debate_answer (the endpoint)",
                  "as collected (parser v1)"):
            if stage[k]:
                m = prf(stage[k])
                res[k] = m
                print(f"    {k:48s} n={m['n']:5d} P={m['precision']:.4f} "
                      f"R={m['recall']:.4f} F1={m['f1']:.4f}")
        held, fl = flips["held"], flips["flipped"]
        if held + fl:
            res["affirmative_revision"] = {
                "held": held, "flipped": fl,
                "flip_share": round(fl / (held + fl), 4),
                "flip_to_correct": flips["flip_to_correct"],
                "flip_to_wrong": flips["flip_to_wrong"],
                "share_of_flips_that_are_wrong": round(
                    flips["flip_to_wrong"] / fl, 4) if fl else None}
            print(f"    affirmative changed its answer on {fl:,} of {held + fl:,} units "
                  f"({fl / (held + fl):.4f}); of those flips {flips['flip_to_wrong']:,} went to the "
                  f"wrong label and {flips['flip_to_correct']:,} to the right one")
        out["defect_2_capitulation"][pop] = res

    # ---- did the mis-parse reach the protocol's own prompts? ---------------------------------
    print("\n  did a parsed label ever enter a prompt (which a re-parse could not undo)?")
    for proto in ("liang_mad", "du_mad", "ford", "bccs", "madm2_naive", "dmad", "dov_vote",
                  "reconcile", "moa"):
        fs = sorted(glob.glob(str(NR / f"raw/wdc_products__{proto}__*.jsonl")))
        if not fs:
            continue
        verdict = "no label-shaped text found in any prompt"
        checked = 0
        for line in open(fs[0], encoding="utf-8"):
            if not line.strip() or checked >= 40:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok":
                continue
            checked += 1
            rec = r.get("record") or {}
            for c in (rec.get("calls") or [])[1:]:
                sent = json.dumps(c.get("messages_sent") or [])
                # a peer's verbatim reply is raw text; a *parsed* label appears as a bare token
                if re.search(r'"(?:non-?match|match)"|answer:\s*(?:non-?match|match)\b',
                             sent, re.I):
                    verdict = "a bare parsed label appears in a later prompt"
                    break
            if verdict.startswith("a bare"):
                break
        out["reached_the_prompts"][proto] = verdict
        print(f"    {proto:16s}{verdict}")

    p = NR / "analysis/label_parse_audit.json"
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
