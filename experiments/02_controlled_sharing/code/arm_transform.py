#!/usr/bin/env python3
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import answer_extract as AX

ARMS = ("FULL", "NOPEER", "ANSONLY", "UNIFSEL")

PEER_PREFIX = {
    "bccs": "These are the solutions to the problem from other agents:",
    "du_mad": "These are the solutions to the problem from other agents:",
    "madm2_objective": "These are the solutions to the problem from other agents:",
    "madm2_naive": "These are the solutions to the problem from other agents:",
}
PEER_CLAUSES = [
    ("Selecting and using the trustable solutions from current collaboration as additional "
     "information, can you provide your answer to the problem?", "Can you provide your answer to the problem?"),
    ("Selecting and using the leading solutions from current collaboration as additional "
     "information, can you provide your answer to the problem?", "Can you provide your answer to the problem?"),
    ("Using the solutions from other agents as additional information, can you provide your answer "
     "to the question?", "Can you provide your answer to the question?"),
    ("Using the solutions from other agents as additional information, can you provide your answer "
     "to the math problem?", "Can you provide your answer to the math problem?"),
    ("Using the solutions from other agents as additional information, give your",
     "Give your"),
    ("Read the question and our opinions carefully", "Read the question carefully"),
]
BLOCK_LABEL_RE = re.compile(
    r"\n\n ?One (?:agent|leader|supporting agent|conflicting agent) solution: ```")
BLOCK_TAIL_RE = re.compile(
    r"\n\n ?(?:Using the solutions from other agents|"
    r"Selecting and using the (?:trustable|leading) solutions)")
BLOCK_RE = re.compile(
    r"\n\n ?One (?:agent|leader|supporting agent|conflicting agent) solution: ```([^`]*?)```", re.S)
MADM2_BLOCK_RE = re.compile(
    r"\n\n\nOne agent's solution: (.*?)(?=\n\n\nOne agent's solution: |\n\nUsing the solutions"
    r" from other agents|\Z)", re.S)
MOA_REFS_HEADER = "Responses from models:"
MOA_REF_RE = re.compile(r"\n(\d+)\. (.*?)(?=\n\d+\. |\Z)", re.S)
MOA_MARKER = "You have been provided with a set of responses"
FORD_INSTR_MARKER = "Read the question and our opinions carefully"
_FORD_INSTR_CACHE: Dict[str, List[str]] = {}


def _ford_instructions(benchmark: str) -> List[str]:
    key = benchmark or ""
    if key in _FORD_INSTR_CACHE:
        return _FORD_INSTR_CACHE[key]
    out: List[str] = []
    try:
        from consensus_delivery.adapters import ford as _f
        for n in (2, 3, 4, 5, 6, 10):
            try:
                t = _f.instruction(key, n)
            except Exception:
                continue
            if t and t not in out:
                out.append(t)
    except Exception:
        pass
    out.sort(key=len, reverse=True)
    _FORD_INSTR_CACHE[key] = out
    return out


def _ford_split_instruction(text: str, benchmark: str) -> Tuple[str, str]:
    t = text or ""
    for instr in _ford_instructions(benchmark):
        if t.endswith("\n" + instr):
            return "\n" + instr, t[: -(len(instr) + 1)]
        if t.endswith(instr):
            return instr, t[: -len(instr)]
    j = t.rfind(FORD_INSTR_MARKER)
    if j > 0:
        cut = j - 1 if t[j - 1] == "\n" else j
        return t[cut:], t[:cut]
    return "", t


def peer_blocks(protocol: str, text: str):
    t = text or ""
    if protocol.startswith("madm2"):
        return [(m.start(), m.end(), m.group(1)) for m in MADM2_BLOCK_RE.finditer(t)]
    labels = list(BLOCK_LABEL_RE.finditer(t))
    if not labels:
        return []
    labels = [m for i, m in enumerate(labels) if i == 0 or t.endswith("```", 0, m.start())]
    tails = [m.start() for m in BLOCK_TAIL_RE.finditer(t) if m.start() >= labels[-1].end()]
    region_end = tails[-1] if tails else len(t)
    out = []
    for i, m in enumerate(labels):
        seg_end = labels[i + 1].start() if i + 1 < len(labels) else region_end
        body_at = m.end()
        close = t.rfind("```", body_at, seg_end)
        if close < 0:
            out.append((m.start(), seg_end, t[body_at:seg_end]))
        else:
            out.append((m.start(), close + 3, t[body_at:close]))
    return out


def moa_refs(text: str, observed_replies: Optional[Sequence[str]] = None):
    t = text or ""
    i = t.find(MOA_REFS_HEADER)
    if i < 0:
        return []
    base = i + len(MOA_REFS_HEADER)
    if observed_replies:
        for k in range(len(observed_replies), 0, -1):
            cand = list(observed_replies[:k])
            suffix = "".join("\n%d. %s" % (n + 1, o or "") for n, o in enumerate(cand))
            if t[base:base + len(suffix)] == suffix:
                out, pos = [], base
                for n, o in enumerate(cand):
                    head = "\n%d. " % (n + 1)
                    body_at = pos + len(head)
                    out.append((pos, body_at + len(o or ""), str(n + 1), o or ""))
                    pos = body_at + len(o or "")
                return out
    out = []
    for m in MOA_REF_RE.finditer(t[base:]):
        out.append((base + m.start(), base + m.end(), m.group(1), m.group(2)))
    return out


def moa_refs_exact(text: str, observed_replies: Optional[Sequence[str]] = None) -> bool:
    t = text or ""
    i = t.find(MOA_REFS_HEADER)
    if i < 0 or not observed_replies:
        return False
    base = i + len(MOA_REFS_HEADER)
    for k in range(len(observed_replies), 0, -1):
        suffix = "".join("\n%d. %s" % (n + 1, o or "")
                         for n, o in enumerate(observed_replies[:k]))
        if t[base:base + len(suffix)] == suffix:
            return True
    return False


_RENDER = {
    "bccs": lambda lab: "\\boxed{%s}" % lab,
    "du_mad": lambda lab: "\\boxed{%s}" % lab,
    "madm2_objective": lambda lab: "Answer: %s" % lab,
    "madm2_naive": lambda lab: "Answer: %s" % lab,
    "ford": lambda lab: 'Answer: %s is more plausible.' % lab,
    "moa": lambda lab: lab,
}


def _label_via_adapter(text: str, protocol: str, kind: str,
                       benchmark: str = "") -> Optional[str]:
    t = text or ""
    if not t.strip():
        return None
    k = kind or "binary_match"
    b = benchmark or "matching_pair"
    try:
        from consensus_delivery.adapters import bccs, du_mad, ford, mad_m2, moa
        if protocol == "bccs":
            lab = bccs.extract_boxed(t, k)
        elif protocol == "du_mad":
            lab = du_mad.extract_for(b, t)
        elif protocol.startswith("madm2"):
            lab = mad_m2.extract_answer(t, k)
        elif protocol == "ford":
            lab = ford.parse_prediction(t, b)[0]
            if lab in (None, "", "None", "none"):
                lab = None
        elif protocol == "moa":
            lab = moa.extract(t, k)
        else:
            return None
    except Exception:
        return None
    if lab in (None, "", "None"):
        return None
    return str(lab).strip() or None


def _extract_answer(text: str, protocol: str, kind: str,
                    benchmark: str = "") -> Tuple[Optional[str], str]:
    lab, tier = AX.extract(text, kind or "binary_match",
                           lambda t: _label_via_adapter(t, protocol, kind, benchmark))
    if lab is None:
        return None, tier
    render = _RENDER.get(protocol)
    return (render(lab) if render else lab), tier


def transform(protocol: str, arm: str, messages: Sequence[Dict[str, str]], *,
              kind: str = "", seed: int = 20261019, call_seq: int = 0, benchmark: str = "",
              observed_replies: Optional[Sequence[str]] = None,
              ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    msgs = [dict(m) for m in messages]
    tr: Dict[str, Any] = {"arm": arm, "protocol": protocol, "peer_blocks_found": 0,
                          "peer_indices_kept": [], "n_peers_rendered": 0,
                          "ansonly_extract_failed": 0, "ansonly_tier": {"contract": 0, "nl": 0},
                          "clauses_rewritten": [],
                          "changed": False}
    if arm == "FULL":
        for m in msgs:
            c = m.get("content") or ""
            tr["peer_blocks_found"] += len(peer_blocks(protocol, c))
            if MOA_MARKER in c:
                tr["peer_blocks_found"] += len(moa_refs(c, observed_replies))
        if protocol == "ford":
            tr["peer_blocks_found"] = max(len(msgs) - 1, 0)
        tr["n_peers_rendered"] = tr["peer_blocks_found"]
        tr["peer_indices_kept"] = list(range(tr["peer_blocks_found"]))
        return msgs, tr

    rng = random.Random((seed, protocol, arm, call_seq).__hash__() & 0xFFFFFFFF)

    if protocol == "ford":
        peer_msgs = list(range(1, len(msgs)))
        tr["peer_blocks_found"] = len(peer_msgs)
        instr = ""
        if msgs:
            instr, body = _ford_split_instruction(msgs[-1].get("content") or "", benchmark)
            if instr:
                msgs[-1]["content"] = body
        if arm == "NOPEER":
            msgs = [msgs[0]] if msgs else msgs
            tr["changed"] = len(peer_msgs) > 0
        elif arm == "ANSONLY":
            kept = []
            for i in peer_msgs:
                got, tier = _extract_answer(msgs[i].get("content") or "", "ford", kind, benchmark)
                if got is None:
                    tr["ansonly_extract_failed"] += 1
                else:
                    msgs[i]["content"] = got
                    tr["ansonly_tier"][tier] += 1
                    kept.append(i - 1)
                    tr["changed"] = True
            tr["peer_indices_kept"] = kept
        else:
            tr["note"] = ("FORD has one peer, so there is no peer to choose between; its "
                          "selection/use factor is the final-answer rule and is applied outside "
                          "the prompt (see use_arm)")
        if instr and msgs:
            msgs[-1]["content"] = (msgs[-1].get("content") or "") + instr
        for a, b in PEER_CLAUSES:
            for m in msgs:
                if a in (m.get("content") or "") and arm == "NOPEER":
                    m["content"] = m["content"].replace(a, b)
                    tr["clauses_rewritten"].append(a[:40])
                    tr["changed"] = True
        tr["n_peers_rendered"] = 0 if arm == "NOPEER" else (
            len(tr["peer_indices_kept"]) if arm == "ANSONLY" else len(peer_msgs))
        return msgs, tr

    if protocol == "moa":
        for m in msgs:
            c = m.get("content") or ""
            if MOA_MARKER not in c:
                continue
            refs = moa_refs(c, observed_replies)
            tr["peer_blocks_found"] = len(refs)
            if not refs:
                continue
            hdr = c.find(MOA_REFS_HEADER)
            if arm == "NOPEER":
                head = c[:hdr].rstrip()
                m["content"] = (head.replace(
                    "You have been provided with a set of responses from various open-source "
                    "models to the latest user query. Your task is to synthesize these responses "
                    "into a single, high-quality response.",
                    "Answer the latest user query directly with a single, high-quality response.")
                    .replace("these responses", "your own reasoning"))
                tr["clauses_rewritten"].append("aggregator synthesis framing -> direct answer")
                tr["n_peers_rendered"] = 0
                tr["changed"] = True
            elif arm == "ANSONLY":
                new = c[:hdr] + MOA_REFS_HEADER
                for _s0, _e0, n, body in refs:
                    got, tier = _extract_answer(body, "moa", kind, benchmark)
                    if got is None:
                        tr["ansonly_extract_failed"] += 1
                        new += f"\n{n}. {body}"
                    else:
                        new += f"\n{n}. {got}"
                        tr["ansonly_tier"][tier] += 1
                        tr["changed"] = True
                m["content"] = new
                tr["peer_indices_kept"] = list(range(len(refs)))
                tr["n_peers_rendered"] = len(refs)
            elif arm == "UNIFSEL":
                keep = sorted(rng.sample(range(len(refs)), 1))
                new = c[:hdr] + MOA_REFS_HEADER
                for j, (_s0, _e0, _n, body) in enumerate(refs):
                    if j in keep:
                        new += f"\n{len(tr['peer_indices_kept']) + 1}. {body}"
                        tr["peer_indices_kept"].append(j)
                m["content"] = new
                tr["n_peers_rendered"] = len(keep)
                tr["changed"] = True
        return msgs, tr

    for m in msgs:
        c = m.get("content") or ""
        blocks = peer_blocks(protocol, c)
        if not blocks:
            continue
        tr["peer_blocks_found"] += len(blocks)
        if arm == "NOPEER":
            new = c
            for st, en, _body in reversed(blocks):
                new = new[:st] + new[en:]
            pref = PEER_PREFIX.get(protocol)
            if pref:
                new = new.replace(pref, "").lstrip()
            m["content"] = new
            tr["n_peers_rendered"] = 0
            tr["changed"] = True
        elif arm == "ANSONLY":
            new, kept = c, []
            for j, (st, en, body) in reversed(list(enumerate(blocks))):
                got, tier = _extract_answer(body, protocol, kind, benchmark)
                if got is None:
                    tr["ansonly_extract_failed"] += 1
                    continue
                lm = None if protocol.startswith("madm2") else BLOCK_LABEL_RE.match(c, st)
                bs = lm.end() if lm else c.index(body, st, en)
                new = new[:bs] + got + new[bs + len(body):]
                kept.append(j)
                tr["ansonly_tier"][tier] += 1
                tr["changed"] = True
            m["content"] = new
            tr["peer_indices_kept"] = sorted(kept)
            tr["n_peers_rendered"] = len(blocks)
        elif arm == "UNIFSEL":
            k = max(1, len(blocks) // 2) if len(blocks) > 1 else len(blocks)
            keep = set(rng.sample(range(len(blocks)), k))
            new = c
            for j, (st, en, _body) in reversed(list(enumerate(blocks))):
                if j not in keep:
                    new = new[:st] + new[en:]
            m["content"] = new
            tr["peer_indices_kept"] = sorted(keep)
            tr["n_peers_rendered"] = len(keep)
            tr["changed"] = len(keep) != len(blocks)
    if arm == "NOPEER":
        for a, b in PEER_CLAUSES:
            for m in msgs:
                if a in (m.get("content") or ""):
                    m["content"] = m["content"].replace(a, b)
                    tr["clauses_rewritten"].append(a[:40])
                    tr["changed"] = True
    return msgs, tr


def use_arm_final_answer(protocol: str, record: Dict[str, Any]) -> Dict[str, Any]:
    preds = []
    for c in record.get("calls") or []:
        p = c.get("prediction")
        if p not in (None, "", "None"):
            preds.append(p)
    released = record.get("majority_answer")
    maj = None
    if preds:
        counts: Dict[str, int] = {}
        for p in preds:
            counts[p] = counts.get(p, 0) + 1
        maj = max(counts.items(), key=lambda kv: kv[1])[0]
    return {"released_rule_answer": released, "all_round_majority": maj,
            "n_predictions": len(preds), "rules_agree": released == maj}
