"""Construct DBLP–ACM entity-disjoint samples and four input presentations. Scoring imports the fixed menu."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
DATA = WORK / "data"
EXP = DATA / "dblp_acm/exp_data"

FIELDS = ("title", "authors", "venue", "year")
MENU = ("no_peer", "peer_decision", "decision_plus_fields", "full_explanation")
N_DEV, N_TEST = 500, 800
SPLIT_SALT = "cis2-dblp-acm-20261028"

TASK_LINE = "Decide whether the two publication records below refer to the SAME real-world publication."
ANSWER_LINE = "The answer is either match or non-match."
STEP_LINE = "Let's think step by step, and put your final answer after 'Answer:'."
PEER_PREFIX = "These are the solutions to the problem from other agents:\n"
PEER_SUFFIX = ("\n\nUsing the solutions from other agents as additional information, give your "
               "updated answer. Let's think step by step, and put your final answer after "
               "'Answer:'.")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().casefold()


def record_block(name: str, row: Dict[str, str]) -> str:
    body = "".join(f"\n  {f}: {row.get(f, '')}" for f in FIELDS)
    return f"Record {name}:{body}"


def base_problem(a: Dict[str, str], b: Dict[str, str]) -> str:
    return (f"{TASK_LINE}\n\n{record_block('A', a)}\n\n{record_block('B', b)}\n\n{ANSWER_LINE}")


def field_block(a: Dict[str, str], b: Dict[str, str]) -> str:
    lines = []
    for f in FIELDS:
        same = norm_text(a.get(f, "")) == norm_text(b.get(f, ""))
        lines.append(f"  {f}: {'identical' if same else 'different'}")
        if f in ("title", "authors"):
            d = len(str(a.get(f, ""))) - len(str(b.get(f, "")))
            lines[-1] += f"; length difference {d:+d} characters"
    return "Field comparison between the two records:\n" + "\n".join(lines)


def build_prompt(arm: str, a: Dict[str, str], b: Dict[str, str],
                 peer_decision: Optional[str], peer_text: Optional[str]) -> str:
    problem = base_problem(a, b)
    if arm == "no_peer":
        return f"{problem}\n\n{STEP_LINE}"
    if arm == "peer_decision":
        peer = f"Answer: {peer_decision}"
    elif arm == "decision_plus_fields":
        peer = f"{field_block(a, b)}\n\nAnswer: {peer_decision}"
    elif arm == "full_explanation":
        peer = str(peer_text or "").strip()
    else:
        raise AssertionError(arm)
    return (PEER_PREFIX + f"\n\nOne agent's solution: {peer}" + PEER_SUFFIX
            + f"\n\nThe problem:\n{problem}")


def load_tables() -> Tuple[Dict[int, Dict[str, str]], Dict[int, Dict[str, str]]]:
    A = {int(r["id"]): r for r in csv.DictReader(open(EXP / "tableA.csv", encoding="utf-8"))}
    B = {int(r["id"]): r for r in csv.DictReader(open(EXP / "tableB.csv", encoding="utf-8"))}
    return A, B


def load_pairs() -> List[Tuple[int, int, int, str]]:
    out = []
    for f in ("train.csv", "valid.csv", "test.csv"):
        for r in csv.DictReader(open(EXP / f, encoding="utf-8")):
            out.append((int(r["ltable_id"]), int(r["rtable_id"]), int(r["label"]), f))
    return out


def entity_split(pairs) -> Tuple[Dict[Any, str], Dict[str, Any]]:
    """Components over the positives, then components to sides by hash of the canonical key."""
    par: Dict[Any, Any] = {}

    def find(x):
        while par.setdefault(x, x) != x:
            par[x] = par.setdefault(par[x], par[x])
            x = par[x]
        return x

    for l, r, lab, _ in pairs:
        find(("A", l))
        find(("B", r))
        if lab == 1:
            ra, rb = find(("A", l)), find(("B", r))
            if ra != rb:
                par[ra] = rb
    comp: Dict[Any, List[Any]] = {}
    for node in list(par):
        comp.setdefault(find(node), []).append(node)
    side = {}
    sizes = {}
    for root, members in comp.items():
        key = "|".join(sorted(f"{t}{i}" for t, i in members))
        h = int(hashlib.sha256((SPLIT_SALT + key).encode()).hexdigest(), 16)
        side[root] = "dev" if h % 2 == 0 else "test"
        sizes[len(members)] = sizes.get(len(members), 0) + 1
    node_side = {n: side[find(n)] for n in par}
    return node_side, dict(components=len(comp), component_size_histogram=dict(sorted(sizes.items())),
                           dev_components=sum(1 for v in side.values() if v == "dev"),
                           test_components=sum(1 for v in side.values() if v == "test"))


def stratified(pairs_side: List[Tuple[int, int, int, str]], n: int,
               salt: str) -> List[Tuple[int, int, int, str]]:
    """Half matches, half non-matches, chosen by hash so the draw carries no RNG state."""
    pos = sorted([p for p in pairs_side if p[2] == 1],
                 key=lambda p: hashlib.sha256(f"{salt}|{p[0]}|{p[1]}".encode()).hexdigest())
    neg = sorted([p for p in pairs_side if p[2] == 0],
                 key=lambda p: hashlib.sha256(f"{salt}|{p[0]}|{p[1]}".encode()).hexdigest())
    k = n // 2
    return pos[:k] + neg[:k]


def unit_key(split: str, l: int, r: int) -> str:
    return f"dblp_acm|{split}|L:{l}|R:{r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("peer", "recipient"))
    ap.add_argument("--peer-raw", default="output/raw/mt_peer_qwen2.5-7b.jsonl")
    a = ap.parse_args()

    A, B = load_tables()
    pairs = load_pairs()
    node_side, split_info = entity_split(pairs)

    kept, straddling = {"dev": [], "test": []}, 0
    for p in pairs:
        sa, sb = node_side[("A", p[0])], node_side[("B", p[1])]
        if sa != sb:
            straddling += 1
            continue
        kept[sa].append(p)

    dev = stratified(kept["dev"], N_DEV, "dev")
    test = stratified(kept["test"], N_TEST, "test")
    chosen = [("dev", p) for p in dev] + [("test", p) for p in test]

    manifest = dict(
        dataset="DeepMatcher Structured DBLP-ACM",
        url="https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/dblp_acm_exp_data.zip",
        zip_sha256=hashlib.sha256((DATA / "dblp_acm/dblp_acm_exp_data.zip").read_bytes()).hexdigest(),
        tableA_records=len(A), tableB_records=len(B), labelled_pairs=len(pairs),
        positives=sum(1 for p in pairs if p[2] == 1),
        released_split="pair split (train/valid/test); NOT entity-disjoint and not used as one",
        entity_split=dict(split_info, salt=SPLIT_SALT,
                          straddling_pairs_dropped=straddling,
                          dev_pairs_available=len(kept["dev"]),
                          test_pairs_available=len(kept["test"])),
        grouping_limit=("every positive component has exactly two members, so no true cluster "
                        "exceeds two records and a grouping/transitive-closure question is not "
                        "defined on this dataset"),
        candidate_coverage=("the released candidate set contains every labelled positive, so a "
                            "perfect judgment reaches direct-link F1 1.0 on it; this is the "
                            "coverage ceiling that WDC's blocked candidate set does not have"),
        menu=list(MENU), n_dev=len(dev), n_test=len(test),
        dev_positives=sum(1 for p in dev if p[2] == 1),
        test_positives=sum(1 for p in test if p[2] == 1),
        selection_rule=("highest F1(match) on the development side; ties resolved by fewer input "
                        "tokens, then alphabetically"),
        label_isolation=("labels appear in no prompt. Only development labels are read for "
                         "selection. Test labels are read once, after the selection is fixed."),
    )
    (DATA / "mt_dblp_acm_manifest.json").write_text(json.dumps(manifest, indent=1),
                                                    encoding="utf-8")

    gold = {unit_key(s, p[0], p[1]): ("match" if p[2] == 1 else "non-match")
            for s, p in chosen}
    (DATA / "mt_dblp_acm_gold.json").write_text(json.dumps(gold), encoding="utf-8")

    if a.stage == "peer":
        out = DATA / "mt_peer_inputs.jsonl"
        with open(out, "w") as f:
            for s, p in chosen:
                prompt = f"{base_problem(A[p[0]], B[p[1]])}\n\n{STEP_LINE}"
                f.write(json.dumps(dict(
                    unit=unit_key(s, p[0], p[1]) + "|peer", split=s, ltable_id=p[0],
                    rtable_id=p[1], role="peer",
                    messages=[{"role": "user", "content": prompt}]), ensure_ascii=False) + "\n")
        print(json.dumps(dict(stage="peer", requests=sum(1 for _ in open(out)),
                              out=str(out.relative_to(WORK)),
                              dev=len(dev), test=len(test),
                              straddling_pairs_dropped=straddling), indent=1))
        return 0

    raw = WORK / a.peer_raw
    if not raw.exists():
        print(f"FATAL: peer raw log missing: {raw}")
        return 1
    peer: Dict[str, Dict[str, Any]] = {}
    for line in open(raw):
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        txt = (r["response"]["choices"][0]["message"].get("content") or "")
        tail = txt.split("Answer:")[-1] if "Answer:" in txt else ""
        low = tail.lower()
        dec = ("non-match" if "non-match" in low or "non match" in low
               else "match" if "match" in low else None)
        peer[r["unit_id"].removesuffix("|peer")] = dict(decision=dec, text=txt.strip())

    out = DATA / "mt_recipient_inputs.jsonl"
    n_unreadable = 0
    with open(out, "w") as f:
        for s, p in chosen:
            uk = unit_key(s, p[0], p[1])
            pr = peer.get(uk)
            if pr is None:
                print(f"FATAL: no peer output for {uk}")
                return 1
            if pr["decision"] is None:
                n_unreadable += 1
            for arm in MENU:
                prompt = build_prompt(arm, A[p[0]], B[p[1]],
                                      pr["decision"] or "(no readable answer)", pr["text"])
                f.write(json.dumps(dict(
                    unit=f"{uk}|{arm}", split=s, arm=arm, ltable_id=p[0], rtable_id=p[1],
                    peer_decision=pr["decision"], peer_chars=len(pr["text"]),
                    prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()[:16],
                    messages=[{"role": "user", "content": prompt}]), ensure_ascii=False) + "\n")
    print(json.dumps(dict(stage="recipient", requests=sum(1 for _ in open(out)),
                          out=str(out.relative_to(WORK)), arms=list(MENU),
                          pairs=len(chosen), peer_unreadable_decisions=n_unreadable), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
