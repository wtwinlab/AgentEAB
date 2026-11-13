r"""Two required controls that the eight-arm matrix does not cover.

    PYTHONPATH=code python3 code/collect_controls.py --stage tokencap --dataset <d> --split dev
    PYTHONPATH=code python3 code/collect_controls.py --stage nestedpool --dataset <d> --split dev
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = (os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
sys.path.insert(0, os.path.join(STAGE, "code"))
sys.path.insert(0, os.path.join(STAGE, "evidence_repair/code"))

import collect_matrix as CM              # noqa: E402

CHARS_PER_TOKEN = 2.971                  # measured on 3,000 stored D5 rows
CAP_BLOCK_CHARS = 380                    # ~128 prompt tokens; frozen on dev
NESTED_FRACTION = 0.10
NESTED_SEED = 20261016
CONTROL_ARMS = ("E5cap", "F5cap")


OMITTED_NOTICE = "  [remaining comparisons omitted to meet a common input budget]"


def truncate_block(block: str, cap: int) -> Tuple[str, List[str]]:
    """Keep the heading plus as many whole leading lines as fit inside `cap` characters.
    """
    lines = block.splitlines()
    if not lines:
        return block, []
    budget = cap - len("\n") - len(OMITTED_NOTICE)       # reserve room for the notice
    kept = [lines[0]]
    used = len(lines[0])
    dropped: List[str] = []
    for ln in lines[1:]:
        if not dropped and used + 1 + len(ln) <= budget:
            kept.append(ln)
            used += 1 + len(ln)
        else:
            dropped.append(ln.strip())                   # once one line drops, all later ones do
    if dropped:
        kept.append(OMITTED_NOTICE)
    return "\n".join(kept), dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["tokencap", "nestedpool", "nestedarms"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    CM.WORKERS = a.workers

    dom = CM.Domain(a.dataset)
    data = os.path.join(STAGE, "data", a.dataset)
    blob = json.load(open(os.path.join(STAGE, "data/protocol", f"{a.dataset}.json"),
                          encoding="utf-8"))
    ent, sp = blob["entity_of"], blob["split"]
    cands = [(str(x), str(y)) for x, y in blob["candidates"]
             if sp.get(ent.get(str(x))) == a.split and sp.get(ent.get(str(y))) == a.split]

    # MusicBrainz uses the corrected parser for its evidence block
    mb_src: Dict[str, str] = {}
    e5p_fn = None
    if a.dataset == "musicbrainz20k":
        import csv                                                    # noqa: PLC0415
        import evidence_e5p as E5P                                    # noqa: PLC0415
        rows = list(csv.DictReader(open(os.path.join(
            STAGE, "data/musicbrainz/musicbrainz-20-A01.csv.dapo"), encoding="utf-8",
            newline="")))
        mb_src = {r["TID"]: r["SourceID"] for r in rows}
        e5p_fn = E5P.evidence_musicbrainz_parsefix

    def evidence_block(x: str, y: str) -> str:
        if e5p_fn is not None:
            return e5p_fn(dom.fields(x), dom.fields(y), mb_src[x], mb_src[y])
        return dom.evidence(dom.fields(x), dom.fields(y))

    # ---------------------------------------------------------------- nested pool
    if a.stage == "nestedpool":
        out_path = os.path.join(data, f"pool2_{a.split}.jsonl")
        have = CM.done_units(out_path)
        rng = random.Random(NESTED_SEED)
        sample = sorted(rng.sample(cands, max(1, int(NESTED_FRACTION * len(cands)))))
        jobs: List[Dict[str, Any]] = []
        for (x, y) in sample:
            base = dom.base_prompt(x, y)
            for m in CM.POOL:
                for rep in range(CM.POOL_REPEATS):
                    u = CM.uid("pool2", a.dataset, a.split, x, y, m, rep)
                    if u in have:
                        continue
                    jobs.append({"unit_id": u, "a": x, "b": y, "model": m, "repeat": rep,
                                 "prompt": base})
        print(f"  nestedpool/{a.dataset}/{a.split}: {len(sample)} of {len(cands)} pairs "
              f"({100*len(sample)/len(cands):.1f}%), {len(jobs)} calls", flush=True)
        if a.limit:
            jobs = jobs[:a.limit]
        if not jobs:
            return 0

        def dp(j: Dict[str, Any]) -> None:
            g = CM.post(j["model"], j["prompt"], CM.MAX_TOKENS["pool"])
            CM.append(out_path, {
                "unit_id": j["unit_id"], "dataset": a.dataset, "split": a.split,
                "a": j["a"], "b": j["b"], "model": j["model"], "repeat": j["repeat"],
                "role": "pool2", "status": "ok", "decision": dom.parse(g["text"]),
                "raw_reply": g["text"], "finish_reason": g["finish_reason"],
                "truncated": g["finish_reason"] == "length",
                "prompt_tokens": g["prompt_tokens"],
                "completion_tokens": g["completion_tokens"],
                "pool_index": 2, "seed_for_sample": NESTED_SEED,
                "not_additional_sample_size": True})
        st = CM.run(jobs, dp, "pool2")
        print(f"  {st}")
        json.dump({"stats": st, "n_pairs": len(sample), "fraction": NESTED_FRACTION,
                   "seed": NESTED_SEED, "pairs": sample[:50]},
                  open(os.path.join(STAGE, "logs",
                                    f"collect_pool2_{a.dataset}_{a.split}.json"), "w",
                       encoding="utf-8"), indent=1, default=str)
        return 0

    # ---------------------------------------------------------------- D5 under the second pool
    if a.stage == "nestedarms":
        # The nested diagnostic is only informative about *outcomes* if an condition is actually re-run on
        # the second pool's peers.
        out_path = os.path.join(data, f"nestedarms_{a.split}.jsonl")
        have = CM.done_units(out_path)
        pool2 = M_load(os.path.join(data, f"pool2_{a.split}.jsonl"))
        if not pool2:
            print("  pool2 not collected yet")
            return 1
        pv2 = build_peer_view(pool2)
        pairs = sorted({(r["a"], r["b"]) for r in pool2})
        jobs, skipped = [], collections.Counter()
        for (x, y) in pairs:
            for rc in CM.RECIPIENTS:
                others = [m for m in CM.POOL if m != rc]
                peers = [pv2[(x, y, m)] for m in sorted(others) if (x, y, m) in pv2]
                if len(peers) < 2:
                    skipped["fewer than two peers in pool2"] += 1
                    continue
                for rep in range(CM.REPEATS):
                    u = CM.uid("nested", a.dataset, a.split, x, y, rc, "D5p2", rep)
                    if u in have:
                        continue
                    jobs.append({"unit_id": u, "a": x, "b": y, "recipient": rc,
                                 "repeat": rep, "peers": peers})
        print(f"  nestedarms/{a.dataset}/{a.split}: {len(pairs)} pairs, {len(jobs)} calls; "
              + ", ".join(f"{v} {k}" for k, v in skipped.items()), flush=True)
        if a.limit:
            jobs = jobs[:a.limit]
        if not jobs:
            return 0

        def dn(j: Dict[str, Any]) -> None:
            pr = (dom.base_prompt(j["a"], j["b"])
                  + CM.FRAME.format(block=CM.peer_decisions_block(j["peers"])))
            g = CM.post(j["recipient"], pr, CM.MAX_TOKENS["D5"])
            CM.append(out_path, {
                "unit_id": j["unit_id"], "dataset": a.dataset, "split": a.split,
                "a": j["a"], "b": j["b"], "recipient": j["recipient"], "arm": "D5p2",
                "repeat": j["repeat"], "draw": None, "role": "recipient", "status": "ok",
                "decision": dom.parse(g["text"]), "raw_reply": g["text"],
                "finish_reason": g["finish_reason"],
                "truncated": g["finish_reason"] == "length",
                "prompt_tokens": g["prompt_tokens"],
                "completion_tokens": g["completion_tokens"],
                "peer_models": [p["model"] for p in j["peers"]],
                "peer_decisions": [p["decision"] for p in j["peers"]],
                "pool_index": 2, "not_additional_sample_size": True})
        st = CM.run(jobs, dn, "D5 on pool2")
        print(f"  {st}")
        json.dump({"stats": st, "n_pairs": len(pairs), "skipped": dict(skipped)},
                  open(os.path.join(STAGE, "logs",
                                    f"collect_nestedarms_{a.dataset}_{a.split}.json"), "w",
                       encoding="utf-8"), indent=1, default=str)
        return 0

    # ---------------------------------------------------------------- token cap
    out_path = os.path.join(data, f"controls_{a.split}.jsonl")
    have = CM.done_units(out_path)
    pool = M_load(os.path.join(data, f"pool_{a.split}.jsonl"))
    peer_view = build_peer_view(pool)

    jobs = []
    skipped = collections.Counter()
    for (x, y) in cands:
        for rc in CM.RECIPIENTS:
            others = [m for m in CM.POOL if m != rc]
            peers = [peer_view[(x, y, m)] for m in sorted(others) if (x, y, m) in peer_view]
            if len(peers) < 2:
                skipped["fewer than two peers"] += 1
                continue
            for arm in CONTROL_ARMS:
                # A cell is only collected where the cap actually changes the payload.
                raw = (evidence_block(x, y) if arm == "E5cap"
                       else CM.peer_full_block(peers))
                if len(raw) <= CAP_BLOCK_CHARS:
                    skipped[f"{arm}: already within the cap, stored arm reused"] += 1
                    continue
                for rep in range(CM.REPEATS):
                    u = CM.uid("ctrl", a.dataset, a.split, x, y, rc, arm, rep)
                    if u in have:
                        skipped["already collected"] += 1
                        continue
                    jobs.append({"unit_id": u, "a": x, "b": y, "recipient": rc, "arm": arm,
                                 "repeat": rep, "peers": peers})
    print(f"  tokencap/{a.dataset}/{a.split}: {len(jobs)} calls (cap {CAP_BLOCK_CHARS} chars "
          f"~ {CAP_BLOCK_CHARS/CHARS_PER_TOKEN:.0f} tokens); D5 reused, not recollected; "
          + ", ".join(f"{v} {k}" for k, v in skipped.items()), flush=True)
    if a.limit:
        jobs = jobs[:a.limit]
    if not jobs:
        return 0

    rng = random.Random(CM.DRAW_SEED)
    blocks = [jobs[i:i + CM.BLOCK] for i in range(0, len(jobs), CM.BLOCK)]
    for b in blocks:
        rng.shuffle(b)
    jobs = [j for b in blocks for j in b]

    def do(j: Dict[str, Any]) -> None:
        base = dom.base_prompt(j["a"], j["b"])
        if j["arm"] == "E5cap":
            raw = evidence_block(j["a"], j["b"])
            kept, dropped = truncate_block(raw, CAP_BLOCK_CHARS)
            pr = (base + CM.FRAME.format(block=CM.peer_decisions_block(j["peers"]))
                  + "\n\n" + kept)
        else:
            raw = CM.peer_full_block(j["peers"])
            kept, dropped = truncate_block(raw, CAP_BLOCK_CHARS)
            pr = base + CM.FRAME.format(block=kept)
        mt = CM.MAX_TOKENS["E5"]
        g = CM.post(j["recipient"], pr, mt)
        CM.append(out_path, {
            "unit_id": j["unit_id"], "dataset": a.dataset, "split": a.split,
            "a": j["a"], "b": j["b"], "recipient": j["recipient"], "arm": j["arm"],
            "repeat": j["repeat"], "draw": None, "role": "recipient", "status": "ok",
            "decision": dom.parse(g["text"]), "raw_reply": g["text"],
            "finish_reason": g["finish_reason"], "truncated": g["finish_reason"] == "length",
            "prompt_tokens": g["prompt_tokens"], "completion_tokens": g["completion_tokens"],
            "peer_models": [p["model"] for p in j["peers"]],
            "peer_decisions": [p["decision"] for p in j["peers"]],
            "prompt_sha": hashlib.sha256(pr.encode()).hexdigest()[:16], "max_tokens": mt,
            "cap_block_chars": CAP_BLOCK_CHARS,
            "cap_tokens_nominal": round(CAP_BLOCK_CHARS / CHARS_PER_TOKEN),
            "block_chars_before_cap": len(raw), "block_chars_after_cap": len(kept),
            "n_dropped_lines": len(dropped), "dropped_lines": dropped[:12]})
    st = CM.run(jobs, do, "+".join(CONTROL_ARMS))
    print(f"  {st}")
    json.dump({"stats": st, "cap_block_chars": CAP_BLOCK_CHARS,
               "chars_per_token_measured": CHARS_PER_TOKEN,
               "cap_chosen_on": "dev, from the E5p block's median natural length; never from a "
                                "test outcome",
               "d5_reused_not_recollected": True, "skipped": dict(skipped)},
              open(os.path.join(STAGE, "logs",
                                f"collect_controls_{a.dataset}_{a.split}.json"), "w",
                   encoding="utf-8"), indent=1, default=str)
    return 0


def M_load(p: str) -> List[Dict[str, Any]]:
    import metrics as M                                               # noqa: PLC0415
    return M.load_jsonl(p)


def build_peer_view(pool: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """collect_matrix's frozen peer rule: modal over repeats, tie -> abstain."""
    cell: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in pool:
        cell[(r["a"], r["b"], r["model"])].append(r)
    out = {}
    for k, rs in cell.items():
        v = [r["decision"] for r in rs if r["decision"] is not None]
        d: Optional[str] = None
        raw: Optional[str] = None
        if v:
            c = collections.Counter(v)
            top = max(c.values())
            w = [kk for kk, n in c.items() if n == top]
            if len(w) == 1:
                d = w[0]
                raw = next(r["raw_reply"] for r in rs if r["decision"] == d)
        out[k] = {"model": k[2], "decision": d, "raw_reply": raw, "n_repeats": len(rs)}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
