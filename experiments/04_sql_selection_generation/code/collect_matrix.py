r"""Unified eight-arm collection for one dataset and one split.

    PYTHONPATH=experiments/06_cost_resources/code:experiments/05_matching_downstream/code:experiments/04_sql_selection_generation/code \
      python3 experiments/04_sql_selection_generation/code/collect_matrix.py \
        --dataset geographic_settlements --split dev --stage pool
      ... --stage conditions --arms N,D5,E5,F5
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import pathlib
import random
import sys
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Derived from this file's own location rather than hardcoded, so the same code runs from the
# laptop and from the GPU host where the collection tree is mirrored.
ROOT = os.environ.get("PAM_REPO_ROOT") or str(
    pathlib.Path(__file__).resolve().parents[3])
STAGE = os.path.join(ROOT, "experiments/04_sql_selection_generation")
PROTO = os.path.join(ROOT, "data/records/matching_protocol")

PORTS = {"qwen2.5-7b": 9101, "llama-3-8b": 9102, "qwen2.5-32b": 9103,
         "mistral-7b": 9104, "qwen2.5-14b": 9105, "qwen2.5-72b": 9106}
POOL = ("qwen2.5-7b", "llama-3-8b", "mistral-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b")
RECIPIENTS = ("qwen2.5-7b", "qwen2.5-32b")
ARMS = ("N", "D5", "E5", "F5", "UD1", "UF1", "AD1", "VD1")
REPEATS = 3
POOL_REPEATS = 3
PEER_DRAWS = 3
SEED = 20261013
DRAW_SEED = 20261015
MAX_TOKENS = {"N": 192, "D5": 192, "E5": 256, "F5": 256,
              "UD1": 192, "UF1": 256, "AD1": 192, "VD1": 192, "pool": 192, "verify": 64}
WORKERS = 32
BLOCK = 4000          # pairs per randomised block

_lock = threading.Lock()


def post(model: str, prompt: str, max_tokens: int, timeout: float = 240.0) -> Dict[str, Any]:
    body = json.dumps({"model": model, "temperature": 0.0, "top_p": 1.0,
                       "max_tokens": max_tokens, "seed": SEED,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://{os.environ.get('INFERENCE_HOST', 'inference-host')}:{PORTS[model]}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    return {"text": ch["message"]["content"], "finish_reason": ch.get("finish_reason"),
            "prompt_tokens": d["usage"]["prompt_tokens"],
            "completion_tokens": d["usage"]["completion_tokens"]}


def append(path: str, obj: Dict[str, Any]) -> None:
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def done_units(path: str) -> set:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["unit_id"])
            except Exception:
                pass
    return out


def uid(*p: Any) -> str:
    return hashlib.sha256("|".join(str(x) for x in p).encode()).hexdigest()[:20]


def run(jobs: List[Dict[str, Any]], fn, label: str) -> Dict[str, Any]:
    t0 = time.time()
    st = {"n": 0, "err": 0, "errs": collections.Counter()}

    def wrap(j):
        try:
            fn(j)
            with _lock:
                st["n"] += 1
                if st["n"] % 5000 == 0:
                    el = time.time() - t0
                    print(f"    {label}: {st['n']}/{len(jobs)} {el:.0f}s ({st['n']/el:.1f}/s)",
                          flush=True)
        except Exception as e:
            with _lock:
                st["err"] += 1
                st["errs"][type(e).__name__] += 1

    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(wrap, jobs))
    el = time.time() - t0
    print(f"  {label}: {st['n']} ok, {st['err']} err {dict(st['errs'])}, {el:.0f}s "
          f"({st['n']/max(el,1e-9):.1f}/s)", flush=True)
    return {"n_ok": st["n"], "n_err": st["err"], "errors": dict(st["errs"]), "wall_s": el}


# ------------------------------------------------------------------ dataset adapters
class Domain:
    """Record text, pair rendering and the evidence template for one dataset."""

    def __init__(self, name: str):
        self.name = name
        import evidence_templates as EV                                 # noqa: PLC0415
        self.evidence = EV.TEMPLATES[name]
        blob = json.load(open(os.path.join(PROTO, f"{name}.json"), encoding="utf-8"))
        self.entity_of = blob["entity_of"]
        self.split_of = blob["split"]
        self.candidates = [tuple(c) for c in blob["candidates"]]
        if name == "geographic_settlements":
            sys.path.insert(0, os.path.join(ROOT, "experiments/05_matching_downstream/code"))
            import gs_common as G                                       # noqa: PLC0415
            self._gs = G.Settlements()
            self._render = G.render
            self._parse = G.parse_decision_v2
            self._fields = lambda rid: self._gs.record_text(int(rid))
            self.task = ("Decide whether the two geographical records below refer to the SAME "
                         "real-world settlement.")
        elif name == "wdc_products":
            sys.path.insert(0, os.path.join(ROOT, "experiments/06_cost_resources/code"))
            import em_common as E                                       # noqa: PLC0415
            m = E.Matching("wdc_products")
            self._rec: Dict[str, Dict[str, Any]] = {}
            for sp, rows in m.splits.items():
                for r in rows:
                    L, R = m.pair_record(r)
                    self._rec[f"L:{r.get('id_left')}"] = L
                    self._rec[f"R:{r.get('id_right')}"] = R
            self._fields = lambda rid: self._rec[rid]
            self._parse = _parse_match
            self.task = ("Decide whether the two product records below refer to the SAME "
                         "real-world product.")
        elif name == "musicbrainz20k":
            rows = list(csv.DictReader(open(
                os.path.join(STAGE, "data/musicbrainz/musicbrainz-20-A01.csv.dapo"),
                encoding="utf-8", newline="")))
            vis = ("title", "artist", "album", "year", "length", "number", "language")
            self._rec = {r["TID"]: {k: r.get(k) for k in vis} for r in rows}
            self._fields = lambda rid: self._rec[rid]
            self._parse = _parse_match
            self.task = ("Decide whether the two music recording records below refer to the SAME "
                         "real-world track.")
        else:
            raise AssertionError(name)

    def fields(self, rid: str) -> Dict[str, Any]:
        return self._fields(rid)

    def base_prompt(self, a: str, b: str) -> str:
        if self.name == "geographic_settlements":
            return self._render(self.fields(a), self.fields(b))
        def fmt(d):
            return "\n".join(f"  {k}: {v}" for k, v in d.items() if v not in (None, ""))
        return (f"{self.task}\n\nRecord A:\n{fmt(self.fields(a))}\n\n"
                f"Record B:\n{fmt(self.fields(b))}\n\n"
                "Answer with exactly one word on the final line, either match or non-match.")

    def parse(self, text: str) -> Optional[str]:
        return self._parse(text)


def _parse_match(text: str) -> Optional[str]:
    """Final-line-first, negation-aware single-word decision parser."""
    t = (text or "").strip().lower()
    if not t:
        return None
    for line in reversed([l.strip() for l in t.splitlines() if l.strip()]):
        if "non-match" in line or "non match" in line or "nonmatch" in line:
            return "non-match"
        if "not the same" in line or "do not match" in line or "does not match" in line:
            return "non-match"
        if "match" in line:
            return "match"
        if line in ("yes", "same"):
            return "match"
        if line in ("no", "different"):
            return "non-match"
    return None


# ------------------------------------------------------------------ peer block rendering
def peer_decisions_block(peers: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(f"  system {i+1}: {p['decision'] or 'no answer'}"
                     for i, p in enumerate(peers))


def peer_full_block(peers: Sequence[Dict[str, Any]]) -> str:
    return "\n\n".join(f"  system {i+1} said: {(p['raw_reply'] or '').strip()[:700]}"
                       for i, p in enumerate(peers))


FRAME = ("\n\nOther systems have already judged this pair:\n{block}\n\n"
         "Using this as additional information, give your own answer. "
         "Answer with exactly one word on the final line, either match or non-match.")


def build_prompt(dom: Domain, arm: str, a: str, b: str,
                 peers: Sequence[Dict[str, Any]]) -> Optional[str]:
    base = dom.base_prompt(a, b)
    if arm == "N":
        return base
    if arm in ("D5", "AD1", "VD1", "UD1"):
        return base + FRAME.format(block=peer_decisions_block(peers))
    if arm in ("F5", "UF1"):
        return base + FRAME.format(block=peer_full_block(peers))
    if arm == "E5":
        ev = dom.evidence(dom.fields(a), dom.fields(b))
        return base + FRAME.format(block=peer_decisions_block(peers)) + "\n\n" + ev
    raise AssertionError(arm)


VERIFY_PROMPT = (
    "A system judged whether two records describe the same real-world entity.\n\n"
    "{base}\n\nThe system answered: {dec}\n\n"
    "Rate how likely that answer is correct, on a scale of 1 to 5, where 5 means almost certainly "
    "correct. Reply with only the digit.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(
        ("geographic_settlements", "wdc_products", "musicbrainz20k")))
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--stage", required=True, choices=["pool", "verify", "arms"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(STAGE, "code"))
    # It exists so a factually-affected condition can be held back without blocking the other conditions or
    # killing an unrelated in-flight stage.
    paused_file = os.path.join(STAGE, "evidence_repair", "PAUSED_ARMS.txt")
    paused = set()
    if os.path.exists(paused_file):
        paused = {ln.strip() for ln in open(paused_file, encoding="utf-8")
                  if ln.strip() and not ln.startswith("#")}
    dom = Domain(a.dataset)
    out_dir = os.path.join(STAGE, "data", a.dataset)
    os.makedirs(out_dir, exist_ok=True)

    cands = [(x, y) for x, y in dom.candidates
             if dom.split_of[dom.entity_of[x]] == a.split
             and dom.split_of[dom.entity_of[y]] == a.split]
    if a.limit:
        cands = cands[:a.limit]
    print(f"{a.dataset}/{a.split}: {len(cands)} candidate pairs, stage={a.stage}", flush=True)

    pool_path = os.path.join(out_dir, f"pool_{a.split}.jsonl")
    ver_path = os.path.join(out_dir, f"verify_{a.split}.jsonl")
    arm_path = os.path.join(out_dir, f"arms_{a.split}.jsonl")
    stats: Dict[str, Any] = {"dataset": a.dataset, "split": a.split, "n_candidates": len(cands)}

    # ------------------------------------------------------------- pool
    if a.stage == "pool":
        have = done_units(pool_path)
        jobs = []
        for (x, y) in cands:
            pr = dom.base_prompt(x, y)
            for m in POOL:
                for rep in range(POOL_REPEATS):
                    u = uid("pool", a.dataset, a.split, x, y, m, rep)
                    if u not in have:
                        jobs.append({"unit_id": u, "a": x, "b": y, "model": m,
                                     "repeat": rep, "prompt": pr})
        print(f"  pool: {len(jobs)} calls ({len(have)} present)", flush=True)
        random.Random(SEED).shuffle(jobs)

        def do(j):
            g = post(j["model"], j["prompt"], MAX_TOKENS["pool"])
            append(pool_path, {
                "unit_id": j["unit_id"], "dataset": a.dataset, "split": a.split,
                "a": j["a"], "b": j["b"], "model": j["model"], "repeat": j["repeat"],
                "role": "pool", "status": "ok", "decision": dom.parse(g["text"]),
                "raw_reply": g["text"], "finish_reason": g["finish_reason"],
                "prompt_tokens": g["prompt_tokens"], "completion_tokens": g["completion_tokens"],
                "prompt_sha": hashlib.sha256(j["prompt"].encode()).hexdigest()[:16],
                "max_tokens": MAX_TOKENS["pool"]})
        stats["pool"] = run(jobs, do, "pool") if jobs else {"n_ok": 0}

    # ------------------------------------------------------------- frozen pool view
    if a.stage in ("verify", "arms"):
        cell = collections.defaultdict(list)
        with open(pool_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") == "ok":
                    cell[(r["a"], r["b"], r["model"])].append(r)

        def modal(rs):
            v = [r["decision"] for r in rs if r["decision"] is not None]
            if not v:
                return None, None
            c = collections.Counter(v)
            top = max(c.values())
            w = [k for k, n in c.items() if n == top]
            if len(w) != 1:
                return None, None                       # frozen tie rule: tie -> abstain
            rep = next(r for r in rs if r["decision"] == w[0])
            return w[0], rep["raw_reply"]

        peer_view: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for (x, y, m), rs in cell.items():
            d, raw = modal(rs)
            peer_view[(x, y, m)] = {"model": m, "decision": d, "raw_reply": raw,
                                    "n_repeats": len(rs)}

    # ------------------------------------------------------------- verifier scores for VD1
    if a.stage == "verify":
        have = done_units(ver_path)
        jobs = []
        for (x, y) in cands:
            base = dom.base_prompt(x, y)
            for m in POOL:
                p = peer_view.get((x, y, m))
                if not p or p["decision"] is None:
                    continue
                u = uid("verify", a.dataset, a.split, x, y, m)
                if u not in have:
                    jobs.append({"unit_id": u, "a": x, "b": y, "peer": m,
                                 "prompt": VERIFY_PROMPT.format(base=base, dec=p["decision"])})
        print(f"  verify: {len(jobs)} calls ({len(have)} present)", flush=True)

        def dov(j):
            g = post("qwen2.5-7b", j["prompt"], MAX_TOKENS["verify"])
            m = next((c for c in g["text"] if c in "12345"), None)
            append(ver_path, {
                "unit_id": j["unit_id"], "dataset": a.dataset, "split": a.split,
                "a": j["a"], "b": j["b"], "peer": j["peer"], "role": "verifier",
                "scorer": "qwen2.5-7b", "score": int(m) if m else None,
                "raw_reply": g["text"], "prompt_tokens": g["prompt_tokens"],
                "completion_tokens": g["completion_tokens"], "status": "ok"})
        stats["verify"] = run(jobs, dov, "verify") if jobs else {"n_ok": 0}

    # ------------------------------------------------------------- the eight conditions
    if a.stage == "arms":
        want = [x for x in a.arms.split(",") if x in ARMS]
        held = [x for x in want if x in paused]
        want = [x for x in want if x not in paused]
        if held:
            print(f"  PAUSED, not submitted: {held}  (see evidence_repair/PAUSED_ARMS.txt)",
                  flush=True)
        vscore: Dict[Tuple[str, str, str], Optional[int]] = {}
        if os.path.exists(ver_path):
            with open(ver_path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    vscore[(r["a"], r["b"], r["peer"])] = r.get("score")
        have = done_units(arm_path)
        jobs: List[Dict[str, Any]] = []
        draw_rng = random.Random(DRAW_SEED)
        for (x, y) in cands:
            for rec in RECIPIENTS:
                others = [m for m in POOL if m != rec]
                peers = [peer_view[(x, y, m)] for m in sorted(others)
                         if (x, y, m) in peer_view]
                if len(peers) < 2:
                    continue
                # the three frozen uniform peer draws, independent of the decoder repeats
                draws = [draw_rng.choice(peers) for _ in range(PEER_DRAWS)]
                # AD1: a peer on the majority side, ties by model name
                dec = [p for p in peers if p["decision"]]
                maj = None
                if dec:
                    c = collections.Counter(p["decision"] for p in dec)
                    top = max(c.values())
                    side = sorted(k for k, n in c.items() if n == top)[0]
                    maj = sorted((p for p in dec if p["decision"] == side),
                                 key=lambda p: p["model"])[0]
                # VD1: highest verifier score, ties by model name
                sc = [(vscore.get((x, y, p["model"])), p) for p in peers
                      if vscore.get((x, y, p["model"])) is not None]
                ver = sorted(sc, key=lambda t: (-t[0], t[1]["model"]))[0][1] if sc else None
                for arm in want:
                    for rep in range(REPEATS):
                        if arm in ("UD1", "UF1"):
                            for di, dp in enumerate(draws):
                                u = uid("arm", a.dataset, a.split, x, y, rec, arm, di, rep)
                                if u in have:
                                    continue
                                jobs.append({"unit_id": u, "a": x, "b": y, "recipient": rec,
                                             "arm": arm, "repeat": rep, "draw": di,
                                             "peers": [dp]})
                            continue
                        sel = {"N": [], "D5": peers, "E5": peers, "F5": peers,
                               "AD1": [maj] if maj else None,
                               "VD1": [ver] if ver else None}[arm]
                        if sel is None:
                            continue
                        u = uid("arm", a.dataset, a.split, x, y, rec, arm, rep)
                        if u in have:
                            continue
                        jobs.append({"unit_id": u, "a": x, "b": y, "recipient": rec,
                                     "arm": arm, "repeat": rep, "draw": None, "peers": sel})
        print(f"  arms {want}: {len(jobs)} calls ({len(have)} present)", flush=True)
        # randomise within blocks so server drift cannot align with an condition
        rng = random.Random(SEED)
        blocks = [jobs[i:i + BLOCK] for i in range(0, len(jobs), BLOCK)]
        for b in blocks:
            rng.shuffle(b)
        jobs = [j for b in blocks for j in b]

        def doa(j):
            pr = build_prompt(dom, j["arm"], j["a"], j["b"], j["peers"])
            mt = MAX_TOKENS[j["arm"]]
            g = post(j["recipient"], pr, mt)
            append(arm_path, {
                "unit_id": j["unit_id"], "dataset": a.dataset, "split": a.split,
                "a": j["a"], "b": j["b"], "recipient": j["recipient"], "arm": j["arm"],
                "repeat": j["repeat"], "draw": j["draw"], "role": "recipient", "status": "ok",
                "decision": dom.parse(g["text"]), "raw_reply": g["text"],
                "finish_reason": g["finish_reason"], "truncated": g["finish_reason"] == "length",
                "prompt_tokens": g["prompt_tokens"], "completion_tokens": g["completion_tokens"],
                "peer_models": [p["model"] for p in j["peers"]],
                "peer_decisions": [p["decision"] for p in j["peers"]],
                "prompt_sha": hashlib.sha256(pr.encode()).hexdigest()[:16], "max_tokens": mt})
        stats["arms"] = run(jobs, doa, f"arms:{'+'.join(want)}") if jobs else {"n_ok": 0}

    sp = os.path.join(STAGE, "logs", f"collect_{a.dataset}_{a.split}_{a.stage}.json")
    json.dump(stats, open(sp, "w", encoding="utf-8"), indent=1, default=str)
    print(f"  wrote {sp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
