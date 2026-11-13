r"""Geographic Settlements: dataset, gold partition, blocking, prompt, cost, parser.

The second matching workload. Chosen because it is the only multi-source entity-clustering benchmark
on the Leipzig page whose records are **real** rather than generated: four real sources (DBpedia,
GeoNames, Freebase, NYTimes), real record-to-entity identifiers, a domain unrelated to WDC products,
and a published gold standard used in OAEI. MusicBrainz and North Carolina Voters were rejected
because their duplicates come from the DAPO and GeCo corruption generators, and a synthetic
perturbation is not an independent workload.

Two issues in the shipped gold standard, both resolved here before any prediction exists
---------------------------------------------------------------------------------------------
1. Record 4216 appears in cluster 459 (with three records that are plainly the same settlement) and
   also as its own singleton cluster 4216. The singleton is redundant, so it is dropped and the
   partition becomes well defined: 819 clusters over 3,054 records, 4,391 gold pairs, which is the
   count the dataset page states.
2. `readMe.txt` says "5 sources"; the records carry four distinct `ontology` values. The data, not
   the readme, is used, and the discrepancy is recorded rather than silently reconciled.

What gold is used for, and what it is never used for
----------------------------------------------------
Gold entity ids are evaluator-only. `gold_cluster()` and `gold_pair()` exist; nothing that builds a
prompt, chooses a block, or runs a policy calls them. Blocking is deliberately **source-agnostic**
even though the gold shows every source to be internally duplicate-free, because that property is
derived from gold and using it would put the answer into the candidate set.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(BASE, "data", "gs", "geographicalSettelments")

#: declared from the task definition, never from reference labels
CLASSES = ("match", "non-match")

#: the serving pool for this round. Same roster as the data-engineering round so the two are
#: comparable; distinct ports and a distinct container prefix so nothing collides with another user.
HOST = os.environ.get("INFERENCE_HOST", "inference-host")
CONTAINER_PREFIX = "dsr-"
POOL: Dict[str, Dict[str, Any]] = {
    "qwen2.5-7b":  {"port": 9201, "params_b": 7.6,  "quant": "none", "gpu": 1,
                    "hf": "Qwen/Qwen2.5-7B-Instruct"},
    "llama-3-8b":  {"port": 9202, "params_b": 8.0,  "quant": "none", "gpu": 4,
                    "hf": "meta-llama/Meta-Llama-3-8B-Instruct"},
    "mistral-7b":  {"port": 9203, "params_b": 7.2,  "quant": "none", "gpu": 5,
                    "hf": "mistralai/Mistral-7B-Instruct-v0.3"},
    "qwen2.5-14b": {"port": 9204, "params_b": 14.7, "quant": "awq",  "gpu": 2,
                    "hf": "Qwen/Qwen2.5-14B-Instruct-AWQ"},
    "qwen2.5-32b": {"port": 9205, "params_b": 32.8, "quant": "awq",  "gpu": 2,
                    "hf": "Qwen/Qwen2.5-32B-Instruct-AWQ"},
    "qwen2.5-72b": {"port": 9206, "params_b": 72.7, "quant": "awq",  "gpu": 3,
                    "hf": "Qwen/Qwen2.5-72B-Instruct-AWQ"},
}
RECIPIENTS = ("qwen2.5-7b", "qwen2.5-32b")
CONDITIONS = ("no_peers", "candidate_only", "full")

SPLIT_SALT = "gs-entity-split-2026-10-13"
BLOCK_NGRAM = 3


def rate(model: str) -> Tuple[float, float]:
    """Relative input/output rate per token, as in the data-engineering round."""
    p = POOL[model]["params_b"]
    eff = p * (0.30 if POOL[model]["quant"] == "awq" else 1.0)
    return (eff * 0.010, eff * 0.030)


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    ri, ro = rate(model)
    return (prompt_tokens * ri + completion_tokens * ro) / 1_000_000.0


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------- dataset
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ngrams(s: str, n: int = BLOCK_NGRAM) -> Set[str]:
    t = f"  {s} "
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))} if s else set()


class Settlements:
    """The node universe, the gold partition, and a source-agnostic candidate blocker."""

    def __init__(self) -> None:
        self.rec: Dict[int, Dict[str, Any]] = {}
        with open(os.path.join(DATA, "settlements.json"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.rec[int(r["id"])] = r["data"]
        raw = []
        with open(os.path.join(DATA, "combinedSettlements(PerfectMatch).json"),
                  encoding="utf-8") as f:
            for line in f:
                raw.append(json.loads(line))
        # resolve the redundant singleton: a cluster of one whose only member also sits in a
        # larger cluster carries no information and makes the partition ill defined
        member_count: Dict[int, int] = {}
        for c in raw:
            for x in c["data"]["clusteredVertices"]:
                member_count[int(x)] = member_count.get(int(x), 0) + 1
        self.dropped_clusters: List[int] = []
        self.clusters: Dict[int, List[int]] = {}
        for c in raw:
            mem = sorted(int(x) for x in c["data"]["clusteredVertices"])
            if len(mem) == 1 and member_count[mem[0]] > 1:
                self.dropped_clusters.append(int(c["id"]))
                continue
            self.clusters[int(c["id"])] = mem
        self.cluster_of: Dict[int, int] = {}
        for cid, mem in self.clusters.items():
            for x in mem:
                assert x not in self.cluster_of, f"record {x} still in two clusters"
            for x in mem:
                self.cluster_of[x] = cid
        assert set(self.cluster_of) == set(self.rec), "gold does not cover the record set"
        self.n_gold_pairs = sum(len(m) * (len(m) - 1) // 2 for m in self.clusters.values())

    # ---------------- splitting, on whole entities
    def split_of(self, cluster_id: int) -> str:
        h = hashlib.sha256(f"{SPLIT_SALT}|{cluster_id}".encode()).hexdigest()
        return "dev" if int(h, 16) % 2 == 0 else "test"

    def records_in(self, split: str) -> List[int]:
        return sorted(r for r in self.rec if self.split_of(self.cluster_of[r]) == split)

    def clusters_in(self, split: str) -> Dict[int, List[int]]:
        return {c: m for c, m in self.clusters.items() if self.split_of(c) == split}

    # ---------------- what a policy may see
    def record_text(self, rid: int) -> Dict[str, Any]:
        """Policy-visible fields only. The gold entity id is not among them."""
        d = self.rec[rid]
        out: Dict[str, Any] = {"name": d.get("label")}
        if d.get("lat") is not None and d.get("lon") is not None:
            out["coordinates"] = f"{float(d['lat']):.4f}, {float(d['lon']):.4f}"
        if d.get("type"):
            out["type"] = d["type"]
        if d.get("typeDetail"):
            out["type detail"] = d["typeDetail"]
        if d.get("ele") is not None:
            out["elevation_m"] = d["ele"]
        out["catalogue"] = _SOURCE_NAME.get(d.get("ontology", ""), d.get("ontology", ""))
        return out

    def pair_key(self, a: int, b: int) -> str:
        x, y = (a, b) if a < b else (b, a)
        return f"{x}|{y}"

    # ---------------- evaluator only
    def gold_cluster(self, rid: int) -> int:
        return self.cluster_of[rid]

    def gold_pair(self, a: int, b: int) -> int:
        return 1 if self.cluster_of[a] == self.cluster_of[b] else 0

    # ---------------- blocking: source-agnostic, label-only, top-K by 3-gram Jaccard
    def block(self, rids: Sequence[int], k: int) -> List[Tuple[int, int]]:
        """Each record proposes its K most similar other records; the candidate set is the union.

        Similarity is Jaccard over character 3-grams of the normalised name. Ties are resolved by
        record id so the result is deterministic. Nothing here reads a coordinate (331 records have
        none) or a source (that would exploit a gold-derived property).
        """
        gr = {r: ngrams(_norm(self.rec[r].get("label", ""))) for r in rids}
        inv: Dict[str, List[int]] = {}
        for r in rids:
            for g in gr[r]:
                inv.setdefault(g, []).append(r)
        out: Set[Tuple[int, int]] = set()
        for r in rids:
            shared: Dict[int, int] = {}
            for g in gr[r]:
                for o in inv[g]:
                    if o != r:
                        shared[o] = shared.get(o, 0) + 1
            if not shared:
                continue
            scored = []
            for o, c in shared.items():
                union = len(gr[r]) + len(gr[o]) - c
                scored.append((-(c / union if union else 0.0), o))
            scored.sort()
            for _s, o in scored[:k]:
                out.add((r, o) if r < o else (o, r))
        return sorted(out)


_SOURCE_NAME = {
    "http://dbpedia.org/": "DBpedia",
    "http://sws.geonames.org/": "GeoNames",
    "http://rdf.freebase.com/": "Freebase",
    "http://data.nytimes.com/": "New York Times",
}

# --------------------------------------------------------------------------------- prompt
PROMPT = (
    "Decide whether the two geographical records below refer to the SAME real-world settlement.\n\n"
    "Record A:\n{a}\n\nRecord B:\n{b}\n\n"
    "Two records match when they denote the same populated place, even if the names are spelled "
    "differently or one includes a region qualifier. Different places that merely share a name do "
    "not match.\n"
    "Answer with exactly one word on the final line, either match or non-match."
)


def render(A: Dict[str, Any], B: Dict[str, Any]) -> str:
    def fmt(d: Dict[str, Any]) -> str:
        return "\n".join(f"  {k}: {v}" for k, v in d.items()) or "  (empty)"
    return PROMPT.format(a=fmt(A), b=fmt(B))


# --------------------------------------------------------------------------------- parser
#: Parser v3. v2 was fixed before the pilot and the pilot broke it, which is what a pilot is for.
#:
#: Two issues the dev pilot exposed in 720 calls:
#:   1. `llama-3-8b` ended a reply "In conclusion, the two records do not match." The v2 alternation
#:      had `match` but no negated verb phrase, so the last class word was `match` and the reply was
#:      recorded as a positive when the model had said the opposite.
#:   2. `same` and `different` were treated as class words. They appear constantly in neutral
#:      reasoning ("the exact same settlement", "slightly different coordinates") and carry no
#:      decision. They are removed.
#:
#: The prompt asks for exactly one word on the final line, so v3 reads the **final non-empty line**
#: first and only falls back to a whole-text scan when that line is not a bare class word. Negated
#: verb phrases are listed before bare `match` so the alternation prefers the longer form.
PARSER_VERSION = "v3-final-line-then-negation-aware"

_NEG_PHRASE = (
    r"do(?:es)?\s+not\s+match", r"did\s+not\s+match", r"do(?:es)?n[’']t\s+match",
    r"not\s+a\s+match", r"no\s+match", r"not\s+match",
    r"is\s+not\s+a\s+match", r"are\s+not\s+a\s+match",
    r"non[\s-]*match", r"nonmatch",
)
#: negated phrases first, bare `match` last; the engine prefers the earlier alternative at a position
_WORD = re.compile(r"(?<![a-z])(" + "|".join(_NEG_PHRASE) + r"|match)(?![a-z])", re.I)
_BARE = re.compile(r"^[\s\"'*`#.:\-]*(non[\s-]*match|nonmatch|match)[\s\"'*`.!:\-]*$", re.I)


def _norm_class(s: str) -> Optional[str]:
    t = re.sub(r"[^a-z]", "", (s or "").lower())
    if t == "match":
        return "match"
    if t in ("nonmatch", "notamatch", "nomatch", "notmatch", "donotmatch", "doesnotmatch",
             "didnotmatch", "dontmatch", "doesntmatch", "isnotamatch", "arenotamatch"):
        return "non-match"
    return None


def parse_decision(text: str) -> Optional[str]:
    """The final line if it is a bare class word, else the last class phrase. Never a silent negative.

    An unreadable reply returns `None`, which is an abstention carried as its own outcome. The rule
    is fixed before main collection and reads no reference label.
    """
    if not text:
        return None
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if lines:
        m = _BARE.match(lines[-1])
        if m:
            return _norm_class(m.group(1))
    hits = _WORD.findall(text)
    if not hits:
        return None
    return _norm_class(hits[-1])


def parse_decision_v2(text: str) -> Optional[str]:
    """The pre-pilot parser, kept so the revision can be audited rather than asserted."""
    w = re.compile(r"(?<![a-z])(match|non-match|nonmatch|no match|not a match|same|different)"
                   r"(?![a-z])", re.I)
    if not text:
        return None
    hits = w.findall(text)
    if not hits:
        return None
    last = hits[-1].lower().replace(" ", "").replace("-", "")
    if last in ("match", "same"):
        return "match"
    if last in ("nonmatch", "notamatch", "nomatch", "different"):
        return "non-match"
    return None


def parse_ambiguity(text: str) -> Dict[str, Any]:
    """Kept per record so the parse can be audited rather than trusted."""
    hits = _WORD.findall(text or "")
    norm = [_norm_class(h) for h in hits]
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return {"n_class_words": len(hits), "surface_forms": hits[-3:],
            "final_line_is_bare": bool(lines and _BARE.match(lines[-1])),
            "conflicting": bool("match" in norm and "non-match" in norm),
            "v2_would_say": parse_decision_v2(text)}
