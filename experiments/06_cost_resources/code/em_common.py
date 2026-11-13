r"""Shared pieces for the entity-matching workload: data access, prompting, parsing, costing.

Everything a policy may see lives here; everything only an evaluator may see is kept out. The two
are separated by construction rather than by discipline: `pair_record()` returns the two records and
nothing else, and the gold label is only ever read through `gold()`, which the collectors never call.

Three decisions that the plan requires to be explicit:

**Invalid outputs are label-independent.** ThriftLLM's shipped loader maps an unparsed prediction to
`(gold + 1) % K`, i.e. it reads the reference label and forces a wrong class, which for a binary
matching task makes every invalid output count as wrong. That is a property of the original artifact
and is reproduced only in the original-artifact replay. Here `parse_decision()` returns `None` for
anything it cannot read, decided before any policy or scorer runs, and `None` is carried as its own
outcome rather than folded into "wrong".

**Cost is recorded, not assumed.** The original cost model fixes output length at one token. For the
common-workload comparison we record actual prompt and completion tokens per call and price them
with a published per-model rate table, keeping the two cost models separate and labelled.

**The class vocabulary is declared, not inferred.** It is `{match, non-match}` from the task
definition, never derived from held-out labels.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROOT_RECORDS = os.path.join(os.environ.get("ARTIFACT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "records")
EM = os.path.join(ROOT_RECORDS, "dataeng_raw", "em")

#: declared from the task definition, never from reference labels
CLASSES = ("match", "non-match")

#: the six locally served models, spanning roughly 10x in parameter count so that a budgeted
#: selection method has a real cost/quality spread to choose from.
POOL: Dict[str, Dict[str, Any]] = {
    "qwen2.5-7b":  {"port": 9001, "params_b": 7.6,  "quant": None},
    "llama-3-8b":  {"port": 9002, "params_b": 8.0,  "quant": None},
    "mistral-7b":  {"port": 9003, "params_b": 7.2,  "quant": None},
    "qwen2.5-14b": {"port": 9004, "params_b": 14.8, "quant": "awq"},
    "qwen2.5-32b": {"port": 9005, "params_b": 32.8, "quant": "awq"},
    "qwen2.5-72b": {"port": 9006, "params_b": 72.7, "quant": "awq"},
}
HOST = os.environ.get("INFERENCE_HOST", "inference-host")

#: USD per million tokens. Local serving has no invoice, so a published-rate table would be fiction.
#: These are *relative* units proportional to served parameter count, which is the quantity that
#: actually drives GPU time here. They are declared as a normalisation, not as market prices, and
#: raw token and call counts are always reported alongside so any other price model can be applied.
def rate(model: str) -> Tuple[float, float]:
    p = POOL[model]["params_b"]
    eff = p * (0.30 if POOL[model]["quant"] == "awq" else 1.0)   # AWQ serves ~4-bit weights
    return (eff * 0.010, eff * 0.030)        # (input, output) relative units per 1M tokens


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    ri, ro = rate(model)
    return ri * prompt_tokens / 1e6 + ro * completion_tokens / 1e6


# --------------------------------------------------------------------------- datasets
def _read_csv(p: str) -> List[Dict[str, str]]:
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Matching:
    """One matching dataset: source tables, entity ids, and labelled pairs per split.

    `records` are what a policy may see. `label` is evaluator-only and is reached through `gold`.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        if name in ("amazon_google", "abt_buy"):
            d = os.path.join(EM, name)
            self.A = {r["id"]: r for r in _read_csv(os.path.join(d, "tableA.csv"))}
            self.B = {r["id"]: r for r in _read_csv(os.path.join(d, "tableB.csv"))}
            self.fields = [k for k in next(iter(self.A.values())) if k != "id"]
            self.splits = {s: _read_csv(os.path.join(d, f"{s}.csv"))
                           for s in ("train", "valid", "test")}
            self.has_clusters = False
        elif name == "wdc_products":
            d = os.path.join(EM, "wdc_products")
            def rd(fn):
                return [json.loads(l) for l in gzip.open(os.path.join(d, fn), "rt")]
            self.fields = ["title", "brand", "description", "price"]
            self._train = rd("wdcproducts80cc20rnd000un_train_small.json.gz")
            self._valid = rd("wdcproducts80cc20rnd000un_valid_small.json.gz")
            self._test = rd("wdcproducts80cc20rnd000un_gs.json.gz")
            self.splits = {"train": self._train, "valid": self._valid, "test": self._test}
            self.has_clusters = True
        else:
            raise AssertionError(name)

    def n(self, split: str) -> int:
        return len(self.splits[split])

    def pair_key(self, row: Dict[str, Any]) -> str:
        if self.name == "wdc_products":
            return f"{row['id_left']}|{row['id_right']}"
        return f"{row['ltable_id']}|{row['rtable_id']}"

    def pair_record(self, row: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """The two records a policy may see. The label is deliberately not returned."""
        if self.name == "wdc_products":
            L = {f: row.get(f"{f}_left") for f in self.fields}
            R = {f: row.get(f"{f}_right") for f in self.fields}
        else:
            a, b = self.A[row["ltable_id"]], self.B[row["rtable_id"]]
            L = {f: a.get(f) for f in self.fields}
            R = {f: b.get(f) for f in self.fields}
        return (_clean(L), _clean(R))

    def gold(self, row: Dict[str, Any]) -> int:
        """Evaluator-only. No collector or policy in this study calls this."""
        return int(row["label"])

    def entity_block(self, row: Dict[str, Any]) -> str:
        """Clustering unit for inference: pairs sharing a record are not independent.

        Amazon-Google and Abt-Buy give no entity clusters, so the connected unit is the left record;
        WDC supplies `cluster_id_left`, which is the true entity and is used instead.
        """
        if self.name == "wdc_products":
            return f"c{row.get('cluster_id_left')}"
        return f"L{row['ltable_id']}"


def _clean(d: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for k, v in d.items():
        s = "" if v is None else str(v)
        s = re.sub(r"\s+", " ", s).strip()
        if s and s.lower() not in ("nan", "none"):
            out[k] = s[:400]
    return out


# --------------------------------------------------------------------------- prompting
PROMPT = (
    "Decide whether the two product records below refer to the SAME real-world product.\n\n"
    "Record A:\n{a}\n\nRecord B:\n{b}\n\n"
    "Answer with exactly one word on the final line, either match or non-match."
)


def render(L: Dict[str, str], R: Dict[str, str]) -> str:
    def fmt(d):
        return "\n".join(f"  {k}: {v}" for k, v in d.items()) or "  (empty)"
    return PROMPT.format(a=fmt(L), b=fmt(R))


#: frozen before the pilot. Reads the last standalone class word in the reply; anything else is
#: `None`, which is an abstention and never silently a non-match.
_WORD = re.compile(r"(?<![a-z])(match|non-match|nonmatch|no match|not a match)(?![a-z])", re.I)

#: Normalised surface forms that denote each class. The negative set is spelled out because the
#: frozen version matched the phrase "no match" in `_WORD` and then failed to map it, so 712 readable
#: replies were recorded as abstentions. See `audit/parser_repair.json` for the full accounting.
_NEG = frozenset({"nonmatch", "notamatch", "nomatch"})


def parse_decision(text: str) -> Optional[str]:
    if not text:
        return None
    hits = _WORD.findall(text)
    if not hits:
        return None
    last = hits[-1].lower().replace(" ", "").replace("-", "")
    if last == "match":
        return "match"
    if last in _NEG:
        return "non-match"
    return None


def parse_decision_v1(text: str) -> Optional[str]:
    """The frozen parser, kept so the revision can be audited rather than asserted.

    It recognises "no match" as a phrase and then drops it, returning an abstention. Collection ran
    under this function; `parse_decision` is what every analysis uses now.
    """
    if not text:
        return None
    hits = _WORD.findall(text)
    if not hits:
        return None
    last = hits[-1].lower().replace(" ", "").replace("-", "")
    if last == "match":
        return "match"
    if last in ("nonmatch", "notamatch"):
        return "non-match"
    return None


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
