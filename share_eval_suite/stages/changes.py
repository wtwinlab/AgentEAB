"""Answer changes: where gains are kept or lost.

A change between two stages, rounds or configurations is read per paired unit as a transition between
correct and wrong. With q the share correct before, c the share of wrong units that become correct and
r the share of correct units that become wrong,

    recovered = (1 - q) c,   lost = q r,   net = recovered - lost   (pp of all units).

Revision changes compare the initial candidates with the last-round candidates of the same run (a
correct candidate lost by revision, or one acquired); rule changes compare the votes of the same run
under two readings (corrections and regressions); peer changes compare an agent's answer without peers
with its answer given peer responses.
"""
from __future__ import annotations

import collections
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from share_eval_suite import paths
from share_eval_suite.records import readers
from share_eval_suite.stages import availability

PEER_OBJECT = "text_peer-source"
BASELINE_READING = availability.BASELINE_READING
RECIPIENTS = ("qwen3-32b", "gpt-oss-120b", "gemma-4-26b")


def transitions(before: Mapping[Any, float], after: Mapping[Any, float]) -> Dict[str, float]:
    """Transition split of a paired binary change over the units present in both.

    Returns `n`, `q_pct` (correct before), `c_pct` (wrong units that become correct), `r_pct` (correct
    units that become wrong), and `recovered_pp`, `lost_pp`, `net_pp` as shares of all units. A rate
    with no units to condition on is NaN.
    """
    keys = sorted(set(before) & set(after))
    b = np.array([before[k] for k in keys])
    t = np.array([after[k] for k in keys])
    q = b.mean()
    c = t[b == 0].mean() if (b == 0).any() else float("nan")
    r = (1 - t[b == 1]).mean() if (b == 1).any() else float("nan")
    out = dict(n=len(keys), q_pct=100 * q, c_pct=100 * c, r_pct=100 * r,
               recovered_pp=100 * (1 - q) * c, lost_pp=100 * q * r, net_pp=100 * ((1 - q) * c - q * r))
    return {k: (int(v) if k == "n" else float(v)) for k, v in out.items()}


# ------------------------------------------------------------------ revision: initial vs last round
def revision_changes(runs: Optional[Iterable[Mapping[str, Any]]] = None,
                     configs: Optional[Sequence[Tuple[str, str]]] = None) -> Dict[str, Any]:
    """Revision's effect on whether a correct candidate is present, BIRD debates with peers.

    For each run the initial pool (C0, lenient extractor) is paired with the last-round pool (CT).
    `lost_pp` is the share of runs whose initial pool holds a correct candidate and whose last-round
    pool holds none; `recovered_pp` the reverse. Returns `per_configuration` ({(protocol, backbone):
    transitions}) and `mean` (equal-weight mean of `recovered_pp`, `lost_pp`, `net_pp` over
    configurations). Every question has the same number of repeats, so run shares equal the shares of
    questions with repeats averaged.
    """
    runs = readers.read_candidate_flow_runs() if runs is None else list(runs)
    configs = list(configs or availability.configurations())
    per = {}
    for c in configs:
        sel = [r for r in runs if (r["protocol"], r["backbone"]) == c]
        key = [(r["question"], r["repeat"]) for r in sel]
        per[c] = transitions(dict(zip(key, (int(r["C0"]) for r in sel))),
                             dict(zip(key, (int(r["CT"]) for r in sel))))
    mean = {k: float(np.mean([per[c][k] for c in configs])) for k in ("recovered_pp", "lost_pp", "net_pp")}
    return {"per_configuration": per, "mean": mean}


# ------------------------------------------------------------------ decision: two readings of the same run
def rule_changes(changed: str = "lenient_execution_first", baseline: str = BASELINE_READING,
                 runs: Optional[Iterable[Mapping[str, Any]]] = None,
                 configs: Optional[Sequence[Tuple[str, str]]] = None) -> Dict[str, int]:
    """Runs whose vote a change of reading corrects or regresses.

    Readings are `{protocol,lenient}_{exact,execution}_{first,last,random}` (extractor, vote, tie
    rule). `corrections` counts runs wrong under `baseline` and correct under `changed`; `regressions`
    the reverse. The default configurations are the four Du-MAD and MAD-M2 ones, on the questions
    present in all six.
    """
    runs = readers.read_candidate_flow_runs() if runs is None else list(runs)
    allc = availability.configurations()
    configs = list(configs or allc[:4])
    # questions common to the six configurations
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        by[(r["protocol"], r["backbone"])][r["question"]].append(r)
    units = sorted(set.intersection(*[set(by[c]) for c in allc]))
    rs = [x for c in configs for u in units for x in by[c][u]]
    return {"runs": len(rs),
            "corrections": sum(int(x[baseline]) == 0 and int(x[changed]) == 1 for x in rs),
            "regressions": sum(int(x[baseline]) == 1 and int(x[changed]) == 0 for x in rs)}


# ------------------------------------------------------------------ peers: without vs with peer responses
def _peer_outcomes(path=None):
    # outcomes per (recipient, peer source, condition), keyed (task, seat); the Qwen3-32B peer set
    # takes the no-peer outcomes collected with the Qwen2.5-7B peer set
    out, bench = collections.defaultdict(dict), {}
    for r in readers.read_csv(path or paths.inputs(PEER_OBJECT) / "peer_source_outcomes.csv"):
        out[(r["recipient"], r["source"], r["condition"])][(r["task"], int(r["seat"]))] = int(r["correct"])
        bench[r["task"]] = r["benchmark"]
    for m in RECIPIENTS:
        out[(m, "qwen3-32b", "no_peers")] = dict(out[(m, "qwen25-7b", "no_peers")])
    return out, bench


def peer_changes(recipient: str, source: str, shared: str = "answer_only", path=None) -> Dict[str, float]:
    """Transitions of a recipient's answers from no peers to peers sharing `shared` (`full` responses
    or `answer_only`), with peers of `source` (`llama3-8b`, `qwen25-7b`, `qwen3-32b`), on the QA
    positions that have all three conditions. Qwen3-32B peers use the no-peer response collected with
    the Qwen2.5-7B peer set."""
    out, _ = _peer_outcomes(path)
    conds = {c: out.get((recipient, source, c), {}) for c in ("no_peers", "full", "answer_only")}
    keys = set(conds["no_peers"]) & set(conds["full"]) & set(conds["answer_only"])
    return transitions({k: conds["no_peers"][k] for k in keys}, {k: conds[shared][k] for k in keys})


def peer_source_change(recipient: str, shared: str = "answer_only", new: str = "qwen3-32b",
                       old: str = "qwen25-7b", path=None) -> Dict[str, float]:
    """Change of the recovered and lost shares when `new` peers replace `old` peers.

    `drec_exact` and `dlost_exact` are the unrounded differences (pp); `drec` and `dlost` subtract the
    two-decimal components, as the supplementary table prints them.
    """
    n = peer_changes(recipient, new, shared, path)
    o = peer_changes(recipient, old, shared, path)
    return {"drec_exact": n["recovered_pp"] - o["recovered_pp"],
            "dlost_exact": n["lost_pp"] - o["lost_pp"],
            "drec": float(round(round(n["recovered_pp"], 2) - round(o["recovered_pp"], 2), 2)),
            "dlost": float(round(round(n["lost_pp"], 2) - round(o["lost_pp"], 2), 2)),
            "regression_rate_falls": bool(n["r_pct"] < o["r_pct"])}


def weaker_recipient_changes(shared: str = "answer_only", path=None) -> Dict[str, float]:
    """Transitions of the Qwen2.5-7B recipient from no peers to Qwen3-32B peers sharing `shared`."""
    rows = readers.read_csv(path or paths.inputs(PEER_OBJECT) / "weaker_recipient_units.csv")
    keys = [(r["task"], int(r["seat"])) for r in rows]
    return transitions({k: int(r["y_no_peers"]) for k, r in zip(keys, rows)},
                       {k: int(r[f"y_{shared}"]) for k, r in zip(keys, rows)})
