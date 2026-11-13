from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .contracts import (
    TERMINAL_RESPONSE,
    Rule,
    Stance,
    StanceTable,
    canonical_decision,
    decision_key,
)


class Stage(str, Enum):

    FORMATION = "formation"
    FINAL_REVIEW = "final_review"
    REPAIR_REVIEW = "repair_review"
    DIAGNOSTIC = "diagnostic"
    POOLING_REVIEW = "pooling_review"


class EndReason(str, Enum):
    ACCEPTED = "accepted"
    SWEEPS_EXHAUSTED = "sweeps-exhausted"
    NO_PROPOSALS = "no-proposals"
    INFRA_FAILED = "infrastructure-failed"
    BUDGET_EXHAUSTED = "budget-exhausted"


@dataclass
class Candidate:
    cid: str
    key: str
    fields: Dict[str, Any]
    author: str
    canonical: Tuple[Tuple[str, str], ...]


@dataclass
class Batch:

    batch_id: str
    stage: Stage
    sweep: int
    cid: str
    key: str
    responses: Dict[str, Stance]
    complete: bool


@dataclass
class FormationResult:
    end_reason: EndReason
    stop_batch_id: Optional[str]
    selected_cid: Optional[str]
    selected_key: Optional[str]
    batches: Tuple[Batch, ...]
    candidates: Tuple[Candidate, ...]
    table: StanceTable
    calls: int
    last_proposed_cid: Optional[str]

    @property
    def reached_agreement(self) -> bool:
        return self.end_reason is EndReason.ACCEPTED


class InfrastructureFailure(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class FormationConfig:
    rule: Rule
    schema: Tuple[str, ...]
    max_sweeps: int = 3

    def __post_init__(self) -> None:
        if self.max_sweeps < 1:
            raise ValueError("max_sweeps must be >= 1")


def run_formation(
    *,
    roster: Sequence[str],
    cfg: FormationConfig,
    propose: Callable[[str, int, Sequence[Candidate]], Optional[Dict[str, Any]]],
    review: Callable[[str, Candidate, int, str], Stance],
    episode_id: str = "ep",
) -> FormationResult:
    roster = tuple(roster)
    table = StanceTable(roster=roster)
    candidates: List[Candidate] = []
    by_key: Dict[str, Candidate] = {}
    batches: List[Batch] = []
    seq = itertools.count()
    calls = 0
    last_proposed: Optional[str] = None

    def intern(fields: Mapping[str, Any], author: str) -> Candidate:
        canon = canonical_decision(fields, cfg.schema)
        k = decision_key(fields, cfg.schema)
        prior = by_key.get(k)
        if prior is not None:
            if prior.canonical != canon:
                raise RuntimeError(f"decision_key collision on {k}: payloads differ")
            return prior
        cand = Candidate(cid=f"{episode_id}-c{len(candidates) + 1:03d}", key=k,
                         fields=dict(fields), author=author, canonical=canon)
        candidates.append(cand)
        by_key[k] = cand
        return cand

    propose_failures = 0
    try:
        for sweep in range(1, cfg.max_sweeps + 1):
            seen_this_sweep: set = set()
            proposals_this_sweep = 0
            for author in roster:
                calls += 1
                try:
                    fields = propose(author, sweep, tuple(candidates))
                except InfrastructureFailure:
                    propose_failures += 1
                    continue
                if fields is None:
                    continue
                cand = intern(fields, author)
                last_proposed = cand.cid
                proposals_this_sweep += 1
                if cand.key in seen_this_sweep:
                    continue
                seen_this_sweep.add(cand.key)

                batch_id = f"{episode_id}/f{sweep}/{cand.cid}/b{len(batches) + 1:03d}"
                responses: Dict[str, Stance] = {}
                for agent in roster:
                    calls += 1
                    st = review(agent, cand, sweep, batch_id)
                    if st is Stance.NOT_ASKED:
                        raise RuntimeError("review() returned NOT_ASKED; absence is not a response")
                    if st not in TERMINAL_RESPONSE:
                        raise RuntimeError(f"review() returned non-terminal {st!r}")
                    responses[agent] = st
                    table.record(cand.key, agent, st, seq=next(seq))

                complete = set(responses) == set(roster)
                batch = Batch(batch_id=batch_id, stage=Stage.FORMATION, sweep=sweep,
                              cid=cand.cid, key=cand.key, responses=responses, complete=complete)
                batches.append(batch)

                if not complete:
                    continue
                if cfg.rule.satisfied(table.accept_count(cand.key), len(roster)):
                    return FormationResult(EndReason.ACCEPTED, batch.batch_id, cand.cid, cand.key,
                                           tuple(batches), tuple(candidates), table, calls,
                                           last_proposed)

            if proposals_this_sweep == 0:
                if candidates:
                    break
                if propose_failures:
                    return FormationResult(EndReason.INFRA_FAILED, None, None, None, tuple(batches),
                                           tuple(candidates), table, calls, last_proposed)
                return FormationResult(EndReason.NO_PROPOSALS, None, None, None, tuple(batches),
                                       tuple(candidates), table, calls, last_proposed)
    except InfrastructureFailure:
        return FormationResult(EndReason.INFRA_FAILED, None, None, None, tuple(batches),
                               tuple(candidates), table, calls, last_proposed)
    except BudgetExhausted:
        return FormationResult(EndReason.BUDGET_EXHAUSTED, None, None, None, tuple(batches),
                               tuple(candidates), table, calls, last_proposed)

    return FormationResult(EndReason.SWEEPS_EXHAUSTED, None, None, None, tuple(batches),
                           tuple(candidates), table, calls, last_proposed)
