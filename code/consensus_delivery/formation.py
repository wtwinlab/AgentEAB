"""Explicit formation state machine.

Two accounting errors of a threshold protocol are structurally impossible here, not merely tested
against:

  * the accept count comes from `StanceTable.accept_count`, which is keyed by (decision, agent) and so
    cannot exceed the roster size no matter how many accept events arrive;
  * the stop predicate is evaluated only on a **complete** batch, and a batch carries an explicit
    `batch_id` and `stage`, so a formation batch and a final-review batch can never merge even when they
    name the same decision in the same round;
  * no per-batch accept counter exists anywhere in this module. One was defined and never called, which
    is an invitation; it is gone.

Proposals are **interleaved**: an agent proposes and the whole roster reviews that proposal before the
next agent proposes. The alternative reading — every agent proposes, then every proposal is reviewed —
costs up to 50% more calls for the same stopping semantics, because a proposal that would have been
accepted immediately is not reviewed until the rest of the roster has also spoken. Both readings fall
inside the design's `3n(1+n)` per-episode bound; this one is the design text's literal reading
("round-robin proposals, each reviewed by the whole roster, stopping at the first accepted batch") and
is the cheaper of the two.

The loop returns a `FormationResult` naming the batch it stopped at and the candidate it selected. Those
two values are what the tests assert, not merely that a trace can be parsed afterwards.
"""
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
    """Which protocol phase a batch belongs to. Never inferred from round numbers."""

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
    """One complete roster review of one candidate at one stage."""

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
    """A transport failure that exhausted its retry budget. Never a stance."""


class BudgetExhausted(RuntimeError):
    """Context or output budget exhausted. A recorded outcome, never a silent truncation."""


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
    """Round-robin proposals, each reviewed by the whole roster, stopping at the first accepted batch.

    `propose` returns decision fields or None (the agent produced nothing usable). `review` returns one
    terminal stance; it must raise `InfrastructureFailure` when its retry budget is exhausted, and must
    never return `NOT_ASKED`.
    """
    roster = tuple(roster)
    table = StanceTable(roster=roster)
    candidates: List[Candidate] = []
    by_key: Dict[str, Candidate] = {}
    batches: List[Batch] = []
    seq = itertools.count()
    calls = 0
    last_proposed: Optional[str] = None

    def intern(fields: Mapping[str, Any], author: str) -> Candidate:
        """One semantic object per canonical decision within an episode.

        Reuse is by canonical payload, not by hash alone: a hash match is verified against the payload
        so a collision cannot merge two different decisions.
        """
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
                    # One agent's transport failing is that agent contributing nothing. Aborting the
                    # episode here would discard proposals already obtained in this same sweep, and
                    # would turn a single flaky call into an unresolved episode.
                    propose_failures += 1
                    continue
                if fields is None:
                    continue
                cand = intern(fields, author)
                last_proposed = cand.cid
                proposals_this_sweep += 1
                if cand.key in seen_this_sweep:
                    # Two agents proposed the same decision in this sweep. One semantic object gets
                    # one complete roster batch per sweep; reviewing it twice would just cost n more
                    # calls for the same current stances.
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

                # The predicate is evaluated here and nowhere else, and only on a complete batch.
                if not complete:
                    continue
                if cfg.rule.satisfied(table.accept_count(cand.key), len(roster)):
                    return FormationResult(EndReason.ACCEPTED, batch.batch_id, cand.cid, cand.key,
                                           tuple(batches), tuple(candidates), table, calls,
                                           last_proposed)

            if proposals_this_sweep == 0:
                # Nothing was proposed this sweep, so no stance can change in any later one: only a
                # newly proposed candidate is reviewed. Continuing would burn n propose calls a sweep
                # to reach the same end state.
                if candidates:
                    break
                if propose_failures:
                    # No candidate exists and the reason is transport, not refusal. Recording this as
                    # "no proposals" would count the episode as a confirmed zero.
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
