from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import Rule, Stance, StanceTable, canonical_decision, decision_key
from .formation import Candidate, EndReason, FormationResult


class OutputSource(str, Enum):

    SELECTED_CANDIDATE = "selected_candidate"
    VOTE_WINNER = "vote_winner"
    AGGREGATOR = "aggregator"
    PARTICIPANT_SUMMARIZER = "participant_summarizer"
    LAST_ROUND = "last_round"
    WHOLE_TRANSCRIPT = "whole_transcript"
    NONE = "none"


class AcceptanceKind(str, Enum):

    PARTICIPANT_ACCEPTANCE = "participant_acceptance"
    ANSWER_AGREEMENT = "answer_agreement"
    THRESHOLD_ON_ANSWERS = "threshold_on_answers"
    AGGREGATION_ONLY = "aggregation_only"
    NONE = "none"


@dataclass(frozen=True)
class RecordContract:

    per_agent_positions: bool
    position_bound_to_decision: bool
    position_ordering: bool
    withdrawal_visible: bool
    positions_on_final_output: bool
    acceptance_kind: AcceptanceKind
    source: str = ""

    def supports_single_solicitation_verification(self) -> bool:
        return (self.per_agent_positions and self.position_bound_to_decision
                and self.positions_on_final_output)

    def supports_withdrawal_verification(self) -> bool:
        return self.supports_single_solicitation_verification() and self.position_ordering

    def supports_delivery_verification(self) -> bool:
        return self.supports_withdrawal_verification()


@dataclass
class MechanismResult:

    mechanism: str
    output: Optional[Dict[str, Any]]
    output_source: OutputSource
    native_stop_reason: str
    contract: RecordContract
    positions: Optional[StanceTable] = None
    candidates: Tuple[Candidate, ...] = ()
    selected_key: Optional[str] = None
    calls: int = 0
    rounds: int = 0
    native_fields: Dict[str, Any] = field(default_factory=dict)
    trace_gaps: Tuple[str, ...] = ()

    def has_accepted_candidate(self) -> bool:
        return self.selected_key is not None and self.positions is not None

    def agreement_evidence(self, rule: Rule, n_agents: int) -> Dict[str, Any]:
        if self.positions is None:
            return {"available": False,
                    "selected_supported": None, "n_accept_at_stop": None,
                    "why": "the mechanism records no per-agent positions"}
        if self.selected_key is None:
            return {"available": True, "selected_supported": None, "n_accept_at_stop": None,
                    "why": "the mechanism has no selected candidate to evaluate support for"}
        n = self.positions.accept_count(self.selected_key)
        satisfied = rule.satisfied(n, n_agents)
        out: Dict[str, Any] = {"available": True, "n_accept_at_stop": n,
                               "acceptance_kind": self.contract.acceptance_kind.value, "why": ""}
        if self.contract.acceptance_kind is AcceptanceKind.PARTICIPANT_ACCEPTANCE:
            out["selected_supported"] = satisfied
        else:
            out["selected_supported"] = None
            out["agreement_on_answer"] = satisfied
            out["why"] = (f"this mechanism's record establishes "
                          f"{self.contract.acceptance_kind.value}, not participant acceptance of the "
                          f"delivered artifact; the rule result is reported as answer agreement")
        return out


STUDY_CONTROLLED_CONTRACT = RecordContract(
    per_agent_positions=True,
    position_bound_to_decision=True,
    position_ordering=True,
    withdrawal_visible=True,
    positions_on_final_output=True,
    acceptance_kind=AcceptanceKind.PARTICIPANT_ACCEPTANCE,
    source="this study, consensus_delivery/formation.py; a study-controlled threshold protocol, "
           "not a published method",
)

_FORMATION_STOP = {
    EndReason.ACCEPTED: "accepted-by-declared-rule",
    EndReason.SWEEPS_EXHAUSTED: "sweep-budget-exhausted",
    EndReason.NO_PROPOSALS: "no-proposals",
    EndReason.INFRA_FAILED: "transport-exhausted",
    EndReason.BUDGET_EXHAUSTED: "output-or-context-budget-exhausted",
}


def from_formation(res: FormationResult, *, mechanism: str = "study-controlled.threshold"
                   ) -> MechanismResult:
    sel: Optional[Dict[str, Any]] = None
    for c in res.candidates:
        if c.cid == res.selected_cid:
            sel = dict(c.fields)
    return MechanismResult(
        mechanism=mechanism,
        output=sel,
        output_source=OutputSource.SELECTED_CANDIDATE if sel else OutputSource.NONE,
        native_stop_reason=_FORMATION_STOP[res.end_reason],
        contract=STUDY_CONTROLLED_CONTRACT,
        positions=res.table,
        candidates=res.candidates,
        selected_key=res.selected_key,
        calls=res.calls,
        rounds=len({b.sweep for b in res.batches}),
        native_fields={"stop_batch_id": res.stop_batch_id, "n_batches": len(res.batches)},
    )


DECLARED_CONTRACTS: Mapping[str, RecordContract] = {}


def register_declared(mechanism: str, contract: RecordContract) -> None:
    if not contract.source or ":" not in contract.source:
        raise ValueError(
            f"{mechanism}: a contract needs a source of the form 'path:line'; got {contract.source!r}")
    if mechanism in DECLARED_CONTRACTS:
        raise ValueError(f"{mechanism} is already registered; a contract is frozen once set")
    dict.__setitem__(DECLARED_CONTRACTS, mechanism, contract)


def contract_table(mechanisms: Sequence[str] = ()) -> str:
    names = list(mechanisms) or sorted(DECLARED_CONTRACTS)
    L = [f"  {'mechanism':34s} {'per-ag':6s} {'bound':6s} {'order':6s} {'withdr':6s} "
         f"{'on-out':6s} {'verifiable':10s} acceptance"]
    for m in names:
        c = DECLARED_CONTRACTS.get(m)
        if c is None:
            L.append(f"  {m:34s} {'?':6s} {'?':6s} {'?':6s} {'?':6s} {'?':6s} "
                     f"{'NOT ESTABLISHED':10s}")
            continue
        L.append(f"  {m:34s} "
                 f"{str(c.per_agent_positions):6s} {str(c.position_bound_to_decision):6s} "
                 f"{str(c.position_ordering):6s} {str(c.withdrawal_visible):6s} "
                 f"{str(c.positions_on_final_output):6s} "
                 f"{str(c.supports_delivery_verification()):10s} {c.acceptance_kind.value}")
    return "\n".join(L)
