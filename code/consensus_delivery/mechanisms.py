"""The mechanism layer: what *any* consensus mechanism returns, published or study-controlled.

The delivery interface does not assume that an accepted prefix exists. The study-controlled
formation has one: `FormationResult` carries `selected_cid` and a `StanceTable`,
because the study-controlled formation in `formation.py` is a threshold protocol and always has both.
Published mechanisms do not. A plurality vote has a winner and no acceptance. A fixed-round debate ends
because the rounds ran out. A confidence-weighted round table ends on a weighted argmax. Handing any of
those to the branch layer as if it had produced "the accepted candidate" would be inventing an
acceptance contract on the method's behalf, which §4.4-7 forbids.

So `MechanismResult` is the interface, and it records four things separately:

  * **the mechanism's own output**, and *how* the mechanism produced it — a vote winner, an
    aggregator's text, a selected candidate, or nothing;
  * **the mechanism's own stop reason, verbatim in its own vocabulary.** Not mapped onto ours. A
    fixed-round debate's `rounds_exhausted` is not our `sweeps-exhausted`, and recording it as such
    would make the two look like the same event;
  * **what the mechanism's record can support**, as a measured property of the record rather than a
    claim about the method (`RecordContract` below). This is the quantity the study is actually about
    at this layer, and it is readable from any faithful run without adding a single step to the method;
  * **the trace that is available**, including what is *not* there.

`from_formation` adapts the study-controlled formation into this interface without changing it, so the
whole existing delivery layer keeps working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import Rule, Stance, StanceTable, canonical_decision, decision_key
from .formation import Candidate, EndReason, FormationResult


class OutputSource(str, Enum):
    """How the mechanism arrived at its final output. Each is a different kind of object.

    The distinction matters because the study's question is whether the participants accepted the
    thing that was delivered. Three of these make the delivered object a participant's **own** text —
    `SELECTED_CANDIDATE`, `VOTE_WINNER` and `PARTICIPANT_SUMMARIZER` — and only `AGGREGATOR` is text
    written by someone who was not in the room. Whichever it is, it is a *property of the published
    method*, not a issue this study introduced, and the member comments carry the source line that
    established it.
    """

    SELECTED_CANDIDATE = "selected_candidate"
    #: A participant's **own** text, chosen by a rule after participants were shown the candidate set.
    #: Verified in all six MALLM voting/consensus protocols: `simple_voting.py:95`
    #: `final_answer=final_answers[most_voted]` over texts authored by the panelists themselves
    #: (`protocol.py:75-103`, `:150-164`). It is therefore **strictly more** than
    #: `SELECTED_CANDIDATE` on this study's own definitions, not less — the earlier docstring called it
    #: "text no participant was ever asked about", which is false and overstated the gap.
    VOTE_WINNER = "vote_winner"
    #: Written by a **non-participant** from the discussion. MoA is the verified instance.
    AGGREGATOR = "aggregator"
    #: Written by **one of the participants** acting as summarizer. MALLM's `judge` protocol is the
    #: verified instance: `decision_protocols/judge.py:59` passes `panelists=[self.panelists[0]]`, and
    #: `process_votes` returns that panelist's summary. Kept distinct from `AGGREGATOR` because the
    #: preservation obligations differ — an open-weight substitution for MoA must preserve an external
    #: writer's full input, while this one must preserve "written by panelist[0], seeing only the n
    #: final answers". Merging them would silently widen the summarizer's input rights.
    PARTICIPANT_SUMMARIZER = "participant_summarizer"
    LAST_ROUND = "last_round"                   # whatever the final round happened to emit
    WHOLE_TRANSCRIPT = "whole_transcript"
    #: The mechanism performs no selection and emits the full exchange; any single answer is the
    #: *evaluator's* choice. Verified for Du MAD (`vendor/mad/gsm/gen_gsm.py:63-65` dumps
    #: `(agent_contexts, answer)`) and for BCCS (`run_mmlu.py:247-249`).
    NONE = "none"                               # the mechanism produced no output


class AcceptanceKind(str, Enum):
    """What, if anything, the mechanism's own final step establishes."""

    PARTICIPANT_ACCEPTANCE = "participant_acceptance"   # agents asked about the final artifact
    ANSWER_AGREEMENT = "answer_agreement"               # agents' answers coincide
    THRESHOLD_ON_ANSWERS = "threshold_on_answers"       # a declared fraction share one answer
    AGGREGATION_ONLY = "aggregation_only"               # a rule combines answers, nobody consents
    NONE = "none"


@dataclass(frozen=True)
class RecordContract:
    """What a mechanism's own record can and cannot establish, measured from the record.

    This is deliberately a statement about the *record*, not about the method's quality. "ReConcile
    does not record per-agent stances on the final artifact" is a verifiable fact about its logs; "
    ReConcile fails to obtain acceptance" would be imposing a contract it never claimed. The paper's
    instrumentation claim rests on the first kind of statement only.
    """

    #: Is there a per-agent record at all, or only an aggregate?
    per_agent_positions: bool
    #: Is a position attached to a *specific* decision/answer, or merely to a round?
    position_bound_to_decision: bool
    #: Can a later position be told from an earlier one by the same agent on the same decision?
    position_ordering: bool
    #: Is a withdrawal — accept then reject — distinguishable from never having accepted?
    withdrawal_visible: bool
    #: Does the mechanism's own protocol include a step soliciting positions on the *final output*?
    positions_on_final_output: bool
    #: What the mechanism's final step establishes, in its own terms.
    acceptance_kind: AcceptanceKind
    #: Free text, with a source citation. Required: an unsourced contract is not a contract.
    source: str = ""

    def supports_single_solicitation_verification(self) -> bool:
        """Can acceptance of the delivered artifact be verified from **one** final solicitation?

        Ordering is deliberately **not** required: if every roster member is asked once about the
        final artifact and each answer is recorded against it, the verdict follows without any time
        dimension. The previous single predicate conjoined ordering and asserted "every clause is
        necessary", which was false — though no record was misclassified by it, because the only two
        contracts that existed were unaffected either way.
        """
        return (self.per_agent_positions and self.position_bound_to_decision
                and self.positions_on_final_output)

    def supports_withdrawal_verification(self) -> bool:
        """Can an acceptance that was later retracted be told from one never given?

        This is where ordering is necessary, and it is a strictly stronger requirement.
        """
        return self.supports_single_solicitation_verification() and self.position_ordering

    def supports_delivery_verification(self) -> bool:
        """Retained for callers that want the strong form. Equivalent to withdrawal verification."""
        return self.supports_withdrawal_verification()


@dataclass
class MechanismResult:
    """One mechanism's run, in the only vocabulary shared across published and study-controlled runs.

    `output` may be None and `positions` may be empty. Nothing downstream may assume otherwise.
    """

    mechanism: str                       # frozen mechanism id, e.g. "mallm.unanimity@<commit>"
    output: Optional[Dict[str, Any]]
    output_source: OutputSource
    native_stop_reason: str              # the mechanism's own vocabulary, verbatim
    contract: RecordContract
    #: Per-agent positions, when the mechanism records them. Keyed exactly as our own table is, so a
    #: mechanism that *does* record them can be analysed with the same code; absent otherwise.
    positions: Optional[StanceTable] = None
    #: Candidates the participants actually had in front of them, when the mechanism exposes them.
    candidates: Tuple[Candidate, ...] = ()
    #: The mechanism's own notion of which candidate it selected, if it has one.
    selected_key: Optional[str] = None
    calls: int = 0
    rounds: int = 0
    #: Anything the mechanism reports that has no counterpart here. Kept verbatim, never mapped.
    native_fields: Dict[str, Any] = field(default_factory=dict)
    #: Named gaps in the trace. An empty tuple states that the trace has no gaps.
    trace_gaps: Tuple[str, ...] = ()

    def has_accepted_candidate(self) -> bool:
        """The precondition the delivery branches used to assume. Now asked, never assumed."""
        return self.selected_key is not None and self.positions is not None

    def agreement_evidence(self, rule: Rule, n_agents: int) -> Dict[str, Any]:
        """What this record says about agreement, with the unavailable parts reported as unavailable.

        A mechanism whose record cannot answer a question gets `None` for it, never `False`. Reading
        "not recorded" as "did not happen" is the single most tempting error at this layer and would
        manufacture a gap wherever instrumentation is thin.
        """
        if self.positions is None:
            return {"available": False,
                    "selected_supported": None, "n_accept_at_stop": None,
                    "why": "the mechanism records no per-agent positions"}
        if self.selected_key is None:
            return {"available": True, "selected_supported": None, "n_accept_at_stop": None,
                    "why": "the mechanism has no selected candidate to evaluate support for"}
        n = self.positions.accept_count(self.selected_key)
        satisfied = rule.satisfied(n, n_agents)
        # The verdict is labelled by what the mechanism's own contract establishes. A mechanism whose
        # record carries answer agreement must not emit a field that reads as participant support:
        # `selected_supported` is only populated when the contract says participants were asked about
        # the artifact. Otherwise the same number is reported under its true name.
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


# --------------------------------------------------------------------- the study-controlled mechanism
#: The contract of our own formation, stated in the same form demanded of every published method so
#: the two are comparable and so ours is not quietly exempted from its own standard.
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

#: Our end reasons, kept verbatim rather than mapped to anyone else's vocabulary.
_FORMATION_STOP = {
    EndReason.ACCEPTED: "accepted-by-declared-rule",
    EndReason.SWEEPS_EXHAUSTED: "sweep-budget-exhausted",
    EndReason.NO_PROPOSALS: "no-proposals",
    EndReason.INFRA_FAILED: "transport-exhausted",
    EndReason.BUDGET_EXHAUSTED: "output-or-context-budget-exhausted",
}


def from_formation(res: FormationResult, *, mechanism: str = "study-controlled.threshold"
                   ) -> MechanismResult:
    """Adapt the study-controlled formation to the shared interface, changing nothing about it."""
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


# --------------------------------------------------------------------- declared contracts, pending source check
#: Contracts asserted from the papers and pending source conformance. **Nothing here may enter a
#: manifest until the vendored code has been read and the `source` upgraded to a `path:line`.**
#: `acceptance_kind` is the field that decides whether a mechanism can answer this study's question
#: natively, so it is the one to be most sceptical about.
DECLARED_CONTRACTS: Mapping[str, RecordContract] = {}


def register_declared(mechanism: str, contract: RecordContract) -> None:
    """Record a contract once it has been established against the vendored source, with a citation."""
    if not contract.source or ":" not in contract.source:
        raise ValueError(
            f"{mechanism}: a contract needs a source of the form 'path:line'; got {contract.source!r}")
    if mechanism in DECLARED_CONTRACTS:
        raise ValueError(f"{mechanism} is already registered; a contract is frozen once set")
    dict.__setitem__(DECLARED_CONTRACTS, mechanism, contract)  # type: ignore[arg-type]


def contract_table(mechanisms: Sequence[str] = ()) -> str:
    """The manifest row per mechanism required before the main collection."""
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
