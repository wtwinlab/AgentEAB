"""Written contracts: agreement rules, stance semantics, decision identity, terminal states.

This module is the *production* statement of the design's §4.3 and §6.2. The independent oracle under
`oracle/` must not import anything from here; it restates the same rules from the design text so that a
single misreading cannot pass both.

Design references are given per definition so a reviewer can check the text, not the author's memory.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- stances
class Stance(str, Enum):
    """A terminal response state for one agent on one decision in one batch.

    `UNKNOWN` is a *returned* response that could not be parsed into a stance. It is not opposition and
    not absence: an unknown stance does not establish support and is not an explicit objection.
    `NOT_ASKED` is the absence of a solicitation, which is a different fact again.
    `INFRA_FAILED` is a transport failure that exhausted its retry budget; the contract forbids
    evaluating agreement on a partial roster, so this state must never enter a count.
    """

    ACCEPT = "accept"
    REJECT = "reject"
    ABSTAIN = "abstain"
    UNKNOWN = "unknown"
    NOT_ASKED = "not_asked"
    INFRA_FAILED = "infra_failed"


#: Only this stance counts toward an agreement rule. Everything else — including silence — does not.
COUNTS_AS_ACCEPT = frozenset({Stance.ACCEPT})

#: States that mean "this agent produced a terminal model response in this batch".
#: A batch is complete only when every roster member is in one of these.
TERMINAL_RESPONSE = frozenset({Stance.ACCEPT, Stance.REJECT, Stance.ABSTAIN, Stance.UNKNOWN})


# ------------------------------------------------------------------- agreement rules
@dataclass(frozen=True)
class Rule:
    """A declared agreement rule over a fixed roster.

    The denominator is always the full roster, never the set of agents who happened to answer. Formulas
    are declared exactly, as declared: unanimity is equality, and the two threshold rules are
    strict inequalities on the fraction of the roster.
    """

    name: str

    def satisfied(self, n_accept: int, n_agents: int) -> bool:
        if n_agents <= 0:
            return False
        if n_accept < 0:
            raise ValueError("n_accept must be non-negative")
        if n_accept > n_agents:
            # Distinct accepting roster members cannot exceed the roster. A count that does sums accept
            # *events* rather than members, so it is rejected rather than satisfying a threshold rule: majority would read 4/2 as satisfied.
            raise ValueError(f"n_accept={n_accept} exceeds roster size {n_agents}")
        if self.name == "unanimity":
            # Equality, not >=. An inflated count must not satisfy it.
            return n_accept == n_agents
        if self.name == "majority":
            return n_accept / n_agents > 0.5
        if self.name == "supermajority":
            return n_accept / n_agents > 2.0 / 3.0
        raise ValueError(f"unknown rule {self.name!r}")


UNANIMITY = Rule("unanimity")
MAJORITY = Rule("majority")
SUPERMAJORITY = Rule("supermajority")
RULES: Mapping[str, Rule] = {r.name: r for r in (UNANIMITY, MAJORITY, SUPERMAJORITY)}


# ------------------------------------------------------------- decision identity
def canonical_decision(fields: Mapping[str, Any], schema: Sequence[str],
                       equivalences: Optional[EquivalenceMap] = None
                       ) -> Tuple[Tuple[str, str], ...]:
    """The authoritative decision, normalised, restricted to the frozen schema's fields.

    Two candidates with the same canonical decision *and* the same context may share support. The canonical form is what a content key is computed from; a candidate instance ID
    does not create a new semantic object, and equally does not merge two different decisions.
    """
    eqs = equivalences or {}
    return tuple((f, _norm_repr(fields.get(f), eqs.get(f))) for f in schema)


def decision_key(fields: Mapping[str, Any], schema: Sequence[str],
                 equivalences: Optional[EquivalenceMap] = None) -> str:
    """A stable content key. Collisions are impossible to rule out in principle, so callers that use
    this as an identity must also compare the canonical payload."""
    canon = canonical_decision(fields, schema, equivalences)
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:32]


#: Per-field semantic equivalences, declared by a **frozen task schema** and nothing else.
#:
#: The default identity is deliberately strict: JSON types are preserved, string case
#: and content are preserved, **list order is preserved**, and only a dict's key *serialisation* order
#: is ignored. Numeric strings are not numbers, `True` is not `1`, nothing is casefolded and no list
#: is sorted. Applying those normalisations globally would merge decisions a task may distinguish, and
#: a reordered ranking would inherit the original's recorded support.
#:
#: Where a task genuinely treats two forms as one decision, the task's adapter declares it per field
#: and an independent oracle re-implements the same declaration. **The official answer scorer's
#: normalisation does not transfer**: a scorer may accept `"8"` for `8` while the participants were
#: asked about a specific written decision.
EQUIVALENCE_KINDS = ("exact", "unordered_collection", "numeric", "casefold")


@dataclass(frozen=True)
class FieldEquivalence:
    """One field's declared equivalence, frozen with the task schema.

    `kind`:
      * `exact` — the default. Types, case, content and order all significant.
      * `unordered_collection` — a list field whose order carries no meaning (an allocation set).
      * `numeric` — the field is a number and `8`, `8.0`, `"8"` are one value at `precision` decimals.
      * `casefold` — a label field whose case carries no meaning.
    """

    field: str
    kind: str = "exact"
    precision: int = 6
    #: Why the task treats these as one decision. Required: an undeclared equivalence is not one.
    justification: str = ""

    def __post_init__(self) -> None:
        if self.kind not in EQUIVALENCE_KINDS:
            raise ValueError(f"kind must be one of {EQUIVALENCE_KINDS}; got {self.kind!r}")
        if self.kind != "exact" and not self.justification:
            raise ValueError(
                f"{self.field!r}: a non-exact equivalence needs a justification naming the task "
                "property that makes the two forms one decision")


#: field -> FieldEquivalence, for the schema currently frozen. Empty means the strict default
#: everywhere, which is what every task gets until its adapter declares otherwise.
EquivalenceMap = Mapping[str, FieldEquivalence]


def _norm_repr(v: Any, eq: Optional[FieldEquivalence] = None) -> str:
    """The declared normal form of one field value, as a string."""
    return repr(_norm(v, eq))


def _as_number(x: Any, precision: int) -> Optional[Tuple[str, str]]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        f = float(x)
    elif isinstance(x, str):
        try:
            f = float(x.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return ("num", f"{round(f, precision) + 0.0:.{precision}f}")


def _norm(v: Any, eq: Optional[FieldEquivalence] = None) -> Any:
    """The declared normal form of one field value.

    Every branch is tagged, so a normalised string can never collide with a normalised number or a
    container. The raw payload is recorded alongside the key, so this policy is auditable and the
    identity can be recomputed under a different declaration.
    """
    kind = eq.kind if eq is not None else "exact"
    if isinstance(v, bool):
        return ("bool", v)                       # never the number one
    if v is None:
        return ("none",)
    if kind == "numeric":
        n = _as_number(v, eq.precision if eq else 6)
        if n is not None:
            return n
    if isinstance(v, str):
        return ("str", v.casefold() if kind == "casefold" else v)
    if isinstance(v, (int, float)):
        return ("num_lit", repr(v))              # type- and literal-preserving by default
    if isinstance(v, (set, frozenset)):
        # A set has no order to preserve; sorting its rendered members is the only stable form.
        return ("set",) + tuple(sorted(repr(_norm(x, eq)) for x in v))
    if isinstance(v, (list, tuple)):
        items = tuple(repr(_norm(x, eq)) for x in v)
        if kind == "unordered_collection":
            return ("bag",) + tuple(sorted(items))
        return ("seq",) + items                  # order preserved by default
    if isinstance(v, dict):
        # Only the key *serialisation* order is ignored.
        return ("map",) + tuple(sorted((str(k), repr(_norm(x, eq))) for k, x in v.items()))
    return ("other", repr(v))


def out_of_schema_digest(fields: Mapping[str, Any], schema: Sequence[str]) -> str:
    """A digest of everything the authoritative decision deliberately excludes.

    `canonical_decision` is restricted to the frozen schema, which is correct — the decision is the
    allocation, not the prose around it. But the prompts ask agents to put their reasoning in a
    non-schema field, so two artifacts with an identical allocation and opposite stated reasons share
    one decision key. Without this digest the "fields unchanged, prose changed" stratum the judge
    sampling plan requires cannot be drawn from the record at all, and that cannot be fixed after
    collection. Recorded, never compared for support.
    """
    extra = {k: v for k, v in fields.items() if k not in set(schema)}
    if not extra:
        return ""
    payload = tuple(sorted((str(k), _norm_repr(v)) for k, v in extra.items()))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


# ------------------------------------------------------------------- terminal states
class Terminal(str, Enum):
    """Exactly one of these per assigned arm. Mutually exclusive and jointly exhaustive.

    Reasons are recorded separately (`Reason`); the design forbids turning every event into its own
    terminal state.
    """

    DELIVERED_SUPPORTED = "delivered-and-supported"
    DELIVERED_UNSUPPORTED = "delivered-without-established-support"
    NO_DELIVERY = "no-delivery-valid-or-no-candidate"
    INVALID_ARTIFACT = "invalid-artifact-not-deliverable"
    INFRA_UNRESOLVED = "infrastructure-unresolved"
    RESOURCE_EXHAUSTED = "resource-exhausted"


class AcceptanceObservation(str, Enum):
    """What was observed about participant acceptance of the **delivered** artifact.

    **This is orthogonal to `Terminal`.** A delivery can happen with acceptance accepted, rejected,
    never solicited, or not measurable at all, and a non-delivery says nothing about acceptance either.
    Folding "we never observed acceptance" into a delivery terminal would turn an absence of
    measurement into a negative result, which is the single error this dimension exists to prevent:
    `DELIVERED_UNSUPPORTED` reads as "delivered and support was not established", and for a published
    method that has no acceptance step at all, that is a verdict the study is not entitled to.

    `NOT_MEASURED` and `NOT_ESTABLISHED` are different and must not be merged:

      * `NOT_MEASURED` — the protocol has no step that asks. Nothing was attempted. This is the
        correct value for every native method whose own contract contains no acceptance solicitation.
        It is a **property of the method**, not a shortcoming, and not a rejection.
      * `NOT_ESTABLISHED` — something was asked, and the answers do not settle it: an unparseable
        reply, a partial roster, an infrastructure failure mid-batch.
    """

    ACCEPTED = "participant-accepted"
    REJECTED = "participant-rejected"
    NOT_ASKED = "not-asked"                 # the study chose not to ask on this arm
    NOT_MEASURED = "not-measured"           # the protocol has no step that asks
    NOT_ESTABLISHED = "not-established"     # asked, but the answers do not settle it


class Reason(str, Enum):
    NO_ACCEPTED_PREFIX = "no-accepted-prefix"
    GATE_REJECT = "gate-reject"
    GATE_ABSTAIN_UNKNOWN = "gate-abstain-or-unknown"
    SCHEMA_INVALID = "parse-or-schema-invalid"
    BUDGET_EXHAUSTED = "context-or-output-budget-exhaustion"
    TRANSPORT_EXHAUSTED = "transport-retries-exhausted"
    GATE_PASSED = "gate-passed"
    UNGATED_PUBLISH = "ungated-publish"
    REPAIRED = "repaired-after-invalid"


#: Terminal states that must never be read as an agent having opposed the decision.
NOT_OPPOSITION = frozenset(
    {Terminal.INFRA_UNRESOLVED, Terminal.RESOURCE_EXHAUSTED, Terminal.INVALID_ARTIFACT}
)


@dataclass
class ArmOutcome:
    """One assigned arm's single final state, with its reason chain and the attempts preserved."""

    study_id: str
    run_id: str
    task_id: str
    model_key: str
    repeat: int
    arm: str
    terminal: Terminal
    reasons: Tuple[Reason, ...] = ()
    delivered_key: Optional[str] = None
    delivered_fields: Optional[Dict[str, Any]] = None
    #: Digest of the delivered artifact's non-schema content. Recorded for stratified judge sampling,
    #: never gated on.
    out_of_schema_digest: Optional[str] = None
    #: Did the **authoritative decision** (the schema fields) change relative to the source candidate?
    decision_changed: Optional[bool] = None
    #: Did the **non-schema content** (the prose) change relative to the source candidate?
    #:
    #: Computed from the prose, not from `canonical_decision` (the schema fields), so the
    #: "fields unchanged, prose changed" stratum of the judge sampling plan can be drawn.
    out_of_schema_changed: Optional[bool] = None

    @property
    def field_equal_prose_changed(self) -> Optional[bool]:
        """The judge-sampling stratum: the decision is identical and the prose is not.

        Derived here so no consumer has to combine the two flags.
        `None` when either input is unknown (nothing delivered, or no source candidate).
        """
        if self.decision_changed is None or self.out_of_schema_changed is None:
            return None
        return (not self.decision_changed) and self.out_of_schema_changed
    n_accept_at_gate: Optional[int] = None
    n_agents: int = 0
    attempts: Tuple[Dict[str, Any], ...] = ()
    calls: int = 0
    #: Orthogonal to `terminal`. Never derived from it.
    acceptance: AcceptanceObservation = AcceptanceObservation.NOT_MEASURED

    def __post_init__(self) -> None:
        if self.terminal in (Terminal.DELIVERED_SUPPORTED, Terminal.DELIVERED_UNSUPPORTED):
            if self.delivered_key is None:
                raise ValueError(f"{self.terminal} requires a delivered artifact")
        else:
            if self.delivered_key is not None:
                raise ValueError(f"{self.terminal} must not carry a delivered artifact")
        # An acceptance verdict requires that something was actually asked about an artifact.
        if self.acceptance in (AcceptanceObservation.ACCEPTED, AcceptanceObservation.REJECTED):
            if self.n_accept_at_gate is None:
                raise ValueError(
                    f"acceptance={self.acceptance.value} claims a solicited verdict, but no gate "
                    "count is recorded; use NOT_MEASURED or NOT_ESTABLISHED")


# ----------------------------------------------------------------------- stance table
@dataclass
class StanceTable:
    """Current stance per agent per decision key, in event order.

    A later stance by the same agent on the *same* decision supersedes the earlier one. Accepting a
    different decision does **not** withdraw acceptance of an earlier one, so this is
    keyed by (decision_key, agent) and never by agent alone.
    """

    roster: Tuple[str, ...]
    _current: Dict[Tuple[str, str], Stance] = field(default_factory=dict)
    _history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        # A duplicated roster member would be solicited twice and counted twice, so two distinct
        # agents could satisfy a three-agent unanimity rule. The independent oracle iterates the
        # roster the same way, so the check is made here.
        if len(set(self.roster)) != len(self.roster):
            raise ValueError(f"roster has duplicate members: {self.roster}")

    def record(self, decision_key_: str, agent: str, stance: Stance, *, seq: int) -> None:
        if agent not in self.roster:
            raise ValueError(f"agent {agent!r} is not on the roster {self.roster}")
        if stance is Stance.NOT_ASKED:
            raise ValueError("NOT_ASKED is an absence; it is never recorded as an event")
        self._history.append((seq, decision_key_, agent, stance))
        self._current[(decision_key_, agent)] = stance

    def current(self, decision_key_: str, agent: str) -> Stance:
        return self._current.get((decision_key_, agent), Stance.NOT_ASKED)

    def accept_count(self, decision_key_: str) -> int:
        """Distinct roster members whose *current* stance on this decision is accept.

        Counting members rather than accept events, it is bounded above
        by the roster size by construction, because the key includes the agent.
        """
        return sum(1 for a in self.roster if self.current(decision_key_, a) is Stance.ACCEPT)

    def counts(self, decision_key_: str) -> Dict[Stance, int]:
        out = {s: 0 for s in Stance}
        for a in self.roster:
            out[self.current(decision_key_, a)] += 1
        return out

    def history(self) -> Tuple[Tuple[int, str, str, Stance], ...]:
        return tuple(self._history)
