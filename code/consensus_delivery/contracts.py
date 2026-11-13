from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class Stance(str, Enum):

    ACCEPT = "accept"
    REJECT = "reject"
    ABSTAIN = "abstain"
    UNKNOWN = "unknown"
    NOT_ASKED = "not_asked"
    INFRA_FAILED = "infra_failed"


COUNTS_AS_ACCEPT = frozenset({Stance.ACCEPT})

TERMINAL_RESPONSE = frozenset({Stance.ACCEPT, Stance.REJECT, Stance.ABSTAIN, Stance.UNKNOWN})


@dataclass(frozen=True)
class Rule:

    name: str

    def satisfied(self, n_accept: int, n_agents: int) -> bool:
        if n_agents <= 0:
            return False
        if n_accept < 0:
            raise ValueError("n_accept must be non-negative")
        if n_accept > n_agents:
            raise ValueError(f"n_accept={n_accept} exceeds roster size {n_agents}")
        if self.name == "unanimity":
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


def canonical_decision(fields: Mapping[str, Any], schema: Sequence[str],
                       equivalences: Optional[EquivalenceMap] = None
                       ) -> Tuple[Tuple[str, str], ...]:
    eqs = equivalences or {}
    return tuple((f, _norm_repr(fields.get(f), eqs.get(f))) for f in schema)


def decision_key(fields: Mapping[str, Any], schema: Sequence[str],
                 equivalences: Optional[EquivalenceMap] = None) -> str:
    canon = canonical_decision(fields, schema, equivalences)
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:32]


EQUIVALENCE_KINDS = ("exact", "unordered_collection", "numeric", "casefold")


@dataclass(frozen=True)
class FieldEquivalence:

    field: str
    kind: str = "exact"
    precision: int = 6
    justification: str = ""

    def __post_init__(self) -> None:
        if self.kind not in EQUIVALENCE_KINDS:
            raise ValueError(f"kind must be one of {EQUIVALENCE_KINDS}; got {self.kind!r}")
        if self.kind != "exact" and not self.justification:
            raise ValueError(
                f"{self.field!r}: a non-exact equivalence needs a justification naming the task "
                "property that makes the two forms one decision")


EquivalenceMap = Mapping[str, FieldEquivalence]


def _norm_repr(v: Any, eq: Optional[FieldEquivalence] = None) -> str:
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
    kind = eq.kind if eq is not None else "exact"
    if isinstance(v, bool):
        return ("bool", v)
    if v is None:
        return ("none",)
    if kind == "numeric":
        n = _as_number(v, eq.precision if eq else 6)
        if n is not None:
            return n
    if isinstance(v, str):
        return ("str", v.casefold() if kind == "casefold" else v)
    if isinstance(v, (int, float)):
        return ("num_lit", repr(v))
    if isinstance(v, (set, frozenset)):
        return ("set",) + tuple(sorted(repr(_norm(x, eq)) for x in v))
    if isinstance(v, (list, tuple)):
        items = tuple(repr(_norm(x, eq)) for x in v)
        if kind == "unordered_collection":
            return ("bag",) + tuple(sorted(items))
        return ("seq",) + items
    if isinstance(v, dict):
        return ("map",) + tuple(sorted((str(k), repr(_norm(x, eq))) for k, x in v.items()))
    return ("other", repr(v))


def out_of_schema_digest(fields: Mapping[str, Any], schema: Sequence[str]) -> str:
    extra = {k: v for k, v in fields.items() if k not in set(schema)}
    if not extra:
        return ""
    payload = tuple(sorted((str(k), _norm_repr(v)) for k, v in extra.items()))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


class Terminal(str, Enum):

    DELIVERED_SUPPORTED = "delivered-and-supported"
    DELIVERED_UNSUPPORTED = "delivered-without-established-support"
    NO_DELIVERY = "no-delivery-valid-or-no-candidate"
    INVALID_ARTIFACT = "invalid-artifact-not-deliverable"
    INFRA_UNRESOLVED = "infrastructure-unresolved"
    RESOURCE_EXHAUSTED = "resource-exhausted"


class AcceptanceObservation(str, Enum):

    ACCEPTED = "participant-accepted"
    REJECTED = "participant-rejected"
    NOT_ASKED = "not-asked"
    NOT_MEASURED = "not-measured"
    NOT_ESTABLISHED = "not-established"


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


NOT_OPPOSITION = frozenset(
    {Terminal.INFRA_UNRESOLVED, Terminal.RESOURCE_EXHAUSTED, Terminal.INVALID_ARTIFACT}
)


@dataclass
class ArmOutcome:

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
    out_of_schema_digest: Optional[str] = None
    decision_changed: Optional[bool] = None
    out_of_schema_changed: Optional[bool] = None

    @property
    def field_equal_prose_changed(self) -> Optional[bool]:
        if self.decision_changed is None or self.out_of_schema_changed is None:
            return None
        return (not self.decision_changed) and self.out_of_schema_changed
    n_accept_at_gate: Optional[int] = None
    n_agents: int = 0
    attempts: Tuple[Dict[str, Any], ...] = ()
    calls: int = 0
    acceptance: AcceptanceObservation = AcceptanceObservation.NOT_MEASURED

    def __post_init__(self) -> None:
        if self.terminal in (Terminal.DELIVERED_SUPPORTED, Terminal.DELIVERED_UNSUPPORTED):
            if self.delivered_key is None:
                raise ValueError(f"{self.terminal} requires a delivered artifact")
        else:
            if self.delivered_key is not None:
                raise ValueError(f"{self.terminal} must not carry a delivered artifact")
        if self.acceptance in (AcceptanceObservation.ACCEPTED, AcceptanceObservation.REJECTED):
            if self.n_accept_at_gate is None:
                raise ValueError(
                    f"acceptance={self.acceptance.value} claims a solicited verdict, but no gate "
                    "count is recorded; use NOT_MEASURED or NOT_ESTABLISHED")


@dataclass
class StanceTable:

    roster: Tuple[str, ...]
    _current: Dict[Tuple[str, str], Stance] = field(default_factory=dict)
    _history: list = field(default_factory=list)

    def __post_init__(self) -> None:
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
        return sum(1 for a in self.roster if self.current(decision_key_, a) is Stance.ACCEPT)

    def counts(self, decision_key_: str) -> Dict[Stance, int]:
        out = {s: 0 for s in Stance}
        for a in self.roster:
            out[self.current(decision_key_, a)] += 1
        return out

    def history(self) -> Tuple[Tuple[int, str, str, Stance], ...]:
        return tuple(self._history)
