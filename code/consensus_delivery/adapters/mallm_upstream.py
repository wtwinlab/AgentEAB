"""Call MALLM's own decision protocols, and nothing else (the released decision functions).

This module does **not** reimplement a threshold or a vote. It imports the upstream classes from the
vendored working copy and calls `make_decision`, which is the function the published protocol *is*.
A from-scratch threshold loop would not reproduce it: upstream decides positionally over a sliding
window and never reads the per-agent stance boolean or the agent id its own record carries. Relabelling
our own loop as MALLM would not be the published protocol.

Two modes, kept apart on purpose:

  * **decision conformance** (this module) drives `make_decision` with no-model fixtures. It needs no
    network and no model. It is what establishes the contract, and what a reviewer can re-run in a
    second. MALLM's heavyweight dependencies are reached only at *import* time, by modules
    `make_decision` never touches, so they are stubbed — and every stub is recorded in the result.
  * **full mechanism run** (not this module) needs MALLM's real environment: `langchain`, `datasets`,
    `bert-score`, `evaluate`. Stubbing is sound for reading a pure function's behaviour and is **not**
    sound for running the mechanism. The two must never be confused in a manifest.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..mechanisms import AcceptanceKind, OutputSource, RecordContract

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

#: The release base, as vendored. Pinned by commit, recorded in every result.
MALLM_ROOT = os.path.join(_REPO, "vendor", "consensus_literature", "mallm")

#: The same base with `patches/mallm_threshold_minimal_fix.patch` applied, in an isolated worktree.
#: `vendor/` is never mutated; the patch and this path are what a reviewer re-applies.
MALLM_MINFIX_ROOT = os.path.join(
    _REPO, "_external", "mechanism_screening_2026-10-08", "repos", "mallm-a57c4a7-minfix")

PATCH_PATH = os.path.join(_REPO, "code", "consensus_delivery", "patches",
                          "mallm_threshold_minimal_fix.patch")

#: Two variants of MALLM's consensus threshold are run on the same base. `release` is the code as
#: shipped at `a57c4a7`. `minimal_fix` applies `patches/mallm_threshold_minimal_fix.patch`, whose
#: predicate equals the one upstream shipped on main from `9abb55d` to `51b9472` and passes upstream's
#: own decision-protocol tests (`test/decision_protocol/test_majority_consensus.py`). At `a57c4a7`
#: every `Agreement` carries a solution, so the release predicate does not reach the threshold for
#: more than one agent; `release` is reported as the as-shipped baseline.
#:
#: Two configuration values stay fixed, because the all-endorsement window is reachable only through
#: them: `skip_decision_making=False`, and no judge regenerate. MALLM is not among the protocols the
#: paper reports.
INTENDED_PROTOCOL_VARIANT = "minimal_fix"

#: The two variants Q2 requires be kept apart, both on one base. Neither may be described as the
#: other, and `minimal_fix` is the intended protocol while `release` is the as-shipped baseline.
VARIANTS = {"release": MALLM_ROOT, "minimal_fix": MALLM_MINFIX_ROOT}

#: The paper's protocol names, as they appear in `decision-protocols`' own experiment configs.
PROTOCOLS = {
    "simple_voting": ("mallm.decision_protocols.simple_voting", "SimpleVoting"),
    "ranked_voting": ("mallm.decision_protocols.ranked_voting", "RankedVoting"),
    "approval_voting": ("mallm.decision_protocols.approval_voting", "ApprovalVoting"),
    "cumulative_voting": ("mallm.decision_protocols.cumulative_voting", "CumulativeVoting"),
    "unanimity_consensus": ("mallm.decision_protocols.consensus", "UnanimityConsensus"),
    "majority_consensus": ("mallm.decision_protocols.consensus", "MajorityConsensus"),
    "supermajority_consensus": ("mallm.decision_protocols.consensus", "SupermajorityConsensus"),
    "hybrid_consensus": ("mallm.decision_protocols.consensus", "HybridMajorityConsensus"),
}


class UpstreamUnavailable(RuntimeError):
    """The vendored upstream could not be loaded. Never silently replaced by our own logic."""


#: Which variant this process has imported. Set once; mixing is refused.
_LOADED_VARIANT: Optional[str] = None

#: Loading mutates `sys.path` and `sys.modules` and sets process-global state, so it is serialised
#: **and** pinned to one thread. A comment saying "do not call this concurrently" is not a guard: it
#: cannot fail a run that does.
_LOAD_LOCK = threading.RLock()
#: The thread that owns the loaded upstream, as a `Thread` object rather than an identifier.
#: `threading.get_ident()` is **reused** once a thread exits, so an identifier comparison let a
#: sequentially-started second thread inherit the first one's claim and load again.
_LOAD_OWNER: Optional[threading.Thread] = None


class UpstreamConcurrencyRefused(RuntimeError):
    """A second thread tried to use the vendored upstream. Give native runs a process each.

    `load_protocol` mutates `sys.path`, installs stubs into `sys.modules` and sets module-level
    variables, all racing `importlib.import_module`. The variant guard happened to survive an
    adversarial two-thread probe, but surviving a probe is not a guarantee, and the failure mode is a
    half-imported package producing results attributed to a published mechanism. This refuses instead.
    """


def _claim_process_for_native() -> None:
    """Refuse a *concurrent* second user of the vendored upstream.

    Sequential reuse from a different thread is fine — the process state is already consistent. What
    is refused is a second **live** thread, which is the case where `sys.path` mutation, stub
    installation into `sys.modules` and `importlib.import_module` race each other.
    """
    global _LOAD_OWNER
    me = threading.current_thread()
    with _LOAD_LOCK:
        owner = _LOAD_OWNER
        if owner is None or owner is me or not owner.is_alive():
            _LOAD_OWNER = me
            return
        raise UpstreamConcurrencyRefused(
            f"the vendored upstream is in use by live thread {owner.name!r}; {me.name!r} must run in "
            f"its own process. Do not route native episodes through `parallel.run_episodes`.")


def commit(root: str = MALLM_ROOT) -> str:
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


class _ImportOnlyStub(types.ModuleType):
    """Stands in for a module reached at import time and never called by `make_decision`.

    Any attribute becomes a trivial class, so class definitions and type annotations resolve. If
    upstream ever *calls* one of these, the call returns None rather than raising, which would be
    dangerous — so `load_protocol` records the stub list and the conformance tests assert that the
    verdicts match upstream's own published assertions. A stub that mattered would break those.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (object,), {"__init__": lambda s, *a, **k: None,
                                      "__call__": lambda s, *a, **k: None})


#: Every module stubbed in this process, cumulatively. A per-call list understates the truth: the
#: first `load_protocol` installs the stubs, every later call then imports cleanly and would report
#: none, and a conformance result that under-reports its own stubs is worse than one that reports too
#: many. Found by this suite reporting zero stubs when fourteen were in use.
_STUBBED_THIS_PROCESS: set = set()

#: Modules MALLM imports transitively that `make_decision` does not use. Enumerated rather than
#: discovered silently, so the list is reviewable and a change in it is visible in a diff.
EXPECTED_STUBS = frozenset({
    "httpx", "openai", "requests", "torch", "sklearn", "sklearn.metrics",
    "sklearn.metrics.pairwise", "langchain_core", "langchain_core.callbacks",
    "langchain_core.callbacks.manager", "langchain_core.language_models",
    "langchain_core.language_models.llms", "langchain_core.outputs",
    "langchain_core.prompt_values",
})


@dataclass
class LoadedProtocol:
    name: str
    cls: Any
    agreement_cls: Any
    source_file: str
    commit: str
    stubs: Tuple[str, ...]
    variant: str = "release"
    #: sha of the applied patch, or None for the unmodified release.
    patch_sha: Optional[str] = None


def load_protocol(name: str, *, variant: str = "release",
                  max_stub_rounds: int = 80) -> LoadedProtocol:
    """Import one upstream protocol class, from the release base or from the patched base.

    Both variants are real working trees on disk, so the object under test is upstream's own code in
    both cases. Only one line differs, and the patch that produces it is checked in. A process may
    hold only one variant at a time, because both present themselves as the package `mallm`; the
    loader refuses to mix them rather than returning whichever was imported first.
    """
    if name not in PROTOCOLS:
        raise KeyError(f"{name!r} is not one of the paper's protocols: {sorted(PROTOCOLS)}")
    if variant not in VARIANTS:
        raise KeyError(f"variant must be one of {sorted(VARIANTS)}; got {variant!r}")
    root = VARIANTS[variant]
    if not os.path.isdir(root):
        raise UpstreamUnavailable(f"MALLM variant {variant!r} not found at {root}")

    _claim_process_for_native()
    global _LOADED_VARIANT
    if _LOADED_VARIANT is not None and _LOADED_VARIANT != variant:
        raise UpstreamUnavailable(
            f"this process already loaded the {_LOADED_VARIANT!r} variant; both variants are the "
            f"package `mallm` and cannot coexist. Compare them in separate processes.")
    _LOADED_VARIANT = variant
    if root not in sys.path:
        sys.path.insert(0, root)

    module_path, cls_name = PROTOCOLS[name]
    stubbed: List[str] = []
    for _ in range(max_stub_rounds):
        try:
            import importlib

            mod = importlib.import_module(module_path)
            types_mod = importlib.import_module("mallm.utils.types")
            break
        except ModuleNotFoundError as e:
            missing = e.name or ""
            if not missing or missing.startswith("mallm"):
                raise UpstreamUnavailable(
                    f"{module_path}: upstream module {missing!r} is missing from the working copy") from None
            stubbed.append(missing)
            parts = missing.split(".")
            for i in range(len(parts)):
                nm = ".".join(parts[: i + 1])
                if nm not in sys.modules:
                    sys.modules[nm] = _ImportOnlyStub(nm)
    else:
        raise UpstreamUnavailable(f"{module_path}: still unimportable after {max_stub_rounds} stubs")

    _STUBBED_THIS_PROCESS.update(stubbed)
    unexpected = sorted(_STUBBED_THIS_PROCESS - EXPECTED_STUBS)
    if unexpected:
        raise UpstreamUnavailable(
            f"{module_path} now needs stubs that were not reviewed: {unexpected}. "
            "Review them and add them to EXPECTED_STUBS, or install the dependency.")
    return LoadedProtocol(name=name, cls=getattr(mod, cls_name),
                          agreement_cls=getattr(types_mod, "Agreement"),
                          source_file=getattr(mod, "__file__", "?"), commit=commit(root),
                          variant=variant, patch_sha=_patch_sha() if variant != "release" else None,
                          stubs=tuple(sorted(_STUBBED_THIS_PROCESS)))


def _patch_sha() -> str:
    import hashlib
    with open(PATCH_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


class _Panelist:
    """`make_decision` reads only `len(self.panelists)`; nothing of a Panelist is touched."""

    id = "p"
    persona = ""


def instantiate(loaded: LoadedProtocol, *, n_panelists: int, n_neutral: int = 0) -> Any:
    return loaded.cls([_Panelist() for _ in range(n_panelists)], n_neutral, None)


def agreement(loaded: LoadedProtocol, *, agree: Optional[bool], solution: Optional[str],
              agent_id: str = "", message_id: int = 0) -> Any:
    """Build an upstream `Agreement`.

    The convention is upstream's own, visible at `mallm/agents/agent.py:75,117,164` and used by its own
    tests: an agent that **agrees** carries the previous solution forward (so `solution=None` in a
    fixture), one that **disagrees** writes its own, and a feedback turn writes `""`.
    """
    return loaded.agreement_cls(agreement=agree, response="", solution=solution,
                                agent_id=agent_id, persona="", message_id=message_id)


def decide(proto: Any, agreements: Sequence[Any], *, turn: int, agent_index: int = 0,
           task: str = "", question: str = "", config: Any = None) -> Dict[str, Any]:
    """One call to upstream's `make_decision`, with its five return values kept as returned."""
    out = proto.make_decision(list(agreements), turn, agent_index, task, question, config)
    solution, is_consensus, returned_agreements, voting_string, voting_results = out
    return {"solution": solution, "is_consensus": bool(is_consensus),
            "n_agreements_returned": len(returned_agreements),
            "voting_string": voting_string, "voting_results": voting_results}


# --------------------------------------------------------------------------- the measured contract
#: M2's contract, as read from the vendored source and confirmed by fixtures. Every boolean below is
#: a statement about the **record and the decision function**, not about the method's quality.
#:
#: `position_bound_to_decision` is False because the verdict is a position in a sliding window: the
#: counted quantity is not attached to a decision identity, and `agent_id` is never read
#: (consensus.py:63-70). `positions_on_final_output` is False because the protocol's final step
#: evaluates the running solution, not a separately published artifact.
MALLM_THRESHOLD_CONTRACT = RecordContract(
    per_agent_positions=True,            # Agreement.agreement is recorded per entry...
    position_bound_to_decision=False,    # ...but make_decision reads neither it nor agent_id
    position_ordering=True,              # entries are appended in order, within a truncated window
    # **Refuted by source, and reconciled with `comparators.INSTRUMENTATION`.**
    # `decision_protocols/consensus.py:59-60` truncates to `agreements[-total_agents:]` and assigns it
    # back to live state (`discussion_paradigms/paradigm.py:105-113`), so the history a withdrawal
    # would live in is discarded before it is written. `protocol.py:100` additionally mutates
    # `prev_answer.solution` in place at vote time. The contract follows upstream here.
    withdrawal_visible=False,
    positions_on_final_output=False,     # no step solicits positions on a published artifact
    acceptance_kind=AcceptanceKind.THRESHOLD_ON_ANSWERS,
    source="vendor/consensus_literature/mallm/mallm/decision_protocols/consensus.py:50-108 "
           "(verdict), mallm/agents/agent.py:75,117,164 (what `solution` means), "
           "mallm/discussion_paradigms/paradigm.py:54 (one entry per agent per turn)",
)

#: What this study adds, declared so it can never be read as part of the mechanism.
STUDY_SIDE_ADDITIONS = (
    "transport: endpoint_url/api_key/model_name are config fields, so a local OpenAI-compatible "
    "server is a config value and no decision logic changes (transport-only adaptation)",
    "instrumentation: we record the upstream verdict and the agreement list verbatim",
    "measurement: this study's own acceptance check is computed separately and labelled; it is never "
    "used to say the mechanism violated a contract it does not claim",
)


# --------------------------------------------------------- the runtime event sequence, not a fixture
def replay_improve(loaded: LoadedProtocol, stances: Sequence[Optional[bool]],
                   solutions: Optional[Sequence[str]] = None) -> List[Any]:
    """Build the agreement list that `Agent.improve()` **actually produces at this base**.

    `mallm/agents/agent.py:73-78` at the release base:

        agree = False if unique_id == 0 else response.agreement
        solution = response.solution if not agree else agreements[-1].solution

    and `mallm/discussion_paradigms/paradigm.py:54` contributes one entry per agent per turn, in roster
    order, with every panelist constructed `drafting_agent=True` (`coordinator.py:136`), so every entry
    comes from `improve()`.

    Conformance fixtures written in upstream's *unit-test* convention — an agreeing entry carrying
    `solution=None` — do not describe this sequence, which is why a suite built on them passed while the
    mechanism could not reach its stop condition. Everything measured about stopping must be driven
    from here.
    """
    sols = list(solutions) if solutions is not None else [f"S{i}" for i in range(len(stances))]
    out: List[Any] = []
    for i, st in enumerate(stances):
        agree = False if i == 0 else bool(st)
        sol = sols[i] if not agree else out[-1].solution
        out.append(agreement(loaded, agree=(None if i == 0 and st is None else st),
                             solution=sol, agent_id=f"a{i % 3}", message_id=i))
    return out
