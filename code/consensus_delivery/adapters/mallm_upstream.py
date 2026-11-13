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

MALLM_ROOT = os.path.join(_REPO, "vendor", "consensus_literature", "mallm")

MALLM_MINFIX_ROOT = os.path.join(
    _REPO, "_external", "mechanism_screening_2026-10-08", "repos", "mallm-a57c4a7-minfix")

PATCH_PATH = os.path.join(_REPO, "code", "consensus_delivery", "patches",
                          "mallm_threshold_minimal_fix.patch")

INTENDED_PROTOCOL_VARIANT = "minimal_fix"

VARIANTS = {"release": MALLM_ROOT, "minimal_fix": MALLM_MINFIX_ROOT}

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
    pass


_LOADED_VARIANT: Optional[str] = None

_LOAD_LOCK = threading.RLock()
_LOAD_OWNER: Optional[threading.Thread] = None


class UpstreamConcurrencyRefused(RuntimeError):
    pass


def _claim_process_for_native() -> None:
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
    except Exception:
        return "unknown"


class _ImportOnlyStub(types.ModuleType):

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (object,), {"__init__": lambda s, *a, **k: None,
                                      "__call__": lambda s, *a, **k: None})


_STUBBED_THIS_PROCESS: set = set()

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
    patch_sha: Optional[str] = None


def load_protocol(name: str, *, variant: str = "release",
                  max_stub_rounds: int = 80) -> LoadedProtocol:
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

    id = "p"
    persona = ""


def instantiate(loaded: LoadedProtocol, *, n_panelists: int, n_neutral: int = 0) -> Any:
    return loaded.cls([_Panelist() for _ in range(n_panelists)], n_neutral, None)


def agreement(loaded: LoadedProtocol, *, agree: Optional[bool], solution: Optional[str],
              agent_id: str = "", message_id: int = 0) -> Any:
    return loaded.agreement_cls(agreement=agree, response="", solution=solution,
                                agent_id=agent_id, persona="", message_id=message_id)


def decide(proto: Any, agreements: Sequence[Any], *, turn: int, agent_index: int = 0,
           task: str = "", question: str = "", config: Any = None) -> Dict[str, Any]:
    out = proto.make_decision(list(agreements), turn, agent_index, task, question, config)
    solution, is_consensus, returned_agreements, voting_string, voting_results = out
    return {"solution": solution, "is_consensus": bool(is_consensus),
            "n_agreements_returned": len(returned_agreements),
            "voting_string": voting_string, "voting_results": voting_results}


MALLM_THRESHOLD_CONTRACT = RecordContract(
    per_agent_positions=True,
    position_bound_to_decision=False,
    position_ordering=True,
    withdrawal_visible=False,
    positions_on_final_output=False,
    acceptance_kind=AcceptanceKind.THRESHOLD_ON_ANSWERS,
    source="vendor/consensus_literature/mallm/mallm/decision_protocols/consensus.py:50-108 "
           "(verdict), mallm/agents/agent.py:75,117,164 (what `solution` means), "
           "mallm/discussion_paradigms/paradigm.py:54 (one entry per agent per turn)",
)

STUDY_SIDE_ADDITIONS = (
    "transport: endpoint_url/api_key/model_name are config fields, so a local OpenAI-compatible "
    "server is a config value and no decision logic changes (transport-only adaptation)",
    "instrumentation: we record the upstream verdict and the agreement list verbatim",
    "measurement: this study's own acceptance check is computed separately and labelled; it is never "
    "used to say the mechanism violated a contract it does not claim",
)


def replay_improve(loaded: LoadedProtocol, stances: Sequence[Optional[bool]],
                   solutions: Optional[Sequence[str]] = None) -> List[Any]:
    sols = list(solutions) if solutions is not None else [f"S{i}" for i in range(len(stances))]
    out: List[Any] = []
    for i, st in enumerate(stances):
        agree = False if i == 0 else bool(st)
        sol = sols[i] if not agree else out[-1].solution
        out.append(agreement(loaded, agree=(None if i == 0 and st is None else st),
                             solution=sol, agent_id=f"a{i % 3}", message_id=i))
    return out
