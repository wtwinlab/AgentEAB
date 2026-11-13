"""Protocol adapters: the debate and consensus methods and the multi-stage LLM pipelines.

Each entry names a workflow, its kind, the stages it runs (generation, revision, extraction, final
decision), the files that implement it and the module `load()` returns. Nothing is imported until
`load()` is called: the debate and consensus methods import as modules of the `consensus_delivery`
package under `code/`; the pipelines run inside external checkouts (CHESS, MAC-SQL, ComEM), so the
repository holds their record extraction and analysis code, and `load()` returns the module that reads
their logged runs when one imports without the checkout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Dict, List, Optional, Tuple

from share_eval_suite import _repo

CONSENSUS = "consensus protocol"
PIPELINE = "pipeline"

GENERATION = "generation"
REVISION = "revision"
EXTRACTION = "extraction"
FINAL_DECISION = "final decision"
STAGES = (GENERATION, REVISION, EXTRACTION, FINAL_DECISION)

_PACKAGE_ROOT = "code"


class ProtocolUnavailable(RuntimeError):
    """A protocol's implementation cannot be imported in this environment."""


@dataclass(frozen=True)
class ProtocolAdapter:
    """One workflow: its stages, where it is implemented, and how to import it.

    `module` is a dotted module of the `consensus_delivery` package (debate and consensus methods) or a
    repository-relative file (pipelines); `None` when the implementation runs only as a script.
    `final_decision` says in plain words how the returned answer is chosen. `in_paper` marks the
    workflows whose runs the paper reports.
    """
    name: str
    kind: str
    stages: Tuple[str, ...]
    files: Tuple[str, ...]
    module: Optional[str]
    final_decision: str
    in_paper: bool
    extra_paths: Tuple[str, ...] = ()
    requires: str = ""
    aliases: Tuple[str, ...] = field(default=())

    def load(self) -> ModuleType:
        """Import the implementation; raises `ProtocolUnavailable` with the reason when it cannot."""
        if self.module is None:
            raise ProtocolUnavailable(
                f"{self.name}: implemented by scripts that run inside the external checkout "
                f"({', '.join(self.files)}); {self.requires or 'no importable module'}")
        try:
            if self.module.endswith(".py"):
                return _repo.load(self.module, *self.extra_paths)
            return _package_module(self.module)
        except ProtocolUnavailable:
            raise
        except Exception as e:                                              # noqa: BLE001
            hint = f" ({self.requires})" if self.requires else ""
            raise ProtocolUnavailable(
                f"{self.name}: importing {self.module} failed{hint}: {type(e).__name__}: {e}") from e


def _package_module(dotted: str) -> ModuleType:
    """Import a module of the `consensus_delivery` package with `code/` on the path.

    The adapters use package-relative imports (`from ..providers import ...`), so they import as
    package modules rather than as single files.
    """
    return _repo.load_package(dotted, _PACKAGE_ROOT)


def _cd(name: str) -> str:
    return f"consensus_delivery.adapters.{name}"


def _f(name: str) -> str:
    return f"code/consensus_delivery/adapters/{name}.py"


_ENTRIES: List[ProtocolAdapter] = [
    # debate and consensus methods reported in the paper
    ProtocolAdapter("Du-MAD", CONSENSUS, STAGES, (_f("du_mad"),), _cd("du_mad"),
                    "majority over the agents' extracted last-round answers, ties to the first seen",
                    True, aliases=("du_mad", "dumad")),
    ProtocolAdapter("Du-MAD (mixed models)", CONSENSUS, STAGES, (_f("du_mad_roster"), _f("du_mad")),
                    _cd("du_mad_roster"),
                    "majority over the agents' extracted last-round answers, one model per agent",
                    True, aliases=("du_mad_roster", "du-mad-roster")),
    ProtocolAdapter("MAD-M2", CONSENSUS, STAGES, (_f("mad_m2"),), _cd("mad_m2"),
                    "most common last-round answer when it holds at least half of the votes",
                    True, aliases=("mad_m2", "madm2", "MAD-M²")),
    ProtocolAdapter("MAD-M2 (mixed models)", CONSENSUS, STAGES, (_f("mad_m2_roster"), _f("mad_m2")),
                    _cd("mad_m2_roster"),
                    "most common last-round answer when it holds at least half of the votes, one model per agent",
                    True, aliases=("mad_m2_roster",)),
    ProtocolAdapter("BCCS", CONSENSUS, STAGES, (_f("bccs"),), _cd("bccs"),
                    "belief-weighted consensus over clustered answers; stops early on full consensus",
                    True, aliases=("bccs",)),
    ProtocolAdapter("MoA", CONSENSUS, (GENERATION, EXTRACTION, FINAL_DECISION), (_f("moa"),), _cd("moa"),
                    "an aggregator model writes one answer from the independent reference answers",
                    True, requires="reads vendor/consensus_literature/moa/moa.py on import (code/README.md)",
                    aliases=("moa",)),
    ProtocolAdapter("FORD", CONSENSUS, STAGES, (_f("ford"),), _cd("ford"),
                    "two models alternate until two consecutive predictions agree, else the round budget ends",
                    True, aliases=("ford",)),
    # debate and consensus methods that were run but are not reported in the paper
    ProtocolAdapter("DMAD", CONSENSUS, STAGES, (_f("dmad"),), _cd("dmad"),
                    "majority over the last round's parsed answers", False, aliases=("dmad",)),
    ProtocolAdapter("Liang MAD", CONSENSUS, STAGES, (_f("liang_mad"),), _cd("liang_mad"),
                    "a moderator ends the debate on a preference, else a judge picks the answer",
                    False, aliases=("liang_mad",)),
    ProtocolAdapter("Debate-or-Vote", CONSENSUS, STAGES, (_f("debate_or_vote"),), _cd("debate_or_vote"),
                    "majority over the last round's answers, ties at random", False,
                    aliases=("debate_or_vote", "dov")),
    ProtocolAdapter("ReConcile", CONSENSUS, STAGES, (_f("reconcile"),), _cd("reconcile"),
                    "confidence-weighted vote over the agents' answers", False, aliases=("reconcile",)),
    ProtocolAdapter("MALLM", CONSENSUS, (GENERATION, REVISION, FINAL_DECISION),
                    (_f("mallm_upstream"), _f("native_run")), _cd("mallm_upstream"),
                    "the released decision protocol's own decision function", False,
                    requires="its runs need vendor/consensus_literature/mallm (code/README.md)",
                    aliases=("mallm", "mallm_upstream")),
    ProtocolAdapter("Task-allocation game", CONSENSUS, (GENERATION, REVISION, FINAL_DECISION),
                    (_f("delib_task_allocation"),), _cd("delib_task_allocation"),
                    "the game's own allocation outcome", False,
                    requires="its runs need vendor/delib (code/README.md)",
                    aliases=("delib_task_allocation",)),
    # multi-stage LLM pipelines
    ProtocolAdapter("CHESS", PIPELINE, (GENERATION, REVISION, FINAL_DECISION),
                    ("experiments/10_chess_unit_tester/code/chess_b1_adapter.py",
                     "experiments/10_chess_unit_tester/code/extract_b1.py",
                     "experiments/10_chess_unit_tester/code/patches/chess_local.patch",
                     "experiments/common/code/adapters/chess.py"),
                    "experiments/10_chess_unit_tester/code/chess_b1_adapter.py",
                    "the Unit Tester picks one of the Candidate Generator's revised queries",
                    True, extra_paths=("experiments/common/code",), aliases=("chess",)),
    ProtocolAdapter("MAC-SQL", PIPELINE, (GENERATION, REVISION),
                    ("experiments/11_macsql_comem_stages/code/extract_macsql_formal.py",
                     "experiments/11_macsql_comem_stages/code/patches/macsql_local.patch",
                     "experiments/11_macsql_comem_stages/code/run_formal_ext.sh"),
                    None, "the Refiner's query (the Decomposer's query when the Refiner is not triggered)",
                    True, requires="run `extract_macsql_formal.py <E> <model_tag> <bench> <out.json>` on a MAC-SQL run",
                    aliases=("macsql", "mac_sql")),
    ProtocolAdapter("ComEM", PIPELINE, (GENERATION, FINAL_DECISION),
                    ("experiments/11_macsql_comem_stages/code/run_comem_formal.py",
                     "experiments/11_macsql_comem_stages/code/build_comem_dev.py",
                     "experiments/11_macsql_comem_stages/code/patches/comem_local.patch"),
                    None, "an LLM selector picks the match among the ranker's top candidates",
                    True, requires="run `run_comem_formal.py` inside a ComEM checkout (imports `src.compound`)",
                    aliases=("comem",)),
]

PROTOCOLS: Dict[str, ProtocolAdapter] = {p.name: p for p in _ENTRIES}


def list_protocols(kind: Optional[str] = None, in_paper: Optional[bool] = None) -> List[ProtocolAdapter]:
    """The registered workflows, optionally restricted to one kind or to those the paper reports."""
    return [p for p in _ENTRIES
            if (kind is None or p.kind == kind) and (in_paper is None or p.in_paper == in_paper)]


def get_protocol(name: str) -> ProtocolAdapter:
    """The workflow called `name` (its display name or an alias, case-insensitive)."""
    key = name.strip().lower()
    for p in _ENTRIES:
        if key == p.name.lower() or key in (a.lower() for a in p.aliases):
            return p
    raise KeyError(f"unknown protocol {name!r}; known: {', '.join(PROTOCOLS)}")


def load(name: str) -> ModuleType:
    """Import the implementation of workflow `name`."""
    return get_protocol(name).load()
