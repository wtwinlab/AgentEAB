"""Task-allocation adapter over the *native* Deliberative Collaboration game.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DELIB = os.path.join(REPO, "vendor", "delib")

DOMAIN = "task_allocation"
#: The frozen decision schema. Changing it is a study-version change.
SCHEMA: Tuple[str, ...] = ("allocation",)
GAMES_ROOT = os.path.join(DELIB, "data", "task_allocation", "games", "task_allocation_games")
BATCH = "level_1_and_2"


def _ensure_path() -> None:
    if DELIB not in sys.path:
        sys.path.insert(0, DELIB)


@dataclass
class Instance:
    """One native task. `private_views` is the only place private information lives."""

    task_id: str
    domain: str
    roster: Tuple[str, ...]
    public_brief: str
    private_views: Dict[str, str]
    schema: Tuple[str, ...]
    _game: Any

    def max_reward(self) -> float:
        return float(self._game.max_reward)

    def reference_allocation(self) -> Dict[str, str]:
        return dict(self._game.best_allocation_dict)


@dataclass
class Quality:
    """Three distinct states the design (§6.3) requires be kept apart.
    """

    valid: bool
    reason: str
    utility: Optional[float]
    normalised: Optional[float]
    reference_utility: float
    feasible: Optional[bool] = None


def load_instances(n: Optional[int] = None, *, games_root: str = GAMES_ROOT,
                   batch: str = BATCH) -> List[Instance]:
    """Load native games through the benchmark's own loader, which registers its pickle aliases."""
    _ensure_path()
    from delib_collab.data_generation.task_allocation.io import load_games  # noqa: E402

    games = load_games(games_root, game_batch_name=batch, n_games=n or 10_000)
    out: List[Instance] = []
    for i, g in enumerate(games):
        roster = tuple(g.agents)
        privates: Dict[str, str] = {}
        for j, a in enumerate(roster):
            obs = getattr(g, f"agent_{j}_obs", None)
            privates[a] = _render_private(obs)
        out.append(Instance(
            task_id=f"{DOMAIN}/{batch}/{i:04d}",
            domain=DOMAIN,
            roster=roster,
            public_brief=_render_public(g),
            private_views=privates,
            schema=SCHEMA,
            _game=g,
        ))
    return out


def _render_public(g: Any) -> str:
    tasks = list(g.tasks)
    lines = [
        "You are one of three agents who must agree on a single assignment of tasks to agents.",
        "",
        "Tasks:",
    ]
    lines += [f"  - {t}" for t in tasks]
    lines += ["", f"Agents: {', '.join(g.agents)}", "",
              "Publicly known resources:"]
    pub = getattr(g, "public_resource_list", None) or []
    lines += [f"  - {r}" for r in pub] or ["  (none)"]
    lines += [
        "",
        "Each agent additionally holds private information about its own resources.",
        "A task can only be performed by an agent whose resources meet that task's requirements.",
    ]
    return "\n".join(lines)


def _render_private(obs: Any) -> str:
    if obs is None:
        return "(no private information)"
    if isinstance(obs, Mapping):
        parts = []
        for k, v in obs.items():
            vs = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
            parts.append(f"{k}: {vs}")
        return "\n".join(parts)
    return str(obs)


def validate(fields: Mapping[str, Any], inst: Instance) -> Tuple[bool, str]:
    """Deterministic syntax/schema check. Not a task-quality gate."""
    alloc = fields.get("allocation")
    if not isinstance(alloc, Mapping) or not alloc:
        return False, "allocation must be a non-empty object"
    tasks = set(inst._game.tasks)
    agents = set(inst.roster)
    for t, a in alloc.items():
        if t not in tasks:
            return False, f"unknown task {t!r}"
        if a not in agents:
            return False, f"unknown agent {a!r}"
    # Leaving a task unassigned is legitimate and is sometimes optimal: the benchmark's own
    # `best_allocation_dict` assigns only 7 of 10 tasks on instance 0000 and still attains
    # `max_reward`.
    return True, ""


def score(fields: Mapping[str, Any], inst: Instance) -> Quality:
    """Native utility. An invalid decision yields `utility=None`, never 0.0.
    """
    ref = inst.max_reward()
    ok, why = validate(fields, inst)
    if not ok:
        return Quality(False, why, None, None, ref, None)
    g = inst._game
    try:
        u = float(g.evaluate_allocation(dict(fields["allocation"])))
        feasible = _feasible(g, dict(fields["allocation"]))
    except Exception as e:  # noqa: BLE001
        return Quality(False, f"native evaluator raised {type(e).__name__}", None, None, ref, None)
    return Quality(True, "", u, (u / ref) if ref else None, ref, feasible)


def _feasible(g: Any, alloc: Mapping[str, str]) -> bool:
    """The benchmark's own constraint check, called directly rather than inferred from a 0.0 reward."""
    _ensure_path()
    from delib_collab.data_generation.task_allocation.game import (  # noqa: E402
        check_allocation_constraints,
        dict_to_allocation_vector,
    )

    vec = dict_to_allocation_vector(dict(alloc), g.tasks, g.agents)
    return bool(check_allocation_constraints(vec, g.constraint_matrix, g.vec_resource_state,
                                             len(g.tasks), len(g.agents)))


def contract() -> Dict[str, Any]:
    """What this adapter exports, for the frozen task manifest."""
    return {
        "domain": DOMAIN,
        "source": "Deliberative Collaboration (preprint benchmark)",
        "source_repo": "vendor/delib",
        "games_root": os.path.relpath(GAMES_ROOT, REPO),
        "batch": BATCH,
        "native_roster_size": 3,
        "private_information": "per-agent resource observations from the benchmark",
        "decision_schema": list(SCHEMA),
        "feasibility": "assigned tasks and assignees must exist; partial allocations are legal "
                       "(the native optimum assigns 7 of 10 tasks on instance 0000); constraint "
                       "satisfaction is inside the reward function",
        "native_quality": "game.evaluate_allocation(allocation) -> reward",
        "native_optimum": "game.max_reward",
        "native_success_criterion": "none defined by the benchmark; utility is continuous",
        "study_added": "unanimity acceptance rule over the roster",
        "invalid_handling": "utility is None, never 0.0",
        "feasibility_separate": "check_allocation_constraints is called directly, so an "
                                "infeasible allocation (reward 0.0 by R(x)=IsValid(x)*v.x) "
                                "is distinguishable from a feasible zero-value one",
    }


def selftest(n: int = 5) -> int:
    """The native reference must validate and must score exactly the native optimum.
    """
    bad = 0
    for i in load_instances(n):
        q = score({"allocation": i.reference_allocation()}, i)
        if not q.valid:
            print(f"  FAIL {i.task_id}: reference allocation invalid ({q.reason})")
            bad += 1
            continue
        if abs(q.utility - i.max_reward()) > 0.01:
            print(f"  FAIL {i.task_id}: reference scores {q.utility:.2f}, optimum {i.max_reward():.2f}")
            bad += 1
            continue
        empty = score({"allocation": {}}, i)
        if empty.valid:
            print(f"  FAIL {i.task_id}: empty allocation accepted as valid")
            bad += 1
            continue
        junk = score({"allocation": {"nope": "agent_0"}}, i)
        if junk.valid or junk.utility is not None:
            print(f"  FAIL {i.task_id}: malformed allocation scored {junk.utility}")
            bad += 1
            continue
        print(f"  ok   {i.task_id}: reference={q.utility:.2f}=={i.max_reward():.2f}, "
              f"normalised={q.normalised:.3f}; invalid -> utility None, not 0.0")
    return bad


if __name__ == "__main__":
    inst = load_instances(3)
    print(f"loaded {len(inst)} instances; schema={SCHEMA}")
    for i in inst:
        q_ref = score({"allocation": i.reference_allocation()}, i)
        bad = score({"allocation": {"nope": "agent_0"}}, i)
        print(f"  {i.task_id}  roster={i.roster}  max_reward={i.max_reward():.2f}")
        print(f"     reference allocation -> valid={q_ref.valid} utility={q_ref.utility} "
              f"normalised={q_ref.normalised}")
        print(f"     malformed            -> valid={bad.valid} utility={bad.utility} ({bad.reason})")
        print(f"     private view (agent_0, first line): "
              f"{i.private_views[i.roster[0]].splitlines()[0][:70]}")
