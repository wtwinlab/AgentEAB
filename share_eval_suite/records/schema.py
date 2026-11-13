"""The common execution record.

Every workflow run is logged per task, agent and round. A record holds what the stage evaluation and
the alternative comparisons read: the agents' responses, the candidate answers extracted from them,
the decision and stopping events, the task score, and the LLM calls and tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Call:
    """One LLM call of one agent in one round."""
    agent: int
    round: int
    response: str = ""                       # the agent's reply as returned by the model
    candidate: Optional[str] = None          # the answer extracted from the reply, if any
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stage: str = ""                          # generation, revision, extraction or final decision
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """The decision of a run: the chosen answer and how the run stopped."""
    answer: Optional[str] = None
    rule: str = ""                           # e.g. exact-match vote, execution vote, judge
    stopped_at_round: Optional[int] = None
    stop_reason: str = ""


@dataclass
class ExecutionRecord:
    """One run of a workflow on one task."""
    task: str                                # task identifier, e.g. "bird|100"
    workflow: str                            # protocol or pipeline, e.g. "du_mad", "chess"
    condition: str = ""                      # e.g. "FULL" (with peers) or "NOPEER"
    backbone: str = ""
    repeat: int = 1
    cluster: str = ""                        # resampling cluster, e.g. the BIRD database
    calls: List[Call] = field(default_factory=list)
    decision: Decision = field(default_factory=Decision)
    scores: Dict[str, float] = field(default_factory=dict)   # task score per reading of the run
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- views used by the stages
    def rounds(self) -> List[int]:
        return sorted({c.round for c in self.calls})

    def candidates(self, round: int) -> List[Optional[str]]:
        """Candidate answers of one round, in agent order."""
        return [c.candidate for c in sorted(self.calls, key=lambda c: c.agent) if c.round == round]

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)
