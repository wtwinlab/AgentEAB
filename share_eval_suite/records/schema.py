from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Call:
    agent: int
    round: int
    response: str = ""
    candidate: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stage: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    answer: Optional[str] = None
    rule: str = ""
    stopped_at_round: Optional[int] = None
    stop_reason: str = ""


@dataclass
class ExecutionRecord:
    task: str
    workflow: str
    condition: str = ""
    backbone: str = ""
    repeat: int = 1
    cluster: str = ""
    calls: List[Call] = field(default_factory=list)
    decision: Decision = field(default_factory=Decision)
    scores: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def rounds(self) -> List[int]:
        return sorted({c.round for c in self.calls})

    def candidates(self, round: int) -> List[Optional[str]]:
        return [c.candidate for c in sorted(self.calls, key=lambda c: c.agent) if c.round == round]

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)
