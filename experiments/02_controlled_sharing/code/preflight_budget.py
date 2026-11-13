#!/usr/bin/env python3
"""Budget ledger and watchdog for the controlled-5x5 preflight. Enforced per call, not per batch.
"""
from __future__ import annotations

import json
import pathlib
import threading
import time
from typing import Any, Dict, Optional


class BudgetExceeded(RuntimeError):
    """Raised when a cap is hit. Carries the reason so the caller can persist it."""

    def __init__(self, reason: str, ledger: Dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.ledger = ledger


#: A call granted less than this cannot succeed, so sending it spends a call and a GPU slot to
#: manufacture a `TimeoutError` that then reads as a protocol failure.
MIN_CALL_SECONDS = 20.0


class Budget:
    def __init__(self, *, max_calls: int, max_output_tokens: int, max_input_tokens: int,
                 max_gpu_hours: float, max_wall_seconds: int, cards: int,
                 state_path: pathlib.Path):
        self.caps = {
            "calls": max_calls, "output_tokens": max_output_tokens,
            "input_tokens": max_input_tokens, "gpu_hours": max_gpu_hours,
            "wall_seconds": max_wall_seconds,
        }
        self.cards = cards
        self.state_path = state_path
        self.used = {"calls": 0, "output_tokens": 0, "input_tokens": 0}
        self._reserved_out = 0
        self.t0 = time.time()
        self.stop_reason: Optional[str] = None
        self._lock = threading.Lock()
        # the wall-clock cap is whichever is tighter: the explicit seconds, or the GPU-hour cap
        # divided by the number of cards actually occupied
        gpu_seconds = max_gpu_hours * 3600.0 / max(cards, 1)
        self.deadline = self.t0 + min(max_wall_seconds, gpu_seconds)
        self.effective_wall_seconds = min(max_wall_seconds, gpu_seconds)
        self._wd = threading.Thread(target=self._watchdog, daemon=True)
        self._wd.start()

    # ---------------------------------------------------------------- watchdog
    def _watchdog(self):
        while True:
            left = self.deadline - time.time()
            if left <= 0:
                with self._lock:
                    if self.stop_reason is None:
                        self.stop_reason = (
                            f"wall-clock/GPU-hour cap: {self.effective_wall_seconds:.0f}s "
                            f"({self.cards} card(s) x {self.caps['gpu_hours']} GPU-h)")
                self.persist()
                return
            time.sleep(min(1.0, max(0.05, left)))

    # ---------------------------------------------------------------- accounting
    def seconds_left(self) -> float:
        return max(0.0, self.deadline - time.time())

    def gpu_hours_used(self) -> float:
        return (time.time() - self.t0) * self.cards / 3600.0

    def check(self):
        """Raise if any cap is already breached. Called between unit-runs."""
        with self._lock:
            if self.stop_reason:
                raise BudgetExceeded(self.stop_reason, self.snapshot())
            for k in ("calls", "output_tokens", "input_tokens"):
                if self.used[k] >= self.caps[k]:
                    self.stop_reason = f"{k} cap: {self.used[k]} >= {self.caps[k]}"
                    raise BudgetExceeded(self.stop_reason, self.snapshot())
            if self.gpu_hours_used() >= self.caps["gpu_hours"]:
                self.stop_reason = (f"gpu_hours cap: {self.gpu_hours_used():.4f} >= "
                                    f"{self.caps['gpu_hours']}")
                raise BudgetExceeded(self.stop_reason, self.snapshot())

    def reserve(self, max_tokens: int, est_prompt_tokens: int = 0) -> float:
        """Refuse the call unless its WORST case fits. Returns the timeout to use for it."""
        self.check()
        with self._lock:
            if self.used["calls"] + 1 > self.caps["calls"]:
                self.stop_reason = f"calls cap would be crossed by this call ({self.caps['calls']})"
                raise BudgetExceeded(self.stop_reason, self.snapshot())
            worst_out = self.used["output_tokens"] + self._reserved_out + max_tokens
            if worst_out > self.caps["output_tokens"]:
                self.stop_reason = (f"output_tokens cap would be crossed in the worst case: "
                                    f"{worst_out} > {self.caps['output_tokens']}")
                raise BudgetExceeded(self.stop_reason, self.snapshot())
            worst_in = self.used["input_tokens"] + est_prompt_tokens
            if worst_in > self.caps["input_tokens"]:
                self.stop_reason = (f"input_tokens cap would be crossed: {worst_in} > "
                                    f"{self.caps['input_tokens']}")
                raise BudgetExceeded(self.stop_reason, self.snapshot())
            self._reserved_out += max_tokens
        # never let one call outlive the window; leave 2 s to persist state
        grant = min(60.0, self.seconds_left() - 2.0)
        if grant < MIN_CALL_SECONDS:
            # Refuse rather than send a request that cannot finish.
            with self._lock:
                self._reserved_out -= max_tokens
                self.stop_reason = (f"wall-clock remainder {self.seconds_left():.0f}s is below the "
                                    f"{MIN_CALL_SECONDS:.0f}s a call needs; refusing to send")
                raise BudgetExceeded(self.stop_reason, self.snapshot())
        return grant

    def settle(self, max_tokens: int, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            self._reserved_out = max(0, self._reserved_out - max_tokens)
            self.used["calls"] += 1
            self.used["output_tokens"] += int(completion_tokens or 0)
            self.used["input_tokens"] += int(prompt_tokens or 0)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "caps": self.caps, "cards_occupied": self.cards,
            "effective_wall_seconds": round(self.effective_wall_seconds, 1),
            "used": dict(self.used),
            "reserved_output_tokens": self._reserved_out,
            "elapsed_seconds": round(time.time() - self.t0, 1),
            "gpu_hours_used": round(self.gpu_hours_used(), 5),
            "seconds_left": round(self.seconds_left(), 1),
            "stop_reason": self.stop_reason,
        }

    def persist(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.snapshot(), indent=1) + "\n",
                                       encoding="utf-8")
        except OSError:
            pass


def self_test() -> None:
    """Prove the guards fire, offline. No model calls."""
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    b = Budget(max_calls=3, max_output_tokens=100, max_input_tokens=1000, max_gpu_hours=1.0,
               max_wall_seconds=60, cards=2, state_path=d / "s.json")
    # a call whose worst case fits
    t = b.reserve(50, est_prompt_tokens=10)
    assert t > 0
    b.settle(50, 10, 20)
    assert b.used["output_tokens"] == 20 and b.used["calls"] == 1
    # a call whose worst case would cross the output cap must be refused
    try:
        b.reserve(90, est_prompt_tokens=10)
        raise AssertionError("output worst-case guard did not fire")
    except BudgetExceeded as e:
        assert "worst case" in str(e)
    # the GPU-hour cap must tighten the wall clock: 0.1 GPU-h on 2 cards = 180 s
    b2 = Budget(max_calls=10, max_output_tokens=10 ** 9, max_input_tokens=10 ** 9,
                max_gpu_hours=0.1, max_wall_seconds=3600, cards=2, state_path=d / "s2.json")
    assert abs(b2.effective_wall_seconds - 180.0) < 1.0, b2.effective_wall_seconds
    # call cap
    b3 = Budget(max_calls=1, max_output_tokens=10 ** 6, max_input_tokens=10 ** 6,
                max_gpu_hours=1.0, max_wall_seconds=60, cards=1, state_path=d / "s3.json")
    b3.reserve(10); b3.settle(10, 1, 1)
    try:
        b3.reserve(10)
        raise AssertionError("call cap guard did not fire")
    except BudgetExceeded:
        pass
    print("  budget self-test: worst-case output guard, GPU-hour->wall-clock tightening, "
          "call cap all fire")


if __name__ == "__main__":
    self_test()
