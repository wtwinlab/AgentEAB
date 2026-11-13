r"""The roster adapters must differ from the originals in the model assignment and nothing else.

Both heterogeneous adapters import their prompts, parsing and aggregation from the homogeneous
module, so they *cannot* drift on those. Sharing the prompt functions proves nothing about the
**loop** — the round structure, who sees whose reply, what history each agent carries, when the run
stops, how the winner is picked. Each roster module reimplements that loop, and a shared prompt
import is no evidence at all that the reimplementation matches.

So the loop is compared directly, **request by request**. Both paths are driven with scripted
responses, every outgoing request is recorded, and the two recordings must agree on everything
except the alias field.

## The control, and why it is a control and not a result

The roster path is run with the same alias in all three seats. That makes the two paths comparable
at all: a real roster would send different models and the recordings could not be compared
message-for-message. A repeated alias is therefore a **test instrument**, and a run of it is not a
heterogeneous result and is never collected as one — the CLI of both roster modules refuses a
homogeneous roster for exactly that reason, and these tests bypass the CLI by calling `run_one`.

Nothing here touches collected data. The transport is replaced by a script, so no request reaches
any endpoint and no production cell is re-run.
"""
from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from consensus_delivery.adapters import du_mad, du_mad_roster, mad_m2, mad_m2_roster  # noqa: E402


class Rec:
    """The fields of a provider record that the adapters read."""

    def __init__(self, text: str, n_logprobs: int = 8) -> None:
        self.served_model = "scripted"
        self.prompt_tokens = 11
        self.completion_tokens = 7
        self.finish_reason = "stop"
        self.attempt = 1
        self.latency_s = 0.25
        self.status = "ok"
        self.raw_ref = None
        self.reasoning_tokens = None
        self.reasoning_chars = None
        self.usd = 0.0
        self.request_sha = "sha"
        self.fingerprint = "fp"
        # Fixed per call index so perplexity is deterministic but not constant across seats --
        # a constant would make MAD-M2's median-based prune degenerate and hide a real difference.
        self.token_logprobs = [{"logprob": -0.5 - 0.1 * (i % 3)} for i in range(n_logprobs)]


class Script:
    """A `chat` stand-in that answers deterministically and records every request verbatim."""

    def __init__(self, replies: List[str]) -> None:
        self.replies = replies
        self.log: List[Dict[str, Any]] = []

    def __call__(self, *, registry: Any, alias: str, messages: List[Dict[str, str]],
                 **kwargs: Any) -> Tuple[str, Rec]:
        n = len(self.log)
        self.log.append({
            "alias": alias,
            "messages": [dict(m) for m in messages],
            # Every generation knob, because a drift in one of these is a protocol drift.
            "kwargs": {k: v for k, v in kwargs.items() if k != "registry"},
        })
        text = self.replies[n % len(self.replies)]
        return text, Rec(text)


REPLIES = [
    "Reasoning about it. My answer is (A)",
    "A different line of reasoning. My answer is (B)",
    "Agreeing with the first. My answer is (A)",
    "Second round, holding. My answer is (A)",
    "Second round, changing. My answer is (A)",
    "Second round, also (A). My answer is (A)",
]


def strip_alias(log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in call.items() if k != "alias"} for call in log]


class DuMadRosterMatchesOriginal(unittest.TestCase):
    """Du MAD: the roster loop must issue the same requests in the same order."""

    ALIAS = "llama3-8b"

    def run_both(self, **kw: Any) -> Tuple[Script, Script, Dict[str, Any], Dict[str, Any]]:
        q = "What is 2+2? (A) 4 (B) 5"
        common = dict(benchmark="mmlu_pro", rounds=2, max_tokens=256, timeout_s=9.0, retries=1)
        common.update(kw)

        s_orig = Script(REPLIES)
        du_mad.chat = s_orig                                   # type: ignore[assignment]
        orig = du_mad.run_one(q, alias=self.ALIAS, agents=3, **common)

        s_rost = Script(REPLIES)
        du_mad_roster.chat = s_rost                            # type: ignore[assignment]
        # Repeated alias: the control described in this module's docstring.
        rost = du_mad_roster.run_one(q, seat_models=[self.ALIAS] * 3, **common)
        return s_orig, s_rost, orig, rost

    def test_same_number_of_requests(self) -> None:
        s_orig, s_rost, _o, _r = self.run_both()
        self.assertEqual(len(s_orig.log), len(s_rost.log),
                         "the roster loop issues a different number of calls")
        self.assertEqual(len(s_orig.log), 6, "3 agents x 2 rounds; the fixture itself drifted")

    def test_every_request_identical_except_the_alias(self) -> None:
        s_orig, s_rost, _o, _r = self.run_both()
        for i, (a, b) in enumerate(zip(strip_alias(s_orig.log), strip_alias(s_rost.log))):
            self.assertEqual(a["messages"], b["messages"],
                             f"call {i}: the conversation sent differs between the two paths")
            self.assertEqual(a["kwargs"], b["kwargs"],
                             f"call {i}: a generation parameter differs at call {i}")

    def test_the_alias_is_the_only_difference_and_it_is_per_seat(self) -> None:
        s_orig, s_rost, _o, _r = self.run_both()
        self.assertEqual([c["alias"] for c in s_orig.log], [self.ALIAS] * 6)
        self.assertEqual([c["alias"] for c in s_rost.log], [self.ALIAS] * 6)
        # And with a real roster the alias follows the seat, round after round.
        s = Script(REPLIES)
        du_mad_roster.chat = s                                 # type: ignore[assignment]
        du_mad_roster.run_one("q? (A) 4 (B) 5", benchmark="mmlu_pro",
                              seat_models=["llama3-8b", "qwen25-7b", "mistral-7b"],
                              rounds=2, max_tokens=256, timeout_s=9.0, retries=1)
        self.assertEqual([c["alias"] for c in s.log],
                         ["llama3-8b", "qwen25-7b", "mistral-7b"] * 2,
                         "the seat-to-model mapping is not stable across rounds")

    def test_history_each_agent_carries_is_identical(self) -> None:
        """The all-to-all context is the thing most likely to drift when a loop is rewritten."""
        s_orig, s_rost, _o, _r = self.run_both()
        for i in range(6):
            a, b = s_orig.log[i]["messages"], s_rost.log[i]["messages"]
            self.assertEqual(len(a), len(b), f"call {i}: history length differs")
            self.assertEqual([m["role"] for m in a], [m["role"] for m in b],
                             f"call {i}: role sequence differs")
        # Round 2 must carry the peers' round-1 replies; if it does not, the fixture is vacuous
        # and would pass even if both paths dropped the history entirely.
        self.assertGreater(len(s_orig.log[3]["messages"]), len(s_orig.log[0]["messages"]),
                           "round 2 did not grow the context -- this test proves nothing")

    def test_outcome_fields_agree(self) -> None:
        """On every field both paths return. `du_mad.main` adds the rest, and `_runner` the same."""
        _so, _sr, orig, rost = self.run_both()
        for k in ("per_agent_final_answer", "majority_answer", "n_distinct_final_answers",
                  "accepted_delivery", "tokens_in", "tokens_out", "usd"):
            self.assertEqual(orig[k], rost[k], f"{k} differs between the two paths")
        # Both must agree that there is no stop rule, however each spells it.
        self.assertIsNone(rost["rule_satisfied"])
        self.assertIsNone(orig.get("rule_satisfied"))

    def test_the_recording_asymmetries_that_are_left_and_why_they_stay(self) -> None:
        """Three differences survive, all in recorded text and none in a reported number.

        They are pinned so that every `du_mad` roster cell spells the same facts the same way.
        Nothing reads them: `native.py` counts stop reasons **within**
        a cell, so a per-arm string is self-consistent, and `agent_contexts` is reconstructible
        from `calls[].messages_sent` and `calls[].raw_reply`, which both paths record.
        """
        _so, _sr, orig, rost = self.run_both()
        self.assertEqual(orig["native_stop_reason"], "fixed-round-budget-exhausted")
        self.assertEqual(rost["native_stop_reason"], "fixed round budget")
        self.assertIn("agent_contexts", orig)
        self.assertNotIn("agent_contexts", rost)
        self.assertNotEqual(orig["accepted_delivery_note"], rost["accepted_delivery_note"])
        # The information `agent_contexts` holds is present either way.
        for c in rost["calls"]:
            self.assertIn("messages_sent", c)
            self.assertIn("raw_reply", c)

    def test_roster_records_what_the_original_cannot(self) -> None:
        _so, _sr, orig, rost = self.run_both()
        self.assertNotIn("per_agent_final_by_model", orig)
        self.assertIn("per_agent_final_by_model", rost)
        self.assertIn("seat_models", rost)
        self.assertTrue(all("agent_alias" in c for c in rost["calls"]))

    def test_the_comparison_can_fail(self) -> None:
        """A drift detector that cannot fail is not a detector."""
        q = "What is 2+2? (A) 4 (B) 5"
        common = dict(benchmark="mmlu_pro", rounds=2, max_tokens=256, timeout_s=9.0, retries=1)
        s_orig = Script(REPLIES)
        du_mad.chat = s_orig                                   # type: ignore[assignment]
        du_mad.run_one(q, alias=self.ALIAS, agents=3, **common)
        s_rost = Script(REPLIES)
        du_mad_roster.chat = s_rost                            # type: ignore[assignment]
        du_mad_roster.run_one(q, seat_models=[self.ALIAS] * 3, **{**common, "rounds": 3})
        self.assertNotEqual(len(s_orig.log), len(s_rost.log),
                            "a deliberately drifted loop compared equal; the test is vacuous")


class MadM2RosterMatchesOriginal(unittest.TestCase):
    """MAD-M2: same comparison, plus the prune must mask the same peers."""

    ALIAS = "llama3-8b"

    def run_both(self, prune: str = "objective") -> Tuple[Script, Script,
                                                          Dict[str, Any], Dict[str, Any]]:
        q = "What is 2+2?"
        common = dict(benchmark="gsm8k", rounds=2, prune=prune, max_tokens=256,
                      timeout_s=9.0, retries=1, temperature=1.0, top_p=1.0)

        s_orig = Script(REPLIES)
        mad_m2.chat = s_orig                                   # type: ignore[assignment]
        orig = mad_m2.run_one(q, task_id="t1", alias=self.ALIAS, agents=3, **common)

        s_rost = Script(REPLIES)
        mad_m2_roster.chat = s_rost                            # type: ignore[assignment]
        rost = mad_m2_roster.run_one(q, task_id="t1", seat_models=[self.ALIAS] * 3, **common)
        return s_orig, s_rost, orig, rost

    def test_every_request_identical_except_the_alias(self) -> None:
        for prune in ("objective", "naive"):
            with self.subTest(prune=prune):
                s_orig, s_rost, _o, _r = self.run_both(prune)
                self.assertEqual(len(s_orig.log), len(s_rost.log),
                                 f"{prune}: different number of calls")
                for i, (a, b) in enumerate(zip(strip_alias(s_orig.log),
                                               strip_alias(s_rost.log))):
                    self.assertEqual(a["messages"], b["messages"],
                                     f"{prune}: call {i} sends a different conversation")
                    self.assertEqual(a["kwargs"], b["kwargs"],
                                     f"{prune}: call {i} differs in a generation parameter")

    def test_logprobs_are_requested_on_both_paths(self) -> None:
        """Perplexity drives the prune, so a path that forgot `logprobs` would prune on nothing."""
        s_orig, s_rost, _o, _r = self.run_both()
        self.assertTrue(all(c["kwargs"].get("logprobs") for c in s_orig.log))
        self.assertTrue(all(c["kwargs"].get("logprobs") for c in s_rost.log))

    def test_the_prune_masks_the_same_peers(self) -> None:
        _so, _sr, orig, rost = self.run_both("objective")
        self.assertEqual([r["masks"] for r in orig["rounds_detail"]],
                         [r["masks"] for r in rost["rounds_detail"]],
                         "the objective prune masked different peers")
        self.assertEqual([r["perplexities"] for r in orig["rounds_detail"]],
                         [r["perplexities"] for r in rost["rounds_detail"]])
        # The fixture must actually exercise masking, or the equality above is vacuous.
        self.assertTrue(any(False in m for r in orig["rounds_detail"] for m in r["masks"]),
                        "no peer was masked in this fixture -- the prune was never exercised")

    def test_outcome_fields_agree(self) -> None:
        _so, _sr, orig, rost = self.run_both()
        for k in ("per_agent_final_answer", "majority_answer", "n_distinct_final_answers",
                  "rule_satisfied", "consensus_per_round", "answer_kind", "prune_strategy",
                  "agents", "rounds", "native_stop_reason"):
            self.assertEqual(orig[k], rost[k], f"{k} differs between the two paths")

    def test_per_call_fields_are_at_parity(self) -> None:
        """Every per-call field the homogeneous path records, this one records too.

        `latency_s`, `n_logprob_tokens` and `mask_over_peers` are recorded by both paths.
        """
        _so, _sr, orig, rost = self.run_both()
        for k in ("prompt_tokens", "completion_tokens", "finish_reason", "usd", "perplexity",
                  "extracted_answer", "raw_reply", "messages_sent", "latency_s",
                  "n_logprob_tokens", "mask_over_peers", "round", "agent_index"):
            self.assertTrue(all(k in c for c in orig["calls"]), f"original dropped {k}")
            self.assertTrue(all(k in c for c in rost["calls"]), f"roster dropped {k}")
        for k in ("absence_cause", "is_task_adaptation", "adaptation_note",
                  "naive_mask_literal_differs_from_roster", "native_stop_reason"):
            self.assertEqual(orig[k], rost[k], f"{k} differs between the two paths")

    def test_the_released_consensus_predicate_is_used_not_unanimity(self) -> None:
        """The regression that cost two cells: a 2-1 split is consensus under the released rule."""
        from consensus_delivery.adapters.mad_m2 import if_reach_consensus
        self.assertEqual(if_reach_consensus(["A", "A", "B"]), (True, "A"))
        # And the roster path must agree with the original on a run that produces a 2-1 split.
        replies = ["Answer: 4", "Answer: 4", "Answer: 5"] * 2
        common = dict(benchmark="gsm8k", rounds=2, prune="objective", max_tokens=99,
                      timeout_s=9.0, retries=1, temperature=1.0, top_p=1.0)
        s_o = Script(replies)
        mad_m2.chat = s_o                                      # type: ignore[assignment]
        o = mad_m2.run_one("q", task_id="t", alias=self.ALIAS, agents=3, **common)
        s_r = Script(replies)
        mad_m2_roster.chat = s_r                               # type: ignore[assignment]
        r = mad_m2_roster.run_one("q", task_id="t", seat_models=[self.ALIAS] * 3, **common)
        self.assertEqual(o["rule_satisfied"], r["rule_satisfied"])
        self.assertEqual(o["majority_answer"], r["majority_answer"])
        self.assertTrue(o["rule_satisfied"], "the fixture did not produce a split to compare on")

    def test_acceptance_is_not_measured_on_either_path(self) -> None:
        """MAD-M2 asks no agent to accept, so neither path may invent a verdict."""
        _so, _sr, orig, rost = self.run_both()
        self.assertEqual(orig["accepted_delivery"], "not-measured")
        self.assertEqual(rost["accepted_delivery"], "not-measured")

    def test_cross_tokenizer_note_is_recorded_and_nothing_is_normalised(self) -> None:
        _so, _sr, orig, rost = self.run_both()
        self.assertIn("cross_tokenizer_perplexity_note", rost)
        self.assertNotIn("cross_tokenizer_perplexity_note", orig)
        # The recorded perplexities must be the raw per-seat values, not rescaled ones.
        for r_o, r_r in zip(orig["rounds_detail"], rost["rounds_detail"]):
            self.assertEqual(r_o["perplexities"], r_r["perplexities"])

    def test_per_model_records_are_keyed_by_seat_model(self) -> None:
        s = Script(REPLIES)
        mad_m2_roster.chat = s                                 # type: ignore[assignment]
        seats = ["llama3-8b", "qwen25-7b", "mistral-7b"]
        r = mad_m2_roster.run_one("q", task_id="t", benchmark="gsm8k", seat_models=seats,
                                  rounds=2, prune="objective", max_tokens=256, timeout_s=9.0,
                                  retries=1, temperature=1.0, top_p=1.0)
        self.assertEqual([c["alias"] for c in s.log], seats * 2)
        for rd in r["rounds_detail"]:
            self.assertEqual(sorted(rd["perplexity_by_model"]), sorted(seats))
        # Round 0 has no peers to mask, and the original records `[]` there; the per-model view
        # follows that shape rather than inventing three empty masks.
        self.assertEqual(r["rounds_detail"][0]["masked_by_model"], {})
        self.assertEqual(sorted(r["rounds_detail"][1]["masked_by_model"]), sorted(seats))


class RosterRefusesHomogeneousAtTheBoundary(unittest.TestCase):
    """The control above is a test instrument; the collectable path must refuse it."""

    def test_both_clis_refuse_a_repeated_alias(self) -> None:
        for mod in (du_mad_roster, mad_m2_roster):
            with self.subTest(module=mod.__name__):
                argv = sys.argv
                sys.argv = [mod.__name__, "--block", "/nonexistent.json",
                            "--roster", "llama3-8b,llama3-8b,llama3-8b", "--out", "/dev/null"]
                try:
                    self.assertEqual(mod.main(), 2,
                                     "a homogeneous roster was not refused, so a repeated alias "
                                     "could be collected as a heterogeneous result")
                finally:
                    sys.argv = argv


if __name__ == "__main__":
    unittest.main(verbosity=2)
