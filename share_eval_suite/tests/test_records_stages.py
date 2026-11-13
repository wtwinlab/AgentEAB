"""Readers of records and scored tables; candidate availability and answer changes."""
import json

import pytest

from share_eval_suite.records import readers
from share_eval_suite.stages import availability, changes
from share_eval_suite.tests._expect import assert_reproduces

FLOW = "tab_candidate-flow"
STRATA = "tab_s-pool-strata"
POOL = "text_generation-beyond-pool"
PEER = "text_peer-source"
TAG = {"Du-MAD": "dumad", "MAD-M2": "madm2", "BCCS": "bccs", "llama3-8b": "llama", "qwen2.5-7b": "qwen"}


def r4(x):
    # tab_candidate-flow stores its values at four decimals
    return round(float(x), 4)


@pytest.fixture(scope="module")
def flow():
    return availability.candidate_flow()


# ------------------------------------------------------------------ readers of scored tables
def test_scored_tables_read():
    runs = readers.read_per_run_scores()
    assert len(runs) == 5544
    recs = readers.per_run_scores_to_records(runs)
    assert len(recs) == len(runs)
    assert all(r.cluster for r in recs)
    assert set(recs[0].scores) >= {"final", "strict|string|last|first", "lenient|exec|r0|random"}
    flow_runs = readers.read_candidate_flow_runs()
    assert_reproduces(FLOW, "runs", len(flow_runs))
    assert len(readers.candidate_flow_to_records(flow_runs)) == len(flow_runs)


def test_flow_votes_equal_per_run_scores():
    # the vote columns of the candidate-flow table are the per-run scores of tab_factorial
    name = {"du_mad": "Du-MAD", "madm2_objective": "MAD-M2", "bccs": "BCCS"}
    scored = {(r["unit"], name[r["protocol"]], r["backbone"], int(r["repeat"])): r
              for r in readers.read_per_run_scores() if r["arm"] == "FULL" and r["protocol"] in name}
    for r in readers.read_candidate_flow_runs():
        s = scored[(r["question"], r["protocol"], r["backbone"], r["repeat"])]
        assert r["protocol_exact_first"] == s["strict|string|last|first"]
        assert r["lenient_execution_last"] == s["lenient|exec|last|last"]


# ------------------------------------------------------------------ readers of logged debates
def _call(rnd, agent, reply, extracted=None, pt=10, ct=5):
    return {"round": rnd, "agent_index": agent, "raw_reply": reply, "extracted_answer": extracted,
            "prompt_tokens": pt, "completion_tokens": ct}


def _logged_run():
    calls = [_call(0, 0, "```sql\nSELECT a FROM t\n```"),
             _call(0, 1, "answer: SELECT b FROM t", extracted="SELECT b FROM t"),
             _call(0, 2, "no query here"),
             _call(1, 0, "```sql\nSELECT b FROM t\n```", extracted="SELECT b FROM t"),
             _call(1, 1, "SELECT b FROM t", extracted="SELECT b FROM t"),
             _call(1, 2, "I keep\nSELECT a FROM t")]
    return {"unit": "bird|7", "protocol": "du_mad", "arm": "FULL", "backbone": "llama3-8b", "repeat": 1,
            "status": "ok", "config": {"unit_meta": {"db_id": "toy"}},
            "record": {"calls": calls, "majority_answer": "SELECT b FROM t", "stopped_at_round": 1,
                       "native_stop_reason": "max rounds"}}


def test_logged_run_to_record(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    run = _logged_run()
    failed = dict(run, status="error")
    (raw / "sql_bird_v2__du_mad__FULL__part0.jsonl").write_text(
        "\n".join(json.dumps(x) for x in (failed, run)) + "\nnot json\n")
    assert readers.logged_debates_available(tmp_path)
    runs, per_unit, seen = readers.load_logged_debates(tmp_path, protocols=("du_mad",))
    assert per_unit is None and seen == {"non_ok_record": 1, "unparseable_line": 1}
    key = ("bird|7", "du_mad", "FULL", "llama3-8b", 1)
    rec = readers.logged_debate_to_record(key, runs[key])
    assert rec.cluster == "toy" and rec.rounds() == [0, 1] and rec.n_calls == 6 and rec.tokens == 90
    assert rec.candidates(0) == [None, "SELECT b FROM t", None]
    assert [c.meta["lenient"] for c in rec.calls if c.round == 0] == ["SELECT a FROM t", "SELECT b FROM t", None]
    assert rec.decision.answer == "SELECT b FROM t" and rec.decision.stopped_at_round == 1
    assert len(readers.read_logged_debates(tmp_path, protocols=("du_mad",))) == 1

    (row,) = readers.candidate_sets(runs, None, ("du_mad",))
    assert row["sets"]["last"]["lenient"] == ["SELECT b FROM t", "SELECT b FROM t", "SELECT a FROM t"]
    k = readers.query_key
    verdicts = {k("toy", "SELECT b FROM t"): {"correct": True}, k("toy", "SELECT a FROM t"): {"correct": False}}
    sigs = {k("toy", "SELECT b FROM t"): {"status": "ok", "sig": "B"},
            k("toy", "SELECT a FROM t"): {"status": "ok", "sig": "A"}}
    s = readers.score_candidate_set(row, sigs, verdicts)
    assert len(s) == 2 * 3 * 2 * 3
    assert s["strict|string|r0|first"] == 1.0          # one readable candidate, correct
    assert s["lenient|string|r0|first"] == 0.0         # tie between a and b goes to the first agent
    assert s["lenient|string|r0|last"] == 1.0
    assert s["lenient|exec|last|first"] == 1.0


def test_hosted_debates_when_present():
    if not readers.logged_debates_available():
        pytest.skip("data/records/ is not in the repository")
    recs = readers.read_logged_debates()
    assert recs and all(r.calls for r in recs)


# ------------------------------------------------------------------ candidate availability
def test_candidate_flow_headline(flow):
    assert_reproduces(FLOW, "questions", flow["questions"])
    for t in ("C0", "CT", "CD", "Y", "Yx", "loss", "recovery"):
        assert_reproduces(FLOW, f"{t}_mean6", r4(flow["mean"][t]))
    L = flow["stage_losses"]
    assert_reproduces(FLOW, "revision_net_mean6", r4(L["revision_net"]))
    assert_reproduces(FLOW, "extraction_gap_mean6", r4(L["extraction_gap"]))
    assert_reproduces(FLOW, "vote_gap_mean6", r4(L["vote_gap"]))
    assert_reproduces(FLOW, "Yx_minus_Y_mean6", r4(L["Yx_minus_Y"]))
    assert_reproduces(FLOW, "flow_identity_holds", int(flow["identity_holds"]))
    assert_reproduces(FLOW, "CT_ge_CD_ge_Y_every_run", int(flow["ordered_every_run"]))


def test_candidate_flow_table(flow):
    for (p, b), row in flow["per_configuration"].items():
        for t in availability.STAGE_FLAGS:
            assert_reproduces(FLOW, f"{t}_{TAG[p]}_{TAG[b]}", r4(row[t]))
    four = availability.candidate_flow(configs=availability.configurations()[:4])
    assert_reproduces(FLOW, "runs_four_configurations", four["runs"])


def test_generation_gain_by_pool_availability():
    ds = availability.generation_gain_by_availability("deepseek-v4-pro", "four_model")
    assert_reproduces(STRATA, "ds_pool4_total", ds["total"]["est"])
    assert_reproduces(STRATA, "ds_pool4_present", ds["present"]["est"])
    assert_reproduces(STRATA, "ds_pool4_absent", ds["absent"]["est"])
    assert_reproduces(STRATA, "ds_pool4_absent_db_lo", ds["absent"]["db_lo"])
    assert_reproduces(STRATA, "ds_pool4_absent_q_hi", ds["absent"]["q_hi"])
    assert_reproduces(STRATA, "ds_pool4_present_detected", ds["present"]["detected"])
    assert_reproduces(STRATA, "ds_pool4_questions", ds["questions"])
    assert_reproduces(STRATA, "ds_pool4_absent_questions", ds["absent_questions"])
    sel = availability.generation_gain_by_availability(against="selector")
    with pytest.raises(ValueError):
        availability.generation_gain_by_availability("gemma-4-26b", "four_model", against="selector")
    assert_reproduces(STRATA, "ds_pool4_present_minus_selector", sel["present"]["est"])
    ref = availability.generation_gain_by_availability(with_reference_only=True)
    assert_reproduces(STRATA, "ds_pool4_with_reference_total", ref["total"]["est"])
    q25 = availability.generation_gain_by_availability("qwen2.5-7b", "six_candidates")
    assert_reproduces(STRATA, "q25_7b_pool6_present", q25["present"]["est"])
    assert_reproduces(STRATA, "q25_7b_pool6_questions", q25["questions"])
    for k, v in availability.generation_gain_range().items():
        assert_reproduces(STRATA, k, v)


def test_method_gain_beyond_pool():
    def r6(x):
        # text_generation-beyond-pool stores its values at six decimals
        return round(float(x), 6)
    g = availability.method_gain_by_availability("DS_G", "Nonempty")
    assert_reproduces(POOL, "bird_DS_G_minus_Nonempty", r6(g["total"]["est"]))
    assert_reproduces(POOL, "bird_DS_G_minus_Nonempty_oracle0", r6(g["absent"]["est"]))
    assert_reproduces(POOL, "bird_DS_G_minus_Nonempty_oracle1", r6(g["present"]["est"]))
    assert_reproduces(POOL, "bird_DS_G_minus_Nonempty_oracle1_detected", int(g["present"]["detected"]))
    assert_reproduces(POOL, "bird_oracle0_questions", g["absent"]["questions"])
    assert_reproduces(POOL, "bird_oracle1_questions", g["present"]["questions"])
    assert_reproduces(POOL, "bird_oracle0_DS_G_correct", g["absent"]["method_correct"])
    assert_reproduces(POOL, "bird_DS_G_minus_Nonempty_oracle1_gained", g["present"]["gained"])
    s = availability.method_gain_by_availability("DS_G", "DS_S_al")
    assert_reproduces(POOL, "bird_DS_G_minus_DS_S_al_oracle1", r6(s["present"]["est"]))
    assert_reproduces(POOL, "bird_DS_G_minus_DS_S_al_oracle1_lost", s["present"]["lost"])
    q3 = availability.method_gain_by_availability("Q3_G", "Nonempty")
    assert_reproduces(POOL, "bird_oracle0_Q3_G_correct", q3["absent"]["method_correct"])
    sp = availability.method_gain_by_availability("DS_G", "Nonempty", benchmark="spider")
    assert_reproduces(POOL, "spider_oracle0_DS_G_correct", sp["absent"]["method_correct"])
    assert_reproduces(POOL, "spider_DS_G_minus_Nonempty_oracle0", r6(sp["absent"]["est"]))


# ------------------------------------------------------------------ answer changes
def test_transitions_split():
    t = changes.transitions({1: 1, 2: 1, 3: 0, 4: 0}, {1: 1, 2: 0, 3: 1, 4: 0, 5: 1})
    assert t["n"] == 4 and t["recovered_pp"] == 25.0 and t["lost_pp"] == 25.0 and t["net_pp"] == 0.0


def test_revision_changes():
    rev = changes.revision_changes()
    assert_reproduces(FLOW, "loss_mean6", r4(rev["mean"]["lost_pp"]))
    assert_reproduces(FLOW, "recovery_mean6", r4(rev["mean"]["recovered_pp"]))
    assert_reproduces(FLOW, "revision_net_mean6", r4(-rev["mean"]["net_pp"]))
    for (p, b), t in rev["per_configuration"].items():
        assert_reproduces(FLOW, f"loss_{TAG[p]}_{TAG[b]}", r4(t["lost_pp"]))
        assert_reproduces(FLOW, f"recovery_{TAG[p]}_{TAG[b]}", r4(t["recovered_pp"]))


def test_rule_changes():
    joint = changes.rule_changes("lenient_execution_first")
    assert_reproduces(FLOW, "runs_four_configurations", joint["runs"])
    assert_reproduces(FLOW, "joint_corrections_runs_mean4", joint["corrections"])
    assert_reproduces(FLOW, "joint_regressions_runs_mean4", joint["regressions"])
    last = changes.rule_changes("protocol_exact_last")
    assert_reproduces(FLOW, "tie_last_corrections_runs_mean4", last["corrections"])
    assert_reproduces(FLOW, "tie_last_regressions_runs_mean4", last["regressions"])


def test_peer_changes():
    name = {"qwen3-32b": "qwen3", "gpt-oss-120b": "gptoss", "gemma-4-26b": "gemma"}
    falls = 0
    for m in changes.RECIPIENTS:
        for shared, s in (("full", "full"), ("answer_only", "answers")):
            c = changes.peer_source_change(m, shared)
            assert_reproduces(PEER, f"chg_{name[m]}_{s}_drec", c["drec"])
            assert_reproduces(PEER, f"chg_{name[m]}_{s}_dlost", c["dlost"])
            assert_reproduces(PEER, f"chg_{name[m]}_{s}_drec_exact", c["drec_exact"])
            assert_reproduces(PEER, f"chg_{name[m]}_{s}_dlost_exact", c["dlost_exact"])
            falls += c["regression_rate_falls"]
    assert_reproduces(PEER, "regression_rate_falls_combinations", falls)
    for shared in ("answer_only", "full"):
        w = changes.weaker_recipient_changes(shared)
        assert_reproduces(PEER, f"w_{shared}_recovered", w["recovered_pp"])
        assert_reproduces(PEER, f"w_{shared}_lost", w["lost_pp"])
    assert_reproduces(PEER, "w_acc_all_no_peers", w["q_pct"])
