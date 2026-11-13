"""Peer effects, joint stage choices and dependency checks reproduce the shipped paper values."""
import pytest

from share_eval_suite.comparisons import dependency, interactions, peers
from share_eval_suite.tests._expect import assert_reproduces, expected

F, RC, PC, AC, SC, MX = ("tab_factorial", "fig_readout-cells", "tab_peer-controls", "tab_answer-content",
                         "tab_s-controls", "tab_matrix")
ON, CEN = "tab_s-online", "tab_sensitivity-census"


def six(obj, key, got):
    """Compare at the six decimals the object writes."""
    assert_reproduces(obj, key, round(float(got), 6))


def interval(obj, key, b, est="est", lo="q_lo", hi="q_hi", suffix=("_q_lo", "_q_hi"), est_suffix=""):
    assert_reproduces(obj, key + est_suffix, b[est])
    assert_reproduces(obj, key + suffix[0], b[lo])
    assert_reproduces(obj, key + suffix[1], b[hi])


# ------------------------------------------------------------------ peers: the four-cell headline
@pytest.mark.parametrize("ext,vote,key", [("protocol", "exact", "rb_dpeer_ref_mean4"),
                                          ("lenient", "execution", "rb_dpeer_EI_mean4")])
def test_headline_peer_effect_replay(ext, vote, key):
    b = peers.peer_effect_replay(ext, vote, "first", "mean4")
    for f in ("est", "lo", "hi", "db_lo", "db_hi", "bound_lo", "bound_hi"):
        six(F, f"{key}_{f}", b[f])


def test_headline_values():
    assert round(peers.peer_effect_replay("protocol", "exact", "first", "mean4")["est"], 2) == 3.03
    assert round(peers.peer_effect_replay("lenient", "execution", "first", "mean4")["est"], 2) == -1.96


@pytest.mark.parametrize("ext,vote,tie", [("protocol", "exact", "first"), ("lenient", "execution", "first"),
                                          ("protocol", "exact", "last"), ("lenient", "execution", "random")])
def test_peer_effect_198(ext, vote, tie):
    b = peers.peer_effect(ext, vote, tie, "both")
    key = f"s198|4cell|{peers.EXTRACTORS[ext]}|{peers.VOTES[vote]}|{tie}|dpeer"
    interval(PC, key, b)
    assert_reproduces(PC, key + "_db_lo", b["db_lo"])
    assert_reproduces(PC, key + "_db_hi", b["db_hi"])


def test_peer_effect_du_mad_198():
    assert_reproduces(PC, "s198|DuMAD_2cell|lenient|exec|first|dpeer", peers.peer_effect("lenient", "execution", configurations="du_mad")["est"])


def test_peer_effect_by_configuration():
    c = peers.peer_effect_by_configuration("first")
    for cell, tag in (("mean", "mean"), ("madm2_objective|qwen2.5-7b", "madm2_objective_qwen257b"),
                      ("du_mad|qwen2.5-7b", "du_mad_qwen257b")):
        for reading, (est, lo, hi) in c[cell].items():
            key = f"{tag}_{reading.replace('|', '_')}_first"
            six(RC, key, est)
            six(RC, key + "_lo", lo)
            six(RC, key + "_hi", hi)
    six(F, "dpeer_madm2_qwen_strict_string_first", c["madm2_objective|qwen2.5-7b"]["strict|string"][0])


def test_levels_by_configuration():
    lv = peers.levels_by_configuration()
    six(F, "level_du_mad_llama38b_lenient_string_last_full", lv["du_mad|llama3-8b"]["lenient_exact_last"])
    six(F, "level_du_mad_llama38b_lenient_exec_r0_full", lv["du_mad|llama3-8b"]["lenient_execution_initial"])
    six(F, "level_madm2_objective_qwen257b_lenient_exec_last_full", lv["madm2_objective|qwen2.5-7b"]["lenient_execution_last"])
    six(F, "level_madm2_objective_llama38b_final_full", lv["madm2_objective|llama3-8b"]["final"])


def test_peer_effect_full_dev():
    b = peers.peer_effect_full_dev("both", "first")
    interval(PC, "4cell|dpeer|first", b)
    assert_reproduces(PC, "4cell|dpeer|first_db_lo", b["db_lo"])
    assert_reproduces(PC, "4cell|dpeer|first_db_hi", b["db_hi"])
    assert_reproduces(PC, "DuMAD_2cell|dpeer|first", peers.peer_effect_full_dev("du_mad")["est"])
    assert_reproduces(PC, "4cell|dpeer|first|1333", peers.peer_effect_full_dev(questions="remaining")["est"])
    assert_reproduces(PC, "DuMAD_2cell|last_diff", peers.difference_full_dev("du_mad", "last")["est"])
    assert_reproduces(PC, "4cell|r0_diff", peers.difference_full_dev("both", "initial")["est"])
    lv = peers.levels_full_dev("du_mad")
    assert_reproduces(PC, "DuMAD_2cell|last_without", lv["last_without"])
    assert_reproduces(PC, "DuMAD_2cell|r0_with", lv["initial_with"])


def test_answer_only_peers():
    b = peers.answer_only_contrast("ANSWER_ONLY", "FULL", "change", "both")
    interval(AC, "E1|4cell|ANSWER_ONLY-FULL", b)
    assert_reproduces(AC, "E1|4cell|ANSWER_ONLY-NOPEER", peers.answer_only_contrast("ANSWER_ONLY", "NOPEER")["est"])
    assert_reproduces(AC, "E1|madm2_objective|qwen2.5-7b|ANSWER_ONLY-FULL",
                      peers.answer_only_contrast(configurations="madm2_objective|qwen2.5-7b")["est"])
    lr = peers.agent_loss_recovery("madm2_objective", "qwen2.5-7b", "ANSWER_ONLY")
    assert_reproduces(AC, "cond_loss|madm2_objective|qwen2.5-7b|ANSWER_ONLY", lr["loss"])
    assert_reproduces(AC, "cond_rec|madm2_objective|qwen2.5-7b|ANSWER_ONLY", lr["recovery"])


def test_peer_controls():
    c = peers.peer_controls()
    for name, key in (("remaining_1333", "1333|4cell|dpeer"), ("placebo_real_minus_placebo", "placebo|4cell|real_minus_placebo"),
                      ("placebo_minus_no_peers", "placebo|4cell|placebo_minus_nopeer"),
                      ("temperature_0.7", "t07|DuMAD_2cell|dpeer"), ("format_change", "fmt|4cell|dpeer_change")):
        interval(SC, key, c[name])
        assert_reproduces(SC, key + "_db_lo", c[name]["db_lo"])
        assert_reproduces(SC, key + "_db_hi", c[name]["db_hi"])


def test_bird_populations():
    for k, v in peers.bird_populations().items():
        assert_reproduces(MX, k, v)


# ------------------------------------------------------------------ interactions
def test_shared_reference_ledger():
    L = interactions.shared_reference_ledger("six")
    six(F, "published_level_mean6", L["reference_level"])
    assert L["n_questions"] == expected(F, "n_questions")
    six(F, "single_lenient", L["rows"]["E"]["joint"])
    six(F, "single_exec_vote", L["rows"]["I"]["joint"])
    six(F, "single_nopeer_lo", L["rows"]["A"]["joint_lo"])
    six(F, "joint_EI", L["rows"]["E+I"]["joint"])
    six(F, "nonadd_EI", L["rows"]["E+I"]["joint_minus_sum"])
    six(F, "nonadd_EI_lo", L["rows"]["E+I"]["joint_minus_sum_lo"])
    six(F, "nonadd_EI_hi", L["rows"]["E+I"]["joint_minus_sum_hi"])
    six(F, "joint_EIR", L["rows"]["E+I+R"]["joint"])
    six(F, "sum_EIR", L["rows"]["E+I+R"]["sum_of_singles"])


def test_replay_ledger():
    L = interactions.replay_ledger("mean4")
    six(F, "rb_E_mean4_est", L["E"]["est"])
    six(F, "rb_I_mean4_est", L["I"]["est"])
    for f in ("est", "lo", "hi", "db_lo", "db_hi"):
        six(F, f"rb_EI_mean4_{f}", L["E+I"][f])
        six(F, f"rb_nonadd_EI_mean4_{f}", L["E+I joint minus sum"][f])
        six(F, f"rb_nonadd_EIR_mean4_{f}", L["E+I+R joint minus sum"][f])
    six(F, "rb_nonadd_EI_mean6_db_lo", interactions.replay_ledger("mean6")["E+I joint minus sum"]["db_lo"])


def test_factorial_round_by_peers_is_mean_peer_effect():
    # round x peers interaction = mean over the four readings of the peer effect (six configurations)
    _, Y, _ = interactions.outcome_array("six")
    fx = interactions.factorial_effects(Y)
    mean4 = sum(expected(RC, f"mean_{r}_first") for r in ("strict_string", "strict_exec", "lenient_string", "lenient_exec")) / 4
    assert fx["RxA"]["est"] == round(mean4, 2)
    assert set(fx) >= {"E", "I", "R", "A", "ExI", "ExIxRxA"}


# ------------------------------------------------------------------ dependency checks
def test_replay_bounds_six_configurations():
    b = dependency.replay_bound("EI", "mean6")
    six(F, "rb_EI_mean6_est", b["est"])
    six(F, "rb_EI_mean6_bound_lo", b["bound_lo"])
    six(F, "rb_EI_mean6_bound_hi", b["bound_hi"])
    six(F, "rb_EI_mean6_bound_ci_lo", b["bound_ci_lo"])
    six(F, "rb_EI_mean6_bound_ci_hi", b["bound_ci_hi"])
    d = dependency.replay_bound("dpeer_EI", "mean6")
    six(F, "rb_dpeer_EI_mean6_bound_lo", d["bound_lo"])
    six(F, "rb_dpeer_EI_mean6_bound_hi", d["bound_hi"])
    six(F, "rb_ref_minus_final_bccs_llama38b_est", dependency.replay_bound("ref_minus_final", "bccs|llama3-8b")["est"])


def test_run_interval_agrees_with_object():
    runs, div, _ = dependency.load_scored_runs()
    from share_eval_suite import _repo
    m = _repo.paper_object(F)
    for r in runs[:400] + [x for x in runs if x["protocol"] == "bccs"][:400]:
        for reading in ("ref", "EI", "ref_tl"):
            for rnd in ("last", "r0", "final"):
                assert dependency.run_interval(r, div, reading, rnd) == m.interval(r, div, reading, rnd)


def test_divergence_and_read_checks():
    s = dependency.divergence_shares()
    six(F, "diverge_llama38b_FULL_EI", s[("llama3-8b", "FULL", "EI")]["P_D"])
    six(F, "diverge_unrec_qwen257b_FULL_EI", s[("qwen2.5-7b", "FULL", "EI")]["P_U"])
    six(F, "diverge_tasks_cp_lo_llama38b_FULL_ref", s[("llama3-8b", "FULL", "ref")]["cp_lo"])
    rc = dependency.read_checks()
    for k in ("verdicts", "stop_flags", "peer_choices"):
        assert_reproduces(F, f"bccs_{k}_reproduced", rc[f"{k}_reproduced"])
    assert_reproduces(F, "bccs_reads_not_reproduced", rc["not_reproduced"])


def test_online_check():
    for tie in ("first", "last", "random"):
        b = dependency.online_peer_effect(tie)
        interval(ON, f"dpeer_tie_{tie}", b, est_suffix="_est")
    b = dependency.online_peer_effect("first")
    assert_reproduces(ON, "dpeer_tie_first_db_lo", b["db_lo"])
    assert_reproduces(ON, "dpeer_tie_first_db_hi", b["db_hi"])
    interval(ON, "dpeer_excl_single", dependency.online_peer_effect("first", True), est_suffix="_est")
    for cond, tag in (("with_peers", "peers"), ("no_peers", "nopeers")):
        o = dependency.online_minus_replay(cond)
        interval(ON, f"online_minus_replay_{tag}", o["overall"], est_suffix="_est")
        assert_reproduces(ON, f"diverging_runs_{tag}_pct", o["diverging_runs_pct"])
        assert_reproduces(ON, f"run_disagreement_{tag}_pct", o["run_disagreement_pct"])
    rec = dependency.online_minus_replay("no_peers")["by_divergence"]["recoverable_only"]
    assert_reproduces(ON, "online_minus_replay_nopeers_recoverable_only_n", rec["n"])
    interval(ON, "online_minus_replay_nopeers_recoverable_only", rec, est_suffix="_est")


def test_sensitivity_census():
    c = dependency.sensitivity_census()
    for tag, d in c["families"].items():
        for f, v in d.items():
            assert_reproduces(CEN, f"{tag}_{f}", v)
    for f, v in c["control"].items():
        assert_reproduces(CEN, f"control_{f}", v)
    for f, v in c["grouping"].items():
        assert_reproduces(CEN, f"grouping_{f}", v)
    assert_reproduces(CEN, "measurement_changed", c["measurement_changed"])
    assert_reproduces(CEN, "measurement_eligible", c["measurement_eligible"])
