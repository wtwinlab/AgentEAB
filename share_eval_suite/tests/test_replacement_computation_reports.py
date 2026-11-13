"""Stage replacements, added computation, intervals, cost and quality reports reproduce the paper's values.

fig_stage-act and tab_replacement-cost ship values rounded to six decimals, so those comparisons
round the suite's result the same way first.
"""
from share_eval_suite.comparisons import computation as C
from share_eval_suite.comparisons import replacement as R
from share_eval_suite.reports import cost, intervals, quality
from share_eval_suite.tests._expect import assert_reproduces


def r6(x):
    return round(float(x), 6)


# ---------------------------------------------------------------- Eq. 4 decomposition (Fig. 6)
def test_refiner_decomposition_bird():
    d = R.stage_decomposition("refiner", "qwen3-32b", "bird")
    assert_reproduces("fig_stage-act", "refiner_q3_bird_n", d.n)
    assert_reproduces("fig_stage-act", "refiner_q3_bird_n_A", d.n_changed)
    assert_reproduces("fig_stage-act", "refiner_q3_bird_P_A", r6(d.change_rate.estimate))
    assert_reproduces("fig_stage-act", "refiner_q3_bird_gains", d.gains)
    assert_reproduces("fig_stage-act", "refiner_q3_bird_losses", d.losses)
    assert_reproduces("fig_stage-act", "refiner_q3_bird_both_wrong", d.both_wrong)
    assert_reproduces("fig_stage-act", "refiner_q3_bird_E_D_A", r6(d.gain_if_changed))
    assert_reproduces("fig_stage-act", "refiner_q3_bird_total", r6(d.total.estimate))
    assert_reproduces("fig_stage-act", "refiner_q3_bird_total_lo", r6(d.total.lo))
    assert_reproduces("fig_stage-act", "refiner_q3_bird_total_hi", r6(d.total.hi))
    d72 = R.stage_decomposition("refiner", "qwen2.5-72b", "bird")
    assert_reproduces("fig_stage-act", "refiner_q72_bird_P_A", r6(d72.change_rate.estimate))
    assert_reproduces("fig_stage-act", "refiner_q72_bird_E_D_A", r6(d72.gain_if_changed))
    # the identity mean D = P(A) * E[D | A]
    assert abs(d.total.estimate - d.change_rate.estimate * d.gain_if_changed / 100) < 1e-9


def test_unit_tester_and_selector_decomposition():
    ut = R.stage_decomposition("unit_tester")
    assert_reproduces("fig_stage-act", "ut_sc_P_A", r6(ut.change_rate.estimate))
    assert_reproduces("fig_stage-act", "ut_sc_gains", ut.gains)
    assert_reproduces("fig_stage-act", "ut_sc_losses", ut.losses)
    assert ut.gain_if_changed_interval is None          # 23 changed tasks, below 30
    sel = R.stage_decomposition("selector", "qwen3-32b", None)
    assert_reproduces("fig_stage-act", "selector_q3_total", r6(sel.total.estimate))
    assert_reproduces("fig_stage-act", "selector_q3_E_D_A", r6(sel.gain_if_changed))
    sel7 = R.stage_decomposition("selector", "qwen2.5-7b", None)
    assert_reproduces("fig_stage-act", "selector_q7_total_lo", r6(sel7.total.lo))
    assert_reproduces("fig_stage-act", "selector_q7_E_D_A_lo", r6(sel7.gain_if_changed_interval[0]))
    sp = R.stage_decomposition("refiner", "qwen2.5-72b", "spider")
    assert_reproduces("fig_stage-act", "refiner_q72_spider_P_A", r6(sp.change_rate.estimate))


def test_stage_share():
    s = R.stage_share("refiner", "qwen3-32b", "bird")
    assert_reproduces("fig_stage-act", "refiner_q3_bird_call_share", r6(s["calls"]))
    assert_reproduces("fig_stage-act", "refiner_q3_bird_token_share", r6(s["tokens"]))
    u = R.stage_share("unit_tester")
    assert_reproduces("fig_stage-act", "ut_sc_call_share", r6(u["calls"]))
    assert_reproduces("fig_stage-act", "ut_sc_token_share", r6(u["tokens"]))


# ---------------------------------------------------------------- Table VII
def test_replacement_cost_refiner_bird():
    row = R.replacement_cost("refiner", "qwen3-32b", "bird")
    assert_reproduces("tab_replacement-cost", "refiner_q3_bird_delta", r6(row.change.estimate))      # +1.83
    assert_reproduces("tab_replacement-cost", "refiner_q3_bird_delta_lo", r6(row.change.lo))         # 1.00
    assert_reproduces("tab_replacement-cost", "refiner_q3_bird_delta_hi", r6(row.change.hi))         # 2.70
    assert_reproduces("tab_replacement-cost", "refiner_q3_bird_tasks", row.tasks)
    assert_reproduces("tab_replacement-cost", "refiner_q3_bird_calls", row.calls)
    assert_reproduces("tab_replacement-cost", "refiner_q3_bird_tok_in_k", r6(row.tokens_in / 1e3))
    assert_reproduces("tab_replacement-cost", "refiner_q3_bird_tok_out_k", r6(row.tokens_out / 1e3))
    assert_reproduces("tab_replacement-cost", "refiner_q3_bird_gpu_share", r6(row.gpu_seconds_share))
    assert_reproduces("tab_replacement-cost", "refiner_q3_bird_per_million_tokens",
                      r6(row.extra_correct_per_million_tokens))                                       # 46.5


def test_replacement_cost_other_rows():
    r72 = R.replacement_cost("refiner", "qwen2.5-72b", "bird")
    assert_reproduces("tab_replacement-cost", "refiner_q72_bird_delta", r6(r72.change.estimate))     # +3.46
    assert_reproduces("tab_replacement-cost", "refiner_q72_bird_gpu_share", r6(r72.gpu_seconds_share))
    assert_reproduces("tab_replacement-cost", "refiner_q72_bird_per_million_tokens",
                      r6(r72.extra_correct_per_million_tokens))                                       # 72.4
    ut = R.replacement_cost("unit_tester")
    assert_reproduces("tab_replacement-cost", "ut_sc_delta", r6(ut.change.estimate))                 # -1.36
    assert_reproduces("tab_replacement-cost", "ut_sc_delta_lo", r6(ut.change.lo))
    assert_reproduces("tab_replacement-cost", "ut_sc_delta_hi", r6(ut.change.hi))
    assert_reproduces("tab_replacement-cost", "chess_ut_calls", ut.calls)                             # 2,599
    assert_reproduces("tab_replacement-cost", "chess_ut_tok_in_M", r6(ut.tokens_in / 1e6))
    assert_reproduces("tab_replacement-cost", "ut_gpu_share", r6(ut.gpu_seconds_share))              # 51.9
    sel = R.replacement_cost("selector", "qwen2.5-7b", None)
    assert_reproduces("tab_replacement-cost", "selector_q7_delta", r6(sel.change.estimate))          # +1.31
    assert_reproduces("tab_replacement-cost", "selector_q7_delta_lo", r6(sel.change.lo))             # -0.44
    assert_reproduces("tab_replacement-cost", "selector_q7_tok_in_k", r6(sel.tokens_in / 1e3))


def test_unit_tester_against_votes():
    r = R.stage_minus_rule("unit_tester", rule="nonempty", weighting="database", population="entered")
    assert_reproduces("tab_replacement-cost", "chess_UT_minus_Nonempty_vote_entered_db", r6(r.change.estimate))
    assert_reproduces("tab_replacement-cost", "chess_UT_minus_Nonempty_vote_entered_db_lo", r6(r.change.lo))
    a = quality.unit_tester_answers()
    assert_reproduces("tab_replacement-cost", "chess_UT_correct", a["unit_tester_correct"])           # 83
    assert_reproduces("tab_replacement-cost", "chess_Nonempty_vote_correct", a["nonempty_correct"])   # 88
    assert_reproduces("tab_replacement-cost", "chess_same_result_SC_official", a["self_consistency_same_result"])
    assert_reproduces("tab_replacement-cost", "chess_ut_entered", a["unit_tester_entered"])


# ---------------------------------------------------------------- cost report
def test_cost_report():
    c = cost.unit_tester_calls()
    assert_reproduces("tab_replacement-cost", "chess_all_calls", c["all_calls"])                     # 7,001
    assert_reproduces("tab_replacement-cost", "chess_ut_call_share", r6(c["call_share"]))            # 37.1
    assert_reproduces("tab_replacement-cost", "chess_ut_completion_share", r6(c["completion_share"]))
    assert_reproduces("tab_replacement-cost", "chess_ut_evaluation_calls", c["evaluation_calls"])
    rates = cost.decoding_rates()
    assert_reproduces("tab_replacement-cost", "rate_Qwen3-32B_decode_single_min", rates["Qwen3-32B"]["min"])
    assert_reproduces("tab_replacement-cost", "rate_Qwen2.5-72B-AWQ_decode_single_max", rates["Qwen2.5-72B-AWQ"]["max"])
    g = cost.stage_gpu_seconds("refiner", "qwen2.5-72b", "spider")
    assert_reproduces("tab_replacement-cost", "refiner_q72_spider_gpu_s", g.point)
    assert_reproduces("tab_replacement-cost", "refiner_q72_spider_gpu_share", r6(g.share))
    d = cost.debate_cost("mad-m2", "qwen2.5-7b", "3x4")
    assert_reproduces("fig_debate-budget", "madm2_qwen_3x4_calls", d["calls"])
    assert_reproduces("fig_debate-budget", "madm2_qwen_3x4_tokens", d["tokens"])


def test_quality_tables():
    rows = quality.replacement_table()
    assert len(rows) == 7
    lab, sel = rows[5]
    assert lab.startswith("ComEM selector, Qwen3-32B")
    assert_reproduces("tab_replacement-cost", "selector_q3_delta", r6(sel.change.estimate))          # +5.19
    dec = quality.decomposition_table()
    assert_reproduces("fig_stage-act", "refiner_q3_spider_P_A", r6(dec[2][1].change_rate.estimate))


# ---------------------------------------------------------------- execution filters on fixed SQL pools
def test_execution_filters():
    g = R.execution_filter_contrast("bird", "pool4", "generation", "unfiltered")
    assert_reproduces("tab_sql-rule-strength", "bird_pool4_32b_gen_minus_unfiltered", g.estimate)    # +8.24
    assert_reproduces("tab_sql-rule-strength", "bird_pool4_32b_gen_minus_unfiltered_lo", g.lo)
    assert_reproduces("tab_sql-rule-strength", "bird_pool4_32b_gen_minus_unfiltered_hi", g.hi)
    n = R.execution_filter_contrast("bird", "pool6", "generation", "nonempty")
    assert_reproduces("tab_sql-rule-strength", "bird_pool6_32b_gen_minus_nonempty", n.estimate)      # -0.04
    s = R.execution_filter_contrast("spider", "pool4", "nonempty", "valid")
    assert_reproduces("tab_sql-rule-strength", "spider_pool4_nonempty_minus_valid_all", s.estimate)  # +0.48
    assert_reproduces("tab_sql-rule-strength", "spider_pool4_nonempty_minus_valid_all_lo", s.lo)
    e = R.execution_filter_contrast("spider", "pool4", "nonempty", "valid", stratum="gold_empty")
    assert_reproduces("tab_sql-rule-strength", "spider_pool4_nonempty_minus_valid_gold_empty", e.estimate)
    q7 = R.execution_filter_contrast("bird", "pool4", "generation", "nonempty", recipient="qwen2.5-7b")
    assert_reproduces("tab_sql-rule-strength", "bird_pool4_7b_gen_minus_nonempty", q7.estimate)      # -3.87
    b = R.empty_answer_break_even("spider", "pool6")
    assert_reproduces("tab_sql-rule-strength", "spider_pool6_nonempty_break_even_share", b["break_even_share"])
    assert_reproduces("tab_sql-rule-strength", "spider_pool4_empty_reference_share",
                      R.empty_answer_break_even("spider", "pool4")["empty_answer_share"])


# ---------------------------------------------------------------- budget-matched debates (Fig. 8)
def test_fixed_response_votes():
    v = C.fixed_response_votes()
    assert_reproduces("fig_debate-budget", "fixed_exec_minus_final", v["execution_vote_minus_output"].estimate)  # 10.10
    assert_reproduces("fig_debate-budget", "fixed_string_minus_final", v["exact_match_vote_minus_output"].estimate)  # -3.28
    last = v["execution_vote_minus_last_round"]
    assert_reproduces("fig_debate-budget", "fixed_exec_minus_last", last.estimate)                   # +1.33
    assert_reproduces("fig_debate-budget", "fixed_exec_minus_last_q_lo", last.question.lo)
    assert_reproduces("fig_debate-budget", "fixed_exec_minus_last_db_hi", last.database.hi)


def test_budget_matched_debates():
    six = C.debate_minus_vote("mad-m2", "3x2")
    assert_reproduces("fig_debate-budget", "call_madm2_3x2_protocol", six.estimate)                  # vote leads by 3.91
    assert_reproduces("fig_debate-budget", "call_madm2_3x2_protocol_q_lo", six.question.lo)
    lead = [-C.debate_minus_vote("mad-m2", c).estimate for c in ("3x3", "5x2", "3x4")]
    assert_reproduces("fig_debate-budget", "madm2_call_lead_9to12_min", min(lead))                   # 3.91
    assert_reproduces("fig_debate-budget", "madm2_call_lead_9to12_max", max(lead))                   # 5.56
    assert_reproduces("fig_debate-budget", "madm2_3x3_minus_base", C.larger_debate_minus_base("mad-m2", "3x3").estimate)
    ks = [C.matched_draft_count("mad-m2", b, c) for b in C.BACKBONES for c in C.BUDGETS]
    assert_reproduces("fig_debate-budget", "madm2_kstar_min", min(ks))
    acc = C.budget_accuracy("mad-m2", "qwen2.5-7b")
    assert_reproduces("fig_debate-budget", "madm2_qwen_3x4_last_acc", acc[12]["debate"])
    assert_reproduces("fig_debate-budget", "madm2_qwen_protocol_vote12_acc", acc[12]["vote"])


# ---------------------------------------------------------------- backbone scale (Table X)
def test_backbone_scale():
    assert_reproduces("tab_scale", "dpeer_mean_14B_est", C.interaction_by_size("mean", "14B").estimate)  # -3.28
    assert_reproduces("tab_scale", "dpeer_madm2_7B_est", C.interaction_by_size("mad-m2", "7B").estimate)
    assert_reproduces("tab_scale", "dpeer_bccs_14B_est", C.interaction_by_size("bccs", "14B").estimate)
    con = C.scale_contrasts()
    assert_reproduces("tab_scale", "contrast_72b_7b_est", con["72b_7b"]["contrast"].estimate)       # +0.51
    assert_reproduces("tab_scale", "contrast_72b_7b_q_hi", con["72b_7b"]["contrast"].question.hi)   # +3.41
    assert_reproduces("tab_scale", "contrast_72b_14b_est", con["72b_14b"]["contrast"].estimate)     # +3.79
    assert_reproduces("tab_scale", "contrast_72b_14b_p_holm", con["72b_14b"]["p_holm"])
    eq = C.scale_equivalence()
    assert_reproduces("tab_scale", "equiv_madm2_72B_ci90_lo", eq["ci90_lo"])
    assert_reproduces("tab_scale", "equiv_madm2_72B_repeat1", eq["repeat1"])


# ---------------------------------------------------------------- stronger models (Table IX)
def test_stronger_models():
    ds = C.stronger_model_contrasts("deepseek-v4-pro")
    assert_reproduces("tab_strength", "ds_bird_DS_Generation_minus_Nonempty", ds["generation_minus_nonempty"]["contrast"].estimate)
    assert_reproduces("tab_strength", "ds_bird_DS_Generation_minus_S_al", ds["generation_minus_selector"]["contrast"].estimate)
    assert_reproduces("tab_strength", "ds_bird_DS_S_al_minus_Nonempty_lo", ds["selector_minus_nonempty"]["contrast"].lo)
    assert_reproduces("tab_strength", "ds_bird_DS_Generation_minus_Nonempty_holm", ds["generation_minus_nonempty"]["holm"])
    q3 = C.stronger_model_contrasts("qwen3-32b")
    assert_reproduces("tab_strength", "ds_bird_Q3_Generation_minus_S_al_hi", q3["generation_minus_selector"]["contrast"].hi)
    assert_reproduces("tab_strength", "ds_bird_level_DS_Generation",
                      C.stronger_model_accuracy("deepseek-v4-pro", "generation").estimate)          # 63.14
    assert_reproduces("tab_strength", "did_bird_Generation_minus_Nonempty",
                      C.stronger_model_gain_difference("generation", "nonempty").estimate)          # +5.26


# ---------------------------------------------------------------- recipient families (Fig. 7)
def test_family_and_feedback():
    g = C.generation_minus_vote("spider", "gemma-4-26b", "nonempty")
    assert_reproduces("fig_sql-family", "spider_gemma_gen_minus_nonempty", g["contrast"].estimate)  # 3.44
    assert_reproduces("fig_sql-family", "spider_gemma_gen_minus_nonempty_holm_joint", g["holm"])
    f, n = C.execution_feedback()
    assert_reproduces("fig_sql-family", "bird_qwen25_7b_feedback_minus_queries_only", f.estimate)   # 3.66
    assert_reproduces("fig_sql-family", "bird_qwen25_7b_feedback_minus_queries_only_lo", f.lo)
    assert_reproduces("fig_sql-family", "bird_feedback_paired_questions", n)                        # 1,532


# ---------------------------------------------------------------- intervals: shared estimators give the same draws
def test_intervals_match_object_bootstraps():
    units = R.stage_units("refiner", "qwen3-32b", "bird")
    D = {k: ys - ya for k, _, _, ys, ya in units}
    cl = {k: c for k, c, _, _, _ in units}
    iv = intervals.cluster_bootstrap(D, cl).scaled(100)
    assert_reproduces("fig_stage-act", "refiner_q3_bird_total_lo", r6(iv.lo))
    assert_reproduces("fig_stage-act", "refiner_q3_bird_total_hi", r6(iv.hi))
    sel = R.stage_units("selector", "qwen3-32b", None)
    st = intervals.stratified_bootstrap({k: ys - ya for k, _, _, ys, ya in sel}, {k: c for k, c, _, _, _ in sel}).scaled(100)
    assert_reproduces("fig_stage-act", "selector_q3_total_lo", r6(st.lo))
    assert_reproduces("fig_stage-act", "selector_q3_total_hi", r6(st.hi))
    rows = R._read("tab_replacement-cost", "chess_questions.csv", None)
    db = intervals.cluster_bootstrap({int(r["question_id"]): int(r["UT"]) - int(r["SC_official"]) for r in rows},
                                     {int(r["question_id"]): r["db_id"] for r in rows}, weighting="database").scaled(100)
    assert_reproduces("tab_replacement-cost", "chess_UT_minus_SC_official_all_db", r6(db.estimate))
    two, _ = intervals.question_and_database_bootstrap(*_scale_inputs())
    assert_reproduces("tab_scale", "dpeer_mean_14B_q_lo", two.question.lo)
    assert_reproduces("tab_scale", "dpeer_mean_14B_db_hi", two.database.hi)


def _scale_inputs():
    _, db, dp = C._scale(str(C._inp("tab_scale", None)))
    return dp[("qwen25-14b-awq", "mean")], db
