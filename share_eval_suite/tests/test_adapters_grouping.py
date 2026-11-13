"""Protocol and task adapters, matching downstream of the final decision, and the command line."""
import csv
import math
import pathlib

import pytest

from share_eval_suite import _repo, cli
from share_eval_suite.adapters import protocols as P
from share_eval_suite.adapters import tasks as T
from share_eval_suite.stages import grouping as G
from share_eval_suite.tests._expect import assert_reproduces

MERGE, RULES, CONTENT, RETURN = G.MERGE, G.RULES, G.CONTENT, G.RETURN_RULE


# ------------------------------------------------------------------ protocol registry
def test_registry_covers_every_protocol_adapter():
    shared = {"__init__", "_runner", "official_scorers", "native_run", "sources", "freeze_official_data"}
    adapters = {p.stem for p in (_repo.REPO / "code/consensus_delivery/adapters").glob("*.py")} - shared
    listed = {pathlib.Path(f).stem for p in P.list_protocols() for f in p.files}
    assert adapters <= listed
    for p in P.list_protocols():
        assert set(p.stages) <= set(P.STAGES) and P.GENERATION in p.stages
        assert all((_repo.REPO / f).is_file() for f in p.files), p.name


def test_registry_names_the_papers_workflows():
    paper = {p.name for p in P.list_protocols(in_paper=True)}
    assert {"Du-MAD", "MAD-M2", "BCCS", "MoA", "FORD"} <= paper
    assert {p.name for p in P.list_protocols(kind=P.PIPELINE)} == {"CHESS", "MAC-SQL", "ComEM"}
    assert P.get_protocol("mad_m2").name == "MAD-M2" and P.get_protocol("comem").kind == P.PIPELINE
    with pytest.raises(KeyError):
        P.get_protocol("no-such-protocol")


@pytest.mark.parametrize("name,attr", [("Du-MAD", "most_frequent"), ("MAD-M2", "if_reach_consensus"),
                                       ("BCCS", "estimate_consensus"), ("FORD", "parse_prediction"),
                                       ("CHESS", "OPS")])
def test_load_returns_the_existing_module(name, attr):
    assert hasattr(P.load(name), attr)


def test_load_fails_with_a_clear_message():
    with pytest.raises(P.ProtocolUnavailable, match="MAC-SQL"):
        P.load("MAC-SQL")
    if not (_repo.REPO / "vendor/consensus_literature/moa").exists():
        with pytest.raises(P.ProtocolUnavailable, match="vendor/consensus_literature/moa"):
            P.load("MoA")


# ------------------------------------------------------------------ question answering
def test_qa_scorers():
    assert T.qa_question("gsm8k", "How many?") == "How many?"
    assert T.qa_question("squad_v2", "Who?", "Passage.") == "Passage.\n\nQuestion: Who?"
    assert T.qa_answer("so the total is 1,234. Answer: 370", "numeric") == "370.0"
    assert T.qa_answer("nothing to read", "numeric") is None
    assert T.qa_correct("gsm8k", "Answer: 370", "steps #### 370") == 1
    assert T.qa_correct("gsm8k", "Answer: 37", "steps #### 370") == 0
    assert T.qa_correct("mmlu_pro", "Answer: (C)", "C", n_options=4) == 1
    assert T.qa_correct("mmlu_pro", "Answer: (E)", "E", n_options=4) == 0        # out of range
    assert T.qa_correct("strategyqa", "Answer: No", "false") == 1
    assert T.qa_correct("gsm8k", "", "#### 1") == 0
    assert T.qa_correct("squad_v2", "Answer: the  Eiffel Tower", ["Eiffel Tower"]) == 1
    assert T.qa_correct("squad_v2", "Answer: Paris", ["Eiffel Tower"]) == 0
    assert T.qa_correct("squad_v2", "Answer: unanswerable", ["<unanswerable>"]) == 1
    assert T.qa_official_correct("gsm8k", "$1,234", "#### 1234") is True
    assert T.qa_official_correct("gsm8k", "12", "") is None
    assert T.squad_scores("The Eiffel Tower", ["Eiffel Tower"], unanswerable=False)["exact"] == 1.0
    assert T.squad_scores("unanswerable", [], unanswerable=True)["f1"] == 1.0


# ------------------------------------------------------------------ text-to-SQL
def test_sql_scorers():
    assert T.sql_question("How many?", "CREATE TABLE t(a int)").endswith(
        "The answer is one SQL query for this schema.")
    assert T.sql_answer("```sql\nSELECT a\n  FROM t;\n```") == "SELECT a FROM t"
    assert T.sql_answer(None) == ""
    assert T.sql_statement("Here it is: SELECT x FROM y;") == "SELECT x FROM y"
    assert T.sql_statement("no query") is None
    assert T.sql_results_match([(1.0000001, "a"), (2, "b")], [(2, "b"), (1.0, "a")])
    assert not T.sql_results_match([(1,)], [(2,)])
    res = [("ok", [(1,)]), ("ok", [(2,)]), ("ok", [(2,)])]
    assert T.sql_execution_vote(["q1", "q2", "q3"], res) == "q2"
    assert T.sql_execution_vote(["q1", "q2", "q3"], [("error", None), ("timeout", []), ("ok", [])]) == "q2"
    assert T.sql_execution_vote(["q1"], [("error", None)]) is None
    fv = [("error", None), ("ok", []), ("ok", [(1,)])]
    assert T.sql_first_valid(["a", "b", "c"], fv, nonempty=True) == "c"
    assert T.sql_first_valid(["a", "b", "c"], fv) == "b"


# ------------------------------------------------------------------ entity matching
def test_matching_scorers():
    q = T.matching_question("Do these match?", {"title": "A", "price": ""}, {"title": "B"})
    assert q == ("Do these match?\n\nRecord A:\n  title: A\n\nRecord B:\n  title: B\n\n"
                 "The answer is either match or non-match.")
    assert T.matching_label("These two records are not a match.") == "non-match"
    assert T.matching_label("Non-match") == "non-match"
    assert T.matching_label("They refer to the same product.") == "match"
    assert T.matching_label("") == ""
    s = T.pair_scores(["match", "match", None, "non-match"], ["match", "non-match", "match", "non-match"])
    assert (s["tp"], s["fp"], s["fn"], s["n"]) == (1, 1, 1, 4)
    assert math.isclose(s["f1"], 50.0) and math.isclose(s["recall"], 50.0)
    assert T.cluster_f1(5, 10, 10) == 0.5 and T.cluster_f1(0, 0, 0) == 0.0


def test_pair_scores_on_shipped_judgments():
    rows = [r for r in csv.DictReader(open(_repo.inputs(CONTENT) / "amazon_google_pairs.csv"))
            if (r["recipient"], r["split"], r["input"]) == ("qwen3-32b", "test", "no_peer")]
    s = T.pair_scores([r["pred"] for r in rows], [r["gold"] for r in rows])
    assert_reproduces(CONTENT, "ag_qwen3_no_peer_f1", s["f1"])
    assert_reproduces(CONTENT, "ag_qwen3_no_peer_tp", s["tp"])
    assert_reproduces(CONTENT, "ag_qwen3_no_peer_recall", s["recall"])


def test_task_registry():
    assert [t.name for t in T.list_tasks()] == [T.QUESTION_ANSWERING, T.TEXT_TO_SQL, T.ENTITY_MATCHING]
    assert T.get_task("text_to_sql").score is T.sql_results_match


# ------------------------------------------------------------------ pair judgments to groups
def test_evaluation_levels_on_a_small_example():
    records, entity = ["1", "2", "3", "4"], {"1": "a", "2": "a", "3": "b", "4": "b"}
    pairs, gold = [("1", "2"), ("2", "3")], {("1", "2"): True, ("2", "3"): False}
    lv = G.evaluation_levels(pairs, gold, {("1", "2"), ("2", "3")}, records, entity)
    assert math.isclose(lv.pair_f1, 2 / 3) and math.isclose(lv.direct_f1, 0.5)
    assert math.isclose(lv.group_f1, 0.4)                    # 3 relations, 1 true, 2 true in total
    assert sorted(map(sorted, lv.groups)) == [["1", "2", "3"], ["4"]]
    assert G.relations(lv.groups) == {("1", "2"), ("1", "3"), ("2", "3")}
    assert sorted(map(sorted, G.connected_components(["1", "2", "3"], [("1", "2")]))) == [["1", "2"], ["3"]]


def test_changed_judgments_create_false_indirect_relations():
    """Abstract and Section sec:downstream: 23 changed pair judgments, 133 false indirect relations,
    pair F1 +0.03 pp, cluster F1 -3.05 pp (Qwen2.5-32B, peer decisions against no peers)."""
    e = G.sharing_effect_on_groups("qwen2.5-32b", "candidate_only", "no_peers")
    assert_reproduces(MERGE, "q32_changed_judgments", e.changed_judgments)
    assert_reproduces(MERGE, "q32_added_indirect_false", e.added_indirect_false)
    assert_reproduces(MERGE, "share_q32b_decisions_pair_pp", e.pair_f1_change_pp)
    assert_reproduces(MERGE, "share_q32b_decisions_cluster_pp", e.group_f1_change_pp)
    assert_reproduces(MERGE, "settlements_sharing_pairs", e.candidate_pairs)
    assert_reproduces(MERGE, "q32_added_relations", e.added_relations)
    assert_reproduces(MERGE, "q32_added_direct", e.added_direct)
    assert_reproduces(MERGE, "q32_added_true", e.added_true)
    assert_reproduces(MERGE, "q32_added_false", e.added_false)
    assert_reproduces(MERGE, "q32_changed_correct", e.changed_correct)
    assert_reproduces(MERGE, "q32_changed_incorrect", e.changed_incorrect)
    assert e.removed_relations == _repo.expected(MERGE)["q32_removed_relations"]


def test_merge_validation():
    v = G.merge_validation()
    for k in ("reversals", "combinations", "both_up", "cc_vs_center_comparisons",
              "cc_minus_center_musicbrainz_max_pp", "wdc_direct_link_f1_cap",
              "fig_settlements_qwen2.5-32b_groups", "fig_wdc_majority_pairs",
              "entity_disjoint_counterpart_cluster_pp", "q32_largest_merge_added_false_indirect"):
        assert_reproduces(MERGE, k, v[k])
    assert set(v) == set(_repo.expected(MERGE))


def test_grouping_rule_selection():
    v = G.grouping_rule_selection()
    for k in ("menu_settings", "selected_mean_regret_max_over_datasets_pp",
              "settlements_fixed_connected_components_mean_regret_pp",
              "musicbrainz_fixed_center_mean_regret_pp", "wdc_selected_connected_components",
              "settlements_devsize_21_entities_mean_regret_pp", "settlements_dev_entities"):
        assert_reproduces(RULES, k, v[k])


def test_matching_peer_content():
    v = G.matching_peer_content()
    for k in ("ag_qwen3_decision_minus_fields", "ag_gemma_decision_minus_fields",
              "ag_fields_correct_to_wrong", "ag_fields_c2w_follow_peer", "ag_qwen3_full_explanation_f1",
              "da_qwen3_no_peer_f1", "shared_content_test_pairs"):
        assert_reproduces(CONTENT, k, v[k])


def test_return_rule_effects():
    v = G.return_rule_effects()
    for k in ("max_abs_move_pp", "min_abs_move_pp", "settlements_madm2_nopeer_effect_protocol_pp",
              "settlements_madm2_nopeer_effect_majority_kept_pp",
              "settlements_dumad_nopeer_effect_majority_last_pp"):
        assert_reproduces(RETURN, k, v[k])


# ------------------------------------------------------------------ command line
def test_cli_lists(capsys):
    assert cli.main(["list-protocols", "--paper"]) == 0
    out = capsys.readouterr().out
    assert "Du-MAD" in out and "CHESS" in out and "DMAD" not in out
    assert cli.main(["list-tasks"]) == 0
    assert "entity matching" in capsys.readouterr().out
    assert cli.main(["replay", "--help"]) == 0
    assert cli.main(["replay", "no_such_step"]) == 2
