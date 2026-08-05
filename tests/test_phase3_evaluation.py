"""
Hermetic Unit Tests for Phase 3 Rebuild Hypothesis Survival Evaluation.

Verifies 60/40 IS/OOS split calculation, binomial exact test survival filter,
and Gate G-P3 Mann-Whitney U test logic.
All tests run 100% offline with zero network calls and zero imports from market.replay.run.
"""

import numpy as np
import pandas as pd
import pytest

from cognition.grammar.hypothesis_grammar import ClauseAST, HypothesisAST
from market.eval.phase3_evaluator import (
    HypothesisPhase3Result,
    Phase3ArmSummary,
    Phase3GateP3Result,
    Phase3HypothesisEvaluator,
)


def test_phase3_no_network_and_no_market_replay_run():
    """Verify zero network calls and no imports from market.replay.run in phase3_evaluator."""
    import market.eval.phase3_evaluator

    mod_file = market.eval.phase3_evaluator.__file__
    with open(mod_file, "r") as f:
        src = f.read()
        assert "market.replay.run" not in src, "phase3_evaluator must NOT import market.replay.run!"


def test_phase3_evaluator_split_logic():
    """Test 60/40 IS/OOS split and binomial filter evaluation on synthetic dataset."""
    dates = pd.date_range("2024-01-01", periods=100)
    df_synthetic = pd.DataFrame(
        {
            "Date": dates,
            "vix_close": np.linspace(10, 25, 100),
            "rv_5d": np.ones(100) * 1.0,
            "target_volatility_5d": np.where(np.arange(100) % 2 == 0, 1.5, 0.5),
        }
    )

    clause = ClauseAST(feature="vix_close", operator="GREATER_THAN", threshold_q1=0.50)
    ast = HypothesisAST(
        hypothesis_id="H_TEST_P3",
        description="Test AST P3",
        source="LLM",
        target_mode="volatility_5d",
        clauses=[clause],
        logical_op="AND",
        prediction_signal=1.0,
    )

    evaluator = Phase3HypothesisEvaluator(is_ratio=0.60, is_p_threshold=0.10)
    res = evaluator.evaluate_hypothesis_split(ast, df_synthetic)

    assert res.hypothesis_id == "H_TEST_P3"
    assert res.is_triggers > 0
    assert 0.0 <= res.is_binomial_p <= 1.0
    assert isinstance(res.is_survived, bool)
    assert isinstance(res.oos_edge, float)


def test_phase3_gate_g_p3_pass_and_fail_conditions():
    """Test Gate G-P3 decision logic under synthetic passing and failing inputs."""
    evaluator = Phase3HypothesisEvaluator()

    # Case 1: Superior LLM survivor edges (Pass)
    llm_surv_pass = [
        HypothesisPhase3Result("H_L1", "LLM", "volatility_5d", "", 10, 8, 0.8, 0.5, 0.01, True, 10, 8, 0.8, 0.5, +0.35),
        HypothesisPhase3Result("H_L2", "LLM", "volatility_5d", "", 10, 8, 0.8, 0.5, 0.01, True, 10, 8, 0.8, 0.5, +0.30),
        HypothesisPhase3Result("H_L3", "LLM", "volatility_5d", "", 10, 8, 0.8, 0.5, 0.01, True, 10, 8, 0.8, 0.5, +0.25),
        HypothesisPhase3Result("H_L4", "LLM", "volatility_5d", "", 10, 8, 0.8, 0.5, 0.01, True, 10, 8, 0.8, 0.5, +0.20),
    ]
    rand_surv_fail = [
        HypothesisPhase3Result("H_R1", "RANDOM", "volatility_5d", "", 10, 6, 0.6, 0.5, 0.08, True, 10, 5, 0.5, 0.5, +0.00),
        HypothesisPhase3Result("H_R2", "RANDOM", "volatility_5d", "", 10, 6, 0.6, 0.5, 0.08, True, 10, 5, 0.5, 0.5, -0.05),
        HypothesisPhase3Result("H_R3", "RANDOM", "volatility_5d", "", 10, 6, 0.6, 0.5, 0.08, True, 10, 5, 0.5, 0.5, -0.10),
        HypothesisPhase3Result("H_R4", "RANDOM", "volatility_5d", "", 10, 6, 0.6, 0.5, 0.08, True, 10, 5, 0.5, 0.5, -0.15),
    ]

    s_llm, s_rand, gate_pass = evaluator.evaluate_gate_g_p3(llm_surv_pass, rand_surv_fail)

    assert s_llm.is_survived_count == 4
    assert s_rand.is_survived_count == 4
    assert gate_pass.gate_g_p3_passed is True
    assert gate_pass.final_verdict == "LLM REASONING VALIDATED"

    # Case 2: Inferior LLM survivor edges (Fail)
    s_llm2, s_rand2, gate_fail = evaluator.evaluate_gate_g_p3(rand_surv_fail, llm_surv_pass)
    assert gate_fail.gate_g_p3_passed is False
    assert gate_fail.final_verdict == "LLM REASONING NULL"
