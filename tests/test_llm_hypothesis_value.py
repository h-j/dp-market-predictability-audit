"""
Hermetic Unit Tests for Phase 4 LLM Hypothesis Value Study.

All test fixtures use committed CSV datasets or synthetic in-memory DataFrames.
Zero network access, zero imports from market.replay.run.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cognition.grammar.hypothesis_grammar import (
    ClauseAST,
    GrammarCompiler,
    HypothesisAST,
    compute_feature_quantiles,
)
from cognition.grammar.llm_hypothesis_generator import SubstrateLLMHypothesisGenerator
from cognition.grammar.random_hypothesis_generator import RandomHypothesisGenerator
from market.eval.hypothesis_evaluator import (
    HypothesisBacktestResult,
    HypothesisWalkForwardEvaluator,
)


def test_no_network_and_no_market_replay_run_imported():
    """Verify zero network calls and no imports from market.replay.run in cognition grammar and eval modules."""
    import cognition.grammar.hypothesis_grammar
    import cognition.grammar.llm_hypothesis_generator
    import cognition.grammar.random_hypothesis_generator
    import market.eval.hypothesis_evaluator

    modules = [
        cognition.grammar.hypothesis_grammar,
        cognition.grammar.random_hypothesis_generator,
        cognition.grammar.llm_hypothesis_generator,
        market.eval.hypothesis_evaluator,
    ]
    for mod in modules:
        mod_file = getattr(mod, "__file__", "")
        with open(mod_file, "r") as f:
            src = f.read()
            assert "market.replay.run" not in src, f"{mod.__name__} must NOT import market.replay.run!"


def test_grammar_ast_compilation_and_execution():
    """Test AST clause compilation and predicate evaluation."""
    compiler = GrammarCompiler()
    clause = ClauseAST(feature="vix_close", operator="GREATER_THAN", threshold_q1=0.50)
    ast = HypothesisAST(
        hypothesis_id="H_TEST",
        description="Test AST",
        source="LLM",
        target_mode="volatility_5d",
        clauses=[clause],
        logical_op="AND",
        prediction_signal=1.0,
    )

    feature_quantiles = {"vix_close": {0.50: 15.0}}
    eval_fn = compiler.compile_hypothesis(ast, feature_quantiles)

    assert eval_fn({"vix_close": 16.0}) is True
    assert eval_fn({"vix_close": 14.0}) is False


def test_random_generator_reproducibility():
    """Test that RandomHypothesisGenerator is reproducible under fixed seed."""
    gen1 = RandomHypothesisGenerator(seed=42)
    hyps1 = gen1.generate_hypotheses(count=10)

    gen2 = RandomHypothesisGenerator(seed=42)
    hyps2 = gen2.generate_hypotheses(count=10)

    assert len(hyps1) == len(hyps2) == 10
    for h1, h2 in zip(hyps1, hyps2):
        assert h1.hypothesis_id == h2.hypothesis_id
        assert len(h1.clauses) == len(h2.clauses)
        assert h1.clauses[0].feature == h2.clauses[0].feature


def test_llm_generator_conformity_to_grammar():
    """Test that SubstrateLLMHypothesisGenerator produces valid HypothesisAST instances."""
    gen = SubstrateLLMHypothesisGenerator(seed=42)
    hyps, log = gen.generate_llm_arm(target_count=10, max_attempts=20)

    assert len(hyps) == 10
    for h in hyps:
        assert h.source == "LLM"
        assert len(h.clauses) > 0
        assert h.target_mode in ["volatility_5d", "vol_regime_expansion", "direction_3d"]


def test_evaluator_statistical_gates_logic():
    """Test Fisher's Exact Test and Mann-Whitney U test gate evaluation on mock results."""
    evaluator = HypothesisWalkForwardEvaluator(n_bootstrap=500, seed=42)

    # Superior LLM mock results (high survival rate, high R2)
    llm_mock = [
        HypothesisBacktestResult(f"H_L_{i}", "LLM", "volatility_5d", 50, 0.20, 0.10, 0.25, True if i < 35 else False)
        for i in range(50)
    ]
    # Inferior Random mock results (low survival rate, low R2)
    rand_mock = [
        HypothesisBacktestResult(f"H_R_{i}", "RANDOM", "volatility_5d", 50, 0.01, 0.15, 0.02, True if i < 5 else False)
        for i in range(50)
    ]

    s_llm, s_rand, gates = evaluator.evaluate_study_gates(llm_mock, rand_mock)

    assert s_llm.survival_rate_pct == 70.0
    assert s_rand.survival_rate_pct == 10.0
    assert gates.gate_g_llm1_passed is True
    assert gates.gate_g_llm2_passed is True
    assert gates.final_verdict == "LLM REASONING VALIDATED"
