"""
Unit and regression tests for prediction review findings (Items 1 to 8).
"""

import numpy as np
import pandas as pd
import pytest

from market.replay.market_observation_synthesizer import MarketObservationSynthesizer
from market.replay.prediction_probe import (
    PredictionDirection,
    PredictionProbe,
    PredictionProbeGenerator,
    calibrate_confidence_from_history,
)
from market.replay.replay_analysis import ReplayAnalysisEngine
from market.replay.visualization import _prediction_dataframe


# ============================================================================
# ITEM 1 TEST
# ============================================================================
def test_item1_score_direction_uncertain():
    """Assert predicted='uncertain'/actual='uncertain' and predicted='uncertain'/actual='higher' are NOT scored identically."""
    probe_gen = PredictionProbeGenerator()

    score_uncertain_uncertain = probe_gen._score_direction(
        PredictionDirection.uncertain, PredictionDirection.uncertain
    )
    score_uncertain_higher = probe_gen._score_direction(
        PredictionDirection.uncertain, PredictionDirection.higher
    )

    assert score_uncertain_uncertain == 0.5, "uncertain/uncertain should score 0.5 (undecidable)"
    assert score_uncertain_higher == 0.0, "uncertain/higher should score 0.0"
    assert score_uncertain_uncertain != score_uncertain_higher, "Uncertain calls should not be scored identically"


# ============================================================================
# ITEM 2 TEST
# ============================================================================
def test_item2_baseline_reporting_and_comparison():
    """Verify data-driven majority class and always range_bound baseline calculations."""
    engine = ReplayAnalysisEngine(market_name="RELIANCE")

    engine.days = [
        {"date": "2023-01-01"},
        {"date": "2023-01-02"},
        {"date": "2023-01-03"},
        {"date": "2023-01-04"},
    ]

    # Mock prediction history with plurality range_bound actual outcomes
    engine.prediction_history = [
        {"date": "2023-01-01", "prediction": {"direction": "range_bound"}},
        {
            "date": "2023-01-02",
            "prediction": {"direction": "higher"},
            "prior_prediction_result": {
                "prior_direction": "range_bound",
                "actual_direction": "range_bound",
                "direction_score": 1.0,
            },
        },
        {
            "date": "2023-01-03",
            "prediction": {"direction": "higher"},
            "prior_prediction_result": {
                "prior_direction": "higher",
                "actual_direction": "range_bound",
                "direction_score": 0.5,
            },
        },
        {
            "date": "2023-01-04",
            "prediction": {"direction": "lower"},
            "prior_prediction_result": {
                "prior_direction": "higher",
                "actual_direction": "higher",
                "direction_score": 1.0,
            },
        },
    ]

    analysis = engine.analyze()
    pa = analysis.get("prediction_analysis", {})

    assert "majority_class_baseline_score" in pa
    assert "always_range_bound_baseline_score" in pa
    assert "exceeds_baselines" in pa
    assert pa["majority_class"] == "range_bound"


# ============================================================================
# ITEM 3 REGRESSION TEST
# ============================================================================
def test_item3_prediction_dataframe_1day_shift_consistency():
    """Regression test asserting for any row in _prediction_dataframe, prediction_direction matches prior_direction."""
    mock_history = [
        {
            "date": "2023-01-01",
            "prediction": {"direction": "higher", "confidence": 0.7},
            "prior_prediction_result": None,
        },
        {
            "date": "2023-01-02",
            "prediction": {"direction": "lower", "confidence": 0.6},
            "prior_prediction_result": {
                "prior_direction": "higher",
                "actual_direction": "higher",
                "direction_score": 1.0,
            },
        },
        {
            "date": "2023-01-03",
            "prediction": {"direction": "range_bound", "confidence": 0.8},
            "prior_prediction_result": {
                "prior_direction": "lower",
                "actual_direction": "lower",
                "direction_score": 1.0,
            },
        },
    ]

    df = _prediction_dataframe({"prediction_history": mock_history, "market_name": "TEST"})
    assert not df.empty
    assert len(df) == 2

    # Assert consistency: row 0 date is 2023-01-02, prediction_direction shown is "higher" (from Day 1), evaluating actual "higher"
    for _, row in df.iterrows():
        evaluated_prior = row["prior_prediction_result"]["prior_direction"]
        shown_direction = row["prediction_direction"]
        assert shown_direction == evaluated_prior, (
            f"Row {row['date']} prediction mismatch: shown '{shown_direction}' vs evaluated '{evaluated_prior}'"
        )


# ============================================================================
# ITEM 4 TEST
# ============================================================================
def test_item4_market_name_explicit_threading():
    """Verify MarketObservationSynthesizer preserves explicit market_name."""
    dates = pd.date_range("2023-01-01", periods=10)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 10000,
        }
    )

    synth_rel = MarketObservationSynthesizer(df, market_name="RELIANCE")
    assert synth_rel.market_name == "RELIANCE"

    synth_nifty = MarketObservationSynthesizer(df, market_name="NIFTY")
    assert synth_nifty.market_name == "NIFTY"


# ============================================================================
# ITEM 5 TEST
# ============================================================================
def test_item5_breadth_proxy_docstring_and_non_compounding():
    """Verify _derive_breadth_state docstring explicitly documents single-asset proxy status."""
    docstring = MarketObservationSynthesizer._derive_breadth_state.__doc__
    assert docstring is not None
    assert "single-asset" in docstring.lower() or "proxy" in docstring.lower()


# ============================================================================
# ITEM 6 TEST
# ============================================================================
def test_item6_volatility_state_percentile_rank_non_degenerate():
    """Assert expanding window percentile rank populates all 3 buckets on synthetic data with non-RELIANCE scale."""
    np.random.seed(42)
    periods = 120

    # Low volatility scale asset (mean 0.4%, compared to RELIANCE ~1.5-2.0%)
    vol_series = np.random.normal(loc=0.4, scale=0.1, size=periods)
    vol_series = np.clip(vol_series, 0.1, 1.0)

    dates = pd.date_range("2023-01-01", periods=periods)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.2,
            "volume": 1000,
            "rolling_volatility_30d": vol_series,
        }
    )

    synth = MarketObservationSynthesizer(df, market_name="LOW_VOL_INDEX")
    states = [synth._derive_volatility_state(i) for i in range(60, periods)]

    unique_states = set(states)
    # Check that compressed, moderate/stable, and expanded/high are populated
    has_compressed = any("compressed" in s for s in unique_states)
    has_moderate_or_stable = any(s in {"moderate", "stable"} for s in unique_states)
    has_expanded_or_high = any(s in {"expanded", "high"} for s in unique_states)

    assert has_compressed, "Compressed bucket should be populated"
    assert has_moderate_or_stable, "Moderate/stable bucket should be populated"
    assert has_expanded_or_high, "Expanded/high bucket should be populated"


# ============================================================================
# ITEM 7 TEST
# ============================================================================
def test_item7_direction_threshold_normalization():
    """Assert _infer_direction uses trailing 10d volatility to normalize raw return thresholds."""
    probe_gen = PredictionProbeGenerator()

    class MockObs:
        trend_state = "neutral"
        breadth_state = "mixed"
        macro_sentiment = "neutral"
        candle_type = "normal"
        descriptors = []
        gap_pct = 0.1
        rolling_volatility_10d = 2.0  # High volatility asset

    class MockTheory:
        summary = "test"

    class MockReflection:
        reflection_summary = "test"

    obs = MockObs()
    theory = MockTheory()
    reflection = MockReflection()

    # Raw return_3d=0.4 is > 0.3 raw, but normalized return_3d = 0.4 / 2.0 = 0.2 (< 0.3 threshold)
    # So strong_up should NOT trigger
    direction = probe_gen._infer_direction(
        observation=obs,
        theory=theory,
        reflection=reflection,
        return_3d=0.4,
        return_5d=0.6,
    )

    assert direction != PredictionDirection.higher, "High volatility scale should normalize return threshold so 0.4 is not strong_up"


# ============================================================================
# ITEM 8 TEST
# ============================================================================
def test_item8_calibrate_confidence_from_history_stub():
    """Test confidence calibration helper stub on sample prediction dataframe."""
    df = pd.DataFrame(
        {
            "confidence": [0.15, 0.25, 0.65, 0.75, 0.85],
            "direction_score": [0.0, 1.0, 1.0, 0.5, 1.0],
        }
    )

    stats = calibrate_confidence_from_history(df)
    assert isinstance(stats, dict)
    assert len(stats["deciles"]) == 10
    assert "0.8-0.9" in stats
    assert stats["0.8-0.9"]["count"] == 1
    assert stats["0.8-0.9"]["actual_hit_rate"] == 1.0
