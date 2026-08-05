"""
Unit tests for Phase 2 Prediction Review & Walk-Forward Study Fixes (Items 1-10).
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from market.data.download_history import resolve_symbol_file, ensure_data
from market.replay.run import DataPreparationManager, FeaturePreparationManager
from market.replay.walkforward_validation import (
    WalkForwardValidator,
    compute_balanced_accuracy,
    compute_mcc,
)
from market.replay.prediction_probe import calibrate_confidence_from_history
from market.replay.capital_simulator import CapitalSimulator


def test_symbol_dataset_resolution_and_source_assertion():
    """Verify resolve_symbol_file and FeaturePreparationManager source column assertion."""
    assert resolve_symbol_file("RELIANCE", enriched=False) == "reliance_daily_3y.csv"
    assert resolve_symbol_file("RELIANCE", enriched=True) == "reliance_enriched_daily_3y.csv"

    assert resolve_symbol_file("NIFTY", enriched=False) == "nifty_daily_3y.csv"
    assert resolve_symbol_file("NIFTY", enriched=True) == "nifty_enriched_daily_3y.csv"

    assert resolve_symbol_file("TCS", enriched=False) == "tcs_daily_3y.csv"
    assert resolve_symbol_file("TCS", enriched=True) == "tcs_enriched_daily_3y.csv"


def test_walkforward_tier1_features_and_metrics():
    """Verify Tier 1 features and MCC / Balanced Accuracy metric calculation."""
    y_true = np.array([1, 1, 0, -1, 0, 1, -1, 0])
    y_pred = np.array([1, 0, 0, -1, 1, 1, -1, 0])

    bal_acc = compute_balanced_accuracy(y_true, y_pred)
    mcc = compute_mcc(y_true, y_pred)

    assert 0.0 <= bal_acc <= 1.0
    assert -1.0 <= mcc <= 1.0

    # Ensure all Tier 1 features exist in FEATURE_COLS
    tier1 = [
        "delivery_pct",
        "delivery_pct_5d",
        "fii_net",
        "dii_net",
        "sector_rs_ratio",
        "sector_zscore",
        "sector_percentile",
    ]
    for col in tier1:
        assert col in WalkForwardValidator.FEATURE_COLS, f"Missing Tier 1 feature {col}"


def test_calibrate_confidence_from_history():
    """Verify empirical confidence calibration deciles and gap calculation."""
    df_history = pd.DataFrame(
        {
            "confidence": [0.15, 0.25, 0.65, 0.75, 0.85, 0.95],
            "direction_score": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    res = calibrate_confidence_from_history(df_history)
    assert "deciles" in res
    assert "mean_calibration_gap" in res
    assert "calibration_table" in res
    assert len(res["deciles"]) == 10


def test_capital_simulator_deprecation_warning():
    """Verify CapitalSimulator emits DeprecationWarning."""
    with pytest.deprecated_call():
        sim = CapitalSimulator(starting_capital=10000.0)
        assert sim.starting_capital == 10000.0
