"""
Unit tests for Walk-Forward Validation Harness (walkforward_validation.py).
"""

import numpy as np
import pandas as pd
import pytest

from market.replay.walkforward_validation import WalkForwardValidator, WalkForwardStudyResult


def test_walkforward_validator_split_generation():
    """Verify generate_expanding_folds generates non-overlapping expanding time-series splits."""
    dates = pd.date_range("2023-01-01", periods=600)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.random.uniform(90, 110, 600),
            "high": np.random.uniform(105, 115, 600),
            "low": np.random.uniform(85, 95, 600),
            "close": np.random.uniform(90, 110, 600),
            "volume": np.random.uniform(1000, 5000, 600),
            "return_3d": np.random.uniform(-2, 2, 600),
            "return_5d": np.random.uniform(-3, 3, 600),
            "daily_return_pct": np.random.uniform(-1, 1, 600),
            "volume_ratio_5d": np.ones(600),
            "volume_ratio_20d": np.ones(600),
            "gap_pct": np.zeros(600),
            "range_pct": np.ones(600),
            "rolling_volatility_10d": np.ones(600),
            "rolling_volatility_30d": np.ones(600),
        }
    )

    validator = WalkForwardValidator(df, initial_train_size=250, test_fold_size=100)
    folds = validator.generate_expanding_folds()

    assert len(folds) >= 3
    for k, (train_idx, test_idx) in enumerate(folds):
        assert train_idx[0] == 0
        assert len(train_idx) >= 250
        assert train_idx[-1] + 1 == test_idx[0]  # No gap or overlap between train and test
        if k > 0:
            # Expanding train window
            assert len(folds[k][0]) > len(folds[k - 1][0])


def test_direction_score_calculation():
    """Verify _calculate_direction_score partial and full credit rules."""
    v = WalkForwardValidator._calculate_direction_score

    assert v(1, 1) == 1.0
    assert v(-1, -1) == 1.0
    assert v(0, 0) == 1.0

    # Range bound partial credit
    assert v(0, 1) == 0.5
    assert v(0, -1) == 0.5
    assert v(1, 0) == 0.5
    assert v(-1, 0) == 0.5

    # Opposing directions
    assert v(1, -1) == 0.0
    assert v(-1, 1) == 0.0


def test_walkforward_study_execution():
    """Verify run_study returns complete metrics and fold results."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 100.0 + np.cumsum(np.random.normal(0, 1, 500)),
            "high": 105.0 + np.cumsum(np.random.normal(0, 1, 500)),
            "low": 95.0 + np.cumsum(np.random.normal(0, 1, 500)),
            "close": 100.0 + np.cumsum(np.random.normal(0, 1, 500)),
            "volume": np.random.uniform(1000, 5000, 500),
            "return_3d": np.random.normal(0, 1.5, 500),
            "return_5d": np.random.normal(0, 2.0, 500),
            "daily_return_pct": np.random.normal(0, 1.0, 500),
            "volume_ratio_5d": np.ones(500),
            "volume_ratio_20d": np.ones(500),
            "gap_pct": np.random.normal(0, 0.5, 500),
            "range_pct": np.random.normal(1.5, 0.3, 500),
            "rolling_volatility_10d": np.ones(500),
            "rolling_volatility_30d": np.ones(500),
        }
    )

    validator = WalkForwardValidator(df, asset_name="TEST_ASSET", initial_train_size=250, test_fold_size=100)
    result = validator.run_study()

    assert isinstance(result, WalkForwardStudyResult)
    assert result.asset_name == "TEST_ASSET"
    assert result.num_folds >= 2
    assert len(result.fold_results) == result.num_folds
