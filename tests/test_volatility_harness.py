"""
Unit tests for Workstream 3 Volatility Harness & Models.
"""

import numpy as np
import pandas as pd
import pytest

from market.replay.walkforward_validation import (
    GBMVolModel,
    HARRVModel,
    WalkForwardValidator,
    compute_qlike,
    compute_r2_vs_persistence,
    compute_spearman_rank,
)


def test_volatility_metric_functions():
    y_true = np.array([1.2, 1.5, 0.8, 2.0, 1.1])
    y_pred = np.array([1.1, 1.4, 0.9, 1.8, 1.2])
    y_pers = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    r2_vs_pers = compute_r2_vs_persistence(y_true, y_pred, y_pers)
    qlike = compute_qlike(y_true, y_pred)
    spearman = compute_spearman_rank(y_true, y_pred)

    assert r2_vs_pers > 0.0, "Predictions closer than persistence should have R2 > 0"
    assert qlike > 0.0, "QLIKE loss should be strictly positive"
    assert 0.0 < spearman <= 1.0, "Spearman rank correlation should be high for aligned predictions"


def test_har_rv_and_gbm_vol_models():
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + 0.05 * np.random.randn(100)
    y = np.clip(y, 0.05, None)

    har = HARRVModel()
    har.fit(X[:80], y[:80])
    pred_har = har.predict(X[80:])

    gbm = GBMVolModel(n_estimators=50)
    gbm.fit(X[:80], y[:80])
    pred_gbm = gbm.predict(X[80:])

    assert len(pred_har) == 20
    assert len(pred_gbm) == 20
    assert np.all(pred_har >= 0.01), "Predictions must be non-negative"
    assert np.all(pred_gbm >= 0.01), "Predictions must be non-negative"


def test_walkforward_validator_volatility_5d_mode():
    dates = pd.date_range("2023-01-01", periods=400, freq="B")
    np.random.seed(42)
    returns = np.random.normal(0, 1, 400)
    prices = 100 * np.exp(np.cumsum(returns) / 100.0)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "close": prices,
            "daily_return_pct": returns,
            "rolling_volatility_10d": np.ones(400),
            "rolling_volatility_30d": np.ones(400),
            "return_3d": np.zeros(400),
            "return_5d": np.zeros(400),
            "volume_ratio_5d": np.ones(400),
            "volume_ratio_20d": np.ones(400),
            "gap_pct": np.zeros(400),
            "range_pct": np.ones(400),
        }
    )

    validator = WalkForwardValidator(
        df, asset_name="TEST_ASSET", target_mode="volatility_5d", initial_train_size=250, test_fold_size=100
    )

    study_res = validator.run_study()
    assert study_res.target_mode == "volatility_5d"
    assert study_res.num_folds > 0
    assert len(study_res.fold_results) == study_res.num_folds
    assert hasattr(study_res, "avg_har_r2_vs_pers")
    assert hasattr(study_res, "avg_gbm_r2_vs_pers")
