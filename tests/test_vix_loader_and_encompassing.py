"""
Unit tests for Workstream 1 India VIX loader and Mincer-Zarnowitz encompassing study.
"""

import numpy as np
import pandas as pd
import pytest

from bootstrap.run_vix_encompassing_study import run_block_bootstrap
from market.data.vix_loader import IndiaVIXLoader


def test_vix_loader_ingestion_and_point_in_time():
    loader = IndiaVIXLoader()
    df_vix = loader.load_vix_data()

    assert not df_vix.empty, "VIX dataframe must not be empty"
    required_cols = ["date", "vix_close", "vix_change_5d", "vix_vs_20d_ma", "vix_percentile_252d"]
    for col in required_cols:
        assert col in df_vix.columns, f"Missing VIX column {col}"

    assert (df_vix["vix_close"] >= 5.0).all() and (df_vix["vix_close"] <= 100.0).all(), "VIX values must be in [5, 100]"
    assert df_vix["date"].is_monotonic_increasing, "VIX dates must be sorted"
    assert df_vix["date"].nunique() == len(df_vix), "No duplicate dates permitted"


def test_mincer_zarnowitz_block_bootstrap():
    np.random.seed(42)
    n = 100
    y_true = np.random.normal(15, 2, n)
    y_pred_a = y_true + np.random.normal(0, 0.5, n)
    y_pred_b = y_true + np.random.normal(0, 0.5, n)

    beta_a, beta_b, ci_low, ci_high, sig_gt_zero = run_block_bootstrap(
        y_true, y_pred_a, y_pred_b, block_size=4, n_resamples=500
    )

    assert isinstance(beta_a, float)
    assert isinstance(beta_b, float)
    assert ci_low <= ci_high
    assert isinstance(sig_gt_zero, bool)
