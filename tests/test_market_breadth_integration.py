"""
Unit tests for Follow-up Ticket 1: Real Market Breadth Integration.
"""

import numpy as np
import pandas as pd
import pytest

from market.data.market_breadth_fetcher import MarketBreadthFetcher
from market.replay.market_observation_synthesizer import MarketObservationSynthesizer


def test_market_breadth_fetcher_computation():
    """Verify MarketBreadthFetcher loads/computes genuine market breadth metrics."""
    fetcher = MarketBreadthFetcher()
    df = fetcher.fetch_and_compute_breadth(force_refresh=False)

    assert not df.empty
    assert len(df) > 500

    required_cols = [
        "date",
        "advance_decline_ratio",
        "net_advances_pct",
        "pct_above_50dma",
        "highs_minus_lows_pct",
        "composite_breadth_score",
        "market_breadth_state",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required breadth column {col}"

    valid_states = {
        "strongly_participatory",
        "strengthened",
        "mixed",
        "weakened",
        "deteriorated",
    }
    unique_states = set(df["market_breadth_state"].unique())
    assert unique_states.issubset(valid_states)


def test_derive_breadth_state_uses_genuine_breadth():
    """Assert _derive_breadth_state consumes market_breadth_state column rather than single-asset OHLCV."""
    dates = pd.date_range("2023-01-01", periods=10)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [102.0] * 10,  # All up days -> proxy would return strongly_participatory
            "volume": [10000] * 10,
            "market_breadth_state": [
                "deteriorated",
                "deteriorated",
                "weakened",
                "weakened",
                "mixed",
                "mixed",
                "strengthened",
                "strengthened",
                "strongly_participatory",
                "strongly_participatory",
            ],
            "composite_breadth_score": [0.2, 0.2, 0.35, 0.35, 0.5, 0.5, 0.6, 0.6, 0.8, 0.8],
        }
    )

    synth = MarketObservationSynthesizer(df, market_name="RELIANCE")

    assert synth._derive_breadth_state(0) == "deteriorated"
    assert synth._derive_breadth_state(2) == "weakened"
    assert synth._derive_breadth_state(4) == "mixed"
    assert synth._derive_breadth_state(6) == "strengthened"
    assert synth._derive_breadth_state(8) == "strongly_participatory"


def test_derive_breadth_state_fallback():
    """Assert _derive_breadth_state falls back gracefully to proxy when breadth data is absent."""
    dates = pd.date_range("2023-01-01", periods=10)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [102.0] * 10,  # All up days -> proxy returns strongly_participatory for index >= 5
            "volume": [10000] * 10,
        }
    )

    synth = MarketObservationSynthesizer(df, market_name="RELIANCE")
    assert synth._derive_breadth_state(6) == "strongly_participatory"
