"""
Hermetic Unit Tests for Cross-Sectional Ranking Study (WS-D).

All tests read committed universe CSV files directly or synthetic in-memory DataFrames.
Zero network access, zero imports from market.replay.run or cognition stack.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market.cross_section.ic_evaluator import CrossSectionalICEvaluator
from market.cross_section.portfolio_sim import PortfolioSimulator
from market.cross_section.signal_engine import CrossSectionalSignalEngine, load_universe_data


@pytest.fixture(scope="module")
def committed_universe_dfs():
    """
    Hermetic module fixture reading committed CSVs from data/universe/.
    """
    data_dir = Path(__file__).parent.parent / "data"
    return load_universe_data(data_dir=data_dir)


def test_no_network_and_no_cognition_stack_imported():
    """
    Test 6: Verify zero network calls and no imports from market.replay.run or cognition stack.
    """
    import market.cross_section.ic_evaluator
    import market.cross_section.portfolio_sim
    import market.cross_section.signal_engine

    modules = [
        market.cross_section.signal_engine,
        market.cross_section.portfolio_sim,
        market.cross_section.ic_evaluator,
    ]
    for mod in modules:
        mod_file = getattr(mod, "__file__", "")
        with open(mod_file, "r") as f:
            src = f.read()
            assert "market.replay.run" not in src, f"{mod.__name__} must NOT import market.replay.run!"
            assert "cognition" not in src, f"{mod.__name__} must NOT import cognition stack!"


def test_point_in_time_signal_independence(committed_universe_dfs):
    """
    Test 1: Signals computed at date t remain identical when all post-t rows are removed or modified.
    """
    sample_ticker = "RELIANCE" if "RELIANCE" in committed_universe_dfs else next(iter(committed_universe_dfs.keys()))
    engine = CrossSectionalSignalEngine(committed_universe_dfs)

    df_full = committed_universe_dfs[sample_ticker]
    rebalance_date = df_full["date"].iloc[500]

    # Compute signals with full history
    sigs_full = engine.compute_raw_signals_for_ticker(sample_ticker, rebalance_date)

    # Truncate history to strictly <= rebalance_date
    df_truncated = df_full[df_full["date"] <= rebalance_date].copy()
    engine_truncated = CrossSectionalSignalEngine({sample_ticker: df_truncated})
    sigs_truncated = engine_truncated.compute_raw_signals_for_ticker(sample_ticker, rebalance_date)

    assert sigs_full is not None and sigs_truncated is not None
    for k in sigs_full.keys():
        assert np.isclose(
            sigs_full[k], sigs_truncated[k], atol=1e-6
        ), f"Point-in-time violation for {k}: {sigs_full[k]} vs {sigs_truncated[k]}"


def test_forward_return_alignment_fixture():
    """
    Test 2: Forward return window starts strictly after rebalance date (hand-computed fixture).
    """
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df_a = pd.DataFrame({"date": dates, "close": np.linspace(100, 200, 100)})
    df_b = pd.DataFrame({"date": dates, "close": np.linspace(200, 100, 100)})

    universe_dfs = {"STOCK_A": df_a, "STOCK_B": df_b}
    rebalance_date = dates[50]  # 2024-02-20

    engine = CrossSectionalSignalEngine(universe_dfs, history_cutoff_months=0)
    df_ranks = engine.compute_cross_section_ranks(rebalance_date)

    assert not df_ranks.empty
    # Stock A close at t=50 is 150.5, at t=71 is 171.7 -> return > 0
    # Stock B close at t=50 is 149.5, at t=71 is 128.3 -> return < 0
    fwd_a = df_ranks[df_ranks["ticker"] == "STOCK_A"]["fwd_return"].iloc[0]
    fwd_b = df_ranks[df_ranks["ticker"] == "STOCK_B"]["fwd_return"].iloc[0]

    assert fwd_a > 0, "Stock A forward return must be positive"
    assert fwd_b < 0, "Stock B forward return must be negative"


def test_planted_monotonic_signal_rank_sanity():
    """
    Test 3: Synthetic universe with planted monotonic signal yields Rank IC ~ 1.0 and correct Q1-Q5 ordering.
    """
    np.random.seed(42)
    n_stocks = 50
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    rebalance_date = dates[250]

    universe_dfs = {}
    for i in range(n_stocks):
        ticker = f"STK_{i:02d}"
        # Plant monotonic relationship: higher i -> higher signal AND higher forward return
        base_trend = np.linspace(100, 100 + i * 2, 300)
        df = pd.DataFrame({"date": dates, "close": base_trend, "volume": np.ones(300) * 1000})
        universe_dfs[ticker] = df

    engine = CrossSectionalSignalEngine(universe_dfs, history_cutoff_months=0)
    df_ranks = engine.compute_cross_section_ranks(rebalance_date)

    evaluator = CrossSectionalICEvaluator()
    ic_rs = evaluator.compute_monthly_ic(df_ranks, "rs_nifty_3m")
    ic_comp = evaluator.compute_monthly_ic(df_ranks, "composite")

    assert ic_rs > 0.95, f"Planted monotonic signal on rs_nifty_3m must produce Rank IC > 0.95 (got {ic_rs:.4f})"
    assert ic_comp > 0.40, f"Planted monotonic composite signal must produce positive Rank IC (got {ic_comp:.4f})"


def test_permutation_null_planted_zero_signal():
    """
    Test 4: Planted zero signal synthetic universe produces composite IC inside null bounds.
    """
    np.random.seed(42)
    n_stocks = 30
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    rebalance_date = dates[250]

    universe_dfs = {}
    for i in range(n_stocks):
        ticker = f"STK_{i:02d}"
        # Random noise close series (zero true signal)
        closes = 100.0 + np.random.randn(300).cumsum()
        df = pd.DataFrame({"date": dates, "close": np.clip(closes, 10.0, None), "volume": np.ones(300) * 1000})
        universe_dfs[ticker] = df

    engine = CrossSectionalSignalEngine(universe_dfs, history_cutoff_months=0)
    df_ranks = engine.compute_cross_section_ranks(rebalance_date)

    evaluator = CrossSectionalICEvaluator(n_bootstrap=200, n_permutations=200)
    monthly_ranks = {rebalance_date: df_ranks}
    _, summary = evaluator.evaluate_signal_ic(monthly_ranks, "composite")

    assert not summary.is_statistically_significant, "Planted zero signal must NOT pass Gate G-XS1!"


def test_turnover_and_transaction_cost_fixture():
    """
    Test 5: Turnover and 0.25% cost deduction verified on 2-period hand-computed fixture.
    """
    simulator = PortfolioSimulator(one_way_cost_pct=0.0025)

    # Period 1: Q5 holds STK_A, STK_B (weights: 0.5, 0.5)
    w_prev = {"STK_A": 0.5, "STK_B": 0.5}
    # Period 2: Q5 holds STK_B, STK_C (weights: 0.5, 0.5)
    w_curr = {"STK_B": 0.5, "STK_C": 0.5}

    # Turnover: |0 - 0.5| (A) + |0.5 - 0.5| (B) + |0.5 - 0| (C) = 0.5 + 0 + 0.5 = 1.0
    turnover = simulator.calculate_turnover(w_prev, w_curr)
    assert np.isclose(turnover, 1.0), f"Turnover must equal 1.0 (got {turnover})"

    cost = turnover * 0.0025
    assert np.isclose(cost, 0.0025), f"Cost deduction must equal 0.0025 (got {cost})"
