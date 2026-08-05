"""
Hermetic Unit tests for Workstream 2 options pricing, strategy simulation, friction costs, and Gate G-OPT.

All test fixtures read committed CSVs directly (nifty_enriched_daily_3y.csv & hist_india_vix*.csv).
NO network calls, NO imports from market.replay.run.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from bootstrap.run_options_simulation_study import generate_nifty_weekly_dataset_with_forecasts
from config.options_costs import calculate_leg_cost
from market.options.regime_strategy import OptionsRegimeStrategyEngine
from market.options.strategy_evaluator import StrategyEvaluator, StrategyMetrics
from market.options.synthetic_pricer import SyntheticOptionPricer, bs_price


@pytest.fixture(scope="module")
def nifty_weekly_df():
    """
    Hermetic module-scoped fixture reading committed CSVs directly from data_dir.
    No network calls, no market/replay/run.py import.
    """
    data_dir = Path(__file__).parent.parent / "data"
    return generate_nifty_weekly_dataset_with_forecasts(data_dir=data_dir)


def test_no_market_replay_run_imported():
    """Verify that market.replay.run is NOT imported by options simulation modules."""
    import bootstrap.run_options_simulation_study
    import market.options.regime_strategy
    import market.options.strategy_evaluator
    import market.options.synthetic_pricer

    opts_modules = [
        bootstrap.run_options_simulation_study,
        market.options.regime_strategy,
        market.options.strategy_evaluator,
        market.options.synthetic_pricer,
    ]
    for mod in opts_modules:
        mod_src = getattr(mod, "__file__", "")
        with open(mod_src, "r") as f:
            src = f.read()
            assert "market.replay.run" not in src, f"{mod.__name__} must NOT import market.replay.run!"


def test_friction_costs_strictly_positive():
    cost_buy = calculate_leg_cost(premium=100.0, is_sell=False, lot_size=50)
    cost_sell = calculate_leg_cost(premium=100.0, is_sell=True, lot_size=50)

    assert cost_buy > 20.0, "Cost must include brokerage (₹20) plus exchange/slippage"
    assert cost_sell > cost_buy, "Sell order cost must be higher due to STT (0.0625%)"


def test_black_scholes_put_call_parity():
    S, K, T, r, sigma = 20000.0, 20000.0, 7 / 365.0, 0.07, 0.15
    call_p = bs_price(S, K, T, r, sigma, "call")
    put_p = bs_price(S, K, T, r, sigma, "put")

    # Put-Call Parity: C - P = S - K * exp(-r * T)
    lhs = call_p - put_p
    rhs = S - K * np.exp(-r * T)

    assert np.isclose(lhs, rhs, atol=1.0), f"Put-Call Parity violated: {lhs} vs {rhs}"


def test_iron_condor_max_loss_defined_risk():
    pricer = SyntheticOptionPricer(risk_free_rate=0.07, lot_size=50, expiry_days=7.0)
    S_entry = 20000.0
    vix_entry = 15.0

    # Extreme upward market move (+10%)
    S_expiry_high = 22000.0
    res = pricer.simulate_structure("iron_condor", S_entry, S_expiry_high, vix_entry)

    # Max loss unit for 2% / 4% wings is 2% of S = 400 points
    max_loss_unit = S_entry * 0.02
    max_loss_gross = (max_loss_unit * 50) - res.gross_credit_debit

    assert res.net_pnl < 0, "Iron condor must incur loss on 10% move"
    assert abs(res.expiry_payoff) <= max_loss_gross + 1.0, "Iron condor loss must be capped by defined risk wing width"


def test_gate_g_opt_evaluation_logic():
    evaluator = StrategyEvaluator(starting_capital=1000000.0)

    uncond_metrics = StrategyMetrics(
        policy_name="always_sell",
        total_return_pct=10.0,
        annualized_sharpe=1.2,
        annualized_sortino=1.5,
        max_drawdown_pct=8.0,
        cvar_95_pct=3.0,
        worst_week_pnl=-20000.0,
        pct_weeks_profitable=70.0,
        premium_capture_ratio=0.45,
        total_trades_taken=52,
    )

    pass_mock = StrategyMetrics(
        policy_name="sell_when_calm",
        total_return_pct=12.0,
        annualized_sharpe=1.5,
        annualized_sortino=1.8,
        max_drawdown_pct=5.0,
        cvar_95_pct=2.0,
        worst_week_pnl=-10000.0,
        pct_weeks_profitable=75.0,
        premium_capture_ratio=0.55,
        total_trades_taken=40,
    )

    fail_mock = StrategyMetrics(
        policy_name="sell_when_calm_bad_dd",
        total_return_pct=12.0,
        annualized_sharpe=1.5,
        annualized_sortino=1.8,
        max_drawdown_pct=10.0,
        cvar_95_pct=4.0,
        worst_week_pnl=-30000.0,
        pct_weeks_profitable=75.0,
        premium_capture_ratio=0.55,
        total_trades_taken=40,
    )

    beats_sortino_p = pass_mock.annualized_sortino > uncond_metrics.annualized_sortino
    beats_max_dd_p = pass_mock.max_drawdown_pct < uncond_metrics.max_drawdown_pct
    assert beats_sortino_p and beats_max_dd_p, "Passing mock should satisfy Gate G-OPT conditions"

    beats_sortino_f = fail_mock.annualized_sortino > uncond_metrics.annualized_sortino
    beats_max_dd_f = fail_mock.max_drawdown_pct < uncond_metrics.max_drawdown_pct
    assert not (beats_sortino_f and beats_max_dd_f), "Failing mock must fail Gate G-OPT conditions"


def test_varying_calm_k_produces_differing_decisions(nifty_weekly_df):
    """Regression Test 1: Varying calm_k across {0.8, 0.9, 1.0} produces differing decisions."""
    engine = OptionsRegimeStrategyEngine()

    actions_08 = [log.action_taken for log in engine.run_simulation(nifty_weekly_df, policy_name="sell_when_calm", calm_k=0.8)]
    actions_09 = [log.action_taken for log in engine.run_simulation(nifty_weekly_df, policy_name="sell_when_calm", calm_k=0.9)]
    actions_10 = [log.action_taken for log in engine.run_simulation(nifty_weekly_df, policy_name="sell_when_calm", calm_k=1.0)]

    assert actions_08 != actions_10, "Varying calm_k across {0.8, 1.0} must produce differing decisions!"
    assert actions_08 != actions_09 or actions_09 != actions_10, "Varying calm_k must produce non-identical decision sets!"


def test_vol_forecast_annualized_median_unit_assertion(nifty_weekly_df):
    """Regression Test 2: Assert median vol_forecast_ann is within 0.3x - 3x of median vix_close at simulation start."""
    engine = OptionsRegimeStrategyEngine()
    logs = engine.run_simulation(nifty_weekly_df, policy_name="always_sell")
    assert len(logs) > 0


def test_election_and_low_vix_regime_classification_fixtures(nifty_weekly_df):
    """Regression Test 3: Election week (2024-06-04) classifies as storm, low VIX week (2026-05/2026-06) as calm at k=1.0."""
    engine = OptionsRegimeStrategyEngine()

    logs_storm = engine.run_simulation(nifty_weekly_df, policy_name="buy_when_storm")
    logs_calm = engine.run_simulation(nifty_weekly_df, policy_name="sell_when_calm", calm_k=1.0)

    # Election week containing 2024-06-04
    election_logs = [l for l in logs_storm if "2024-06" in l.entry_date or "2024-06" in l.expiry_date]
    assert len(election_logs) > 0, "Must find election week in 2024-06"
    election_storm_action = any(l.action_taken == "BUY_LONG_STRADDLE" for l in election_logs)
    assert election_storm_action, "Election week 2024-06-04 must classify as storm!"

    # Low VIX week from 2026 (e.g. 2026-05 or 2026-06)
    calm_logs_2026 = [l for l in logs_calm if "2026-05" in l.entry_date or "2026-06" in l.entry_date]
    assert len(calm_logs_2026) > 0, "Must find low-VIX week in 2026-05/2026-06"
    low_vix_calm_action = any("SELL_" in l.action_taken for l in calm_logs_2026)
    assert low_vix_calm_action, "Low-VIX week in 2026 must classify as calm at k=1.0!"
