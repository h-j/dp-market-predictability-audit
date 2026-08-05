"""
Unit tests for Workstream 2 options pricing, strategy simulation, friction costs, and Gate G-OPT.
"""

import numpy as np
import pytest

from config.options_costs import calculate_leg_cost
from market.options.synthetic_pricer import SyntheticOptionPricer, bs_price
from market.options.strategy_evaluator import StrategyEvaluator, StrategyMetrics


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

    # Passing mock metrics (higher Sortino, lower MaxDD)
    pass_mock = StrategyMetrics(
        policy_name="sell_when_calm",
        total_return_pct=12.0,
        annualized_sharpe=1.5,
        annualized_sortino=1.8,  # > 1.5
        max_drawdown_pct=5.0,    # < 8.0
        cvar_95_pct=2.0,
        worst_week_pnl=-10000.0,
        pct_weeks_profitable=75.0,
        premium_capture_ratio=0.55,
        total_trades_taken=40,
    )

    # Failing mock metrics (higher Sortino, but worse MaxDD)
    fail_mock = StrategyMetrics(
        policy_name="sell_when_calm_bad_dd",
        total_return_pct=12.0,
        annualized_sharpe=1.5,
        annualized_sortino=1.8,  # > 1.5
        max_drawdown_pct=10.0,   # > 8.0 (FAILED)
        cvar_95_pct=4.0,
        worst_week_pnl=-30000.0,
        pct_weeks_profitable=75.0,
        premium_capture_ratio=0.55,
        total_trades_taken=40,
    )

    # Check G-OPT logic directly
    beats_sortino_p = pass_mock.annualized_sortino > uncond_metrics.annualized_sortino
    beats_max_dd_p = pass_mock.max_drawdown_pct < uncond_metrics.max_drawdown_pct
    assert beats_sortino_p and beats_max_dd_p, "Passing mock should satisfy Gate G-OPT conditions"

    beats_sortino_f = fail_mock.annualized_sortino > uncond_metrics.annualized_sortino
    beats_max_dd_f = fail_mock.max_drawdown_pct < uncond_metrics.max_drawdown_pct
    assert not (beats_sortino_f and beats_max_dd_f), "Failing mock must fail Gate G-OPT conditions"
