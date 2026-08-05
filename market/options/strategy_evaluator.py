"""
Tail-Aware Options Strategy Evaluator & Pre-Registered Gate G-OPT Verification.

Computes total return, Sharpe, Sortino, Max Drawdown, CVaR(95%), worst single week P&L,
% profitable weeks, and premium capture ratio vs three benchmarks.
Evaluates pre-registered Gate G-OPT.
"""

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

from market.options.regime_strategy import WeeklyStrategyLog


HONESTY_CAVEATS_VERBATIM = (
    "Synthetic pricing assumes BS with VIX as ATM IV; real chains have skew, smile, and liquidity "
    "effects not modeled. Results are upper bounds on realism until replaced with actual option chain data. "
    "This simulation does not constitute a profitable-strategy claim."
)


@dataclass
class StrategyMetrics:
    policy_name: str
    total_return_pct: float
    annualized_sharpe: float
    annualized_sortino: float
    max_drawdown_pct: float
    cvar_95_pct: float
    worst_week_pnl: float
    pct_weeks_profitable: float
    premium_capture_ratio: float
    total_trades_taken: int
    gate_g_opt_passed: bool = False
    gate_g_opt_reason: str = ""


class StrategyEvaluator:
    """
    Evaluates options strategy weekly execution logs and tests Gate G-OPT.
    """

    def __init__(self, starting_capital: float = 1000000.0, risk_free_rate: float = 0.07):
        self.starting_capital = starting_capital
        self.rf_annual = risk_free_rate

    def evaluate_logs(
        self, logs: List[WeeklyStrategyLog], unconditional_baseline_metrics: StrategyMetrics = None
    ) -> StrategyMetrics:
        if not logs:
            return StrategyMetrics(
                policy_name="empty",
                total_return_pct=0.0,
                annualized_sharpe=0.0,
                annualized_sortino=0.0,
                max_drawdown_pct=0.0,
                cvar_95_pct=0.0,
                worst_week_pnl=0.0,
                pct_weeks_profitable=0.0,
                premium_capture_ratio=0.0,
                total_trades_taken=0,
            )

        policy_name = logs[0].policy_name
        pnls = np.array([log.pnl_rupees for log in logs])
        traded_logs = [log for log in logs if log.action_taken != "NO_TRADE"]
        trades_count = len(traded_logs)

        # Weekly return rates relative to capital base
        weekly_returns = pnls / self.starting_capital
        rf_weekly = (1.0 + self.rf_annual) ** (1.0 / 52.0) - 1.0
        excess_returns = weekly_returns - rf_weekly

        total_pnl = float(np.sum(pnls))
        total_return_pct = float((total_pnl / self.starting_capital) * 100.0)

        # Sharpe & Sortino
        std_ret = float(np.std(weekly_returns))
        sharpe = float((np.mean(excess_returns) / std_ret) * np.sqrt(52.0)) if std_ret > 0 else 0.0

        downside_returns = weekly_returns[weekly_returns < 0]
        downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 1e-6
        sortino = float((np.mean(excess_returns) / downside_std) * np.sqrt(52.0)) if downside_std > 0 else 0.0

        # Maximum Drawdown calculation on equity curve
        cum_equity = self.starting_capital + np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_equity)
        drawdowns = (peak - cum_equity) / peak
        max_dd_pct = float(np.max(drawdowns) * 100.0)

        # CVaR (95%)
        cvar_threshold = np.percentile(weekly_returns, 5.0)
        cvar_losses = weekly_returns[weekly_returns <= cvar_threshold]
        cvar_95_pct = float(-np.mean(cvar_losses) * 100.0) if len(cvar_losses) > 0 else 0.0

        worst_week_pnl = float(np.min(pnls))
        pct_profitable = float((np.sum(pnls > 0) / len(pnls)) * 100.0)

        # Premium Capture Ratio
        gross_premiums = np.array([log.trade_result.gross_credit_debit for log in traded_logs if log.trade_result and log.trade_result.gross_credit_debit > 0])
        total_gross_premium = float(np.sum(gross_premiums)) if len(gross_premiums) > 0 else 0.0
        premium_capture_ratio = float(total_pnl / total_gross_premium) if total_gross_premium > 0 else 0.0

        # Gate G-OPT Verification
        g_opt_passed = False
        g_opt_reason = ""

        if unconditional_baseline_metrics is not None:
            beats_sortino = sortino > unconditional_baseline_metrics.annualized_sortino
            beats_max_dd = max_dd_pct < unconditional_baseline_metrics.max_drawdown_pct

            if beats_sortino and beats_max_dd:
                g_opt_passed = True
                g_opt_reason = (
                    f"PASSED: Model policy beats unconditional baseline on Sortino "
                    f"({sortino:.2f} > {unconditional_baseline_metrics.annualized_sortino:.2f}) "
                    f"AND Max Drawdown ({max_dd_pct:.2f}% < {unconditional_baseline_metrics.max_drawdown_pct:.2f}%)."
                )
            else:
                g_opt_passed = False
                g_opt_reason = (
                    f"FAILED: Model policy did not beat unconditional baseline on both metrics simultaneously. "
                    f"Sortino: {sortino:.2f} vs {unconditional_baseline_metrics.annualized_sortino:.2f} (Beats: {beats_sortino}); "
                    f"MaxDD: {max_dd_pct:.2f}% vs {unconditional_baseline_metrics.max_drawdown_pct:.2f}% (Beats: {beats_max_dd})."
                )

        return StrategyMetrics(
            policy_name=policy_name,
            total_return_pct=round(total_return_pct, 2),
            annualized_sharpe=round(sharpe, 2),
            annualized_sortino=round(sortino, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            cvar_95_pct=round(cvar_95_pct, 2),
            worst_week_pnl=round(worst_week_pnl, 2),
            pct_weeks_profitable=round(pct_profitable, 2),
            premium_capture_ratio=round(premium_capture_ratio, 4),
            total_trades_taken=trades_count,
            gate_g_opt_passed=g_opt_passed,
            gate_g_opt_reason=g_opt_reason,
        )
