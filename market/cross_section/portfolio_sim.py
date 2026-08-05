"""
Cross-Sectional Portfolio Simulator & Transaction Cost Engine.

Simulates monthly quintile portfolio performance (Q1 to Q5) and Long/Short spread.
Deducts 0.25% one-way transaction cost on turnover per rebalance.
Computes CAGR, Sharpe Ratio, Max Drawdown, and Active Return vs NIFTY benchmark.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class MonthlyPortfolioLog:
    rebalance_date: str
    q1_gross_return: float
    q5_gross_return: float
    q5_turnover: float
    q5_net_return: float
    nifty_return: float
    active_net_return: float


@dataclass
class PortfolioStats:
    total_months: int
    cagr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    mean_monthly_turnover: float
    total_transaction_costs_pct: float
    active_return_pct: float
    win_rate_pct: float


class PortfolioSimulator:
    """
    Simulates monthly rebalanced equal-weight quintile portfolios with 0.25% one-way transaction costs.
    """

    def __init__(self, one_way_cost_pct: float = 0.0025):  # 0.25% one-way cost
        self.one_way_cost_pct = one_way_cost_pct

    def calculate_turnover(self, prev_weights: Dict[str, float], curr_weights: Dict[str, float]) -> float:
        """
        Calculate one-way portfolio turnover: sum(|w_curr - w_prev|) / 2.0 or total weight changes.
        """
        all_tickers = set(prev_weights.keys()).union(set(curr_weights.keys()))
        if not all_tickers:
            return 0.0

        turnover = sum(abs(curr_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers)
        return float(turnover)

    def simulate_signal_portfolios(
        self, monthly_ranks: Dict[pd.Timestamp, pd.DataFrame], signal_col: str = "composite"
    ) -> Tuple[List[MonthlyPortfolioLog], PortfolioStats, PortfolioStats]:
        """
        Simulate monthly top quintile (Q5) vs bottom quintile (Q1) portfolios.
        Returns: (logs, q5_stats, q1_stats)
        """
        sorted_dates = sorted(monthly_ranks.keys())
        logs: List[MonthlyPortfolioLog] = []

        prev_q5_weights: Dict[str, float] = {}

        q5_net_returns: List[float] = []
        q1_gross_returns: List[float] = []

        for date in sorted_dates:
            df = monthly_ranks[date].copy()
            if df.empty or signal_col not in df.columns:
                continue

            n = len(df)
            if n < 5:
                continue

            # Assign quintiles: Q1 (bottom 20%) to Q5 (top 20%)
            df["quintile"] = pd.qcut(df[signal_col], q=5, labels=[1, 2, 3, 4, 5])

            df_q1 = df[df["quintile"] == 1]
            df_q5 = df[df["quintile"] == 5]

            q1_gross = float(df_q1["fwd_return"].mean())
            q5_gross = float(df_q5["fwd_return"].mean())

            # Current equal-weight Q5 holdings
            q5_tickers = df_q5["ticker"].tolist()
            w_unit = 1.0 / len(q5_tickers) if q5_tickers else 0.0
            curr_q5_weights = {t: w_unit for t in q5_tickers}

            # Turnover and cost deduction
            turnover = self.calculate_turnover(prev_q5_weights, curr_q5_weights)
            cost_deduction = turnover * self.one_way_cost_pct
            q5_net = q5_gross - cost_deduction

            nifty_ret = 0.0
            if "nifty_fwd_return" in df.columns:
                nifty_ret = float(df["nifty_fwd_return"].iloc[0])

            active_net = q5_net - nifty_ret

            logs.append(
                MonthlyPortfolioLog(
                    rebalance_date=date.strftime("%Y-%m-%d"),
                    q1_gross_return=round(q1_gross * 100, 2),
                    q5_gross_return=round(q5_gross * 100, 2),
                    q5_turnover=round(turnover, 4),
                    q5_net_return=round(q5_net * 100, 2),
                    nifty_return=round(nifty_ret * 100, 2),
                    active_net_return=round(active_net * 100, 2),
                )
            )

            q5_net_returns.append(q5_net)
            q1_gross_returns.append(q1_gross)
            prev_q5_weights = curr_q5_weights

        q5_stats = self.compute_portfolio_stats(q5_net_returns, [l.q5_turnover for l in logs])
        q1_stats = self.compute_portfolio_stats(q1_gross_returns, [0.0] * len(logs))

        return logs, q5_stats, q1_stats

    @staticmethod
    def compute_portfolio_stats(monthly_returns: List[float], monthly_turnovers: List[float]) -> PortfolioStats:
        """
        Compute CAGR, Sharpe ratio, Max Drawdown from monthly net returns.
        """
        if not monthly_returns:
            return PortfolioStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        rets = np.array(monthly_returns)
        n_months = len(rets)

        # Cumulative Growth
        cum_growth = np.cumprod(1.0 + rets)
        total_return = cum_growth[-1] - 1.0

        # Annualized CAGR (assuming 12 months per year)
        years = n_months / 12.0
        cagr = ((1.0 + total_return) ** (1.0 / years) - 1.0) if (years > 0 and total_return > -1.0) else 0.0

        # Annualized Sharpe Ratio (rf = 0.06 / 12)
        rf_monthly = 0.06 / 12.0
        excess_rets = rets - rf_monthly
        mean_excess = np.mean(excess_rets)
        std_rets = np.std(rets, ddof=1) if len(rets) > 1 else 0.01

        sharpe = (mean_excess / std_rets) * np.sqrt(12.0) if std_rets > 1e-6 else 0.0

        # Max Drawdown
        peak = np.maximum.accumulate(cum_growth)
        drawdowns = (cum_growth - peak) / peak
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        mean_turnover = float(np.mean(monthly_turnovers)) if monthly_turnovers else 0.0
        total_costs = sum(t * 0.0025 for t in monthly_turnovers) * 100.0

        win_rate = float(np.mean(rets > 0)) * 100.0 if len(rets) > 0 else 0.0

        return PortfolioStats(
            total_months=n_months,
            cagr_pct=round(cagr * 100, 2),
            sharpe_ratio=round(float(sharpe), 2),
            max_drawdown_pct=round(abs(max_dd) * 100, 2),
            mean_monthly_turnover=round(mean_turnover, 2),
            total_transaction_costs_pct=round(total_costs, 2),
            active_return_pct=round(cagr * 100 - 6.0, 2),  # Active return vs ~6% risk-free
            win_rate_pct=round(win_rate, 2),
        )
