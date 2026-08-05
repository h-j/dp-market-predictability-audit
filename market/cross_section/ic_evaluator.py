"""
Cross-Sectional Information Coefficient (IC) Evaluator & Significance Engine.

Computes monthly Spearman Rank IC, 95% stationary bootstrap CIs, t-statistics,
and empirical permutation null distribution (1,000 rank shuffles per month).
Evaluates Pre-Registered Gate G-XS1.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class ICSignalSummary:
    signal_name: str
    mean_ic: float
    std_ic: float
    t_stat: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    null_p975: float
    is_statistically_significant: bool
    q5_q1_spread_annualized_pct: float


class CrossSectionalICEvaluator:
    """
    Evaluates cross-sectional Rank IC performance, bootstrap confidence intervals, and permutation nulls.
    """

    def __init__(self, n_bootstrap: int = 2000, n_permutations: int = 1000, seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.seed = seed

    @staticmethod
    def compute_monthly_ic(df_ranks: pd.DataFrame, signal_col: str) -> float:
        """
        Compute Spearman rank correlation for a single rebalance month.
        """
        if df_ranks.empty or signal_col not in df_ranks.columns or "fwd_return" not in df_ranks.columns:
            return 0.0

        valid_df = df_ranks[[signal_col, "fwd_return"]].dropna()
        if len(valid_df) < 5:
            return 0.0

        corr, _ = spearmanr(valid_df[signal_col], valid_df["fwd_return"])
        return float(corr) if not np.isnan(corr) else 0.0

    def evaluate_signal_ic(
        self, monthly_ranks: Dict[pd.Timestamp, pd.DataFrame], signal_col: str
    ) -> Tuple[List[float], ICSignalSummary]:
        """
        Evaluate full time-series of monthly ICs for a signal across all rebalance dates.
        Computes mean IC, t-stat, bootstrap 95% CI, and empirical 97.5th percentile of permutation null.
        """
        sorted_dates = sorted(monthly_ranks.keys())
        monthly_ics: List[float] = []

        q5_minus_q1_monthly: List[float] = []

        for date in sorted_dates:
            df = monthly_ranks[date]
            ic = self.compute_monthly_ic(df, signal_col)
            monthly_ics.append(ic)

            # Quintile spread computation
            if not df.empty and signal_col in df.columns and len(df) >= 5:
                df_q = df.copy()
                df_q["q"] = pd.qcut(df_q[signal_col], q=5, labels=[1, 2, 3, 4, 5])
                q1 = df_q[df_q["q"] == 1]["fwd_return"].mean()
                q5 = df_q[df_q["q"] == 5]["fwd_return"].mean()
                q5_minus_q1_monthly.append(q5 - q1)

        ics = np.array(monthly_ics)
        mean_ic = float(np.mean(ics)) if len(ics) > 0 else 0.0
        std_ic = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.01

        # t-statistic: mean_ic / (std_ic / sqrt(N))
        se_ic = std_ic / np.sqrt(len(ics)) if len(ics) > 0 else 0.01
        t_stat = mean_ic / se_ic if se_ic > 1e-6 else 0.0

        # Stationary Bootstrap (2,000 resamples over months)
        rng = np.random.default_rng(self.seed)
        boot_means = []
        for _ in range(self.n_bootstrap):
            resample = rng.choice(ics, size=len(ics), replace=True)
            boot_means.append(np.mean(resample))

        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        # Permutation Null Distribution (1,000 shuffles per month)
        null_means = []
        for _ in range(self.n_permutations):
            perm_ics = []
            for date in sorted_dates:
                df = monthly_ranks[date].copy()
                if not df.empty and signal_col in df.columns and len(df) >= 5:
                    shuffled_ranks = rng.permutation(df[signal_col].values)
                    corr, _ = spearmanr(shuffled_ranks, df["fwd_return"].values)
                    perm_ics.append(corr if not np.isnan(corr) else 0.0)
            null_means.append(np.mean(perm_ics))

        null_p975 = float(np.percentile(null_means, 97.5))

        # Gate G-XS1 check: mean_ic > 0 AND ci_lower > 0 AND mean_ic > null_p975
        is_sig = (mean_ic > 0.0) and (ci_lower > 0.0) and (mean_ic > null_p975)

        # Annualized Q5 - Q1 spread
        mean_monthly_spread = float(np.mean(q5_minus_q1_monthly)) if q5_minus_q1_monthly else 0.0
        annualized_spread_pct = mean_monthly_spread * 12.0 * 100.0

        summary = ICSignalSummary(
            signal_name=signal_col,
            mean_ic=round(mean_ic, 4),
            std_ic=round(std_ic, 4),
            t_stat=round(t_stat, 2),
            bootstrap_ci_lower=round(ci_lower, 4),
            bootstrap_ci_upper=round(ci_upper, 4),
            null_p975=round(null_p975, 4),
            is_statistically_significant=is_sig,
            q5_q1_spread_annualized_pct=round(annualized_spread_pct, 2),
        )

        return monthly_ics, summary
