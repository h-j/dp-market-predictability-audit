"""
Cross-Sectional Signal Computation Engine.

Computes point-in-time percentile-ranked signals across eligible NIFTY 100 constituents per monthly rebalance date.
Pure functions operating on pandas DataFrames; zero network calls and zero external framework dependencies.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_universe_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all committed constituent CSV files from data/universe/.
    Returns dict mapping ticker symbol -> cleaned DataFrame sorted by date.
    """
    universe_dir = Path(data_dir) / "universe"
    dfs: Dict[str, pd.DataFrame] = {}

    for csv_file in sorted(universe_dir.glob("*.csv")):
        ticker = csv_file.stem.upper()
        df = pd.read_csv(csv_file)
        if "date" in df.columns and "close" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            dfs[ticker] = df

    return dfs


def load_nifty_benchmark(data_dir: Path) -> pd.DataFrame:
    """
    Load NIFTY index benchmark series from data/nifty_daily_3y.csv or nifty_enriched.
    """
    p_enriched = Path(data_dir) / "nifty_enriched_daily_3y.csv"
    p_raw = Path(data_dir) / "nifty_daily_3y.csv"
    target_p = p_enriched if p_enriched.exists() else p_raw

    df = pd.read_csv(target_p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_monthly_rebalance_dates(universe_dfs: Dict[str, pd.DataFrame], start_date: str = "2023-01-01") -> List[pd.Timestamp]:
    """
    Extract last trading day of each calendar month across the common calendar.
    """
    # Combine dates from reliable large cap (e.g. RELIANCE or TCS)
    sample_df = universe_dfs.get("RELIANCE", next(iter(universe_dfs.values())))
    df_filtered = sample_df[sample_df["date"] >= pd.to_datetime(start_date)].copy()

    df_filtered["year_month"] = df_filtered["date"].dt.to_period("M")
    rebalance_dates = df_filtered.groupby("year_month")["date"].max().tolist()
    return sorted(rebalance_dates)


class CrossSectionalSignalEngine:
    """
    Computes cross-sectional signals and percentile ranks at date t strictly using data <= t.
    """

    def __init__(
        self,
        universe_dfs: Dict[str, pd.DataFrame],
        nifty_df: Optional[pd.DataFrame] = None,
        history_cutoff_months: int = 15,
    ):
        self.universe_dfs = universe_dfs
        self.nifty_df = nifty_df
        self.history_cutoff_months = history_cutoff_months

    def compute_raw_signals_for_ticker(
        self, ticker: str, rebalance_date: pd.Timestamp
    ) -> Optional[Dict[str, float]]:
        """
        Compute 5 raw signals for single ticker at rebalance_date using history <= rebalance_date.
        """
        df = self.universe_dfs.get(ticker)
        if df is None:
            return None

        # Point-in-time filter: strictly <= rebalance_date
        df_hist = df[df["date"] <= rebalance_date].copy()
        if len(df_hist) < 30:  # Need minimal history
            return None

        # Check history length at rebalance date (>= history_cutoff_months rule)
        if self.history_cutoff_months > 0:
            min_date = df_hist["date"].min()
            days_history = (rebalance_date - min_date).days
            if days_history < self.history_cutoff_months * 30:
                return None

        closes = df_hist["close"].values
        volumes = df_hist["volume"].values if "volume" in df_hist.columns else np.ones(len(closes))

        # 1. mom_6m1m: 6-month return skipping trailing 1 month (21d to 126d)
        c_curr = closes[-1]
        c_1m = closes[-22] if len(closes) >= 22 else closes[-1]
        c_6m = closes[-126] if len(closes) >= 126 else closes[0]

        mom_6m1m = (c_1m / c_6m - 1.0) if c_6m > 0 else 0.0

        # 2. rev_1m: Negative of trailing 1-month return
        rev_1m = 1.0 - (c_curr / c_1m) if c_1m > 0 else 0.0

        # 3. rs_nifty_3m: 3-month return relative to NIFTY
        c_3m = closes[-63] if len(closes) >= 63 else closes[0]
        ret_3m_stock = (c_curr / c_3m - 1.0) if c_3m > 0 else 0.0

        ret_3m_nifty = 0.0
        if self.nifty_df is not None:
            nifty_hist = self.nifty_df[self.nifty_df["date"] <= rebalance_date]["close"].values
            if len(nifty_hist) >= 63:
                ret_3m_nifty = (nifty_hist[-1] / nifty_hist[-63]) - 1.0

        rs_nifty_3m = ret_3m_stock - ret_3m_nifty

        # 4. vol_3m_inv: Inverse of 3-month daily return volatility
        daily_returns_3m = np.diff(closes[-63:]) / closes[-63:-1] if len(closes) >= 64 else np.diff(closes) / closes[:-1]
        vol_3m = np.std(daily_returns_3m) if len(daily_returns_3m) > 5 else 0.01
        vol_3m_inv = (1.0 / vol_3m) if vol_3m > 1e-6 else 100.0

        # 5. vol_trend: Ratio of 1-month avg volume to 6-month avg volume
        v_1m = np.mean(volumes[-21:]) if len(volumes) >= 21 else np.mean(volumes)
        v_6m = np.mean(volumes[-126:]) if len(volumes) >= 126 else np.mean(volumes)
        vol_trend = (v_1m / v_6m) if v_6m > 0 else 1.0

        return {
            "mom_6m1m": float(mom_6m1m),
            "rev_1m": float(rev_1m),
            "rs_nifty_3m": float(rs_nifty_3m),
            "vol_3m_inv": float(vol_3m_inv),
            "vol_trend": float(vol_trend),
        }

    def compute_cross_section_ranks(self, rebalance_date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute cross-sectional percentile ranks (0.0 to 1.0) for all eligible stocks at rebalance_date.
        Includes forward 1-month return (target) starting strictly after rebalance_date.
        """
        raw_records = []

        for ticker in sorted(self.universe_dfs.keys()):
            df_full = self.universe_dfs[ticker]
            df_hist = df_full[df_full["date"] <= rebalance_date]
            if df_hist.empty:
                continue

            # Check forward return starting strictly after rebalance_date
            df_fwd = df_full[df_full["date"] > rebalance_date].copy()
            if len(df_fwd) < 15:  # Need at least ~15 trading days in forward month
                continue

            # Forward 1-month return (end of next month or ~21 trading days)
            fwd_close_entry = df_hist["close"].iloc[-1]
            fwd_close_exit = df_fwd["close"].iloc[min(21, len(df_fwd) - 1)]
            fwd_return = (fwd_close_exit / fwd_close_entry) - 1.0

            sigs = self.compute_raw_signals_for_ticker(ticker, rebalance_date)
            if sigs is not None:
                sigs["ticker"] = ticker
                sigs["fwd_return"] = float(fwd_return)
                raw_records.append(sigs)

        if not raw_records:
            return pd.DataFrame()

        df_raw = pd.DataFrame(raw_records)
        signal_cols = ["mom_6m1m", "rev_1m", "rs_nifty_3m", "vol_3m_inv", "vol_trend"]

        # Percentile rank (0.0 to 1.0) cross-sectionally
        df_ranks = pd.DataFrame({"ticker": df_raw["ticker"], "fwd_return": df_raw["fwd_return"]})
        for col in signal_cols:
            df_ranks[col] = df_raw[col].rank(pct=True, method="average")

        # Composite signal: equal-weighted mean of ranks
        df_ranks["composite"] = df_ranks[signal_cols].mean(axis=1)

        return df_ranks
