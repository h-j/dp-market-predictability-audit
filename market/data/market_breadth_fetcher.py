"""
Market Breadth Fetcher Module.

Computes genuine constituent-level daily market breadth features for NIFTY 50
(Advance/Decline ratio, % above 50DMA, New Highs minus Lows ratio, Composite Breadth Score, and Breadth State).
Caches metrics locally for deterministic, walk-forward historical replay.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("market_breadth_fetcher")
logger.setLevel(logging.INFO)

# Representative NIFTY 50 Constituents for Market/Sector Breadth
NIFTY_50_TICKERS: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "HCLTECH.NS",
    "SUNPHARMA.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "TITAN.NS",
    "BAJFINANCE.NS",
    "ULTRACEMCO.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "TATASTEEL.NS",
    "M&M.NS",
    "INDUSINDBK.NS",
    "ONGC.NS",
    "ADANIENT.NS",
    "JSWSTEEL.NS",
    "COALINDIA.NS",
    "GRASIM.NS",
    "HDFCLIFE.NS",
    "SBIN.NS",
    "BAJAJFINSV.NS",
    "BRITANNIA.NS",
    "TECHM.NS",
    "CIPLA.NS",
    "HINDUNILVR.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "APOLLOHOSP.NS",
    "SBILIFE.NS",
    "BPCL.NS",
    "TATACONSUM.NS",
    "WIPRO.NS",
    "BEL.NS",
    "TRENT.NS",
    "SHRIRAMFIN.NS",
    "ADANIPORTS.NS",
    "NESTLEIND.NS",
    "HINDALCO.NS",
    "BAJAJ-AUTO.NS",
]


class MarketBreadthFetcher:
    """
    Fetcher & Calculator for daily Market Breadth.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = (
                Path(__file__).parent.parent.parent
                / "data"
                / "market_data"
                / "breadth"
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "nifty50_breadth_daily_3y.parquet"

    def fetch_and_compute_breadth(
        self,
        start_date: str = "2022-10-01",  # Warm-up period for 50DMA
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch constituent daily OHLCV and compute genuine market breadth metrics.
        Returns a DataFrame with date, net_advances_pct, pct_above_50dma,
        highs_minus_lows_pct, composite_breadth_score, and market_breadth_state.
        """
        if self.cache_file.exists() and not force_refresh:
            try:
                df = pd.read_parquet(self.cache_file)
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                logger.info(
                    f"Loaded {len(df)} market breadth records from cache {self.cache_file}"
                )
                return df
            except Exception as exc:
                logger.warning(
                    f"Failed to load cached market breadth parquet ({exc}), recalculating..."
                )

        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        logger.info(
            f"Downloading {len(NIFTY_50_TICKERS)} NIFTY 50 constituent OHLCV from {start_date} to {end_date}..."
        )

        try:
            raw_data = yf.download(
                NIFTY_50_TICKERS,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logger.error(f"Failed to download constituent OHLCV from yfinance: {exc}")
            raise

        if raw_data.empty or "Close" not in raw_data:
            raise ValueError("yfinance returned empty data for constituents.")

        close_df = raw_data["Close"]
        high_df = raw_data["High"] if "High" in raw_data else close_df
        low_df = raw_data["Low"] if "Low" in raw_data else close_df

        # Calculate daily constituent returns (net advances/declines)
        daily_returns = close_df.pct_change()
        advances = (daily_returns > 0.0).sum(axis=1)
        declines = (daily_returns < 0.0).sum(axis=1)
        total_active = advances + declines

        net_advances_pct = np.where(
            total_active > 0, (advances - declines) / total_active, 0.0
        )
        advance_decline_ratio = np.where(
            declines > 0, advances / declines, advances.astype(float)
        )

        # % of constituents above 50-day moving average
        sma_50 = close_df.rolling(window=50, min_periods=20).mean()
        above_50dma = (close_df > sma_50).sum(axis=1)
        total_valid_50dma = sma_50.notna().sum(axis=1)
        pct_above_50dma = np.where(
            total_valid_50dma > 0, above_50dma / total_valid_50dma, 0.5
        )

        # 20-day Highs minus Lows ratio
        high_20 = high_df.rolling(window=20, min_periods=10).max()
        low_20 = low_df.rolling(window=20, min_periods=10).min()
        new_highs = (high_df >= high_20).sum(axis=1)
        new_lows = (low_df <= low_20).sum(axis=1)
        total_valid_hl = high_df.notna().sum(axis=1)
        highs_minus_lows_pct = np.where(
            total_valid_hl > 0, (new_highs - new_lows) / total_valid_hl, 0.0
        )

        # Composite Market Breadth Score [0.0 to 1.0]
        norm_advances = (net_advances_pct + 1.0) / 2.0  # Mapped [-1, 1] -> [0, 1]
        norm_hl = (np.clip(highs_minus_lows_pct, -1.0, 1.0) + 1.0) / 2.0  # Mapped [-1, 1] -> [0, 1]

        composite_score = (
            0.40 * norm_advances + 0.40 * pct_above_50dma + 0.20 * norm_hl
        )
        composite_score = np.clip(composite_score, 0.0, 1.0)

        # Map to canonical breadth_state bucket names
        breadth_states = []
        for score in composite_score:
            if score >= 0.70:
                breadth_states.append("strongly_participatory")
            elif score >= 0.55:
                breadth_states.append("strengthened")
            elif score >= 0.45:
                breadth_states.append("mixed")
            elif score >= 0.30:
                breadth_states.append("weakened")
            else:
                breadth_states.append("deteriorated")

        breadth_df = pd.DataFrame(
            {
                "date": pd.to_datetime(close_df.index).strftime("%Y-%m-%d"),
                "advance_decline_ratio": np.round(advance_decline_ratio, 4),
                "net_advances_pct": np.round(net_advances_pct, 4),
                "pct_above_50dma": np.round(pct_above_50dma, 4),
                "highs_minus_lows_pct": np.round(highs_minus_lows_pct, 4),
                "composite_breadth_score": np.round(composite_score, 4),
                "market_breadth_state": breadth_states,
            }
        )

        breadth_df = breadth_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        try:
            breadth_df.to_parquet(self.cache_file, index=False)
            logger.info(
                f"Saved {len(breadth_df)} market breadth daily records to {self.cache_file}"
            )
        except Exception as exc:
            logger.warning(f"Could not save parquet cache: {exc}")

        return breadth_df
