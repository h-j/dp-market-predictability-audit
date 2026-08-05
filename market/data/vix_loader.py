"""
India VIX Ingestion & Feature Engineering Module.

Globs, concatenates, dedupes, and validates historical India VIX data.
Computes point-in-time expanding features (vix_close, vix_change_5d, vix_vs_20d_ma, vix_percentile_252d)
without lookahead bias and merges them into asset DataFrames.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class IndiaVIXLoader:
    """
    Ingests and validates India VIX historical CSV files.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir = Path(data_dir)

    def load_vix_data(self) -> pd.DataFrame:
        """
        Glob matching files, concatenate, dedupe, validate, and compute point-in-time features.
        """
        patterns = ["*india_vix*.csv", "india_vix*.csv", "india_vix.csv"]
        matching_files = []
        for pat in patterns:
            for p in self.data_dir.glob(pat):
                if p not in matching_files:
                    matching_files.append(p)

        if not matching_files:
            raise FileNotFoundError(f"No India VIX CSV files found in {self.data_dir}")

        dfs = []
        for file_path in sorted(matching_files):
            try:
                raw_df = pd.read_csv(file_path)
                # Clean column headers
                raw_df.columns = [str(c).strip().lower() for c in raw_df.columns]

                # Map column names
                date_col = next((c for c in raw_df.columns if "date" in c), None)
                close_col = next((c for c in raw_df.columns if "close" in c and "prev" not in c), None)

                if date_col and close_col:
                    subset = pd.DataFrame()
                    subset["date"] = pd.to_datetime(raw_df[date_col], format="mixed", errors="coerce")
                    subset["vix_close"] = pd.to_numeric(raw_df[close_col], errors="coerce")
                    subset = subset.dropna(subset=["date", "vix_close"])
                    dfs.append(subset)
            except Exception as exc:
                print(f"⚠ Warning: Could not parse VIX file {file_path}: {exc}")

        if not dfs:
            raise ValueError("Failed to parse any valid India VIX data.")

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

        # Range Validation [5.0, 100.0]
        invalid_mask = (combined["vix_close"] < 5.0) | (combined["vix_close"] > 100.0)
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            print(f"⚠ Dropping {invalid_count} VIX rows out of plausible range [5, 100]")
            combined = combined[~invalid_mask].reset_index(drop=True)

        # Continuity check
        date_diffs = combined["date"].diff().dt.days
        max_gap = date_diffs.max()
        print(f"✓ Ingested India VIX history: {len(combined)} trading days ({combined['date'].min().strftime('%Y-%m-%d')} to {combined['date'].max().strftime('%Y-%m-%d')}, max gap: {max_gap} days)")

        # Derive Point-In-Time Features
        vix_close = combined["vix_close"]
        combined["vix_change_5d"] = ((vix_close - vix_close.shift(5)) / vix_close.shift(5)).fillna(0.0).round(4)
        vix_ma20 = vix_close.rolling(window=20, min_periods=1).mean()
        combined["vix_vs_20d_ma"] = (vix_close / vix_ma20).fillna(1.0).round(4)

        # Expanding-window 252-day percentile rank (no lookahead)
        percentiles = []
        for i in range(len(combined)):
            window_start = max(0, i - 251)
            hist_window = vix_close.iloc[window_start : i + 1]
            rank = (hist_window <= vix_close.iloc[i]).mean()
            percentiles.append(round(rank, 4))
        combined["vix_percentile_252d"] = percentiles

        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
        return combined

    def merge_vix_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge point-in-time VIX features into an asset DataFrame by date.
        """
        vix_df = self.load_vix_data()
        df_merged = df.copy()

        # Convert date column to string format for consistent joining
        df_merged["date"] = pd.to_datetime(df_merged["date"]).dt.strftime("%Y-%m-%d")

        vix_cols = ["date", "vix_close", "vix_change_5d", "vix_vs_20d_ma", "vix_percentile_252d"]
        df_merged = pd.merge(df_merged, vix_df[vix_cols], on="date", how="left")

        # Forward fill and backward fill missing VIX values
        for col in ["vix_close", "vix_change_5d", "vix_vs_20d_ma", "vix_percentile_252d"]:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].ffill().bfill().fillna(15.0 if col == "vix_close" else (1.0 if col == "vix_vs_20d_ma" else 0.5))

        print(f"✓ Merged India VIX features into asset dataset ({len(df_merged)} rows)")
        return df_merged
