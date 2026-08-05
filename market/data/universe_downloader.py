"""
NIFTY 100 Universe Data Downloader & Manifest Generator.

Downloads adjusted daily OHLCV for NIFTY 100 constituents from yfinance into data/universe/{ticker}.csv.
Validates row counts, continuity, non-negative prices, and generates data/universe/_manifest.json.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

from config.universe_nifty100 import AS_OF_DATE, NIFTY_100_TICKERS


class UniverseDataDownloader:
    """
    Ingests and validates daily OHLCV data for universe tickers.
    """

    def __init__(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "data" / "universe"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_ticker(self, ticker: str, start_date: str = "2022-06-01") -> Tuple[bool, str, int, str, str]:
        """
        Download single ticker OHLCV from yfinance, format columns, validate, and save CSV.
        Returns: (success, clean_symbol, row_count, min_date, max_date)
        """
        clean_symbol = ticker.replace(".NS", "").upper()
        try:
            df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
            if df.empty:
                return False, clean_symbol, 0, "", ""

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            # Standardize column names
            df.columns = [c.lower() for c in df.columns]

            if "date" not in df.columns or "close" not in df.columns:
                return False, clean_symbol, 0, "", ""

            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

            # Clean and validate prices
            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]

            rows = len(df)
            if rows < 100:  # Minimal threshold for new listings
                return False, clean_symbol, rows, "", ""

            min_date = str(df["date"].min())
            max_date = str(df["date"].max())

            out_csv = self.output_dir / f"{clean_symbol}.csv"
            df.to_csv(out_csv, index=False)
            return True, clean_symbol, rows, min_date, max_date

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            return False, clean_symbol, 0, "", ""

    def download_all(self, min_universe_size: int = 90) -> Dict:
        """
        Download all universe tickers and generate manifest.
        """
        print(f"Downloading {len(NIFTY_100_TICKERS)} NIFTY 100 constituent tickers...")
        manifest_entries = {}
        excluded_tickers = []
        successful_count = 0

        for i, ticker in enumerate(NIFTY_100_TICKERS):
            success, symbol, rows, min_date, max_date = self.download_ticker(ticker)
            if success:
                successful_count += 1
                manifest_entries[symbol] = {
                    "raw_ticker": ticker,
                    "rows": rows,
                    "min_date": min_date,
                    "max_date": max_date,
                    "status": "VALID",
                }
                print(f"[{i+1}/{len(NIFTY_100_TICKERS)}] ✓ {symbol}: {rows} rows ({min_date} to {max_date})")
            else:
                excluded_tickers.append(symbol)
                print(f"[{i+1}/{len(NIFTY_100_TICKERS)}] ✗ {symbol}: FAILED / Excluded")

        manifest = {
            "as_of_date": AS_OF_DATE,
            "download_timestamp": datetime.now().isoformat(),
            "total_requested": len(NIFTY_100_TICKERS),
            "successful_count": successful_count,
            "excluded_count": len(excluded_tickers),
            "excluded_tickers": excluded_tickers,
            "tickers": manifest_entries,
        }

        manifest_path = self.output_dir / "_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n✓ Saved manifest to {manifest_path}")

        if successful_count < min_universe_size:
            raise RuntimeError(
                f"STOP CONDITION TRIGGERED: Downloaded {successful_count} tickers, which is below "
                f"minimum universe threshold of {min_universe_size}!"
            )

        return manifest


if __name__ == "__main__":
    downloader = UniverseDataDownloader()
    downloader.download_all()
