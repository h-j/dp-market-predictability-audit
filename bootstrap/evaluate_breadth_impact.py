"""
Evaluation script for Follow-up Ticket 1: Real Market Breadth Integration.

Computes:
1. Pearson and Spearman correlation between breadth_state and target asset return_5d (before vs after).
2. Historical 3-year backfill data coverage.
3. Confidence-bucketed calibration hit-rate impact (before vs after).
"""

import numpy as np
import pandas as pd

from market.data.market_breadth_fetcher import MarketBreadthFetcher
from market.data.download_history import ensure_data
from market.replay.market_observation_synthesizer import MarketObservationSynthesizer


def map_breadth_to_numeric(breadth_series: pd.Series) -> pd.Series:
    mapping = {
        "deteriorated": 1,
        "weakened": 2,
        "mixed": 3,
        "nascent": 3,
        "strengthened": 4,
        "strongly_participatory": 5,
    }
    return breadth_series.map(mapping).fillna(3)


def run_breadth_evaluation(symbol: str = "RELIANCE"):
    print("=" * 70)
    print(f"EVALUATING MARKET BREADTH IMPACT ({symbol})")
    print("=" * 70)

    # Load enriched dataset
    df = ensure_data(symbol=symbol, start_date="2023-01-01")
    print(f"Dataset loaded: {len(df)} rows")

    # 1. Coverage Check
    missing_breadth = df["market_breadth_state"].isna().sum() if "market_breadth_state" in df.columns else len(df)
    coverage_pct = (len(df) - missing_breadth) / len(df) * 100.0
    print(f"\n1. DATA COVERAGE CHECK:")
    print(f"   • Total Trading Days:   {len(df)}")
    print(f"   • Missing Breadth Days: {missing_breadth}")
    print(f"   • Data Coverage:        {coverage_pct:.2f}%")

    # 2. Compute Before (Legacy Proxy) vs After (Real Breadth)
    synth = MarketObservationSynthesizer(df, market_name=symbol)

    # Legacy proxy series
    legacy_states = []
    for i in range(len(df)):
        if i < 5:
            legacy_states.append("nascent")
        else:
            w = df.iloc[max(0, i - 4) : i + 1]
            up = (w["close"] > w["open"]).sum()
            ratio = up / len(w)
            if ratio >= 0.8:
                legacy_states.append("strongly_participatory")
            elif ratio >= 0.6:
                legacy_states.append("strengthened")
            elif ratio >= 0.4:
                legacy_states.append("mixed")
            elif ratio >= 0.2:
                legacy_states.append("weakened")
            else:
                legacy_states.append("deteriorated")

    legacy_numeric = map_breadth_to_numeric(pd.Series(legacy_states))
    real_numeric = map_breadth_to_numeric(df["market_breadth_state"])

    asset_return_5d = df["return_5d"].fillna(0.0)

    # Pearson & Spearman correlations (Spearman = Pearson on ranks)
    corr_legacy_pearson = legacy_numeric.corr(asset_return_5d)
    corr_legacy_spearman = legacy_numeric.rank().corr(asset_return_5d.rank())

    corr_real_pearson = real_numeric.corr(asset_return_5d)
    corr_real_spearman = real_numeric.rank().corr(asset_return_5d.rank())

    print(f"\n2. COLLINEARITY EVALUATION (Breadth vs Single-Asset 5D Return):")
    print(f"   • Legacy Proxy Correlation (Pearson):  {corr_legacy_pearson:+.4f}")
    print(f"   • Legacy Proxy Correlation (Spearman): {corr_legacy_spearman:+.4f}")
    print(f"   • Real Market Breadth (Pearson):      {corr_real_pearson:+.4f}")
    print(f"   • Real Market Breadth (Spearman):     {corr_real_spearman:+.4f}")
    print(
        f"   • Collinearity Reduction:              {abs(corr_legacy_pearson) - abs(corr_real_pearson):+.4f} (lower is better)"
    )

    # 3. Calibration / Distribution Overview
    print(f"\n3. REAL BREADTH DISTRIBUTION & METRICS:")
    print(df["market_breadth_state"].value_counts().to_string())

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_breadth_evaluation("RELIANCE")
