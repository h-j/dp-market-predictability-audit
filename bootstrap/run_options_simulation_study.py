"""
Options Strategy Simulation Study Runner (Workstream 2).

Simulates weekly options trading policies on NIFTY index using Model C volatility forecasts and India VIX.
Evaluates unconditional variance risk premium harvesting vs model-filtered policies across naked and defined-risk structures.
Tests Pre-Registered Gate G-OPT and outputs options_simulation_study_report.md.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from market.data.vix_loader import IndiaVIXLoader
from market.options.regime_strategy import OptionsRegimeStrategyEngine
from market.options.strategy_evaluator import HONESTY_CAVEATS_VERBATIM, StrategyEvaluator, StrategyMetrics
from market.replay.walkforward_validation import WalkForwardValidator


def generate_nifty_weekly_dataset_with_forecasts(data_dir: Path = None) -> pd.DataFrame:
    """
    Generate weekly NIFTY dataset with walk-forward Model C 5-day volatility forecasts.
    Hermetic offline loader: reads committed data/nifty_enriched_daily_3y.csv directly.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    nifty_path = data_dir / "nifty_enriched_daily_3y.csv"
    if not nifty_path.exists():
        nifty_path = data_dir / "nifty_daily_3y.csv"

    df_raw = pd.read_csv(nifty_path)
    vix_loader = IndiaVIXLoader(data_dir=data_dir)
    df_enriched = vix_loader.merge_vix_features(df_raw)

    validator = WalkForwardValidator(
        df_enriched,
        asset_name="NIFTY",
        target_mode="volatility_5d",
        initial_train_size=250,
        test_fold_size=100,
    )
    df = validator.df.copy()

    # Model C features
    vix_cols = ["vix_close", "vix_change_5d", "vix_vs_20d_ma", "vix_percentile_252d"]
    X_all_vix = df[validator.FEATURE_COLS + validator.HAR_FEATURE_COLS + vix_cols].values
    y_vol = df["target_volatility_5d"].values

    folds = validator.generate_expanding_folds()
    forecasts = np.full(len(df), np.nan)

    for train_idx, test_idx in folds:
        gbm_c = GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, min_samples_leaf=20, random_state=42
        )
        gbm_c.fit(X_all_vix[train_idx], y_vol[train_idx])
        preds = gbm_c.predict(X_all_vix[test_idx])
        forecasts[test_idx] = np.clip(preds, 0.01, None)

    df["vol_forecast_5d"] = forecasts
    df = df.dropna(subset=["vol_forecast_5d"]).reset_index(drop=True)

    # Sample weekly rows (stride 5)
    df_weekly = df.iloc[::5].reset_index(drop=True)
    return df_weekly


def run_options_simulation_study():
    print("=" * 80)
    print("OPTIONS STRATEGY SIMULATION STUDY & GATE G-OPT EVALUATION (Workstream 2)")
    print("=" * 80)

    df_weekly = generate_nifty_weekly_dataset_with_forecasts()
    print(f"✓ Created weekly NIFTY dataset with Model C forecasts ({len(df_weekly)} weeks)")

    evaluator = StrategyEvaluator(starting_capital=1000000.0, risk_free_rate=0.07)

    # 1. Unconditional Baseline (Naked Short Strangle)
    engine_naked = OptionsRegimeStrategyEngine(starting_capital=1000000.0, structure="short_strangle")
    uncond_logs = engine_naked.run_simulation(df_weekly, policy_name="always_sell")
    uncond_metrics = evaluator.evaluate_logs(uncond_logs)
    print(f"\n---> Unconditional Short Strangle Baseline:")
    print(f"     Return: {uncond_metrics.total_return_pct:+.2f}% | Sortino: {uncond_metrics.annualized_sortino:.2f} | MaxDD: {uncond_metrics.max_drawdown_pct:.2f}% | PCR: {uncond_metrics.premium_capture_ratio:.4f}")

    # Policies to test across naked and defined-risk structures
    policies = [
        ("sell_when_calm", 1.0),
        ("sell_when_calm", 0.9),
        ("sell_when_calm", 0.8),
        ("buy_when_storm", 1.2),
        ("combined_regime", 1.0),
    ]

    naked_results: Dict[str, StrategyMetrics] = {"always_sell": uncond_metrics}
    condor_results: Dict[str, StrategyMetrics] = {}

    engine_condor = OptionsRegimeStrategyEngine(starting_capital=1000000.0, structure="iron_condor")
    uncond_condor_logs = engine_condor.run_simulation(df_weekly, policy_name="always_sell")
    condor_results["always_sell"] = evaluator.evaluate_logs(uncond_condor_logs, unconditional_baseline_metrics=uncond_metrics)

    print("\n---> Model-Filtered Strategy Evaluation & Gate G-OPT Tests:")

    for pol, k in policies:
        name_key = f"{pol}_k{k}" if "calm" in pol or "combined" in pol else pol

        # Naked variant
        logs_naked = engine_naked.run_simulation(df_weekly, policy_name=pol, calm_k=k)
        metrics_naked = evaluator.evaluate_logs(logs_naked, unconditional_baseline_metrics=uncond_metrics)
        metrics_naked.policy_name = name_key
        naked_results[name_key] = metrics_naked

        # Defined-risk variant (Iron Condor)
        logs_condor = engine_condor.run_simulation(df_weekly, policy_name=pol, calm_k=k)
        metrics_condor = evaluator.evaluate_logs(logs_condor, unconditional_baseline_metrics=uncond_metrics)
        metrics_condor.policy_name = name_key
        condor_results[name_key] = metrics_condor

        print(f"  • [{name_key.upper()} - Short Strangle] Return: {metrics_naked.total_return_pct:+.2f}% | Sortino: {metrics_naked.annualized_sortino:.2f} | MaxDD: {metrics_naked.max_drawdown_pct:.2f}% | Gate G-OPT: {'PASSED' if metrics_naked.gate_g_opt_passed else 'FAILED'}")

    generate_markdown_report(uncond_metrics, naked_results, condor_results, df_weekly)


def generate_markdown_report(
    uncond_metrics: StrategyMetrics,
    naked_results: Dict[str, StrategyMetrics],
    condor_results: Dict[str, StrategyMetrics],
    df_weekly: pd.DataFrame,
):
    report = []
    report.append("# Options Strategy Simulation Study & Gate G-OPT Report (Workstream 2)")
    report.append("\n## Executive Summary\n")
    report.append(
        "Evaluated synthetic options trading strategies on NIFTY weekly expiry using Model C volatility forecasts vs India VIX. "
        "Tested unconditional variance risk premium harvesting against model-filtered policies across naked and defined-risk structures."
    )

    g_opt_overall_passed = any(m.gate_g_opt_passed for m in naked_results.values()) or any(
        m.gate_g_opt_passed for m in condor_results.values()
    )

    if g_opt_overall_passed:
        gate_header = "### 🟢 GATE G-OPT VERDICT: PASSED"
        gate_body = "At least one model-filtered policy beat the unconditional short-strangle baseline on Sortino ratio and Max Drawdown simultaneously."
    else:
        gate_header = "### 🔴 GATE G-OPT VERDICT: FAILED"
        gate_body = (
            "No model-filtered policy beat the unconditional short-strangle baseline on both Sortino ratio and Max Drawdown simultaneously.\n\n"
            "**Research Conclusion**: Harvesting the unconditional variance risk premium yields positive return, but model-filtering based on "
            "daily bar features does not produce a statistically superior risk-adjusted profile over unconditional premium selling."
        )

    report.append(f"\n{gate_header}\n\n{gate_body}\n")

    report.append("\n## Short Strangle Strategy Results (Naked Structure)\n")
    report.append("| Policy | Total Return (%) | Sortino | Sharpe | Max Drawdown (%) | CVaR (95%) | PCR | Gate G-OPT |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for k, m in naked_results.items():
        g_str = "**PASSED**" if m.gate_g_opt_passed else "FAILED"
        report.append(
            f"| **{k}** | {m.total_return_pct:+.2f}% | {m.annualized_sortino:.2f} | {m.annualized_sharpe:.2f} | "
            f"{m.max_drawdown_pct:.2f}% | {m.cvar_95_pct:.2f}% | {m.premium_capture_ratio:.4f} | {g_str} |"
        )

    report.append("\n## Iron Condor Strategy Results (Defined-Risk Structure)\n")
    report.append("| Policy | Total Return (%) | Sortino | Sharpe | Max Drawdown (%) | CVaR (95%) | PCR | Gate G-OPT |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for k, m in condor_results.items():
        g_str = "**PASSED**" if m.gate_g_opt_passed else "FAILED"
        report.append(
            f"| **{k}** | {m.total_return_pct:+.2f}% | {m.annualized_sortino:.2f} | {m.annualized_sharpe:.2f} | "
            f"{m.max_drawdown_pct:.2f}% | {m.cvar_95_pct:.2f}% | {m.premium_capture_ratio:.4f} | {g_str} |"
        )

    report.append("\n---\n")
    report.append("## Verbatim Honesty Caveats\n")
    report.append(f"> {HONESTY_CAVEATS_VERBATIM}\n")

    report_text = "\n".join(report)
    out_path = Path(__file__).parent.parent / "options_simulation_study_report.md"
    out_path.write_text(report_text)
    print(f"\n✓ Saved Options Simulation Study Report to {out_path}")

    update_consolidated_track_report(naked_results, condor_results, g_opt_overall_passed)


def update_consolidated_track_report(
    naked_results: Dict[str, StrategyMetrics],
    condor_results: Dict[str, StrategyMetrics],
    g_opt_overall_passed: bool,
):
    track_report_path = Path(__file__).parent.parent / "volatility_options_track_report.md"
    if not track_report_path.exists():
        return

    content = track_report_path.read_text()
    
    # Build updated table programmatically
    table_lines = [
        "| Strategy Structure | Policy Name | Total Return (%) | Sortino | Sharpe | Max Drawdown (%) | CVaR (95%) | PCR | Gate G-OPT Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, m in naked_results.items():
        struct_label = "**Short Strangle (Naked)**" if k == "always_sell" else "Short Strangle"
        name_label = "**always_sell (BASELINE)**" if k == "always_sell" else k
        g_str = "BASELINE" if k == "always_sell" else ("**PASSED**" if m.gate_g_opt_passed else "FAILED")
        table_lines.append(
            f"| {struct_label} | {name_label} | {m.total_return_pct:+.2f}% | {m.annualized_sortino:.2f} | {m.annualized_sharpe:.2f} | "
            f"{m.max_drawdown_pct:.2f}% | {m.cvar_95_pct:.2f}% | {m.premium_capture_ratio:.4f} | {g_str} |"
        )
    for k, m in condor_results.items():
        struct_label = "**Iron Condor (Defined Risk)**" if k == "always_sell" else "Iron Condor"
        name_label = "**always_sell**" if k == "always_sell" else k
        g_str = "FAILED" if k == "always_sell" else ("**PASSED**" if m.gate_g_opt_passed else "FAILED")
        table_lines.append(
            f"| {struct_label} | {name_label} | {m.total_return_pct:+.2f}% | {m.annualized_sortino:.2f} | {m.annualized_sharpe:.2f} | "
            f"{m.max_drawdown_pct:.2f}% | {m.cvar_95_pct:.2f}% | {m.premium_capture_ratio:.4f} | {g_str} |"
        )

    updated_table_text = "\n".join(table_lines)
    
    # Replace options table section in consolidated report
    import re
    table_pattern = re.compile(
        r"\| Strategy Structure \| Policy Name \|.*?\n(?:\|.*?\n)+", re.DOTALL
    )
    if table_pattern.search(content):
        new_content = table_pattern.sub(updated_table_text + "\n", content, count=1)
        track_report_path.write_text(new_content)
        print(f"✓ Programmatically updated Gate G-OPT table in {track_report_path}")


if __name__ == "__main__":
    run_options_simulation_study()
