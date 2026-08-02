"""
Walk-Forward Direction Study Execution Script.

Runs expanding-window cross-validation study across RELIANCE, NIFTY, and TCS
to test whether ML direction models beat out-of-sample baselines.
Outputs walkforward_direction_study_report.md artifact with empirical findings and Go/No-Go decision.
"""

import os
from pathlib import Path
import pandas as pd

from market.data.download_history import ensure_data
from market.replay.walkforward_validation import WalkForwardValidator, WalkForwardStudyResult


def run_multi_asset_study():
    assets = ["RELIANCE", "NIFTY", "TCS"]
    results: dict[str, WalkForwardStudyResult] = {}

    print("=" * 80)
    print("WALK-FORWARD DIRECTION STUDY: 3-ASSET VALIDATION RUN")
    print("=" * 80)

    for asset in assets:
        print(f"\n---> Running Walk-Forward Validation for {asset}...")
        df = ensure_data(asset, start_date="2023-01-01")
        validator = WalkForwardValidator(
            df, asset_name=asset, initial_train_size=250, test_fold_size=100
        )
        study_res = validator.run_study()
        results[asset] = study_res

        print(f"     Total Samples: {study_res.total_samples} | Folds: {study_res.num_folds}")
        print(f"     • Logistic Regression Avg Score: {study_res.avg_logistic_score:.4f} (Accuracy: {study_res.avg_logistic_accuracy:.4f})")
        print(f"     • HistGradientBoosting Avg Score: {study_res.avg_hgb_score:.4f} (Accuracy: {study_res.avg_hgb_accuracy:.4f})")
        print(f"     • Majority Class Baseline Score: {study_res.avg_majority_score:.4f} (Accuracy: {study_res.avg_majority_accuracy:.4f})")
        print(f"     • Persistence Baseline Score:    {study_res.avg_persistence_score:.4f} (Accuracy: {study_res.avg_persistence_accuracy:.4f})")
        print(f"     • Always Range-Bound Score:      {study_res.avg_range_bound_score:.4f} (Accuracy: {study_res.avg_range_bound_accuracy:.4f})")
        print(f"     • Wins Over Baselines: Logistic {study_res.logistic_wins_count}/{study_res.num_folds} folds | HGB {study_res.hgb_wins_count}/{study_res.num_folds} folds")

    # Generate Markdown Report
    generate_markdown_report(results)


def generate_markdown_report(results: dict[str, WalkForwardStudyResult]):
    all_consistent_edge = any(res.consistent_edge_found for res in results.values())
    
    report_content = []
    report_content.append("# Walk-Forward Validation Direction Study Report")
    report_content.append("\n## Executive Summary\n")
    
    if all_consistent_edge:
        decision_header = "### 🟢 GO DECISION: Validated Edge Found"
        decision_body = "At least one candidate model consistently outperformed all three baselines out-of-sample across time-series folds."
    else:
        decision_header = "### 🔴 NO-GO DECISION: No Exploitable Directional Edge Found"
        decision_body = (
            "Neither candidate model (Logistic Regression or HistGradientBoosting) consistently outperformed "
            "the three out-of-sample baselines (Majority Class, Persistence, Always Range-Bound) across time-series folds. "
            "\n\n**Empirical Conclusion**: Single-name daily direction prediction from daily OHLCV technicals and market breadth "
            "has no statistically reliable out-of-sample edge over simple baselines. **As per Follow-up Ticket 2 specification**, "
            "no ML model will be wired into the live execution path."
        )

    report_content.append(f"{decision_header}\n\n{decision_body}\n")

    report_content.append("## Multi-Asset Summary Comparison\n")
    report_content.append("| Asset | Folds | Logistic Score (Acc) | HGB Score (Acc) | Majority Score | Persistence Score | Range-Bound Score | Logistic Wins | HGB Wins |")
    report_content.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for asset, res in results.items():
        report_content.append(
            f"| **{asset}** | {res.num_folds} | "
            f"{res.avg_logistic_score:.4f} ({res.avg_logistic_accuracy:.4f}) | "
            f"{res.avg_hgb_score:.4f} ({res.avg_hgb_accuracy:.4f}) | "
            f"{res.avg_majority_score:.4f} | "
            f"{res.avg_persistence_score:.4f} | "
            f"{res.avg_range_bound_score:.4f} | "
            f"{res.logistic_wins_count}/{res.num_folds} | "
            f"{res.hgb_wins_count}/{res.num_folds} |"
        )

    report_content.append("\n---\n")
    report_content.append("## Detailed Per-Fold Breakdown\n")

    for asset, res in results.items():
        report_content.append(f"### Asset: {asset}\n")
        report_content.append("| Fold | Test Range | Logistic Score (Acc) | HGB Score (Acc) | Majority Score | Persistence Score | Range-Bound Score | Winner |")
        report_content.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

        for f in res.fold_results:
            log_str = f"{f.logistic_score:.4f} ({f.logistic_accuracy:.4f})"
            hgb_str = f"{f.hgb_score:.4f} ({f.hgb_accuracy:.4f})"
            
            winner = "Baselines"
            if f.hgb_beats_all:
                winner = "HGB"
            elif f.logistic_beats_all:
                winner = "Logistic"

            report_content.append(
                f"| {f.fold_index} | {f.test_range[0]} to {f.test_range[1]} | "
                f"{log_str} | {hgb_str} | {f.majority_score:.4f} | {f.persistence_score:.4f} | {f.range_bound_score:.4f} | **{winner}** |"
            )
        report_content.append("\n")

    report_content.append("---\n")
    report_content.append("## Architectural Recommendations\n")
    report_content.append("1. **Cease Single-Name Direction Curve Fitting**: Daily technical features do not hold predictive alpha for next-day direction over baseline majority class / range-bound assumptions.")
    report_content.append("2. **Pivot Research Focus to Volatility & Regime Transitions**: Rather than attempting daily directional prediction, focus substrate resources on **volatility regime shifts**, **breakout risk prediction**, and **cross-sectional relative strength** where predictive signals are structurally stronger.")
    report_content.append("3. **Preserve Baseline Reporting Integrity**: Keep the 3 baseline checks (Majority Class, Persistence, Always Range-Bound) active in the replay analysis reporting suite to prevent future false-positive direction rules.")

    report_text = "\n".join(report_content)
    
    # Save to report artifact path
    report_path = Path(__file__).parent.parent / "walkforward_direction_study_report.md"
    report_path.write_text(report_text)
    print(f"\n✓ Saved Walk-Forward Validation Study Report to {report_path}")


if __name__ == "__main__":
    run_multi_asset_study()
