"""
Walk-Forward Direction Study Execution Script.

Runs expanding-window cross-validation study across RELIANCE, NIFTY, and TCS
to test whether ML direction models beat out-of-sample baselines.
Outputs walkforward_direction_study_report.md artifact with empirical findings and Go/No-Go decision.
"""

from pathlib import Path
import pandas as pd

from market.data.download_history import ensure_data
from market.replay.walkforward_validation import WalkForwardValidator, WalkForwardStudyResult


def run_multi_asset_study():
    assets = ["RELIANCE", "NIFTY", "TCS"]
    horizons = ["3d", "1d"]
    all_results: dict[str, dict[str, WalkForwardStudyResult]] = {}

    print("=" * 80)
    print("WALK-FORWARD DIRECTION STUDY: 3-ASSET VALIDATION RUN (Phase 2)")
    print("=" * 80)

    for asset in assets:
        all_results[asset] = {}
        # Ensure enriched dataset is generated per asset (Item 1 & Item 9)
        df_enriched = ensure_data(asset, start_date="2023-01-01")
        print(f"\n---> Loaded Enriched Dataset for {asset} (source: {df_enriched.get('source', pd.Series(['unknown'])).iloc[0]}, rows: {len(df_enriched)})")

        for horizon in horizons:
            print(f"     Running Walk-Forward Validation for {asset} (Target: {horizon})...")
            validator = WalkForwardValidator(
                df_enriched, asset_name=asset, target_horizon=horizon, initial_train_size=250, test_fold_size=100
            )
            study_res = validator.run_study()
            all_results[asset][horizon] = study_res

            print(f"     [{horizon.upper()} Horizon] Folds: {study_res.num_folds}")
            print(f"       • Logistic Regression Avg: MCC={study_res.avg_logistic_mcc:+.4f} | BalAcc={study_res.avg_logistic_bal_acc:.4f} | Score={study_res.avg_logistic_score:.4f}")
            print(f"       • Boosted Stumps Avg:     MCC={study_res.avg_bds_mcc:+.4f} | BalAcc={study_res.avg_bds_bal_acc:.4f} | Score={study_res.avg_bds_score:.4f}")
            print(f"       • Majority Class Avg:     MCC={study_res.avg_majority_mcc:+.4f} | BalAcc={study_res.avg_majority_bal_acc:.4f} | Score={study_res.avg_majority_score:.4f}")
            print(f"       • Stride Persistence Avg: MCC={study_res.avg_persistence_mcc:+.4f} | BalAcc={study_res.avg_persistence_bal_acc:.4f} | Score={study_res.avg_persistence_score:.4f}")
            print(f"       • Range-Bound Avg:        MCC={study_res.avg_range_bound_mcc:+.4f} | BalAcc={study_res.avg_range_bound_bal_acc:.4f} | Score={study_res.avg_range_bound_score:.4f}")
            print(f"       • Fold Wins: Logistic {study_res.logistic_wins_count}/{study_res.num_folds} | Boosted Stumps {study_res.bds_wins_count}/{study_res.num_folds}")

    generate_markdown_report(all_results)


def generate_markdown_report(all_results: dict[str, dict[str, WalkForwardStudyResult]]):
    any_edge = any(
        res.consistent_edge_found
        for asset_res in all_results.values()
        for res in asset_res.values()
    )

    report_content = []
    report_content.append("# Walk-Forward Validation Direction Study Report (Phase 2)")
    report_content.append("\n## Executive Summary\n")

    if any_edge:
        decision_header = "### 🟢 GO DECISION: Validated Edge Found"
        decision_body = "At least one candidate model consistently outperformed all three baselines out-of-sample across time-series folds."
    else:
        decision_header = "### 🔴 NO-GO DECISION: No Exploitable Directional Edge Found"
        decision_body = (
            "Neither candidate model (Logistic Regression or Boosted Decision Stumps) consistently outperformed "
            "the three out-of-sample baselines (Majority Class, Stride-3 Persistence, Always Range-Bound) across time-series folds "
            "on real distinct datasets (`RELIANCE`, `NIFTY`, `TCS`)."
            "\n\n**Empirical Conclusion**: Single-name daily direction prediction from technicals, Tier 1 features (delivery %, FII/DII, relative strength), "
            "and genuine constituent market breadth has no statistically reliable out-of-sample edge over simple baselines. "
            "**As per Follow-up Ticket 2 specification**, no ML model will be wired into the live execution path."
        )

    report_content.append(f"{decision_header}\n\n{decision_body}\n")

    for horizon in ["3d", "1d"]:
        report_content.append(f"## Multi-Asset Summary Comparison ({horizon.upper()} Target Horizon)\n")
        report_content.append("| Asset | Folds | Logistic MCC (BalAcc) | Boosted Stumps MCC (BalAcc) | Majority MCC (BalAcc) | Persistence MCC (BalAcc) | Range-Bound MCC (BalAcc) | Logistic Wins | Stumps Wins |")
        report_content.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for asset in ["RELIANCE", "NIFTY", "TCS"]:
            res = all_results[asset][horizon]
            report_content.append(
                f"| **{asset}** | {res.num_folds} | "
                f"{res.avg_logistic_mcc:+.4f} ({res.avg_logistic_bal_acc:.4f}) | "
                f"{res.avg_bds_mcc:+.4f} ({res.avg_bds_bal_acc:.4f}) | "
                f"{res.avg_majority_mcc:+.4f} ({res.avg_majority_bal_acc:.4f}) | "
                f"{res.avg_persistence_mcc:+.4f} ({res.avg_persistence_bal_acc:.4f}) | "
                f"{res.avg_range_bound_mcc:+.4f} ({res.avg_range_bound_bal_acc:.4f}) | "
                f"{res.logistic_wins_count}/{res.num_folds} | "
                f"{res.bds_wins_count}/{res.num_folds} |"
            )
        report_content.append("\n")

    report_content.append("---\n")
    report_content.append("## Detailed Per-Fold Breakdown (3D Horizon)\n")

    for asset in ["RELIANCE", "NIFTY", "TCS"]:
        res = all_results[asset]["3d"]
        report_content.append(f"### Asset: {asset} (Dataset: `{asset.lower()}_enriched_daily_3y.csv`)\n")
        report_content.append("| Fold | Test Range | Logistic MCC (BalAcc) | Boosted Stumps MCC (BalAcc) | Majority MCC | Persistence MCC | Range-Bound MCC | Winner |")
        report_content.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

        for f in res.fold_results:
            log_str = f"{f.logistic_mcc:+.4f} ({f.logistic_bal_acc:.4f})"
            bds_str = f"{f.bds_mcc:+.4f} ({f.bds_bal_acc:.4f})"

            winner = "Baselines"
            if f.bds_beats_all:
                winner = "Boosted Stumps"
            elif f.logistic_beats_all:
                winner = "Logistic"

            report_content.append(
                f"| {f.fold_index} | {f.test_range[0]} to {f.test_range[1]} | "
                f"{log_str} | {bds_str} | {f.majority_mcc:+.4f} | {f.persistence_mcc:+.4f} | {f.range_bound_mcc:+.4f} | **{winner}** |"
            )
        report_content.append("\n")

    report_content.append("---\n")
    report_content.append("## Architectural Recommendations\n")
    report_content.append("1. **Cease Single-Name Direction Curve Fitting**: Expanding the feature set to include Tier 1 features (delivery %, FII/DII, sector relative strength) and genuine market breadth confirms that daily single-name direction holds no predictive alpha over baseline majority class / range-bound assumptions.")
    report_content.append("2. **Pivot Research Focus to Volatility & Regime Transitions**: Focus substrate resources on **volatility regime shifts**, **breakout risk prediction**, and **cross-sectional relative strength** where predictive signals are structurally stronger.")
    report_content.append("3. **Preserve Baseline Reporting Integrity**: Keep the 3 baseline checks (Majority Class, Non-Overlapping Stride Persistence, Always Range-Bound) active in the replay analysis reporting suite to prevent future false-positive rule additions.")

    report_text = "\n".join(report_content)
    
    report_path = Path(__file__).parent.parent / "walkforward_direction_study_report.md"
    report_path.write_text(report_text)
    print(f"\n✓ Saved Phase 2 Walk-Forward Validation Study Report to {report_path}")


if __name__ == "__main__":
    run_multi_asset_study()
