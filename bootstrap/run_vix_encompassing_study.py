"""
India VIX Encompassing Study Runner (Workstream 1).

Executes Mincer-Zarnowitz encompassing regressions across NIFTY (primary), RELIANCE, and TCS.
Compares Model A (VIX-only), Model B (HAR-RV + GBM without VIX), and Model C (Model B + VIX).
Performs block-bootstrap (block-4, 2000 resamples, 95% CI) on encompassing coefficients
and tests hypotheses H-V1, H-V2, and H-V3.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from market.data.download_history import ensure_data
from market.data.vix_loader import IndiaVIXLoader
from market.replay.walkforward_validation import (
    WalkForwardValidator,
    compute_qlike,
    compute_r2_vs_persistence,
    compute_spearman_rank,
)


@dataclass
class EncompassingResult:
    asset_name: str
    num_folds: int
    r2_a: float  # VIX-only
    r2_b: float  # Repo best without VIX
    r2_c: float  # Model B + VIX
    qlike_a: float
    qlike_b: float
    qlike_c: float
    beta_a: float
    beta_b: float
    beta_b_ci_lower: float
    beta_b_ci_upper: float
    beta_b_sig_gt_zero: bool
    h_v1_verdict: str
    h_v2_verdict: str
    h_v3_verdict: str
    is_edge_candidate: bool


def run_block_bootstrap(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, block_size: int = 4, n_resamples: int = 2000
) -> Tuple[float, float, float, float, float]:
    """
    Mincer-Zarnowitz joint regression with circular block bootstrap confidence interval.
    y = alpha + beta_a * y_pred_a + beta_b * y_pred_b
    """
    n = len(y_true)
    if n < 10:
        return 0.0, 0.0, 0.0, 0.0, False

    X = np.column_stack([y_pred_a, y_pred_b])
    reg = LinearRegression()
    reg.fit(X, y_true)
    beta_a, beta_b = float(reg.coef_[0]), float(reg.coef_[1])

    # Circular Block Bootstrap
    boot_betas_b = []
    n_blocks = int(np.ceil(n / block_size))

    np.random.seed(42)
    for _ in range(n_resamples):
        start_indices = np.random.randint(0, n, size=n_blocks)
        sample_indices = []
        for idx in start_indices:
            block_idx = [(idx + j) % n for j in range(block_size)]
            sample_indices.extend(block_idx)
        sample_indices = np.array(sample_indices[:n])

        y_boot = y_true[sample_indices]
        X_boot = X[sample_indices]

        try:
            r_boot = LinearRegression()
            r_boot.fit(X_boot, y_boot)
            boot_betas_b.append(r_boot.coef_[1])
        except Exception:
            pass

    if boot_betas_b:
        ci_lower = float(np.percentile(boot_betas_b, 2.5))
        ci_upper = float(np.percentile(boot_betas_b, 97.5))
    else:
        ci_lower, ci_upper = beta_b, beta_b

    sig_gt_zero = ci_lower > 0.0
    return beta_a, beta_b, ci_lower, ci_upper, sig_gt_zero


def evaluate_asset_encompassing(asset_name: str) -> EncompassingResult:
    df_raw = ensure_data(asset_name, start_date="2023-01-01")
    vix_loader = IndiaVIXLoader()
    df_enriched = vix_loader.merge_vix_features(df_raw)

    # Instantiate validator to get expanding fold indices and prepared targets
    validator = WalkForwardValidator(
        df_enriched,
        asset_name=asset_name,
        target_mode="volatility_5d",
        initial_train_size=250,
        test_fold_size=100,
    )
    folds = validator.generate_expanding_folds()
    df = validator.df

    X_har = df[validator.HAR_FEATURE_COLS].values
    X_base = df[validator.FEATURE_COLS + validator.HAR_FEATURE_COLS].values
    vix_cols = ["vix_close", "vix_change_5d", "vix_vs_20d_ma", "vix_percentile_252d"]
    X_vix_only = df[["vix_close"]].values
    X_all_vix = df[validator.FEATURE_COLS + validator.HAR_FEATURE_COLS + vix_cols].values

    y_vol = df["target_volatility_5d"].values
    pers_vol = df["rv_5d"].values

    preds_a, preds_b, preds_c = [], [], []
    y_stride_all, pers_stride_all = [], []

    for train_idx, test_idx in folds:
        test_stride_subidx = np.arange(0, len(test_idx), 5)
        test_stride_idx = test_idx[test_stride_subidx]

        # Target & Baselines
        y_train = y_vol[train_idx]
        y_test_stride = y_vol[test_stride_idx]
        pers_test_stride = pers_vol[test_stride_idx]

        # Model A: VIX-only regression
        reg_a = LinearRegression()
        reg_a.fit(X_vix_only[train_idx], y_train)
        p_a = np.clip(reg_a.predict(X_vix_only[test_stride_idx]), 0.01, None)

        # Model B: Repo best model (HAR-RV + GBM without VIX)
        gbm_b = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, min_samples_leaf=20, random_state=42)
        gbm_b.fit(X_base[train_idx], y_train)
        p_b = np.clip(gbm_b.predict(X_base[test_stride_idx]), 0.01, None)

        # Model C: Model B + VIX features
        gbm_c = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, min_samples_leaf=20, random_state=42)
        gbm_c.fit(X_all_vix[train_idx], y_train)
        p_c = np.clip(gbm_c.predict(X_all_vix[test_stride_idx]), 0.01, None)

        preds_a.extend(p_a)
        preds_b.extend(p_b)
        preds_c.extend(p_c)
        y_stride_all.extend(y_test_stride)
        pers_stride_all.extend(pers_test_stride)

    y_arr = np.array(y_stride_all)
    pers_arr = np.array(pers_stride_all)
    pa_arr = np.array(preds_a)
    pb_arr = np.array(preds_b)
    pc_arr = np.array(preds_c)

    r2_a = compute_r2_vs_persistence(y_arr, pa_arr, pers_arr)
    r2_b = compute_r2_vs_persistence(y_arr, pb_arr, pers_arr)
    r2_c = compute_r2_vs_persistence(y_arr, pc_arr, pers_arr)

    qlike_a = compute_qlike(y_arr, pa_arr)
    qlike_b = compute_qlike(y_arr, pb_arr)
    qlike_c = compute_qlike(y_arr, pc_arr)

    # Mincer-Zarnowitz Encompassing Test
    beta_a, beta_b, ci_low, ci_high, sig_gt_zero = run_block_bootstrap(y_arr, pa_arr, pb_arr, block_size=4, n_resamples=2000)

    # Evaluate Hypotheses
    if asset_name == "NIFTY":
        h_v1 = "CONFIRMED" if r2_a >= r2_b else "REJECTED"
        h_v2 = "N/A (Index)"
        h_v3 = "EDGE-CANDIDATE (Beta_B > 0)" if sig_gt_zero else "CONFIRMED (No Edge, Beta_B <= 0)"
        is_edge = sig_gt_zero
    else:
        h_v1 = "N/A (Single-Name)"
        h_v2 = "CONFIRMED" if (r2_c > r2_a and r2_c > r2_b) else "REJECTED"
        h_v3 = "N/A (Single-Name)"
        is_edge = False

    return EncompassingResult(
        asset_name=asset_name,
        num_folds=len(folds),
        r2_a=round(r2_a, 4),
        r2_b=round(r2_b, 4),
        r2_c=round(r2_c, 4),
        qlike_a=round(qlike_a, 4),
        qlike_b=round(qlike_b, 4),
        qlike_c=round(qlike_c, 4),
        beta_a=round(beta_a, 4),
        beta_b=round(beta_b, 4),
        beta_b_ci_lower=round(ci_low, 4),
        beta_b_ci_upper=round(ci_high, 4),
        beta_b_sig_gt_zero=sig_gt_zero,
        h_v1_verdict=h_v1,
        h_v2_verdict=h_v2,
        h_v3_verdict=h_v3,
        is_edge_candidate=is_edge,
    )


def run_vix_encompassing_study():
    assets = ["NIFTY", "RELIANCE", "TCS"]
    results: Dict[str, EncompassingResult] = {}

    print("=" * 80)
    print("INDIA VIX ENCOMPASSING STUDY: MINCER-ZARNOWITZ BOOTSTRAP (Workstream 1)")
    print("=" * 80)

    for asset in assets:
        res = evaluate_asset_encompassing(asset)
        results[asset] = res

        print(f"\n---> Asset: {asset} (Folds: {res.num_folds})")
        print(f"     • Model A (VIX-Only):    R2_vs_pers={res.r2_a:+.4f} | QLIKE={res.qlike_a:.4f}")
        print(f"     • Model B (No VIX):      R2_vs_pers={res.r2_b:+.4f} | QLIKE={res.qlike_b:.4f}")
        print(f"     • Model C (Model B+VIX): R2_vs_pers={res.r2_c:+.4f} | QLIKE={res.qlike_c:.4f}")
        print(f"     • Encompassing Coeffs:  Beta_A={res.beta_a:.4f} | Beta_B={res.beta_b:.4f} (95% CI: [{res.beta_b_ci_lower:+.4f}, {res.beta_b_ci_upper:+.4f}])")
        print(f"     • H-V1 Verdict: {res.h_v1_verdict}")
        print(f"     • H-V2 Verdict: {res.h_v2_verdict}")
        print(f"     • H-V3 Verdict: {res.h_v3_verdict}")

    generate_markdown_report(results)


def generate_markdown_report(results: Dict[str, EncompassingResult]):
    report = []
    report.append("# India VIX Encompassing Study Report (Workstream 1)")
    report.append("\n## Executive Summary\n")
    report.append(
        "Evaluated out-of-sample Mincer-Zarnowitz encompassing regressions across `NIFTY` (primary), `RELIANCE`, and `TCS` "
        "to test whether daily bar features retain incremental predictive power over implied volatility (India VIX)."
    )

    report.append("\n## Summary Table: Model Comparisons & Encompassing Tests\n")
    report.append("| Asset | Folds | Model A R² (VIX Only) | Model B R² (No VIX) | Model C R² (Model B+VIX) | Beta A (VIX) | Beta B (Model B) | 95% CI (Beta B) | Beta B > 0? | Status |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for asset in ["NIFTY", "RELIANCE", "TCS"]:
        r = results[asset]
        flag = "**EDGE-CANDIDATE**" if r.is_edge_candidate else "NO-EDGE"
        report.append(
            f"| **{r.asset_name}** | {r.num_folds} | {r.r2_a:+.4f} | {r.r2_b:+.4f} | **{r.r2_c:+.4f}** | "
            f"{r.beta_a:.4f} | {r.beta_b:.4f} | [{r.beta_b_ci_lower:+.4f}, {r.beta_b_ci_upper:+.4f}] | "
            f"{'YES' if r.beta_b_sig_gt_zero else 'NO'} | {flag} |"
        )

    report.append("\n---\n")
    report.append("## Pre-Registered Hypotheses Verdicts\n")
    nifty_res = results["NIFTY"]
    report.append(f"- **H-V1 (NIFTY Implied Vol Subsumption)**: {nifty_res.h_v1_verdict} (Model A R² = {nifty_res.r2_a:+.4f} vs Model B R² = {nifty_res.r2_b:+.4f})")
    rel_res = results["RELIANCE"]
    tcs_res = results["TCS"]
    report.append(f"- **H-V2 (Single-Name Dual Channel)**: RELIANCE = {rel_res.h_v2_verdict}, TCS = {tcs_res.h_v2_verdict}")
    report.append(f"- **H-V3 (NIFTY Monetizable Edge)**: {nifty_res.h_v3_verdict}")

    report_text = "\n".join(report)
    out_path = Path(__file__).parent.parent / "vix_encompassing_study_report.md"
    out_path.write_text(report_text)
    print(f"\n✓ Saved VIX Encompassing Study Report to {out_path}")


if __name__ == "__main__":
    run_vix_encompassing_study()
