# Walk-Forward Validation Direction Study Report (Phase 2)

## Executive Summary

### 🔴 NO-GO DECISION: No Exploitable Directional Edge Found

Neither candidate model (Logistic Regression or Boosted Decision Stumps) consistently outperformed the three out-of-sample baselines (Majority Class, Stride-3 Persistence, Always Range-Bound) across time-series folds on real distinct datasets (`RELIANCE`, `NIFTY`, `TCS`).

**Empirical Conclusion**: Single-name daily direction prediction from technicals, Tier 1 features (delivery %, FII/DII, relative strength), and genuine constituent market breadth has no statistically reliable out-of-sample edge over simple baselines. **As per Follow-up Ticket 2 specification**, no ML model will be wired into the live execution path.

## Multi-Asset Summary Comparison (3D Target Horizon)

| Asset | Folds | Logistic MCC (BalAcc) | Boosted Stumps MCC (BalAcc) | Majority MCC (BalAcc) | Persistence MCC (BalAcc) | Range-Bound MCC (BalAcc) | Logistic Wins | Stumps Wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | 7 | -0.0290 (0.3258) | -0.0144 (0.3284) | +0.0000 (0.3333) | -0.0309 (0.3200) | +0.0000 (0.3333) | 1/7 | 2/7 |
| **NIFTY** | 6 | -0.0302 (0.3205) | -0.0196 (0.3300) | +0.0000 (0.3333) | -0.0093 (0.3320) | +0.0000 (0.3333) | 2/6 | 0/6 |
| **TCS** | 7 | +0.0042 (0.3396) | -0.0371 (0.3182) | +0.0000 (0.3333) | +0.0120 (0.3612) | +0.0000 (0.3333) | 2/7 | 1/7 |


## Multi-Asset Summary Comparison (1D Target Horizon)

| Asset | Folds | Logistic MCC (BalAcc) | Boosted Stumps MCC (BalAcc) | Majority MCC (BalAcc) | Persistence MCC (BalAcc) | Range-Bound MCC (BalAcc) | Logistic Wins | Stumps Wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | 7 | -0.0272 (0.3226) | -0.0213 (0.3254) | +0.0000 (0.3333) | -0.0197 (0.3318) | +0.0000 (0.3333) | 2/7 | 1/7 |
| **NIFTY** | 7 | -0.0101 (0.3287) | +0.0390 (0.3423) | +0.0000 (0.3333) | +0.0327 (0.3574) | +0.0000 (0.3333) | 0/7 | 2/7 |
| **TCS** | 7 | -0.0491 (0.3144) | +0.0379 (0.3420) | +0.0000 (0.3333) | +0.0309 (0.3594) | +0.0000 (0.3333) | 1/7 | 3/7 |


---

## Detailed Per-Fold Breakdown (3D Horizon)

### Asset: RELIANCE (Dataset: `reliance_enriched_daily_3y.csv`)

| Fold | Test Range | Logistic MCC (BalAcc) | Boosted Stumps MCC (BalAcc) | Majority MCC | Persistence MCC | Range-Bound MCC | Winner |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 2024-01-08 to 2024-06-06 | +0.0681 (0.3873) | -0.0682 (0.3082) | +0.0000 | -0.1898 | +0.0000 | **Logistic** |
| 2 | 2024-06-07 to 2024-10-30 | -0.0367 (0.3217) | -0.0630 (0.3138) | +0.0000 | -0.1014 | +0.0000 | **Baselines** |
| 3 | 2024-10-31 to 2025-03-25 | +0.0385 (0.3479) | -0.0306 (0.3224) | +0.0000 | +0.0956 | +0.0000 | **Baselines** |
| 4 | 2025-03-26 to 2025-08-20 | -0.0197 (0.3256) | -0.0987 (0.2946) | +0.0000 | +0.0901 | +0.0000 | **Baselines** |
| 5 | 2025-08-21 to 2026-01-14 | -0.0731 (0.3042) | +0.0370 (0.3446) | +0.0000 | +0.0328 | +0.0000 | **Boosted Stumps** |
| 6 | 2026-01-15 to 2026-06-11 | -0.1643 (0.2670) | -0.0291 (0.3212) | +0.0000 | +0.0481 | +0.0000 | **Baselines** |
| 7 | 2026-06-12 to 2026-07-28 | -0.0157 (0.3270) | +0.1518 (0.3937) | +0.0000 | -0.1919 | +0.0000 | **Boosted Stumps** |


### Asset: NIFTY (Dataset: `nifty_enriched_daily_3y.csv`)

| Fold | Test Range | Logistic MCC (BalAcc) | Boosted Stumps MCC (BalAcc) | Majority MCC | Persistence MCC | Range-Bound MCC | Winner |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 2024-01-08 to 2024-06-06 | -0.0629 (0.3130) | +0.0148 (0.3406) | +0.0000 | +0.0272 | +0.0000 | **Baselines** |
| 2 | 2024-06-07 to 2024-10-30 | +0.0605 (0.3489) | -0.0409 (0.3215) | +0.0000 | -0.0288 | +0.0000 | **Logistic** |
| 3 | 2024-10-31 to 2025-03-25 | +0.1145 (0.3462) | +0.0000 (0.3333) | +0.0000 | -0.0259 | +0.0000 | **Logistic** |
| 4 | 2025-03-26 to 2025-08-20 | -0.0607 (0.3185) | -0.0446 (0.3259) | +0.0000 | -0.1479 | +0.0000 | **Baselines** |
| 5 | 2025-08-21 to 2026-01-14 | -0.0816 (0.3074) | -0.0215 (0.3293) | +0.0000 | -0.1372 | +0.0000 | **Baselines** |
| 6 | 2026-01-16 to 2026-06-16 | -0.1511 (0.2891) | -0.0256 (0.3293) | +0.0000 | +0.2570 | +0.0000 | **Baselines** |


### Asset: TCS (Dataset: `tcs_enriched_daily_3y.csv`)

| Fold | Test Range | Logistic MCC (BalAcc) | Boosted Stumps MCC (BalAcc) | Majority MCC | Persistence MCC | Range-Bound MCC | Winner |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 2024-01-08 to 2024-06-06 | +0.1755 (0.4079) | -0.0183 (0.3252) | +0.0000 | -0.0409 | +0.0000 | **Logistic** |
| 2 | 2024-06-07 to 2024-10-30 | -0.1125 (0.2826) | -0.1695 (0.2661) | +0.0000 | +0.0531 | +0.0000 | **Baselines** |
| 3 | 2024-10-31 to 2025-03-25 | +0.0416 (0.3472) | +0.0168 (0.3382) | +0.0000 | -0.1368 | +0.0000 | **Boosted Stumps** |
| 4 | 2025-03-26 to 2025-08-20 | +0.1652 (0.4024) | +0.0986 (0.3737) | +0.0000 | +0.1026 | +0.0000 | **Baselines** |
| 5 | 2025-08-21 to 2026-01-14 | -0.1430 (0.3069) | +0.0000 (0.3333) | +0.0000 | +0.0901 | +0.0000 | **Baselines** |
| 6 | 2026-01-15 to 2026-06-11 | -0.0972 (0.2972) | -0.1870 (0.2573) | +0.0000 | -0.1140 | +0.0000 | **Baselines** |
| 7 | 2026-06-12 to 2026-07-28 | +0.0000 (0.3333) | +0.0000 (0.3333) | +0.0000 | +0.1296 | +0.0000 | **Baselines** |


---

## Architectural Recommendations

1. **Cease Single-Name Direction Curve Fitting**: Expanding the feature set to include Tier 1 features (delivery %, FII/DII, sector relative strength) and genuine market breadth confirms that daily single-name direction holds no predictive alpha over baseline majority class / range-bound assumptions.
2. **Pivot Research Focus to Volatility & Regime Transitions**: Focus substrate resources on **volatility regime shifts**, **breakout risk prediction**, and **cross-sectional relative strength** where predictive signals are structurally stronger.
3. **Preserve Baseline Reporting Integrity**: Keep the 3 baseline checks (Majority Class, Non-Overlapping Stride Persistence, Always Range-Bound) active in the replay analysis reporting suite to prevent future false-positive rule additions.