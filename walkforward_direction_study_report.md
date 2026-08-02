# Walk-Forward Validation Direction Study Report

## Executive Summary

### 🔴 NO-GO DECISION: No Exploitable Directional Edge Found

Neither candidate model (Logistic Regression or HistGradientBoosting) consistently outperformed the three out-of-sample baselines (Majority Class, Persistence, Always Range-Bound) across time-series folds. 

**Empirical Conclusion**: Single-name daily direction prediction from daily OHLCV technicals and market breadth has no statistically reliable out-of-sample edge over simple baselines. **As per Follow-up Ticket 2 specification**, no ML model will be wired into the live execution path.

## Multi-Asset Summary Comparison

| Asset | Folds | Logistic Score (Acc) | HGB Score (Acc) | Majority Score | Persistence Score | Range-Bound Score | Logistic Wins | HGB Wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | 7 | 0.4778 (0.4106) | 0.4994 (0.4322) | 0.4893 | 0.7138 | 0.5672 | 0/7 | 0/7 |
| **NIFTY** | 7 | 0.4778 (0.4106) | 0.4994 (0.4322) | 0.4893 | 0.7138 | 0.5672 | 0/7 | 0/7 |
| **TCS** | 7 | 0.4778 (0.4106) | 0.4994 (0.4322) | 0.4893 | 0.7138 | 0.5672 | 0/7 | 0/7 |

---

## Detailed Per-Fold Breakdown

### Asset: RELIANCE

| Fold | Test Range | Logistic Score (Acc) | HGB Score (Acc) | Majority Score | Persistence Score | Range-Bound Score | Winner |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 2024-01-08 to 2024-06-06 | 0.4800 (0.4200) | 0.5100 (0.4500) | 0.5500 | 0.6900 | 0.5600 | **Baselines** |
| 2 | 2024-06-07 to 2024-10-30 | 0.5250 (0.4300) | 0.4250 (0.3300) | 0.4850 | 0.7100 | 0.5950 | **Baselines** |
| 3 | 2024-10-31 to 2025-03-25 | 0.5650 (0.5100) | 0.6050 (0.5500) | 0.4250 | 0.7700 | 0.5550 | **Baselines** |
| 4 | 2025-03-26 to 2025-08-20 | 0.4600 (0.3900) | 0.4700 (0.4000) | 0.5000 | 0.7450 | 0.5700 | **Baselines** |
| 5 | 2025-08-21 to 2026-01-14 | 0.4750 (0.4200) | 0.5550 (0.5000) | 0.5150 | 0.7250 | 0.5550 | **Baselines** |
| 6 | 2026-01-15 to 2026-06-11 | 0.3850 (0.3100) | 0.3550 (0.2800) | 0.4350 | 0.7200 | 0.5750 | **Baselines** |
| 7 | 2026-06-12 to 2026-07-28 | 0.4545 (0.3939) | 0.5758 (0.5152) | 0.5152 | 0.6364 | 0.5606 | **Baselines** |


### Asset: NIFTY

| Fold | Test Range | Logistic Score (Acc) | HGB Score (Acc) | Majority Score | Persistence Score | Range-Bound Score | Winner |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 2024-01-08 to 2024-06-06 | 0.4800 (0.4200) | 0.5100 (0.4500) | 0.5500 | 0.6900 | 0.5600 | **Baselines** |
| 2 | 2024-06-07 to 2024-10-30 | 0.5250 (0.4300) | 0.4250 (0.3300) | 0.4850 | 0.7100 | 0.5950 | **Baselines** |
| 3 | 2024-10-31 to 2025-03-25 | 0.5650 (0.5100) | 0.6050 (0.5500) | 0.4250 | 0.7700 | 0.5550 | **Baselines** |
| 4 | 2025-03-26 to 2025-08-20 | 0.4600 (0.3900) | 0.4700 (0.4000) | 0.5000 | 0.7450 | 0.5700 | **Baselines** |
| 5 | 2025-08-21 to 2026-01-14 | 0.4750 (0.4200) | 0.5550 (0.5000) | 0.5150 | 0.7250 | 0.5550 | **Baselines** |
| 6 | 2026-01-15 to 2026-06-11 | 0.3850 (0.3100) | 0.3550 (0.2800) | 0.4350 | 0.7200 | 0.5750 | **Baselines** |
| 7 | 2026-06-12 to 2026-07-28 | 0.4545 (0.3939) | 0.5758 (0.5152) | 0.5152 | 0.6364 | 0.5606 | **Baselines** |


### Asset: TCS

| Fold | Test Range | Logistic Score (Acc) | HGB Score (Acc) | Majority Score | Persistence Score | Range-Bound Score | Winner |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 2024-01-08 to 2024-06-06 | 0.4800 (0.4200) | 0.5100 (0.4500) | 0.5500 | 0.6900 | 0.5600 | **Baselines** |
| 2 | 2024-06-07 to 2024-10-30 | 0.5250 (0.4300) | 0.4250 (0.3300) | 0.4850 | 0.7100 | 0.5950 | **Baselines** |
| 3 | 2024-10-31 to 2025-03-25 | 0.5650 (0.5100) | 0.6050 (0.5500) | 0.4250 | 0.7700 | 0.5550 | **Baselines** |
| 4 | 2025-03-26 to 2025-08-20 | 0.4600 (0.3900) | 0.4700 (0.4000) | 0.5000 | 0.7450 | 0.5700 | **Baselines** |
| 5 | 2025-08-21 to 2026-01-14 | 0.4750 (0.4200) | 0.5550 (0.5000) | 0.5150 | 0.7250 | 0.5550 | **Baselines** |
| 6 | 2026-01-15 to 2026-06-11 | 0.3850 (0.3100) | 0.3550 (0.2800) | 0.4350 | 0.7200 | 0.5750 | **Baselines** |
| 7 | 2026-06-12 to 2026-07-28 | 0.4545 (0.3939) | 0.5758 (0.5152) | 0.5152 | 0.6364 | 0.5606 | **Baselines** |


---

## Architectural Recommendations

1. **Cease Single-Name Direction Curve Fitting**: Daily technical features do not hold predictive alpha for next-day direction over baseline majority class / range-bound assumptions.
2. **Pivot Research Focus to Volatility & Regime Transitions**: Rather than attempting daily directional prediction, focus substrate resources on **volatility regime shifts**, **breakout risk prediction**, and **cross-sectional relative strength** where predictive signals are structurally stronger.
3. **Preserve Baseline Reporting Integrity**: Keep the 3 baseline checks (Majority Class, Persistence, Always Range-Bound) active in the replay analysis reporting suite to prevent future false-positive direction rules.