# India VIX Encompassing Study Report (Workstream 1)

## Executive Summary

Evaluated out-of-sample Mincer-Zarnowitz encompassing regressions across `NIFTY` (primary), `RELIANCE`, and `TCS` to test whether daily bar features retain incremental predictive power over implied volatility (India VIX).

## Summary Table: Model Comparisons & Encompassing Tests

| Asset | Folds | Model A R² (VIX Only) | Model B R² (No VIX) | Model C R² (Model B+VIX) | Beta A (VIX) | Beta B (Model B) | 95% CI (Beta B) | Beta B > 0? | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **NIFTY** | 6 | +0.3497 | +0.2697 | **+0.2565** | 0.1952 | 0.0410 | [-0.2277, +0.3504] | NO | NO-EDGE |
| **RELIANCE** | 7 | +0.2416 | +0.2711 | **+0.2394** | -0.4383 | 0.2377 | [-0.2571, +0.7315] | NO | NO-EDGE |
| **TCS** | 7 | +0.4038 | +0.3162 | **+0.3114** | 0.6092 | 0.2304 | [-0.1528, +0.5689] | NO | NO-EDGE |

---

## Pre-Registered Hypotheses Verdicts

- **H-V1 (NIFTY Implied Vol Subsumption)**: CONFIRMED (Model A R² = +0.3497 vs Model B R² = +0.2697)
- **H-V2 (Single-Name Dual Channel)**: RELIANCE = REJECTED, TCS = REJECTED
- **H-V3 (NIFTY Monetizable Edge)**: CONFIRMED (No Edge, Beta_B <= 0)