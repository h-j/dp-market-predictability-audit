# Study Pre-registration: India VIX Encompassing Study (Workstream 1)

**Registration Date**: 2026-08-05  
**Research Track**: Volatility & Options-Flow Research Track (Phase 3)  
**Target Variable**: Continuous 5-Day Out-of-Sample Forward Realized Volatility ($RealizedVol_{5d, t+1..t+5}$)  
**Evaluation Cadence**: Expanding-window walk-forward cross-validation (initial train size=250, test fold size=100), stride-5 non-overlapping evaluation.

---

## Models Evaluated

- **Model A (VIX-Only)**: Univariate regression ($RealizedVol_{5d} \sim VIX_{close}$, walk-forward refit per fold).
- **Model B (Repo Best - No VIX)**: HAR-RV lag features + Gradient Boosting Regressor (GBM) calibrated on daily/weekly/monthly realized volatility and asset features without VIX.
- **Model C (Repo Best + VIX)**: Model B features + point-in-time VIX features (`vix_close`, `vix_change_5d`, `vix_vs_20d_ma`, `vix_percentile_252d`).

---

## Pre-Registered Hypotheses (Verbatim)

- **H-V1**: For NIFTY, Model A (VIX-only) $\ge$ Model B — implied vol subsumes daily-bar features at index level.
- **H-V2**: For single names, Model C > A and C > B — both channels contribute.
- **H-V3 (the money question)**: Model B's encompassing coefficient for NIFTY is **NOT** significantly > 0. If it IS, that increment is the first candidate monetizable edge and gets flagged `EDGE-CANDIDATE` in the report.

---

## Statistical Methodology: Mincer-Zarnowitz Encompassing Test

For each asset, out-of-sample forecast vectors $\hat{y}_A$ and $\hat{y}_B$ are generated strictly out-of-sample. A joint regression is estimated:

$$RealizedVol_{5d} = \alpha + \beta_A \hat{y}_A + \beta_B \hat{y}_B + \epsilon$$

Hypothesis testing on $\beta_B > 0$ is conducted using non-parametric **circular block bootstrap** (block size=4, 2,000 resamples, 95% confidence interval). If the lower bound of the 95% CI for $\beta_B$ is strictly $> 0$, we reject the null hypothesis that Model B contains no incremental predictive information beyond Model A.
