# Reflective Cognition Substrate — Empirical Research Program Findings

**Repository**: `dp-market-predictability-audit`  
**Git Tag**: `v1.0-research-complete`  
**Date**: 2026-08-05  
**Audience**: Executive & Technical Reviewers (5-Minute Overview)

---

## 1. Core Question & Research Philosophy

The **DP-Core Reflective Cognition Substrate** is an experimental quantitative intelligence framework designed to explore longitudinal cognition, reflective memory, and LLM-guided hypothesis synthesis.

This empirical research program evaluated two foundational questions:
1. **Market Predictability**: Where do monetizable predictive edges exist in Indian equities (NIFTY 50, NIFTY 100, RELIANCE, TCS) across time-series direction, realized volatility, volatility regimes, and cross-sectional rankings?
2. **LLM Reasoning Edge**: Does structured hypothesis synthesis from a Substrate LLM (Ollama/Llama3, reflective memory) generate out-of-sample trading edge statistically superior to pseudo-hypotheses randomly generated from the exact same structural domain grammar $G$?

---

## 2. Methodological & Engineering Laws

All research tracks strictly enforced the substrate's non-negotiable engineering doctrine:
- **Point-in-Time Integrity**: All features, quantile thresholds, and signals computed strictly on historical expanding training windows with zero forward-looking leakage.
- **Pre-Registered Gates**: Hypotheses, evaluation protocols, data splits, and statistical gates frozen in pre-registration documents prior to study execution.
- **Hermetic Offline Testing**: 100% offline verification across 48 unit tests executing with network sockets explicitly disabled.
- **Programmatic Reports**: All summary markdown reports generated directly by code.

---

## 3. Master Program Status & Research Verdict Matrix (All 7 Tracks)

| Track # | Research Scope | Method & Model | Pre-Registered Gate | Empirical Result | Final Verdict | Study Report Link |
| :-: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | **Daily / 3-Day Direction** | Logistic & Boosted Stumps over OHLCV, Breadth, Delivery, FII/DII | Directional Predictability ($MCC > 0.10$) | Mean $MCC \approx 0.00$ on RELIANCE, NIFTY, TCS | 🔴 **`DIRECTION IS DEAD`** | [Section 4.1](#track-1-single-name--index-daily3-day-direction) |
| **2** | **5-Day Realized Volatility** | HAR-RV, EWMA, Gradient Boosting vs Persistence | Realized Vol Edge ($R^2_{\text{vs\_pers}} > 0.20$) | $R^2_{\text{vs\_pers}} = +0.26 \text{ to } +0.40$ out-of-sample | 🟢 **`VOLATILITY IS ALIVE`** | [volatility_options_track_report.md#1-workstream-3-volatility-harness--positive-control-study](volatility_options_track_report.md#1-workstream-3-volatility-harness--positive-control-study) |
| **3** | **India VIX Encompassing** | Mincer-Zarnowitz Joint Regressions & Block Bootstrap | Gate G-VIX / H-V3 ($\beta_B > 0, 95\% \text{ CI} > 0$) | $\beta_B = 0.0410$, $95\% \text{ CI } [-0.2277, +0.3504]$ | 🟢 **`H-V3 CONFIRMED (NO EDGE BEYOND VIX)`** | [volatility_options_track_report.md#2-workstream-1-india-vix-ingestion--mincer-zarnowitz-encompassing-study](volatility_options_track_report.md#2-workstream-1-india-vix-ingestion--mincer-zarnowitz-encompassing-study) |
| **4** | **Options Volatility Strategy** | Corrected Annualized Vol Options Filter (Short Strangle & Iron Condor) | Gate G-OPT (Sortino & MaxDD superiority vs Baseline) | Sortino $-0.23$ vs $-0.08$, MaxDD $3.30\%$ vs $3.26\%$ | 🔴 **`G-OPT FAILED`** | [volatility_options_track_report.md#3-workstream-2-corrected-options-strategy-simulation--gate-g-opt](volatility_options_track_report.md#3-workstream-2-corrected-options-strategy-simulation--gate-g-opt) |
| **5** | **Cross-Sectional Ranking** | Point-in-Time Quintile Portfolios (0.25% Friction Costs) over NIFTY 100 | Gate G-XS1 (IC) & G-XS2 (Sharpe vs Benchmark) | Mean IC = $-0.0063$, Q5 Sharpe = $0.45$ vs Equal-Weight $0.72$ | 🔴 **`CROSS-SECTIONAL NULL`** | [cross_sectional_ranking_report.md](cross_sectional_ranking_report.md) |
| **6** | **LLM Hypothesis Track (Initial)** | Ollama Candidate Hypotheses vs Random Noise Rules ($N=50$) | Gate G-LLM1 & G-LLM2 (Survival Rate Superiority) | Fisher $p=1.0000$, survival rate 0% | ⚠️ **`SUPERSEDED BY PHASE 3 REBUILD`** | [experiments/preregistration/llm_hypothesis_value_prereg.md](experiments/preregistration/llm_hypothesis_value_prereg.md) |
| **7** | **Substrate LLM Hypothesis Survival** | 60/40 IS/OOS Split, Binomial $p<0.10$ Filter, $N=150$ LLM vs $N=150$ Random | Gate G-P3 (Pooled Mann-Whitney $p<0.05$ on OOS Edge) | LLM Survivor Edge = $+0.1948$ vs Random = $+0.2493$ ($p=0.6876$) | 🔴 **`LLM REASONING NULL`** | [llm_hypothesis_survival_report.md](llm_hypothesis_survival_report.md) |

---

## 4. Track-by-Track Executive Summaries

### Track 1: Single-Name & Index Daily/3-Day Direction
- **Question**: Can daily or 3-day return direction be predicted using price/volume, market breadth, delivery ratios, or institutional flow features?
- **Finding**: Out-of-sample Matthews Correlation Coefficient ($MCC$) is $\approx 0.00$ across all models, assets, and feature combinations.
- **Verdict**: 🔴 **`DIRECTION IS DEAD`** — Daily equity direction is statistically unpredictable in daily bar data.

### Track 2: 5-Day Realized Volatility Forecasting
- **Question**: Is 5-day forward realized volatility predictable relative to a persistence baseline?
- **Finding**: Heterogeneous Autoregressive (HAR-RV), EWMA, and Gradient Boosting models reduce forecast error by $39\%\text{--}48\%$ ($R^2_{\text{vs\_pers}} = +0.26 \text{ to } +0.40$).
- **Verdict**: 🟢 **`VOLATILITY IS ALIVE`** — Realized volatility contains strong structural persistence and predictability.

### Track 3: India VIX Encompassing Test (Mincer-Zarnowitz)
- **Question**: Do daily-bar technical features contain monetizable predictive information for 5-day realized volatility beyond India VIX?
- **Finding**: Joint Mincer-Zarnowitz regression shows India VIX encompasses daily-bar features ($\beta_B = 0.0410$, 95% block bootstrap CI includes zero $[-0.2277, +0.3504]$).
- **Verdict**: 🟢 **`H-V3 CONFIRMED (NO EDGE BEYOND VIX)`** — Daily-bar technical features add zero incremental edge over India VIX.

### Track 4: Options Volatility Regime Strategy (Corrected Gate G-OPT)
- **Question**: Does filtering options variance risk premium selling using a model volatility forecast beat unconditional option selling?
- **Finding**: Correcting daily-to-annualized vol conversion ($\hat{\sigma}_{ann} = \hat{\sigma}_{daily} \times \sqrt{252}$) reveals unconditional short-strangle selling yields $+14.30\%$ (Sortino $-0.08$, MaxDD $3.26\%$), outperforming model-filtered policies ($+12.27\%$, Sortino $-0.23$, MaxDD $3.30\%$).
- **Verdict**: 🔴 **`G-OPT FAILED`** — Model regime filtering degrades options selling profitability compared to unconditional variance premium harvesting.

### Track 5: Cross-Sectional Ranking Study
- **Question**: Does cross-sectional ranking across NIFTY 100 constituents over 1-month horizons yield predictive rank IC or tradeable alpha?
- **Finding**: Composite signal Mean Rank IC is $-0.0063$ (statistically indistinguishable from zero permutation null). Top quintile portfolio Sharpe is $0.45$ vs. NIFTY 100 Equal-Weight benchmark Sharpe of $0.72$.
- **Verdict**: 🔴 **`CROSS-SECTIONAL NULL`** — Technical cross-sectional signals contain no predictive edge.

### Track 6 & 7: Substrate LLM Hypothesis Survival Study (Phase 3 Rebuild)
- **Question**: Do trading hypotheses generated by Substrate LLM reasoning survive backtesting at a rate or edge superior to random grammar baseline hypotheses?
- **Finding**: After generating 150 unique LLM hypotheses and 150 unique control random hypotheses, 60/40 IS/OOS backtesting with binomial $p < 0.10$ filtering yields 10 LLM survivors (6.7%) and 6 Random survivors (4.0%). Out-of-sample survivor edge for LLM hypotheses is $+0.1948$ vs $+0.2493$ for random hypotheses (Mann-Whitney $p = 0.6876$).
- **Verdict**: 🔴 **`LLM REASONING NULL`** — Substrate LLM hypothesis generation demonstrates no detectable advantage over random generation (Mann-Whitney p=0.69); test powered only for large effects given survivor pool sizes (10 vs 6).

---

## 5. Key Methodological Limitations

1. **Daily Bar Granularity**: Daily OHLCV data cannot capture intraday market microstructure, order flow imbalances, or intraday gamma exposure.
2. **Black-Scholes Options Model**: Option simulations use synthetic Black-Scholes pricing with VIX as ATM IV; live option chains exhibit skew, smile, and bid-ask friction.
3. **Survivorship Bias**: Backfilling current NIFTY 100 constituents introduces mild survivorship bias, inflating long-side baseline returns.

---

## 6. Repository Evidence Map & Navigation

The codebase and committed datasets serve as ground-truth empirical evidence:

- **Consolidated Master Report**: [`volatility_options_track_report.md`](volatility_options_track_report.md)
- **Cross-Sectional Study Report**: [`cross_sectional_ranking_report.md`](cross_sectional_ranking_report.md)
- **Phase 3 LLM Survival Report**: [`llm_hypothesis_survival_report.md`](llm_hypothesis_survival_report.md)
- **Pre-Registration Documents**: [`experiments/preregistration/`](experiments/preregistration/)
- **Frozen Hypothesis Arms**: [`data/hypotheses/`](data/hypotheses/)
- **Execution Entry Points**: [`bootstrap/`](bootstrap/)
- **Hermetic Unit Test Suite**: [`tests/`](tests/)
