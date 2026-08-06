# Pre-Registration: Cross-Sectional Ranking Study (Phase 2b)

**Date**: 2026-08-05  
**Repository Substrate**: `dp-market-predictability-audit`  
**Study Scope**: Cross-Sectional Ranking Signal & Portfolio Evaluation  
**Status**: PRE-REGISTERED (FROZEN PRIOR TO STUDY RUN)

---

## 1. Study Motivation & Context

Following certified empirical conclusions from prior substrate tracks:
1. Single-name daily/3-day direction prediction is dead ($MCC \approx 0.00$ across all assets/models).
2. 5-day realized volatility is predictable ($R^2_{vs\_pers} = +0.26 \text{ to } +0.40$), but daily-bar features contain no monetizable edge beyond India VIX (**H-V3** confirmed).
3. Option volatility regime filtering (**Gate G-OPT**) failed to beat unconditional variance risk premium harvesting.

This final information-value experiment tests the last open edge class: **cross-sectional ranking**—predicting which stocks relative to their peers outperform over a monthly horizon, rather than whether individual stocks or indices rise in direction.

---

## 2. Universe & Data Specification

- **Universe**: Current NIFTY 100 constituent tickers (`config/universe_nifty100.py`), hardcoded as-of `2026-08-05`.
- **Exclusion Rule**: Exclude stocks with $< 15$ months of trading history at any given rebalance date.
- **Survivorship Bias Caveat**: Using current constituents backfilled creates survivorship bias on the long side. This limitation is acknowledged explicitly; any marginal pass verdict will be treated as suspect for this reason.
- **Data Range**: Daily adjusted OHLCV from `2022-06-01` to present (providing $\ge 6$ months lookback prior to first rebalance).

---

## 3. Signals & Formulas

Signals are computed at the last trading day of each month $t$ using data strictly $\le t$. Each raw signal is cross-sectionally percentile-ranked to $[0.0, 1.0]$ across eligible universe stocks at date $t$.

1. `mom_6m1m`: 6-month return skipping trailing 1 month:
   $$\text{Signal}_i(t) = \frac{\text{Close}_i(t - 21\text{d})}{\text{Close}_i(t - 126\text{d})} - 1.0$$
2. `rev_1m`: Negative of trailing 1-month return (short-term reversal):
   $$\text{Signal}_i(t) = 1.0 - \frac{\text{Close}_i(t)}{\text{Close}_i(t - 21\text{d})}$$
3. `rs_nifty_3m`: 3-month return relative to NIFTY index:
   $$\text{Signal}_i(t) = \left(\frac{\text{Close}_i(t)}{\text{Close}_i(t - 63\text{d})} - 1.0\right) - \left(\frac{\text{NIFTY}(t)}{\text{NIFTY}(t - 63\text{d})} - 1.0\right)$$
4. `vol_3m_inv`: Inverse of 3-month realized volatility:
   $$\text{Signal}_i(t) = \frac{1.0}{\text{std}\big(\text{DailyReturn}_i[t-63\text{d} : t]\big)}$$
5. `vol_trend`: Ratio of 1-month average volume to 6-month average volume:
   $$\text{Signal}_i(t) = \frac{\text{mean}\big(\text{Volume}_i[t-21\text{d} : t]\big)}{\text{mean}\big(\text{Volume}_i[t-126\text{d} : t]\big)}$$
6. `composite`: Equal-weighted average of percentile ranks of Signals 1–5:
   $$\text{Rank}_{\text{composite}, i}(t) = \frac{1}{5} \sum_{k=1}^{5} \text{Rank}_{k, i}(t)$$

---

## 4. Evaluation Cadence & Portfolio Engine

- **Cadence**: Monthly rebalance on the last trading day of each month (~34 rebalance dates from 2023 to present).
- **Target**: Next calendar month's total return per stock ($r_{i, t+1\text{m}}$).
- **Portfolio Construction**: Equal-weighted quintile portfolios (Q1 = Bottom 20%, Q5 = Top 20%).
- **Transaction Costs**: **0.25% one-way cost** applied to portfolio turnover at each rebalance.

---

## 5. Metrics & Baselines

1. **Rank IC**: Spearman rank correlation between signal rank and forward-month return, computed per rebalance month.
   - Metrics: Mean IC, $t$-statistic across ~34 months, 95% stationary bootstrap CI over months.
2. **Quintile Spread**: Mean forward return of Q5 (Top) minus Q1 (Bottom), annualized.
3. **Long-Only Implementable Test**: Equal-weighted Top Quintile (Q5) portfolio vs NIFTY total return:
   - Metrics: CAGR, Sharpe ratio, Max Drawdown, Turnover, Active Return.
4. **Baselines**:
   - **Permutation Null**: 1,000 random rank permutations per month forming the empirical null IC distribution.
   - **NIFTY Buy-and-Hold**: Benchmark for long-only comparison.

---

## 6. Pre-Registered Gates

> [!IMPORTANT]
> - **G-XS1 (Signal Existence)**: Composite mean IC > 0 with 95% stationary bootstrap CI excluding 0 **AND** exceeding the 97.5th percentile of the 1,000-permutation empirical null distribution.
> - **G-XS2 (Retail Implementable)**: Equal-weighted Top-Quintile (Q5) long-only portfolio beats NIFTY buy-and-hold on Sharpe ratio after **0.25% one-way transaction cost on turnover** over the full OOS evaluation window.

### Pre-Registered Verdict Matrix

| G-XS1 (Signal) | G-XS2 (Implementable) | Final Track Verdict | Next Action |
| :---: | :---: | :--- | :--- |
| **PASS** | **PASS** | **`EDGE-CANDIDATE`** | Proceed to 6-month paper-trade protocol (separate track). |
| **PASS** | **FAIL** | **`UNIMPLEMENTABLE SIGNAL`** | Signal is statistically real but not retail-implementable at 0.25% turnover cost. |
| **FAIL** | **FAIL** | **`CROSS-SECTIONAL NULL`** | Cross-sectional ranking contains no monetizable edge. Empirical phase closes. |

---

## 7. Verbatim Honesty & Survivorship Caveat

> Backfilling current NIFTY 100 constituent tickers introduces survivorship bias, inflating long-side returns. Any passing verdict must be treated as upper-bound candidate edge pending historical constituent verification.
