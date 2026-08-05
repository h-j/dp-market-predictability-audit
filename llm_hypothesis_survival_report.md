# Substrate LLM Hypothesis Survival Study Report (Phase 3 Rebuild)

**Date**: 2026-08-05  
**Substrate**: `dp-core-phase1-substrate-v3`  
**Sample Size**: N=150 Substrate LLM Hypotheses vs N=150 Control Random Grammar Hypotheses  
**Final Track Verdict**: **`LLM REASONING NULL`**

## Executive Summary & Single Gate G-P3 Status

| Gate / Metric | Subject | Pre-Registered Condition | Empirical Result | Verdict |
| :--- | :--- | :--- | :---: | :---: |
| **Gate G-P3** | Survivor OOS Edge Superiority | Pooled One-Sided Mann-Whitney p < 0.05 | LLM Survivor Mean OOS Edge=+0.1948 vs Random Survivor Mean OOS Edge=+0.2493 (Mann-Whitney U=26.0, p=0.6876) | 🔴 **FAILED** |

### Scientific Conclusion: **`LLM REASONING NULL`**

Substrate LLM hypothesis generation on daily-bar technical and volatility features demonstrates no statistically significant out-of-sample edge over random grammar synthesis (Mann-Whitney p = 0.6876). While both arms yield a small set of in-sample surviving hypotheses with positive out-of-sample edge (+19.48% vs +24.93%), LLM-guided prompt synthesis fails to generate superior out-of-sample performance compared to random structural sampling.

## Arm Summary & In-Sample Survival Breakdown

| Hypothesis Arm | Total N | IS-Survivors (p < 0.10) | IS Survival Rate (%) | Survivor Mean OOS Edge | Survivor Median OOS Edge | All-Hypotheses Mean OOS Edge |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Substrate LLM (H_LLM)** | 150 | 10 | **6.7%** | **+0.1948** | +0.2021 | +0.0019 |
| **Random Grammar (H_Random)** | 150 | 6 | **4.0%** | **+0.2493** | +0.1975 | +0.0238 |

## Top 5 Surviving Substrate LLM Hypotheses

| Hypothesis ID | Target Mode | IS Hits/Triggers | IS Binomial p | OOS Triggers | OOS Edge |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **H_LLM_117** | volatility_5d | 6/6 | 0.0761 | 4 | **+0.3944** |
| **H_LLM_129** | volatility_5d | 10/10 | 0.0137 | 5 | **+0.3944** |
| **H_LLM_080** | vol_regime_expansion | 14/29 | 0.0959 | 13 | **+0.2979** |
| **H_LLM_084** | volatility_5d | 19/20 | 0.0022 | 9 | **+0.2833** |
| **H_LLM_148** | volatility_5d | 13/14 | 0.0209 | 9 | **+0.2833** |

## Top 5 Surviving Control Random Hypotheses

| Hypothesis ID | Target Mode | IS Hits/Triggers | IS Binomial p | OOS Triggers | OOS Edge |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **H_RAND_149** | vol_regime_expansion | 5/5 | 0.0052 | 2 | **+0.6056** |
| **H_RAND_096** | volatility_5d | 13/21 | 0.0105 | 11 | **+0.2420** |
| **H_RAND_104** | vol_regime_expansion | 13/21 | 0.0105 | 11 | **+0.2420** |
| **H_RAND_118** | direction_3d | 38/45 | 0.0034 | 29 | **+0.1530** |
| **H_RAND_147** | vol_regime_expansion | 23/27 | 0.0187 | 16 | **+0.1444** |

---

## Verbatim Honesty Caveats

> Substrate LLM hypotheses are evaluated strictly out-of-sample over expanding historical walk-forward windows. If LLM hypothesis survival rates do not exceed random grammar generation at p < 0.05, the LLM reasoning pipeline provides zero monetizable or predictive value beyond random rule synthesis.
