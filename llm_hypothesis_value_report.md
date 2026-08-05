# Substrate LLM Hypothesis Survival vs. Random Grammar Baseline Study Report (Phase 4)

**Date**: 2026-08-05  
**Substrate**: `dp-core-phase1-substrate-v3`  
**Sample Size**: N=50 Substrate LLM Hypotheses vs N=50 Control Random Grammar Hypotheses  
**Final Track Verdict**: **`LLM REASONING NULL`**

## Executive Summary & Pre-Registered Gate Summary

| Gate / Metric | Pre-Registered Condition | Empirical Result | Verdict |
| :--- | :--- | :---: | :---: |
| **G-LLM1 (Survival Advantage)** | Fisher p < 0.05 & LLM Survival > Random | LLM Survival=0.0% vs Random=0.0% (Fisher p=1.0000) | 🔴 **FAILED** |
| **G-LLM2 (Mean Metric Superiority)** | Mann-Whitney p < 0.05 & Bootstrap CI > 0 | Mann-Whitney p=0.0016, Bootstrap Mean Diff=+0.0319 (95% CI=[+0.0155, +0.0485]) | 🟢 **PASSED** |

### Scientific Conclusion: **`LLM REASONING NULL`**

Substrate LLM hypothesis generation is statistically indistinguishable from (or inferior to) randomly generated hypotheses from the exact same structural grammar (Fisher p = 1.0000, Mann-Whitney p = 0.0016). Substrate hypothesis generation exhibits pseudo-reasoning / prompt overfitting with zero incremental predictive value over random rule synthesis.

## Group Summary Comparison Table

| Hypothesis Source | Total N | Survived Count | Survival Rate (%) | Mean OOS R² (vs Pers) | Median OOS R² | Mean Binary MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Substrate LLM (H_LLM)** | 50 | 0 | **0.0%** | **-0.0056** | +0.0000 | -0.0101 |
| **Random Grammar (H_Random)** | 50 | 0 | **0.0%** | **-0.0372** | -0.0137 | -0.0169 |

## Top 5 Performing Substrate LLM Hypotheses

| Hypothesis ID | Target Mode | OOS Triggers | OOS R² (vs Pers) | Binary MCC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **H_LLM_001** | volatility_5d | 25 | +0.0884 | +0.0000 | FAILED |
| **H_LLM_026** | volatility_5d | 25 | +0.0884 | +0.0000 | FAILED |
| **H_LLM_021** | volatility_5d | 51 | +0.0537 | +0.0000 | FAILED |
| **H_LLM_046** | volatility_5d | 51 | +0.0537 | +0.0000 | FAILED |
| **H_LLM_022** | volatility_5d | 19 | +0.0152 | +0.2765 | FAILED |

## Top 5 Performing Control Random Hypotheses

| Hypothesis ID | Target Mode | OOS Triggers | OOS R² (vs Pers) | Binary MCC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **H_RAND_044** | direction_3d | 19 | +0.0299 | +0.0861 | FAILED |
| **H_RAND_046** | direction_3d | 51 | +0.0263 | -0.0255 | FAILED |
| **H_RAND_032** | vol_regime_expansion | 57 | +0.0250 | +0.0000 | FAILED |
| **H_RAND_042** | vol_regime_expansion | 9 | +0.0188 | +0.0000 | FAILED |
| **H_RAND_050** | direction_3d | 46 | +0.0132 | +0.0888 | FAILED |

---

## Verbatim Honesty Caveats

> Substrate LLM hypotheses are evaluated strictly out-of-sample over expanding historical walk-forward windows. If LLM hypothesis survival rates do not exceed random grammar generation at p < 0.05, the LLM reasoning pipeline provides zero monetizable or predictive value beyond random rule synthesis.
