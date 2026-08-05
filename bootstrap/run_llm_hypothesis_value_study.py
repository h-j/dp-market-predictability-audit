"""
Substrate LLM Hypothesis Survival vs. Random Grammar Baseline Study Runner (Phase 4 Track).

Executes backtesting of 50 Substrate LLM hypotheses vs 50 Random Grammar hypotheses over expanding walk-forward folds.
Evaluates Pre-Registered Gates G-LLM1 and G-LLM2 via Fisher's Exact Test, Mann-Whitney U test, and bootstrap CIs.
Outputs llm_hypothesis_value_report.md and updates volatility_options_track_report.md.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from cognition.grammar.llm_hypothesis_generator import SubstrateLLMHypothesisGenerator
from cognition.grammar.random_hypothesis_generator import RandomHypothesisGenerator
from market.data.vix_loader import IndiaVIXLoader
from market.eval.hypothesis_evaluator import (
    GroupEvaluationSummary,
    HypothesisBacktestResult,
    HypothesisWalkForwardEvaluator,
    StudyStatisticalGates,
)
from market.replay.walkforward_validation import WalkForwardValidator

HONESTY_CAVEAT_VERBATIM = (
    "Substrate LLM hypotheses are evaluated strictly out-of-sample over expanding historical walk-forward windows. "
    "If LLM hypothesis survival rates do not exceed random grammar generation at p < 0.05, the LLM reasoning pipeline "
    "provides zero monetizable or predictive value beyond random rule synthesis."
)


def load_enriched_asset_data(asset_name: str, data_dir: Path) -> pd.DataFrame:
    """
    Load committed asset CSV and merge India VIX features.
    """
    p_enriched = data_dir / f"{asset_name.lower()}_enriched_daily_3y.csv"
    p_raw = data_dir / f"{asset_name.lower()}_daily_3y.csv"
    target_p = p_enriched if p_enriched.exists() else p_raw

    df_raw = pd.read_csv(target_p)
    vix_loader = IndiaVIXLoader(data_dir=data_dir)
    df_enriched = vix_loader.merge_vix_features(df_raw)

    validator = WalkForwardValidator(
        df_enriched, asset_name=asset_name, target_mode="volatility_5d", initial_train_size=250, test_fold_size=100
    )
    return validator.df.copy()


def run_llm_hypothesis_value_study(data_dir: Path = None):
    """
    Main study runner for Phase 4 Track: LLM Reasoning Value vs Random Grammar Baseline.
    """
    print("=" * 80)
    print("SUBSTRATE LLM HYPOTHESIS SURVIVAL VS. RANDOM GRAMMAR BASELINE STUDY (Phase 4)")
    print("=" * 80)

    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    df_nifty = load_enriched_asset_data("NIFTY", data_dir)
    print(f"✓ Loaded enriched NIFTY dataset ({len(df_nifty)} rows)")

    # 1. Generate Candidates (N=50 LLM vs N=50 Random)
    llm_gen = SubstrateLLMHypothesisGenerator(seed=42)
    llm_hyps = llm_gen.generate_llm_hypotheses(count=50)

    rand_gen = RandomHypothesisGenerator(seed=42)
    rand_hyps = rand_gen.generate_hypotheses(count=50)

    print(f"✓ Generated {len(llm_hyps)} Substrate LLM hypotheses (H_LLM)")
    print(f"✓ Generated {len(rand_hyps)} Control Random Grammar hypotheses (H_Random)")

    # 2. Backtest Hypotheses
    evaluator = HypothesisWalkForwardEvaluator(n_bootstrap=2000, seed=42)
    llm_results: List[HypothesisBacktestResult] = []
    rand_results: List[HypothesisBacktestResult] = []

    print("\n--- Backtesting Substrate LLM Hypotheses (H_LLM, N=50) ---")
    for i, h in enumerate(llm_hyps):
        res = evaluator.evaluate_hypothesis(h, df_nifty, asset_name="NIFTY")
        llm_results.append(res)
        surv_str = "SURVIVED" if res.survived else "FAILED"
        if (i + 1) % 10 == 0 or res.survived:
            print(f"  • [{i+1:02d}/50] {res.hypothesis_id}: R²(vs pers)={res.r2_vs_persistence:+.4f} | MCC={res.binary_mcc:+.4f} | Triggers={res.total_triggers_oos:3d} [{surv_str}]")

    print("\n--- Backtesting Control Random Hypotheses (H_Random, N=50) ---")
    for i, h in enumerate(rand_hyps):
        res = evaluator.evaluate_hypothesis(h, df_nifty, asset_name="NIFTY")
        rand_results.append(res)
        surv_str = "SURVIVED" if res.survived else "FAILED"
        if (i + 1) % 10 == 0 or res.survived:
            print(f"  • [{i+1:02d}/50] {res.hypothesis_id}: R²(vs pers)={res.r2_vs_persistence:+.4f} | MCC={res.binary_mcc:+.4f} | Triggers={res.total_triggers_oos:3d} [{surv_str}]")

    # 3. Statistical Gate Evaluations
    sum_llm, sum_rand, gates = evaluator.evaluate_study_gates(llm_results, rand_results)

    print("\n" + "=" * 80)
    print("HYPOTHESIS SURVIVAL & STATISTICAL GATE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Treatment (H_LLM)    : Survived {sum_llm.survived_count}/{sum_llm.total_hypotheses} ({sum_llm.survival_rate_pct:.1f}%) | Mean R²={sum_llm.mean_r2_vs_pers:+.4f} | Median R²={sum_llm.median_r2_vs_pers:+.4f}")
    print(f"Control   (H_Random) : Survived {sum_rand.survived_count}/{sum_rand.total_hypotheses} ({sum_rand.survival_rate_pct:.1f}%) | Mean R²={sum_rand.mean_r2_vs_pers:+.4f} | Median R²={sum_rand.median_r2_vs_pers:+.4f}")
    print("-" * 80)
    print(f"Fisher's Exact Test p-value: {gates.fisher_p_value:.4f} (Odds Ratio={gates.fisher_odds_ratio:.4f})")
    print(f"PRE-REGISTERED GATE G-LLM1 (Survival Advantage): {'PASSED' if gates.gate_g_llm1_passed else 'FAILED'}")
    print("-" * 80)
    print(f"Mann-Whitney U Test p-value: {gates.mann_whitney_p_value:.4f} (U={gates.mann_whitney_u_stat:.1f})")
    print(f"Bootstrap Mean Diff (LLM - Random): {gates.bootstrap_diff_mean:+.4f} (95% CI=[{gates.bootstrap_diff_ci_lower:+.4f}, {gates.bootstrap_diff_ci_upper:+.4f}])")
    print(f"PRE-REGISTERED GATE G-LLM2 (Mean Metric Superiority): {'PASSED' if gates.gate_g_llm2_passed else 'FAILED'}")
    print("=" * 80)
    print(f"FINAL TRACK VERDICT: {gates.final_verdict}")
    print("=" * 80 + "\n")

    generate_markdown_reports(llm_results, rand_results, sum_llm, sum_rand, gates)


def generate_markdown_reports(
    llm_results: List[HypothesisBacktestResult],
    rand_results: List[HypothesisBacktestResult],
    sum_llm: GroupEvaluationSummary,
    sum_rand: GroupEvaluationSummary,
    gates: StudyStatisticalGates,
):
    report_lines = []
    report_lines.append("# Substrate LLM Hypothesis Survival vs. Random Grammar Baseline Study Report (Phase 4)")
    report_lines.append("\n**Date**: 2026-08-05  ")
    report_lines.append("**Substrate**: `dp-core-phase1-substrate-v3`  ")
    report_lines.append(f"**Sample Size**: N=50 Substrate LLM Hypotheses vs N=50 Control Random Grammar Hypotheses  ")
    report_lines.append(f"**Final Track Verdict**: **`{gates.final_verdict}`**\n")

    report_lines.append("## Executive Summary & Pre-Registered Gate Summary\n")
    report_lines.append("| Gate / Metric | Pre-Registered Condition | Empirical Result | Verdict |")
    report_lines.append("| :--- | :--- | :---: | :---: |")

    g1_res = f"LLM Survival={sum_llm.survival_rate_pct:.1f}% vs Random={sum_rand.survival_rate_pct:.1f}% (Fisher p={gates.fisher_p_value:.4f})"
    g1_v = "🟢 **PASSED**" if gates.gate_g_llm1_passed else "🔴 **FAILED**"
    report_lines.append(f"| **G-LLM1 (Survival Advantage)** | Fisher p < 0.05 & LLM Survival > Random | {g1_res} | {g1_v} |")

    g2_res = f"Mann-Whitney p={gates.mann_whitney_p_value:.4f}, Bootstrap Mean Diff={gates.bootstrap_diff_mean:+.4f} (95% CI=[{gates.bootstrap_diff_ci_lower:+.4f}, {gates.bootstrap_diff_ci_upper:+.4f}])"
    g2_v = "🟢 **PASSED**" if gates.gate_g_llm2_passed else "🔴 **FAILED**"
    report_lines.append(f"| **G-LLM2 (Mean Metric Superiority)** | Mann-Whitney p < 0.05 & Bootstrap CI > 0 | {g2_res} | {g2_v} |")

    report_lines.append(f"\n### Scientific Conclusion: **`{gates.final_verdict}`**\n")
    if gates.final_verdict == "LLM REASONING VALIDATED":
        report_lines.append("Substrate LLM reasoning produces trading hypotheses that survive out-of-sample walk-forward backtesting at a statistically significantly higher rate than randomly generated hypotheses from the exact same structural grammar.")
    else:
        report_lines.append("Substrate LLM hypothesis generation is statistically indistinguishable from (or inferior to) randomly generated hypotheses from the exact same structural grammar (Fisher p = {:.4f}, Mann-Whitney p = {:.4f}). Substrate hypothesis generation exhibits pseudo-reasoning / prompt overfitting with zero incremental predictive value over random rule synthesis.".format(gates.fisher_p_value, gates.mann_whitney_p_value))

    report_lines.append("\n## Group Summary Comparison Table\n")
    report_lines.append("| Hypothesis Source | Total N | Survived Count | Survival Rate (%) | Mean OOS R² (vs Pers) | Median OOS R² | Mean Binary MCC |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    report_lines.append(f"| **Substrate LLM (H_LLM)** | {sum_llm.total_hypotheses} | {sum_llm.survived_count} | **{sum_llm.survival_rate_pct:.1f}%** | **{sum_llm.mean_r2_vs_pers:+.4f}** | {sum_llm.median_r2_vs_pers:+.4f} | {sum_llm.mean_mcc:+.4f} |")
    report_lines.append(f"| **Random Grammar (H_Random)** | {sum_rand.total_hypotheses} | {sum_rand.survived_count} | **{sum_rand.survival_rate_pct:.1f}%** | **{sum_rand.mean_r2_vs_pers:+.4f}** | {sum_rand.median_r2_vs_pers:+.4f} | {sum_rand.mean_mcc:+.4f} |")

    report_lines.append("\n## Top 5 Performing Substrate LLM Hypotheses\n")
    report_lines.append("| Hypothesis ID | Target Mode | OOS Triggers | OOS R² (vs Pers) | Binary MCC | Status |")
    report_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    sorted_llm = sorted(llm_results, key=lambda x: x.r2_vs_persistence, reverse=True)[:5]
    for r in sorted_llm:
        st = "**SURVIVED**" if r.survived else "FAILED"
        report_lines.append(f"| **{r.hypothesis_id}** | {r.target_mode} | {r.total_triggers_oos} | {r.r2_vs_persistence:+.4f} | {r.binary_mcc:+.4f} | {st} |")

    report_lines.append("\n## Top 5 Performing Control Random Hypotheses\n")
    report_lines.append("| Hypothesis ID | Target Mode | OOS Triggers | OOS R² (vs Pers) | Binary MCC | Status |")
    report_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    sorted_rand = sorted(rand_results, key=lambda x: x.r2_vs_persistence, reverse=True)[:5]
    for r in sorted_rand:
        st = "**SURVIVED**" if r.survived else "FAILED"
        report_lines.append(f"| **{r.hypothesis_id}** | {r.target_mode} | {r.total_triggers_oos} | {r.r2_vs_persistence:+.4f} | {r.binary_mcc:+.4f} | {st} |")

    report_lines.append("\n---\n")
    report_lines.append("## Verbatim Honesty Caveats\n")
    report_lines.append(f"> {HONESTY_CAVEAT_VERBATIM}\n")

    report_text = "\n".join(report_lines)
    out_path = Path(__file__).parent.parent / "llm_hypothesis_value_report.md"
    out_path.write_text(report_text)
    print(f"✓ Saved LLM Hypothesis Value Study Report to {out_path}")

    update_volatility_options_track_report(gates, sum_llm, sum_rand)


def update_volatility_options_track_report(
    gates: StudyStatisticalGates, sum_llm: GroupEvaluationSummary, sum_rand: GroupEvaluationSummary
):
    track_report_path = Path(__file__).parent.parent / "volatility_options_track_report.md"
    if not track_report_path.exists():
        return

    content = track_report_path.read_text()

    summary_addition = f"""

---

## 7. Phase 4: Substrate LLM Hypothesis Survival vs. Random Grammar Study

**Status**: Completed  
**Final Track Verdict**: **`{gates.final_verdict}`**

| Gate / Hypothesis | Subject | Pre-Registered Condition | Result | Verdict |
| :--- | :--- | :--- | :---: | :---: |
| **G-LLM1** | LLM Hypothesis Survival Advantage | Fisher p < 0.05 & LLM Survival > Random | LLM={sum_llm.survival_rate_pct:.1f}% vs Random={sum_rand.survival_rate_pct:.1f}% (Fisher p={gates.fisher_p_value:.4f}) | {"🟢 **PASSED**" if gates.gate_g_llm1_passed else "🔴 **FAILED**"} |
| **G-LLM2** | Mean Out-of-Sample Metric Superiority | Mann-Whitney p < 0.05 & Bootstrap CI > 0 | MW p={gates.mann_whitney_p_value:.4f}, Mean Diff={gates.bootstrap_diff_mean:+.4f} (95% CI=[{gates.bootstrap_diff_ci_lower:+.4f}, {gates.bootstrap_diff_ci_upper:+.4f}]) | {"🟢 **PASSED**" if gates.gate_g_llm2_passed else "🔴 **FAILED**"} |

> {HONESTY_CAVEAT_VERBATIM}
"""

    if "Phase 4: Substrate LLM Hypothesis Survival" not in content:
        content += summary_addition
        track_report_path.write_text(content)
        print(f"✓ Appended Phase 4 summary section to {track_report_path}")


if __name__ == "__main__":
    run_llm_hypothesis_value_study()
