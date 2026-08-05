"""
Walk-Forward Hypothesis Evaluator & Statistical Significance Engine.

Backtests candidate AST hypotheses (H_LLM vs H_Random) over expanding-window walk-forward folds.
Computes Fisher's Exact Test for survival rates (Gate G-LLM1), Mann-Whitney U test, and 2,000-resample bootstrap CIs (Gate G-LLM2).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from cognition.grammar.hypothesis_grammar import GrammarCompiler, HypothesisAST, compute_feature_quantiles
from market.replay.walkforward_validation import (
    WalkForwardValidator,
    compute_mcc,
    compute_qlike,
    compute_r2_vs_persistence,
)


@dataclass
class HypothesisBacktestResult:
    hypothesis_id: str
    source: str
    target_mode: str
    total_triggers_oos: int
    r2_vs_persistence: float
    qlike_loss: float
    binary_mcc: float
    survived: bool


@dataclass
class GroupEvaluationSummary:
    group_name: str  # "LLM" or "RANDOM"
    total_hypotheses: int
    survived_count: int
    survival_rate_pct: float
    mean_r2_vs_pers: float
    std_r2_vs_pers: float
    median_r2_vs_pers: float
    mean_mcc: float


@dataclass
class StudyStatisticalGates:
    fisher_odds_ratio: float
    fisher_p_value: float
    gate_g_llm1_passed: bool
    mann_whitney_u_stat: float
    mann_whitney_p_value: float
    bootstrap_diff_mean: float
    bootstrap_diff_ci_lower: float
    bootstrap_diff_ci_upper: float
    gate_g_llm2_passed: bool
    final_verdict: str


class HypothesisWalkForwardEvaluator:
    """
    Backtests AST hypotheses across assets using point-in-time expanding windows.
    """

    def __init__(self, n_bootstrap: int = 2000, seed: int = 42):
        self.compiler = GrammarCompiler()
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def evaluate_hypothesis(
        self, ast: HypothesisAST, df_asset: pd.DataFrame, asset_name: str = "NIFTY"
    ) -> HypothesisBacktestResult:
        """
        Backtest a single AST hypothesis on df_asset using expanding walk-forward folds.
        """
        validator = WalkForwardValidator(
            df_asset, asset_name=asset_name, target_mode="volatility_5d", initial_train_size=250, test_fold_size=100
        )
        df = validator.df.copy()
        folds = validator.generate_expanding_folds()

        y_true_all = df["target_volatility_5d"].values
        pers_all = df["rv_5d"].values
        y_regime_all = df["target_vol_regime"].values

        oos_preds = []
        oos_actuals = []
        oos_pers = []
        oos_regimes = []
        trigger_count = 0

        for train_idx, test_idx in folds:
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            # Point-in-time threshold map computed strictly on train history
            feature_quantiles = compute_feature_quantiles(df_train)
            evaluator_fn = self.compiler.compile_hypothesis(ast, feature_quantiles)

            # Evaluate test fold (stride 5)
            test_stride_idx = np.arange(0, len(df_test), 5)
            for idx in test_stride_idx:
                row = df_test.iloc[idx]
                global_idx = test_idx[idx]

                is_triggered = evaluator_fn(row)

                # Prediction logic: if triggered, apply prediction signal multiplier to current RV
                if is_triggered:
                    trigger_count += 1
                    base_rv = float(row.get("rv_5d", 1.0))
                    # signal > 0 -> expansion (+20%), signal < 0 -> compression (-20%)
                    mult = 1.20 if ast.prediction_signal > 0 else 0.80
                    pred_vol = base_rv * mult
                else:
                    pred_vol = float(row.get("rv_5d", 1.0))  # Fallback to persistence

                oos_preds.append(pred_vol)
                oos_actuals.append(y_true_all[global_idx])
                oos_pers.append(pers_all[global_idx])
                oos_regimes.append(y_regime_all[global_idx])

        preds_arr = np.array(oos_preds)
        actuals_arr = np.array(oos_actuals)
        pers_arr = np.array(oos_pers)
        regimes_arr = np.array(oos_regimes)

        r2_vs_pers = compute_r2_vs_persistence(actuals_arr, preds_arr, pers_arr)
        qlike_loss = compute_qlike(actuals_arr, preds_arr)

        pred_regime_rise = (preds_arr > pers_arr).astype(int)
        binary_mcc = compute_mcc(regimes_arr, pred_regime_rise)

        # Pre-registered Survival Criterion: R2_vs_pers > 0.05 AND binary_mcc > 0.10
        survived = (r2_vs_pers > 0.05) and (binary_mcc > 0.10)

        return HypothesisBacktestResult(
            hypothesis_id=ast.hypothesis_id,
            source=ast.source,
            target_mode=ast.target_mode,
            total_triggers_oos=trigger_count,
            r2_vs_persistence=round(float(r2_vs_pers), 4),
            qlike_loss=round(float(qlike_loss), 4),
            binary_mcc=round(float(binary_mcc), 4),
            survived=survived,
        )

    def evaluate_study_gates(
        self, llm_results: List[HypothesisBacktestResult], random_results: List[HypothesisBacktestResult]
    ) -> Tuple[GroupEvaluationSummary, GroupEvaluationSummary, StudyStatisticalGates]:
        """
        Evaluate Gate G-LLM1 (Fisher's Exact Test for survival rates) and Gate G-LLM2 (Mann-Whitney U test & Bootstrap CIs).
        """
        llm_survived = sum(1 for r in llm_results if r.survived)
        rand_survived = sum(1 for r in random_results if r.survived)

        n_llm = len(llm_results)
        n_rand = len(random_results)

        llm_survival_pct = (llm_survived / n_llm * 100.0) if n_llm > 0 else 0.0
        rand_survival_pct = (rand_survived / n_rand * 100.0) if n_rand > 0 else 0.0

        # 2x2 Contingency Table for Fisher's Exact Test: [[LLM_surv, LLM_failed], [RAND_surv, RAND_failed]]
        contingency_table = [
            [llm_survived, n_llm - llm_survived],
            [rand_survived, n_rand - rand_survived],
        ]
        odds_ratio, fisher_p = fisher_exact(contingency_table, alternative="greater")

        # Gate G-LLM1 Passed IF Fisher p-value < 0.05 AND LLM survival rate > Random survival rate
        gate_g_llm1_passed = (fisher_p < 0.05) and (llm_survival_pct > rand_survival_pct)

        # Out-of-sample R2 metrics arrays
        llm_r2s = np.array([r.r2_vs_persistence for r in llm_results])
        rand_r2s = np.array([r.r2_vs_persistence for r in random_results])

        # Mann-Whitney U test (one-sided: LLM > Random)
        u_stat, mw_p = mannwhitneyu(llm_r2s, rand_r2s, alternative="greater")

        # 2,000-resample stationary bootstrap of mean difference (LLM_mean - RAND_mean)
        rng = np.random.default_rng(self.seed)
        boot_diffs = []
        for _ in range(self.n_bootstrap):
            b_llm = rng.choice(llm_r2s, size=len(llm_r2s), replace=True)
            b_rand = rng.choice(rand_r2s, size=len(rand_r2s), replace=True)
            boot_diffs.append(np.mean(b_llm) - np.mean(b_rand))

        diff_mean = float(np.mean(boot_diffs))
        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))

        # Gate G-LLM2 Passed IF Mann-Whitney p < 0.05 AND 95% bootstrap CI > 0
        gate_g_llm2_passed = (mw_p < 0.05) and (ci_lower > 0.0)

        if gate_g_llm1_passed and gate_g_llm2_passed:
            final_verdict = "LLM REASONING VALIDATED"
        else:
            final_verdict = "LLM REASONING NULL"

        summary_llm = GroupEvaluationSummary(
            group_name="LLM",
            total_hypotheses=n_llm,
            survived_count=llm_survived,
            survival_rate_pct=round(llm_survival_pct, 2),
            mean_r2_vs_pers=round(float(np.mean(llm_r2s)), 4),
            std_r2_vs_pers=round(float(np.std(llm_r2s, ddof=1)), 4),
            median_r2_vs_pers=round(float(np.median(llm_r2s)), 4),
            mean_mcc=round(float(np.mean([r.binary_mcc for r in llm_results])), 4),
        )

        summary_rand = GroupEvaluationSummary(
            group_name="RANDOM",
            total_hypotheses=n_rand,
            survived_count=rand_survived,
            survival_rate_pct=round(rand_survival_pct, 2),
            mean_r2_vs_pers=round(float(np.mean(rand_r2s)), 4),
            std_r2_vs_pers=round(float(np.std(rand_r2s, ddof=1)), 4),
            median_r2_vs_pers=round(float(np.median(rand_r2s)), 4),
            mean_mcc=round(float(np.mean([r.binary_mcc for r in random_results])), 4),
        )

        gates_result = StudyStatisticalGates(
            fisher_odds_ratio=round(float(odds_ratio), 4),
            fisher_p_value=round(float(fisher_p), 4),
            gate_g_llm1_passed=gate_g_llm1_passed,
            mann_whitney_u_stat=round(float(u_stat), 2),
            mann_whitney_p_value=round(float(mw_p), 4),
            bootstrap_diff_mean=round(diff_mean, 4),
            bootstrap_diff_ci_lower=round(ci_lower, 4),
            bootstrap_diff_ci_upper=round(ci_upper, 4),
            gate_g_llm2_passed=gate_g_llm2_passed,
            final_verdict=final_verdict,
        )

        return summary_llm, summary_rand, gates_result
