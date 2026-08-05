"""
Phase 3 Evaluator — In-Sample Binomial Survival Filter & Out-of-Sample Mann-Whitney Gate G-P3.

Evaluates frozen hypothesis arms (150 LLM vs 150 Random) using a 60/40 In-Sample/Out-of-Sample split.
Filters hypotheses via binomial exact test (p < 0.10) in-sample, then evaluates out-of-sample edge.
Executes single Gate G-P3 (pooled one-sided Mann-Whitney U test on surviving OOS edges).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu

from cognition.grammar.hypothesis_grammar import ClauseAST, GrammarCompiler, HypothesisAST, compute_feature_quantiles


@dataclass
class HypothesisPhase3Result:
    hypothesis_id: str
    source: str
    target_mode: str
    description: str
    is_triggers: int
    is_hits: int
    is_hit_rate: float
    is_base_rate: float
    is_binomial_p: float
    is_survived: bool  # True if binomial_p < 0.10
    oos_triggers: int
    oos_hits: int
    oos_hit_rate: float
    oos_base_rate: float
    oos_edge: float  # oos_hit_rate - oos_base_rate


@dataclass
class Phase3ArmSummary:
    arm_name: str  # "LLM" or "RANDOM"
    total_hypotheses: int
    is_survived_count: int
    is_survival_rate_pct: float
    mean_oos_edge_survivors: float
    median_oos_edge_survivors: float
    mean_oos_edge_all: float


@dataclass
class Phase3GateP3Result:
    mann_whitney_u_stat: float
    mann_whitney_p_value: float
    llm_survivors_mean_edge: float
    random_survivors_mean_edge: float
    gate_g_p3_passed: bool
    final_verdict: str


class Phase3HypothesisEvaluator:
    """
    Executes 60/40 IS/OOS split evaluation per Phase 3 Pre-Registration.
    """

    def __init__(self, is_ratio: float = 0.60, is_p_threshold: float = 0.10):
        self.compiler = GrammarCompiler()
        self.is_ratio = is_ratio
        self.is_p_threshold = is_p_threshold

    def evaluate_hypothesis_split(
        self, ast: HypothesisAST, df_asset: pd.DataFrame
    ) -> HypothesisPhase3Result:
        """
        Evaluate a single hypothesis AST using 60/40 IS/OOS split.
        """
        n_total = len(df_asset)
        split_idx = int(n_total * self.is_ratio)

        df_is = df_asset.iloc[:split_idx].copy()
        df_oos = df_asset.iloc[split_idx:].copy()

        # In-sample feature quantiles
        is_quantiles = compute_feature_quantiles(df_is)
        evaluator_fn = self.compiler.compile_hypothesis(ast, is_quantiles)

        # 1. In-Sample Evaluation
        is_triggers = 0
        is_hits = 0
        is_targets = []

        for idx in range(0, len(df_is), 5):
            row = df_is.iloc[idx]
            actual_vol = float(row.get("target_volatility_5d", 1.0))
            base_rv = float(row.get("rv_5d", 1.0))

            # Target hit logic: signal=1.0 -> actual_vol > base_rv; signal=-1.0 -> actual_vol < base_rv
            actual_hit = (actual_vol > base_rv) if ast.prediction_signal > 0 else (actual_vol < base_rv)
            is_targets.append(1 if actual_hit else 0)

            if evaluator_fn(row):
                is_triggers += 1
                if actual_hit:
                    is_hits += 1

        is_base_rate = float(np.mean(is_targets)) if len(is_targets) > 0 else 0.50
        is_hit_rate = (is_hits / is_triggers) if is_triggers > 0 else 0.0

        # One-sided binomial test IS hit-rate > IS base-rate
        if is_triggers > 0:
            binom_res = binomtest(is_hits, n=is_triggers, p=is_base_rate, alternative="greater")
            is_p_val = float(binom_res.pvalue)
        else:
            is_p_val = 1.0

        is_survived = (is_p_val < self.is_p_threshold) and (is_triggers >= 3)

        # 2. Out-of-Sample Evaluation
        oos_triggers = 0
        oos_hits = 0
        oos_targets = []

        for idx in range(0, len(df_oos), 5):
            row = df_oos.iloc[idx]
            actual_vol = float(row.get("target_volatility_5d", 1.0))
            base_rv = float(row.get("rv_5d", 1.0))

            actual_hit = (actual_vol > base_rv) if ast.prediction_signal > 0 else (actual_vol < base_rv)
            oos_targets.append(1 if actual_hit else 0)

            if evaluator_fn(row):
                oos_triggers += 1
                if actual_hit:
                    oos_hits += 1

        oos_base_rate = float(np.mean(oos_targets)) if len(oos_targets) > 0 else 0.50
        oos_hit_rate = (oos_hits / oos_triggers) if oos_triggers > 0 else oos_base_rate
        oos_edge = oos_hit_rate - oos_base_rate

        return HypothesisPhase3Result(
            hypothesis_id=ast.hypothesis_id,
            source=ast.source,
            target_mode=ast.target_mode,
            description=ast.description,
            is_triggers=is_triggers,
            is_hits=is_hits,
            is_hit_rate=round(float(is_hit_rate), 4),
            is_base_rate=round(float(is_base_rate), 4),
            is_binomial_p=round(float(is_p_val), 4),
            is_survived=is_survived,
            oos_triggers=oos_triggers,
            oos_hits=oos_hits,
            oos_hit_rate=round(float(oos_hit_rate), 4),
            oos_base_rate=round(float(oos_base_rate), 4),
            oos_edge=round(float(oos_edge), 4),
        )

    def evaluate_gate_g_p3(
        self, llm_results: List[HypothesisPhase3Result], rand_results: List[HypothesisPhase3Result]
    ) -> Tuple[Phase3ArmSummary, Phase3ArmSummary, Phase3GateP3Result]:
        """
        Compute summary metrics and evaluate single Gate G-P3 (Mann-Whitney U test on survivor OOS edges).
        """
        # LLM Arm Filtering
        llm_survivors = [r for r in llm_results if r.is_survived]
        rand_survivors = [r for r in rand_results if r.is_survived]

        n_llm = len(llm_results)
        n_rand = len(rand_results)

        llm_surv_pct = (len(llm_survivors) / n_llm * 100.0) if n_llm > 0 else 0.0
        rand_surv_pct = (len(rand_survivors) / n_rand * 100.0) if n_rand > 0 else 0.0

        llm_surv_edges = [r.oos_edge for r in llm_survivors]
        rand_surv_edges = [r.oos_edge for r in rand_survivors]

        llm_all_edges = [r.oos_edge for r in llm_results]
        rand_all_edges = [r.oos_edge for r in rand_results]

        llm_mean_surv_edge = float(np.mean(llm_surv_edges)) if len(llm_surv_edges) > 0 else 0.0
        llm_median_surv_edge = float(np.median(llm_surv_edges)) if len(llm_surv_edges) > 0 else 0.0

        rand_mean_surv_edge = float(np.mean(rand_surv_edges)) if len(rand_surv_edges) > 0 else 0.0
        rand_median_surv_edge = float(np.median(rand_surv_edges)) if len(rand_surv_edges) > 0 else 0.0

        # Pooled One-Sided Mann-Whitney U test: LLM survivor edges > Random survivor edges
        if len(llm_surv_edges) > 0 and len(rand_surv_edges) > 0:
            mw_res = mannwhitneyu(llm_surv_edges, rand_surv_edges, alternative="greater")
            mw_stat = float(mw_res.statistic)
            mw_p = float(mw_res.pvalue)
        else:
            mw_stat = 0.0
            mw_p = 1.0

        gate_g_p3_passed = (mw_p < 0.05) and (llm_mean_surv_edge > rand_mean_surv_edge)
        final_verdict = "LLM REASONING VALIDATED" if gate_g_p3_passed else "LLM REASONING NULL"

        sum_llm = Phase3ArmSummary(
            arm_name="LLM",
            total_hypotheses=n_llm,
            is_survived_count=len(llm_survivors),
            is_survival_rate_pct=round(llm_surv_pct, 2),
            mean_oos_edge_survivors=round(llm_mean_surv_edge, 4),
            median_oos_edge_survivors=round(llm_median_surv_edge, 4),
            mean_oos_edge_all=round(float(np.mean(llm_all_edges)), 4),
        )

        sum_rand = Phase3ArmSummary(
            arm_name="RANDOM",
            total_hypotheses=n_rand,
            is_survived_count=len(rand_survivors),
            is_survival_rate_pct=round(rand_surv_pct, 2),
            mean_oos_edge_survivors=round(rand_mean_surv_edge, 4),
            median_oos_edge_survivors=round(rand_median_surv_edge, 4),
            mean_oos_edge_all=round(float(np.mean(rand_all_edges)), 4),
        )

        gate_result = Phase3GateP3Result(
            mann_whitney_u_stat=round(mw_stat, 2),
            mann_whitney_p_value=round(mw_p, 4),
            llm_survivors_mean_edge=round(llm_mean_surv_edge, 4),
            random_survivors_mean_edge=round(rand_mean_surv_edge, 4),
            gate_g_p3_passed=gate_g_p3_passed,
            final_verdict=final_verdict,
        )

        return sum_llm, sum_rand, gate_result
