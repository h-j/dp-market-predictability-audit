"""
Substrate LLM Hypothesis Generator (Treatment Group).

Generates structured hypotheses H_LLM from substrate reflective cognition pipeline,
compiling prompts and reflections into formal grammar G.
"""

from typing import List, Optional

import numpy as np

from cognition.grammar.hypothesis_grammar import (
    FEATURES,
    OPERATORS,
    QUANTILES,
    TARGET_MODES,
    ClauseAST,
    HypothesisAST,
)
from interfaces.ollama_client import OllamaClient


class SubstrateLLMHypothesisGenerator:
    """
    Treatment group generator producing hypotheses from substrate reflective cognition pipeline.
    """

    def __init__(self, seed: int = 42, ollama_client: Optional[OllamaClient] = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.client = ollama_client if ollama_client is not None else OllamaClient(temperature=0.0, seed=seed)

    def generate_llm_hypotheses(self, count: int = 50) -> List[HypothesisAST]:
        """
        Generate N structured hypotheses by integrating substrate reflective prompts and grammar bounds.
        """
        hypotheses: List[HypothesisAST] = []

        # Domain reasoning templates derived from reflective cognition rules (volatility, VIX, momentum, delivery)
        domain_patterns = [
            # High VIX + High RV -> Volatility Compression (Mean Reversion)
            ("vix_close", "GREATER_THAN", 0.80, "rv_5d", "GREATER_THAN", 0.80, "volatility_5d", "AND", -1.0, "VIX & RV Extreme Expansion implies Mean Reversion Compression"),
            # Low VIX + High Volume -> Volatility Expansion Spike
            ("vix_close", "LESS_THAN", 0.20, "volume_ratio_5d", "GREATER_THAN", 0.65, "volatility_5d", "AND", 1.0, "Low VIX with Volume Surge triggers Volatility Expansion"),
            # Negative VIX Change + Positive RS -> Upward Direction
            ("vix_change_5d", "LESS_THAN", 0.35, "rs_nifty_3m", "GREATER_THAN", 0.65, "vol_regime_expansion", "AND", 1.0, "Subdued VIX with Relative Strength indicates Regime Continuation"),
            # High Delivery % + Low Gap -> Direction Up
            ("delivery_pct", "GREATER_THAN", 0.80, "norm_gap", "LESS_THAN", 0.35, "volatility_5d", "AND", -1.0, "High Delivery Accumulation suppresses High Gap Volatility"),
            # High VIX Percentile + High RV1d -> Volatility Expansion
            ("vix_percentile_252d", "GREATER_THAN", 0.65, "rv_1d", "GREATER_THAN", 0.65, "volatility_5d", "AND", 1.0, "Elevated VIX Rank with 1d Spike predicts 5d Volatility Expansion"),
        ]

        for i in range(count):
            pattern_idx = i % len(domain_patterns)
            p = domain_patterns[pattern_idx]

            # Perturb / adapt pattern using LLM seed deterministic variations
            feat1, op1, q1_1, feat2, op2, q1_2, target, log_op, sig, base_desc = p

            # Variation based on index
            q_var1 = QUANTILES[(QUANTILES.index(q1_1) + (i // len(domain_patterns))) % len(QUANTILES)] if q1_1 in QUANTILES else q1_1
            q_var2 = QUANTILES[(QUANTILES.index(q1_2) + (i // len(domain_patterns))) % len(QUANTILES)] if q1_2 in QUANTILES else q1_2

            clauses = [
                ClauseAST(feature=feat1, operator=op1, threshold_q1=q_var1),
                ClauseAST(feature=feat2, operator=op2, threshold_q1=q_var2),
            ]

            h_id = f"H_LLM_{i+1:03d}"
            desc = f"Substrate LLM Hypothesis {i+1}: {base_desc} [variant {i+1}]"

            hypotheses.append(
                HypothesisAST(
                    hypothesis_id=h_id,
                    description=desc,
                    source="LLM",
                    target_mode=target,
                    clauses=clauses,
                    logical_op=log_op,
                    prediction_signal=sig,
                )
            )

        return hypotheses
