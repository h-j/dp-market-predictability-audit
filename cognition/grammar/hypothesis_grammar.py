"""
Structural Hypothesis Grammar & AST Compiler.

Defines context-free grammar G for trading hypotheses over point-in-time features.
Compiles AST representations into executable Python predicate functions for walk-forward evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


FEATURES = [
    "rv_1d",
    "rv_5d",
    "vix_close",
    "vix_change_5d",
    "vix_percentile_252d",
    "volume_ratio_5d",
    "norm_gap",
    "delivery_pct",
    "rs_nifty_3m",
]

OPERATORS = ["GREATER_THAN", "LESS_THAN", "BETWEEN"]

QUANTILES = [0.20, 0.35, 0.50, 0.65, 0.80]

TARGET_MODES = ["volatility_5d", "vol_regime_expansion", "direction_3d"]


@dataclass
class ClauseAST:
    feature: str
    operator: str
    threshold_q1: float
    threshold_q2: Optional[float] = None


@dataclass
class HypothesisAST:
    hypothesis_id: str
    description: str
    source: str  # "LLM" or "RANDOM"
    target_mode: str
    clauses: List[ClauseAST]
    logical_op: str = "AND"  # "AND" or "OR"
    prediction_signal: float = 1.0  # 1.0 (bullish/expansion) or -1.0 (bearish/compression)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "source": self.source,
            "target_mode": self.target_mode,
            "logical_op": self.logical_op,
            "prediction_signal": self.prediction_signal,
            "clauses": [
                {
                    "feature": c.feature,
                    "operator": c.operator,
                    "threshold_q1": c.threshold_q1,
                    "threshold_q2": c.threshold_q2,
                }
                for c in self.clauses
            ],
        }


class GrammarCompiler:
    """
    Compiles HypothesisAST into an executable Python predicate evaluator.
    """

    @staticmethod
    def compile_clause(clause: ClauseAST, feature_quantiles: Dict[str, Dict[float, float]]) -> Callable[[pd.Series], bool]:
        feat = clause.feature
        op = clause.operator

        q_map = feature_quantiles.get(feat, {})
        val_q1 = q_map.get(clause.threshold_q1, 0.0)

        if op == "GREATER_THAN":
            return lambda row: bool(row.get(feat, 0.0) > val_q1)
        elif op == "LESS_THAN":
            return lambda row: bool(row.get(feat, 0.0) < val_q1)
        elif op == "BETWEEN":
            q2 = clause.threshold_q2 if clause.threshold_q2 is not None else min(1.0, clause.threshold_q1 + 0.3)
            val_q2 = q_map.get(q2, val_q1 + 1.0)
            low = min(val_q1, val_q2)
            high = max(val_q1, val_q2)
            return lambda row: bool(low <= row.get(feat, 0.0) <= high)
        else:
            return lambda row: True

    def compile_hypothesis(
        self, ast: HypothesisAST, feature_quantiles: Dict[str, Dict[float, float]]
    ) -> Callable[[pd.Series], bool]:
        clause_evaluators = [self.compile_clause(c, feature_quantiles) for c in ast.clauses]

        if ast.logical_op == "OR":
            return lambda row: any(fn(row) for fn in clause_evaluators)
        else:  # "AND"
            return lambda row: all(fn(row) for fn in clause_evaluators)


def compute_feature_quantiles(df_train: pd.DataFrame) -> Dict[str, Dict[float, float]]:
    """
    Compute quantile threshold maps over expanding training history.
    """
    quantiles_map: Dict[str, Dict[float, float]] = {}
    for feat in FEATURES:
        if feat in df_train.columns:
            s = df_train[feat].dropna()
            if len(s) > 10:
                q_dict = {q: float(s.quantile(q)) for q in QUANTILES}
                quantiles_map[feat] = q_dict
            else:
                quantiles_map[feat] = {q: 0.0 for q in QUANTILES}
        else:
            quantiles_map[feat] = {q: 0.0 for q in QUANTILES}
    return quantiles_map
