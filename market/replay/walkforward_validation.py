"""
Walk-Forward Validation Harness Module.

Provides reusable expanding-window time-series cross-validation for market direction models.
Evaluates ML candidate models (Logistic Regression, Gradient Boosted Decision Stumps) against three baselines
(majority-class, persistence, and always-range_bound) across time-series folds without lookahead bias.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("walkforward_validation")
logger.setLevel(logging.INFO)


class LogisticRegressionModel:
    """Multi-class Logistic Regression Classifier using Gradient Descent."""

    def __init__(self, lr: float = 0.05, n_epochs: int = 150, reg: float = 0.1):
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.weights = None
        self.biases = None
        self.classes = None
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.mean) / self.std

        self.classes = np.unique(y)
        n_samples, n_features = X_norm.shape
        n_classes = len(self.classes)

        Y = np.zeros((n_samples, n_classes))
        for idx, cls in enumerate(self.classes):
            Y[y == cls, idx] = 1.0

        self.weights = np.zeros((n_features, n_classes))
        self.biases = np.zeros(n_classes)

        for _ in range(self.n_epochs):
            logits = X_norm @ self.weights + self.biases
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            dw = (1 / n_samples) * (X_norm.T @ (probs - Y)) + self.reg * self.weights
            db = (1 / n_samples) * np.sum(probs - Y, axis=0)

            self.weights -= self.lr * dw
            self.biases -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.classes is None:
            return np.zeros(len(X), dtype=int)
        X_norm = (X - self.mean) / self.std
        logits = X_norm @ self.weights + self.biases
        preds_idx = np.argmax(logits, axis=1)
        return self.classes[preds_idx]


class DecisionStump:
    """Single Decision Stump for Gradient Boosting."""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_val = 0.0
        self.right_val = 0.0

    def fit(self, X: np.ndarray, residuals: np.ndarray):
        n_samples, n_features = X.shape
        best_loss = float("inf")

        for f_idx in range(n_features):
            vals = X[:, f_idx]
            thresholds = np.percentile(vals, [25, 50, 75])
            for thresh in thresholds:
                left_mask = vals <= thresh
                right_mask = ~left_mask

                l_val = float(np.mean(residuals[left_mask])) if np.any(left_mask) else 0.0
                r_val = float(np.mean(residuals[right_mask])) if np.any(right_mask) else 0.0

                pred = np.where(left_mask, l_val, r_val)
                loss = float(np.sum((residuals - pred) ** 2))

                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx = f_idx
                    self.threshold = thresh
                    self.left_val = l_val
                    self.right_val = r_val

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(X[:, self.feature_idx] <= self.threshold, self.left_val, self.right_val)


class GradientBoostedTreeModel:
    """Gradient Boosted Decision Stump Ensemble Classifier."""

    def __init__(self, n_estimators: int = 25, lr: float = 0.1):
        self.n_estimators = n_estimators
        self.lr = lr
        self.classes = None
        self.estimators = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        n_samples = len(y)

        for cls in self.classes:
            binary_y = (y == cls).astype(float)
            stumps = []
            pred = np.full(n_samples, np.mean(binary_y))

            for _ in range(self.n_estimators):
                residual = binary_y - pred
                stump = DecisionStump()
                stump.fit(X, residual)
                pred += self.lr * stump.predict(X)
                stumps.append(stump)

            self.estimators[cls] = (np.mean(binary_y), stumps)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators or self.classes is None:
            return np.zeros(len(X), dtype=int)
        n_samples = len(X)
        scores = np.zeros((n_samples, len(self.classes)))

        for i, cls in enumerate(self.classes):
            init_val, stumps = self.estimators[cls]
            cls_pred = np.full(n_samples, init_val)
            for stump in stumps:
                cls_pred += self.lr * stump.predict(X)
            scores[:, i] = cls_pred

        preds_idx = np.argmax(scores, axis=1)
        return self.classes[preds_idx]


@dataclass
class FoldResult:
    fold_index: int
    train_range: Tuple[str, str]
    test_range: Tuple[str, str]
    train_size: int
    test_size: int
    logistic_accuracy: float
    logistic_score: float
    hgb_accuracy: float
    hgb_score: float
    majority_accuracy: float
    majority_score: float
    persistence_accuracy: float
    persistence_score: float
    range_bound_accuracy: float
    range_bound_score: float
    logistic_beats_all: bool
    hgb_beats_all: bool


@dataclass
class WalkForwardStudyResult:
    asset_name: str
    total_samples: int
    num_folds: int
    fold_results: List[FoldResult] = field(default_factory=list)
    avg_logistic_accuracy: float = 0.0
    avg_logistic_score: float = 0.0
    avg_hgb_accuracy: float = 0.0
    avg_hgb_score: float = 0.0
    avg_majority_accuracy: float = 0.0
    avg_majority_score: float = 0.0
    avg_persistence_accuracy: float = 0.0
    avg_persistence_score: float = 0.0
    avg_range_bound_accuracy: float = 0.0
    avg_range_bound_score: float = 0.0
    logistic_wins_count: int = 0
    hgb_wins_count: int = 0
    consistent_edge_found: bool = False


class WalkForwardValidator:
    """
    Expanding-Window Time-Series Cross-Validator.
    """

    FEATURE_COLS = [
        "norm_return_3d",
        "norm_return_5d",
        "norm_daily_return",
        "volume_ratio_5d",
        "volume_ratio_20d",
        "norm_gap",
        "norm_range",
        "net_advances_pct",
        "pct_above_50dma",
        "highs_minus_lows_pct",
        "composite_breadth_score",
        "volatility_30d_rank",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        asset_name: str = "UNKNOWN",
        initial_train_size: int = 250,
        test_fold_size: int = 100,
    ):
        self.asset_name = asset_name
        self.initial_train_size = initial_train_size
        self.test_fold_size = test_fold_size
        self.df = df.copy().sort_values("date").reset_index(drop=True)
        self.prepare_features_and_targets()

    def prepare_features_and_targets(self):
        """
        Construct non-lookahead features and forward-looking directional targets.
        """
        df = self.df

        vol_10d = df["rolling_volatility_10d"].replace(0, np.nan).fillna(1.0)

        df["norm_return_3d"] = (df["return_3d"].fillna(0.0) / vol_10d).round(4)
        df["norm_return_5d"] = (df["return_5d"].fillna(0.0) / vol_10d).round(4)
        df["norm_daily_return"] = (df["daily_return_pct"].fillna(0.0) / vol_10d).round(4)

        df["volume_ratio_5d"] = df["volume_ratio_5d"].fillna(1.0)
        df["volume_ratio_20d"] = df["volume_ratio_20d"].fillna(1.0)

        df["norm_gap"] = (df["gap_pct"].fillna(0.0) / vol_10d).round(4)
        df["norm_range"] = (df["range_pct"].fillna(0.0) / vol_10d).round(4)

        df["net_advances_pct"] = df.get("net_advances_pct", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["pct_above_50dma"] = df.get("pct_above_50dma", pd.Series(0.5, index=df.index)).fillna(0.5)
        df["highs_minus_lows_pct"] = df.get("highs_minus_lows_pct", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["composite_breadth_score"] = df.get("composite_breadth_score", pd.Series(0.5, index=df.index)).fillna(0.5)

        vol_30d = df["rolling_volatility_30d"].fillna(1.0)
        vol_ranks = []
        for i in range(len(df)):
            if i < 10:
                vol_ranks.append(0.5)
            else:
                historical = vol_30d.iloc[: i + 1]
                rank = (historical <= vol_30d.iloc[i]).mean()
                vol_ranks.append(round(rank, 4))
        df["volatility_30d_rank"] = vol_ranks

        # Forward 3-day return target
        fwd_return_3d = ((df["close"].shift(-3) - df["close"]) / df["close"] * 100.0).fillna(0.0)
        fwd_norm_return = fwd_return_3d / vol_10d

        targets = []
        for norm_ret in fwd_norm_return:
            if norm_ret > 0.3:
                targets.append(1)  # higher
            elif norm_ret < -0.3:
                targets.append(-1)  # lower
            else:
                targets.append(0)  # range_bound
        df["target_direction"] = targets

        self.df = df

    def generate_expanding_folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-series expanding training window fold indices.
        """
        n_samples = len(self.df) - 3
        folds = []

        start = self.initial_train_size
        while start + 30 <= n_samples:
            end = min(start + self.test_fold_size, n_samples)
            train_idx = np.arange(0, start)
            test_idx = np.arange(start, end)
            folds.append((train_idx, test_idx))
            start = end

        return folds

    @staticmethod
    def _calculate_direction_score(pred: int, actual: int) -> float:
        if pred == actual:
            return 1.0
        if pred == 0 or actual == 0:
            return 0.5
        return 0.0

    def run_study(self) -> WalkForwardStudyResult:
        """
        Execute expanding-window walk-forward validation study.
        """
        folds = self.generate_expanding_folds()
        fold_results: List[FoldResult] = []

        X = self.df[self.FEATURE_COLS].values
        y = self.df["target_direction"].values
        dates = self.df["date"].values

        for k, (train_idx, test_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            train_range = (str(dates[train_idx[0]]), str(dates[train_idx[-1]]))
            test_range = (str(dates[test_idx[0]]), str(dates[test_idx[-1]]))

            # Candidate 1: Logistic Regression
            lr = LogisticRegressionModel(lr=0.05, n_epochs=150, reg=0.1)
            lr.fit(X_train, y_train)
            pred_lr = lr.predict(X_test)

            # Candidate 2: Gradient Boosted Trees
            hgb = GradientBoostedTreeModel(n_estimators=25, lr=0.1)
            hgb.fit(X_train, y_train)
            pred_hgb = hgb.predict(X_test)

            # Baseline 1: Majority Class (in training set)
            vals, counts = np.unique(y_train, return_counts=True)
            maj_class = vals[np.argmax(counts)]
            pred_maj = np.full_like(y_test, maj_class)

            # Baseline 2: Persistence (prior 1-day direction in target array)
            pred_pers = y[test_idx - 1]

            # Baseline 3: Always Range-Bound (0)
            pred_rb = np.zeros_like(y_test)

            # Calculate Accuracies and Direction Scores
            acc_lr = float((pred_lr == y_test).mean())
            score_lr = float(np.mean([self._calculate_direction_score(p, a) for p, a in zip(pred_lr, y_test)]))

            acc_hgb = float((pred_hgb == y_test).mean())
            score_hgb = float(np.mean([self._calculate_direction_score(p, a) for p, a in zip(pred_hgb, y_test)]))

            acc_maj = float((pred_maj == y_test).mean())
            score_maj = float(np.mean([self._calculate_direction_score(p, a) for p, a in zip(pred_maj, y_test)]))

            acc_pers = float((pred_pers == y_test).mean())
            score_pers = float(np.mean([self._calculate_direction_score(p, a) for p, a in zip(pred_pers, y_test)]))

            acc_rb = float((pred_rb == y_test).mean())
            score_rb = float(np.mean([self._calculate_direction_score(p, a) for p, a in zip(pred_rb, y_test)]))

            # Wins check (score must exceed ALL 3 baselines)
            lr_win = (score_lr > score_maj) and (score_lr > score_pers) and (score_lr > score_rb)
            hgb_win = (score_hgb > score_maj) and (score_hgb > score_pers) and (score_hgb > score_rb)

            fold_results.append(
                FoldResult(
                    fold_index=k + 1,
                    train_range=train_range,
                    test_range=test_range,
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    logistic_accuracy=round(acc_lr, 4),
                    logistic_score=round(score_lr, 4),
                    hgb_accuracy=round(acc_hgb, 4),
                    hgb_score=round(score_hgb, 4),
                    majority_accuracy=round(acc_maj, 4),
                    majority_score=round(score_maj, 4),
                    persistence_accuracy=round(acc_pers, 4),
                    persistence_score=round(score_pers, 4),
                    range_bound_accuracy=round(acc_rb, 4),
                    range_bound_score=round(score_rb, 4),
                    logistic_beats_all=lr_win,
                    hgb_beats_all=hgb_win,
                )
            )

        n_folds = len(fold_results)
        res = WalkForwardStudyResult(
            asset_name=self.asset_name,
            total_samples=len(self.df),
            num_folds=n_folds,
            fold_results=fold_results,
            avg_logistic_accuracy=round(float(np.mean([r.logistic_accuracy for r in fold_results])), 4),
            avg_logistic_score=round(float(np.mean([r.logistic_score for r in fold_results])), 4),
            avg_hgb_accuracy=round(float(np.mean([r.hgb_accuracy for r in fold_results])), 4),
            avg_hgb_score=round(float(np.mean([r.hgb_score for r in fold_results])), 4),
            avg_majority_accuracy=round(float(np.mean([r.majority_accuracy for r in fold_results])), 4),
            avg_majority_score=round(float(np.mean([r.majority_score for r in fold_results])), 4),
            avg_persistence_accuracy=round(float(np.mean([r.persistence_accuracy for r in fold_results])), 4),
            avg_persistence_score=round(float(np.mean([r.persistence_score for r in fold_results])), 4),
            avg_range_bound_accuracy=round(float(np.mean([r.range_bound_accuracy for r in fold_results])), 4),
            avg_range_bound_score=round(float(np.mean([r.range_bound_score for r in fold_results])), 4),
            logistic_wins_count=sum(1 for r in fold_results if r.logistic_beats_all),
            hgb_wins_count=sum(1 for r in fold_results if r.hgb_beats_all),
        )

        res.consistent_edge_found = (res.logistic_wins_count >= 0.75 * n_folds) or (
            res.hgb_wins_count >= 0.75 * n_folds
        )
        return res
