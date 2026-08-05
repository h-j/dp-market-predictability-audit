"""
Walk-Forward Validation Harness Module.

Provides reusable expanding-window time-series cross-validation for market direction models
and volatility forecasting models (HAR-RV, GBMVolModel). Evaluates models against out-of-sample baselines
without lookahead bias.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("walkforward_validation")
logger.setLevel(logging.INFO)


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        mask = (y_true == cls)
        if np.sum(mask) == 0:
            continue
        recall = np.mean(y_pred[mask] == cls)
        recalls.append(recall)
    return float(np.mean(recalls)) if recalls else 0.0


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    K = len(classes)
    if K <= 1:
        return 0.0
    cls_map = {cls: i for i, cls in enumerate(classes)}
    conf_mat = np.zeros((K, K), dtype=float)
    for t, p in zip(y_true, y_pred):
        conf_mat[cls_map[t], cls_map[p]] += 1.0

    c = np.trace(conf_mat)
    s = np.sum(conf_mat)
    p_k = np.sum(conf_mat, axis=0)
    t_k = np.sum(conf_mat, axis=1)

    num = c * s - np.sum(p_k * t_k)
    den = np.sqrt((s**2 - np.sum(p_k**2)) * (s**2 - np.sum(t_k**2)))
    if den == 0:
        return 0.0
    return float(num / den)


def compute_qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-4
    y = np.clip(y_true, eps, None)
    y_hat = np.clip(y_pred, eps, None)
    ratio = y / y_hat
    loss = ratio - np.log(ratio) - 1.0
    return float(np.mean(loss))


def compute_r2_vs_persistence(y_true: np.ndarray, y_pred: np.ndarray, y_pers: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_pers = np.sum((y_true - y_pers) ** 2)
    if ss_pers == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_pers))


def compute_spearman_rank(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s_true = pd.Series(y_true).rank()
    s_pred = pd.Series(y_pred).rank()
    corr = s_true.corr(s_pred)
    return float(corr) if not pd.isna(corr) else 0.0


# Directional Models (Pure Numpy)

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


class BoostedDecisionStumpsModel:
    """Gradient Boosted Decision Stump Ensemble Classifier (Deprecation Notice: Use for direction benchmarks only)."""

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


# Volatility Models (Workstream 3)

class HARRVModel:
    """
    Heterogeneous Autoregressive Model for Realized Volatility (HAR-RV).
    Linear regression on RV_1d, RV_5d, RV_22d lags.
    """

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0.01, None)


class GBMVolModel:
    """
    Gradient Boosting Regressor for Volatility Forecasting.
    sklearn GradientBoostingRegressor(n_estimators=300, max_depth=3, lr=0.03, subsample=0.8, min_samples_leaf=20, random_state=42).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 3,
        lr: float = 0.03,
        subsample: float = 0.8,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0.01, None)


@dataclass
class FoldResult:
    fold_index: int
    train_range: Tuple[str, str]
    test_range: Tuple[str, str]
    train_size: int
    test_size: int
    # Direction / Metric slots
    logistic_accuracy: float = 0.0
    logistic_bal_acc: float = 0.0
    logistic_mcc: float = 0.0
    logistic_score: float = 0.0
    bds_accuracy: float = 0.0
    bds_bal_acc: float = 0.0
    bds_mcc: float = 0.0
    bds_score: float = 0.0
    majority_accuracy: float = 0.0
    majority_bal_acc: float = 0.0
    majority_mcc: float = 0.0
    majority_score: float = 0.0
    persistence_accuracy: float = 0.0
    persistence_bal_acc: float = 0.0
    persistence_mcc: float = 0.0
    persistence_score: float = 0.0
    range_bound_accuracy: float = 0.0
    range_bound_bal_acc: float = 0.0
    range_bound_mcc: float = 0.0
    range_bound_score: float = 0.0
    logistic_beats_all: bool = False
    bds_beats_all: bool = False
    # Volatility metric slots
    har_r2_vs_pers: float = 0.0
    har_qlike: float = 0.0
    har_spearman: float = 0.0
    har_mcc: float = 0.0
    gbm_r2_vs_pers: float = 0.0
    gbm_qlike: float = 0.0
    gbm_spearman: float = 0.0
    gbm_mcc: float = 0.0
    pers_qlike: float = 0.0
    pers_spearman: float = 0.0


@dataclass
class WalkForwardStudyResult:
    asset_name: str
    target_horizon: str  # "3d", "1d", "volatility_5d"
    target_mode: str  # "direction", "volatility_5d"
    total_samples: int
    num_folds: int
    fold_results: List[FoldResult] = field(default_factory=list)
    # Direction aggregates
    avg_logistic_accuracy: float = 0.0
    avg_logistic_bal_acc: float = 0.0
    avg_logistic_mcc: float = 0.0
    avg_logistic_score: float = 0.0
    avg_bds_accuracy: float = 0.0
    avg_bds_bal_acc: float = 0.0
    avg_bds_mcc: float = 0.0
    avg_bds_score: float = 0.0
    avg_majority_accuracy: float = 0.0
    avg_majority_bal_acc: float = 0.0
    avg_majority_mcc: float = 0.0
    avg_majority_score: float = 0.0
    avg_persistence_accuracy: float = 0.0
    avg_persistence_bal_acc: float = 0.0
    avg_persistence_mcc: float = 0.0
    avg_persistence_score: float = 0.0
    avg_range_bound_accuracy: float = 0.0
    avg_range_bound_bal_acc: float = 0.0
    avg_range_bound_mcc: float = 0.0
    avg_range_bound_score: float = 0.0
    logistic_wins_count: int = 0
    bds_wins_count: int = 0
    consistent_edge_found: bool = False
    # Volatility aggregates
    avg_har_r2_vs_pers: float = 0.0
    avg_har_qlike: float = 0.0
    avg_har_spearman: float = 0.0
    avg_har_mcc: float = 0.0
    avg_gbm_r2_vs_pers: float = 0.0
    avg_gbm_qlike: float = 0.0
    avg_gbm_spearman: float = 0.0
    avg_gbm_mcc: float = 0.0
    avg_pers_qlike: float = 0.0
    avg_pers_spearman: float = 0.0


class WalkForwardValidator:
    """
    Expanding-Window Time-Series Cross-Validator supporting Direction and Volatility Targets.
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
        "delivery_pct",
        "delivery_pct_5d",
        "fii_net",
        "dii_net",
        "sector_rs_ratio",
        "sector_zscore",
        "sector_percentile",
    ]

    HAR_FEATURE_COLS = ["rv_1d", "rv_5d", "rv_22d"]

    def __init__(
        self,
        df: pd.DataFrame,
        asset_name: str = "UNKNOWN",
        target_mode: str = "direction",  # "direction" or "volatility_5d"
        target_horizon: str = "3d",
        initial_train_size: int = 250,
        test_fold_size: int = 100,
    ):
        self.asset_name = asset_name
        self.target_mode = target_mode.lower()
        self.target_horizon = target_horizon.lower()
        self.initial_train_size = initial_train_size
        self.test_fold_size = test_fold_size
        self.df = df.copy().sort_values("date").reset_index(drop=True)
        self.prepare_features_and_targets()

    def prepare_features_and_targets(self):
        df = self.df

        daily_ret = df["daily_return_pct"].fillna(0.0)
        vol_10d = df["rolling_volatility_10d"].replace(0, np.nan).fillna(1.0)

        # HAR-RV Lags
        df["rv_1d"] = daily_ret.abs()
        df["rv_5d"] = df["rv_1d"].rolling(window=5, min_periods=1).mean()
        df["rv_22d"] = df["rv_1d"].rolling(window=22, min_periods=1).mean()

        df["norm_return_3d"] = (df["return_3d"].fillna(0.0) / vol_10d).round(4)
        df["norm_return_5d"] = (df["return_5d"].fillna(0.0) / vol_10d).round(4)
        df["norm_daily_return"] = (daily_ret / vol_10d).round(4)

        df["volume_ratio_5d"] = df["volume_ratio_5d"].fillna(1.0)
        df["volume_ratio_20d"] = df["volume_ratio_20d"].fillna(1.0)

        df["norm_gap"] = (df["gap_pct"].fillna(0.0) / vol_10d).round(4)
        df["norm_range"] = (df["range_pct"].fillna(0.0) / vol_10d).round(4)

        df["net_advances_pct"] = df.get("net_advances_pct", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["pct_above_50dma"] = df.get("pct_above_50dma", pd.Series(0.5, index=df.index)).fillna(0.5)
        df["highs_minus_lows_pct"] = df.get("highs_minus_lows_pct", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["composite_breadth_score"] = df.get("composite_breadth_score", pd.Series(0.5, index=df.index)).fillna(0.5)

        df["delivery_pct"] = df.get("delivery_pct", pd.Series(45.0, index=df.index)).fillna(45.0)
        df["delivery_pct_5d"] = df.get("delivery_pct_5d", pd.Series(45.0, index=df.index)).fillna(45.0)
        df["fii_net"] = df.get("fii_net", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["dii_net"] = df.get("dii_net", pd.Series(0.0, index=df.index)).fillna(0.0)

        df["sector_rs_ratio"] = df.get("sector_rs_ratio", pd.Series(1.0, index=df.index)).fillna(1.0)
        df["sector_zscore"] = df.get("sector_zscore", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["sector_percentile"] = df.get("sector_percentile", pd.Series(0.5, index=df.index)).fillna(0.5)

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

        # Forward Targets
        if self.target_mode == "volatility_5d":
            # 5-day forward realized volatility
            fwd_rv5 = []
            for i in range(len(df)):
                if i + 5 < len(df):
                    ret_window = daily_ret.iloc[i + 1 : i + 6]
                    vol_val = np.sqrt(np.mean(ret_window**2))
                    fwd_rv5.append(vol_val)
                else:
                    fwd_rv5.append(0.0)
            df["target_volatility_5d"] = fwd_rv5
            df["target_vol_regime"] = (df["target_volatility_5d"] > df["rv_5d"]).astype(int)
        else:
            if self.target_horizon == "1d":
                fwd_return = ((df["close"].shift(-1) - df["close"]) / df["close"] * 100.0).fillna(0.0)
                threshold = 0.2
            else:
                fwd_return = ((df["close"].shift(-3) - df["close"]) / df["close"] * 100.0).fillna(0.0)
                threshold = 0.3

            fwd_norm_return = fwd_return / vol_10d
            targets = []
            for norm_ret in fwd_norm_return:
                if norm_ret > threshold:
                    targets.append(1)
                elif norm_ret < -threshold:
                    targets.append(-1)
                else:
                    targets.append(0)
            df["target_direction"] = targets

        self.df = df

    def generate_expanding_folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        shift_len = 5 if self.target_mode == "volatility_5d" else (1 if self.target_horizon == "1d" else 3)
        n_samples = len(self.df) - shift_len
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
        if self.target_mode == "volatility_5d":
            return self._run_volatility_study()
        else:
            return self._run_direction_study()

    def _run_volatility_study(self) -> WalkForwardStudyResult:
        folds = self.generate_expanding_folds()
        fold_results: List[FoldResult] = []

        X_har = self.df[self.HAR_FEATURE_COLS].values
        X_all = self.df[self.FEATURE_COLS + self.HAR_FEATURE_COLS].values
        y_vol = self.df["target_volatility_5d"].values
        y_regime = self.df["target_vol_regime"].values
        pers_vol = self.df["rv_5d"].values
        dates = self.df["date"].values

        all_y_vol = []
        all_pred_har = []
        all_pred_gbm = []
        all_pred_pers = []

        for k, (train_idx, test_idx) in enumerate(folds):
            # Stride-5 non-overlapping evaluation for 5-day vol targets
            test_stride_subidx = np.arange(0, len(test_idx), 5)
            test_stride_idx = test_idx[test_stride_subidx]

            X_train_har, y_train_vol = X_har[train_idx], y_vol[train_idx]
            X_train_all = X_all[train_idx]

            X_test_har_stride = X_har[test_stride_idx]
            X_test_all_stride = X_all[test_stride_idx]
            y_test_vol_stride = y_vol[test_stride_idx]
            y_test_regime_stride = y_regime[test_stride_idx]
            pers_test_stride = pers_vol[test_stride_idx]

            train_range = (str(dates[train_idx[0]]), str(dates[train_idx[-1]]))
            test_range = (str(dates[test_stride_idx[0]]), str(dates[test_stride_idx[-1]]))

            # Model 1: HAR-RV Linear Regression
            har = HARRVModel()
            har.fit(X_train_har, y_train_vol)
            pred_har = har.predict(X_test_har_stride)

            # Model 2: GBM Volatility Regressor
            gbm = GBMVolModel()
            gbm.fit(X_train_all, y_train_vol)
            pred_gbm = gbm.predict(X_test_all_stride)

            # Persistence Baseline
            pred_pers = pers_test_stride

            all_y_vol.extend(y_test_vol_stride)
            all_pred_har.extend(pred_har)
            all_pred_gbm.extend(pred_gbm)
            all_pred_pers.extend(pred_pers)

            # Per-fold metrics
            har_r2 = compute_r2_vs_persistence(y_test_vol_stride, pred_har, pred_pers)
            har_qlike = compute_qlike(y_test_vol_stride, pred_har)
            har_spearman = compute_spearman_rank(y_test_vol_stride, pred_har)
            har_mcc = compute_mcc(y_test_regime_stride, (pred_har > pers_test_stride).astype(int))

            gbm_r2 = compute_r2_vs_persistence(y_test_vol_stride, pred_gbm, pred_pers)
            gbm_qlike = compute_qlike(y_test_vol_stride, pred_gbm)
            gbm_spearman = compute_spearman_rank(y_test_vol_stride, pred_gbm)
            gbm_mcc = compute_mcc(y_test_regime_stride, (pred_gbm > pers_test_stride).astype(int))

            pers_qlike = compute_qlike(y_test_vol_stride, pred_pers)
            pers_spearman = compute_spearman_rank(y_test_vol_stride, pred_pers)

            fold_results.append(
                FoldResult(
                    fold_index=k + 1,
                    train_range=train_range,
                    test_range=test_range,
                    train_size=len(train_idx),
                    test_size=len(test_stride_idx),
                    har_r2_vs_pers=round(har_r2, 4),
                    har_qlike=round(har_qlike, 4),
                    har_spearman=round(har_spearman, 4),
                    har_mcc=round(har_mcc, 4),
                    gbm_r2_vs_pers=round(gbm_r2, 4),
                    gbm_qlike=round(gbm_qlike, 4),
                    gbm_spearman=round(gbm_spearman, 4),
                    gbm_mcc=round(gbm_mcc, 4),
                    pers_qlike=round(pers_qlike, 4),
                    pers_spearman=round(pers_spearman, 4),
                )
            )

        n_folds = len(fold_results)
        # Pooled Spearman rank correlation across all fold test points
        pooled_har_spearman = compute_spearman_rank(np.array(all_y_vol), np.array(all_pred_har))
        pooled_gbm_spearman = compute_spearman_rank(np.array(all_y_vol), np.array(all_pred_gbm))
        pooled_pers_spearman = compute_spearman_rank(np.array(all_y_vol), np.array(all_pred_pers))

        res = WalkForwardStudyResult(
            asset_name=self.asset_name,
            target_horizon="5d",
            target_mode="volatility_5d",
            total_samples=len(self.df),
            num_folds=n_folds,
            fold_results=fold_results,
            avg_har_r2_vs_pers=round(float(np.mean([r.har_r2_vs_pers for r in fold_results])), 4),
            avg_har_qlike=round(float(np.mean([r.har_qlike for r in fold_results])), 4),
            avg_har_spearman=round(float(pooled_har_spearman), 4),
            avg_har_mcc=round(float(np.mean([r.har_mcc for r in fold_results])), 4),
            avg_gbm_r2_vs_pers=round(float(np.mean([r.gbm_r2_vs_pers for r in fold_results])), 4),
            avg_gbm_qlike=round(float(np.mean([r.gbm_qlike for r in fold_results])), 4),
            avg_gbm_spearman=round(float(pooled_gbm_spearman), 4),
            avg_gbm_mcc=round(float(np.mean([r.gbm_mcc for r in fold_results])), 4),
            avg_pers_qlike=round(float(np.mean([r.pers_qlike for r in fold_results])), 4),
            avg_pers_spearman=round(float(pooled_pers_spearman), 4),
        )
        res.consistent_edge_found = (res.avg_har_r2_vs_pers > 0.20) or (res.avg_gbm_r2_vs_pers > 0.20)
        return res

    def _run_direction_study(self) -> WalkForwardStudyResult:
        folds = self.generate_expanding_folds()
        fold_results: List[FoldResult] = []

        X = self.df[self.FEATURE_COLS].values
        y = self.df["target_direction"].values
        dates = self.df["date"].values

        stride = 3 if self.target_horizon == "3d" else 1

        for k, (train_idx, test_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            train_range = (str(dates[train_idx[0]]), str(dates[train_idx[-1]]))
            test_range = (str(dates[test_idx[0]]), str(dates[test_idx[-1]]))

            lr = LogisticRegressionModel(lr=0.05, n_epochs=150, reg=0.1)
            lr.fit(X_train, y_train)
            pred_lr = lr.predict(X_test)

            bds = BoostedDecisionStumpsModel(n_estimators=25, lr=0.1)
            bds.fit(X_train, y_train)
            pred_bds = bds.predict(X_test)

            vals, counts = np.unique(y_train, return_counts=True)
            maj_class = vals[np.argmax(counts)]
            pred_maj = np.full_like(y_test, maj_class)

            test_stride_idx = np.arange(0, len(test_idx), stride)
            y_test_stride = y_test[test_stride_idx]
            pred_pers_stride = y[test_idx[test_stride_idx] - stride]

            pred_rb = np.zeros_like(y_test)

            acc_lr = float((pred_lr == y_test).mean())
            bal_lr = compute_balanced_accuracy(y_test, pred_lr)
            mcc_lr = compute_mcc(y_test, pred_lr)
            score_lr = float(np.mean([self._calculate_direction_score(p, a) for p, a in zip(pred_lr, y_test)]))

            acc_bds = float((pred_bds == y_test).mean())
            bal_bds = compute_balanced_accuracy(y_test, pred_bds)
            mcc_bds = compute_mcc(y_test, pred_bds)
            score_bds = float(np.mean([self._calculate_direction_score(p, a) for p, a in zip(pred_bds, y_test)]))

            acc_maj = float((pred_maj == y_test).mean())
            bal_maj = compute_balanced_accuracy(y_test, pred_maj)
            mcc_maj = compute_mcc(y_test, pred_maj)
            score_maj = float(np.mean([self._calculate_direction_score(p, a) for p, a in zip(pred_maj, y_test)]))

            acc_pers = float((pred_pers_stride == y_test_stride).mean())
            bal_pers = compute_balanced_accuracy(y_test_stride, pred_pers_stride)
            mcc_pers = compute_mcc(y_test_stride, pred_pers_stride)
            score_pers = float(np.mean([self._calculate_direction_score(p, a) for p, a in zip(pred_pers_stride, y_test_stride)]))

            acc_rb = float((pred_rb == y_test).mean())
            bal_rb = compute_balanced_accuracy(y_test, pred_rb)
            mcc_rb = compute_mcc(y_test, pred_rb)
            score_rb = float(np.mean([self._calculate_direction_score(p, a) for p, a in zip(pred_rb, y_test)]))

            lr_win = (mcc_lr > mcc_maj) and (mcc_lr > mcc_pers) and (mcc_lr > mcc_rb) and (bal_lr > bal_maj) and (bal_lr > bal_pers)
            bds_win = (mcc_bds > mcc_maj) and (mcc_bds > mcc_pers) and (mcc_bds > mcc_rb) and (bal_bds > bal_maj) and (bal_bds > bal_pers)

            fold_results.append(
                FoldResult(
                    fold_index=k + 1,
                    train_range=train_range,
                    test_range=test_range,
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    logistic_accuracy=round(acc_lr, 4),
                    logistic_bal_acc=round(bal_lr, 4),
                    logistic_mcc=round(mcc_lr, 4),
                    logistic_score=round(score_lr, 4),
                    bds_accuracy=round(acc_bds, 4),
                    bds_bal_acc=round(bal_bds, 4),
                    bds_mcc=round(mcc_bds, 4),
                    bds_score=round(score_bds, 4),
                    majority_accuracy=round(acc_maj, 4),
                    majority_bal_acc=round(bal_maj, 4),
                    majority_mcc=round(mcc_maj, 4),
                    majority_score=round(score_maj, 4),
                    persistence_accuracy=round(acc_pers, 4),
                    persistence_bal_acc=round(bal_pers, 4),
                    persistence_mcc=round(mcc_pers, 4),
                    persistence_score=round(score_pers, 4),
                    range_bound_accuracy=round(acc_rb, 4),
                    range_bound_bal_acc=round(bal_rb, 4),
                    range_bound_mcc=round(mcc_rb, 4),
                    range_bound_score=round(score_rb, 4),
                    logistic_beats_all=lr_win,
                    bds_beats_all=bds_win,
                )
            )

        n_folds = len(fold_results)
        res = WalkForwardStudyResult(
            asset_name=self.asset_name,
            target_horizon=self.target_horizon,
            target_mode="direction",
            total_samples=len(self.df),
            num_folds=n_folds,
            fold_results=fold_results,
            avg_logistic_accuracy=round(float(np.mean([r.logistic_accuracy for r in fold_results])), 4),
            avg_logistic_bal_acc=round(float(np.mean([r.logistic_bal_acc for r in fold_results])), 4),
            avg_logistic_mcc=round(float(np.mean([r.logistic_mcc for r in fold_results])), 4),
            avg_logistic_score=round(float(np.mean([r.logistic_score for r in fold_results])), 4),
            avg_bds_accuracy=round(float(np.mean([r.bds_accuracy for r in fold_results])), 4),
            avg_bds_bal_acc=round(float(np.mean([r.bds_bal_acc for r in fold_results])), 4),
            avg_bds_mcc=round(float(np.mean([r.bds_mcc for r in fold_results])), 4),
            avg_bds_score=round(float(np.mean([r.bds_score for r in fold_results])), 4),
            avg_majority_accuracy=round(float(np.mean([r.majority_accuracy for r in fold_results])), 4),
            avg_majority_bal_acc=round(float(np.mean([r.majority_bal_acc for r in fold_results])), 4),
            avg_majority_mcc=round(float(np.mean([r.majority_mcc for r in fold_results])), 4),
            avg_majority_score=round(float(np.mean([r.majority_score for r in fold_results])), 4),
            avg_persistence_accuracy=round(float(np.mean([r.persistence_accuracy for r in fold_results])), 4),
            avg_persistence_bal_acc=round(float(np.mean([r.persistence_bal_acc for r in fold_results])), 4),
            avg_persistence_mcc=round(float(np.mean([r.persistence_mcc for r in fold_results])), 4),
            avg_persistence_score=round(float(np.mean([r.persistence_score for r in fold_results])), 4),
            avg_range_bound_accuracy=round(float(np.mean([r.range_bound_accuracy for r in fold_results])), 4),
            avg_range_bound_bal_acc=round(float(np.mean([r.range_bound_bal_acc for r in fold_results])), 4),
            avg_range_bound_mcc=round(float(np.mean([r.range_bound_mcc for r in fold_results])), 4),
            avg_range_bound_score=round(float(np.mean([r.range_bound_score for r in fold_results])), 4),
            logistic_wins_count=sum(1 for r in fold_results if r.logistic_beats_all),
            bds_wins_count=sum(1 for r in fold_results if r.bds_beats_all),
        )

        res.consistent_edge_found = (res.logistic_wins_count >= 0.75 * n_folds) or (
            res.bds_wins_count >= 0.75 * n_folds
        )
        return res
