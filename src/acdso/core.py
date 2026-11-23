"""
ACDSO v3.0 - Core Engine Module
================================

Production-grade, scikit-learn-compatible AutoML estimators with
robustness-aware optimization.

This module contains:
- _ACDSOEngineV3: Internal optimization engine with v3 innovations
- ACDSOClassifier: scikit-learn compatible classifier
- ACDSORegressor: scikit-learn compatible regressor

Key features (v3):
- Stress testing + robustness objective (6-objective optimization)
- Successive halving (multi-fidelity search) for efficient evaluation
- Learned risk calibration (predicts robustness from dataset features)
- Continuous, severity-aware risk penalties
- Multi-objective evaluation: metric, time, memory, latency, risk, robustness
- Constraint-based pruning + Pareto knee-point selection
- JSON-serializable audit trail artifacts
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import time
import tracemalloc
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, ElasticNet, Lasso
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from acdso.pareto import CandidateResult, knee_point_selection, pareto_front
from acdso.risk import RiskReport, analyze_risk, compute_risk_penalty
from acdso.utils import build_preprocessor, measure_latency_ms_per_row, sanitize_features, split_columns

# Optional boosting libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# ============================================================================
# Stress Testing Infrastructure (v3 Innovation)
# ============================================================================

@dataclass
class StressTestResult:
    """Result from a single stress test."""
    name: str
    stressed_metric: float
    delta: float  # baseline - stressed (positive = degradation)


def _safe_numpy(X: Any) -> np.ndarray:
    """Convert X to numpy array without copying when possible."""
    if isinstance(X, np.ndarray):
        return X
    try:
        return np.asarray(X)
    except Exception:
        return np.array(X)


def _extract_meta_features_for_task(
    task: str,
    model_step: Any,
    X_mat: Any,
    global_classes: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Extract leakage-free 2D meta-features from a fitted base estimator."""
    try:
        if task == "regression":
            pred = np.asarray(model_step.predict(X_mat), dtype=float).reshape(-1, 1)
            return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        if global_classes is None or len(global_classes) < 2:
            return None
        n_classes = len(global_classes)
        if hasattr(model_step, "predict_proba"):
            proba = np.asarray(model_step.predict_proba(X_mat), dtype=float)
            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)
            est_classes = getattr(model_step, "classes_", None)
            if est_classes is not None:
                cls_map = {c: idx for idx, c in enumerate(global_classes)}
                aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
                for j, c in enumerate(est_classes):
                    if c in cls_map and j < proba.shape[1]:
                        aligned[:, cls_map[c]] = proba[:, j]
                out = aligned[:, 1:2] if n_classes == 2 else aligned
            else:
                out = proba[:, 1:2] if (n_classes == 2 and proba.shape[1] >= 2) else proba
            return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        if hasattr(model_step, "decision_function"):
            dec = np.asarray(model_step.decision_function(X_mat), dtype=float)
            if dec.ndim == 1:
                dec = dec.reshape(-1, 1)
            if n_classes > 2 and dec.shape[1] != n_classes:
                return None
            return np.nan_to_num(dec, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None
    return None


class _Fold0StackAdapter:
    """Top-level picklable Fold-0 stacking adapter for latency, stress tests, and serialization."""

    def __init__(
        self,
        base_models: List[Any],
        meta_model: Any,
        task: str,
        global_classes: Optional[np.ndarray],
    ) -> None:
        self.base_models = base_models
        self.meta_model = meta_model
        self.task = task
        self.global_classes = global_classes
        self.classes_ = getattr(meta_model, "classes_", global_classes)

    def _meta_X(self, X_proc: Any) -> np.ndarray:
        cols = [
            _extract_meta_features_for_task(self.task, m, X_proc, self.global_classes)
            for m in self.base_models
        ]
        mat = np.hstack(cols)
        return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, X_proc: Any) -> np.ndarray:
        return self.meta_model.predict(self._meta_X(X_proc))

    def predict_proba(self, X_proc: Any) -> np.ndarray:
        return self.meta_model.predict_proba(self._meta_X(X_proc))


def _apply_missingness_noise(X: Any, missing_frac: float, rng: np.random.RandomState) -> Any:
    """Inject missing values into X while preserving DataFrame columns or array shape."""
    if isinstance(X, pd.DataFrame):
        df_out = X.copy()
        for col in df_out.columns:
            if pd.api.types.is_bool_dtype(df_out[col]) or pd.api.types.is_integer_dtype(df_out[col]):
                df_out[col] = df_out[col].astype("float64")
        n, d = df_out.shape
        if n == 0 or d == 0:
            return df_out
        k = int(min(n * d, max(1, missing_frac * n * d)))
        flat_idx = rng.choice(n * d, size=k, replace=False)
        rows, cols = np.unravel_index(flat_idx, (n, d))
        for c_idx in np.unique(cols):
            r_sel = rows[cols == c_idx]
            df_out.iloc[r_sel, int(c_idx)] = np.nan
        return df_out

    Xn = _safe_numpy(X).copy()
    if Xn.ndim == 1:
        Xn = Xn.reshape(-1, 1)
    if np.issubdtype(Xn.dtype, np.integer) or np.issubdtype(Xn.dtype, np.bool_):
        Xn = Xn.astype(np.float64)
    n, d = Xn.shape
    if n == 0 or d == 0:
        return Xn
    k = int(min(n * d, max(1, missing_frac * n * d)))
    idx = rng.choice(n * d, size=k, replace=False)
    Xn.flat[idx] = np.nan
    return Xn


def _apply_feature_jitter(X: Any, frac: float, rng: np.random.RandomState) -> Any:
    """Small gaussian jitter on a subset of numeric cells while preserving DataFrame structure."""
    if isinstance(X, pd.DataFrame):
        df_out = X.copy()
        num_cols = [
            c for c in df_out.columns
            if pd.api.types.is_numeric_dtype(df_out[c]) and not pd.api.types.is_bool_dtype(df_out[c])
        ]
        if not num_cols or len(df_out) == 0:
            return df_out
        sub = df_out[num_cols].astype(np.float64, copy=False).to_numpy(dtype=np.float64, copy=True)
        n, d_num = sub.shape
        k = int(min(n * d_num, max(1, frac * n * d_num)))
        flat_idx = rng.choice(n * d_num, size=k, replace=False)
        rows, cols = np.unravel_index(flat_idx, (n, d_num))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_stds = np.nanstd(sub, axis=0)
        col_stds = np.where(np.isfinite(col_stds) & (col_stds > 0), col_stds, 1.0)
        noise = rng.normal(loc=0.0, scale=1.0, size=k) * (0.01 * col_stds[cols])
        sub[rows, cols] = sub[rows, cols] + noise
        for j, col in enumerate(num_cols):
            df_out[col] = sub[:, j]
        return df_out

    Xn = _safe_numpy(X).copy()
    if Xn.ndim == 1:
        Xn = Xn.reshape(-1, 1)
    if not np.issubdtype(Xn.dtype, np.number) or np.issubdtype(Xn.dtype, np.bool_):
        return Xn
    if np.issubdtype(Xn.dtype, np.integer):
        Xn = Xn.astype(np.float64)
    n, d = Xn.shape
    if n == 0 or d == 0:
        return Xn
    k = int(min(n * d, max(1, frac * n * d)))
    flat_idx = rng.choice(n * d, size=k, replace=False)
    rows, cols = np.unravel_index(flat_idx, (n, d))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_stds = np.nanstd(Xn, axis=0)
    col_stds = np.where(np.isfinite(col_stds) & (col_stds > 0), col_stds, 1.0)
    noise = rng.normal(loc=0.0, scale=1.0, size=k) * (0.01 * col_stds[cols])
    Xn[rows, cols] = Xn[rows, cols] + noise
    return Xn


def _apply_label_noise(y: np.ndarray, frac: float, rng: np.random.RandomState) -> np.ndarray:
    """Flip a small percentage of labels to different classes."""
    yn = y.copy()
    n = len(yn)
    if n == 0:
        return yn
    k = int(min(n, max(1, frac * n)))
    idx = rng.choice(n, size=k, replace=False)

    classes = np.unique(yn)
    if len(classes) <= 1:
        return yn

    for i in idx:
        current = yn[i]
        choices = classes[classes != current]
        if len(choices) > 0:
            yn[i] = rng.choice(choices)
    return yn


def run_stress_tests(
    fitted_pipe: Pipeline,
    X_val: Any,
    y_val: Any,
    task: str,
    scorer_name: str,
    random_state: int,
) -> Tuple[float, List[StressTestResult], float]:
    """
    Run stress tests to measure robustness under perturbations (v3).

    Uses validation data (held-out from training) to get realistic
    estimates of robustness degradation on unseen data, preserving DataFrame
    columns and index when DataFrames are supplied.
    """
    scorer = get_scorer(scorer_name)
    baseline = float(scorer(fitted_pipe, X_val, y_val))

    rng = np.random.RandomState(random_state)
    ynp = np.asarray(y_val)

    tests: List[StressTestResult] = []

    # 1) Missingness injection
    try:
        Xm = _apply_missingness_noise(X_val, missing_frac=0.02, rng=rng)
        m_score = float(scorer(fitted_pipe, Xm, y_val))
        tests.append(StressTestResult("missingness_2pct", m_score, baseline - m_score))
    except Exception:
        pass

    # 2) Feature jitter (numeric)
    try:
        Xj = _apply_feature_jitter(X_val, frac=0.02, rng=rng)
        j_score = float(scorer(fitted_pipe, Xj, y_val))
        tests.append(StressTestResult("feature_jitter_2pct", j_score, baseline - j_score))
    except Exception:
        pass

    # 3) Label noise (classification only)
    if task == "classification":
        try:
            y_noisy = _apply_label_noise(ynp, frac=0.02, rng=rng)
            ln_score = float(scorer(fitted_pipe, X_val, y_noisy))
            tests.append(StressTestResult("label_noise_2pct", ln_score, baseline - ln_score))
        except Exception:
            pass

    worst_drop = float(max(0.0, max(t.delta for t in tests))) if tests else 0.0
    return baseline, tests, worst_drop


def _estimate_array_nbytes(arr: Any) -> int:
    """Estimate memory footprint of an array, sparse matrix, or DataFrame in bytes."""
    if arr is None:
        return 0
    if isinstance(arr, np.ndarray):
        return int(arr.nbytes)
    if hasattr(arr, "data") and hasattr(arr, "indices") and hasattr(arr, "indptr"):
        return int(
            getattr(arr.data, "nbytes", 0)
            + getattr(arr.indices, "nbytes", 0)
            + getattr(arr.indptr, "nbytes", 0)
        )
    if isinstance(arr, pd.DataFrame):
        try:
            return int(arr.memory_usage(deep=False).sum())
        except Exception:
            return sys.getsizeof(arr)
    if isinstance(arr, pd.Series):
        try:
            return int(arr.memory_usage(deep=False))
        except Exception:
            return sys.getsizeof(arr)
    return int(sys.getsizeof(arr))


def _estimate_model_nbytes(model: Any) -> int:
    """Estimate deep memory footprint of a fitted estimator in bytes."""
    if model is None:
        return 0
    base = sys.getsizeof(model)
    try:
        if hasattr(model, "estimators_"):
            ests = model.estimators_
            if isinstance(ests, np.ndarray):
                ests_iter = ests.ravel().tolist()
            else:
                ests_iter = list(ests)
            tree_bytes = 0
            for t in ests_iter:
                tree_obj = getattr(t, "tree_", None)
                if tree_obj is not None:
                    tree_bytes += int(getattr(tree_obj, "node_count", 0)) * 64
            if tree_bytes > 0:
                return max(base, tree_bytes)
    except Exception:
        pass
    try:
        return max(base, len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)))
    except Exception:
        return base


@dataclass
class _FoldCacheEntry:
    """Cached leak-free per-fold preprocessed data and fitted transforms."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    X_tr_raw: Any
    X_val_raw: Any
    y_tr: np.ndarray
    y_val: np.ndarray
    X_tr_proc: Any
    X_val_proc: Any
    pre_f: Any
    sel_f: Any
    data_mem_bytes: int


def _auto_adjust_scorer(
    scorer: str,
    y: Any,
    task: str,
    llm_provider: Any = None,
    column_name: str = "target",
) -> str:
    """
    Automatically adjust the scorer based on target labels.
    
    Uses LLM-powered analysis if a provider is available, otherwise
    falls back to rule-based logic.
    
    Handles:
    - Multiclass classification: convert f1 → f1_weighted
    - Binary with non-standard labels (not 0/1): convert f1 → f1_weighted
    - Regression: no changes needed
    
    Parameters
    ----------
    scorer : str
        Original scorer name.
    y : array-like
        Target values.
    task : str
        Task type ("classification" or "regression").
    llm_provider : Any, optional
        LLM provider for intelligent label analysis.
    column_name : str, default="target"
        Name of the target column (for LLM context).
        
    Returns
    -------
    str
        Adjusted scorer name.
    """
    if task != "classification":
        return scorer
    
    # Try LLM-powered analysis if provider is available
    if llm_provider is not None:
        try:
            from acdso.advisors import LLMLabelAnalyzer
            analyzer = LLMLabelAnalyzer(provider=llm_provider)
            result = analyzer.analyze(y, column_name=column_name)
            
            # If LLM recommends a different scorer, use it
            if result.recommended_scorer and result.recommended_scorer != scorer:
                return result.recommended_scorer
        except Exception:
            # Fall back to rule-based on any error
            pass
    
    # Rule-based fallback
    y_arr = np.asarray(y)
    unique_labels = np.unique(y_arr)
    n_classes = len(unique_labels)
    
    # Scorers that require pos_label=1 (binary-only)
    binary_only_scorers = ["f1", "precision", "recall", "roc_auc"]
    
    # Check if scorer needs adjustment
    if scorer in binary_only_scorers:
        # Multiclass: always use weighted version
        if n_classes > 2:
            if scorer == "roc_auc":
                return "roc_auc_ovr_weighted"
            return f"{scorer}_weighted"
        
        # Binary but labels are not {0, 1}: use weighted version
        if n_classes == 2:
            labels_set = set(unique_labels)
            if labels_set != {0, 1}:
                if scorer == "roc_auc":
                    return "roc_auc_ovr_weighted"
                return f"{scorer}_weighted"
    
    return scorer


class _ACDSOEngineV3:
    """
    Internal ACDSO optimization engine (v3).

    This class handles the core optimization loop including:
    - Risk analysis with learned calibration
    - Candidate generation with adaptive feature selection
    - Stress testing for robustness measurement
    - Multi-objective evaluation (6 objectives)
    - Successive halving for efficient candidate evaluation
    - Pareto front construction with robustness awareness
    - Knee-point selection

    v3 Innovations:
    - Stress testing: missingness, feature jitter, label noise
    - Robustness drop as 6th optimization objective
    - Multi-fidelity search via successive halving
    - Learned risk calibration from dataset features

    Not intended for direct use; use ACDSOClassifier or ACDSORegressor instead.
    """

    def __init__(
        self,
        task: str,
        scorer: str,
        cv: int,
        random_state: Optional[int],
        time_budget_s: Optional[float],
        max_candidates: int,
        preference: str,
        max_train_time_s: Optional[float],
        max_peak_mem_mb: Optional[float],
        max_latency_ms_per_row: Optional[float],
        enable_feature_selection: bool,
        llm_advisor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]],
        verbose: int,
        selection_mode: str = "knee_point",
        n_jobs: Optional[int] = None,
        enable_stacking: bool = True,
    ):
        self.task = task
        self.scorer = scorer  # Will be adjusted in fit() based on y
        self._original_scorer = scorer  # Keep original for reference
        self.cv = cv
        self.random_state = 42 if random_state is None else int(random_state)
        self.time_budget_s = time_budget_s
        self.max_candidates = max_candidates
        self.preference = preference
        self.selection_mode = selection_mode  # "knee_point" or "best_score"
        self.n_jobs = n_jobs
        self.enable_stacking = bool(enable_stacking)

        self.max_train_time_s = max_train_time_s
        self.max_peak_mem_mb = max_peak_mem_mb
        self.max_latency_ms_per_row = max_latency_ms_per_row

        self.enable_feature_selection = enable_feature_selection
        self.llm_advisor = llm_advisor
        self.verbose = verbose

        # Results (populated after fit)
        self.risk_report_: Optional[RiskReport] = None
        self.pareto_front_: List[CandidateResult] = []
        self.all_results_: List[CandidateResult] = []
        self.best_result_: Optional[CandidateResult] = None
        self.audit_trail_: Dict[str, Any] = {}

    def _log(self, msg: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ACDSO v3] {msg}")

    def _resolve_max_workers(self, n_tasks: int, n_rows: Optional[int] = None) -> int:
        """Resolve bounded worker count to prevent thread/GIL/tracemalloc contention."""
        cpu_cnt = os.cpu_count() or 4
        if self.n_jobs == 1:
            cap = 1
        elif self.n_jobs is None:
            if tracemalloc.is_tracing():
                cap = 1
            elif n_rows is not None and n_rows < 300:
                cap = 1
            else:
                cap = min(cpu_cnt, 4)
        elif self.n_jobs == -1:
            cap = 1 if tracemalloc.is_tracing() else min(cpu_cnt, 8)
        elif self.n_jobs < -1:
            cap = max(1, min(cpu_cnt + 1 + self.n_jobs, 8))
        else:
            cap = min(int(self.n_jobs), cpu_cnt, 12)
        return max(1, min(max(1, n_tasks), cap))

    def _build_preprocessor(self, X: Any) -> Any:
        """Build a preprocessing ColumnTransformer from X (used by fallback paths)."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        num_cols, cat_cols = split_columns(df)
        return build_preprocessor(num_cols, cat_cols)

    def _generate_candidates(
        self,
        X: Any,
        y: Any,
        rr: RiskReport,
    ) -> List[Tuple[str, Pipeline]]:
        """
        Generate candidate pipelines based on risk profile and dataset characteristics.

        Candidate generation is adaptive:
        - High multicollinearity → add regularization or feature selection
        - Severe imbalance → prefer class weights
        - Small datasets → simpler models
        - Large datasets → scalable learners (boosting if available)
        - Very few features → skip feature selection, simpler models
        - More features than samples → increased regularization
        """
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        num_cols, cat_cols = split_columns(df)

        pre = build_preprocessor(num_cols, cat_cols)

        # Dataset size detection for adaptive model configuration
        n_samples = rr.n_rows
        n_features = rr.n_cols

        # Prevent 72-core joblib/OpenMP thrashing on Windows during parallel or small fits
        cpu_cnt = os.cpu_count() or 4
        if self.n_jobs != 1 or n_samples < 5000:
            inner_n_jobs = 1
        else:
            inner_n_jobs = min(cpu_cnt, 4)

        # Edge case flags
        very_few_features = n_features <= 3
        very_few_samples = n_samples < 50
        few_samples = n_samples < 200
        small = n_samples < 1000
        large = n_samples >= 50000
        more_features_than_samples = n_features > n_samples
        
        # Log edge cases for debugging
        if self.verbose:
            edge_cases = []
            if very_few_features:
                edge_cases.append(f"very_few_features ({n_features})")
            if very_few_samples:
                edge_cases.append(f"very_few_samples ({n_samples})")
            if more_features_than_samples:
                edge_cases.append(f"more_features_than_samples ({n_features}>{n_samples})")
            if edge_cases:
                self._log(f"Edge cases detected: {', '.join(edge_cases)}")

        # Add feature selection if high multicollinearity detected
        # BUT skip if very few features (no point selecting from ≤3 features)
        selector = None
        if self.enable_feature_selection and rr.vif_max >= 10.0 and not very_few_features:
            score_func = f_classif if self.task == "classification" else f_regression
            # Determine k: more aggressive selection for higher VIF
            # VIF 10-20: keep 75% of features, VIF 20-50: keep 60%, VIF >50: keep 50%
            if rr.vif_max >= 50.0:
                k_ratio = 0.5
            elif rr.vif_max >= 20.0:
                k_ratio = 0.6
            else:
                k_ratio = 0.75
            
            # Calculate actual k (ensure at least 2 features remain)
            total_features = len(num_cols) if num_cols else n_features
            k = max(2, int(total_features * k_ratio))
            k = min(k, total_features)  # Don't exceed available features
            
            # Only add selector if k < total_features (otherwise it's useless)
            if k < total_features:
                selector = SelectKBest(score_func=score_func, k=k)

        # Adaptive n_estimators based on dataset size
        if very_few_samples:
            n_est_small, n_est_med, n_est_large = 30, 50, 80
        elif few_samples:
            n_est_small, n_est_med, n_est_large = 50, 80, 100
        elif small:
            n_est_small, n_est_med, n_est_large = 80, 100, 150
        elif not large and n_samples < 5000:
            n_est_small, n_est_med, n_est_large = 60, 100, 150
        else:
            n_est_small, n_est_med, n_est_large = 100, 150, 250
        
        # Adaptive max_depth based on features and samples
        if very_few_features:
            max_depth_shallow, max_depth_medium = 2, 3
        elif very_few_samples or more_features_than_samples:
            max_depth_shallow, max_depth_medium = 3, 5
        elif few_samples:
            max_depth_shallow, max_depth_medium = 4, 7
        else:
            max_depth_shallow, max_depth_medium = 5, 10
        
        # Adaptive min_samples_leaf for small datasets
        if very_few_samples:
            min_samples_leaf = max(1, n_samples // 10)
        elif few_samples:
            min_samples_leaf = 3
        else:
            min_samples_leaf = 1
        
        # Linearity detection - prioritize linear models if data is linear
        is_linear = getattr(rr, 'is_linear', False)
        linearity_score = getattr(rr, 'linearity_score', 0.0)
        
        if is_linear and self.verbose:
            self._log(f"Linear data detected (score={linearity_score:.3f}). Prioritizing linear models.")

        models: List[Tuple[str, Any]] = []

        if self.task == "classification":
            # Logistic Regression with adaptive regularization
            # Increase regularization (lower C) when more features than samples
            c_default = 0.1 if more_features_than_samples else 1.0
            c_weak = 0.01 if more_features_than_samples else 0.1
            
            lr_l2 = LogisticRegression(max_iter=2000, penalty='l2', C=c_default, random_state=self.random_state)
            lr_weak = LogisticRegression(max_iter=2000, penalty='l2', C=c_weak, random_state=self.random_state)
            
            # Add more linear model variants for linear datasets
            if is_linear:
                # Additional regularization strengths for linear data
                lr_strong = LogisticRegression(max_iter=2000, penalty='l2', C=10.0, random_state=self.random_state)
                lr_elastic = LogisticRegression(
                    max_iter=2000, penalty='elasticnet', solver='saga',
                    l1_ratio=0.5, C=1.0, random_state=self.random_state
                )
            
            # Random Forest with adaptive configurations
            rf_shallow = RandomForestClassifier(
                n_estimators=n_est_small,
                max_depth=max_depth_shallow,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=inner_n_jobs,
            )
            rf_medium = RandomForestClassifier(
                n_estimators=n_est_med,
                max_depth=max_depth_medium,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=inner_n_jobs,
            )
            rf_deep = RandomForestClassifier(
                n_estimators=n_est_large,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=inner_n_jobs,
            )

            # Handle class imbalance
            if rr.class_imbalance_max_ratio is not None and rr.class_imbalance_max_ratio >= 0.85:
                for model in [lr_l2, lr_weak]:
                    try:
                        model.set_params(class_weight="balanced")
                    except Exception:
                        pass
                if is_linear:
                    for model in [lr_strong, lr_elastic]:
                        try:
                            model.set_params(class_weight="balanced")
                        except Exception:
                            pass
                for model in [rf_shallow, rf_medium, rf_deep]:
                    try:
                        model.set_params(class_weight="balanced")
                    except Exception:
                        pass

            # For linear data, prioritize linear models and keep non-linear check in top 6
            if is_linear:
                models.append(("LogisticRegression_C10", lr_strong))
                models.append(("LogisticRegression", lr_l2))
                models.append(("LogisticRegression_ElasticNet", lr_elastic))
                models.append(("LogisticRegression_C0.1", lr_weak))
                models.append(("RandomForest_d10", rf_medium))
                models.append(("RandomForest_d5", rf_shallow))
            else:
                models.append(("LogisticRegression", lr_l2))
                models.append(("LogisticRegression_C0.1", lr_weak))
                models.append(("RandomForest_d5", rf_shallow))
                models.append(("RandomForest_d10", rf_medium))
                models.append(("RandomForest", rf_deep))

            # Add boosting for medium+ datasets with enough features
            # Skip LightGBM if <5 features, skip XGBoost if <3 features
            can_use_lgbm = not small and _HAS_LGBM and n_features >= 5 and not very_few_samples
            can_use_xgb = not small and _HAS_XGB and n_features >= 3 and not very_few_samples
            
            if can_use_lgbm:
                # LightGBM with settings to prevent crashes on small/sparse data
                lgbm_min_data = max(5, n_samples // 50) if few_samples else 20
                models.append(("LightGBM", LGBMClassifier(
                    random_state=self.random_state,
                    n_jobs=inner_n_jobs,
                    verbose=-1,
                    n_estimators=n_est_med,
                    feature_pre_filter=False,  # Prevent "Forced splits" error
                    min_data_in_leaf=lgbm_min_data,
                )))
            if can_use_xgb:
                models.append(("XGBoost", XGBClassifier(
                    random_state=self.random_state,
                    n_jobs=inner_n_jobs,
                    verbosity=0,
                    n_estimators=n_est_med,
                    max_depth=max_depth_medium,
                )))

        else:  # regression
            # Ridge Regression with adaptive alpha
            # Increase alpha (more regularization) when more features than samples
            if more_features_than_samples:
                alpha_low, alpha_mid, alpha_high = 1.0, 10.0, 100.0
            elif very_few_samples:
                alpha_low, alpha_mid, alpha_high = 0.5, 5.0, 50.0
            else:
                alpha_low, alpha_mid, alpha_high = 0.1, 1.0, 10.0
            
            # ElasticNet/Lasso with adaptive settings
            # Reduce max_iter and increase tol for high-dimensional or small datasets
            if more_features_than_samples or very_few_samples:
                elastic_max_iter, elastic_tol = 200, 1e-2
            else:
                elastic_max_iter, elastic_tol = 500, 1e-3
            
            # For linear datasets, prioritize linear models while keeping RandomForest_d10 in top 6
            if is_linear:
                from sklearn.linear_model import LinearRegression, HuberRegressor

                # LinearRegression (OLS) - best for truly linear data
                models.append(("LinearRegression", LinearRegression()))

                # Ridge variants with core alphas
                models.append(("Ridge_alpha0.1", Ridge(alpha=alpha_low, random_state=self.random_state)))
                models.append(("Ridge_alpha1.0", Ridge(alpha=alpha_mid, random_state=self.random_state)))
                models.append(("Ridge_alpha10", Ridge(alpha=alpha_high, random_state=self.random_state)))

                models.append(("ElasticNet", ElasticNet(
                    random_state=self.random_state,
                    max_iter=elastic_max_iter,
                    tol=elastic_tol,
                )))

                # Retain a capable non-linear baseline within the top 6 candidates
                models.append(("RandomForest_d10", RandomForestRegressor(
                    n_estimators=n_est_med,
                    max_depth=max_depth_medium,
                    min_samples_leaf=min_samples_leaf,
                    random_state=self.random_state,
                    n_jobs=inner_n_jobs,
                )))

                models.append(("Ridge_alpha0.01", Ridge(alpha=0.01, random_state=self.random_state)))
                models.append(("ElasticNet_l1_0.2", ElasticNet(
                    l1_ratio=0.2, random_state=self.random_state,
                    max_iter=elastic_max_iter, tol=elastic_tol,
                )))
                models.append(("ElasticNet_l1_0.8", ElasticNet(
                    l1_ratio=0.8, random_state=self.random_state,
                    max_iter=elastic_max_iter, tol=elastic_tol,
                )))
                models.append(("Lasso", Lasso(
                    random_state=self.random_state,
                    max_iter=elastic_max_iter,
                    tol=elastic_tol,
                )))
                if not very_few_samples:
                    models.append(("HuberRegressor", HuberRegressor(max_iter=500)))
                models.append(("RandomForest_d5", RandomForestRegressor(
                    n_estimators=n_est_small,
                    max_depth=max_depth_shallow,
                    min_samples_leaf=min_samples_leaf,
                    random_state=self.random_state,
                    n_jobs=inner_n_jobs,
                )))
            else:
                # Standard model selection for non-linear data
                models.append(("Ridge_alpha0.1", Ridge(alpha=alpha_low, random_state=self.random_state)))
                models.append(("Ridge_alpha1.0", Ridge(alpha=alpha_mid, random_state=self.random_state)))
                models.append(("Ridge_alpha10", Ridge(alpha=alpha_high, random_state=self.random_state)))
                
                models.append(("ElasticNet", ElasticNet(
                    random_state=self.random_state, 
                    max_iter=elastic_max_iter, 
                    tol=elastic_tol
                )))
                models.append(("Lasso", Lasso(
                    random_state=self.random_state, 
                    max_iter=elastic_max_iter, 
                    tol=elastic_tol
                )))

                # Random Forest Regressor with adaptive configurations
                models.append(("RandomForest_d5", RandomForestRegressor(
                    n_estimators=n_est_small,
                    max_depth=max_depth_shallow,
                    min_samples_leaf=min_samples_leaf,
                    random_state=self.random_state,
                    n_jobs=inner_n_jobs,
                )))
                models.append(("RandomForest_d10", RandomForestRegressor(
                    n_estimators=n_est_med,
                    max_depth=max_depth_medium,
                    min_samples_leaf=min_samples_leaf,
                    random_state=self.random_state,
                    n_jobs=inner_n_jobs,
                )))
                models.append(("RandomForestRegressor", RandomForestRegressor(
                    n_estimators=n_est_large,
                    min_samples_leaf=min_samples_leaf,
                    random_state=self.random_state,
                    n_jobs=inner_n_jobs,
                )))

            # Add boosting for medium+ datasets with enough features (unless linear)
            # Skip LightGBM if <5 features, skip XGBoost if <3 features
            # For linear data, skip tree-based models as they're unlikely to help
            if not is_linear:
                can_use_lgbm = not small and _HAS_LGBM and n_features >= 5 and not very_few_samples
                can_use_xgb = not small and _HAS_XGB and n_features >= 3 and not very_few_samples
                
                if can_use_lgbm:
                    # LightGBM with settings to prevent crashes on small/sparse data
                    lgbm_min_data = max(5, n_samples // 50) if few_samples else 20
                    models.append(("LightGBMRegressor", LGBMRegressor(
                        random_state=self.random_state,
                        n_jobs=inner_n_jobs,
                        verbose=-1,
                        n_estimators=n_est_med,
                        feature_pre_filter=False,  # Prevent "Forced splits" error
                        min_data_in_leaf=lgbm_min_data,
                    )))
                if can_use_xgb:
                    models.append(("XGBoostRegressor", XGBRegressor(
                        random_state=self.random_state,
                        n_jobs=inner_n_jobs,
                        verbosity=0,
                        n_estimators=n_est_med,
                        max_depth=max_depth_medium,
                    )))

        # Build pipelines
        candidates: List[Tuple[str, Pipeline]] = []
        for name, model in models[: max(1, self.max_candidates)]:
            steps = [("pre", pre)]
            if selector is not None:
                steps.append(("select", selector))
            steps.append(("model", model))
            candidates.append((name, Pipeline(steps=steps)))

        return candidates

    def _successive_halving_plan(self, n_candidates: int) -> List[Dict[str, Any]]:
        """
        Two-stage plan for successive halving (v3 multi-fidelity search):
        - Stage 0 (cheap): 25% rows, reduced CV folds, keep top ~60%
        - Stage 1 (full): 100% rows, full CV folds, keep all for final Pareto selection
        
        The "full" stage uses row_frac=1.0 which ensures _subsample_rows returns
        the complete dataset for final evaluation.
        
        Keep at least 3 candidates to ensure meaningful comparison.
        """
        # Keep 60% of candidates, but at least 3 (or all if fewer than 3)
        keep0 = max(min(3, n_candidates), int(math.ceil(n_candidates * 0.6)))
        return [
            {"name": "cheap", "row_frac": 0.25, "cv": max(2, self.cv // 2), "keep": keep0},
            {"name": "full", "row_frac": 1.0, "cv": self.cv, "keep": n_candidates},
        ]

    def _subsample_rows(self, X: Any, y: Any, frac: float) -> Tuple[Any, Any]:
        """
        Subsample rows for multi-fidelity evaluation.

        For frac >= 1.0, small datasets (n < 100), or imbalanced classification
        datasets with < 8 minority samples, returns the original data unchanged
        so minority classes are never starved.
        """
        if frac >= 1.0:
            return X, y

        n = len(y)
        if n < 100:
            return X, y

        y_arr = np.asarray(y)
        if self.task == "classification":
            uniq, counts = np.unique(y_arr, return_counts=True)
            if len(uniq) >= 2 and int(counts.min()) < 8:
                return X, y

        sample_size = max(10, int(n * frac))
        if sample_size >= n:
            return X, y

        idx = None
        if self.task == "classification":
            try:
                uniq, counts = np.unique(y_arr, return_counts=True)
                min_strat = max(sample_size, len(uniq) * 4)
                if min_strat >= n:
                    return X, y
                if len(uniq) >= 2 and int(counts.min()) >= 2 and min_strat >= len(uniq):
                    from sklearn.model_selection import StratifiedShuffleSplit
                    sss = StratifiedShuffleSplit(
                        n_splits=1, train_size=min_strat, random_state=self.random_state
                    )
                    cand_idx, _ = next(sss.split(np.zeros(n), y_arr))
                    _, sub_counts = np.unique(y_arr[cand_idx], return_counts=True)
                    if len(sub_counts) == len(uniq) and int(sub_counts.min()) >= 2:
                        idx = cand_idx
            except Exception:
                idx = None

        if idx is None:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n, size=sample_size, replace=False)

        if hasattr(X, "iloc"):
            Xs = X.iloc[idx]
        else:
            Xs = _safe_numpy(X)[idx]
        ys = y_arr[idx]
        return Xs, ys

    def _evaluate_candidate(
        self,
        X: Any,
        y: Any,
        name: str,
        pipe: Pipeline,
        cv_override: Optional[int] = None,
    ) -> CandidateResult:
        """
        Evaluate a single candidate pipeline (sequential fallback / direct helper).
        """
        start = time.time()
        y_arr = np.asarray(y)

        cv_used = int(cv_override or self.cv)
        if self.task == "classification":
            uniq, counts = np.unique(y_arr, return_counts=True)
            if len(uniq) >= 2 and int(counts.min()) >= 2:
                cv_used = max(2, min(cv_used, int(counts.min())))
                splitter = StratifiedKFold(
                    n_splits=cv_used,
                    shuffle=True,
                    random_state=self.random_state,
                )
            else:
                cv_used = max(2, min(cv_used, len(y_arr) // 2))
                splitter = KFold(
                    n_splits=cv_used,
                    shuffle=True,
                    random_state=self.random_state,
                )
        else:
            cv_used = max(2, min(cv_used, len(y_arr) // 2))
            splitter = KFold(
                n_splits=cv_used,
                shuffle=True,
                random_state=self.random_state,
            )

        scorer = get_scorer(self.scorer)

        rss_before = 0.0
        rss_after = 0.0
        proc = None
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            rss_before = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            proc = None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            scores = cross_val_score(pipe, X, y_arr, cv=splitter, scoring=scorer, n_jobs=None)

        if proc is not None:
            try:
                rss_after = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                rss_after = rss_before

        train_time_s = float(time.time() - start)
        metric = float(np.mean(scores))
        cv_std = float(np.std(scores))

        assert self.risk_report_ is not None
        risk_pen = compute_risk_penalty(
            self.risk_report_,
            cv_score_std=cv_std,
            preference=self.preference,
        )

        # Validation split for stress testing
        n_samples = len(y_arr)
        val_size = max(10, int(n_samples * 0.2))
        if val_size >= n_samples:
            val_size = max(1, n_samples // 2)

        rng = np.random.RandomState(self.random_state)
        all_idx = np.arange(n_samples)
        rng.shuffle(all_idx)

        train_idx = all_idx[val_size:]
        val_idx = all_idx[:val_size]

        if hasattr(X, "iloc"):
            X_train_stress = X.iloc[train_idx]
            X_val_stress = X.iloc[val_idx]
        else:
            X_arr = _safe_numpy(X)
            X_train_stress = X_arr[train_idx]
            X_val_stress = X_arr[val_idx]

        y_train_stress = y_arr[train_idx]
        y_val_stress = y_arr[val_idx]

        fitted = clone(pipe)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            fitted.fit(X_train_stress, y_train_stress)

        det_bytes = (
            _estimate_array_nbytes(X) * 2
            + _estimate_array_nbytes(y_arr)
            + _estimate_model_nbytes(fitted)
        )
        peak_mem_mb = float(max(det_bytes / (1024 * 1024), max(0.0, rss_after - rss_before), 0.01))

        latency = measure_latency_ms_per_row(fitted, X_val_stress)

        baseline_fit_metric, stress_results, worst_drop = run_stress_tests(
            fitted_pipe=fitted,
            X_val=X_val_stress,
            y_val=y_val_stress,
            task=self.task,
            scorer_name=self.scorer,
            random_state=self.random_state,
        )

        objectives = (
            metric,
            math.log1p(train_time_s),
            math.log1p(peak_mem_mb),
            math.log1p(latency),
            risk_pen,
            worst_drop,
        )

        return CandidateResult(
            name=name,
            pipeline=fitted,
            metric=metric,
            train_time_s=train_time_s,
            peak_mem_mb=peak_mem_mb,
            latency_ms_per_row=latency,
            risk_penalty=risk_pen,
            objectives=objectives,
            meta={
                "cv_scores": scores.tolist(),
                "cv_score_std": cv_std,
                "stress_baseline_metric": baseline_fit_metric,
                "stress_tests": [asdict(r) for r in stress_results],
                "stress_worst_drop": worst_drop,
            },
        )

    def _evaluate_stage(
        self,
        Xs: Any,
        ys: Any,
        survivors: List[Tuple[str, Pipeline]],
        stage: Dict[str, Any],
        t0: float,
    ) -> Tuple[List[CandidateResult], List[Dict[str, Any]], bool]:
        """
        Evaluate a stage with leak-free fold preprocessor caching and bounded
        parallelism across (candidate, fold) tasks.
        """
        stage_name = stage["name"]
        is_full = stage_name == "full"
        stage_results: List[CandidateResult] = []
        stage_failures: List[Dict[str, Any]] = []
        budget_exhausted = False

        if not survivors:
            return stage_results, stage_failures, budget_exhausted

        # Check standard Pipeline structure ("pre", optional "select", "model")
        can_cache = all(
            isinstance(p, Pipeline)
            and "pre" in p.named_steps
            and "model" in p.named_steps
            for _, p in survivors
        )
        if not can_cache:
            for name, pipe in survivors:
                if self.time_budget_s is not None and (time.time() - t0) >= self.time_budget_s:
                    self._log("Time budget reached; stopping evaluation.")
                    budget_exhausted = True
                    break
                try:
                    res = self._evaluate_candidate(Xs, ys, name, pipe, cv_override=stage["cv"])
                    if is_full and not self._passes_constraints(res):
                        stage_failures.append({
                            "name": name,
                            "reason": "constraints",
                            "metrics": {
                                "train_time_s": res.train_time_s,
                                "peak_mem_mb": res.peak_mem_mb,
                                "latency_ms_per_row": res.latency_ms_per_row,
                                "risk_penalty": res.risk_penalty,
                                "metric": res.metric,
                                "stress_worst_drop": res.meta.get("stress_worst_drop"),
                            },
                        })
                        continue
                    stage_results.append(res)
                except Exception as e:
                    stage_failures.append({"name": name, "reason": f"error:{type(e).__name__}", "detail": str(e)})
                    self._log(f"Failed: {name} ({type(e).__name__})")
            return stage_results, stage_failures, budget_exhausted

        ys_arr = np.asarray(ys)
        n_rows = len(ys_arr)
        cv_used = int(stage["cv"])
        if self.task == "classification":
            uniq, counts = np.unique(ys_arr, return_counts=True)
            if len(uniq) >= 2 and int(counts.min()) >= 2:
                cv_used = max(2, min(cv_used, int(counts.min())))
                splitter = StratifiedKFold(n_splits=cv_used, shuffle=True, random_state=self.random_state)
            else:
                cv_used = max(2, min(cv_used, max(2, n_rows // 2)))
                splitter = KFold(n_splits=cv_used, shuffle=True, random_state=self.random_state)
        else:
            cv_used = max(2, min(cv_used, max(2, n_rows // 2)))
            splitter = KFold(n_splits=cv_used, shuffle=True, random_state=self.random_state)

        splits = list(splitter.split(Xs, ys_arr))
        n_folds = len(splits)

        base_pre = survivors[0][1].named_steps["pre"]
        base_sel = survivors[0][1].named_steps.get("select")
        is_df = hasattr(Xs, "iloc")
        Xs_np = None if is_df else _safe_numpy(Xs)

        def _build_fold_entry(fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray) -> Optional[_FoldCacheEntry]:
            if self.time_budget_s is not None and (time.time() - t0) >= self.time_budget_s:
                return None
            if is_df:
                X_tr_raw = Xs.iloc[train_idx]
                X_val_raw = Xs.iloc[val_idx]
            else:
                X_tr_raw = Xs_np[train_idx]
                X_val_raw = Xs_np[val_idx]
            y_tr = ys_arr[train_idx]
            y_val = ys_arr[val_idx]

            pre_f = clone(base_pre)
            X_tr_proc = pre_f.fit_transform(X_tr_raw, y_tr)
            X_val_proc = pre_f.transform(X_val_raw)

            sel_f = None
            if base_sel is not None:
                sel_f = clone(base_sel)
                X_tr_proc = sel_f.fit_transform(X_tr_proc, y_tr)
                X_val_proc = sel_f.transform(X_val_proc)

            data_mem_bytes = _estimate_array_nbytes(X_tr_proc) * 2 + _estimate_array_nbytes(y_tr)
            return _FoldCacheEntry(
                train_idx=train_idx,
                val_idx=val_idx,
                X_tr_raw=X_tr_raw,
                X_val_raw=X_val_raw,
                y_tr=y_tr,
                y_val=y_val,
                X_tr_proc=X_tr_proc,
                X_val_proc=X_val_proc,
                pre_f=pre_f,
                sel_f=sel_f,
                data_mem_bytes=data_mem_bytes,
            )

        fold_cache: List[Optional[_FoldCacheEntry]] = [None] * n_folds
        prep_workers = self._resolve_max_workers(n_folds, n_rows=n_rows)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if prep_workers == 1:
                for fi, (tr_idx, va_idx) in enumerate(splits):
                    fold_cache[fi] = _build_fold_entry(fi, tr_idx, va_idx)
                    if fold_cache[fi] is None:
                        budget_exhausted = True
                        break
            else:
                with ThreadPoolExecutor(max_workers=prep_workers) as prep_pool:
                    fut_map = {
                        prep_pool.submit(_build_fold_entry, fi, tr_idx, va_idx): fi
                        for fi, (tr_idx, va_idx) in enumerate(splits)
                    }
                    for fut in as_completed(fut_map):
                        fi = fut_map[fut]
                        entry = fut.result()
                        if entry is None:
                            budget_exhausted = True
                        else:
                            fold_cache[fi] = entry

        if any(e is None for e in fold_cache):
            self._log("Time budget reached during fold preparation; stopping evaluation.")
            return stage_results, stage_failures, True

        entries: List[_FoldCacheEntry] = [e for e in fold_cache if e is not None]
        scorer_fn = get_scorer(self.scorer)
        capture_oof = bool(is_full and self.enable_stacking and n_rows >= 60)
        global_classes = np.unique(ys_arr) if (capture_oof and self.task == "classification") else None

        n_cands = len(survivors)
        fold_scores: List[List[Optional[float]]] = [[None] * n_folds for _ in range(n_cands)]
        fold_times: List[List[float]] = [[0.0] * n_folds for _ in range(n_cands)]
        fold_mems: List[List[float]] = [[0.0] * n_folds for _ in range(n_cands)]
        fold_oof: List[List[Optional[np.ndarray]]] = [[None] * n_folds for _ in range(n_cands)]
        fold0_models: List[Any] = [None] * n_cands
        cand_errors: Dict[int, Exception] = {}

        def _eval_pair(
            cand_idx: int, fold_idx: int
        ) -> Tuple[int, int, Optional[float], float, float, Any, Optional[np.ndarray], Optional[Exception]]:
            if self.time_budget_s is not None and (time.time() - t0) >= self.time_budget_s:
                return cand_idx, fold_idx, None, 0.0, 0.0, None, None, None
            entry = entries[fold_idx]
            try:
                model_step = clone(survivors[cand_idx][1].named_steps["model"])
                t_start = time.perf_counter()
                model_step.fit(entry.X_tr_proc, entry.y_tr)
                score = float(scorer_fn(model_step, entry.X_val_proc, entry.y_val))
                elapsed = float(time.perf_counter() - t_start)
                total_bytes = entry.data_mem_bytes + _estimate_model_nbytes(model_step)
                mem_mb = float(max(0.01, total_bytes / (1024 * 1024)))
                keep_model = model_step if fold_idx == 0 else None
                oof_arr = None
                if capture_oof:
                    oof_arr = self._extract_meta_features(model_step, entry.X_val_proc, global_classes)
                return cand_idx, fold_idx, score, elapsed, mem_mb, keep_model, oof_arr, None
            except Exception as exc:
                return cand_idx, fold_idx, None, 0.0, 0.0, None, None, exc

        n_tasks = n_cands * n_folds
        eval_workers = self._resolve_max_workers(n_tasks, n_rows=n_rows)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if eval_workers == 1:
                for ci in range(n_cands):
                    if self.time_budget_s is not None and (time.time() - t0) >= self.time_budget_s:
                        budget_exhausted = True
                        break
                    for fi in range(n_folds):
                        c_i, f_i, sc, el, mb, mod0, oof_arr, exc = _eval_pair(ci, fi)
                        if exc is not None:
                            cand_errors[c_i] = exc
                            break
                        if sc is None:
                            budget_exhausted = True
                            break
                        fold_scores[c_i][f_i] = sc
                        fold_times[c_i][f_i] = el
                        fold_mems[c_i][f_i] = mb
                        fold_oof[c_i][f_i] = oof_arr
                        if f_i == 0 and mod0 is not None:
                            fold0_models[c_i] = mod0
                    if budget_exhausted:
                        break
            else:
                with ThreadPoolExecutor(max_workers=eval_workers) as pool:
                    futures = [
                        pool.submit(_eval_pair, ci, fi)
                        for ci in range(n_cands)
                        for fi in range(n_folds)
                    ]
                    for fut in as_completed(futures):
                        c_i, f_i, sc, el, mb, mod0, oof_arr, exc = fut.result()
                        if exc is not None:
                            if c_i not in cand_errors:
                                cand_errors[c_i] = exc
                        elif sc is None:
                            budget_exhausted = True
                        else:
                            fold_scores[c_i][f_i] = sc
                            fold_times[c_i][f_i] = el
                            fold_mems[c_i][f_i] = mb
                            fold_oof[c_i][f_i] = oof_arr
                            if f_i == 0 and mod0 is not None:
                                fold0_models[c_i] = mod0
                        if self.time_budget_s is not None and (time.time() - t0) >= self.time_budget_s:
                            budget_exhausted = True
                            for rem in futures:
                                rem.cancel()
                            break

        if budget_exhausted:
            self._log("Time budget reached; stopping evaluation.")

        # Precompute Fold 0's perturbed validation matrices ONCE for Stage 1
        e0 = entries[0]
        Xm_proc = None
        Xj_proc = None
        y_noisy = None
        if is_full:
            rng_stress = np.random.RandomState(self.random_state)
            try:
                Xm_raw = _apply_missingness_noise(e0.X_val_raw, missing_frac=0.02, rng=rng_stress)
                Xm_proc = e0.pre_f.transform(Xm_raw)
                if e0.sel_f is not None:
                    Xm_proc = e0.sel_f.transform(Xm_proc)
            except Exception:
                Xm_proc = None
            try:
                Xj_raw = _apply_feature_jitter(e0.X_val_raw, frac=0.02, rng=rng_stress)
                Xj_proc = e0.pre_f.transform(Xj_raw)
                if e0.sel_f is not None:
                    Xj_proc = e0.sel_f.transform(Xj_proc)
            except Exception:
                Xj_proc = None
            if self.task == "classification":
                try:
                    y_noisy = _apply_label_noise(np.asarray(e0.y_val), frac=0.02, rng=rng_stress)
                except Exception:
                    y_noisy = None

        assert self.risk_report_ is not None
        for ci, (name, raw_pipe) in enumerate(survivors):
            if ci in cand_errors:
                exc = cand_errors[ci]
                stage_failures.append({"name": name, "reason": f"error:{type(exc).__name__}", "detail": str(exc)})
                self._log(f"Failed: {name} ({type(exc).__name__})")
                continue

            scores_ci = fold_scores[ci]
            if any(s is None for s in scores_ci) or fold0_models[ci] is None:
                continue

            scores_arr = np.asarray(scores_ci, dtype=float)
            metric = float(np.mean(scores_arr))
            cv_std = float(np.std(scores_arr))
            train_time_s = float(sum(fold_times[ci]))
            peak_mem_mb = float(max(fold_mems[ci])) if fold_mems[ci] else 0.01

            risk_pen = compute_risk_penalty(
                self.risk_report_,
                cv_score_std=cv_std,
                preference=self.preference,
            )

            model_fold0 = fold0_models[ci]
            steps_f0 = [("pre", e0.pre_f)]
            if e0.sel_f is not None:
                steps_f0.append(("select", e0.sel_f))
            steps_f0.append(("model", model_fold0))
            fold0_pipe = Pipeline(steps=steps_f0)

            if is_full:
                latency = measure_latency_ms_per_row(fold0_pipe, e0.X_val_raw)
                baseline_f0 = float(scores_arr[0])
                stress_results: List[StressTestResult] = []
                if Xm_proc is not None:
                    try:
                        m_score = float(scorer_fn(model_fold0, Xm_proc, e0.y_val))
                        stress_results.append(StressTestResult("missingness_2pct", m_score, baseline_f0 - m_score))
                    except Exception:
                        pass
                if Xj_proc is not None:
                    try:
                        j_score = float(scorer_fn(model_fold0, Xj_proc, e0.y_val))
                        stress_results.append(StressTestResult("feature_jitter_2pct", j_score, baseline_f0 - j_score))
                    except Exception:
                        pass
                if self.task == "classification" and y_noisy is not None:
                    try:
                        ln_score = float(scorer_fn(model_fold0, e0.X_val_proc, y_noisy))
                        stress_results.append(StressTestResult("label_noise_2pct", ln_score, baseline_f0 - ln_score))
                    except Exception:
                        pass
                worst_drop = float(max(0.0, max(t.delta for t in stress_results))) if stress_results else 0.0
            else:
                latency = 0.0
                baseline_f0 = float(scores_arr[0])
                stress_results = []
                worst_drop = 0.0

            objectives = (
                metric,
                math.log1p(train_time_s),
                math.log1p(peak_mem_mb),
                math.log1p(latency),
                risk_pen,
                worst_drop,
            )

            res = CandidateResult(
                name=name,
                pipeline=fold0_pipe,
                metric=metric,
                train_time_s=train_time_s,
                peak_mem_mb=peak_mem_mb,
                latency_ms_per_row=latency,
                risk_penalty=risk_pen,
                objectives=objectives,
                meta={
                    "cv_scores": scores_arr.tolist(),
                    "cv_score_std": cv_std,
                    "stress_baseline_metric": baseline_f0,
                    "stress_tests": [asdict(r) for r in stress_results],
                    "stress_worst_drop": worst_drop,
                },
            )

            if is_full and not self._passes_constraints(res):
                self._log(f"Pruned by constraints: {name}")
                stage_failures.append({
                    "name": name,
                    "reason": "constraints",
                    "metrics": {
                        "train_time_s": res.train_time_s,
                        "peak_mem_mb": res.peak_mem_mb,
                        "latency_ms_per_row": res.latency_ms_per_row,
                        "risk_penalty": res.risk_penalty,
                        "metric": res.metric,
                        "stress_worst_drop": res.meta.get("stress_worst_drop"),
                    },
                })
                continue

            stage_results.append(res)
            self._log(
                f"{name}: metric={res.metric:.4f}, time={res.train_time_s:.2f}s, "
                f"mem={res.peak_mem_mb:.1f}MB, lat={res.latency_ms_per_row:.2f}ms/row, "
                f"risk={res.risk_penalty:.2f}, robustness_drop={res.meta.get('stress_worst_drop', 0.0):.4f}"
            )

        if capture_oof and stage_results:
            oof_by_name: Dict[str, np.ndarray] = {}
            fold0_by_name: Dict[str, Any] = {}
            for ci, (name, _) in enumerate(survivors):
                if ci in cand_errors or fold0_models[ci] is None:
                    continue
                parts = fold_oof[ci]
                if any(p is None for p in parts):
                    continue
                widths = {p.shape[1] for p in parts if p is not None and p.ndim == 2}
                if len(widths) != 1:
                    continue
                d = next(iter(widths))
                oof_mat = np.zeros((n_rows, d), dtype=float)
                ok_fill = True
                for fi, p in enumerate(parts):
                    assert p is not None
                    if p.shape[0] != len(entries[fi].val_idx):
                        ok_fill = False
                        break
                    oof_mat[entries[fi].val_idx] = p
                if ok_fill:
                    oof_by_name[name] = oof_mat
                    fold0_by_name[name] = fold0_models[ci]
            self._last_full_oof_cache = {
                "entries": entries,
                "ys_arr": ys_arr,
                "Xm_proc": Xm_proc,
                "Xj_proc": Xj_proc,
                "y_noisy": y_noisy,
                "global_classes": global_classes,
                "oof_by_name": oof_by_name,
                "fold0_by_name": fold0_by_name,
            }

        return stage_results, stage_failures, budget_exhausted

    def _extract_meta_features(
        self,
        model_step: Any,
        X_mat: Any,
        global_classes: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Extract leakage-free 2D meta-features from a fitted base estimator."""
        return _extract_meta_features_for_task(self.task, model_step, X_mat, global_classes)

    @staticmethod
    def _model_family(name: str, estimator: Any) -> str:
        """Map a candidate model to a coarse family for diverse stacking selection."""
        cls = type(estimator).__name__.lower()
        lname = (name or "").lower()
        if any(k in cls or k in lname for k in ("logistic", "ridge", "lasso", "elastic", "linear", "huber", "sgd")):
            return "linear"
        if "extratrees" in cls:
            return "extratrees"
        if "randomforest" in cls:
            return "randomforest"
        if "histgradient" in cls or "gradientboosting" in cls:
            return "gboost"
        if "lgbm" in cls or "lightgbm" in lname:
            return "lightgbm"
        if "xgb" in cls:
            return "xgboost"
        if "catboost" in cls:
            return "catboost"
        if "mlp" in cls:
            return "mlp"
        if "svc" in cls or "svr" in cls:
            return "svm"
        if "kneighbors" in cls or "knn" in lname:
            return "knn"
        return cls

    def _build_stacking_candidate(
        self,
        results: List[CandidateResult],
        raw_templates: Dict[str, Pipeline],
        X: Any,
        y: Any,
        stage0_results: Optional[List[CandidateResult]] = None,
    ) -> Optional[Tuple[str, Pipeline]]:
        """Synthesize a Pareto-aware multi-family StackingEnsemble from top diverse survivors."""
        if not self.enable_stacking or len(results) < 2:
            return None
        y_arr = np.asarray(y)
        n_samples = len(y_arr)
        if n_samples < 60:
            return None

        # Determine safe inner CV folds for stacking meta-features (2-fold keeps final refit fast)
        stack_cv = min(2, max(2, int(self.cv)))
        if self.task == "classification":
            _, counts = np.unique(y_arr, return_counts=True)
            if len(counts) < 2 or int(counts.min()) < 2:
                return None
            stack_cv = min(stack_cv, int(counts.min()))
            if stack_cv < 2:
                return None

        ranked = sorted(
            (r for r in results if math.isfinite(float(r.metric))),
            key=lambda r: float(r.metric - 0.5 * float(r.meta.get("stress_worst_drop", 0.0))),
            reverse=True,
        )
        if len(ranked) < 2:
            return None

        # Build candidate priority list: full-stage survivors first, then Stage 0 survivors for family diversity
        combined_pool: List[CandidateResult] = list(ranked)
        seen_names = {r.name for r in combined_pool}
        if stage0_results:
            s0_ranked = sorted(
                (r for r in stage0_results if math.isfinite(float(r.metric)) and r.name not in seen_names),
                key=lambda r: float(r.metric),
                reverse=True,
            )
            combined_pool.extend(s0_ranked)

        base_estimators: List[Tuple[str, Any]] = []
        seen_families: set[str] = set()
        anchor_pipe: Optional[Pipeline] = None

        for r in combined_pool:
            tpl = raw_templates.get(r.name)
            if not isinstance(tpl, Pipeline) or "model" not in tpl.named_steps:
                continue
            model_step = tpl.named_steps["model"]
            fam = self._model_family(r.name, model_step)
            if fam in seen_families:
                continue
            est = clone(model_step)
            if hasattr(est, "n_jobs"):
                try:
                    est.set_params(n_jobs=1)
                except Exception:
                    pass
            est_key = f"{fam}_{len(base_estimators)}"
            base_estimators.append((est_key, est))
            seen_families.add(fam)
            if anchor_pipe is None:
                anchor_pipe = tpl
            if len(base_estimators) >= 3:
                break

        # Last-resort diversity from generated templates if all Stage 0/full survivors share 1 family
        if len(base_estimators) < 2 and anchor_pipe is not None:
            for tname, tpl in raw_templates.items():
                if not isinstance(tpl, Pipeline) or "model" not in tpl.named_steps:
                    continue
                model_step = tpl.named_steps["model"]
                fam = self._model_family(tname, model_step)
                if fam in seen_families:
                    continue
                est = clone(model_step)
                if hasattr(est, "n_jobs"):
                    try:
                        est.set_params(n_jobs=1)
                    except Exception:
                        pass
                base_estimators.append((f"{fam}_{len(base_estimators)}", est))
                seen_families.add(fam)
                if len(base_estimators) >= 2:
                    break

        if len(base_estimators) < 2 or anchor_pipe is None:
            return None

        if self.task == "classification":
            meta_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            stacker: Any = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_model,
                cv=stack_cv,
                n_jobs=1,
                passthrough=False,
            )
        else:
            meta_model = RidgeCV(alphas=np.logspace(-3, 2, 12))
            stacker = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_model,
                cv=stack_cv,
                n_jobs=1,
                passthrough=False,
            )

        steps: List[Tuple[str, Any]] = []
        if "pre" in anchor_pipe.named_steps:
            steps.append(("pre", clone(anchor_pipe.named_steps["pre"])))
        else:
            steps.append(("pre", self._build_preprocessor(X)))
        if "select" in anchor_pipe.named_steps:
            steps.append(("select", clone(anchor_pipe.named_steps["select"])))
        steps.append(("model", stacker))
        return ("StackingEnsemble", Pipeline(steps=steps))

    def _evaluate_oof_stacking(
        self,
        results: List[CandidateResult],
        raw_templates: Dict[str, Pipeline],
        X: Any,
        y: Any,
        t0: float,
        stage0_results: Optional[List[CandidateResult]] = None,
    ) -> Optional[Tuple[str, Pipeline, CandidateResult]]:
        """Evaluate a multi-family StackingEnsemble using cached Stage-1 OOF predictions.

        Reuses the exact K-fold Stage-1 out-of-fold predictions so CV evaluation
        takes milliseconds instead of running nested KxM inner base-model fits,
        while still constructing a genuine sklearn StackingClassifier/StackingRegressor
        for full-data refit if selected on the Pareto front.
        """
        cache = getattr(self, "_last_full_oof_cache", None)
        if not self.enable_stacking or not cache or len(results) < 2:
            return None

        entries: List[_FoldCacheEntry] = cache["entries"]
        ys_arr: np.ndarray = cache["ys_arr"]
        n_rows = len(ys_arr)
        if n_rows < 60 or len(entries) < 2:
            return None

        global_classes = cache["global_classes"]
        if self.task == "classification":
            if global_classes is None or len(global_classes) < 2:
                return None
            _, counts = np.unique(ys_arr, return_counts=True)
            if int(counts.min()) < 2:
                return None

        remaining = (
            float(self.time_budget_s) - (time.time() - t0)
            if self.time_budget_s is not None
            else float("inf")
        )
        if remaining < 2.0:
            return None

        oof_by_name: Dict[str, np.ndarray] = cache["oof_by_name"]
        fold0_by_name: Dict[str, Any] = cache["fold0_by_name"]

        ranked = sorted(
            (r for r in results if math.isfinite(float(r.metric)) and r.name in oof_by_name and r.name in fold0_by_name),
            key=lambda r: float(r.metric - 0.5 * float(r.meta.get("stress_worst_drop", 0.0))),
            reverse=True,
        )
        if not ranked:
            return None

        chosen: List[Dict[str, Any]] = []
        seen_families: set[str] = set()
        anchor_pipe: Optional[Pipeline] = None

        for r in ranked:
            tpl = raw_templates.get(r.name)
            if not isinstance(tpl, Pipeline) or "model" not in tpl.named_steps:
                continue
            model_step = tpl.named_steps["model"]
            fam = self._model_family(r.name, model_step)
            if fam in seen_families:
                continue
            chosen.append(
                {
                    "name": r.name,
                    "family": fam,
                    "model_step": model_step,
                    "oof": oof_by_name[r.name],
                    "fold0_model": fold0_by_name[r.name],
                    "train_time_s": float(r.train_time_s),
                    "peak_mem_mb": float(r.peak_mem_mb),
                }
            )
            seen_families.add(fam)
            if anchor_pipe is None:
                anchor_pipe = tpl
            if len(chosen) >= 3:
                break

        # If all Stage-1 survivors came from 1 family, evaluate ONE fast complementary
        # family across the ALREADY-preprocessed fold entries (zero preprocessing overhead).
        if len(chosen) == 1 and anchor_pipe is not None and remaining >= 4.0:
            fallback_candidates: List[Tuple[str, Pipeline, float]] = []
            if stage0_results:
                for r0 in sorted(
                    (r for r in stage0_results if math.isfinite(float(r.metric))),
                    key=lambda r: float(r.metric),
                    reverse=True,
                ):
                    tpl = raw_templates.get(r0.name)
                    if isinstance(tpl, Pipeline) and "model" in tpl.named_steps:
                        fallback_candidates.append((r0.name, tpl, float(r0.train_time_s)))
            for tname, tpl in raw_templates.items():
                if isinstance(tpl, Pipeline) and "model" in tpl.named_steps:
                    fallback_candidates.append((tname, tpl, 0.5))

            for cand_name, tpl, s0_time in fallback_candidates:
                model_step = tpl.named_steps["model"]
                fam = self._model_family(cand_name, model_step)
                if fam in seen_families:
                    continue
                # Guard: skip expensive complementary models when budget is tight
                if self.time_budget_s is not None and (s0_time * len(entries) * 2.5) > (0.35 * remaining):
                    continue
                try:
                    parts: List[np.ndarray] = []
                    f0_mod = None
                    t_base = 0.0
                    mem_base = 0.01
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        for fi, entry in enumerate(entries):
                            if self.time_budget_s is not None and (time.time() - t0) >= (self.time_budget_s - 1.5):
                                parts = []
                                break
                            est_f = clone(model_step)
                            if hasattr(est_f, "n_jobs"):
                                try:
                                    est_f.set_params(n_jobs=1)
                                except Exception:
                                    pass
                            ts = time.perf_counter()
                            est_f.fit(entry.X_tr_proc, entry.y_tr)
                            t_base += float(time.perf_counter() - ts)
                            oof_p = self._extract_meta_features(est_f, entry.X_val_proc, global_classes)
                            if oof_p is None:
                                parts = []
                                break
                            parts.append(oof_p)
                            if fi == 0:
                                f0_mod = est_f
                            mem_base = max(mem_base, (entry.data_mem_bytes + _estimate_model_nbytes(est_f)) / (1024 * 1024))
                    if len(parts) == len(entries) and f0_mod is not None:
                        widths = {p.shape[1] for p in parts if p.ndim == 2}
                        if len(widths) == 1:
                            d = next(iter(widths))
                            oof_mat = np.zeros((n_rows, d), dtype=float)
                            for fi, p in enumerate(parts):
                                oof_mat[entries[fi].val_idx] = p
                            chosen.append(
                                {
                                    "name": cand_name,
                                    "family": fam,
                                    "model_step": model_step,
                                    "oof": oof_mat,
                                    "fold0_model": f0_mod,
                                    "train_time_s": float(t_base),
                                    "peak_mem_mb": float(mem_base),
                                }
                            )
                            seen_families.add(fam)
                            break
                except Exception:
                    continue

        if len(chosen) < 2 or anchor_pipe is None:
            return None

        # Check that even if StackingEnsemble wins, its single cv=2 final refit will fit inside budget
        base_time_sum = float(sum(b["train_time_s"] for b in chosen))
        est_final_refit_s = (3.0 / max(1, len(entries))) * base_time_sum
        remaining_now = (
            float(self.time_budget_s) - (time.time() - t0)
            if self.time_budget_s is not None
            else float("inf")
        )
        if self.time_budget_s is not None and est_final_refit_s > max(2.5, 0.55 * remaining_now):
            # Keep only the 2 fastest diverse families if 3 exceeded final refit budget
            if len(chosen) > 2:
                chosen = sorted(chosen, key=lambda b: b["train_time_s"])[:2]
                base_time_sum = float(sum(b["train_time_s"] for b in chosen))
                est_final_refit_s = (3.0 / max(1, len(entries))) * base_time_sum
            if est_final_refit_s > max(2.5, 0.55 * remaining_now):
                return None

        Z_oof = np.hstack([b["oof"] for b in chosen])
        Z_oof = np.nan_to_num(Z_oof, nan=0.0, posinf=0.0, neginf=0.0)
        scorer_fn = get_scorer(self.scorer)

        if self.task == "classification":
            meta_tpl: Any = LogisticRegression(max_iter=1000, random_state=self.random_state)
        else:
            meta_tpl = RidgeCV(alphas=np.logspace(-3, 2, 12))

        meta_scores: List[float] = []
        fold0_meta = None
        t_meta_start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for fi, entry in enumerate(entries):
                meta_f = clone(meta_tpl)
                meta_f.fit(Z_oof[entry.train_idx], ys_arr[entry.train_idx])
                sc = float(scorer_fn(meta_f, Z_oof[entry.val_idx], entry.y_val))
                meta_scores.append(sc)
                if fi == 0:
                    fold0_meta = meta_f
        meta_cv_time = float(time.perf_counter() - t_meta_start)

        if fold0_meta is None or not all(math.isfinite(s) for s in meta_scores):
            return None

        scores_arr = np.asarray(meta_scores, dtype=float)
        metric = float(np.mean(scores_arr))
        cv_std = float(np.std(scores_arr))
        train_time_s = float(base_time_sum + meta_cv_time)
        peak_mem_mb = float(max(0.01, sum(b["peak_mem_mb"] for b in chosen)))
        assert self.risk_report_ is not None
        risk_pen = compute_risk_penalty(
            self.risk_report_,
            cv_score_std=cv_std,
            preference=self.preference,
        )

        # Lightweight top-level picklable Fold-0 inference adapter for genuine latency & stress evaluation
        base_f0_models = [b["fold0_model"] for b in chosen]
        adapter = _Fold0StackAdapter(
            base_models=base_f0_models,
            meta_model=fold0_meta,
            task=self.task,
            global_classes=global_classes,
        )
        e0 = entries[0]
        steps_f0: List[Tuple[str, Any]] = [("pre", e0.pre_f)]
        if e0.sel_f is not None:
            steps_f0.append(("select", e0.sel_f))
        steps_f0.append(("model", adapter))
        fold0_pipe = Pipeline(steps=steps_f0)

        latency = measure_latency_ms_per_row(fold0_pipe, e0.X_val_raw)
        baseline_f0 = float(scores_arr[0])
        stress_results: List[StressTestResult] = []
        Xm_proc = cache["Xm_proc"]
        Xj_proc = cache["Xj_proc"]
        y_noisy = cache["y_noisy"]
        if Xm_proc is not None:
            try:
                m_score = float(scorer_fn(adapter, Xm_proc, e0.y_val))
                stress_results.append(StressTestResult("missingness_2pct", m_score, baseline_f0 - m_score))
            except Exception:
                pass
        if Xj_proc is not None:
            try:
                j_score = float(scorer_fn(adapter, Xj_proc, e0.y_val))
                stress_results.append(StressTestResult("feature_jitter_2pct", j_score, baseline_f0 - j_score))
            except Exception:
                pass
        if self.task == "classification" and y_noisy is not None:
            try:
                ln_score = float(scorer_fn(adapter, e0.X_val_proc, y_noisy))
                stress_results.append(StressTestResult("label_noise_2pct", ln_score, baseline_f0 - ln_score))
            except Exception:
                pass
        worst_drop = float(max(0.0, max(t.delta for t in stress_results))) if stress_results else 0.0

        objectives = (
            metric,
            math.log1p(train_time_s),
            math.log1p(peak_mem_mb),
            math.log1p(latency),
            risk_pen,
            worst_drop,
        )
        stack_res = CandidateResult(
            name="StackingEnsemble",
            pipeline=fold0_pipe,
            metric=metric,
            train_time_s=train_time_s,
            peak_mem_mb=peak_mem_mb,
            latency_ms_per_row=latency,
            risk_penalty=risk_pen,
            objectives=objectives,
            meta={
                "cv_scores": scores_arr.tolist(),
                "cv_score_std": cv_std,
                "stress_baseline_metric": baseline_f0,
                "stress_tests": [asdict(r) for r in stress_results],
                "stress_worst_drop": worst_drop,
                "base_estimators": [b["name"] for b in chosen],
            },
        )
        if not self._passes_constraints(stack_res):
            return None

        # Construct the genuine deployable sklearn StackingClassifier/StackingRegressor template
        base_estimators: List[Tuple[str, Any]] = []
        for idx, b in enumerate(chosen):
            est = clone(b["model_step"])
            if hasattr(est, "n_jobs"):
                try:
                    est.set_params(n_jobs=1)
                except Exception:
                    pass
            base_estimators.append((f"{b['family']}_{idx}", est))

        stack_cv = 2
        if self.task == "classification":
            stacker: Any = StackingClassifier(
                estimators=base_estimators,
                final_estimator=clone(meta_tpl),
                cv=stack_cv,
                n_jobs=1,
                passthrough=False,
            )
        else:
            stacker = StackingRegressor(
                estimators=base_estimators,
                final_estimator=clone(meta_tpl),
                cv=stack_cv,
                n_jobs=1,
                passthrough=False,
            )

        tpl_steps: List[Tuple[str, Any]] = []
        if "pre" in anchor_pipe.named_steps:
            tpl_steps.append(("pre", clone(anchor_pipe.named_steps["pre"])))
        else:
            tpl_steps.append(("pre", self._build_preprocessor(X)))
        if "select" in anchor_pipe.named_steps:
            tpl_steps.append(("select", clone(anchor_pipe.named_steps["select"])))
        tpl_steps.append(("model", stacker))
        return ("StackingEnsemble", Pipeline(steps=tpl_steps), stack_res)

    def _passes_constraints(self, res: CandidateResult) -> bool:
        """Check if a candidate passes all constraint guardrails."""
        if self.max_train_time_s is not None and res.train_time_s > self.max_train_time_s:
            return False
        if self.max_peak_mem_mb is not None and res.peak_mem_mb > self.max_peak_mem_mb:
            return False
        if self.max_latency_ms_per_row is not None and res.latency_ms_per_row > self.max_latency_ms_per_row:
            return False
        return True

    def _risk_features(self, rr: RiskReport) -> np.ndarray:
        """Extract risk features for calibration."""
        # Handle both RiskReport object and dict (from asdict)
        if isinstance(rr, dict):
            missing_ratio = rr.get('missing_ratio', 0.0)
            n_rows = rr.get('n_rows', 0)
            n_cols = rr.get('n_cols', 0)
            vif_max = rr.get('vif_max', 1.0)
            class_imbalance = rr.get('class_imbalance_max_ratio', 0.0)
            cv_std = rr.get('cv_score_std', 0.0)
        else:
            missing_ratio = getattr(rr, 'missing_ratio', 0.0)
            n_rows = rr.n_rows
            n_cols = rr.n_cols
            vif_max = rr.vif_max
            class_imbalance = rr.class_imbalance_max_ratio or 0.0
            cv_std = rr.cv_score_std or 0.0
        
        feats = [
            float(n_rows),
            float(n_cols),
            float(vif_max),
            float(missing_ratio),
            float(class_imbalance),
            float(cv_std),
        ]
        return np.array(feats, dtype=float)

    def _fit_risk_calibrator(self, rr: RiskReport, results: List[CandidateResult]) -> Optional[Ridge]:
        """
        Fit a model that predicts worst-case robustness drop from risk features + candidate signals.
        Uses Ridge for stability.
        """
        try:
            if len(results) < 2:
                return None  # Need at least 2 samples to fit
            
            Xcal = []
            ycal = []
            base_feats = self._risk_features(rr)

            for r in results:
                worst_drop = float(r.meta.get("stress_worst_drop", 0.0))
                # Add candidate-level predictors
                x = np.concatenate([
                    base_feats,
                    np.array([
                        float(r.meta.get("cv_score_std", 0.0)),
                        float(r.latency_ms_per_row)
                    ], dtype=float),
                ])
                Xcal.append(x)
                ycal.append(worst_drop)

            Xcal = np.vstack(Xcal)
            ycal = np.array(ycal, dtype=float)

            # Only fit if we have variation in targets
            if np.std(ycal) < 1e-6:
                return None

            model = Ridge(alpha=1.0, random_state=self.random_state)
            model.fit(Xcal, ycal)
            return model
        except Exception:
            return None

    def _calibrated_risk_penalty(self, rr: RiskReport, res: CandidateResult, calibrator: Optional[Ridge]) -> float:
        """
        Combine original risk_penalty with predicted robustness drop.
        """
        base = float(res.risk_penalty)
        if calibrator is None:
            return base

        try:
            x = np.concatenate([
                self._risk_features(rr),
                np.array([
                    float(res.meta.get("cv_score_std", 0.0)),
                    float(res.latency_ms_per_row)
                ], dtype=float),
            ]).reshape(1, -1)

            pred_drop = float(calibrator.predict(x)[0])
            pred_drop = max(0.0, pred_drop)

            # Blend: keep original risk penalty, add calibrated robustness risk
            return float(base + 2.0 * pred_drop)
        except Exception:
            return base

    def fit(self, X: Any, y: Any) -> "_ACDSOEngineV3":
        """
        Run the full ACDSO optimization loop.

        1. Analyze dataset risk
        2. Generate adaptive candidates
        3. Evaluate candidates with multi-objective metrics
        4. Filter by constraints
        5. Construct Pareto front
        6. Select knee-point solution
        7. Build audit trail
        """
        # Start total wall-clock timer BEFORE risk analysis so time_budget_s covers full fit()
        t0 = time.time()

        # Step 0: Validate X and y upfront and sanitize input features
        n_x = len(X) if hasattr(X, '__len__') else np.asarray(X).shape[0]
        y_arr = np.asarray(y)
        if y_arr.ndim == 0 or len(y_arr) == 0:
            raise ValueError("Target y cannot be empty.")
        if n_x == 0:
            raise ValueError("Input X cannot be empty.")
        if len(y_arr) != n_x:
            raise ValueError(f"X and y must have the same number of samples (got {n_x} vs {len(y_arr)}).")

        if np.issubdtype(y_arr.dtype, np.floating):
            if not np.all(np.isfinite(y_arr)):
                raise ValueError("Target y contains NaN or infinite values.")
        elif y_arr.dtype == object or isinstance(y, (pd.Series, pd.Categorical)):
            y_s = pd.Series(y_arr)
            if y_s.isna().any():
                raise ValueError("Target y contains NaN/missing values.")
            for v in y_arr:
                if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)):
                    raise ValueError("Target y contains infinite values.")

        X = sanitize_features(X)

        # Step 0a: Adapt CV folds for small datasets
        n_samples = len(y_arr)
        original_cv = self.cv

        # Ensure we have at least 2 samples per fold
        max_cv_for_data = n_samples // 2
        if self.cv > max_cv_for_data:
            self.cv = max(2, max_cv_for_data)
            self._log(f"Reduced CV folds from {original_cv} to {self.cv} (small dataset: {n_samples} samples)")

        # For very small datasets, use leave-one-out style (but cap at 5)
        if n_samples < 20:
            self.cv = min(n_samples - 1, 5)
            self._log(f"Using {self.cv}-fold CV for very small dataset ({n_samples} samples)")

        # Step 0b: Auto-adjust scorer for multiclass or non-standard binary labels
        adjusted_scorer = _auto_adjust_scorer(self.scorer, y, self.task)
        if adjusted_scorer != self.scorer:
            self._log(f"Auto-adjusted scorer: {self.scorer} → {adjusted_scorer}")
            self.scorer = adjusted_scorer
            self._adjusted_scorer = adjusted_scorer
        else:
            self._adjusted_scorer = self.scorer

        # Step 1: Risk analysis (time-bounded)
        risk_budget = min(self.time_budget_s * 0.25, 10.0) if self.time_budget_s is not None else None
        self.risk_report_ = analyze_risk(
            X, y,
            task=self.task,
            scorer=self.scorer,
            cv=self.cv,
            random_state=self.random_state,
            llm_advisor=self.llm_advisor,
            time_budget_s=risk_budget,
        )
        self._log(f"Risk report: {asdict(self.risk_report_)}")

        # Step 2: Generate candidates
        candidates = self._generate_candidates(X, y, self.risk_report_)
        self._log(f"Generated {len(candidates)} candidates.")
        raw_templates: Dict[str, Pipeline] = {name: pipe for name, pipe in candidates}

        # Step 3: Evaluate candidates with successive halving (v3 innovation)
        results: List[CandidateResult] = []
        stage0_results: List[CandidateResult] = []
        failures: List[Dict[str, Any]] = []

        plan = self._successive_halving_plan(len(candidates))
        survivors = candidates[:]

        def _rank_key(r: CandidateResult) -> float:
            worst_drop = float(r.meta.get("stress_worst_drop", 0.0))
            return float(r.metric - 0.5 * worst_drop)

        for stage in plan:
            if not survivors:
                break

            self._log(f"Halving stage: {stage['name']} | row_frac={stage['row_frac']} cv={stage['cv']} keep={stage['keep']}")

            Xs, ys = self._subsample_rows(X, y, stage["row_frac"])
            stage_results, stage_failures, budget_exhausted = self._evaluate_stage(
                Xs, ys, survivors, stage, t0
            )

            stage_results.sort(key=_rank_key, reverse=True)

            keep = min(stage["keep"], len(stage_results))
            survivors = [
                (r.name, clone(raw_templates[r.name]) if r.name in raw_templates else clone(r.pipeline))
                for r in stage_results[:keep]
            ]

            failures.extend(stage_failures)

            if stage["name"] == "cheap":
                stage0_results = stage_results
            elif stage["name"] == "full":
                if stage_results:
                    results = stage_results
                elif budget_exhausted and stage0_results and not any(
                    f.get("reason") == "constraints" for f in stage_failures
                ):
                    # Budget expired during full stage; preserve Stage 0 survivors that pass constraints
                    passing_s0 = [r for r in stage0_results if self._passes_constraints(r)]
                    if passing_s0:
                        results = passing_s0

        # Step 3.25: Fast OOF-evaluated Pareto-aware multi-family StackingEnsemble
        try:
            if self.enable_stacking and len(results) >= 2:
                stack_eval = self._evaluate_oof_stacking(
                    results, raw_templates, X, y, t0, stage0_results=stage0_results
                )
                if stack_eval is not None:
                    stack_name, stack_pipe, stack_res = stack_eval
                    raw_templates[stack_name] = clone(stack_pipe)
                    results.append(stack_res)
                    self._log(
                        f"Added {stack_name} to Pareto pool (metric={stack_res.metric:.4f}, "
                        f"time={stack_res.train_time_s:.2f}s, bases={stack_res.meta.get('base_estimators')})"
                    )
        except Exception as stack_exc:
            failures.append({
                "name": "StackingEnsemble",
                "reason": f"error:{type(stack_exc).__name__}",
                "detail": str(stack_exc),
            })
        finally:
            self._last_full_oof_cache = None

        if not results:
            # Fallback: Create a simple baseline model that always works
            self._log("WARNING: All candidates failed. Using fallback baseline model.")
            from sklearn.dummy import DummyRegressor, DummyClassifier

            if self.task == "classification":
                fallback_model = DummyClassifier(strategy="most_frequent")
            else:
                fallback_model = DummyRegressor(strategy="mean")

            fallback_pipe = Pipeline([("pre", self._build_preprocessor(X)), ("model", fallback_model)])
            raw_templates["Baseline_Fallback"] = clone(fallback_pipe)

            try:
                fallback_res = self._evaluate_candidate(X, y, "Baseline_Fallback", fallback_pipe)
                results = [fallback_res]
                self._log(f"Fallback model: metric={fallback_res.metric:.4f}")
            except Exception as fallback_e:
                raise RuntimeError(
                    f"No candidates passed evaluation/constraints, and fallback also failed: {fallback_e}. "
                    f"Failures: {failures[:5]}"
                )

        self.all_results_ = results

        # Step 3.5: Fit risk calibrator and adjust risk penalties (v3 innovation)
        calibrator = self._fit_risk_calibrator(self.risk_report_, results)
        for r in results:
            r.risk_penalty = self._calibrated_risk_penalty(self.risk_report_, r, calibrator)
            obj = list(r.objectives)
            if len(obj) >= 5:
                obj[4] = r.risk_penalty
            r.objectives = tuple(obj)

        # Update risk report with best stability
        best_std = float(np.min([r.meta.get("cv_score_std", 0.0) for r in results]))
        self.risk_report_.cv_score_std = best_std

        # Step 5: Construct Pareto front
        finite_metric_results = [r for r in results if math.isfinite(float(r.metric))]
        pareto_pool = finite_metric_results if finite_metric_results else results
        front = pareto_front(pareto_pool)
        self.pareto_front_ = front
        self._log(f"Pareto front size: {len(front)}")

        # Step 6: Select best model based on selection_mode
        pref_norm = str(getattr(self, "preference", "balanced") or "balanced").strip().lower()
        if self.selection_mode == "best_score" or pref_norm in ("accuracy", "metric"):
            finite_res = [r for r in results if math.isfinite(float(r.metric))]
            pool = finite_res if finite_res else results
            best = max(
                pool,
                key=lambda r: (
                    float(r.metric) if math.isfinite(float(r.metric)) else float("-inf"),
                    -float(r.risk_penalty) if math.isfinite(float(r.risk_penalty)) else float("-inf"),
                    -float(r.train_time_s) if math.isfinite(float(r.train_time_s)) else float("-inf"),
                    str(r.name),
                ),
            )
            self._log(f"Selected: {best.name} (best-score: {best.metric:.4f})")
        else:
            # Exclude catastrophically weak baseline models from distorting the knee anchor
            finite_front = [c for c in front if math.isfinite(float(c.metric))]
            sel_front = finite_front if finite_front else front
            max_front_metric = max(float(c.metric) for c in sel_front)
            rel_cap = 0.03 if pref_norm in ("speed", "fast", "low_risk", "safe") else 0.025
            if math.isfinite(max_front_metric):
                scale_base = (
                    max(1e-12, abs(max_front_metric))
                    if max_front_metric < 0.0
                    else max(0.2, abs(max_front_metric))
                )
                tol = rel_cap * scale_base
            else:
                tol = rel_cap
            viable_front = [c for c in sel_front if float(c.metric) >= max_front_metric - tol]
            if not viable_front:
                viable_front = list(sel_front)
            m_ref = max(
                (abs(float(c.metric)) for c in viable_front if math.isfinite(float(c.metric))),
                default=1.0,
            )
            if m_ref <= 0.0:
                m_ref = 1.0
            viable_front.sort(
                key=lambda c: (
                    -round(float(c.metric) / m_ref, 8) if math.isfinite(float(c.metric)) else float("inf"),
                    round(float(c.risk_penalty), 4) if math.isfinite(float(c.risk_penalty)) else float("inf"),
                    round(float(c.robustness_drop), 4) if math.isfinite(float(c.robustness_drop)) else float("inf"),
                    round(float(c.train_time_s), 1) if math.isfinite(float(c.train_time_s)) else float("inf"),
                    str(c.name),
                )
            )
            best = knee_point_selection(viable_front, preference=self.preference)
            self._log(f"Selected: {best.name} (knee-point)")

        # Step 6.5: Refit ONLY the winning pipeline on 100% of (X, y)
        refitted_full = False
        try:
            template = raw_templates.get(best.name)
            final_pipe = clone(template) if template is not None else clone(best.pipeline)
            if n_samples >= 5000 and self.n_jobs != 1 and isinstance(final_pipe, Pipeline):
                model_step = final_pipe.named_steps.get("model")
                if model_step is not None and hasattr(model_step, "n_jobs"):
                    model_step.set_params(n_jobs=min(os.cpu_count() or 4, 4))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                final_pipe.fit(X, y)
            best.pipeline = final_pipe
            refitted_full = True
        except Exception:
            refitted_full = False

        self.best_result_ = best

        # Step 7: Build audit trail
        self.audit_trail_ = {
            "risk_report": asdict(self.risk_report_),
            "constraints": {
                "max_train_time_s": self.max_train_time_s,
                "max_peak_mem_mb": self.max_peak_mem_mb,
                "max_latency_ms_per_row": self.max_latency_ms_per_row,
            },
            "candidates_evaluated": [
                {
                    "name": r.name,
                    "metric": r.metric,
                    "train_time_s": r.train_time_s,
                    "peak_mem_mb": r.peak_mem_mb,
                    "latency_ms_per_row": r.latency_ms_per_row,
                    "risk_penalty": r.risk_penalty,
                    "cv_scores": r.meta.get("cv_scores"),
                    "cv_score_std": r.meta.get("cv_score_std"),
                    "stress_worst_drop": r.meta.get("stress_worst_drop"),
                    "stress_tests": r.meta.get("stress_tests"),
                }
                for r in results
            ],
            "pareto_front": [
                {
                    "name": r.name,
                    "metric": r.metric,
                    "train_time_s": r.train_time_s,
                    "peak_mem_mb": r.peak_mem_mb,
                    "latency_ms_per_row": r.latency_ms_per_row,
                    "risk_penalty": r.risk_penalty,
                }
                for r in front
            ],
            "pruned_or_failed": failures,
            "selected": {
                "name": best.name,
                "metric": best.metric,
                "train_time_s": best.train_time_s,
                "peak_mem_mb": best.peak_mem_mb,
                "latency_ms_per_row": best.latency_ms_per_row,
                "risk_penalty": best.risk_penalty,
                "stress_worst_drop": best.meta.get("stress_worst_drop"),
                "refitted_full": refitted_full,
            },
            "risk_calibrator": {
                "type": "ridge" if calibrator is not None else None,
                "used": calibrator is not None,
            },
        }

        return self


class ACDSOClassifier(BaseEstimator, ClassifierMixin):
    """
    ACDSO v3.0 Classifier - Risk-Optimized AutoML for Classification.

    A production-oriented AutoML classifier that unifies contextual risk
    diagnostics, adaptive pipeline synthesis, and multi-objective optimization.
    
    v3 Features:
    - Stress testing with robustness objective (6-objective optimization)
    - Successive halving for efficient candidate evaluation
    - Validation-based robustness measurement
    - Learned risk calibration

    Parameters
    ----------
    scorer : str, default="accuracy"
        Scoring metric for evaluation. Common options:
        "accuracy", "f1", "f1_macro", "f1_weighted", "roc_auc", "precision", "recall"

    cv : int, default=5
        Number of cross-validation folds.

    preference : str, default="balanced"
        Risk preference mode:
        - "safe": Higher weight on leakage and stability risks
        - "balanced": Balanced weighting across all risk factors

    time_budget_s : float, optional
        Maximum total time budget in seconds. If exceeded, evaluation stops.

    max_candidates : int, default=12
        Maximum number of candidate pipelines to evaluate.
        Increased from 4 to provide better AutoML search coverage.

    random_state : int, optional, default=42
        Random seed for reproducibility.

    max_train_time_s : float, optional
        Maximum allowed training time per candidate (constraint).

    max_peak_mem_mb : float, optional
        Maximum allowed peak memory usage (constraint).

    max_latency_ms_per_row : float, optional
        Maximum allowed inference latency per row (constraint).

    enable_feature_selection : bool, default=True
        Whether to enable automatic feature selection for high-multicollinearity data.

    llm_advisor : callable, optional
        Optional LLM advisor function for semantic analysis.
        Should accept Dict[str, Any] and return Dict[str, Any].

    verbose : int, default=0
        Verbosity level. 0 = silent, 1+ = progress messages.

    Attributes
    ----------
    best_estimator_ : Pipeline
        The selected best pipeline after fitting.

    risk_report_ : dict
        Comprehensive risk report for the dataset.

    pareto_front_ : list
        List of Pareto-optimal candidates.

    audit_trail_ : dict
        JSON-serializable audit artifact containing all evaluation details.

    Examples
    --------
    >>> from acdso import ACDSOClassifier
    >>> clf = ACDSOClassifier(scorer="f1", cv=5, preference="safe", verbose=1)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> print(clf.risk_report_)
    >>> print(clf.audit_trail_)

    >>> # With constraints
    >>> clf = ACDSOClassifier(
    ...     max_latency_ms_per_row=10.0,
    ...     max_peak_mem_mb=500.0,
    ...     time_budget_s=300,
    ... )
    >>> clf.fit(X, y)
    """

    def __init__(
        self,
        scorer: str = "accuracy",
        cv: int = 5,
        preference: str = "balanced",
        time_budget_s: Optional[float] = None,
        max_candidates: int = 12,
        random_state: Optional[int] = 42,
        max_train_time_s: Optional[float] = None,
        max_peak_mem_mb: Optional[float] = None,
        max_latency_ms_per_row: Optional[float] = None,
        enable_feature_selection: bool = True,
        llm_advisor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        verbose: int = 0,
        selection_mode: str = "knee_point",
        n_jobs: Optional[int] = None,
        enable_stacking: bool = True,
    ) -> None:
        self.scorer = scorer
        self.cv = cv
        self.preference = preference
        self.time_budget_s = time_budget_s
        self.max_candidates = max_candidates
        self.random_state = random_state
        self.max_train_time_s = max_train_time_s
        self.max_peak_mem_mb = max_peak_mem_mb
        self.max_latency_ms_per_row = max_latency_ms_per_row
        self.enable_feature_selection = enable_feature_selection
        self.llm_advisor = llm_advisor
        self.verbose = verbose
        self.selection_mode = selection_mode  # "knee_point" or "best_score"
        self.n_jobs = n_jobs
        self.enable_stacking = enable_stacking

        # Fitted attributes
        self.best_estimator_: Optional[Pipeline] = None
        self.risk_report_: Optional[Dict[str, Any]] = None
        self.pareto_front_: Optional[List[Dict[str, Any]]] = None
        self.audit_trail_: Optional[Dict[str, Any]] = None
        self._engine: Optional[_ACDSOEngineV3] = None

    def fit(self, X: Any, y: Any) -> "ACDSOClassifier":
        """
        Fit the ACDSO classifier.

        This runs the full optimization loop:
        1. Analyze dataset risk
        2. Generate adaptive candidates
        3. Evaluate with multi-objective metrics
        4. Select optimal model via Pareto knee-point or best score

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ACDSOClassifier
            Fitted estimator.
        """
        self._engine = _ACDSOEngineV3(
            task="classification",
            scorer=self.scorer,
            cv=self.cv,
            random_state=self.random_state,
            time_budget_s=self.time_budget_s,
            max_candidates=self.max_candidates,
            preference=self.preference,
            max_train_time_s=self.max_train_time_s,
            max_peak_mem_mb=self.max_peak_mem_mb,
            max_latency_ms_per_row=self.max_latency_ms_per_row,
            enable_feature_selection=self.enable_feature_selection,
            llm_advisor=self.llm_advisor,
            verbose=self.verbose,
            selection_mode=self.selection_mode,
            n_jobs=self.n_jobs,
            enable_stacking=self.enable_stacking,
        ).fit(X, y)

        assert self._engine.best_result_ is not None
        self.best_estimator_ = self._engine.best_result_.pipeline
        self.risk_report_ = asdict(self._engine.risk_report_) if self._engine.risk_report_ else None
        self.pareto_front_ = self._engine.audit_trail_.get("pareto_front")
        self.audit_trail_ = self._engine.audit_trail_
        
        # Store adjusted scorer internally without mutating public estimator param `scorer`
        self._effective_scorer_ = getattr(self._engine, "_adjusted_scorer", self.scorer)

        return self

    def predict(self, X: Any) -> Any:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("ACDSOClassifier is not fitted. Call fit() first.")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.

        Raises
        ------
        AttributeError
            If the selected estimator does not support probability prediction.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("ACDSOClassifier is not fitted. Call fit() first.")
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X)
        raise AttributeError("Selected estimator does not support predict_proba().")

    def score(self, X: Any, y: Any) -> float:
        """
        Return the score on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Score according to the scorer parameter.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("ACDSOClassifier is not fitted. Call fit() first.")
        scorer_name = getattr(self, "_effective_scorer_", self.scorer)
        scorer = get_scorer(scorer_name)
        return float(scorer(self.best_estimator_, X, y))


class ACDSORegressor(BaseEstimator, RegressorMixin):
    """
    ACDSO v3.0 Regressor - Risk-Optimized AutoML for Regression.

    A production-oriented AutoML regressor that unifies contextual risk
    diagnostics, adaptive pipeline synthesis, and multi-objective optimization.
    
    v3 Features:
    - Stress testing with robustness objective (6-objective optimization)
    - Successive halving for efficient candidate evaluation
    - Validation-based robustness measurement
    - Learned risk calibration

    Parameters
    ----------
    scorer : str, default="r2"
        Scoring metric for evaluation. Common options:
        "r2", "neg_mean_squared_error", "neg_mean_absolute_error",
        "neg_root_mean_squared_error"

    cv : int, default=5
        Number of cross-validation folds.

    preference : str, default="balanced"
        Risk preference mode:
        - "safe": Higher weight on leakage and stability risks
        - "balanced": Balanced weighting across all risk factors

    time_budget_s : float, optional
        Maximum total time budget in seconds.

    max_candidates : int, default=12
        Maximum number of candidate pipelines to evaluate.
        Increased from 4 to provide better AutoML search coverage.

    random_state : int, optional, default=42
        Random seed for reproducibility.

    max_train_time_s : float, optional
        Maximum allowed training time per candidate (constraint).

    max_peak_mem_mb : float, optional
        Maximum allowed peak memory usage (constraint).

    max_latency_ms_per_row : float, optional
        Maximum allowed inference latency per row (constraint).

    enable_feature_selection : bool, default=True
        Whether to enable automatic feature selection.

    llm_advisor : callable, optional
        Optional LLM advisor function for semantic analysis.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    best_estimator_ : Pipeline
        The selected best pipeline after fitting.

    risk_report_ : dict
        Comprehensive risk report for the dataset.

    pareto_front_ : list
        List of Pareto-optimal candidates.

    audit_trail_ : dict
        JSON-serializable audit artifact.

    Examples
    --------
    >>> from acdso import ACDSORegressor
    >>> reg = ACDSORegressor(scorer="r2", cv=5, verbose=1)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
    >>> print(reg.risk_report_)
    """

    def __init__(
        self,
        scorer: str = "r2",
        cv: int = 5,
        preference: str = "balanced",
        time_budget_s: Optional[float] = None,
        max_candidates: int = 12,
        random_state: Optional[int] = 42,
        max_train_time_s: Optional[float] = None,
        max_peak_mem_mb: Optional[float] = None,
        max_latency_ms_per_row: Optional[float] = None,
        enable_feature_selection: bool = True,
        llm_advisor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        verbose: int = 0,
        selection_mode: str = "knee_point",
        n_jobs: Optional[int] = None,
        enable_stacking: bool = True,
    ) -> None:
        self.scorer = scorer
        self.cv = cv
        self.preference = preference
        self.time_budget_s = time_budget_s
        self.max_candidates = max_candidates
        self.random_state = random_state
        self.max_train_time_s = max_train_time_s
        self.max_peak_mem_mb = max_peak_mem_mb
        self.max_latency_ms_per_row = max_latency_ms_per_row
        self.enable_feature_selection = enable_feature_selection
        self.llm_advisor = llm_advisor
        self.verbose = verbose
        self.selection_mode = selection_mode  # "knee_point" or "best_score"
        self.n_jobs = n_jobs
        self.enable_stacking = enable_stacking

        # Fitted attributes
        self.best_estimator_: Optional[Pipeline] = None
        self.risk_report_: Optional[Dict[str, Any]] = None
        self.pareto_front_: Optional[List[Dict[str, Any]]] = None
        self.audit_trail_: Optional[Dict[str, Any]] = None
        self._engine: Optional[_ACDSOEngineV3] = None

    def fit(self, X: Any, y: Any) -> "ACDSORegressor":
        """
        Fit the ACDSO regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ACDSORegressor
            Fitted estimator.
        """
        self._engine = _ACDSOEngineV3(
            task="regression",
            scorer=self.scorer,
            cv=self.cv,
            random_state=self.random_state,
            time_budget_s=self.time_budget_s,
            max_candidates=self.max_candidates,
            preference=self.preference,
            max_train_time_s=self.max_train_time_s,
            max_peak_mem_mb=self.max_peak_mem_mb,
            max_latency_ms_per_row=self.max_latency_ms_per_row,
            enable_feature_selection=self.enable_feature_selection,
            llm_advisor=self.llm_advisor,
            verbose=self.verbose,
            selection_mode=self.selection_mode,
            n_jobs=self.n_jobs,
            enable_stacking=self.enable_stacking,
        ).fit(X, y)

        assert self._engine.best_result_ is not None
        self.best_estimator_ = self._engine.best_result_.pipeline
        self.risk_report_ = asdict(self._engine.risk_report_) if self._engine.risk_report_ else None
        self.pareto_front_ = self._engine.audit_trail_.get("pareto_front")
        self.audit_trail_ = self._engine.audit_trail_
        
        # Store adjusted scorer internally without mutating public estimator param `scorer`
        self._effective_scorer_ = getattr(self._engine, "_adjusted_scorer", self.scorer)

        return self

    def predict(self, X: Any) -> Any:
        """
        Predict regression targets for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("ACDSORegressor is not fitted. Call fit() first.")
        return self.best_estimator_.predict(X)

    def score(self, X: Any, y: Any) -> float:
        """
        Return the score on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            Score according to the scorer parameter.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("ACDSORegressor is not fitted. Call fit() first.")
        scorer_name = getattr(self, "_effective_scorer_", self.scorer)
        scorer = get_scorer(scorer_name)
        return float(scorer(self.best_estimator_, X, y))

