"""
ACDSO - Learning-to-Rank Module (R5b)
=====================================

Provides ``ndcg_at_k`` and ``ACDSORanker``, a query-grouped Learning-to-Rank
estimator enforcing leak-free ``GroupKFold`` cross-validation over query IDs,
supporting pairwise preference ranking, query-centered pointwise Ridge /
HistGradientBoosting, optional LightGBM/XGBoost rankers, and free-text NLP
featurization.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from acdso.cv import (
    build_cv_splitter,
    iter_cv_splits,
    resolve_cv_strategy_name,
    splitter_supports_groups,
    validate_groups,
    validate_no_group_leakage,
)
from acdso.utils import build_preprocessor, sanitize_features, split_columns

_VALID_ALGORITHMS = (
    "auto",
    "pairwise",
    "pointwise",
    "pointwise_ridge",
    "pointwise_gbr",
    "lgbm_ranker",
    "xgb_ranker",
)


def _sanitize_json_obj(obj: Any) -> Any:
    """Recursively sanitize nested objects for strict RFC 8259 JSON."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_obj(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize_json_obj(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _coerce_1d_finite_float(arr_like: Any, name: str) -> np.ndarray:
    if arr_like is None:
        raise ValueError(f"{name} cannot be None.")
    arr = np.asarray(arr_like, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _validate_k(k: Optional[int]) -> Optional[int]:
    if k is None:
        return None
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an integer >= 1 or None, got {k!r}.")
    k_int = int(k)
    if k_int < 1:
        raise ValueError(f"k must be >= 1, got {k_int}.")
    return k_int


def _query_ndcg_at_k(rel: np.ndarray, scr: np.ndarray, k: Optional[int]) -> float:
    """Compute tie-aware exponential-gain NDCG@k for a single query group."""
    m = len(rel)
    if m == 0:
        return 0.0
    eff_k = m if k is None else min(int(k), m)
    if eff_k < 1:
        return 0.0

    discounts = np.log2(np.arange(eff_k, dtype=np.float64) + 2.0)
    ideal_rel = np.sort(rel)[::-1][:eff_k]
    idcg = float(np.sum((np.power(2.0, ideal_rel) - 1.0) / discounts))
    if not math.isfinite(idcg) or idcg <= 0.0:
        return 0.0

    # Tie-aware DCG@k: average gains across tied predicted scores (within 1e-12)
    order = np.argsort(-scr, kind="mergesort")
    s_sorted = scr[order]
    gains = np.power(2.0, rel[order]) - 1.0

    i = 0
    while i < m:
        j = i + 1
        while j < m and abs(float(s_sorted[j] - s_sorted[i])) <= 1e-12:
            j += 1
        if j - i > 1:
            gains[i:j] = float(np.mean(gains[i:j]))
        i = j

    dcg = float(np.sum(gains[:eff_k] / discounts))
    val = dcg / idcg
    if not math.isfinite(val):
        return 0.0
    return float(np.clip(val, 0.0, 1.0))


def ndcg_at_k(
    y_true: Any,
    y_score: Any,
    groups: Any = None,
    k: Optional[int] = 10,
) -> float:
    """Compute mean per-query Normalized Discounted Cumulative Gain at ``k``.

    Uses exponential gain ``(2 ** rel - 1) / log2(rank + 2)``, tie-aware
    averaging across tied predicted scores, and handles single-item queries
    and all-zero relevances cleanly.
    """
    eff_k = _validate_k(k)
    rel_all = _coerce_1d_finite_float(y_true, "y_true")
    scr_all = _coerce_1d_finite_float(y_score, "y_score")
    if len(rel_all) != len(scr_all):
        raise ValueError(
            f"y_true ({len(rel_all)}) and y_score ({len(scr_all)}) must have equal length."
        )

    if groups is None:
        return _query_ndcg_at_k(rel_all, scr_all, eff_k)

    groups_arr = validate_groups(groups, n_samples=len(rel_all), min_unique=1)
    codes, _ = pd.factorize(pd.Series(groups_arr, dtype=object), sort=False)
    order = np.argsort(codes, kind="mergesort")
    sorted_codes = codes[order]
    split_pts = np.flatnonzero(np.diff(sorted_codes)) + 1
    query_slices = np.split(order, split_pts)

    ndcgs = [_query_ndcg_at_k(rel_all[idx], scr_all[idx], eff_k) for idx in query_slices]
    return float(np.mean(ndcgs)) if ndcgs else 0.0


class _PairwiseRanker(BaseEstimator, RegressorMixin):
    """Picklable pairwise preference ranker via balanced within-query differences."""

    def __init__(
        self,
        max_pairs_per_query: int = 200,
        C: float = 1.0,
        random_state: Optional[int] = 42,
    ) -> None:
        self.max_pairs_per_query = max_pairs_per_query
        self.C = C
        self.random_state = random_state

    def fit(self, X: Any, y: Any, groups: Any) -> "_PairwiseRanker":
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        codes, _ = pd.factorize(pd.Series(groups, dtype=object), sort=False)
        order = np.argsort(codes, kind="mergesort")
        sorted_codes = codes[order]
        split_pts = np.flatnonzero(np.diff(sorted_codes)) + 1
        query_slices = np.split(order, split_pts)

        rng = np.random.RandomState(self.random_state)
        pos_left: List[int] = []
        pos_right: List[int] = []
        cap = max(1, int(self.max_pairs_per_query))

        for q_idx in query_slices:
            if len(q_idx) < 2:
                continue
            rel_q = y_arr[q_idx]
            ii, jj = np.where(rel_q[:, None] > rel_q[None, :])
            if len(ii) == 0:
                continue
            if len(ii) > cap:
                chosen = rng.choice(len(ii), size=cap, replace=False)
                ii = ii[chosen]
                jj = jj[chosen]
            pos_left.extend(q_idx[ii].tolist())
            pos_right.extend(q_idx[jj].tolist())

        if not pos_left:
            self._is_trivial_ = True
            self.clf_ = None
            return self

        left_arr = np.asarray(pos_left, dtype=np.intp)
        right_arr = np.asarray(pos_right, dtype=np.intp)
        diff_pos = X[left_arr] - X[right_arr]
        diff_neg = -diff_pos

        if sparse.issparse(X):
            X_pairs = sparse.vstack([diff_pos, diff_neg], format="csr")
        else:
            X_pairs = np.vstack([diff_pos, diff_neg])

        n_pos = len(left_arr)
        y_pairs = np.concatenate(
            [np.ones(n_pos, dtype=np.int64), np.zeros(n_pos, dtype=np.int64)]
        )

        clf = LogisticRegression(
            C=self.C, max_iter=1000, solver="lbfgs", random_state=self.random_state
        )
        clf.fit(X_pairs, y_pairs)
        self._is_trivial_ = False
        self.clf_ = clf
        return self

    def predict(self, X: Any) -> np.ndarray:
        n_samples = X.shape[0]
        if getattr(self, "_is_trivial_", False) or self.clf_ is None:
            return np.zeros(n_samples, dtype=np.float64)
        scores = np.asarray(self.clf_.decision_function(X), dtype=np.float64).ravel()
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


class _QueryCenteredPointwiseRanker(BaseEstimator, RegressorMixin):
    """Picklable pointwise ranker trained on within-query centered relevance targets."""

    def __init__(self, estimator: Any, require_dense: bool = False) -> None:
        self.estimator = estimator
        self.require_dense = require_dense

    def fit(self, X: Any, y: Any, groups: Any) -> "_QueryCenteredPointwiseRanker":
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        codes, _ = pd.factorize(pd.Series(groups, dtype=object), sort=False)
        y_centered = y_arr.copy()
        for c in np.unique(codes):
            mask = codes == c
            y_centered[mask] = y_arr[mask] - float(np.mean(y_arr[mask]))

        X_fit = X.toarray() if (self.require_dense and sparse.issparse(X)) else X
        est = clone(self.estimator)
        est.fit(X_fit, y_centered)
        self.estimator_ = est
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_eval = X.toarray() if (self.require_dense and sparse.issparse(X)) else X
        scores = np.asarray(self.estimator_.predict(X_eval), dtype=np.float64).ravel()
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


class _OptionalBoosterRanker(BaseEstimator, RegressorMixin):
    """Picklable wrapper for optional LightGBM / XGBoost learning-to-rank models."""

    def __init__(self, backend: str = "lightgbm", random_state: Optional[int] = 42) -> None:
        self.backend = backend
        self.random_state = random_state

    def fit(self, X: Any, y: Any, groups: Any) -> "_OptionalBoosterRanker":
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        codes, _ = pd.factorize(pd.Series(groups, dtype=object), sort=False)
        order = np.argsort(codes, kind="mergesort")
        sorted_codes = codes[order]
        _, group_counts = np.unique(sorted_codes, return_counts=True)
        X_ord = X[order]
        y_ord = y_arr[order]

        if self.backend == "lightgbm":
            import lightgbm as lgb

            est = lgb.LGBMRanker(
                n_estimators=100,
                random_state=self.random_state,
                verbose=-1,
            )
            est.fit(X_ord, np.rint(np.maximum(y_ord, 0.0)).astype(np.int64), group=group_counts.tolist())
        elif self.backend == "xgboost":
            import xgboost as xgb

            est = xgb.XGBRanker(
                n_estimators=100,
                random_state=self.random_state,
                tree_method="hist",
            )
            est.fit(X_ord, y_ord, group=group_counts.tolist())
        else:
            raise ValueError(f"Unknown booster backend: {self.backend!r}")

        self.estimator_ = est
        return self

    def predict(self, X: Any) -> np.ndarray:
        scores = np.asarray(self.estimator_.predict(X), dtype=np.float64).ravel()
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


class ACDSORanker(BaseEstimator, RegressorMixin):
    """Query-grouped Learning-to-Rank estimator with GroupKFold CV and NLP text branch.

    Parameters
    ----------
    algorithm : str, default="auto"
        One of ``"auto"``, ``"pairwise"``, ``"pointwise"``, ``"pointwise_ridge"``,
        ``"pointwise_gbr"``, ``"lgbm_ranker"``, ``"xgb_ranker"``.
    k : int, default=10
        Cutoff ``k`` for CV selection and default ``score()`` NDCG@k.
    cv : int or splitter, default=5
        Number of folds or CV splitter object.
    random_state : int, optional, default=42
        Random seed for reproducibility.
    text_cols : list of str, optional
        Explicit free-text columns. When ``None``, auto-detected via
        ``split_columns(..., detect_text=True)``.
    text_method : str, default="tfidf"
        Text featurization mode passed to ``TextFeaturizer``.
    verbose : int, default=0
        Verbosity level.
    """

    def __init__(
        self,
        algorithm: str = "auto",
        k: int = 10,
        cv: Any = 5,
        random_state: Optional[int] = 42,
        text_cols: Optional[List[str]] = None,
        text_method: str = "tfidf",
        verbose: int = 0,
    ) -> None:
        self.algorithm = algorithm
        self.k = k
        self.cv = cv
        self.random_state = random_state
        self.text_cols = text_cols
        self.text_method = text_method
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose and int(self.verbose) > 0:
            print(f"[ACDSORanker] {msg}")

    def _validate_hyperparameters(self) -> Tuple[str, int]:
        if not isinstance(self.algorithm, str) or self.algorithm not in _VALID_ALGORITHMS:
            raise ValueError(
                f"Invalid algorithm={self.algorithm!r}. Must be one of {_VALID_ALGORITHMS}."
            )
        eff_k = _validate_k(self.k)
        if eff_k is None:
            raise ValueError("ACDSORanker k cannot be None; specify an integer >= 1.")
        return self.algorithm, eff_k

    def _coerce_input(self, X: Any, *, is_fit: bool) -> pd.DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None.")

        if isinstance(X, pd.DataFrame):
            n_rows, n_cols = X.shape
            if is_fit:
                if n_rows < 2 or n_cols < 1:
                    raise ValueError(
                        f"ACDSORanker.fit requires >= 2 rows and >= 1 column, got {X.shape}."
                    )
                if X.columns.has_duplicates:
                    raw_cols = list(X.columns)
                    internal_cols = [f"__acdso_col_{i}" for i in range(n_cols)]
                    self._has_dup_cols_ = True
                    self._raw_raw_cols_ = raw_cols
                    self._raw_feature_names_ = raw_cols
                    self._internal_cols_ = internal_cols
                    df = X.copy()
                    df.columns = internal_cols
                else:
                    self._has_dup_cols_ = False
                    self._raw_raw_cols_ = list(X.columns)
                    self._raw_feature_names_ = list(X.columns)
                    self._internal_cols_ = list(X.columns)
                    df = X.copy()
                self._input_was_df_ = True
                self.n_features_in_ = int(n_cols)
                if all(isinstance(c, str) for c in X.columns):
                    self.feature_names_in_ = np.asarray(X.columns, dtype=object)
                elif hasattr(self, "feature_names_in_"):
                    delattr(self, "feature_names_in_")
            else:
                if self._has_dup_cols_:
                    if n_cols != len(self._raw_raw_cols_) or list(X.columns) != self._raw_raw_cols_:
                        raise ValueError(
                            f"Expected columns {self._raw_raw_cols_}, got {list(X.columns)}."
                        )
                    df = X.copy()
                    df.columns = self._internal_cols_
                elif self._input_was_df_:
                    if X.columns.has_duplicates:
                        raise ValueError(
                            "Duplicate columns found at inference when training data had unique columns."
                        )
                    missing = [c for c in self._raw_feature_names_ if c not in X.columns]
                    if missing:
                        raise ValueError(f"Missing expected columns at inference: {missing}")
                    df = X[self._raw_feature_names_].copy()
                else:
                    if n_cols != self.n_features_in_:
                        raise ValueError(f"Expected {self.n_features_in_} features, got {n_cols}.")
                    df = X.copy()
                    df.columns = self._internal_cols_
            return sanitize_features(df)

        arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input X, got {arr.ndim}D.")
        n_rows, n_cols = arr.shape
        if is_fit:
            if n_rows < 2 or n_cols < 1:
                raise ValueError(
                    f"ACDSORanker.fit requires >= 2 rows and >= 1 column, got {arr.shape}."
                )
            cols = [f"f_{i}" for i in range(n_cols)]
            self._has_dup_cols_ = False
            self._input_was_df_ = False
            self._raw_raw_cols_ = cols
            self._raw_feature_names_ = cols
            self._internal_cols_ = cols
            self.n_features_in_ = int(n_cols)
            if hasattr(self, "feature_names_in_"):
                delattr(self, "feature_names_in_")
        else:
            if n_cols != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, got {n_cols}.")
            cols = self._internal_cols_

        df = pd.DataFrame(arr, columns=cols)
        return sanitize_features(df)

    @staticmethod
    def _normalize_matrix(X_proc: Any, *, force_dense: bool = False) -> Any:
        if sparse.issparse(X_proc):
            n_rows, n_cols = X_proc.shape
            if force_dense or n_rows * n_cols <= 1_000_000:
                arr = X_proc.toarray().astype(np.float64, copy=False)
                if arr.size > 0:
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                return arr
            csr = X_proc.tocsr()
            if csr.data.size > 0 and not np.all(np.isfinite(csr.data)):
                csr = csr.copy()
                csr.data = np.nan_to_num(csr.data, nan=0.0, posinf=0.0, neginf=0.0)
            return csr
        arr = np.asarray(X_proc, dtype=np.float64)
        if arr.size > 0:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _resolve_columns(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
        if self.text_cols is not None:
            if self._has_dup_cols_:
                raw_cols = self._raw_raw_cols_
                missing_text = [c for c in self.text_cols if c not in raw_cols]
                if missing_text:
                    raise ValueError(f"text_cols not found in input DataFrame: {missing_text}")
                req_set = set(self.text_cols)
                eff_text_cols = [
                    internal
                    for orig, internal in zip(raw_cols, self._internal_cols_)
                    if orig in req_set
                ]
            else:
                missing_text = [c for c in self.text_cols if c not in df.columns]
                if missing_text:
                    raise ValueError(f"text_cols not found in input DataFrame: {missing_text}")
                eff_text_cols = list(self.text_cols)
        else:
            eff_text_cols = []

        active_cols = [c for c in df.columns if not bool(df[c].isna().all())]
        if not active_cols:
            raise ValueError("No valid features remain after preprocessing (e.g. all-NaN columns).")
        self._active_cols_ = active_cols
        df_active = df[active_cols]

        if self.text_cols is not None:
            active_set = set(active_cols)
            eff_text_cols = [c for c in eff_text_cols if c in active_set]
            num_cols, cat_cols = split_columns(df_active, detect_text=False)
            text_set = set(eff_text_cols)
            num_cols = [c for c in num_cols if c not in text_set]
            cat_cols = [c for c in cat_cols if c not in text_set]
        else:
            num_cols, cat_cols, eff_text_cols = split_columns(df_active, detect_text=True)

        return df_active, num_cols, cat_cols, eff_text_cols

    def _candidate_list(self, algo: str) -> List[str]:
        if algo == "pairwise":
            return ["pairwise"]
        if algo == "pointwise":
            return ["pointwise_ridge", "pointwise_gbr"]
        if algo in ("pointwise_ridge", "pointwise_gbr"):
            return [algo]
        if algo == "lgbm_ranker":
            try:
                import lightgbm  # noqa: F401
            except ImportError as exc:
                raise ImportError("algorithm='lgbm_ranker' requires lightgbm to be installed.") from exc
            return ["lgbm_ranker"]
        if algo == "xgb_ranker":
            try:
                import xgboost  # noqa: F401
            except ImportError as exc:
                raise ImportError("algorithm='xgb_ranker' requires xgboost to be installed.") from exc
            return ["xgb_ranker"]

        cands = ["pairwise", "pointwise_ridge", "pointwise_gbr"]
        try:
            import lightgbm  # noqa: F401

            cands.append("lgbm_ranker")
        except ImportError:
            pass
        try:
            import xgboost  # noqa: F401

            cands.append("xgb_ranker")
        except ImportError:
            pass
        return cands

    def _build_candidate(self, name: str) -> Any:
        rs = self.random_state
        if name == "pairwise":
            return _PairwiseRanker(max_pairs_per_query=200, C=1.0, random_state=rs)
        if name == "pointwise_ridge":
            return _QueryCenteredPointwiseRanker(Ridge(alpha=1.0), require_dense=False)
        if name == "pointwise_gbr":
            return _QueryCenteredPointwiseRanker(
                HistGradientBoostingRegressor(max_iter=100, random_state=rs),
                require_dense=True,
            )
        if name == "lgbm_ranker":
            return _OptionalBoosterRanker(backend="lightgbm", random_state=rs)
        if name == "xgb_ranker":
            return _OptionalBoosterRanker(backend="xgboost", random_state=rs)
        raise ValueError(f"Unsupported ranking candidate: {name!r}")

    def fit(self, X: Any, y: Any, groups: Any = None) -> "ACDSORanker":
        """Fit query-grouped Learning-to-Rank estimator using GroupKFold CV."""
        algo, eff_k = self._validate_hyperparameters()
        if groups is None:
            raise ValueError("ACDSORanker.fit requires query groups (groups cannot be None).")

        df = self._coerce_input(X, is_fit=True)
        y_arr = _coerce_1d_finite_float(y, "y")
        if len(y_arr) != len(df):
            raise ValueError(
                f"X ({len(df)}) and y ({len(y_arr)}) must have equal length."
            )
        groups_arr = validate_groups(groups, n_samples=len(df), min_unique=2)
        n_queries = int(len(pd.unique(pd.Series(groups_arr, dtype=object))))

        df_active, num_cols, cat_cols, eff_text_cols = self._resolve_columns(df)
        self.text_cols_ = list(eff_text_cols)

        def _make_pre() -> Any:
            return build_preprocessor(
                numeric_cols=num_cols,
                categorical_cols=cat_cols,
                sparse_threshold=0.3,
                text_cols=eff_text_cols or None,
                text_method=self.text_method,
            )

        # Always pass cv_strategy="group_kfold" so integer relevance grades never trigger stratification
        splitter = build_cv_splitter(
            cv=self.cv,
            y=y_arr,
            groups=groups_arr,
            task="ranking",
            cv_strategy="group_kfold",
            random_state=self.random_state,
        )
        if not splitter_supports_groups(splitter):
            raise ValueError(
                "ACDSORanker requires a group-aware CV splitter supporting query groups."
            )
        eff_cv_name = resolve_cv_strategy_name(splitter)
        splits = iter_cv_splits(
            splitter, df_active, y_arr, groups_arr, validate_groups_leakage=True
        )
        validate_no_group_leakage(splits, groups_arr)

        fold_matrices: List[Tuple[Any, np.ndarray, np.ndarray, Any, np.ndarray, np.ndarray]] = []
        for tr_idx, va_idx in splits:
            fold_pre = _make_pre()
            X_tr_raw = fold_pre.fit_transform(df_active.iloc[tr_idx])
            if X_tr_raw.shape[1] == 0:
                raise ValueError("No valid features remain after preprocessing in CV fold.")
            X_va_raw = fold_pre.transform(df_active.iloc[va_idx])
            fold_matrices.append(
                (
                    X_tr_raw,
                    y_arr[tr_idx],
                    groups_arr[tr_idx],
                    X_va_raw,
                    y_arr[va_idx],
                    groups_arr[va_idx],
                )
            )

        cand_names = self._candidate_list(algo)
        tie_order = {
            "lgbm_ranker": 5,
            "xgb_ranker": 4,
            "pairwise": 3,
            "pointwise_gbr": 2,
            "pointwise_ridge": 1,
        }

        candidate_summaries: List[Dict[str, Any]] = []
        for name in cand_names:
            force_dense = name == "pointwise_gbr"
            fold_scores: List[float] = []
            try:
                for X_tr_raw, y_tr, g_tr, X_va_raw, y_va, g_va in fold_matrices:
                    X_tr = self._normalize_matrix(X_tr_raw, force_dense=force_dense)
                    X_va = self._normalize_matrix(X_va_raw, force_dense=force_dense)
                    if force_dense and X_tr.shape[1] > 1024:
                        raise ValueError("Too many features for dense pointwise_gbr candidate.")
                    est = self._build_candidate(name)
                    est.fit(X_tr, y_tr, g_tr)
                    preds_va = est.predict(X_va)
                    fold_scores.append(ndcg_at_k(y_va, preds_va, groups=g_va, k=eff_k))

                candidate_summaries.append(
                    {
                        "algorithm": name,
                        "cv_ndcg_at_k": float(np.mean(fold_scores)),
                        "fold_scores": fold_scores,
                    }
                )
            except Exception as exc:
                self._log(f"Candidate {name} failed during CV: {exc}")

        if not candidate_summaries:
            raise ValueError("All ranking candidate algorithms failed during cross-validation.")

        best_summary = max(
            candidate_summaries,
            key=lambda item: (float(item["cv_ndcg_at_k"]), tie_order.get(item["algorithm"], 0)),
        )
        best_name = str(best_summary["algorithm"])

        self.preprocessor_ = _make_pre()
        X_full_raw = self.preprocessor_.fit_transform(df_active)
        if X_full_raw.shape[1] == 0:
            raise ValueError("No valid features remain after preprocessing.")
        force_dense_best = best_name == "pointwise_gbr"
        X_full = self._normalize_matrix(X_full_raw, force_dense=force_dense_best)

        best_est = self._build_candidate(best_name)
        best_est.fit(X_full, y_arr, groups_arr)

        self.best_algorithm_ = best_name
        self.best_estimator_ = best_est
        self._force_dense_ = force_dense_best
        self.candidates_ = _sanitize_json_obj(candidate_summaries)
        self.audit_trail_ = _sanitize_json_obj(
            {
                "algorithm_requested": self.algorithm,
                "best_algorithm": self.best_algorithm_,
                "k": eff_k,
                "cv_strategy": eff_cv_name,
                "n_splits": len(splits),
                "n_samples": int(len(df_active)),
                "n_queries": n_queries,
                "n_features_preprocessed": int(X_full.shape[1]),
                "text_cols": list(eff_text_cols),
                "selected_candidate": best_summary,
                "candidates": candidate_summaries,
            }
        )
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "best_estimator_") or self.best_estimator_ is None:
            raise RuntimeError("ACDSORanker is not fitted. Call fit() first.")

    def _prepare_eval_matrix(self, X: Any) -> Any:
        self._require_fitted()
        df = self._coerce_input(X, is_fit=False)
        df_active = df[self._active_cols_]
        X_proc = self.preprocessor_.transform(df_active)
        return self._normalize_matrix(
            X_proc, force_dense=getattr(self, "_force_dense_", False)
        )

    def predict(self, X: Any) -> np.ndarray:
        """Predict 1D relevance scores for ``X``."""
        X_mat = self._prepare_eval_matrix(X)
        scores = np.asarray(self.best_estimator_.predict(X_mat), dtype=np.float64).ravel()
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def rank(self, X: Any, groups: Any = None) -> np.ndarray:
        """Return 0-based within-query integer ranks (0 = highest predicted score)."""
        scores = self.predict(X)
        n = len(scores)
        if groups is None:
            order = np.argsort(-scores, kind="mergesort")
            ranks = np.empty(n, dtype=np.int64)
            ranks[order] = np.arange(n, dtype=np.int64)
            return ranks

        groups_arr = validate_groups(groups, n_samples=n, min_unique=1)
        codes, _ = pd.factorize(pd.Series(groups_arr, dtype=object), sort=False)
        ranks = np.empty(n, dtype=np.int64)
        for c in np.unique(codes):
            idx = np.flatnonzero(codes == c)
            q_order = np.argsort(-scores[idx], kind="mergesort")
            ranks[idx[q_order]] = np.arange(len(idx), dtype=np.int64)
        return ranks

    def score(self, X: Any, y: Any, groups: Any = None, sample_weight: Any = None) -> float:
        """Return mean per-query ``ndcg_at_k`` at ``self.k``."""
        self._require_fitted()
        preds = self.predict(X)
        return ndcg_at_k(y, preds, groups=groups, k=self.k)

    def evaluate(
        self,
        X: Any,
        y: Any,
        groups: Any = None,
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate ranking performance returning NDCG@k, NDCG@5, NDCG@10, MRR, and Spearman."""
        self._require_fitted()
        eff_k = _validate_k(k if k is not None else self.k)
        if eff_k is None:
            eff_k = 10
        preds = self.predict(X)
        y_arr = _coerce_1d_finite_float(y, "y")
        if len(y_arr) != len(preds):
            raise ValueError(
                f"X ({len(preds)}) and y ({len(y_arr)}) must have equal length."
            )

        if groups is None:
            groups_arr = np.zeros(len(y_arr), dtype=np.intp)
        else:
            groups_arr = validate_groups(groups, n_samples=len(y_arr), min_unique=1)

        codes, _ = pd.factorize(pd.Series(groups_arr, dtype=object), sort=False)
        order = np.argsort(codes, kind="mergesort")
        sorted_codes = codes[order]
        split_pts = np.flatnonzero(np.diff(sorted_codes)) + 1
        query_slices = np.split(order, split_pts)

        ndcg_eff = ndcg_at_k(y_arr, preds, groups=groups_arr, k=eff_k)
        ndcg_5 = ndcg_at_k(y_arr, preds, groups=groups_arr, k=5)
        ndcg_10 = ndcg_at_k(y_arr, preds, groups=groups_arr, k=10)

        mrr_vals: List[float] = []
        spearman_vals: List[float] = []
        for idx in query_slices:
            rel_q = y_arr[idx]
            scr_q = preds[idx]
            q_order = np.argsort(-scr_q, kind="mergesort")
            rel_sorted = rel_q[q_order]
            pos_hits = np.flatnonzero(rel_sorted > 0.0)
            mrr_vals.append(1.0 / float(pos_hits[0] + 1) if pos_hits.size > 0 else 0.0)

            if len(rel_q) >= 2 and np.ptp(rel_q) > 0.0 and np.ptp(scr_q) > 0.0:
                sp = stats.spearmanr(rel_q, scr_q).statistic
                if sp is not None and math.isfinite(float(sp)):
                    spearman_vals.append(float(sp))

        metrics: Dict[str, float] = {
            "ndcg_at_k": float(ndcg_eff),
            f"ndcg@{eff_k}": float(ndcg_eff),
            "ndcg_at_5": float(ndcg_5),
            "ndcg_at_10": float(ndcg_10),
            "mrr": float(np.mean(mrr_vals)) if mrr_vals else 0.0,
            "mean_spearman": float(np.mean(spearman_vals)) if spearman_vals else 0.0,
        }
        return metrics
