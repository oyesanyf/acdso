"""
ACDSO - Single-Machine Out-of-Core & Chunked Scale Engine
=========================================================

Provides streaming file iteration, deterministic Algorithm R reservoir sampling,
incremental ``partial_fit`` learners (SGD, MiniBatchKMeans, IncrementalPCA) with
stateless hashing featurizers and Welford running statistics, bounded subsampling
fallbacks, and streaming file prediction.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.pipeline import Pipeline

from acdso.cli.io import _widen_chunk_dtype, iter_file_chunks, sample_file_bounded
from acdso.nlp import detect_text_columns
from acdso.utils import _datetime_series_to_epoch, build_preprocessor, infer_target_type, split_columns

__all__ = ["iter_file_chunks", "reservoir_sample_file", "ACDSOOutOfCoreTrainer"]

_VALID_TASKS = frozenset({"auto", "classification", "regression", "clustering", "dim_reduction", "pca"})
_VALID_MODES = frozenset({"auto", "partial_fit", "chunked", "subsample"})


def _sanitize_json_obj(obj: Any) -> Any:
    """Recursively sanitize objects for strict RFC 8259 JSON serialization."""
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_obj(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize_json_obj(v) for v in obj.tolist()]
    return str(obj)


def reservoir_sample_file(
    path: Union[str, os.PathLike[str]],
    max_rows: int = 100_000,
    chunk_size: int = 50_000,
    random_state: Optional[int] = 42,
    usecols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Stream a file in bounded chunks and return ``(sample_df, total_rows)``.

    Uses Vitter's Algorithm R so memory remains bounded by ``max_rows + chunk_size``
    regardless of total dataset size.
    """
    sample_df, total_rows = sample_file_bounded(
        filepath=path,
        max_rows=max_rows,
        chunk_size=chunk_size,
        random_state=random_state,
        usecols=usecols,
        return_total=True,
    )
    return sample_df, int(total_rows)


class _StreamingNumericScaler:
    """Online Welford imputer and standard scaler for numeric feature columns."""

    def __init__(self, columns: List[Any]) -> None:
        self.columns = list(columns)
        n_cols = len(self.columns)
        self.count_ = np.zeros(n_cols, dtype=np.int64)
        self.mean_ = np.zeros(n_cols, dtype=np.float64)
        self.M2_ = np.zeros(n_cols, dtype=np.float64)

    def _extract_array(self, df: pd.DataFrame) -> np.ndarray:
        n_rows = len(df)
        n_cols = len(self.columns)
        if n_cols == 0:
            return np.empty((n_rows, 0), dtype=np.float64)
        arr = np.empty((n_rows, n_cols), dtype=np.float64)
        for j, col in enumerate(self.columns):
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s.dtype):
                vals = _datetime_series_to_epoch(s)
            elif pd.api.types.is_bool_dtype(s.dtype):
                vals = s.to_numpy(dtype=np.float64, na_value=np.nan)
            else:
                vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
            vals = np.where(np.isfinite(vals), vals, np.nan)
            arr[:, j] = vals
        return arr

    def partial_fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        arr = self._extract_array(df)
        if arr.shape[1] == 0:
            return arr
        for j in range(arr.shape[1]):
            col_vals = arr[:, j]
            valid = np.isfinite(col_vals)
            n_b = int(np.sum(valid))
            if n_b == 0:
                continue
            vals = col_vals[valid]
            mean_b = float(np.mean(vals))
            m2_b = float(np.sum((vals - mean_b) ** 2))
            n_a = int(self.count_[j])
            if n_a == 0:
                self.count_[j] = n_b
                self.mean_[j] = mean_b
                self.M2_[j] = m2_b
            else:
                n_ab = n_a + n_b
                delta = mean_b - self.mean_[j]
                self.mean_[j] = self.mean_[j] + delta * (n_b / n_ab)
                self.M2_[j] = self.M2_[j] + m2_b + (delta ** 2) * (n_a * n_b / n_ab)
                self.count_[j] = n_ab
        return self._scale_array(arr)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        arr = self._extract_array(df)
        if arr.shape[1] == 0:
            return arr
        return self._scale_array(arr)

    def _scale_array(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        for j in range(out.shape[1]):
            nan_mask = ~np.isfinite(out[:, j])
            if nan_mask.any():
                out[nan_mask, j] = self.mean_[j]
            n_c = int(self.count_[j])
            if n_c > 1:
                std = math.sqrt(max(0.0, float(self.M2_[j]) / float(n_c - 1)))
            else:
                std = 1.0
            if std <= 1e-12:
                std = 1.0
            out[:, j] = (out[:, j] - self.mean_[j]) / std
        return out


class _StreamingFeatureEncoder:
    """Stateless-vocabulary streaming encoder for numeric, categorical, and text columns."""

    def __init__(
        self,
        n_hash_features: int = 2**14,
        cat_hash_features: int = 1024,
    ) -> None:
        self.n_hash_features = int(n_hash_features)
        self.cat_hash_features = min(int(cat_hash_features), max(16, int(n_hash_features)))
        self.feature_columns_: List[Any] = []
        self.numeric_cols_: List[Any] = []
        self.categorical_cols_: List[Any] = []
        self.text_cols_: List[Any] = []
        self.numeric_scaler_: Optional[_StreamingNumericScaler] = None
        self.cat_hasher_: Optional[FeatureHasher] = None
        self.text_vectorizer_: Optional[HashingVectorizer] = None

    def fit_warmup(
        self,
        df_features: pd.DataFrame,
        text_cols: Optional[List[str]] = None,
    ) -> None:
        self.feature_columns_ = list(df_features.columns)
        if len(self.feature_columns_) == 0:
            raise ValueError("Input features cannot have 0 columns.")

        num_cols = [
            c
            for c in self.feature_columns_
            if pd.api.types.is_numeric_dtype(df_features[c])
            or pd.api.types.is_datetime64_any_dtype(df_features[c])
        ]
        if text_cols is not None:
            missing = [c for c in text_cols if c not in df_features.columns]
            if missing:
                raise ValueError(f"text_cols not found in input features: {missing}")
            eff_text = list(text_cols)
            text_set = set(eff_text)
            num_cols = [c for c in num_cols if c not in text_set]
        else:
            eff_text = detect_text_columns(df_features, exclude_cols=num_cols)
            text_set = set(eff_text)

        excluded = set(num_cols) | text_set
        cat_cols = [c for c in self.feature_columns_ if c not in excluded]

        self.numeric_cols_ = num_cols
        self.categorical_cols_ = cat_cols
        self.text_cols_ = eff_text

        self.numeric_scaler_ = _StreamingNumericScaler(num_cols) if num_cols else None
        self.cat_hasher_ = (
            FeatureHasher(n_features=self.cat_hash_features, input_type="dict", alternate_sign=False)
            if cat_cols
            else None
        )
        self.text_vectorizer_ = (
            HashingVectorizer(
                n_features=self.n_hash_features,
                ngram_range=(1, 2),
                alternate_sign=False,
                norm="l2",
            )
            if eff_text
            else None
        )

    def _transform_blocks(
        self,
        df_features: pd.DataFrame,
        update_stats: bool,
        dense: bool = False,
    ) -> Union[sparse.csr_matrix, np.ndarray]:
        missing = [c for c in self.feature_columns_ if c not in df_features.columns]
        if missing:
            raise ValueError(f"Input is missing expected feature columns: {missing}")
        df_aligned = df_features[self.feature_columns_]

        blocks: List[Any] = []
        if self.numeric_scaler_ is not None:
            if update_stats:
                num_arr = self.numeric_scaler_.partial_fit_transform(df_aligned)
            else:
                num_arr = self.numeric_scaler_.transform(df_aligned)
            blocks.append(num_arr if dense else sparse.csr_matrix(num_arr))

        if self.cat_hasher_ is not None:
            cat_dicts: List[Dict[str, str]] = []
            sub_cat = df_aligned[self.categorical_cols_]
            col_strs = [str(c) for c in self.categorical_cols_]
            for row in sub_cat.itertuples(index=False, name=None):
                d: Dict[str, str] = {}
                for cname, val in zip(col_strs, row):
                    if val is None or val is pd.NA or val is pd.NaT:
                        continue
                    if isinstance(val, (float, np.floating)) and not math.isfinite(float(val)):
                        continue
                    d[cname] = str(val)
                cat_dicts.append(d)
            cat_mat = self.cat_hasher_.transform(cat_dicts)
            blocks.append(cat_mat.toarray() if dense else cat_mat)

        if self.text_vectorizer_ is not None:
            for tcol in self.text_cols_:
                s = df_aligned[tcol]
                docs = s.astype(object).where(s.notna(), "").astype(str).tolist()
                txt_mat = self.text_vectorizer_.transform(docs)
                blocks.append(txt_mat.toarray() if dense else txt_mat)

        if not blocks:
            raise ValueError("No feature blocks produced by _StreamingFeatureEncoder.")

        if dense:
            if len(blocks) == 1:
                return np.asarray(blocks[0], dtype=np.float64)
            return np.hstack(blocks)

        if len(blocks) == 1:
            return blocks[0].tocsr() if sparse.issparse(blocks[0]) else sparse.csr_matrix(blocks[0])
        return sparse.hstack(blocks, format="csr")

    def partial_fit_transform(
        self,
        df_features: pd.DataFrame,
        dense: bool = False,
    ) -> Union[sparse.csr_matrix, np.ndarray]:
        return self._transform_blocks(df_features, update_stats=True, dense=dense)

    def transform(
        self,
        df_features: pd.DataFrame,
        dense: bool = False,
    ) -> Union[sparse.csr_matrix, np.ndarray]:
        return self._transform_blocks(df_features, update_stats=False, dense=dense)


def _coerce_to_dataframe(
    X: Any,
    expected_columns: Optional[List[Any]] = None,
    drop_col: Optional[str] = None,
) -> pd.DataFrame:
    """Coerce DataFrame or array-like input into a DataFrame preserving column names."""
    if isinstance(X, pd.DataFrame):
        df = X
        if drop_col is not None and drop_col in df.columns:
            df = df.drop(columns=[drop_col])
        return df

    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D input X, got {arr.ndim}D array.")
    if expected_columns is not None:
        if arr.shape[1] != len(expected_columns):
            raise ValueError(
                f"Expected {len(expected_columns)} features, got {arr.shape[1]}."
            )
        cols = list(expected_columns)
    else:
        cols = [f"x{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _clean_supervised_chunk(
    chunk: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract feature DataFrame and finite/non-null 1D target array for a chunk."""
    if target_col not in chunk.columns:
        raise ValueError(f"Target column '{target_col}' not found in chunk columns.")
    y_s = chunk[target_col]
    valid_mask = y_s.notna()
    if pd.api.types.is_numeric_dtype(y_s) and not pd.api.types.is_bool_dtype(y_s):
        vals = y_s.to_numpy(dtype=np.float64, na_value=np.nan)
        valid_mask = valid_mask & np.isfinite(vals)
    elif y_s.dtype == object:
        is_finite_obj = [
            not (isinstance(v, (float, np.floating)) and not math.isfinite(float(v)))
            for v in y_s.tolist()
        ]
        valid_mask = valid_mask & pd.Series(is_finite_obj, index=chunk.index)

    if not bool(valid_mask.all()):
        chunk = chunk.loc[valid_mask]
        y_s = chunk[target_col]

    X_chunk = chunk.drop(columns=[target_col])
    return X_chunk, y_s.to_numpy()


def _expand_sgd_classes(
    est: SGDClassifier,
    new_classes: np.ndarray,
) -> SGDClassifier:
    """Expand an already-fitted SGDClassifier when unseen classes appear mid-stream."""
    old_classes = np.asarray(est.classes_)
    merged = np.unique(np.concatenate([old_classes, np.asarray(new_classes)]))
    if len(merged) == len(old_classes) and np.array_equal(merged, old_classes):
        return est

    n_features = est.coef_.shape[1]
    if len(old_classes) == 2 and est.coef_.shape[0] == 1:
        # Binary SGDClassifier stores a single hyperplane for old_classes[1] vs old_classes[0]
        old_coef = np.vstack([-est.coef_[0], est.coef_[0]])
        old_intercept = np.array([-float(est.intercept_[0]), float(est.intercept_[0])], dtype=np.float64)
    else:
        old_coef = est.coef_.copy()
        old_intercept = est.intercept_.copy()

    new_coef = np.zeros((len(merged), n_features), dtype=np.float64)
    new_intercept = np.zeros(len(merged), dtype=np.float64)
    old_pos = {c: idx for idx, c in enumerate(old_classes.tolist())}
    for new_idx, cls in enumerate(merged.tolist()):
        if cls in old_pos:
            oi = old_pos[cls]
            new_coef[new_idx] = old_coef[oi]
            new_intercept[new_idx] = old_intercept[oi]

    est.classes_ = merged
    est.coef_ = new_coef
    est.intercept_ = new_intercept
    return est


class ACDSOOutOfCoreTrainer(BaseEstimator):
    """Single-machine out-of-core and bounded-subsampling trainer.

    Supports incremental ``partial_fit`` learners (``SGDClassifier``, ``SGDRegressor``,
    ``MiniBatchKMeans``, ``IncrementalPCA``) paired with ``HashingVectorizer``,
    ``FeatureHasher``, and online Welford numeric scaling, as well as deterministic
    Algorithm R bounded subsampling (``mode="subsample"``).
    """

    def __init__(
        self,
        task: str = "classification",
        algorithm: str = "auto",
        mode: str = "auto",
        chunk_size: int = 50_000,
        max_in_memory_rows: int = 200_000,
        n_hash_features: int = 2**14,
        n_clusters: int = 5,
        n_components: Optional[int] = None,
        random_state: Optional[int] = 42,
        verbose: int = 0,
    ) -> None:
        self.task = task
        self.algorithm = algorithm
        self.mode = mode
        self.chunk_size = chunk_size
        self.max_in_memory_rows = max_in_memory_rows
        self.n_hash_features = n_hash_features
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose

    def _validate_params(self, mode_override: Optional[str] = None) -> Tuple[str, str]:
        task_norm = str(self.task).strip().lower()
        if task_norm not in _VALID_TASKS:
            raise ValueError(f"Invalid task {self.task!r}; expected one of {sorted(_VALID_TASKS)}.")
        if task_norm == "pca":
            task_norm = "dim_reduction"

        raw_mode = self.mode if mode_override is None else mode_override
        mode_norm = str(raw_mode).strip().lower()
        if mode_norm not in _VALID_MODES:
            raise ValueError(f"Invalid mode {raw_mode!r}; expected one of {sorted(_VALID_MODES)}.")

        algo_norm = str(self.algorithm).strip().lower()
        if mode_norm == "auto":
            eff_mode = "subsample" if algo_norm == "subsample" else "partial_fit"
        elif mode_norm == "chunked":
            eff_mode = "partial_fit"
        else:
            eff_mode = mode_norm

        for pname, pval in (
            ("chunk_size", self.chunk_size),
            ("max_in_memory_rows", self.max_in_memory_rows),
            ("n_hash_features", self.n_hash_features),
            ("n_clusters", self.n_clusters),
        ):
            if isinstance(pval, bool) or int(pval) < 1:
                raise ValueError(f"{pname} must be a positive integer, got {pval!r}.")

        if self.n_components is not None:
            if isinstance(self.n_components, bool) or int(self.n_components) < 1:
                raise ValueError(f"n_components must be a positive integer, got {self.n_components!r}.")

        return task_norm, eff_mode

    def _resolve_auto_task(
        self,
        sample_y: Optional[np.ndarray],
        target_col: Optional[str],
    ) -> str:
        if target_col is None or sample_y is None:
            return "clustering"
        inferred = infer_target_type(sample_y)
        return "regression" if inferred == "regression" else "classification"

    def _fit_subsample_df(
        self,
        sample_df: pd.DataFrame,
        total_rows: int,
        n_chunks: int,
        task: str,
        target_col: Optional[str],
        text_cols: Optional[List[str]],
    ) -> "ACDSOOutOfCoreTrainer":
        if total_rows == 0 or len(sample_df) == 0:
            raise ValueError("Cannot fit ACDSOOutOfCoreTrainer on an empty dataset (0 rows).")

        if task in ("classification", "regression"):
            if not target_col:
                raise ValueError(f"target_col is required for task={task!r}.")
            X_df, y_arr = _clean_supervised_chunk(sample_df, target_col)
            if len(y_arr) == 0:
                raise ValueError("No valid rows remain after dropping missing/non-finite targets.")
        else:
            if target_col and target_col in sample_df.columns:
                X_df = sample_df.drop(columns=[target_col])
            else:
                X_df = sample_df
            y_arr = None

        if task == "auto":
            task = self._resolve_auto_task(y_arr, target_col)

        feature_cols = list(X_df.columns)
        if not feature_cols:
            raise ValueError("Input features cannot have 0 columns.")

        if text_cols is not None:
            missing = [c for c in text_cols if c not in X_df.columns]
            if missing:
                raise ValueError(f"text_cols not found in input features: {missing}")
            eff_text = list(text_cols)
            num_cols, cat_cols = split_columns(X_df, detect_text=False)
            text_set = set(eff_text)
            num_cols = [c for c in num_cols if c not in text_set]
            cat_cols = [c for c in cat_cols if c not in text_set]
        else:
            num_cols, cat_cols, eff_text = split_columns(X_df, detect_text=True)

        pre = build_preprocessor(
            num_cols,
            cat_cols,
            text_cols=eff_text or None,
            text_method="tfidf",
        )

        classes: Optional[np.ndarray] = None
        if task == "classification":
            assert y_arr is not None
            classes = np.unique(y_arr)
            if len(classes) < 2:
                model: Any = DummyClassifier(strategy="most_frequent")
                algo_name = "DummyClassifier_subsample"
            else:
                model = LogisticRegression(max_iter=1000, random_state=self.random_state)
                algo_name = "LogisticRegression_subsample"
            pipe: Any = Pipeline([("pre", pre), ("model", model)])
            pipe.fit(X_df, y_arr)
        elif task == "regression":
            assert y_arr is not None
            model = Ridge(alpha=1.0, random_state=self.random_state)
            algo_name = "Ridge_subsample"
            pipe = Pipeline([("pre", pre), ("model", model)])
            pipe.fit(X_df, y_arr)
        elif task == "clustering":
            eff_k = min(int(self.n_clusters), len(X_df))
            model = KMeans(n_clusters=eff_k, n_init="auto", random_state=self.random_state)
            algo_name = "KMeans_subsample"
            pipe = Pipeline([("pre", pre), ("model", model)])
            pipe.fit(X_df)
        else:  # dim_reduction
            X_pre = pre.fit_transform(X_df)
            if sparse.issparse(X_pre):
                X_pre = X_pre.toarray()
            n_feat = X_pre.shape[1]
            req_comp = int(self.n_components) if self.n_components is not None else min(10, n_feat, len(X_df))
            eff_comp = max(1, min(req_comp, n_feat, len(X_df)))
            model = PCA(n_components=eff_comp, random_state=self.random_state)
            model.fit(X_pre)
            algo_name = "PCA_subsample"
            pipe = Pipeline([("pre", pre), ("model", model)])

        self.is_fitted_ = True
        self.task_ = task
        self.mode_ = "subsample"
        self.algorithm_ = algo_name
        self.n_rows_seen_ = int(total_rows)
        self.n_chunks_seen_ = int(n_chunks)
        self.n_rows_sampled_ = int(len(sample_df))
        self.feature_columns_ = feature_cols
        self.numeric_cols_ = list(num_cols)
        self.categorical_cols_ = list(cat_cols)
        self.text_cols_ = list(eff_text)
        self.target_col_ = target_col
        self.classes_ = classes
        self.encoder_ = None
        self.estimator_ = pipe
        self.audit_trail_ = _sanitize_json_obj(
            {
                "task": self.task_,
                "mode": self.mode_,
                "algorithm": self.algorithm_,
                "chunk_size": int(self.chunk_size),
                "max_in_memory_rows": int(self.max_in_memory_rows),
                "n_rows_seen": int(self.n_rows_seen_),
                "n_rows_sampled": int(self.n_rows_sampled_),
                "n_chunks_seen": int(self.n_chunks_seen_),
                "subsampled": bool(self.n_rows_seen_ > self.n_rows_sampled_),
                "numeric_cols": [str(c) for c in self.numeric_cols_],
                "categorical_cols": [str(c) for c in self.categorical_cols_],
                "text_cols": [str(c) for c in self.text_cols_],
                "target_col": self.target_col_,
                "n_classes": int(len(classes)) if classes is not None else None,
            }
        )
        return self

    def _fit_partial_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        task: str,
        target_col: Optional[str],
        text_cols: Optional[List[str]],
        known_classes: Optional[np.ndarray] = None,
    ) -> "ACDSOOutOfCoreTrainer":
        max_buf = int(self.max_in_memory_rows)
        eff_chunk_size = int(self.chunk_size)
        is_dim_red = task == "dim_reduction"

        min_warmup = min(
            max_buf,
            max(
                eff_chunk_size,
                64,
                int(self.n_clusters) if task == "clustering" else 2,
                (2 * int(self.n_components)) if (is_dim_red and self.n_components is not None) else (int(self.n_components) if self.n_components is not None else 2),
            ),
        )

        warmup_X_parts: List[pd.DataFrame] = []
        warmup_y_parts: List[np.ndarray] = []
        warmup_rows = 0
        dim_buf: List[np.ndarray] = []
        dim_buf_rows = 0
        n_rows_seen = 0
        n_valid_rows = 0
        n_chunks_seen = 0

        encoder: Optional[_StreamingFeatureEncoder] = None
        estimator: Optional[Any] = None
        resolved_task = task
        classes_arr: Optional[np.ndarray] = known_classes
        is_single_class_fallback = False
        algo_name = ""

        def _init_from_warmup() -> None:
            nonlocal encoder, estimator, resolved_task, classes_arr, is_single_class_fallback, algo_name, dim_buf_rows

            df_warm = pd.concat(warmup_X_parts, ignore_index=True) if len(warmup_X_parts) > 1 else warmup_X_parts[0]
            y_warm = np.concatenate(warmup_y_parts) if warmup_y_parts else None
            warmup_X_parts.clear()
            warmup_y_parts.clear()

            if resolved_task == "auto":
                resolved_task = self._resolve_auto_task(y_warm, target_col)

            hash_dim = min(int(self.n_hash_features), 512) if resolved_task == "dim_reduction" else int(self.n_hash_features)
            encoder = _StreamingFeatureEncoder(
                n_hash_features=hash_dim,
                cat_hash_features=256 if resolved_task == "dim_reduction" else 1024,
            )
            encoder.fit_warmup(df_warm, text_cols=text_cols)

            dense_needed = resolved_task == "dim_reduction"
            X_tr = encoder.partial_fit_transform(df_warm, dense=dense_needed)

            if resolved_task == "classification":
                assert y_warm is not None
                if classes_arr is None:
                    classes_arr = np.unique(y_warm)
                if len(classes_arr) < 2:
                    est = DummyClassifier(strategy="most_frequent")
                    est.fit(X_tr, y_warm)
                    estimator = est
                    is_single_class_fallback = True
                    algo_name = "DummyClassifier_partial_fallback"
                else:
                    est = SGDClassifier(
                        loss="log_loss",
                        alpha=1e-4,
                        random_state=self.random_state,
                    )
                    est.partial_fit(X_tr, y_warm, classes=classes_arr)
                    estimator = est
                    algo_name = "SGDClassifier"
            elif resolved_task == "regression":
                assert y_warm is not None
                est = SGDRegressor(alpha=1e-4, random_state=self.random_state)
                est.partial_fit(X_tr, y_warm)
                estimator = est
                algo_name = "SGDRegressor"
            elif resolved_task == "clustering":
                eff_k = max(1, min(int(self.n_clusters), X_tr.shape[0]))
                est = MiniBatchKMeans(
                    n_clusters=eff_k,
                    batch_size=max(100, eff_chunk_size),
                    random_state=self.random_state,
                    n_init="auto",
                )
                est.partial_fit(X_tr)
                estimator = est
                algo_name = "MiniBatchKMeans"
            else:  # dim_reduction
                n_feat = X_tr.shape[1]
                req_comp = int(self.n_components) if self.n_components is not None else min(10, n_feat, X_tr.shape[0])
                eff_comp = max(1, min(req_comp, n_feat, X_tr.shape[0]))
                est = IncrementalPCA(n_components=eff_comp)
                if X_tr.shape[0] >= 2 * eff_comp:
                    n_init_fit = int(X_tr.shape[0]) - eff_comp
                    est.partial_fit(X_tr[:n_init_fit])
                    X_keep = X_tr[n_init_fit:]
                    dim_buf.append(X_keep)
                    dim_buf_rows += int(X_keep.shape[0])
                else:
                    est.partial_fit(X_tr)
                estimator = est
                algo_name = "IncrementalPCA"

        for raw_chunk in chunks:
            if not isinstance(raw_chunk, pd.DataFrame):
                raw_chunk = _coerce_to_dataframe(raw_chunk)
            if len(raw_chunk) == 0:
                continue
            n_chunks_seen += 1
            n_rows_seen += len(raw_chunk)

            if resolved_task in ("classification", "regression") or (
                resolved_task == "auto" and target_col is not None
            ):
                if not target_col:
                    raise ValueError(f"target_col is required for task={resolved_task!r}.")
                X_chunk, y_chunk = _clean_supervised_chunk(raw_chunk, target_col)
                if len(y_chunk) == 0:
                    continue
                n_valid_rows += len(y_chunk)
            else:
                if target_col and target_col in raw_chunk.columns:
                    X_chunk = raw_chunk.drop(columns=[target_col])
                else:
                    X_chunk = raw_chunk
                y_chunk = None
                n_valid_rows += len(X_chunk)

            if estimator is None:
                warmup_X_parts.append(X_chunk)
                if y_chunk is not None:
                    warmup_y_parts.append(y_chunk)
                warmup_rows += len(X_chunk)

                ready = warmup_rows >= min_warmup
                if (
                    ready
                    and resolved_task in ("classification", "auto")
                    and y_chunk is not None
                    and classes_arr is None
                ):
                    seen_y = np.unique(np.concatenate(warmup_y_parts))
                    if len(seen_y) < 2 and warmup_rows < max_buf:
                        ready = False
                if ready or warmup_rows >= max_buf:
                    _init_from_warmup()
                continue

            assert encoder is not None
            dense_needed = resolved_task == "dim_reduction"
            X_tr = encoder.partial_fit_transform(X_chunk, dense=dense_needed)
            if resolved_task == "classification":
                assert y_chunk is not None
                chunk_classes = np.unique(y_chunk)
                if is_single_class_fallback:
                    assert classes_arr is not None
                    combined_cls = np.unique(np.concatenate([classes_arr, chunk_classes]))
                    if len(combined_cls) >= 2:
                        # Upgrade from single-class dummy fallback cleanly
                        classes_arr = combined_cls
                        est_new = SGDClassifier(
                            loss="log_loss",
                            alpha=1e-4,
                            random_state=self.random_state,
                        )
                        est_new.partial_fit(X_tr, y_chunk, classes=classes_arr)
                        estimator = est_new
                        is_single_class_fallback = False
                        algo_name = "SGDClassifier"
                else:
                    if not set(chunk_classes.tolist()).issubset(set(estimator.classes_.tolist())):
                        estimator = _expand_sgd_classes(estimator, chunk_classes)
                        classes_arr = np.asarray(estimator.classes_)
                    estimator.partial_fit(X_tr, y_chunk)
            elif resolved_task == "regression":
                assert y_chunk is not None
                estimator.partial_fit(X_tr, y_chunk)
            elif resolved_task == "clustering":
                estimator.partial_fit(X_tr)
            else:
                dim_buf.append(X_tr)
                dim_buf_rows += int(X_tr.shape[0])
                eff_comp = int(estimator.n_components)
                flush_thresh = min(max_buf, max(32, eff_comp) + eff_comp)
                if dim_buf_rows >= flush_thresh:
                    X_buf = np.vstack(dim_buf) if len(dim_buf) > 1 else dim_buf[0]
                    n_buf = int(X_buf.shape[0])
                    if n_buf >= 2 * eff_comp:
                        n_flush = n_buf - eff_comp
                        estimator.partial_fit(X_buf[:n_flush])
                        X_keep = X_buf[n_flush:]
                        dim_buf = [X_keep]
                        dim_buf_rows = int(X_keep.shape[0])
                    elif n_buf >= eff_comp:
                        estimator.partial_fit(X_buf)
                        dim_buf.clear()
                        dim_buf_rows = 0

        if n_rows_seen == 0 or n_valid_rows == 0:
            raise ValueError("Cannot fit ACDSOOutOfCoreTrainer on an empty dataset (0 valid rows).")

        if estimator is None:
            _init_from_warmup()

        if resolved_task == "dim_reduction" and dim_buf_rows > 0:
            assert estimator is not None
            eff_comp = int(estimator.n_components)
            X_rem = np.vstack(dim_buf) if len(dim_buf) > 1 else dim_buf[0]
            dim_buf.clear()
            dim_buf_rows = 0
            if X_rem.shape[0] >= eff_comp:
                estimator.partial_fit(X_rem)

        assert encoder is not None and estimator is not None

        self.is_fitted_ = True
        self.task_ = resolved_task
        self.mode_ = "partial_fit"
        self.algorithm_ = algo_name
        self.n_rows_seen_ = int(n_rows_seen)
        self.n_valid_rows_ = int(n_valid_rows)
        self.n_chunks_seen_ = int(n_chunks_seen)
        self.feature_columns_ = list(encoder.feature_columns_)
        self.numeric_cols_ = list(encoder.numeric_cols_)
        self.categorical_cols_ = list(encoder.categorical_cols_)
        self.text_cols_ = list(encoder.text_cols_)
        self.target_col_ = target_col
        self.classes_ = np.asarray(estimator.classes_) if hasattr(estimator, "classes_") else classes_arr
        self.encoder_ = encoder
        self.estimator_ = estimator
        self.audit_trail_ = _sanitize_json_obj(
            {
                "task": self.task_,
                "mode": self.mode_,
                "algorithm": self.algorithm_,
                "chunk_size": int(self.chunk_size),
                "max_in_memory_rows": int(self.max_in_memory_rows),
                "n_hash_features": int(self.n_hash_features),
                "n_rows_seen": int(self.n_rows_seen_),
                "n_valid_rows": int(self.n_valid_rows_),
                "n_chunks_seen": int(self.n_chunks_seen_),
                "subsampled": False,
                "numeric_cols": [str(c) for c in self.numeric_cols_],
                "categorical_cols": [str(c) for c in self.categorical_cols_],
                "text_cols": [str(c) for c in self.text_cols_],
                "target_col": self.target_col_,
                "n_classes": int(len(self.classes_)) if self.classes_ is not None else None,
            }
        )
        return self

    def fit_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        target_col: Optional[str] = None,
        text_cols: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> "ACDSOOutOfCoreTrainer":
        """Train incrementally over a stream of DataFrame chunks."""
        task_norm, eff_mode = self._validate_params(mode_override=mode)
        if task_norm in ("classification", "regression") and not target_col:
            raise ValueError(f"target_col is required for task={task_norm!r}.")

        if eff_mode == "subsample":
            # Single-pass Algorithm R reservoir over the incoming chunks
            limit = int(self.max_in_memory_rows)
            rng = np.random.RandomState(self.random_state)
            total_rows = 0
            n_chunks = 0
            columns: Optional[List[Any]] = None
            dtypes_map: Dict[Any, Any] = {}
            buffered: List[pd.DataFrame] = []
            reservoir: Optional[List[Tuple[Any, ...]]] = None

            for raw_chunk in chunks:
                if not isinstance(raw_chunk, pd.DataFrame):
                    raw_chunk = _coerce_to_dataframe(raw_chunk)
                n_chunk = len(raw_chunk)
                if n_chunk == 0:
                    continue
                n_chunks += 1
                if columns is None:
                    columns = list(raw_chunk.columns)
                    dtypes_map = {col: raw_chunk[col].dtype for col in columns}
                else:
                    new_cols = [col for col in raw_chunk.columns if col not in dtypes_map]
                    if new_cols:
                        columns.extend(new_cols)
                        for col in new_cols:
                            c_dtype = raw_chunk[col].dtype
                            if pd.api.types.is_bool_dtype(c_dtype):
                                dtypes_map[col] = np.dtype(object)
                            elif pd.api.types.is_integer_dtype(c_dtype):
                                dtypes_map[col] = np.dtype(np.float64)
                            else:
                                dtypes_map[col] = c_dtype
                        if reservoir is not None:
                            pad = (None,) * len(new_cols)
                            reservoir = [row + pad for row in reservoir]
                    for col in columns:
                        if col in raw_chunk.columns:
                            if raw_chunk[col].dtype != dtypes_map[col]:
                                dtypes_map[col] = _widen_chunk_dtype(dtypes_map[col], raw_chunk[col].dtype)
                        else:
                            if pd.api.types.is_bool_dtype(dtypes_map[col]):
                                dtypes_map[col] = np.dtype(object)
                            elif pd.api.types.is_integer_dtype(dtypes_map[col]):
                                dtypes_map[col] = np.dtype(np.float64)
                    if list(raw_chunk.columns) != columns:
                        raw_chunk = raw_chunk.reindex(columns=columns)
                if reservoir is None:
                    if total_rows + n_chunk <= limit:
                        buffered.append(raw_chunk)
                        total_rows += n_chunk
                        continue
                    reservoir = []
                    for b in buffered:
                        if list(b.columns) != columns:
                            b = b.reindex(columns=columns)
                        reservoir.extend(b.itertuples(index=False, name=None))
                    buffered.clear()
                    need = limit - len(reservoir)
                    if need > 0:
                        reservoir.extend(raw_chunk.iloc[:need].itertuples(index=False, name=None))
                        total_rows += need
                        rem_df = raw_chunk.iloc[need:]
                    else:
                        rem_df = raw_chunk
                else:
                    rem_df = raw_chunk

                n_rem = len(rem_df)
                if n_rem > 0:
                    upper = np.arange(total_rows + 1, total_rows + n_rem + 1, dtype=np.int64)
                    j_draws = rng.randint(0, upper, dtype=np.int64)
                    hits = np.flatnonzero(j_draws < limit)
                    if hits.size > 0:
                        rows_list = list(rem_df.itertuples(index=False, name=None))
                        for pos in hits.tolist():
                            reservoir[int(j_draws[pos])] = rows_list[pos]
                    total_rows += n_rem

            if total_rows == 0:
                raise ValueError("Cannot fit ACDSOOutOfCoreTrainer on an empty chunk stream (0 rows).")
            if reservoir is None:
                if len(buffered) > 1:
                    aligned_buf = [
                        b.reindex(columns=columns) if list(b.columns) != columns else b
                        for b in buffered
                    ]
                    sample_df = pd.concat(aligned_buf, ignore_index=True)
                else:
                    sample_df = buffered[0].reset_index(drop=True)
            else:
                sample_df = pd.DataFrame.from_records(reservoir, columns=columns)
                for col, dtype in dtypes_map.items():
                    if col in sample_df.columns and sample_df[col].dtype != dtype:
                        if pd.api.types.is_integer_dtype(dtype) and (
                            pd.api.types.is_float_dtype(sample_df[col].dtype)
                            or sample_df[col].dtype == object
                        ):
                            continue
                        try:
                            sample_df[col] = sample_df[col].astype(dtype)
                        except (TypeError, ValueError):
                            pass
            return self._fit_subsample_df(
                sample_df=sample_df,
                total_rows=total_rows,
                n_chunks=n_chunks,
                task=task_norm,
                target_col=target_col,
                text_cols=text_cols,
            )

        return self._fit_partial_chunks(
            chunks=chunks,
            task=task_norm,
            target_col=target_col,
            text_cols=text_cols,
        )

    def fit_from_file(
        self,
        path: Union[str, os.PathLike[str]],
        target_col: Optional[str] = None,
        text_cols: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> "ACDSOOutOfCoreTrainer":
        """Train directly from a file on disk using streaming chunks or reservoir sampling."""
        file_path = os.fspath(path) if path is not None else ""
        if not file_path or not os.path.isfile(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        task_norm, eff_mode = self._validate_params(mode_override=mode)
        if task_norm in ("classification", "regression") and not target_col:
            raise ValueError(f"target_col is required for task={task_norm!r}.")

        if eff_mode == "subsample":
            sample_df, total_rows = reservoir_sample_file(
                file_path,
                max_rows=int(self.max_in_memory_rows),
                chunk_size=int(self.chunk_size),
                random_state=self.random_state,
            )
            n_chunks = max(1, int(math.ceil(total_rows / float(self.chunk_size)))) if total_rows > 0 else 0
            return self._fit_subsample_df(
                sample_df=sample_df,
                total_rows=total_rows,
                n_chunks=n_chunks,
                task=task_norm,
                target_col=target_col,
                text_cols=text_cols,
            )

        known_classes: Optional[np.ndarray] = None
        if target_col is not None and task_norm in ("classification", "auto"):
            uniq_parts: List[np.ndarray] = []
            for y_chunk_df in iter_file_chunks(
                file_path,
                chunk_size=int(self.chunk_size),
                usecols=[target_col],
            ):
                _, y_vals = _clean_supervised_chunk(y_chunk_df, target_col)
                if len(y_vals) > 0:
                    uniq_parts.append(np.unique(y_vals))
            if not uniq_parts:
                raise ValueError(f"Dataset '{file_path}' contains 0 valid rows for target '{target_col}'.")
            all_y_sample = np.concatenate(uniq_parts)
            if task_norm == "auto":
                task_norm = self._resolve_auto_task(all_y_sample, target_col)
            if task_norm == "classification":
                known_classes = np.unique(all_y_sample)

        return self._fit_partial_chunks(
            chunks=iter_file_chunks(file_path, chunk_size=int(self.chunk_size)),
            task=task_norm,
            target_col=target_col,
            text_cols=text_cols,
            known_classes=known_classes,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Any] = None,
        text_cols: Optional[List[str]] = None,
    ) -> "ACDSOOutOfCoreTrainer":
        """Sklearn-style fit slicing an in-memory DataFrame/ndarray into bounded chunks."""
        self._validate_params()
        X_df = _coerce_to_dataframe(X)
        if len(X_df) == 0:
            raise ValueError("Cannot fit ACDSOOutOfCoreTrainer on an empty dataset (0 rows).")

        target_col: Optional[str] = None
        if y is not None:
            y_arr = np.asarray(y)
            if len(y_arr) != len(X_df):
                raise ValueError(f"X and y length mismatch ({len(X_df)} vs {len(y_arr)}).")
            target_col = "__acdso_target__"
            while target_col in X_df.columns:
                target_col = f"_{target_col}"
            df_full = X_df.copy()
            df_full[target_col] = y_arr
        else:
            df_full = X_df

        step = int(self.chunk_size)
        n_total = len(df_full)
        chunk_iter = (df_full.iloc[i : i + step] for i in range(0, n_total, step))
        return self.fit_chunks(chunk_iter, target_col=target_col, text_cols=text_cols)

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False) or self.estimator_ is None:
            raise RuntimeError("ACDSOOutOfCoreTrainer is not fitted yet. Call fit() or fit_from_file() first.")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict target labels, continuous values, or cluster IDs for X."""
        self._check_is_fitted()
        if self.task_ == "dim_reduction":
            raise ValueError("predict() is not supported for dim_reduction; call transform() instead.")

        df_X = _coerce_to_dataframe(
            X,
            expected_columns=self.feature_columns_,
            drop_col=self.target_col_,
        )
        if self.mode_ == "subsample":
            return np.asarray(self.estimator_.predict(df_X[self.feature_columns_]))

        assert self.encoder_ is not None
        X_tr = self.encoder_.transform(df_X, dense=False)
        return np.asarray(self.estimator_.predict(X_tr))

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Project X into reduced components (valid when ``task`` is ``'dim_reduction'``/``'pca'``)."""
        self._check_is_fitted()
        if self.task_ != "dim_reduction":
            raise ValueError(f"transform() is only supported for task='dim_reduction'/'pca', got {self.task_!r}.")

        df_X = _coerce_to_dataframe(
            X,
            expected_columns=self.feature_columns_,
            drop_col=self.target_col_,
        )
        if self.mode_ == "subsample":
            X_pre = self.estimator_.named_steps["pre"].transform(df_X[self.feature_columns_])
            if sparse.issparse(X_pre):
                X_pre = X_pre.toarray()
            return np.asarray(self.estimator_.named_steps["model"].transform(X_pre))

        assert self.encoder_ is not None
        X_tr = self.encoder_.transform(df_X, dense=True)
        return np.asarray(self.estimator_.transform(X_tr))

    def predict_to_file(
        self,
        input_path: Union[str, os.PathLike[str]],
        output_path: Union[str, os.PathLike[str]],
        chunk_size: Optional[int] = None,
        prediction_col: Optional[str] = None,
    ) -> Path:
        """Stream an input file in chunks, predict per chunk, and write a CSV to ``output_path``."""
        self._check_is_fitted()
        in_path = os.fspath(input_path) if input_path is not None else ""
        if not in_path or not os.path.isfile(in_path):
            raise FileNotFoundError(f"Input file not found: {in_path}")

        if chunk_size is not None and (isinstance(chunk_size, bool) or int(chunk_size) < 1):
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
        eff_chunk = int(self.chunk_size) if chunk_size is None else int(chunk_size)
        if eff_chunk < 1:
            raise ValueError(f"chunk_size must be a positive integer, got {eff_chunk!r}.")

        if prediction_col is not None:
            pred_col = str(prediction_col)
        elif self.target_col_ and not str(self.target_col_).startswith("__acdso_"):
            pred_col = f"{self.target_col_}_predicted"
        elif self.task_ == "clustering":
            pred_col = "cluster"
        else:
            pred_col = "prediction"

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        wrote_header = False
        for chunk in iter_file_chunks(in_path, chunk_size=eff_chunk):
            out_chunk = chunk.copy()
            if self.task_ == "dim_reduction":
                proj = self.transform(chunk)
                for j in range(proj.shape[1]):
                    out_chunk[f"component_{j}"] = proj[:, j]
            else:
                out_chunk[pred_col] = self.predict(chunk)
            out_chunk.to_csv(
                out_p,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True

        if not wrote_header:
            cols = list(self.feature_columns_)
            if self.task_ == "dim_reduction":
                pca_model = (
                    self.estimator_.named_steps["model"]
                    if self.mode_ == "subsample"
                    else self.estimator_
                )
                n_comp = int(getattr(pca_model, "n_components_", getattr(pca_model, "n_components", 0)))
                if n_comp <= 0 and hasattr(pca_model, "components_"):
                    n_comp = int(pca_model.components_.shape[0])
                cols.extend(f"component_{j}" for j in range(n_comp))
            else:
                cols.append(pred_col)
            pd.DataFrame(columns=cols).to_csv(out_p, index=False)

        return out_p

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> float:
        """Compute task-appropriate score (accuracy, R2, silhouette, or explained variance)."""
        self._check_is_fitted()
        if self.task_ == "classification":
            if y is None:
                raise ValueError("y is required to score classification.")
            return float(accuracy_score(np.asarray(y), self.predict(X)))
        if self.task_ == "regression":
            if y is None:
                raise ValueError("y is required to score regression.")
            return float(r2_score(np.asarray(y), self.predict(X)))
        if self.task_ == "clustering":
            labels = self.predict(X)
            n_uniq = len(np.unique(labels))
            if n_uniq >= 2 and len(labels) > n_uniq:
                df_X = _coerce_to_dataframe(X, expected_columns=self.feature_columns_, drop_col=self.target_col_)
                if self.mode_ == "subsample":
                    X_mat = self.estimator_.named_steps["pre"].transform(df_X[self.feature_columns_])
                else:
                    assert self.encoder_ is not None
                    X_mat = self.encoder_.transform(df_X, dense=False)
                if len(labels) > 5000:
                    rng = np.random.RandomState(self.random_state)
                    idx = rng.choice(len(labels), size=5000, replace=False)
                    X_mat = X_mat[idx]
                    labels = labels[idx]
                return float(silhouette_score(X_mat, labels))
            return 0.0
        # dim_reduction
        ratio = getattr(self.estimator_.named_steps["model"] if self.mode_ == "subsample" else self.estimator_, "explained_variance_ratio_", None)
        if ratio is not None:
            return float(np.sum(ratio))
        return 0.0

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> Dict[str, Any]:
        """Return a JSON-serializable evaluation dictionary."""
        self._check_is_fitted()
        primary = self.score(X, y)
        res: Dict[str, Any] = {
            "task": self.task_,
            "mode": self.mode_,
            "algorithm": self.algorithm_,
            "primary_score": primary,
            "n_rows_evaluated": int(len(X)),
        }
        if self.task_ == "classification":
            res["accuracy"] = primary
        elif self.task_ == "regression":
            res["r2"] = primary
        elif self.task_ == "clustering":
            res["silhouette"] = primary
            res["n_clusters"] = int(self.n_clusters)
        else:
            res["explained_variance_ratio_sum"] = primary
        return _sanitize_json_obj(res)

    def get_scale_summary(self) -> Dict[str, Any]:
        """Return a strict JSON-serializable copy of the out-of-core execution summary."""
        self._check_is_fitted()
        return _sanitize_json_obj(dict(self.audit_trail_))
