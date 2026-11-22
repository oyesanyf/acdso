"""
ACDSO v2.0 - Utility Functions
==============================

Core utility functions for data preprocessing, type inference,
and column handling used throughout ACDSO.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def _datetime_series_to_epoch(s: pd.Series) -> np.ndarray:
    """Convert a datetime64 Series to float64 epoch seconds, preserving NaT as NaN."""
    if getattr(s.dtype, "tz", None) is not None:
        s_ns = s.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    else:
        s_ns = s.astype("datetime64[ns]")
    na_mask = s_ns.isna().to_numpy()
    vals = s_ns.astype("int64").to_numpy(dtype=np.float64) / 1e9
    if na_mask.any():
        vals[na_mask] = np.nan
    return vals


def sanitize_features(X: Any) -> Any:
    """Sanitize input features for sklearn compatibility.

    - Replaces +/-inf in numeric columns/arrays with np.nan.
    - Casts pandas nullable extension dtypes (Int64/Float64/boolean/string)
      to standard float64 (numeric/bool) or object (string/categorical) with np.nan.
    - Converts datetime64 columns to float64 epoch seconds (NaT -> np.nan).
    """
    if isinstance(X, pd.DataFrame):
        df = X
        copied = False

        def _ensure_copy() -> pd.DataFrame:
            nonlocal df, copied
            if not copied:
                df = X.copy()
                copied = True
            return df

        for i in range(X.shape[1]):
            s = X.iloc[:, i]
            dtype = s.dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                _ensure_copy()
                epoch_vals = _datetime_series_to_epoch(s)
                if not df.columns.has_duplicates:
                    df[df.columns[i]] = pd.Series(epoch_vals, index=df.index, dtype=np.float64)
                else:
                    df = df.copy()
                    cols_list = [df.iloc[:, j] if j != i else pd.Series(epoch_vals, index=df.index, name=df.columns[i], dtype=np.float64) for j in range(df.shape[1])]
                    df = pd.concat(cols_list, axis=1)
            elif pd.api.types.is_bool_dtype(dtype):
                if isinstance(dtype, pd.api.extensions.ExtensionDtype) or s.isna().any():
                    _ensure_copy()
                    df.iloc[:, i] = s.to_numpy(dtype=np.float64, na_value=np.nan)
            elif pd.api.types.is_numeric_dtype(dtype):
                if isinstance(dtype, pd.api.extensions.ExtensionDtype):
                    _ensure_copy()
                    arr = s.to_numpy(dtype=np.float64, na_value=np.nan)
                    arr[~np.isfinite(arr)] = np.nan
                    df.iloc[:, i] = arr
                elif np.issubdtype(dtype, np.floating):
                    arr = s.to_numpy(dtype=np.float64, copy=False)
                    if np.isinf(arr).any():
                        _ensure_copy()
                        arr_c = arr.copy()
                        arr_c[np.isinf(arr_c)] = np.nan
                        df.iloc[:, i] = arr_c
            elif isinstance(dtype, pd.StringDtype):
                _ensure_copy()
                df.iloc[:, i] = s.astype(object).where(s.notna(), np.nan)
            elif dtype == object or isinstance(dtype, pd.CategoricalDtype):
                if s.isna().any():
                    # Normalize pd.NA / None inside object/categorical columns
                    if dtype == object:
                        _ensure_copy()
                        df.iloc[:, i] = s.where(s.notna(), np.nan)
        return df

    arr = np.asarray(X)
    if arr.ndim == 0:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if np.isinf(arr).any():
            arr = arr.copy()
            arr[np.isinf(arr)] = np.nan
        return arr
    if np.issubdtype(arr.dtype, np.datetime64):
        s_ns = pd.Series(arr.ravel()).astype("datetime64[ns]")
        vals = _datetime_series_to_epoch(s_ns).reshape(arr.shape)
        return vals
    if arr.dtype == object:
        needs_fix = False
        flat = arr.ravel()
        for v in flat:
            if v is pd.NA or v is None or (isinstance(v, (float, np.floating)) and np.isinf(v)):
                needs_fix = True
                break
        if needs_fix:
            fixed = np.empty(flat.shape, dtype=object)
            for idx, v in enumerate(flat):
                if v is pd.NA or v is None or (isinstance(v, (float, np.floating)) and np.isinf(v)):
                    fixed[idx] = np.nan
                else:
                    fixed[idx] = v
            return fixed.reshape(arr.shape)
    return arr


class _NumericSanitizer(BaseEstimator, TransformerMixin):
    """Sanitize numeric/datetime/nullable columns to float64 with np.nan."""

    def fit(self, X: Any, y: Any = None) -> "_NumericSanitizer":
        return self

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 0:
                return np.empty((len(X), 0), dtype=np.float64)
            cols = []
            for i in range(X.shape[1]):
                s = X.iloc[:, i]
                if pd.api.types.is_datetime64_any_dtype(s.dtype):
                    arr = _datetime_series_to_epoch(s)
                else:
                    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
                arr[~np.isfinite(arr)] = np.nan
                cols.append(arr)
            return np.column_stack(cols)
        arr = np.asarray(sanitize_features(X), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size > 0:
            arr = arr.copy()
            arr[~np.isfinite(arr)] = np.nan
        return arr


class _CategoricalSanitizer(BaseEstimator, TransformerMixin):
    """Sanitize categorical/string/nullable columns to object array with np.nan for missing."""

    def fit(self, X: Any, y: Any = None) -> "_CategoricalSanitizer":
        return self

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 0:
                return np.empty((len(X), 0), dtype=object)
            cols = []
            for i in range(X.shape[1]):
                s = X.iloc[:, i]
                if pd.api.types.is_datetime64_any_dtype(s.dtype):
                    col_obj = s.astype(str).where(s.notna(), np.nan).to_numpy(dtype=object)
                else:
                    col_obj = s.astype(object).where(s.notna(), np.nan).to_numpy(dtype=object)
                cols.append(col_obj)
            return np.column_stack(cols)
        arr = np.asarray(sanitize_features(X), dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr


def infer_target_type(y: Any) -> str:
    """
    Infer the target type from the target array.

    Parameters
    ----------
    y : array-like
        Target values.

    Returns
    -------
    str
        One of "binary", "multiclass", or "regression".
    """
    y_arr = np.asarray(y)
    if y_arr.ndim == 0 or y_arr.size == 0:
        raise ValueError("Target array y cannot be empty.")

    if np.issubdtype(y_arr.dtype, np.floating):
        finite_vals = y_arr[np.isfinite(y_arr)]
        if finite_vals.size == 0:
            raise ValueError("Target array y contains no finite values.")
        unique_vals = np.unique(finite_vals)
        if len(unique_vals) <= 20 and np.allclose(unique_vals, np.rint(unique_vals)):
            if len(unique_vals) == 2:
                return "binary"
            return "multiclass"
        return "regression"

    # Handle object, nullable pandas extension dtypes, strings, and ints safely
    s_clean = pd.Series(y_arr).dropna()
    if len(s_clean) == 0:
        raise ValueError("Target array y contains only missing values.")

    if pd.api.types.is_float_dtype(s_clean.dtype):
        vals = s_clean.to_numpy(dtype=np.float64)
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size == 0:
            raise ValueError("Target array y contains no finite values.")
        unique_vals = np.unique(finite_vals)
        if len(unique_vals) <= 20 and np.allclose(unique_vals, np.rint(unique_vals)):
            return "binary" if len(unique_vals) == 2 else "multiclass"
        return "regression"

    n_unique = int(s_clean.nunique())
    if n_unique == 2:
        return "binary"
    return "multiclass"


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Split DataFrame columns into numeric and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of (numeric_columns, categorical_columns).
    """
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    num_set = set(num_cols)
    cat_cols = [c for c in df.columns if c not in num_set]
    return num_cols, cat_cols


def compute_missing_ratio(df: pd.DataFrame) -> float:
    """
    Compute the ratio of missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    float
        Ratio of missing values (0.0 to 1.0).
    """
    total = df.shape[0] * df.shape[1]
    return float(df.isna().sum().sum() / total) if total else 0.0


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    sparse_threshold: float = 0.3,
) -> ColumnTransformer:
    """
    Build a preprocessing ColumnTransformer for numeric and categorical columns.

    The preprocessor applies:
    - Numeric: Median imputation + StandardScaler (without centering for sparse compatibility)
    - Categorical: Mode imputation + OneHotEncoder

    Parameters
    ----------
    numeric_cols : List[str]
        List of numeric column names.
    categorical_cols : List[str]
        List of categorical column names.
    sparse_threshold : float, default=0.3
        Threshold for sparse matrix output (0.0-1.0).

    Returns
    -------
    ColumnTransformer
        Configured preprocessing transformer.
    """
    num_pipe = Pipeline([
        ("sanitize", _NumericSanitizer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    cat_pipe = Pipeline([
        ("sanitize", _CategoricalSanitizer()),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
    )


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect a datetime-like column in the DataFrame.

    First checks for actual datetime64 columns, then falls back to
    word-boundary name heuristics (date, time, timestamp, etc.),
    skipping numeric columns to avoid false positives like
    ``sentiment_score_control`` matching on "time".

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    Optional[str]
        Name of detected datetime column, or None if not found.
    """
    import re

    # First, check for actual datetime columns
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return str(c)

    # Fall back to word-boundary name heuristics, skipping numeric columns
    _dt_pattern = re.compile(
        r'(?<![a-z])(?:date|time|timestamp|ts|datetime|created|updated|week|period)(?![a-z])',
        re.IGNORECASE,
    )
    name_hits = [
        c for c in df.columns
        if _dt_pattern.search(str(c)) and not pd.api.types.is_numeric_dtype(df[c])
    ]
    return str(name_hits[0]) if name_hits else None


def class_imbalance(y: Any) -> Tuple[Optional[float], float]:
    """
    Compute class imbalance metrics.

    Parameters
    ----------
    y : array-like
        Target values.

    Returns
    -------
    Tuple[Optional[float], float]
        Tuple of (max_class_ratio, imbalance_severity).
        - max_class_ratio: Ratio of the most common class (0.0 to 1.0)
        - imbalance_severity: Severity score (0.0 = balanced, 1.0 = severely imbalanced)
    """
    counts = pd.Series(y).dropna().value_counts(dropna=True).to_numpy()
    if len(counts) == 0:
        return None, 0.0
    max_ratio = float(np.max(counts) / np.sum(counts))
    # Severity: 0 when max_ratio=0.5 (balanced), 1 when max_ratio=1.0 (one class)
    severity = float(np.clip((max_ratio - 0.5) / 0.5, 0.0, 1.0))
    return max_ratio, severity


def measure_latency_ms_per_row(
    pipeline: Pipeline,
    X: Any,
    n_rows: int = 256,
) -> float:
    """
    Measure inference latency in milliseconds per row.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    X : array-like
        Input data for prediction.
    n_rows : int, default=256
        Number of rows to use for measurement.

    Returns
    -------
    float
        Latency in milliseconds per row.
    """
    import time

    try:
        if hasattr(X, "iloc"):
            Xs = X.iloc[:n_rows]
            rows = min(n_rows, len(Xs))
        else:
            Xs = X[:n_rows]
            rows = min(n_rows, len(Xs))

        if rows <= 0:
            return 0.0

        t0 = time.perf_counter()
        repeats = 5 if rows <= 256 else 1
        for _ in range(repeats):
            _ = pipeline.predict(Xs)
        t1 = time.perf_counter()

        return float((t1 - t0) * 1000.0 / (rows * repeats))
    except Exception:
        return float("inf")

