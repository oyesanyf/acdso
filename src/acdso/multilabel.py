"""
ACDSO - Multi-Label Classification Module (R5a)
=============================================

Provides ``ACDSOMultiLabelClassifier``, a scikit-learn compatible estimator
supporting ``MultiOutputClassifier``, ``ClassifierChain``, multi-output
``RandomForestClassifier``, multi-output ``ExtraTreesClassifier``, and
automated CV-based strategy selection with entity-grouped CV (``groups=``),
free-text NLP preprocessing (``text_cols``), constant-label fold safety,
and strict RFC 8259 JSON audit trails.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
)
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from acdso.cv import build_cv_splitter, iter_cv_splits, resolve_cv_strategy_name
from acdso.utils import build_preprocessor, sanitize_features, split_columns

_VALID_STRATEGIES = (
    "auto",
    "multi_output",
    "classifier_chain",
    "random_forest",
    "extra_trees",
)

_VALID_SCORERS = (
    "f1_micro",
    "f1_macro",
    "hamming_loss",
    "subset_accuracy",
    "jaccard_samples",
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


class _SafeBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Picklable binary classifier wrapper handling constant-class labels safely.

    Prevents ``LogisticRegression`` or ``ClassifierChain`` from raising
    ``ValueError`` when a label column is constant (all 0 or all 1) globally
    or inside a CV training fold.
    """

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> "_SafeBinaryClassifier":
        y_arr = np.asarray(y, dtype=np.int64).ravel()
        uniq = np.unique(y_arr)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        if len(uniq) < 2:
            self._is_constant_ = True
            self.constant_class_ = int(uniq[0]) if len(uniq) == 1 else 0
            self.estimator_ = None
            return self

        base = (
            clone(self.estimator)
            if self.estimator is not None
            else LogisticRegression(max_iter=1000, solver="lbfgs")
        )
        base.fit(X, y_arr)
        self._is_constant_ = False
        self.constant_class_ = 0
        self.estimator_ = base
        return self

    def predict(self, X: Any) -> np.ndarray:
        n_samples = X.shape[0]
        if getattr(self, "_is_constant_", False):
            return np.full(n_samples, self.constant_class_, dtype=np.int64)
        return np.asarray(self.estimator_.predict(X), dtype=np.int64).ravel()

    def predict_proba(self, X: Any) -> np.ndarray:
        n_samples = X.shape[0]
        if getattr(self, "_is_constant_", False):
            out = np.zeros((n_samples, 2), dtype=np.float64)
            if self.constant_class_ == 1:
                out[:, 1] = 1.0
            else:
                out[:, 0] = 1.0
            return out

        raw = np.asarray(self.estimator_.predict_proba(X), dtype=np.float64)
        if raw.ndim == 1:
            p1 = np.clip(raw, 0.0, 1.0)
            return np.column_stack([1.0 - p1, p1])

        cls_list = list(getattr(self.estimator_, "classes_", [0, 1]))
        out = np.zeros((n_samples, 2), dtype=np.float64)
        if 0 in cls_list:
            out[:, 0] = raw[:, cls_list.index(0)]
        if 1 in cls_list:
            out[:, 1] = raw[:, cls_list.index(1)]
        if 0 not in cls_list and 1 in cls_list:
            out[:, 0] = 1.0 - out[:, 1]
        elif 1 not in cls_list and 0 in cls_list:
            out[:, 1] = 1.0 - out[:, 0]
        return np.clip(np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _is_null_scalar(v: Any) -> bool:
    if v is None or v is pd.NA:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(math.isnan(float(v)))
    return False


def _is_delimited_str_batch(items: List[Any], classes: Optional[np.ndarray]) -> bool:
    if any(isinstance(v, str) and ("|" in v or "," in v) for v in items):
        return True
    if classes is not None and len(classes) >= 2:
        has_str = any(isinstance(v, str) for v in items)
        all_str_or_null = all(isinstance(v, str) or _is_null_scalar(v) for v in items)
        if has_str and all_str_or_null:
            return True
    return False


def _split_delimited_labels(
    items: List[Any],
    classes: Optional[np.ndarray] = None,
) -> List[Tuple[str, ...]]:
    has_pipe = any(isinstance(v, str) and "|" in v for v in items) or (
        classes is not None and any(isinstance(c, str) and "," in str(c) for c in classes)
    )
    sep = "|" if has_pipe else ","
    parsed: List[Tuple[str, ...]] = []
    for v in items:
        if _is_null_scalar(v):
            parsed.append(())
        elif isinstance(v, str):
            tokens = tuple(tok.strip() for tok in v.split(sep) if tok.strip())
            parsed.append(tokens)
        else:
            raise ValueError(f"Expected delimited label strings, got {type(v).__name__}.")
    return parsed


def _as_1d_object_array(items: Any) -> np.ndarray:
    """Create a 1D object ndarray without collapsing equal-length tuple items to 2D."""
    if isinstance(items, np.ndarray) and items.ndim == 1 and items.dtype == object:
        return items.copy()
    seq = list(items)
    arr = np.empty(len(seq), dtype=object)
    arr[:] = seq
    return arr


def _coerce_multilabel_y_with_origin(
    y: Any,
    n_samples: int,
    classes: Optional[np.ndarray] = None,
    allow_positional_df: bool = False,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Coerce multi-label targets into a 2D binary int64 array, 1D class labels, and unnamed flag."""
    if y is None:
        raise ValueError("Multi-label target y cannot be None.")

    y_unnamed = False
    if sparse.issparse(y):
        y_unnamed = True
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError(
                f"Multi-label target y must be 2D with at least 2 label columns, got shape {y.shape}."
            )
        arr = y.toarray()
        label_names = (
            _as_1d_object_array(classes)
            if classes is not None
            else np.arange(arr.shape[1], dtype=object)
        )
    elif isinstance(y, pd.DataFrame):
        y_unnamed = False
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError(
                f"Multi-label target DataFrame must have at least 2 label columns, got shape {y.shape}."
            )
        if y.columns.has_duplicates:
            dup_cols = list(y.columns[y.columns.duplicated()])
            raise ValueError(
                f"Multi-label target DataFrame contains duplicate column names: {dup_cols}."
            )
        if classes is not None:
            cls_list = list(classes)
            if len(cls_list) != y.shape[1]:
                raise ValueError(
                    f"Expected {len(cls_list)} label columns in y, got {y.shape[1]}."
                )
            if set(cls_list) == set(y.columns):
                col_pos = {col: idx for idx, col in enumerate(y.columns)}
                arr = y.iloc[:, [col_pos[c] for c in cls_list]].to_numpy()
            elif (
                allow_positional_df
                and cls_list == list(range(len(cls_list)))
                and all(
                    isinstance(c, (int, np.integer)) and not isinstance(c, (bool, np.bool_))
                    for c in cls_list
                )
            ):
                arr = y.to_numpy()
            else:
                raise ValueError(
                    f"Multi-label target DataFrame columns {list(y.columns)} do not match "
                    f"expected label classes {cls_list}."
                )
            label_names = _as_1d_object_array(classes)
        else:
            arr = y.to_numpy()
            label_names = _as_1d_object_array(y.columns)
    elif isinstance(y, np.ndarray):
        y_unnamed = True
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError(
                f"Multi-label target y must be 2D with at least 2 label columns, got shape {y.shape}."
            )
        arr = y
        label_names = (
            _as_1d_object_array(classes)
            if classes is not None
            else np.arange(arr.shape[1], dtype=object)
        )
    elif isinstance(y, pd.Series):
        y_unnamed = False
        items = y.tolist()
        if not items:
            raise ValueError("Multi-label target y cannot be empty.")
        if all(isinstance(v, (list, tuple, set, frozenset)) for v in items):
            mlb = MultiLabelBinarizer(classes=list(classes) if classes is not None else None)
            arr = mlb.fit_transform([list(v) for v in items])
            label_names = _as_1d_object_array(mlb.classes_)
        elif _is_delimited_str_batch(items, classes):
            parsed = _split_delimited_labels(items, classes=classes)
            mlb = MultiLabelBinarizer(classes=list(classes) if classes is not None else None)
            arr = mlb.fit_transform(parsed)
            label_names = _as_1d_object_array(mlb.classes_)
        else:
            raise ValueError(
                "1D pd.Series without label collections or delimited strings cannot be used as multi-label y."
            )
    elif isinstance(y, (list, tuple)):
        items = list(y)
        if not items:
            raise ValueError("Multi-label target y cannot be empty.")
        is_rect_numeric = (
            all(
                isinstance(row, (list, tuple, np.ndarray))
                and not isinstance(row, (set, frozenset, str, bytes))
                for row in items
            )
            and len({len(row) for row in items}) == 1
            and all(
                isinstance(el, (int, float, bool, np.integer, np.floating, np.bool_))
                for row in items
                for el in row
            )
        )
        if is_rect_numeric:
            y_unnamed = True
            arr = np.asarray(items)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(
                    f"Multi-label target y must have at least 2 label columns, got shape {arr.shape}."
                )
            label_names = (
                _as_1d_object_array(classes)
                if classes is not None
                else np.arange(arr.shape[1], dtype=object)
            )
        elif all(isinstance(v, (list, tuple, set, frozenset)) for v in items):
            y_unnamed = False
            mlb = MultiLabelBinarizer(classes=list(classes) if classes is not None else None)
            arr = mlb.fit_transform([list(v) for v in items])
            label_names = _as_1d_object_array(mlb.classes_)
        elif _is_delimited_str_batch(items, classes):
            y_unnamed = False
            parsed = _split_delimited_labels(items, classes=classes)
            mlb = MultiLabelBinarizer(classes=list(classes) if classes is not None else None)
            arr = mlb.fit_transform(parsed)
            label_names = _as_1d_object_array(mlb.classes_)
        else:
            y_unnamed = True
            arr = np.asarray(items)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(
                    f"Multi-label target y must be 2D with at least 2 label columns, got shape {arr.shape}."
                )
            label_names = (
                _as_1d_object_array(classes)
                if classes is not None
                else np.arange(arr.shape[1], dtype=object)
            )
    else:
        y_unnamed = True
        arr = np.asarray(y)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                f"Multi-label target y must be 2D with at least 2 label columns, got shape {arr.shape}."
            )
        label_names = (
            _as_1d_object_array(classes)
            if classes is not None
            else np.arange(arr.shape[1], dtype=object)
        )

    if arr.shape[0] != n_samples:
        raise ValueError(
            f"X and y have inconsistent numbers of samples: {n_samples} vs {arr.shape[0]}."
        )
    if arr.shape[1] < 2:
        raise ValueError(
            f"Multi-label target y must have at least 2 label columns, got {arr.shape[1]}."
        )

    try:
        arr_f = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Multi-label target matrix must contain binary values {0, 1}.") from exc

    if not np.all(np.isfinite(arr_f)):
        raise ValueError("Multi-label target y contains NaN or infinite values.")
    if not np.all((arr_f == 0.0) | (arr_f == 1.0)):
        raise ValueError("Multi-label target y must contain only binary values {0, 1}.")

    return arr_f.astype(np.int64, copy=False), label_names, y_unnamed


def _coerce_multilabel_y(
    y: Any,
    n_samples: int,
    classes: Optional[np.ndarray] = None,
    allow_positional_df: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce multi-label targets into a 2D binary int64 array and 1D class labels."""
    arr, label_names, _ = _coerce_multilabel_y_with_origin(
        y,
        n_samples,
        classes=classes,
        allow_positional_df=allow_positional_df,
    )
    return arr, label_names


def _extract_positive_proba(est: Any, X_mat: Any, n_labels: int) -> np.ndarray:
    """Normalize multi-label predict_proba output to float64 (n_samples, n_labels)."""
    n_samples = X_mat.shape[0]
    if isinstance(est, ClassifierChain):
        proba = np.asarray(est.predict_proba(X_mat), dtype=np.float64)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        if proba.shape != (n_samples, n_labels):
            out = np.zeros((n_samples, n_labels), dtype=np.float64)
            cols = min(n_labels, proba.shape[1])
            out[:, :cols] = proba[:, :cols]
            proba = out
        return np.clip(np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    raw = est.predict_proba(X_mat)
    out = np.zeros((n_samples, n_labels), dtype=np.float64)
    if isinstance(raw, list):
        sub_ests = getattr(est, "estimators_", None)
        est_classes = getattr(est, "classes_", None)
        for j in range(min(n_labels, len(raw))):
            arr_j = np.asarray(raw[j], dtype=np.float64)
            if arr_j.ndim == 1:
                out[:, j] = arr_j
                continue
            cls_j: Any = None
            if isinstance(est, MultiOutputClassifier) and sub_ests is not None and j < len(sub_ests):
                cls_j = getattr(sub_ests[j], "classes_", None)
            elif est_classes is not None and j < len(est_classes):
                cls_j = est_classes[j]

            if cls_j is not None:
                cls_vals = [int(round(float(c))) for c in cls_j]
                if 1 in cls_vals:
                    out[:, j] = arr_j[:, cls_vals.index(1)]
                else:
                    out[:, j] = 0.0
            elif arr_j.shape[1] >= 2:
                out[:, j] = arr_j[:, 1]
            elif arr_j.shape[1] == 1:
                out[:, j] = 0.0
    else:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == n_labels:
            out = arr
        elif arr.ndim == 2:
            cols = min(n_labels, arr.shape[1])
            out[:, :cols] = arr[:, :cols]

    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _compute_multilabel_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, float]:
    """Compute bounded [0, 1] finite multi-label evaluation metrics."""
    f1_mic = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
    f1_mac = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
    h_loss = float(hamming_loss(Y_true, Y_pred))
    sub_acc = float(accuracy_score(Y_true, Y_pred))
    jac_samp = float(jaccard_score(Y_true, Y_pred, average="samples", zero_division=0))

    metrics = {
        "f1_micro": float(np.clip(f1_mic if math.isfinite(f1_mic) else 0.0, 0.0, 1.0)),
        "f1_macro": float(np.clip(f1_mac if math.isfinite(f1_mac) else 0.0, 0.0, 1.0)),
        "hamming_loss": float(np.clip(h_loss if math.isfinite(h_loss) else 1.0, 0.0, 1.0)),
        "subset_accuracy": float(np.clip(sub_acc if math.isfinite(sub_acc) else 0.0, 0.0, 1.0)),
        "jaccard_samples": float(np.clip(jac_samp if math.isfinite(jac_samp) else 0.0, 0.0, 1.0)),
    }
    return metrics


class ACDSOMultiLabelClassifier(BaseEstimator, ClassifierMixin):
    """Multi-label classifier with entity-grouped CV and NLP text featurization.

    Parameters
    ----------
    strategy : str, default="auto"
        One of ``"auto"``, ``"multi_output"``, ``"classifier_chain"``,
        ``"random_forest"``, ``"extra_trees"``.
    scorer : str, default="f1_micro"
        Selection and ``score()`` metric: ``"f1_micro"``, ``"f1_macro"``,
        ``"hamming_loss"``, ``"subset_accuracy"``, or ``"jaccard_samples"``.
    cv : int or splitter, default=5
        Cross-validation folds or splitter object.
    random_state : int, optional, default=42
        Random seed for reproducibility.
    text_cols : list of str, optional
        Explicit free-text column names. When ``None``, auto-detected via
        ``split_columns(..., detect_text=True)``.
    text_method : str, default="tfidf"
        Text featurization mode passed to ``TextFeaturizer``.
    verbose : int, default=0
        Verbosity level.
    """

    def __init__(
        self,
        strategy: str = "auto",
        scorer: str = "f1_micro",
        cv: Any = 5,
        random_state: Optional[int] = 42,
        text_cols: Optional[List[str]] = None,
        text_method: str = "tfidf",
        verbose: int = 0,
    ) -> None:
        self.strategy = strategy
        self.scorer = scorer
        self.cv = cv
        self.random_state = random_state
        self.text_cols = text_cols
        self.text_method = text_method
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose and int(self.verbose) > 0:
            print(f"[ACDSOMultiLabelClassifier] {msg}")

    def _validate_hyperparameters(self) -> Tuple[str, str]:
        if not isinstance(self.strategy, str) or self.strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy={self.strategy!r}. Must be one of {_VALID_STRATEGIES}."
            )
        if not isinstance(self.scorer, str) or self.scorer not in _VALID_SCORERS:
            raise ValueError(
                f"Invalid scorer={self.scorer!r}. Must be one of {_VALID_SCORERS}."
            )
        return self.strategy, self.scorer

    def _coerce_input(self, X: Any, *, is_fit: bool) -> pd.DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None.")

        if isinstance(X, pd.DataFrame):
            n_rows, n_cols = X.shape
            if is_fit:
                if n_rows < 2 or n_cols < 1:
                    raise ValueError(
                        f"ACDSOMultiLabelClassifier.fit requires >= 2 rows and >= 1 column, got {X.shape}."
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
                    f"ACDSOMultiLabelClassifier.fit requires >= 2 rows and >= 1 column, got {arr.shape}."
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

    def _build_candidate(self, name: str) -> Any:
        rs = self.random_state
        if name == "multi_output":
            return MultiOutputClassifier(
                _SafeBinaryClassifier(
                    LogisticRegression(max_iter=1000, solver="lbfgs", random_state=rs)
                )
            )
        if name == "classifier_chain":
            return ClassifierChain(
                _SafeBinaryClassifier(
                    LogisticRegression(max_iter=1000, solver="lbfgs", random_state=rs)
                ),
                order="random",
                random_state=rs,
            )
        if name == "random_forest":
            return RandomForestClassifier(
                n_estimators=100, random_state=rs, n_jobs=1
            )
        if name == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=100, random_state=rs, n_jobs=1
            )
        if name == "multi_output_hgb":
            return MultiOutputClassifier(
                _SafeBinaryClassifier(
                    HistGradientBoostingClassifier(max_iter=100, random_state=rs)
                )
            )
        raise ValueError(f"Unsupported candidate strategy: {name!r}")

    @staticmethod
    def _objective_from_metrics(metrics: Dict[str, float], scorer: str) -> float:
        if scorer == "hamming_loss":
            return float(1.0 - metrics["hamming_loss"])
        return float(metrics[scorer])

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

    def fit(
        self,
        X: Any,
        y: Any,
        groups: Any = None,
        cv_strategy: str = "auto",
    ) -> "ACDSOMultiLabelClassifier":
        """Fit preprocessing, evaluate CV strategies, and refit the winning multi-label model."""
        strategy, scorer = self._validate_hyperparameters()
        df = self._coerce_input(X, is_fit=True)
        Y, label_names, y_unnamed = _coerce_multilabel_y_with_origin(y, len(df))

        self.classes_ = label_names
        self.n_labels_ = int(Y.shape[1])
        self._y_unnamed_ = bool(y_unnamed)

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

        splitter = build_cv_splitter(
            cv=self.cv,
            y=Y,
            groups=groups,
            task="multilabel",
            cv_strategy=cv_strategy,
            random_state=self.random_state,
        )
        eff_cv_name = resolve_cv_strategy_name(splitter)
        splits = iter_cv_splits(splitter, df_active, Y, groups, validate_groups_leakage=True)

        if strategy == "auto":
            cand_names = ["multi_output", "classifier_chain", "random_forest", "extra_trees"]
            if len(df_active) <= 5000 and len(num_cols) + len(cat_cols) <= 256:
                cand_names.append("multi_output_hgb")
        else:
            cand_names = [strategy]

        tie_order = {
            "multi_output": 5,
            "classifier_chain": 4,
            "random_forest": 3,
            "extra_trees": 2,
            "multi_output_hgb": 1,
        }

        # Pre-transform CV folds once so all candidates evaluate on identical leak-free splits
        fold_matrices: List[Tuple[Any, np.ndarray, Any, np.ndarray]] = []
        for tr_idx, va_idx in splits:
            fold_pre = _make_pre()
            X_tr_raw = fold_pre.fit_transform(df_active.iloc[tr_idx])
            if X_tr_raw.shape[1] == 0:
                raise ValueError("No valid features remain after preprocessing in CV fold.")
            X_va_raw = fold_pre.transform(df_active.iloc[va_idx])
            fold_matrices.append((X_tr_raw, Y[tr_idx], X_va_raw, Y[va_idx]))

        candidate_summaries: List[Dict[str, Any]] = []
        for name in cand_names:
            force_dense = name == "multi_output_hgb"
            fold_scores: List[float] = []
            fold_metrics_list: List[Dict[str, float]] = []
            try:
                for X_tr_raw, Y_tr, X_va_raw, Y_va in fold_matrices:
                    X_tr = self._normalize_matrix(X_tr_raw, force_dense=force_dense)
                    X_va = self._normalize_matrix(X_va_raw, force_dense=force_dense)
                    if force_dense and X_tr.shape[1] > 512:
                        raise ValueError("Too many preprocessed features for dense HGB candidate.")
                    est = self._build_candidate(name)
                    est.fit(X_tr, Y_tr)
                    preds_va = np.asarray(est.predict(X_va), dtype=np.int64)
                    m_fold = _compute_multilabel_metrics(Y_va, preds_va)
                    fold_metrics_list.append(m_fold)
                    fold_scores.append(self._objective_from_metrics(m_fold, scorer))

                mean_obj = float(np.mean(fold_scores))
                mean_metrics = {
                    k: float(np.mean([fm[k] for fm in fold_metrics_list]))
                    for k in fold_metrics_list[0]
                }
                candidate_summaries.append(
                    {
                        "strategy": name,
                        "cv_objective": mean_obj,
                        "fold_scores": fold_scores,
                        "cv_metrics": mean_metrics,
                    }
                )
            except Exception as exc:
                self._log(f"Candidate {name} failed during CV: {exc}")

        if not candidate_summaries:
            raise ValueError("All multi-label candidate strategies failed during cross-validation.")

        best_summary = max(
            candidate_summaries,
            key=lambda item: (float(item["cv_objective"]), tie_order.get(item["strategy"], 0)),
        )
        best_name = str(best_summary["strategy"])

        self.preprocessor_ = _make_pre()
        X_full_raw = self.preprocessor_.fit_transform(df_active)
        if X_full_raw.shape[1] == 0:
            raise ValueError("No valid features remain after preprocessing.")
        force_dense_best = best_name == "multi_output_hgb"
        X_full = self._normalize_matrix(X_full_raw, force_dense=force_dense_best)

        best_est = self._build_candidate(best_name)
        best_est.fit(X_full, Y)

        self.best_strategy_ = best_name
        self.best_estimator_ = best_est
        self._force_dense_ = force_dense_best
        self.candidates_ = _sanitize_json_obj(candidate_summaries)
        self.audit_trail_ = _sanitize_json_obj(
            {
                "strategy_requested": self.strategy,
                "best_strategy": self.best_strategy_,
                "scorer": scorer,
                "cv_strategy": eff_cv_name,
                "n_splits": len(splits),
                "n_samples": int(len(df_active)),
                "n_labels": int(self.n_labels_),
                "classes": self.classes_.tolist(),
                "n_features_preprocessed": int(X_full.shape[1]),
                "text_cols": list(eff_text_cols),
                "selected_candidate": best_summary,
                "candidates": candidate_summaries,
            }
        )
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "best_estimator_") or self.best_estimator_ is None:
            raise RuntimeError("ACDSOMultiLabelClassifier is not fitted. Call fit() first.")

    def _prepare_eval_matrix(self, X: Any) -> Any:
        self._require_fitted()
        df = self._coerce_input(X, is_fit=False)
        df_active = df[self._active_cols_]
        X_proc = self.preprocessor_.transform(df_active)
        return self._normalize_matrix(
            X_proc, force_dense=getattr(self, "_force_dense_", False)
        )

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary multi-label indicator matrix of shape ``(n_samples, n_labels)``."""
        X_mat = self._prepare_eval_matrix(X)
        raw_preds = np.asarray(self.best_estimator_.predict(X_mat), dtype=np.int64)
        if raw_preds.ndim == 1:
            raw_preds = raw_preds.reshape(-1, self.n_labels_)
        return np.clip(raw_preds, 0, 1).astype(np.int64, copy=False)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict positive-class probabilities of shape ``(n_samples, n_labels)`` in ``[0, 1]``."""
        X_mat = self._prepare_eval_matrix(X)
        return _extract_positive_proba(self.best_estimator_, X_mat, self.n_labels_)

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """Evaluate multi-label predictions on ``(X, y)`` returning bounded metrics in ``[0, 1]``."""
        self._require_fitted()
        Y_pred = self.predict(X)
        Y_true, _ = _coerce_multilabel_y(
            y,
            len(Y_pred),
            classes=self.classes_,
            allow_positional_df=getattr(self, "_y_unnamed_", False),
        )
        if Y_true.shape[1] != self.n_labels_:
            raise ValueError(
                f"Expected {self.n_labels_} label columns in y, got {Y_true.shape[1]}."
            )
        return _compute_multilabel_metrics(Y_true, Y_pred)

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return configured multi-label score (higher is better; ``1 - hamming_loss`` for hamming_loss)."""
        metrics = self.evaluate(X, y)
        return self._objective_from_metrics(metrics, self.scorer)
