"""
ACDSO - Trainable Unsupervised Anomaly & Outlier Detection Module (R2)
======================================================================

Provides ``ACDSOAnomalyDetector``, a scikit-learn compatible ``OutlierMixin``
estimator supporting Isolation Forest, Local Outlier Factor (``novelty=True``),
and One-Class SVM with automated candidate selection, mixed numeric/categorical/
free-text preprocessing, and consistent score polarity across all backends.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from acdso.utils import build_preprocessor, sanitize_features, split_columns

_ALGORITHM_ALIASES: Dict[str, str] = {
    "auto": "auto",
    "isolation_forest": "isolation_forest",
    "iforest": "isolation_forest",
    "lof": "lof",
    "local_outlier_factor": "lof",
    "ocsvm": "ocsvm",
    "one_class_svm": "ocsvm",
}


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


class ACDSOAnomalyDetector(BaseEstimator, OutlierMixin):
    """Trainable Unsupervised Anomaly & Outlier Detector for ACDSO.

    Supports ``"isolation_forest"`` (alias ``"iforest"``), ``"lof"``
    (``LocalOutlierFactor(novelty=True)``), ``"ocsvm"`` (``OneClassSVM``),
    and ``"auto"`` selection with mixed numeric/categorical/free-text inputs.

    Score semantics across all backends:
    - ``predict(X)``: ``1`` for normal inliers, ``-1`` for anomalies.
    - ``predict_is_anomaly(X)``: boolean array (``True`` when anomaly).
    - ``decision_function(X)``: signed score, **higher = more normal** (``< 0`` is anomaly).
    - ``score_samples(X)``: raw density/normality score, **higher = more normal**.
    - ``anomaly_scores(X)``: ``-decision_function(X)``, **higher = more anomalous**.

    Parameters
    ----------
    algorithm : str, default="auto"
        Detector algorithm: ``"auto"``, ``"isolation_forest"`` (``"iforest"``),
        ``"lof"``, or ``"ocsvm"``.
    contamination : str or float, default="auto"
        Expected proportion of outliers in ``(0.0, 0.5]`` or ``"auto"``.
    n_neighbors : int, default=20
        Number of neighbors for Local Outlier Factor (clamped to ``n_samples - 1``).
    nu : float, default=0.05
        Upper bound on fraction of training errors for One-Class SVM in ``(0.0, 1.0]``
        when ``contamination="auto"``.
    random_state : int, optional, default=42
        Random seed for reproducibility.
    text_cols : list of str, optional
        Explicit free-text column names. When ``None``, auto-detected via ``split_columns(detect_text=True)``.
    verbose : int, default=0
        Verbosity level.
    text_method : str, default="tfidf"
        Text featurization mode passed to ``TextFeaturizer``.
    """

    def __init__(
        self,
        algorithm: str = "auto",
        contamination: Union[str, float] = "auto",
        n_neighbors: int = 20,
        nu: float = 0.05,
        random_state: Optional[int] = 42,
        text_cols: Optional[List[str]] = None,
        verbose: int = 0,
        text_method: str = "tfidf",
    ) -> None:
        self.algorithm = algorithm
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.nu = nu
        self.random_state = random_state
        self.text_cols = text_cols
        self.verbose = verbose
        self.text_method = text_method

    def _log(self, msg: str) -> None:
        if self.verbose and int(self.verbose) > 0:
            print(f"[ACDSOAnomalyDetector] {msg}")

    def _validate_hyperparameters(self) -> Tuple[str, Union[str, float], int, float]:
        if not isinstance(self.algorithm, str):
            raise ValueError(f"algorithm must be a string, got {self.algorithm!r}.")
        canon_algo = _ALGORITHM_ALIASES.get(self.algorithm)
        if canon_algo is None:
            raise ValueError(
                f"Invalid algorithm={self.algorithm!r}. "
                f"Must be one of {tuple(_ALGORITHM_ALIASES.keys())}."
            )

        if self.contamination == "auto":
            eff_contam: Union[str, float] = "auto"
        else:
            if isinstance(self.contamination, bool) or not isinstance(
                self.contamination, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    f"contamination must be 'auto' or float in (0.0, 0.5], got {self.contamination!r}."
                )
            c_val = float(self.contamination)
            if not math.isfinite(c_val) or not (0.0 < c_val <= 0.5):
                raise ValueError(
                    f"contamination must be in (0.0, 0.5], got {self.contamination!r}."
                )
            eff_contam = c_val

        if isinstance(self.n_neighbors, bool) or not isinstance(self.n_neighbors, (int, np.integer)):
            raise ValueError(f"n_neighbors must be an integer >= 1, got {self.n_neighbors!r}.")
        eff_nn = int(self.n_neighbors)
        if eff_nn < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {self.n_neighbors!r}.")

        if isinstance(self.nu, bool) or not isinstance(self.nu, (int, float, np.integer, np.floating)):
            raise ValueError(f"nu must be a float in (0.0, 1.0], got {self.nu!r}.")
        eff_nu = float(self.nu)
        if not math.isfinite(eff_nu) or not (0.0 < eff_nu <= 1.0):
            raise ValueError(f"nu must be in (0.0, 1.0], got {self.nu!r}.")

        return canon_algo, eff_contam, eff_nn, eff_nu

    def _coerce_input(self, X: Any, *, is_fit: bool) -> pd.DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None.")

        if isinstance(X, pd.DataFrame):
            n_rows, n_cols = X.shape
            if is_fit:
                if n_rows < 2 or n_cols < 1:
                    raise ValueError(
                        f"ACDSOAnomalyDetector.fit requires at least 2 rows and 1 column, got shape {X.shape}."
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
                    if df.shape[1] != len(self._raw_feature_names_):
                        raise ValueError(
                            f"Expected {len(self._raw_feature_names_)} features, got {df.shape[1]}."
                        )
                else:
                    if n_cols != self.n_features_in_:
                        raise ValueError(
                            f"Expected {self.n_features_in_} features, got {n_cols}."
                        )
                    df = X.copy()
                    df.columns = self._internal_cols_
            return sanitize_features(df)

        if sparse.issparse(X):
            arr = X.toarray()
        else:
            arr = np.asarray(X)

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input X, got {arr.ndim}D.")
        n_rows, n_cols = arr.shape
        if is_fit:
            if n_rows < 2 or n_cols < 1:
                raise ValueError(
                    f"ACDSOAnomalyDetector.fit requires at least 2 rows and 1 column, got shape {arr.shape}."
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
    def _row_l2_norms(mat: Any) -> np.ndarray:
        if sparse.issparse(mat):
            return np.asarray(np.sqrt(mat.multiply(mat).sum(axis=1))).ravel().astype(np.float64)
        arr = np.asarray(mat, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return np.linalg.norm(arr, axis=1)

    def _reset_fitted_text_state(self) -> None:
        self._has_text_compactor_ = False
        self._text_svd_: Optional[TruncatedSVD] = None
        self._text_mu_unit_: Optional[np.ndarray] = None
        self._text_block_scale_: float = 1.0
        self._text_extra_mean_: Optional[np.ndarray] = None
        self._text_extra_scale_: Optional[np.ndarray] = None
        self._fit_is_sparse_: bool = False

    def _fit_and_apply_text_compactor(self, X_proc: Any) -> Any:
        output_indices = getattr(self.preprocessor_, "output_indices_", None)
        if not output_indices or "text" not in output_indices:
            self._has_text_compactor_ = False
            return X_proc

        text_sl = output_indices["text"]
        text_width = int(text_sl.stop - text_sl.start)
        if text_width <= 0:
            self._has_text_compactor_ = False
            return X_proc

        n_rows, n_total = X_proc.shape
        n_non_text = int(n_total - text_width)
        k_max = min(8, max(3, n_non_text * 2)) if n_non_text > 0 else 8

        blk = X_proc[:, text_sl]
        norms = self._row_l2_norms(blk)
        if sparse.issparse(blk):
            mu = np.asarray(blk.mean(axis=0)).ravel().astype(np.float64)
        else:
            mu = np.mean(np.asarray(blk, dtype=np.float64), axis=0).ravel()
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        mu_norm = float(np.linalg.norm(mu))
        mu_unit = (mu / mu_norm) if mu_norm > 1e-12 else mu
        self._text_mu_unit_ = mu_unit

        align = (blk @ mu_unit) if sparse.issparse(blk) else (np.asarray(blk, dtype=np.float64) @ mu_unit)
        align = np.nan_to_num(np.asarray(align, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)

        if text_width > k_max and n_rows > 1 and text_width > 1:
            k = min(k_max, n_rows - 1, text_width - 1)
            svd = TruncatedSVD(n_components=k, random_state=self.random_state)
            blk_red = np.asarray(svd.fit_transform(blk), dtype=np.float64)
            self._text_svd_ = svd
        else:
            self._text_svd_ = None
            blk_red = (
                blk.toarray().astype(np.float64, copy=False)
                if sparse.issparse(blk)
                else np.asarray(blk, dtype=np.float64)
            )
        blk_red = np.nan_to_num(blk_red, nan=0.0, posinf=0.0, neginf=0.0)

        blk_std = float(np.sqrt(np.sum(np.var(blk_red, axis=0)))) if blk_red.size > 0 else 1.0
        self._text_block_scale_ = 1.0 / max(blk_std, 1.0)
        blk_red = blk_red * self._text_block_scale_

        extra = np.column_stack([norms, align])
        self._text_extra_mean_ = np.mean(extra, axis=0)
        self._text_extra_scale_ = np.maximum(np.std(extra, axis=0), 0.20)
        extra_scaled = (extra - self._text_extra_mean_) / self._text_extra_scale_
        self._has_text_compactor_ = True

        return self._assemble_compacted_matrix(X_proc, blk_red, extra_scaled)

    def _apply_text_compactor(self, X_proc: Any) -> Any:
        if not getattr(self, "_has_text_compactor_", False):
            return X_proc

        text_sl = self.preprocessor_.output_indices_["text"]
        blk = X_proc[:, text_sl]
        norms = self._row_l2_norms(blk)
        mu_unit = self._text_mu_unit_
        align = (blk @ mu_unit) if sparse.issparse(blk) else (np.asarray(blk, dtype=np.float64) @ mu_unit)
        align = np.nan_to_num(np.asarray(align, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)

        if self._text_svd_ is not None:
            blk_red = np.asarray(self._text_svd_.transform(blk), dtype=np.float64)
        else:
            blk_red = (
                blk.toarray().astype(np.float64, copy=False)
                if sparse.issparse(blk)
                else np.asarray(blk, dtype=np.float64)
            )
        blk_red = np.nan_to_num(blk_red, nan=0.0, posinf=0.0, neginf=0.0) * float(self._text_block_scale_)

        extra = np.column_stack([norms, align])
        extra_scaled = (extra - self._text_extra_mean_) / self._text_extra_scale_

        return self._assemble_compacted_matrix(X_proc, blk_red, extra_scaled)

    def _assemble_compacted_matrix(
        self,
        X_proc: Any,
        blk_red: np.ndarray,
        extra_scaled: np.ndarray,
    ) -> Any:
        text_dense = np.nan_to_num(
            np.hstack([blk_red, extra_scaled]), nan=0.0, posinf=0.0, neginf=0.0
        )
        non_text_parts = [
            X_proc[:, sl]
            for t_name, sl in self.preprocessor_.output_indices_.items()
            if t_name != "text" and sl.stop > sl.start
        ]
        if not non_text_parts:
            return text_dense

        if any(sparse.issparse(p) for p in non_text_parts):
            X_nt = sparse.hstack(
                [p if sparse.issparse(p) else sparse.csr_matrix(p) for p in non_text_parts],
                format="csr",
            )
            if X_nt.shape[0] * (X_nt.shape[1] + text_dense.shape[1]) > 1_000_000:
                return sparse.hstack([X_nt, sparse.csr_matrix(text_dense)], format="csr")
            return np.hstack([X_nt.toarray().astype(np.float64, copy=False), text_dense])

        nt_dense = np.hstack([np.asarray(p, dtype=np.float64) for p in non_text_parts])
        return np.hstack([nt_dense, text_dense])

    @staticmethod
    def _normalize_matrix(X_proc: Any) -> Any:
        n_rows, n_cols = X_proc.shape
        if sparse.issparse(X_proc):
            if n_rows * n_cols <= 1_000_000:
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

    def _fit_candidate(
        self,
        name: str,
        X_mat: Any,
        eff_contam: Union[str, float],
        eff_nn: int,
        eff_nu: float,
    ) -> Any:
        n_samples = X_mat.shape[0]
        if name == "isolation_forest":
            est = IsolationForest(
                contamination=eff_contam,
                random_state=self.random_state,
            )
            X_fit = X_mat.tocsc() if sparse.issparse(X_mat) else X_mat
            est.fit(X_fit)
            return est
        if name == "lof":
            k = max(1, min(eff_nn, n_samples - 1))
            est = LocalOutlierFactor(
                n_neighbors=k,
                contamination=eff_contam,
                novelty=True,
            )
            X_fit = X_mat.tocsr() if sparse.issparse(X_mat) else X_mat
            est.fit(X_fit)
            return est
        if name == "ocsvm":
            oc_nu = float(eff_contam) if eff_contam != "auto" else eff_nu
            est = OneClassSVM(nu=oc_nu)
            X_fit = X_mat.tocsr() if sparse.issparse(X_mat) else X_mat
            est.fit(X_fit)
            return est
        raise ValueError(f"Unsupported candidate algorithm: {name!r}")

    def _score_candidate(
        self,
        name: str,
        est: Any,
        X_mat: Any,
        eff_contam: Union[str, float],
    ) -> Dict[str, Any]:
        X_eval = X_mat.tocsr() if sparse.issparse(X_mat) else X_mat
        n_rows = X_eval.shape[0]
        dec = np.asarray(est.decision_function(X_eval), dtype=np.float64)
        dec = np.nan_to_num(dec, nan=0.0, posinf=0.0, neginf=0.0)
        anom = -dec
        preds = np.asarray(est.predict(X_eval), dtype=int)
        anom_rate = float(np.mean(preds == -1))

        q = float(eff_contam) if eff_contam != "auto" else 0.05
        k_tail = max(1, min(n_rows - 1, int(round(q * n_rows))))
        sorted_anom = np.sort(anom)
        bulk = sorted_anom[:-k_tail]
        tail = sorted_anom[-k_tail:]
        bulk_std = max(float(np.std(bulk)), 1e-6)
        separation = float((np.mean(tail) - np.mean(bulk)) / bulk_std)

        penalty = 0.0
        if anom_rate == 0.0:
            penalty += 2.0
        elif anom_rate > max(0.30, 2.5 * q):
            penalty += 3.0 * (anom_rate - max(0.30, 2.5 * q))
        if eff_contam != "auto":
            penalty += 2.0 * abs(anom_rate - float(eff_contam))

        objective = float(separation - penalty)
        offset_val = float(np.asarray(est.offset_).ravel()[0])
        return {
            "algorithm": name,
            "objective": objective,
            "separation": separation,
            "anomaly_rate": anom_rate,
            "offset": offset_val,
            "threshold": float(-offset_val),
        }

    def fit(self, X: Any, y: Any = None) -> "ACDSOAnomalyDetector":
        """Fit preprocessing and unsupervised anomaly detection model on ``X``.

        Parameters
        ----------
        X : pd.DataFrame or array-like of shape (n_samples, n_features)
            Training observations.
        y : Ignored
            Not used, present for scikit-learn API consistency.

        Returns
        -------
        self : ACDSOAnomalyDetector
            Fitted detector instance.
        """
        canon_algo, eff_contam, eff_nn, eff_nu = self._validate_hyperparameters()
        self._reset_fitted_text_state()
        df = self._coerce_input(X, is_fit=True)
        n_rows, n_raw_cols = df.shape

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

        active_cols = [c for c in df.columns if not bool(df[c].isna().all())]
        if not active_cols:
            raise ValueError("No valid features remain after preprocessing (e.g. all-NaN columns).")
        self._active_cols_ = active_cols
        df = df[self._active_cols_]

        if self.text_cols is not None:
            active_set = set(self._active_cols_)
            eff_text_cols = [c for c in eff_text_cols if c in active_set]
            num_cols, cat_cols = split_columns(df, detect_text=False)
            text_set = set(eff_text_cols)
            num_cols = [c for c in num_cols if c not in text_set]
            cat_cols = [c for c in cat_cols if c not in text_set]
        else:
            num_cols, cat_cols, eff_text_cols = split_columns(df, detect_text=True)

        self.text_cols_ = list(eff_text_cols)
        self.preprocessor_ = build_preprocessor(
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            sparse_threshold=0.3,
            text_cols=eff_text_cols or None,
            text_method=self.text_method,
        )
        X_proc = self.preprocessor_.fit_transform(df)
        if X_proc.shape[1] == 0:
            raise ValueError("No valid features remain after preprocessing (e.g. all-NaN columns).")

        X_comp = self._fit_and_apply_text_compactor(X_proc)
        X_mat = self._normalize_matrix(X_comp)
        self._fit_is_sparse_ = bool(sparse.issparse(X_mat))

        if canon_algo == "auto":
            algo_names = ["isolation_forest"]
            if n_rows <= 10_000:
                algo_names.append("lof")
            if n_rows <= 5_000:
                algo_names.append("ocsvm")
        else:
            algo_names = [canon_algo]

        tie_order = {"isolation_forest": 3, "ocsvm": 2, "lof": 1}
        candidates: List[Dict[str, Any]] = []
        fitted: List[Tuple[Dict[str, Any], Any]] = []

        for name in algo_names:
            try:
                est = self._fit_candidate(name, X_mat, eff_contam, eff_nn, eff_nu)
                summary = self._score_candidate(name, est, X_mat, eff_contam)
                candidates.append(summary)
                fitted.append((summary, est))
            except Exception as exc:
                self._log(f"Candidate {name} failed: {exc}")

        if not fitted:
            raise ValueError("All anomaly detector candidates failed on the provided dataset.")

        best_summary, best_est = max(
            fitted,
            key=lambda item: (float(item[0]["objective"]), tie_order.get(item[0]["algorithm"], 0)),
        )

        self.best_algorithm_ = str(best_summary["algorithm"])
        self.best_estimator_ = best_est
        self.offset_ = float(best_summary["offset"])
        self.threshold_ = float(best_summary["threshold"])
        self.candidates_ = _sanitize_json_obj(candidates)
        self.audit_trail_ = _sanitize_json_obj(
            {
                "algorithm_requested": self.algorithm,
                "best_algorithm": self.best_algorithm_,
                "contamination": eff_contam,
                "n_neighbors": eff_nn,
                "nu": eff_nu,
                "n_samples": int(n_rows),
                "n_features_raw": int(n_raw_cols),
                "n_features_preprocessed": int(X_proc.shape[1]),
                "n_features_model": int(X_mat.shape[1]),
                "fit_is_sparse": bool(self._fit_is_sparse_),
                "text_cols": list(eff_text_cols),
                "offset": self.offset_,
                "threshold": self.threshold_,
                "selected_candidate": best_summary,
                "candidates_evaluated": candidates,
            }
        )
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "best_estimator_") or self.best_estimator_ is None:
            raise RuntimeError("ACDSOAnomalyDetector is not fitted. Call fit() first.")

    def _transform_to_model_space(self, X: Any) -> Any:
        self._require_fitted()
        df = self._coerce_input(X, is_fit=False)
        if df.shape[0] == 0:
            return None
        if hasattr(self, "_active_cols_"):
            df = df[self._active_cols_]
        X_proc = self.preprocessor_.transform(df)
        return self._apply_text_compactor(X_proc)

    @staticmethod
    def _ensure_finite_csr(X_mat: Any) -> sparse.csr_matrix:
        if sparse.issparse(X_mat):
            csr = X_mat.tocsr()
            if csr.data.size > 0 and not np.all(np.isfinite(csr.data)):
                csr = csr.copy()
                csr.data = np.nan_to_num(csr.data, nan=0.0, posinf=0.0, neginf=0.0)
            return csr
        arr = np.asarray(X_mat, dtype=np.float64)
        if arr.size > 0:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return sparse.csr_matrix(arr)

    @staticmethod
    def _ensure_finite_dense(X_mat: Any) -> np.ndarray:
        if sparse.issparse(X_mat):
            arr = X_mat.toarray().astype(np.float64, copy=False)
        else:
            arr = np.asarray(X_mat, dtype=np.float64)
        if arr.size > 0:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _prepare_eval_matrix(self, X: Any) -> Any:
        X_comp = self._transform_to_model_space(X)
        if X_comp is None:
            return None
        if getattr(self, "_fit_is_sparse_", False):
            return self._ensure_finite_csr(X_comp)
        return self._ensure_finite_dense(X_comp)

    def _eval_estimator(self, X: Any, method_name: str) -> np.ndarray:
        X_comp = self._transform_to_model_space(X)
        if X_comp is None:
            if method_name == "predict":
                return np.empty((0,), dtype=int)
            return np.empty((0,), dtype=np.float64)

        fn = getattr(self.best_estimator_, method_name)
        fit_sparse = bool(getattr(self, "_fit_is_sparse_", False))

        if fit_sparse:
            X_csr = self._ensure_finite_csr(X_comp)
            res = np.asarray(fn(X_csr))
        elif not sparse.issparse(X_comp):
            X_dense = self._ensure_finite_dense(X_comp)
            res = np.asarray(fn(X_dense))
        else:
            n_rows, n_cols = X_comp.shape
            if n_rows * n_cols <= 1_000_000:
                X_dense = self._ensure_finite_dense(X_comp)
                res = np.asarray(fn(X_dense))
            else:
                chunk_rows = max(1, 1_000_000 // max(1, n_cols))
                X_csr = X_comp.tocsr()
                chunks: List[np.ndarray] = []
                for start in range(0, n_rows, chunk_rows):
                    stop = min(start + chunk_rows, n_rows)
                    chunk_dense = self._ensure_finite_dense(X_csr[start:stop])
                    chunks.append(np.asarray(fn(chunk_dense)))
                res = np.concatenate(chunks, axis=0)

        if method_name == "predict":
            return res.astype(int, copy=False)
        return np.nan_to_num(res.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    def score_samples(self, X: Any) -> np.ndarray:
        """Return shifted normality scores (higher = more normal)."""
        return self._eval_estimator(X, "score_samples")

    def decision_function(self, X: Any) -> np.ndarray:
        """Return signed decision scores (higher = more normal, ``< 0`` = anomaly)."""
        return self._eval_estimator(X, "decision_function")

    def anomaly_scores(self, X: Any) -> np.ndarray:
        """Return anomaly scores (``-decision_function(X)``, higher = more anomalous)."""
        return -self.decision_function(X)

    def predict(self, X: Any) -> np.ndarray:
        """Predict ``1`` for normal inliers and ``-1`` for anomalies."""
        return self._eval_estimator(X, "predict")

    def predict_is_anomaly(self, X: Any) -> np.ndarray:
        """Predict boolean anomaly flags (``True`` for anomaly, ``False`` for normal)."""
        return np.asarray(self.predict(X) == -1, dtype=bool)

    def score(self, X: Any, y: Any = None) -> float:
        """Score observations on ``X``.

        When ``y is None``, returns mean ``decision_function(X)`` (higher = more normal).
        When ``y`` is provided (``{1, -1}`` inlier/anomaly or boolean/``{0, 1}`` anomaly
        labels), returns classification accuracy against predicted labels.
        """
        self._require_fitted()
        if y is None:
            dec = self.decision_function(X)
            return float(np.mean(dec)) if dec.size > 0 else 0.0

        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            if y_arr.ndim == 2 and y_arr.shape[1] == 1:
                y_arr = y_arr.ravel()
            else:
                raise ValueError(
                    f"Expected 1D target array y in score(X, y), got shape {y_arr.shape}."
                )
        preds = self.predict(X)
        if len(y_arr) != len(preds):
            raise ValueError(
                f"Length mismatch in score(X, y): X has {len(preds)} samples, y has {len(y_arr)}."
            )
        if preds.size == 0:
            return 0.0
        if y_arr.dtype == bool:
            preds_bool = preds == -1
            return float(np.mean(preds_bool == y_arr))
        unique_vals = set(np.unique(y_arr).tolist())
        if unique_vals.issubset({-1, 1}):
            return float(np.mean(preds == y_arr))
        if unique_vals.issubset({0, 1}):
            return float(np.mean((preds == -1).astype(int) == y_arr))
        dec = self.decision_function(X)
        return float(np.mean(dec)) if dec.size > 0 else 0.0
