"""
ACDSO - Unsupervised Clustering & Segmentation Module (R1)
========================================================

Provides ``ACDSOClusterer``, a scikit-learn compatible clustering estimator
supporting K-Means, inductive HDBSCAN, Gaussian Mixture Models (GMM), automated
model and cluster-count tuning via silhouette and Davies-Bouldin scores,
PCA / optional UMAP (with deterministic PCA fallback) dimensionality reduction,
and mixed numeric, categorical, and free-text preprocessing.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer

from acdso.utils import build_preprocessor, sanitize_features, split_columns

_VALID_ALGORITHMS = ("auto", "kmeans", "hdbscan", "gmm")
_VALID_DIM_REDUCTIONS = ("none", "pca", "umap")
_VALID_METRICS = ("silhouette", "davies_bouldin")


def _finite_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert scalar to finite Python float, or return ``default``."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


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


class _NumericSkewStabilizer(BaseEstimator, TransformerMixin):
    """Domain-agnostic, picklable skewness & dynamic-range stabilizer for continuous numeric columns.

    Inserted after median imputation and before ``StandardScaler`` in the numeric
    preprocessing pipeline so heavily skewed or multi-order-of-magnitude positive
    continuous columns do not compress lower-magnitude clusters under linear scaling
    when mixed with L2-normalized text or categorical blocks.
    """

    def __init__(
        self,
        skew_threshold: float = 1.0,
        range_ratio_threshold: float = 10.0,
        min_unique: int = 6,
    ) -> None:
        self.skew_threshold = skew_threshold
        self.range_ratio_threshold = range_ratio_threshold
        self.min_unique = min_unique

    def fit(self, X: Any, y: Any = None) -> "_NumericSkewStabilizer":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        n_rows, n_cols = X_arr.shape
        self.n_features_in_ = int(n_cols)
        self.modes_: Dict[int, str] = {}
        self.power_transformers_: Dict[int, PowerTransformer] = {}

        if n_rows < 3 or n_cols == 0:
            return self

        for j in range(n_cols):
            col = X_arr[:, j]
            finite = np.isfinite(col)
            if int(finite.sum()) < 3:
                continue
            vals = col[finite]
            if np.ptp(vals) <= 1e-12 or float(np.std(vals)) <= 1e-12:
                continue
            if len(np.unique(vals)) < int(self.min_unique):
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sk = _finite_float(stats.skew(vals), default=0.0) or 0.0

            cmin = float(np.min(vals))
            cmax = float(np.max(vals))
            q05, q95 = np.percentile(vals, [5.0, 95.0])
            is_pos_multiscale = (
                cmin >= 0.0
                and sk > 0.25
                and (
                    (cmin > 0.0 and (cmax / cmin) >= self.range_ratio_threshold)
                    or (q05 > 0.0 and (q95 / q05) >= self.range_ratio_threshold)
                )
            )
            is_heavy_skew = abs(sk) > self.skew_threshold
            if not (is_pos_multiscale or is_heavy_skew):
                continue

            if cmin >= 0.0 and sk > 0.0:
                cand = np.sign(vals) * np.log1p(np.abs(vals))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cand_sk = _finite_float(stats.skew(cand), default=abs(sk))
                if cand_sk is not None and abs(cand_sk) < abs(sk) and np.ptp(cand) > 1e-12:
                    self.modes_[j] = "signed_log1p"
                    continue

            best_mode: Optional[str] = None
            best_abs_sk = abs(sk)
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cand_pt = pt.fit_transform(vals.reshape(-1, 1)).ravel()
                if np.all(np.isfinite(cand_pt)) and np.ptp(cand_pt) > 1e-12:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        pt_sk = _finite_float(stats.skew(cand_pt), default=best_abs_sk)
                    if pt_sk is not None and abs(pt_sk) < best_abs_sk:
                        best_mode = "yeo_johnson"
                        best_abs_sk = abs(pt_sk)
                        self.power_transformers_[j] = pt
            except Exception:
                pass

            cand_sl = np.sign(vals) * np.log1p(np.abs(vals))
            if np.all(np.isfinite(cand_sl)) and np.ptp(cand_sl) > 1e-12:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sl_sk = _finite_float(stats.skew(cand_sl), default=best_abs_sk)
                if sl_sk is not None and abs(sl_sk) < best_abs_sk:
                    best_mode = "signed_log1p"
                    self.power_transformers_.pop(j, None)

            if best_mode is not None:
                self.modes_[j] = best_mode

        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.array(X, dtype=np.float64, copy=True)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0 or not getattr(self, "modes_", None):
            return X_arr
        n_cols = X_arr.shape[1]
        for j, mode in self.modes_.items():
            if j >= n_cols:
                continue
            col = X_arr[:, j]
            if mode == "signed_log1p":
                X_arr[:, j] = np.sign(col) * np.log1p(np.abs(col))
            elif mode == "yeo_johnson":
                pt = self.power_transformers_.get(j)
                if pt is not None:
                    tr = pt.transform(col.reshape(-1, 1)).ravel()
                    X_arr[:, j] = np.nan_to_num(tr, nan=0.0, posinf=0.0, neginf=0.0)
        return X_arr

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return np.asarray(self.feature_names_in_, dtype=object)
        n = getattr(self, "n_features_in_", 0)
        return np.asarray([f"x{i}" for i in range(n)], dtype=object)


class _InductiveHDBSCAN(BaseEstimator, ClusterMixin):
    """Picklable wrapper around ``sklearn.cluster.HDBSCAN`` with inductive ``predict``.

    ``sklearn.cluster.HDBSCAN`` (1.5.2) does not provide ``.predict()`` on unseen
    samples. This wrapper stores the non-noise training instances in a 1-NN index
    along with per-cluster distance thresholds and cluster centroids so unseen
    samples can be assigned to existing clusters or labeled ``-1`` (noise).
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def fit(self, X: Any, y: Any = None) -> "_InductiveHDBSCAN":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] < 2:
            raise ValueError("HDBSCAN requires a 2D array with at least 2 samples.")
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples = X_arr.shape[0]
        eff_mcs = max(2, min(int(self.min_cluster_size), n_samples))
        eff_ms = (
            max(1, min(int(self.min_samples), n_samples))
            if self.min_samples is not None
            else min(eff_mcs, n_samples)
        )

        hdb = HDBSCAN(
            min_cluster_size=eff_mcs,
            min_samples=eff_ms,
            store_centers=None,
        )
        raw_labels = np.asarray(hdb.fit_predict(X_arr), dtype=int)
        # Map special HDBSCAN negative codes (-2 inf, -3 nan) to -1 noise if any
        labels = np.where(raw_labels < 0, -1, raw_labels)

        self.hdbscan_ = hdb
        self.labels_ = labels
        self.n_features_in_ = X_arr.shape[1]

        non_noise_mask = labels >= 0
        cluster_ids = sorted(int(c) for c in np.unique(labels[non_noise_mask])) if non_noise_mask.any() else []
        self.cluster_ids_ = cluster_ids
        self.n_clusters_ = len(cluster_ids)

        if self.n_clusters_ == 0:
            self.nn_ = None
            self.non_noise_labels_ = np.empty((0,), dtype=int)
            self.centroids_ = np.empty((0, X_arr.shape[1]), dtype=np.float64)
            self.noise_thresholds_: Dict[int, float] = {}
            return self

        X_valid = X_arr[non_noise_mask]
        y_valid = labels[non_noise_mask]
        self.non_noise_labels_ = y_valid
        self.nn_ = NearestNeighbors(n_neighbors=1).fit(X_valid)

        centroids = []
        thresholds: Dict[int, float] = {}
        for cid in cluster_ids:
            pts = X_arr[labels == cid]
            centroid = np.mean(pts, axis=0)
            centroids.append(centroid)
            centroid_dists = np.linalg.norm(pts - centroid, axis=1)
            max_centroid_dist = float(np.max(centroid_dists)) if len(pts) > 0 else 0.0
            if len(pts) >= 2:
                nn_intra = NearestNeighbors(n_neighbors=2).fit(pts)
                dists_intra, _ = nn_intra.kneighbors(pts)
                intra_nn = dists_intra[:, 1]
                p95_intra = float(np.percentile(intra_nn, 95))
                max_intra = float(np.max(intra_nn))
                thresh = max(3.5 * p95_intra, 2.5 * max_intra, 1.5 * max_centroid_dist, 1e-6)
            else:
                thresh = max(1.5 * max_centroid_dist, 1e-6)
            thresholds[cid] = float(thresh)

        self.centroids_ = np.vstack(centroids)
        self.noise_thresholds_ = thresholds
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "labels_"):
            raise RuntimeError("_InductiveHDBSCAN is not fitted. Call fit() first.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[0] == 0:
            return np.empty((0,), dtype=int)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self.n_clusters_ == 0 or self.nn_ is None:
            return np.full(X_arr.shape[0], -1, dtype=int)

        dists, idxs = self.nn_.kneighbors(X_arr, n_neighbors=1)
        nearest_dists = dists[:, 0]
        nearest_labels = self.non_noise_labels_[idxs[:, 0]]
        out = np.empty(X_arr.shape[0], dtype=int)
        for i in range(X_arr.shape[0]):
            lbl = int(nearest_labels[i])
            thresh = self.noise_thresholds_.get(lbl, float("inf"))
            out[i] = lbl if nearest_dists[i] <= thresh else -1
        return out


class ACDSOClusterer(BaseEstimator, ClusterMixin):
    """Unsupervised Clustering & Segmentation Estimator for ACDSO.

    Supports K-Means, inductive HDBSCAN, and Gaussian Mixture Models (GMM),
    automated algorithm and cluster-count tuning using silhouette or
    Davies-Bouldin scores, optional PCA or UMAP (with deterministic PCA fallback)
    dimensionality reduction, and mixed numeric/categorical/free-text inputs.

    Parameters
    ----------
    algorithm : str, default="auto"
        Clustering algorithm: ``"auto"``, ``"kmeans"``, ``"hdbscan"``, or ``"gmm"``.
    n_clusters : int, optional
        Fixed number of clusters for K-Means / GMM. When ``None``, tunes over ``k_range``.
    k_range : tuple of (int, int), default=(2, 8)
        Inclusive ``(k_min, k_max)`` range searched when ``n_clusters is None``.
    dim_reduction : str, default="none"
        Dimensionality reduction method: ``"none"``, ``"pca"``, or ``"umap"``.
    n_components : int or float, optional
        Number of reduced dimensions (or variance fraction for PCA when float in (0, 1)).
    metric : str, default="silhouette"
        Model selection metric: ``"silhouette"`` (maximize) or ``"davies_bouldin"`` (minimize).
    min_cluster_size : int, default=5
        Minimum cluster size for HDBSCAN.
    random_state : int, optional, default=42
        Random seed for reproducibility.
    text_cols : list of str, optional
        Explicit free-text column names. When ``None``, auto-detected via ``split_columns(detect_text=True)``.
    text_method : str, default="tfidf"
        Text featurization mode passed to ``TextFeaturizer``.
    verbose : int, default=0
        Verbosity level.
    """

    def __init__(
        self,
        algorithm: str = "auto",
        n_clusters: Optional[int] = None,
        k_range: Tuple[int, int] = (2, 8),
        dim_reduction: str = "none",
        n_components: Optional[Union[int, float]] = None,
        metric: str = "silhouette",
        min_cluster_size: int = 5,
        random_state: Optional[int] = 42,
        text_cols: Optional[List[str]] = None,
        text_method: str = "tfidf",
        verbose: int = 0,
    ) -> None:
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.k_range = k_range
        self.dim_reduction = dim_reduction
        self.n_components = n_components
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.text_cols = text_cols
        self.text_method = text_method
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose and int(self.verbose) > 0:
            print(f"[ACDSOClusterer] {msg}")

    def _validate_hyperparameters(self) -> Tuple[str, str, str, Tuple[int, int], int]:
        if not isinstance(self.algorithm, str) or self.algorithm not in _VALID_ALGORITHMS:
            raise ValueError(
                f"Invalid algorithm={self.algorithm!r}. Must be one of {_VALID_ALGORITHMS}."
            )
        if not isinstance(self.dim_reduction, str) or self.dim_reduction not in _VALID_DIM_REDUCTIONS:
            raise ValueError(
                f"Invalid dim_reduction={self.dim_reduction!r}. Must be one of {_VALID_DIM_REDUCTIONS}."
            )
        if not isinstance(self.metric, str) or self.metric not in _VALID_METRICS:
            raise ValueError(
                f"Invalid metric={self.metric!r}. Must be one of {_VALID_METRICS}."
            )

        if self.n_clusters is not None:
            if isinstance(self.n_clusters, bool) or not isinstance(self.n_clusters, (int, np.integer)):
                raise ValueError(f"n_clusters must be an integer >= 2, got {self.n_clusters!r}.")
            if int(self.n_clusters) < 2:
                raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters!r}.")

        if (
            not isinstance(self.k_range, (tuple, list))
            or len(self.k_range) != 2
            or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) for k in self.k_range)
        ):
            raise ValueError(f"k_range must be a 2-tuple of integers (k_min, k_max), got {self.k_range!r}.")
        k_min, k_max = int(self.k_range[0]), int(self.k_range[1])
        if k_min < 2 or k_min > k_max:
            raise ValueError(
                f"k_range must satisfy 2 <= k_min <= k_max, got ({k_min}, {k_max})."
            )

        if isinstance(self.min_cluster_size, bool) or not isinstance(self.min_cluster_size, (int, np.integer)):
            raise ValueError(f"min_cluster_size must be an integer >= 2, got {self.min_cluster_size!r}.")
        eff_mcs = int(self.min_cluster_size)
        if eff_mcs < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {self.min_cluster_size!r}.")

        if self.n_components is not None:
            if isinstance(self.n_components, bool):
                raise ValueError("n_components cannot be a boolean.")
            if isinstance(self.n_components, (int, np.integer)):
                if int(self.n_components) < 1:
                    raise ValueError(f"n_components must be >= 1 when int, got {self.n_components!r}.")
            elif isinstance(self.n_components, (float, np.floating)):
                val = float(self.n_components)
                if not math.isfinite(val) or not (0.0 < val < 1.0):
                    raise ValueError(
                        f"n_components must be in (0, 1) when float, got {self.n_components!r}."
                    )
            else:
                raise ValueError(f"Invalid n_components={self.n_components!r}.")

        return self.algorithm, self.dim_reduction, self.metric, (k_min, k_max), eff_mcs

    def _coerce_input(self, X: Any, *, is_fit: bool) -> pd.DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None.")

        if isinstance(X, pd.DataFrame):
            n_rows, n_cols = X.shape
            if is_fit:
                if n_rows < 2 or n_cols < 1:
                    raise ValueError(
                        f"ACDSOClusterer.fit requires at least 2 rows and 1 column, got shape {X.shape}."
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
                    f"ACDSOClusterer.fit requires at least 2 rows and 1 column, got shape {arr.shape}."
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

    def _to_dense_finite(self, X_proc: Any, *, fit_svd: bool = False) -> np.ndarray:
        if sparse.issparse(X_proc):
            n_rows, n_cols = X_proc.shape
            if fit_svd:
                k_svd = min(64, n_rows - 1, n_cols - 1) if (n_rows > 1 and n_cols > 1) else 0
                if n_rows * n_cols > 20_000_000 and k_svd >= 1:
                    self._sparse_svd_ = TruncatedSVD(n_components=k_svd, random_state=self.random_state)
                    arr = np.asarray(self._sparse_svd_.fit_transform(X_proc), dtype=np.float64)
                else:
                    self._sparse_svd_ = None
                    arr = X_proc.toarray().astype(np.float64, copy=False)
            elif getattr(self, "_sparse_svd_", None) is not None:
                arr = np.asarray(self._sparse_svd_.transform(X_proc), dtype=np.float64)
            else:
                arr = X_proc.toarray().astype(np.float64, copy=False)
        else:
            arr = np.asarray(X_proc, dtype=np.float64)
            if fit_svd:
                self._sparse_svd_ = None
            elif getattr(self, "_sparse_svd_", None) is not None:
                arr = np.asarray(self._sparse_svd_.transform(arr), dtype=np.float64)
        if arr.size > 0:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _fit_reducer(self, X_dense: np.ndarray, dim_reduction: str) -> Tuple[np.ndarray, Optional[Any], str]:
        n_rows, n_feat = X_dense.shape
        if dim_reduction == "none":
            return X_dense, None, "none"

        has_variance = n_feat > 0 and n_rows > 0 and bool(np.any(np.ptp(X_dense, axis=0) > 1e-12))
        if self.n_components is None:
            if dim_reduction == "umap":
                eff_n: Union[int, float] = max(1, min(2, n_rows, n_feat))
            else:
                eff_n = max(1, min(32, n_rows, n_feat))
        elif isinstance(self.n_components, (float, np.floating)):
            eff_n = float(self.n_components) if (min(n_rows, n_feat) >= 2 and has_variance) else 1
        else:
            eff_n = max(1, min(int(self.n_components), n_rows, n_feat))

        if dim_reduction == "pca":
            pca = PCA(n_components=eff_n, random_state=self.random_state)
            with warnings.catch_warnings():
                if not has_variance:
                    warnings.simplefilter("ignore", RuntimeWarning)
                X_red = np.asarray(pca.fit_transform(X_dense), dtype=np.float64)
            return np.nan_to_num(X_red, nan=0.0, posinf=0.0, neginf=0.0), pca, "pca"

        # dim_reduction == "umap"
        umap_n = eff_n if isinstance(eff_n, int) else max(1, min(2, n_rows, n_feat))
        if n_rows >= 4:
            try:
                import umap  # type: ignore[import-not-found]

                reducer = umap.UMAP(
                    n_components=umap_n,
                    n_neighbors=max(2, min(15, n_rows - 1)),
                    random_state=self.random_state,
                )
                X_red = np.asarray(reducer.fit_transform(X_dense), dtype=np.float64)
                return np.nan_to_num(X_red, nan=0.0, posinf=0.0, neginf=0.0), reducer, "umap"
            except Exception:
                pass

        pca_fallback = PCA(n_components=eff_n, random_state=self.random_state)
        with warnings.catch_warnings():
            if not has_variance:
                warnings.simplefilter("ignore", RuntimeWarning)
            X_red = np.asarray(pca_fallback.fit_transform(X_dense), dtype=np.float64)
        return np.nan_to_num(X_red, nan=0.0, posinf=0.0, neginf=0.0), pca_fallback, "pca_fallback"

    @staticmethod
    def _evaluate_labels(X_eval: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        labels_arr = np.asarray(labels, dtype=int)
        n_total = len(labels_arr)
        valid_mask = labels_arr >= 0
        n_valid = int(valid_mask.sum())
        noise_count = int(n_total - n_valid)
        noise_ratio = float(noise_count / n_total) if n_total > 0 else 0.0

        unique_valid = np.unique(labels_arr[valid_mask]) if n_valid > 0 else np.empty((0,), dtype=int)
        n_clusters = int(len(unique_valid))

        sil: Optional[float] = None
        db: Optional[float] = None
        if n_clusters >= 2 and 2 <= n_clusters < n_valid:
            X_v = X_eval[valid_mask]
            y_v = labels_arr[valid_mask]
            if np.any(np.ptp(X_v, axis=0) > 1e-12):
                try:
                    sil = _finite_float(silhouette_score(X_v, y_v))
                except Exception:
                    sil = None
                try:
                    db = _finite_float(davies_bouldin_score(X_v, y_v))
                except Exception:
                    db = None

        return {
            "n_clusters": n_clusters,
            "n_valid": n_valid,
            "noise_count": noise_count,
            "noise_ratio": noise_ratio,
            "silhouette_score": sil,
            "davies_bouldin_score": db,
        }

    def _fit_gmm(self, X_eval: np.ndarray, k: int) -> Tuple[GaussianMixture, np.ndarray]:
        for cov_type, reg in (("full", 1e-6), ("diag", 1e-4), ("diag", 1e-2)):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    reg_covar=reg,
                    random_state=self.random_state,
                )
                labels = np.asarray(gmm.fit_predict(X_eval), dtype=int)
                return gmm, labels
            except (np.linalg.LinAlgError, ValueError):
                continue
        gmm = GaussianMixture(
            n_components=1,
            covariance_type="diag",
            reg_covar=1e-2,
            random_state=self.random_state,
        )
        labels = np.asarray(gmm.fit_predict(X_eval), dtype=int)
        return gmm, labels

    def fit(self, X: Any, y: Any = None) -> "ACDSOClusterer":
        """Fit preprocessing, optional dimensionality reduction, and clustering model.

        Parameters
        ----------
        X : pd.DataFrame or array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for scikit-learn API consistency.

        Returns
        -------
        self : ACDSOClusterer
            Fitted clusterer instance.
        """
        algorithm, dim_reduction, metric, (k_min, k_max), eff_mcs = self._validate_hyperparameters()
        self._sparse_svd_ = None

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
        if num_cols:
            for t_name, t_obj, _ in self.preprocessor_.transformers:
                if t_name == "num" and hasattr(t_obj, "steps"):
                    t_obj.steps.insert(-1, ("skew", _NumericSkewStabilizer()))
                    break
        X_proc = self.preprocessor_.fit_transform(df)
        if X_proc.shape[1] == 0:
            raise ValueError("No valid features remain after preprocessing (e.g. all-NaN columns).")

        X_dense = self._to_dense_finite(X_proc, fit_svd=True)
        X_red, reducer, dim_backend = self._fit_reducer(X_dense, dim_reduction)
        self.reducer_ = reducer
        self._dim_reduction_backend_ = dim_backend

        has_variance = bool(X_red.shape[1] > 0 and np.any(np.ptp(X_red, axis=0) > 1e-12))
        n_unique_rows = int(np.unique(X_red, axis=0).shape[0]) if X_red.shape[1] > 0 else 1

        # Determine k sweep values for kmeans / gmm
        if self.n_clusters is not None:
            k_req = int(self.n_clusters)
            if k_req > n_rows:
                raise ValueError(
                    f"n_clusters={k_req} cannot exceed n_samples={n_rows}."
                )
            ks = [k_req]
        elif not has_variance:
            ks = [1]
        else:
            k_upper = min(k_max, max(2, min(n_rows - 1, n_unique_rows)))
            k_lower = min(k_min, k_upper)
            ks = list(range(k_lower, k_upper + 1))

        # Generate and evaluate candidates
        candidate_records: List[Dict[str, Any]] = []
        fitted_candidates: List[Tuple[Dict[str, Any], Any, np.ndarray]] = []

        if algorithm in ("auto", "kmeans"):
            for k in ks:
                try:
                    km = KMeans(n_clusters=k, n_init="auto", random_state=self.random_state)
                    if has_variance and k <= n_unique_rows:
                        labels = np.asarray(km.fit_predict(X_red), dtype=int)
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=ConvergenceWarning)
                            labels = np.asarray(km.fit_predict(X_red), dtype=int)
                    stats = self._evaluate_labels(X_red, labels)
                    rec = {"algorithm": "kmeans", "k": int(k), **stats}
                    candidate_records.append(rec)
                    fitted_candidates.append((rec, km, labels))
                except Exception as exc:
                    self._log(f"kmeans(k={k}) failed: {exc}")

        if algorithm in ("auto", "gmm"):
            for k in ks:
                try:
                    if has_variance and k <= n_unique_rows:
                        gmm, labels = self._fit_gmm(X_red, k)
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=ConvergenceWarning)
                            gmm, labels = self._fit_gmm(X_red, k)
                    stats = self._evaluate_labels(X_red, labels)
                    rec = {"algorithm": "gmm", "k": int(gmm.n_components), **stats}
                    candidate_records.append(rec)
                    fitted_candidates.append((rec, gmm, labels))
                except Exception as exc:
                    self._log(f"gmm(k={k}) failed: {exc}")

        if algorithm in ("auto", "hdbscan"):
            if algorithm == "hdbscan" or n_rows >= max(5, eff_mcs):
                try:
                    hdb = _InductiveHDBSCAN(min_cluster_size=max(2, min(eff_mcs, n_rows)))
                    labels = np.asarray(hdb.fit_predict(X_red), dtype=int)
                    stats = self._evaluate_labels(X_red, labels)
                    rec = {"algorithm": "hdbscan", "k": stats["n_clusters"], **stats}
                    candidate_records.append(rec)
                    fitted_candidates.append((rec, hdb, labels))
                except Exception as exc:
                    self._log(f"hdbscan failed: {exc}")

        if not fitted_candidates:
            raise ValueError("All clustering candidates failed on the provided dataset.")

        # Rank candidates
        def _selection_key(item: Tuple[Dict[str, Any], Any, np.ndarray]) -> Tuple[int, float]:
            rec = item[0]
            sil = rec["silhouette_score"]
            db = rec["davies_bouldin_score"]
            n_cl = rec["n_clusters"]
            noise_ratio = rec["noise_ratio"]
            if metric == "silhouette":
                if n_cl >= 2 and sil is not None:
                    # Penalize excessive noise so degenerate sparse clusters don't beat clean partitions
                    penalty = 0.35 * noise_ratio if noise_ratio > 0.15 else 0.0
                    return (1, float(sil - penalty))
                return (0, -float(noise_ratio))
            else:
                if n_cl >= 2 and db is not None:
                    penalty = 2.0 * noise_ratio if noise_ratio > 0.15 else 0.0
                    return (1, -float(db + penalty))
                return (0, -float(noise_ratio))

        best_rec, best_est, best_labels = max(fitted_candidates, key=_selection_key)

        self.best_algorithm_ = str(best_rec["algorithm"])
        self.best_estimator_ = best_est
        self.labels_ = np.asarray(best_labels, dtype=int)
        self.n_clusters_ = int(best_rec["n_clusters"])
        self.silhouette_score_ = (
            float(best_rec["silhouette_score"])
            if best_rec["silhouette_score"] is not None
            else 0.0
        )
        self.davies_bouldin_score_ = (
            float(best_rec["davies_bouldin_score"])
            if best_rec["davies_bouldin_score"] is not None
            else 0.0
        )
        self.candidates_ = _sanitize_json_obj(candidate_records)

        # Compute centroids in clustered space for summary
        centroids: Dict[str, List[float]] = {}
        for cid in sorted(int(c) for c in np.unique(self.labels_) if c >= 0):
            pts = X_red[self.labels_ == cid]
            centroids[str(cid)] = [float(v) for v in np.mean(pts, axis=0)]
        self._centroids_ = centroids

        self.audit_trail_ = _sanitize_json_obj(
            {
                "algorithm_requested": algorithm,
                "best_algorithm": self.best_algorithm_,
                "metric": metric,
                "dim_reduction_requested": dim_reduction,
                "dim_reduction_backend": dim_backend,
                "n_samples": int(n_rows),
                "n_features_raw": int(n_raw_cols),
                "n_features_preprocessed": int(X_proc.shape[1]),
                "n_features_clustered": int(X_red.shape[1]),
                "text_cols": list(eff_text_cols),
                "n_clusters": self.n_clusters_,
                "noise_count": int(best_rec["noise_count"]),
                "noise_ratio": float(best_rec["noise_ratio"]),
                "silhouette_score": self.silhouette_score_,
                "davies_bouldin_score": self.davies_bouldin_score_,
                "selected_candidate": best_rec,
                "candidates_evaluated": candidate_records,
            }
        )
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "best_estimator_") or self.best_estimator_ is None:
            raise RuntimeError("ACDSOClusterer is not fitted. Call fit() first.")

    def transform(self, X: Any) -> np.ndarray:
        """Preprocess and optionally project ``X`` into the clustering feature space."""
        self._require_fitted()
        df = self._coerce_input(X, is_fit=False)
        if df.shape[0] == 0:
            n_out = self.audit_trail_.get("n_features_clustered", 0)
            return np.empty((0, int(n_out)), dtype=np.float64)
        if hasattr(self, "_active_cols_"):
            df = df[self._active_cols_]
        X_proc = self.preprocessor_.transform(df)
        X_dense = self._to_dense_finite(X_proc, fit_svd=False)
        if self.reducer_ is not None:
            X_red = np.asarray(self.reducer_.transform(X_dense), dtype=np.float64)
            return np.nan_to_num(X_red, nan=0.0, posinf=0.0, neginf=0.0)
        return X_dense

    def predict(self, X: Any) -> np.ndarray:
        """Assign cluster labels to ``X`` using the fitted pipeline."""
        self._require_fitted()
        X_red = self.transform(X)
        if X_red.shape[0] == 0:
            return np.empty((0,), dtype=int)
        return np.asarray(self.best_estimator_.predict(X_red), dtype=int)

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit the clusterer on ``X`` and return training cluster labels."""
        self.fit(X, y=y)
        return self.labels_.copy()

    def score(self, X: Any, y: Any = None) -> float:
        """Evaluate clustering quality on ``X`` using the configured ``metric``.

        Returns silhouette score (higher is better) when ``metric="silhouette"``,
        or negative Davies-Bouldin score (higher is better) when
        ``metric="davies_bouldin"``. Returns ``0.0`` when fewer than 2 valid
        non-noise clusters exist.
        """
        self._require_fitted()
        X_red = self.transform(X)
        if X_red.shape[0] < 3:
            return 0.0
        labels = np.asarray(self.best_estimator_.predict(X_red), dtype=int)
        stats = self._evaluate_labels(X_red, labels)
        if self.metric == "silhouette":
            val = stats["silhouette_score"]
            return float(val) if val is not None else 0.0
        val = stats["davies_bouldin_score"]
        return -float(val) if val is not None else 0.0

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of fitted clusters."""
        self._require_fitted()
        unique, counts = np.unique(self.labels_, return_counts=True)
        cluster_sizes = {str(int(k)): int(v) for k, v in zip(unique, counts) if int(k) >= 0}
        noise_count = int(np.sum(self.labels_ < 0))
        n_samples = int(len(self.labels_))
        return _sanitize_json_obj(
            {
                "best_algorithm": self.best_algorithm_,
                "n_clusters": int(self.n_clusters_),
                "n_samples": n_samples,
                "noise_count": noise_count,
                "noise_ratio": float(noise_count / n_samples) if n_samples > 0 else 0.0,
                "cluster_sizes": cluster_sizes,
                "silhouette_score": float(self.silhouette_score_),
                "davies_bouldin_score": float(self.davies_bouldin_score_),
                "dim_reduction": self.dim_reduction,
                "dim_reduction_backend": getattr(self, "_dim_reduction_backend_", "none"),
                "centroids": getattr(self, "_centroids_", {}),
            }
        )
