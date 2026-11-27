"""
ACDSO - Survival / Censored Time-to-Event Analysis Module (R5c)
===============================================================

Provides Harrell's ``concordance_index`` (O(n log n) Fenwick-tree
implementation) and ``ACDSOSurvivalAnalyzer``, a scikit-learn compatible Cox
Proportional Hazards estimator using an overflow-free Breslow partial
likelihood L-BFGS-B solver (with optional ``lifelines`` backend), Breslow
baseline cumulative hazard / monotonic survival curves, entity-grouped CV, and
free-text NLP featurization.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, sparse
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD

from acdso.cv import (
    build_cv_splitter,
    iter_cv_splits,
    resolve_cv_strategy_name,
    validate_groups,
    validate_no_group_leakage,
)
from acdso.utils import build_preprocessor, sanitize_features, split_columns

_VALID_BACKENDS = ("auto", "scipy", "lifelines")


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


def _coerce_1d_survival_vec(arr_like: Any, name: str) -> np.ndarray:
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


def _validate_time_event(time: Any, event: Any, n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    t_arr = _coerce_1d_survival_vec(time, "time")
    e_arr = _coerce_1d_survival_vec(event, "event")
    if len(t_arr) != len(e_arr):
        raise ValueError(
            f"time ({len(t_arr)}) and event ({len(e_arr)}) must have equal length."
        )
    if n_samples is not None and len(t_arr) != int(n_samples):
        raise ValueError(
            f"X ({int(n_samples)}) and survival targets ({len(t_arr)}) must have equal length."
        )
    if np.any(t_arr < 0.0):
        raise ValueError("event_times cannot contain negative values.")
    if not np.all((e_arr == 0.0) | (e_arr == 1.0)):
        raise ValueError("event_observed must contain only binary values {0, 1}.")
    return t_arr, e_arr.astype(np.int64, copy=False)


class _FenwickTree:
    """1-indexed Binary Indexed Tree for O(log M) prefix counts."""

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.tree = np.zeros(self.size + 1, dtype=np.int64)

    def add(self, idx: int, delta: int = 1) -> None:
        n = self.size
        tree = self.tree
        while idx <= n:
            tree[idx] += delta
            idx += idx & -idx

    def query(self, idx: int) -> int:
        s = 0
        tree = self.tree
        while idx > 0:
            s += int(tree[idx])
            idx -= idx & -idx
        return s


def concordance_index(
    event_times: Any,
    predicted_scores: Any,
    event_observed: Any,
) -> float:
    """Compute Harrell's C-index in O(n log n) via a descending-time Fenwick tree.

    Polarity convention: higher ``predicted_scores`` indicate higher hazard /
    risk and therefore earlier expected event times.

    Returns ``0.5`` when zero comparable pairs exist (e.g. all-censored inputs).
    """
    t_arr, e_arr = _validate_time_event(event_times, event_observed)
    s_arr = _coerce_1d_survival_vec(predicted_scores, "predicted_scores")
    if len(s_arr) != len(t_arr):
        raise ValueError(
            f"predicted_scores ({len(s_arr)}) must match event_times ({len(t_arr)}) length."
        )

    n = len(t_arr)
    if n < 2 or int(np.sum(e_arr)) == 0:
        return 0.5

    _, s_ranks = np.unique(s_arr, return_inverse=True)
    ranks_1b = s_ranks.astype(np.intp) + 1
    m_ranks = int(ranks_1b.max())

    bit = _FenwickTree(m_ranks)
    freq = np.zeros(m_ranks + 1, dtype=np.int64)

    # Sort descending by time so each unique time block enters the risk set together
    order = np.argsort(-t_arr, kind="mergesort")
    t_desc = t_arr[order]
    split_pts = np.flatnonzero(np.diff(t_desc)) + 1
    time_blocks = np.split(order, split_pts)

    concordant = 0.0
    comparable = 0
    total_in_tree = 0

    for block in time_blocks:
        e_block = e_arr[block]
        cens_idx = block[e_block == 0]
        evt_idx = block[e_block == 1]

        # 1) Censored subjects at time t are comparable to events at t and earlier times
        for j in cens_idx:
            r = int(ranks_1b[j])
            bit.add(r, 1)
            freq[r] += 1
            total_in_tree += 1

        # 2) Query events at time t against all j currently in the tree
        for i in evt_idx:
            r = int(ranks_1b[i])
            n_lt = bit.query(r - 1)
            n_eq = int(freq[r])
            concordant += float(n_lt) + 0.5 * float(n_eq)
            comparable += total_in_tree

        # 3) Insert events at time t after querying so tied events do not compare with each other
        for i in evt_idx:
            r = int(ranks_1b[i])
            bit.add(r, 1)
            freq[r] += 1
            total_in_tree += 1

    if comparable == 0:
        return 0.5
    return float(concordant / float(comparable))


def _build_descending_time_blocks(
    time_arr: np.ndarray, event_arr: np.ndarray
) -> List[Tuple[float, np.ndarray, int]]:
    """Group sample indices by unique time in descending order."""
    order = np.argsort(-time_arr, kind="mergesort")
    t_desc = time_arr[order]
    split_pts = np.flatnonzero(np.diff(t_desc)) + 1
    blocks = np.split(order, split_pts)
    result: List[Tuple[float, np.ndarray, int]] = []
    for blk in blocks:
        d_r = int(np.sum(event_arr[blk]))
        result.append((float(time_arr[blk[0]]), blk, d_r))
    return result


def _breslow_nll_and_grad(
    beta: np.ndarray,
    X_std: np.ndarray,
    event_arr: np.ndarray,
    time_blocks: List[Tuple[float, np.ndarray, int]],
    sum_X_events: np.ndarray,
    l2_penalty: float,
) -> Tuple[float, np.ndarray]:
    """Compute normalized Breslow partial NLL + L2 penalty and exact analytical gradient."""
    n = float(X_std.shape[0])
    p = X_std.shape[1]
    eta = X_std @ beta

    m_cur = -np.inf
    s_exp = 0.0
    s_xexp = np.zeros(p, dtype=np.float64)
    log_risk_sum = 0.0
    grad_risk = np.zeros(p, dtype=np.float64)

    for _, idx_r, d_r in time_blocks:
        eta_r = eta[idx_r]
        m_r = float(np.max(eta_r))
        if m_r > m_cur:
            if m_cur > -np.inf:
                alpha = math.exp(m_cur - m_r)
                s_exp *= alpha
                s_xexp *= alpha
            m_cur = m_r
        w_r = np.exp(eta_r - m_cur)
        s_exp += float(np.sum(w_r))
        s_xexp += X_std[idx_r].T @ w_r

        if d_r > 0:
            log_s_risk = m_cur + math.log(s_exp)
            log_risk_sum += float(d_r) * log_s_risk
            grad_risk += float(d_r) * (s_xexp / s_exp)

    sum_eta_events = float(np.sum(eta[event_arr == 1]))
    nll = -(sum_eta_events - log_risk_sum) / n + 0.5 * l2_penalty * float(np.dot(beta, beta))
    grad = -(sum_X_events - grad_risk) / n + l2_penalty * beta
    return float(nll), grad


def _compute_breslow_baseline(
    beta: np.ndarray,
    X_std: np.ndarray,
    time_blocks: List[Tuple[float, np.ndarray, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sorted ascending event times and Breslow baseline cumulative hazard H0(t)."""
    eta = np.clip(X_std @ beta, -50.0, 50.0)
    m_cur = -np.inf
    s_exp = 0.0
    desc_events: List[Tuple[float, float]] = []

    for u_r, idx_r, d_r in time_blocks:
        eta_r = eta[idx_r]
        m_r = float(np.max(eta_r))
        if m_r > m_cur:
            if m_cur > -np.inf:
                s_exp *= math.exp(m_cur - m_r)
            m_cur = m_r
        s_exp += float(np.sum(np.exp(eta_r - m_cur)))
        if d_r > 0:
            log_s_risk = max(-700.0, min(700.0, m_cur + math.log(s_exp)))
            delta_h0 = math.exp(math.log(float(d_r)) - log_s_risk)
            desc_events.append((u_r, delta_h0))

    if not desc_events:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    asc_events = desc_events[::-1]
    event_times = np.asarray([t for t, _ in asc_events], dtype=np.float64)
    cumhaz = np.cumsum(np.asarray([dh for _, dh in asc_events], dtype=np.float64))
    return event_times, cumhaz


def _disambiguate_two_fields(
    name0: str,
    name1: str,
    is_bool_0: bool,
    is_bool_1: bool,
) -> Tuple[int, int]:
    """Return ``(time_idx, event_idx)`` for a 2-field survival target."""
    if is_bool_0 != is_bool_1:
        return (1, 0) if is_bool_0 else (0, 1)

    n0, n1 = name0.lower(), name1.lower()
    event_tokens = ("event", "status", "cens", "indic", "obs")
    time_tokens = ("time", "dur", "surv", "futime")

    has_ev_0 = any(tok in n0 for tok in event_tokens)
    has_ev_1 = any(tok in n1 for tok in event_tokens)
    has_tm_0 = any(tok in n0 for tok in time_tokens)
    has_tm_1 = any(tok in n1 for tok in time_tokens)

    pure_ev_0 = has_ev_0 and not has_tm_0
    pure_ev_1 = has_ev_1 and not has_tm_1
    if pure_ev_0 != pure_ev_1:
        return (1, 0) if pure_ev_0 else (0, 1)

    pure_tm_0 = has_tm_0 and not has_ev_0
    pure_tm_1 = has_tm_1 and not has_ev_1
    if pure_tm_0 != pure_tm_1:
        return (0, 1) if pure_tm_0 else (1, 0)

    if has_ev_0 != has_ev_1:
        return (1, 0) if has_ev_0 else (0, 1)

    return (0, 1)


def _parse_survival_targets(
    y: Any,
    time: Any,
    event: Any,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse (time, event) from keyword args, structured array, DataFrame, or 2-col array."""
    if time is not None or event is not None:
        if time is None or event is None:
            raise ValueError("Both time and event must be provided when either is specified.")
        return _validate_time_event(time, event, n_samples=n_samples)

    if y is None:
        raise ValueError("Provide survival targets via time=/event= or a 2-column/structured y.")

    if isinstance(y, np.ndarray) and y.dtype.names is not None:
        names = list(y.dtype.names)
        if len(names) != 2:
            raise ValueError(f"Structured array y must have 2 fields, got {names}.")
        tm_idx, ev_idx = _disambiguate_two_fields(
            names[0],
            names[1],
            y.dtype[names[0]] == np.bool_,
            y.dtype[names[1]] == np.bool_,
        )
        return _validate_time_event(y[names[tm_idx]], y[names[ev_idx]], n_samples=n_samples)

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 2:
            raise ValueError(f"Survival DataFrame y must have 2 columns, got shape {y.shape}.")
        tm_idx, ev_idx = _disambiguate_two_fields(
            str(y.columns[0]),
            str(y.columns[1]),
            bool(pd.api.types.is_bool_dtype(y.iloc[:, 0].dtype)),
            bool(pd.api.types.is_bool_dtype(y.iloc[:, 1].dtype)),
        )
        return _validate_time_event(y.iloc[:, tm_idx], y.iloc[:, ev_idx], n_samples=n_samples)

    arr = np.asarray(y)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"Survival target y must have shape (n_samples, 2), got {arr.shape}."
        )
    return _validate_time_event(arr[:, 0], arr[:, 1], n_samples=n_samples)


class ACDSOSurvivalAnalyzer(BaseEstimator):
    """Cox Proportional Hazards survival analyzer with SciPy Breslow & optional lifelines.

    Parameters
    ----------
    l2_penalty : float, default=1e-2
        Non-negative L2 ridge regularization weight.
    backend : str, default="auto"
        Solver backend: ``"auto"``, ``"scipy"``, or ``"lifelines"``.
    cv : int or splitter, default=5
        Number of folds or CV splitter object.
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
        l2_penalty: float = 1e-2,
        backend: str = "auto",
        cv: Any = 5,
        random_state: Optional[int] = 42,
        text_cols: Optional[List[str]] = None,
        text_method: str = "tfidf",
        verbose: int = 0,
    ) -> None:
        self.l2_penalty = l2_penalty
        self.backend = backend
        self.cv = cv
        self.random_state = random_state
        self.text_cols = text_cols
        self.text_method = text_method
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose and int(self.verbose) > 0:
            print(f"[ACDSOSurvivalAnalyzer] {msg}")

    def _validate_hyperparameters(self) -> Tuple[float, str]:
        if isinstance(self.l2_penalty, bool) or not isinstance(
            self.l2_penalty, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"l2_penalty must be a float >= 0, got {self.l2_penalty!r}.")
        l2 = float(self.l2_penalty)
        if not math.isfinite(l2) or l2 < 0.0:
            raise ValueError(f"l2_penalty must be finite and >= 0, got {l2!r}.")
        if not isinstance(self.backend, str) or self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend={self.backend!r}. Must be one of {_VALID_BACKENDS}."
            )
        return l2, self.backend

    def _coerce_input(self, X: Any, *, is_fit: bool) -> pd.DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None.")

        if isinstance(X, pd.DataFrame):
            n_rows, n_cols = X.shape
            if is_fit:
                if n_rows < 2 or n_cols < 1:
                    raise ValueError(
                        f"ACDSOSurvivalAnalyzer.fit requires >= 2 rows and >= 1 column, got {X.shape}."
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
                    f"ACDSOSurvivalAnalyzer.fit requires >= 2 rows and >= 1 column, got {arr.shape}."
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

    @staticmethod
    def _to_dense_features(
        X_proc: Any, svd: Optional[TruncatedSVD] = None, fit_svd: bool = False, random_state: Optional[int] = 42
    ) -> Tuple[np.ndarray, Optional[TruncatedSVD]]:
        if sparse.issparse(X_proc):
            n_rows, n_cols = X_proc.shape
            if svd is not None:
                arr = np.asarray(svd.transform(X_proc), dtype=np.float64)
            elif fit_svd and n_rows * n_cols > 20_000_000 and min(n_rows, n_cols) > 2:
                n_comp = min(64, n_rows - 1, n_cols - 1)
                svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
                arr = np.asarray(svd.fit_transform(X_proc), dtype=np.float64)
            else:
                arr = X_proc.toarray().astype(np.float64, copy=False)
        else:
            arr = np.asarray(X_proc, dtype=np.float64)
            if svd is not None:
                arr = np.asarray(svd.transform(arr), dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr, svd

    def _fit_scipy_core(
        self,
        X_dense: np.ndarray,
        time_arr: np.ndarray,
        event_arr: np.ndarray,
        l2_penalty: float,
    ) -> Dict[str, Any]:
        mean_vec = np.mean(X_dense, axis=0)
        std_vec = np.std(X_dense, axis=0)
        std_vec = np.where(std_vec < 1e-12, 1.0, std_vec)
        X_std = (X_dense - mean_vec) / std_vec
        p = X_std.shape[1]

        if int(np.sum(event_arr)) == 0:
            beta = np.zeros(p, dtype=np.float64)
            t_min = max(1e-6, float(np.min(time_arr)))
            t_max = max(t_min + 1e-6, float(np.max(time_arr)))
            event_times = np.linspace(t_min, t_max, 5)
            cumhaz = np.zeros(5, dtype=np.float64)
            return {
                "mean": mean_vec,
                "std": std_vec,
                "beta": beta,
                "event_times": event_times,
                "cumhaz": cumhaz,
                "converged": True,
                "n_iter": 0,
            }

        time_blocks = _build_descending_time_blocks(time_arr, event_arr)
        sum_X_events = np.sum(X_std[event_arr == 1], axis=0)

        res = optimize.minimize(
            fun=_breslow_nll_and_grad,
            x0=np.zeros(p, dtype=np.float64),
            args=(X_std, event_arr, time_blocks, sum_X_events, l2_penalty),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-50.0, 50.0)] * p,
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
        beta = np.nan_to_num(np.asarray(res.x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        event_times, cumhaz = _compute_breslow_baseline(beta, X_std, time_blocks)
        return {
            "mean": mean_vec,
            "std": std_vec,
            "beta": beta,
            "event_times": event_times,
            "cumhaz": cumhaz,
            "converged": bool(res.success),
            "n_iter": int(getattr(res, "nit", 0)),
        }

    def _fit_lifelines_core(
        self,
        X_dense: np.ndarray,
        time_arr: np.ndarray,
        event_arr: np.ndarray,
        l2_penalty: float,
    ) -> Any:
        from lifelines import CoxPHFitter

        cols = [f"f_{i}" for i in range(X_dense.shape[1])]
        df_ll = pd.DataFrame(X_dense, columns=cols)
        df_ll["__time__"] = time_arr
        df_ll["__event__"] = event_arr
        cph = CoxPHFitter(penalizer=l2_penalty)
        cph.fit(df_ll, duration_col="__time__", event_col="__event__")
        return cph

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        time: Any = None,
        event: Any = None,
        groups: Any = None,
        cv_strategy: str = "auto",
    ) -> "ACDSOSurvivalAnalyzer":
        """Fit Cox Proportional Hazards model with optional grouped CV evaluation."""
        l2, backend_req = self._validate_hyperparameters()
        df = self._coerce_input(X, is_fit=True)
        time_arr, event_arr = _parse_survival_targets(y, time, event, len(df))

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

        # Resolve backend
        backend_used = backend_req
        if backend_req == "lifelines":
            try:
                import lifelines  # noqa: F401
            except ImportError as exc:
                raise ImportError("backend='lifelines' requires the lifelines package.") from exc
        elif backend_req == "auto":
            try:
                import lifelines  # noqa: F401

                backend_used = "lifelines"
            except ImportError:
                backend_used = "scipy"

        groups_arr: Optional[np.ndarray] = None
        if groups is not None:
            groups_arr = validate_groups(groups, n_samples=len(df_active), min_unique=2)

        # Entity-grouped or standard CV evaluation (strictly leak-free per fold)
        eff_cv_strategy = (
            "group_kfold" if (groups_arr is not None and cv_strategy == "auto") else cv_strategy
        )
        splitter = build_cv_splitter(
            cv=self.cv,
            y=event_arr,
            groups=groups_arr,
            task="regression",
            cv_strategy=eff_cv_strategy,
            random_state=self.random_state,
        )
        cv_scores: List[float] = []
        cv_name = "none"
        if len(df_active) >= 4:
            cv_name = resolve_cv_strategy_name(splitter)
            splits = iter_cv_splits(
                splitter, df_active, event_arr, groups_arr, validate_groups_leakage=True
            )
            if groups_arr is not None and eff_cv_strategy in ("group_kfold", "stratified_group_kfold"):
                validate_no_group_leakage(splits, groups_arr)
            for tr_idx, va_idx in splits:
                if len(tr_idx) < 2 or len(va_idx) < 2:
                    continue
                fold_pre = _make_pre()
                X_tr_raw = fold_pre.fit_transform(df_active.iloc[tr_idx])
                if X_tr_raw.shape[1] == 0:
                    continue
                X_va_raw = fold_pre.transform(df_active.iloc[va_idx])
                X_tr_d, fold_svd = self._to_dense_features(
                    X_tr_raw, fit_svd=True, random_state=self.random_state
                )
                X_va_d, _ = self._to_dense_features(X_va_raw, svd=fold_svd)
                fold_model = self._fit_scipy_core(
                    X_tr_d, time_arr[tr_idx], event_arr[tr_idx], l2
                )
                X_va_std = (X_va_d - fold_model["mean"]) / fold_model["std"]
                eta_va = X_va_std @ fold_model["beta"]
                cv_scores.append(
                    concordance_index(time_arr[va_idx], eta_va, event_arr[va_idx])
                )

        # Refit on full data
        self.preprocessor_ = _make_pre()
        X_full_raw = self.preprocessor_.fit_transform(df_active)
        if X_full_raw.shape[1] == 0:
            raise ValueError("No valid features remain after preprocessing.")
        X_full_d, svd_full = self._to_dense_features(
            X_full_raw, fit_svd=True, random_state=self.random_state
        )
        self._svd_ = svd_full
        self._min_time_ = float(np.min(time_arr))
        self._max_time_ = float(np.max(time_arr))

        self._lifelines_model_ = None
        if backend_used == "lifelines":
            try:
                cph = self._fit_lifelines_core(X_full_d, time_arr, event_arr, l2)
                self._lifelines_model_ = cph
                self.backend_used_ = "lifelines"
                self.coef_ = np.asarray(cph.params_.to_numpy(), dtype=np.float64)
                self._mean_ = np.zeros(X_full_d.shape[1], dtype=np.float64)
                self._std_ = np.ones(X_full_d.shape[1], dtype=np.float64)
                bh = cph.baseline_cumulative_hazard_
                self.event_times_ = np.asarray(bh.index.to_numpy(), dtype=np.float64)
                self.baseline_cumhaz_ = np.asarray(bh.iloc[:, 0].to_numpy(), dtype=np.float64)
                converged = True
                n_iter = 0
            except Exception as exc:
                if backend_req == "lifelines":
                    raise
                self._log(f"lifelines backend failed ({exc}); falling back to scipy Breslow.")
                backend_used = "scipy"

        if backend_used == "scipy":
            core = self._fit_scipy_core(X_full_d, time_arr, event_arr, l2)
            self.backend_used_ = "scipy"
            self._mean_ = core["mean"]
            self._std_ = core["std"]
            self.coef_ = core["beta"]
            self.event_times_ = core["event_times"]
            self.baseline_cumhaz_ = core["cumhaz"]
            converged = core["converged"]
            n_iter = core["n_iter"]
            train_risks = np.nan_to_num(
                ((X_full_d - self._mean_) / self._std_) @ self.coef_,
                nan=0.0,
                posinf=50.0,
                neginf=-50.0,
            )
        else:
            train_risks = self.predict_risk(X)

        train_c = concordance_index(time_arr, train_risks, event_arr)

        self.audit_trail_ = _sanitize_json_obj(
            {
                "backend_requested": self.backend,
                "backend_used": self.backend_used_,
                "l2_penalty": l2,
                "cv_strategy": cv_name,
                "cv_fold_c_index": cv_scores,
                "cv_mean_c_index": float(np.mean(cv_scores)) if cv_scores else None,
                "train_c_index": train_c,
                "n_samples": int(len(df_active)),
                "n_events": int(np.sum(event_arr)),
                "censoring_rate": float(1.0 - np.mean(event_arr)),
                "n_features_preprocessed": int(X_full_d.shape[1]),
                "text_cols": list(eff_text_cols),
                "converged": converged,
                "n_iter": n_iter,
            }
        )
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "coef_") or self.coef_ is None:
            raise RuntimeError("ACDSOSurvivalAnalyzer is not fitted. Call fit() first.")

    def _prepare_eval_dense(self, X: Any) -> np.ndarray:
        self._require_fitted()
        df = self._coerce_input(X, is_fit=False)
        df_active = df[self._active_cols_]
        X_proc = self.preprocessor_.transform(df_active)
        X_dense, _ = self._to_dense_features(X_proc, svd=getattr(self, "_svd_", None))
        return X_dense

    def predict_risk(self, X: Any) -> np.ndarray:
        """Predict 1D linear risk scores (higher = higher hazard / earlier event)."""
        X_dense = self._prepare_eval_dense(X)
        if getattr(self, "_lifelines_model_", None) is not None:
            cols = [f"f_{i}" for i in range(X_dense.shape[1])]
            df_ll = pd.DataFrame(X_dense, columns=cols)
            ph = np.asarray(
                self._lifelines_model_.predict_partial_hazard(df_ll), dtype=np.float64
            ).ravel()
            eta = np.log(np.maximum(ph, 1e-300))
            return np.nan_to_num(eta, nan=0.0, posinf=50.0, neginf=-50.0)

        X_std = (X_dense - self._mean_) / self._std_
        eta = np.asarray(X_std @ self.coef_, dtype=np.float64).ravel()
        return np.nan_to_num(eta, nan=0.0, posinf=50.0, neginf=-50.0)

    def predict(self, X: Any) -> np.ndarray:
        """Alias for ``predict_risk(X)`` returning 1D linear risk scores."""
        return self.predict_risk(X)

    def predict_survival_function(self, X: Any, times: Any = None) -> np.ndarray:
        """Predict survival probabilities ``S(t | x)`` of shape ``(n_samples, len(times))``."""
        X_dense = self._prepare_eval_dense(X)
        if times is None:
            if len(self.event_times_) >= 2:
                times_arr = self.event_times_.copy()
            else:
                t_lo = max(1e-6, getattr(self, "_min_time_", 0.05))
                t_hi = max(t_lo + 1e-6, getattr(self, "_max_time_", 1.0))
                times_arr = np.linspace(t_lo, t_hi, 10, dtype=np.float64)
        else:
            times_arr = _coerce_1d_survival_vec(times, "times")
            if np.any(times_arr < 0.0):
                raise ValueError("times cannot contain negative values.")

        if getattr(self, "_lifelines_model_", None) is not None:
            cols = [f"f_{i}" for i in range(X_dense.shape[1])]
            df_ll = pd.DataFrame(X_dense, columns=cols)
            sf_df = self._lifelines_model_.predict_survival_function(df_ll, times=times_arr)
            sf = np.asarray(sf_df.to_numpy().T, dtype=np.float64)
            return np.clip(np.nan_to_num(sf, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        X_std = (X_dense - self._mean_) / self._std_
        eta = np.clip(np.asarray(X_std @ self.coef_, dtype=np.float64).ravel(), -50.0, 50.0)
        if len(self.event_times_) == 0:
            return np.ones((len(X_dense), len(times_arr)), dtype=np.float64)

        idx = np.searchsorted(self.event_times_, times_arr, side="right") - 1
        h0 = np.where(idx >= 0, self.baseline_cumhaz_[np.maximum(idx, 0)], 0.0)
        cum_haz = np.outer(np.exp(eta), h0)
        sf = np.exp(-cum_haz)
        return np.clip(np.nan_to_num(sf, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    def score(
        self,
        X: Any,
        y: Any = None,
        *,
        time: Any = None,
        event: Any = None,
    ) -> float:
        """Return Harrell's C-index on ``(X, time, event)``."""
        self._require_fitted()
        risks = self.predict_risk(X)
        t_arr, e_arr = _parse_survival_targets(y, time, event, len(risks))
        return concordance_index(t_arr, risks, e_arr)

    def evaluate(
        self,
        X: Any,
        y: Any = None,
        *,
        time: Any = None,
        event: Any = None,
    ) -> Dict[str, Any]:
        """Evaluate survival predictions returning JSON-safe C-index and censoring summary."""
        self._require_fitted()
        risks = self.predict_risk(X)
        t_arr, e_arr = _parse_survival_targets(y, time, event, len(risks))
        c_val = concordance_index(t_arr, risks, e_arr)
        return _sanitize_json_obj(
            {
                "c_index": float(c_val),
                "concordance_index": float(c_val),
                "n_samples": int(len(risks)),
                "n_events": int(np.sum(e_arr)),
                "censoring_rate": float(1.0 - np.mean(e_arr)),
                "mean_risk": float(np.mean(risks)),
            }
        )
