"""
ACDSO v3.0 - Risk Analysis Module
==================================

Advanced risk diagnostics for production-relevant analysis including:
- Leakage proxy detection (univariate predictive power + mutual information)
- Temporal leakage check (chronological vs shuffled CV)
- VIF-based multicollinearity detection
- Instability risk via repeated CV variance
- Class imbalance fragility assessment
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.utils import check_random_state

from acdso.utils import (
    infer_target_type,
    split_columns,
    compute_missing_ratio,
    build_preprocessor,
    detect_datetime_column,
    class_imbalance,
    sanitize_features,
)


@dataclass
class RiskReport:
    """
    Comprehensive risk report for a dataset.

    Contains quantitative risk assessments across multiple dimensions
    including leakage, imbalance, multicollinearity, stability, and linearity.

    Attributes
    ----------
    task : str
        Task type ("classification" or "regression").
    target_type : str
        Target type ("binary", "multiclass", or "regression").
    n_rows : int
        Number of rows in the dataset.
    n_cols : int
        Number of columns in the dataset.
    n_numeric : int
        Number of numeric columns.
    n_categorical : int
        Number of categorical columns.
    missing_ratio : float
        Ratio of missing values (0.0 to 1.0).
    leakage_proxy_features : List[str]
        Features suspected of being leakage proxies.
    leakage_proxy_score_max : float
        Maximum univariate predictive power score.
    temporal_leakage_gap : Optional[float]
        Score gap between shuffled and chronological CV (if applicable).
    class_imbalance_max_ratio : Optional[float]
        Maximum class ratio (for classification).
    imbalance_severity : float
        Imbalance severity score (0.0 to 1.0).
    vif_max : float
        Maximum Variance Inflation Factor.
    vif_features : List[str]
        Features with high VIF (>=10).
    cv_score_std : float
        Standard deviation of cross-validation scores.
    is_linear : bool
        Whether the dataset exhibits strong linear relationships.
    linearity_score : float
        Score indicating how linear the data is (0.0 to 1.0).
    notes : List[str]
        Human-readable risk notes and warnings.
    """

    task: str
    target_type: str
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    missing_ratio: float

    # Leakage
    leakage_proxy_features: List[str] = field(default_factory=list)
    leakage_proxy_score_max: float = 0.0
    temporal_leakage_gap: Optional[float] = None

    # Imbalance
    class_imbalance_max_ratio: Optional[float] = None
    imbalance_severity: float = 0.0

    # Multicollinearity
    vif_max: float = 1.0
    vif_features: List[str] = field(default_factory=list)

    # Stability
    cv_score_std: float = 0.0
    
    # Linearity detection
    is_linear: bool = False
    linearity_score: float = 0.0

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def _temporal_leakage_gap(
    df: pd.DataFrame,
    y: Any,
    scorer: str,
    cv: int,
    task: str,
    random_state: int,
) -> Optional[float]:
    """
    Detect temporal leakage by comparing chronological vs shuffled CV.

    If the shuffled CV score is significantly higher than chronological,
    it indicates potential temporal leakage in the data.

    Parameters
    ----------
    df : pd.DataFrame
        Input features.
    y : array-like
        Target values.
    scorer : str
        Scoring metric name.
    cv : int
        Number of cross-validation folds.
    task : str
        Task type ("classification" or "regression").
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Optional[float]
        Score gap (shuffled - chronological), or None if not applicable.
    """
    tcol = detect_datetime_column(df)
    if tcol is None:
        return None

    try:
        if not pd.api.types.is_datetime64_any_dtype(df[tcol]):
            parsed = pd.to_datetime(df[tcol], errors="coerce")
        else:
            parsed = df[tcol]
        if parsed.isna().mean() > 0.5:
            return None
    except Exception:
        return None

    order = np.argsort(parsed.values.astype("datetime64[ns]").astype(np.int64))
    df_sorted = df.iloc[order].reset_index(drop=True)
    y_sorted = np.asarray(y)[order]

    num_cols, cat_cols = split_columns(df_sorted)
    pre = build_preprocessor(num_cols, cat_cols)

    if task == "classification":
        model = LogisticRegression(max_iter=1000)
        splitter_shuf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        splitter_chr = KFold(n_splits=cv, shuffle=False)
    else:
        model = Ridge()
        splitter_shuf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        splitter_chr = KFold(n_splits=cv, shuffle=False)

    pipe = Pipeline([("pre", pre), ("model", model)])

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            s_shuf = float(np.mean(cross_val_score(pipe, df_sorted, y_sorted, scoring=scorer, cv=splitter_shuf)))
            s_chr = float(np.mean(cross_val_score(pipe, df_sorted, y_sorted, scoring=scorer, cv=splitter_chr)))
        return float(s_shuf - s_chr)
    except Exception:
        return None


def _univariate_predictive_power_detailed(
    df: pd.DataFrame,
    y: Any,
    task: str,
    scorer: str,
    cv: int,
    random_state: int,
    deadline: Optional[float] = None,
) -> Tuple[List[Tuple[str, float]], float]:
    """
    Detect potential leakage via fast vectorized pre-screening + top-K CV.

    Returns
    -------
    Tuple[List[Tuple[str, float]], float]
        Sorted list of (feature_name, cv_score) pairs and max score.
    """
    from sklearn.feature_selection import f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

    n = int(df.shape[0])
    if n < 4 or df.shape[1] == 0:
        return [], 0.0

    rng = check_random_state(random_state)
    max_rows = min(n, 3000)
    if n > max_rows:
        idx = rng.choice(n, size=max_rows, replace=False)
        df_s = df.iloc[idx].reset_index(drop=True)
        y_s = np.asarray(y)[idx]
    else:
        df_s = df
        y_s = np.asarray(y)

    y_ser = pd.Series(y_s)
    valid_mask = y_ser.notna().to_numpy()
    if not np.all(valid_mask):
        df_s = df_s.iloc[valid_mask].reset_index(drop=True)
        y_ser = y_ser.iloc[valid_mask].reset_index(drop=True)
    if len(y_ser) < 4:
        return [], 0.0

    if task == "classification":
        y_arr = y_ser.astype(object).to_numpy() if not pd.api.types.is_numeric_dtype(y_ser) else y_ser.to_numpy()
        uniq_y, counts_y = np.unique(y_arr, return_counts=True)
        if len(uniq_y) < 2:
            return [], 0.0
        min_class_count = int(counts_y.min())
    else:
        y_float = pd.to_numeric(y_ser, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        finite_y = np.isfinite(y_float)
        if int(finite_y.sum()) < 4:
            return [], 0.0
        if not np.all(finite_y):
            df_s = df_s.iloc[finite_y].reset_index(drop=True)
            y_float = y_float[finite_y]
        if float(np.std(y_float)) < 1e-12:
            return [], 0.0
        y_arr = y_float
        min_class_count = len(y_arr)

    # Normalize duplicate column names for safe positional indexing
    orig_col_names = [str(c) for c in df_s.columns]
    if df_s.columns.has_duplicates:
        internal_cols = [f"__acdso_c_{i}" for i in range(df_s.shape[1])]
        internal_to_orig = dict(zip(internal_cols, orig_col_names))
        df_scan = df_s.copy()
        df_scan.columns = internal_cols
    else:
        internal_to_orig = {str(c): str(c) for c in df_s.columns}
        df_scan = df_s

    df_scan = sanitize_features(df_scan)
    num_cols, cat_cols = split_columns(df_scan)

    # Pre-screen candidates: (internal_col, orig_name, prescreen_score, is_numeric)
    candidates: List[Tuple[Any, str, float, bool]] = []

    if num_cols:
        X_num = df_scan[num_cols].astype(np.float64, copy=False).to_numpy(dtype=np.float64, copy=True)
        X_num[~np.isfinite(X_num)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(X_num, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        nan_r, nan_c = np.where(np.isnan(X_num))
        if nan_r.size > 0:
            X_num[nan_r, nan_c] = medians[nan_c]

        stds = np.std(X_num, axis=0)
        valid_idx = np.flatnonzero(stds > 1e-12)
        if valid_idx.size > 0:
            Xv = X_num[:, valid_idx]
            if task == "classification":
                y_enc = LabelEncoder().fit_transform(y_arr)
                n_cls = max(2, len(np.unique(y_enc)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    f_vals, _ = f_classif(Xv, y_enc)
                f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=1e6, neginf=0.0)
                df_between = float(max(1, n_cls - 1))
                df_within = float(max(1, len(y_enc) - n_cls))
                eta2 = (f_vals * df_between) / np.maximum(f_vals * df_between + df_within, 1.0)
                prescores = np.clip(eta2, 0.0, 1.0)
            else:
                Xc = Xv - Xv.mean(axis=0)
                yc = y_arr - np.mean(y_arr)
                x_norms = np.linalg.norm(Xc, axis=0)
                y_norm = float(np.linalg.norm(yc))
                r = (Xc.T @ yc) / np.maximum(x_norms * y_norm, 1e-12)
                prescores = np.clip(r * r, 0.0, 1.0)

            for idx_pos, sc in zip(valid_idx.tolist(), prescores.tolist()):
                cname = num_cols[idx_pos]
                orig_name = internal_to_orig.get(str(cname), str(cname))
                candidates.append((cname, orig_name, float(sc), True))

    if cat_cols:
        if task == "regression":
            y_mean = float(np.mean(y_arr))
            ss_tot = float(np.sum((y_arr - y_mean) ** 2))
        for c in cat_cols:
            s = df_scan[c]
            n_non_null = int(s.notna().sum())
            if n_non_null < 2:
                continue
            try:
                nunique = int(s.nunique(dropna=True))
            except Exception:
                continue
            # Skip constant or ID-like non-numeric columns (high-cardinality strings/UUIDs)
            if nunique <= 1 or nunique > max(20, int(0.9 * n_non_null)):
                continue
            try:
                s_clean = s.astype(object).fillna("__MISSING__")
                orig_name = internal_to_orig.get(str(c), str(c))
                if task == "classification":
                    ct = pd.crosstab(s_clean.to_numpy(), y_arr)
                    purity = float(ct.max(axis=1).sum() / max(1, len(y_arr)))
                    candidates.append((c, orig_name, purity, False))
                else:
                    codes, uniques = pd.factorize(s_clean, sort=False)
                    if len(uniques) <= 1 or ss_tot <= 1e-12:
                        continue
                    counts = np.bincount(codes, minlength=len(uniques))
                    sums = np.bincount(codes, weights=y_arr, minlength=len(uniques))
                    means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
                    ss_between = float(np.sum(counts * (means - y_mean) ** 2))
                    eta2 = float(np.clip(ss_between / ss_tot, 0.0, 1.0))
                    candidates.append((c, orig_name, eta2, False))
            except Exception:
                continue

    if not candidates:
        return [], 0.0

    candidates.sort(key=lambda item: item[2], reverse=True)
    top_k = candidates[: min(10, len(candidates))]

    if task == "classification":
        if min_class_count < 2:
            fallback = [(orig_name, float(sc)) for _, orig_name, sc, _ in top_k]
            return fallback, float(fallback[0][1])
        cv_eval = max(2, min(cv, 3, min_class_count))
        splitter = StratifiedKFold(n_splits=cv_eval, shuffle=True, random_state=random_state)
        eval_scorer = "f1_weighted" if str(scorer).startswith("neg_") else scorer
        eval_scorer = _auto_adjust_scorer_for_risk(eval_scorer, y_arr, task)
        base_model = LogisticRegression(max_iter=500, random_state=random_state)
    else:
        cv_eval = max(2, min(cv, 3, len(y_arr) // 2))
        splitter = KFold(n_splits=cv_eval, shuffle=True, random_state=random_state)
        eval_scorer = "r2"
        base_model = Ridge(random_state=random_state)

    scored: List[Tuple[str, float]] = []

    for orig_col, orig_name, _, is_num in top_k:
        if deadline is not None and time.perf_counter() > deadline:
            break
        try:
            if is_num:
                x_col = df_scan[[orig_col]].astype(np.float64, copy=False).to_numpy(dtype=np.float64, copy=True)
                x_col[~np.isfinite(x_col)] = np.nan
                pipe = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("model", clone(base_model)),
                ])
            else:
                x_col = df_scan[orig_col].astype(object).fillna("__MISSING__").to_numpy().reshape(-1, 1)
                pipe = Pipeline([
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ("model", clone(base_model)),
                ])

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                cv_scores = cross_val_score(pipe, x_col, y_arr, scoring=eval_scorer, cv=splitter)
            scored.append((orig_name, float(np.mean(cv_scores))))
        except Exception:
            continue

    if not scored:
        fallback = [(orig_name, float(sc)) for _, orig_name, sc, _ in top_k]
        return fallback, float(fallback[0][1])

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored, float(scored[0][1])


def _univariate_predictive_power(
    df: pd.DataFrame,
    y: Any,
    task: str,
    scorer: str,
    cv: int,
    random_state: int,
) -> Tuple[List[str], float]:
    """
    Detect potential leakage via univariate predictive power.

    Features that achieve near-perfect prediction alone are suspicious
    and may be direct or proxy leaks.

    Parameters
    ----------
    df : pd.DataFrame
        Input features.
    y : array-like
        Target values.
    task : str
        Task type ("classification" or "regression").
    scorer : str
        Scoring metric name.
    cv : int
        Number of cross-validation folds.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[List[str], float]
        Tuple of (top_features, max_score).
    """
    scored, max_score = _univariate_predictive_power_detailed(
        df, y, task=task, scorer=scorer, cv=cv, random_state=random_state, deadline=None
    )
    return [f for f, _ in scored[:5]], max_score


def _vif_numeric(
    df_num: pd.DataFrame,
    max_features: int = 40,
) -> Tuple[float, List[str]]:
    """
    Compute Variance Inflation Factor for numeric features.

    Uses a ridge-damped correlation matrix inversion (Cholesky with eigh fallback)
    for vectorized O(d^3) computation while detecting exact collinear duplicates.
    """
    if df_num.shape[1] < 2 or df_num.shape[0] < 2:
        return 1.0, []

    cols = list(df_num.columns)[:max_features]
    X_arr = df_num[cols].astype(np.float64, copy=False).to_numpy(dtype=np.float64, copy=True)
    X_arr[~np.isfinite(X_arr)] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(X_arr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    nan_r, nan_c = np.where(np.isnan(X_arr))
    if nan_r.size > 0:
        X_arr[nan_r, nan_c] = medians[nan_c]

    stds = np.std(X_arr, axis=0)
    keep = np.flatnonzero(stds > 1e-12)
    if keep.size < 2:
        return 1.0, []

    X_nc = X_arr[:, keep]
    Xc = (X_nc - X_nc.mean(axis=0)) / stds[keep]
    n_rows, d = Xc.shape
    corr = (Xc.T @ Xc) / float(n_rows)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)

    reg = 1e-6
    try:
        corr_reg = corr + reg * np.eye(d)
        L = np.linalg.cholesky(corr_reg)
        inv_L = np.linalg.solve(L, np.eye(d))
        vifs_nc = np.sum(inv_L * inv_L, axis=0)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr)
        inv_eig = 1.0 / np.maximum(eigvals, reg)
        vifs_nc = (eigvecs * eigvecs) @ inv_eig

    vifs_nc = np.clip(vifs_nc, 1.0, 1e6)
    vif_max = float(np.max(vifs_nc)) if vifs_nc.size else 1.0
    high = [str(cols[i]) for i, v in zip(keep.tolist(), vifs_nc.tolist()) if v >= 10.0]
    return vif_max, high


def _detect_linearity(
    df: pd.DataFrame,
    y: Any,
    task: str,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[bool, float]:
    """
    Detect whether the dataset exhibits strong linear relationships.

    Compares linear model performance to a compact non-linear baseline to
    determine if linear models are likely to perform well.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return False, 0.0

    X = df[num_cols].astype(np.float64, copy=False).to_numpy(dtype=np.float64, copy=True)
    X[~np.isfinite(X)] = np.nan
    y_ser = pd.Series(y)
    valid_y = y_ser.notna().to_numpy()
    if task != "classification":
        y_num = pd.to_numeric(y_ser, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        valid_y = valid_y & np.isfinite(y_num)
        if not np.all(valid_y):
            X = X[valid_y]
            y_arr = y_num[valid_y]
        else:
            y_arr = y_num
    else:
        if not np.all(valid_y):
            X = X[valid_y]
            y_ser = y_ser.iloc[valid_y]
        y_arr = y_ser.astype(object).to_numpy() if not pd.api.types.is_numeric_dtype(y_ser) else y_ser.to_numpy()

    n = len(y_arr)
    if n < 4:
        return False, 0.0

    max_samples = min(n, 3000)
    rng = np.random.RandomState(random_state)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X = X[idx]
        y_arr = y_arr[idx]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputer = SimpleImputer(strategy="median")
            X = imputer.fit_transform(X)
        if X.shape[1] == 0:
            return False, 0.0
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    except Exception:
        return False, 0.0

    n_samples = len(y_arr)
    cv_actual = min(cv, 3, n_samples // 2)
    if task == "classification":
        uniq, counts = np.unique(y_arr, return_counts=True)
        if len(uniq) < 2 or int(counts.min()) < 2:
            return False, 0.0
        cv_actual = min(cv_actual, int(counts.min()))
    cv_actual = max(2, cv_actual)

    try:
        if task == "classification":
            splitter = StratifiedKFold(n_splits=cv_actual, shuffle=True, random_state=random_state)
            linear_model = LogisticRegression(max_iter=500, random_state=random_state)
            nonlinear_model = RandomForestClassifier(
                n_estimators=30, max_depth=10, random_state=random_state, n_jobs=1
            )
            scorer = "accuracy"
        else:
            splitter = KFold(n_splits=cv_actual, shuffle=True, random_state=random_state)
            linear_model = Ridge(random_state=random_state)
            nonlinear_model = RandomForestRegressor(
                n_estimators=30,
                max_depth=10,
                max_features=("sqrt" if X.shape[1] > 20 else 1.0),
                random_state=random_state,
                n_jobs=1,
            )
            scorer = "r2"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            linear_scores = cross_val_score(linear_model, X, y_arr, cv=splitter, scoring=scorer)
            linear_mean = float(np.mean(linear_scores))
            nonlinear_scores = cross_val_score(nonlinear_model, X, y_arr, cv=splitter, scoring=scorer)
            nonlinear_mean = float(np.mean(nonlinear_scores))

        if nonlinear_mean <= 0:
            ratio = 1.0 if linear_mean >= 0 else 0.0
        else:
            ratio = linear_mean / nonlinear_mean

        if task == "regression":
            min_lin = 0.5
            is_linear = linear_mean >= 0.7 or (ratio >= 0.95 and linear_mean >= min_lin)
            base_score = max(0.0, linear_mean)
        else:
            min_lin = 0.7
            is_linear = linear_mean >= 0.85 or (ratio >= 0.95 and linear_mean >= min_lin)
            base_score = float(np.clip(ratio, 0.0, 1.0)) * float(np.clip(linear_mean, 0.0, 1.0))

        if ratio > 1.05 and linear_mean >= min_lin:
            is_linear = True
            linearity_score = min(1.0, ratio)
        elif is_linear:
            linearity_score = float(np.clip(max(base_score, min(1.0, ratio)), 0.0, 1.0))
        else:
            linearity_score = float(np.clip(min(base_score, max(0.0, ratio)), 0.0, 1.0))

        return is_linear, float(np.clip(linearity_score, 0.0, 1.0))

    except Exception:
        return False, 0.0


def _auto_adjust_scorer_for_risk(scorer: str, y: Any, task: str) -> str:
    """
    Auto-adjust scorer for risk analysis to handle multiclass/non-standard labels.
    
    Parameters
    ----------
    scorer : str
        Original scorer name.
    y : array-like
        Target values.
    task : str
        Task type.
        
    Returns
    -------
    str
        Adjusted scorer name.
    """
    if task != "classification":
        return scorer

    y_clean = pd.Series(y).dropna()
    unique_labels = y_clean.unique()
    n_classes = len(unique_labels)

    # Scorers that require pos_label=1 (binary-only)
    binary_only_scorers = ["f1", "precision", "recall", "roc_auc"]

    if scorer in binary_only_scorers:
        # Multiclass or non-standard binary labels
        if n_classes > 2 or (n_classes == 2 and set(unique_labels.tolist()) != {0, 1}):
            if scorer == "roc_auc":
                return "roc_auc_ovr_weighted"
            return f"{scorer}_weighted"

    return scorer


def analyze_risk(
    X: Any,
    y: Any,
    task: str,
    scorer: str = "accuracy",
    cv: int = 5,
    random_state: int = 42,
    llm_advisor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    time_budget_s: Optional[float] = None,
) -> RiskReport:
    """
    Perform comprehensive risk analysis on a dataset.

    This is the main entry point for ACDSO risk diagnostics.

    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like
        Target values.
    task : str
        Task type ("classification" or "regression").
    scorer : str, default="accuracy"
        Scoring metric name.
    cv : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Random seed for reproducibility.
    llm_advisor : Optional[Callable], default=None
        Optional LLM advisor function for semantic analysis.
    time_budget_s : Optional[float], default=None
        Optional time budget in seconds for risk diagnostics.

    Returns
    -------
    RiskReport
        Comprehensive risk report.
    """
    deadline: Optional[float] = None
    skip_heavy = False
    if time_budget_s is not None:
        if time_budget_s <= 0:
            skip_heavy = True
        else:
            deadline = time.perf_counter() + float(time_budget_s)

    # Auto-adjust scorer for multiclass or non-standard binary labels
    scorer = _auto_adjust_scorer_for_risk(scorer, y, task)

    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    num_cols, cat_cols = split_columns(df)

    missing_ratio = compute_missing_ratio(df)
    target_type = infer_target_type(y)

    # Prepare sanitized, uniquely-named working DataFrame for internal diagnostics
    orig_names = [str(c) for c in df.columns]
    if df.columns.has_duplicates:
        internal_cols = [f"__acdso_c_{i}" for i in range(df.shape[1])]
        col_map = dict(zip(internal_cols, orig_names))
        df_work = df.copy()
        df_work.columns = internal_cols
    else:
        col_map = {str(c): str(c) for c in df.columns}
        df_work = df
    df_work = sanitize_features(df_work)
    work_num_cols, _ = split_columns(df_work)

    notes: List[str] = []

    # Univariate predictive power (leakage proxy detection)
    max_pp = 0.0
    leak_feats: List[str] = []
    if not skip_heavy and (deadline is None or time.perf_counter() <= deadline):
        scored_feats, max_pp = _univariate_predictive_power_detailed(
            df_work, y,
            task=task,
            scorer=scorer,
            cv=max(3, cv),
            random_state=random_state,
            deadline=deadline,
        )
        leak_feats = [col_map.get(name, name) for name, sc in scored_feats if sc >= 0.95]
        if leak_feats:
            notes.append(
                f"Potential leakage/proxy: univariate predictive power max={max_pp:.3f} "
                f"(features={leak_feats})."
            )

    # Temporal leakage detection
    gap: Optional[float] = None
    if not skip_heavy and (deadline is None or time.perf_counter() <= deadline):
        gap = _temporal_leakage_gap(
            df_work, y,
            scorer=scorer,
            cv=min(max(3, cv), 3),
            task=task,
            random_state=random_state,
        )
        if gap is not None and gap >= 0.05:
            notes.append(f"Temporal leakage suspicion: shuffled-chronological score gap={gap:.3f}.")

    # Class imbalance
    max_ratio, imb_sev = (None, 0.0)
    if task == "classification":
        max_ratio, imb_sev = class_imbalance(y)
        if max_ratio is not None and max_ratio >= 0.90:
            notes.append(f"Severe imbalance: max class ratio={max_ratio:.3f}.")

    # Multicollinearity (VIF)
    vif_max, vif_feats_raw = _vif_numeric(df_work[work_num_cols] if work_num_cols else df_work.iloc[:, :0])
    vif_feats = [col_map.get(str(x), str(x)) for x in vif_feats_raw]
    if vif_max >= 10.0:
        notes.append(f"High multicollinearity: VIF max={vif_max:.1f}.")

    # Linearity detection
    is_linear, linearity_score = (False, 0.0)
    if not skip_heavy and (deadline is None or time.perf_counter() <= deadline):
        is_linear, linearity_score = _detect_linearity(
            df_work, y,
            task=task,
            cv=min(max(3, cv), 3),
            random_state=random_state,
        )
        if is_linear:
            notes.append(
                f"Linear relationship detected: linearity score={linearity_score:.3f}. Linear models recommended."
            )

    # LLM advisor (if provided)
    if llm_advisor is not None:
        try:
            llm_out = llm_advisor({
                "columns": [str(c) for c in df.columns],
                "task": task,
                "target_type": target_type,
                "notes": notes,
            }) or {}
            extra = llm_out.get("note")
            if isinstance(extra, str) and extra.strip():
                notes.append(f"LLM: {extra.strip()}")
        except Exception as e:
            notes.append(f"LLM advisor error ignored: {type(e).__name__}")

    return RiskReport(
        task=task,
        target_type=target_type,
        n_rows=n_rows,
        n_cols=n_cols,
        n_numeric=len(num_cols),
        n_categorical=len(cat_cols),
        missing_ratio=float(missing_ratio),
        leakage_proxy_features=leak_feats,
        leakage_proxy_score_max=float(max_pp),
        temporal_leakage_gap=float(gap) if gap is not None else None,
        class_imbalance_max_ratio=float(max_ratio) if max_ratio is not None else None,
        imbalance_severity=float(imb_sev),
        vif_max=float(vif_max),
        vif_features=[str(x) for x in vif_feats],
        cv_score_std=0.0,
        is_linear=is_linear,
        linearity_score=float(linearity_score),
        notes=notes,
    )


def compute_risk_penalty(
    rr: RiskReport,
    cv_score_std: float,
    preference: str = "balanced",
) -> float:
    """
    Compute a continuous, severity-weighted risk penalty.

    The penalty combines multiple risk signals into a single [0, 1] score
    that can be used in multi-objective optimization.

    Parameters
    ----------
    rr : RiskReport
        Risk report from analyze_risk().
    cv_score_std : float
        Standard deviation of cross-validation scores.
    preference : str, default="balanced"
        Risk preference mode ("safe" or "balanced").

    Returns
    -------
    float
        Risk penalty score (0.0 = low risk, 1.0 = high risk).
    """
    # Missing data risk
    miss = float(np.clip((rr.missing_ratio - 0.10) / 0.40, 0.0, 1.0))

    # Imbalance risk
    imb = float(np.clip(rr.imbalance_severity, 0.0, 1.0))

    # Leakage proxy risk
    leak = 0.0
    if rr.leakage_proxy_score_max >= 0.95:
        leak = float(np.clip((rr.leakage_proxy_score_max - 0.95) / 0.05, 0.0, 1.0))

    # Temporal leakage risk
    temp = 0.0
    if rr.temporal_leakage_gap is not None:
        temp = float(np.clip(rr.temporal_leakage_gap / 0.20, 0.0, 1.0))

    # Multicollinearity risk (VIF)
    vif = float(np.clip((rr.vif_max - 5.0) / 25.0, 0.0, 1.0))

    # Instability risk
    stab = float(np.clip(cv_score_std / 0.05, 0.0, 1.0))

    # Preference-based weighting
    if preference == "safe":
        w = {"leak": 0.30, "temp": 0.15, "imb": 0.20, "vif": 0.15, "miss": 0.10, "stab": 0.10}
    else:
        w = {"leak": 0.25, "temp": 0.10, "imb": 0.20, "vif": 0.15, "miss": 0.15, "stab": 0.15}

    penalty = (
        w["leak"] * leak +
        w["temp"] * temp +
        w["imb"] * imb +
        w["vif"] * vif +
        w["miss"] * miss +
        w["stab"] * stab
    )
    return float(np.clip(penalty, 0.0, 1.0))

