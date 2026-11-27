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

from acdso.cv import (
    build_cv_splitter,
    iter_cv_splits,
    resolve_cv_strategy_name,
    sample_by_groups,
    validate_groups,
    validate_no_group_leakage,
)
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
    cv_strategy: str = "auto"
    n_groups: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def _to_classification_target_array(y_ser: pd.Series) -> np.ndarray:
    """Normalize a filtered classification target Series (incl. object-of-bool) to a NumPy array."""
    if pd.api.types.is_numeric_dtype(y_ser) or pd.api.types.is_bool_dtype(y_ser):
        return y_ser.to_numpy()
    y_inf = y_ser.infer_objects()
    if pd.api.types.is_numeric_dtype(y_inf) or pd.api.types.is_bool_dtype(y_inf):
        return y_inf.to_numpy()
    vals = y_ser.tolist()
    if vals and all(isinstance(v, (bool, np.bool_)) for v in vals):
        return np.asarray(vals, dtype=bool)
    return y_ser.astype(object).to_numpy()


def _temporal_leakage_gap(
    df: pd.DataFrame,
    y: Any,
    scorer: str,
    cv: int,
    task: str,
    random_state: int,
    groups: Optional[Any] = None,
    cv_strategy: str = "auto",
    tcol_hint: Optional[str] = None,
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
    groups : array-like, optional
        Entity/group identifiers aligned with df/y.
    cv_strategy : str, default="auto"
        Cross-validation strategy name.

    Returns
    -------
    Optional[float]
        Score gap (shuffled - chronological), or None if not applicable.
    """
    tcol = tcol_hint if (tcol_hint is not None and tcol_hint in df.columns) else detect_datetime_column(df)
    if tcol is None:
        return None

    try:
        if pd.api.types.is_datetime64_any_dtype(df[tcol]):
            parsed = df[tcol]
            if parsed.isna().mean() > 0.5:
                return None
            order = np.argsort(parsed.values.astype("datetime64[ns]").astype(np.int64))
        elif pd.api.types.is_numeric_dtype(df[tcol]):
            vals = df[tcol].to_numpy(dtype=np.float64, copy=False)
            if np.isnan(vals).mean() > 0.5:
                return None
            order = np.argsort(np.where(np.isfinite(vals), vals, np.inf))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                parsed = pd.to_datetime(df[tcol], errors="coerce")
            if parsed.isna().mean() > 0.5:
                return None
            order = np.argsort(parsed.values.astype("datetime64[ns]").astype(np.int64))
    except Exception:
        return None

    df_sorted = df.iloc[order].reset_index(drop=True)
    y_sorted = np.asarray(y)[order]
    groups_sorted = np.asarray(groups)[order] if groups is not None else None

    y_ser = pd.Series(y_sorted)
    valid_mask = y_ser.notna().to_numpy()
    if task != "classification":
        y_num = pd.to_numeric(y_ser, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        valid_mask = valid_mask & np.isfinite(y_num)
        if not np.all(valid_mask):
            df_sorted = df_sorted.iloc[valid_mask].reset_index(drop=True)
            y_sorted = y_num[valid_mask]
            if groups_sorted is not None:
                groups_sorted = groups_sorted[valid_mask]
        else:
            y_sorted = y_num
    else:
        if not np.all(valid_mask):
            df_sorted = df_sorted.iloc[valid_mask].reset_index(drop=True)
            y_ser = y_ser.iloc[valid_mask].reset_index(drop=True)
            if groups_sorted is not None:
                groups_sorted = groups_sorted[valid_mask]
        y_sorted = _to_classification_target_array(y_ser)

    if len(y_sorted) < max(4, int(cv) * 2):
        return None
    cls_counts: Optional[np.ndarray] = None
    if task == "classification":
        uniq_cls, cls_counts = np.unique(y_sorted, return_counts=True)
        if len(uniq_cls) < 2 or (groups_sorted is None and int(cls_counts.min()) < 2):
            return None

    num_cols, cat_cols = split_columns(df_sorted)
    pre = build_preprocessor(num_cols, cat_cols)

    if task == "classification":
        model = LogisticRegression(max_iter=1000)
    else:
        model = Ridge()

    pipe = Pipeline([("pre", pre), ("model", model)])

    try:
        if groups_sorted is not None:
            n_uniq_g = int(len(np.unique(groups_sorted)))
            if n_uniq_g < 2:
                return None
            cv_eff = max(2, min(int(cv), n_uniq_g))
            shuf_strategy = "auto" if str(cv_strategy).strip().lower() == "temporal" else cv_strategy
            splitter_shuf = build_cv_splitter(
                cv=cv_eff,
                y=y_sorted,
                groups=groups_sorted,
                task=task,
                cv_strategy=shuf_strategy,
                random_state=random_state,
            )
            shuf_splits = iter_cv_splits(
                splitter_shuf,
                df_sorted,
                y_sorted,
                groups_sorted,
                validate_groups_leakage=True,
            )

            # Chronological block folds grouped by first-appearance entity order
            codes, _ = pd.factorize(pd.Series(groups_sorted, dtype=object), sort=False)
            blocks = np.array_split(np.arange(n_uniq_g), cv_eff)
            chr_splits: List[Tuple[np.ndarray, np.ndarray]] = []
            for blk in blocks:
                if blk.size == 0:
                    continue
                val_mask = np.isin(codes, blk)
                tr_idx = np.flatnonzero(~val_mask)
                va_idx = np.flatnonzero(val_mask)
                if tr_idx.size > 0 and va_idx.size > 0:
                    chr_splits.append((tr_idx, va_idx))
            if len(chr_splits) < 2:
                return None
            validate_no_group_leakage(chr_splits, groups_sorted)
            cv_shuf: Any = shuf_splits
            cv_chr: Any = chr_splits
        else:
            if task == "classification":
                assert cls_counts is not None
                cv_eff_ng = max(2, min(int(cv), int(cls_counts.min())))
                cv_shuf = StratifiedKFold(n_splits=cv_eff_ng, shuffle=True, random_state=random_state)
                cv_chr = KFold(n_splits=cv_eff_ng, shuffle=False)
            else:
                cv_eff_ng = max(2, min(int(cv), len(y_sorted)))
                cv_shuf = KFold(n_splits=cv_eff_ng, shuffle=True, random_state=random_state)
                cv_chr = KFold(n_splits=cv_eff_ng, shuffle=False)

        eff_scorer = _auto_adjust_scorer_for_risk(scorer, y_sorted, task)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            scores_shuf = np.asarray(
                cross_val_score(pipe, df_sorted, y_sorted, scoring=eff_scorer, cv=cv_shuf),
                dtype=float,
            )
            scores_chr = np.asarray(
                cross_val_score(pipe, df_sorted, y_sorted, scoring=eff_scorer, cv=cv_chr),
                dtype=float,
            )
        if (
            scores_shuf.size == 0
            or scores_chr.size == 0
            or not np.all(np.isfinite(scores_shuf))
            or not np.all(np.isfinite(scores_chr))
        ):
            return None
        s_shuf = float(np.mean(scores_shuf))
        s_chr = float(np.mean(scores_chr))
        gap = float(s_shuf - s_chr)
        if not np.isfinite(gap):
            return None
        return gap
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
    groups: Optional[Any] = None,
    cv_strategy: str = "auto",
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

    groups_s: Optional[np.ndarray] = np.asarray(groups) if groups is not None else None
    rng = check_random_state(random_state)
    max_rows = min(n, 3000)
    if n > max_rows:
        if groups_s is not None:
            df_s, y_sub, groups_s = sample_by_groups(
                df,
                y=y,
                groups=groups_s,
                max_rows=max_rows,
                random_state=random_state,
            )
            df_s = df_s.reset_index(drop=True)
            y_s = np.asarray(y_sub)
        else:
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
        if groups_s is not None:
            groups_s = groups_s[valid_mask]
    if len(y_ser) < 4:
        return [], 0.0

    if task == "classification":
        y_arr = _to_classification_target_array(y_ser)
        uniq_y, counts_y = np.unique(y_arr, return_counts=True)
        if len(uniq_y) < 2:
            return [], 0.0
        min_class_count = int(counts_y.min())
        max_class_count = int(counts_y.max())
    else:
        y_float = pd.to_numeric(y_ser, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        finite_y = np.isfinite(y_float)
        if int(finite_y.sum()) < 4:
            return [], 0.0
        if not np.all(finite_y):
            df_s = df_s.iloc[finite_y].reset_index(drop=True)
            y_float = y_float[finite_y]
            if groups_s is not None:
                groups_s = groups_s[finite_y]
        if float(np.std(y_float)) < 1e-12:
            return [], 0.0
        y_arr = y_float
        min_class_count = len(y_arr)
        max_class_count = len(y_arr)

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

    if groups_s is not None and len(np.unique(groups_s)) < 2:
        fallback = [(orig_name, float(sc)) for _, orig_name, sc, _ in top_k]
        return fallback, float(fallback[0][1])

    if task == "classification":
        if min_class_count < 2:
            fallback = [(orig_name, float(sc)) for _, orig_name, sc, _ in top_k]
            return fallback, float(fallback[0][1])
        if groups_s is not None:
            n_uniq_g = int(len(np.unique(groups_s)))
            cv_eval = max(2, min(cv, 3, n_uniq_g))
        else:
            cv_eval = max(2, min(cv, 3, min_class_count))
        splitter = build_cv_splitter(
            cv=cv_eval,
            y=y_arr,
            groups=groups_s,
            task=task,
            cv_strategy=cv_strategy,
            random_state=random_state,
        )
        eval_scorer = "f1_weighted" if str(scorer).startswith("neg_") else scorer
        eval_scorer = _auto_adjust_scorer_for_risk(eval_scorer, y_arr, task)
        base_model = LogisticRegression(max_iter=500, random_state=random_state)
    else:
        if groups_s is not None:
            n_uniq_g = int(len(np.unique(groups_s)))
            cv_eval = max(2, min(cv, 3, n_uniq_g, max(2, len(y_arr) // 2)))
        else:
            cv_eval = max(2, min(cv, 3, len(y_arr) // 2))
        splitter = build_cv_splitter(
            cv=cv_eval,
            y=y_arr,
            groups=groups_s,
            task=task,
            cv_strategy=cv_strategy,
            random_state=random_state,
        )
        eval_scorer = "r2"
        base_model = Ridge(random_state=random_state)

    try:
        eval_splits = iter_cv_splits(
            splitter,
            df_scan,
            y_arr,
            groups_s,
            validate_groups_leakage=(groups_s is not None),
        )
    except Exception:
        fallback = [(orig_name, float(sc)) for _, orig_name, sc, _ in top_k]
        return fallback, float(fallback[0][1])

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
                cv_scores = cross_val_score(pipe, x_col, y_arr, scoring=eval_scorer, cv=eval_splits)
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
    groups: Optional[Any] = None,
    cv_strategy: str = "auto",
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
    groups : array-like, optional
        Entity/group identifiers aligned with df/y.
    cv_strategy : str, default="auto"
        Cross-validation strategy name.

    Returns
    -------
    Tuple[List[str], float]
        Tuple of (top_features, max_score).
    """
    scored, max_score = _univariate_predictive_power_detailed(
        df,
        y,
        task=task,
        scorer=scorer,
        cv=cv,
        random_state=random_state,
        deadline=None,
        groups=groups,
        cv_strategy=cv_strategy,
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
    groups: Optional[Any] = None,
    cv_strategy: str = "auto",
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
    groups_arr: Optional[np.ndarray] = np.asarray(groups) if groups is not None else None
    y_ser = pd.Series(y)
    valid_y = y_ser.notna().to_numpy()
    if task != "classification":
        y_num = pd.to_numeric(y_ser, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        valid_y = valid_y & np.isfinite(y_num)
        if not np.all(valid_y):
            X = X[valid_y]
            y_arr = y_num[valid_y]
            if groups_arr is not None:
                groups_arr = groups_arr[valid_y]
        else:
            y_arr = y_num
    else:
        if not np.all(valid_y):
            X = X[valid_y]
            y_ser = y_ser.iloc[valid_y]
            if groups_arr is not None:
                groups_arr = groups_arr[valid_y]
        y_arr = _to_classification_target_array(y_ser)

    n = len(y_arr)
    if n < 4:
        return False, 0.0

    max_samples = min(n, 3000)
    rng = np.random.RandomState(random_state)
    if n > max_samples:
        if groups_arr is not None:
            X, y_sub, groups_arr = sample_by_groups(
                X,
                y=y_arr,
                groups=groups_arr,
                max_rows=max_samples,
                random_state=random_state,
            )
            y_arr = np.asarray(y_sub)
        else:
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
        if groups_arr is None:
            cv_actual = min(cv_actual, int(counts.min()))
    if groups_arr is not None:
        n_uniq_g = int(len(np.unique(groups_arr)))
        if n_uniq_g < 2:
            return False, 0.0
        cv_actual = min(cv_actual, n_uniq_g)
    cv_actual = max(2, cv_actual)

    try:
        splitter = build_cv_splitter(
            cv=cv_actual,
            y=y_arr,
            groups=groups_arr,
            task=task,
            cv_strategy=cv_strategy,
            random_state=random_state,
        )
        lin_splits = iter_cv_splits(
            splitter,
            X,
            y_arr,
            groups_arr,
            validate_groups_leakage=(groups_arr is not None),
        )
        if task == "classification":
            linear_model = LogisticRegression(max_iter=500, random_state=random_state)
            nonlinear_model = RandomForestClassifier(
                n_estimators=30, max_depth=10, random_state=random_state, n_jobs=1
            )
            scorer = "accuracy"
        else:
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
            linear_scores = cross_val_score(linear_model, X, y_arr, cv=lin_splits, scoring=scorer)
            linear_mean = float(np.mean(linear_scores))
            nonlinear_scores = cross_val_score(nonlinear_model, X, y_arr, cv=lin_splits, scoring=scorer)
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


_CLASSIFICATION_ONLY_SCORERS = frozenset({
    "accuracy",
    "balanced_accuracy",
    "top_k_accuracy",
    "average_precision",
    "neg_brier_score",
    "brier_score_loss",
    "f1",
    "f1_micro",
    "f1_macro",
    "f1_weighted",
    "f1_samples",
    "precision",
    "precision_micro",
    "precision_macro",
    "precision_weighted",
    "precision_samples",
    "recall",
    "recall_micro",
    "recall_macro",
    "recall_weighted",
    "recall_samples",
    "jaccard",
    "jaccard_micro",
    "jaccard_macro",
    "jaccard_weighted",
    "jaccard_samples",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "roc_auc_ovr_weighted",
    "roc_auc_ovo_weighted",
    "neg_log_loss",
    "log_loss",
    "matthews_corrcoef",
    "cohen_kappa",
})


def _auto_adjust_scorer_for_risk(scorer: str, y: Any, task: str) -> str:
    """
    Auto-adjust scorer for risk analysis to handle regression defaults and multiclass/non-standard labels.
    
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
    task_norm = str(task or "").strip().lower()
    if task_norm != "classification":
        if task_norm == "regression":
            scorer_norm = str(scorer or "").strip().lower()
            if (
                scorer_norm in _CLASSIFICATION_ONLY_SCORERS
                or scorer_norm.startswith(("f1_", "precision_", "recall_", "roc_auc_", "jaccard_"))
            ):
                return "r2"
        return scorer

    y_clean = pd.Series(y).dropna()
    unique_labels = y_clean.unique()
    n_classes = len(unique_labels)

    # Scorers that require pos_label=1 (binary-only)
    binary_only_scorers = ["f1", "precision", "recall", "roc_auc"]

    if scorer in binary_only_scorers:
        # Multiclass requires ovr_weighted / weighted; binary non-{0,1} keeps roc_auc
        # because sklearn roc_auc handles arbitrary 2-class labels natively.
        if n_classes > 2:
            if scorer == "roc_auc":
                return "roc_auc_ovr_weighted"
            return f"{scorer}_weighted"
        if n_classes == 2 and set(unique_labels.tolist()) != {0, 1}:
            if scorer == "roc_auc":
                return "roc_auc"
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
    groups: Optional[Any] = None,
    cv_strategy: str = "auto",
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
    groups : array-like, optional
        Entity/group identifiers aligned with X and y for grouped CV.
    cv_strategy : str, default="auto"
        Cross-validation strategy ("auto", "kfold", "stratified",
        "group_kfold", "stratified_group_kfold", or "temporal").

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

    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])

    y_len = len(y) if hasattr(y, "__len__") else int(np.asarray(y).shape[0])
    if y_len != n_rows:
        raise ValueError(f"X and y must have the same number of rows ({n_rows} vs {y_len}).")

    groups_arr: Optional[np.ndarray] = None
    n_groups: Optional[int] = None
    if groups is not None:
        groups_arr = validate_groups(groups, n_samples=n_rows, min_unique=2)
        n_groups = int(len(np.unique(groups_arr)))

    # Filter/normalize target for probe_splitter consistency with internal CV diagnostics
    strategy_norm = cv_strategy.strip().lower() if isinstance(cv_strategy, str) else ""
    is_temporal = strategy_norm == "temporal"

    y_ser_probe = pd.Series(y)
    valid_probe = y_ser_probe.notna().to_numpy()
    groups_probe: Optional[np.ndarray] = groups_arr
    if str(task or "").strip().lower() == "classification":
        if not np.all(valid_probe):
            y_ser_probe = y_ser_probe.iloc[valid_probe].reset_index(drop=True)
            if groups_probe is not None:
                groups_probe = groups_probe[valid_probe]
        y_probe: Any = (
            _to_classification_target_array(y_ser_probe)
            if len(y_ser_probe) > 0
            else np.asarray([], dtype=object)
        )
    else:
        y_num_probe = pd.to_numeric(y_ser_probe, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        valid_probe = valid_probe & np.isfinite(y_num_probe)
        if not np.all(valid_probe):
            y_probe = y_num_probe[valid_probe]
            if groups_probe is not None:
                groups_probe = groups_probe[valid_probe]
        else:
            y_probe = y_num_probe

    n_groups_probe = int(len(np.unique(groups_probe))) if (groups_probe is not None and len(groups_probe) > 0) else 0
    has_enough_filtered = len(y_probe) >= 2 and (groups_probe is None or n_groups_probe >= 2)
    if has_enough_filtered:
        if is_temporal and len(y_probe) < 3 and n_rows >= 3:
            y_for_probe: Any = None
            groups_for_probe: Optional[np.ndarray] = groups_arr
        else:
            y_for_probe = y_probe
            groups_for_probe = groups_probe
    else:
        y_for_probe = None
        groups_for_probe = groups_arr

    eff_n_groups = (
        int(len(np.unique(groups_for_probe)))
        if (groups_for_probe is not None and len(groups_for_probe) > 0)
        else None
    )

    # Validate cv / cv_strategy / groups compatibility upfront before try/except blocks
    if (
        groups_for_probe is not None
        and eff_n_groups is not None
        and not hasattr(cv, "split")
        and not isinstance(cv, bool)
        and isinstance(cv, (int, np.integer))
        and int(cv) >= 2
        and int(cv) > eff_n_groups
        and not is_temporal
    ):
        cv_probe: Any = max(2, min(int(cv), eff_n_groups))
    else:
        cv_probe = cv

    probe_splitter = build_cv_splitter(
        cv=cv_probe,
        y=y_for_probe,
        groups=groups_for_probe,
        task=task,
        cv_strategy=cv_strategy,
        random_state=random_state,
    )
    resolved_cv_strategy = resolve_cv_strategy_name(probe_splitter)
    if hasattr(probe_splitter, "get_n_splits"):
        try:
            cv_int = int(probe_splitter.get_n_splits(df, y, groups_arr))
        except TypeError:
            cv_int = int(probe_splitter.get_n_splits(df))
    else:
        cv_int = int(cv)
    cv_int = max(2, cv_int)

    # Auto-adjust scorer for multiclass or non-standard binary labels
    scorer = _auto_adjust_scorer_for_risk(scorer, y, task)

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
    raw_tcol = detect_datetime_column(df_work)
    df_work = sanitize_features(df_work)
    work_num_cols, _ = split_columns(df_work)

    notes: List[str] = []
    warnings_list: List[str] = []

    if groups_arr is not None and not hasattr(cv, "split"):
        if not is_temporal and n_groups is not None and n_groups < int(cv):
            msg = f"Only {n_groups} unique groups found (< cv={cv}); CV folds clamped to {n_groups}."
            warnings_list.append(msg)
            notes.append(msg)
        if strategy_norm in ("kfold", "stratified"):
            msg = f"groups provided with non-grouped cv_strategy={cv_strategy!r}; entity grouping is not enforced."
            warnings_list.append(msg)
            notes.append(msg)

    # Univariate predictive power (leakage proxy detection)
    max_pp = 0.0
    leak_feats: List[str] = []
    if not skip_heavy and (deadline is None or time.perf_counter() <= deadline):
        scored_feats, max_pp = _univariate_predictive_power_detailed(
            df_work, y,
            task=task,
            scorer=scorer,
            cv=max(3, cv_int),
            random_state=random_state,
            deadline=deadline,
            groups=groups_arr,
            cv_strategy=cv_strategy,
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
            cv=min(max(3, cv_int), 3),
            task=task,
            random_state=random_state,
            groups=groups_arr,
            cv_strategy=cv_strategy,
            tcol_hint=raw_tcol,
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
            cv=min(max(3, cv_int), 3),
            random_state=random_state,
            groups=groups_arr,
            cv_strategy=cv_strategy,
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

    details = {
        "cv_strategy_requested": str(cv_strategy),
        "cv_strategy_resolved": resolved_cv_strategy,
        "groups_provided": groups_arr is not None,
        "n_groups": n_groups,
    }

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
        cv_strategy=resolved_cv_strategy,
        n_groups=n_groups,
        details=details,
        warnings=warnings_list,
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
    if rr.temporal_leakage_gap is not None and np.isfinite(float(rr.temporal_leakage_gap)):
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

