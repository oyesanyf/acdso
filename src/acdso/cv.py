"""
ACDSO Cross-Validation & Entity-Grouping Utilities
==================================================

Unified cross-validation splitter factory, group-leakage validator,
and group-aware subsampling / holdout helpers supporting:
- K-Fold and Stratified K-Fold
- Entity-grouped CV (GroupKFold and StratifiedGroupKFold)
- Temporal / chronological splits (TimeSeriesSplit and TimeSeriesSplitter adapters)
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.utils.multiclass import type_of_target

VALID_CV_STRATEGIES = (
    "auto",
    "kfold",
    "stratified",
    "group_kfold",
    "stratified_group_kfold",
    "temporal",
)


class _TimeSeriesSplitterAdapter:
    """Adapter exposing a 3-argument sklearn-compatible .split(X, y, groups) API
    for ``acdso.timeseries.splits.TimeSeriesSplitter`` instances."""

    def __init__(self, splitter: Any) -> None:
        self.splitter = splitter

    def split(self, X: Any, y: Any = None, groups: Any = None):
        return self.splitter.split(X, y)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return int(self.splitter.get_n_splits(X))

    def __repr__(self) -> str:
        return f"_TimeSeriesSplitterAdapter({self.splitter!r})"


def validate_groups(
    groups: Any,
    n_samples: Optional[int] = None,
    *,
    min_unique: int = 2,
) -> np.ndarray:
    """Validate and normalize entity group identifiers into a 1D NumPy array.

    Parameters
    ----------
    groups : array-like
        Entity/group identifiers aligned with samples.
    n_samples : int, optional
        Expected number of samples. If provided, length mismatches raise ValueError.
    min_unique : int, default=2
        Minimum number of unique groups required.

    Returns
    -------
    np.ndarray
        Validated 1D array of group identifiers.
    """
    if groups is None:
        raise ValueError("groups cannot be None.")

    if isinstance(groups, pd.DataFrame):
        if groups.shape[1] != 1:
            raise ValueError(
                f"groups DataFrame must have exactly 1 column, got {groups.shape[1]}."
            )
        raw: Any = groups.iloc[:, 0]
    else:
        raw = groups

    if isinstance(raw, (pd.Series, pd.Index, pd.Categorical)):
        if pd.isna(raw).any():
            raise ValueError("groups contains missing/NaN values.")
        arr = np.asarray(raw)
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            raise ValueError("groups contains infinite values.")
    else:
        arr = np.asarray(raw)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        elif arr.ndim != 1:
            raise ValueError(f"groups must be 1-dimensional, got shape {arr.shape}.")
        if arr.size > 0:
            if np.issubdtype(arr.dtype, np.floating):
                if not np.all(np.isfinite(arr)):
                    raise ValueError("groups contains NaN or infinite values.")
            elif pd.isna(pd.Series(arr)).any():
                raise ValueError("groups contains missing/NaN values.")

    if arr.size == 0:
        raise ValueError("groups cannot be empty.")

    if n_samples is not None and len(arr) != int(n_samples):
        raise ValueError(
            f"groups length ({len(arr)}) does not match n_samples ({int(n_samples)})."
        )

    try:
        uniq = np.unique(arr)
    except TypeError:
        # Mixed incomparable Python types (e.g. int and str in object array)
        codes, _ = pd.factorize(pd.Series(arr, dtype=object), sort=False)
        arr = codes
        uniq = np.unique(arr)

    if len(uniq) < int(min_unique):
        raise ValueError(
            f"groups must contain at least {int(min_unique)} unique groups, got {len(uniq)}."
        )

    return arr


def splitter_supports_groups(splitter: Any) -> bool:
    """Return True if ``splitter.split`` consumes the ``groups`` parameter."""
    if isinstance(
        splitter,
        (
            GroupKFold,
            StratifiedGroupKFold,
            GroupShuffleSplit,
            LeaveOneGroupOut,
            LeavePGroupsOut,
        ),
    ):
        return True

    if isinstance(
        splitter,
        (
            KFold,
            StratifiedKFold,
            RepeatedKFold,
            RepeatedStratifiedKFold,
            TimeSeriesSplit,
            _TimeSeriesSplitterAdapter,
            ShuffleSplit,
            StratifiedShuffleSplit,
            LeaveOneOut,
            LeavePOut,
            PredefinedSplit,
        ),
    ):
        return False

    mro_names = {cls.__name__ for cls in getattr(type(splitter), "__mro__", ())}
    if "GroupsConsumerMixin" in mro_names or "_GroupsConsumerMixin" in mro_names:
        return True

    req = getattr(splitter, "__metadata_request__split", None)
    if isinstance(req, dict) and req.get("groups") not in (None, False, "UNUSED"):
        return True

    return bool(getattr(splitter, "requires_groups", False))


def resolve_cv_strategy_name(splitter: Any) -> str:
    """Return canonical strategy name for a CV splitter instance."""
    if isinstance(splitter, StratifiedGroupKFold):
        return "stratified_group_kfold"
    if isinstance(splitter, GroupKFold):
        return "group_kfold"
    if isinstance(splitter, StratifiedKFold):
        return "stratified"
    if isinstance(splitter, KFold):
        return "kfold"
    if isinstance(splitter, (TimeSeriesSplit, _TimeSeriesSplitterAdapter)):
        return "temporal"
    try:
        from acdso.timeseries.splits import TimeSeriesSplitter

        if isinstance(splitter, TimeSeriesSplitter):
            return "temporal"
    except ImportError:
        pass
    return type(splitter).__name__


def _inspect_target(
    y: Any,
    task: str,
) -> Tuple[Optional[int], bool, bool, int, int, int]:
    """Inspect target y -> (n_samples, is_1d_stratifiable, is_auto_clf, n_classes, min_cnt, max_cnt)."""
    if y is None:
        return None, False, False, 0, 0, 0

    y_arr = np.asarray(y)
    if y_arr.ndim == 0:
        raise ValueError("Target y cannot be 0-dimensional.")

    n_samples = int(y_arr.shape[0])
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_1d = y_arr.ravel()
    elif y_arr.ndim == 1:
        y_1d = y_arr
    else:
        return n_samples, False, False, 0, 0, 0

    try:
        tgt_type = type_of_target(y_1d)
    except Exception:
        tgt_type = "unknown"

    is_1d_stratifiable = tgt_type in ("binary", "multiclass")
    task_norm = str(task or "classification").strip().lower()
    is_auto_clf = is_1d_stratifiable and (task_norm != "regression")

    if is_1d_stratifiable and n_samples > 0:
        try:
            _, counts = np.unique(y_1d, return_counts=True)
        except TypeError:
            _, counts = np.unique(pd.Series(y_1d, dtype=object).astype(str).to_numpy(), return_counts=True)
        n_classes = int(len(counts))
        min_cnt = int(counts.min()) if counts.size else 0
        max_cnt = int(counts.max()) if counts.size else 0
    else:
        n_classes, min_cnt, max_cnt = 0, 0, 0

    return n_samples, is_1d_stratifiable, is_auto_clf, n_classes, min_cnt, max_cnt


def build_cv_splitter(
    cv: Any = 5,
    y: Any = None,
    groups: Any = None,
    task: str = "classification",
    cv_strategy: str = "auto",
    random_state: Optional[int] = 42,
) -> Any:
    """Build a scikit-learn compatible CV splitter respecting groups and temporal order.

    Parameters
    ----------
    cv : int or splitter, default=5
        Number of folds (>= 2) or an existing CV splitter object with ``.split``.
    y : array-like, optional
        Target values used for stratification checks and sample count clamping.
    groups : array-like, optional
        Entity/group identifiers for grouped CV.
    task : str, default="classification"
        Task type ("classification" or "regression").
    cv_strategy : str, default="auto"
        One of ``{"auto", "kfold", "stratified", "group_kfold",
        "stratified_group_kfold", "temporal"}``.
    random_state : int, optional, default=42
        Random seed for shuffling splitters where supported.

    Returns
    -------
    splitter
        A CV splitter implementing ``.split(X, y=None, groups=None)``.
    """
    if not isinstance(cv_strategy, str):
        raise ValueError(f"cv_strategy must be a string, got {type(cv_strategy).__name__}.")
    strategy = cv_strategy.strip().lower()
    if strategy not in VALID_CV_STRATEGIES:
        raise ValueError(
            f"Unknown cv_strategy {cv_strategy!r}. Valid options: {VALID_CV_STRATEGIES}."
        )

    if hasattr(cv, "split") and not isinstance(cv, (str, bytes)):
        if groups is not None:
            n_y = _num_samples(y) if y is not None else None
            validate_groups(groups, n_samples=n_y, min_unique=2)
        try:
            from acdso.timeseries.splits import TimeSeriesSplitter

            if isinstance(cv, TimeSeriesSplitter):
                return _TimeSeriesSplitterAdapter(cv)
        except ImportError:
            pass
        return cv

    if isinstance(cv, bool) or not isinstance(cv, (int, np.integer)):
        raise ValueError(f"cv must be an integer >= 2 or a splitter object, got {cv!r}.")
    n_splits = int(cv)
    if n_splits < 2:
        raise ValueError(f"cv must be >= 2, got {n_splits}.")

    n_samples, is_1d_stratifiable, is_auto_clf, n_classes, min_cnt, max_cnt = _inspect_target(y, task)

    groups_arr: Optional[np.ndarray] = None
    n_unique_groups = 0
    if groups is not None:
        groups_arr = validate_groups(groups, n_samples=n_samples, min_unique=2)
        n_unique_groups = int(len(np.unique(groups_arr)))

    shuf = random_state is not None
    rs_val = random_state if shuf else None

    if strategy == "group_kfold":
        if groups_arr is None:
            raise ValueError("cv_strategy='group_kfold' requires non-None groups.")
        k = max(2, min(n_splits, n_unique_groups))
        return GroupKFold(n_splits=k)

    if strategy == "stratified_group_kfold":
        if groups_arr is None:
            raise ValueError("cv_strategy='stratified_group_kfold' requires non-None groups.")
        if y is None or not is_1d_stratifiable or n_classes < 2 or max_cnt < 2:
            raise ValueError(
                "cv_strategy='stratified_group_kfold' requires a 1D binary or multiclass target y "
                "with at least 2 classes."
            )
        k = max(2, min(n_splits, n_unique_groups, max_cnt))
        return StratifiedGroupKFold(n_splits=k, shuffle=shuf, random_state=rs_val)

    if strategy == "temporal":
        if n_samples is not None:
            if n_samples < 3:
                raise ValueError(
                    f"cv_strategy='temporal' requires at least 3 samples, got {n_samples}."
                )
            k = max(2, min(n_splits, n_samples - 1))
        elif groups_arr is not None:
            if len(groups_arr) < 3:
                raise ValueError(
                    f"cv_strategy='temporal' requires at least 3 samples, got {len(groups_arr)}."
                )
            k = max(2, min(n_splits, len(groups_arr) - 1))
        else:
            k = n_splits
        return TimeSeriesSplit(n_splits=k)

    if strategy == "stratified":
        if y is None or not is_1d_stratifiable or n_classes < 2 or min_cnt < 2:
            raise ValueError(
                "cv_strategy='stratified' requires a 1D binary or multiclass target y "
                "with at least 2 samples per class."
            )
        k = max(2, min(n_splits, min_cnt))
        return StratifiedKFold(n_splits=k, shuffle=shuf, random_state=rs_val)

    if strategy == "kfold":
        eff_n = n_samples if n_samples is not None else (len(groups_arr) if groups_arr is not None else None)
        if eff_n is not None:
            if eff_n < 2:
                raise ValueError(f"cv_strategy='kfold' requires at least 2 samples, got {eff_n}.")
            k = max(2, min(n_splits, eff_n))
        else:
            k = n_splits
        return KFold(n_splits=k, shuffle=shuf, random_state=rs_val)

    # strategy == "auto"
    if groups_arr is not None:
        k = max(2, min(n_splits, n_unique_groups))
        if (
            is_auto_clf
            and n_classes >= 2
            and min_cnt >= 2
            and n_unique_groups >= n_splits
            and k <= max_cnt
        ):
            return StratifiedGroupKFold(n_splits=k, shuffle=shuf, random_state=rs_val)
        return GroupKFold(n_splits=k)

    if is_auto_clf and n_classes >= 2 and min_cnt >= 2:
        k = max(2, min(n_splits, min_cnt))
        return StratifiedKFold(n_splits=k, shuffle=shuf, random_state=rs_val)

    eff_n = n_samples
    if eff_n is not None:
        if eff_n < 2:
            raise ValueError(f"At least 2 samples required for CV, got {eff_n}.")
        k = max(2, min(n_splits, eff_n))
    else:
        k = n_splits
    return KFold(n_splits=k, shuffle=shuf, random_state=rs_val)


def validate_no_group_leakage(
    splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    groups: Any,
) -> None:
    """Verify that no entity/group appears in both train and validation indices of any fold."""
    groups_arr = validate_groups(groups, min_unique=1)
    n_groups_total = len(groups_arr)

    for fold_idx, (tr, va) in enumerate(splits):
        tr_idx = np.asarray(tr, dtype=np.intp)
        va_idx = np.asarray(va, dtype=np.intp)
        if tr_idx.size == 0 or va_idx.size == 0:
            raise ValueError(
                f"CV fold {fold_idx} has an empty split (train={tr_idx.size}, val={va_idx.size})."
            )
        if int(tr_idx.min()) < 0 or int(va_idx.min()) < 0:
            raise ValueError(
                f"Split indices cannot be negative in fold {fold_idx}."
            )
        if int(tr_idx.max()) >= n_groups_total or int(va_idx.max()) >= n_groups_total:
            raise ValueError(
                f"Split indices exceed groups length ({n_groups_total}) in fold {fold_idx}."
            )
        tr_groups = set(groups_arr[tr_idx].tolist())
        va_groups = set(groups_arr[va_idx].tolist())
        overlap = tr_groups & va_groups
        if overlap:
            sample = ", ".join(repr(g) for g in sorted(overlap, key=repr)[:5])
            raise ValueError(
                f"Group leakage detected in fold {fold_idx}: {len(overlap)} overlapping group(s) "
                f"(e.g. {sample})."
            )


def iter_cv_splits(
    splitter: Any,
    X: Any,
    y: Any = None,
    groups: Any = None,
    *,
    validate_groups_leakage: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Materialize CV splits without emitting ignored-groups warnings and optionally verify zero leakage."""
    if groups is not None and splitter_supports_groups(splitter):
        raw_splits = splitter.split(X, y, groups)
    else:
        try:
            raw_splits = splitter.split(X, y)
        except TypeError:
            raw_splits = splitter.split(X)

    splits = [
        (np.asarray(tr, dtype=np.intp), np.asarray(va, dtype=np.intp))
        for tr, va in raw_splits
    ]
    if not splits:
        raise ValueError("CV splitter produced zero folds.")

    for fold_idx, (tr, va) in enumerate(splits):
        if tr.size == 0 or va.size == 0:
            raise ValueError(
                f"CV fold {fold_idx} is empty (train={tr.size}, val={va.size})."
            )

    if validate_groups_leakage and groups is not None and splitter_supports_groups(splitter):
        validate_no_group_leakage(splits, groups)

    return splits


def _num_samples(X: Any) -> int:
    if sparse.issparse(X) or hasattr(X, "shape"):
        shape = getattr(X, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
    if hasattr(X, "__len__"):
        return len(X)
    return int(np.asarray(X).shape[0])


def _slice_sample(obj: Any, idx: np.ndarray) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "iloc"):
        return obj.iloc[idx]
    if sparse.issparse(obj):
        try:
            return obj[idx]
        except TypeError:
            return obj.tocsr()[idx]
    return np.asarray(obj)[idx]


def sample_by_groups(
    X: Any,
    y: Any = None,
    groups: Any = None,
    max_rows: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: Optional[int] = 42,
    stratify: bool = False,
    min_groups_per_class: int = 2,
    min_groups_total: int = 2,
) -> Tuple[Any, Any, Any]:
    """Subsample rows while keeping whole entity groups intact.

    Always returns a 3-tuple ``(X_sub, y_sub, groups_sub)``.
    """
    n = _num_samples(X)
    if n == 0:
        return X, y, (validate_groups(groups, n_samples=0, min_unique=0) if groups is not None else None)

    target_rows = n
    if frac is not None:
        frac_f = float(frac)
        if frac_f <= 0.0:
            raise ValueError(f"frac must be > 0, got {frac}.")
        if frac_f < 1.0:
            target_rows = min(target_rows, max(1, int(n * frac_f)))

    if max_rows is not None:
        max_r = int(max_rows)
        if max_r <= 0:
            raise ValueError(f"max_rows must be > 0, got {max_rows}.")
        target_rows = min(target_rows, max_r)

    groups_arr: Optional[np.ndarray] = None
    if groups is not None:
        groups_arr = validate_groups(groups, n_samples=n, min_unique=1)

    if target_rows >= n:
        return X, y, groups_arr

    rng = np.random.RandomState(random_state)

    if groups_arr is not None:
        # Preserve original appearance order per group, shuffle unique groups deterministically
        codes, uniques = pd.factorize(pd.Series(groups_arr, dtype=object), sort=False)
        n_groups = len(uniques)
        group_indices: List[List[int]] = [[] for _ in range(n_groups)]
        for pos, code in enumerate(codes.tolist()):
            group_indices[code].append(pos)

        min_g_total = min(n_groups, max(2, int(min_groups_total)))
        selected_pos: List[int] = []

        if stratify and y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim == 2 and y_arr.shape[1] == 1:
                y_1d = y_arr.ravel()
            else:
                y_1d = y_arr
            if y_1d.ndim == 1 and len(y_1d) == n:
                try:
                    y_codes, y_uniques = pd.factorize(pd.Series(y_1d, dtype=object), sort=False)
                except TypeError:
                    y_codes = np.empty(0, dtype=np.int64)
                    y_uniques = np.empty(0, dtype=object)
                n_cls = len(y_uniques)
                if n_cls >= 2:
                    group_classes: List[set] = [set() for _ in range(n_groups)]
                    group_class_counts: List[np.ndarray] = [
                        np.zeros(n_cls, dtype=np.int64) for _ in range(n_groups)
                    ]
                    orig_class_rows = np.zeros(n_cls, dtype=np.int64)
                    groups_containing_c: List[List[int]] = [[] for _ in range(n_cls)]
                    groups_by_maj: List[List[int]] = [[] for _ in range(n_cls)]

                    for g_code in range(n_groups):
                        ys_g = y_codes[group_indices[g_code]]
                        ys_valid = ys_g[ys_g >= 0]
                        if ys_valid.size == 0:
                            continue
                        vals, cnts = np.unique(ys_valid, return_counts=True)
                        for ci_raw, cnt_raw in zip(vals.tolist(), cnts.tolist()):
                            ci = int(ci_raw)
                            cnt = int(cnt_raw)
                            group_classes[g_code].add(ci)
                            group_class_counts[g_code][ci] = cnt
                            orig_class_rows[ci] += cnt
                            groups_containing_c[ci].append(g_code)
                        maj_ci = int(vals[int(np.argmax(cnts))])
                        groups_by_maj[maj_ci].append(g_code)

                    total_valid_rows = int(orig_class_rows.sum())
                    if total_valid_rows > 0:
                        orig_class_freq = orig_class_rows.astype(np.float64) / float(total_valid_rows)

                        for ci in range(n_cls):
                            if groups_containing_c[ci]:
                                groups_containing_c[ci] = rng.permutation(groups_containing_c[ci]).tolist()
                            if groups_by_maj[ci]:
                                groups_by_maj[ci] = rng.permutation(groups_by_maj[ci]).tolist()

                        selected_set: set = set()
                        selected_groups: List[int] = []
                        selected_rows = 0
                        selected_class_rows = np.zeros(n_cls, dtype=np.int64)

                        # Phase 1: guarantee >= min_groups_per_class groups containing each class (rarest first)
                        cls_order = sorted(range(n_cls), key=lambda ci: len(groups_containing_c[ci]))
                        for ci in cls_order:
                            need = min(len(groups_containing_c[ci]), max(1, int(min_groups_per_class)))
                            have = sum(1 for g in selected_groups if ci in group_classes[g])
                            if have >= need:
                                continue
                            for g_code in groups_containing_c[ci]:
                                if g_code in selected_set:
                                    continue
                                selected_set.add(g_code)
                                selected_groups.append(g_code)
                                selected_rows += len(group_indices[g_code])
                                selected_class_rows += group_class_counts[g_code]
                                have += 1
                                if have >= need:
                                    break

                        # Phase 2: proportional fill matching original dataset class proportions
                        class_pools: List[List[int]] = []
                        for ci in range(n_cls):
                            seen_ci: set = set()
                            pool_ci: List[int] = []
                            for g_code in groups_by_maj[ci]:
                                if g_code not in seen_ci:
                                    seen_ci.add(g_code)
                                    pool_ci.append(g_code)
                            for g_code in groups_containing_c[ci]:
                                if g_code not in seen_ci:
                                    seen_ci.add(g_code)
                                    pool_ci.append(g_code)
                            class_pools.append(pool_ci)

                        pool_cursors = [0] * n_cls
                        fallback_perm = rng.permutation(n_groups).tolist()
                        fallback_cursor = 0
                        cls_tie_order = rng.permutation(n_cls).tolist()
                        cls_tie_order.sort(key=lambda ci: float(orig_class_freq[ci]), reverse=True)
                        target_valid_rows = float(target_rows) * (float(total_valid_rows) / float(n))

                        while len(selected_groups) < min_g_total or selected_rows < target_rows:
                            selected_valid_rows = int(selected_class_rows.sum())
                            horizon = max(target_valid_rows, float(selected_valid_rows))
                            best_ci: Optional[int] = None
                            best_shortfall = -float("inf")

                            for ci in cls_tie_order:
                                pool_ci = class_pools[ci]
                                while pool_cursors[ci] < len(pool_ci) and pool_ci[pool_cursors[ci]] in selected_set:
                                    pool_cursors[ci] += 1
                                if pool_cursors[ci] >= len(pool_ci):
                                    continue
                                shortfall = horizon * float(orig_class_freq[ci]) - float(selected_class_rows[ci])
                                if shortfall > best_shortfall:
                                    best_shortfall = shortfall
                                    best_ci = ci

                            if best_ci is not None:
                                g_code = class_pools[best_ci][pool_cursors[best_ci]]
                                pool_cursors[best_ci] += 1
                            else:
                                while fallback_cursor < n_groups and fallback_perm[fallback_cursor] in selected_set:
                                    fallback_cursor += 1
                                if fallback_cursor >= n_groups:
                                    break
                                g_code = fallback_perm[fallback_cursor]
                                fallback_cursor += 1

                            selected_set.add(g_code)
                            selected_groups.append(g_code)
                            selected_rows += len(group_indices[g_code])
                            selected_class_rows += group_class_counts[g_code]

                        if len(selected_groups) >= n_groups:
                            return X, y, groups_arr

                        for g_code in selected_groups:
                            selected_pos.extend(group_indices[g_code])

                        idx = np.sort(np.asarray(selected_pos, dtype=np.intp))
                        return _slice_sample(X, idx), _slice_sample(y, idx), groups_arr[idx]

        perm = rng.permutation(n_groups)
        selected_count = 0
        for g_code in perm.tolist():
            rows_for_g = group_indices[g_code]
            selected_pos.extend(rows_for_g)
            selected_count += 1
            if len(selected_pos) >= target_rows and selected_count >= min_g_total:
                break

        if selected_count >= n_groups:
            return X, y, groups_arr

        idx = np.sort(np.asarray(selected_pos, dtype=np.intp))
        return _slice_sample(X, idx), _slice_sample(y, idx), groups_arr[idx]

    idx = np.sort(rng.choice(n, size=target_rows, replace=False).astype(np.intp))
    return _slice_sample(X, idx), _slice_sample(y, idx), None


def group_holdout_split(
    X: Any,
    y: Any = None,
    groups: Any = None,
    test_size: float = 0.2,
    task: str = "classification",
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic single holdout (train_idx, val_idx) split respecting groups."""
    n_samples = _num_samples(X)
    if n_samples < 2:
        raise ValueError(f"Cannot create holdout split with {n_samples} sample(s).")

    if groups is not None:
        groups_arr = validate_groups(groups, n_samples=n_samples, min_unique=2)
        n_unique_groups = int(len(np.unique(groups_arr)))
        if isinstance(test_size, (int, np.integer)) and not isinstance(test_size, bool):
            raw_test_groups = int(test_size)
        else:
            ts_f = float(test_size)
            raw_test_groups = int(round(ts_f * n_unique_groups)) if ts_f < 1.0 else int(ts_f)
        n_test_groups = max(1, min(n_unique_groups - 1, raw_test_groups))
        gss = GroupShuffleSplit(n_splits=1, test_size=n_test_groups, random_state=random_state)
        tr_idx, va_idx = next(gss.split(X, y, groups_arr))
        return np.asarray(tr_idx, dtype=np.intp), np.asarray(va_idx, dtype=np.intp)

    val_size = max(1, min(n_samples - 1, int(round(n_samples * float(test_size))) if test_size < 1.0 else int(test_size)))
    if y is not None and str(task or "classification").strip().lower() == "classification":
        _, is_1d_strat, _, n_cls, min_cnt, _ = _inspect_target(y, task)
        if is_1d_strat and n_cls >= 2 and min_cnt >= 2 and val_size >= n_cls and (n_samples - val_size) >= n_cls:
            try:
                y_arr = np.asarray(y).ravel()
                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=val_size,
                    random_state=random_state,
                )
                tr_idx, va_idx = next(sss.split(np.zeros(n_samples), y_arr))
                return np.asarray(tr_idx, dtype=np.intp), np.asarray(va_idx, dtype=np.intp)
            except Exception:
                pass

    rng = np.random.RandomState(random_state)
    all_idx = np.arange(n_samples, dtype=np.intp)
    rng.shuffle(all_idx)
    return all_idx[val_size:], all_idx[:val_size]
