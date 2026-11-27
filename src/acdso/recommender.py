"""
ACDSO - Collaborative-Filtering Recommender Module (R5d)
========================================================

Provides ``ACDSORecommender``, a collaborative-filtering recommender supporting
explicit ratings and implicit feedback with truncated SVD, NMF, item-based
cosine kNN, user-based cosine kNN, automated 80/20 holdout selection with
full-data refit, cold-start bias/popularity fallbacks, and ranking + rating
evaluation metrics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

_VALID_METHODS = ("auto", "svd", "nmf", "item_knn", "user_knn")


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


def _to_py_scalar(val: Any) -> Any:
    if isinstance(val, np.generic):
        return val.item()
    return val


def _coerce_interactions_df(
    interactions: Any,
    user_col: str,
    item_col: str,
    rating_col: Optional[str],
    *,
    allow_empty: bool = False,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Coerce DataFrame or 2D array/tuples into a validated interactions DataFrame."""
    if interactions is None:
        raise ValueError("interactions cannot be None.")

    if isinstance(interactions, pd.DataFrame):
        if user_col not in interactions.columns or item_col not in interactions.columns:
            raise ValueError(
                f"Required columns ({user_col!r}, {item_col!r}) not found in DataFrame columns {list(interactions.columns)}."
            )
        if rating_col is not None and rating_col not in interactions.columns:
            raise ValueError(
                f"rating_col={rating_col!r} not found in DataFrame columns {list(interactions.columns)}."
            )
        if len(interactions) == 0:
            if allow_empty:
                return interactions.copy(), rating_col
            raise ValueError("interactions DataFrame cannot be empty.")
        cols = [user_col, item_col] + ([rating_col] if rating_col is not None else [])
        df = interactions[cols].copy()
        eff_rating_col = rating_col
    else:
        arr = np.asarray(interactions, dtype=object)
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError(
                f"Array/tuple interactions must have shape (n, 2) or (n, 3), got {arr.shape}."
            )
        if arr.shape[0] == 0:
            if allow_empty:
                empty_cols = {user_col: pd.Series(dtype=object), item_col: pd.Series(dtype=object)}
                eff_r = rating_col
                if arr.shape[1] == 3:
                    eff_r = rating_col or "rating"
                    empty_cols[eff_r] = pd.Series(dtype=np.float64)
                return pd.DataFrame(empty_cols), eff_r
            raise ValueError("interactions cannot be empty.")

        if arr.shape[1] == 2:
            if rating_col is not None:
                raise ValueError(
                    f"rating_col={rating_col!r} specified but interactions only has 2 columns."
                )
            df = pd.DataFrame({user_col: arr[:, 0], item_col: arr[:, 1]})
            eff_rating_col = None
        else:
            eff_rating_col = rating_col or "rating"
            df = pd.DataFrame(
                {
                    user_col: arr[:, 0],
                    item_col: arr[:, 1],
                    eff_rating_col: pd.to_numeric(arr[:, 2], errors="coerce"),
                }
            )

    if df[user_col].isna().any() or df[item_col].isna().any():
        raise ValueError("interactions contains missing user or item identifiers.")

    if eff_rating_col is not None:
        r_vals = pd.to_numeric(df[eff_rating_col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(r_vals)):
            raise ValueError("interactions rating column contains NaN or infinite values.")
        df[eff_rating_col] = r_vals

    return df, eff_rating_col


class ACDSORecommender(BaseEstimator):
    """Collaborative-filtering recommender supporting explicit & implicit interactions.

    Supports truncated SVD (``"svd"``), Non-negative Matrix Factorization
    (``"nmf"``), item-based cosine kNN (``"item_knn"``), user-based cosine kNN
    (``"user_knn"``), and automated holdout model selection (``"auto"``).

    Parameters
    ----------
    method : str, default="auto"
        One of ``"auto"``, ``"svd"``, ``"nmf"``, ``"item_knn"``, ``"user_knn"``.
    n_factors : int, default=20
        Latent dimensionality for SVD and NMF.
    n_neighbors : int, default=20
        Neighborhood size for ``item_knn`` and ``user_knn``.
    l2_reg : float, default=0.05
        L2 regularization strength for biases, singular values, and NMF factors.
    random_state : int, optional, default=42
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level.
    """

    def __init__(
        self,
        method: str = "auto",
        n_factors: int = 20,
        n_neighbors: int = 20,
        l2_reg: float = 0.05,
        random_state: Optional[int] = 42,
        verbose: int = 0,
    ) -> None:
        self.method = method
        self.n_factors = n_factors
        self.n_neighbors = n_neighbors
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose and int(self.verbose) > 0:
            print(f"[ACDSORecommender] {msg}")

    def _validate_hyperparameters(self) -> Tuple[str, int, int, float]:
        if not isinstance(self.method, str) or self.method not in _VALID_METHODS:
            raise ValueError(
                f"Invalid method={self.method!r}. Must be one of {_VALID_METHODS}."
            )
        if isinstance(self.n_factors, bool) or not isinstance(self.n_factors, (int, np.integer)):
            raise ValueError(f"n_factors must be an integer >= 1, got {self.n_factors!r}.")
        eff_factors = int(self.n_factors)
        if eff_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {eff_factors}.")

        if isinstance(self.n_neighbors, bool) or not isinstance(self.n_neighbors, (int, np.integer)):
            raise ValueError(f"n_neighbors must be an integer >= 1, got {self.n_neighbors!r}.")
        eff_neighbors = int(self.n_neighbors)
        if eff_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {eff_neighbors}.")

        if isinstance(self.l2_reg, bool) or not isinstance(
            self.l2_reg, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"l2_reg must be a float >= 0, got {self.l2_reg!r}.")
        eff_l2 = float(self.l2_reg)
        if not math.isfinite(eff_l2) or eff_l2 < 0.0:
            raise ValueError(f"l2_reg must be finite and >= 0, got {eff_l2!r}.")

        return self.method, eff_factors, eff_neighbors, eff_l2

    @staticmethod
    def _aggregate_interactions(
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        eff_rating_col: Optional[str],
    ) -> pd.DataFrame:
        if eff_rating_col is not None:
            agg_df = (
                df.groupby([user_col, item_col], sort=False, as_index=False, observed=True)[
                    eff_rating_col
                ]
                .mean()
            )
            agg_df["__val__"] = agg_df[eff_rating_col].to_numpy(dtype=np.float64)
            return agg_df
        agg_df = df[[user_col, item_col]].drop_duplicates(
            subset=[user_col, item_col], keep="first"
        ).reset_index(drop=True)
        agg_df["__val__"] = np.ones(len(agg_df), dtype=np.float64)
        return agg_df

    def _fit_concrete_method(
        self,
        agg_df: pd.DataFrame,
        user_col: str,
        item_col: str,
        is_implicit: bool,
        chosen_method: str,
        eff_factors: int,
        eff_neighbors: int,
        eff_l2: float,
    ) -> Dict[str, Any]:
        """Fit a single concrete CF method on aggregated interaction rows."""
        u_vals = agg_df[user_col].tolist()
        i_vals = agg_df[item_col].tolist()
        r_vals = agg_df["__val__"].to_numpy(dtype=np.float64)

        u_codes, u_uniques = pd.factorize(pd.Series(u_vals, dtype=object), sort=False)
        i_codes, i_uniques = pd.factorize(pd.Series(i_vals, dtype=object), sort=False)

        idx_to_user = [_to_py_scalar(u) for u in u_uniques.tolist()]
        idx_to_item = [_to_py_scalar(it) for it in i_uniques.tolist()]
        user_to_idx = {u: idx for idx, u in enumerate(idx_to_user)}
        item_to_idx = {it: idx for idx, it in enumerate(idx_to_item)}

        n_u = len(idx_to_user)
        n_i = len(idx_to_item)
        u_idx = u_codes.astype(np.intp, copy=False)
        i_idx = i_codes.astype(np.intp, copy=False)

        user_seen: Dict[int, set] = {u: set() for u in range(n_u)}
        for u_k, i_k in zip(u_idx.tolist(), i_idx.tolist()):
            user_seen[u_k].add(i_k)

        item_counts = np.bincount(i_idx, minlength=n_i).astype(np.float64)
        item_popularity = item_counts / float(max(1, n_u))

        R = sparse.csr_matrix((r_vals, (u_idx, i_idx)), shape=(n_u, n_i), dtype=np.float64)

        if is_implicit:
            global_mean = 0.0
            user_biases = np.zeros(n_u, dtype=np.float64)
            item_biases = np.zeros(n_i, dtype=np.float64)
            min_rating = 0.0
            max_rating = 1.0
            E = R
        else:
            global_mean = float(np.mean(r_vals))
            reg_b = max(1.0, 10.0 * eff_l2)
            item_sum = np.bincount(i_idx, weights=r_vals - global_mean, minlength=n_i)
            item_biases = item_sum / (item_counts + reg_b)

            u_counts = np.bincount(u_idx, minlength=n_u).astype(np.float64)
            u_resid = r_vals - global_mean - item_biases[i_idx]
            user_sum = np.bincount(u_idx, weights=u_resid, minlength=n_u)
            user_biases = user_sum / (u_counts + reg_b)

            min_rating = float(np.min(r_vals))
            max_rating = float(np.max(r_vals))
            res_vals = r_vals - (global_mean + user_biases[u_idx] + item_biases[i_idx])
            E = sparse.csr_matrix(
                (res_vals, (u_idx, i_idx)), shape=(n_u, n_i), dtype=np.float64
            )

        state: Dict[str, Any] = {
            "method": chosen_method,
            "is_implicit": is_implicit,
            "idx_to_user": idx_to_user,
            "idx_to_item": idx_to_item,
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
            "n_users": n_u,
            "n_items": n_i,
            "user_seen": user_seen,
            "item_popularity": item_popularity,
            "global_mean": global_mean,
            "user_biases": user_biases,
            "item_biases": item_biases,
            "min_rating": min_rating,
            "max_rating": max_rating,
            "R": R,
            "E": E,
            "P": None,
            "Q": None,
            "nmf_shift": 0.0,
            "sim_matrix": None,
            "eff_neighbors": eff_neighbors,
        }

        min_dim = min(n_u, n_i)
        if chosen_method == "svd":
            has_residual = bool(E.nnz > 0 and np.any(np.abs(E.data) > 1e-12))
            if min_dim >= 2 and has_residual:
                U = s = Vt = None
                if min_dim <= 3 or eff_factors >= min_dim - 1:
                    U, s, Vt = np.linalg.svd(E.toarray(), full_matrices=False)
                    k_keep = min(eff_factors, len(s))
                    U = U[:, :k_keep]
                    s = s[:k_keep]
                    Vt = Vt[:k_keep, :]
                else:
                    k_keep = min(eff_factors, min_dim - 1)
                    rng = np.random.RandomState(self.random_state)
                    v0 = rng.randn(min_dim)
                    try:
                        U, s, Vt = svds(E, k=k_keep, v0=v0)
                        order = np.argsort(-s)
                        U = U[:, order]
                        s = s[order]
                        Vt = Vt[order, :]
                    except Exception:
                        if min_dim <= 500:
                            U, s, Vt = np.linalg.svd(E.toarray(), full_matrices=False)
                            k_keep = min(eff_factors, len(s))
                            U = U[:, :k_keep]
                            s = s[:k_keep]
                            Vt = Vt[:k_keep, :]
                if U is not None and s is not None and Vt is not None:
                    s_damped = np.maximum(s, 0.0) / (1.0 + eff_l2)
                    sqrt_s = np.sqrt(s_damped)
                    state["P"] = U * sqrt_s[None, :]
                    state["Q"] = Vt.T * sqrt_s[None, :]
        elif chosen_method == "nmf":
            eff_k = max(1, min(eff_factors, n_u, n_i))
            if (not is_implicit) and (n_u * n_i <= 250_000):
                mat = np.broadcast_to(
                    global_mean + user_biases[:, None] + item_biases[None, :], (n_u, n_i)
                ).copy()
                mat[u_idx, i_idx] = r_vals
                shift = min(0.0, float(np.min(mat)))
                mat_pos = np.maximum(mat - shift, 0.0)
            else:
                shift = 0.0
                mat_pos = sparse.csr_matrix(
                    (np.maximum(r_vals, 0.0), (u_idx, i_idx)),
                    shape=(n_u, n_i),
                    dtype=np.float64,
                )
            init_mode = "nndsvda" if eff_k <= min(n_u, n_i) else "random"
            nmf = NMF(
                n_components=eff_k,
                init=init_mode,
                alpha_W=eff_l2,
                alpha_H=eff_l2,
                random_state=self.random_state,
                max_iter=300,
            )
            W = nmf.fit_transform(mat_pos)
            H = nmf.components_.T
            state["P"] = np.asarray(W, dtype=np.float64)
            state["Q"] = np.asarray(H, dtype=np.float64)
            state["nmf_shift"] = float(shift)
        elif chosen_method == "item_knn":
            if n_i >= 2:
                state["eff_neighbors"] = max(1, min(eff_neighbors, n_i - 1))
                if n_i <= 4000:
                    sim = cosine_similarity(E.T)
                    np.fill_diagonal(sim, 0.0)
                    state["sim_matrix"] = np.asarray(sim, dtype=np.float64)
                else:
                    state["sim_matrix"] = None
                    if is_implicit:
                        X_norm = normalize(E.T, norm="l2", axis=1, copy=True)
                        s_vec = np.asarray(X_norm.sum(axis=0)).ravel()
                        row_norm_sq = np.asarray(X_norm.multiply(X_norm).sum(axis=1)).ravel()
                        state["pos_sim_row_sum"] = np.maximum(
                            np.asarray(X_norm @ s_vec).ravel() - row_norm_sq, 0.0
                        )
        elif chosen_method == "user_knn":
            if n_u >= 2:
                state["eff_neighbors"] = max(1, min(eff_neighbors, n_u - 1))
                if n_u <= 4000:
                    sim = cosine_similarity(E)
                    np.fill_diagonal(sim, 0.0)
                    state["sim_matrix"] = np.asarray(sim, dtype=np.float64)
                else:
                    state["sim_matrix"] = None
        else:
            raise ValueError(f"Unsupported concrete recommender method: {chosen_method!r}")

        return state

    @staticmethod
    def _fallback_score_item_vec(state: Dict[str, Any], u_idx: Optional[int]) -> np.ndarray:
        """Vectorized baseline/popularity fallback across all known items for a user."""
        if state["is_implicit"]:
            return state["item_popularity"].copy()
        bu = float(state["user_biases"][u_idx]) if u_idx is not None else 0.0
        return state["global_mean"] + bu + state["item_biases"]

    @classmethod
    def _score_all_items_for_user(cls, state: Dict[str, Any], u_idx: Optional[int]) -> np.ndarray:
        """Compute predicted scores across all known items for user index ``u_idx`` (or None if unseen)."""
        n_i = state["n_items"]
        if u_idx is None:
            return cls._fallback_score_item_vec(state, None)

        method = state["method"]
        is_implicit = state["is_implicit"]

        if method == "svd":
            if state["P"] is None or state["Q"] is None:
                scores = cls._fallback_score_item_vec(state, u_idx)
            else:
                latent = state["Q"] @ state["P"][u_idx]
                if is_implicit:
                    scores = latent
                else:
                    scores = cls._fallback_score_item_vec(state, u_idx) + latent
        elif method == "nmf":
            if state["P"] is None or state["Q"] is None:
                scores = cls._fallback_score_item_vec(state, u_idx)
            else:
                scores = (state["Q"] @ state["P"][u_idx]) + float(state["nmf_shift"])
        elif method == "item_knn":
            scores = cls._fallback_score_item_vec(state, u_idx).copy()
            E = state["E"]
            u_row = E.getrow(u_idx)
            seen_cols = u_row.indices
            seen_vals = u_row.data
            if seen_cols.size > 0 and n_i >= 2:
                k = int(state["eff_neighbors"])
                sim_mat = state["sim_matrix"]
                if sim_mat is not None:
                    sim_sub = sim_mat[:, seen_cols]  # shape (n_i, len(seen))
                else:
                    # On-demand cosine similarity for > 4000 items
                    sim_sub = cosine_similarity(E.T, E[:, seen_cols].T)
                    for pos, col_id in enumerate(seen_cols):
                        sim_sub[col_id, pos] = 0.0

                if sim_sub.shape[1] > k:
                    abs_sub = np.abs(sim_sub)
                    top_idx = np.argpartition(-abs_sub, kth=k - 1, axis=1)[:, :k]
                    rows_idx = np.arange(n_i)[:, None]
                    sel_sim = sim_sub[rows_idx, top_idx]
                    sel_vals = seen_vals[top_idx]
                else:
                    sel_sim = sim_sub
                    sel_vals = np.broadcast_to(seen_vals[None, :], sim_sub.shape)

                if is_implicit:
                    pos_sim = np.maximum(sel_sim, 0.0)
                    num = np.sum(pos_sim, axis=1)
                    if sim_mat is not None:
                        denom = np.sum(np.maximum(sim_mat, 0.0), axis=1)
                    elif state.get("pos_sim_row_sum") is not None:
                        denom = state["pos_sim_row_sum"]
                    else:
                        denom = np.full(n_i, float(max(1, len(seen_cols))), dtype=np.float64)
                    valid = denom > 1e-12
                    scores_knn = np.zeros(n_i, dtype=np.float64)
                    scores_knn[valid] = num[valid] / (denom[valid] + 1e-12)
                    scores = np.where(valid, scores_knn, state["item_popularity"])
                else:
                    denom = np.sum(np.abs(sel_sim), axis=1)
                    num = np.sum(sel_sim * sel_vals, axis=1)
                    valid = denom > 1e-12
                    scores[valid] = scores[valid] + num[valid] / denom[valid]
        elif method == "user_knn":
            scores = cls._fallback_score_item_vec(state, u_idx).copy()
            E = state["E"]
            n_u = state["n_users"]
            if n_u >= 2:
                k = int(state["eff_neighbors"])
                sim_mat = state["sim_matrix"]
                if sim_mat is not None:
                    u_sims = sim_mat[u_idx].copy()
                else:
                    u_sims = cosine_similarity(E.getrow(u_idx), E).ravel()
                u_sims[u_idx] = 0.0

                if is_implicit:
                    pos_sims = np.maximum(u_sims, 0.0)
                    if n_u > k:
                        top_u = np.argpartition(-pos_sims, kth=k - 1)[:k]
                    else:
                        top_u = np.arange(n_u)
                    w = pos_sims[top_u]
                    w_sum = float(np.sum(w))
                    if w_sum > 1e-12:
                        scores = np.asarray(E[top_u].T @ w, dtype=np.float64).ravel() / (w_sum + 1e-12)
                else:
                    abs_sims = np.abs(u_sims)
                    if n_u > k:
                        top_u = np.argpartition(-abs_sims, kth=k - 1)[:k]
                    else:
                        top_u = np.arange(n_u)
                    w = u_sims[top_u]
                    E_sub = E[top_u]
                    nz_mask = (E_sub != 0).astype(np.float64)
                    num = np.asarray(E_sub.T @ w, dtype=np.float64).ravel()
                    denom = np.asarray(nz_mask.T @ np.abs(w), dtype=np.float64).ravel()
                    valid = denom > 1e-12
                    scores[valid] = scores[valid] + num[valid] / denom[valid]

        if not is_implicit and state["max_rating"] > state["min_rating"]:
            scores = np.clip(scores, state["min_rating"], state["max_rating"])
        return np.nan_to_num(scores, nan=state["global_mean"], posinf=state["max_rating"], neginf=state["min_rating"])

    @classmethod
    def _predict_from_state(
        cls,
        state: Dict[str, Any],
        users: Any,
        items: Any,
    ) -> np.ndarray:
        u_list = list(users)
        i_list = list(items)
        if len(u_list) != len(i_list):
            raise ValueError(
                f"users ({len(u_list)}) and items ({len(i_list)}) must have equal length."
            )
        n = len(u_list)
        if n == 0:
            return np.empty(0, dtype=np.float64)

        u_map = state["user_to_idx"]
        i_map = state["item_to_idx"]
        is_implicit = state["is_implicit"]
        fallback_val = 0.0 if is_implicit else float(state["global_mean"])

        # Group queries by user so item vectors are computed at most once per user
        preds = np.full(n, fallback_val, dtype=np.float64)
        user_queries: Dict[Optional[int], List[Tuple[int, Any]]] = {}
        for pos, (u, it) in enumerate(zip(u_list, i_list)):
            u_key = _to_py_scalar(u)
            it_key = _to_py_scalar(it)
            u_idx = u_map.get(u_key)
            user_queries.setdefault(u_idx, []).append((pos, it_key))

        for u_idx, pos_items in user_queries.items():
            if u_idx is None:
                bu = 0.0
                for pos, it_key in pos_items:
                    it_idx = i_map.get(it_key)
                    if it_idx is not None:
                        if is_implicit:
                            preds[pos] = float(state["item_popularity"][it_idx])
                        else:
                            val = state["global_mean"] + bu + float(state["item_biases"][it_idx])
                            if state["max_rating"] > state["min_rating"]:
                                val = min(max(val, state["min_rating"]), state["max_rating"])
                            preds[pos] = val
                continue

            item_scores = cls._score_all_items_for_user(state, u_idx)
            bu = 0.0 if is_implicit else float(state["user_biases"][u_idx])
            for pos, it_key in pos_items:
                it_idx = i_map.get(it_key)
                if it_idx is not None:
                    preds[pos] = float(item_scores[it_idx])
                else:
                    if is_implicit:
                        preds[pos] = 0.0
                    else:
                        val = state["global_mean"] + bu
                        if state["max_rating"] > state["min_rating"]:
                            val = min(max(val, state["min_rating"]), state["max_rating"])
                        preds[pos] = val

        return np.nan_to_num(preds, nan=fallback_val, posinf=state["max_rating"], neginf=state["min_rating"])

    @classmethod
    def _recommend_from_state(
        cls,
        state: Dict[str, Any],
        user_id: Any,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[Any, float]]:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 0:
            raise ValueError(f"n must be an integer >= 0, got {n!r}.")
        top_n = int(n)
        if top_n == 0 or state["n_items"] == 0:
            return []

        u_key = _to_py_scalar(user_id)
        u_idx = state["user_to_idx"].get(u_key)
        scores = cls._score_all_items_for_user(state, u_idx)

        candidate_mask = np.ones(state["n_items"], dtype=bool)
        if exclude_seen and u_idx is not None:
            seen = state["user_seen"].get(u_idx)
            if seen:
                candidate_mask[list(seen)] = False

        cand_indices = np.flatnonzero(candidate_mask)
        if cand_indices.size == 0:
            return []

        cand_scores = scores[cand_indices]
        # Stable descending sort by score (ties broken deterministically by item index)
        order = np.argsort(-cand_scores, kind="mergesort")
        top_indices = cand_indices[order[:top_n]]
        top_scores = scores[top_indices]
        idx_to_item = state["idx_to_item"]
        return [
            (idx_to_item[int(i_k)], float(sc))
            for i_k, sc in zip(top_indices.tolist(), top_scores.tolist())
        ]

    @classmethod
    def _evaluate_state(
        cls,
        state: Dict[str, Any],
        df_eval: pd.DataFrame,
        user_col: str,
        item_col: str,
        eff_rating_col: Optional[str],
        k: int,
    ) -> Dict[str, Any]:
        if len(df_eval) == 0:
            return {
                "rmse": 0.0,
                "mae": 0.0,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "ndcg_at_k": 0.0,
                "k": int(k),
                "n_interactions": 0,
                "n_users_evaluated": 0,
            }

        preds = cls._predict_from_state(state, df_eval[user_col], df_eval[item_col])
        if eff_rating_col is not None:
            y_true = df_eval[eff_rating_col].to_numpy(dtype=np.float64)
        else:
            y_true = np.ones(len(df_eval), dtype=np.float64)

        err = preds - y_true
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))

        if eff_rating_col is not None:
            r_min = float(np.min(y_true))
            r_max = float(np.max(y_true))
            rel_thresh = r_min if r_max <= r_min else (r_min + 0.5 * (r_max - r_min))
        else:
            rel_thresh = 0.5

        discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
        precisions: List[float] = []
        recalls: List[float] = []
        ndcgs: List[float] = []

        for u_val, grp in df_eval.groupby(user_col, sort=False, observed=True):
            if eff_rating_col is not None:
                rel_mask = grp[eff_rating_col].to_numpy(dtype=np.float64) >= rel_thresh
                rel_items = set(_to_py_scalar(it) for it in grp.loc[rel_mask, item_col].tolist())
            else:
                rel_items = set(_to_py_scalar(it) for it in grp[item_col].tolist())

            if not rel_items:
                continue

            recs = cls._recommend_from_state(state, u_val, n=k, exclude_seen=True)
            rec_items = [it for it, _ in recs]
            hits = np.array([1.0 if it in rel_items else 0.0 for it in rec_items], dtype=np.float64)
            n_hits = float(np.sum(hits))

            precisions.append(n_hits / float(k))
            recalls.append(n_hits / float(len(rel_items)))

            if hits.size > 0:
                dcg = float(np.sum(hits * discounts[: hits.size]))
                ideal_len = min(k, len(rel_items))
                idcg = float(np.sum(discounts[:ideal_len]))
                ndcgs.append(dcg / idcg if idcg > 0.0 else 0.0)
            else:
                ndcgs.append(0.0)

        return {
            "rmse": float(rmse if math.isfinite(rmse) else 0.0),
            "mae": float(mae if math.isfinite(mae) else 0.0),
            "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
            "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "k": int(k),
            "n_interactions": int(len(df_eval)),
            "n_users_evaluated": int(len(precisions)),
        }

    def fit(
        self,
        interactions: Any,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None,
    ) -> "ACDSORecommender":
        """Fit collaborative-filtering recommender on explicit or implicit interactions."""
        method_req, eff_factors, eff_neighbors, eff_l2 = self._validate_hyperparameters()
        df_raw, eff_rating_col = _coerce_interactions_df(
            interactions, user_col, item_col, rating_col, allow_empty=False
        )
        n_raw = int(len(df_raw))
        agg_df = self._aggregate_interactions(df_raw, user_col, item_col, eff_rating_col)
        n_unique = int(len(agg_df))
        is_implicit = eff_rating_col is None

        candidate_summaries: List[Dict[str, Any]] = []
        if method_req == "auto":
            n_u_total = int(agg_df[user_col].nunique())
            n_i_total = int(agg_df[item_col].nunique())
            if n_unique < 10 or min(n_u_total, n_i_total) < 2:
                best_method = "svd" if min(n_u_total, n_i_total) >= 2 else "item_knn"
            else:
                rng = np.random.RandomState(self.random_state)
                perm = rng.permutation(n_unique)
                split_pt = max(1, min(n_unique - 1, int(round(0.8 * n_unique))))
                tr_agg = agg_df.iloc[perm[:split_pt]].reset_index(drop=True)
                va_agg = agg_df.iloc[perm[split_pt:]].reset_index(drop=True)
                va_eval_df = va_agg[[user_col, item_col]].copy()
                if eff_rating_col is not None:
                    va_eval_df[eff_rating_col] = va_agg["__val__"].to_numpy(dtype=np.float64)

                tie_order = {"svd": 4, "item_knn": 3, "user_knn": 2, "nmf": 1}
                scored_cands: List[Dict[str, Any]] = []
                for m_name in ("svd", "nmf", "item_knn", "user_knn"):
                    try:
                        cand_state = self._fit_concrete_method(
                            tr_agg,
                            user_col,
                            item_col,
                            is_implicit,
                            m_name,
                            eff_factors,
                            eff_neighbors,
                            eff_l2,
                        )
                        ev = self._evaluate_state(
                            cand_state,
                            va_eval_df,
                            user_col,
                            item_col,
                            eff_rating_col,
                            k=10,
                        )
                        scored_cands.append({"method": m_name, **ev})
                    except Exception as exc:
                        self._log(f"Auto candidate {m_name} failed: {exc}")

                if scored_cands:
                    if is_implicit:
                        best_cand = max(
                            scored_cands,
                            key=lambda d: (
                                float(d["ndcg_at_k"]),
                                float(d["recall_at_k"]),
                                tie_order.get(d["method"], 0),
                            ),
                        )
                    else:
                        best_cand = min(
                            scored_cands,
                            key=lambda d: (
                                float(d["rmse"]),
                                float(d["mae"]),
                                -tie_order.get(d["method"], 0),
                            ),
                        )
                    best_method = str(best_cand["method"])
                    candidate_summaries = scored_cands
                else:
                    best_method = "svd"
        else:
            best_method = method_req

        final_state = self._fit_concrete_method(
            agg_df,
            user_col,
            item_col,
            is_implicit,
            best_method,
            eff_factors,
            eff_neighbors,
            eff_l2,
        )

        self._state_ = final_state
        self.best_method_ = best_method
        self.user_col_ = user_col
        self.item_col_ = item_col
        self.rating_col_ = eff_rating_col
        self.is_implicit_ = is_implicit
        self.n_users_ = int(final_state["n_users"])
        self.n_items_ = int(final_state["n_items"])
        self.candidates_ = _sanitize_json_obj(candidate_summaries)
        self.audit_trail_ = _sanitize_json_obj(
            {
                "method_requested": self.method,
                "best_method": self.best_method_,
                "is_implicit": is_implicit,
                "n_factors": eff_factors,
                "n_neighbors": eff_neighbors,
                "l2_reg": eff_l2,
                "n_users": self.n_users_,
                "n_items": self.n_items_,
                "n_raw_interactions": n_raw,
                "n_unique_interactions": n_unique,
                "global_mean": float(final_state["global_mean"]),
                "candidates": candidate_summaries,
            }
        )
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "_state_") or self._state_ is None:
            raise RuntimeError("ACDSORecommender is not fitted. Call fit() first.")

    def predict(self, users: Any, items: Any) -> np.ndarray:
        """Predict rating or affinity scores for parallel ``(users, items)`` pairs."""
        self._require_fitted()
        return self._predict_from_state(self._state_, users, items)

    def recommend(
        self,
        user_id: Any,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[Any, float]]:
        """Return top-``n`` ``(item_id, score)`` recommendations sorted descending."""
        self._require_fitted()
        return self._recommend_from_state(
            self._state_, user_id, n=n, exclude_seen=exclude_seen
        )

    def evaluate(
        self,
        interactions: Any,
        k: int = 10,
        user_col: Optional[str] = None,
        item_col: Optional[str] = None,
        rating_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate recommender on holdout interactions returning RMSE, MAE, Precision@k, Recall@k, NDCG@k."""
        self._require_fitted()
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
            raise ValueError(f"k must be an integer >= 1, got {k!r}.")
        eff_k = int(k)

        eff_u_col = user_col if user_col is not None else self.user_col_
        eff_i_col = item_col if item_col is not None else self.item_col_
        if rating_col is not None:
            eff_r_col: Optional[str] = rating_col
        elif isinstance(interactions, pd.DataFrame) and self.rating_col_ is not None and self.rating_col_ in interactions.columns:
            eff_r_col = self.rating_col_
        else:
            eff_r_col = None

        df_eval, resolved_r_col = _coerce_interactions_df(
            interactions, eff_u_col, eff_i_col, eff_r_col, allow_empty=True
        )
        metrics = self._evaluate_state(
            self._state_, df_eval, eff_u_col, eff_i_col, resolved_r_col, k=eff_k
        )
        return _sanitize_json_obj(metrics)
