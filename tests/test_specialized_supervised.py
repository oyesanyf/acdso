"""Unit and integration tests for Milestone M4 / R5 specialized supervised modules."""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, sparse
from sklearn.base import clone
from sklearn.model_selection import KFold

from acdso.multilabel import ACDSOMultiLabelClassifier
from acdso.ranking import ACDSORanker, ndcg_at_k
from acdso.recommender import ACDSORecommender
from acdso.survival import (
    ACDSOSurvivalAnalyzer,
    _breslow_nll_and_grad,
    _build_descending_time_blocks,
    concordance_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_strict_json(obj: Any) -> None:
    dumped = json.dumps(obj, allow_nan=False)
    assert isinstance(dumped, str) and len(dumped) > 0


def _bruteforce_c_index(times: np.ndarray, scores: np.ndarray, events: np.ndarray) -> float:
    """Reference O(n^2) Harrell's C-index implementation for cross-validation."""
    concordant = 0.0
    comparable = 0
    n = len(times)
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if (times[i] < times[j]) or (times[i] == times[j] and events[j] == 0):
                comparable += 1
                if scores[i] > scores[j]:
                    concordant += 1.0
                elif scores[i] == scores[j]:
                    concordant += 0.5
    return 0.5 if comparable == 0 else concordant / float(comparable)


def _make_multilabel_data(n_rows: int = 120, seed: int = 42):
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n_rows)
    x1 = rng.randn(n_rows)
    x2 = rng.randn(n_rows)
    y0 = (x0 + 0.15 * rng.randn(n_rows) > 0).astype(int)
    y1 = (x1 + 0.15 * rng.randn(n_rows) > 0).astype(int)
    y2 = (x0 + x1 + 0.15 * rng.randn(n_rows) > 0).astype(int)
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    Y = np.column_stack([y0, y1, y2])
    return X, Y


# ===========================================================================
# R5a: ACDSOMultiLabelClassifier
# ===========================================================================


@pytest.mark.parametrize(
    "strategy",
    ["auto", "multi_output", "classifier_chain", "random_forest", "extra_trees"],
)
def test_multilabel_all_strategies_fit_predict_proba(strategy: str) -> None:
    X, Y = _make_multilabel_data(n_rows=100, seed=42)
    clf = ACDSOMultiLabelClassifier(strategy=strategy, cv=3, random_state=42)
    assert clf.fit(X.iloc[:75], Y[:75]) is clf

    preds = clf.predict(X.iloc[75:])
    proba = clf.predict_proba(X.iloc[75:])
    assert preds.shape == (25, 3)
    assert set(np.unique(preds)).issubset({0, 1})
    assert proba.shape == (25, 3)
    assert proba.dtype == np.float64
    assert np.all(np.isfinite(proba))
    assert np.all((proba >= 0.0) & (proba <= 1.0))

    ev = clf.evaluate(X.iloc[75:], Y[75:])
    for key in ("f1_micro", "f1_macro", "hamming_loss", "subset_accuracy", "jaccard_samples"):
        assert key in ev
        assert 0.0 <= ev[key] <= 1.0
    assert clf.score(X.iloc[75:], Y[75:]) > 0.5
    _assert_strict_json(clf.audit_trail_)


def test_multilabel_target_coercions_and_text_cols() -> None:
    rng = np.random.RandomState(11)
    n = 90
    x0 = rng.randn(n)
    y0 = (x0 > 0).astype(int)
    y1 = (rng.randn(n) > 0).astype(int)
    texts = [
        ("urgent billing invoice payment" if y0[i] else "general product feedback question")
        + " "
        + ("shipping delivery tracking delayed" if y1[i] else "account profile settings update")
        for i in range(n)
    ]
    X = pd.DataFrame({"x0": x0, "notes": texts})

    # Delimited string targets
    delim_y = [
        "|".join(([("billing")] if y0[i] else []) + ([("shipping")] if y1[i] else []))
        for i in range(n)
    ]
    delim_y = [val if val else "general" for val in delim_y]
    clf_str = ACDSOMultiLabelClassifier(
        strategy="multi_output", cv=3, text_cols=["notes"], random_state=11
    )
    clf_str.fit(X.iloc[:70], delim_y[:70])
    assert clf_str.n_labels_ == 3
    assert clf_str.predict(X.iloc[70:]).shape == (20, 3)
    assert "notes" in clf_str.text_cols_

    # Sparse matrix targets + auto-detected text column
    Y_sparse = sparse.csr_matrix(np.column_stack([y0, y1]))
    clf_sp = ACDSOMultiLabelClassifier(strategy="random_forest", cv=3, random_state=11)
    clf_sp.fit(X.iloc[:70], Y_sparse[:70])
    assert "notes" in clf_sp.text_cols_
    assert clf_sp.predict_proba(X.iloc[70:]).shape == (20, 2)

    # List of sets + DataFrame targets
    list_sets = [
        {"alpha"} if (y0[i] and not y1[i]) else ({"beta"} if y1[i] else {"alpha", "beta"})
        for i in range(n)
    ]
    clf_sets = ACDSOMultiLabelClassifier(strategy="classifier_chain", cv=3, random_state=11)
    clf_sets.fit(X.iloc[:70], list_sets[:70])
    assert clf_sets.predict(X.iloc[70:]).shape == (20, 2)

    Y_df = pd.DataFrame({"tag_a": y0, "tag_b": y1})
    clf_df = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=11)
    clf_df.fit(X.iloc[:70], Y_df.iloc[:70])
    assert list(clf_df.classes_) == ["tag_a", "tag_b"]


def test_multilabel_constant_columns_and_grouped_cv() -> None:
    rng = np.random.RandomState(5)
    n = 80
    X = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
    Y = np.column_stack(
        [
            (X["a"] > 0).astype(int),
            np.zeros(n, dtype=int),
            np.ones(n, dtype=int),
        ]
    )
    groups = np.repeat(np.arange(16), 5)

    for strat in ("multi_output", "classifier_chain", "random_forest", "extra_trees"):
        clf = ACDSOMultiLabelClassifier(strategy=strat, scorer="hamming_loss", cv=4, random_state=5)
        clf.fit(X.iloc[:60], Y[:60], groups=groups[:60], cv_strategy="group_kfold")
        preds = clf.predict(X.iloc[60:])
        proba = clf.predict_proba(X.iloc[60:])
        assert preds.shape == (20, 3)
        assert proba.shape == (20, 3)
        assert np.all(proba[:, 1] <= 0.1)
        assert np.all(proba[:, 2] >= 0.9)
        ev = clf.evaluate(X.iloc[60:], Y[60:])
        assert all(0.0 <= v <= 1.0 for v in ev.values())
        assert np.isclose(clf.score(X.iloc[60:], Y[60:]), 1.0 - ev["hamming_loss"])


def test_multilabel_validation_clone_pickle_and_errors() -> None:
    X, Y = _make_multilabel_data(n_rows=50, seed=3)
    with pytest.raises(ValueError):
        ACDSOMultiLabelClassifier(strategy="invalid").fit(X, Y)
    with pytest.raises(ValueError):
        ACDSOMultiLabelClassifier(scorer="bad_scorer").fit(X, Y)
    with pytest.raises(ValueError):
        ACDSOMultiLabelClassifier(cv=3).fit(X, Y[:, 0])
    with pytest.raises(ValueError):
        ACDSOMultiLabelClassifier(cv=3).fit(X, Y[:, :1])
    with pytest.raises(ValueError):
        ACDSOMultiLabelClassifier(cv=3).fit(X, Y[:40])
    with pytest.raises(ValueError):
        ACDSOMultiLabelClassifier(cv=3).fit(X, Y, groups=np.arange(30))

    unfitted = ACDSOMultiLabelClassifier()
    with pytest.raises(RuntimeError):
        unfitted.predict(X)
    with pytest.raises(RuntimeError):
        unfitted.predict_proba(X)
    with pytest.raises(RuntimeError):
        unfitted.score(X, Y)
    with pytest.raises(RuntimeError):
        unfitted.evaluate(X, Y)

    clf = ACDSOMultiLabelClassifier(strategy="auto", cv=3, random_state=3).fit(X, Y)
    cloned = clone(clf)
    assert cloned.get_params() == clf.get_params()
    assert not hasattr(cloned, "best_estimator_")

    roundtrip = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(roundtrip.predict(X), clf.predict(X))
    assert roundtrip.n_features_in_ == 3
    assert list(roundtrip.feature_names_in_) == ["x0", "x1", "x2"]


# ===========================================================================
# R5b: ndcg_at_k & ACDSORanker
# ===========================================================================


def test_ndcg_at_k_edge_cases_and_ties() -> None:
    # Single-item queries
    assert np.isclose(ndcg_at_k([3.0], [1.0], groups=[0], k=5), 1.0)
    assert np.isclose(ndcg_at_k([0.0], [1.0], groups=[0], k=5), 0.0)
    assert np.isclose(ndcg_at_k([2.0], [0.5], groups=None, k=1), 1.0)

    # All-zero relevance
    assert ndcg_at_k(np.zeros(6), np.linspace(0, 1, 6), groups=[0, 0, 0, 1, 1, 1], k=5) == 0.0

    # Tie-aware scoring: tied predicted scores must average DCG regardless of index order
    rel_a = np.array([3.0, 0.0, 2.0])
    rel_b = np.array([0.0, 3.0, 2.0])
    tied_scores = np.array([1.0, 1.0, 1.0])
    assert np.isclose(ndcg_at_k(rel_a, tied_scores, k=2), ndcg_at_k(rel_b, tied_scores, k=2))

    # Perfect vs reversed across multiple queries
    y_true = np.array([3.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0])
    qids = np.array(["q0"] * 4 + ["q1"] * 4)
    perf = np.array([10.0, 8.0, 5.0, 1.0, 9.0, 4.0, 2.0, 0.0])
    assert np.isclose(ndcg_at_k(y_true, perf, groups=qids, k=3), 1.0)
    assert ndcg_at_k(y_true, -perf, groups=qids, k=3) < 0.7

    # Error cases
    with pytest.raises(ValueError):
        ndcg_at_k(y_true, perf, groups=qids[:4], k=3)
    with pytest.raises(ValueError):
        ndcg_at_k(y_true, perf, k=0)
    with pytest.raises(ValueError):
        ndcg_at_k([1.0, np.nan], [1.0, 2.0])


@pytest.mark.parametrize("algorithm", ["auto", "pairwise", "pointwise", "pointwise_ridge", "pointwise_gbr"])
def test_ranker_algorithms_and_query_isolation(algorithm: str) -> None:
    rng = np.random.RandomState(17)
    n_queries = 20
    docs_per_q = 8
    qids = np.repeat([f"q_{i:02d}" for i in range(n_queries)], docs_per_q)
    n = len(qids)
    f0 = rng.randn(n)
    f1 = rng.randn(n)
    signal = 2.5 * f0 - 1.5 * f1 + 0.1 * rng.randn(n)
    rel = np.digitize(signal, [-1.2, 0.0, 1.2]).astype(float)
    desc = [
        "verified exact match official top product" if rel[i] >= 2 else "unrelated generic catalog stub"
        for i in range(n)
    ]
    X = pd.DataFrame({"f0": f0, "f1": f1, "desc": desc})

    tr = np.arange(15 * docs_per_q)
    te = np.arange(15 * docs_per_q, n)

    ranker = ACDSORanker(algorithm=algorithm, k=5, cv=3, text_cols=["desc"], random_state=17)
    assert ranker.fit(X.iloc[tr], rel[tr], groups=qids[tr]) is ranker

    preds = ranker.predict(X.iloc[te])
    assert preds.shape == (len(te),)
    assert np.all(np.isfinite(preds))

    sc = ranker.score(X.iloc[te], rel[te], groups=qids[te])
    assert np.isclose(sc, ndcg_at_k(rel[te], preds, groups=qids[te], k=5), atol=1e-9)
    assert sc > 0.75

    ranks = ranker.rank(X.iloc[te], groups=qids[te])
    assert ranks.shape == (len(te),)
    for q in np.unique(qids[te]):
        q_mask = qids[te] == q
        assert set(ranks[q_mask].tolist()) == set(range(docs_per_q))

    ev = ranker.evaluate(X.iloc[te], rel[te], groups=qids[te], k=5)
    assert "ndcg_at_k" in ev and "mrr" in ev
    assert all(np.isfinite(v) for v in ev.values())
    assert ranker.audit_trail_["cv_strategy"] == "group_kfold"
    _assert_strict_json(ranker.audit_trail_)


def test_ranker_validation_clone_pickle_and_errors() -> None:
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.randn(30), "b": rng.randn(30)})
    y = rng.randint(0, 3, size=30)
    qids = np.repeat(np.arange(6), 5)

    with pytest.raises(ValueError):
        ACDSORanker(cv=3).fit(X, y, groups=None)
    with pytest.raises(ValueError):
        ACDSORanker(cv=3).fit(X, y, groups=qids[:10])
    with pytest.raises(ValueError):
        ACDSORanker(algorithm="unknown").fit(X, y, groups=qids)

    unfitted = ACDSORanker()
    with pytest.raises(RuntimeError):
        unfitted.predict(X)
    with pytest.raises(RuntimeError):
        unfitted.score(X, y, groups=qids)
    with pytest.raises(RuntimeError):
        unfitted.evaluate(X, y, groups=qids)

    ranker = ACDSORanker(algorithm="pairwise", cv=3, k=4, random_state=42).fit(X, y, groups=qids)
    cloned = clone(ranker)
    assert cloned.get_params() == ranker.get_params()
    roundtrip = pickle.loads(pickle.dumps(ranker))
    np.testing.assert_allclose(roundtrip.predict(X), ranker.predict(X))


# ===========================================================================
# R5c: concordance_index & ACDSOSurvivalAnalyzer
# ===========================================================================


def test_concordance_index_matches_bruteforce_with_ties_and_censoring() -> None:
    rng = np.random.RandomState(99)
    for trial in range(20):
        n = rng.randint(15, 45)
        # Discrete times + quantized scores guarantee heavy ties in both T and S
        times = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n)
        scores = rng.choice([0.1, 0.2, 0.3, 0.4, 0.5], size=n)
        events = rng.binomial(1, 0.65, size=n)
        c_fast = concordance_index(times, scores, events)
        c_ref = _bruteforce_c_index(times, scores, events)
        assert np.isclose(c_fast, c_ref, atol=1e-12), f"Mismatch on trial {trial}: {c_fast} vs {c_ref}"

    # All censored / single-sample edge cases
    assert concordance_index([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0, 0, 0]) == 0.5
    assert concordance_index([2.0], [1.0], [1]) == 0.5

    # Invalid inputs
    with pytest.raises(ValueError):
        concordance_index([-1.0, 2.0], [1.0, 2.0], [1, 1])
    with pytest.raises(ValueError):
        concordance_index([1.0, np.nan], [1.0, 2.0], [1, 1])
    with pytest.raises(ValueError):
        concordance_index([1.0, 2.0], [1.0, 2.0], [1, 2])
    with pytest.raises(ValueError):
        concordance_index([1.0, 2.0], [1.0], [1, 1])


def test_breslow_analytical_gradient_matches_finite_difference() -> None:
    rng = np.random.RandomState(7)
    n, p = 50, 4
    X_std = rng.randn(n, p)
    # Include tied event times and censoring
    time_arr = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], size=n)
    event_arr = rng.binomial(1, 0.7, size=n).astype(np.int64)
    time_blocks = _build_descending_time_blocks(time_arr, event_arr)
    sum_X_events = np.sum(X_std[event_arr == 1], axis=0)
    l2 = 0.05

    def fun(b: np.ndarray) -> float:
        val, _ = _breslow_nll_and_grad(b, X_std, event_arr, time_blocks, sum_X_events, l2)
        return val

    def grad(b: np.ndarray) -> np.ndarray:
        _, g = _breslow_nll_and_grad(b, X_std, event_arr, time_blocks, sum_X_events, l2)
        return g

    for beta0 in [np.zeros(p), rng.normal(0.0, 0.5, size=p)]:
        err = optimize.check_grad(fun, grad, beta0, epsilon=1e-7)
        assert err < 1e-6, f"Analytical gradient error {err} exceeded 1e-6"


def test_survival_analyzer_fit_forms_curves_collinear_and_cv() -> None:
    rng = np.random.RandomState(42)
    n = 150
    x0 = rng.randn(n)
    x1 = rng.randn(n)
    x2 = x0 + 1e-8 * rng.randn(n)  # near-collinear
    lin = 1.2 * x0 - 0.9 * x1
    true_t = rng.exponential(scale=5.0, size=n) * np.exp(-lin)
    cens_t = rng.uniform(2.0, 12.0, size=n)
    obs_t = np.maximum(0.05, np.minimum(true_t, cens_t))
    evt = (true_t <= cens_t).astype(int)
    notes = ["high severity critical alert" if lin[i] > 0 else "routine stable checkup note" for i in range(n)]
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "notes": notes})
    groups = np.repeat(np.arange(15), 10)

    surv = ACDSOSurvivalAnalyzer(l2_penalty=0.05, backend="scipy", cv=3, random_state=42)
    assert surv.fit(X.iloc[:110], time=obs_t[:110], event=evt[:110], groups=groups[:110]) is surv
    assert "notes" in surv.text_cols_

    risks = surv.predict_risk(X.iloc[110:])
    assert risks.shape == (40,)
    assert np.corrcoef(risks, lin[110:])[0, 1] > 0.7
    c_val = surv.score(X.iloc[110:], time=obs_t[110:], event=evt[110:])
    assert c_val > 0.70

    sf = surv.predict_survival_function(X.iloc[110:116])
    assert sf.shape[0] == 6 and sf.shape[1] >= 2
    assert np.all((sf >= 0.0) & (sf <= 1.0))
    assert np.all(np.diff(sf, axis=1) <= 1e-9)

    ev = surv.evaluate(X.iloc[110:], time=obs_t[110:], event=evt[110:])
    assert np.isclose(ev["c_index"], c_val)
    _assert_strict_json(surv.audit_trail_)
    assert surv.audit_trail_["cv_strategy"] == "group_kfold"

    # Alternative y target forms: structured array & 2-col DataFrame
    X_num = X[["x0", "x1"]].iloc[:80]
    y_struct = np.zeros(80, dtype=[("event", bool), ("time", float)])
    y_struct["event"] = evt[:80].astype(bool)
    y_struct["time"] = obs_t[:80]
    s_struct = ACDSOSurvivalAnalyzer(backend="scipy", cv=3, random_state=42).fit(X_num, y_struct)
    assert s_struct.score(X_num, y_struct) > 0.65

    y_df = pd.DataFrame({"event": evt[:80], "duration": obs_t[:80]})
    s_df = ACDSOSurvivalAnalyzer(backend="scipy", cv=3, random_state=42).fit(X_num, y_df)
    assert np.isclose(s_df.score(X_num, y_df), s_struct.score(X_num, y_struct), atol=1e-6)

    # All-censored stability
    s_cens = ACDSOSurvivalAnalyzer(backend="scipy", cv=3, random_state=42).fit(
        X_num, time=obs_t[:80], event=np.zeros(80, dtype=int)
    )
    sf_cens = s_cens.predict_survival_function(X_num.iloc[:3])
    assert np.allclose(sf_cens, 1.0)

    # Clone & pickle
    assert clone(surv).get_params() == surv.get_params()
    rt = pickle.loads(pickle.dumps(surv))
    np.testing.assert_allclose(rt.predict_risk(X.iloc[110:]), risks)

    # Unfitted & bad input errors
    with pytest.raises(RuntimeError):
        ACDSOSurvivalAnalyzer().predict_risk(X_num)
    with pytest.raises(ValueError):
        ACDSOSurvivalAnalyzer(backend="bad").fit(X_num, time=obs_t[:80], event=evt[:80])


# ===========================================================================
# R5d: ACDSORecommender
# ===========================================================================


def _make_rec_df(seed: int = 42):
    rng = np.random.RandomState(seed)
    users = [f"u_{i:02d}" for i in range(18)]
    items = [f"i_{j:02d}" for j in range(14)]
    rows = []
    for u_idx, u in enumerate(users):
        for i_idx, it in enumerate(items):
            if (u_idx + i_idx) % 2 == 0 or rng.rand() < 0.35:
                rating = 4.5 if (u_idx % 3) == (i_idx % 3) else 1.8
                rating = float(np.clip(rating + rng.normal(0.0, 0.2), 1.0, 5.0))
                rows.append({"user_id": u, "item_id": it, "rating": round(rating, 2)})
    df = pd.DataFrame(rows)
    split = int(len(df) * 0.8)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


@pytest.mark.parametrize("method", ["auto", "svd", "nmf", "item_knn", "user_knn"])
def test_recommender_explicit_and_implicit_all_methods(method: str) -> None:
    train_df, test_df = _make_rec_df(seed=42)
    rec = ACDSORecommender(method=method, n_factors=4, n_neighbors=6, random_state=42)
    assert rec.fit(train_df, user_col="user_id", item_col="item_id", rating_col="rating") is rec

    preds = rec.predict(test_df["user_id"], test_df["item_id"])
    assert preds.shape == (len(test_df),)
    assert np.all(np.isfinite(preds))

    seen = set(train_df.loc[train_df["user_id"] == "u_00", "item_id"])
    recs = rec.recommend("u_00", n=5, exclude_seen=True)
    assert 1 <= len(recs) <= 5
    assert seen.isdisjoint({it for it, _ in recs})
    scores = [sc for _, sc in recs]
    assert scores == sorted(scores, reverse=True)

    ev = rec.evaluate(test_df, k=5)
    for key in ("rmse", "mae", "precision_at_k", "recall_at_k", "ndcg_at_k"):
        assert key in ev and np.isfinite(ev[key])
    assert ev["rmse"] < 1.5
    _assert_strict_json(rec.audit_trail_)

    # Implicit feedback mode (rating_col=None)
    imp_train = train_df[["user_id", "item_id"]].copy()
    imp_test = test_df[["user_id", "item_id"]].copy()
    rec_imp = ACDSORecommender(method=method, n_factors=4, n_neighbors=6, random_state=42)
    rec_imp.fit(imp_train, user_col="user_id", item_col="item_id", rating_col=None)
    imp_recs = rec_imp.recommend("u_00", n=5, exclude_seen=True)
    assert 1 <= len(imp_recs) <= 5
    imp_ev = rec_imp.evaluate(imp_test, k=5)
    assert all(np.isfinite(v) for v in imp_ev.values())


def test_recommender_boundaries_cold_start_duplicates_and_small_matrix() -> None:
    # 3x3 matrix with n_factors=20 > min_dim, plus duplicate (u, i) rows
    small_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
            "item_id": ["i1", "i1", "i2", "i1", "i3", "i2"],
            "rating": [5.0, 3.0, 4.0, 4.0, 2.0, 5.0],
        }
    )
    rec = ACDSORecommender(method="svd", n_factors=20, random_state=42).fit(
        small_df, user_col="user_id", item_col="item_id", rating_col="rating"
    )
    # Cold-start user, item, and both
    cold = rec.predict(["new_u", "u1", "new_u"], ["new_i", "new_i", "i1"])
    assert cold.shape == (3,) and np.all(np.isfinite(cold))
    assert rec.recommend("u1", n=0) == []
    assert len(rec.recommend("u1", n=100, exclude_seen=True)) == 1
    assert len(rec.recommend("u1", n=100, exclude_seen=False)) == 3
    assert len(rec.recommend("brand_new_user", n=2)) == 2

    # Empty holdout and zero-relevant-items holdout
    empty_eval = rec.evaluate(small_df.iloc[:0], k=3)
    assert empty_eval["n_interactions"] == 0 and empty_eval["ndcg_at_k"] == 0.0
    low_eval = rec.evaluate(
        pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["i3", "i2"], "rating": [1.0, 5.0]}),
        k=2,
    )
    assert all(np.isfinite(v) for v in low_eval.values())

    # 2-tuple array input & single-user matrix fallback
    arr_imp = [("a", "x"), ("a", "y"), ("b", "y")]
    rec_arr = ACDSORecommender(method="auto", n_factors=5, random_state=42).fit(arr_imp)
    assert len(rec_arr.recommend("b", n=2)) >= 1

    # Clone & pickle
    assert clone(rec).get_params() == rec.get_params()
    rt = pickle.loads(pickle.dumps(rec))
    np.testing.assert_allclose(rt.predict(["u1"], ["i3"]), rec.predict(["u1"], ["i3"]))

    # Errors
    with pytest.raises(RuntimeError):
        ACDSORecommender().predict(["u1"], ["i1"])
    with pytest.raises(ValueError):
        ACDSORecommender(method="invalid").fit(small_df)
    with pytest.raises(ValueError):
        ACDSORecommender().fit(small_df.iloc[:0])
    with pytest.raises(ValueError):
        ACDSORecommender().fit(small_df, user_col="missing_col")


# ===========================================================================
# Iteration 2 Regression Tests (All 7 Verified Fixes)
# ===========================================================================


def test_multilabel_delimited_single_label_and_reordered_df_evaluate() -> None:
    X = pd.DataFrame({"x0": np.linspace(-1.0, 1.0, 30), "x1": np.linspace(1.0, -1.0, 30)})
    y_tr = ["a|b" if i % 3 == 0 else ("a" if i % 3 == 1 else "b") for i in range(20)]
    clf = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(X.iloc[:20], y_tr)
    assert list(clf.classes_) == ["a", "b"]

    # Batch without any '|' or ',' delimiters (list and pd.Series, incl. empty/null)
    y_te_list = ["a", "b", "", None, "a", "b", "", "a", "b", ""]
    ev_list = clf.evaluate(X.iloc[20:], y_te_list)
    assert all(0.0 <= v <= 1.0 for v in ev_list.values())
    assert 0.0 <= clf.score(X.iloc[20:], y_te_list) <= 1.0

    y_te_series = pd.Series(["a", "b", "", pd.NA, "a", "b", "", "a", "b", np.nan], dtype=object)
    ev_series = clf.evaluate(X.iloc[20:], y_te_series)
    assert np.isclose(ev_series["f1_micro"], ev_list["f1_micro"])

    # 1D numeric/bool and (n, 1) targets must still be rejected at evaluate()
    with pytest.raises(ValueError):
        clf.evaluate(X.iloc[20:], [0, 1] * 5)
    with pytest.raises(ValueError):
        clf.evaluate(X.iloc[20:], [True, False] * 5)
    with pytest.raises(ValueError):
        clf.evaluate(X.iloc[20:], np.zeros((10, 1), dtype=int))

    # Reordered DataFrame columns at evaluate()/score() must align by class name
    ya = (X["x0"] > 0.0).astype(int)
    yb = (X["x1"] > 0.2).astype(int)
    y_df_orig = pd.DataFrame({"tag_a": ya, "tag_b": yb})
    y_df_swap = pd.DataFrame({"tag_b": yb, "tag_a": ya})
    clf_df = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(X, y_df_orig)
    ev_orig = clf_df.evaluate(X, y_df_orig)
    ev_swap = clf_df.evaluate(X, y_df_swap)
    assert np.isclose(ev_orig["f1_micro"], ev_swap["f1_micro"])
    assert np.isclose(clf_df.score(X, y_df_orig), clf_df.score(X, y_df_swap))


def test_ranker_rejects_nongrouped_custom_cv_splitter() -> None:
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.randn(30), "b": rng.randn(30)})
    y = rng.randint(0, 3, size=30)
    qids = np.repeat(np.arange(6), 5)

    with pytest.raises(ValueError):
        ACDSORanker(algorithm="pairwise", cv=KFold(n_splits=3)).fit(X, y, groups=qids)


def test_survival_structured_censoring_time_and_strict_cv_validation() -> None:
    X = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    y_struct = np.array(
        [(1.0, True), (2.0, False), (3.0, True), (4.0, False), (5.0, True), (6.0, False)],
        dtype=[("censoring_time", float), ("event", bool)],
    )
    surv = ACDSOSurvivalAnalyzer(backend="scipy", cv=2, random_state=42).fit(X, y_struct)
    assert np.isfinite(surv.score(X, y_struct))

    y_df_rev = pd.DataFrame({"event": [1, 0, 1, 0, 1, 0], "censoring_time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    surv_rev = ACDSOSurvivalAnalyzer(backend="scipy", cv=2, random_state=42).fit(X, y_df_rev)
    assert np.isclose(surv.score(X, y_struct), surv_rev.score(X, y_df_rev))

    # Invalid CV / strategy / small-n groups must raise ValueError instead of being swallowed
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    e = np.array([1, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        ACDSOSurvivalAnalyzer(cv=1, backend="scipy").fit(X, time=t, event=e)
    with pytest.raises(ValueError):
        ACDSOSurvivalAnalyzer(backend="scipy").fit(X, time=t, event=e, cv_strategy="group_kfold")
    with pytest.raises(ValueError):
        ACDSOSurvivalAnalyzer(backend="scipy").fit(X, time=t, event=e, cv_strategy="stratified_group_kfold")
    with pytest.raises(ValueError):
        ACDSOSurvivalAnalyzer(backend="scipy").fit(X, time=t, event=e, cv_strategy="bogus")
    with pytest.raises(ValueError):
        ACDSOSurvivalAnalyzer(cv=2, backend="scipy").fit(
            X.iloc[:3], time=t[:3], event=e[:3], groups=[0, 1]
        )


def test_recommender_categorical_and_large_implicit_item_knn() -> None:
    # Categorical user/item IDs (including unobserved categories) must not produce Cartesian NaNs
    df_cat = pd.DataFrame(
        {
            "u": pd.Categorical(["u1", "u1", "u2", "u3", "u2", "u3"], categories=["u1", "u2", "u3", "u_unused"]),
            "i": pd.Categorical(["i1", "i2", "i2", "i3", "i1", "i2"], categories=["i1", "i2", "i3", "i_unused"]),
            "r": [5.0, 4.0, 2.0, 5.0, 3.0, 4.5],
        }
    )
    for m in ("svd", "nmf", "item_knn", "user_knn", "auto"):
        rec = ACDSORecommender(method=m, n_factors=2, n_neighbors=2, random_state=42).fit(
            df_cat, user_col="u", item_col="i", rating_col="r"
        )
        preds = rec.predict(df_cat["u"], df_cat["i"])
        assert np.all(np.isfinite(preds))
        ev = rec.evaluate(df_cat, k=2)
        assert np.isfinite(ev["rmse"])

    # n_items > 4000 implicit item_knn must retain differentiated scores < 1.0
    rng = np.random.RandomState(11)
    rows = [("u0", f"i_{j}") for j in range(10)]
    for u in range(1, 20):
        for j in range(10):
            if rng.rand() < 0.6:
                rows.append((f"u{u}", f"i_{j}"))
    for j in range(10, 25):
        for u in range(1, 20):
            if rng.rand() < (0.85 - 0.04 * (j - 10)):
                rows.append((f"u{u}", f"i_{j}"))
        for u_bg in range(20, 20 + (j - 9) * 3):
            rows.append((f"u{u_bg}", f"i_{j}"))
    for j in range(25, 4005):
        u_a = int(rng.randint(20, 250))
        u_b = int(rng.randint(20, 250))
        rows.append((f"u{u_a}", f"i_{j}"))
        if u_b != u_a:
            rows.append((f"u{u_b}", f"i_{j}"))

    rec_large = ACDSORecommender(method="item_knn", n_neighbors=5, random_state=11).fit(rows)
    assert rec_large.n_items_ == 4005
    recs = rec_large.recommend("u0", n=10)
    assert len(recs) == 10
    scores = [sc for _, sc in recs]
    assert len({round(sc, 6) for sc in scores}) > 1
    assert all(0.0 <= sc < 1.0 for sc in scores)

    # Verify pos_sim_row_sum matches exact dense item-similarity row sums
    from sklearn.metrics.pairwise import cosine_similarity

    E_large = rec_large._state_["E"]
    sim_ref = cosine_similarity(E_large.T)
    np.fill_diagonal(sim_ref, 0.0)
    expected_denom = np.sum(np.maximum(sim_ref, 0.0), axis=1)
    np.testing.assert_allclose(rec_large._state_["pos_sim_row_sum"], expected_denom, atol=1e-10)


# ===========================================================================
# Iteration 3 Regression Tests (Challenger 1 R5a Fixes)
# ===========================================================================


def test_multilabel_df_duplicate_mismatch_and_pipe_comma_classes() -> None:
    import warnings

    X = pd.DataFrame({"x0": np.linspace(-1.0, 1.0, 30), "x1": np.sin(np.linspace(0.0, 6.0, 30))})
    ya = (X["x0"] > 0.0).astype(int)
    yb = (X["x1"] > 0.0).astype(int)

    # Duplicate DataFrame y columns must be rejected at fit()
    y_df_dup = pd.DataFrame(np.column_stack([ya, yb]), columns=["tag_a", "tag_a"])
    with pytest.raises(ValueError, match="duplicate"):
        ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(X, y_df_dup)

    # Named DataFrame fit -> duplicate, partially mismatched, fully mismatched, and extra-col y must fail
    y_df = pd.DataFrame({"tag_a": ya, "tag_b": yb})
    clf_df = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(X, y_df)
    for bad_y in (
        pd.DataFrame(np.column_stack([yb, ya]), columns=["tag_a", "tag_a"]),
        pd.DataFrame({"tag_a": ya, "wrong_b": yb}),
        pd.DataFrame({"wrong_b": yb, "wrong_a": ya}),
        pd.DataFrame({"tag_a": ya, "tag_b": yb, "extra": ya}),
    ):
        with pytest.raises(ValueError):
            clf_df.evaluate(X, bad_y)
        with pytest.raises(ValueError):
            clf_df.score(X, bad_y)

    # Delimited-fitted named classes must also reject mismatched DataFrame y columns
    y_delim_named = [
        "alpha|beta" if (a and b) else ("alpha" if a else ("beta" if b else ""))
        for a, b in zip(ya, yb)
    ]
    clf_delim = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(X, y_delim_named)
    assert np.isclose(
        clf_delim.score(X, pd.DataFrame({"beta": yb, "alpha": ya})),
        clf_delim.score(X, y_delim_named),
    )
    with pytest.raises(ValueError):
        clf_delim.evaluate(X, pd.DataFrame({"wrong_a": ya, "wrong_b": yb}))

    # Unnamed 2D ndarray fit (classes_ == [0, 1]) still allows DataFrame evaluate()/score()
    Y_arr = np.column_stack([ya, yb])
    clf_arr = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(X, Y_arr)
    assert list(clf_arr.classes_) == [0, 1]
    ev_arr = clf_arr.evaluate(X, Y_arr)
    ev_named_df = clf_arr.evaluate(X, pd.DataFrame({"c0": ya, "c1": yb}))
    ev_range_df = clf_arr.evaluate(X, pd.DataFrame(Y_arr))
    assert np.isclose(ev_named_df["f1_micro"], ev_arr["f1_micro"])
    assert np.isclose(ev_range_df["f1_micro"], ev_arr["f1_micro"])
    assert np.isclose(clf_arr.score(X, pd.DataFrame({"c0": ya, "c1": yb})), clf_arr.score(X, Y_arr))

    # Pipe-delimited class containing ',' evaluated on a batch without '|' must not split on ','
    y_pipe_comma_tr = [
        "x|a,b" if i % 4 == 0 else ("y" if i % 4 == 1 else ("x" if i % 4 == 2 else "a,b|y"))
        for i in range(24)
    ]
    clf_pc = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(
        X.iloc[:24], y_pipe_comma_tr
    )
    assert list(clf_pc.classes_) == ["a,b", "x", "y"]

    y_pc_te = ["a,b", "y", "", "a,b", "x", "y"]
    y_pc_te_df = pd.DataFrame(
        {
            "a,b": [1, 0, 0, 1, 0, 0],
            "x": [0, 0, 0, 0, 1, 0],
            "y": [0, 1, 0, 0, 0, 1],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev_pc_list = clf_pc.evaluate(X.iloc[24:], y_pc_te)
        ev_pc_series = clf_pc.evaluate(X.iloc[24:], pd.Series(y_pc_te, dtype=object))
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    ev_pc_df = clf_pc.evaluate(X.iloc[24:], y_pc_te_df)
    assert np.isclose(ev_pc_list["f1_micro"], ev_pc_df["f1_micro"])
    assert np.isclose(ev_pc_series["f1_micro"], ev_pc_df["f1_micro"])
    assert np.isclose(clf_pc.score(X.iloc[24:], y_pc_te), clf_pc.score(X.iloc[24:], y_pc_te_df))


# ===========================================================================
# Iteration 4 Regression Tests (Challenger 1 MultiIndex/Tuple & Bool/Int DF Fixes)
# ===========================================================================


def test_multilabel_multiindex_and_tuple_columns_end_to_end() -> None:
    X = pd.DataFrame({"x0": np.linspace(-1.0, 1.0, 24), "x1": np.sin(np.linspace(0.0, 6.0, 24))})
    ya = (X["x0"] > 0.0).astype(int)
    yb = (X["x1"] > 0.0).astype(int)

    # 1. MultiIndex columns on y
    y_mi = pd.DataFrame(
        np.column_stack([ya, yb]),
        columns=pd.MultiIndex.from_tuples([("grp", "a"), ("grp", "b")]),
    )
    y_mi_swap = pd.DataFrame(
        np.column_stack([yb, ya]),
        columns=pd.MultiIndex.from_tuples([("grp", "b"), ("grp", "a")]),
    )
    clf_mi = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(X, y_mi)
    assert clf_mi.classes_.shape == (2,)
    assert list(clf_mi.classes_) == [("grp", "a"), ("grp", "b")]
    assert clf_mi.predict(X).shape == (24, 2)
    assert clf_mi.predict_proba(X).shape == (24, 2)
    _assert_strict_json(clf_mi.audit_trail_)
    assert clf_mi.audit_trail_["classes"] == [["grp", "a"], ["grp", "b"]]

    ev_mi = clf_mi.evaluate(X, y_mi)
    ev_mi_swap = clf_mi.evaluate(X, y_mi_swap)
    assert np.isclose(ev_mi["f1_micro"], ev_mi_swap["f1_micro"])
    assert np.isclose(clf_mi.score(X, y_mi), clf_mi.score(X, y_mi_swap))

    cloned_mi = clone(clf_mi)
    assert cloned_mi.get_params() == clf_mi.get_params()
    assert not hasattr(cloned_mi, "_y_unnamed_")
    rt_mi = pickle.loads(pickle.dumps(clf_mi))
    np.testing.assert_array_equal(rt_mi.predict(X), clf_mi.predict(X))
    assert np.isclose(rt_mi.score(X, y_mi_swap), clf_mi.score(X, y_mi))

    for bad_mi in (
        pd.DataFrame(
            np.column_stack([ya, yb]),
            columns=pd.MultiIndex.from_tuples([("grp", "a"), ("grp", "wrong")]),
        ),
        pd.DataFrame(
            np.column_stack([ya, yb, ya]),
            columns=pd.MultiIndex.from_tuples([("grp", "a"), ("grp", "b"), ("grp", "c")]),
        ),
        pd.DataFrame({"a": ya, "b": yb}),
    ):
        with pytest.raises(ValueError):
            clf_mi.evaluate(X, bad_mi)
        with pytest.raises(ValueError):
            clf_mi.score(X, bad_mi)

    # 2. Plain Index of equal-length and mixed-length tuple column names
    for cols_pair in (
        ([("g", 0), ("g", 1)], [("g", 1), ("g", 0)], [("g", 0), ("g", 2)]),
        ([("a",), ("b", "c")], [("b", "c"), ("a",)], [("a",), ("b", "d")]),
    ):
        cols_tr, cols_sw, cols_bad = cols_pair
        y_tup = pd.DataFrame(
            np.column_stack([ya, yb]),
            columns=pd.Index(cols_tr, dtype=object, tupleize_cols=False),
        )
        y_tup_sw = pd.DataFrame(
            np.column_stack([yb, ya]),
            columns=pd.Index(cols_sw, dtype=object, tupleize_cols=False),
        )
        y_tup_bad = pd.DataFrame(
            np.column_stack([ya, yb]),
            columns=pd.Index(cols_bad, dtype=object, tupleize_cols=False),
        )

        clf_tup = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(X, y_tup)
        assert clf_tup.classes_.shape == (2,)
        assert list(clf_tup.classes_) == cols_tr
        assert clf_tup.predict(X).shape == (24, 2)
        assert clf_tup.predict_proba(X).shape == (24, 2)
        _assert_strict_json(clf_tup.audit_trail_)
        assert np.isclose(clf_tup.score(X, y_tup), clf_tup.score(X, y_tup_sw))
        with pytest.raises(ValueError):
            clf_tup.evaluate(X, y_tup_bad)
        with pytest.raises(ValueError):
            clf_tup.score(X, y_tup_bad)


def test_multilabel_bool_int_df_indexing_and_unnamed_positional_gating() -> None:
    X = pd.DataFrame({"x0": np.linspace(-1.0, 1.0, 24), "x1": np.sin(np.linspace(0.0, 6.0, 24))})
    ya = (X["x0"] > 0.0).astype(int)
    yb = (X["x1"] > 0.0).astype(int)

    # 1. Boolean DataFrame column names [False, True] must not trigger 2-row boolean row-masking
    y_bool_tr = pd.DataFrame({False: ya, True: yb})
    clf_bool = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(X, y_bool_tr)
    y_bool_2 = pd.DataFrame({False: [1, 0], True: [0, 1]})
    y_bool_2_swap = pd.DataFrame({True: [0, 1], False: [1, 0]})
    ev_b2 = clf_bool.evaluate(X.iloc[:2], y_bool_2)
    ev_b2_swap = clf_bool.evaluate(X.iloc[:2], y_bool_2_swap)
    assert np.isclose(ev_b2["f1_micro"], ev_b2_swap["f1_micro"])
    assert np.isclose(clf_bool.score(X.iloc[:2], y_bool_2), clf_bool.score(X.iloc[:2], y_bool_2_swap))

    # Explicit boolean/integer DataFrame fits must fail closed on mismatched named DataFrames
    y_wrong_named = pd.DataFrame({"wrong_b": yb, "wrong_a": ya})
    with pytest.raises(ValueError):
        clf_bool.evaluate(X, y_wrong_named)
    with pytest.raises(ValueError):
        clf_bool.score(X, y_wrong_named)

    clf_int_df = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(
        X, pd.DataFrame({0: ya, 1: yb})
    )
    assert np.isclose(
        clf_int_df.score(X, pd.DataFrame({1: yb, 0: ya})),
        clf_int_df.score(X, pd.DataFrame({0: ya, 1: yb})),
    )
    with pytest.raises(ValueError):
        clf_int_df.evaluate(X, y_wrong_named)
    with pytest.raises(ValueError):
        clf_int_df.score(X, y_wrong_named)

    # Unnamed 2D ndarray / sparse / rectangular list fits retain backwards-compatible DataFrame eval
    Y_arr = np.column_stack([ya, yb])
    for unnamed_y in (Y_arr, sparse.csr_matrix(Y_arr), Y_arr.tolist()):
        clf_un = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42).fit(X, unnamed_y)
        assert clf_un._y_unnamed_ is True
        assert np.isclose(
            clf_un.score(X, pd.DataFrame({"c0": ya, "c1": yb})),
            clf_un.score(X, Y_arr),
        )
        assert not hasattr(clone(clf_un), "_y_unnamed_")
