"""Tier 2 E2E Tests: R1-R7 Boundary Conditions, Corner Cases & Negative Testing (50 tests: 5 per feature area 1-10)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .conftest import stop_job_via_client


# ===========================================================================
# R1: ACDSOClusterer Boundaries & Negative Cases - 5 tests
# ===========================================================================


@pytest.mark.tier2
@pytest.mark.f1
def test_r1_t2_unfitted_predict_transform_score_raises() -> None:
    """Calling predict, transform, score, or get_cluster_summary before fit() raises RuntimeError/ValueError."""
    from acdso import ACDSOClusterer

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    clusterer = ACDSOClusterer()
    with pytest.raises((RuntimeError, ValueError)):
        clusterer.predict(X)
    with pytest.raises((RuntimeError, ValueError)):
        clusterer.transform(X)
    with pytest.raises((RuntimeError, ValueError)):
        clusterer.score(X)
    with pytest.raises((RuntimeError, ValueError)):
        clusterer.get_cluster_summary()


@pytest.mark.tier2
@pytest.mark.f1
def test_r1_t2_empty_and_single_row_dataframe_raises_value_error() -> None:
    """Empty (0-row) and 1-row DataFrames raise ValueError in ACDSOClusterer.fit()."""
    from acdso import ACDSOClusterer

    empty_df = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=float)})
    with pytest.raises(ValueError):
        ACDSOClusterer(algorithm="kmeans", n_clusters=2).fit(empty_df)

    one_row_df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(ValueError):
        ACDSOClusterer(algorithm="kmeans", n_clusters=2).fit(one_row_df)


@pytest.mark.tier2
@pytest.mark.f1
def test_r1_t2_invalid_algorithm_metric_dim_reduction_and_k_range() -> None:
    """Invalid algorithm, metric, dim_reduction, or inverted k_range=(5, 2) raises ValueError."""
    from acdso import ACDSOClusterer

    X = pd.DataFrame({"a": np.linspace(0, 10, 30), "b": np.linspace(10, 0, 30)})

    with pytest.raises(ValueError):
        ACDSOClusterer(algorithm="bogus_algo").fit(X)
    with pytest.raises(ValueError):
        ACDSOClusterer(metric="bogus_metric").fit(X)
    with pytest.raises(ValueError):
        ACDSOClusterer(dim_reduction="bogus_dim").fit(X)
    with pytest.raises(ValueError):
        ACDSOClusterer(algorithm="auto", k_range=(5, 2)).fit(X)
    with pytest.raises(ValueError):
        ACDSOClusterer(algorithm="auto", k_range=(1, 3)).fit(X)


@pytest.mark.tier2
@pytest.mark.f1
def test_r1_t2_constant_identical_features_graceful() -> None:
    """All-identical constant feature rows are handled gracefully without NaN corruption or unhandled crash."""
    from acdso import ACDSOClusterer

    X_const = pd.DataFrame({"a": np.ones(40), "b": np.full(40, 3.5), "c": np.zeros(40)})
    clusterer = ACDSOClusterer(algorithm="kmeans", n_clusters=2, random_state=42)
    try:
        clusterer.fit(X_const)
    except ValueError:
        return
    assert len(clusterer.labels_) == 40
    assert np.all(np.isfinite(clusterer.labels_))
    preds = clusterer.predict(X_const.iloc[:10])
    assert len(preds) == 10
    assert np.all(np.isfinite(preds))


@pytest.mark.tier2
@pytest.mark.f1
def test_r1_t2_hdbscan_all_noise_edge_input() -> None:
    """When HDBSCAN marks all training points as noise (-1), labels_ and predict() remain valid without NaN crashes."""
    from acdso import ACDSOClusterer

    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.uniform(-100, 100, size=(30, 4)), columns=["f0", "f1", "f2", "f3"])
    clusterer = ACDSOClusterer(algorithm="hdbscan", min_cluster_size=25, random_state=42)
    clusterer.fit(X)
    assert len(clusterer.labels_) == 30
    assert np.all(np.isfinite(clusterer.labels_))
    preds = clusterer.predict(X.iloc[:5])
    assert len(preds) == 5
    assert np.all(np.isfinite(preds))


# ===========================================================================
# R2: ACDSOAnomalyDetector Boundaries & Negative Cases - 5 tests
# ===========================================================================


@pytest.mark.tier2
@pytest.mark.f1
def test_r2_t2_unfitted_methods_raise() -> None:
    """Unfitted ACDSOAnomalyDetector methods raise RuntimeError/ValueError."""
    from acdso import ACDSOAnomalyDetector

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    det = ACDSOAnomalyDetector()
    with pytest.raises((RuntimeError, ValueError)):
        det.predict(X)
    with pytest.raises((RuntimeError, ValueError)):
        det.predict_is_anomaly(X)
    with pytest.raises((RuntimeError, ValueError)):
        det.decision_function(X)
    with pytest.raises((RuntimeError, ValueError)):
        det.score_samples(X)
    with pytest.raises((RuntimeError, ValueError)):
        det.anomaly_scores(X)


@pytest.mark.tier2
@pytest.mark.f1
def test_r2_t2_invalid_algorithm_and_contamination_bounds() -> None:
    """Invalid algorithm or out-of-range contamination (<=0 or >0.5) raises ValueError."""
    from acdso import ACDSOAnomalyDetector

    X = pd.DataFrame({"a": np.linspace(0, 1, 30), "b": np.linspace(1, 2, 30)})
    with pytest.raises(ValueError):
        ACDSOAnomalyDetector(algorithm="unknown_detector").fit(X)
    with pytest.raises(ValueError):
        ACDSOAnomalyDetector(algorithm="isolation_forest", contamination=0.0).fit(X)
    with pytest.raises(ValueError):
        ACDSOAnomalyDetector(algorithm="isolation_forest", contamination=-0.1).fit(X)
    with pytest.raises(ValueError):
        ACDSOAnomalyDetector(algorithm="isolation_forest", contamination=0.6).fit(X)


@pytest.mark.tier2
@pytest.mark.f1
def test_r2_t2_empty_fit_and_mismatched_columns_at_predict() -> None:
    """Empty fit DataFrame and mismatched column counts at predict() raise ValueError/KeyError."""
    from acdso import ACDSOAnomalyDetector

    empty_X = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=float)})
    with pytest.raises(ValueError):
        ACDSOAnomalyDetector(algorithm="isolation_forest").fit(empty_X)

    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.randn(50, 3), columns=["a", "b", "c"])
    det = ACDSOAnomalyDetector(algorithm="isolation_forest", random_state=42).fit(X_train)

    X_bad = pd.DataFrame(rng.randn(10, 2), columns=["a", "b"])
    with pytest.raises((ValueError, KeyError)):
        det.predict(X_bad)


@pytest.mark.tier2
@pytest.mark.f1
def test_r2_t2_extreme_unseen_outliers_flagged() -> None:
    """Extreme 1e6-magnitude unseen outliers are flagged -1 with strictly higher anomaly_scores than inliers."""
    from acdso import ACDSOAnomalyDetector

    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.randn(120, 3), columns=["x", "y", "z"])
    X_in = pd.DataFrame(rng.randn(15, 3) * 0.5, columns=["x", "y", "z"])
    X_extreme = pd.DataFrame(np.full((5, 3), 1e6), columns=["x", "y", "z"])

    det = ACDSOAnomalyDetector(algorithm="isolation_forest", contamination=0.05, random_state=42).fit(X_train)
    assert np.all(det.predict(X_extreme) == -1)
    assert np.all(det.predict_is_anomaly(X_extreme))
    assert float(det.anomaly_scores(X_extreme).min()) > float(det.anomaly_scores(X_in).max())


@pytest.mark.tier2
@pytest.mark.f1
def test_r2_t2_nan_values_in_numeric_and_text_columns() -> None:
    """NaNs in numeric and text columns are handled or rejected cleanly without silent NaN output scores."""
    from acdso import ACDSOAnomalyDetector

    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "num1": rng.randn(60),
            "num2": rng.randn(60),
            "note": ["normal transaction ok" if i % 2 == 0 else "routine account check" for i in range(60)],
        }
    )
    df.loc[[0, 5, 10], "num1"] = np.nan
    df.loc[[1, 6], "note"] = np.nan

    det = ACDSOAnomalyDetector(algorithm="isolation_forest", random_state=42)
    try:
        det.fit(df)
    except ValueError:
        return
    scores = det.anomaly_scores(df.iloc[:12])
    preds = det.predict(df.iloc[:12])
    assert len(scores) == 12
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({-1, 1})


# ===========================================================================
# R3: Entity-Grouped CV Boundaries & Negative Cases - 5 tests
# ===========================================================================


@pytest.mark.tier2
@pytest.mark.f1
def test_r3_t2_group_cv_without_groups_raises_value_error() -> None:
    """Requesting group_kfold or stratified_group_kfold with groups=None raises ValueError."""
    from acdso.cv import build_cv_splitter

    y = np.array([0, 1] * 15)
    with pytest.raises(ValueError):
        build_cv_splitter(cv=3, y=y, groups=None, cv_strategy="group_kfold")
    with pytest.raises(ValueError):
        build_cv_splitter(cv=3, y=y, groups=None, cv_strategy="stratified_group_kfold")


@pytest.mark.tier2
@pytest.mark.f1
def test_r3_t2_fewer_than_two_unique_groups_raises_value_error() -> None:
    """Passing fewer than 2 unique groups raises ValueError."""
    from acdso.cv import build_cv_splitter

    y = np.array([0, 1] * 10)
    single_group = np.array(["only_one"] * 20)
    with pytest.raises(ValueError):
        build_cv_splitter(cv=3, y=y, groups=single_group, cv_strategy="group_kfold")
    with pytest.raises(ValueError):
        build_cv_splitter(cv=3, y=y, groups=single_group, cv_strategy="stratified_group_kfold")


@pytest.mark.tier2
@pytest.mark.f1
def test_r3_t2_unique_groups_less_than_n_splits_raises_or_adapts() -> None:
    """When n_unique_groups < n_splits under group_kfold, splitter either raises ValueError or adapts n_splits cleanly."""
    from acdso.cv import build_cv_splitter, validate_no_group_leakage

    X = pd.DataFrame({"a": np.arange(20, dtype=float)})
    y = np.array([0, 1] * 10)
    groups = np.array(["g0"] * 10 + ["g1"] * 10)
    try:
        splitter = build_cv_splitter(cv=5, y=y, groups=groups, cv_strategy="group_kfold")
        splits = list(splitter.split(X, y, groups))
    except ValueError:
        return
    assert len(splits) <= 2
    validate_no_group_leakage(splits, groups)


@pytest.mark.tier2
@pytest.mark.f1
def test_r3_t2_validate_no_group_leakage_detects_overlap() -> None:
    """validate_no_group_leakage raises ValueError whenever any group spans train and validation folds."""
    from acdso.cv import validate_no_group_leakage

    groups = np.array(["g0", "g1", "g2", "g0", "g3", "g4"])
    leaky_splits = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]
    with pytest.raises(ValueError):
        validate_no_group_leakage(leaky_splits, groups)


@pytest.mark.tier2
@pytest.mark.f1
def test_r3_t2_classifier_mismatched_groups_length_raises_value_error() -> None:
    """ACDSOClassifier.fit() raises ValueError when len(groups) != len(X)."""
    from acdso import ACDSOClassifier

    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.randn(50), "b": rng.randn(50)})
    y = rng.randint(0, 2, size=50)
    bad_groups = np.array([f"g{i % 5}" for i in range(40)])

    clf = ACDSOClassifier(cv=3, max_candidates=2, verbose=0)
    with pytest.raises(ValueError):
        clf.fit(X, y, groups=bad_groups)


# ===========================================================================
# R4: Classical NLP Featurization Boundaries & Negative Cases - 5 tests
# ===========================================================================


@pytest.mark.tier2
@pytest.mark.f1
def test_r4_t2_empty_nan_whitespace_text_column_safe() -> None:
    """TextFeaturizer safely handles NaN, None, empty strings, and whitespace-only entries without crashing."""
    from acdso.nlp import TextFeaturizer

    s = pd.Series(
        [
            "reliable fast device",
            np.nan,
            "",
            "   ",
            None,
            "broken loud fan issue",
            np.nan,
            "great keyboard build",
        ]
    )
    feat = TextFeaturizer(method="tfidf", max_features=20)
    mat = feat.fit_transform(s)
    arr = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
    assert arr.shape[0] == len(s)
    assert np.all(np.isfinite(arr))


@pytest.mark.tier2
@pytest.mark.f1
def test_r4_t2_numeric_and_short_id_not_misclassified_as_text() -> None:
    """Numeric columns, short IDs, and short categorical codes are NOT flagged by detect_text_columns."""
    from acdso.nlp import detect_text_columns

    df = pd.DataFrame(
        {
            "id_num": np.arange(1, 51),
            "id_str": [f"ID_{i:04d}" for i in range(50)],
            "code": ["A1" if i % 2 == 0 else "B2" for i in range(50)],
            "amount": np.linspace(10.0, 200.0, 50),
        }
    )
    assert detect_text_columns(df) == []


@pytest.mark.tier2
@pytest.mark.f1
def test_r4_t2_unseen_oov_words_at_transform_same_width() -> None:
    """Completely out-of-vocabulary unseen words at transform() return finite matrix of identical width."""
    from acdso.nlp import TextFeaturizer

    train_txt = pd.Series(
        [
            "alpha beta gamma delta epsilon",
            "beta gamma zeta eta theta",
            "alpha delta iota kappa lambda",
            "epsilon zeta theta mu nu",
        ]
        * 3
    )
    test_oov = pd.Series(["xylophone quasar nebula", "fjord zephyr sphinx"])

    feat = TextFeaturizer(method="tfidf", max_features=25)
    X_tr = feat.fit_transform(train_txt)
    X_te = feat.transform(test_oov)
    arr_tr = X_tr.toarray() if hasattr(X_tr, "toarray") else np.asarray(X_tr)
    arr_te = X_te.toarray() if hasattr(X_te, "toarray") else np.asarray(X_te)
    assert arr_te.shape == (2, arr_tr.shape[1])
    assert np.all(np.isfinite(arr_te))


@pytest.mark.tier2
@pytest.mark.f1
def test_r4_t2_n_topics_exceeds_vocab_or_docs_handled_cleanly() -> None:
    """TextFeaturizer(method='lda'/'nmf') handles n_topics > n_docs or vocab size cleanly without crash."""
    from acdso.nlp import TextFeaturizer

    tiny = pd.Series(
        [
            "red apple sweet",
            "blue ocean wave",
            "green forest tree",
            "red berry tart",
            "blue sky cloud",
            "green leaf moss",
        ]
    )
    for method in ("lda", "nmf"):
        feat = TextFeaturizer(method=method, n_topics=25, max_features=12, random_state=42)
        mat = feat.fit_transform(tiny)
        arr = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
        assert arr.shape[0] == 6
        assert arr.shape[1] >= 1
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0.0)


@pytest.mark.tier2
@pytest.mark.f1
def test_r4_t2_invalid_method_raises_value_error() -> None:
    """TextFeaturizer with unsupported method raises ValueError."""
    from acdso.nlp import TextFeaturizer

    with pytest.raises(ValueError):
        TextFeaturizer(method="unsupported_nlp_xyz").fit(pd.Series(["hello world sample text"]))


# ===========================================================================
# R5a: Multi-Label Classification Boundaries & Negative Cases - 5 tests
# ===========================================================================


@pytest.mark.tier2
@pytest.mark.f1
def test_r5a_t2_1d_or_single_column_y_rejected() -> None:
    """ACDSOMultiLabelClassifier rejects 1D y or single-column (n, 1) y with ValueError."""
    from acdso import ACDSOMultiLabelClassifier

    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.randn(40), "b": rng.randn(40)})
    y_1d = rng.randint(0, 2, size=40)
    y_single_col = y_1d.reshape(-1, 1)

    with pytest.raises(ValueError):
        ACDSOMultiLabelClassifier(cv=3).fit(X, y_1d)
    with pytest.raises(ValueError):
        ACDSOMultiLabelClassifier(cv=3).fit(X, y_single_col)


@pytest.mark.tier2
@pytest.mark.f1
def test_r5a_t2_unfitted_predict_and_proba_raises() -> None:
    """Unfitted ACDSOMultiLabelClassifier raises RuntimeError/ValueError on predict/predict_proba/score/evaluate."""
    from acdso import ACDSOMultiLabelClassifier

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    Y = np.array([[0, 1], [1, 0], [1, 1]])
    clf = ACDSOMultiLabelClassifier()
    with pytest.raises((RuntimeError, ValueError)):
        clf.predict(X)
    with pytest.raises((RuntimeError, ValueError)):
        clf.predict_proba(X)
    with pytest.raises((RuntimeError, ValueError)):
        clf.score(X, Y)
    with pytest.raises((RuntimeError, ValueError)):
        clf.evaluate(X, Y)


@pytest.mark.tier2
@pytest.mark.f1
def test_r5a_t2_mismatched_xy_length_raises_value_error() -> None:
    """ACDSOMultiLabelClassifier.fit() raises ValueError when len(X) != len(y)."""
    from acdso import ACDSOMultiLabelClassifier

    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.randn(50), "b": rng.randn(50)})
    Y = rng.randint(0, 2, size=(40, 3))
    with pytest.raises(ValueError):
        ACDSOMultiLabelClassifier(cv=3).fit(X, Y)


@pytest.mark.tier2
@pytest.mark.f1
def test_r5a_t2_constant_all_zero_label_column_handled() -> None:
    """An all-zero label column in multi-label Y is handled cleanly by fit() and evaluate() without NaN/ZeroDivisionError."""
    from acdso import ACDSOMultiLabelClassifier

    rng = np.random.RandomState(42)
    X = pd.DataFrame({"x0": rng.randn(60), "x1": rng.randn(60)})
    Y = np.column_stack(
        [
            (X["x0"] > 0).astype(int),
            (X["x1"] > 0).astype(int),
            np.zeros(60, dtype=int),
        ]
    )
    clf = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42)
    clf.fit(X.iloc[:45], Y[:45])
    ev = clf.evaluate(X.iloc[45:], Y[45:])
    assert isinstance(ev, dict)
    for val in ev.values():
        if isinstance(val, (int, float)):
            assert np.isfinite(float(val))


@pytest.mark.tier2
@pytest.mark.f1
def test_r5a_t2_groups_mismatch_raises_value_error() -> None:
    """ACDSOMultiLabelClassifier.fit() raises ValueError when len(groups) != len(X)."""
    from acdso import ACDSOMultiLabelClassifier

    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.randn(50), "b": rng.randn(50)})
    Y = rng.randint(0, 2, size=(50, 2))
    groups = np.array([f"g{i % 5}" for i in range(30)])
    with pytest.raises(ValueError):
        ACDSOMultiLabelClassifier(cv=3).fit(X, Y, groups=groups)


# ===========================================================================
# R5b: Learning-to-Rank Boundaries & Negative Cases - 5 tests
# ===========================================================================


@pytest.mark.tier2
@pytest.mark.f1
def test_r5b_t2_missing_none_groups_raises_value_error() -> None:
    """ACDSORanker.fit() with groups=None raises ValueError."""
    from acdso import ACDSORanker

    rng = np.random.RandomState(42)
    X = pd.DataFrame({"x0": rng.randn(30), "x1": rng.randn(30)})
    y = rng.randint(0, 3, size=30)
    with pytest.raises(ValueError):
        ACDSORanker(cv=3).fit(X, y, groups=None)


@pytest.mark.tier2
@pytest.mark.f1
def test_r5b_t2_single_query_and_single_item_ndcg_no_zerodiv() -> None:
    """ndcg_at_k handles a single-query group and single-item query cleanly without ZeroDivisionError."""
    from acdso.ranking import ndcg_at_k

    assert np.isclose(ndcg_at_k(np.array([3.0]), np.array([1.0]), groups=np.array([0]), k=5), 1.0)
    assert np.isclose(ndcg_at_k(np.array([0.0]), np.array([1.0]), groups=np.array([0]), k=5), 0.0)
    single_q = ndcg_at_k(np.array([2.0, 1.0, 0.0]), np.array([3.0, 2.0, 1.0]), groups=np.array([7, 7, 7]), k=3)
    assert np.isclose(single_q, 1.0)


@pytest.mark.tier2
@pytest.mark.f1
def test_r5b_t2_all_zero_relevance_ties_finite_ndcg() -> None:
    """ndcg_at_k with all-zero true relevances returns finite 0.0 without NaN or ZeroDivisionError."""
    from acdso.ranking import ndcg_at_k

    y_true = np.zeros(10, dtype=float)
    y_score = np.linspace(0.0, 1.0, 10)
    groups = np.array([0] * 5 + [1] * 5)
    val = ndcg_at_k(y_true, y_score, groups=groups, k=5)
    assert np.isfinite(val)
    assert val == 0.0


@pytest.mark.tier2
@pytest.mark.f1
def test_r5b_t2_mismatched_groups_length_raises_value_error() -> None:
    """Mismatched groups length raises ValueError in ACDSORanker.fit() and ndcg_at_k()."""
    from acdso import ACDSORanker
    from acdso.ranking import ndcg_at_k

    X = pd.DataFrame({"a": np.arange(30, dtype=float), "b": np.arange(30, dtype=float)})
    y = np.tile([0, 1, 2], 10)
    bad_groups = np.arange(20)

    with pytest.raises(ValueError):
        ndcg_at_k(y, np.arange(30, dtype=float), groups=bad_groups, k=5)
    with pytest.raises(ValueError):
        ACDSORanker(cv=3).fit(X, y, groups=bad_groups)


@pytest.mark.tier2
@pytest.mark.f1
def test_r5b_t2_unfitted_predict_raises() -> None:
    """Unfitted ACDSORanker raises RuntimeError/ValueError on predict/score/evaluate."""
    from acdso import ACDSORanker

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = np.array([0.0, 1.0, 2.0])
    g = np.array([0, 0, 0])
    ranker = ACDSORanker()
    with pytest.raises((RuntimeError, ValueError)):
        ranker.predict(X)
    with pytest.raises((RuntimeError, ValueError)):
        ranker.score(X, y, groups=g)
    with pytest.raises((RuntimeError, ValueError)):
        ranker.evaluate(X, y, groups=g)


# ===========================================================================
# R5c: Survival Analysis Boundaries & Negative Cases - 5 tests
# ===========================================================================


@pytest.mark.tier2
@pytest.mark.f1
def test_r5c_t2_all_censored_events_handled_or_rejected_cleanly() -> None:
    """All-censored (event == 0 everywhere) input never causes NaN crash or ZeroDivisionError."""
    from acdso import ACDSOSurvivalAnalyzer
    from acdso.survival import concordance_index

    times = np.linspace(1.0, 10.0, 40)
    events = np.zeros(40, dtype=int)
    scores = np.linspace(0.0, 1.0, 40)

    try:
        c_val = concordance_index(times, scores, events)
    except ValueError:
        pass
    else:
        assert np.isfinite(float(c_val))

    rng = np.random.RandomState(42)
    X = pd.DataFrame({"x0": rng.randn(40), "x1": rng.randn(40)})
    surv = ACDSOSurvivalAnalyzer(l2_penalty=0.1, cv=3, random_state=42)
    try:
        surv.fit(X, time=times, event=events)
    except ValueError:
        return
    risks = surv.predict_risk(X.iloc[:5])
    assert np.all(np.isfinite(risks))


@pytest.mark.tier2
@pytest.mark.f1
def test_r5c_t2_negative_nan_time_and_nonbinary_event_raises() -> None:
    """Negative time, NaN time, or non-binary event values raise ValueError."""
    from acdso import ACDSOSurvivalAnalyzer

    rng = np.random.RandomState(42)
    X = pd.DataFrame({"x0": rng.randn(30), "x1": rng.randn(30)})
    valid_t = rng.uniform(0.5, 5.0, size=30)
    valid_e = rng.randint(0, 2, size=30)

    neg_t = valid_t.copy()
    neg_t[0] = -1.0
    with pytest.raises(ValueError):
        ACDSOSurvivalAnalyzer().fit(X, time=neg_t, event=valid_e)

    nan_t = valid_t.copy()
    nan_t[3] = np.nan
    with pytest.raises(ValueError):
        ACDSOSurvivalAnalyzer().fit(X, time=nan_t, event=valid_e)

    bad_e = valid_e.copy()
    bad_e[1] = 2
    with pytest.raises(ValueError):
        ACDSOSurvivalAnalyzer().fit(X, time=valid_t, event=bad_e)


@pytest.mark.tier2
@pytest.mark.f1
def test_r5c_t2_all_uncensored_events_valid_c_index() -> None:
    """All-uncensored (event == 1 everywhere) survival data produces valid C-index > 0.65."""
    from acdso import ACDSOSurvivalAnalyzer

    rng = np.random.RandomState(42)
    n = 100
    x0 = rng.randn(n)
    x1 = rng.randn(n)
    t = rng.exponential(scale=1.0 / np.exp(1.3 * x0 - 0.9 * x1))
    t = np.maximum(t, 1e-3)
    e = np.ones(n, dtype=int)
    X = pd.DataFrame({"x0": x0, "x1": x1})

    surv = ACDSOSurvivalAnalyzer(backend="scipy", l2_penalty=1e-2, cv=3, random_state=42)
    surv.fit(X.iloc[:75], time=t[:75], event=e[:75])
    c_idx = float(surv.score(X.iloc[75:], time=t[75:], event=e[75:]))
    assert 0.65 < c_idx <= 1.0


@pytest.mark.tier2
@pytest.mark.f1
def test_r5c_t2_tied_event_times_handled_cleanly() -> None:
    """Tied event times fit cleanly under Breslow partial likelihood and compute finite concordance_index."""
    from acdso import ACDSOSurvivalAnalyzer
    from acdso.survival import concordance_index

    rng = np.random.RandomState(42)
    n = 60
    X = pd.DataFrame({"x0": rng.randn(n), "x1": rng.randn(n)})
    # Heavy ties across discrete integer times 1..4
    t = np.where(X["x0"] > 0, rng.choice([1.0, 2.0], size=n), rng.choice([3.0, 4.0], size=n))
    e = np.ones(n, dtype=int)

    surv = ACDSOSurvivalAnalyzer(backend="scipy", l2_penalty=1e-2, cv=3, random_state=42)
    surv.fit(X, time=t, event=e)
    risks = surv.predict_risk(X)
    assert np.all(np.isfinite(risks))
    c_val = concordance_index(t, risks, e)
    assert 0.0 <= c_val <= 1.0 and np.isfinite(c_val)


@pytest.mark.tier2
@pytest.mark.f1
def test_r5c_t2_collinear_features_with_l2_penalty_converges() -> None:
    """Near-collinear features with l2_penalty > 0 converge without singular matrix errors or NaNs."""
    from acdso import ACDSOSurvivalAnalyzer

    rng = np.random.RandomState(42)
    n = 80
    x0 = rng.randn(n)
    x1 = x0 + 1e-8 * rng.randn(n)
    t = np.maximum(rng.exponential(scale=1.0 / np.exp(0.8 * x0)), 1e-3)
    e = rng.binomial(1, 0.8, size=n)
    X = pd.DataFrame({"x0": x0, "x1": x1})

    surv = ACDSOSurvivalAnalyzer(backend="scipy", l2_penalty=0.1, cv=3, random_state=42)
    surv.fit(X, time=t, event=e)
    risks = surv.predict_risk(X)
    assert risks.shape == (n,)
    assert np.all(np.isfinite(risks))


# ===========================================================================
# R5d: Recommender System Boundaries & Negative Cases - 5 tests
# ===========================================================================


@pytest.mark.tier2
@pytest.mark.f1
def test_r5d_t2_cold_start_unseen_user_and_item_bias_fallback() -> None:
    """Cold-start completely unseen user_id and item_id return finite bias fallback predictions without KeyError."""
    from acdso import ACDSORecommender

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "item_id": ["i1", "i2", "i1", "i3", "i2", "i3"],
            "rating": [5.0, 3.0, 4.0, 2.0, 3.0, 4.0],
        }
    )
    rec = ACDSORecommender(method="svd", n_factors=2, random_state=42).fit(
        df, user_col="user_id", item_col="item_id", rating_col="rating"
    )
    preds = rec.predict(["brand_new_user", "u1", "brand_new_user"], ["brand_new_item", "brand_new_item", "i1"])
    assert preds.shape == (3,)
    assert np.all(np.isfinite(preds))


@pytest.mark.tier2
@pytest.mark.f1
def test_r5d_t2_recommend_unknown_user_safe() -> None:
    """recommend(unknown_user) returns a bounded valid list without raising KeyError."""
    from acdso import ACDSORecommender

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "item_id": ["i1", "i2", "i2", "i3", "i1", "i3"],
            "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 2.0],
        }
    )
    rec = ACDSORecommender(method="svd", n_factors=2, random_state=42).fit(
        df, user_col="user_id", item_col="item_id", rating_col="rating"
    )
    recs = rec.recommend("never_seen_user", n=5)
    assert isinstance(recs, list)
    assert len(recs) <= 5
    for item, score in recs:
        assert np.isfinite(float(score))


@pytest.mark.tier2
@pytest.mark.f1
def test_r5d_t2_empty_or_missing_columns_raises_value_error() -> None:
    """Empty interactions DataFrame or missing user_col/item_col raises ValueError."""
    from acdso import ACDSORecommender

    empty_df = pd.DataFrame({"user_id": pd.Series(dtype=str), "item_id": pd.Series(dtype=str)})
    with pytest.raises(ValueError):
        ACDSORecommender().fit(empty_df, user_col="user_id", item_col="item_id")

    bad_cols = pd.DataFrame({"uid": ["u1", "u2"], "iid": ["i1", "i2"]})
    with pytest.raises(ValueError):
        ACDSORecommender().fit(bad_cols, user_col="user_id", item_col="item_id")


@pytest.mark.tier2
@pytest.mark.f1
def test_r5d_t2_recommend_n_zero_and_n_exceeds_items() -> None:
    """recommend(..., n=0) returns [] and n > n_items bounds cleanly to available unseen items."""
    from acdso import ACDSORecommender

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "item_id": ["i1", "i2", "i2", "i3", "i1", "i4"],
            "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 2.0],
        }
    )
    rec = ACDSORecommender(method="item_knn", n_neighbors=2).fit(
        df, user_col="user_id", item_col="item_id", rating_col="rating"
    )
    assert rec.recommend("u1", n=0) == []
    all_unseen = rec.recommend("u1", n=10_000, exclude_seen=True)
    assert len(all_unseen) <= 2


@pytest.mark.tier2
@pytest.mark.f1
def test_r5d_t2_duplicate_user_item_interactions_handled() -> None:
    """Duplicate (user_id, item_id) interaction rows are aggregated/handled cleanly without crash."""
    from acdso import ACDSORecommender

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
            "item_id": ["i1", "i1", "i2", "i1", "i3", "i2"],
            "rating": [5.0, 3.0, 4.0, 4.0, 2.0, 5.0],
        }
    )
    rec = ACDSORecommender(method="svd", n_factors=2, random_state=42).fit(
        df, user_col="user_id", item_col="item_id", rating_col="rating"
    )
    preds = rec.predict(["u1", "u2"], ["i3", "i2"])
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))
    assert isinstance(rec.recommend("u1", n=2), list)


# ===========================================================================
# R6: Out-of-Core / Chunked Scale Boundaries & Negative Cases - 5 tests
# ===========================================================================


@pytest.mark.tier2
@pytest.mark.f1
def test_r6_t2_nonexistent_file_raises_file_not_found(tmp_path: Path) -> None:
    """iter_file_chunks, reservoir_sample_file, and ACDSOOutOfCoreTrainer raise FileNotFoundError on missing path."""
    from acdso import ACDSOOutOfCoreTrainer
    from acdso.scale import iter_file_chunks, reservoir_sample_file

    missing = tmp_path / "does_not_exist_ooc.csv"
    with pytest.raises(FileNotFoundError):
        list(iter_file_chunks(missing, chunk_size=10))
    with pytest.raises(FileNotFoundError):
        reservoir_sample_file(missing, max_rows=10)
    with pytest.raises(FileNotFoundError):
        ACDSOOutOfCoreTrainer(task="classification").fit_from_file(missing, target_col="target")


@pytest.mark.tier2
@pytest.mark.f1
def test_r6_t2_empty_header_only_csv_handled_cleanly(tmp_path: Path) -> None:
    """Header-only CSV (0 rows) yields 0 rows in reservoir_sample_file/iter_file_chunks and raises ValueError on fit."""
    from acdso import ACDSOOutOfCoreTrainer
    from acdso.scale import iter_file_chunks, reservoir_sample_file

    header_csv = tmp_path / "header_only.csv"
    header_csv.write_text("x1,x2,target\n", encoding="utf-8")

    chunks = [c for c in iter_file_chunks(header_csv, chunk_size=50) if len(c) > 0]
    assert chunks == []
    sample_df, total_rows = reservoir_sample_file(header_csv, max_rows=10, chunk_size=50)
    assert total_rows == 0
    assert len(sample_df) == 0

    with pytest.raises(ValueError):
        ACDSOOutOfCoreTrainer(task="classification").fit_from_file(header_csv, target_col="target")


@pytest.mark.tier2
@pytest.mark.f1
def test_r6_t2_chunk_size_one_and_exceeding_total_rows(dataset_factory) -> None:
    """chunk_size=1 and chunk_size > total_rows both fit cleanly and produce valid matching-length predictions."""
    from acdso import ACDSOOutOfCoreTrainer

    df = dataset_factory.make_classification_df(n_rows=40, seed=42)
    csv_path = dataset_factory.write_csv(df, "ooc_boundary_chunks.csv")
    X_eval = df.drop(columns=["target"])

    t_one = ACDSOOutOfCoreTrainer(task="classification", chunk_size=1, random_state=42)
    t_one.fit_from_file(csv_path, target_col="target")
    p_one = t_one.predict(X_eval)

    t_big = ACDSOOutOfCoreTrainer(task="classification", chunk_size=10_000, random_state=42)
    t_big.fit_from_file(csv_path, target_col="target")
    p_big = t_big.predict(X_eval)

    assert len(p_one) == 40 and len(p_big) == 40
    assert set(np.unique(p_one)).issubset({0, 1})
    assert set(np.unique(p_big)).issubset({0, 1})


@pytest.mark.tier2
@pytest.mark.f1
def test_r6_t2_reservoir_sample_max_rows_exceeds_total(dataset_factory) -> None:
    """reservoir_sample_file with max_rows > total_rows returns all rows and exact total_rows."""
    from acdso.scale import reservoir_sample_file

    df = dataset_factory.make_regression_df(n_rows=50, seed=42)
    csv_path = dataset_factory.write_csv(df, "res_small_50.csv")

    sample_df, total_rows = reservoir_sample_file(csv_path, max_rows=500, chunk_size=20, random_state=42)
    assert total_rows == 50
    assert len(sample_df) == 50
    assert list(sample_df.columns) == list(df.columns)


@pytest.mark.tier2
@pytest.mark.f1
def test_r6_t2_unfitted_out_of_core_predict_raises(dataset_factory, tmp_path: Path) -> None:
    """Calling predict() or predict_to_file() on an unfitted ACDSOOutOfCoreTrainer raises RuntimeError/ValueError."""
    from acdso import ACDSOOutOfCoreTrainer

    df = dataset_factory.make_classification_df(n_rows=20)
    in_csv = dataset_factory.write_csv(df.drop(columns=["target"]), "ooc_unfit.csv")
    out_csv = tmp_path / "ooc_unfit_out.csv"

    trainer = ACDSOOutOfCoreTrainer(task="classification")
    with pytest.raises((RuntimeError, ValueError)):
        trainer.predict(df.drop(columns=["target"]))
    with pytest.raises((RuntimeError, ValueError)):
        trainer.predict_to_file(in_csv, out_csv)


# ===========================================================================
# R7: Interface Wiring Boundaries & Negative Cases - 5 tests
# ===========================================================================


@pytest.mark.tier2
@pytest.mark.f2
def test_r7_t2_cli_invalid_task_exits_nonzero(dataset_factory, run_cli, tmp_path: Path) -> None:
    """CLI --task bogus_task exits with a non-zero return code."""
    csv_path = dataset_factory.write_csv(dataset_factory.make_classification_df(n_rows=30), "cli_bad_task.csv")
    res = run_cli(["--file", str(csv_path), "--task", "bogus_task_xyz"], cwd=tmp_path, timeout=45.0)
    assert res.returncode != 0


@pytest.mark.tier2
@pytest.mark.f2
def test_r7_t2_cli_survival_missing_time_or_event_col_exits_nonzero(dataset_factory, run_cli, tmp_path: Path) -> None:
    """CLI --task survival without required --time-col / --event-col exits non-zero cleanly."""
    csv_path = dataset_factory.write_csv(dataset_factory.make_classification_df(n_rows=40), "cli_bad_surv.csv")
    res = run_cli(["--file", str(csv_path), "--task", "survival", "--simple"], cwd=tmp_path, timeout=45.0)
    assert res.returncode != 0


@pytest.mark.tier2
@pytest.mark.f4
def test_r7_t2_webui_run_task_validation_and_targetless_clustering(webui_client, dataset_factory) -> None:
    """WebUI POST /api/run returns 400 on invalid task / missing survival cols while permitting targetless clustering."""
    _, client = webui_client
    df = dataset_factory.make_classification_df(n_rows=40)
    csv_path = dataset_factory.write_csv(df, "e2e_r7_webui_run.csv")
    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, csv_path.name)}, content_type="multipart/form-data")
    assert up.status_code == 200
    uid = up.get_json()["upload_id"]

    # Invalid task name -> 400
    r_bad_task = client.post("/api/run", json={"upload_id": uid, "predict": ["target"], "task": "bogus_task"})
    assert r_bad_task.status_code == 400

    # Survival without time_col / event_col -> 400
    r_bad_surv = client.post("/api/run", json={"upload_id": uid, "task": "survival"})
    assert r_bad_surv.status_code == 400

    # Clustering without target/predict -> 200 (permitted)
    r_cluster = client.post("/api/run", json={"upload_id": uid, "task": "clustering", "simple": True})
    assert r_cluster.status_code == 200, r_cluster.get_json()
    job_id = r_cluster.get_json()["job_id"]
    try:
        assert r_cluster.get_json()["status"] in {"running", "finished"}
    finally:
        stop_job_via_client(client, job_id)
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            r_j = client.get(f"/api/job/{job_id}")
            if r_j.status_code != 200 or (r_j.get_json() or {}).get("status") != "running":
                break
            time.sleep(0.15)


@pytest.mark.tier2
@pytest.mark.f3
def test_r7_t2_webui_validate_columns_role_conflicts_on_new_cols(isolated_webui_client, dataset_factory) -> None:
    """POST /api/validate-columns detects role conflicts involving new R1-R6 roles (group_col, time_col, event_col)."""
    _, client, _ = isolated_webui_client
    df = pd.DataFrame(
        {
            "group_id": [f"g_{i % 6}" for i in range(30)],
            "t_col": np.linspace(1.0, 10.0, 30),
            "e_col": [i % 2 for i in range(30)],
            "num1": np.linspace(0.0, 1.0, 30),
            "target": [i % 2 for i in range(30)],
        }
    )
    csv_path = dataset_factory.write_csv(df, "r7_col_conflicts.csv")
    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, csv_path.name)}, content_type="multipart/form-data")
    assert up.status_code == 200
    uid = up.get_json()["upload_id"]

    r_grp_conflict = client.post(
        "/api/validate-columns",
        json={"upload_id": uid, "predict": ["target"], "group_col": ["target"]},
    )
    assert r_grp_conflict.status_code == 200
    assert r_grp_conflict.get_json()["ok"] is False

    r_surv_conflict = client.post(
        "/api/validate-columns",
        json={"upload_id": uid, "time_col": ["t_col"], "event_col": ["t_col"]},
    )
    assert r_surv_conflict.status_code == 200
    assert r_surv_conflict.get_json()["ok"] is False


@pytest.mark.tier2
@pytest.mark.f2
def test_r7_t2_tui_validates_missing_target_for_supervised_vs_clustering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TUI rejects missing target on task='classification' while permitting empty target on clustering and anomaly."""
    import acdso.tui.app as tui_app
    from acdso.tui.app import FallbackTerminalUI, build_cli_command

    csv_path = tmp_path / "tui_target_gate.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0], "y": [0, 1, 0]}).to_csv(csv_path, index=False)

    # Empty target allowed for clustering and anomaly in build_cli_command
    for t_name in ("clustering", "anomaly"):
        argv = build_cli_command(str(csv_path), "", task=t_name, simple=True)
        assert "--task" in argv and argv[argv.index("--task") + 1] == t_name
        assert "--target" not in argv and "--predict" not in argv

    class _FakeProc:
        def __init__(self) -> None:
            self.stdout: list[str] = []

        def wait(self) -> int:
            return 0

    captured_cmds: list[list[str]] = []

    def _fake_popen(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        return _FakeProc()

    monkeypatch.setattr(tui_app, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(tui_app.subprocess, "Popen", _fake_popen)

    # 1. Missing target rejected cleanly for supervised classification without launching subprocess
    logs_sup: list[str] = []
    ui_sup = FallbackTerminalUI(initial_file=str(csv_path), input_fn=lambda _: "q", output_fn=logs_sup.append)
    ui_sup.file_path = str(csv_path)
    ui_sup.task = "classification"
    ui_sup.target = ""
    rc_sup = ui_sup._run_pipeline()
    assert rc_sup != 0
    assert len(captured_cmds) == 0
    assert any("target" in line.lower() for line in logs_sup)

    # 2. Empty target permitted for unsupervised clustering and launches subprocess with --task clustering
    logs_clu: list[str] = []
    ui_clu = FallbackTerminalUI(initial_file=str(csv_path), input_fn=lambda _: "q", output_fn=logs_clu.append)
    ui_clu.file_path = str(csv_path)
    ui_clu.task = "clustering"
    ui_clu.target = ""
    rc_clu = ui_clu._run_pipeline()
    assert rc_clu == 0, "\n".join(logs_clu)
    assert len(captured_cmds) == 1
    launched = captured_cmds[0]
    assert "--task" in launched and launched[launched.index("--task") + 1] == "clustering"
    assert "--target" not in launched and "--predict" not in launched
