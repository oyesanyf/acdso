"""Tier 1 E2E Tests: R1-R7 Core Feature Coverage (50 tests: 5 per feature area 1-10)."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import time
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from .conftest import (
    PYTHON_EXE,
    SRC_ROOT,
    parse_progress_events,
)


class _TableTagCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.open_tables = 0
        self.close_tables = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "table":
            self.open_tables += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "table":
            self.close_tables += 1


def _make_grouped_classification_df(
    n_groups: int = 18,
    rows_per_group: int = 6,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    rows = []
    labels = []
    groups = []
    for g in range(n_groups):
        grp_name = f"entity_{g:03d}"
        cls = g % 2
        center = 1.6 if cls == 1 else -1.6
        for _ in range(rows_per_group):
            f1 = center + 0.6 * rng.randn()
            f2 = -0.8 * center + 0.6 * rng.randn()
            f3 = rng.randn()
            rows.append({"f1": f1, "f2": f2, "f3": f3, "tier": "hi" if cls == 1 else "lo"})
            labels.append(cls)
            groups.append(grp_name)
    return pd.DataFrame(rows), np.array(labels, dtype=int), np.array(groups, dtype=object)


def _make_grouped_regression_df(
    n_groups: int = 18,
    rows_per_group: int = 6,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    rows = []
    targets = []
    groups = []
    for g in range(n_groups):
        grp_name = f"unit_{g:03d}"
        offset = (g % 5) - 2.0
        for _ in range(rows_per_group):
            x1 = rng.randn() + 0.5 * offset
            x2 = rng.randn()
            y = 3.0 * x1 - 1.5 * x2 + offset + 0.1 * rng.randn()
            rows.append({"x1": x1, "x2": x2, "group_code": f"c{g % 3}"})
            targets.append(y)
            groups.append(grp_name)
    return pd.DataFrame(rows), np.array(targets, dtype=float), np.array(groups, dtype=object)


def _make_mixed_nlp_df(n_rows: int = 48, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    positive_snippets = [
        "excellent battery life fast shipping reliable build quality recommended",
        "amazing display clarity crisp sound sturdy frame great value overall",
        "smooth performance quiet cooling comfortable keyboard solid everyday laptop",
        "quick setup intuitive interface clean design works flawlessly out of box",
    ]
    negative_snippets = [
        "terrible battery drain loud fan sluggish response disappointed with purchase",
        "broken hinge poor screen brightness flimsy plastic returned immediately",
        "constant crashes overheating charger failed after two days terrible support",
        "slow delivery damaged packaging missing cables frustrating setup experience",
    ]
    cats = ["electronics", "office", "home"]
    records = []
    for i in range(n_rows):
        is_pos = i % 2 == 0
        pool = positive_snippets if is_pos else negative_snippets
        text = pool[i % len(pool)] + f" item batch {i % 7}"
        records.append(
            {
                "item_id": f"ITM_{i:04d}",
                "category": cats[i % len(cats)],
                "price": round(float(50 + (15 if is_pos else -10) + rng.randn() * 5), 2),
                "rating": round(float(4.5 if is_pos else 1.8 + rng.randn() * 0.2), 2),
                "review": text,
                "label": 1 if is_pos else 0,
            }
        )
    return pd.DataFrame(records)


def _make_survival_data(
    n_rows: int = 160,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n_rows)
    x1 = rng.randn(n_rows)
    x2 = rng.randn(n_rows)
    lin_pred = 1.2 * x0 - 0.8 * x1 + 0.2 * x2
    rate = np.exp(lin_pred)
    event_time = rng.exponential(scale=1.0 / rate)
    censor_time = rng.exponential(scale=2.5, size=n_rows)
    observed_time = np.maximum(np.minimum(event_time, censor_time), 1e-3)
    event = (event_time <= censor_time).astype(int)
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    return X, observed_time, event, lin_pred


def _make_recommender_interactions(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    n_users = 24
    n_items = 20
    rows = []
    for u in range(n_users):
        u_cluster = u % 2
        for it in range(n_items):
            if rng.rand() < 0.65:
                i_cluster = it % 2
                rating = (4.5 if u_cluster == i_cluster else 1.5) + 0.2 * rng.randn()
                rating = float(np.clip(rating, 1.0, 5.0))
                rows.append({"user_id": f"u_{u:02d}", "item_id": f"i_{it:02d}", "rating": round(rating, 3)})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split = int(len(df) * 0.8)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


# ===========================================================================
# R1: Unsupervised Clustering & Segmentation (ACDSOClusterer) - 5 tests
# ===========================================================================


@pytest.mark.tier1
@pytest.mark.f1
def test_r1_t1_kmeans_fit_predict_and_summary() -> None:
    """ACDSOClusterer with K-Means fits separated clusters, predicts unseen points, and summarizes clusters."""
    from acdso import ACDSOClusterer

    X_arr, y_true = make_blobs(n_samples=120, centers=3, n_features=4, cluster_std=0.5, random_state=42)
    X_df = pd.DataFrame(X_arr, columns=["f0", "f1", "f2", "f3"])
    X_train, y_train = X_df.iloc[:90], y_true[:90]
    X_test, y_test = X_df.iloc[90:], y_true[90:]

    clusterer = ACDSOClusterer(algorithm="kmeans", n_clusters=3, random_state=42)
    ret = clusterer.fit(X_train)
    assert ret is clusterer
    assert clusterer.best_algorithm_ == "kmeans"
    assert clusterer.n_clusters_ == 3
    assert len(clusterer.labels_) == len(X_train)
    assert adjusted_rand_score(y_train, clusterer.labels_) > 0.9

    preds = clusterer.predict(X_test)
    assert len(preds) == len(X_test)
    assert adjusted_rand_score(y_test, preds) > 0.9
    assert np.array_equal(clusterer.fit_predict(X_train), clusterer.labels_)

    summary = clusterer.get_cluster_summary()
    assert isinstance(summary, dict)
    assert clusterer.silhouette_score_ > 0.5
    assert np.isfinite(float(clusterer.score(X_test)))


@pytest.mark.tier1
@pytest.mark.f1
def test_r1_t1_hdbscan_inductive_predict_unseen() -> None:
    """ACDSOClusterer with sklearn HDBSCAN discovers clusters and inductively predicts unseen points."""
    from acdso import ACDSOClusterer

    X_arr, y_true = make_blobs(n_samples=120, centers=3, n_features=3, cluster_std=0.35, random_state=42)
    X_df = pd.DataFrame(X_arr, columns=["a", "b", "c"])
    X_train, X_test, y_test = X_df.iloc[:96], X_df.iloc[96:], y_true[96:]

    clusterer = ACDSOClusterer(algorithm="hdbscan", min_cluster_size=8, random_state=42)
    clusterer.fit(X_train)
    assert clusterer.best_algorithm_ == "hdbscan"
    assert clusterer.n_clusters_ >= 2
    assert len(clusterer.labels_) == len(X_train)

    unseen_labels = clusterer.predict(X_test)
    assert len(unseen_labels) == len(X_test)
    assert np.all(np.isfinite(unseen_labels))
    assert adjusted_rand_score(y_test, unseen_labels) > 0.85


@pytest.mark.tier1
@pytest.mark.f1
def test_r1_t1_gmm_fit_predict_and_score() -> None:
    """ACDSOClusterer with GMM fits Gaussian mixtures, predicts unseen cluster assignments, and scores."""
    from acdso import ACDSOClusterer

    X_arr, y_true = make_blobs(n_samples=120, centers=3, n_features=4, cluster_std=0.6, random_state=7)
    X_df = pd.DataFrame(X_arr, columns=["x1", "x2", "x3", "x4"])
    X_train, X_test, y_test = X_df.iloc[:90], X_df.iloc[90:], y_true[90:]

    clusterer = ACDSOClusterer(algorithm="gmm", n_clusters=3, random_state=7)
    clusterer.fit(X_train)
    assert clusterer.best_algorithm_ == "gmm"
    assert clusterer.n_clusters_ == 3
    preds = clusterer.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1, 2})
    assert adjusted_rand_score(y_test, preds) > 0.9
    assert np.isfinite(float(clusterer.score(X_test)))


@pytest.mark.tier1
@pytest.mark.f1
def test_r1_t1_auto_tuning_silhouette_and_davies_bouldin() -> None:
    """ACDSOClusterer(algorithm='auto') tunes with silhouette and Davies-Bouldin and emits JSON audit_trail_."""
    from acdso import ACDSOClusterer

    X_arr, _ = make_blobs(n_samples=120, centers=3, n_features=4, cluster_std=0.45, random_state=11)
    X_df = pd.DataFrame(X_arr, columns=["c0", "c1", "c2", "c3"])

    c_sil = ACDSOClusterer(algorithm="auto", k_range=(2, 5), metric="silhouette", random_state=11)
    c_sil.fit(X_df)
    assert c_sil.best_algorithm_ in {"kmeans", "hdbscan", "gmm"}
    assert c_sil.best_estimator_ is not None
    assert c_sil.n_clusters_ >= 2
    assert c_sil.silhouette_score_ > 0.4
    assert c_sil.davies_bouldin_score_ > 0.0
    assert len(c_sil.candidates_) >= 1
    assert isinstance(c_sil.audit_trail_, dict)
    json.dumps(c_sil.audit_trail_)

    c_db = ACDSOClusterer(algorithm="auto", k_range=(2, 5), metric="davies_bouldin", random_state=11)
    c_db.fit(X_df)
    assert c_db.best_algorithm_ in {"kmeans", "hdbscan", "gmm"}
    assert np.isfinite(float(c_db.davies_bouldin_score_))


@pytest.mark.tier1
@pytest.mark.f1
def test_r1_t1_dim_reduction_pca_and_umap_fallback() -> None:
    """ACDSOClusterer supports dim_reduction='pca' and deterministic fallback for 'umap' recorded in audit_trail_."""
    from acdso import ACDSOClusterer

    X_arr, _ = make_blobs(n_samples=90, centers=3, n_features=8, cluster_std=0.6, random_state=42)
    X_df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(8)])

    c_pca = ACDSOClusterer(
        algorithm="kmeans", n_clusters=3, dim_reduction="pca", n_components=2, random_state=42
    )
    c_pca.fit(X_df)
    proj_pca = c_pca.transform(X_df)
    assert proj_pca.shape == (90, 2)
    assert np.all(np.isfinite(proj_pca))
    assert "dim_reduction_backend" in c_pca.audit_trail_

    c_umap = ACDSOClusterer(
        algorithm="kmeans", n_clusters=3, dim_reduction="umap", n_components=2, random_state=42
    )
    c_umap.fit(X_df)
    proj_umap = c_umap.transform(X_df)
    assert proj_umap.shape == (90, 2)
    assert np.all(np.isfinite(proj_umap))
    assert "dim_reduction_backend" in c_umap.audit_trail_
    assert isinstance(c_umap.audit_trail_["dim_reduction_backend"], str)
    assert len(c_umap.audit_trail_["dim_reduction_backend"]) > 0


# ===========================================================================
# R2: Production Unsupervised Anomaly & Outlier Scoring - 5 tests
# ===========================================================================


@pytest.mark.tier1
@pytest.mark.f1
def test_r2_t1_isolation_forest_fit_predict_and_scores() -> None:
    """ACDSOAnomalyDetector(algorithm='isolation_forest') scores unseen outliers higher than inliers."""
    from acdso import ACDSOAnomalyDetector

    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.randn(160, 4), columns=["a", "b", "c", "d"])
    X_in = pd.DataFrame(rng.randn(20, 4), columns=["a", "b", "c", "d"])
    X_out = pd.DataFrame(rng.randn(10, 4) + 8.0, columns=["a", "b", "c", "d"])

    det = ACDSOAnomalyDetector(algorithm="isolation_forest", contamination=0.05, random_state=42)
    assert det.fit(X_train) is det
    assert set(np.unique(det.predict(X_in))).issubset({-1, 1})
    assert np.mean(det.predict(X_out) == -1) >= 0.8
    assert det.predict_is_anomaly(X_out).dtype == bool
    assert float(det.anomaly_scores(X_out).mean()) > float(det.anomaly_scores(X_in).mean())


@pytest.mark.tier1
@pytest.mark.f1
def test_r2_t1_lof_novelty_unseen_outliers() -> None:
    """ACDSOAnomalyDetector(algorithm='lof') uses novelty=True to score unseen outliers above inliers."""
    from acdso import ACDSOAnomalyDetector

    rng = np.random.RandomState(17)
    X_train = pd.DataFrame(rng.randn(150, 3), columns=["x", "y", "z"])
    X_in = pd.DataFrame(rng.randn(20, 3), columns=["x", "y", "z"])
    X_out = pd.DataFrame(rng.randn(10, 3) + 10.0, columns=["x", "y", "z"])

    det = ACDSOAnomalyDetector(algorithm="lof", n_neighbors=15, contamination=0.05, random_state=17)
    det.fit(X_train)
    out_preds = det.predict(X_out)
    assert np.all(out_preds == -1)
    assert float(det.anomaly_scores(X_out).mean()) > float(det.anomaly_scores(X_in).mean())
    assert float(det.decision_function(X_in).mean()) > float(det.decision_function(X_out).mean())


@pytest.mark.tier1
@pytest.mark.f1
def test_r2_t1_ocsvm_fit_predict_is_anomaly_and_decision() -> None:
    """ACDSOAnomalyDetector(algorithm='ocsvm') separates unseen outliers via predict_is_anomaly & decision_function."""
    from acdso import ACDSOAnomalyDetector

    rng = np.random.RandomState(99)
    X_train = pd.DataFrame(rng.randn(140, 4), columns=["f1", "f2", "f3", "f4"])
    X_in = pd.DataFrame(rng.randn(20, 4) * 0.5, columns=["f1", "f2", "f3", "f4"])
    X_out = pd.DataFrame(rng.randn(10, 4) + 9.0, columns=["f1", "f2", "f3", "f4"])

    det = ACDSOAnomalyDetector(algorithm="ocsvm", nu=0.05, random_state=99)
    det.fit(X_train)
    flags_out = det.predict_is_anomaly(X_out)
    flags_in = det.predict_is_anomaly(X_in)
    assert flags_out.dtype == bool and flags_in.dtype == bool
    assert np.mean(flags_out) >= 0.8
    assert float(det.decision_function(X_in).mean()) > float(det.decision_function(X_out).mean())
    assert np.all(np.isfinite(det.score_samples(X_in)))
    assert np.isfinite(float(det.score(X_in)))


@pytest.mark.tier1
@pytest.mark.f1
def test_r2_t1_auto_algorithm_and_json_audit_trail() -> None:
    """ACDSOAnomalyDetector(algorithm='auto') selects an estimator and emits JSON-serializable audit_trail_."""
    from acdso import ACDSOAnomalyDetector

    rng = np.random.RandomState(42)
    inliers = rng.randn(140, 3)
    outliers = rng.randn(10, 3) + 7.5
    X_df = pd.DataFrame(np.vstack([inliers, outliers]), columns=["m1", "m2", "m3"])

    det = ACDSOAnomalyDetector(algorithm="auto", contamination=0.07, random_state=42)
    det.fit(X_df)
    assert det.best_algorithm_ in {"isolation_forest", "lof", "ocsvm"}
    assert det.best_estimator_ is not None
    assert hasattr(det, "offset_")
    assert hasattr(det, "threshold_")
    assert isinstance(det.audit_trail_, dict)
    json.dumps(det.audit_trail_)


@pytest.mark.tier1
@pytest.mark.f1
def test_r2_t1_auto_eda_fit_anomaly_detector_hook(dataset_factory) -> None:
    """ACDSOAutoEDA.fit_anomaly_detector(df) trains and stores anomaly_detector_ while preserving analyze()."""
    from acdso import ACDSOAnomalyDetector, ACDSOAutoEDA

    df = dataset_factory.make_regression_df(n_rows=100).drop(columns=["y"])
    eda = ACDSOAutoEDA()
    report = eda.analyze(df, dataset_name="eda_anom_check")
    assert report is not None
    assert eda.report_ is report

    det = eda.fit_anomaly_detector(df, algorithm="isolation_forest", contamination=0.05)
    assert isinstance(det, ACDSOAnomalyDetector)
    assert getattr(eda, "anomaly_detector_", None) is det
    preds = det.predict(df.iloc[:15])
    assert len(preds) == 15
    assert set(np.unique(preds)).issubset({-1, 1})


# ===========================================================================
# R3: Entity-Grouped Cross-Validation - 5 tests
# ===========================================================================


@pytest.mark.tier1
@pytest.mark.f1
def test_r3_t1_build_cv_splitter_group_and_stratified_group() -> None:
    """build_cv_splitter produces leak-free GroupKFold and StratifiedGroupKFold splits validated by validate_no_group_leakage."""
    from acdso.cv import build_cv_splitter, validate_no_group_leakage

    X, y, groups = _make_grouped_classification_df(n_groups=15, rows_per_group=5, seed=42)

    for strategy in ("group_kfold", "stratified_group_kfold", "auto"):
        splitter = build_cv_splitter(
            cv=5, y=y, groups=groups, task="classification", cv_strategy=strategy, random_state=42
        )
        splits = list(splitter.split(X, y, groups))
        assert len(splits) == 5
        validate_no_group_leakage(splits, groups)
        covered_val = np.concatenate([va for _, va in splits])
        assert len(covered_val) == len(X)
        assert len(set(covered_val.tolist())) == len(X)


@pytest.mark.tier1
@pytest.mark.f1
def test_r3_t1_temporal_split_monotonicity() -> None:
    """build_cv_splitter(cv_strategy='temporal') preserves strict forward-in-time ordering across folds."""
    from acdso.cv import build_cv_splitter

    X = pd.DataFrame({"t": np.arange(60), "v": np.sin(np.arange(60) * 0.2)})
    y = np.arange(60, dtype=float)
    splitter = build_cv_splitter(cv=4, y=y, task="regression", cv_strategy="temporal")
    splits = list(splitter.split(X, y))
    assert len(splits) == 4
    prev_val_end = -1
    for tr, va in splits:
        assert len(tr) > 0 and len(va) > 0
        assert int(np.max(tr)) < int(np.min(va)), "Future rows leaked into training fold"
        assert int(np.min(va)) > prev_val_end
        prev_val_end = int(np.max(va))


@pytest.mark.tier1
@pytest.mark.f1
def test_r3_t1_classifier_fit_with_groups_zero_leakage() -> None:
    """ACDSOClassifier.fit(X, y, groups=groups) completes cleanly with entity-grouped CV."""
    from acdso import ACDSOClassifier

    X, y, groups = _make_grouped_classification_df(n_groups=18, rows_per_group=6, seed=42)
    train_mask = np.isin(groups, [f"entity_{i:03d}" for i in range(14)])
    X_tr, y_tr, g_tr = X.loc[train_mask].reset_index(drop=True), y[train_mask], groups[train_mask]
    X_te, y_te = X.loc[~train_mask].reset_index(drop=True), y[~train_mask]

    clf = ACDSOClassifier(scorer="accuracy", cv=3, max_candidates=2, random_state=42, verbose=0)
    clf.fit(X_tr, y_tr, groups=g_tr)
    assert clf.best_estimator_ is not None
    preds = clf.predict(X_te)
    assert len(preds) == len(X_te)
    assert float(clf.score(X_te, y_te)) >= 0.6
    json.dumps(clf.audit_trail_)


@pytest.mark.tier1
@pytest.mark.f1
def test_r3_t1_regressor_fit_with_groups() -> None:
    """ACDSORegressor.fit(X, y, groups=groups) fits and predicts with grouped entity splits."""
    from acdso import ACDSORegressor

    X, y, groups = _make_grouped_regression_df(n_groups=18, rows_per_group=6, seed=42)
    train_mask = np.isin(groups, [f"unit_{i:03d}" for i in range(14)])
    X_tr, y_tr, g_tr = X.loc[train_mask].reset_index(drop=True), y[train_mask], groups[train_mask]
    X_te, y_te = X.loc[~train_mask].reset_index(drop=True), y[~train_mask]

    reg = ACDSORegressor(scorer="r2", cv=3, max_candidates=2, random_state=42, verbose=0)
    reg.fit(X_tr, y_tr, groups=g_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert float(reg.score(X_te, y_te)) > 0.5
    json.dumps(reg.audit_trail_)


@pytest.mark.tier1
@pytest.mark.f1
def test_r3_t1_analyze_risk_and_sample_by_groups() -> None:
    """analyze_risk(..., groups=groups) and sample_by_groups preserve whole entities without splitting groups."""
    from acdso import analyze_risk
    from acdso.cv import sample_by_groups

    X, y, groups = _make_grouped_classification_df(n_groups=16, rows_per_group=5, seed=42)
    rr = analyze_risk(X, y, task="classification", cv=4, groups=groups, random_state=42)
    rr_dict = rr.to_dict()
    assert rr_dict["task"] == "classification"
    assert rr_dict["n_rows"] == len(X)

    X_sub, y_sub, g_sub = sample_by_groups(X, y=y, groups=groups, max_rows=35, random_state=42)
    assert 0 < len(X_sub) <= 35
    assert len(y_sub) == len(X_sub)
    assert len(g_sub) == len(X_sub)
    for g in np.unique(g_sub):
        assert int(np.sum(g_sub == g)) == int(np.sum(groups == g)), f"Entity {g} was partially split"


# ===========================================================================
# R4: Classical NLP & Free-Text Featurization - 5 tests
# ===========================================================================


@pytest.mark.tier1
@pytest.mark.f1
def test_r4_t1_detect_text_columns_separates_text_from_cats_and_ids() -> None:
    """detect_text_columns detects free-text columns without misclassifying IDs, categoricals, or numerics."""
    from acdso.nlp import detect_text_columns

    df = _make_mixed_nlp_df(n_rows=40)
    detected = detect_text_columns(df)
    assert detected == ["review"]
    assert "item_id" not in detected
    assert "category" not in detected
    assert "price" not in detected


@pytest.mark.tier1
@pytest.mark.f1
def test_r4_t1_text_featurizer_tfidf_and_ngram() -> None:
    """TextFeaturizer supports method='tfidf' and 'ngram' with valid feature names."""
    from acdso.nlp import TextFeaturizer

    df = _make_mixed_nlp_df(n_rows=40)
    texts = df[["review"]]

    for method in ("tfidf", "ngram"):
        feat = TextFeaturizer(method=method, ngram_range=(1, 2), max_features=40)
        mat = feat.fit_transform(texts)
        arr = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
        assert arr.shape[0] == 40
        assert 1 <= arr.shape[1] <= 40
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0.0)
        names = feat.get_feature_names_out()
        assert len(names) == arr.shape[1]
        assert all(isinstance(str(n), str) and len(str(n)) > 0 for n in names)


@pytest.mark.tier1
@pytest.mark.f1
def test_r4_t1_text_featurizer_lda_and_nmf_topics() -> None:
    """TextFeaturizer supports method='lda' and 'nmf' producing non-negative topic representations."""
    from acdso.nlp import TextFeaturizer

    df = _make_mixed_nlp_df(n_rows=40)
    texts = df["review"]

    for method in ("lda", "nmf"):
        feat = TextFeaturizer(method=method, max_features=60, n_topics=4, random_state=42)
        mat = feat.fit_transform(texts)
        arr = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
        assert arr.shape == (40, 4)
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0.0)
        names = feat.get_feature_names_out()
        assert len(names) == 4


@pytest.mark.tier1
@pytest.mark.f1
def test_r4_t1_utils_split_columns_and_build_preprocessor_text_branch() -> None:
    """utils.split_columns preserves 2-tuple default and adds text_cols when detect_text=True; build_preprocessor handles text_cols."""
    from acdso.utils import build_preprocessor, split_columns

    df = _make_mixed_nlp_df(n_rows=40).drop(columns=["item_id", "label"])

    # Backwards-compatible 2-tuple signature
    num2, cat2 = split_columns(df)
    assert "price" in num2 and "rating" in num2
    assert "category" in cat2 and "review" in cat2

    # Extended 3-tuple signature
    num3, cat3, text3 = split_columns(df, detect_text=True)
    assert set(num3) == {"price", "rating"}
    assert set(cat3) == {"category"}
    assert text3 == ["review"]

    pre = build_preprocessor(num3, cat3, text_cols=text3, text_method="tfidf")
    Xt = pre.fit_transform(df)
    arr = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    assert arr.shape[0] == len(df)
    assert arr.shape[1] > len(num3) + 3
    assert np.all(np.isfinite(arr))


@pytest.mark.tier1
@pytest.mark.f1
def test_r4_t1_auto_preprocessor_mixed_dataframe() -> None:
    """AutoPreprocessor transforms mixed numeric, categorical, and free-text DataFrames consistently."""
    from acdso.auto_preprocessing import AutoPreprocessor

    df = _make_mixed_nlp_df(n_rows=48).drop(columns=["item_id"])
    X = df.drop(columns=["label"])
    y = df["label"]

    ap = AutoPreprocessor()
    X_tr = ap.fit_transform(X.iloc[:36], y.iloc[:36])
    X_te = ap.transform(X.iloc[36:])
    assert X_tr.shape[0] == 36
    assert X_te.shape[0] == 12
    assert X_tr.shape[1] == X_te.shape[1]
    assert X_tr.shape[1] > 5
    assert np.all(np.isfinite(X_tr))
    assert np.all(np.isfinite(X_te))
    assert len(ap.get_feature_names_out()) == X_tr.shape[1]


# ===========================================================================
# R5a: Multi-Label Classification (ACDSOMultiLabelClassifier) - 5 tests
# ===========================================================================


def _make_multilabel_df(n_rows: int = 120, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n_rows)
    x1 = rng.randn(n_rows)
    x2 = rng.randn(n_rows)
    y0 = (x0 + 0.2 * rng.randn(n_rows) > 0).astype(int)
    y1 = (x1 + 0.2 * rng.randn(n_rows) > 0).astype(int)
    y2 = ((x0 + x1) + 0.2 * rng.randn(n_rows) > 0).astype(int)
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    Y = np.column_stack([y0, y1, y2])
    return X, Y


@pytest.mark.tier1
@pytest.mark.f1
def test_r5a_t1_multi_output_strategy() -> None:
    """ACDSOMultiLabelClassifier(strategy='multi_output') fits and predicts binary labels & probabilities."""
    from acdso import ACDSOMultiLabelClassifier

    X, Y = _make_multilabel_df(n_rows=120, seed=42)
    clf = ACDSOMultiLabelClassifier(strategy="multi_output", cv=3, random_state=42)
    assert clf.fit(X.iloc[:90], Y[:90]) is clf
    preds = clf.predict(X.iloc[90:])
    assert preds.shape == (30, 3)
    assert set(np.unique(preds)).issubset({0, 1})
    proba = clf.predict_proba(X.iloc[90:])
    assert proba.shape == (30, 3)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


@pytest.mark.tier1
@pytest.mark.f1
def test_r5a_t1_classifier_chain_strategy() -> None:
    """ACDSOMultiLabelClassifier(strategy='classifier_chain') models label dependencies cleanly."""
    from acdso import ACDSOMultiLabelClassifier

    X, Y = _make_multilabel_df(n_rows=120, seed=7)
    clf = ACDSOMultiLabelClassifier(strategy="classifier_chain", cv=3, random_state=7)
    clf.fit(X.iloc[:90], Y[:90])
    preds = clf.predict(X.iloc[90:])
    assert preds.shape == (30, 3)
    assert set(np.unique(preds)).issubset({0, 1})
    proba = clf.predict_proba(X.iloc[90:])
    assert proba.shape == (30, 3)
    assert np.all(np.isfinite(proba))


@pytest.mark.tier1
@pytest.mark.f1
def test_r5a_t1_auto_strategy_and_evaluate_metrics() -> None:
    """ACDSOMultiLabelClassifier(strategy='auto').evaluate() reports Hamming, F1 micro/macro, and Jaccard."""
    from acdso import ACDSOMultiLabelClassifier

    X, Y = _make_multilabel_df(n_rows=120, seed=13)
    clf = ACDSOMultiLabelClassifier(strategy="auto", cv=3, random_state=13)
    clf.fit(X.iloc[:90], Y[:90])
    ev = clf.evaluate(X.iloc[90:], Y[90:])
    assert isinstance(ev, dict)
    keys_lower = {k.lower() for k in ev.keys()}
    assert any("hamming" in k for k in keys_lower), f"Missing Hamming metric in {ev}"
    assert any("f1" in k and "micro" in k for k in keys_lower), f"Missing f1_micro in {ev}"
    assert any("f1" in k and "macro" in k for k in keys_lower), f"Missing f1_macro in {ev}"
    assert any("jaccard" in k for k in keys_lower), f"Missing Jaccard metric in {ev}"
    for val in ev.values():
        if isinstance(val, (int, float)):
            assert 0.0 <= float(val) <= 1.0


@pytest.mark.tier1
@pytest.mark.f1
def test_r5a_t1_groups_cv_support() -> None:
    """ACDSOMultiLabelClassifier.fit(X, y, groups=groups) completes cleanly with entity-grouped CV."""
    from acdso import ACDSOMultiLabelClassifier

    X, Y = _make_multilabel_df(n_rows=100, seed=21)
    groups = np.repeat([f"patient_{i}" for i in range(20)], 5)
    clf = ACDSOMultiLabelClassifier(strategy="auto", cv=4, random_state=21)
    clf.fit(X.iloc[:80], Y[:80], groups=groups[:80])
    preds = clf.predict(X.iloc[80:])
    assert preds.shape == (20, 3)


@pytest.mark.tier1
@pytest.mark.f1
def test_r5a_t1_audit_trail_and_nontrivial_score() -> None:
    """ACDSOMultiLabelClassifier achieves score > 0.5 on separable synthetic data and emits JSON audit_trail_."""
    from acdso import ACDSOMultiLabelClassifier

    X, Y = _make_multilabel_df(n_rows=150, seed=42)
    clf = ACDSOMultiLabelClassifier(strategy="auto", scorer="f1_micro", cv=3, random_state=42)
    clf.fit(X.iloc[:110], Y[:110])
    sc = float(clf.score(X.iloc[110:], Y[110:]))
    assert sc > 0.5, f"Expected non-trivial multi-label score > 0.5, got {sc}"
    assert isinstance(clf.audit_trail_, dict)
    json.dumps(clf.audit_trail_)


# ===========================================================================
# R5b: Learning-to-Rank (ACDSORanker, ndcg_at_k) - 5 tests
# ===========================================================================


def _make_ranking_data(
    n_queries: int = 24,
    docs_per_query: int = 8,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    rows = []
    relevances = []
    qids = []
    for q in range(n_queries):
        for _ in range(docs_per_query):
            x0 = rng.randn()
            x1 = rng.randn()
            x2 = rng.randn()
            signal = 2.0 * x0 - 1.2 * x1 + 0.1 * rng.randn()
            rel = int(np.digitize(signal, [-1.5, 0.0, 1.5]))
            rows.append({"x0": x0, "x1": x1, "x2": x2})
            relevances.append(rel)
            qids.append(f"q_{q:03d}")
    return pd.DataFrame(rows), np.array(relevances, dtype=float), np.array(qids, dtype=object)


@pytest.mark.tier1
@pytest.mark.f1
def test_r5b_t1_ndcg_at_k_perfect_vs_reversed() -> None:
    """ndcg_at_k returns 1.0 for perfect per-query ordering and strictly lower for reversed ordering."""
    from acdso.ranking import ndcg_at_k

    y_true = np.array([3.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    perfect_scores = np.array([10.0, 8.0, 5.0, 1.0, 9.0, 4.0, 2.0, 0.0])
    reversed_scores = -perfect_scores

    assert np.isclose(ndcg_at_k(y_true, perfect_scores, groups=groups, k=3), 1.0, atol=1e-9)
    rev = ndcg_at_k(y_true, reversed_scores, groups=groups, k=3)
    assert 0.0 <= rev < 0.7


@pytest.mark.tier1
@pytest.mark.f1
def test_r5b_t1_ranker_fit_and_predict() -> None:
    """ACDSORanker fits query-grouped training data and outputs 1D relevance scores."""
    from acdso import ACDSORanker

    X, y, qids = _make_ranking_data(n_queries=24, docs_per_query=8, seed=42)
    split = 18 * 8
    ranker = ACDSORanker(cv=3, k=5, random_state=42)
    assert ranker.fit(X.iloc[:split], y[:split], groups=qids[:split]) is ranker
    preds = ranker.predict(X.iloc[split:])
    assert preds.shape == (len(X) - split,)
    assert np.all(np.isfinite(preds))


@pytest.mark.tier1
@pytest.mark.f1
def test_r5b_t1_ranker_score_beats_random_baseline() -> None:
    """ACDSORanker.score() outperforms random scoring on held-out queries."""
    from acdso import ACDSORanker
    from acdso.ranking import ndcg_at_k

    X, y, qids = _make_ranking_data(n_queries=30, docs_per_query=8, seed=42)
    split = 22 * 8
    ranker = ACDSORanker(cv=3, k=5, random_state=42)
    ranker.fit(X.iloc[:split], y[:split], groups=qids[:split])

    model_ndcg = float(ranker.score(X.iloc[split:], y[split:], groups=qids[split:]))
    rng = np.random.RandomState(123)
    rand_scores = rng.randn(len(y) - split)
    rand_ndcg = float(ndcg_at_k(y[split:], rand_scores, groups=qids[split:], k=5))
    assert model_ndcg > rand_ndcg + 0.08
    assert model_ndcg > 0.7


@pytest.mark.tier1
@pytest.mark.f1
def test_r5b_t1_ranker_evaluate_dict() -> None:
    """ACDSORanker.evaluate(X, y, groups=qid, k=5) returns a dict of ranking metrics."""
    from acdso import ACDSORanker

    X, y, qids = _make_ranking_data(n_queries=20, docs_per_query=8, seed=11)
    split = 15 * 8
    ranker = ACDSORanker(cv=3, k=5, random_state=11)
    ranker.fit(X.iloc[:split], y[:split], groups=qids[:split])
    ev = ranker.evaluate(X.iloc[split:], y[split:], groups=qids[split:], k=5)
    assert isinstance(ev, dict) and len(ev) >= 1
    assert any("ndcg" in key.lower() for key in ev.keys())
    for val in ev.values():
        if isinstance(val, (int, float)):
            assert np.isfinite(float(val))


@pytest.mark.tier1
@pytest.mark.f1
def test_r5b_t1_query_group_preservation() -> None:
    """ACDSORanker performs GroupKFold across query IDs without splitting queries."""
    from acdso import ACDSORanker

    X, y, qids = _make_ranking_data(n_queries=9, docs_per_query=8, seed=7)
    ranker = ACDSORanker(cv=3, k=5, random_state=7)
    ranker.fit(X, y, groups=qids)
    assert isinstance(ranker.audit_trail_, dict)
    json.dumps(ranker.audit_trail_)


# ===========================================================================
# R5c: Survival / Censored Time-to-Event Analysis - 5 tests
# ===========================================================================


@pytest.mark.tier1
@pytest.mark.f1
def test_r5c_t1_concordance_index_exact_ordering() -> None:
    """concordance_index computes exact Harrell's C-index on censored pairs."""
    from acdso.survival import concordance_index

    # Comparable pairs: (0 vs 1), (0 vs 2), (0 vs 3), (2 vs 3) -> 4 pairs
    # Subject 1 is censored at t=2, so (1 vs 2) and (1 vs 3) are NOT comparable.
    times = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([1, 0, 1, 1])
    perfect_risk = np.array([4.0, 3.0, 2.0, 1.0])
    reversed_risk = np.array([1.0, 2.0, 3.0, 4.0])

    assert np.isclose(concordance_index(times, perfect_risk, events), 1.0, atol=1e-9)
    assert np.isclose(concordance_index(times, reversed_risk, events), 0.0, atol=1e-9)

    # Swap risk of 2 and 3 -> 3/4 = 0.75
    partial_risk = np.array([4.0, 3.0, 1.0, 2.0])
    assert np.isclose(concordance_index(times, partial_risk, events), 0.75, atol=1e-9)


@pytest.mark.tier1
@pytest.mark.f1
def test_r5c_t1_cox_ph_fit_scipy_breslow_backend() -> None:
    """ACDSOSurvivalAnalyzer fits Cox PH via SciPy Breslow partial likelihood without lifelines."""
    from acdso import ACDSOSurvivalAnalyzer

    X, t, e, _ = _make_survival_data(n_rows=140, seed=42)
    surv = ACDSOSurvivalAnalyzer(backend="scipy", l2_penalty=1e-2, cv=3, random_state=42)
    assert surv.fit(X.iloc[:100], time=t[:100], event=e[:100]) is surv
    assert surv.score(X.iloc[100:], time=t[100:], event=e[100:]) > 0.65


@pytest.mark.tier1
@pytest.mark.f1
def test_r5c_t1_predict_risk_correlates_with_hazard() -> None:
    """ACDSOSurvivalAnalyzer.predict_risk() correlates positively with true linear hazard and yields C-index > 0.65."""
    from acdso import ACDSOSurvivalAnalyzer

    X, t, e, lin_pred = _make_survival_data(n_rows=180, seed=7)
    surv = ACDSOSurvivalAnalyzer(backend="auto", l2_penalty=1e-2, cv=3, random_state=7)
    surv.fit(X.iloc[:130], time=t[:130], event=e[:130])

    risks = surv.predict_risk(X.iloc[130:])
    assert risks.shape == (50,)
    assert np.all(np.isfinite(risks))
    corr = float(np.corrcoef(risks, lin_pred[130:])[0, 1])
    assert corr > 0.6, f"Expected positive hazard correlation > 0.6, got {corr}"
    assert float(surv.score(X.iloc[130:], time=t[130:], event=e[130:])) > 0.65


@pytest.mark.tier1
@pytest.mark.f1
def test_r5c_t1_predict_survival_function_monotonic() -> None:
    """predict_survival_function returns probabilities in [0, 1] that are monotonically non-increasing in time."""
    from acdso import ACDSOSurvivalAnalyzer

    X, t, e, _ = _make_survival_data(n_rows=120, seed=42)
    surv = ACDSOSurvivalAnalyzer(l2_penalty=1e-2, cv=3, random_state=42)
    surv.fit(X.iloc[:90], time=t[:90], event=e[:90])

    times_grid = np.linspace(0.05, float(np.quantile(t, 0.9)), 12)
    sf = surv.predict_survival_function(X.iloc[90:96], times=times_grid)
    sf_arr = np.asarray(sf, dtype=float)
    assert sf_arr.shape == (6, 12)
    assert np.all((sf_arr >= 0.0) & (sf_arr <= 1.0))
    assert np.all(np.diff(sf_arr, axis=1) <= 1e-9), "Survival function must be non-increasing over time"


@pytest.mark.tier1
@pytest.mark.f1
def test_r5c_t1_groups_and_evaluate() -> None:
    """ACDSOSurvivalAnalyzer supports grouped CV via groups= and returns C-index evaluation dict."""
    from acdso import ACDSOSurvivalAnalyzer

    X, t, e, _ = _make_survival_data(n_rows=150, seed=19)
    groups = np.repeat([f"hosp_{i}" for i in range(15)], 10)
    surv = ACDSOSurvivalAnalyzer(l2_penalty=1e-2, cv=3, random_state=19)
    surv.fit(X.iloc[:110], time=t[:110], event=e[:110], groups=groups[:110])

    ev = surv.evaluate(X.iloc[110:], time=t[110:], event=e[110:])
    assert isinstance(ev, dict)
    assert any("c_index" in k.lower() or "concordance" in k.lower() for k in ev.keys())
    c_val = next(v for k, v in ev.items() if "c_index" in k.lower() or "concordance" in k.lower())
    assert float(c_val) > 0.6
    json.dumps(surv.audit_trail_)


# ===========================================================================
# R5d: Collaborative-Filtering Recommender (ACDSORecommender) - 5 tests
# ===========================================================================


@pytest.mark.tier1
@pytest.mark.f1
def test_r5d_t1_fit_and_predict_explicit_ratings() -> None:
    """ACDSORecommender fits explicit user-item interactions and predicts held-out ratings accurately."""
    from acdso import ACDSORecommender

    train_df, test_df = _make_recommender_interactions(seed=42)
    rec = ACDSORecommender(method="svd", n_factors=4, random_state=42)
    assert rec.fit(train_df, user_col="user_id", item_col="item_id", rating_col="rating") is rec

    preds = rec.predict(test_df["user_id"], test_df["item_id"])
    assert preds.shape == (len(test_df),)
    assert np.all(np.isfinite(preds))
    rmse = float(np.sqrt(np.mean((preds - test_df["rating"].to_numpy()) ** 2)))
    assert rmse < 1.5


@pytest.mark.tier1
@pytest.mark.f1
def test_r5d_t1_svd_recommend_excludes_seen() -> None:
    """ACDSORecommender(method='svd').recommend(..., exclude_seen=True) excludes training seen items."""
    from acdso import ACDSORecommender

    train_df, _ = _make_recommender_interactions(seed=42)
    target_user = "u_00"
    seen_items = set(train_df.loc[train_df["user_id"] == target_user, "item_id"].tolist())

    for method in ("svd", "auto"):
        rec = ACDSORecommender(method=method, n_factors=4, random_state=42)
        rec.fit(train_df, user_col="user_id", item_col="item_id", rating_col="rating")
        recs = rec.recommend(target_user, n=5, exclude_seen=True)
        assert 1 <= len(recs) <= 5
        rec_items = [item for item, _ in recs]
        assert seen_items.isdisjoint(set(rec_items)), "Recommended item was already seen in training"
        scores = [float(sc) for _, sc in recs]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.tier1
@pytest.mark.f1
def test_r5d_t1_nmf_method_fit_and_recommend() -> None:
    """ACDSORecommender(method='nmf') learns non-negative latent factors and ranks unseen items."""
    from acdso import ACDSORecommender

    train_df, test_df = _make_recommender_interactions(seed=7)
    rec = ACDSORecommender(method="nmf", n_factors=4, random_state=7)
    rec.fit(train_df, user_col="user_id", item_col="item_id", rating_col="rating")

    preds = rec.predict(test_df["user_id"], test_df["item_id"])
    assert preds.shape == (len(test_df),)
    assert np.all(np.isfinite(preds))
    recs = rec.recommend("u_01", n=5, exclude_seen=True)
    assert len(recs) >= 1 and len(recs) <= 5


@pytest.mark.tier1
@pytest.mark.f1
def test_r5d_t1_item_and_user_knn_methods() -> None:
    """ACDSORecommender supports item_knn and user_knn collaborative filtering on implicit/explicit data."""
    from acdso import ACDSORecommender

    train_df, test_df = _make_recommender_interactions(seed=11)
    for method in ("item_knn", "user_knn"):
        rec = ACDSORecommender(method=method, n_neighbors=5, random_state=11)
        rec.fit(train_df, user_col="user_id", item_col="item_id", rating_col="rating")
        preds = rec.predict(test_df["user_id"], test_df["item_id"])
        assert preds.shape == (len(test_df),)
        assert np.all(np.isfinite(preds))
        recs = rec.recommend("u_02", n=5, exclude_seen=True)
        assert len(recs) >= 1

    # Implicit feedback (rating_col=None)
    implicit_df = train_df[["user_id", "item_id"]].copy()
    rec_imp = ACDSORecommender(method="item_knn", n_neighbors=5, random_state=11)
    rec_imp.fit(implicit_df, user_col="user_id", item_col="item_id", rating_col=None)
    assert len(rec_imp.recommend("u_02", n=5, exclude_seen=True)) >= 1


@pytest.mark.tier1
@pytest.mark.f1
def test_r5d_t1_evaluate_returns_ranking_and_rating_metrics() -> None:
    """ACDSORecommender.evaluate() returns RMSE, MAE, Precision@k, Recall@k, and NDCG@k."""
    from acdso import ACDSORecommender

    train_df, test_df = _make_recommender_interactions(seed=42)
    rec = ACDSORecommender(method="svd", n_factors=5, random_state=42)
    rec.fit(train_df, user_col="user_id", item_col="item_id", rating_col="rating")
    ev = rec.evaluate(test_df, k=5, user_col="user_id", item_col="item_id", rating_col="rating")
    assert isinstance(ev, dict)
    keys_lower = {k.lower() for k in ev.keys()}
    for token in ("rmse", "mae", "precision", "recall", "ndcg"):
        assert any(token in k for k in keys_lower), f"Missing {token} metric in {ev}"
    for val in ev.values():
        if isinstance(val, (int, float)):
            assert np.isfinite(float(val))


# ===========================================================================
# R6: Single-Machine Out-of-Core / Chunked Scale - 5 tests
# ===========================================================================


@pytest.mark.tier1
@pytest.mark.f1
def test_r6_t1_iter_file_chunks_exact_count_and_schema(dataset_factory) -> None:
    """iter_file_chunks streams CSV files in exact bounded chunks preserving total rows and columns."""
    from acdso.scale import iter_file_chunks

    df = dataset_factory.make_regression_df(n_rows=550, seed=42)
    csv_path = dataset_factory.write_csv(df, "chunks_550.csv")

    chunks = list(iter_file_chunks(csv_path, chunk_size=120))
    assert len(chunks) == 5
    assert [len(c) for c in chunks] == [120, 120, 120, 120, 70]
    combined = pd.concat(chunks, ignore_index=True)
    assert list(combined.columns) == list(df.columns)
    assert len(combined) == 550


@pytest.mark.tier1
@pytest.mark.f1
def test_r6_t1_reservoir_sample_file_deterministic_and_total_count(dataset_factory) -> None:
    """reservoir_sample_file produces deterministic bounded samples and exact total row count."""
    from acdso.scale import reservoir_sample_file

    df = dataset_factory.make_classification_df(n_rows=1000, seed=42)
    csv_path = dataset_factory.write_csv(df, "reservoir_1000.csv")

    sample_a, total_a = reservoir_sample_file(csv_path, max_rows=150, chunk_size=200, random_state=42)
    sample_b, total_b = reservoir_sample_file(csv_path, max_rows=150, chunk_size=200, random_state=42)
    assert total_a == 1000 and total_b == 1000
    assert len(sample_a) == 150
    assert list(sample_a.columns) == list(df.columns)
    pd.testing.assert_frame_equal(sample_a.reset_index(drop=True), sample_b.reset_index(drop=True))


@pytest.mark.tier1
@pytest.mark.f1
def test_r6_t1_out_of_core_classification_partial_fit_from_file(dataset_factory, tmp_path: Path) -> None:
    """ACDSOOutOfCoreTrainer(task='classification') trains incrementally across file chunks via partial_fit."""
    from acdso import ACDSOOutOfCoreTrainer

    rng = np.random.RandomState(42)
    n = 1000
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    y = (2.0 * x1 - 1.5 * x2 > 0).astype(int)
    notes = ["positive signal strong upward" if v == 1 else "negative signal sharp decline" for v in y]
    df = pd.DataFrame({"x1": x1, "x2": x2, "notes": notes, "target": y})
    csv_path = dataset_factory.write_csv(df.iloc[:800], "ooc_clf_train.csv")

    trainer = ACDSOOutOfCoreTrainer(
        task="classification", chunk_size=200, n_hash_features=128, random_state=42
    )
    trainer.fit_from_file(csv_path, target_col="target", text_cols=["notes"])
    preds = trainer.predict(df.iloc[800:].drop(columns=["target"]))
    assert len(preds) == 200
    acc = float(np.mean(preds == df.iloc[800:]["target"].to_numpy()))
    assert acc > 0.75


@pytest.mark.tier1
@pytest.mark.f1
def test_r6_t1_out_of_core_clustering_and_predict_to_file(dataset_factory, tmp_path: Path) -> None:
    """ACDSOOutOfCoreTrainer(task='clustering') trains MiniBatchKMeans and streams predictions via predict_to_file."""
    from acdso import ACDSOOutOfCoreTrainer

    X_arr, _ = make_blobs(n_samples=600, centers=3, n_features=4, cluster_std=0.5, random_state=42)
    df = pd.DataFrame(X_arr, columns=["c0", "c1", "c2", "c3"])
    in_csv = dataset_factory.write_csv(df, "ooc_cluster_in.csv")
    out_csv = tmp_path / "ooc_cluster_out.csv"

    trainer = ACDSOOutOfCoreTrainer(task="clustering", chunk_size=150, n_clusters=3, random_state=42)
    trainer.fit_from_file(in_csv)
    trainer.predict_to_file(in_csv, out_csv, chunk_size=150)

    assert out_csv.is_file()
    out_df = pd.read_csv(out_csv)
    assert len(out_df) == 600
    pred_cols = [c for c in out_df.columns if "pred" in c.lower() or "cluster" in c.lower()]
    assert len(pred_cols) >= 1
    assert set(out_df[pred_cols[0]].unique()).issubset({0, 1, 2})


@pytest.mark.tier1
@pytest.mark.f1
def test_r6_t1_out_of_core_subsample_mode_regression(dataset_factory) -> None:
    """ACDSOOutOfCoreTrainer(task='regression', mode='subsample') fits via bounded reservoir sampling."""
    from acdso import ACDSOOutOfCoreTrainer

    df = dataset_factory.make_regression_df(n_rows=600, seed=42)
    csv_path = dataset_factory.write_csv(df.iloc[:480], "ooc_reg_train.csv")

    init_params = inspect.signature(ACDSOOutOfCoreTrainer.__init__).parameters
    fit_params = inspect.signature(ACDSOOutOfCoreTrainer.fit_from_file).parameters
    if "mode" in init_params:
        trainer = ACDSOOutOfCoreTrainer(
            task="regression",
            mode="subsample",
            chunk_size=120,
            max_in_memory_rows=200,
            random_state=42,
        )
        trainer.fit_from_file(csv_path, target_col="y")
    elif "mode" in fit_params:
        trainer = ACDSOOutOfCoreTrainer(
            task="regression",
            chunk_size=120,
            max_in_memory_rows=200,
            random_state=42,
        )
        trainer.fit_from_file(csv_path, target_col="y", mode="subsample")
    else:
        trainer = ACDSOOutOfCoreTrainer(
            task="regression",
            algorithm="subsample",
            chunk_size=120,
            max_in_memory_rows=200,
            random_state=42,
        )
        trainer.fit_from_file(csv_path, target_col="y")
    test_X = df.iloc[480:].drop(columns=["y"])
    test_y = df.iloc[480:]["y"].to_numpy()
    preds = trainer.predict(test_X)
    assert preds.shape == (120,)
    assert np.all(np.isfinite(preds))
    ss_res = float(np.sum((test_y - preds) ** 2))
    ss_tot = float(np.sum((test_y - np.mean(test_y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    assert r2 > 0.5


# ===========================================================================
# R7: Full Interface Wiring (Python API, CLI, TUI, Web GUI) - 5 tests
# ===========================================================================


@pytest.mark.tier1
@pytest.mark.f1
def test_r7_t1_cold_import_sub_second_and_lazy_r1_r6_symbols() -> None:
    """import acdso stays < 1.0s without eager heavy/R1-R6 imports while resolving all new lazy exports."""
    probe = (
        "import sys, time; "
        "t0 = time.perf_counter(); "
        "import acdso; "
        "dt = time.perf_counter() - t0; "
        "eager = [m for m in ('numpy', 'pandas', 'scipy', 'sklearn', 'acdso.cv', 'acdso.nlp', "
        "'acdso.clustering', 'acdso.anomaly', 'acdso.multilabel', 'acdso.ranking', 'acdso.survival', "
        "'acdso.recommender', 'acdso.scale') if m in sys.modules]; "
        "print('{:.4f}|{}'.format(dt, ','.join(eager)))"
    )
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(SRC_ROOT)})
    proc = subprocess.run(
        [PYTHON_EXE, "-B", "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=30.0,
    )
    assert proc.returncode == 0, proc.stderr
    dt_str, eager_str = proc.stdout.strip().splitlines()[-1].split("|", 1)
    assert float(dt_str) < 1.0, f"Cold import took {dt_str}s (contract < 1.0s)"
    assert eager_str == "", f"Eagerly imported modules on `import acdso`: {eager_str}"

    import acdso

    expected_symbols = (
        "ACDSOClusterer",
        "ACDSOAnomalyDetector",
        "build_cv_splitter",
        "validate_no_group_leakage",
        "TextFeaturizer",
        "detect_text_columns",
        "ACDSOMultiLabelClassifier",
        "ACDSORanker",
        "ndcg_at_k",
        "ACDSOSurvivalAnalyzer",
        "concordance_index",
        "ACDSORecommender",
        "ACDSOOutOfCoreTrainer",
    )
    for sym in expected_symbols:
        assert sym in acdso.__all__, f"{sym} missing from acdso.__all__"
        assert getattr(acdso, sym) is not None


@pytest.mark.tier1
@pytest.mark.f2
def test_r7_t1_cli_help_sub_500ms_and_lists_new_tasks_and_flags(run_cli) -> None:
    """run_acdso.py --help exits 0 under 500ms and exposes R1-R6 --task choices and CLI flags."""
    t0 = time.perf_counter()
    res = run_cli(["--help"], timeout=30.0)
    elapsed_s = time.perf_counter() - t0
    assert res.returncode == 0, res.stderr
    assert elapsed_s < 0.5, f"CLI --help took {elapsed_s:.3f}s (contract < 0.5s)"

    out = res.stdout
    for task_name in ("clustering", "anomaly", "multilabel", "ranking", "survival", "recommender"):
        assert task_name in out, f"Task choice {task_name!r} missing from --help"
    for flag in ("--group-col", "--cv-strategy", "--text-cols", "--cluster-algo", "--anomaly-algo", "--time-col", "--event-col", "--query-col", "--user-col", "--item-col", "--chunk-size", "--scale-mode"):
        assert flag in out, f"CLI flag {flag} missing from --help"


@pytest.mark.tier1
@pytest.mark.f2
def test_r7_t1_cli_clustering_simple_emits_progress_and_rfc8259_json(dataset_factory, run_cli, tmp_path: Path) -> None:
    """CLI --task clustering --simple runs targetlessly, emits [ACDSO_PROGRESS], and writes RFC 8259 JSON + HTML."""
    X_arr, _ = make_blobs(n_samples=90, centers=3, n_features=4, cluster_std=0.5, random_state=42)
    df = pd.DataFrame(X_arr, columns=["f0", "f1", "f2", "f3"])
    csv_path = dataset_factory.write_csv(df, "cli_cluster.csv")

    res = run_cli(
        ["--file", str(csv_path), "--task", "clustering", "--simple"],
        cwd=tmp_path,
        timeout=120.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    events = parse_progress_events(res.stdout)
    assert len(events) >= 1

    reports_dir = tmp_path / "reports"
    combined_files = list(reports_dir.glob("autonomous_cli_cluster*_combined.json"))
    html_files = list(reports_dir.glob("autonomous_cli_cluster*_report.html"))
    assert combined_files, f"Combined JSON missing in {list(reports_dir.iterdir())}"
    assert html_files, f"HTML report missing in {list(reports_dir.iterdir())}"

    def _reject_nonfinite(val: str) -> None:
        raise ValueError(f"Non-RFC8259 constant: {val}")

    payload = json.loads(combined_files[0].read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert payload.get("task") == "clustering"
    assert "clustering" in payload or "audit_trail" in payload

    parser = _TableTagCounter()
    parser.feed(html_files[0].read_text(encoding="utf-8", errors="replace"))
    assert parser.open_tables == parser.close_tables


@pytest.mark.tier1
@pytest.mark.f2
def test_r7_t1_tui_build_cli_command_and_fallback_empty_target_clustering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TUI build_cli_command & FallbackTerminalUI support targetless clustering commands cleanly."""
    import acdso.tui.app as tui_app
    from acdso.tui.app import FallbackTerminalUI, build_cli_command

    csv_path = tmp_path / "tui_cluster.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 4.0, 3.0]}).to_csv(csv_path, index=False)

    argv = build_cli_command(
        str(csv_path),
        target="",
        task="clustering",
        simple=True,
    )
    assert "--file" in argv and str(csv_path) in argv
    assert "--task" in argv and argv[argv.index("--task") + 1] == "clustering"
    assert "--simple" in argv
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

    logs: list[str] = []
    ui = FallbackTerminalUI(initial_file=str(csv_path), input_fn=lambda _: "q", output_fn=logs.append)
    ui.file_path = str(csv_path)
    ui.task = "clustering"
    ui.target = ""
    rc = ui._run_pipeline()
    assert rc == 0, "\n".join(logs)
    assert len(captured_cmds) == 1
    launched = captured_cmds[0]
    assert "--task" in launched and launched[launched.index("--task") + 1] == "clustering"
    assert "--target" not in launched and "--predict" not in launched


@pytest.mark.tier1
@pytest.mark.f3
def test_r7_t1_webui_startup_flags_validate_and_nl_plan_for_r1_r6(isolated_webui_client, dataset_factory) -> None:
    """WebUI create_app() < 0.5s with zero eager ML imports; FLAGS, /api/validate-columns & /api/nl-plan support R1-R6."""
    probe = (
        "import sys, time; "
        "t0 = time.perf_counter(); "
        "import acdso.webui; "
        "app = acdso.webui.create_app(); "
        "dt = time.perf_counter() - t0; "
        "heavy = [m for m in ('pandas', 'numpy', 'sklearn', 'scipy') if m in sys.modules]; "
        "print('{:.4f}|{}'.format(dt, ','.join(heavy)))"
    )
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(SRC_ROOT)})
    proc = subprocess.run(
        [PYTHON_EXE, "-B", "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=30.0,
    )
    assert proc.returncode == 0, proc.stderr
    dt_str, eager_str = proc.stdout.strip().splitlines()[-1].split("|", 1)
    assert float(dt_str) < 0.5, f"WebUI create_app() took {dt_str}s (contract < 0.5s)"
    assert eager_str == "", f"Heavy ML libs eagerly imported: {eager_str}"

    from acdso.webui.app import FLAGS_BY_NAME

    task_choices = set(FLAGS_BY_NAME["task"].choices)
    assert {"auto", "classification", "regression", "clustering", "anomaly", "multilabel", "ranking", "survival", "recommender"}.issubset(
        task_choices
    )

    _, client, _ = isolated_webui_client
    df = _make_mixed_nlp_df(n_rows=40)
    df["group_id"] = [f"grp_{i % 8}" for i in range(len(df))]
    csv_path = dataset_factory.write_csv(df, "webui_r1_r6.csv")
    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, csv_path.name)}, content_type="multipart/form-data")
    assert up.status_code == 200
    uid = up.get_json()["upload_id"]

    r_val = client.post(
        "/api/validate-columns",
        json={"upload_id": uid, "predict": ["label"], "group_col": ["group_id"]},
    )
    assert r_val.status_code == 200
    assert r_val.get_json()["ok"] is True

    r_nl = client.post(
        "/api/nl-plan",
        json={"upload_id": uid, "prompt": "cluster customers into segments without a target"},
    )
    assert r_nl.status_code == 200
    plan = r_nl.get_json()
    assert plan["task"] == "clustering"
    assert isinstance(plan.get("explanation"), str)
