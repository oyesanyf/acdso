"""Tests for Milestone M3: Unsupervised Clustering & Segmentation (R1)."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from acdso.clustering import ACDSOClusterer


class TestClusteringAlgorithms:
    def test_kmeans_fit_predict_score_and_summary(self):
        X_arr, y_true = make_blobs(
            n_samples=120, centers=3, n_features=4, cluster_std=0.5, random_state=42
        )
        X_df = pd.DataFrame(X_arr, columns=["f0", "f1", "f2", "f3"])
        X_train, y_train = X_df.iloc[:90], y_true[:90]
        X_test, y_test = X_df.iloc[90:], y_true[90:]

        clusterer = ACDSOClusterer(algorithm="kmeans", n_clusters=3, random_state=42)
        assert clusterer.fit(X_train) is clusterer
        assert clusterer.best_algorithm_ == "kmeans"
        assert clusterer.n_clusters_ == 3
        assert len(clusterer.labels_) == len(X_train)
        assert adjusted_rand_score(y_train, clusterer.labels_) > 0.9

        preds = clusterer.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert adjusted_rand_score(y_test, preds) > 0.9
        assert np.array_equal(clusterer.fit_predict(X_train), clusterer.labels_)

        summary = clusterer.get_cluster_summary()
        assert isinstance(summary, dict)
        assert summary["n_clusters"] == 3
        assert summary["n_samples"] == len(X_train)
        assert sum(summary["cluster_sizes"].values()) == len(X_train)
        assert clusterer.silhouette_score_ > 0.5
        assert clusterer.davies_bouldin_score_ > 0.0
        assert np.isfinite(float(clusterer.score(X_test)))
        json.dumps(summary, allow_nan=False)

    def test_hdbscan_inductive_predict_and_all_noise_edge(self):
        X_arr, y_true = make_blobs(
            n_samples=120, centers=3, n_features=3, cluster_std=0.35, random_state=42
        )
        X_df = pd.DataFrame(X_arr, columns=["a", "b", "c"])
        X_train, X_test, y_test = X_df.iloc[:96], X_df.iloc[96:], y_true[96:]

        clusterer = ACDSOClusterer(algorithm="hdbscan", min_cluster_size=8, random_state=42)
        clusterer.fit(X_train)
        assert clusterer.best_algorithm_ == "hdbscan"
        assert clusterer.n_clusters_ >= 2
        assert len(clusterer.labels_) == len(X_train)

        unseen_labels = clusterer.predict(X_test)
        assert unseen_labels.shape == (len(X_test),)
        assert np.all(np.isfinite(unseen_labels))
        assert adjusted_rand_score(y_test, unseen_labels) > 0.85

        # All-noise edge input where min_cluster_size exceeds dense core sizes
        rng = np.random.RandomState(42)
        X_sparse = pd.DataFrame(
            rng.uniform(-100, 100, size=(30, 4)),
            columns=["f0", "f1", "f2", "f3"],
        )
        all_noise = ACDSOClusterer(algorithm="hdbscan", min_cluster_size=25, random_state=42)
        all_noise.fit(X_sparse)
        assert len(all_noise.labels_) == 30
        assert np.all(np.isfinite(all_noise.labels_))
        preds_noise = all_noise.predict(X_sparse.iloc[:5])
        assert preds_noise.shape == (5,)
        assert np.all(np.isfinite(preds_noise))
        assert np.isfinite(float(all_noise.score(X_sparse)))
        json.dumps(all_noise.audit_trail_, allow_nan=False)

    def test_gmm_fit_predict_score_and_high_dim_collinear_fallback(self):
        X_arr, y_true = make_blobs(
            n_samples=120, centers=3, n_features=4, cluster_std=0.6, random_state=7
        )
        X_df = pd.DataFrame(X_arr, columns=["x1", "x2", "x3", "x4"])
        X_train, X_test, y_test = X_df.iloc[:90], X_df.iloc[90:], y_true[90:]

        clusterer = ACDSOClusterer(algorithm="gmm", n_clusters=3, random_state=7)
        clusterer.fit(X_train)
        assert clusterer.best_algorithm_ == "gmm"
        assert clusterer.n_clusters_ == 3
        preds = clusterer.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert set(np.unique(preds)).issubset({0, 1, 2})
        assert adjusted_rand_score(y_test, preds) > 0.9
        assert np.isfinite(float(clusterer.score(X_test)))

        # High-dimensional collinear features (n_features > n_samples)
        rng = np.random.RandomState(3)
        base = rng.randn(12, 2)
        wide = np.hstack([base] * 10)
        gmm_wide = ACDSOClusterer(algorithm="gmm", n_clusters=2, random_state=3)
        gmm_wide.fit(wide)
        assert gmm_wide.labels_.shape == (12,)
        assert gmm_wide.predict(wide[:4]).shape == (4,)

    def test_auto_tuning_silhouette_and_davies_bouldin(self):
        X_arr, _ = make_blobs(
            n_samples=120, centers=3, n_features=4, cluster_std=0.45, random_state=11
        )
        X_df = pd.DataFrame(X_arr, columns=["c0", "c1", "c2", "c3"])

        c_sil = ACDSOClusterer(
            algorithm="auto", k_range=(2, 5), metric="silhouette", random_state=11
        )
        c_sil.fit(X_df)
        assert c_sil.best_algorithm_ in {"kmeans", "hdbscan", "gmm"}
        assert c_sil.best_estimator_ is not None
        assert c_sil.n_clusters_ >= 2
        assert c_sil.silhouette_score_ > 0.4
        assert c_sil.davies_bouldin_score_ > 0.0
        assert len(c_sil.candidates_) >= 3
        json.dumps(c_sil.audit_trail_, allow_nan=False)

        c_db = ACDSOClusterer(
            algorithm="auto", k_range=(2, 5), metric="davies_bouldin", random_state=11
        )
        c_db.fit(X_df)
        assert c_db.best_algorithm_ in {"kmeans", "hdbscan", "gmm"}
        assert np.isfinite(float(c_db.davies_bouldin_score_))
        assert float(c_db.score(X_df)) <= 0.0  # negative Davies-Bouldin (higher is better)
        json.dumps(c_db.audit_trail_, allow_nan=False)


class TestDimReductionAndMixedInputs:
    def test_dim_reduction_pca_and_umap_deterministic_fallback(self):
        X_arr, _ = make_blobs(
            n_samples=90, centers=3, n_features=8, cluster_std=0.6, random_state=42
        )
        X_df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(8)])

        c_pca = ACDSOClusterer(
            algorithm="kmeans", n_clusters=3, dim_reduction="pca", n_components=2, random_state=42
        )
        c_pca.fit(X_df)
        proj_pca = c_pca.transform(X_df)
        assert proj_pca.shape == (90, 2)
        assert np.all(np.isfinite(proj_pca))
        assert c_pca.audit_trail_["dim_reduction_backend"] == "pca"

        c_umap_1 = ACDSOClusterer(
            algorithm="kmeans", n_clusters=3, dim_reduction="umap", n_components=2, random_state=42
        ).fit(X_df)
        c_umap_2 = ACDSOClusterer(
            algorithm="kmeans", n_clusters=3, dim_reduction="umap", n_components=2, random_state=42
        ).fit(X_df)
        proj_1 = c_umap_1.transform(X_df)
        proj_2 = c_umap_2.transform(X_df)
        assert proj_1.shape == (90, 2)
        assert np.all(np.isfinite(proj_1))
        assert c_umap_1.audit_trail_["dim_reduction_backend"] in {"umap", "pca_fallback"}
        assert np.allclose(proj_1, proj_2)

        # Clamping when requested n_components > n_features
        c_clamp = ACDSOClusterer(
            algorithm="kmeans", n_clusters=2, dim_reduction="pca", n_components=50, random_state=42
        ).fit(X_df.iloc[:15, :4])
        assert c_clamp.transform(X_df.iloc[:5, :4]).shape == (5, 4)

    def test_mixed_numeric_categorical_text_nans_nullable_and_ndarray(self):
        rng = np.random.RandomState(42)
        topics = [
            (0, 1400.0, "tier_a", "ultrabook laptop solid state drive fast processor battery"),
            (1, 80.0, "tier_b", "running sneakers cushioned sole breathable mesh athletic shoe"),
            (2, 260.0, "tier_c", "stainless steel espresso coffee maker grinder kitchen appliance"),
        ]
        rows = []
        y_true = []
        for cid, price_mu, tier, text in topics:
            for i in range(25):
                rows.append(
                    {
                        "price": float(price_mu + rng.normal(0.0, price_mu * 0.06)),
                        "qty": pd.NA if i == 0 else int(cid * 5 + 1),
                        "tier": None if i == 1 else tier,
                        "created_at": pd.Timestamp("2024-01-01") + pd.Timedelta(days=cid * 15 + (i % 5)),
                        "description": np.nan if i == 2 else f"{text} item {i}",
                    }
                )
                y_true.append(cid)

        df = pd.DataFrame(rows)
        df["qty"] = df["qty"].astype("Int64")

        # Explicit text_cols
        c_exp = ACDSOClusterer(
            algorithm="kmeans",
            n_clusters=3,
            dim_reduction="pca",
            n_components=4,
            text_cols=["description"],
            text_method="tfidf",
            random_state=42,
        )
        labels = c_exp.fit_predict(df)
        assert labels.shape == (len(df),)
        assert adjusted_rand_score(y_true, labels) >= 0.85
        assert c_exp.predict(df.iloc[:6]).shape == (6,)

        # Auto text detection (text_cols=None)
        c_auto = ACDSOClusterer(
            algorithm="kmeans",
            n_clusters=3,
            dim_reduction="pca",
            n_components=4,
            text_cols=None,
            random_state=42,
        ).fit(df)
        assert "description" in c_auto.audit_trail_["text_cols"]
        assert c_auto.predict(df.iloc[:4]).shape == (4,)

        # Pure numpy ndarray input
        arr = rng.randn(40, 3)
        c_arr = ACDSOClusterer(algorithm="kmeans", n_clusters=2, random_state=42).fit(arr)
        assert c_arr.predict(arr[:5]).shape == (5,)
        assert c_arr.transform(arr[:5]).shape == (5, 3)


class TestClusteringBoundariesClonePickle:
    def test_unfitted_and_validation_errors(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
        unfitted = ACDSOClusterer()
        with pytest.raises(RuntimeError):
            unfitted.predict(X)
        with pytest.raises(RuntimeError):
            unfitted.transform(X)
        with pytest.raises(RuntimeError):
            unfitted.score(X)
        with pytest.raises(RuntimeError):
            unfitted.get_cluster_summary()

        # Empty & 1-row DataFrames
        with pytest.raises(ValueError):
            ACDSOClusterer(algorithm="kmeans", n_clusters=2).fit(X.iloc[:0])
        with pytest.raises(ValueError):
            ACDSOClusterer(algorithm="kmeans", n_clusters=2).fit(X.iloc[:1])

        # All-NaN DataFrame
        all_nan = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError):
            ACDSOClusterer(algorithm="kmeans", n_clusters=2).fit(all_nan)

        # Invalid hyperparameters
        X30 = pd.DataFrame({"a": np.linspace(0, 10, 30), "b": np.linspace(10, 0, 30)})
        for bad_kwargs in [
            {"algorithm": "bogus"},
            {"metric": "bogus"},
            {"dim_reduction": "bogus"},
            {"k_range": (5, 2)},
            {"k_range": (1, 3)},
            {"n_clusters": 1},
            {"min_cluster_size": 1},
            {"n_components": 0},
        ]:
            with pytest.raises(ValueError):
                ACDSOClusterer(**bad_kwargs).fit(X30)

        # Mismatched columns at predict/transform
        fitted = ACDSOClusterer(algorithm="kmeans", n_clusters=2, random_state=42).fit(X30)
        with pytest.raises(ValueError):
            fitted.predict(X30[["a"]])
        with pytest.raises(ValueError):
            fitted.transform(X30[["a"]])

    def test_constant_rows_graceful(self):
        X_const = pd.DataFrame({"a": np.ones(30), "b": np.full(30, 3.5), "c": np.zeros(30)})
        clusterer = ACDSOClusterer(algorithm="auto", k_range=(2, 4), random_state=42)
        clusterer.fit(X_const)
        assert clusterer.labels_.shape == (30,)
        assert np.all(np.isfinite(clusterer.labels_))
        assert clusterer.predict(X_const.iloc[:5]).shape == (5,)
        json.dumps(clusterer.audit_trail_, allow_nan=False)

    def test_sklearn_clone_and_pickle_roundtrip(self):
        X_arr, _ = make_blobs(n_samples=60, centers=3, n_features=4, random_state=42)
        X_df = pd.DataFrame(X_arr, columns=["f0", "f1", "f2", "f3"])

        est = ACDSOClusterer(
            algorithm="auto",
            k_range=(2, 4),
            dim_reduction="pca",
            n_components=2,
            text_cols=None,
            random_state=42,
        )
        cloned_pre = clone(est)
        assert cloned_pre.get_params() == est.get_params()

        est.fit(X_df)
        cloned_post = clone(est)
        assert cloned_post.get_params() == est.get_params()
        assert not hasattr(cloned_post, "best_estimator_")

        restored = pickle.loads(pickle.dumps(est))
        assert np.array_equal(restored.predict(X_df), est.predict(X_df))
        assert np.allclose(restored.transform(X_df), est.transform(X_df))
        assert restored.get_cluster_summary() == est.get_cluster_summary()

    def test_all_nan_object_column_dropped_and_inferred(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "num1": np.concatenate([rng.randn(20), rng.randn(20) + 5.0]),
                "num2": np.concatenate([rng.randn(20), rng.randn(20) + 5.0]),
                "empty_obj": pd.Series([None] * 40, dtype=object),
            }
        )
        c = ACDSOClusterer(algorithm="kmeans", n_clusters=2, random_state=42).fit(df)
        assert c.labels_.shape == (40,)
        assert c._active_cols_ == ["num1", "num2"]
        preds = c.predict(df.iloc[:6])
        assert preds.shape == (6,)
        # Even if unseen rows provide values in the previously all-NaN column, transform/predict works
        df_unseen = df.iloc[:6].copy()
        df_unseen["empty_obj"] = "unexpected_val"
        assert c.predict(df_unseen).shape == (6,)

        # All-NaN object+numeric DataFrame still raises ValueError
        all_nan = pd.DataFrame(
            {"a": [np.nan, np.nan, np.nan], "b": pd.Series([None, None, None], dtype=object)}
        )
        with pytest.raises(ValueError):
            ACDSOClusterer(algorithm="kmeans", n_clusters=2).fit(all_nan)

    def test_duplicate_columns_schema_and_refit_feature_names(self):
        rng = np.random.RandomState(7)
        # 1. Unique columns at fit -> reject duplicate columns at inference
        X_uniq = pd.DataFrame({"a": rng.randn(20), "b": rng.randn(20)})
        c_uniq = ACDSOClusterer(algorithm="kmeans", n_clusters=2, random_state=7).fit(X_uniq)
        assert hasattr(c_uniq, "feature_names_in_")
        X_dup_inf = pd.DataFrame(
            np.column_stack([rng.randn(5), rng.randn(5), rng.randn(5)]),
            columns=["a", "b", "b"],
        )
        with pytest.raises(ValueError):
            c_uniq.predict(X_dup_inf)

        # 2. Re-fit on ndarray clears stale feature_names_in_
        c_uniq.fit(rng.randn(20, 3))
        assert not hasattr(c_uniq, "feature_names_in_")
        assert c_uniq.predict(rng.randn(4, 3)).shape == (4,)

        # 3. Duplicate columns at fit -> enforce exact column sequence at inference
        X_dup = pd.DataFrame(
            np.column_stack([rng.randn(24), rng.randn(24), rng.randn(24)]),
            columns=["a", "b", "a"],
        )
        c_dup = ACDSOClusterer(algorithm="kmeans", n_clusters=2, random_state=7).fit(X_dup)
        assert c_dup.predict(X_dup.iloc[:5]).shape == (5,)
        X_dup_mismatch = pd.DataFrame(
            np.column_stack([rng.randn(5), rng.randn(5), rng.randn(5)]),
            columns=["a", "b", "c"],
        )
        with pytest.raises(ValueError):
            c_dup.predict(X_dup_mismatch)

        # 4. Duplicate non-string columns with text_cols=[0]
        texts = [
            "machine learning clustering evaluation pipeline",
            "database transaction log processing latency",
        ] * 12
        X_int_dup = pd.DataFrame(
            [[texts[i], float(i % 2), texts[i]] for i in range(24)],
            columns=[0, 1, 0],
        )
        c_int = ACDSOClusterer(
            algorithm="kmeans", n_clusters=2, text_cols=[0], random_state=7
        ).fit(X_int_dup)
        assert not hasattr(c_int, "feature_names_in_")
        assert c_int.predict(X_int_dup.iloc[:4]).shape == (4,)

    def test_pca_float_n_components_on_constant_data(self):
        X_const = pd.DataFrame({"a": np.ones(20), "b": np.full(20, 2.5), "c": np.zeros(20)})
        for dr in ("pca", "umap"):
            c = ACDSOClusterer(
                algorithm="kmeans",
                n_clusters=2,
                dim_reduction=dr,
                n_components=0.95,
                random_state=42,
            ).fit(X_const)
            proj = c.transform(X_const.iloc[:5])
            assert proj.shape == (5, 1)
            assert np.all(np.isfinite(proj))
            assert c.predict(X_const.iloc[:5]).shape == (5,)
            json.dumps(c.audit_trail_, allow_nan=False)

    def test_skewed_multiscale_numeric_with_noisy_text_and_svd_dense_inference(self):
        rng = np.random.RandomState(99)
        # Domain-agnostic multi-order-of-magnitude right-skewed positive metrics
        # paired with multi-template free-text containing cross-cluster shared tokens.
        groups = [
            (
                0,
                25.0,
                0.35,
                [
                    "alpha telemetry sensor module compact low power probe",
                    "delta telemetry housing bracket lightweight polymer mount",
                    "omega telemetry indicator gauge analog dial readout",
                ],
            ),
            (
                1,
                190.0,
                3.2,
                [
                    "hydraulic actuator valve manifold brass pressure regulator",
                    "pneumatic actuator cylinder stroke control assembly",
                    "industrial actuator relay controller din rail switch",
                ],
            ),
            (
                2,
                1850.0,
                18.0,
                [
                    "turbomolecular vacuum pump high throughput flange unit",
                    "centrifugal vacuum chiller compressor heavy duty condenser",
                    "servo vacuum spindle drive inverter three phase cabinet",
                ],
            ),
        ]
        rows = []
        y_true = []
        for cid, m1, m2, templates in groups:
            for i in range(30):
                rows.append(
                    {
                        "metric_a": float(max(0.5, m1 + rng.normal(0.0, m1 * 0.08))),
                        "metric_b": float(max(0.05, m2 + rng.normal(0.0, m2 * 0.10))),
                        "notes": f"{templates[i % len(templates)]} lot ref {i}",
                    }
                )
                y_true.append(cid)

        df = pd.DataFrame(rows)
        for algo, n_cl in [("kmeans", 3), ("gmm", 3), ("auto", None)]:
            c = ACDSOClusterer(
                algorithm=algo,
                n_clusters=n_cl,
                k_range=(2, 5),
                dim_reduction="pca",
                n_components=4,
                metric="silhouette",
                min_cluster_size=15,
                text_cols=["notes"],
                text_method="tfidf",
                random_state=99,
            )
            labels = c.fit_predict(df)
            assert labels.shape == (len(df),)
            assert c.n_clusters_ >= 2
            assert c.silhouette_score_ > 0.15
            assert adjusted_rand_score(y_true, labels) >= 0.80

            # Consistent inference on unseen samples + pickle roundtrip
            X_unseen = pd.DataFrame(
                [
                    {"metric_a": 24.0, "metric_b": 0.34, "notes": "alpha sensor module compact low power telemetry"},
                    {"metric_a": 195.0, "metric_b": 3.1, "notes": "hydraulic valve manifold brass pressure regulator"},
                    {"metric_a": 1820.0, "metric_b": 17.5, "notes": "turbomolecular vacuum pump high throughput flange"},
                ]
            )
            preds = c.predict(X_unseen)
            assert preds.shape == (3,)
            assert len(set(int(p) for p in preds)) == 3
            restored = pickle.loads(pickle.dumps(c))
            assert np.array_equal(restored.predict(X_unseen), preds)

        # Verify _to_dense_finite still applies _sparse_svd_ when inference batch is dense
        from scipy import sparse
        from sklearn.decomposition import TruncatedSVD

        c_svd = ACDSOClusterer(algorithm="kmeans", n_clusters=2, random_state=42)
        c_svd._sparse_svd_ = TruncatedSVD(n_components=2, random_state=42).fit(
            sparse.csr_matrix(rng.randn(10, 6))
        )
        dense_inf = c_svd._to_dense_finite(rng.randn(3, 6), fit_svd=False)
        assert dense_inf.shape == (3, 2)
        assert np.all(np.isfinite(dense_inf))
