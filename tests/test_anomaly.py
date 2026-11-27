"""Tests for Milestone M3: Trainable Anomaly Detection & AutoEDA Hook (R2)."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone

from acdso.anomaly import ACDSOAnomalyDetector
from acdso.eda import ACDSOAutoEDA


class TestAnomalyDetectors:
    def test_isolation_forest_and_iforest_alias(self):
        rng = np.random.RandomState(42)
        X_train = pd.DataFrame(rng.randn(160, 4), columns=["a", "b", "c", "d"])
        X_in = pd.DataFrame(rng.randn(20, 4), columns=["a", "b", "c", "d"])
        X_out = pd.DataFrame(rng.randn(10, 4) + 8.0, columns=["a", "b", "c", "d"])

        det = ACDSOAnomalyDetector(algorithm="isolation_forest", contamination=0.05, random_state=42)
        assert det.fit(X_train) is det
        assert det.best_algorithm_ == "isolation_forest"
        assert set(np.unique(det.predict(X_in))).issubset({-1, 1})
        assert np.mean(det.predict(X_out) == -1) >= 0.8
        assert det.predict_is_anomaly(X_out).dtype == bool
        assert float(det.anomaly_scores(X_out).mean()) > float(det.anomaly_scores(X_in).mean())
        assert np.allclose(det.anomaly_scores(X_in), -det.decision_function(X_in))

        det_alias = ACDSOAnomalyDetector(algorithm="iforest", contamination=0.05, random_state=42)
        det_alias.fit(X_train)
        assert det_alias.best_algorithm_ == "isolation_forest"
        assert np.array_equal(det_alias.predict(X_out), det.predict(X_out))

    def test_lof_novelty_and_small_n_clamping(self):
        rng = np.random.RandomState(17)
        X_train = pd.DataFrame(rng.randn(150, 3), columns=["x", "y", "z"])
        X_in = pd.DataFrame(rng.randn(20, 3), columns=["x", "y", "z"])
        X_out = pd.DataFrame(rng.randn(10, 3) + 10.0, columns=["x", "y", "z"])

        det = ACDSOAnomalyDetector(algorithm="lof", n_neighbors=15, contamination=0.05, random_state=17)
        det.fit(X_train)
        assert det.best_algorithm_ == "lof"
        assert np.all(det.predict(X_out) == -1)
        assert float(det.anomaly_scores(X_out).mean()) > float(det.anomaly_scores(X_in).mean())
        assert float(det.decision_function(X_in).mean()) > float(det.decision_function(X_out).mean())

        # n_neighbors > n_samples should clamp safely without error
        X_small = pd.DataFrame(rng.randn(5, 3), columns=["x", "y", "z"])
        det_small = ACDSOAnomalyDetector(algorithm="lof", n_neighbors=20, random_state=17).fit(X_small)
        assert det_small.predict(X_small).shape == (5,)

    def test_ocsvm_fit_predict_score_and_polarity(self):
        rng = np.random.RandomState(99)
        X_train = pd.DataFrame(rng.randn(140, 4), columns=["f1", "f2", "f3", "f4"])
        X_in = pd.DataFrame(rng.randn(20, 4) * 0.5, columns=["f1", "f2", "f3", "f4"])
        X_out = pd.DataFrame(rng.randn(10, 4) + 9.0, columns=["f1", "f2", "f3", "f4"])

        det = ACDSOAnomalyDetector(algorithm="ocsvm", nu=0.05, random_state=99)
        det.fit(X_train)
        assert det.best_algorithm_ == "ocsvm"
        flags_out = det.predict_is_anomaly(X_out)
        flags_in = det.predict_is_anomaly(X_in)
        assert flags_out.dtype == bool and flags_in.dtype == bool
        assert np.mean(flags_out) >= 0.8
        assert float(det.decision_function(X_in).mean()) > float(det.decision_function(X_out).mean())
        assert np.all(np.isfinite(det.score_samples(X_in)))
        assert np.isfinite(float(det.score(X_in)))

        # Supervised score evaluation with {1, -1} and boolean labels
        y_mix = np.concatenate([np.ones(len(X_in), dtype=int), -np.ones(len(X_out), dtype=int)])
        X_mix = pd.concat([X_in, X_out], ignore_index=True)
        acc = det.score(X_mix, y_mix)
        assert 0.5 <= acc <= 1.0

    def test_auto_selection_attributes_and_json_audit_trail(self):
        rng = np.random.RandomState(42)
        inliers = rng.randn(140, 3)
        outliers = rng.randn(10, 3) + 7.5
        X_df = pd.DataFrame(np.vstack([inliers, outliers]), columns=["m1", "m2", "m3"])

        det = ACDSOAnomalyDetector(algorithm="auto", contamination=0.07, random_state=42)
        det.fit(X_df)
        assert det.best_algorithm_ in {"isolation_forest", "lof", "ocsvm"}
        assert det.best_estimator_ is not None
        assert isinstance(det.offset_, float)
        assert isinstance(det.threshold_, float)
        assert math_isclose(det.threshold_, -det.offset_)
        assert len(det.candidates_) == 3
        json.dumps(det.audit_trail_, allow_nan=False)


def math_isclose(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-12


class TestAnomalyMixedInputsAndBoundaries:
    def test_mixed_text_numeric_categorical_nans_and_extreme_outliers(self):
        rng = np.random.RandomState(17)
        normal_phrases = [
            "request completed successfully status ok user session verified",
            "health check passed database connection pool nominal latency",
            "scheduled backup finished cleanly checksum verified no warnings",
        ]
        anomaly_phrases = [
            "CRITICAL kernel panic segmentation fault unauthorized root privilege escalation",
            "FATAL memory corruption buffer overflow exploit payload blocked emergency halt",
        ]

        train_df = pd.DataFrame(
            [
                {
                    "latency_ms": np.nan if i == 3 else float(45.0 + rng.normal(0.0, 6.0)),
                    "env": None if i == 4 else "prod",
                    "text": np.nan if i == 5 else normal_phrases[i % len(normal_phrases)],
                }
                for i in range(90)
            ]
        )
        det = ACDSOAnomalyDetector(
            algorithm="isolation_forest",
            contamination=0.05,
            text_cols=["text"],
            random_state=42,
        ).fit(train_df)

        unseen_norm = pd.DataFrame(
            [
                {
                    "latency_ms": float(46.0 + rng.normal(0.0, 5.0)),
                    "env": "prod",
                    "text": normal_phrases[i % len(normal_phrases)],
                }
                for i in range(15)
            ]
        )
        unseen_anom = pd.DataFrame(
            [
                {
                    "latency_ms": float(5000.0 + i * 250.0),
                    "env": "unknown_region",
                    "text": anomaly_phrases[i % len(anomaly_phrases)],
                }
                for i in range(10)
            ]
        )
        unseen = pd.concat([unseen_norm, unseen_anom], ignore_index=True)

        preds = det.predict(unseen)
        is_anom = det.predict_is_anomaly(unseen)
        anom_scores = det.anomaly_scores(unseen)
        dec_scores = det.decision_function(unseen)

        assert preds.shape == (25,)
        assert np.all(np.isfinite(anom_scores))
        assert is_anom[15:].mean() >= 0.80
        assert anom_scores[15:].mean() > anom_scores[:15].mean()
        assert dec_scores[:15].mean() > dec_scores[15:].mean()

        # Numpy ndarray input & extreme 1e6 outliers
        arr_train = rng.randn(100, 3)
        arr_in = rng.randn(10, 3) * 0.5
        arr_ext = np.full((5, 3), 1e6)
        det_arr = ACDSOAnomalyDetector(algorithm="isolation_forest", contamination=0.05, random_state=42).fit(arr_train)
        assert np.all(det_arr.predict(arr_ext) == -1)
        assert float(det_arr.anomaly_scores(arr_ext).min()) > float(det_arr.anomaly_scores(arr_in).max())

    def test_unfitted_validation_and_schema_errors(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        unfitted = ACDSOAnomalyDetector()
        for fn in (
            unfitted.predict,
            unfitted.predict_is_anomaly,
            unfitted.decision_function,
            unfitted.score_samples,
            unfitted.anomaly_scores,
            unfitted.score,
        ):
            with pytest.raises(RuntimeError):
                fn(X)

        X30 = pd.DataFrame({"a": np.linspace(0, 1, 30), "b": np.linspace(1, 2, 30)})
        for bad_kwargs in [
            {"algorithm": "unknown_detector"},
            {"contamination": 0.0},
            {"contamination": -0.1},
            {"contamination": 0.6},
            {"contamination": True},
            {"nu": 0.0},
            {"nu": 1.2},
            {"n_neighbors": 0},
        ]:
            with pytest.raises(ValueError):
                ACDSOAnomalyDetector(**bad_kwargs).fit(X30)

        # Empty, 1-row, all-NaN
        with pytest.raises(ValueError):
            ACDSOAnomalyDetector(algorithm="isolation_forest").fit(X30.iloc[:0])
        with pytest.raises(ValueError):
            ACDSOAnomalyDetector(algorithm="lof").fit(X30.iloc[:1])
        all_nan = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError):
            ACDSOAnomalyDetector(algorithm="isolation_forest").fit(all_nan)

        # Mismatched columns at predict
        fitted = ACDSOAnomalyDetector(algorithm="isolation_forest", random_state=42).fit(X30)
        with pytest.raises(ValueError):
            fitted.predict(X30[["a"]])

    def test_auto_eda_hook_and_backwards_compatibility(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "f1": rng.randn(100),
                "f2": rng.randn(100) * 2.0,
                "cat": ["x", "y"] * 50,
            }
        )
        df_copy = df.copy(deep=True)

        eda = ACDSOAutoEDA()
        assert eda.anomaly_detector_ is None
        with pytest.raises(RuntimeError):
            eda.score_anomalies(df)

        report = eda.analyze(df, dataset_name="anom_hook_test")
        assert report is not None
        assert eda.report_ is report
        pd.testing.assert_frame_equal(df, df_copy)

        det = eda.fit_anomaly_detector(df, algorithm="isolation_forest", contamination=0.05)
        assert isinstance(det, ACDSOAnomalyDetector)
        assert eda.anomaly_detector_ is det
        preds = det.predict(df.iloc[:15])
        assert preds.shape == (15,)
        assert set(np.unique(preds)).issubset({-1, 1})

        scores = eda.score_anomalies(df.iloc[:15])
        assert np.allclose(scores, det.anomaly_scores(df.iloc[:15]))

    def test_sklearn_clone_and_pickle_roundtrip(self):
        rng = np.random.RandomState(42)
        X_df = pd.DataFrame(rng.randn(80, 4), columns=["a", "b", "c", "d"])

        est = ACDSOAnomalyDetector(
            algorithm="auto",
            contamination=0.05,
            n_neighbors=10,
            nu=0.05,
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
        assert np.allclose(restored.anomaly_scores(X_df), est.anomaly_scores(X_df))

    def test_all_nan_object_column_dropped_and_inferred(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "num1": rng.randn(40),
                "num2": rng.randn(40),
                "empty_obj": pd.Series([None] * 40, dtype=object),
            }
        )
        det = ACDSOAnomalyDetector(
            algorithm="isolation_forest", contamination=0.05, random_state=42
        ).fit(df)
        assert det._active_cols_ == ["num1", "num2"]
        assert det.predict(df.iloc[:6]).shape == (6,)
        df_unseen = df.iloc[:6].copy()
        df_unseen["empty_obj"] = "seen_only_at_inference"
        assert det.predict(df_unseen).shape == (6,)

        all_nan = pd.DataFrame(
            {"a": [np.nan, np.nan, np.nan], "b": pd.Series([None, None, None], dtype=object)}
        )
        with pytest.raises(ValueError):
            ACDSOAnomalyDetector(algorithm="isolation_forest").fit(all_nan)

    def test_duplicate_columns_schema_refit_and_score_length_mismatch(self):
        rng = np.random.RandomState(19)
        X_uniq = pd.DataFrame({"a": rng.randn(30), "b": rng.randn(30)})
        det = ACDSOAnomalyDetector(algorithm="isolation_forest", random_state=19).fit(X_uniq)
        assert hasattr(det, "feature_names_in_")

        # Score length mismatch raises ValueError
        with pytest.raises(ValueError):
            det.score(X_uniq.iloc[:10], np.ones(5, dtype=int))

        # Duplicate columns at inference when fit had unique columns
        X_dup_inf = pd.DataFrame(
            np.column_stack([rng.randn(5), rng.randn(5), rng.randn(5)]),
            columns=["a", "b", "b"],
        )
        with pytest.raises(ValueError):
            det.predict(X_dup_inf)

        # Re-fit on ndarray deletes stale feature_names_in_
        det.fit(rng.randn(30, 3))
        assert not hasattr(det, "feature_names_in_")
        assert det.predict(rng.randn(4, 3)).shape == (4,)

        # Duplicate columns at fit -> enforce exact column sequence at inference
        X_dup = pd.DataFrame(
            np.column_stack([rng.randn(30), rng.randn(30), rng.randn(30)]),
            columns=["a", "b", "a"],
        )
        det_dup = ACDSOAnomalyDetector(algorithm="isolation_forest", random_state=19).fit(X_dup)
        assert det_dup.predict(X_dup.iloc[:5]).shape == (5,)
        X_dup_mismatch = pd.DataFrame(
            np.column_stack([rng.randn(5), rng.randn(5), rng.randn(5)]),
            columns=["a", "b", "c"],
        )
        with pytest.raises(ValueError):
            det_dup.predict(X_dup_mismatch)

        # Duplicate non-string columns with text_cols=[0]
        phrases = [
            "request completed successfully user session verified",
            "database connection pool nominal latency health check",
        ] * 15
        X_int_dup = pd.DataFrame(
            [[phrases[i], float(rng.randn()), phrases[i]] for i in range(30)],
            columns=[0, 1, 0],
        )
        det_int = ACDSOAnomalyDetector(
            algorithm="isolation_forest", text_cols=[0], random_state=19
        ).fit(X_int_dup)
        assert not hasattr(det_int, "feature_names_in_")
        assert det_int.predict(X_int_dup.iloc[:4]).shape == (4,)

    @pytest.mark.parametrize("algorithm", ["isolation_forest", "lof", "ocsvm", "auto"])
    def test_high_dim_text_and_low_dim_numeric_oov_all_algorithms(self, algorithm: str):
        rng = np.random.RandomState(17)
        normal_phrases = [
            "request completed successfully status ok user session verified",
            "health check passed database connection pool nominal latency",
            "scheduled backup finished cleanly checksum verified no warnings",
            "cache refresh completed items updated response time normal",
        ]
        anomaly_phrases = [
            "CRITICAL kernel panic segmentation fault unauthorized root privilege escalation",
            "FATAL memory corruption buffer overflow exploit payload blocked emergency halt",
            "ALERT brute force credential stuffing exfiltration dump detected firewall breach",
        ]
        train_df = pd.DataFrame(
            [
                {
                    "latency_ms": float(45.0 + rng.normal(0.0, 6.0)),
                    "error_count": int(rng.choice([0, 0, 0, 1])),
                    "text": normal_phrases[i % len(normal_phrases)],
                }
                for i in range(90)
            ]
        )
        unseen_norm = pd.DataFrame(
            [
                {
                    "latency_ms": float(46.0 + rng.normal(0.0, 5.0)),
                    "error_count": 0,
                    "text": normal_phrases[i % len(normal_phrases)],
                }
                for i in range(15)
            ]
        )
        unseen_anom = pd.DataFrame(
            [
                {
                    "latency_ms": float(4800.0 + i * 250.0),
                    "error_count": 85 + i * 5,
                    "text": anomaly_phrases[i % len(anomaly_phrases)],
                }
                for i in range(10)
            ]
        )
        unseen = pd.concat([unseen_norm, unseen_anom], ignore_index=True)

        det = ACDSOAnomalyDetector(
            algorithm=algorithm,
            contamination=0.05,
            text_cols=["text"],
            random_state=42,
        ).fit(train_df)

        preds = det.predict(unseen)
        is_anom = det.predict_is_anomaly(unseen)
        anom_scores = det.anomaly_scores(unseen)
        dec_scores = det.decision_function(unseen)

        assert preds.shape == (25,)
        assert set(np.unique(preds)).issubset({-1, 1})
        assert is_anom.dtype == bool
        assert float(is_anom[:15].mean()) <= 0.15
        assert float(is_anom[15:].mean()) >= 0.80
        assert float(anom_scores[15:].mean()) > float(anom_scores[:15].mean())
        assert float(dec_scores[:15].mean()) > float(dec_scores[15:].mean())
        json.dumps(det.audit_trail_, allow_nan=False)

        # Clone and pickle roundtrip with text compactor
        cloned = clone(det)
        assert cloned.get_params() == det.get_params()
        restored = pickle.loads(pickle.dumps(det))
        assert np.array_equal(restored.predict(unseen), preds)
        assert np.allclose(restored.anomaly_scores(unseen), anom_scores)

    @pytest.mark.parametrize("algorithm", ["ocsvm", "isolation_forest", "lof", "auto"])
    def test_dense_fit_sparse_predict_and_sparse_fit_dense_predict(self, algorithm: str):
        rng = np.random.RandomState(31)

        # 1) Dense fit (100 x 600 <= 1_000_000) -> Sparse predict (2500 x 600 > 1_000_000)
        n_fit_dense = 100
        n_pred_sparse = 2500
        fit_data = {f"num_{j}": rng.randn(n_fit_dense) for j in range(100)}
        pred_data = {
            f"num_{j}": np.concatenate([rng.randn(n_pred_sparse - 10), np.full(10, 50.0)])
            for j in range(100)
        }
        for c_idx in range(5):
            fit_data[f"high_cat_{c_idx}"] = [f"c{c_idx}_{i % 100}" for i in range(n_fit_dense)]
            pred_data[f"high_cat_{c_idx}"] = [f"c{c_idx}_{i % 100}" for i in range(n_pred_sparse)]
        df_dense_fit = pd.DataFrame(fit_data)
        df_sparse_pred = pd.DataFrame(pred_data)

        det_df = ACDSOAnomalyDetector(
            algorithm=algorithm, contamination=0.05, random_state=31
        ).fit(df_dense_fit)
        assert det_df.audit_trail_["n_features_preprocessed"] == 600
        assert det_df._fit_is_sparse_ is False
        assert n_pred_sparse * det_df.audit_trail_["n_features_model"] > 1_000_000
        assert sparse.issparse(det_df._transform_to_model_space(df_sparse_pred))
        preds_sp = det_df.predict(df_sparse_pred)
        dec_sp = det_df.decision_function(df_sparse_pred)
        samp_sp = det_df.score_samples(df_sparse_pred)
        anom_sp = det_df.anomaly_scores(df_sparse_pred)
        assert preds_sp.shape == (n_pred_sparse,)
        assert dec_sp.shape == (n_pred_sparse,)
        assert samp_sp.shape == (n_pred_sparse,)
        assert np.all(np.isfinite(dec_sp))
        assert np.all(np.isfinite(samp_sp))
        assert np.allclose(anom_sp, -dec_sp)
        assert np.mean(preds_sp[-10:] == -1) >= 0.80

        # 2) Sparse fit (1100 x 1002 > 1_000_000) -> Dense-sized predict (5 x 1002 <= 1_000_000)
        n_fit_sparse = 1100
        cats_1001 = [f"k_{i}" for i in range(1001)]
        df_sparse_fit = pd.DataFrame(
            {
                "num": rng.randn(n_fit_sparse),
                "high_cat": [cats_1001[i % 1001] for i in range(n_fit_sparse)],
            }
        )
        df_dense_pred = pd.DataFrame(
            {
                "num": [0.1, -0.2, 0.0, 0.3, 45.0],
                "high_cat": [cats_1001[i] for i in range(5)],
            }
        )
        det_sf = ACDSOAnomalyDetector(
            algorithm=algorithm, contamination=0.05, random_state=31
        ).fit(df_sparse_fit)
        assert det_sf._fit_is_sparse_ is True
        preds_dn = det_sf.predict(df_dense_pred)
        dec_dn = det_sf.decision_function(df_dense_pred)
        assert preds_dn.shape == (5,)
        assert set(np.unique(preds_dn)).issubset({-1, 1})
        assert np.all(np.isfinite(dec_dn))
        assert preds_dn[-1] == -1
        restored = pickle.loads(pickle.dumps(det_sf))
        assert np.array_equal(restored.predict(df_dense_pred), preds_dn)
