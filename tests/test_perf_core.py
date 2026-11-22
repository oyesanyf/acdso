"""
Performance, Quality & Regression Tests for ACDSO M1 Improvements
==================================================================

Covers Features #1-#5:
1. Lazy top-level imports (`acdso/__init__.py`)
2. Vectorized VIF & proxy leakage detection (`acdso/risk.py`)
3. Leak-free fold caching, 100% final refit, DataFrame stress tests & fallback (`acdso/core.py`)
4. Thread-safe memory tracking & log1p-aligned knee-point selection (`acdso/core.py`, `acdso/pareto.py`)
5. Vectorized preprocessing, parallel DataFrame-safe feature importance, zero-copy EDA,
   and deep timeseries model memory tracking.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from acdso import (
    ACDSOAutoEDA,
    ACDSOClassifier,
    ACDSORegressor,
    AutoPreprocessor,
    OutlierClipper,
    SafeFeatureImportance,
    SkewnessTransformer,
    TargetEncoder,
)
from acdso.core import _ACDSOEngineV3, run_stress_tests
from acdso.pareto import CandidateResult, knee_point_selection
from acdso.risk import RiskReport, _vif_numeric, analyze_risk
from acdso.timeseries import AutoTS


class TestLazyImports:
    """Feature #1: Lazy top-level imports in acdso/__init__.py."""

    def test_cold_import_defers_heavy_modules(self):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        code = (
            "import sys, time; "
            "t0 = time.perf_counter(); "
            "import acdso; "
            "dt = time.perf_counter() - t0; "
            "deferred = ['acdso.eda', 'acdso.feature_importance', 'acdso.auto_preprocessing', "
            "'acdso.advisors', 'acdso.timeseries', 'acdso.decision', 'acdso.integrations', "
            "'sklearn', 'pandas', 'scipy']; "
            "loaded = [m for m in deferred if m in sys.modules]; "
            "assert not loaded, f'Eagerly loaded: {loaded}'; "
            "assert dt < 1.0, f'Cold import too slow: {dt:.3f}s'; "
            "print(f'OK {dt:.4f}')"
        )
        proc = subprocess.run(
            [sys.executable, "-B", "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"

    def test_all_exports_and_dir_preserved(self):
        import acdso

        expected = {
            "ACDSOClassifier",
            "ACDSORegressor",
            "ACDSOAutoEDA",
            "RiskReport",
            "analyze_risk",
            "SafeFeatureImportance",
            "AutoPreprocessor",
            "TargetEncoder",
            "StubLLMAdvisor",
            "timeseries",
            "decision",
            "integrations",
        }
        assert expected.issubset(set(acdso.__all__))
        assert set(acdso.__all__).issubset(set(dir(acdso)))
        for name in ["ACDSOAutoEDA", "SafeFeatureImportance", "AutoPreprocessor", "StubLLMAdvisor", "LLMLabelAnalyzer"]:
            assert getattr(acdso, name) is not None


class TestVectorizedVIF:
    """Feature #2: Vectorized correlation-matrix VIF parity & edge cases."""

    def test_vif_matches_ols_and_flags_rank_deficient(self):
        rng = np.random.RandomState(42)
        n = 250
        x1 = rng.randn(n)
        x2 = x1 + 0.05 * rng.randn(n)
        x3 = rng.randn(n)
        df_collinear = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        # Reference OLS VIF
        ols_vifs = {}
        for col in df_collinear.columns:
            lr = LinearRegression().fit(df_collinear.drop(columns=[col]).values, df_collinear[col].values)
            r2 = float(np.clip(lr.score(df_collinear.drop(columns=[col]).values, df_collinear[col].values), 0.0, 0.999999))
            ols_vifs[col] = 1.0 / (1.0 - r2)

        vif_max, high_vif = _vif_numeric(df_collinear)
        assert abs(vif_max - max(ols_vifs.values())) / max(ols_vifs.values()) < 0.05
        assert set(high_vif) == {"x1", "x2"}

        # Exact rank-deficient duplicate + constant + NaN/Inf
        df_dup = pd.DataFrame({
            "a": x1,
            "b": 2.0 * x1,
            "c": x3,
            "const": np.full(n, 7.0),
        })
        df_dup.loc[3, "c"] = np.nan
        df_dup.loc[4, "c"] = np.inf
        dup_max, dup_high = _vif_numeric(df_dup)
        assert dup_max >= 10.0
        assert {"a", "b"}.issubset(set(dup_high))
        assert "const" not in dup_high

        # Single column
        single_max, single_high = _vif_numeric(pd.DataFrame({"only": x1}))
        assert single_max == 1.0
        assert single_high == []


class TestProxyLeakageDetection:
    """Feature #2: Proxy leakage detection across classification & regression."""

    def test_synthetic_proxy_leakage_matches_benchmark_case(self):
        rng = np.random.RandomState(42)
        n = 500
        X, y = make_classification(
            n_samples=n,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
        df["leaky_target"] = y + rng.randn(n) * 0.01

        report = analyze_risk(df, y, task="classification", random_state=42)
        assert "leaky_target" in report.leakage_proxy_features
        assert report.leakage_proxy_score_max >= 0.95
        assert len(report.leakage_proxy_features) == 1

    def test_categorical_and_regression_and_clean_cases(self):
        rng = np.random.RandomState(42)
        n = 250
        y_clf = rng.randint(0, 2, n)
        df_cat = pd.DataFrame({
            "cat_proxy": np.where(y_clf == 1, "pos", "neg"),
            "uid": [f"id_{i}" for i in range(n)],
            "rand": rng.randn(n),
        })
        rep_cat = analyze_risk(df_cat, y_clf, task="classification", random_state=42)
        assert "cat_proxy" in rep_cat.leakage_proxy_features
        assert "uid" not in rep_cat.leakage_proxy_features

        X_reg, y_reg = make_regression(n_samples=n, n_features=5, noise=5.0, random_state=42)
        df_reg = pd.DataFrame(X_reg, columns=[f"r{i}" for i in range(5)])
        df_reg["leaky_reg"] = y_reg + rng.randn(n) * 0.01
        rep_reg = analyze_risk(df_reg, y_reg, task="regression", scorer="r2", random_state=42)
        assert "leaky_reg" in rep_reg.leakage_proxy_features
        assert rep_reg.leakage_proxy_score_max >= 0.95

        df_clean = pd.DataFrame(rng.randn(n, 5), columns=[f"n{i}" for i in range(5)])
        rep_clean = analyze_risk(df_clean, rng.randint(0, 2, n), task="classification", random_state=42)
        assert rep_clean.leakage_proxy_features == []


class TestFinalRefitAndStress:
    """Features #3 & #4: 100% refit, DataFrame stress tests, and _build_preprocessor fallback."""

    def test_best_estimator_refitted_on_full_training_data(self):
        X, y = make_classification(
            n_samples=240,
            n_features=8,
            n_informative=5,
            random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        X_train, X_test, y_train, _ = train_test_split(df, y, test_size=0.25, random_state=42)

        clf = ACDSOClassifier(cv=3, max_candidates=4, random_state=42, verbose=0)
        clf.fit(X_train, y_train)

        assert clf.audit_trail_["selected"]["refitted_full"] is True
        refit_clone = clone(clf.best_estimator_)
        refit_clone.fit(X_train, y_train)
        np.testing.assert_allclose(clf.predict(X_test), refit_clone.predict(X_test))

    def test_run_stress_tests_preserves_mixed_dataframe(self):
        rng = np.random.RandomState(42)
        n = 160
        df = pd.DataFrame({
            "num_f": rng.randn(n),
            "int_f": rng.randint(0, 20, size=n),
            "bool_f": rng.randint(0, 2, size=n).astype(bool),
            "cat_f": rng.choice(["A", "B", "C"], size=n),
        })
        y = (df["num_f"] + 0.1 * df["int_f"] > 0).astype(int).to_numpy()

        clf = ACDSOClassifier(cv=3, max_candidates=3, random_state=42, verbose=0)
        clf.fit(df, y)

        _, tests, worst_drop = run_stress_tests(
            clf.best_estimator_, df, y, task="classification", scorer_name="accuracy", random_state=42
        )
        names = {t.name for t in tests}
        assert {"missingness_2pct", "feature_jitter_2pct", "label_noise_2pct"}.issubset(names)
        assert worst_drop >= 0.0

    def test_fallback_build_preprocessor_when_all_pruned(self):
        X, y = make_classification(n_samples=80, n_features=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"c{i}" for i in range(5)])
        clf = ACDSOClassifier(
            cv=2,
            max_candidates=2,
            max_train_time_s=1e-9,
            random_state=42,
            verbose=0,
        )
        clf.fit(df, y)
        assert clf.audit_trail_["selected"]["name"] == "Baseline_Fallback"
        assert clf.predict(df).shape == (80,)

    def test_predict_works_on_dataframe_with_categorical_and_missing(self):
        rng = np.random.RandomState(42)
        n = 150
        df = pd.DataFrame({
            "x1": rng.randn(n),
            "x2": rng.randn(n),
            "cat": rng.choice(["red", "blue", "green"], size=n),
        })
        df.loc[0:5, "x1"] = np.nan
        y = (df["x1"].fillna(0) > 0).astype(int).to_numpy()

        clf = ACDSOClassifier(cv=3, max_candidates=3, random_state=42, verbose=0)
        clf.fit(df, y)
        preds = clf.predict(df)
        assert preds.shape == (n,)


class TestMemoryAndPareto:
    """Feature #5 (core/pareto): non-zero peak_mem_mb and log1p knee selection."""

    def test_peak_mem_mb_positive_for_evaluated_candidates(self):
        X, y = make_classification(n_samples=150, n_features=10, random_state=42)
        clf = ACDSOClassifier(cv=3, max_candidates=4, random_state=42, verbose=0)
        clf.fit(X, y)

        cands = clf.audit_trail_["candidates_evaluated"]
        assert len(cands) > 0
        assert all(c["peak_mem_mb"] > 0.0 for c in cands)

    def test_knee_point_selection_log1p_resists_outlier(self):
        def make_cand(name: str, metric: float, t: float, mem: float, lat: float) -> CandidateResult:
            return CandidateResult(
                name=name,
                pipeline=None,
                metric=metric,
                train_time_s=t,
                peak_mem_mb=mem,
                latency_ms_per_row=lat,
                risk_penalty=0.05,
                objectives=(metric, math.log1p(t), math.log1p(mem), math.log1p(lat), 0.05, 0.0),
            )

        outlier = make_cand("Outlier", metric=0.985, t=200.0, mem=1000.0, lat=10.0)
        balanced = make_cand("Balanced", metric=0.980, t=2.0, mem=25.0, lat=0.2)
        fast = make_cand("Fast", metric=0.820, t=0.5, mem=10.0, lat=0.1)

        selected = knee_point_selection([outlier, balanced, fast])
        assert selected.name == "Balanced"

        # Non-finite latency guard
        inf_cand = make_cand("InfLat", metric=0.90, t=1.0, mem=10.0, lat=float("inf"))
        assert knee_point_selection([balanced, inf_cand]).name in {"Balanced", "InfLat"}

    def test_knee_point_selection_nan_candidate_never_beats_optimal_finite(self):
        from acdso.pareto import dominates, pareto_front

        def make_cand(name: str, metric: float, t: float, mem: float, lat: float, risk: float) -> CandidateResult:
            return CandidateResult(
                name=name,
                pipeline=None,
                metric=metric,
                train_time_s=t,
                peak_mem_mb=mem,
                latency_ms_per_row=lat,
                risk_penalty=risk,
                objectives=(metric, math.log1p(t), math.log1p(mem), math.log1p(lat), risk, 0.0),
            )

        nan_cand = make_cand("NaNCand", metric=float("nan"), t=1.0, mem=10.0, lat=0.1, risk=0.05)
        opt_cand = make_cand("Optimal", metric=0.95, t=0.5, mem=5.0, lat=0.05, risk=0.01)

        # Optimal dominates NaNCand on metric and costs; NaNCand never dominates Optimal
        assert dominates(opt_cand, nan_cand)
        assert not dominates(nan_cand, opt_cand)

        # Even if both are passed directly to knee_point_selection (a == b case), Optimal wins
        assert knee_point_selection([nan_cand, opt_cand]).name == "Optimal"
        assert pareto_front([nan_cand, opt_cand])[0].name == "Optimal"


class TestAutoPreprocessingVectorized:
    """Feature #5: Vectorized TargetEncoder, SkewnessTransformer, OutlierClipper, AutoPreprocessor."""

    def test_target_encoder_parity_and_types(self):
        df = pd.DataFrame({
            "cat_str": ["a", "a", "b", "b", "c", None],
            "cat_dtype": pd.Series(["x", "x", "y", "y", "z", "z"], dtype="category"),
        })
        y = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        gmean = float(np.mean(y))

        te = TargetEncoder(smoothing=10.0, min_samples=1)
        out = te.fit_transform(df, y)
        assert out.shape == (6, 2)
        assert np.all(np.isfinite(out))

        # Manual formula check for 'a' (count=2, mean=0.5)
        expected_a = (2 * 0.5 + 10.0 * gmean) / (2 + 10.0)
        assert np.isclose(out[0, 0], expected_a)
        # Missing / unseen falls back to global_mean_
        out_unseen = te.transform(pd.DataFrame({"cat_str": ["unseen"], "cat_dtype": ["unseen"]}))
        assert np.allclose(out_unseen, gmean)

        # NumPy ndarray support + sklearn clone
        arr = np.array([["a"], ["a"], ["b"], ["b"]], dtype=object)
        y_arr = np.array([1.0, 1.0, 0.0, 0.0])
        te2 = clone(te).fit(arr, y_arr)
        out_arr = te2.transform(arr)
        assert out_arr.shape == (4, 1)
        assert out_arr[0, 0] > out_arr[2, 0]

    def test_skewness_transformer_and_outlier_clipper(self):
        rng = np.random.RandomState(42)
        skewed = np.exp(rng.randn(200) * 1.5)
        neg_skewed = skewed - 50.0
        normal = rng.randn(200)
        const = np.ones(200)
        X = np.column_stack([skewed, neg_skewed, normal, const])
        X[0, 0] = np.nan

        st = SkewnessTransformer(threshold=1.0, method="auto")
        Xt = st.fit_transform(X)
        assert Xt.shape == X.shape
        assert np.isnan(Xt[0, 0])
        assert 0 in st.transforms_ and 1 in st.transforms_
        assert 2 not in st.transforms_ and 3 not in st.transforms_

        oc = OutlierClipper(lower_percentile=5, upper_percentile=95)
        Xo = oc.fit_transform(X)
        low0, high0 = np.nanpercentile(X[:, 0], [5, 95])
        assert np.nanmin(Xo[:, 0]) >= low0 - 1e-12
        assert np.nanmax(Xo[:, 0]) <= high0 + 1e-12

    def test_auto_preprocessor_encodes_high_cardinality(self):
        rng = np.random.RandomState(42)
        n = 120
        df = pd.DataFrame({
            "num": rng.randn(n),
            "skew": np.exp(rng.randn(n)),
            "low_cat": rng.choice(["A", "B", "C"], size=n),
            "zip_code": [f"Z_{i % 25}" for i in range(n)],
        })
        y = (df["num"] > 0).astype(int).to_numpy()

        ap = AutoPreprocessor(high_cardinality_threshold=10, verbose=0)
        out = ap.fit_transform(df, y)
        assert isinstance(out, np.ndarray)
        assert np.issubdtype(out.dtype, np.number)
        assert np.all(np.isfinite(out))
        assert "zip_code" in ap.get_plan().target_encode_columns

    def test_target_encoder_and_auto_preprocessor_string_y_and_all_nan_multicol(self):
        rng = np.random.RandomState(42)
        n = 40
        base = rng.randn(n)
        df = pd.DataFrame({
            "num_a": base,
            "num_b": base * 2.0 + 1e-4 * rng.randn(n),
            "all_nan": np.full(n, np.nan),
            "hc": [f"c_{i % 15}" for i in range(n)],
        })
        y = np.array(["yes" if i % 2 == 0 else "no" for i in range(n)])

        # Direct TargetEncoder with string y
        te = TargetEncoder(smoothing=5.0)
        encoded = te.fit_transform(df[["hc"]], y)
        assert encoded.shape == (n, 1)
        assert np.all(np.isfinite(encoded))

        # AutoPreprocessor with string y AND all-NaN column alongside correlated columns
        ap = AutoPreprocessor(high_cardinality_threshold=10, handle_multicollinearity=True, verbose=0)
        out = ap.fit_transform(df, y)
        assert isinstance(out, np.ndarray)
        assert np.issubdtype(out.dtype, np.number)
        assert np.all(np.isfinite(out))
        plan = ap.get_plan()
        assert "all_nan" in plan.drop_columns
        assert len(plan.correlated_groups) >= 1


class TestSafeFeatureImportancePerf:
    """Feature #5: DataFrame-safe, bounded-parallel, deterministic SafeFeatureImportance."""

    def test_safe_feature_importance_ranking_and_determinism(self):
        rng = np.random.RandomState(42)
        n = 200
        x_imp = rng.randn(n)
        df = pd.DataFrame({
            "imp_a": x_imp,
            "corr_1": x_imp + 0.02 * rng.randn(n),
            "corr_2": x_imp - 0.02 * rng.randn(n),
            "noise_1": rng.randn(n),
            "noise_2": rng.randn(n),
        })
        y = (x_imp > 0).astype(int)

        clf = ACDSOClassifier(cv=3, max_candidates=3, random_state=42, verbose=0)
        clf.fit(df, y)

        fi1 = SafeFeatureImportance(clf, n_repeats=3, cv_folds=3, n_bootstrap=5, random_state=42)
        rep1 = fi1.compute(df, y)

        fi2 = SafeFeatureImportance(clf, n_repeats=3, cv_folds=3, n_bootstrap=5, random_state=42)
        rep2 = fi2.compute(df, y)

        assert len(rep1.features) == 5
        assert all(0.0 <= f.stability <= 1.0 for f in rep1.features)
        # Correlated features flagged unreliable
        corr_feats = [f for f in rep1.features if f.name in {"imp_a", "corr_1", "corr_2"}]
        assert any(not f.is_reliable and len(f.correlated_with) > 0 for f in corr_feats)
        # Deterministic across identical seeds
        assert [f.name for f in rep1.features] == [f.name for f in rep2.features]
        np.testing.assert_allclose(
            [f.importance for f in rep1.features],
            [f.importance for f in rep2.features],
        )


class TestAutoEDAVectorized:
    """Feature #5: Zero-copy, vectorized profiling, correlation p-values & duplicate shortcut in eda.py."""

    def test_eda_parity_and_no_mutation(self):
        rng = np.random.RandomState(42)
        n = 150
        x1 = rng.randn(n)
        x2 = 2.0 * x1 + 0.1 * rng.randn(n)
        df = pd.DataFrame({
            "x1": x1,
            "x2": x2,
            "flag": rng.randint(0, 2, size=n).astype(bool),
            "cat": rng.choice(["alpha", "beta", "gamma"], size=n),
            "const": np.ones(n),
        })
        df.loc[0, "x1"] = np.nan
        df.loc[1, "x2"] = np.nan
        # Inject exact duplicate row
        df_with_dup = pd.concat([df, df.iloc[[2]]], ignore_index=True)
        df_orig = df_with_dup.copy(deep=True)

        eda = ACDSOAutoEDA(verbose=0)
        report = eda.analyze(df_with_dup, dataset_name="mixed_test")

        # Input not mutated
        pd.testing.assert_frame_equal(df_with_dup, df_orig)
        assert report.data_quality["n_duplicate_rows"] == 1
        assert eda.data_quality_.n_duplicate_rows == 1

        # Pearson correlation & p-value parity with scipy.stats.pearsonr
        clean_pair = df_with_dup[["x1", "x2"]].dropna()
        ref_r, ref_p = stats.pearsonr(clean_pair["x1"], clean_pair["x2"])
        pair_res = [
            c for c in report.high_correlations
            if {c["feature_1"], c["feature_2"]} == {"x1", "x2"}
        ]
        assert len(pair_res) == 1
        assert abs(pair_res[0]["correlation"] - round(float(ref_r), 4)) <= 1e-4
        assert abs(pair_res[0]["p_value"] - round(float(ref_p), 6)) <= 1e-5

    def test_eda_5000x30_fast_smoke(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(5000, 30), columns=[f"col_{i}" for i in range(30)])
        t0 = time.perf_counter()
        report = ACDSOAutoEDA(verbose=0).analyze(df, dataset_name="perf_smoke")
        dt = time.perf_counter() - t0
        assert len(report.column_profiles) == 30
        assert dt < 5.0

    def test_eda_unused_categorical_nullable_boolean_and_string_index_anomalies(self):
        # 1. Unused pd.Categorical categories must not inflate n_unique or break n_duplicate_rows
        df_cat = pd.DataFrame({
            "c": pd.Series(pd.Categorical(["a", "b", "a", "b"], categories=["a", "b", "u1", "u2"])),
            "v": [1, 2, 1, 2],
        })
        rep_cat = ACDSOAutoEDA(verbose=0).analyze(df_cat, dataset_name="unused_cat")
        c_prof = [p for p in rep_cat.column_profiles if p["name"] == "c"][0]
        assert c_prof["n_unique"] == 2
        assert rep_cat.data_quality["n_duplicate_rows"] == 2

        # 2. Nullable boolean column with pd.NA must profile without crashing
        df_bool = pd.DataFrame({
            "nb": pd.Series([True, False, pd.NA, True, False], dtype="boolean"),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        rep_bool = ACDSOAutoEDA(verbose=0).analyze(df_bool, dataset_name="nullable_bool")
        nb_prof = [p for p in rep_bool.column_profiles if p["name"] == "nb"][0]
        assert nb_prof["n_missing"] == 1

        # 3. Non-integer string index with statistical anomalies must not crash _detect_anomalies
        vals = [0.0] * 30 + [100.0]
        df_str_idx = pd.DataFrame({"x": vals}, index=[f"row_{i}" for i in range(31)])
        rep_anom = ACDSOAutoEDA(verbose=0).analyze(df_str_idx, dataset_name="str_idx_anom")
        assert len(rep_anom.anomalies) >= 1
        assert all(isinstance(a["row_index"], int) for a in rep_anom.anomalies)


class TestTimeseriesMemoryMetric:
    """Feature #5: Deep model memory tracking & non-destructive tracemalloc in AutoTS."""

    def test_autots_memory_metric_and_tracemalloc_preservation(self):
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        values = np.sin(np.arange(90) * 0.2) * 10.0 + np.arange(90) * 0.5
        df = pd.DataFrame({"date": dates, "sales": values})

        was_already_tracing = tracemalloc.is_tracing()
        if not was_already_tracing:
            tracemalloc.start()
        try:
            ts = AutoTS(
                datetime_col="date",
                target_col="sales",
                horizon=7,
                n_splits=2,
                verbose=False,
            )
            res = ts.fit(df)
            assert tracemalloc.is_tracing() is True
            assert len(res.all_pipelines) > 0
            assert all(p.memory_mb > 0.0 for p in res.all_pipelines)

            naive = [p for p in res.all_pipelines if p.is_baseline]
            rf = [p for p in res.all_pipelines if "RandomForest" in p.name]
            if naive and rf:
                assert max(p.memory_mb for p in rf) > min(p.memory_mb for p in naive)
        finally:
            if not was_already_tracing and tracemalloc.is_tracing():
                tracemalloc.stop()
