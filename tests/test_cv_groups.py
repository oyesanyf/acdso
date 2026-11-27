"""
Tests for Entity-Grouped & Temporal Cross-Validation (Milestone M1 / R3)
========================================================================

Genuine unit and integration tests for ``acdso.cv`` and its wiring across
``core.py``, ``risk.py``, ``feature_importance.py``, and
``integrations/meta_optimizer.py``.
"""

from __future__ import annotations

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, TimeSeriesSplit

from acdso import (
    ACDSOClassifier,
    ACDSORegressor,
    SafeFeatureImportance,
    analyze_risk,
    compute_safe_importance,
)
from acdso.cv import (
    _TimeSeriesSplitterAdapter,
    build_cv_splitter,
    group_holdout_split,
    iter_cv_splits,
    resolve_cv_strategy_name,
    sample_by_groups,
    splitter_supports_groups,
    validate_groups,
    validate_no_group_leakage,
)
from acdso.integrations.base import SklearnNativeAdapter
from acdso.integrations.meta_optimizer import (
    ACDSOMetaClassifier,
    ACDSOMetaOptimizer,
    ACDSOMetaRegressor,
)
from acdso.timeseries.splits import ExpandingWindowSplit, WalkForwardSplit


def _make_grouped_classification(
    n_groups: int = 12,
    rows_per_group: int = 12,
    n_features: int = 6,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    n = n_groups * rows_per_group
    groups = np.repeat([f"entity_{i:02d}" for i in range(n_groups)], rows_per_group)
    X_arr = rng.randn(n, n_features)
    # Signal + entity-level offset
    group_signal = np.repeat(rng.choice([-1.2, 1.2], size=n_groups), rows_per_group)
    logits = 1.4 * X_arr[:, 0] - 0.9 * X_arr[:, 1] + 0.8 * group_signal + 0.3 * rng.randn(n)
    y = (logits > 0.0).astype(int)
    # Ensure both classes exist reliably
    if len(np.unique(y)) < 2:
        y[: rows_per_group] = 0
        y[-rows_per_group:] = 1
    df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(n_features)])
    return df, y, groups


def _make_grouped_regression(
    n_groups: int = 12,
    rows_per_group: int = 10,
    n_features: int = 6,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    n = n_groups * rows_per_group
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    X_arr = rng.randn(n, n_features)
    group_offset = np.repeat(rng.randn(n_groups) * 0.5, rows_per_group)
    y = 2.5 * X_arr[:, 0] - 1.5 * X_arr[:, 1] + group_offset + 0.2 * rng.randn(n)
    df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(n_features)])
    return df, y, groups


class TestGroupValidation:
    """Tests for validate_groups input normalization and error guards."""

    def test_accepts_various_1d_containers(self):
        expected_len = 6
        for container in (
            [1, 1, 2, 2, 3, 3],
            (0, 0, 1, 1, 2, 2),
            np.array(["a", "a", "b", "b", "c", "c"]),
            np.array([[1], [1], [2], [2], [3], [3]]),
            pd.Series(["p1", "p1", "p2", "p2", "p3", "p3"]),
            pd.DataFrame({"gid": [10, 10, 20, 20, 30, 30]}),
            pd.Index(["x", "x", "y", "y", "z", "z"]),
            pd.Categorical(["g1", "g1", "g2", "g2", "g3", "g3"]),
        ):
            arr = validate_groups(container, n_samples=expected_len)
            assert isinstance(arr, np.ndarray)
            assert arr.ndim == 1
            assert len(arr) == expected_len
            assert len(np.unique(arr)) == 3

    def test_mixed_int_and_str_groups_normalized(self):
        arr = validate_groups([1, 1, "a", "a", 2, 2], n_samples=6)
        assert arr.ndim == 1
        assert len(arr) == 6
        assert len(np.unique(arr)) == 3

    def test_rejects_invalid_groups(self):
        with pytest.raises(ValueError, match="None"):
            validate_groups(None)
        with pytest.raises(ValueError, match="empty"):
            validate_groups([])
        with pytest.raises(ValueError, match="n_samples"):
            validate_groups([1, 1, 2, 2], n_samples=5)
        with pytest.raises(ValueError, match="1 column"):
            validate_groups(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        with pytest.raises(ValueError, match="1-dimensional"):
            validate_groups(np.ones((4, 2)))
        with pytest.raises(ValueError, match="missing|NaN"):
            validate_groups(pd.Series(["g1", np.nan, "g2", "g2"]))
        with pytest.raises(ValueError, match="missing|NaN"):
            validate_groups(["g1", None, "g2", "g2"])
        with pytest.raises(ValueError, match="NaN|infinite"):
            validate_groups(np.array([1.0, np.inf, 2.0, 2.0]))
        with pytest.raises(ValueError, match="unique"):
            validate_groups([7, 7, 7, 7], min_unique=2)


class TestBuildCVSplitter:
    """Tests for build_cv_splitter, adapters, and sklearn 1.5.2 edge guards."""

    def test_auto_strategy_with_groups_classification_and_regression(self):
        df_c, y_c, g_c = _make_grouped_classification(n_groups=10, rows_per_group=6)
        sp_c = build_cv_splitter(cv=5, y=y_c, groups=g_c, task="classification", cv_strategy="auto")
        assert isinstance(sp_c, StratifiedGroupKFold)
        assert resolve_cv_strategy_name(sp_c) == "stratified_group_kfold"
        assert splitter_supports_groups(sp_c) is True
        splits_c = iter_cv_splits(sp_c, df_c, y_c, g_c)
        assert len(splits_c) == 5

        df_r, y_r, g_r = _make_grouped_regression(n_groups=10, rows_per_group=6)
        sp_r = build_cv_splitter(cv=5, y=y_r, groups=g_r, task="regression", cv_strategy="auto")
        assert isinstance(sp_r, GroupKFold)
        assert resolve_cv_strategy_name(sp_r) == "group_kfold"
        splits_r = iter_cv_splits(sp_r, df_r, y_r, g_r)
        assert len(splits_r) == 5

    def test_clamps_small_group_counts_without_empty_folds(self):
        X = np.zeros((12, 3))
        y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        groups = np.array(["A"] * 6 + ["B"] * 6)

        sp_auto = build_cv_splitter(cv=5, y=y, groups=groups, task="classification", cv_strategy="auto")
        assert isinstance(sp_auto, GroupKFold)
        splits_auto = iter_cv_splits(sp_auto, X, y, groups)
        assert len(splits_auto) == 2
        assert all(len(tr) > 0 and len(va) > 0 for tr, va in splits_auto)

        sp_sgk = build_cv_splitter(
            cv=5, y=y, groups=groups, task="classification", cv_strategy="stratified_group_kfold"
        )
        assert isinstance(sp_sgk, StratifiedGroupKFold)
        splits_sgk = iter_cv_splits(sp_sgk, X, y, groups)
        assert len(splits_sgk) == 2
        assert all(len(tr) > 0 and len(va) > 0 for tr, va in splits_sgk)

    def test_explicit_strategies_and_determinism(self):
        df, y, g = _make_grouped_classification(n_groups=8, rows_per_group=5)

        sp_gk = build_cv_splitter(cv=4, y=y, groups=g, cv_strategy="group_kfold", random_state=42)
        assert isinstance(sp_gk, GroupKFold)
        assert len(iter_cv_splits(sp_gk, df, y, g)) == 4

        sp_sgk1 = build_cv_splitter(cv=4, y=y, groups=g, cv_strategy="stratified_group_kfold", random_state=7)
        sp_sgk2 = build_cv_splitter(cv=4, y=y, groups=g, cv_strategy="stratified_group_kfold", random_state=7)
        s1 = iter_cv_splits(sp_sgk1, df, y, g)
        s2 = iter_cv_splits(sp_sgk2, df, y, g)
        for (tr1, va1), (tr2, va2) in zip(s1, s2):
            assert np.array_equal(tr1, tr2)
            assert np.array_equal(va1, va2)

        sp_temp = build_cv_splitter(cv=4, y=y, groups=g, cv_strategy="temporal")
        assert isinstance(sp_temp, TimeSeriesSplit)
        assert resolve_cv_strategy_name(sp_temp) == "temporal"
        assert splitter_supports_groups(sp_temp) is False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            temp_splits = iter_cv_splits(sp_temp, df, y, g)
        assert len(temp_splits) == 4
        assert not [w for w in caught if "ignored by TimeSeriesSplit" in str(w.message)]
        # Chronological property: max(train_idx) < min(val_idx)
        for tr, va in temp_splits:
            assert int(tr.max()) < int(va.min())

    def test_custom_splitter_and_timeseries_adapter(self):
        X = np.arange(60).reshape(30, 2)
        y = np.arange(30) % 2
        groups = np.repeat(np.arange(6), 5)

        custom_gk = GroupKFold(n_splits=3)
        assert build_cv_splitter(cv=custom_gk, y=y, groups=groups) is custom_gk

        exp = ExpandingWindowSplit(n_splits=3)
        wrapped_exp = build_cv_splitter(cv=exp, y=y)
        assert isinstance(wrapped_exp, _TimeSeriesSplitterAdapter)
        assert resolve_cv_strategy_name(wrapped_exp) == "temporal"
        assert splitter_supports_groups(wrapped_exp) is False
        splits_exp = iter_cv_splits(wrapped_exp, X, y, groups)
        assert len(splits_exp) == 3

        wf = WalkForwardSplit(n_splits=3)
        wrapped_wf = build_cv_splitter(cv=wf, y=y)
        assert isinstance(wrapped_wf, _TimeSeriesSplitterAdapter)
        assert len(iter_cv_splits(wrapped_wf, X, y, groups)) == 3

    def test_error_paths(self):
        y_clf = np.array([0, 1] * 10)
        y_reg = np.linspace(0.0, 1.0, 20)
        g = np.repeat(np.arange(5), 4)

        with pytest.raises(ValueError, match="Unknown cv_strategy"):
            build_cv_splitter(cv=3, y=y_clf, cv_strategy="invalid_mode")
        with pytest.raises(ValueError, match="cv must be >= 2"):
            build_cv_splitter(cv=1, y=y_clf)
        with pytest.raises(ValueError, match="integer >= 2"):
            build_cv_splitter(cv=True, y=y_clf)
        with pytest.raises(ValueError, match="requires non-None groups"):
            build_cv_splitter(cv=3, y=y_clf, groups=None, cv_strategy="group_kfold")
        with pytest.raises(ValueError, match="requires non-None groups"):
            build_cv_splitter(cv=3, y=y_clf, groups=None, cv_strategy="stratified_group_kfold")
        with pytest.raises(ValueError, match="binary or multiclass"):
            build_cv_splitter(cv=3, y=y_reg, groups=g, cv_strategy="stratified_group_kfold")
        with pytest.raises(ValueError, match="binary or multiclass"):
            build_cv_splitter(cv=3, y=None, groups=g, cv_strategy="stratified_group_kfold")
        with pytest.raises(ValueError, match="at least 3 samples"):
            build_cv_splitter(cv=3, y=np.array([0, 1]), cv_strategy="temporal")


class TestGroupLeakageValidator:
    """Tests for validate_no_group_leakage."""

    def test_passes_on_grouped_splitters(self):
        X = np.zeros((20, 2))
        y = np.array([0, 1] * 10)
        groups = np.repeat(["g1", "g2", "g3", "g4", "g5"], 4)

        for sp in (GroupKFold(n_splits=5), StratifiedGroupKFold(n_splits=5)):
            splits = list(sp.split(X, y, groups))
            validate_no_group_leakage(splits, groups)

    def test_detects_kfold_leakage_and_empty_folds(self):
        X = np.zeros((20, 2))
        y = np.array([0, 1] * 10)
        groups = np.repeat(["g1", "g2", "g3", "g4", "g5"], 4)

        leaky_splits = list(KFold(n_splits=4, shuffle=False).split(X, y))
        with pytest.raises(ValueError, match="Group leakage detected"):
            validate_no_group_leakage(leaky_splits, groups)

        with pytest.raises(ValueError, match="empty split"):
            validate_no_group_leakage([(np.array([0, 1, 2]), np.array([], dtype=int))], groups[:3])


class TestSampleByGroupsAndHoldout:
    """Tests for sample_by_groups and group_holdout_split."""

    def test_sample_by_groups_keeps_entities_whole_and_preserves_types(self):
        df, y, groups = _make_grouped_classification(n_groups=20, rows_per_group=8, seed=11)
        Xs, ys, gs = sample_by_groups(df, y=pd.Series(y), groups=groups, frac=0.25, random_state=42)

        assert isinstance(Xs, pd.DataFrame)
        assert isinstance(ys, pd.Series)
        assert isinstance(gs, np.ndarray)
        assert len(Xs) == len(ys) == len(gs)
        assert len(np.unique(gs)) >= 2

        # Every sampled entity retains 100% of its original rows
        orig_counts = pd.Series(groups).value_counts().to_dict()
        sub_counts = pd.Series(gs).value_counts().to_dict()
        for gid, cnt in sub_counts.items():
            assert cnt == orig_counts[gid]

        # Deterministic with identical random_state
        Xs2, ys2, gs2 = sample_by_groups(df, y=pd.Series(y), groups=groups, frac=0.25, random_state=42)
        assert np.array_equal(gs, gs2)
        pd.testing.assert_frame_equal(Xs, Xs2)

    def test_sample_by_groups_ndarray_and_groups_none(self):
        X_arr = np.arange(100, dtype=float).reshape(50, 2)
        y_arr = np.arange(50)
        Xs, ys, gs = sample_by_groups(X_arr, y=y_arr, groups=None, max_rows=20, random_state=7)
        assert isinstance(Xs, np.ndarray)
        assert len(Xs) == len(ys) == 20
        assert gs is None

    def test_group_holdout_split_disjoint_even_with_two_groups(self):
        X = np.zeros((12, 3))
        y = np.array([0, 1] * 6)
        groups = np.array(["G1"] * 6 + ["G2"] * 6)

        tr, va = group_holdout_split(X, y=y, groups=groups, test_size=0.2, random_state=42)
        assert len(tr) > 0 and len(va) > 0
        assert set(groups[tr]).isdisjoint(set(groups[va]))


class TestCoreEndToEndGroupsAndTemporal:
    """End-to-end integration tests for ACDSOClassifier & ACDSORegressor with groups/temporal CV."""

    def test_classifier_grouped_cv_subsampling_stacking_pickle_clone(self):
        # n=144 >= 100 exercises Stage 0 _subsample_rows, Stage 1 caching, and OOF stacking
        df, y, groups = _make_grouped_classification(n_groups=12, rows_per_group=12, seed=42)
        clf = ACDSOClassifier(
            cv=4,
            max_candidates=6,
            enable_stacking=True,
            n_jobs=1,
            random_state=42,
        )
        clf.fit(df, y, groups=groups)

        assert clf.best_estimator_ is not None
        assert clf.cv_strategy_ == "stratified_group_kfold"
        assert clf.audit_trail_ is not None
        assert clf.audit_trail_["cv_strategy"] == "stratified_group_kfold"
        assert clf.audit_trail_["groups_provided"] is True
        assert clf.audit_trail_["n_groups"] == 12
        json.dumps(clf.audit_trail_)

        preds = clf.predict(df.iloc[:10])
        proba = clf.predict_proba(df.iloc[:10])
        assert len(preds) == 10
        assert proba.shape == (10, 2)
        assert 0.0 <= clf.score(df, y) <= 1.0

        restored = pickle.loads(pickle.dumps(clf))
        assert np.array_equal(restored.predict(df.iloc[:10]), preds)

        cloned = clone(clf)
        assert cloned.get_params()["cv_strategy"] == "auto"
        assert cloned.best_estimator_ is None
        cloned.set_params(cv_strategy="group_kfold")
        assert cloned.get_params()["cv_strategy"] == "group_kfold"

    def test_classifier_small_group_count_clamping_and_stacking(self):
        df, y, groups = _make_grouped_classification(n_groups=3, rows_per_group=24, seed=99)
        clf = ACDSOClassifier(
            cv=5,
            max_candidates=6,
            enable_stacking=True,
            n_jobs=1,
            random_state=99,
        )
        clf.fit(df, y, groups=groups)
        assert clf.best_estimator_ is not None
        assert clf.audit_trail_["n_groups"] == 3
        assert clf.audit_trail_["cv_strategy"] in ("group_kfold", "stratified_group_kfold")
        for cand in clf.audit_trail_["candidates_evaluated"]:
            assert len(cand["cv_scores"]) == 3

    def test_regressor_grouped_cv(self):
        df, y, groups = _make_grouped_regression(n_groups=10, rows_per_group=12, seed=17)
        reg = ACDSORegressor(
            cv=4,
            max_candidates=5,
            enable_stacking=True,
            cv_strategy="group_kfold",
            n_jobs=1,
            random_state=17,
        )
        reg.fit(df, y, groups=groups)

        assert reg.best_estimator_ is not None
        assert reg.cv_strategy_ == "group_kfold"
        assert reg.audit_trail_["cv_strategy"] == "group_kfold"
        assert reg.audit_trail_["groups_provided"] is True
        assert reg.audit_trail_["n_groups"] == 10
        preds = reg.predict(df.iloc[:10])
        assert len(preds) == 10
        assert reg.score(df, y) > 0.5

    def test_temporal_cv_with_stacking_enabled_does_not_crash(self):
        rng = np.random.RandomState(42)
        n = 120
        t = np.arange(n, dtype=float)
        df = pd.DataFrame({
            "x1": np.sin(t / 10.0) + rng.randn(n) * 0.2,
            "x2": rng.randn(n),
            "x3": rng.randn(n),
        })
        y_clf = (df["x1"] > 0.0).astype(int).to_numpy()
        y_reg = 2.0 * df["x1"].to_numpy() + rng.randn(n) * 0.2

        clf = ACDSOClassifier(cv=3, max_candidates=6, enable_stacking=True, cv_strategy="temporal", n_jobs=1)
        clf.fit(df, y_clf)
        assert clf.cv_strategy_ == "temporal"
        assert clf.audit_trail_["cv_strategy"] == "temporal"
        assert len(clf.predict(df.iloc[-10:])) == 10

        reg = ACDSORegressor(cv=3, max_candidates=6, enable_stacking=True, cv_strategy="temporal", n_jobs=1)
        reg.fit(df, y_reg)
        assert reg.cv_strategy_ == "temporal"
        assert clf.best_estimator_ is not None and reg.best_estimator_ is not None

    def test_grouped_cv_prevents_entity_memorization_leakage(self):
        # Each entity has a unique constant ID feature and a constant noisy label uncorrelated
        # with unseen entities; grouped CV must not report inflated entity-memorized accuracy.
        rng = np.random.RandomState(123)
        n_groups = 16
        rows_per_group = 6
        entity_labels = rng.randint(0, 2, size=n_groups)
        groups = np.repeat(np.arange(n_groups), rows_per_group)
        y = np.repeat(entity_labels, rows_per_group)
        df = pd.DataFrame({
            "entity_proxy": np.repeat(np.arange(n_groups, dtype=float) * 100.0, rows_per_group),
            "noise_1": rng.randn(n_groups * rows_per_group),
            "noise_2": rng.randn(n_groups * rows_per_group),
        })

        clf_ungrouped = ACDSOClassifier(cv=4, max_candidates=4, enable_stacking=False, n_jobs=1, random_state=42)
        clf_ungrouped.fit(df, y)
        ungrouped_metric = clf_ungrouped.audit_trail_["selected"]["metric"]

        clf_grouped = ACDSOClassifier(cv=4, max_candidates=4, enable_stacking=False, n_jobs=1, random_state=42)
        clf_grouped.fit(df, y, groups=groups)
        grouped_metric = clf_grouped.audit_trail_["selected"]["metric"]

        assert grouped_metric < ungrouped_metric

    def test_invalid_groups_or_strategy_raises_upfront(self):
        df, y, groups = _make_grouped_classification(n_groups=6, rows_per_group=6)
        clf = ACDSOClassifier(cv=3, max_candidates=3, n_jobs=1)

        with pytest.raises(ValueError, match="requires non-None groups"):
            clf.fit(df, y, cv_strategy="group_kfold")
        with pytest.raises(ValueError, match="n_samples"):
            clf.fit(df, y, groups=groups[:-1])
        with pytest.raises(ValueError, match="missing|NaN"):
            bad_g = groups.astype(object)
            bad_g[0] = np.nan
            clf.fit(df, y, groups=bad_g)
        with pytest.raises(ValueError, match="unique groups"):
            clf.fit(df, y, groups=np.ones(len(y)))


class _CompatMockAdapter(SklearnNativeAdapter):
    """Adapter preserving the 3-arg generate_candidates(X, y, task) contract."""

    def __init__(self, models):
        super().__init__(name="CompatMockAdapter")
        self.models = models

    def is_available(self):
        return True

    def generate_candidates(self, X, y, task):
        out = []
        for idx, model in enumerate(self.models):
            model.fit(X, y)
            out.append(self.wrap_estimator(model, name=f"compat_m_{idx}", rank=idx))
        return out


class TestRiskFeatureImportanceMetaWithGroups:
    """Integration tests for risk.py, feature_importance.py, and meta_optimizer.py with groups."""

    def test_analyze_risk_with_groups_large_and_temporal(self):
        rng = np.random.RandomState(42)
        # >3000 rows exercises sample_by_groups inside _univariate_predictive_power_detailed & _detect_linearity
        n_groups = 120
        rows_per_group = 30
        n = n_groups * rows_per_group
        groups = np.repeat(np.arange(n_groups), rows_per_group)
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        x1 = rng.randn(n)
        df = pd.DataFrame({
            "ts": dates,
            "x1": x1,
            "x2": rng.randn(n),
            "leak": (x1 > 0).astype(float) + rng.randn(n) * 0.01,
        })
        y = (x1 > 0).astype(int)

        report = analyze_risk(df, y, task="classification", cv=4, groups=groups, random_state=42)
        assert report.cv_strategy == "stratified_group_kfold"
        assert report.n_groups == n_groups
        assert report.details["groups_provided"] is True
        assert report.details["n_groups"] == n_groups
        assert report.temporal_leakage_gap is not None
        assert "leak" in report.leakage_proxy_features
        json.dumps(report.to_dict())

        # Small group count triggers fold clamping warning + note
        small_g = np.repeat(["g1", "g2", "g3"], 20)
        rep_small = analyze_risk(
            df.iloc[:60], y[:60], task="classification", cv=5, groups=small_g, random_state=42
        )
        assert rep_small.n_groups == 3
        assert rep_small.cv_strategy == "stratified_group_kfold"
        assert any("clamped" in w for w in rep_small.warnings)

        # Fail fast before try/except when invalid
        with pytest.raises(ValueError, match="requires non-None groups"):
            analyze_risk(df.iloc[:40], y[:40], task="classification", cv_strategy="group_kfold")

    def test_safe_feature_importance_with_groups(self):
        df, y, groups = _make_grouped_classification(n_groups=10, rows_per_group=10, seed=55)
        model = RandomForestClassifier(n_estimators=15, random_state=55).fit(df, y)

        fi = SafeFeatureImportance(
            model,
            n_repeats=2,
            cv_folds=4,
            n_bootstrap=4,
            random_state=55,
            n_jobs=1,
        )
        rep = fi.compute(df, y, groups=groups)
        assert rep.cv_strategy == "stratified_group_kfold"
        assert rep.n_groups == 10
        d = rep.to_dict()
        assert d["cv_strategy"] == "stratified_group_kfold"
        assert d["n_groups"] == 10
        assert rep.get_feature_importance("f0") is not None

        rep2 = compute_safe_importance(model, df, y, groups=groups, cv_strategy="group_kfold")
        assert rep2.cv_strategy == "group_kfold"
        assert rep2.n_groups == 10

    def test_meta_optimizer_with_groups_and_mock_adapter(self):
        from sklearn.linear_model import LogisticRegression

        df, y, groups = _make_grouped_classification(n_groups=10, rows_per_group=10, seed=77)
        adapter = _CompatMockAdapter([
            LogisticRegression(max_iter=500, random_state=77),
            RandomForestClassifier(n_estimators=15, random_state=77),
        ])

        meta_clf = ACDSOMetaClassifier(
            adapters=[adapter],
            include_acdso_candidates=True,
            cv=4,
            random_state=77,
        )
        meta_clf.fit(df, y, groups=groups)
        assert meta_clf.cv_strategy_ == "stratified_group_kfold"
        assert meta_clf.audit_trail_["cv_strategy"] == "stratified_group_kfold"
        assert meta_clf.audit_trail_["n_groups"] == 10
        assert len(meta_clf.predict(df.iloc[:8])) == 8

        df_r, y_r, g_r = _make_grouped_regression(n_groups=8, rows_per_group=10, seed=88)
        meta_reg = ACDSOMetaRegressor(include_acdso_candidates=True, cv=4, random_state=88)
        meta_reg.fit(df_r, y_r, groups=g_r, cv_strategy="group_kfold")
        assert meta_reg.cv_strategy_ == "group_kfold"
        assert meta_reg.audit_trail_["cv_strategy"] == "group_kfold"
        assert meta_reg.audit_trail_["n_groups"] == 8

        with pytest.raises(ValueError, match="requires non-None groups"):
            ACDSOMetaOptimizer(adapters=[]).fit(df, y, cv_strategy="group_kfold")


class TestM1Iteration2Regressions:
    """Regression tests for M1 Iteration 2 fixes across cv, core, risk, fi, and meta_optimizer."""

    def test_validate_no_group_leakage_rejects_negative_indices(self):
        groups = np.array(["a", "a", "b", "c"])
        with pytest.raises(ValueError, match="negative"):
            validate_no_group_leakage([(np.array([-1]), np.array([0, 1]))], groups)
        with pytest.raises(ValueError, match="negative"):
            validate_no_group_leakage([(np.array([0, 1]), np.array([-2]))], groups)

    def test_group_holdout_split_clamps_high_test_size_on_two_groups(self):
        X = np.zeros((20, 3))
        y = np.array([0, 1] * 10)
        groups = np.array(["G1"] * 10 + ["G2"] * 10)

        for ts in (0.8, 0.95, 5):
            tr, va = group_holdout_split(X, y=y, groups=groups, test_size=ts, random_state=42)
            assert len(tr) > 0 and len(va) > 0
            assert set(groups[tr]).isdisjoint(set(groups[va]))

    def test_temporal_leakage_gap_filters_nan_and_inf_targets(self):
        from acdso.risk import _temporal_leakage_gap

        rng = np.random.RandomState(123)
        n = 240
        groups = np.repeat(np.arange(48), 5)
        df = pd.DataFrame({
            "event_dt": pd.date_range("2024-01-01", periods=n, freq="h"),
            "sig": rng.randn(n),
            "noise": rng.randn(n),
        })
        y = 2.0 * df["sig"].to_numpy() + rng.randn(n) * 0.2
        y_dirty = y.copy()
        y_dirty[5] = np.nan
        y_dirty[15] = np.inf
        y_dirty[25] = -np.inf

        rep = analyze_risk(df, y_dirty, task="regression", scorer="r2", cv=4, groups=groups, random_state=42)
        assert rep.temporal_leakage_gap is not None
        assert np.isfinite(rep.temporal_leakage_gap)

        gap_direct = _temporal_leakage_gap(df, y_dirty, scorer="r2", cv=4, task="regression", random_state=42, groups=groups)
        assert gap_direct is not None and np.isfinite(gap_direct)

        # Too few valid rows after filtering cleanly returns None
        y_mostly_nan = np.full(n, np.nan)
        y_mostly_nan[:5] = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _temporal_leakage_gap(df, y_mostly_nan, scorer="r2", cv=4, task="regression", random_state=42, groups=groups) is None

    def test_cv_strategy_reporting_matches_clamped_splitter(self):
        import acdso.integrations.meta_optimizer as m
        import acdso.risk as r
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(1)
        X = pd.DataFrame({"x1": rng.randn(80), "x2": rng.randn(80)})
        y = np.repeat([0, 1, 0, 1], 20)
        groups = np.repeat([0, 1, 2, 3], 20)

        seen_risk = []
        orig_build = r.build_cv_splitter
        try:
            def _spy_build(*args, **kwargs):
                sp = orig_build(*args, **kwargs)
                seen_risk.append(type(sp).__name__)
                return sp

            r.build_cv_splitter = _spy_build
            rep = r.analyze_risk(X, y, task="classification", cv=5, groups=groups, random_state=1)
        finally:
            r.build_cv_splitter = orig_build

        assert rep.cv_strategy == "stratified_group_kfold"
        assert all(name == "StratifiedGroupKFold" for name in seen_risk)

        seen_meta = []
        orig_iter = m.iter_cv_splits
        adapter = _CompatMockAdapter([LogisticRegression(max_iter=500, random_state=1)])
        try:
            def _spy_iter(sp, *args, **kwargs):
                seen_meta.append(type(sp).__name__)
                return orig_iter(sp, *args, **kwargs)

            m.iter_cv_splits = _spy_iter
            opt = m.ACDSOMetaClassifier(
                adapters=[adapter],
                include_acdso_candidates=False,
                cv=5,
                random_state=1,
            ).fit(X, y, groups=groups)
        finally:
            m.iter_cv_splits = orig_iter

        assert opt.cv_strategy_ == "stratified_group_kfold"
        assert opt.audit_trail_["cv_strategy"] == "stratified_group_kfold"
        assert seen_meta == ["StratifiedGroupKFold"]

    def test_core_temporal_splitter_object_propagates_temporal_subsampling(self, monkeypatch):
        from acdso.core import _ACDSOEngineV3

        rng = np.random.RandomState(42)
        n = 120
        t = np.arange(n, dtype=float)
        df = pd.DataFrame({
            "x1": np.sin(t / 8.0) + rng.randn(n) * 0.15,
            "x2": rng.randn(n),
        })
        y = (df["x1"] > 0.0).astype(int).to_numpy()

        clf = ACDSOClassifier(
            cv=ExpandingWindowSplit(n_splits=3),
            max_candidates=4,
            enable_stacking=False,
            n_jobs=1,
            random_state=42,
        )

        observed_strategies = []
        orig_subsample = _ACDSOEngineV3._subsample_rows
        def _spy_subsample(self_eng, X_in, y_in, frac, groups=None, cv_strategy=None, stage_cv=None):
            observed_strategies.append(cv_strategy)
            return orig_subsample(self_eng, X_in, y_in, frac, groups=groups, cv_strategy=cv_strategy, stage_cv=stage_cv)

        monkeypatch.setattr(_ACDSOEngineV3, "_subsample_rows", _spy_subsample)
        clf.fit(df, y)
        assert clf.cv_strategy_ == "temporal"
        assert "temporal" in observed_strategies

        # Direct _subsample_rows with resolved temporal strategy returns chronological tail
        Xs, ys, gs = orig_subsample(clf._engine, df, y, 0.25, groups=None, cv_strategy=clf.cv_strategy_)
        assert gs is None
        assert list(Xs.index) == list(range(90, 120))

    def test_entity_pure_12_groups_subsampling_does_not_collapse_to_fallback(self, monkeypatch):
        from acdso.core import _ACDSOEngineV3

        rng = np.random.RandomState(7)
        n_groups = 12
        rows_per_group = 12
        # 6 groups all-0, 6 groups all-1 (entity-pure labels)
        group_labels = np.array([0] * 6 + [1] * 6)
        groups = np.repeat([f"ent_{i:02d}" for i in range(n_groups)], rows_per_group)
        y = np.repeat(group_labels, rows_per_group)
        n = len(y)
        signal = np.where(y == 1, 2.0, -2.0) + rng.randn(n) * 0.3
        df = pd.DataFrame({
            "sig": signal,
            "f1": 0.5 * signal + rng.randn(n) * 0.2,
            "noise": rng.randn(n),
        })

        clf = ACDSOClassifier(
            cv=4,
            max_candidates=6,
            enable_stacking=True,
            n_jobs=1,
            random_state=7,
        )

        stage0_groups = []
        orig_sub = _ACDSOEngineV3._subsample_rows
        def _spy_sub(self_eng, X_in, y_in, frac, groups=None, cv_strategy=None, stage_cv=None):
            Xs, ys, gs = orig_sub(self_eng, X_in, y_in, frac, groups=groups, cv_strategy=cv_strategy, stage_cv=stage_cv)
            if frac < 1.0 and gs is not None:
                stage0_groups.append((np.asarray(ys), np.asarray(gs)))
            return Xs, ys, gs

        monkeypatch.setattr(_ACDSOEngineV3, "_subsample_rows", _spy_sub)
        clf.fit(df, y, groups=groups)

        assert len(stage0_groups) == 1
        ys0, gs0 = stage0_groups[0]
        # Both classes retain >= 2 distinct groups in Stage 0 subsample
        for cls_val in (0, 1):
            assert len(np.unique(gs0[ys0 == cls_val])) >= 2

        assert clf.audit_trail_["selected"]["name"] != "Baseline_Fallback"
        assert clf.score(df, y) >= 0.90

    def test_safe_feature_importance_temporal_and_zero_bootstrap(self):
        rng = np.random.RandomState(21)
        n = 80
        df = pd.DataFrame({
            "sig": rng.randn(n),
            "noise": rng.randn(n),
        })
        y = (df["sig"] > 0.0).astype(int).to_numpy()
        model = RandomForestClassifier(n_estimators=15, random_state=21).fit(df, y)

        fi = SafeFeatureImportance(
            model,
            n_repeats=2,
            cv_folds=3,
            n_bootstrap=0,
            random_state=21,
            n_jobs=1,
            cv_strategy="temporal",
        )
        rep = fi.compute(df, y)
        assert rep.cv_strategy == "temporal"
        assert rep.n_groups is None
        assert len(rep.features) == 2
        assert all(np.isfinite(f.importance) and f.stability == 1.0 for f in rep.features)


class TestM1Iteration3Regressions:
    """Regression tests for M1 Iteration 3 fixes (proportional stratified group sampling & minors)."""

    def test_sample_by_groups_stratify_preserves_90_10_proportions_and_whole_groups(self):
        groups = np.repeat([f"e_{i:03d}" for i in range(100)], 10)
        y = np.repeat([0] * 90 + [1] * 10, 10)
        X = pd.DataFrame({"x": np.arange(1000, dtype=float)})

        Xs, ys, gs = sample_by_groups(
            X,
            y=y,
            groups=groups,
            frac=0.20,
            random_state=42,
            stratify=True,
            min_groups_per_class=2,
            min_groups_total=4,
        )
        assert len(Xs) == len(ys) == len(gs) == 200
        # Whole groups preserved
        orig_counts = pd.Series(groups).value_counts().to_dict()
        sub_counts = pd.Series(gs).value_counts().to_dict()
        for gid, cnt in sub_counts.items():
            assert cnt == orig_counts[gid]
        # Minimum groups per class preserved
        assert len(np.unique(gs[ys == 0])) >= 2
        assert len(np.unique(gs[ys == 1])) >= 2
        # Minority proportion closely matches 0.10 (not distorted to 0.50)
        sub_c1 = float(np.mean(ys == 1))
        assert 0.08 <= sub_c1 <= 0.15

    def test_sample_by_groups_stratify_preserves_61_23_15_multiclass_and_mixed_groups(self):
        # 20 entity-pure groups: 12 class-0 (123 rows, 61.5%), 5 class-1 (47 rows, 23.5%), 3 class-2 (30 rows, 15.0%)
        sizes = [10] * 9 + [11] * 3 + [9] * 3 + [10] * 2 + [10] * 3
        groups = np.concatenate([np.repeat(f"g_{i:02d}", sz) for i, sz in enumerate(sizes)])
        y = np.concatenate([np.repeat(0, 123), np.repeat(1, 47), np.repeat(2, 30)])
        X = pd.DataFrame({"x": np.arange(len(y), dtype=float)})

        Xs, ys, gs = sample_by_groups(
            X,
            y=y,
            groups=groups,
            frac=0.35,
            random_state=42,
            stratify=True,
            min_groups_per_class=2,
            min_groups_total=6,
        )
        orig_counts = pd.Series(groups).value_counts().to_dict()
        for gid, cnt in pd.Series(gs).value_counts().to_dict().items():
            assert cnt == orig_counts[gid]
        for c in (0, 1, 2):
            assert len(np.unique(gs[ys == c])) >= 2
        # Majority class 0 must remain the dominant class (not inverted below classes 1 & 2)
        p0, p1, p2 = (float(np.mean(ys == c)) for c in (0, 1, 2))
        assert p0 > p1 and p0 > p2 and p0 >= 0.45

        # Mixed-class groups (16 groups, n=160, frac=0.35) + dirty mixed/NaN y
        g_mix = np.repeat([f"m_{i:02d}" for i in range(16)], 10)
        y_mix = np.tile([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], 16)
        X_mix = pd.DataFrame({"x": np.arange(160, dtype=float)})
        Xs_m, ys_m, gs_m = sample_by_groups(
            X_mix,
            y=y_mix,
            groups=g_mix,
            frac=0.35,
            random_state=42,
            stratify=True,
            min_groups_per_class=2,
            min_groups_total=4,
        )
        assert len(Xs_m) == 60
        assert all(len(np.unique(gs_m[ys_m == c])) >= 2 for c in (0, 1, 2))

        y_dirty = np.array([0, "lbl", 0, "lbl", np.nan] * 6, dtype=object)
        g_dirty = np.repeat(["a", "b", "c", "d", "e", "f"], 5)
        Xs_d, ys_d, gs_d = sample_by_groups(
            pd.DataFrame({"x": np.arange(30, dtype=float)}),
            y=y_dirty,
            groups=g_dirty,
            frac=0.5,
            random_state=42,
            stratify=True,
            min_groups_per_class=2,
            min_groups_total=3,
        )
        assert len(Xs_d) == len(ys_d) == len(gs_d) >= 15

    def test_risk_temporal_leakage_gap_object_bool_with_missing(self):
        rng = np.random.RandomState(42)
        n = 240
        groups = np.repeat(np.arange(48), 5)
        df = pd.DataFrame({
            "event_dt": pd.date_range("2024-01-01", periods=n, freq="h"),
            "sig": rng.randn(n),
            "noise": rng.randn(n),
        })
        y_obj = (df["sig"].to_numpy() > 0.0).astype(object)
        y_obj[5] = np.nan
        y_obj[15] = None

        rep = analyze_risk(df, y_obj, task="classification", cv=4, groups=groups, random_state=42)
        assert rep.temporal_leakage_gap is not None
        assert np.isfinite(rep.temporal_leakage_gap)

    def test_meta_optimizer_temporal_does_not_clamp_splits_by_n_groups(self):
        import acdso.integrations.meta_optimizer as m
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(11)
        n = 120
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = (X["x1"] > 0.0).astype(int).to_numpy()
        groups = np.repeat(["g0", "g1", "g2"], n // 3)  # n_groups=3 < cv=5

        observed_n_splits = []
        orig_iter = m.iter_cv_splits
        adapter = _CompatMockAdapter([LogisticRegression(max_iter=500, random_state=11)])
        try:
            def _spy_iter(sp, *args, **kwargs):
                observed_n_splits.append((type(sp).__name__, sp.get_n_splits()))
                return orig_iter(sp, *args, **kwargs)

            m.iter_cv_splits = _spy_iter
            opt = m.ACDSOMetaClassifier(
                adapters=[adapter],
                include_acdso_candidates=False,
                cv=5,
                cv_strategy="temporal",
                random_state=11,
            ).fit(X, y, groups=groups)
        finally:
            m.iter_cv_splits = orig_iter

        assert opt.cv_strategy_ == "temporal"
        assert opt.audit_trail_["cv_strategy"] == "temporal"
        assert observed_n_splits == [("TimeSeriesSplit", 5)]

    def test_core_subsample_rows_temporal_with_groups_takes_trailing_whole_groups(self):
        from acdso.core import _ACDSOEngineV3

        rng = np.random.RandomState(9)
        n = 120
        groups = np.repeat([f"g_{i:02d}" for i in range(12)], 10)
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = np.tile([0, 1], n // 2)

        clf = ACDSOClassifier(
            cv=TimeSeriesSplit(n_splits=3),
            max_candidates=4,
            enable_stacking=False,
            n_jobs=1,
            random_state=9,
        ).fit(X, y, groups=groups)
        assert clf.cv_strategy_ == "temporal"

        Xs, ys, gs = clf._engine._subsample_rows(
            X, y, frac=0.25, groups=groups, cv_strategy="temporal", stage_cv=2
        )
        assert len(Xs) == 30
        assert list(Xs.index) == list(range(90, 120))
        assert set(np.unique(gs)) == {"g_09", "g_10", "g_11"}


class TestM1Iteration4Regressions:
    """Regression tests for M1 Iteration 4 fixes across feature_importance, risk, and core."""

    def test_feature_importance_temporal_does_not_clamp_by_small_n_groups(self, monkeypatch):
        import acdso.feature_importance as fi_mod

        rng = np.random.RandomState(42)
        n = 120
        df = pd.DataFrame({
            "sig": rng.randn(n),
            "noise": rng.randn(n),
        })
        y = (df["sig"] > 0.0).astype(int).to_numpy()
        groups_3 = np.repeat(["g0", "g1", "g2"], n // 3)
        model = RandomForestClassifier(n_estimators=15, random_state=42).fit(df, y)

        seen_splitters = []
        orig_iter = fi_mod.iter_cv_splits

        def _spy_iter(sp, *args, **kwargs):
            seen_splitters.append((type(sp).__name__, sp.get_n_splits()))
            return orig_iter(sp, *args, **kwargs)

        monkeypatch.setattr(fi_mod, "iter_cv_splits", _spy_iter)

        fi = SafeFeatureImportance(
            model,
            n_repeats=2,
            cv_folds=5,
            n_bootstrap=0,
            random_state=42,
            n_jobs=1,
            cv_strategy="temporal",
        )
        rep = fi.compute(df, y, groups=groups_3)
        assert rep.cv_strategy == "temporal"
        assert rep.n_groups == 3
        assert seen_splitters == [("TimeSeriesSplit", 5)]

    def test_analyze_risk_object_bool_with_missing_auto_and_sgk_consistency(self, monkeypatch):
        import acdso.risk as r_mod

        rng = np.random.RandomState(42)
        n_groups = 20
        rows_per_group = 10
        n = n_groups * rows_per_group
        groups = np.repeat(np.arange(n_groups), rows_per_group)
        df = pd.DataFrame({
            "event_dt": pd.date_range("2024-01-01", periods=n, freq="h"),
            "sig": rng.randn(n),
            "noise": rng.randn(n),
        })
        y_obj = (df["sig"].to_numpy() > 0.0).astype(object)
        y_obj[5] = np.nan
        y_obj[15] = None
        y_obj[47] = np.nan

        seen_internal = []
        orig_build = r_mod.build_cv_splitter

        def _spy_build(*args, **kwargs):
            sp = orig_build(*args, **kwargs)
            seen_internal.append(r_mod.resolve_cv_strategy_name(sp))
            return sp

        monkeypatch.setattr(r_mod, "build_cv_splitter", _spy_build)
        rep_auto = analyze_risk(
            df, y_obj, task="classification", cv=4, groups=groups, cv_strategy="auto", random_state=42
        )
        assert rep_auto.cv_strategy == "stratified_group_kfold"
        assert seen_internal
        assert all(s == "stratified_group_kfold" for s in seen_internal)

        rep_sgk = analyze_risk(
            df,
            y_obj,
            task="classification",
            cv=4,
            groups=groups,
            cv_strategy="stratified_group_kfold",
            random_state=42,
        )
        assert rep_sgk.cv_strategy == "stratified_group_kfold"

        # Edge case: < 2 groups remain after filtering missing targets
        y_one_grp = np.full(40, np.nan, dtype=object)
        y_one_grp[:10] = [True, False] * 5
        g_two = np.repeat(["g0", "g1"], 20)
        rep_edge = analyze_risk(
            df.iloc[:40], y_one_grp, task="classification", cv=4, groups=g_two, cv_strategy="auto", random_state=42
        )
        assert rep_edge.cv_strategy in ("group_kfold", "stratified_group_kfold")

    def test_analyze_risk_regression_default_scorer_and_nonfinite_gap_guard(self, monkeypatch):
        import acdso.risk as r_mod
        from acdso.risk import RiskReport, _temporal_leakage_gap, compute_risk_penalty

        rng = np.random.RandomState(42)
        n = 180
        groups_3 = np.repeat(["g0", "g1", "g2"], n // 3)
        df = pd.DataFrame({
            "event_dt": pd.date_range("2024-01-01", periods=n, freq="h"),
            "sig": rng.randn(n),
            "noise": rng.randn(n),
        })
        y = 2.0 * df["sig"].to_numpy() + rng.randn(n) * 0.2
        y_dirty = y.copy()
        y_dirty[3] = np.nan
        y_dirty[9] = np.inf

        # Calling analyze_risk with task="regression" and default scorer="accuracy"
        rep = analyze_risk(df, y_dirty, task="regression", cv=5, groups=groups_3, random_state=42)
        assert rep.temporal_leakage_gap is not None
        assert np.isfinite(rep.temporal_leakage_gap)
        pen = compute_risk_penalty(rep, cv_score_std=0.05)
        assert np.isfinite(pen) and 0.0 <= pen <= 1.0

        # Non-finite CV scores from cross_val_score return None (never nan)
        monkeypatch.setattr(r_mod, "cross_val_score", lambda *a, **kw: np.array([np.nan, np.nan]))
        gap_nan = _temporal_leakage_gap(
            df, y, scorer="accuracy", cv=3, task="regression", random_state=42, groups=groups_3
        )
        assert gap_nan is None

        # Even if a RiskReport manually carries NaN temporal_leakage_gap, compute_risk_penalty stays finite
        rr_nan = RiskReport(
            task="regression",
            target_type="regression",
            n_rows=n,
            n_cols=3,
            n_numeric=2,
            n_categorical=0,
            missing_ratio=0.0,
            temporal_leakage_gap=float("nan"),
        )
        pen_nan = compute_risk_penalty(rr_nan, cv_score_std=0.01)
        assert np.isfinite(pen_nan) and 0.0 <= pen_nan <= 1.0

    def test_analyze_risk_temporal_suppresses_stale_clamp_warning(self):
        rng = np.random.RandomState(42)
        n = 120
        groups_3 = np.repeat(["g0", "g1", "g2"], n // 3)
        df = pd.DataFrame({
            "event_dt": pd.date_range("2024-01-01", periods=n, freq="h"),
            "sig": rng.randn(n),
            "noise": rng.randn(n),
        })
        y = (df["sig"].to_numpy() > 0.0).astype(int)

        rep = analyze_risk(
            df, y, task="classification", cv=5, groups=groups_3, cv_strategy="temporal", random_state=42
        )
        assert rep.cv_strategy == "temporal"
        assert not any("clamped" in w for w in rep.warnings)
        assert not any("clamped" in note for note in rep.notes)

    def test_core_classifier_temporal_with_small_groups_preserves_5_folds(self):
        rng = np.random.RandomState(17)
        n = 120
        groups_3 = np.repeat(["g0", "g1", "g2"], n // 3)
        t = np.arange(n, dtype=float)
        X = pd.DataFrame({
            "x1": np.sin(t / 7.0) + rng.randn(n) * 0.2,
            "x2": rng.randn(n),
            "x3": rng.randn(n),
        })
        y = (X["x1"].to_numpy() > 0.0).astype(int)

        clf = ACDSOClassifier(
            cv=5,
            cv_strategy="temporal",
            max_candidates=3,
            enable_stacking=False,
            n_jobs=1,
            random_state=17,
        )
        clf.fit(X, y, groups=groups_3)

        assert clf.cv_strategy_ == "temporal"
        assert clf.audit_trail_["cv_strategy"] == "temporal"
        assert clf._engine.cv == 5
        assert clf.audit_trail_["candidates_evaluated"]
        for cand in clf.audit_trail_["candidates_evaluated"]:
            assert len(cand["cv_scores"]) == 5
        assert not any("clamped" in w for w in clf.risk_report_.get("warnings", []))
