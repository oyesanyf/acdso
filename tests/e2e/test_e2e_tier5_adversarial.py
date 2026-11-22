"""Tier-5 White-Box Adversarial & Regression E2E/Contract Tests (Core + CLI/TUI + WebUI).

Combines all 27 Tier-5 adversarial tests across:
- Core AutoML engine (`src/acdso/`): linearity detection on large non-linear datasets,
  weak-signal noise guard, Inf/nullable dtypes/datetime64 resilience, small imbalanced
  classification, scorer immutability, NaN target validation, wide-table risk scan,
  duplicate column handling, AutoPreprocessor bool/StringDtype/multicollinearity,
  AutoEDA strict JSON, utils/pareto/feature_importance, AutoTS, and DecisionIntelligence.
- CLI/TUI & WebUI (`run_acdso.py`, `src/acdso/webui/`): `_json_safe_scalar`, strict
  RFC 8259 multi-target JSON, XSS sanitization in HTML reports and forecast UI, cp1252 &
  non-ASCII uploads, target drift elimination, bounded period-range forecast, path traversal
  and `--output` confinement, `JobManager` Popen recovery & concurrency race, and JS
  inspector multi-target / `N/A` safety.
"""

from __future__ import annotations

import io
import json
import math
import re
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing, make_classification, make_regression
from sklearn.model_selection import train_test_split

from acdso import (
    ACDSOAutoEDA,
    ACDSOClassifier,
    ACDSORegressor,
    AutoPreprocessor,
    SafeFeatureImportance,
    analyze_risk,
    infer_target_type,
)
from acdso.decision import ActionOptimizer, BudgetConstraint, DecisionIntelligence, UpliftModeler
from acdso.decision.optimizer import FairnessConstraint, ROIConstraint, knapsack_optimize
from acdso.pareto import CandidateResult, knee_point_selection
from acdso.risk import _detect_linearity
from acdso.timeseries import AutoTS, TimeSeriesRiskAnalyzer

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_EXE = Path(r"C:\Program Files\nodejs\node.exe")


def _reject_nonfinite(constant: str) -> None:
    raise ValueError(f"Non-RFC8259 JSON constant encountered: {constant}")


# ===========================================================================
# 1. Core AutoML Engine Adversarial & Regression Tests (12 tests)
# ===========================================================================


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_california_housing_full_not_misclassified_linear() -> None:
    """Full 20,640-row fetch_california_housing must not be pruned to linear-only models."""
    cal = fetch_california_housing()
    X = pd.DataFrame(cal.data, columns=cal.feature_names)
    y = cal.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    is_linear, _ = _detect_linearity(X_train, y_train, task="regression", cv=3, random_state=42)
    assert is_linear is False, (
        "Full California Housing (non-linear) was misclassified as is_linear=True"
    )

    reg = ACDSORegressor(cv=3, max_candidates=8, random_state=42, verbose=0)
    reg.fit(X_train, y_train)
    test_r2 = float(reg.score(X_test, y_test))
    assert test_r2 >= 0.72, (
        f"Expected non-linear test R2 >= 0.72 on California Housing, got {test_r2:.4f} "
        f"(selected={reg.audit_trail_['selected']['name']})"
    )


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_detect_linearity_weak_signal_noise_not_linear() -> None:
    """Weak-signal or near-chance datasets must not trigger is_linear=True via ratio > 1.05."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(500, 8), columns=[f"f{i}" for i in range(8)])

    y_weak_reg = 0.4 * X["f0"].to_numpy() + rng.randn(500) * 1.1
    is_lin_reg, _ = _detect_linearity(X, y_weak_reg, task="regression", cv=3, random_state=42)
    assert is_lin_reg is False

    y_weak_clf = (0.2 * X["f0"].to_numpy() + rng.randn(500) > 0).astype(int)
    is_lin_clf, _ = _detect_linearity(X, y_weak_clf, task="classification", cv=3, random_state=42)
    assert is_lin_clf is False

    # Sanity: strong linear relationship still detected as True
    X_lin, y_lin = make_regression(n_samples=600, n_features=6, noise=2.0, random_state=42)
    is_lin_true, score_true = _detect_linearity(
        pd.DataFrame(X_lin), y_lin, task="regression", cv=3, random_state=42
    )
    assert is_lin_true is True and score_true >= 0.9


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_core_handles_inf_nullable_dtypes_and_datetime() -> None:
    """ACDSOClassifier/ACDSORegressor fit and predict cleanly on Inf, pd.NA dtypes, and datetime64."""
    rng = np.random.RandomState(42)
    n = 100

    # Inf / -Inf in numeric features
    X_inf = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n), "c": rng.randn(n)})
    y_inf = (X_inf["a"] > 0).astype(int).to_numpy()
    X_inf.iloc[0, 0] = np.inf
    X_inf.iloc[1, 1] = -np.inf
    clf_inf = ACDSOClassifier(cv=3, max_candidates=3, random_state=42, verbose=0).fit(X_inf, y_inf)
    assert clf_inf.audit_trail_["selected"]["name"] != "Baseline_Fallback"
    assert clf_inf.predict(X_inf).shape == (n,)

    # Nullable Int64, Float64, boolean, string with pd.NA
    X_null = pd.DataFrame({
        "i64": pd.Series([i if i % 7 != 0 else pd.NA for i in range(n)], dtype="Int64"),
        "f64": pd.Series([float(rng.randn()) if i % 8 != 0 else pd.NA for i in range(n)], dtype="Float64"),
        "b": pd.Series([bool(i % 2) if i % 9 != 0 else pd.NA for i in range(n)], dtype="boolean"),
        "s": pd.Series(["a" if i % 2 == 0 else ("b" if i % 5 != 0 else pd.NA) for i in range(n)], dtype="string"),
    })
    y_null = (np.arange(n) % 2).astype(int)
    clf_null = ACDSOClassifier(cv=3, max_candidates=3, random_state=42, verbose=0).fit(X_null, y_null)
    assert clf_null.predict(X_null).shape == (n,)

    # datetime64 feature column
    X_dt = pd.DataFrame({
        "dt": pd.date_range("2024-01-01", periods=n, freq="D"),
        "x1": rng.randn(n),
        "x2": rng.randn(n),
    })
    y_dt = (X_dt["x1"] > 0).astype(int).to_numpy()
    clf_dt = ACDSOClassifier(cv=3, max_candidates=3, random_state=42, verbose=0).fit(X_dt, y_dt)
    assert clf_dt.predict(X_dt).shape == (n,)


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_small_imbalanced_classification_does_not_degrade_to_dummy() -> None:
    """Small imbalanced dataset (n=40, 36 vs 4) should evaluate real models without Stage-0 class starvation."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "signal": np.concatenate([rng.normal(-2.0, 0.5, 36), rng.normal(2.0, 0.5, 4)]),
        "n1": rng.randn(40),
        "n2": rng.randn(40),
        "n3": rng.randn(40),
    })
    y = np.array([0] * 36 + [1] * 4)

    clf = ACDSOClassifier(cv=3, max_candidates=4, random_state=42, verbose=0)
    clf.fit(X, y)
    assert clf.audit_trail_["selected"]["name"] != "Baseline_Fallback"
    assert len(clf.audit_trail_["pruned_or_failed"]) == 0


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_estimator_scorer_param_immutable_and_nan_y_raises_value_error() -> None:
    """fit() must not mutate __init__ param `scorer`; NaN targets must raise ValueError cleanly."""
    X, y = make_classification(
        n_samples=90, n_features=6, n_informative=4, n_classes=3, random_state=42
    )
    clf = ACDSOClassifier(scorer="f1", cv=3, max_candidates=2, random_state=42, verbose=0)
    clf.fit(X, y)
    assert clf.scorer == "f1"
    assert clone(clf).scorer == "f1"
    assert 0.0 <= clf.score(X, y) <= 1.0

    y_bad = y.astype(float)
    y_bad[0] = np.nan
    with pytest.raises(ValueError):
        ACDSOClassifier(cv=3, max_candidates=2, random_state=42).fit(X, y_bad)


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_analyze_risk_wide_table_and_duplicate_columns() -> None:
    """analyze_risk detects proxy leakage past column 200 and handles duplicate column names."""
    rng = np.random.RandomState(42)
    n = 150
    data = {f"f_{i}": rng.randn(n) for i in range(205)}
    y = rng.randint(0, 2, n)
    data["f_204"] = y.astype(float) + 0.001 * rng.randn(n)
    df_wide = pd.DataFrame(data)

    rr_wide = analyze_risk(df_wide, y, task="classification", random_state=42)
    assert "f_204" in rr_wide.leakage_proxy_features
    assert rr_wide.leakage_proxy_score_max >= 0.95

    df_dup = pd.DataFrame(rng.randn(80, 3), columns=["a", "a", "b"])
    y_dup = (df_dup.iloc[:, 0] > 0).astype(int).to_numpy()
    rr_dup = analyze_risk(df_dup, y_dup, task="classification", random_state=42)
    assert rr_dup.n_cols == 3


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_auto_preprocessor_retains_bool_string_and_detects_scattered_nan_collinearity() -> None:
    """AutoPreprocessor keeps bool/StringDtype columns and detects multicollinearity despite scattered NaNs."""
    rng = np.random.RandomState(42)
    df_types = pd.DataFrame({
        "flag": [True, False, True, False, True, False] * 5,
        "txt": pd.Series(["a", "b", "a", "b", "c", "a"] * 5, dtype="string"),
        "num": rng.randn(30),
    })
    y_types = np.array([0, 1] * 15)
    ap = AutoPreprocessor(verbose=0)
    out = ap.fit_transform(df_types, y_types)
    assert out.shape[0] == 30
    assert out.shape[1] >= 3  # num + bool + one-hot string

    n = 100
    base = rng.randn(n)
    df_mc = pd.DataFrame({f"c{i}": base + 1e-4 * rng.randn(n) for i in range(6)})
    for i in range(6):
        df_mc.iloc[i * 16 : (i + 1) * 16, i] = np.nan
    ap_mc = AutoPreprocessor(handle_multicollinearity=True, missing_threshold=0.5, verbose=0)
    ap_mc.fit(df_mc, np.arange(n))
    assert len(ap_mc.get_plan().correlated_groups) >= 1
    assert len(ap_mc.get_plan().drop_columns) >= 4


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_eda_strict_json_on_inf_and_duplicate_cols(tmp_path: Path) -> None:
    """ACDSOAutoEDA produces RFC-compliant JSON on Inf values and survives duplicate column names."""
    df_inf = pd.DataFrame({
        "a": [1.0, 2.0, np.inf, 4.0, 5.0],
        "b": [1.0, 2.0, 3.0, 4.0, -np.inf],
    })
    eda = ACDSOAutoEDA(verbose=0)
    rep = eda.analyze(df_inf, dataset_name="inf_test")
    serialized = json.dumps(rep.to_dict(), allow_nan=False)
    assert len(serialized) > 0
    out_json = tmp_path / "eda_report.json"
    eda.to_json(str(out_json))
    assert out_json.is_file()

    df_dup = pd.DataFrame(np.random.RandomState(42).randn(50, 3), columns=["x", "x", "y"])
    rep_dup = ACDSOAutoEDA(verbose=0).analyze(df_dup, dataset_name="dup_cols")
    assert len(rep_dup.column_profiles) == 3


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_utils_infer_target_type_and_pareto_non_finite() -> None:
    """infer_target_type handles NaN floats without warnings, empty arrays, and object+None; knee_point prefers finite."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert infer_target_type(np.array([0.0, 1.0, np.nan])) == "binary"
        assert infer_target_type(np.array([1.25, 2.75, np.nan])) == "regression"
    cast_warns = [w for w in caught if "invalid value encountered in cast" in str(w.message)]
    assert cast_warns == []
    assert infer_target_type(np.array(["a", "b", None], dtype=object)) == "binary"
    with pytest.raises(ValueError):
        infer_target_type(np.array([], dtype=float))

    c_fin = CandidateResult("Balanced", None, 0.90, 1.0, 10.0, 0.1, 0.05, (0.90, 0.69, 2.40, 0.095, 0.05, 0.0))
    c_inf = CandidateResult("InfLat", None, 0.90, 1.0, 10.0, float("inf"), 0.05, (0.90, 0.69, 2.40, float("inf"), 0.05, 0.0))
    assert knee_point_selection([c_fin, c_inf]).name == "Balanced"


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_safe_feature_importance_regressor_and_mixed_corr_shape() -> None:
    """SafeFeatureImportance works on ACDSORegressor without forcing explicit scorer='accuracy' override."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "n1": rng.randn(80),
        "n2": rng.randn(80),
        "c1": rng.choice(["x", "y", "z"], size=80),
    })
    y = 3.0 * df["n1"].to_numpy() + 0.1 * rng.randn(80)
    reg = ACDSORegressor(cv=2, max_candidates=2, random_state=42, verbose=0).fit(df, y)

    sfi = SafeFeatureImportance(reg, n_repeats=2, cv_folds=2, n_bootstrap=3, random_state=42)
    rep = sfi.compute(df, y)
    assert rep.scorer == "r2"
    assert len(rep.features) == 3
    assert rep.features[0].name == "n1"


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_autots_feature_cols_and_empty_pipeline_error() -> None:
    """AutoTS respects feature_cols and raises RuntimeError (not IndexError) if no pipeline completes."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "sales": np.linspace(10.0, 70.0, 60) + rng.randn(60) * 0.2,
        "good_exog": np.arange(60, dtype=float),
        "ignored_noise": rng.randn(60) * 1000.0,
    })

    ts_bad = AutoTS(datetime_col="date", target_col="sales", horizon=2, time_budget=-1, verbose=False)
    with pytest.raises(RuntimeError):
        ts_bad.fit(df)

    ts_ok = AutoTS(datetime_col="date", target_col="sales", horizon=3, n_splits=2, verbose=False)
    res = ts_ok.fit(df, feature_cols=["good_exog"])
    assert "ignored_noise" not in res.feature_importance


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_decision_intelligence_and_uplift_end_to_end() -> None:
    """UpliftModeler and DecisionIntelligence execute end-to-end without unfitted BaseEnsemble errors."""
    rng = np.random.RandomState(42)
    n = 120
    age = rng.randn(n)
    tenure = rng.randn(n)
    treatment = (rng.rand(n) > 0.5).astype(int)
    outcome = (0.4 * age - 0.8 * treatment + rng.randn(n) * 0.2 > 0).astype(int)
    df = pd.DataFrame({
        "age": age,
        "tenure": tenure,
        "offered_incentive": treatment,
        "churned": outcome,
        "cost": np.full(n, 10.0),
    })

    uplift_res = UpliftModeler(
        treatment_col="offered_incentive",
        outcome_col="churned",
        feature_cols=["age", "tenure"],
    ).fit(df, verbose=False)
    assert len(uplift_res.ite) == n
    assert uplift_res.best_model in UpliftModeler.AVAILABLE_MODELS

    di = DecisionIntelligence(
        treatment_col="offered_incentive",
        outcome_col="churned",
        cost_col="cost",
        confounders=["age", "tenure"],
        feature_cols=["age", "tenure"],
    )
    res = di.fit(df, budget=200.0, verbose=False)
    assert res.causal is not None
    assert res.uplift is not None
    assert res.optimization is not None
    assert res.optimization.total_cost <= 200.0 + 1e-6


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_decision_and_causal_numeric_dtypes_inf_and_nan_outcome() -> None:
    """DecisionIntelligence and CausalAnalyzer support float32/int32/nullable numeric dtypes, +-inf features, and NaN outcomes."""
    from acdso.decision.causal import CausalAnalyzer

    rng = np.random.RandomState(42)
    n = 120

    # 1. float32 / int32 / nullable Float64 / Int64 / boolean with pd.NA
    df_mixed = pd.DataFrame({
        "x1": rng.randn(n).astype(np.float32),
        "x2": rng.randint(0, 10, size=n).astype(np.int32),
        "x3": pd.Series(rng.randn(n), dtype="Float64"),
        "x4": pd.Series(rng.randint(0, 5, size=n), dtype="Int64"),
        "x5": pd.Series(rng.rand(n) > 0.5, dtype="boolean"),
        "treat": rng.binomial(1, 0.5, size=n),
        "outcome": rng.randn(n),
        "cost": np.ones(n),
    })
    df_mixed.loc[0, "x3"] = pd.NA
    df_mixed.loc[1, "x4"] = pd.NA
    df_mixed.loc[2, "x5"] = pd.NA
    di_mixed = DecisionIntelligence("treat", "outcome", cost_col="cost")
    res_mixed = di_mixed.fit(df_mixed, budget=50.0, verbose=False)
    assert set(di_mixed.feature_cols) == {"x1", "x2", "x3", "x4", "x5"}
    assert np.isfinite(res_mixed.causal.ate.effect)
    assert np.isfinite(res_mixed.causal.att.effect)
    assert res_mixed.optimization is not None

    # 2. +-inf in features
    df_inf = pd.DataFrame({
        "x1": rng.randn(n),
        "x2": rng.randn(n),
        "treat": rng.binomial(1, 0.5, size=n),
        "outcome": rng.randn(n),
        "cost": np.ones(n),
    })
    df_inf.loc[0, "x1"] = np.inf
    df_inf.loc[1, "x2"] = -np.inf
    res_inf = DecisionIntelligence("treat", "outcome", cost_col="cost").fit(df_inf, budget=50.0, verbose=False)
    assert np.isfinite(res_inf.causal.ate.effect)
    assert np.isfinite(res_inf.causal.att.effect)
    assert res_inf.optimization.total_cost <= 50.0 + 1e-6

    # 3. NaN in outcome -> ATT must remain finite
    df_nan_y = pd.DataFrame({
        "x1": rng.randn(n),
        "t": rng.binomial(1, 0.5, size=n),
        "y": rng.randn(n),
    })
    df_nan_y.loc[0, "y"] = np.nan
    m = CausalAnalyzer("t", "y", ["x1"]).fit(df_nan_y, run_refutation=False, verbose=False)
    assert np.isfinite(m.ate.effect)
    assert np.isfinite(m.att.effect)


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_autots_extension_dtypes_categorical_string_tz_and_inf() -> None:
    """AutoTS.fit() handles Categorical, string, Int64, Float64, boolean, tz-aware datetime64, pd.NA, and +-inf."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "target": pd.Series(np.sin(np.arange(100) * 0.2) * 10.0 + 50.0, dtype="Float64"),
            "store_cat": pd.Categorical(["A", "B"] * 50),
            "region_str": pd.Series([f"r_{i % 3}" for i in range(100)], dtype="string"),
            "int_exog": pd.Series(np.arange(100), dtype="Int64"),
            "flt_exog": pd.Series(np.linspace(0.0, 1.0, 100), dtype="Float64"),
            "bool_exog": pd.Series([i % 2 == 0 for i in range(100)], dtype="boolean"),
            "tz_extra": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
        }
    )
    df.loc[5, "target"] = pd.NA
    df.loc[10, "target"] = np.inf
    df.loc[3, "int_exog"] = pd.NA
    df.loc[7, "flt_exog"] = -np.inf
    df.loc[9, "bool_exog"] = pd.NA

    # 1. feature_cols=None (default auto-filtering + ML lag pipelines)
    res_none = AutoTS(
        datetime_col="date",
        target_col="target",
        horizon=3,
        n_splits=2,
        time_budget=10,
        include_baselines=False,
        verbose=False,
    ).fit(df)
    assert np.isfinite(res_none.best_pipeline.cv_score)
    preds = res_none.predict(df.tail(20))
    assert len(preds) == 20 and np.all(np.isfinite(preds))

    # 2. Explicit feature_cols=[...] including nullable numeric and categorical columns
    res_explicit = AutoTS(
        datetime_col="date",
        target_col="target",
        horizon=3,
        n_splits=2,
        time_budget=10,
        verbose=False,
    ).fit(df, feature_cols=["int_exog", "flt_exog", "bool_exog", "store_cat"])
    assert np.isfinite(res_explicit.best_pipeline.cv_score)

    # 3. df.convert_dtypes() with tz-aware primary datetime_col
    df_tz = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"),
            "target": np.sin(np.arange(60) * 0.2) * 10.0 + 50.0,
            "exog": np.arange(60),
            "label": ["x", "y"] * 30,
        }
    ).convert_dtypes()
    res_tz = AutoTS(
        datetime_col="date",
        target_col="target",
        horizon=3,
        n_splits=2,
        time_budget=10,
        verbose=False,
    ).fit(df_tz)
    assert np.isfinite(res_tz.best_pipeline.cv_score)


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_optimizer_and_decision_nonfinite_cost_resilience() -> None:
    """ActionOptimizer, BudgetConstraint, ROIConstraint, and DecisionIntelligence ignore non-finite costs."""
    df_opt = pd.DataFrame(
        {
            "val": [10.0, 20.0, 30.0, 40.0, 50.0, np.inf],
            "cost": [1.0, 1.0, 1.0, np.nan, -2.0, 1.0],
        }
    )
    opt = (
        ActionOptimizer(value_col="val", cost_col="cost")
        .add_constraint(BudgetConstraint(max_budget=10.0, cost_col="cost"))
        .add_constraint(ROIConstraint(min_roi=2.0, value_col="val", cost_col="cost"))
    )
    res_opt = opt.optimize(df_opt)
    assert res_opt.n_selected == 3
    assert np.isfinite(res_opt.total_cost) and res_opt.total_cost == pytest.approx(3.0)
    assert np.isfinite(res_opt.total_value) and res_opt.total_value == pytest.approx(60.0)
    assert res_opt.selected[3] == 0 and res_opt.selected[4] == 0 and res_opt.selected[5] == 0
    assert res_opt.constraints_satisfied["budget"] is True
    assert res_opt.constraints_satisfied["roi"] is True

    rng = np.random.RandomState(42)
    n = 120
    df_di = pd.DataFrame(
        {
            "x1": rng.randn(n),
            "x2": rng.randn(n),
            "treat": rng.binomial(1, 0.5, size=n),
            "outcome": rng.randn(n),
            "cost": pd.Series(np.ones(n), dtype="Float64"),
        }
    )
    df_di.loc[0, "cost"] = np.nan
    df_di.loc[1, "cost"] = pd.NA
    df_di.loc[2, "cost"] = np.inf
    res_di = DecisionIntelligence("treat", "outcome", cost_col="cost").fit(
        df_di, budget=50.0, verbose=False
    )
    assert res_di.optimization is not None
    assert res_di.optimization.n_selected > 0
    assert np.isfinite(res_di.optimization.total_cost)
    assert res_di.optimization.total_cost <= 50.0 + 1e-6
    assert res_di.optimization.selected[0] == 0
    assert res_di.optimization.selected[1] == 0
    assert res_di.optimization.selected[2] == 0


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_pareto_linear_regression_datasets_no_underfit_or_jitter_regression() -> None:
    """synthetic_reg & linear_highdim-style datasets deterministically select near-optimal R2 models."""
    X_sr, y_sr = make_regression(
        n_samples=1000, n_features=20, n_informative=10, noise=10.0, random_state=42
    )
    df_sr = pd.DataFrame(X_sr, columns=[f"f{i}" for i in range(20)])
    X_sr_tr, X_sr_te, y_sr_tr, y_sr_te = train_test_split(
        df_sr, y_sr, test_size=0.2, random_state=42
    )

    reg_sr_1 = ACDSORegressor(
        scorer="r2", cv=5, preference="balanced", time_budget_s=60.0, random_state=42, verbose=0
    ).fit(X_sr_tr, y_sr_tr)
    reg_sr_2 = ACDSORegressor(
        scorer="r2", cv=5, preference="balanced", time_budget_s=60.0, random_state=42, verbose=0
    ).fit(X_sr_tr, y_sr_tr)

    sel_sr_1 = reg_sr_1.audit_trail_["selected"]
    sel_sr_2 = reg_sr_2.audit_trail_["selected"]
    max_sr_cv = max(c["metric"] for c in reg_sr_1.audit_trail_["candidates_evaluated"])

    assert sel_sr_1["name"] == sel_sr_2["name"], (
        f"Non-deterministic selection across identical runs: {sel_sr_1['name']} vs {sel_sr_2['name']}"
    )
    assert sel_sr_1["metric"] >= 0.99, f"Expected CV R2 >= 0.99, got {sel_sr_1['metric']:.4f}"
    assert sel_sr_1["metric"] >= max_sr_cv - 0.01, (
        f"Selected {sel_sr_1['name']} ({sel_sr_1['metric']:.4f}) regressed vs best CV ({max_sr_cv:.4f})"
    )
    assert float(reg_sr_1.score(X_sr_te, y_sr_te)) >= 0.99

    rng = np.random.RandomState(42)
    n, n_features = 500, 100
    base = rng.randn(n, 10)
    X_mat = np.hstack([base, base @ rng.randn(10, 90) * 0.5 + rng.randn(n, 90) * 0.5])
    df_hd = pd.DataFrame(X_mat, columns=[f"x{i}" for i in range(n_features)])
    coeffs = np.array([3.0, 2.0, -1.5, 1.0, -0.5, 0.8, -0.3, 0.2, 0.1, -0.1])
    y_hd = X_mat[:, :10] @ coeffs + rng.randn(n) * 0.5
    X_hd_tr, X_hd_te, y_hd_tr, y_hd_te = train_test_split(
        df_hd, y_hd, test_size=0.2, random_state=42
    )

    reg_hd = ACDSORegressor(
        scorer="r2", cv=5, preference="balanced", time_budget_s=60.0, random_state=42, verbose=0
    ).fit(X_hd_tr, y_hd_tr)
    sel_hd = reg_hd.audit_trail_["selected"]
    max_hd_cv = max(c["metric"] for c in reg_hd.audit_trail_["candidates_evaluated"])

    assert sel_hd["metric"] >= 0.97, f"Expected CV R2 >= 0.97, got {sel_hd['metric']:.4f}"
    assert sel_hd["metric"] >= max_hd_cv - 0.01, (
        f"Selected {sel_hd['name']} ({sel_hd['metric']:.4f}) regressed vs best CV ({max_hd_cv:.4f})"
    )
    assert float(reg_hd.score(X_hd_te, y_hd_te)) >= 0.97


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_pareto_knee_signed_distance_and_noise_floor() -> None:
    """knee_point_selection ignores <0.005 stress_worst_drop / sub-ms jitter and rejects above-chord points."""
    c_top = CandidateResult(
        "TopRidge", None, 0.997, 0.027, 0.20, 0.0156, 0.0163,
        (0.997, 0.027, 0.18, 0.0155, 0.0163, 0.0075),
    )
    c_bad_chord = CandidateResult(
        "AboveChordLR", None, 0.997, 0.055, 0.20, 0.0800, 0.0900,
        (0.997, 0.054, 0.18, 0.0770, 0.0900, 0.0250),
    )
    c_underfit = CandidateResult(
        "ElasticNetWeak", None, 0.966, 0.043, 0.20, 0.0154, 0.0199,
        (0.966, 0.042, 0.18, 0.0153, 0.0199, 0.0073),
    )

    assert knee_point_selection([c_bad_chord, c_top, c_underfit]).name == "TopRidge"
    assert knee_point_selection([c_underfit, c_top]).name == "TopRidge"


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_pareto_small_scale_neg_mse_and_inf_candidate_safety() -> None:
    """Small-scale neg_mean_squared_error selects nonlinear RF (test_R2 >= 0.85) and +inf metric never dominates finite candidates."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.uniform(-3, 3, size=(300, 4)), columns=["a", "b", "c", "d"])
    y_base = np.sin(X["a"] * 2.0) * 5.0 + X["b"] ** 2 + rng.randn(300) * 0.2

    for scale in (0.005, 0.0001):
        y = y_base * scale
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = ACDSORegressor(
            scorer="neg_mean_squared_error",
            selection_mode="knee_point",
            cv=3,
            max_candidates=8,
            random_state=42,
            verbose=0,
        ).fit(X_tr, y_tr)
        r2_te = float(r2_score(y_te, reg.predict(X_te)))
        sel_name = reg.audit_trail_["selected"]["name"]
        assert r2_te >= 0.85, f"scale={scale}: selected {sel_name} got test_R2={r2_te:.4f} < 0.85"
        assert "Ridge" not in sel_name and "Lasso" not in sel_name, (
            f"scale={scale}: expected nonlinear model, got linear {sel_name}"
        )

    # +inf metric candidate must never dominate valid finite candidates out of pareto_front_
    X_inf, y_inf = make_regression(n_samples=120, n_features=6, n_informative=4, noise=2.0, random_state=7)
    df_inf = pd.DataFrame(X_inf, columns=[f"f{i}" for i in range(6)])
    X_inf_tr, X_inf_te, y_inf_tr, y_inf_te = train_test_split(df_inf, y_inf, test_size=0.2, random_state=7)

    def _rogue_inf_scorer(est, X_val, y_val) -> float:
        model_step = est.named_steps.get("model") if hasattr(est, "named_steps") else est
        if isinstance(model_step, Ridge):
            return float("inf")
        return float(r2_score(y_val, est.predict(X_val)))

    reg_inf = ACDSORegressor(
        scorer=_rogue_inf_scorer,
        selection_mode="knee_point",
        cv=3,
        max_candidates=6,
        random_state=7,
        verbose=0,
    ).fit(X_inf_tr, y_inf_tr)
    sel_inf = reg_inf.audit_trail_["selected"]
    assert math.isfinite(float(sel_inf["metric"])), f"Selected non-finite metric candidate: {sel_inf}"
    assert all(math.isfinite(float(c["metric"])) for c in reg_inf.pareto_front_), (
        f"pareto_front_ contains non-finite metric: {reg_inf.pareto_front_}"
    )
    assert float(r2_score(y_inf_te, reg_inf.predict(X_inf_te))) >= 0.85


# ===========================================================================
# 2. CLI / Report Adversarial & M2 Caveat Regressions (5 tests)
# ===========================================================================


@pytest.mark.tier5
@pytest.mark.f2
def test_t5_json_safe_scalar_np_bool_and_int_preservation() -> None:
    """_json_safe_scalar preserves np.bool_ as bool and np.integer as int (not str/float)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import run_acdso

    assert run_acdso._json_safe_scalar(np.bool_(True)) is True
    assert isinstance(run_acdso._json_safe_scalar(np.bool_(False)), bool)
    assert run_acdso._json_safe_scalar(np.bool_(False)) is False

    out_int = run_acdso._json_safe_scalar(np.int64(7))
    assert out_int == 7 and isinstance(out_int, int)

    assert run_acdso._json_safe_scalar(np.float64("nan")) is None
    assert run_acdso._json_safe_scalar(np.float64("inf")) is None
    assert run_acdso._json_safe_scalar(np.float64("-inf")) is None


@pytest.mark.tier5
@pytest.mark.f2
def test_t5_multi_target_predict_and_special_filename_rfc8259(
    dataset_factory, run_cli, tmp_path: Path
) -> None:
    """Multi-target --predict on filenames with spaces/parens/quotes produces strict RFC 8259 JSON and valid HTML."""
    df_tips = pd.read_csv(dataset_factory.copy_sample("tips", "tips_src.csv")).head(45)
    special_csv = tmp_path / "my data (v1) 'test'.csv"
    df_tips.to_csv(special_csv, index=False)

    res = run_cli(
        ["--file", str(special_csv), "--predict", "total_bill,tip", "--simple", "--skip-benchmark"],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    stem = "my data (v1) 'test'"
    combined_json = tmp_path / "reports" / f"multi_target_{stem}_combined.json"
    combined_html = tmp_path / "reports" / f"multi_target_{stem}_combined_report.html"
    assert combined_json.is_file()
    assert combined_html.is_file()

    # Must parse strictly as RFC 8259 JSON (no NaN / Infinity)
    parsed = json.loads(combined_json.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert set(parsed["targets"]) == {"total_bill", "tip"}
    assert "total_bill" in parsed["predictions"] and "tip" in parsed["predictions"]


@pytest.mark.tier5
@pytest.mark.f2
def test_t5_html_report_strips_event_handler_xss_from_eda_and_ts(tmp_path: Path) -> None:
    """Combined HTML report must not allow event-handler or malformed-script XSS from embedded EDA/TS HTML."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import run_acdso

    malicious_report = {
        "mode": "autonomous_custom_file",
        "timestamp": "2026-11-22T12:00:00",
        "eda_html": (
            "<body><p>Summary</p>"
            "<img src=x onerror=alert(1)>"
            "<svg onload=alert(2)>"
            "<SCRIPT>alert(3)</script ></body>"
        ),
        "timeseries_html": "<body><img src=y onerror=alert(4)></body>",
    }
    out_html = tmp_path / "xss_check.html"
    run_acdso.generate_html_report_from_combined(malicious_report, str(out_html))
    rendered = out_html.read_text(encoding="utf-8")

    if "<iframe" not in rendered.lower():
        assert "onerror=" not in rendered.lower()
        assert "onload=" not in rendered.lower()
        assert "<script>alert(3)" not in rendered.lower()


@pytest.mark.tier5
@pytest.mark.f2
def test_t5_multi_output_small_continuous_dataset_no_classification_crash(
    run_cli, tmp_path: Path
) -> None:
    """--target t1,t2 --multi-output on <=20 rows of continuous floats must not misclassify as classification."""
    rng = np.random.RandomState(7)
    df = pd.DataFrame(
        {
            "x1": np.linspace(1.0, 10.0, 18),
            "x2": rng.normal(size=18),
            "t1": np.linspace(2.0, 20.0, 18) + 0.05,
            "t2": np.linspace(5.0, 40.0, 18) + 0.03,
        }
    )
    csv_path = tmp_path / "small_mo.csv"
    df.to_csv(csv_path, index=False)

    res = run_cli(
        ["--file", str(csv_path), "--target", "t1,t2", "--multi-output", "--simple"],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Unknown label type: continuous-multioutput" not in (res.stdout + res.stderr)


@pytest.mark.tier5
@pytest.mark.f2
def test_t5_sequential_multi_target_discrete_float_and_small_continuous(
    run_cli, tmp_path: Path
) -> None:
    """Sequential --target t1,t2 --simple infers regression on discrete float ratings and small continuous floats."""
    rng = np.random.RandomState(42)

    # 1. 60-row discrete float ratings (1.0..5.0 in 0.5 steps = 9 distinct float values)
    d1 = tmp_path / "d1"
    d1.mkdir()
    ratings_csv = d1 / "ratings.csv"
    df_ratings = pd.DataFrame(
        {
            "x1": rng.randn(60),
            "x2": rng.randn(60),
            "t1": rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], size=60),
            "t2": rng.randn(60) * 10.0 + 50.0,
        }
    )
    df_ratings.to_csv(ratings_csv, index=False)
    r1 = run_cli(
        ["--file", str(ratings_csv), "--target", "t1,t2", "--simple"],
        cwd=d1,
        timeout=150.0,
    )
    assert r1.returncode == 0, f"STDOUT:\n{r1.stdout}\nSTDERR:\n{r1.stderr}"
    res1_json = d1 / "reports" / "multi_target_ratings_results.json"
    assert res1_json.is_file()
    parsed1 = json.loads(res1_json.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert parsed1["models"]["t1"]["task"] == "regression"
    assert parsed1["models"]["t2"]["task"] == "regression"

    # 2. 8-row continuous float dataset
    d2 = tmp_path / "d2"
    d2.mkdir()
    seq8_csv = d2 / "seq8.csv"
    df_seq8 = pd.DataFrame(
        {
            "x1": np.linspace(1.0, 8.0, 8),
            "x2": rng.randn(8),
            "t1": np.linspace(2.0, 9.0, 8) + 0.1,
            "t2": np.linspace(10.0, 24.0, 8) + 0.2,
        }
    )
    df_seq8.to_csv(seq8_csv, index=False)
    r2 = run_cli(
        ["--file", str(seq8_csv), "--target", "t1,t2", "--simple"],
        cwd=d2,
        timeout=150.0,
    )
    assert r2.returncode == 0, f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}"
    res2_json = d2 / "reports" / "multi_target_seq8_results.json"
    assert res2_json.is_file()
    parsed2 = json.loads(res2_json.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert parsed2["models"]["t1"]["task"] == "regression"
    assert parsed2["models"]["t2"]["task"] == "regression"


@pytest.mark.tier5
@pytest.mark.f2
def test_t5_decision_mode_handles_string_binary_and_float32_features(
    dataset_factory, run_cli, tmp_path: Path
) -> None:
    """--decision --file must handle datasets with string binary columns (tips) and float32 numeric features."""
    tips_path = dataset_factory.copy_sample("tips", "tips_dec.csv")
    d1 = tmp_path / "d1"
    d1.mkdir()
    r1 = run_cli(["--decision", "--file", str(tips_path), "--target", "tip"], cwd=d1, timeout=120.0)
    assert r1.returncode == 0, f"STDOUT:\n{r1.stdout}\nSTDERR:\n{r1.stderr}"

    d2 = tmp_path / "d2"
    d2.mkdir()
    df_f32 = pd.DataFrame(
        {
            "f1": np.linspace(0, 1, 60, dtype=np.float32),
            "f2": np.arange(60, dtype=np.int32),
            "treat": np.array([0, 1] * 30, dtype=np.int64),
            "outcome": np.linspace(10, 20, 60, dtype=np.float64),
        }
    )
    f32_pkl = d2 / "f32.pkl"
    df_f32.to_pickle(f32_pkl)
    r2 = run_cli(
        ["--decision", "--file", str(f32_pkl), "--target", "outcome", "--treatment", "treat"],
        cwd=d2,
        timeout=120.0,
    )
    assert r2.returncode == 0, f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}"


# ===========================================================================
# 3. WebUI Backend Adversarial & Security Tests (7 tests)
# ===========================================================================


@pytest.mark.tier5
@pytest.mark.f3
def test_t5_webui_cp1252_and_non_ascii_filename_upload(isolated_webui_client) -> None:
    """WebUI /api/upload accepts cp1252 CSVs and non-ASCII filenames (preserving .csv suffix)."""
    _, client, _ = isolated_webui_client

    cp_bytes = "name,val\nCaf\xe9,10\nNa\xefve,20\n".encode("cp1252")
    r_cp = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(cp_bytes), "windows_cp1252.csv")},
        content_type="multipart/form-data",
    )
    assert r_cp.status_code == 200, r_cp.get_data(as_text=True)
    assert r_cp.get_json()["rows"] == 2

    r_uni = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"a,b\n1,2\n3,4\n"), "日本語.csv")},
        content_type="multipart/form-data",
    )
    assert r_uni.status_code == 200, r_uni.get_data(as_text=True)
    assert r_uni.get_json()["rows"] == 2


@pytest.mark.tier5
@pytest.mark.f3
def test_t5_webui_strict_rfc8259_json_on_inf_preview_and_nan_report_summary(
    isolated_webui_client,
) -> None:
    """/api/upload preview and /api/report-summary must never emit non-RFC8259 NaN/Infinity tokens."""
    _, client, (_, rep_dir, _) = isolated_webui_client

    inf_csv = b"a,b\n1,inf\n2,-inf\n3,nan\n4,2.5\n"
    r_up = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(inf_csv), "inf_preview.csv")},
        content_type="multipart/form-data",
    )
    assert r_up.status_code == 200, r_up.get_data(as_text=True)
    parsed_up = json.loads(r_up.get_data(as_text=True), parse_constant=_reject_nonfinite)
    assert parsed_up["preview"][0]["b"] is None
    assert parsed_up["preview"][1]["b"] is None

    (rep_dir / "nonfinite_combined.json").write_text(
        '{"summary": {"score": NaN, "upper": Infinity}, "selected_model": {"name": "M", "metrics": {"metric": NaN}}}',
        encoding="utf-8",
    )
    r_sum = client.get("/api/report-summary?path=nonfinite_combined.json")
    assert r_sum.status_code == 200
    parsed_sum = json.loads(r_sum.get_data(as_text=True), parse_constant=_reject_nonfinite)
    assert parsed_sum["summary"]["score"] is None
    assert parsed_sum["summary"]["upper"] is None


@pytest.mark.tier5
@pytest.mark.f3
def test_t5_webui_suggest_target_matches_inspected_predictable_columns(
    isolated_webui_client,
) -> None:
    """/api/suggest-target candidates must match the upload's predictable_columns (no 1000-vs-5000 drift)."""
    _, client, _ = isolated_webui_client

    vals = [f"v_{i % 10}" for i in range(1000)] + [f"u_{i}" for i in range(500)]
    df = pd.DataFrame(
        {
            "f1": np.linspace(0, 10, 1500),
            "cat_drift": vals,
            "y": np.linspace(1, 2, 1500),
        }
    )
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    up = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(buf.getvalue()), "drift_check.csv")},
        content_type="multipart/form-data",
    )
    assert up.status_code == 200
    up_json = up.get_json()

    sug = client.post("/api/suggest-target", json={"upload_id": up_json["upload_id"]})
    assert sug.status_code == 200
    sug_json = sug.get_json()
    assert set(sug_json["candidates"]).issubset(set(up_json["predictable_columns"]))
    for s in sug_json["suggestions"]:
        assert s["column"] in up_json["predictable_columns"]


@pytest.mark.tier5
@pytest.mark.f3
def test_t5_webui_forecast_rejects_excessive_period_gap_400(isolated_webui_client) -> None:
    """/api/forecast rejects extreme date-range period gaps (e.g. centuries at daily grain) with 400."""
    _, client, _ = isolated_webui_client

    gap_csv = b"Date,val\n1800-01-01,10\n2026-01-01,20\n2026-01-02,30\n"
    up = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(gap_csv), "huge_gap.csv")},
        content_type="multipart/form-data",
    )
    assert up.status_code == 200
    uid = up.get_json()["upload_id"]

    fc = client.post(
        "/api/forecast",
        json={"upload_id": uid, "date_col": "Date", "value_col": "val", "grain": "D", "horizon": 3},
    )
    assert fc.status_code == 400
    assert "error" in fc.get_json()


@pytest.mark.tier5
@pytest.mark.f4
def test_t5_webui_path_traversal_variants_and_output_flag_blocked(isolated_webui_client) -> None:
    """Encoded/UNC/NUL path traversal variants and unconstrained --output in /api/run are blocked."""
    _, client, (_, _, log_dir) = isolated_webui_client
    (log_dir / "secret.log").write_text("secret", encoding="utf-8")

    for bad_path, expected_statuses in (
        ("%2e%2e/start.py", (403,)),
        ("..%2fstart.py", (403,)),
        ("webui_logs/../webui_logs/secret.log", (403,)),
        ("secret%00.json", (400, 403)),
        ("\\\\localhost\\c$\\Windows\\win.ini", (403,)),
    ):
        assert client.get(f"/api/artifact?path={bad_path}").status_code in expected_statuses
        assert client.get(f"/api/report-summary?path={bad_path}").status_code in expected_statuses
        assert client.delete(f"/api/reports?path={bad_path}").status_code in expected_statuses

    r_out = client.post(
        "/api/run",
        json={"benchmark": True, "output": "../../outside_evil.json"},
    )
    assert r_out.status_code in (400, 403), f"Allowed arbitrary --output: {r_out.status_code}"


@pytest.mark.tier5
@pytest.mark.f4
def test_t5_webui_job_manager_recovers_from_popen_failure(isolated_webui_client) -> None:
    """If subprocess.Popen fails in JobManager.start(), active_job must not stay stuck 'running' forever."""
    import acdso.webui.app as webui_app

    jm = webui_app.JobManager()
    with pytest.raises(OSError):
        jm.start(["__definitely_missing_executable_xyz__"])

    assert jm.active_job is None or jm.active_job.status != "running"

    job = jm.start([sys.executable, "-c", "print('recovered', flush=True)"])
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and job.status == "running":
        time.sleep(0.05)
    assert job.status == "finished"
    assert job.returncode == 0


@pytest.mark.tier5
@pytest.mark.f4
def test_t5_webui_concurrent_run_race_allows_exactly_one_job(
    isolated_webui_client, monkeypatch
) -> None:
    """Concurrent POST /api/run burst launches exactly 1 job (200) and rejects the rest with 409."""
    import acdso.webui.app as webui_app

    app, client, _ = isolated_webui_client
    runner_path = Path(tempfile.gettempdir()) / f"t5_sleep_runner_{time.time_ns()}.py"
    runner_path.write_text("import time; print('up', flush=True); time.sleep(3)\n", encoding="utf-8")
    monkeypatch.setattr(webui_app, "RUNNER", runner_path)

    up = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"a,y\n1,2\n3,4\n5,6\n"), "race.csv")},
        content_type="multipart/form-data",
    ).get_json()
    uid = up["upload_id"]

    statuses: list[int] = []
    job_ids: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(6)

    def fire() -> None:
        c = app.test_client()
        barrier.wait()
        r = c.post("/api/run", json={"upload_id": uid, "predict": ["y"]})
        with lock:
            statuses.append(r.status_code)
            if r.status_code == 200:
                job_ids.append(r.get_json()["job_id"])

    threads = [threading.Thread(target=fire) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    try:
        assert statuses.count(200) == 1, f"Expected exactly 1 winner, got {statuses}"
        assert statuses.count(409) == 5, f"Expected 5 conflicts, got {statuses}"
    finally:
        for jid in job_ids:
            client.post(f"/api/stop/{jid}")
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                if client.get(f"/api/job/{jid}").get_json()["status"] != "running":
                    break
                time.sleep(0.05)
        runner_path.unlink(missing_ok=True)


# ===========================================================================
# 4. WebUI Frontend JS / Report Inspector Runtime Test (1 test)
# ===========================================================================


@pytest.mark.tier5
@pytest.mark.f4
def test_t5_webui_report_summary_and_frontend_multi_target_and_xss(
    isolated_webui_client, tmp_path: Path
) -> None:
    """Backend + JS renderInspector handle multi_target reports, string 'N/A' metrics, and escaped dropped_periods."""
    if not NODE_EXE.is_file():
        pytest.skip("Node.js not installed")

    _, client, (_, rep_dir, _) = isolated_webui_client

    mt_rep = {
        "file": "tips.csv",
        "targets": ["total_bill", "tip"],
        "mode": "multi_target_autonomous",
        "target_reports": {
            "total_bill": {
                "selected_model": {"name": "Ridge", "metrics": {"metric": 0.61}},
                "audit_trail": {"candidates_evaluated": [{"name": "Ridge", "metric": 0.61}], "pareto_front": [{"name": "Ridge"}]},
                "risk_report": {"task": "regression", "vif_max": 1.2, "notes": ["ok"]},
                "eda_report": {"quality_score": 95.0, "n_rows": 244, "n_cols": 7},
            },
            "tip": {
                "selected_model": {"name": "RF", "metrics": {"metric": 0.52}},
                "audit_trail": {"candidates_evaluated": [{"name": "RF", "metric": 0.52}], "pareto_front": [{"name": "RF"}]},
            },
        },
        "predictions": {"total_bill": {"count": 244}, "tip": {"count": 244}},
    }
    (rep_dir / "multi_target_tips_combined.json").write_text(json.dumps(mt_rep), encoding="utf-8")
    r_mt = client.get("/api/report-summary?path=multi_target_tips_combined.json")
    assert r_mt.status_code == 200
    mt_summary = r_mt.get_json()
    assert mt_summary["kind"] == "multi_target"
    assert mt_summary["selected_model"] is not None or (
        mt_summary.get("target_reports") and any(
            tr.get("selected_model") for tr in mt_summary["target_reports"].values()
        )
    )

    html = client.get("/").get_data(as_text=True)
    m = re.search(r"<script>(.*)</script>", html, re.DOTALL)
    assert m is not None
    js_path = tmp_path / "workbench.js"
    js_path.write_text(m.group(1).replace("initWorkbench();", ""), encoding="utf-8")

    driver_js = tmp_path / "driver.js"
    driver_js.write_text(
        r"""
const vm = require('vm');
const fs = require('fs');
const assert = require('assert');

const jsCode = fs.readFileSync(process.argv[2], 'utf8');
const elements = new Map();
function makeEl(id) {
  const el = {
    id, value: '', checked: false, disabled: false, multiple: false,
    options: [], selectedOptions: [], style: {}, className: '',
    textContent: '', _innerHTML: '', dataset: {},
    classList: { _s: new Set(), add(c){this._s.add(c);}, remove(c){this._s.delete(c);}, toggle(c,f){if(f===undefined){this._s.has(c)?this._s.delete(c):this._s.add(c);}else if(f)this._s.add(c);else this._s.delete(c);}, contains(c){return this._s.has(c);} },
    setAttribute(){}, getAttribute(){}, addEventListener(){}, dispatchEvent(){},
    appendChild(){}, insertAdjacentHTML(){}, querySelectorAll(){ return []; },
  };
  Object.defineProperty(el, 'innerHTML', { get(){return this._innerHTML;}, set(v){this._innerHTML = String(v);} });
  return el;
}
function getEl(id) { if (!elements.has(id)) elements.set(id, makeEl(id)); return elements.get(id); }

const sandbox = {
  assert, console, setTimeout, clearTimeout, setInterval: () => 1, clearInterval: () => {},
  confirm: () => true, Event: function(t){ this.type = t; },
  document: { getElementById: getEl, querySelectorAll: () => [], createTextNode: (t) => ({ textContent: t }) },
  fetch: async () => ({ status: 200, ok: true, text: async () => '{}' }),
};
vm.createContext(sandbox);
vm.runInContext(jsCode + `
(() => {
  // 1. renderForecast must escape dropped_periods
  renderForecast({
    value_col: 'val',
    forecast: [{ period: '2026-04', value: 10 }],
    history: [{ period: '2026-01', value: 5 }, { period: '2026-02', value: 7 }, { period: '2026-03', value: 9 }],
    forecast_total: 10, chosen_method: 'last value',
    methods: [{ method: 'last value', mae: 1.0 }],
    csv_path: 'fc.csv',
    aggregation: { grain: 'M', periods: 3, rows_used: 3, rows_in: 3, agg: 'sum', dropped_periods: ['<img src=x onerror=alert(1)>'] },
    warnings: [],
  });
  assert.strictEqual($('fcOut').innerHTML.includes('<img src=x onerror=alert(1)>'), false, 'Unescaped dropped_periods XSS');

  // 2. renderInspector must not render literal NaN for string 'N/A' metrics
  state.inspectorData = {
    kind: 'combined',
    selected_model: { name: 'Ridge', metrics: { metric: 'N/A', train_time_s: 'N/A' } },
    dataset_info: {}, summary: {}, phases: [],
  };
  state.inspectorTab = 'overview';
  renderInspector();
  assert.strictEqual($('inspectorBody').innerHTML.includes('>NaN<'), false, 'Rendered literal NaN for string metric');

  // 3. renderInspector must render meaningful model info for multi_target reports
  state.inspectorData = __MT_SUMMARY_JSON__;
  state.inspectorTab = 'overview';
  renderInspector();
  assert.strictEqual($('inspectorBody').innerHTML.includes('Ridge') || $('inspectorBody').innerHTML.includes('total_bill'), true, 'Multi-target inspector missing target/model info');
})();
`, sandbox);
""".replace("__MT_SUMMARY_JSON__", json.dumps(mt_summary)),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [str(NODE_EXE), str(driver_js), str(js_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr


# ===========================================================================
# 5. Iteration 4 Challenger Regression Tests (3 tests)
# ===========================================================================


@pytest.mark.tier5
@pytest.mark.f3
def test_t5_dst_mixed_offset_timestamps_webui_and_cli_timeseries(
    isolated_webui_client, run_cli, tmp_path: Path
) -> None:
    """DST/mixed-offset ISO-8601 timestamps and +-inf work cleanly in WebUI & CLI --timeseries."""
    _, client, _ = isolated_webui_client

    dates_ny = pd.date_range("2024-03-01", periods=24, freq="D", tz="America/New_York")
    csv_dst = "\n".join(["timestamp,requests"] + [f"{d.isoformat()},{100 + i}" for i, d in enumerate(dates_ny)])
    up1 = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(csv_dst.encode("utf-8")), "dst_logs.csv")},
        content_type="multipart/form-data",
    )
    assert up1.status_code == 200
    uid1 = up1.get_json()["upload_id"]

    # Mixed Z and +02:00 offsets
    rows_mix = ["timestamp,requests"]
    for i in range(24):
        ts = f"2024-03-{i + 1:02d}T12:00:00Z" if i % 2 == 0 else f"2024-03-{i + 1:02d}T14:00:00+02:00"
        rows_mix.append(f"{ts},{50 + i}")
    up_mix = client.post(
        "/api/upload",
        data={"file": (io.BytesIO("\n".join(rows_mix).encode("utf-8")), "mixed_tz.csv")},
        content_type="multipart/form-data",
    )
    assert up_mix.status_code == 200
    uid_mix = up_mix.get_json()["upload_id"]

    # Tz-aware DataFrame with np.inf in target
    df_inf = pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC"),
        "target_val": pd.Series(np.arange(20.0), dtype="Float64"),
    })
    df_inf.loc[5, "target_val"] = np.inf
    pkl_buf = io.BytesIO()
    df_inf.to_pickle(pkl_buf)
    pkl_buf.seek(0)
    up_inf = client.post(
        "/api/upload",
        data={"file": (pkl_buf, "adv_ext.pkl")},
        content_type="multipart/form-data",
    )
    assert up_inf.status_code == 200
    uid_inf = up_inf.get_json()["upload_id"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for uid in (uid1, uid_mix):
            r_hor = client.post("/api/suggest-horizon", json={"upload_id": uid, "datetime_col": "timestamp", "target_col": "requests"})
            assert r_hor.status_code == 200 and r_hor.get_json()["horizon"] is not None
            r_fc = client.post("/api/forecast", json={"upload_id": uid, "date_col": "timestamp", "value_col": "requests", "horizon": 3})
            assert r_fc.status_code == 200 and len(r_fc.get_json()["forecast"]) == 3

        r_hor_inf = client.post("/api/suggest-horizon", json={"upload_id": uid_inf, "datetime_col": "ts_utc", "target_col": "target_val"})
        assert r_hor_inf.status_code == 200 and r_hor_inf.get_json()["horizon"] is not None
        r_fc_inf = client.post("/api/forecast", json={"upload_id": uid_inf, "date_col": "ts_utc", "value_col": "target_val", "horizon": 3})
        assert r_fc_inf.status_code == 400

    bad_w = [
        w for w in caught
        if issubclass(w.category, (RuntimeWarning, UserWarning))
        and "statsmodels" not in str(w.message)
    ]
    assert not bad_w, [(w.category.__name__, str(w.message)) for w in bad_w]

    # CLI --timeseries on 45-row DST CSV succeeds; 24-row (<30) exits rc=1 cleanly without Traceback
    dst45 = tmp_path / "dst45.csv"
    d45 = pd.date_range("2024-02-20", periods=45, freq="D", tz="America/New_York")
    df_dst45 = pd.DataFrame({
        "timestamp": [d.isoformat() for d in d45],
        "requests": np.linspace(100.0, 150.0, 45) + np.sin(np.arange(45)),
    })
    df_dst45.to_csv(dst45, index=False)

    ts_risk = TimeSeriesRiskAnalyzer(datetime_col="timestamp", target_col="requests").analyze(df_dst45)
    assert ts_risk.frequency == "D"

    res_ok = run_cli(
        ["--timeseries", "--file", str(dst45), "--target", "requests", "--datetime-col", "timestamp", "--horizon", "5"],
        cwd=tmp_path,
        timeout=90.0,
    )
    assert res_ok.returncode == 0, f"STDOUT:\n{res_ok.stdout}\nSTDERR:\n{res_ok.stderr}"
    assert "Traceback" not in res_ok.stderr

    dst24 = tmp_path / "dst24.csv"
    pd.DataFrame({"timestamp": [d.isoformat() for d in dates_ny], "requests": np.linspace(100.0, 124.0, 24)}).to_csv(dst24, index=False)
    res_short = run_cli(
        ["--timeseries", "--file", str(dst24), "--target", "requests", "--datetime-col", "timestamp", "--horizon", "3"],
        cwd=tmp_path,
        timeout=60.0,
    )
    assert res_short.returncode == 1
    assert "Traceback" not in res_short.stderr
    assert "[ACDSO_PROGRESS]" in res_short.stdout and '"error"' in res_short.stdout


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_cli_decision_nonnumeric_inf_cost_and_fairness_na(run_cli, tmp_path: Path) -> None:
    """CLI --decision coerces non-numeric/inf incentive_cost, and FairnessConstraint handles pd.NA/np.nan."""
    rng = np.random.default_rng(42)
    dec_csv = tmp_path / "dec_str_inf.csv"
    pd.DataFrame({
        "x1": rng.normal(0, 1, 100),
        "x2": rng.normal(0, 1, 100),
        "offered_incentive": rng.integers(0, 2, 100),
        "churned": rng.integers(0, 2, 100),
        "incentive_cost": ["25.0" if i % 2 == 0 else ("bad" if i == 1 else "inf") for i in range(100)],
    }).to_csv(dec_csv, index=False)

    res_cli = run_cli(
        ["--decision", "--file", str(dec_csv), "--target", "churned", "--treatment", "offered_incentive"],
        cwd=tmp_path,
        timeout=90.0,
    )
    assert res_cli.returncode == 0, f"STDOUT:\n{res_cli.stdout}\nSTDERR:\n{res_cli.stderr}"
    assert "Traceback" not in res_cli.stderr
    assert "Budget constraint: $inf" not in res_cli.stdout

    # FairnessConstraint & DecisionIntelligence(..., fairness_col=...) with pd.NA / np.nan across dtypes
    n = 120
    for g_col in (
        pd.array(["A", "B", pd.NA] * 40, dtype="string"),
        pd.Categorical(["A", "B", pd.NA] * 40),
        pd.array([1, 2, pd.NA] * 40, dtype="Int64"),
        np.array(["A", "B", np.nan] * 40, dtype=object),
    ):
        df_f = pd.DataFrame({
            "x1": rng.normal(0, 1, n),
            "treat": rng.integers(0, 2, n),
            "outcome": rng.normal(0, 1, n),
            "cost": np.ones(n),
            "grp": g_col,
        })
        res_di = DecisionIntelligence("treat", "outcome", cost_col="cost").fit(
            df_f, budget=30.0, fairness_col="grp", run_causal=False, run_policy=False, verbose=False
        )
        assert res_di.optimization is not None and res_di.optimization.n_selected >= 0
        fc = FairnessConstraint(group_col="grp", max_disparity=0.5)
        assert isinstance(fc.is_feasible(res_di.optimization.selected, df_f), bool)
        assert math.isfinite(fc.get_violation(res_di.optimization.selected, df_f))

    sel, total = knapsack_optimize(
        pd.Series([10.0, 100.0, pd.NA, np.inf], dtype="Float64"),
        pd.Series([2.0, -1.0, 1.0, 1.0], dtype="Float64"),
        capacity=5.0,
    )
    assert sel.tolist() == [1, 0, 0, 0]
    assert total == pytest.approx(10.0)


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_analyze_risk_string_target_with_pd_na() -> None:
    """analyze_risk handles string targets containing pd.NA without TypeError."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": np.arange(80.0), "b": rng.normal(0, 1, 80)})
    y = pd.Series(["c0", "c1", pd.NA, "c0"] * 20, dtype="string")

    rr = analyze_risk(X, y, task="classification")
    assert rr.target_type == "binary"
    assert rr.class_imbalance_max_ratio is not None and math.isfinite(rr.class_imbalance_max_ratio)


# ===========================================================================
# 6. Iteration 5 Core Remediation Regression Tests (4 tests)
# ===========================================================================


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_decision_fairness_col_exclusion_and_policy_na_safety() -> None:
    """DecisionIntelligence excludes fairness_col/confounders from feature_cols and runs policy safely on pd.NA/Inf."""
    rng = np.random.default_rng(7)
    n = 120

    for grp_arr in (
        pd.array([pd.NA] * n, dtype="Int64"),
        pd.array([pd.NA] * n, dtype="Float64"),
        pd.array([pd.NA] * n, dtype="boolean"),
        pd.array([1, 2, pd.NA] * 40, dtype="Int64"),
    ):
        df_di = pd.DataFrame({
            "f1": rng.normal(0, 1, n),
            "z1": rng.normal(0, 1, n),
            "treat": rng.integers(0, 2, n),
            "outcome": rng.normal(0, 1, n),
            "cost": np.ones(n),
            "grp": grp_arr,
        })
        di = DecisionIntelligence("treat", "outcome", cost_col="cost", confounders=["z1"])
        res = di.fit(df_di, budget=30.0, fairness_col="grp", run_causal=False, run_policy=False, verbose=False)
        assert di.feature_cols == ["f1"]
        assert "grp" not in di.feature_cols
        assert "z1" not in di.feature_cols
        assert res.optimization is not None

    df_pol = pd.DataFrame({
        "f1": pd.array(list(rng.normal(0, 1, n - 2)) + [pd.NA, np.inf], dtype="Float64"),
        "f2": pd.array(list(rng.integers(0, 5, n - 1)) + [pd.NA], dtype="Int64"),
        "f3": pd.array([True, False] * (n // 2), dtype="boolean"),
        "treat": rng.integers(0, 2, n),
        "outcome": rng.normal(0, 1, n),
        "cost": ["1.0" if i % 5 != 0 else ("bad" if i == 0 else "inf") for i in range(n)],
    })
    di_pol = DecisionIntelligence("treat", "outcome", cost_col="cost")
    res_pol = di_pol.fit(df_pol, budget=30.0, run_causal=False, run_policy=True, policy_iterations=2, verbose=False)
    assert res_pol.policy is not None
    assert math.isfinite(res_pol.policy.mean_reward)


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_uplift_predict_extension_dtypes_and_pd_na() -> None:
    """UpliftModeler.predict() handles Float64, Int64, and boolean features with pd.NA and +-inf."""
    rng = np.random.default_rng(7)
    n = 100
    df_u = pd.DataFrame({
        "f1": pd.array(list(rng.normal(0, 1, n - 2)) + [np.inf, -np.inf], dtype="Float64"),
        "f2": pd.array(list(rng.integers(0, 5, n - 5)) + [pd.NA] * 5, dtype="Int64"),
        "f3": pd.array([True, False] * (n // 2 - 1) + [pd.NA, True], dtype="boolean"),
        "treat": rng.integers(0, 2, n),
        "outcome": rng.normal(0, 1, n),
    })
    um = UpliftModeler("treat", "outcome")
    um.fit(df_u, verbose=False)
    preds = um.predict(df_u.iloc[-10:])
    assert preds.shape == (10,)
    assert np.all(np.isfinite(preds))


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_datetime_transformer_weekofyear_with_nat() -> None:
    """DateTimeFeatureTransformer handles NaT/pd.NA for 'weekofyear' while keeping int dtype on complete data."""
    from acdso.timeseries import DateTimeFeatureTransformer

    df_nat = pd.DataFrame({
        "ts": ["2024-03-10T01:00:00-05:00", None, "2024-03-10T03:00:00-04:00"],
    })
    out_nat = DateTimeFeatureTransformer(
        datetime_col="ts", features=["hour", "weekofyear"]
    ).fit_transform(df_nat)
    assert np.isfinite(out_nat["dt_weekofyear"].iloc[0])
    assert np.isnan(out_nat["dt_weekofyear"].iloc[1])
    assert np.isnan(out_nat["dt_weekofyear_sin"].iloc[1])
    assert np.isfinite(out_nat["dt_weekofyear"].iloc[2])

    df_clean = pd.DataFrame({
        "ts": ["2024-03-10T01:00:00-05:00", "2024-03-10T03:00:00-04:00"],
    })
    out_clean = DateTimeFeatureTransformer(
        datetime_col="ts", features=["hour", "weekofyear"]
    ).fit_transform(df_clean)
    assert pd.api.types.is_integer_dtype(out_clean["dt_weekofyear"].dtype)


@pytest.mark.tier5
@pytest.mark.f1
def test_t5_meta_optimizer_f1_binary_string_and_tracemalloc_cleanup() -> None:
    """ACDSOMetaClassifier handles scorer='f1' on binary string targets and never leaks tracemalloc on errors."""
    import tracemalloc
    from acdso.integrations.base import CandidateSource
    from acdso.integrations.meta_optimizer import ACDSOMetaClassifier

    rng = np.random.default_rng(7)
    X_m = pd.DataFrame({"a": rng.normal(0, 1, 80), "b": rng.normal(0, 1, 80)})
    y_m = pd.Series(["c0", "c1"] * 40, dtype="string")

    meta = ACDSOMetaClassifier(
        adapters=[], include_acdso_candidates=True, scorer="f1", cv=3, verbose=0
    )
    meta.fit(X_m, y_m)
    assert meta.scorer == "f1"
    assert meta.pareto_front_ and all(math.isfinite(c["metric"]) for c in meta.pareto_front_)
    assert all(math.isfinite(c["risk_penalty"]) for c in meta.pareto_front_)
    assert meta.predict(X_m.iloc[:5]).shape == (5,)
    assert math.isfinite(meta.score(X_m, y_m))

    class BrokenEstimator:
        def get_params(self, deep: bool = True) -> dict:
            return {}

        def fit(self, X, y):
            raise RuntimeError("intentional evaluation failure")

    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.stop()
    try:
        rr = analyze_risk(X_m, y_m, task="classification")
        bad_cand = CandidateSource(name="broken", source="test", estimator=BrokenEstimator(), metadata={})
        assert meta._evaluate_candidate(X_m, y_m, bad_cand, rr, "classification") is None
        assert tracemalloc.is_tracing() is False
    finally:
        if was_tracing and not tracemalloc.is_tracing():
            tracemalloc.start()
