"""
Tests for Risk Analysis Module
==============================

Tests covering risk detection, VIF computation, leakage detection,
and temporal leakage checks.
"""

import numpy as np
import pandas as pd
import pytest

from acdso.risk import (
    RiskReport,
    analyze_risk,
    compute_risk_penalty,
    _vif_numeric,
    _univariate_predictive_power,
)
from acdso.utils import class_imbalance, infer_target_type


class TestRiskReport:
    """Tests for RiskReport dataclass."""

    def test_creation(self):
        """Test RiskReport creation."""
        report = RiskReport(
            task="classification",
            target_type="binary",
            n_rows=1000,
            n_cols=10,
            n_numeric=8,
            n_categorical=2,
            missing_ratio=0.05,
        )

        assert report.task == "classification"
        assert report.n_rows == 1000
        assert report.missing_ratio == 0.05

    def test_to_dict(self):
        """Test RiskReport serialization."""
        report = RiskReport(
            task="regression",
            target_type="regression",
            n_rows=500,
            n_cols=5,
            n_numeric=5,
            n_categorical=0,
            missing_ratio=0.0,
            notes=["Test note"],
        )

        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["task"] == "regression"
        assert "Test note" in d["notes"]


class TestVIF:
    """Tests for VIF computation."""

    def test_vif_independent_features(self):
        """Test VIF for independent features."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n),
        })

        vif_max, high_vif = _vif_numeric(X)

        # Independent features should have low VIF (close to 1)
        assert vif_max < 5.0
        assert len(high_vif) == 0

    def test_vif_collinear_features(self):
        """Test VIF for collinear features."""
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = x1 + 0.01 * np.random.randn(n)  # Highly correlated
        x3 = np.random.randn(n)

        X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        vif_max, high_vif = _vif_numeric(X)

        # Should detect high VIF due to collinearity
        assert vif_max > 10.0
        assert len(high_vif) > 0

    def test_vif_single_column(self):
        """Test VIF with single column."""
        X = pd.DataFrame({"a": np.random.randn(100)})
        vif_max, high_vif = _vif_numeric(X)

        # Single column should have VIF = 1
        assert vif_max == 1.0
        assert len(high_vif) == 0


class TestClassImbalance:
    """Tests for class imbalance detection."""

    def test_balanced_classes(self):
        """Test balanced class detection."""
        y = np.array([0] * 50 + [1] * 50)
        max_ratio, severity = class_imbalance(y)

        assert max_ratio == 0.5
        assert severity == 0.0

    def test_imbalanced_classes(self):
        """Test imbalanced class detection."""
        y = np.array([0] * 90 + [1] * 10)
        max_ratio, severity = class_imbalance(y)

        assert max_ratio == 0.9
        assert severity == 0.8  # (0.9 - 0.5) / 0.5 = 0.8

    def test_extreme_imbalance(self):
        """Test extreme imbalance detection."""
        y = np.array([0] * 99 + [1] * 1)
        max_ratio, severity = class_imbalance(y)

        assert max_ratio == 0.99
        assert severity >= 0.98


class TestTargetTypeInference:
    """Tests for target type inference."""

    def test_binary_classification(self):
        """Test binary classification detection."""
        y = np.array([0, 1, 0, 1, 1, 0])
        assert infer_target_type(y) == "binary"

        y = np.array(["a", "b", "a", "b"])
        assert infer_target_type(y) == "binary"

    def test_multiclass_classification(self):
        """Test multiclass classification detection."""
        y = np.array([0, 1, 2, 0, 1, 2])
        assert infer_target_type(y) == "multiclass"

        y = np.array(["a", "b", "c", "a"])
        assert infer_target_type(y) == "multiclass"

    def test_regression(self):
        """Test regression detection."""
        y = np.array([1.5, 2.3, 0.7, 4.2])
        assert infer_target_type(y) == "regression"


class TestAnalyzeRisk:
    """Tests for the main analyze_risk function."""

    def test_basic_analysis(self):
        """Test basic risk analysis."""
        np.random.seed(42)
        X = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
        })
        y = np.random.randint(0, 2, 100)

        report = analyze_risk(X, y, task="classification")

        assert isinstance(report, RiskReport)
        assert report.task == "classification"
        assert report.n_rows == 100
        assert report.n_cols == 2

    def test_missing_data_detection(self):
        """Test missing data detection."""
        X = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0, 5.0] * 20,
            "b": [1.0, np.nan, np.nan, 4.0, 5.0] * 20,
        })
        y = np.random.randint(0, 2, 100)

        report = analyze_risk(X, y, task="classification")

        # Should detect significant missing data
        assert report.missing_ratio > 0.1

    def test_with_llm_advisor(self):
        """Test with custom LLM advisor."""
        X = pd.DataFrame({
            "feature_1": np.random.randn(50),
            "target_leak": np.random.randn(50),  # Suspicious name
        })
        y = np.random.randint(0, 2, 50)

        def mock_advisor(payload):
            return {
                "note": "Test advisor note",
                "suspicious_columns": ["target_leak"],
            }

        report = analyze_risk(
            X, y,
            task="classification",
            llm_advisor=mock_advisor,
        )

        # Should have advisor note
        llm_notes = [n for n in report.notes if "Test advisor" in n]
        assert len(llm_notes) > 0


class TestComputeRiskPenalty:
    """Tests for risk penalty computation."""

    def test_low_risk_penalty(self):
        """Test penalty for low-risk data."""
        report = RiskReport(
            task="classification",
            target_type="binary",
            n_rows=1000,
            n_cols=10,
            n_numeric=10,
            n_categorical=0,
            missing_ratio=0.01,
            imbalance_severity=0.0,
            vif_max=2.0,
            leakage_proxy_score_max=0.6,
        )

        penalty = compute_risk_penalty(report, cv_score_std=0.01)
        assert penalty < 0.2

    def test_high_risk_penalty(self):
        """Test penalty for high-risk data."""
        report = RiskReport(
            task="classification",
            target_type="binary",
            n_rows=100,
            n_cols=5,
            n_numeric=5,
            n_categorical=0,
            missing_ratio=0.4,
            imbalance_severity=0.8,
            vif_max=50.0,
            leakage_proxy_score_max=0.99,
            temporal_leakage_gap=0.15,
        )

        penalty = compute_risk_penalty(report, cv_score_std=0.1)
        assert penalty > 0.5

    def test_safe_preference_weights_leakage_higher(self):
        """Test that safe preference weights leakage higher."""
        report = RiskReport(
            task="classification",
            target_type="binary",
            n_rows=100,
            n_cols=5,
            n_numeric=5,
            n_categorical=0,
            missing_ratio=0.0,
            imbalance_severity=0.0,
            vif_max=1.0,
            leakage_proxy_score_max=0.98,  # High leakage signal
        )

        penalty_safe = compute_risk_penalty(
            report, cv_score_std=0.01, preference="safe"
        )
        penalty_balanced = compute_risk_penalty(
            report, cv_score_std=0.01, preference="balanced"
        )

        # Safe preference should penalize leakage more
        assert penalty_safe >= penalty_balanced


class TestVectorizedRiskDiagnostics:
    """Tests for vectorized VIF, proxy leakage detection, and time budget."""

    def test_vif_exact_duplicates_and_constants(self):
        """Test VIF flags exact duplicates while ignoring constants and NaNs/Infs."""
        rng = np.random.RandomState(42)
        x1 = rng.randn(150)
        df = pd.DataFrame({
            "x1": x1,
            "x2": x1.copy(),
            "x_indep": rng.randn(150),
            "const_col": np.ones(150),
            "all_nan": np.full(150, np.nan),
        })
        df.loc[0, "x_indep"] = np.inf

        vif_max, high_vif = _vif_numeric(df)
        assert vif_max >= 10.0
        assert "x1" in high_vif and "x2" in high_vif
        assert "const_col" not in high_vif
        assert "all_nan" not in high_vif

    def test_proxy_leakage_continuous_numeric(self):
        """Continuous numeric proxy leakage is detected without flagging innocent columns."""
        rng = np.random.RandomState(42)
        n = 400
        X = pd.DataFrame({f"f{i}": rng.randn(n) for i in range(6)})
        y = rng.randint(0, 2, n)
        X["leaky_feature"] = y + rng.randn(n) * 0.01

        report = analyze_risk(X, y, task="classification", random_state=42)
        assert report.leakage_proxy_score_max >= 0.95
        assert report.leakage_proxy_features == ["leaky_feature"]

    def test_proxy_leakage_regression(self):
        """Continuous regression proxy leakage is detected."""
        rng = np.random.RandomState(42)
        n = 300
        X = pd.DataFrame({f"f{i}": rng.randn(n) for i in range(5)})
        y = 2.0 * X["f0"].to_numpy() + 2.0 * X["f1"].to_numpy() + 2.0 * X["f2"].to_numpy() + rng.randn(n) * 0.5
        X["leaky_reg"] = y + rng.randn(n) * 0.01

        report = analyze_risk(X, y, task="regression", scorer="r2", random_state=42)
        assert report.leakage_proxy_score_max >= 0.95
        assert report.leakage_proxy_features == ["leaky_reg"]

    def test_proxy_leakage_categorical_and_id_skip(self):
        """Categorical proxy leak is detected while high-cardinality string IDs are skipped."""
        rng = np.random.RandomState(42)
        n = 200
        y = rng.randint(0, 2, n)
        X = pd.DataFrame({
            "user_id": [f"uid_{i}" for i in range(n)],
            "cat_leak": np.where(y == 1, "yes", "no"),
            "noise": rng.randn(n),
        })

        report = analyze_risk(X, y, task="classification", random_state=42)
        assert report.leakage_proxy_score_max >= 0.95
        assert "cat_leak" in report.leakage_proxy_features
        assert "user_id" not in report.leakage_proxy_features

    def test_analyze_risk_time_budget(self):
        """analyze_risk honors small time_budget_s without raising."""
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(200, 15), columns=[f"c{i}" for i in range(15)])
        y = rng.randint(0, 2, 200)

        report = analyze_risk(X, y, task="classification", time_budget_s=0.05, random_state=42)
        assert isinstance(report, RiskReport)
        assert report.n_rows == 200
        assert report.n_cols == 15

    def test_proxy_leakage_categorical_non_range_and_sliced_index(self):
        """Categorical proxy leak is detected on sliced iloc and custom string-indexed >10-col DataFrames."""
        rng = np.random.RandomState(42)
        n = 300
        y = rng.randint(0, 2, n)
        data = {f"c{i}": rng.choice(["A", "B", "C", "D"], n) for i in range(15)}
        data["leak_cat"] = np.where(y == 1, "yes", "no")

        # Case 1: sliced DataFrame (index 150..299)
        df_full = pd.DataFrame(data)
        df_sliced = df_full.iloc[150:]
        y_sliced = y[150:]
        rep_sliced = analyze_risk(df_sliced, y_sliced, task="classification", random_state=42)
        assert "leak_cat" in rep_sliced.leakage_proxy_features
        assert rep_sliced.leakage_proxy_score_max >= 0.95

        # Case 2: custom string index
        df_str = pd.DataFrame(data, index=[f"id_{i}" for i in range(n)])
        rep_str = analyze_risk(df_str, y, task="classification", random_state=42)
        assert "leak_cat" in rep_str.leakage_proxy_features
        assert rep_str.leakage_proxy_score_max >= 0.95

    def test_analyze_risk_all_nan_column_no_imputer_warning(self):
        """All-NaN numeric columns do not emit SimpleImputer UserWarning in _detect_linearity."""
        import warnings

        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame({
            "x1": rng.randn(n),
            "x2": rng.randn(n),
            "all_nan": np.full(n, np.nan),
        })
        y = (X["x1"] > 0).astype(int).to_numpy()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = analyze_risk(X, y, task="classification", random_state=42)

        assert isinstance(report, RiskReport)
        imputer_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning) and "Skipping features without any observed values" in str(w.message)
        ]
        assert imputer_warnings == []

    def test_analyze_risk_categorical_time_column_no_datetime_warning(self):
        """Categorical columns named 'time' (e.g. Dinner/Lunch) do not emit pandas dateutil warnings."""
        import warnings

        rng = np.random.RandomState(42)
        n = 60
        X = pd.DataFrame({
            "total_bill": rng.uniform(5.0, 50.0, size=n),
            "time": rng.choice(["Dinner", "Lunch"], size=n),
            "size": rng.randint(1, 6, size=n),
        })
        y = 0.15 * X["total_bill"] + rng.randn(n) * 0.5

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = analyze_risk(X, y, task="regression", random_state=42)

        assert isinstance(report, RiskReport)
        dt_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning) and "Could not infer format" in str(w.message)
        ]
        assert dt_warnings == []


