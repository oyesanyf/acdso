"""
ACDSO v3.0: Risk-Optimized AutoML for Production Systems
=========================================================

A production-oriented AutoML library that unifies contextual risk diagnostics,
adaptive pipeline synthesis, and multi-objective optimization into a single
closed-loop optimization procedure.

ACDSO optimizes predictive performance alongside:
- Training cost
- Inference latency
- Memory footprint
- Quantifiable data/model risk (semantic leakage, temporal leakage,
  multicollinearity, instability)

Key Features:
- Continuous, severity-aware risk functions (not binary flags)
- Constraint-based pruning for production guardrails
- Pareto knee-point selection (not naive "best accuracy")
- JSON-serializable Risk Audit Trail for MLOps integration
- Extensible hooks for meta-learning and LLM-based advisory

Basic Usage:
    >>> from acdso import ACDSOClassifier, ACDSORegressor
    >>>
    >>> # Classification
    >>> clf = ACDSOClassifier(scorer="f1", cv=5, preference="safe")
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> print(clf.risk_report_)
    >>> print(clf.audit_trail_)
    >>>
    >>> # Regression
    >>> reg = ACDSORegressor(scorer="r2", cv=5)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)

Authors: DS-TUI Development Team
Version: 2.0.0 (Production-Ready)
License: MIT
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

__version__ = "2.0.0"
__author__ = "DS-TUI Development Team"
__license__ = "MIT"

_LAZY_SUBMODULES = frozenset({
    "core",
    "eda",
    "risk",
    "utils",
    "pareto",
    "feature_importance",
    "auto_preprocessing",
    "advisors",
    "integrations",
    "timeseries",
    "decision",
    # M1-M5 submodules
    "cv",
    "nlp",
    "clustering",
    "anomaly",
    "multilabel",
    "ranking",
    "survival",
    "recommender",
    "scale",
})

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Core estimators
    "ACDSOClassifier": ("acdso.core", "ACDSOClassifier"),
    "ACDSORegressor": ("acdso.core", "ACDSORegressor"),
    # Auto EDA
    "ACDSOAutoEDA": ("acdso.eda", "ACDSOAutoEDA"),
    "ColumnProfile": ("acdso.eda", "ColumnProfile"),
    "DataQualityReport": ("acdso.eda", "DataQualityReport"),
    "EDAReport": ("acdso.eda", "EDAReport"),
    "EDACandidate": ("acdso.eda", "EDACandidate"),
    "CorrelationResult": ("acdso.eda", "CorrelationResult"),
    "AnomalyResult": ("acdso.eda", "AnomalyResult"),
    "EDABackend": ("acdso.eda", "EDABackend"),
    "get_available_backends": ("acdso.eda", "get_available_backends"),
    "get_backend_adapter": ("acdso.eda", "get_backend_adapter"),
    "YDataProfilingAdapter": ("acdso.eda", "YDataProfilingAdapter"),
    "SweetvizAdapter": ("acdso.eda", "SweetvizAdapter"),
    "DataprepAdapter": ("acdso.eda", "DataprepAdapter"),
    "DTaleAdapter": ("acdso.eda", "DTaleAdapter"),
    "quick_profile": ("acdso.eda", "quick_profile"),
    "compare_datasets": ("acdso.eda", "compare_datasets"),
    "validate_data": ("acdso.eda", "validate_data"),
    "infer_data_types": ("acdso.eda", "infer_data_types"),
    "get_preprocessing_recommendations": ("acdso.eda", "get_preprocessing_recommendations"),
    # Risk analysis components
    "RiskReport": ("acdso.risk", "RiskReport"),
    "analyze_risk": ("acdso.risk", "analyze_risk"),
    "compute_risk_penalty": ("acdso.risk", "compute_risk_penalty"),
    # Utility functions
    "build_preprocessor": ("acdso.utils", "build_preprocessor"),
    "detect_datetime_column": ("acdso.utils", "detect_datetime_column"),
    "infer_target_type": ("acdso.utils", "infer_target_type"),
    "split_columns": ("acdso.utils", "split_columns"),
    # Pareto optimization
    "CandidateResult": ("acdso.pareto", "CandidateResult"),
    "pareto_front": ("acdso.pareto", "pareto_front"),
    "knee_point_selection": ("acdso.pareto", "knee_point_selection"),
    # Safe Feature Importance
    "SafeFeatureImportance": ("acdso.feature_importance", "SafeFeatureImportance"),
    "FeatureImportanceResult": ("acdso.feature_importance", "FeatureImportanceResult"),
    "SafeImportanceReport": ("acdso.feature_importance", "SafeImportanceReport"),
    "compute_safe_importance": ("acdso.feature_importance", "compute_safe_importance"),
    # AutoEDA-Informed Preprocessing
    "AutoPreprocessor": ("acdso.auto_preprocessing", "AutoPreprocessor"),
    "PreprocessingPlan": ("acdso.auto_preprocessing", "PreprocessingPlan"),
    "PreprocessingAction": ("acdso.auto_preprocessing", "PreprocessingAction"),
    "TargetEncoder": ("acdso.auto_preprocessing", "TargetEncoder"),
    "SkewnessTransformer": ("acdso.auto_preprocessing", "SkewnessTransformer"),
    "OutlierClipper": ("acdso.auto_preprocessing", "OutlierClipper"),
    "create_eda_preprocessor": ("acdso.auto_preprocessing", "create_eda_preprocessor"),
    # LLM advisors
    "LLMAdvisor": ("acdso.advisors", "LLMAdvisor"),
    "LLMConfig": ("acdso.advisors", "LLMConfig"),
    "PromptTemplate": ("acdso.advisors", "PromptTemplate"),
    "BaseLLMProvider": ("acdso.advisors", "BaseLLMProvider"),
    "OpenAIProvider": ("acdso.advisors", "OpenAIProvider"),
    "AnthropicProvider": ("acdso.advisors", "AnthropicProvider"),
    "AzureOpenAIProvider": ("acdso.advisors", "AzureOpenAIProvider"),
    "OllamaProvider": ("acdso.advisors", "OllamaProvider"),
    "OpenAICompatibleProvider": ("acdso.advisors", "OpenAICompatibleProvider"),
    "create_advisor": ("acdso.advisors", "create_advisor"),
    "llm_advisor_stub": ("acdso.advisors", "llm_advisor_stub"),
    "BaseLLMAdvisor": ("acdso.advisors", "BaseLLMAdvisor"),
    "StubLLMAdvisor": ("acdso.advisors", "StubLLMAdvisor"),
    "LLMLabelAnalyzer": ("acdso.advisors", "LLMLabelAnalyzer"),
    "LabelAnalysisResult": ("acdso.advisors", "LabelAnalysisResult"),
    "analyze_labels_with_llm": ("acdso.advisors", "analyze_labels_with_llm"),
    "LLMColumnAnalyzer": ("acdso.advisors", "LLMColumnAnalyzer"),
    "ColumnAnalysisResult": ("acdso.advisors", "ColumnAnalysisResult"),
    "auto_exclude_columns": ("acdso.advisors", "auto_exclude_columns"),
    # Entity-grouped & temporal cross-validation (R3 / M1)
    "VALID_CV_STRATEGIES": ("acdso.cv", "VALID_CV_STRATEGIES"),
    "build_cv_splitter": ("acdso.cv", "build_cv_splitter"),
    "validate_no_group_leakage": ("acdso.cv", "validate_no_group_leakage"),
    "validate_groups": ("acdso.cv", "validate_groups"),
    "splitter_supports_groups": ("acdso.cv", "splitter_supports_groups"),
    "resolve_cv_strategy_name": ("acdso.cv", "resolve_cv_strategy_name"),
    "iter_cv_splits": ("acdso.cv", "iter_cv_splits"),
    "sample_by_groups": ("acdso.cv", "sample_by_groups"),
    "group_holdout_split": ("acdso.cv", "group_holdout_split"),
    # Classical NLP & free-text featurization (R4 / M2)
    "TextFeaturizer": ("acdso.nlp", "TextFeaturizer"),
    "detect_text_columns": ("acdso.nlp", "detect_text_columns"),
    # Unsupervised clustering & trainable anomaly detection (R1, R2 / M3)
    "ACDSOClusterer": ("acdso.clustering", "ACDSOClusterer"),
    "ACDSOAnomalyDetector": ("acdso.anomaly", "ACDSOAnomalyDetector"),
    # Specialized supervised formulations (R5 / M4)
    "ACDSOMultiLabelClassifier": ("acdso.multilabel", "ACDSOMultiLabelClassifier"),
    "ACDSORanker": ("acdso.ranking", "ACDSORanker"),
    "ndcg_at_k": ("acdso.ranking", "ndcg_at_k"),
    "ACDSOSurvivalAnalyzer": ("acdso.survival", "ACDSOSurvivalAnalyzer"),
    "concordance_index": ("acdso.survival", "concordance_index"),
    "ACDSORecommender": ("acdso.recommender", "ACDSORecommender"),
    # Single-machine out-of-core / chunked scale (R6 / M5)
    "ACDSOOutOfCoreTrainer": ("acdso.scale", "ACDSOOutOfCoreTrainer"),
    "iter_file_chunks": ("acdso.scale", "iter_file_chunks"),
    "reservoir_sample_file": ("acdso.scale", "reservoir_sample_file"),
}


def __getattr__(name: str):
    """Lazy import for ACDSO public symbols and submodules."""
    if name in _LAZY_ATTRS:
        mod_name, attr_name = _LAZY_ATTRS[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core estimators
    "ACDSOClassifier",
    "ACDSORegressor",
    # Auto EDA
    "ACDSOAutoEDA",
    "ColumnProfile",
    "DataQualityReport",
    "EDAReport",
    "EDACandidate",
    "CorrelationResult",
    "AnomalyResult",
    "EDABackend",
    "get_available_backends",
    "get_backend_adapter",
    "YDataProfilingAdapter",
    "SweetvizAdapter",
    "DataprepAdapter",
    "DTaleAdapter",
    "quick_profile",
    "compare_datasets",
    "validate_data",
    "infer_data_types",
    "get_preprocessing_recommendations",
    # Risk components
    "RiskReport",
    "analyze_risk",
    "compute_risk_penalty",
    # Utilities
    "build_preprocessor",
    "detect_datetime_column",
    "infer_target_type",
    "split_columns",
    # Pareto optimization
    "CandidateResult",
    "pareto_front",
    "knee_point_selection",
    # Safe Feature Importance
    "SafeFeatureImportance",
    "FeatureImportanceResult",
    "SafeImportanceReport",
    "compute_safe_importance",
    # AutoEDA-Informed Preprocessing
    "AutoPreprocessor",
    "PreprocessingPlan",
    "PreprocessingAction",
    "TargetEncoder",
    "SkewnessTransformer",
    "OutlierClipper",
    "create_eda_preprocessor",
    # Advisors
    "LLMAdvisor",
    "LLMConfig",
    "PromptTemplate",
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "create_advisor",
    "llm_advisor_stub",
    "BaseLLMAdvisor",
    "StubLLMAdvisor",
    # Integrations submodule
    "integrations",
    # Time series submodule (AutoTS)
    "timeseries",
    # Decision Intelligence submodule (DoWhy, EconML, OR-Tools, RLlib)
    "decision",
    # Cross-validation (R3 / M1)
    "VALID_CV_STRATEGIES",
    "build_cv_splitter",
    "validate_no_group_leakage",
    "validate_groups",
    "splitter_supports_groups",
    "resolve_cv_strategy_name",
    "iter_cv_splits",
    "sample_by_groups",
    "group_holdout_split",
    # Classical NLP (R4 / M2)
    "TextFeaturizer",
    "detect_text_columns",
    # Clustering & Anomaly Detection (R1, R2 / M3)
    "ACDSOClusterer",
    "ACDSOAnomalyDetector",
    # Specialized Supervised Formulations (R5 / M4)
    "ACDSOMultiLabelClassifier",
    "ACDSORanker",
    "ndcg_at_k",
    "ACDSOSurvivalAnalyzer",
    "concordance_index",
    "ACDSORecommender",
    # Out-of-Core Scale (R6 / M5)
    "ACDSOOutOfCoreTrainer",
    "iter_file_chunks",
    "reservoir_sample_file",
    # M1-M5 lazy submodules
    "cv",
    "nlp",
    "clustering",
    "anomaly",
    "multilabel",
    "ranking",
    "survival",
    "recommender",
    "scale",
]

