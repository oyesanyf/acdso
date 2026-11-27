"""Tier 5 Adversarial Coverage Hardening Suite for R1-R7 (Core ML, Scale & Interfaces)."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RUNNER = REPO_ROOT / "run_acdso.py"
NODE_EXE = Path(r"C:\Program Files\nodejs\node.exe")

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _loads_strict_json(raw_text: str) -> dict[str, Any]:
    def _reject_nonfinite(constant: str) -> None:
        raise ValueError(f"Non-RFC8259 JSON constant: {constant}")

    return json.loads(raw_text, parse_constant=_reject_nonfinite)


def _run_cli(args: list[str], cwd: Path, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env["OLLAMA_HOST"] = "http://127.0.0.1:1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(SRC_ROOT)
    cwd.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, "-B", str(RUNNER), *args],
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout,
        check=False,
    )


@pytest.fixture()
def isolated_tier5_webui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import acdso.webui.app as webui_app

    up_dir = tmp_path / "uploads"
    rep_dir = tmp_path / "reports"
    log_dir = rep_dir / "webui_logs"
    up_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(webui_app, "UPLOAD_DIR", up_dir)
    monkeypatch.setattr(webui_app, "REPORTS_DIR", rep_dir)
    monkeypatch.setattr(webui_app, "LOG_DIR", log_dir)
    webui_app._invalidate_reports_cache()

    app = webui_app.create_app()
    yield webui_app, app, up_dir, rep_dir
    webui_app._invalidate_reports_cache()


# ===========================================================================
# Part A: Core ML & Scale (R1-R6) Adversarial Regressions
# ===========================================================================


class TestTier5CoreMLAndScale:
    def test_cv_sample_by_groups_and_holdout_sparse_matrices(self) -> None:
        from scipy import sparse
        from sklearn.model_selection import KFold
        from acdso.core import ACDSOClassifier
        from acdso.cv import build_cv_splitter, group_holdout_split, sample_by_groups

        X_csr = sparse.csr_matrix(np.vstack([np.eye(6), np.eye(6)]))
        X_csc = sparse.csc_matrix(X_csr)
        X_coo = sparse.coo_matrix(X_csr)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        groups = np.array(["g0", "g0", "g0", "g1", "g1", "g1", "g2", "g2", "g2", "g3", "g3", "g3"])

        # build_cv_splitter with existing splitter + sparse 2D targets (csr_matrix, csr_array, csc_matrix)
        kf = KFold(n_splits=2)
        assert build_cv_splitter(kf, y=X_csr, groups=groups) is kf
        assert build_cv_splitter(kf, y=X_csc, groups=groups) is kf
        if hasattr(sparse, "csr_array"):
            assert build_cv_splitter(kf, y=sparse.csr_array(X_csr), groups=groups) is kf

        # max_rows=1 -> returns 2 groups (6 rows) sliced from sparse matrices
        X_sub, y_sub, g_sub = sample_by_groups(X_csr, y, groups, max_rows=1, random_state=42)
        assert sparse.issparse(X_sub)
        assert X_sub.shape[0] == len(y_sub) == len(g_sub)
        assert X_sub.shape[0] < X_csr.shape[0]
        assert len(np.unique(g_sub)) >= 2

        # Stratified sparse group sampling & CSC/COO support
        X_sub_s, y_sub_s, g_sub_s = sample_by_groups(
            X_csc, y, groups, max_rows=4, stratify=True, random_state=42
        )
        assert sparse.issparse(X_sub_s)
        assert X_sub_s.shape[0] == len(y_sub_s) == len(g_sub_s)
        assert len(np.unique(y_sub_s)) == 2

        X_sub_coo, _, _ = sample_by_groups(X_coo, y, groups, max_rows=4, random_state=42)
        assert sparse.issparse(X_sub_coo)

        # group_holdout_split on sparse matrix
        tr, va = group_holdout_split(X_csr, y, groups=groups, test_size=0.25, random_state=42)
        assert len(tr) > 0 and len(va) > 0
        assert set(groups[tr].tolist()).isdisjoint(set(groups[va].tolist()))

        # ACDSOClassifier.fit/predict/score on sparse csr_matrix input
        rng = np.random.RandomState(42)
        X_dense_clf = np.hstack([rng.randn(24, 4), (y_clf := np.array([0, 1] * 12))[:, None] * 2.0])
        X_csr_clf = sparse.csr_matrix(X_dense_clf)
        g_clf = np.array([f"g{i // 3}" for i in range(24)])
        clf = ACDSOClassifier(cv=2, max_candidates=1, enable_stacking=False, random_state=42, verbose=0)
        clf.fit(X_csr_clf, y_clf, groups=g_clf)
        assert clf.predict(X_csr_clf[:4]).shape == (4,)
        assert math.isfinite(clf.score(X_csr_clf, y_clf))

    def test_clustering_constant_features_zero_convergence_warnings(self) -> None:
        from sklearn.exceptions import ConvergenceWarning
        from acdso.clustering import ACDSOClusterer

        X_const = pd.DataFrame(
            {
                "a": np.ones(24, dtype=float),
                "b": np.full(24, 5.0, dtype=float),
                "c": ["same"] * 24,
            }
        )
        X_two_pts = pd.DataFrame(
            {
                "a": [0.0, 5.0] * 10,
                "b": [1.0, 2.0] * 10,
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clu_auto = ACDSOClusterer(algorithm="auto", random_state=42).fit(X_const)
            clu_km = ACDSOClusterer(algorithm="kmeans", n_clusters=3, random_state=42).fit(X_const)
            clu_two = ACDSOClusterer(algorithm="auto", random_state=42).fit(X_two_pts)

        conv_warns = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        assert conv_warns == [], f"Unexpected ConvergenceWarning: {[str(w.message) for w in conv_warns]}"
        assert set(np.unique(clu_auto.labels_)).issubset({-1, 0})
        assert set(np.unique(clu_km.labels_)).issubset({0})
        assert clu_two.n_clusters_ <= 2
        assert clu_auto.predict(X_const.iloc[:5]).shape == (5,)
        _loads_strict_json(json.dumps(clu_auto.audit_trail_, allow_nan=False))

    def test_iter_file_chunks_early_break_closes_handles_on_windows(self, tmp_path: Path) -> None:
        from acdso.cli.io import inspect_file_schema, iter_file_chunks

        csv_p = tmp_path / "early_close.csv"
        pd.DataFrame({"a": np.arange(30), "b": np.linspace(0.0, 1.0, 30)}).to_csv(csv_p, index=False)

        gen = iter_file_chunks(csv_p, chunk_size=5)
        try:
            first = next(gen)
            assert len(first) == 5
        finally:
            gen.close()
        os.unlink(csv_p)
        assert not csv_p.exists()

        jsonl_p = tmp_path / "early_close.jsonl"
        pd.DataFrame({"a": np.arange(20), "b": ["txt"] * 20}).to_json(jsonl_p, orient="records", lines=True)
        schema = inspect_file_schema(jsonl_p, sample_rows=5, count_rows=False)
        assert schema["columns"] == ["a", "b"]
        os.unlink(jsonl_p)
        assert not jsonl_p.exists()

    def test_reservoir_sample_file_widens_int_to_float_and_mixed_without_truncation(self, tmp_path: Path) -> None:
        from acdso.scale import ACDSOOutOfCoreTrainer, reservoir_sample_file

        csv_p = tmp_path / "widen.csv"
        # Chunk 1 (rows 0..4): pure ints 1..5; Chunk 2 (rows 5..9): floats 3.5..7.5 + NaN
        lines = ["val,grp"]
        for i in range(5):
            lines.append(f"{i + 1},A")
        lines.extend(["3.5,B", "4.5,B", "5.5,B", ",B", "7.5,B"])
        csv_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

        sample_df, total_rows = reservoir_sample_file(csv_p, max_rows=6, chunk_size=5, random_state=7)
        assert total_rows == 10
        assert pd.api.types.is_float_dtype(sample_df["val"].dtype)
        # Ensure at least one fractional value survived without integer truncation
        non_null_vals = sample_df["val"].dropna().tolist()
        assert any(abs(v - round(v)) > 0.1 for v in non_null_vals), f"Fractional values truncated: {non_null_vals}"

        # Also check ACDSOOutOfCoreTrainer subsample mode on chunked stream with widening
        c1 = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
        c2 = pd.DataFrame({"x": [1.5, 2.5, 3.5, 4.5], "y": [1, 0, 1, 0]})
        tr = ACDSOOutOfCoreTrainer(task="classification", mode="subsample", max_in_memory_rows=5, chunk_size=4)
        tr.fit_chunks([c1, c2], target_col="y")
        preds = tr.predict(pd.DataFrame({"x": [1.5, 3.5]}))
        assert preds.shape == (2,)

    def test_reservoir_sample_file_and_out_of_core_heterogeneous_reordered_chunks(self, tmp_path: Path) -> None:
        from acdso.scale import ACDSOOutOfCoreTrainer, reservoir_sample_file

        # Chunk 1 (rows 0..4): keys ["a", "b"]
        # Chunk 2 (rows 5..13): reordered keys ["b", "a"] + extra optional key "c"
        jsonl_p = tmp_path / "hetero.jsonl"
        lines = []
        for i in range(5):
            lines.append(json.dumps({"a": i, "b": f"s{i}"}))
        for i in range(5, 14):
            lines.append(json.dumps({"b": f"s{i}", "a": i, "c": float(i) + 0.5}))
        jsonl_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 1. Reservoir path (max_rows=6 triggers transition & Algorithm R across chunks)
        sample_df, total_rows = reservoir_sample_file(jsonl_p, max_rows=6, chunk_size=5, random_state=42)
        assert total_rows == 14
        assert len(sample_df) == 6
        assert list(sample_df.columns) == ["a", "b", "c"]
        for _, row in sample_df.iterrows():
            a_val = int(row["a"])
            assert row["b"] == f"s{a_val}", f"Column scrambling detected: {row.to_dict()}"
            if a_val < 5:
                assert pd.isna(row["c"])
            else:
                assert float(row["c"]) == pytest.approx(float(a_val) + 0.5)

        # 2. Buffered path (max_rows > total_rows)
        buf_df, total_buf = reservoir_sample_file(jsonl_p, max_rows=20, chunk_size=5, random_state=42)
        assert total_buf == 14
        assert len(buf_df) == 14
        assert list(buf_df.columns) == ["a", "b", "c"]
        assert buf_df["a"].tolist() == list(range(14))
        assert buf_df["b"].tolist() == [f"s{i}" for i in range(14)]

        # 3. ACDSOOutOfCoreTrainer(mode="subsample") with reordered & heterogeneous DataFrame chunks
        df_c1 = pd.DataFrame({
            "x1": [0.0, 0.1, 10.0, 10.1, 0.2],
            "x2": [1.0, 1.1, 11.0, 11.1, 1.2],
            "y": [0, 0, 1, 1, 0],
        })
        df_c2 = pd.DataFrame({
            "y": [1, 0, 1, 0, 1],
            "x2": [12.0, 2.0, 12.1, 2.1, 12.2],
            "x1": [11.0, 1.0, 11.1, 1.1, 11.2],
        })
        df_c3 = pd.DataFrame({
            "x2": [3.0, 13.0, 3.1, 13.1],
            "x1": [2.0, 12.0, 2.1, 12.1],
            "x3": [5.0, 6.0, 7.0, 8.0],
            "y": [0, 1, 0, 1],
        })
        trainer = ACDSOOutOfCoreTrainer(
            task="classification",
            mode="subsample",
            max_in_memory_rows=8,
            chunk_size=5,
            random_state=42,
        )
        trainer.fit_chunks([df_c1, df_c2, df_c3], target_col="y")
        preds = trainer.predict(pd.DataFrame({"x1": [0.0, 12.0], "x2": [1.0, 13.0], "x3": [5.0, 8.0]}))
        assert preds.tolist() == [0, 1]

    def test_out_of_core_mid_stream_class_expansion_single_class_and_datetime(self) -> None:
        from acdso.scale import ACDSOOutOfCoreTrainer

        # 1. Warmup fits SGDClassifier with [0, 1] because max_in_memory_rows=20 < total rows=40;
        #    later chunks introduce unseen class 2 -> exercises mid-stream _expand_sgd_classes.
        rng = np.random.RandomState(42)
        c1 = pd.DataFrame({
            "f1": rng.randn(10) - 2.0,
            "f2": rng.randn(10),
            "ts": pd.date_range("2025-01-01", periods=10, freq="D"),
            "y": [0] * 5 + [1] * 5,
        })
        c2 = pd.DataFrame({
            "f1": rng.randn(10),
            "f2": rng.randn(10),
            "ts": pd.date_range("2025-01-11", periods=10, freq="D"),
            "y": [0] * 5 + [1] * 5,
        })
        c3 = pd.DataFrame({
            "f1": rng.randn(20) + 6.0,
            "f2": rng.randn(20),
            "ts": pd.date_range("2025-01-21", periods=20, freq="D"),
            "y": [2] * 20,
        })
        trainer = ACDSOOutOfCoreTrainer(
            task="classification",
            mode="partial_fit",
            chunk_size=10,
            max_in_memory_rows=20,
            random_state=42,
        )
        trainer.fit_chunks([c1, c2, c3], target_col="y")
        assert set(trainer.classes_.tolist()) == {0, 1, 2}
        assert "ts" in trainer.numeric_cols_
        assert "ts" not in trainer.categorical_cols_
        preds = trainer.predict(pd.concat([c1, c3], ignore_index=True).drop(columns=["y"]))
        assert preds.shape == (30,)
        assert 2 in set(preds.tolist())

        # 2. Single-class warmup fallback -> later chunk introduces second class
        s1 = pd.DataFrame({"f1": rng.randn(12) - 2.0, "y": [0] * 12})
        s2 = pd.DataFrame({"f1": rng.randn(12) + 2.0, "y": [1] * 12})
        s3 = pd.DataFrame({"f1": rng.randn(12), "y": [2] * 12})
        tr_single = ACDSOOutOfCoreTrainer(
            task="classification",
            mode="partial_fit",
            chunk_size=12,
            max_in_memory_rows=12,
            random_state=42,
        )
        tr_single.fit_chunks([s1, s2, s3], target_col="y")
        assert tr_single.algorithm_ == "SGDClassifier"
        assert set(tr_single.classes_.tolist()) == {0, 1, 2}

    def test_recommender_constant_ratings_svd_no_arpack_crash(self) -> None:
        from acdso.recommender import ACDSORecommender

        rows = [
            {"user_id": f"u{u}", "item_id": f"i{i}", "rating": 5.0}
            for u in range(8)
            for i in range(8)
            if (u + i) % 2 == 0
        ]
        df = pd.DataFrame(rows)
        rec = ACDSORecommender(method="svd", n_factors=2, random_state=42).fit(df, rating_col="rating")
        preds = rec.predict(["u0", "unseen_u"], ["i1", "unseen_i"])
        assert preds.shape == (2,)
        assert np.allclose(preds, 5.0)
        recs = rec.recommend("u0", n=3)
        assert len(recs) == 3
        ev = rec.evaluate(df, k=3, rating_col="rating")
        assert math.isfinite(ev["rmse"])
        _loads_strict_json(json.dumps(ev, allow_nan=False))

    def test_auto_preprocessing_inf_preservation_and_mixed_object_ohe(self) -> None:
        from acdso.auto_preprocessing import AutoPreprocessor
        from acdso.utils import build_preprocessor, split_columns

        # 1. Numeric column with np.inf and finite variance must NOT be dropped as constant
        df_inf = pd.DataFrame({
            "x_inf": [1.0, 2.0, 3.0, np.inf, 5.0, -np.inf, 7.0, 8.0],
            "x_all_inf": [np.inf, -np.inf, np.inf, np.inf, -np.inf, np.inf, np.inf, np.inf],
            "cat_mix": [1, "alpha", 2, None, "beta", 3, "alpha", 1],
        })
        ap = AutoPreprocessor(missing_threshold=0.5, verbose=False)
        X_out = ap.fit_transform(df_inf)
        assert "x_inf" not in ap.plan_.drop_columns
        assert "x_all_inf" in ap.plan_.drop_columns
        assert np.all(np.isfinite(X_out))

        # 2. utils.build_preprocessor handles mixed int/str object columns cleanly
        num_c, cat_c = split_columns(df_inf[["x_inf", "cat_mix"]])
        pre = build_preprocessor(num_c, cat_c)
        mat = pre.fit_transform(df_inf[["x_inf", "cat_mix"]])
        assert mat.shape[0] == len(df_inf)
        mat_val = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
        assert np.all(np.isfinite(mat_val))

    def test_binary_roc_auc_on_non_zero_one_labels(self) -> None:
        from acdso.core import ACDSOClassifier, _auto_adjust_scorer
        from acdso.risk import _auto_adjust_scorer_for_risk

        y_str = np.array(["neg", "pos"] * 12)
        y_int = np.array([1, 2] * 12)
        y_mc = np.array(["a", "b", "c"] * 8)

        assert _auto_adjust_scorer("roc_auc", y_str, "classification") == "roc_auc"
        assert _auto_adjust_scorer("roc_auc", y_int, "classification") == "roc_auc"
        assert _auto_adjust_scorer("roc_auc", y_mc, "classification") == "roc_auc_ovr_weighted"
        assert _auto_adjust_scorer("f1", y_str, "classification") == "f1_weighted"
        assert _auto_adjust_scorer("roc_auc", pd.Series(["neg", "pos"] * 6 + [None]), "classification") == "roc_auc"
        assert _auto_adjust_scorer("roc_auc", [0, "1"] * 6, "classification") == "roc_auc"
        assert _auto_adjust_scorer("f1", [0, "1"] * 6 + [None], "classification") == "f1_weighted"

        assert _auto_adjust_scorer_for_risk("roc_auc", y_str, "classification") == "roc_auc"
        assert _auto_adjust_scorer_for_risk("roc_auc", y_int, "classification") == "roc_auc"
        assert _auto_adjust_scorer_for_risk("roc_auc", y_mc, "classification") == "roc_auc_ovr_weighted"

        rng = np.random.RandomState(42)
        X = pd.DataFrame({"x1": rng.randn(30), "x2": rng.randn(30)})
        y_bin = np.where(X["x1"] > 0, "pos", "neg")
        clf = ACDSOClassifier(scorer="roc_auc", cv=3, max_candidates=2, random_state=42, verbose=0)
        clf.fit(X, y_bin)
        sc = clf.score(X, y_bin)
        assert math.isfinite(sc) and 0.0 <= sc <= 1.0

    def test_feature_importance_nullable_int64_text_step_and_one_class_fold(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from acdso.feature_importance import SafeFeatureImportance
        from acdso.nlp import TextFeaturizer

        # 1. Nullable Int64 column with fractional median (median([1, 2]) = 1.5)
        X_corr = pd.DataFrame({
            "a": pd.Series([1, 2, pd.NA, 1, 2], dtype="Int64"),
            "b": pd.Series([10.0, 20.0, np.inf, 10.0, 20.0]),
        })
        fi_helper = SafeFeatureImportance(LogisticRegression(), n_bootstrap=0)
        corr_mat, _ = fi_helper._compute_correlations(X_corr)
        assert corr_mat.shape == (2, 2)
        assert np.all(np.isfinite(corr_mat))

        # 2. _has_expanding_or_dropping_step recognizes TextFeaturizer
        pipe_txt = Pipeline([("txt", TextFeaturizer()), ("clf", LogisticRegression())])
        assert SafeFeatureImportance._has_expanding_or_dropping_step(pipe_txt) is True

        # 3. CV 1-class fold fallback (GroupKFold where one training fold has only 1 class)
        X_fold = pd.DataFrame({"x1": np.arange(12, dtype=float), "x2": np.arange(12, dtype=float) * -1.0})
        y_fold = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        # Group g0 has all 0s; g1 and g2 have all 1s -> leaving out g1/g2 still has 0+1, leaving out g0 leaves ONLY 1s!
        g_fold = np.array(["g0"] * 4 + ["g1"] * 4 + ["g2"] * 4)
        base_clf = LogisticRegression().fit(X_fold, y_fold)
        sfi = SafeFeatureImportance(
            base_clf,
            cv_folds=3,
            n_repeats=2,
            n_bootstrap=0,
            random_state=42,
        )
        rep = sfi.compute(X_fold, y_fold, groups=g_fold, cv_strategy="group_kfold")
        assert len(rep.features) == 2
        assert all(math.isfinite(f.importance) for f in rep.features)

    def test_degenerate_survival_ranking_multilabel_and_constant_anomaly(self) -> None:
        from acdso.anomaly import ACDSOAnomalyDetector
        from acdso.multilabel import ACDSOMultiLabelClassifier
        from acdso.ranking import ACDSORanker
        from acdso.survival import ACDSOSurvivalAnalyzer

        # All-censored survival (0 events)
        X_sv = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]})
        sv = ACDSOSurvivalAnalyzer(cv=2, random_state=42).fit(
            X_sv, time=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), event=np.zeros(6, dtype=int)
        )
        surv_fn = sv.predict_survival_function(X_sv.iloc[:2])
        assert np.all(np.isfinite(surv_fn))

        # Constant-relevance ranking
        X_rk = pd.DataFrame({"f1": np.arange(8, dtype=float), "f2": np.ones(8)})
        rk = ACDSORanker(algorithm="pointwise", cv=2, random_state=42).fit(
            X_rk, np.ones(8), groups=np.array(["q1"] * 4 + ["q2"] * 4)
        )
        assert rk.predict(X_rk).shape == (8,)

        # All-zero label column in multilabel
        y_ml = np.column_stack([[0, 1, 0, 1, 0, 1], np.zeros(6, dtype=int)])
        ml = ACDSOMultiLabelClassifier(strategy="multi_output", cv=2, random_state=42).fit(X_sv, y_ml)
        preds_ml = ml.predict(X_sv)
        assert preds_ml.shape == (6, 2)
        assert np.all(preds_ml[:, 1] == 0)

        # Constant-input anomaly detectors (all 3 backends)
        X_const = pd.DataFrame({"a": np.ones(15), "b": np.full(15, 2.0)})
        for algo in ("isolation_forest", "lof", "ocsvm"):
            det = ACDSOAnomalyDetector(algorithm=algo, n_neighbors=5, random_state=42).fit(X_const)
            p = det.predict(X_const.iloc[:4])
            assert p.shape == (4,)
            assert set(np.unique(p)).issubset({-1, 1})
            assert np.all(np.isfinite(det.anomaly_scores(X_const.iloc[:4])))


# ===========================================================================
# Part B: Interfaces R7 Adversarial Regressions
# ===========================================================================


class TestTier5InterfacesR7:
    def test_workbench_apply_nl_plan_reset_and_charts_null_empty_guards_node_vm(
        self, isolated_tier5_webui, tmp_path: Path
    ) -> None:
        if not NODE_EXE.is_file():
            pytest.skip("Node.js not installed")

        _, app, _, _ = isolated_tier5_webui
        client = app.test_client()
        html = client.get("/").get_data(as_text=True)
        m = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
        assert m is not None
        charts_js = client.get("/static/charts.js").get_data(as_text=True)
        wb_js = client.get("/static/workbench.js").get_data(as_text=True)

        bundle_path = tmp_path / "t5_bundle.js"
        bundle_path.write_text("\n".join([m.group(1), charts_js, wb_js.replace("initWorkbench();", "")]), encoding="utf-8")

        driver_path = tmp_path / "t5_driver.js"
        driver_path.write_text(
            r"""
const vm = require('vm');
const fs = require('fs');
const assert = require('assert');

const jsCode = fs.readFileSync(process.argv[2], 'utf8');
const elements = new Map();
function makeEl(id) {
  const el = {
    id, name: id.replace(/^f_/, ''), value: '', checked: false, disabled: false, multiple: false,
    options: [], style: {}, className: '', textContent: '', _innerHTML: '', dataset: {},
    classList: { _s: new Set(), add(c){this._s.add(c);}, remove(c){this._s.delete(c);}, toggle(){}, contains(c){return this._s.has(c);} },
    setAttribute(){}, getAttribute(){}, addEventListener(){}, dispatchEvent(){},
    appendChild(){}, insertAdjacentHTML(){}, querySelectorAll(){ return []; },
  };
  Object.defineProperty(el, 'selectedOptions', { get(){ return this.options.filter((o) => o.selected); } });
  Object.defineProperty(el, 'innerHTML', { get(){return this._innerHTML;}, set(v){this._innerHTML = String(v);} });
  return el;
}
function getEl(id) { if (!elements.has(id)) elements.set(id, makeEl(id)); return elements.get(id); }

const sandbox = {
  assert, console, setTimeout, clearTimeout, setInterval: () => 1, clearInterval: () => {},
  confirm: () => true, Event: function(t){ this.type = t; },
  $: getEl,
  document: { getElementById: getEl, querySelectorAll: () => [], createTextNode: (t) => ({ textContent: t }) },
  fetch: async () => ({ status: 200, ok: true, text: async () => '{"ok":true,"issues":[]}' }),
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(jsCode + `
(() => {
  state.uploadId = 'u_t5';
  state.uploadFileName = 'data.csv';
  state.predictableSet = new Set(['target']);

  const controlSpecs = [
    ['f_task', 'choice'], ['f_cv_strategy', 'choice'], ['f_nlp_method', 'choice'],
    ['f_cluster_algo', 'choice'], ['f_dim_reduce', 'choice'], ['f_cluster_metric', 'choice'],
    ['f_anomaly_algo', 'choice'], ['f_scale_mode', 'choice'],
    ['f_n_clusters', 'int'], ['f_contamination', 'text'],
    ['f_predict', 'columns'],
  ];
  const formCtrls = controlSpecs.map(([id, kind]) => {
    const el = $(id);
    el.dataset = { kind };
    return el;
  });
  $('f_predict').multiple = true;
  $('f_predict').options = [{ value: 'target', selected: false }];
  $('form').querySelectorAll = () => formCtrls;

  // 1. Apply specialized + grouped + NLP + chunked plan with skipped non-predictable target
  applyNlPlan({
    task: 'clustering',
    predict: ['bad_id'],
    cv_strategy: 'group_kfold',
    nlp_method: 'tfidf',
    cluster_algo: 'kmeans',
    dim_reduce: 'pca',
    cluster_metric: 'davies_bouldin',
    anomaly_algo: 'lof',
    scale_mode: 'chunked',
    n_clusters: 4,
    contamination: '0.1',
  });
  assert.strictEqual($('f_cv_strategy').value, 'group_kfold');
  assert.strictEqual($('f_scale_mode').value, 'chunked');
  assert.strictEqual($('f_n_clusters').value, '4');
  assert($('nlPlanStatus').textContent.includes('bad_id'));

  // 2. Re-plan back to plain classification (omitting specialized keys) -> must reset ALL controls
  applyNlPlan({ task: 'classification', predict: ['target'], exclude: [], datetime_col: [] });
  assert.strictEqual($('f_task').value, 'classification');
  assert.strictEqual($('f_cv_strategy').value, 'auto');
  assert.strictEqual($('f_nlp_method').value, 'auto');
  assert.strictEqual($('f_cluster_algo').value, 'auto');
  assert.strictEqual($('f_dim_reduce').value, 'none');
  assert.strictEqual($('f_cluster_metric').value, 'silhouette');
  assert.strictEqual($('f_anomaly_algo').value, 'auto');
  assert.strictEqual($('f_scale_mode').value, 'auto');
  assert.strictEqual($('f_n_clusters').value, '');
  assert.strictEqual($('f_contamination').value, '');
  assert.strictEqual($('nlPlanStatus').textContent, '');

  const payload = collectPayload();
  assert.strictEqual(payload.cv_strategy, 'auto');
  assert.strictEqual(payload.scale_mode, 'auto');
  assert.strictEqual(payload.cluster_algo, undefined);

  // 3. charts.js & workbench.js must NOT coerce null/""/boolean to 0
  const paretoNull = ACDSOCharts.renderParetoPlotHtml(
    [{ name: 'NullScoreModel', metric: null, train_time_s: 1.0 }, { name: 'EmptyScoreModel', metric: '', train_time_s: 1.0 }],
    [],
    '',
    'train_time_s'
  );
  assert.strictEqual(paretoNull, '', 'null/empty metrics must be filtered from Pareto SVG');

  const cmpNull = ACDSOCharts.renderRunComparisonHtml({
    reports: [
      { path: 'a.json', target: 'y', selected_model: 'A', metric: null, deltas: {} },
      { path: 'b.json', target: 'y', selected_model: 'B', metric: '', deltas: {} },
    ],
  });
  assert(!cmpNull.includes('<svg'), 'all-null comparison must not draw phantom 0 score bar SVG');

  state.inspectorData = {
    kind: 'combined',
    task: 'classification',
    selected_model: { name: 'Ridge', metrics: { metric: 0.9 } },
    dataset_info: { rows: '', columns: 4 },
    eda_report: { n_rows: null, n_cols: 4, quality_score: '' },
    risk_report: { missing_ratio: '' },
    summary: {},
    phases: [],
  };
  state.inspectorTab = 'overview';
  renderInspector();
  assert(!$('inspectorBody').innerHTML.includes('0 x 4'), 'empty ds.rows coerced to 0');
  state.inspectorTab = 'eda';
  renderInspector();
  assert($('inspectorBody').innerHTML.includes('n/a'), 'null eda.n_rows must render n/a');
  state.inspectorData.meta = { html_report_path: 123 };
  renderInspector();
  assert(!$('inspectorBody').innerHTML.includes('/api/artifact'), 'non-string html_report_path must be ignored safely');
  assert.strictEqual(fmtVotes({ rule_based: null, learnability: { metric: 'R2', score: '' } }), ' · votes: learnability: n/a');
  console.log('TIER5_NODE_VM_OK');
})();
`, sandbox);
""",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [str(NODE_EXE), str(driver_path), str(bundle_path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
        assert "TIER5_NODE_VM_OK" in proc.stdout

    def test_nl_plan_adversarial_clauses_and_word_boundaries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from acdso.webui.nl_plan import build_plan

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        df = pd.DataFrame({
            "col_a": np.arange(24, dtype=float),
            "patient_id": [f"P{i % 4}" for i in range(24)],
            "session_id": [f"S{i % 4}" for i in range(24)],
            "site_id": [f"ST{i % 4}" for i in range(24)],
            "store_id": [f"STR{i % 4}" for i in range(24)],
            "item": [f"IT{i % 4}" for i in range(24)],
            "user_id": [f"U{i % 4}" for i in range(24)],
            "customer_id": [f"C{i % 4}" for i in range(24)],
            "art": [i % 2 for i in range(24)],
            "review_text": ["excellent quality and fast delivery service"] * 24,
            "target": [i % 2 for i in range(24)],
        })

        # Spaceless ,per and ,using
        p1 = build_plan("predict target excluding col_a,per patient_id", df)
        assert p1["exclude"] == ["col_a"]
        assert p1.get("group_col") == ["patient_id"]

        p2 = build_plan("predict target excluding col_a,using tfidf", df)
        assert p2["exclude"] == ["col_a"]
        assert p2.get("nlp_method") == "tfidf"

        # per session_id / site_id / store_id / item must not leak into exclude
        for noun in ("session_id", "site_id", "store_id", "item"):
            px = build_plan(f"predict target excluding col_a per {noun}", df)
            assert px["exclude"] == ["col_a"], f"Leaked {noun} into exclude: {px}"

        # per session_id sets group_col consistently with _EXCLUDE_CLAUSE_RE
        p_sess = build_plan("predict target per session_id", df)
        assert p_sess.get("group_col") == ["session_id"]
        assert p_sess["predict"] == ["target"]

        # by user_id and by customer_id
        p_u = build_plan("predict target by user_id", df)
        assert p_u.get("group_col") == ["user_id"]
        p_c = build_plan("predict target by customer_id", df)
        assert p_c.get("group_col") == ["customer_id"]

        # "chart" must not falsely match column "art"
        p_art = build_plan("cluster into segments and build a chart", df)
        assert p_art["task"] == "clustering"
        assert p_art["predict"] == []

        # "rank." at EOL
        p_rk = build_plan("train a model to rank.", pd.DataFrame({
            "query_id": ["q1", "q1", "q2", "q2"],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "relevance": [0, 1, 1, 0],
        }))
        assert p_rk["task"] == "ranking"

    def test_column_validator_consistency_and_inf_time_col(self) -> None:
        from acdso.webui.column_validator import validate_columns

        df = pd.DataFrame({
            "dt_native": pd.date_range("2025-01-01", periods=30, freq="D"),
            "dt_str": [f"2025-01-{(i % 28) + 1:02d}" for i in range(30)],
            "t_inf": [1.0, 2.0, float("inf")] + [3.0] * 27,
            "ev": [i % 2 for i in range(30)],
            "q": [f"Q{i % 3}" for i in range(30)],
            "f1": np.arange(30, dtype=float),
            "y": [i % 2 for i in range(30)],
        })

        # Both datetime64 and high-cardinality string date targets warn (not block)
        res_nat = validate_columns(df, {"predict": ["dt_native"]})
        res_str = validate_columns(df, {"predict": ["dt_str"]})
        assert res_nat.ok is True and any(i.severity == "warn" for i in res_nat.issues)
        assert res_str.ok is True and any(i.severity == "warn" for i in res_str.issues)

        # Equal (predict, target) and ranking query_col==group_col allowed
        assert validate_columns(df, {"predict": ["y"], "target": ["y"]}).ok is True
        assert validate_columns(df, {"predict": ["y"], "query_col": ["q"], "group_col": ["q"]}, task="ranking").ok is True

        # clustering ignores stripped predict/target when counting remaining features
        df_two = df[["f1", "y"]]
        assert validate_columns(df_two, {"predict": ["y"], "exclude": ["f1"]}, task="clustering").ok is True

        # +inf in time_col must block
        res_inf = validate_columns(df, {"time_col": ["t_inf"], "event_col": ["ev"]}, task="survival")
        assert res_inf.ok is False

    def test_webui_malformed_report_json_types_no_500_with_testing_false(self, isolated_tier5_webui) -> None:
        _, app, _, rep_dir = isolated_tier5_webui
        app.config["TESTING"] = False
        client = app.test_client()

        bad_rep = rep_dir / "autonomous_bad_combined.json"
        bad_rep.write_text(
            json.dumps({
                "selected_model": "bad",
                "metrics": "bad",
                "risk_report": ["bad"],
                "dataset_info": "bad",
                "candidates_evaluated": 123,
                "pareto_front": "bad",
                "summary": ["bad"],
            }),
            encoding="utf-8",
        )
        ok_rep = rep_dir / "autonomous_ok_combined.json"
        ok_rep.write_text(
            json.dumps({
                "selected_model": {"name": "Ridge", "metrics": {"metric": 0.85}},
                "dataset_info": {"rows": 10, "columns": 3},
                "candidates_evaluated": [{"name": "Ridge", "metric": 0.85}],
            }),
            encoding="utf-8",
        )

        r_sum = client.get("/api/report-summary?path=autonomous_bad_combined.json")
        assert r_sum.status_code == 200, r_sum.get_data(as_text=True)

        r_cmp = client.post(
            "/api/compare-reports",
            json={"paths": ["autonomous_bad_combined.json", "autonomous_ok_combined.json"]},
        )
        assert r_cmp.status_code == 200, r_cmp.get_data(as_text=True)

        r_404 = client.get("/api/nonexistent_endpoint")
        assert r_404.status_code == 404
        assert r_404.is_json

        # Unhashable non-string target entries in models.multi_output.targets must not crash (200 JSON)
        mo_rep = rep_dir / "autonomous_mo_bad_targets_combined.json"
        mo_rep.write_text(
            json.dumps({
                "task": "classification",
                "models": {
                    "multi_output": {
                        "type": "MultiOutputClassifier",
                        "targets": [{"unhashable": 1}, "y1"],
                        "y1": {"f1": 0.88},
                    }
                },
                "html_report_path": 123,
            }),
            encoding="utf-8",
        )
        r_mo = client.get("/api/report-summary?path=autonomous_mo_bad_targets_combined.json")
        assert r_mo.status_code == 200, r_mo.get_data(as_text=True)
        mo_json = r_mo.get_json()
        assert mo_json["meta"]["html_report_path"] is None
        assert "y1" in (mo_json.get("target_reports") or {})

    def test_cli_reporting_fmt_int_numpy_scalars_and_kv_table(self) -> None:
        from acdso.cli.reporting import _fmt_int, _render_kv_table_card

        assert _fmt_int(np.int64(12345)) == "12,345"
        assert _fmt_int(np.float64("nan")) == "N/A"
        assert _fmt_int(np.float64("inf")) == "N/A"
        assert _fmt_int(np.bool_(True)) == "True"

        card = _render_kv_table_card(
            "Stats",
            "Subtitle",
            {"count": np.int64(12345), "nan_val": np.float64("nan"), "flag": np.bool_(True)},
        )
        assert "12,345" in card
        assert "12345.0000" not in card
        assert "<td>N/A</td>" in card
        assert "<td>True</td>" in card

    def test_cli_ranking_mixed_query_col_and_invalid_survival_ranking_guards(self, tmp_path: Path) -> None:
        # 1. Mixed str/int query_col in .pkl succeeds cleanly
        rng = np.random.RandomState(42)
        qids = ["q1", 2, "q3", 4] * 6
        df_rk = pd.DataFrame({
            "qid": pd.Series(qids, dtype=object),
            "f1": rng.randn(24),
            "f2": rng.randn(24),
            "rel": [i % 3 for i in range(24)],
        })
        pkl_p = tmp_path / "mixed_qid.pkl"
        df_rk.to_pickle(pkl_p)
        ws_ok = tmp_path / "ws_ok"
        ws_ok.mkdir()
        res_ok = _run_cli(
            ["--file", str(pkl_p), "--task", "ranking", "--predict", "rel", "--query-col", "qid", "--simple"],
            cwd=ws_ok,
        )
        assert res_ok.returncode == 0, res_ok.stdout + "\n" + res_ok.stderr
        assert (ws_ok / "reports" / "autonomous_mixed_qid_rel_combined.json").is_file()

        # 2. Invalid survival & ranking values emit [ACDSO_PROGRESS] error and return 1 without traceback
        bad_sv = tmp_path / "bad_sv.csv"
        pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "t": ["bad_dur", "2", "3", "4"], "e": [0, 1, 0, 1]}).to_csv(bad_sv, index=False)
        res_sv1 = _run_cli(["--file", str(bad_sv), "--task", "survival", "--time-col", "t", "--event-col", "e"], cwd=tmp_path / "ws_sv1")
        assert res_sv1.returncode == 1
        assert "Traceback" not in res_sv1.stderr
        assert '"error"' in res_sv1.stdout

        neg_sv = tmp_path / "neg_sv.csv"
        pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "t": [-1.0, 2.0, 3.0, 4.0], "e": [0, 1, 0, 2]}).to_csv(neg_sv, index=False)
        res_sv2 = _run_cli(["--file", str(neg_sv), "--task", "survival", "--time-col", "t", "--event-col", "e"], cwd=tmp_path / "ws_sv2")
        assert res_sv2.returncode == 1
        assert "Traceback" not in res_sv2.stderr
        assert '"error"' in res_sv2.stdout

        bad_rk = tmp_path / "bad_rk.csv"
        pd.DataFrame({"qid": ["q1", "q1", "q2", "q2"], "f1": [1.0, 2.0, 3.0, 4.0], "rel": ["bad", "1", "0", "1"]}).to_csv(bad_rk, index=False)
        res_rk_bad = _run_cli(["--file", str(bad_rk), "--task", "ranking", "--predict", "rel", "--query-col", "qid"], cwd=tmp_path / "ws_rk")
        assert res_rk_bad.returncode == 1
        assert "Traceback" not in res_rk_bad.stderr
        assert '"error"' in res_rk_bad.stdout

    def test_tui_validation_multilabel_lookup_none_normalization_and_nan_display(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import acdso.tui.app as tui_app

        # 1. _validate_before_run checks
        assert tui_app._validate_before_run("d.csv", "y", "classification", cv_strategy="group_kfold")[0] is False
        assert tui_app._validate_before_run("d.csv", "y", "classification", cv_strategy="stratified_group_kfold", group_col="")[0] is False
        assert tui_app._validate_before_run("d.csv", "", "survival", time_col="t", event_col="t")[0] is False
        assert tui_app._validate_before_run("d.csv", "t,t", "survival")[0] is False
        assert tui_app._validate_before_run("d.csv", "", "recommender", user_col="u", item_col="u")[0] is False

        # 2. load_latest_combined_summary prioritizes _multilabel_combined.json when "," in target
        rep_dir = tmp_path / "reports"
        rep_dir.mkdir()
        monkeypatch.setattr(tui_app, "REPORTS_DIR", rep_dir)
        (rep_dir / "autonomous_dset_y0_combined.json").write_text(
            json.dumps({"task": "classification", "selected_model": {"name": "StaleSingle"}}), encoding="utf-8"
        )
        (rep_dir / "autonomous_dset_multilabel_combined.json").write_text(
            json.dumps({"task": "multilabel", "selected_model": {"name": "FreshMulti", "metrics": {"metric": float("nan")}}, "audit_trail": {"candidates_evaluated": [{"name": "FreshMulti", "metric": float("nan"), "train_time_s": float("inf")}]}}),
            encoding="utf-8",
        )
        summary = tui_app.load_latest_combined_summary(tmp_path / "dset.csv", "y0,y1")
        assert summary is not None
        assert summary["selected_model"]["name"] == "FreshMulti"

        # 3. _show_results formats NaN/Inf as n/a
        logs: list[str] = []
        ui = tui_app.FallbackTerminalUI(input_fn=lambda _: "q", output_fn=logs.append)
        ui.last_summary = summary
        ui._show_results()
        joined = "\n".join(logs)
        assert "nan" not in joined.lower()
        assert "inf" not in joined.lower()
        assert "n/a" in joined

        # 4. _configure_r1_r6 normalizes "none" on survival/ranking/recommender cols
        answers = iter(["survival", "none", "auto", "none", "none", "none", "auto"])
        ui2 = tui_app.FallbackTerminalUI(input_fn=lambda _: next(answers), output_fn=lambda _: None)
        ui2._configure_r1_r6()
        assert ui2.time_col == ""
        assert ui2.event_col == ""
        assert ui2.group_col == ""
        assert ui2.text_cols == ""
