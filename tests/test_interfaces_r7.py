"""Comprehensive unit and integration tests for R7b (CLI + Reporting) and R7c (TUI)."""

from __future__ import annotations

import asyncio
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_acdso.py"
SRC_ROOT = REPO_ROOT / "src"


class _TagBalanceAuditor(HTMLParser):
    VOID_TAGS = frozenset(
        {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.open_tables = 0
        self.close_tables = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "table":
            self.open_tables += 1
        if t not in self.VOID_TAGS:
            self.stack.append(t)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "table":
            self.close_tables += 1
        if t in self.VOID_TAGS:
            return
        if not self.stack or self.stack[-1] != t:
            self.errors.append(f"Mismatched </{t}> with stack={self.stack[-5:]}")
            if t in self.stack:
                while self.stack and self.stack[-1] != t:
                    self.stack.pop()
                if self.stack:
                    self.stack.pop()
        else:
            self.stack.pop()


def _run_cli(args: list[str], cwd: Path, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(SRC_ROOT)
    return subprocess.run(
        [sys.executable, "-B", str(RUNNER), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _loads_strict_json(path: Path) -> dict[str, Any]:
    def _reject_nonfinite(constant: str) -> None:
        raise ValueError(f"Non-RFC8259 JSON constant: {constant}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)


def _parse_progress(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if line.startswith("[ACDSO_PROGRESS] "):
            events.append(json.loads(line[len("[ACDSO_PROGRESS] "):]))
    return events


# ---------------------------------------------------------------------------
# R7b: CLI Startup, Validation & Task Pipelines
# ---------------------------------------------------------------------------


class TestCLIR7StartupAndValidation:
    def test_cli_help_sub_500ms_and_lists_all_r1_r6_tasks_and_flags(self, tmp_path: Path) -> None:
        t0 = time.perf_counter()
        res = _run_cli(["--help"], cwd=tmp_path, timeout=30.0)
        elapsed_s = time.perf_counter() - t0
        assert res.returncode == 0, res.stderr
        assert elapsed_s < 0.5, f"run_acdso.py --help took {elapsed_s:.3f}s (>0.5s)"

        for task_name in (
            "auto",
            "classification",
            "regression",
            "clustering",
            "anomaly",
            "multilabel",
            "ranking",
            "survival",
            "recommender",
        ):
            assert task_name in res.stdout, f"Missing task {task_name!r} in --help"

        for flag in (
            "--group-col",
            "--cv-strategy",
            "--text-cols",
            "--nlp-method",
            "--cluster-algo",
            "--n-clusters",
            "--dim-reduce",
            "--cluster-metric",
            "--anomaly-algo",
            "--contamination",
            "--event-col",
            "--time-col",
            "--query-col",
            "--user-col",
            "--item-col",
            "--rating-col",
            "--chunk-size",
            "--scale-mode",
        ):
            assert flag in res.stdout, f"Missing flag {flag} in --help"

    @pytest.mark.parametrize(
        "extra_args",
        [
            ["--task", "bogus_task"],
            ["--task", "clustering", "--n-clusters", "1"],
            ["--task", "anomaly", "--contamination", "0.9"],
            ["--scale-mode", "chunked", "--chunk-size", "0"],
            ["--task", "survival", "--simple"],
            ["--task", "recommender", "--simple"],
            ["--task", "ranking", "--predict", "y", "--simple"],
            ["--task", "multilabel", "--simple"],
        ],
    )
    def test_cli_preflight_validation_exits_nonzero(self, tmp_path: Path, extra_args: list[str]) -> None:
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]}).to_csv(csv_path, index=False)
        res = _run_cli(["--file", str(csv_path), *extra_args], cwd=tmp_path, timeout=30.0)
        assert res.returncode != 0


class TestCLIR7TasksAndScaling:
    def test_cli_grouped_cv_classification_writes_cv_summary_and_predictions(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(42)
        n_groups = 12
        rows_per_grp = 8
        patient_id = np.repeat([f"PAT_{i:02d}" for i in range(n_groups)], rows_per_grp)
        risk = np.repeat([1.5 if i % 2 == 0 else -1.5 for i in range(n_groups)], rows_per_grp)
        x1 = risk + rng.normal(0.0, 0.3, size=len(patient_id))
        x2 = rng.normal(0.0, 1.0, size=len(patient_id))
        y = (x1 > 0).astype(int)
        csv_path = tmp_path / "grouped_clf.csv"
        pd.DataFrame({"patient_id": patient_id, "x1": x1, "x2": x2, "target": y}).to_csv(csv_path, index=False)

        res = _run_cli(
            [
                "--file",
                str(csv_path),
                "--task",
                "classification",
                "--predict",
                "target",
                "--group-col",
                "patient_id",
                "--cv-strategy",
                "stratified_group_kfold",
                "--simple",
                "--skip-benchmark",
            ],
            cwd=tmp_path,
            timeout=150.0,
        )
        assert res.returncode == 0, res.stdout + "\n" + res.stderr
        assert "\x1b[" not in res.stdout

        events = _parse_progress(res.stdout)
        assert len(events) >= 10
        done_phases = [int(e["phase"]) for e in events if e["status"] == "done"]
        assert done_phases == list(range(1, 11))

        combined_path = tmp_path / "reports" / "autonomous_grouped_clf_target_combined.json"
        html_path = tmp_path / "reports" / "autonomous_grouped_clf_target_report.html"
        pred_path = tmp_path / "reports" / "predictions_grouped_clf_target_full_dataset.csv"
        assert combined_path.is_file() and html_path.is_file() and pred_path.is_file()

        combined = _loads_strict_json(combined_path)
        assert combined["task"] == "classification"
        assert isinstance(combined.get("metrics"), dict) and len(combined["metrics"]) >= 1
        assert combined.get("cv", {}).get("group_col") == "patient_id"
        import joblib
        bundle = joblib.load(tmp_path / "reports" / "model_grouped_clf_target.pkl")
        assert "patient_id" not in bundle["feature_columns"]
        assert bundle["group_col"] == "patient_id"

        pred_df = pd.read_csv(pred_path)
        assert len(pred_df) == len(patient_id)
        assert "target_predicted" in pred_df.columns

    def test_cli_nlp_text_cols_keeps_text_raw_and_preserves_high_card_categoricals(self, tmp_path: Path) -> None:
        import joblib

        rng = np.random.RandomState(11)
        n_rows = 120
        pos_txt = "fantastic reliable build quality smooth performance highly recommended purchase"
        neg_txt = "defective broken unit terrible customer support refund requested immediately"
        y = np.array([1 if i % 2 == 0 else 0 for i in range(n_rows)])
        df = pd.DataFrame(
            {
                "account_code": [f"ACCT_{i:04d}" for i in range(n_rows)],
                "score_num": np.where(y == 1, rng.uniform(6.0, 10.0, n_rows), rng.uniform(1.0, 4.0, n_rows)),
                "review_text": [pos_txt if y[i] == 1 else neg_txt for i in range(n_rows)],
                "sentiment": y,
            }
        )
        csv_path = tmp_path / "nlp_hc.csv"
        df.to_csv(csv_path, index=False)

        res = _run_cli(
            [
                "--file",
                str(csv_path),
                "--predict",
                "sentiment",
                "--text-cols",
                "review_text",
                "--nlp-method",
                "auto",
                "--simple",
                "--skip-benchmark",
                "--no-auto-exclude",
            ],
            cwd=tmp_path,
            timeout=150.0,
        )
        assert res.returncode == 0, res.stdout + "\n" + res.stderr

        combined = _loads_strict_json(tmp_path / "reports" / "autonomous_nlp_hc_sentiment_combined.json")
        assert combined.get("nlp", {}).get("text_cols") == ["review_text"]
        assert combined["nlp"]["nlp_method"] == "tfidf"

        bundle = joblib.load(tmp_path / "reports" / "model_nlp_hc_sentiment.pkl")
        assert "account_code" in bundle["categorical_encoding"]["high_card_freqs"]
        assert "review_text" not in bundle["categorical_encoding"]["high_card_freqs"]
        assert bundle["text_cols"] == ["review_text"]

        pred_df = pd.read_csv(tmp_path / "reports" / "predictions_nlp_hc_sentiment_full_dataset.csv")
        assert len(pred_df) == n_rows and "sentiment_predicted" in pred_df.columns

    def test_cli_clustering_and_anomaly_targetless_pipelines(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(7)
        c1 = rng.normal([0.0, 0.0], 0.3, size=(25, 2))
        c2 = rng.normal([5.0, 5.0], 0.3, size=(25, 2))
        c3 = rng.normal([0.0, 6.0], 0.3, size=(25, 2))
        pts = np.vstack([c1, c2, c3])
        df_clu = pd.DataFrame(
            {
                "f1": pts[:, 0],
                "f2": pts[:, 1],
                "notes": [
                    "enterprise renewal account" if i < 25 else ("starter promo discount" if i < 50 else "cancel billing dispute")
                    for i in range(len(pts))
                ],
            }
        )
        clu_csv = tmp_path / "clu_input.csv"
        df_clu.to_csv(clu_csv, index=False)

        ws_clu = tmp_path / "ws_clu"
        ws_clu.mkdir()
        res_clu = _run_cli(
            [
                "--file",
                str(clu_csv),
                "--task",
                "clustering",
                "--cluster-algo",
                "kmeans",
                "--n-clusters",
                "3",
                "--dim-reduce",
                "pca",
                "--cluster-metric",
                "silhouette",
                "--text-cols",
                "notes",
                "--simple",
            ],
            cwd=ws_clu,
            timeout=150.0,
        )
        assert res_clu.returncode == 0, res_clu.stdout + "\n" + res_clu.stderr
        events_clu = _parse_progress(res_clu.stdout)
        assert events_clu and events_clu[-1]["status"] == "done"
        comb_clu = _loads_strict_json(ws_clu / "reports" / "autonomous_clu_input_combined.json")
        assert comb_clu["task"] == "clustering"
        assert comb_clu["clustering"]["n_clusters"] == 3
        assert isinstance(comb_clu.get("metrics"), dict) and "silhouette_score" in comb_clu["metrics"]
        assert (ws_clu / "reports" / "predictions_clu_input_cluster_full_dataset.csv").is_file()

        html_clu = (ws_clu / "reports" / "autonomous_clu_input_report.html").read_text(encoding="utf-8")
        aud = _TagBalanceAuditor()
        aud.feed(html_clu)
        assert not aud.errors and not aud.stack and aud.open_tables >= 1 and aud.open_tables == aud.close_tables

        # Anomaly targetless
        norm = rng.normal(0.0, 0.5, size=(60, 3))
        outliers = np.array([[9.0, 9.0, 9.0], [-9.0, 9.0, -9.0], [8.5, -8.5, 8.5]])
        anom_df = pd.DataFrame(np.vstack([norm, outliers]), columns=["x1", "x2", "x3"])
        anom_csv = tmp_path / "anom_input.csv"
        anom_df.to_csv(anom_csv, index=False)

        ws_anom = tmp_path / "ws_anom"
        ws_anom.mkdir()
        res_anom = _run_cli(
            [
                "--file",
                str(anom_csv),
                "--task",
                "anomaly",
                "--anomaly-algo",
                "isolation_forest",
                "--contamination",
                "0.05",
                "--simple",
            ],
            cwd=ws_anom,
            timeout=150.0,
        )
        assert res_anom.returncode == 0, res_anom.stdout + "\n" + res_anom.stderr
        comb_anom = _loads_strict_json(ws_anom / "reports" / "autonomous_anom_input_combined.json")
        assert comb_anom["task"] == "anomaly"
        assert comb_anom["anomaly"]["best_algorithm"] == "isolation_forest"
        assert (ws_anom / "reports" / "predictions_anom_input_anomaly_full_dataset.csv").is_file()

    def test_cli_specialized_r5_tasks_multilabel_ranking_survival_recommender(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(123)

        # 1. Multilabel
        n_ml = 60
        x1 = rng.randn(n_ml)
        x2 = rng.randn(n_ml)
        df_ml = pd.DataFrame(
            {"x1": x1, "x2": x2, "y0": (x1 > 0).astype(int), "y1": (x2 > 0).astype(int)}
        )
        ml_csv = tmp_path / "ml.csv"
        df_ml.to_csv(ml_csv, index=False)
        ws_ml = tmp_path / "ws_ml"
        ws_ml.mkdir()
        res_ml = _run_cli(
            ["--file", str(ml_csv), "--task", "multilabel", "--predict", "y0,y1", "--simple"],
            cwd=ws_ml,
            timeout=150.0,
        )
        assert res_ml.returncode == 0, res_ml.stdout + "\n" + res_ml.stderr
        comb_ml = _loads_strict_json(ws_ml / "reports" / "autonomous_ml_multilabel_combined.json")
        assert comb_ml["task"] == "multilabel"
        assert comb_ml["specialized"]["n_labels"] == 2
        assert (ws_ml / "reports" / "predictions_ml_multilabel_full_dataset.csv").is_file()

        # 2. Ranking
        qids = np.repeat([f"Q{q:02d}" for q in range(8)], 8)
        f1 = rng.uniform(0.0, 1.0, size=len(qids))
        f2 = rng.uniform(0.0, 1.0, size=len(qids))
        rel = np.clip(np.round(2.0 * f1 + f2), 0, 3).astype(int)
        rk_csv = tmp_path / "rank.csv"
        pd.DataFrame({"qid": qids, "f1": f1, "f2": f2, "rel": rel}).to_csv(rk_csv, index=False)
        ws_rk = tmp_path / "ws_rk"
        ws_rk.mkdir()
        res_rk = _run_cli(
            ["--file", str(rk_csv), "--task", "ranking", "--predict", "rel", "--query-col", "qid", "--simple"],
            cwd=ws_rk,
            timeout=150.0,
        )
        assert res_rk.returncode == 0, res_rk.stdout + "\n" + res_rk.stderr
        comb_rk = _loads_strict_json(ws_rk / "reports" / "autonomous_rank_rel_combined.json")
        assert comb_rk["task"] == "ranking"
        assert "ndcg_at_k" in comb_rk["metrics"]
        assert (ws_rk / "reports" / "predictions_rank_rel_full_dataset.csv").is_file()

        # 3. Survival
        n_sv = 80
        z1 = rng.randn(n_sv)
        z2 = rng.randn(n_sv)
        sv_time = np.round(np.maximum(0.2, 10.0 * np.exp(-0.8 * z1) * rng.exponential(1.0, size=n_sv)), 3)
        sv_event = rng.choice([0, 1], size=n_sv, p=[0.3, 0.7])
        sv_site = np.repeat([f"S{i}" for i in range(8)], 10)
        sv_csv = tmp_path / "surv.csv"
        pd.DataFrame({"z1": z1, "z2": z2, "site": sv_site, "time": sv_time, "event": sv_event}).to_csv(sv_csv, index=False)
        ws_sv = tmp_path / "ws_sv"
        ws_sv.mkdir()
        res_sv = _run_cli(
            [
                "--file",
                str(sv_csv),
                "--task",
                "survival",
                "--time-col",
                "time",
                "--event-col",
                "event",
                "--group-col",
                "site",
                "--simple",
            ],
            cwd=ws_sv,
            timeout=150.0,
        )
        assert res_sv.returncode == 0, res_sv.stdout + "\n" + res_sv.stderr
        comb_sv = _loads_strict_json(ws_sv / "reports" / "autonomous_surv_combined.json")
        assert comb_sv["task"] == "survival"
        assert "c_index" in comb_sv["metrics"]
        assert (ws_sv / "reports" / "predictions_surv_time_full_dataset.csv").is_file()

        # 4. Recommender
        rec_rows = [
            {"user_id": f"U{u}", "item_id": f"I{it}", "rating": round(1.0 + ((u + it) % 4), 2)}
            for u in range(8)
            for it in range(6)
            if (u + it) % 2 == 0 or u % 3 == 0
        ]
        rec_csv = tmp_path / "rec.csv"
        pd.DataFrame(rec_rows).to_csv(rec_csv, index=False)
        ws_rec = tmp_path / "ws_rec"
        ws_rec.mkdir()
        res_rec = _run_cli(
            [
                "--file",
                str(rec_csv),
                "--task",
                "recommender",
                "--user-col",
                "user_id",
                "--item-col",
                "item_id",
                "--rating-col",
                "rating",
                "--simple",
            ],
            cwd=ws_rec,
            timeout=150.0,
        )
        assert res_rec.returncode == 0, res_rec.stdout + "\n" + res_rec.stderr
        comb_rec = _loads_strict_json(ws_rec / "reports" / "autonomous_rec_combined.json")
        assert comb_rec["task"] == "recommender"
        assert isinstance(comb_rec.get("specialized"), dict)
        assert (ws_rec / "reports" / "predictions_rec_recommender_full_dataset.csv").is_file()

    @pytest.mark.parametrize("scale_mode", ["chunked", "subsample"])
    def test_cli_scale_modes_chunked_and_subsample(self, tmp_path: Path, scale_mode: str) -> None:
        rng = np.random.RandomState(42)
        n_rows = 450
        x1 = rng.randn(n_rows)
        x2 = rng.randn(n_rows)
        y = (1.5 * x1 - x2 > 0).astype(int)
        csv_path = tmp_path / f"scale_{scale_mode}.csv"
        pd.DataFrame({"x1": x1, "x2": x2, "target": y}).to_csv(csv_path, index=False)

        ws = tmp_path / f"ws_{scale_mode}"
        ws.mkdir()
        res = _run_cli(
            [
                "--file",
                str(csv_path),
                "--task",
                "classification",
                "--predict",
                "target",
                "--scale-mode",
                scale_mode,
                "--chunk-size",
                "120",
                "--simple",
            ],
            cwd=ws,
            timeout=150.0,
        )
        assert res.returncode == 0, res.stdout + "\n" + res.stderr

        combined = _loads_strict_json(ws / "reports" / f"autonomous_scale_{scale_mode}_target_combined.json")
        assert combined["task"] == "classification"
        assert isinstance(combined.get("scale"), dict) and len(combined["scale"]) >= 1
        pred_df = pd.read_csv(ws / "reports" / f"predictions_scale_{scale_mode}_target_full_dataset.csv")
        assert len(pred_df) == n_rows and "target_predicted" in pred_df.columns


# ---------------------------------------------------------------------------
# R7b: Reporting Module Tests
# ---------------------------------------------------------------------------


class TestReportingR7:
    def test_sanitize_json_payload_handles_arrays_series_sets_nonfinite_and_bool(self) -> None:
        from acdso.cli.reporting import _sanitize_json_payload

        payload = {
            "arr": np.array([1.0, float("nan"), float("inf"), 2.5]),
            "series": pd.Series([10, 20]),
            "index": pd.Index(["a", "b"]),
            "tags": {"b", "a"},
            "flag_py": True,
            "flag_np": np.bool_(False),
            "bad_float": float("-inf"),
        }
        sanitized = _sanitize_json_payload(payload)
        encoded = json.dumps(sanitized, allow_nan=False)
        decoded = json.loads(encoded)
        assert decoded["arr"] == [1.0, None, None, 2.5]
        assert decoded["series"] == [10, 20]
        assert decoded["index"] == ["a", "b"]
        assert decoded["tags"] == ["a", "b"]
        assert decoded["flag_py"] is True
        assert decoded["flag_np"] is False
        assert decoded["bad_float"] is None

    def test_generate_combined_report_populates_r1_r2_r6_blocks(self) -> None:
        from acdso.anomaly import ACDSOAnomalyDetector
        from acdso.cli.reporting import generate_combined_report
        from acdso.clustering import ACDSOClusterer
        from acdso.scale import ACDSOOutOfCoreTrainer

        rng = np.random.RandomState(42)
        df = pd.DataFrame({"x1": rng.randn(60), "x2": rng.randn(60)})

        clu = ACDSOClusterer(algorithm="kmeans", n_clusters=3, random_state=42).fit(df)
        rep_clu = generate_combined_report(clu, "clustering log")
        assert rep_clu["task"] == "clustering"
        assert isinstance(rep_clu["clustering"], dict)
        assert isinstance(rep_clu["metrics"], dict) and "silhouette_score" in rep_clu["metrics"]

        anom = ACDSOAnomalyDetector(algorithm="isolation_forest", contamination=0.1, random_state=42).fit(df)
        rep_anom = generate_combined_report(anom, "anomaly log")
        assert rep_anom["task"] == "anomaly"
        assert rep_anom["anomaly"]["best_algorithm"] == "isolation_forest"

        df_sup = df.copy()
        df_sup["y"] = (df_sup["x1"] > 0).astype(int)
        trainer = ACDSOOutOfCoreTrainer(task="classification", chunk_size=20, random_state=42).fit(df.copy(), df_sup["y"])
        rep_scale = generate_combined_report(trainer, "scale log")
        assert rep_scale["task"] == "classification"
        assert isinstance(rep_scale["scale"], dict)

    def test_html_report_renders_r1_r6_cards_xss_escaped_balanced_with_none_selected(self, tmp_path: Path) -> None:
        from acdso.cli.reporting import generate_html_report_from_combined, save_combined_report

        report = {
            "timestamp": "2026-11-24T12:00:00",
            "mode": "autonomous_clustering",
            "task": "clustering <script>alert(1)</script>",
            "selected_model": None,
            "metrics": {"silhouette_score": 0.65, "bad_nan": float("nan"), "is_valid": True},
            "clustering": {"best_algorithm": "kmeans", "n_clusters": 3, "cluster_sizes": {"0": 20, "1": 25}},
            "anomaly": {"best_algorithm": "isolation_forest", "n_anomalies": 4},
            "cv": {"group_col": "patient<id>", "cv_strategy": "group_kfold", "leakage_validated": True},
            "nlp": {"text_cols": ["notes<1>"], "nlp_method": "tfidf"},
            "specialized": {"task": "survival", "metrics": {"c_index": 0.81}},
            "scale": {"mode": "chunked", "chunk_size": 500, "n_rows_seen": 2200},
            "audit_trail": {
                "selected_candidate": "kmeans_3",
                "candidates": [{"algorithm": "kmeans", "n_clusters": 3, "silhouette_score": 0.65, "selected": True}],
                "pareto_front": [],
            },
            "predictions": {
                "target_column": "cluster",
                "total_predictions": 45,
                "task": "clustering",
                "output_file": str(tmp_path / "preds.csv"),
                "statistics": {"mean": float("nan"), "distribution": {"0": 20, "1": 25}},
                "sample_values": [{"row": 0, "predicted": 0, "actual": None}],
            },
        }
        json_out = tmp_path / "rep.json"
        html_out = tmp_path / "rep.html"
        save_combined_report(report, str(json_out))
        _loads_strict_json(json_out)

        generate_html_report_from_combined(report, str(html_out))
        html_text = html_out.read_text(encoding="utf-8")

        assert "<script>alert(1)</script>" not in html_text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_text
        assert "patient&lt;id&gt;" in html_text
        assert "Clustering Summary" in html_text
        assert "Anomaly Detection Summary" in html_text
        assert "Cross-Validation &amp; Grouping" in html_text
        assert "Out-of-Core &amp; Scale Summary" in html_text

        aud = _TagBalanceAuditor()
        aud.feed(html_text)
        assert aud.errors == []
        assert aud.stack == []
        assert aud.open_tables >= 7 and aud.open_tables == aud.close_tables


# ---------------------------------------------------------------------------
# R7c: TUI Wiring Tests
# ---------------------------------------------------------------------------


class TestTUIR7:
    def test_build_cli_command_matrix_all_tasks_and_flags(self, tmp_path: Path) -> None:
        from acdso.tui.app import build_cli_command

        csv_p = str(tmp_path / "d.csv")

        cmd_cls = build_cli_command(
            csv_p,
            "target",
            task="classification",
            group_col="patient_id",
            cv_strategy="stratified_group_kfold",
            text_cols=["notes", "title"],
            nlp_method="tfidf",
            scale_mode="chunked",
            chunk_size=1024,
        )
        assert "--task" in cmd_cls and cmd_cls[cmd_cls.index("--task") + 1] == "classification"
        assert "--predict" in cmd_cls and "target" in cmd_cls
        assert "--group-col" in cmd_cls and "patient_id" in cmd_cls
        assert "--cv-strategy" in cmd_cls and "stratified_group_kfold" in cmd_cls
        assert "--text-cols" in cmd_cls and "notes,title" in cmd_cls
        assert "--nlp-method" in cmd_cls and "tfidf" in cmd_cls
        assert "--scale-mode" in cmd_cls and "chunked" in cmd_cls
        assert "--chunk-size" in cmd_cls and "1024" in cmd_cls

        cmd_clu = build_cli_command(
            csv_p,
            "ignored_target",
            task="clustering",
            cluster_algo="kmeans",
            n_clusters=4,
            dim_reduce="pca",
            cluster_metric="davies_bouldin",
        )
        assert "--task" in cmd_clu and "clustering" in cmd_clu
        assert "--predict" not in cmd_clu and "--target" not in cmd_clu
        assert "--cluster-algo" in cmd_clu and "kmeans" in cmd_clu
        assert "--n-clusters" in cmd_clu and "4" in cmd_clu
        assert "--dim-reduce" in cmd_clu and "pca" in cmd_clu
        assert "--cluster-metric" in cmd_clu and "davies_bouldin" in cmd_clu

        cmd_anom = build_cli_command(csv_p, "", task="anomaly", anomaly_algo="ocsvm", contamination=0.08)
        assert "--task" in cmd_anom and "anomaly" in cmd_anom
        assert "--predict" not in cmd_anom and "--target" not in cmd_anom
        assert "--anomaly-algo" in cmd_anom and "ocsvm" in cmd_anom
        assert "--contamination" in cmd_anom and "0.08" in cmd_anom

        cmd_surv = build_cli_command(csv_p, "", task="survival", time_col="t", event_col="e")
        assert "--task" in cmd_surv and "survival" in cmd_surv
        assert "--time-col" in cmd_surv and "--event-col" in cmd_surv
        assert "--predict" not in cmd_surv

        cmd_rec = build_cli_command(csv_p, "r", task="recommender", user_col="u", item_col="i", rating_col="r")
        assert "--task" in cmd_rec and "recommender" in cmd_rec
        assert "--user-col" in cmd_rec and "--item-col" in cmd_rec and "--rating-col" in cmd_rec
        assert "--predict" not in cmd_rec and "--target" not in cmd_rec

        cmd_rank = build_cli_command(csv_p, "rel", task="ranking", query_col="qid")
        assert "--task" in cmd_rank and "ranking" in cmd_rank
        assert "--query-col" in cmd_rank and "qid" in cmd_rank
        assert "--predict" in cmd_rank

    def test_validate_before_run_matrix(self) -> None:
        from acdso.tui.app import _validate_before_run

        assert _validate_before_run("", "y", "classification")[0] is False
        assert _validate_before_run("d.csv", "", "classification")[0] is False
        assert _validate_before_run("d.csv", "", "clustering")[0] is True
        assert _validate_before_run("d.csv", "", "anomaly")[0] is True
        assert _validate_before_run("d.csv", "", "survival")[0] is False
        assert _validate_before_run("d.csv", "", "survival", time_col="t", event_col="e")[0] is True
        assert _validate_before_run("d.csv", "", "recommender", user_col="u")[0] is False
        assert _validate_before_run("d.csv", "", "recommender", user_col="u", item_col="i")[0] is True
        assert _validate_before_run("d.csv", "rel", "ranking")[0] is False
        assert _validate_before_run("d.csv", "rel", "ranking", query_col="q")[0] is True
        assert _validate_before_run("d.csv", "", "clustering", n_clusters=1)[0] is False
        assert _validate_before_run("d.csv", "", "anomaly", contamination=0.8)[0] is False
        assert _validate_before_run("d.csv", "y", "classification", chunk_size=0)[0] is False

    def test_fallback_terminal_ui_validation_and_task_aware_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import acdso.tui.app as tui_app

        csv_path = tmp_path / "tui_test.csv"
        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "y": [0, 1, 0]}).to_csv(csv_path, index=False)

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (reports_dir / "autonomous_tui_test_combined.json").write_text(
            json.dumps(
                {
                    "task": "clustering",
                    "metrics": {"silhouette_score": 0.72, "n_clusters": 3},
                    "clustering": {"best_algorithm": "kmeans", "n_clusters": 3, "silhouette_score": 0.72},
                    "selected_model": {"name": "kmeans", "metrics": {"silhouette_score": 0.72}},
                    "audit_trail": {"candidates_evaluated": [{"algorithm": "kmeans", "silhouette_score": 0.72}]},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(tui_app, "REPORTS_DIR", reports_dir)

        launched: list[list[str]] = []

        class _FakeProc:
            def __init__(self) -> None:
                self.stdout = iter([])

            def wait(self) -> int:
                return 0

        monkeypatch.setattr(tui_app.subprocess, "Popen", lambda cmd, **kw: (launched.append(list(cmd)), _FakeProc())[1])

        logs_bad: list[str] = []
        ui_bad = tui_app.FallbackTerminalUI(initial_file=str(csv_path), input_fn=lambda _: "q", output_fn=logs_bad.append)
        ui_bad.task = "classification"
        ui_bad.target = ""
        assert ui_bad._run_pipeline() != 0
        assert launched == []
        assert any("target" in ln.lower() for ln in logs_bad)

        logs_ok: list[str] = []
        ui_ok = tui_app.FallbackTerminalUI(initial_file=str(csv_path), input_fn=lambda _: "q", output_fn=logs_ok.append)
        ui_ok.task = "clustering"
        ui_ok.target = ""
        assert ui_ok._run_pipeline() == 0
        assert len(launched) == 1
        assert "--task" in launched[0] and "clustering" in launched[0]
        assert any("Task: clustering" in ln for ln in logs_ok)
        assert any("Clustering:" in ln for ln in logs_ok)

    def test_profile_dataset_bounded_and_load_latest_combined_summary_targetless(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import acdso.cli.io as cli_io
        import acdso.tui.app as tui_app

        csv_path = tmp_path / "bounded.csv"
        pd.DataFrame(
            {
                "num": np.arange(50, dtype=float),
                "notes": ["detailed customer feedback comment about delivery speed"] * 50,
                "target": [i % 2 for i in range(50)],
            }
        ).to_csv(csv_path, index=False)

        def _forbid_full_load(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("profile_dataset must not call full-file load_user_file on CSV")

        monkeypatch.setattr(cli_io, "load_user_file", _forbid_full_load)
        prof = tui_app.profile_dataset(csv_path, max_rows=10)
        assert prof["rows"] == 50
        assert prof["cols"] == 3
        assert "notes" in prof["text_cols"]

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (reports_dir / "autonomous_bounded_combined.json").write_text(
            json.dumps(
                {
                    "task": "anomaly",
                    "metrics": {"n_anomalies": 2},
                    "anomaly": {"best_algorithm": "isolation_forest", "n_anomalies": 2},
                    "selected_model": {"name": "isolation_forest", "metrics": {"n_anomalies": 2}},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(tui_app, "REPORTS_DIR", reports_dir)
        summary = tui_app.load_latest_combined_summary(csv_path, "")
        assert summary is not None
        assert summary["task"] == "anomaly"
        assert summary["anomaly"]["best_algorithm"] == "isolation_forest"
        assert summary["metrics"]["n_anomalies"] == 2

    def test_textual_app_smoke_preserves_widgets_and_populates_targetless(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import acdso.tui.app as tui_app

        if not tui_app.has_textual():
            pytest.skip("textual optional dependency not installed")

        from textual.widgets import DataTable, Input, Select, Static

        csv_path = tmp_path / "tex.csv"
        pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [4.0, 3.0, 2.0, 1.0], "y": [0, 1, 0, 1]}).to_csv(csv_path, index=False)
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (reports_dir / "autonomous_tex_combined.json").write_text(
            json.dumps(
                {
                    "task": "clustering",
                    "metrics": {"silhouette_score": 0.68},
                    "clustering": {"best_algorithm": "kmeans", "n_clusters": 2, "silhouette_score": 0.68},
                    "selected_model": {"name": "kmeans", "metrics": {"silhouette_score": 0.68}},
                    "audit_trail": {"candidates_evaluated": [{"algorithm": "kmeans", "silhouette_score": 0.68}]},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(tui_app, "REPORTS_DIR", reports_dir)

        async def _drive() -> None:
            app = tui_app.create_textual_app(initial_file=str(csv_path))
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause()
                assert app.query_one("#cols-table", DataTable).row_count == 3
                assert "tex.csv" in str(app.query_one("#dataset-meta", Static).render())
                assert app.query_one("#task-select", Select).value == "auto"
                await pilot.click("#suggest-btn")
                await pilot.pause()
                assert app.query_one("#target-input", Input).value != ""
                app._populate_results(str(csv_path), "")
                await pilot.pause()
                assert "clustering" in str(app.query_one("#kpi-banner", Static).render()).lower()
                assert app.query_one("#pareto-table", DataTable).row_count >= 1
                await pilot.press("q")

        asyncio.run(_drive())


# ---------------------------------------------------------------------------
# M6a Iteration 2 Regression Tests
# ---------------------------------------------------------------------------


class TestM6aRegressions:
    def test_survival_two_col_predict_build_cli_command_roundtrip(self, tmp_path: Path) -> None:
        from acdso.tui.app import _validate_before_run, build_cli_command

        rng = np.random.RandomState(301)
        n_sv = 60
        z1 = rng.randn(n_sv)
        z2 = rng.randn(n_sv)
        sv_time = np.round(np.maximum(0.2, 8.0 * np.exp(-0.7 * z1) * rng.exponential(1.0, size=n_sv)), 3)
        sv_event = rng.choice([0, 1], size=n_sv, p=[0.3, 0.7])
        sv_csv = tmp_path / "surv_2col.csv"
        pd.DataFrame({"z1": z1, "z2": z2, "time": sv_time, "event": sv_event}).to_csv(sv_csv, index=False)

        ok, msg = _validate_before_run(str(sv_csv), "time,event", "survival")
        assert ok is True, msg

        cmd = build_cli_command(str(sv_csv), "time,event", task="survival")
        assert "--predict" in cmd and "time,event" in cmd
        cli_args = cmd[cmd.index("--file") :]

        ws = tmp_path / "ws_sv"
        ws.mkdir()
        res = _run_cli(cli_args, cwd=ws, timeout=150.0)
        assert res.returncode == 0, res.stdout + "\n" + res.stderr

        comb = _loads_strict_json(ws / "reports" / "autonomous_surv_2col_combined.json")
        assert comb["task"] == "survival"
        assert "c_index" in comb["metrics"]

    def test_out_of_core_chunked_drops_exclude_and_group_col_and_clustering_block(self, tmp_path: Path) -> None:
        import joblib

        rng = np.random.RandomState(402)
        n_rows = 240
        x1 = rng.randn(n_rows)
        y = (x1 > 0).astype(int)
        df = pd.DataFrame(
            {
                "leak_col": y * 100.0,
                "grp": [f"G{i % 10}" for i in range(n_rows)],
                "x1": x1,
                "target": y,
            }
        )
        csv_p = tmp_path / "sc_excl.csv"
        df.to_csv(csv_p, index=False)

        ws_sup = tmp_path / "ws_sup"
        ws_sup.mkdir()
        res_sup = _run_cli(
            [
                "--file",
                str(csv_p),
                "--predict",
                "target",
                "--exclude",
                "leak_col",
                "--group-col",
                "grp",
                "--scale-mode",
                "chunked",
                "--chunk-size",
                "60",
                "--simple",
            ],
            cwd=ws_sup,
            timeout=150.0,
        )
        assert res_sup.returncode == 0, res_sup.stdout + "\n" + res_sup.stderr
        bundle = joblib.load(ws_sup / "reports" / "model_sc_excl_target.pkl")
        assert "leak_col" not in bundle["feature_columns"]
        assert "grp" not in bundle["feature_columns"]
        assert "x1" in bundle["feature_columns"]

        # Chunked clustering emits clustering summary block and silhouette_score
        c1 = rng.normal([0.0, 0.0], 0.3, size=(60, 2))
        c2 = rng.normal([5.0, 5.0], 0.3, size=(60, 2))
        pts = np.vstack([c1, c2])
        clu_csv = tmp_path / "sc_clu.csv"
        pd.DataFrame({"f1": pts[:, 0], "f2": pts[:, 1], "drop_me": np.arange(len(pts))}).to_csv(clu_csv, index=False)

        ws_clu = tmp_path / "ws_clu"
        ws_clu.mkdir()
        res_clu = _run_cli(
            [
                "--file",
                str(clu_csv),
                "--task",
                "clustering",
                "--exclude",
                "drop_me",
                "--scale-mode",
                "chunked",
                "--chunk-size",
                "40",
                "--n-clusters",
                "2",
                "--simple",
            ],
            cwd=ws_clu,
            timeout=150.0,
        )
        assert res_clu.returncode == 0, res_clu.stdout + "\n" + res_clu.stderr
        comb_clu = _loads_strict_json(ws_clu / "reports" / "autonomous_sc_clu_combined.json")
        assert comb_clu["task"] == "clustering"
        assert isinstance(comb_clu.get("clustering"), dict)
        assert comb_clu["clustering"]["n_clusters"] == 2
        assert "silhouette_score" in comb_clu["metrics"]

    def test_tui_stale_target_report_collision_fallback_and_textual(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import acdso.tui.app as tui_app

        csv_p = tmp_path / "dset.csv"
        pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "target": [0, 1, 0]}).to_csv(csv_p, index=False)

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (reports_dir / "autonomous_dset_target_combined.json").write_text(
            json.dumps(
                {
                    "task": "classification",
                    "metrics": {"accuracy": 0.99},
                    "selected_model": {"name": "StaleClf", "metrics": {"metric": 0.99}},
                }
            ),
            encoding="utf-8",
        )
        (reports_dir / "autonomous_dset_combined.json").write_text(
            json.dumps(
                {
                    "task": "clustering",
                    "metrics": {"silhouette_score": 0.71},
                    "clustering": {"best_algorithm": "kmeans", "n_clusters": 2, "silhouette_score": 0.71},
                    "selected_model": {"name": "FreshKMeans", "metrics": {"silhouette_score": 0.71}},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(tui_app, "REPORTS_DIR", reports_dir)

        class _FakePopen:
            def __init__(self) -> None:
                self.stdout = iter([])

            def wait(self) -> int:
                return 0

        monkeypatch.setattr(tui_app.subprocess, "Popen", lambda *a, **kw: _FakePopen())

        logs: list[str] = []
        ui = tui_app.FallbackTerminalUI(initial_file=str(csv_p), input_fn=lambda _: "q", output_fn=logs.append)
        ui.task = "clustering"
        ui.target = "target"
        assert ui._run_pipeline() == 0
        assert ui.last_summary is not None
        assert ui.last_summary["task"] == "clustering"
        assert ui.last_summary["selected_model"]["name"] == "FreshKMeans"

        if tui_app.has_textual():
            from textual.widgets import Select, Static

            class _FakeAsyncProc:
                def __init__(self) -> None:
                    self.returncode = 0
                    self.stdout = self

                async def readline(self) -> bytes:
                    return b""

                async def wait(self) -> int:
                    return 0

            async def _fake_exec(*args: Any, **kwargs: Any) -> _FakeAsyncProc:
                return _FakeAsyncProc()

            monkeypatch.setattr(tui_app.asyncio, "create_subprocess_exec", _fake_exec)

            async def _drive() -> None:
                app = tui_app.create_textual_app(initial_file=str(csv_p))
                async with app.run_test(size=(120, 36)) as pilot:
                    await pilot.pause()
                    app.query_one("#task-select", Select).value = "clustering"
                    argv = tui_app.build_cli_command(str(csv_p), "target", task="clustering")
                    await app._run_worker(argv, str(csv_p), "target")
                    await pilot.pause()
                    banner_txt = str(app.query_one("#kpi-banner", Static).render())
                    assert "clustering" in banner_txt.lower()
                    assert "FreshKMeans" in banner_txt
                    assert "StaleClf" not in banner_txt
                    await pilot.press("q")

            asyncio.run(_drive())

    def test_tui_validation_bool_guards_and_anomaly_show_results(self) -> None:
        from acdso.tui.app import (
            FallbackTerminalUI,
            _validate_before_run,
            build_cli_command,
            parse_progress_marker,
        )

        ok_nc, msg_nc = _validate_before_run("d.csv", "", "clustering", n_clusters="abc")  # type: ignore[arg-type]
        assert ok_nc is False and "n_clusters" in msg_nc

        ok_cs, msg_cs = _validate_before_run("d.csv", "y", "classification", chunk_size="abc")  # type: ignore[arg-type]
        assert ok_cs is False and "chunk_size" in msg_cs

        cmd_bool_contam = build_cli_command("d.csv", "", task="anomaly", contamination=True)  # type: ignore[arg-type]
        assert "--contamination" not in cmd_bool_contam

        assert parse_progress_marker('[ACDSO_PROGRESS] {"phase": "abc", "total": 4}') is None
        assert parse_progress_marker('[ACDSO_PROGRESS] {"phase": true, "total": 4}') is None

        # FallbackTerminalUI _configure_r1_r6 'auto' reset + anomaly _show_results
        answers = iter(["clustering", "none", "auto", "auto", "kmeans", "auto", "auto"])
        logs: list[str] = []
        ui = FallbackTerminalUI(input_fn=lambda _: next(answers), output_fn=logs.append)
        ui.n_clusters = 4
        ui._configure_r1_r6()
        assert ui.n_clusters is None

        anom_logs: list[str] = []
        ui_anom = FallbackTerminalUI(input_fn=lambda _: "q", output_fn=anom_logs.append)
        ui_anom.last_summary = {
            "task": "anomaly",
            "metrics": {"n_anomalies": 7},
            "anomaly": {"best_algorithm": "isolation_forest", "n_anomalies": 7},
            "selected_model": {"name": "isolation_forest", "metrics": {}},
        }
        ui_anom._show_results()
        joined = "\n".join(anom_logs)
        assert "Selected Model: isolation_forest | CV Score: 7" in joined
        assert "n_anomalies=7" in joined

    def test_reporting_multi_target_html_guards_nonfinite_and_bool(self) -> None:
        from acdso.cli.reporting import _generate_multi_target_html_report

        mt_results = {
            "targets": ["t1"],
            "target_reports": {
                "t1": {
                    "selected_model": {"name": "Ridge", "metrics": {"metric": 0.85}},
                    "benchmark_results": [
                        {
                            "system": "ACDSO",
                            "test_score": float("nan"),
                            "fit_time_s": True,
                            "latency_ms_per_row": float("inf"),
                            "model_name": "Ridge",
                        }
                    ],
                    "eda_report": {"quality_score": 90, "n_rows": float("nan"), "n_cols": 5},
                    "phase_results": {
                        "All": [{"name": "Train", "status": "SUCCESS", "model_name": "Ridge", "time": float("nan")}]
                    },
                }
            },
            "predictions": {"t1": {"output_file": "preds.csv", "count": float("nan")}},
        }
        html_out = _generate_multi_target_html_report(mt_results, "ds")
        assert ">nan<" not in html_out.lower()
        assert ">inf<" not in html_out.lower()
        assert "<td>True</td>" in html_out
        assert "<td>N/A</td>" in html_out

    def test_cli_three_groups_and_nan_group_col_handled(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(505)
        n_per_grp = 15
        groups = np.repeat(["HOSP_A", "HOSP_B", "HOSP_C"], n_per_grp).astype(object)
        x1 = rng.randn(len(groups))
        y = ((x1 + np.repeat([-1.0, 0.0, 1.0], n_per_grp)) > 0).astype(int)
        df = pd.DataFrame({"hosp": groups, "x1": x1, "x2": rng.randn(len(groups)), "y": y})
        df.loc[0, "hosp"] = np.nan
        csv_p = tmp_path / "grp3.csv"
        df.to_csv(csv_p, index=False)

        ws = tmp_path / "ws_grp3"
        ws.mkdir()
        res = _run_cli(
            [
                "--file",
                str(csv_p),
                "--task",
                "classification",
                "--predict",
                "y",
                "--group-col",
                "hosp",
                "--cv-strategy",
                "group_kfold",
                "--simple",
            ],
            cwd=ws,
            timeout=150.0,
        )
        assert res.returncode == 0, res.stdout + "\n" + res.stderr
        comb = _loads_strict_json(ws / "reports" / "autonomous_grp3_y_combined.json")
        assert comb["cv"]["group_col"] == "hosp"
        assert comb["cv"]["n_unique_groups"] == 3

    def test_out_of_core_subsample_auto_task_with_predict_supervised(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(606)
        n_rows = 180
        x1 = rng.randn(n_rows)
        x2 = rng.randn(n_rows)

        # 1. Binary classification with --predict target --scale-mode subsample (default --task auto)
        y_clf = (1.5 * x1 - 0.8 * x2 > 0).astype(int)
        clf_csv = tmp_path / "sub_auto_clf.csv"
        pd.DataFrame({"x1": x1, "x2": x2, "target": y_clf}).to_csv(clf_csv, index=False)
        ws_clf = tmp_path / "ws_sub_clf"
        ws_clf.mkdir()
        res_clf = _run_cli(
            [
                "--file",
                str(clf_csv),
                "--predict",
                "target",
                "--scale-mode",
                "subsample",
                "--chunk-size",
                "60",
                "--simple",
            ],
            cwd=ws_clf,
            timeout=150.0,
        )
        assert res_clf.returncode == 0, res_clf.stdout + "\n" + res_clf.stderr
        comb_clf = _loads_strict_json(ws_clf / "reports" / "autonomous_sub_auto_clf_target_combined.json")
        assert comb_clf["task"] == "classification"
        assert comb_clf["selected_model"]["name"] != "KMeans_subsample"
        assert comb_clf["scale"]["mode"] == "subsample"
        pred_clf = pd.read_csv(ws_clf / "reports" / "predictions_sub_auto_clf_target_full_dataset.csv")
        assert len(pred_clf) == n_rows
        assert set(pred_clf["target_predicted"].unique()).issubset({0, 1})

        # 2. Continuous regression with --predict target --scale-mode subsample (default --task auto)
        y_reg = 3.0 * x1 - 1.5 * x2 + rng.normal(0.0, 0.2, size=n_rows)
        reg_csv = tmp_path / "sub_auto_reg.csv"
        pd.DataFrame({"x1": x1, "x2": x2, "target": y_reg}).to_csv(reg_csv, index=False)
        ws_reg = tmp_path / "ws_sub_reg"
        ws_reg.mkdir()
        res_reg = _run_cli(
            [
                "--file",
                str(reg_csv),
                "--predict",
                "target",
                "--scale-mode",
                "subsample",
                "--chunk-size",
                "60",
                "--simple",
            ],
            cwd=ws_reg,
            timeout=150.0,
        )
        assert res_reg.returncode == 0, res_reg.stdout + "\n" + res_reg.stderr
        comb_reg = _loads_strict_json(ws_reg / "reports" / "autonomous_sub_auto_reg_target_combined.json")
        assert comb_reg["task"] == "regression"
        assert comb_reg["selected_model"]["name"] == "Ridge_subsample"
        pred_reg = pd.read_csv(ws_reg / "reports" / "predictions_sub_auto_reg_target_full_dataset.csv")
        assert len(pred_reg) == n_rows
        corr = float(np.corrcoef(y_reg, pred_reg["target_predicted"].to_numpy(dtype=float))[0, 1])
        assert corr > 0.7

        # 3. Targetless --scale-mode subsample (default --task auto, no --target/--predict) still resolves to clustering
        unsup_csv = tmp_path / "sub_auto_unsup.csv"
        pd.DataFrame({"x1": x1, "x2": x2}).to_csv(unsup_csv, index=False)
        ws_unsup = tmp_path / "ws_sub_unsup"
        ws_unsup.mkdir()
        res_unsup = _run_cli(
            [
                "--file",
                str(unsup_csv),
                "--scale-mode",
                "subsample",
                "--chunk-size",
                "60",
                "--n-clusters",
                "3",
                "--simple",
            ],
            cwd=ws_unsup,
            timeout=150.0,
        )
        assert res_unsup.returncode == 0, res_unsup.stdout + "\n" + res_unsup.stderr
        comb_unsup = _loads_strict_json(ws_unsup / "reports" / "autonomous_sub_auto_unsup_combined.json")
        assert comb_unsup["task"] == "clustering"

    @pytest.mark.parametrize("group_kind", ["string", "numeric"])
    def test_survival_and_multilabel_nan_group_col_no_overlap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, group_kind: str
    ) -> None:
        import acdso.cv as acdso_cv
        import run_acdso

        rng = np.random.RandomState(707)
        n_per_grp = 18
        raw_grps = (
            np.repeat(["SITE_A", "SITE_B", "SITE_C"], n_per_grp).astype(object)
            if group_kind == "string"
            else np.repeat([101.0, 202.0, 303.0], n_per_grp).astype(float)
        )
        n_total = len(raw_grps)
        z1 = rng.randn(n_total)
        z2 = rng.randn(n_total)

        # Survival dataset with 1 NaN in group_col
        sv_time = np.round(np.maximum(0.2, 8.0 * np.exp(-0.6 * z1) * rng.exponential(1.0, size=n_total)), 3)
        sv_event = rng.choice([0, 1], size=n_total, p=[0.3, 0.7])
        sv_df = pd.DataFrame({"grp": raw_grps, "z1": z1, "z2": z2, "time": sv_time, "event": sv_event})
        sv_df.loc[0, "grp"] = np.nan
        sv_csv = tmp_path / f"surv_{group_kind}.csv"
        sv_df.to_csv(sv_csv, index=False)

        ws_sv = tmp_path / f"ws_sv_{group_kind}"
        ws_sv.mkdir()
        res_sv = _run_cli(
            [
                "--file",
                str(sv_csv),
                "--task",
                "survival",
                "--time-col",
                "time",
                "--event-col",
                "event",
                "--group-col",
                "grp",
                "--simple",
            ],
            cwd=ws_sv,
            timeout=150.0,
        )
        assert res_sv.returncode == 0, res_sv.stdout + "\n" + res_sv.stderr
        comb_sv = _loads_strict_json(ws_sv / "reports" / f"autonomous_surv_{group_kind}_combined.json")
        assert comb_sv["task"] == "survival"
        assert comb_sv["cv"]["leakage_validated"] is True
        assert comb_sv["cv"]["n_unique_groups"] == 3
        pred_sv = pd.read_csv(ws_sv / "reports" / f"predictions_surv_{group_kind}_time_full_dataset.csv")
        assert len(pred_sv) == n_total - 1
        assert pred_sv["grp"].isna().sum() == 0

        # Multilabel dataset with 1 NaN in group_col
        ml_df = pd.DataFrame(
            {
                "grp": raw_grps,
                "z1": z1,
                "z2": z2,
                "y0": (z1 > 0).astype(int),
                "y1": (z2 > 0).astype(int),
            }
        )
        ml_df.loc[0, "grp"] = np.nan
        ml_csv = tmp_path / f"ml_{group_kind}.csv"
        ml_df.to_csv(ml_csv, index=False)

        ws_ml = tmp_path / f"ws_ml_{group_kind}"
        ws_ml.mkdir()
        res_ml = _run_cli(
            [
                "--file",
                str(ml_csv),
                "--task",
                "multilabel",
                "--predict",
                "y0,y1",
                "--group-col",
                "grp",
                "--simple",
            ],
            cwd=ws_ml,
            timeout=150.0,
        )
        assert res_ml.returncode == 0, res_ml.stdout + "\n" + res_ml.stderr
        comb_ml = _loads_strict_json(ws_ml / "reports" / f"autonomous_ml_{group_kind}_multilabel_combined.json")
        assert comb_ml["task"] == "multilabel"
        assert comb_ml["cv"]["leakage_validated"] is True
        assert comb_ml["cv"]["n_unique_groups"] == 3
        pred_ml = pd.read_csv(ws_ml / "reports" / f"predictions_ml_{group_kind}_multilabel_full_dataset.csv")
        assert len(pred_ml) == n_total - 1
        assert pred_ml["grp"].isna().sum() == 0

        # Verify in-process zero group overlap in group_holdout_split for both pipelines
        seen_overlaps: list[set[Any]] = []
        orig_holdout = acdso_cv.group_holdout_split

        def _spy_holdout(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
            tr_idx, te_idx = orig_holdout(*args, **kwargs)
            g_arg = kwargs.get("groups")
            if g_arg is not None:
                seen_overlaps.append(set(g_arg[tr_idx]).intersection(set(g_arg[te_idx])))
            return tr_idx, te_idx

        monkeypatch.setattr(acdso_cv, "group_holdout_split", _spy_holdout)
        ws_spy = tmp_path / f"ws_spy_{group_kind}"
        ws_spy.mkdir()
        monkeypatch.chdir(ws_spy)
        assert run_acdso._run_survival_pipeline(
            file_path=str(sv_csv),
            target=None,
            time_col="time",
            event_col="event",
            group_col="grp",
        )
        assert run_acdso._run_multilabel_pipeline(
            file_path=str(ml_csv),
            target=None,
            predict="y0,y1",
            group_col="grp",
        )
        assert len(seen_overlaps) == 2
        assert all(len(ov) == 0 for ov in seen_overlaps)

    def test_reporting_ts_eda_nonfinite_and_tui_overflow_guards(self) -> None:
        from acdso.cli.reporting import _generate_multi_target_html_report
        from acdso.tui.app import _validate_before_run, parse_progress_marker

        mt_results = {
            "targets": ["t1"],
            "target_reports": {
                "t1": {
                    "selected_model": {"name": "Ridge", "metrics": {"metric": 0.85}},
                    "timeseries": {"best_model": "ARIMA", "cv_score": float("nan")},
                    "timeseries_best": {"best_model": "ARIMA", "rmse": float("inf")},
                    "timeseries_leaderboard": [
                        {
                            "System": "AutoTS",
                            "RMSE": float("nan"),
                            "MAE": float("inf"),
                            "Time(s)": True,
                        }
                    ],
                    "eda_report": {
                        "quality_score": float("nan"),
                        "n_rows": float("inf"),
                        "n_cols": float("inf"),
                    },
                }
            },
            "predictions": {"t1": {"output_file": "preds.csv", "count": 10}},
        }
        html_out = _generate_multi_target_html_report(mt_results, "ds_ts_eda")
        assert ">nan<" not in html_out.lower()
        assert ">inf<" not in html_out.lower()
        assert "nan/100" not in html_out.lower()
        assert "N/A/100" in html_out
        assert "<td>True</td>" in html_out
        assert "<td>N/A</td>" in html_out

        aud = _TagBalanceAuditor()
        aud.feed(html_out)
        assert aud.errors == []
        assert aud.stack == []

        # TUI OverflowError / non-finite guards
        assert parse_progress_marker('[ACDSO_PROGRESS] {"phase": 1e1000, "total": 4}') is None
        assert parse_progress_marker('[ACDSO_PROGRESS] {"phase": 1, "total": 1e1000}') is None
        assert parse_progress_marker('[ACDSO_PROGRESS] {"phase": NaN, "total": 4}') is None
        assert _validate_before_run("d.csv", "", "clustering", n_clusters=float("inf"))[0] is False  # type: ignore[arg-type]
        assert _validate_before_run("d.csv", "", "clustering", n_clusters=float("nan"))[0] is False  # type: ignore[arg-type]
        assert _validate_before_run("d.csv", "y", "classification", chunk_size=float("inf"))[0] is False  # type: ignore[arg-type]
        assert _validate_before_run("d.csv", "y", "classification", chunk_size=float("nan"))[0] is False  # type: ignore[arg-type]



# ===========================================================================
# R7d: WebUI Startup, FLAGS, Validation, NL Plan, Reporting & Frontend Sync
# ===========================================================================
import io
import re


def _loads_strict_json_text(raw_text: str) -> dict[str, Any]:
    def _reject_nonfinite(constant: str) -> None:
        raise ValueError(f"Non-RFC8259 JSON constant: {constant}")

    return json.loads(raw_text, parse_constant=_reject_nonfinite)


@pytest.fixture()
def isolated_r7_webui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide an isolated Flask test client whose uploads/reports/logs stay in tmp_path."""
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
    app.config["TESTING"] = True
    yield webui_app, app.test_client(), up_dir, rep_dir
    webui_app._invalidate_reports_cache()


class TestWebUIR7StartupAndFlags:
    def test_create_app_sub_500ms_and_lazy_imports(self) -> None:
        probe = (
            "import sys, time; "
            "t0 = time.perf_counter(); "
            "import acdso.webui; "
            "app = acdso.webui.create_app(); "
            "dt = time.perf_counter() - t0; "
            "heavy = [m for m in ('pandas', 'numpy', 'sklearn', 'scipy') if m in sys.modules]; "
            "print(f'{dt:.4f}|' + ','.join(heavy))"
        )
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        proc = subprocess.run(
            [sys.executable, "-B", "-c", probe],
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=30.0,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
        dt_str, heavy_str = proc.stdout.strip().splitlines()[-1].split("|", 1)
        assert float(dt_str) < 0.5, f"create_app() took {dt_str}s (>= 0.5s)"
        assert heavy_str == "", f"Eager heavy imports on create_app(): {heavy_str}"

    def test_flags_cover_all_r7_tasks_and_18_flags(self) -> None:
        from acdso.webui.app import FLAGS_BY_NAME

        expected_tasks = (
            "auto",
            "classification",
            "regression",
            "clustering",
            "anomaly",
            "multilabel",
            "ranking",
            "survival",
            "recommender",
        )
        assert tuple(FLAGS_BY_NAME["task"].choices) == expected_tasks

        expected_18 = {
            "group_col": ("--group-col", "columns"),
            "cv_strategy": ("--cv-strategy", "choice"),
            "text_cols": ("--text-cols", "columns"),
            "nlp_method": ("--nlp-method", "choice"),
            "cluster_algo": ("--cluster-algo", "choice"),
            "n_clusters": ("--n-clusters", "int"),
            "dim_reduce": ("--dim-reduce", "choice"),
            "cluster_metric": ("--cluster-metric", "choice"),
            "anomaly_algo": ("--anomaly-algo", "choice"),
            "contamination": ("--contamination", "text"),
            "event_col": ("--event-col", "columns"),
            "time_col": ("--time-col", "columns"),
            "query_col": ("--query-col", "columns"),
            "user_col": ("--user-col", "columns"),
            "item_col": ("--item-col", "columns"),
            "rating_col": ("--rating-col", "columns"),
            "chunk_size": ("--chunk-size", "int"),
            "scale_mode": ("--scale-mode", "choice"),
        }
        for name, (cli_flag, kind) in expected_18.items():
            assert name in FLAGS_BY_NAME, f"Missing WebUI flag {name!r}"
            assert FLAGS_BY_NAME[name].cli == cli_flag
            assert FLAGS_BY_NAME[name].kind == kind

    def test_build_argv_r1_r6_flags_bounds_and_suppression(self, tmp_path: Path) -> None:
        from acdso.webui.app import build_argv

        dummy_file = tmp_path / "data.csv"
        dummy_file.write_text("a,b\n1,2\n", encoding="utf-8")

        # List and comma-string columns + target suppression on clustering
        argv_clu = build_argv(
            {
                "task": "clustering",
                "predict": ["ignored_target"],
                "target": ["ignored_target"],
                "group_col": ["entity_id"],
                "text_cols": "review_a, review_b",
                "nlp_method": "tfidf",
                "cluster_algo": "kmeans",
                "n_clusters": 4,
                "dim_reduce": "pca",
                "cluster_metric": "davies_bouldin",
                "chunk_size": 1000,
                "scale_mode": "subsample",
            },
            dummy_file,
        )
        assert "--predict" not in argv_clu and "--target" not in argv_clu
        for tok in (
            "--task", "clustering",
            "--group-col", "entity_id",
            "--text-cols", "review_a,review_b",
            "--nlp-method", "tfidf",
            "--cluster-algo", "kmeans",
            "--n-clusters", "4",
            "--dim-reduce", "pca",
            "--cluster-metric", "davies_bouldin",
            "--chunk-size", "1000",
            "--scale-mode", "subsample",
        ):
            assert tok in argv_clu, f"Missing {tok!r} in {argv_clu}"

        # Contamination handling ("auto" omitted; finite float emitted)
        argv_anom_auto = build_argv({"task": "anomaly", "contamination": "auto"}, dummy_file)
        assert "--contamination" not in argv_anom_auto
        argv_anom_num = build_argv({"task": "anomaly", "contamination": "0.08"}, dummy_file)
        assert argv_anom_num[argv_anom_num.index("--contamination") + 1] == "0.08"

        # Validation rejects invalid choices, bools, OverflowError, and out-of-range values
        for bad_payload in (
            {"task": "invalid_task"},
            {"task": True},
            {"n_clusters": 1},
            {"n_clusters": True},
            {"n_clusters": 10**400},
            {"chunk_size": 0},
            {"contamination": "0.0"},
            {"contamination": "0.75"},
            {"contamination": "nan"},
            {"contamination": True},
            {"cv_strategy": "bad_cv"},
        ):
            with pytest.raises(ValueError):
                build_argv(bad_payload, dummy_file)


class TestWebUIR7ValidationAndNLPlan:
    def test_nl_plan_r1_r6_tasks_and_exclusions(self, isolated_r7_webui, monkeypatch: pytest.MonkeyPatch) -> None:
        import pandas as pd

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _, client, _, _ = isolated_r7_webui

        df = pd.DataFrame({
            "patient_id": [f"P{i % 5}" for i in range(30)],
            "user_id": [f"U{i % 6}" for i in range(30)],
            "item_id": [f"I{i % 4}" for i in range(30)],
            "rating": [float((i % 5) + 1) for i in range(30)],
            "time": [float(i + 1) for i in range(30)],
            "event": [i % 2 for i in range(30)],
            "query_id": [f"Q{i % 5}" for i in range(30)],
            "review_text": ["excellent battery quality and fast screen performance" for _ in range(30)],
            "x1": [float(i) for i in range(30)],
            "target": [i % 2 for i in range(30)],
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        up = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(buf.getvalue()), "nl_r1_r6.csv")},
            content_type="multipart/form-data",
        )
        assert up.status_code == 200
        uid = up.get_json()["upload_id"]

        r_clu = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "cluster customers into segments without a target"})
        assert r_clu.status_code == 200
        assert r_clu.get_json()["task"] == "clustering"
        assert r_clu.get_json()["predict"] == []

        r_anom = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "detect anomalies using isolation forest"})
        assert r_anom.status_code == 200
        assert r_anom.get_json()["task"] == "anomaly"
        assert r_anom.get_json()["predict"] == []

        r_surv = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "run cox survival analysis for censored patient hazard"})
        assert r_surv.status_code == 200
        surv_plan = r_surv.get_json()
        assert surv_plan["task"] == "survival"
        assert surv_plan.get("time_col") == ["time"]
        assert surv_plan.get("event_col") == ["event"]

        r_rec = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "build a collaborative filtering recommender"})
        assert r_rec.status_code == 200
        rec_plan = r_rec.get_json()
        assert rec_plan["task"] == "recommender"
        assert rec_plan.get("user_col") == ["user_id"]
        assert rec_plan.get("item_col") == ["item_id"]

        r_excl = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "predict target excluding patient_id"})
        assert r_excl.status_code == 200
        excl_plan = r_excl.get_json()
        assert "target" in excl_plan["predict"]
        assert "patient_id" in excl_plan["exclude"]

    def test_validate_columns_supports_all_r1_r6_roles_and_conflicts(self, isolated_r7_webui) -> None:
        import pandas as pd

        _, client, _, _ = isolated_r7_webui
        df = pd.DataFrame({
            "patient_id": [f"P{i % 5}" for i in range(25)],
            "t_col": [float(i + 1) for i in range(25)],
            "neg_t": [float(i - 5) for i in range(25)],
            "e_col": [i % 2 for i in range(25)],
            "bad_e": [i % 3 for i in range(25)],
            "q_col": [f"Q{i % 4}" for i in range(25)],
            "u_col": [f"U{i % 5}" for i in range(25)],
            "i_col": [f"I{i % 5}" for i in range(25)],
            "r_col": [float((i % 5) + 1) for i in range(25)],
            "bad_r": ["good" if i % 2 == 0 else "bad" for i in range(25)],
            "review_txt": ["solid product value and reliable delivery" for _ in range(25)],
            "target": [i % 2 for i in range(25)],
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        up = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(buf.getvalue()), "val_r1_r6.csv")},
            content_type="multipart/form-data",
        )
        uid = up.get_json()["upload_id"]

        # Valid string & list forms
        assert client.post(
            "/api/validate-columns",
            json={"upload_id": uid, "predict": ["target"], "group_col": "patient_id", "text_cols": ["review_txt"]},
        ).get_json()["ok"] is True

        assert client.post(
            "/api/validate-columns",
            json={"upload_id": uid, "task": "survival", "time_col": ["t_col"], "event_col": "e_col"},
        ).get_json()["ok"] is True

        assert client.post(
            "/api/validate-columns",
            json={"upload_id": uid, "task": "recommender", "user_col": "u_col", "item_col": "i_col", "rating_col": "r_col"},
        ).get_json()["ok"] is True

        # Blocking conflicts & invalid role types
        for bad_body in (
            {"upload_id": uid, "predict": ["target"], "group_col": ["target"]},
            {"upload_id": uid, "time_col": ["t_col"], "event_col": ["t_col"]},
            {"upload_id": uid, "task": "recommender", "user_col": "u_col", "item_col": "u_col"},
            {"upload_id": uid, "group_col": ["missing_col"]},
            {"upload_id": uid, "task": "survival", "time_col": ["neg_t"], "event_col": ["e_col"]},
            {"upload_id": uid, "task": "survival", "time_col": ["t_col"], "event_col": ["bad_e"]},
            {"upload_id": uid, "task": "recommender", "user_col": "u_col", "item_col": "i_col", "rating_col": "bad_r"},
        ):
            res = client.post("/api/validate-columns", json=bad_body)
            assert res.status_code == 200
            assert res.get_json()["ok"] is False, f"Expected ok=False for {bad_body}"

    def test_run_targetless_and_specialized_validation_gates(
        self, isolated_r7_webui, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pandas as pd

        webui_app, client, _, _ = isolated_r7_webui
        stub = tmp_path / "stub_runner.py"
        stub.write_text("import sys; print('ARGS:', sys.argv[1:], flush=True)\n", encoding="utf-8")
        monkeypatch.setattr(webui_app, "RUNNER", stub)

        df = pd.DataFrame({
            "t_col": [1.0 + i for i in range(20)],
            "e_col": [i % 2 for i in range(20)],
            "q_col": [f"Q{i % 4}" for i in range(20)],
            "u_col": [f"U{i % 4}" for i in range(20)],
            "i_col": [f"I{i % 4}" for i in range(20)],
            "x1": [float(i) for i in range(20)],
            "target": [i % 2 for i in range(20)],
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        uid = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(buf.getvalue()), "gate_r1_r6.csv")},
            content_type="multipart/form-data",
        ).get_json()["upload_id"]

        # Must return 400
        for bad in (
            {"upload_id": uid},
            {"upload_id": uid, "task": "invalid_task", "predict": ["target"]},
            {"upload_id": uid, "task": "survival"},
            {"upload_id": uid, "task": "recommender", "user_col": "u_col"},
            {"upload_id": uid, "task": "ranking", "predict": ["target"]},
            {"upload_id": uid, "task": "multilabel"},
            {"upload_id": uid, "task": "clustering", "n_clusters": 1},
            {"upload_id": uid, "task": "clustering", "chunk_size": 0},
            {"upload_id": uid, "task": "anomaly", "contamination": "0.9"},
        ):
            assert client.post("/api/run", json=bad).status_code == 400, f"Expected 400 for {bad}"

        # Permitted targetless / specialized runs
        for good in (
            {"upload_id": uid, "task": "clustering", "cluster_algo": "kmeans", "n_clusters": 3},
            {"upload_id": uid, "task": "anomaly", "anomaly_algo": "isolation_forest", "contamination": "0.1"},
            {"upload_id": uid, "task": "survival", "time_col": "t_col", "event_col": "e_col"},
            {"upload_id": uid, "task": "recommender", "user_col": "u_col", "item_col": "i_col"},
        ):
            r = client.post("/api/run", json=good)
            assert r.status_code == 200, f"Expected 200 for {good}, got {r.get_data(as_text=True)}"
            jid = r.get_json()["job_id"]
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                st = client.get(f"/api/job/{jid}").get_json()
                if st["status"] != "running":
                    break
                time.sleep(0.05)


class TestWebUIR7ReportingAndComparison:
    def test_summarize_json_report_preserves_r1_r7_sections_and_sanitizes_nonfinite(
        self, isolated_r7_webui
    ) -> None:
        _, client, _, rep_dir = isolated_r7_webui

        raw_report = {
            "task": "clustering",
            "mode": "clustering",
            "metrics": {"silhouette_score": 0.74, "davies_bouldin_score": 0.41, "bad_nan": float("nan")},
            "selected_model": {"name": "kmeans", "metrics": {"metric": 0.74, "train_time_s": 0.25}},
            "audit_trail": {"candidates": [{"algorithm": "kmeans", "silhouette_score": 0.74}]},
            "clustering": {"best_algorithm": "kmeans", "n_clusters": 3, "cluster_sizes": {"0": 30, "1": 30}},
            "anomaly": {"best_algorithm": "isolation_forest", "n_anomalies": 3},
            "cv": {"group_col": "patient_id", "cv_strategy": "stratified_group_kfold", "leakage_validated": True},
            "nlp": {"nlp_method": "tfidf", "text_cols": ["review_text"], "n_text_cols": 1},
            "specialized": {"strategy": "classifier_chain", "n_labels": 3},
            "scale": {"mode": "chunked", "chunk_size": 500, "total_rows": 2200},
            "dataset_info": {"rows": 90, "columns": 4},
        }
        (rep_dir / "r7_full_blocks_combined.json").write_text(json.dumps(raw_report), encoding="utf-8")

        resp = client.get("/api/report-summary?path=r7_full_blocks_combined.json")
        assert resp.status_code == 200
        strict_parsed = _loads_strict_json_text(resp.get_data(as_text=True))
        assert strict_parsed["task"] == "clustering"
        assert strict_parsed["meta"]["task"] == "clustering"
        assert strict_parsed["metrics"]["bad_nan"] is None
        for key in ("metrics", "clustering", "anomaly", "cv", "nlp", "specialized", "scale"):
            assert isinstance(strict_parsed.get(key), dict) and len(strict_parsed[key]) >= 1
        assert len(strict_parsed["candidates_evaluated"]) == 1

    def test_compare_reports_mixed_tasks_and_specialized_kpis(self, isolated_r7_webui) -> None:
        _, client, _, rep_dir = isolated_r7_webui

        (rep_dir / "cmp_cls_combined.json").write_text(
            json.dumps({
                "task": "classification",
                "target": "label",
                "selected_model": {"name": "LightGBM", "metrics": {"metric": 0.91, "train_time_s": 0.5, "peak_mem_mb": 12.0}},
                "dataset_info": {"rows": 120, "columns": 5},
            }),
            encoding="utf-8",
        )
        (rep_dir / "cmp_clu_combined.json").write_text(
            json.dumps({
                "task": "clustering",
                "metrics": {"silhouette_score": 0.68},
                "selected_model": {"name": "kmeans", "metrics": {"train_time_s": 0.2, "peak_mem_mb": True}},
                "dataset_info": {"rows": 120, "columns": 4},
            }),
            encoding="utf-8",
        )

        resp = client.post("/api/compare-reports", json={"paths": ["cmp_cls_combined.json", "cmp_clu_combined.json"]})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["reports"][0]["task"] == "classification"
        assert data["reports"][1]["task"] == "clustering"
        assert data["reports"][1]["metric"] == pytest.approx(0.68)
        assert data["reports"][1]["deltas"]["metric"] == pytest.approx(-0.23)
        assert data["reports"][1]["deltas"]["peak_mem_mb"] is None


class TestWebUIR7TemplatesAndFrontendSync:
    def test_index_html_sync_with_flags_and_valid_html(self, isolated_r7_webui) -> None:
        from html.parser import HTMLParser
        from acdso.webui.app import FLAGS_BY_NAME

        _, client, _, _ = isolated_r7_webui
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert html.count("<script>") == 1, "Must keep exactly one inline <script> bootstrap block"
        assert "initWorkbench();" in client.get("/static/workbench.js").get_data(as_text=True)

        # Verify /api/flags endpoint exposes choices and groups
        flags_resp = client.get("/api/flags")
        assert flags_resp.status_code == 200
        flag_names = {f["name"] for f in flags_resp.get_json()["flags"]}
        assert set(FLAGS_BY_NAME.keys()).issubset(flag_names)

        # Verify all 9 task options and all 18 new flag controls exist in index.html
        for t in FLAGS_BY_NAME["task"].choices:
            assert f'value="{t}"' in html, f"Missing task option {t!r}"
        for fname in (
            "group_col", "cv_strategy", "text_cols", "nlp_method",
            "cluster_algo", "n_clusters", "dim_reduce", "cluster_metric",
            "anomaly_algo", "contamination", "event_col", "time_col",
            "query_col", "user_col", "item_col", "rating_col",
            "chunk_size", "scale_mode",
        ):
            assert f'id="f_{fname}"' in html and f'name="{fname}"' in html, f"Missing control f_{fname}"

        class _Check(HTMLParser):
            VOID = {"meta", "link", "br", "hr", "input", "img"}

            def __init__(self) -> None:
                super().__init__()
                self.stack: list[str] = []
                self.errors: list[str] = []

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag not in self.VOID:
                    self.stack.append(tag)

            def handle_endtag(self, tag: str) -> None:
                if tag in self.VOID:
                    return
                if not self.stack or self.stack[-1] != tag:
                    self.errors.append(f"Mismatched </{tag}> vs stack={self.stack[-3:]}")
                else:
                    self.stack.pop()

        checker = _Check()
        checker.feed(html)
        assert not checker.errors, checker.errors
        assert checker.stack == [], f"Unclosed HTML tags: {checker.stack}"

    def test_workbench_and_charts_js_r7_node_vm_and_xss_safety(self, isolated_r7_webui, tmp_path: Path) -> None:
        node_exe = Path(r"C:\Program Files\nodejs\node.exe")
        if not node_exe.is_file():
            pytest.skip("Node.js not installed")

        _, client, _, _ = isolated_r7_webui
        html = client.get("/").get_data(as_text=True)
        m = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
        assert m is not None
        charts_js = client.get("/static/charts.js").get_data(as_text=True)
        wb_js = client.get("/static/workbench.js").get_data(as_text=True)

        combined = "\n".join([m.group(1), charts_js, wb_js.replace("initWorkbench();", "")])
        bundle_path = tmp_path / "r7_bundle.js"
        bundle_path.write_text(combined, encoding="utf-8")

        driver_path = tmp_path / "r7_driver.js"
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
    options: [], selectedOptions: [], style: {}, className: '',
    textContent: '', _innerHTML: '', dataset: {},
    classList: { _s: new Set(), add(c){this._s.add(c);}, remove(c){this._s.delete(c);}, toggle(c,f){if(f===undefined){this._s.has(c)?this._s.delete(c):this._s.add(c);}else if(f)this._s.add(c);else this._s.delete(c);}, contains(c){return this._s.has(c);} },
    setAttribute(){}, getAttribute(){}, addEventListener(){}, dispatchEvent(){},
    appendChild(){}, insertAdjacentHTML(){},
    querySelectorAll(){ return []; },
  };
  Object.defineProperty(el, 'innerHTML', { get(){return this._innerHTML;}, set(v){this._innerHTML = String(v);} });
  return el;
}
function getEl(id) { if (!elements.has(id)) elements.set(id, makeEl(id)); return elements.get(id); }

const sandbox = {
  assert, console, setTimeout, clearTimeout, setInterval: () => 1, clearInterval: () => {},
  confirm: () => true, Event: function(t){ this.type = t; },
  $: getEl,
  document: { getElementById: getEl, querySelectorAll: () => [], createTextNode: (t) => ({ textContent: t }) },
  fetch: async () => ({ status: 200, ok: true, text: async () => '{}' }),
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(jsCode + `
(() => {
  state.uploadId = 'abc123def456';
  state.columnsBlocked = false;

  // 1. Preflight permits targetless clustering & anomaly, enforces survival/recommender/ranking
  assert.strictEqual(validateRunPreflight({ upload_id: 'u1', task: 'clustering', predict: [], target: [] }), null);
  assert.strictEqual(validateRunPreflight({ upload_id: 'u1', task: 'anomaly', predict: [], target: [], contamination: '0.1' }), null);
  assert.notStrictEqual(validateRunPreflight({ upload_id: 'u1', task: 'anomaly', contamination: '0.9' }), null);
  assert.notStrictEqual(validateRunPreflight({ upload_id: 'u1', task: 'clustering', n_clusters: 1 }), null);
  assert.notStrictEqual(validateRunPreflight({ upload_id: 'u1', task: 'survival', predict: [], target: [] }), null);
  assert.strictEqual(validateRunPreflight({ upload_id: 'u1', task: 'survival', time_col: ['t'], event_col: ['e'], predict: [], target: [] }), null);
  assert.notStrictEqual(validateRunPreflight({ upload_id: 'u1', task: 'recommender', user_col: ['u'], predict: [], target: [] }), null);
  assert.strictEqual(validateRunPreflight({ upload_id: 'u1', task: 'recommender', user_col: ['u'], item_col: ['i'], predict: [], target: [] }), null);
  assert.notStrictEqual(validateRunPreflight({ upload_id: 'u1', task: 'ranking', predict: ['rel'], target: [] }), null);
  assert.strictEqual(validateRunPreflight({ upload_id: 'u1', task: 'ranking', predict: ['rel'], query_col: ['q'], target: [] }), null);
  assert.notStrictEqual(validateRunPreflight({ upload_id: 'u1', task: 'auto', predict: [], target: [] }), null);

  // 2. applyNlPlan sets task & targetless mode without throwing
  const fTask = $('f_task');
  fTask.dataset = { kind: 'choice' };
  $('form').querySelectorAll = () => [fTask];
  applyNlPlan({ task: 'clustering', predict: [], exclude: [], datetime_col: [] });
  assert.strictEqual(fTask.value, 'clustering');
  const cluPayload = collectPayload();
  assert.strictEqual(cluPayload.task, 'clustering');
  refreshCommand();
  assert($('cmd').textContent.includes('--task clustering'), 'refreshCommand missing --task clustering');

  // 3. renderInspector renders R1-R6 blocks & escapes XSS payloads
  state.inspectorData = {
    kind: 'combined',
    task: 'clustering',
    meta: { task: 'clustering' },
    metrics: { silhouette_score: 0.72 },
    selected_model: { name: 'kmeans<img src=x onerror=alert(1)>', metrics: { train_time_s: 0.4 } },
    clustering: { best_algorithm: 'kmeans', n_clusters: 3, cluster_sizes: { 'grp<script>': 40 } },
    anomaly: { best_algorithm: 'isolation_forest', n_anomalies: 2 },
    cv: { group_col: 'pat<script>id', cv_strategy: 'stratified_group_kfold', leakage_validated: true },
    nlp: { nlp_method: 'tfidf', text_cols: ['review<script>'] },
    specialized: { strategy: 'classifier_chain' },
    scale: { mode: 'chunked', chunk_size: 500 },
    audit_trail: { candidates: [{ algorithm: 'kmeans<script>', silhouette_score: 0.72 }] },
    predictions: { total_predictions: 40, statistics: { distribution: { 'c0<script>': 20, 'c1': 20 } }, sample_values: [{ row: 0, predicted: 'c0<script>' }] },
    dataset_info: { rows: 40, columns: 4 },
    summary: {},
    phases: [],
  };
  state.inspectorTab = 'overview';
  renderInspector();
  const ovHtml = $('inspectorBody').innerHTML;
  assert(ovHtml.includes('Clustering Summary') && ovHtml.includes('Out-of-Core Scale'), 'Missing R1-R6 blocks');
  assert(!ovHtml.includes('<script>') && !ovHtml.includes('<img src=x onerror=alert(1)>'), 'Unescaped XSS in overview');

  state.inspectorTab = 'models';
  renderInspector();
  const modHtml = $('inspectorBody').innerHTML;
  assert(modHtml.includes('0.7200') && !modHtml.includes('<script>'), 'Models fallback or XSS check failed');

  state.inspectorTab = 'predictions';
  renderInspector();
  const predHtml = $('inspectorBody').innerHTML;
  assert(predHtml.includes('<svg') && !predHtml.includes('<script>'), 'Predictions distribution SVG or XSS check failed');
  console.log('R7D_NODE_VM_OK');
})();
`, sandbox);
""",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [str(node_exe), str(driver_path), str(bundle_path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
        assert "R7D_NODE_VM_OK" in proc.stdout


class TestWebUIR7Iteration2Regressions:
    def test_numeric_contamination_and_task_normalization_in_build_argv_and_api_run(
        self, isolated_r7_webui, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pandas as pd
        import acdso.webui.app as webui_app

        stub = tmp_path / "stub_runner.py"
        stub.write_text("import sys; print('ARGS:', sys.argv[1:], flush=True)\n", encoding="utf-8")
        monkeypatch.setattr(webui_app, "RUNNER", stub)

        dummy_file = tmp_path / "data.csv"
        dummy_file.write_text("a,b\n1,2\n", encoding="utf-8")

        # Direct build_argv numeric float/int contamination & case-insensitive task
        argv_float = webui_app.build_argv({"task": "anomaly", "contamination": 0.1}, dummy_file)
        assert argv_float[argv_float.index("--contamination") + 1] == "0.1"
        argv_bound = webui_app.build_argv({"task": "anomaly", "contamination": 0.5}, dummy_file)
        assert argv_bound[argv_bound.index("--contamination") + 1] == "0.5"
        argv_upper_task = webui_app.build_argv({"task": "CLUSTERING"}, dummy_file)
        assert argv_upper_task[argv_upper_task.index("--task") + 1] == "clustering"

        for bad_contam in (True, False, 0, 0.0, -0.1, 0.51, float("nan"), float("inf"), float("-inf"), 10**400, 10**5000):
            with pytest.raises(ValueError):
                webui_app.build_argv({"task": "anomaly", "contamination": bad_contam}, dummy_file)

        # POST /api/run with numeric float contamination=0.1 and mixed-case task
        _, client, _, _ = isolated_r7_webui
        df = pd.DataFrame({"x1": [float(i) for i in range(20)], "x2": [float(i * 2) for i in range(20)]})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        uid = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(buf.getvalue()), "contam_num.csv")},
            content_type="multipart/form-data",
        ).get_json()["upload_id"]

        for good_payload in (
            {"upload_id": uid, "task": "anomaly", "contamination": 0.1},
            {"upload_id": uid, "task": "CLUSTERING"},
        ):
            r = client.post("/api/run", json=good_payload)
            assert r.status_code == 200, f"Expected 200 for {good_payload}, got {r.get_data(as_text=True)}"
            jid = r.get_json()["job_id"]
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                st = client.get(f"/api/job/{jid}").get_json()
                if st["status"] != "running":
                    break
                time.sleep(0.05)

        for bad_payload in (
            {"upload_id": uid, "task": "anomaly", "contamination": True},
            {"upload_id": uid, "task": "anomaly", "contamination": 0.0},
            {"upload_id": uid, "task": "anomaly", "contamination": 0.51},
            {"upload_id": uid, "task": "anomaly", "contamination": "nan"},
            {"upload_id": uid, "task": "anomaly", "contamination": "inf"},
            {"upload_id": uid, "task": "anomaly", "contamination": 10**400},
            {"upload_id": uid, "task": True},
            {"upload_id": uid, "task": "BOGUS_TASK"},
        ):
            r_bad = client.post("/api/run", json=bad_payload)
            assert r_bad.status_code == 400, f"Expected 400 for {bad_payload}, got {r_bad.status_code}"

    def test_nl_plan_exclusion_and_predict_segment_disambiguation(
        self, isolated_r7_webui, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pandas as pd

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _, client, _, _ = isolated_r7_webui

        df = pd.DataFrame({
            "patient_id": [f"P{i % 5}" for i in range(30)],
            "const_group": ["G1" for _ in range(30)],
            "x1": [float(i) for i in range(30)],
            "segment": ["retail" if i % 2 == 0 else "enterprise" for i in range(30)],
            "target": [i % 2 for i in range(30)],
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        uid = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(buf.getvalue()), "nl_polish.csv")},
            content_type="multipart/form-data",
        ).get_json()["upload_id"]

        # Excluded patient_id token must not infer group_col='const_group'
        r_excl = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "predict target excluding patient_id"})
        assert r_excl.status_code == 200
        j_excl = r_excl.get_json()
        assert "target" in j_excl["predict"]
        assert "patient_id" in j_excl["exclude"]
        assert "group_col" not in j_excl

        # Supervised "predict customer segment" with explicit target column must not force clustering
        r_seg_sup = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "predict customer segment"})
        assert r_seg_sup.status_code == 200
        j_seg_sup = r_seg_sup.get_json()
        assert j_seg_sup["task"] != "clustering"
        assert j_seg_sup["predict"] == ["segment"]

        # Explicit clustering prompt still resolves to clustering with empty predict
        r_clu = client.post(
            "/api/nl-plan",
            json={"upload_id": uid, "prompt": "cluster customers into segments without a target"},
        )
        assert r_clu.status_code == 200
        j_clu = r_clu.get_json()
        assert j_clu["task"] == "clustering"
        assert j_clu["predict"] == []

    def test_frontend_polish_boolean_guards_finite_locale_and_single_check_columns(
        self, isolated_r7_webui, tmp_path: Path
    ) -> None:
        node_exe = Path(r"C:\Program Files\nodejs\node.exe")
        if not node_exe.is_file():
            pytest.skip("Node.js not installed")

        _, client, _, _ = isolated_r7_webui
        html = client.get("/").get_data(as_text=True)
        m = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
        assert m is not None
        charts_js = client.get("/static/charts.js").get_data(as_text=True)
        wb_js = client.get("/static/workbench.js").get_data(as_text=True)

        combined = "\n".join([m.group(1), charts_js, wb_js.replace("initWorkbench();", "")])
        bundle_path = tmp_path / "r7_polish_bundle.js"
        bundle_path.write_text(combined, encoding="utf-8")

        driver_path = tmp_path / "r7_polish_driver.js"
        driver_path.write_text(
            r"""
const vm = require('vm');
const fs = require('fs');
const assert = require('assert');

const jsCode = fs.readFileSync(process.argv[2], 'utf8');
const elements = new Map();
function makeEl(id) {
  const listeners = {};
  const el = {
    id, name: id.replace(/^f_/, ''), value: '', checked: false, disabled: false, multiple: false,
    options: [], style: {}, className: '',
    textContent: '', _innerHTML: '', dataset: {},
    classList: { _s: new Set(), add(c){this._s.add(c);}, remove(c){this._s.delete(c);}, toggle(c,f){if(f===undefined){this._s.has(c)?this._s.delete(c):this._s.add(c);}else if(f)this._s.add(c);else this._s.delete(c);}, contains(c){return this._s.has(c);} },
    setAttribute(){}, getAttribute(){},
    addEventListener(t, fn){ (listeners[t] = listeners[t] || []).push(fn); },
    dispatchEvent(ev){ (listeners[ev.type] || []).forEach((fn) => fn(ev)); },
    appendChild(){}, insertAdjacentHTML(){},
    querySelectorAll(){ return []; },
  };
  Object.defineProperty(el, 'selectedOptions', { get(){ return this.options.filter((o) => o.selected); } });
  Object.defineProperty(el, 'innerHTML', { get(){return this._innerHTML;}, set(v){this._innerHTML = String(v);} });
  return el;
}
function getEl(id) { if (!elements.has(id)) elements.set(id, makeEl(id)); return elements.get(id); }

const sandbox = {
  assert, console, setTimeout, clearTimeout, setInterval: () => 1, clearInterval: () => {},
  confirm: () => true, Event: function(t){ this.type = t; },
  fetchCalls: [],
  $: getEl,
  document: { getElementById: getEl, querySelectorAll: () => [], createTextNode: (t) => ({ textContent: t }) },
};
sandbox.fetch = async (url) => {
  sandbox.fetchCalls.push(url);
  return { status: 200, ok: true, text: async () => '{"ok":true,"issues":[]}' };
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(jsCode + `
(() => {
  // 1. Boolean rejection in fmtNum and fmtDelta
  assert.strictEqual(ACDSOCharts.fmtNum(true, 4), 'n/a');
  assert.strictEqual(ACDSOCharts.fmtNum(false, 4), 'n/a');
  const cmpHtml = ACDSOCharts.renderRunComparisonHtml({
    reports: [
      { path: 'a.json', target: 'y', selected_model: 'A', metric: 0.8, train_time_s: 1.0, peak_mem_mb: 10, risk_penalty: 0.1, deltas: { metric: 0 } },
      { path: 'b.json', target: 'y', selected_model: 'B', metric: 0.9, train_time_s: 1.2, peak_mem_mb: 12, risk_penalty: 0.1, deltas: { metric: true, train_time_s: false, peak_mem_mb: '', risk_penalty: null } },
    ],
  });
  assert(!cmpHtml.includes('+1.0000'), 'Boolean delta formatted as +1.0000');
  assert(cmpHtml.includes('&mdash;'), 'Boolean/empty delta must render mdash');

  // 2. Non-numeric total_rows / total_predictions do not display NaN, and predictions tab uses d.task fallback
  state.inspectorData = { kind: 'csv', path: 'p.csv', columns: ['a'], rows: [{ a: 1 }], total_rows: 'not-a-num' };
  state.inspectorTab = 'predictions';
  renderInspector();
  assert(!$('inspectorBody').innerHTML.includes('NaN'), 'NaN leaked in CSV total_rows');
  assert($('inspectorBody').innerHTML.includes('n/a'), 'Expected n/a for non-numeric total_rows');

  state.inspectorData = {
    kind: 'combined',
    task: 'anomaly',
    predictions: { total_predictions: 'bad-count', statistics: { distribution: { '-1': 5, '1': 45 } }, sample_values: [{ row: 0, predicted: -1 }] },
  };
  renderInspector();
  const pHtml = $('inspectorBody').innerHTML;
  assert(!pHtml.includes('NaN'), 'NaN leaked in predictions total_predictions');
  assert(pHtml.includes('#ff7b7b'), 'd.task fallback failed to color anomaly distribution bar');

  // 3. applyNlPlan avoids redundant checkColumns() calls even with pre-existing role selection
  state.uploadId = 'u_single';
  fetchCalls.length = 0;
  const fPred = $('f_predict');
  fPred.multiple = true;
  fPred.options = [{ value: 'target', selected: false }];
  $('f_group_col').options = [{ value: 'existing_group', selected: false }, { value: 'patient_id', selected: false }];
  $('f_group_col').value = 'existing_group';
  applyNlPlan({ task: 'classification', predict: ['target'], group_col: ['patient_id'], exclude: [], datetime_col: [] });
  const validateCalls = fetchCalls.filter((u) => u === '/api/validate-columns').length;
  assert.strictEqual(validateCalls, 1, 'Expected single /api/validate-columns call, got ' + validateCalls);
  console.log('R7D_POLISH_NODE_OK');
})();
`, sandbox);
""",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [str(node_exe), str(driver_path), str(bundle_path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
        assert "R7D_POLISH_NODE_OK" in proc.stdout


class TestWebUIR7Iteration3Regressions:
    def test_nl_plan_ranking_query_and_grouped_patient_precedence_and_api_run(
        self, isolated_r7_webui, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pandas as pd
        import acdso.webui.app as webui_app

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        stub = tmp_path / "stub_runner_r3.py"
        stub.write_text("import sys; print('ARGS:', sys.argv[1:], flush=True)\n", encoding="utf-8")
        monkeypatch.setattr(webui_app, "RUNNER", stub)

        _, client, _, _ = isolated_r7_webui
        df = pd.DataFrame({
            "query_id": [f"Q{i % 4}" for i in range(24)],
            "patient_id": [f"P{i % 6}" for i in range(24)],
            "const_group": ["G1" for _ in range(24)],
            "outlier_flag": [i % 2 for i in range(24)],
            "segment": ["retail" if i % 2 == 0 else "enterprise" for i in range(24)],
            "review_text": ["excellent product quality and reliable performance" for _ in range(24)],
            "rating": [float((i % 5) + 1) for i in range(24)],
            "x1": [float(i) for i in range(24)],
            "target": [i % 2 for i in range(24)],
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        uid = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(buf.getvalue()), "nl_r3_precedence.csv")},
            content_type="multipart/form-data",
        ).get_json()["upload_id"]

        # 1. Ranking: query_id resolves to query_col BEFORE predict, rating becomes predict, /api/run returns 200
        r_rank = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "learning to rank rating by query_id"})
        assert r_rank.status_code == 200
        rank_plan = r_rank.get_json()
        assert rank_plan["task"] == "ranking"
        assert rank_plan.get("query_col") == ["query_id"]
        assert "rating" in rank_plan["predict"]
        assert "query_id" not in rank_plan["predict"]

        run_payload = {
            "upload_id": uid,
            "task": rank_plan["task"],
            "predict": rank_plan["predict"],
            "query_col": rank_plan["query_col"],
            "exclude": rank_plan.get("exclude", []),
        }
        r_run = client.post("/api/run", json=run_payload)
        assert r_run.status_code == 200, f"Expected 200 for ranking plan, got {r_run.get_data(as_text=True)}"
        jid = r_run.get_json()["job_id"]
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            st = client.get(f"/api/job/{jid}").get_json()
            if st["status"] != "running":
                break
            time.sleep(0.05)
        assert st["status"] == "finished" and st["returncode"] == 0

        # 2. Grouped CV: patient_id resolves to group_col (not const_group), target becomes predict
        r_grp = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "predict target grouped by patient_id"})
        assert r_grp.status_code == 200
        grp_plan = r_grp.get_json()
        assert grp_plan.get("group_col") == ["patient_id"]
        assert "target" in grp_plan["predict"]
        assert "patient_id" not in grp_plan["predict"]
        assert grp_plan.get("cv_strategy") == "group_kfold"

        # 3. Excluded token containing anomaly keyword ("excluding outlier_flag") must not misclassify as anomaly
        r_out = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "predict target excluding outlier_flag"})
        assert r_out.status_code == 200
        out_plan = r_out.get_json()
        assert out_plan["task"] == "auto"
        assert "target" in out_plan["predict"]
        assert "outlier_flag" in out_plan["exclude"]

        # 4. Preserved contracts (excluding patient_id, predict customer segment, clustering, tfidf+chunked)
        r_excl = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "predict target excluding patient_id"})
        assert r_excl.status_code == 200
        j_excl = r_excl.get_json()
        assert "target" in j_excl["predict"]
        assert "patient_id" in j_excl["exclude"]
        assert "group_col" not in j_excl

        r_seg = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "predict customer segment"})
        assert r_seg.status_code == 200
        j_seg = r_seg.get_json()
        assert j_seg["task"] != "clustering"
        assert j_seg["predict"] == ["segment"]

        r_clu = client.post(
            "/api/nl-plan",
            json={"upload_id": uid, "prompt": "cluster customers into segments without a target"},
        )
        assert r_clu.status_code == 200
        j_clu = r_clu.get_json()
        assert j_clu["task"] == "clustering"
        assert j_clu["predict"] == []

        r_nlp = client.post(
            "/api/nl-plan",
            json={
                "upload_id": uid,
                "prompt": "predict target excluding patient_id, const_group using tfidf and chunked scale",
            },
        )
        assert r_nlp.status_code == 200
        j_nlp = r_nlp.get_json()
        assert "target" in j_nlp["predict"]
        assert set(j_nlp["exclude"]) == {"patient_id", "const_group"}
        assert j_nlp.get("text_cols") == ["review_text"]
        assert j_nlp.get("nlp_method") == "tfidf"
        assert j_nlp.get("scale_mode") == "chunked"

    def test_whitespace_contamination_consistency_between_build_argv_and_api_run(
        self, isolated_r7_webui, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pandas as pd
        import acdso.webui.app as webui_app

        stub = tmp_path / "stub_runner_ws.py"
        stub.write_text("import sys; print('ARGS:', sys.argv[1:], flush=True)\n", encoding="utf-8")
        monkeypatch.setattr(webui_app, "RUNNER", stub)

        dummy_file = tmp_path / "data.csv"
        dummy_file.write_text("a,b\n1,2\n", encoding="utf-8")

        argv_ws = webui_app.build_argv({"task": "anomaly", "contamination": "   "}, dummy_file)
        assert "--contamination" not in argv_ws

        _, client, _, _ = isolated_r7_webui
        df = pd.DataFrame({"x1": [float(i) for i in range(20)], "x2": [float(i * 2) for i in range(20)]})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        uid = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(buf.getvalue()), "ws_contam.csv")},
            content_type="multipart/form-data",
        ).get_json()["upload_id"]

        r = client.post("/api/run", json={"upload_id": uid, "task": "anomaly", "contamination": "   "})
        assert r.status_code == 200, f"Expected 200 for whitespace contamination, got {r.get_data(as_text=True)}"
        jid = r.get_json()["job_id"]
        deadline = time.monotonic() + 5.0
        st = {}
        while time.monotonic() < deadline:
            st = client.get(f"/api/job/{jid}").get_json()
            if st["status"] != "running":
                break
            time.sleep(0.05)
        assert st["status"] == "finished" and st["returncode"] == 0
        assert "--contamination" not in " ".join(st.get("lines", []))


class TestWebUIR7Iteration4Regressions:
    def test_nl_plan_exclude_lookahead_and_ranking_query_vs_grouped_precedence(
        self, isolated_r7_webui, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pandas as pd
        import acdso.webui.app as webui_app

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        stub = tmp_path / "stub_runner_r4.py"
        stub.write_text("import sys; print('ARGS:', sys.argv[1:], flush=True)\n", encoding="utf-8")
        monkeypatch.setattr(webui_app, "RUNNER", stub)

        _, client, _, _ = isolated_r7_webui

        def _upload(df: pd.DataFrame, name: str) -> str:
            buf = io.BytesIO()
            df.to_csv(buf, index=False)
            resp = client.post(
                "/api/upload",
                data={"file": (io.BytesIO(buf.getvalue()), name)},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 200
            return resp.get_json()["upload_id"]

        # 1. excluding ... grouped by device_id preserves "grouped by"
        uid_dev = _upload(
            pd.DataFrame({
                "patient_id": [f"P{i % 4}" for i in range(20)],
                "device_id": [f"D{i % 5}" for i in range(20)],
                "x1": [float(i) for i in range(20)],
                "target": [i % 2 for i in range(20)],
            }),
            "r4_excl_dev.csv",
        )
        p1 = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_dev, "prompt": "predict target excluding patient_id grouped by device_id"},
        ).get_json()
        assert p1["exclude"] == ["patient_id"]
        assert p1.get("group_col") == ["device_id"]
        assert p1.get("cv_strategy") == "group_kfold"
        assert p1["predict"] == ["target"]

        # 2. excluding col_a grouped by entity_id
        uid_ent = _upload(
            pd.DataFrame({
                "entity_id": [f"E{i % 5}" for i in range(20)],
                "col_a": [float(i) for i in range(20)],
                "col_b": [float(i * 2) for i in range(20)],
                "score": [float(i % 7) for i in range(20)],
                "label": [i % 2 for i in range(20)],
            }),
            "r4_excl_ent.csv",
        )
        p2 = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_ent, "prompt": "predict score excluding col_a grouped by entity_id"},
        ).get_json()
        assert p2.get("cv_strategy") == "group_kfold"
        assert p2.get("group_col") == ["entity_id"]
        assert p2["predict"] == ["score"]
        assert p2["exclude"] == ["col_a"]

        # 3. excluding col_a grouped by grp_1 (generic group name without entity/patient token)
        uid_grp = _upload(
            pd.DataFrame({
                "grp_1": [f"G{i % 5}" for i in range(20)],
                "col_a": [float(i) for i in range(20)],
                "score": [float(i % 7) for i in range(20)],
                "f1": [float(i) * 0.5 for i in range(20)],
            }),
            "r4_excl_grp1.csv",
        )
        p3 = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_grp, "prompt": "predict score excluding col_a grouped by grp_1"},
        ).get_json()
        assert p3.get("cv_strategy") == "group_kfold"
        assert p3.get("group_col") == ["grp_1"]
        assert p3["predict"] == ["score"]
        assert p3["exclude"] == ["col_a"]

        # 4. excluding col_a by patient_id (bare "by" delimiter preserved)
        uid_pat = _upload(
            pd.DataFrame({
                "patient_id": [f"P{i % 5}" for i in range(20)],
                "col_a": [float(i) for i in range(20)],
                "score": [float(i % 7) for i in range(20)],
                "f1": [float(i) * 0.5 for i in range(20)],
            }),
            "r4_excl_pat.csv",
        )
        p4 = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_pat, "prompt": "predict score excluding col_a by patient_id"},
        ).get_json()
        assert p4.get("group_col") == ["patient_id"]
        assert p4["predict"] == ["score"]
        assert p4["exclude"] == ["col_a"]

        # 5. "excluding patient_id, predict target" does not swallow "predict target"
        p_comma = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_dev, "prompt": "excluding patient_id, predict target"},
        ).get_json()
        assert p_comma["exclude"] == ["patient_id"]
        assert p_comma["predict"] == ["target"]

        # 6. "train classifier group by patient_id" -> classification + group_col + group_kfold
        p_clf = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_dev, "prompt": "train classifier group by patient_id"},
        ).get_json()
        assert p_clf["task"] == "classification"
        assert p_clf.get("group_col") == ["patient_id"]
        assert p_clf.get("cv_strategy") == "group_kfold"

        # 7. Ranking with both query_id and grouped by session_id -> query_col == ["query_id"], relevance in predict, /api/run 200
        uid_dual = _upload(
            pd.DataFrame({
                "query_id": [f"Q{i % 4}" for i in range(24)],
                "session_id": [f"S{i % 6}" for i in range(24)],
                "relevance": [float(i % 4) for i in range(24)],
                "f1": [float(i) for i in range(24)],
                "f2": [float(i) * 0.5 for i in range(24)],
            }),
            "r4_dual_rank.csv",
        )
        p_dual = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_dual, "prompt": "learning to rank relevance by query_id grouped by session_id"},
        ).get_json()
        assert p_dual["task"] == "ranking"
        assert p_dual.get("query_col") == ["query_id"]
        assert p_dual["predict"] == ["relevance"]
        assert "query_id" not in p_dual["predict"]
        assert "session_id" not in p_dual["predict"]

        r_run = client.post(
            "/api/run",
            json={
                "upload_id": uid_dual,
                "task": p_dual["task"],
                "predict": p_dual["predict"],
                "query_col": p_dual["query_col"],
                "group_col": p_dual.get("group_col", []),
                "cv_strategy": p_dual.get("cv_strategy", "auto"),
            },
        )
        assert r_run.status_code == 200, r_run.get_data(as_text=True)

        # 8. Ranking fallback when no query_id column exists: "ranking relevance grouped by session_id"
        uid_sess = _upload(
            pd.DataFrame({
                "session_id": [f"S{i % 6}" for i in range(24)],
                "relevance": [float(i % 4) for i in range(24)],
                "f1": [float(i) for i in range(24)],
            }),
            "r4_sess_rank.csv",
        )
        p_sess = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_sess, "prompt": "ranking relevance grouped by session_id"},
        ).get_json()
        assert p_sess["task"] == "ranking"
        assert p_sess.get("query_col") == ["session_id"]
        assert p_sess["predict"] == ["relevance"]

        # 9. Ranking skip_dup_ranking_group when grouped by query_id
        uid_qdup = _upload(
            pd.DataFrame({
                "query_id": [f"Q{i % 4}" for i in range(24)],
                "patient_id": [f"P{i % 6}" for i in range(24)],
                "rating": [float((i % 5) + 1) for i in range(24)],
                "f1": [float(i) for i in range(24)],
            }),
            "r4_qdup_rank.csv",
        )
        p_qdup = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_qdup, "prompt": "learning to rank rating grouped by query_id"},
        ).get_json()
        assert p_qdup["task"] == "ranking"
        assert p_qdup.get("query_col") == ["query_id"]
        assert "group_col" not in p_qdup
        assert p_qdup["predict"] == ["rating"]

    def test_frontend_whitespace_only_hardening_in_node_vm(
        self, isolated_r7_webui, tmp_path: Path
    ) -> None:
        node_exe = Path(r"C:\Program Files\nodejs\node.exe")
        if not node_exe.is_file():
            pytest.skip("Node.js not installed")

        _, client, _, _ = isolated_r7_webui
        html = client.get("/").get_data(as_text=True)
        m = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
        assert m is not None
        charts_js = client.get("/static/charts.js").get_data(as_text=True)
        wb_js = client.get("/static/workbench.js").get_data(as_text=True)

        combined = "\n".join([m.group(1), charts_js, wb_js.replace("initWorkbench();", "")])
        bundle_path = tmp_path / "r7_iter4_bundle.js"
        bundle_path.write_text(combined, encoding="utf-8")

        driver_path = tmp_path / "r7_iter4_driver.js"
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
    options: [], style: {}, className: '',
    textContent: '', _innerHTML: '', dataset: {},
    classList: { _s: new Set(), add(c){this._s.add(c);}, remove(c){this._s.delete(c);}, toggle(c,f){if(f===undefined){this._s.has(c)?this._s.delete(c):this._s.add(c);}else if(f)this._s.add(c);else this._s.delete(c);}, contains(c){return this._s.has(c);} },
    setAttribute(){}, getAttribute(){}, addEventListener(){}, dispatchEvent(){},
    appendChild(){}, insertAdjacentHTML(){},
    querySelectorAll(){ return []; },
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
  assert.strictEqual(ACDSOCharts.fmtNum('   ', 2), 'n/a');
  const cmpHtml = ACDSOCharts.renderRunComparisonHtml({
    reports: [
      { path: 'a.json', target: 'y', selected_model: 'A', metric: 0.8, train_time_s: 1.0, peak_mem_mb: 10, risk_penalty: 0.1, deltas: { metric: 0 } },
      { path: 'b.json', target: 'y', selected_model: 'B', metric: '   ', train_time_s: 1.2, peak_mem_mb: 12, risk_penalty: 0.1, deltas: { metric: '   ', train_time_s: 0.2, peak_mem_mb: 2, risk_penalty: 0 } },
    ],
  });
  assert(cmpHtml.includes('n/a'), 'Whitespace metric must render n/a');
  assert(cmpHtml.includes('&mdash;'), 'Whitespace delta must render mdash');

  state.inspectorData = {
    kind: 'combined',
    task: 'classification',
    predictions: { total_predictions: '   ', statistics: {}, sample_values: [] },
  };
  state.inspectorTab = 'predictions';
  renderInspector();
  assert($('inspectorBody').innerHTML.includes('n/a'), 'Whitespace total_predictions must render n/a');

  state.uploadId = 'u_ws';
  const fTask = $('f_task');
  fTask.dataset = { kind: 'choice' };
  fTask.value = 'anomaly';
  const fContam = $('f_contamination');
  fContam.dataset = { kind: 'text' };
  fContam.value = '   ';
  $('form').querySelectorAll = () => [fTask, fContam];
  refreshCommand();
  assert(!$('cmd').textContent.includes('--contamination'), 'Whitespace contamination must be omitted from cmd preview: ' + $('cmd').textContent);
  console.log('R7_ITER4_WS_OK');
})();
`, sandbox);
""",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [str(node_exe), str(driver_path), str(bundle_path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
        assert "R7_ITER4_WS_OK" in proc.stdout


class TestWebUIR7Iteration5Regressions:
    def test_nl_plan_per_entity_id_group_cue_and_api_run_parity(
        self, isolated_r7_webui, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pandas as pd
        import acdso.webui.app as webui_app

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        stub = tmp_path / "stub_runner_r5.py"
        stub.write_text("import sys; print('ARGS:', sys.argv[1:], flush=True)\n", encoding="utf-8")
        monkeypatch.setattr(webui_app, "RUNNER", stub)

        _, client, _, _ = isolated_r7_webui

        def _upload(df: pd.DataFrame, name: str) -> str:
            buf = io.BytesIO()
            df.to_csv(buf, index=False)
            resp = client.post(
                "/api/upload",
                data={"file": (io.BytesIO(buf.getvalue()), name)},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 200
            return resp.get_json()["upload_id"]

        def _run_and_wait(payload: dict) -> dict:
            r_run = client.post("/api/run", json=payload)
            assert r_run.status_code == 200, r_run.get_data(as_text=True)
            jid = r_run.get_json()["job_id"]
            deadline = time.monotonic() + 5.0
            st = {}
            while time.monotonic() < deadline:
                st = client.get(f"/api/job/{jid}").get_json()
                if st["status"] != "running":
                    break
                time.sleep(0.05)
            assert st["status"] == "finished" and st["returncode"] == 0, st
            return st

        # 1. "predict target per patient_id" -> group_col == ["patient_id"], predict == ["target"]
        uid_pat = _upload(
            pd.DataFrame({
                "patient_id": [f"P{i % 5}" for i in range(20)],
                "target": [i % 2 for i in range(20)],
                "f1": [float(i) for i in range(20)],
            }),
            "r5_per_patient.csv",
        )
        p_pat = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_pat, "prompt": "predict target per patient_id"},
        ).get_json()
        assert p_pat.get("group_col") == ["patient_id"], p_pat
        assert p_pat["predict"] == ["target"], p_pat

        st_pat = _run_and_wait({
            "upload_id": uid_pat,
            "task": p_pat["task"],
            "predict": p_pat["predict"],
            "group_col": p_pat["group_col"],
        })
        out_pat = " ".join(st_pat.get("lines", []))
        assert "'--predict', 'target'" in out_pat and "'--group-col', 'patient_id'" in out_pat, out_pat

        # 2. "ignore col_a, col_b; predict score per device_id"
        uid_dev = _upload(
            pd.DataFrame({
                "col_a": [float(i) for i in range(20)],
                "col_b": [float(i * 2) for i in range(20)],
                "device_id": [f"D{i % 5}" for i in range(20)],
                "score": [float(i % 7) for i in range(20)],
                "f1": [float(i) * 0.5 for i in range(20)],
            }),
            "r5_per_device.csv",
        )
        p_dev = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_dev, "prompt": "ignore col_a, col_b; predict score per device_id"},
        ).get_json()
        assert p_dev["exclude"] == ["col_a", "col_b"], p_dev
        assert p_dev.get("group_col") == ["device_id"], p_dev
        assert p_dev["predict"] == ["score"], p_dev

        st_dev = _run_and_wait({
            "upload_id": uid_dev,
            "task": p_dev["task"],
            "predict": p_dev["predict"],
            "exclude": p_dev["exclude"],
            "group_col": p_dev["group_col"],
        })
        out_dev = " ".join(st_dev.get("lines", []))
        assert "'--predict', 'score'" in out_dev and "'--group-col', 'device_id'" in out_dev, out_dev
        assert "'--exclude'" in out_dev and "col_a" in out_dev and "col_b" in out_dev, out_dev

        # 3. Additional suffixed "per <entity>_id" columns
        for col in ("customer_id", "hospital_id", "user_id", "subject_id"):
            uid_e = _upload(
                pd.DataFrame({
                    col: [f"E{i % 5}" for i in range(20)],
                    "target": [i % 2 for i in range(20)],
                    "f1": [float(i) for i in range(20)],
                }),
                f"r5_per_{col}.csv",
            )
            p_e = client.post(
                "/api/nl-plan",
                json={"upload_id": uid_e, "prompt": f"predict target per {col}"},
            ).get_json()
            assert p_e.get("group_col") == [col], (col, p_e)
            assert p_e["predict"] == ["target"], (col, p_e)

        # 4. Negative controls: excluding patient_id & predict customer segment must NOT trigger spurious group_col
        p_excl = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_pat, "prompt": "predict target excluding patient_id"},
        ).get_json()
        assert p_excl["exclude"] == ["patient_id"], p_excl
        assert p_excl["predict"] == ["target"], p_excl
        assert p_excl.get("group_col", []) == [], p_excl

        uid_seg = _upload(
            pd.DataFrame({
                "customer_id": [f"C{i}" for i in range(20)],
                "segment": [f"S{i % 3}" for i in range(20)],
                "f1": [float(i) for i in range(20)],
            }),
            "r5_cust_segment.csv",
        )
        p_seg = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_seg, "prompt": "predict customer segment"},
        ).get_json()
        assert p_seg["predict"] == ["segment"], p_seg
        assert p_seg.get("group_col", []) == [], p_seg


class TestWebUIR7Iteration6Regressions:
    def test_nl_plan_excluding_before_per_entity_and_hospital_group_kw(
        self, isolated_r7_webui, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pandas as pd
        import acdso.webui.app as webui_app

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        stub = tmp_path / "stub_runner_r6.py"
        stub.write_text(
            "import json, sys; print('ARGS_JSON:' + json.dumps(sys.argv[1:]), flush=True)\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(webui_app, "RUNNER", stub)

        _, client, _, _ = isolated_r7_webui

        def _upload(df: pd.DataFrame, name: str) -> str:
            buf = io.BytesIO()
            df.to_csv(buf, index=False)
            resp = client.post(
                "/api/upload",
                data={"file": (io.BytesIO(buf.getvalue()), name)},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 200
            return resp.get_json()["upload_id"]

        def _run_and_wait(payload: dict) -> list[str]:
            r_run = client.post("/api/run", json=payload)
            assert r_run.status_code == 200, r_run.get_data(as_text=True)
            jid = r_run.get_json()["job_id"]
            deadline = time.monotonic() + 5.0
            st = {}
            while time.monotonic() < deadline:
                st = client.get(f"/api/job/{jid}").get_json()
                if st["status"] != "running":
                    break
                time.sleep(0.05)
            assert st["status"] == "finished" and st["returncode"] == 0, st
            for line in st.get("lines", []):
                if line.startswith("ARGS_JSON:"):
                    return json.loads(line[len("ARGS_JSON:"):])
            raise AssertionError(f"Missing ARGS_JSON in job output: {st}")

        # 1. "predict score excluding col_a per device_id" -> /api/nl-plan -> /api/run
        uid_dev = _upload(
            pd.DataFrame({
                "col_a": [float(i) for i in range(20)],
                "device_id": [f"D{i % 5}" for i in range(20)],
                "score": [float(i % 7) for i in range(20)],
                "f1": [float(i) * 0.5 for i in range(20)],
            }),
            "r6_excl_per_device.csv",
        )
        p_dev = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_dev, "prompt": "predict score excluding col_a per device_id"},
        ).get_json()
        assert p_dev["exclude"] == ["col_a"], p_dev
        assert p_dev.get("group_col") == ["device_id"], p_dev
        assert p_dev["predict"] == ["score"], p_dev

        argv_dev = _run_and_wait({
            "upload_id": uid_dev,
            "task": p_dev["task"],
            "predict": p_dev["predict"],
            "exclude": p_dev["exclude"],
            "group_col": p_dev["group_col"],
        })
        assert argv_dev[argv_dev.index("--predict") + 1] == "score", argv_dev
        assert argv_dev[argv_dev.index("--group-col") + 1] == "device_id", argv_dev
        assert argv_dev[argv_dev.index("--exclude") + 1] == "col_a", argv_dev

        # 2. "predict target excluding col_a per patient_id"
        uid_pat = _upload(
            pd.DataFrame({
                "col_a": [float(i) for i in range(20)],
                "patient_id": [f"P{i % 5}" for i in range(20)],
                "target": [i % 2 for i in range(20)],
                "f1": [float(i) for i in range(20)],
            }),
            "r6_excl_per_patient.csv",
        )
        p_pat = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_pat, "prompt": "predict target excluding col_a per patient_id"},
        ).get_json()
        assert p_pat["exclude"] == ["col_a"], p_pat
        assert p_pat.get("group_col") == ["patient_id"], p_pat
        assert p_pat["predict"] == ["target"], p_pat

        # 3. "predict target by hospital_id" when user_id co-exists in schema
        uid_hosp = _upload(
            pd.DataFrame({
                "user_id": [f"U{i % 4}" for i in range(20)],
                "hospital_id": [f"H{i % 5}" for i in range(20)],
                "target": [i % 2 for i in range(20)],
                "f1": [float(i) for i in range(20)],
            }),
            "r6_by_hospital.csv",
        )
        p_hosp = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_hosp, "prompt": "predict target by hospital_id"},
        ).get_json()
        assert p_hosp.get("group_col") == ["hospital_id"], p_hosp
        assert p_hosp["predict"] == ["target"], p_hosp

        # 4. Negative controls: excluding patient_id, per row, and excluding col_a per item
        p_excl = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_pat, "prompt": "predict target excluding patient_id"},
        ).get_json()
        assert p_excl["exclude"] == ["patient_id"], p_excl
        assert p_excl["predict"] == ["target"], p_excl
        assert p_excl.get("group_col", []) == [], p_excl

        p_row = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_pat, "prompt": "predict target per row"},
        ).get_json()
        assert p_row["predict"] == ["target"], p_row
        assert p_row.get("group_col", []) == [], p_row

        p_item = client.post(
            "/api/nl-plan",
            json={"upload_id": uid_pat, "prompt": "predict target excluding col_a per item"},
        ).get_json()
        assert p_item["exclude"] == ["col_a"], p_item
        assert p_item["predict"] == ["target"], p_item
        assert p_item.get("group_col", []) == [], p_item


