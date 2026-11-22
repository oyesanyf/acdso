"""Tests for the ACDSO Web GUI backend and frontend workbench."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_SAMPLES = REPO_ROOT / "uploads"
REPORTS_SAMPLES = REPO_ROOT / "reports"
NODE_EXE = Path(r"C:\Program Files\nodejs\node.exe")


@pytest.fixture(autouse=True)
def isolated_webui_dirs(tmp_path, monkeypatch):
    """Isolate UPLOAD_DIR, REPORTS_DIR, and LOG_DIR so tests never pollute the repo."""
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
    yield {"uploads": up_dir, "reports": rep_dir, "logs": log_dir}
    webui_app._invalidate_reports_cache()


@pytest.fixture
def client():
    from acdso.webui import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _upload_sample(client, filename: str) -> dict:
    src = UPLOADS_SAMPLES / filename
    data = {"file": (io.BytesIO(src.read_bytes()), filename)}
    resp = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json()


def _upload_bytes(client, filename: str, content: bytes) -> dict:
    data = {"file": (io.BytesIO(content), filename)}
    resp = client.post("/api/upload", data=data, content_type="multipart/form-data")
    return {"status": resp.status_code, "json": resp.get_json()}


class TestWebUIStartupAndAssets:
    def test_start_py_check_dependencies_avoids_importing_pandas(self):
        code = (
            "import sys; sys.path.insert(0, r'd:\\harfile\\acdso'); "
            "import start; start._check_dependencies(); "
            "assert 'pandas' not in sys.modules, 'pandas imported eagerly'; "
            "assert 'numpy' not in sys.modules, 'numpy imported eagerly'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout

    def test_webui_submodules_defer_heavy_ml_imports_and_create_app_fast(self):
        code = "\n".join([
            "import sys, time",
            "t0 = time.perf_counter()",
            "import acdso.webui",
            "import acdso.webui.column_validator",
            "import acdso.webui.forecast",
            "import acdso.webui.nl_plan",
            "import acdso.webui.app as wa",
            "t1 = time.perf_counter()",
            "for m in ('pandas', 'numpy', 'sklearn', 'scipy'):",
            "    assert m not in sys.modules, f'{m} eagerly imported'",
            "app = wa.create_app()",
            "t2 = time.perf_counter()",
            "for m in ('pandas', 'numpy', 'sklearn', 'scipy'):",
            "    assert m not in sys.modules, f'{m} imported during create_app()'",
            "total_ms = (t2 - t0) * 1000",
            "print(f'total_ms={total_ms:.2f} create_app_ms={(t2-t1)*1000:.2f}')",
            "assert (t2 - t1) < 0.5, f'create_app took {(t2-t1)*1000:.1f}ms'",
        ])
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout

    def test_wsgi_lazy_app_attribute(self):
        import acdso.webui.app as webui_app

        assert hasattr(webui_app.app, "test_client")
        assert webui_app.app is webui_app.app

    def test_index_renders_guided_workbench_without_contrast_or_encoding_issues(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        for token in (
            "recentUploads",
            "targetPickerList",
            "goalPredictBtn",
            "goalTrainBtn",
            "runMode",
            "advancedFlags",
            "pipelineStepper",
            "reportInspector",
            ".msg.warn",
            "safeFetchJson",
            "syncDecisionOptsVisibility",
            'data-match="data loading"',
            "if (data.command) $('cmd').textContent = data.command;",
            "state.columnsBlocked = false;",
        ):
            assert token in html, f"Missing workbench token {token!r}"
        assert "#c33" not in html.lower()
        assert "\ufffd" not in html

    def test_index_script_passes_node_syntax_check(self, client, tmp_path):
        if not NODE_EXE.is_file():
            pytest.skip("Node.js not installed at expected path")
        html = client.get("/").get_data(as_text=True)
        match = re.search(r"<script>(.*)</script>", html, re.DOTALL)
        assert match is not None, "Could not find <script> block in index.html"
        js_file = tmp_path / "workbench.js"
        js_file.write_text(match.group(1), encoding="utf-8")
        proc = subprocess.run(
            [str(NODE_EXE), "--check", str(js_file)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout


class TestWebUIUploadsAndPersistence:
    def test_upload_real_tips_csv_and_persist_across_app_restarts(self, client):
        uploaded = _upload_sample(client, "tips_0cccb3b1499e.csv")
        assert uploaded["rows"] == 244
        assert len(uploaded["columns"]) == 7
        assert len(uploaded["preview"]) == 10
        assert uploaded["size_bytes"] > 0
        assert "column_stats" in uploaded
        stats_by_name = {s["name"]: s for s in uploaded["column_stats"]}
        assert stats_by_name["total_bill"]["role_hint"] == "numeric"
        assert stats_by_name["total_bill"]["predictable"] is True

        # New app instance simulates server restart reading .uploads_index.json
        from acdso.webui import create_app

        restarted = create_app().test_client()
        listed = restarted.get("/api/uploads").get_json()["uploads"]
        assert any(u["upload_id"] == uploaded["upload_id"] for u in listed)

        detail = restarted.get(f"/api/uploads/{uploaded['upload_id']}")
        assert detail.status_code == 200
        djson = detail.get_json()
        assert djson["rows"] == 244
        assert len(djson["preview"]) == 10

    def test_upload_rejects_missing_or_bad_extension(self, client):
        r_empty = client.post("/api/upload", data={}, content_type="multipart/form-data")
        assert r_empty.status_code == 400
        r_bad = _upload_bytes(client, "malware.exe", b"MZ")
        assert r_bad["status"] == 400
        assert "Unsupported file type" in r_bad["json"]["error"]

    def test_upload_explains_every_excluded_column(self, client):
        csv_text = (
            "record_id,dt,const_col,empty_col\n"
            "1,2026-01-01,same,\n"
            "2,2026-01-02,same,\n"
            "3,2026-01-03,same,\n"
        )
        res = _upload_bytes(client, "all_excluded.csv", csv_text.encode("utf-8"))
        assert res["status"] == 200
        j = res["json"]
        assert j["predictable_columns"] == []
        excl = j["excluded_columns"]
        assert excl["record_id"]["reason"] in ("id_like", "unique_non_float")
        assert excl["dt"]["reason"] == "datetime"
        assert excl["const_col"]["reason"] == "constant"
        assert excl["empty_col"]["reason"] == "all_null"

    def test_lazy_discovery_of_existing_upload_files_without_index(self, isolated_webui_dirs):
        from acdso.webui import create_app

        src = UPLOADS_SAMPLES / "tips_0cccb3b1499e.csv"
        dest = isolated_webui_dirs["uploads"] / "tips_abc123def456.csv"
        dest.write_bytes(src.read_bytes())

        c = create_app().test_client()
        listed = c.get("/api/uploads").get_json()["uploads"]
        entry = next(u for u in listed if u["upload_id"] == "abc123def456")
        assert entry["needs_inspect"] is True
        assert entry["columns"] == []

        detail = c.get("/api/uploads/abc123def456")
        assert detail.status_code == 200
        dj = detail.get_json()
        assert dj["needs_inspect"] is False
        assert dj["rows"] == 244
        assert len(dj["columns"]) == 7


class TestWebUISuggestionsAndValidation:
    def test_suggest_target_real_datasets_fast_and_zero_warnings(self, client):
        for fn in (
            "tips_0cccb3b1499e.csv",
            "scratch_domains_92b347bb484a.csv",
            "yearJFrog-data-transfer-20250821-20260814_146d8b6117aa.csv",
        ):
            up = _upload_sample(client, fn)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                t0 = time.perf_counter()
                resp = client.post("/api/suggest-target", json={"upload_id": up["upload_id"]})
                cold_ms = (time.perf_counter() - t0) * 1000
                t1 = time.perf_counter()
                resp_warm = client.post("/api/suggest-target", json={"upload_id": up["upload_id"]})
                warm_ms = (time.perf_counter() - t1) * 1000

            assert resp.status_code == 200
            assert resp_warm.status_code == 200
            assert not caught, f"Unexpected warnings on {fn}: {[str(w.message) for w in caught]}"
            assert cold_ms < 500.0, f"Cold suggest-target too slow on {fn}: {cold_ms:.1f}ms"
            assert warm_ms < 25.0, f"Cached suggest-target too slow on {fn}: {warm_ms:.1f}ms"
            data = resp.get_json()
            assert data["cached"] is False
            assert resp_warm.get_json()["cached"] is True
            assert len(data["suggestions"]) >= 1
            for s in data["suggestions"]:
                assert s["column"] in up["predictable_columns"]

    def test_suggest_target_wide_100_col_csv_sub_500ms(self, client):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(7)
        cols = {f"f_{i}": rng.normal(size=1200) for i in range(99)}
        cols["target_revenue"] = 2.0 * cols["f_0"] - 1.5 * cols["f_1"] + rng.normal(scale=0.1, size=1200)
        buf = io.BytesIO()
        pd.DataFrame(cols).to_csv(buf, index=False)
        up = _upload_bytes(client, "wide100.csv", buf.getvalue())["json"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            t0 = time.perf_counter()
            resp = client.post("/api/suggest-target", json={"upload_id": up["upload_id"]})
            ms = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200
        assert not caught
        assert ms < 500.0, f"100-col suggest-target took {ms:.1f}ms"

    def test_suggest_horizon_real_jfrog_and_bounded_reads(self, client):
        up = _upload_sample(client, "yearJFrog-data-transfer-20250821-20260814_146d8b6117aa.csv")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            resp = client.post(
                "/api/suggest-horizon",
                json={
                    "upload_id": up["upload_id"],
                    "datetime_col": "Date",
                    "target_col": "Total Data Transfer (B)",
                },
            )
        assert resp.status_code == 200
        assert not caught, f"Unexpected warnings: {[str(w.message) for w in caught]}"
        data = resp.get_json()
        assert data["grain"] == "M"
        assert isinstance(data["horizon"], int) and data["horizon"] >= 1

        # Cached hit
        resp2 = client.post(
            "/api/suggest-horizon",
            json={
                "upload_id": up["upload_id"],
                "datetime_col": "Date",
                "target_col": "Total Data Transfer (B)",
            },
        )
        assert resp2.get_json()["cached"] is True

        # Unknown column returns 400
        bad = client.post(
            "/api/suggest-horizon",
            json={"upload_id": up["upload_id"], "datetime_col": "NoSuchCol", "target_col": "Total Data Transfer (B)"},
        )
        assert bad.status_code == 400

    def test_validate_columns_blocks_conflicts_and_allows_valid(self, client):
        up = _upload_sample(client, "tips_0cccb3b1499e.csv")
        uid = up["upload_id"]

        # Same-column dual role
        r_dual = client.post("/api/validate-columns", json={"upload_id": uid, "predict": ["tip"], "exclude": ["tip"]})
        assert r_dual.status_code == 200
        assert r_dual.get_json()["ok"] is False

        # Predict vs target mismatch
        r_mismatch = client.post(
            "/api/validate-columns",
            json={"upload_id": uid, "predict": ["tip"], "target": ["total_bill"]},
        )
        assert r_mismatch.status_code == 200
        j_mm = r_mismatch.get_json()
        assert j_mm["ok"] is False
        assert any(i["severity"] == "block" for i in j_mm["issues"])

        # Exclude leaves zero features
        all_other = [c for c in up["columns"] if c != "tip"]
        r_nofeat = client.post(
            "/api/validate-columns",
            json={"upload_id": uid, "predict": ["tip"], "exclude": all_other},
        )
        assert r_nofeat.get_json()["ok"] is False

        # Valid selection
        r_ok = client.post(
            "/api/validate-columns",
            json={"upload_id": uid, "predict": ["tip"], "exclude": ["smoker"]},
        )
        assert r_ok.get_json()["ok"] is True

    def test_nl_plan_real_scratch_domains_and_numeric_time_guard(self, client):
        up = _upload_sample(client, "scratch_domains_92b347bb484a.csv")
        resp = client.post(
            "/api/nl-plan",
            json={"upload_id": up["upload_id"], "prompt": "predict ai_provider_label from domain features"},
        )
        assert resp.status_code == 200
        plan = resp.get_json()
        assert plan["predict"] == ["ai_provider_label"]

        # Numeric column with "time" substring must not become datetime_col
        lines = ["Date,lifetime_value,response_time_ms,label"]
        for i in range(30):
            lines.append(f"2026-01-{(i % 28) + 1:02d},{100 + i},{10 + (i % 5)},{i % 2}")
        up2 = _upload_bytes(client, "nl_guard.csv", "\n".join(lines).encode("utf-8"))["json"]
        resp2 = client.post("/api/nl-plan", json={"upload_id": up2["upload_id"], "prompt": "predict label"})
        assert resp2.status_code == 200
        dt = resp2.get_json()["datetime_col"]
        assert dt == ["Date"], f"Unexpected datetime_col: {dt}"


class TestWebUIForecast:
    def test_forecast_real_jfrog_csv_and_cleanup(self, client):
        up = _upload_sample(client, "yearJFrog-data-transfer-20250821-20260814_146d8b6117aa.csv")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            resp = client.post(
                "/api/forecast",
                json={
                    "upload_id": up["upload_id"],
                    "date_col": "Date",
                    "value_col": "Total Data Transfer (B)",
                    "agg": "sum",
                    "horizon": 3,
                },
            )
        assert resp.status_code == 200
        assert not caught
        data = resp.get_json()
        assert len(data["forecast"]) == 3
        assert len(data["history"]) >= 12
        assert data["chosen_method"]
        csv_path = data["csv_path"]

        art = client.get(f"/api/artifact?path={csv_path}")
        assert art.status_code == 200
        art.close()

        del_resp = client.delete(f"/api/reports?path={csv_path}")
        assert del_resp.status_code == 200

    def test_forecast_count_on_non_numeric_and_invalid_bounds_400(self, client):
        up = _upload_sample(client, "yearJFrog-data-transfer-20250821-20260814_146d8b6117aa.csv")
        uid = up["upload_id"]

        ok_count = client.post(
            "/api/forecast",
            json={"upload_id": uid, "date_col": "Date", "value_col": "Service", "agg": "count", "horizon": 2},
        )
        assert ok_count.status_code == 200

        for bad_payload in (
            {"upload_id": uid, "date_col": "Date", "value_col": "Total Data Transfer (B)", "horizon": 0},
            {"upload_id": uid, "date_col": "Date", "value_col": "Total Data Transfer (B)", "horizon": -3},
            {"upload_id": uid, "date_col": "Date", "value_col": "Total Data Transfer (B)", "horizon": 99},
            {"upload_id": uid, "date_col": "Date", "value_col": "Total Data Transfer (B)", "horizon": True},
            {"upload_id": uid, "date_col": "Date", "value_col": "Total Data Transfer (B)", "grain": "INVALID"},
            {"upload_id": uid, "date_col": "Date", "value_col": "Total Data Transfer (B)", "agg": "median"},
            {"upload_id": uid, "date_col": "Total Data Transfer (B)", "value_col": "Number Of Files Transferred", "horizon": 2},
            {"upload_id": uid, "date_col": "Date", "value_col": "NoSuchCol", "horizon": 2},
        ):
            bad = client.post("/api/forecast", json=bad_payload)
            assert bad.status_code == 400, f"Expected 400 for {bad_payload}, got {bad.status_code}"


class TestWebUIRunJobsProgressAndStop:
    def test_run_rejects_invalid_inputs_400(self, client):
        r_no_file = client.post("/api/run", json={})
        assert r_no_file.status_code == 400

        up = _upload_sample(client, "tips_0cccb3b1499e.csv")
        uid = up["upload_id"]

        for bad_payload in (
            {"upload_id": uid},
            {"upload_id": uid, "predict": ["tip"], "target": ["total_bill"]},
            {"upload_id": uid, "predict": ["tip"], "horizon": 0},
            {"upload_id": uid, "predict": ["tip"], "horizon": -2},
            {"upload_id": uid, "predict": ["tip"], "budget": -1.5},
            {"upload_id": uid, "predict": ["tip"], "time_budget": 0},
            {"upload_id": uid, "predict": ["no_such_column"]},
        ):
            resp = client.post("/api/run", json=bad_payload)
            assert resp.status_code == 400, f"Expected 400 for {bad_payload}, got {resp.status_code}"

    def test_job_progress_streaming_conflict_and_stop(self, client, tmp_path, monkeypatch):
        import acdso.webui.app as webui_app

        dummy_runner = tmp_path / "dummy_runner.py"
        dummy_runner.write_text(
            "import sys, time\n"
            "print('[ACDSO_PROGRESS] {\"phase\": 2, \"total\": 6, \"name\": \"Auto EDA\", \"status\": \"running\", \"detail\": \"profiling\"}', flush=True)\n"
            "print('[ACDSO_PROGRESS] invalid json line', flush=True)\n"
            "if '--horizon' in sys.argv:\n"
            "    time.sleep(10)\n"
            "print('[ACDSO_PROGRESS] {\"phase\": 6, \"total\": 6, \"name\": \"Report\", \"status\": \"done\", \"detail\": \"complete\"}', flush=True)\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(webui_app, "RUNNER", dummy_runner)

        up = _upload_sample(client, "tips_0cccb3b1499e.csv")
        uid = up["upload_id"]

        # Start slow run (horizon=5 triggers sleep) so we can verify progress, 409, and stop
        r_start = client.post("/api/run", json={"upload_id": uid, "predict": ["tip"], "horizon": 5})
        assert r_start.status_code == 200, r_start.get_data(as_text=True)
        job_id = r_start.get_json()["job_id"]

        # Wait briefly for progress line to stream
        prog = None
        for _ in range(30):
            js = client.get(f"/api/job/{job_id}").get_json()
            if js.get("progress"):
                prog = js["progress"]
                break
            time.sleep(0.05)

        assert prog is not None
        assert prog["phase"] == 2
        assert prog["total"] == 6
        assert prog["percent"] == 33
        assert prog["name"] == "Auto EDA"
        assert any("[phase 2/6] Auto EDA (running) - profiling" in ln for ln in js["lines"])
        assert all('{"phase":' not in ln for ln in js["lines"])
        assert any("invalid json line" in ln for ln in js["lines"])

        # /api/health exposes active_job
        health = client.get("/api/health").get_json()
        assert health["active_job"] == job_id

        # Concurrent run returns 409
        r_conflict = client.post("/api/run", json={"upload_id": uid, "predict": ["tip"]})
        assert r_conflict.status_code == 409

        # Stop via /api/stop/<id> alias
        r_stop = client.post(f"/api/stop/{job_id}")
        assert r_stop.status_code == 200
        assert r_stop.get_json()["status"] == "cancelled"

        # Wait until pump finishes
        for _ in range(30):
            if client.get(f"/api/job/{job_id}").get_json()["status"] != "running":
                break
            time.sleep(0.05)

        # Verify jobs list includes progress
        jobs = client.get("/api/jobs").get_json()["jobs"]
        assert any(j["job_id"] == job_id and j["progress"] is not None for j in jobs)


class TestWebUIReportSummaryAndSecurity:
    def test_reports_listing_skips_webui_logs_and_invalidates_on_delete(self, client, isolated_webui_dirs):
        rep = isolated_webui_dirs["reports"]
        logs = isolated_webui_dirs["logs"]
        (logs / "run_secret.log").write_text("log", encoding="utf-8")
        (rep / "sample_123456.json").write_text('{"summary": {}}', encoding="utf-8")

        listed = client.get("/api/reports").get_json()["reports"]
        paths = [r["path"] for r in listed]
        assert "sample_123456.json" in paths
        assert all("webui_logs" not in p and not p.endswith(".log") for p in paths)

        del_r = client.delete("/api/reports?path=sample_123456.json")
        assert del_r.status_code == 200
        listed2 = client.get("/api/reports").get_json()["reports"]
        assert all(r["path"] != "sample_123456.json" for r in listed2)

    def test_report_summary_combined_audit_and_csv(self, client, isolated_webui_dirs):
        rep = isolated_webui_dirs["reports"]
        real_combined = REPORTS_SAMPLES / "autonomous_AI_list_ea6b78f6968f_Application_combined.json"
        if real_combined.is_file():
            (rep / real_combined.name).write_bytes(real_combined.read_bytes())
            resp = client.get(f"/api/report-summary?path={real_combined.name}")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["kind"] == "combined"
            assert "eda_html" not in data
            assert "console_output" not in data
            assert len(data["phases"]) == 9
            assert data["eda_report"]["quality_score"] == pytest.approx(52.7)

        audit_payload = {
            "risk_report": {"task": "classification", "vif_max": 2.4, "notes": ["Low risk"]},
            "candidates_evaluated": [{"model_name": "LightGBM", "metric": 0.91, "train_time_s": 1.2}],
            "pareto_front": [{"model_name": "LightGBM"}],
            "selected": {"model_name": "LightGBM", "metric": 0.91},
        }
        (rep / "audit_demo.json").write_text(json.dumps(audit_payload), encoding="utf-8")
        r_audit = client.get("/api/report-summary?path=audit_demo.json")
        assert r_audit.status_code == 200
        aj = r_audit.get_json()
        assert aj["kind"] == "audit"
        assert aj["selected_model"]["model_name"] == "LightGBM"
        assert len(aj["pareto_front"]) == 1

        csv_rows = ["a,b,target_predicted"] + [f"{i},{i*2},{i%2}" for i in range(120)]
        (rep / "predictions_demo_target_full_dataset.csv").write_text("\n".join(csv_rows), encoding="utf-8")
        r_csv = client.get("/api/report-summary?path=predictions_demo_target_full_dataset.csv&limit=50")
        assert r_csv.status_code == 200
        cj = r_csv.get_json()
        assert cj["kind"] == "csv"
        assert cj["total_rows"] == 120
        assert len(cj["rows"]) == 50
        assert "target_predicted" in cj["prediction_stats"]

    def test_report_summary_blocks_traversal_and_webui_logs(self, client, isolated_webui_dirs):
        (isolated_webui_dirs["logs"] / "run_1.log").write_text("x", encoding="utf-8")
        assert client.get("/api/report-summary?path=../start.py").status_code == 403
        assert client.get("/api/report-summary?path=../../etc/passwd").status_code == 403
        assert client.get("/api/report-summary?path=webui_logs/run_1.log").status_code == 403
        assert client.get("/api/report-summary?path=does_not_exist.json").status_code == 404
        assert client.get("/api/report-summary").status_code == 400


class TestWebUIHardeningRegressions:
    def test_parse_progress_line_and_pump_survive_inf_nan_bool(self, client, tmp_path, monkeypatch):
        import acdso.webui.app as webui_app

        # Direct unit checks on _parse_progress_line
        for bad_line in (
            '[ACDSO_PROGRESS] {"phase": 1e309, "total": 6, "name": "InfPhase"}',
            '[ACDSO_PROGRESS] {"phase": -1e309, "total": 6, "name": "NegInf"}',
            '[ACDSO_PROGRESS] {"phase": Infinity, "total": 6, "name": "Infinity"}',
            '[ACDSO_PROGRESS] {"phase": NaN, "total": 6, "name": "NaNPhase"}',
            '[ACDSO_PROGRESS] {"phase": true, "total": 6, "name": "BoolPhase"}',
            '[ACDSO_PROGRESS] {"phase": 2, "total": false, "name": "BoolTotal"}',
            '[ACDSO_PROGRESS] {"phase": "not-a-num", "total": 6, "name": "StrPhase"}',
            '[ACDSO_PROGRESS] [1, 2, 3]',
        ):
            assert webui_app._parse_progress_line(bad_line) is None, f"Should reject {bad_line}"

        # None phase/total handled cleanly without crash
        parsed_none = webui_app._parse_progress_line(
            '[ACDSO_PROGRESS] {"phase": null, "total": null, "name": "Init", "status": "running"}'
        )
        assert parsed_none == {
            "phase": 0,
            "total": 0,
            "name": "Init",
            "status": "running",
            "percent": 0,
            "detail": "",
        }

        # End-to-end JobManager._pump survives malformed lines and keeps processing subsequent lines
        fuzz_runner = tmp_path / "fuzz_runner.py"
        fuzz_runner.write_text(
            "print('starting', flush=True)\n"
            "print('[ACDSO_PROGRESS] {\"phase\": 1e309, \"total\": 6, \"name\": \"InfPhase\"}', flush=True)\n"
            "print('[ACDSO_PROGRESS] {\"phase\": NaN, \"total\": 6, \"name\": \"NaNPhase\"}', flush=True)\n"
            "print('[ACDSO_PROGRESS] {\"phase\": true, \"total\": 6, \"name\": \"BoolPhase\"}', flush=True)\n"
            "print('[ACDSO_PROGRESS] {\"phase\": 3, \"total\": 6, \"name\": \"AutoML Training\", \"status\": \"running\"}', flush=True)\n"
            "print('[ACDSO_PROGRESS] {\"phase\": 6, \"total\": 6, \"name\": \"Report Generation\", \"status\": \"done\"}', flush=True)\n"
            "print('finished_all_lines', flush=True)\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(webui_app, "RUNNER", fuzz_runner)

        up = _upload_sample(client, "tips_0cccb3b1499e.csv")
        r = client.post("/api/run", json={"upload_id": up["upload_id"], "predict": ["tip"]})
        assert r.status_code == 200
        job_id = r.get_json()["job_id"]

        js = None
        for _ in range(40):
            js = client.get(f"/api/job/{job_id}").get_json()
            if js["status"] != "running":
                break
            time.sleep(0.05)

        assert js is not None
        assert js["status"] == "finished"
        assert js["progress"]["phase"] == 6
        assert js["progress"]["percent"] == 100
        assert js["progress"]["name"] == "Report Generation"
        assert any("finished_all_lines" in ln for ln in js["lines"])
        assert any("[phase 6/6] Report Generation (done)" in ln for ln in js["lines"])

    def test_artifact_returns_json_on_400_403_404(self, isolated_webui_dirs):
        from acdso.webui import create_app

        app = create_app()
        app.config["TESTING"] = False
        c = app.test_client()

        (isolated_webui_dirs["logs"] / "run_x.log").write_text("secret", encoding="utf-8")

        for path, expected_status in (
            ("", 400),
            (".", 403),
            ("../start.py", 403),
            ("webui_logs/run_x.log", 403),
            ("missing_artifact.json", 404),
        ):
            url = "/api/artifact" if path == "" else f"/api/artifact?path={path}"
            resp = c.get(url)
            assert resp.status_code == expected_status, f"path={path!r} -> {resp.status_code}"
            assert resp.content_type.startswith("application/json"), (
                f"Expected JSON error for {path!r}, got {resp.content_type}"
            )
            body = resp.get_json()
            assert isinstance(body, dict) and body.get("error")

    def test_post_routes_return_400_json_on_non_dict_and_non_iterable_payloads(self):
        from acdso.webui import create_app

        app = create_app()
        app.config["TESTING"] = False
        c = app.test_client()

        up = _upload_sample(c, "tips_0cccb3b1499e.csv")
        uid = up["upload_id"]

        endpoints = (
            "/api/run",
            "/api/validate-columns",
            "/api/suggest-target",
            "/api/suggest-horizon",
            "/api/nl-plan",
            "/api/forecast",
        )
        for ep in endpoints:
            for raw_payload in ("[1, 2, 3]", '"hello"', "123", "true", "null", "{bad json"):
                resp = c.post(ep, data=raw_payload, content_type="application/json")
                assert resp.status_code == 400, f"{ep} with {raw_payload!r} -> {resp.status_code}"
                assert resp.content_type.startswith("application/json")
                assert isinstance(resp.get_json(), dict) and resp.get_json().get("error")

        # Non-iterable / non-string role fields must return 400 JSON, never 500
        for ep in ("/api/run", "/api/validate-columns"):
            for bad_body in (
                {"upload_id": uid, "predict": 123},
                {"upload_id": uid, "predict": ["tip"], "exclude": 123},
                {"upload_id": uid, "target": {"a": 1}},
                {"upload_id": uid, "predict": True},
                {"upload_id": [1, 2], "predict": ["tip"]},
            ):
                resp = c.post(ep, json=bad_body)
                assert resp.status_code == 400, f"{ep} with {bad_body!r} -> {resp.status_code}"
                assert resp.content_type.startswith("application/json")
                assert isinstance(resp.get_json(), dict) and resp.get_json().get("error")

        # Forecast filters validation & scalar filter safety (use JFrog dates + Service filter)
        up_jf = _upload_sample(c, "yearJFrog-data-transfer-20250821-20260814_146d8b6117aa.csv")
        jf_uid = up_jf["upload_id"]

        for bad_fc in (
            {"upload_id": jf_uid, "date_col": "Date", "value_col": "Total Data Transfer (B)", "filters": {"Service": 123}},
            {"upload_id": jf_uid, "date_col": "Date", "value_col": "Total Data Transfer (B)", "filters": [1, 2]},
            {"upload_id": jf_uid, "date_col": "Date", "value_col": "Total Data Transfer (B)", "filters": {"Service": {"bad": 1}}},
            {"upload_id": jf_uid, "date_col": 123, "value_col": "Total Data Transfer (B)"},
        ):
            resp = c.post("/api/forecast", json=bad_fc)
            assert resp.status_code == 400, f"/api/forecast {bad_fc!r} -> {resp.status_code}"
            assert resp.content_type.startswith("application/json")
            assert isinstance(resp.get_json(), dict) and resp.get_json().get("error")

        # Scalar string filter should work cleanly (no 500)
        ok_scalar = c.post(
            "/api/forecast",
            json={
                "upload_id": jf_uid,
                "date_col": "Date",
                "value_col": "Total Data Transfer (B)",
                "filters": {"Service": "Artifactory"},
                "horizon": 2,
            },
        )
        assert ok_scalar.status_code == 200, ok_scalar.get_data(as_text=True)

    def test_concurrent_uploads_and_index_listing_thread_safe(self):
        from concurrent.futures import ThreadPoolExecutor

        from acdso.webui import create_app

        app = create_app()
        csv_bytes = (UPLOADS_SAMPLES / "tips_0cccb3b1499e.csv").read_bytes()
        errors = []

        def worker(idx: int):
            c = app.test_client()
            try:
                up = _upload_bytes(c, f"tips_thread_{idx}.csv", csv_bytes)
                assert up["status"] == 200
                lst = c.get("/api/uploads")
                assert lst.status_code == 200
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(worker, range(12)))
        assert not errors, f"Concurrent upload/list errors: {errors}"
