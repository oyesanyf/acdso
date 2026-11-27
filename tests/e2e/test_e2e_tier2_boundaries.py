"""Tier 2 E2E Tests: Boundary Conditions, Negative Inputs & Error Handling (24 tests: 6 per feature F1-F4)."""

from __future__ import annotations

import importlib.util
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .conftest import stop_job_via_client


# ===========================================================================
# F1: Core AutoML Boundaries & Error Handling (6 tests)
# ===========================================================================

@pytest.mark.tier2
@pytest.mark.f1
def test_f1_t2_unfitted_estimator_raises_runtime_error() -> None:
    """Calling predict/predict_proba/score before fit() raises RuntimeError."""
    from acdso import ACDSOClassifier, ACDSORegressor

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = np.array([0, 1, 0])

    clf = ACDSOClassifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_proba(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.score(X, y)

    reg = ACDSORegressor()
    with pytest.raises(RuntimeError, match="not fitted"):
        reg.predict(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        reg.score(X, y)


@pytest.mark.tier2
@pytest.mark.f1
def test_f1_t2_empty_and_mismatched_inputs_fail_cleanly() -> None:
    """Empty DataFrames and X/y length mismatches fail with controlled ValueError/RuntimeError."""
    from acdso import ACDSOClassifier, analyze_risk

    empty_X = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=float)})
    empty_y = np.array([], dtype=int)
    with pytest.raises((ValueError, RuntimeError)):
        ACDSOClassifier(cv=3, max_candidates=2, verbose=0).fit(empty_X, empty_y)

    X = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float)})
    y_short = np.array([0, 1] * 5)
    with pytest.raises((ValueError, RuntimeError)):
        analyze_risk(X, y_short, task="classification")


@pytest.mark.tier2
@pytest.mark.f1
def test_f1_t2_single_class_target_and_bad_scorer_handled_cleanly() -> None:
    """Single-class target either raises cleanly or falls back without hallucinating classes; invalid scorer raises."""
    from acdso import ACDSOClassifier

    X = pd.DataFrame({"f1": np.linspace(0, 1, 30), "f2": np.linspace(1, 2, 30)})
    y = np.zeros(30, dtype=int)
    clf = ACDSOClassifier(cv=3, max_candidates=2, verbose=0)
    try:
        clf.fit(X, y)
    except (ValueError, RuntimeError):
        pass
    else:
        preds = clf.predict(X)
        assert len(preds) == 30
        assert set(preds) == {0}

    with pytest.raises((ValueError, KeyError, RuntimeError)):
        ACDSOClassifier(scorer="nonexistent_metric_xyz", cv=3, max_candidates=2, verbose=0).fit(
            X, np.array([0, 1] * 15)
        )


@pytest.mark.tier2
@pytest.mark.f1
def test_f1_t2_high_missingness_and_tiny_dataset_cv_adaptation() -> None:
    """Tiny dataset (n=16, cv=5) with 30% NaNs across mixed dtypes fits and predicts cleanly."""
    from acdso import ACDSOClassifier

    rng = np.random.RandomState(42)
    n = 16
    X = pd.DataFrame(
        {
            "num1": rng.randn(n),
            "num2": rng.randn(n),
            "cat": rng.choice(["a", "b", "c"], size=n).astype(object),
        }
    )
    y = np.array([0, 1] * (n // 2))
    # Inject ~30% missing values across features
    for col, idxs in (("num1", [0, 3, 7, 11, 14]), ("num2", [1, 4, 8, 12, 15]), ("cat", [2, 5, 9, 13])):
        X.loc[idxs, col] = np.nan

    clf = ACDSOClassifier(cv=5, max_candidates=2, random_state=42, verbose=0)
    clf.fit(X, y)
    assert clf.best_estimator_ is not None
    assert clf.risk_report_["missing_ratio"] >= 0.2
    preds = clf.predict(X)
    assert len(preds) == n


@pytest.mark.tier2
@pytest.mark.f1
def test_f1_t2_impossible_constraints_fallback_no_attribute_error(dataset_factory) -> None:
    """Impossibly tight constraints and short time_budget_s fall back cleanly without AttributeError."""
    from acdso import ACDSOClassifier

    df = dataset_factory.make_classification_df(n_rows=80)
    X = df.drop(columns=["target"])
    y = df["target"]

    clf_tb = ACDSOClassifier(
        cv=3,
        max_candidates=2,
        time_budget_s=0.05,
        random_state=42,
        verbose=0,
    )
    clf_tb.fit(X, y)
    assert clf_tb.best_estimator_ is not None
    assert len(clf_tb.predict(X)) == len(X)

    clf = ACDSOClassifier(
        cv=3,
        max_candidates=2,
        max_train_time_s=1e-9,
        max_latency_ms_per_row=1e-9,
        random_state=42,
        verbose=0,
    )
    try:
        clf.fit(X, y)
    except RuntimeError:
        return
    assert clf.best_estimator_ is not None
    assert len(clf.predict(X)) == len(X)


@pytest.mark.tier2
@pytest.mark.f1
def test_f1_t2_empty_pareto_raises_and_best_score_selection(dataset_factory) -> None:
    """knee_point_selection([]) raises ValueError; selection_mode='best_score' picks top CV metric."""
    from acdso import ACDSOClassifier
    from acdso.pareto import knee_point_selection

    with pytest.raises(ValueError):
        knee_point_selection([])

    df = dataset_factory.make_classification_df(n_rows=90)
    X = df.drop(columns=["target"])
    y = df["target"]

    clf = ACDSOClassifier(
        scorer="accuracy",
        cv=3,
        max_candidates=3,
        selection_mode="best_score",
        random_state=42,
        verbose=0,
    )
    clf.fit(X, y)
    evaluated = clf.audit_trail_["candidates_evaluated"]
    selected_metric = clf.audit_trail_["selected"]["metric"]
    assert selected_metric >= max(c["metric"] for c in evaluated) - 1e-12


# ===========================================================================
# F2: CLI Boundary & Negative Cases (6 tests)
# ===========================================================================

@pytest.mark.tier2
@pytest.mark.f2
def test_f2_t2_missing_file_exits_one(run_cli, tmp_path: Path) -> None:
    """run_acdso.py --file nonexistent.csv exits with non-zero code 1."""
    res = run_cli(["--file", str(tmp_path / "does_not_exist_e2e.csv"), "--simple"], cwd=tmp_path, timeout=45.0)
    assert res.returncode == 1, f"Expected exit 1, got {res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


@pytest.mark.tier2
@pytest.mark.f2
def test_f2_t2_empty_and_header_only_csv_exit_one(run_cli, tmp_path: Path) -> None:
    """Zero-byte CSV and header-only CSV both exit with code 1."""
    empty_csv = tmp_path / "zero_bytes.csv"
    empty_csv.write_bytes(b"")
    r1 = run_cli(["--file", str(empty_csv), "--target", "y", "--simple"], cwd=tmp_path, timeout=45.0)
    assert r1.returncode == 1, f"Zero-byte CSV returned {r1.returncode}\n{r1.stdout}\n{r1.stderr}"

    header_only = tmp_path / "header_only.csv"
    header_only.write_text("x1,x2,y\n", encoding="utf-8")
    r2 = run_cli(["--file", str(header_only), "--target", "y", "--simple"], cwd=tmp_path, timeout=45.0)
    assert r2.returncode == 1, f"Header-only CSV returned {r2.returncode}\n{r2.stdout}\n{r2.stderr}"


@pytest.mark.tier2
@pytest.mark.f2
def test_f2_t2_invalid_target_and_predict_columns_exit_one(dataset_factory, run_cli, tmp_path: Path) -> None:
    """Unknown --target and unknown --predict columns exit with code 1 and report available columns."""
    csv_path = dataset_factory.copy_sample("tips", "tips_bad_col.csv")

    r_target = run_cli(["--file", str(csv_path), "--target", "no_such_col", "--simple"], cwd=tmp_path, timeout=45.0)
    assert r_target.returncode == 1, r_target.stdout
    assert "total_bill" in (r_target.stdout + r_target.stderr)

    r_pred = run_cli(["--file", str(csv_path), "--predict", "no_such_col", "--simple"], cwd=tmp_path, timeout=45.0)
    assert r_pred.returncode == 1, r_pred.stdout
    assert "total_bill" in (r_pred.stdout + r_pred.stderr)


@pytest.mark.tier2
@pytest.mark.f2
def test_f2_t2_invalid_horizon_and_negative_budget_exit_one(dataset_factory, run_cli, tmp_path: Path) -> None:
    """Invalid --horizon <= 0 and negative --budget < 0 exit with code 1."""
    ts_path = dataset_factory.write_csv(dataset_factory.make_timeseries_df(n_periods=12), "ts_bad_h.csv")
    r_h0 = run_cli(
        ["--timeseries", "--file", str(ts_path), "--target", "metric", "--datetime-col", "Date", "--horizon", "0"],
        cwd=tmp_path,
        timeout=45.0,
    )
    assert r_h0.returncode == 1, r_h0.stdout

    r_hneg = run_cli(
        ["--timeseries", "--file", str(ts_path), "--target", "metric", "--datetime-col", "Date", "--horizon", "-3"],
        cwd=tmp_path,
        timeout=45.0,
    )
    assert r_hneg.returncode == 1, r_hneg.stdout

    clf_path = dataset_factory.write_csv(dataset_factory.make_classification_df(n_rows=40), "dec_bad_b.csv")
    r_budget = run_cli(
        ["--decision", "--file", str(clf_path), "--target", "target", "--budget", "-50"],
        cwd=tmp_path,
        timeout=45.0,
    )
    assert r_budget.returncode == 1, r_budget.stdout


@pytest.mark.tier2
@pytest.mark.f2
def test_f2_t2_predict_without_file_exits_one(run_cli, tmp_path: Path) -> None:
    """run_acdso.py --predict tip without --file exits with code 1."""
    res = run_cli(["--predict", "tip"], cwd=tmp_path, timeout=45.0)
    assert res.returncode == 1, f"Expected exit 1, got {res.returncode}\n{res.stdout}\n{res.stderr}"


@pytest.mark.tier2
@pytest.mark.f2
def test_f2_t2_nan_targets_and_numeric_datetime_false_positive_guard(dataset_factory, run_cli, tmp_path: Path) -> None:
    """NaN target rows are handled cleanly and integer year/ID columns are not misclassified as datetime."""
    df = dataset_factory.make_classification_df(n_rows=90, include_nan_target=True)
    df["year_code"] = np.arange(2000, 2000 + len(df))
    csv_path = dataset_factory.write_csv(df, "nan_and_numdate.csv")

    res = run_cli(
        ["--file", str(csv_path), "--target", "target", "--simple", "--skip-benchmark"],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    combined_files = list((tmp_path / "reports").glob("autonomous_nan_and_numdate*_combined.json"))
    assert combined_files
    combined = json.loads(combined_files[0].read_text(encoding="utf-8"))
    assert combined.get("datetime_col") is None, f"Numeric column misclassified as datetime: {combined.get('datetime_col')}"


# ===========================================================================
# F3: WebUI Upload, Validation & Suggestion Boundaries (6 tests)
# ===========================================================================

@pytest.mark.tier2
@pytest.mark.f3
def test_f3_t2_upload_missing_file_and_bad_extension_400(isolated_webui_client) -> None:
    """POST /api/upload rejects missing files, empty filenames, and disallowed suffixes with 400."""
    _, client, (up_dir, _, _) = isolated_webui_client

    r_missing = client.post("/api/upload", data={}, content_type="multipart/form-data")
    assert r_missing.status_code == 400
    assert "error" in r_missing.get_json()

    r_txt = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"hello world"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert r_txt.status_code == 400
    assert "error" in r_txt.get_json()

    r_exe = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"MZ\x90\x00"), "payload.exe")},
        content_type="multipart/form-data",
    )
    assert r_exe.status_code == 400
    assert list(up_dir.glob("*payload*")) == []


@pytest.mark.tier2
@pytest.mark.f3
def test_f3_t2_upload_corrupt_or_missing_engine_400_and_cleans_disk(isolated_webui_client) -> None:
    """Corrupt/empty files or missing optional engines (.xlsx) return 400 and leave no partial upload."""
    _, client, (up_dir, _, _) = isolated_webui_client

    r_empty = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b""), "empty_bad.csv")},
        content_type="multipart/form-data",
    )
    assert r_empty.status_code == 400
    assert list(up_dir.glob("*empty_bad*")) == []

    r_pkl = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"not-a-valid-pickle"), "corrupt_bad.pkl")},
        content_type="multipart/form-data",
    )
    assert r_pkl.status_code == 400
    assert list(up_dir.glob("*corrupt_bad*")) == []

    if importlib.util.find_spec("openpyxl") is None:
        r_xlsx = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b"PK\x03\x04fake"), "no_engine.xlsx")},
            content_type="multipart/form-data",
        )
        assert r_xlsx.status_code == 400
        assert "error" in r_xlsx.get_json()
        assert list(up_dir.glob("*no_engine*")) == []


@pytest.mark.tier2
@pytest.mark.f3
def test_f3_t2_suggestion_endpoints_unknown_upload_and_blank_prompt_400(isolated_webui_client, dataset_factory) -> None:
    """GET /api/uploads/<id> returns 404 and suggestion/validation/NL-plan/forecast endpoints return 400 for unknown upload_id / missing fields."""
    _, client, _ = isolated_webui_client

    r_missing_upload = client.get("/api/uploads/nonexistent_id")
    assert r_missing_upload.status_code == 404
    assert "error" in r_missing_upload.get_json()

    for route in ("/api/suggest-target", "/api/suggest-horizon", "/api/validate-columns", "/api/nl-plan", "/api/forecast"):
        resp = client.post(route, json={"upload_id": "nonexistent_id"})
        assert resp.status_code == 400, f"{route} returned {resp.status_code} for unknown upload_id"
        assert "error" in resp.get_json()

    tips_path = dataset_factory.copy_sample("tips", "tips_neg.csv")
    with tips_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, "tips_neg.csv")}, content_type="multipart/form-data")
    uid = up.get_json()["upload_id"]

    r_blank = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "   "})
    assert r_blank.status_code == 400

    r_hor = client.post("/api/suggest-horizon", json={"upload_id": uid, "datetime_col": "day"})
    assert r_hor.status_code == 400


@pytest.mark.tier2
@pytest.mark.f3
def test_f3_t2_suggest_horizon_short_series_returns_null(isolated_webui_client, dataset_factory) -> None:
    """Series with < 4 unique periods returns 200 with horizon=None and explanatory reason."""
    _, client, _ = isolated_webui_client
    df_short = dataset_factory.make_timeseries_df(n_periods=2)
    csv_path = dataset_factory.write_csv(df_short, "short_ts.csv")

    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, "short_ts.csv")}, content_type="multipart/form-data")
    uid = up.get_json()["upload_id"]

    resp = client.post(
        "/api/suggest-horizon",
        json={"upload_id": uid, "datetime_col": "Date", "target_col": "metric"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["horizon"] is None
    assert isinstance(data.get("reason"), str) and len(data["reason"]) > 0


@pytest.mark.tier2
@pytest.mark.f3
def test_f3_t2_validate_columns_blocks_dual_role_mismatch_and_high_card_id(isolated_webui_client, dataset_factory) -> None:
    """validate-columns blocks dual-role columns, predict != target mismatch, and high-cardinality IDs."""
    _, client, _ = isolated_webui_client
    jfrog_path = dataset_factory.copy_sample("jfrog", "jfrog_val.csv")
    with jfrog_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, "jfrog_val.csv")}, content_type="multipart/form-data")
    uid = up.get_json()["upload_id"]

    r_dual = client.post(
        "/api/validate-columns",
        json={"upload_id": uid, "predict": ["Action"], "exclude": ["Action"]},
    )
    assert r_dual.status_code == 200
    assert r_dual.get_json()["ok"] is False

    r_mismatch = client.post(
        "/api/validate-columns",
        json={"upload_id": uid, "predict": ["Action"], "target": ["Service"]},
    )
    assert r_mismatch.status_code == 200
    assert r_mismatch.get_json()["ok"] is False, "Expected predict != target mismatch to be blocked"

    r_high_card = client.post(
        "/api/validate-columns",
        json={"upload_id": uid, "predict": ["User"]},
    )
    assert r_high_card.status_code == 200
    hc_data = r_high_card.get_json()
    assert hc_data["ok"] is False
    assert any(i["severity"] == "block" for i in hc_data["issues"])


@pytest.mark.tier2
@pytest.mark.f3
def test_f3_t2_forecast_invalid_options_and_horizon_bounds_400(isolated_webui_client, dataset_factory) -> None:
    """/api/forecast rejects bad agg, non-int horizon, non-date column, <3 periods, and horizon=100/0 with 400 while horizon=36 returns 200."""
    _, client, _ = isolated_webui_client
    df = dataset_factory.make_timeseries_df(n_periods=12)
    csv_path = dataset_factory.write_csv(df, "fc_bounds.csv")

    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, "fc_bounds.csv")}, content_type="multipart/form-data")
    uid = up.get_json()["upload_id"]

    assert client.post(
        "/api/forecast",
        json={"upload_id": uid, "date_col": "Date", "value_col": "metric", "agg": "median"},
    ).status_code == 400

    assert client.post(
        "/api/forecast",
        json={"upload_id": uid, "date_col": "Date", "value_col": "metric", "horizon": "abc"},
    ).status_code == 400

    assert client.post(
        "/api/forecast",
        json={"upload_id": uid, "date_col": "category", "value_col": "metric", "horizon": 3},
    ).status_code == 400

    # Out-of-bounds horizon (>36 or <1) -> 400; max valid horizon=36 -> 200
    assert client.post(
        "/api/forecast",
        json={"upload_id": uid, "date_col": "Date", "value_col": "metric", "horizon": 100},
    ).status_code == 400
    assert client.post(
        "/api/forecast",
        json={"upload_id": uid, "date_col": "Date", "value_col": "metric", "horizon": 0},
    ).status_code == 400

    valid_max = client.post(
        "/api/forecast",
        json={"upload_id": uid, "date_col": "Date", "value_col": "metric", "horizon": 36},
    )
    assert valid_max.status_code == 200
    assert len(valid_max.get_json()["forecast"]) == 36

    # < 3 periods after filter -> 400
    short_df = dataset_factory.make_timeseries_df(n_periods=2)
    short_path = dataset_factory.write_csv(short_df, "fc_short.csv")
    with short_path.open("rb") as fh:
        up_s = client.post("/api/upload", data={"file": (fh, "fc_short.csv")}, content_type="multipart/form-data")
    assert client.post(
        "/api/forecast",
        json={"upload_id": up_s.get_json()["upload_id"], "date_col": "Date", "value_col": "metric", "horizon": 3},
    ).status_code == 400


# ===========================================================================
# F4: WebUI Jobs, Security & Report Inspection Boundaries (6 tests)
# ===========================================================================

@pytest.mark.tier2
@pytest.mark.f4
def test_f4_t2_run_invalid_payloads_return_400(webui_client, dataset_factory) -> None:
    """POST /api/run rejects empty payloads, missing targets, invalid bounds, and role conflicts with 400."""
    _, client = webui_client

    assert client.post("/api/run", json={}).status_code == 400
    assert client.post("/api/run", json={"upload_id": "missing_id", "predict": ["tip"]}).status_code == 400

    tips_path = dataset_factory.copy_sample("tips", "e2e_tips_run_bad.csv")
    with tips_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, "e2e_tips_run_bad.csv")}, content_type="multipart/form-data")
    uid = up.get_json()["upload_id"]

    assert client.post("/api/run", json={"upload_id": uid}).status_code == 400
    assert client.post("/api/run", json={"upload_id": uid, "predict": ["tip"], "task": "invalid_task"}).status_code == 400
    assert client.post("/api/run", json={"upload_id": uid, "predict": ["tip"], "horizon": 0}).status_code == 400
    assert client.post("/api/run", json={"upload_id": uid, "predict": ["tip"], "budget": -10}).status_code == 400
    assert client.post(
        "/api/run",
        json={"upload_id": uid, "predict": ["tip"], "target": ["total_bill"]},
    ).status_code == 400


@pytest.mark.tier2
@pytest.mark.f4
def test_f4_t2_concurrent_run_and_delete_while_running_return_409(isolated_webui_client, dataset_factory) -> None:
    """While a job is running, a second POST /api/run and DELETE /api/reports both return 409 (on isolated dirs)."""
    _, client, _ = isolated_webui_client
    df = dataset_factory.make_classification_df(n_rows=240)
    csv_path = dataset_factory.write_csv(df, "e2e_409.csv")

    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, csv_path.name)}, content_type="multipart/form-data")
    uid = up.get_json()["upload_id"]

    r1 = client.post("/api/run", json={"upload_id": uid, "target": ["target"]})
    assert r1.status_code == 200
    job_id = r1.get_json()["job_id"]

    try:
        r_job = client.get(f"/api/job/{job_id}")
        assert r_job.status_code == 200 and r_job.get_json()["status"] == "running"

        r2 = client.post("/api/run", json={"upload_id": uid, "target": ["target"]})
        assert r2.status_code == 409

        r_del = client.delete("/api/reports?all=1")
        assert r_del.status_code == 409
    finally:
        stop_job_via_client(client, job_id)
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            st = client.get(f"/api/job/{job_id}").get_json()
            if st["status"] != "running":
                break
            time.sleep(0.15)


@pytest.mark.tier2
@pytest.mark.f4
def test_f4_t2_unknown_job_id_returns_404(isolated_webui_client) -> None:
    """GET /api/job/<unknown> and POST /api/stop/<unknown> return 404."""
    _, client, _ = isolated_webui_client

    assert client.get("/api/job/unknown_job_xyz").status_code == 404
    assert client.post("/api/stop/unknown_job_xyz").status_code == 404


@pytest.mark.tier2
@pytest.mark.f4
def test_f4_t2_path_traversal_blocked_403(isolated_webui_client) -> None:
    """Path traversal attempts on /api/artifact, /api/report-summary, and DELETE /api/reports return 403."""
    _, client, _ = isolated_webui_client

    for bad_path in ("../run_acdso.py", "..\\start.py", "../../etc/passwd", "C:/Windows/win.ini"):
        r_art = client.get(f"/api/artifact?path={bad_path}")
        assert r_art.status_code == 403, f"/api/artifact allowed {bad_path} -> {r_art.status_code}"

        r_sum = client.get(f"/api/report-summary?path={bad_path}")
        assert r_sum.status_code == 403, f"/api/report-summary allowed {bad_path} -> {r_sum.status_code}"

        r_del = client.delete(f"/api/reports?path={bad_path}")
        assert r_del.status_code == 403, f"DELETE /api/reports allowed {bad_path} -> {r_del.status_code}"


@pytest.mark.tier2
@pytest.mark.f4
def test_f4_t2_missing_or_nonexistent_paths_400_and_404(isolated_webui_client) -> None:
    """Missing path params return 400; nonexistent files or non-file dirs return 404; webui_logs returns 403."""
    _, client, (_, rep_dir, _) = isolated_webui_client
    (rep_dir / "empty_subdir").mkdir(parents=True, exist_ok=True)

    assert client.get("/api/artifact").status_code == 400
    assert client.get("/api/report-summary").status_code == 400
    assert client.delete("/api/reports").status_code == 400

    assert client.get("/api/artifact?path=missing_artifact.json").status_code == 404
    assert client.get("/api/artifact?path=empty_subdir").status_code == 404
    assert client.get("/api/artifact?path=webui_logs").status_code == 403
    assert client.get("/api/report-summary?path=missing_summary.json").status_code == 404
    assert client.delete("/api/reports?path=missing_artifact.json").status_code == 404


@pytest.mark.tier2
@pytest.mark.f4
def test_f4_t2_report_summary_corrupt_and_job_cursor_beyond_eof(isolated_webui_client, dataset_factory) -> None:
    """/api/report-summary rejects unsupported/corrupt files with 400; /api/job?cursor=999999 returns empty lines."""
    _, client, (_, rep_dir, _) = isolated_webui_client
    (rep_dir / "bad_blob.bin").write_bytes(b"\x00\x01\x02\x03")
    (rep_dir / "corrupt_combined.json").write_text("{not valid json", encoding="utf-8")

    assert client.get("/api/report-summary?path=bad_blob.bin").status_code == 400
    assert client.get("/api/report-summary?path=corrupt_combined.json").status_code == 400

    df = dataset_factory.make_regression_df(n_rows=60)
    csv_path = dataset_factory.write_csv(df, "e2e_cursor_eof.csv")
    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, csv_path.name)}, content_type="multipart/form-data")
    uid = up.get_json()["upload_id"]

    run_resp = client.post("/api/run", json={"upload_id": uid, "target": ["y"], "simple": True, "skip_benchmark": True})
    assert run_resp.status_code == 200
    job_id = run_resp.get_json()["job_id"]

    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        r = client.get(f"/api/job/{job_id}?cursor=999999")
        assert r.status_code == 200
        data = r.get_json()
        assert data["lines"] == []
        if data["status"] != "running":
            break
        time.sleep(0.25)
