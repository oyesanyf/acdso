"""Tier 3 E2E Tests: Cross-Feature Pairwise Integration Scenarios (6 tests)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from .conftest import poll_job_until_done


@pytest.mark.tier3
@pytest.mark.f1
@pytest.mark.f2
def test_t3_1_f1_f2_best_score_and_no_auto_exclude_cli(dataset_factory, run_cli, tmp_path: Path) -> None:
    """Pairwise F1 x F2: --no-auto-exclude retains ID column while default excludes it; --exclude & --best-score work."""
    src_df = pd.read_csv(dataset_factory.copy_sample("scratch_domains", "sd_src.csv"))
    src_df["record_id"] = np.arange(1, len(src_df) + 1)
    csv_default = dataset_factory.write_csv(src_df, "sd_default.csv")
    csv_noexcl = dataset_factory.write_csv(src_df, "sd_noexcl.csv")

    dir_def = tmp_path / "run_def"
    dir_noexcl = tmp_path / "run_noexcl"

    r_def = run_cli(
        ["--file", str(csv_default), "--predict", "ai_provider_label", "--simple", "--skip-benchmark"],
        cwd=dir_def,
        timeout=150.0,
    )
    assert r_def.returncode == 0, f"STDOUT:\n{r_def.stdout}\nSTDERR:\n{r_def.stderr}"

    r_noexcl = run_cli(
        [
            "--file",
            str(csv_noexcl),
            "--predict",
            "ai_provider_label",
            "--no-auto-exclude",
            "--exclude",
            "similarity_score",
            "--best-score",
            "--simple",
            "--skip-benchmark",
        ],
        cwd=dir_noexcl,
        timeout=150.0,
    )
    assert r_noexcl.returncode == 0, f"STDOUT:\n{r_noexcl.stdout}\nSTDERR:\n{r_noexcl.stderr}"

    m_def = joblib.load(dir_def / "reports" / "model_sd_default_ai_provider_label.pkl")
    m_noexcl = joblib.load(dir_noexcl / "reports" / "model_sd_noexcl_ai_provider_label.pkl")

    assert "record_id" in (m_def.get("exclude") or []), f"Expected 'record_id' auto-excluded, got {m_def.get('exclude')}"
    assert "record_id" not in (m_noexcl.get("exclude") or []), (
        f"Expected --no-auto-exclude to keep 'record_id', got {m_noexcl.get('exclude')}"
    )
    assert "similarity_score" in (m_noexcl.get("exclude") or []), (
        f"Expected --exclude similarity_score to be recorded, got {m_noexcl.get('exclude')}"
    )
    assert "similarity_score" not in (m_noexcl.get("feature_columns") or [])

    audit_noexcl = json.loads((dir_noexcl / "reports" / "audit_sd_noexcl.json").read_text(encoding="utf-8"))
    evaluated = audit_noexcl["candidates_evaluated"]
    assert audit_noexcl["selected"]["metric"] >= max(c["metric"] for c in evaluated) - 1e-12


@pytest.mark.tier3
@pytest.mark.f1
@pytest.mark.f4
def test_t3_2_f1_f4_automl_risk_and_pareto_surfaced_via_report_summary(
    isolated_webui_client, dataset_factory, run_cli, tmp_path: Path
) -> None:
    """Pairwise F1 x F4: AutoML risk & Pareto produced by CLI are inspectable via WebUI /api/report-summary."""
    _, client, (_, rep_dir, _) = isolated_webui_client
    X_leak, y_leak = dataset_factory.make_leaky_df(n_rows=150, seed=42)
    df = X_leak.copy()
    df["target"] = y_leak
    csv_path = dataset_factory.write_csv(df, "leaky_summary.csv")

    res = run_cli(
        ["--file", str(csv_path), "--target", "target", "--simple", "--skip-benchmark"],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    combined_files = list(rep_dir.glob("autonomous_leaky_summary*_combined.json"))
    assert combined_files
    rel_name = combined_files[0].relative_to(rep_dir).as_posix()

    sum_resp = client.get(f"/api/report-summary?path={rel_name}")
    assert sum_resp.status_code == 200, sum_resp.get_data(as_text=True)
    summary = sum_resp.get_json()
    assert summary["kind"] == "combined"
    assert summary["meta"]["target"] == "target"
    assert len(summary["pareto_front"]) >= 1
    assert isinstance(summary["risk_report"], dict)
    assert any("leakage" in str(n).lower() for n in (summary["risk_report"].get("notes") or []))
    audit_rr = (summary.get("audit_trail") or {}).get("risk_report") or {}
    assert len(audit_rr["leakage_proxy_features"]) >= 1 or audit_rr["leakage_proxy_score_max"] >= 0.95
    assert isinstance(summary["selected_model"], dict) and summary["selected_model"].get("name")


@pytest.mark.tier3
@pytest.mark.f2
@pytest.mark.f3
def test_t3_3_f2_f3_persisted_upload_consumed_by_cli_with_splitext(
    isolated_webui_client, dataset_factory, run_cli, tmp_path: Path
) -> None:
    """Pairwise F2 x F3: Persisted WebUI upload fed to CLI --file uses clean splitext stems."""
    _, client, (up_dir, _, _) = isolated_webui_client
    df = dataset_factory.make_regression_df(n_rows=80)
    json_path = dataset_factory.write_json(df, "upload_pair.json")

    with json_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, "upload_pair.json")}, content_type="multipart/form-data")
    assert up.status_code == 200
    stored_as = up.get_json()["stored_as"]
    stored_path = up_dir / stored_as
    assert stored_path.is_file()

    listed = client.get("/api/uploads")
    assert listed.status_code == 200

    res = run_cli(
        ["--file", str(stored_path), "--predict", "y", "--simple", "--skip-benchmark"],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    stem = stored_path.stem
    pred_csv = tmp_path / "reports" / f"predictions_{stem}_y_full_dataset.csv"
    assert pred_csv.is_file()
    assert len(pd.read_csv(pred_csv)) == 80


@pytest.mark.tier3
@pytest.mark.f2
@pytest.mark.f1
def test_t3_4_f2_f1_multi_target_multi_output_pipeline(dataset_factory, run_cli, tmp_path: Path) -> None:
    """Pairwise F2 x F1: CLI --target t1,t2 --multi-output trains joint multi-output AutoML model."""
    df = dataset_factory.make_multi_target_df(n_rows=90)
    csv_path = dataset_factory.write_csv(df, "multi_out.csv")

    res = run_cli(
        ["--file", str(csv_path), "--target", "t1,t2", "--multi-output", "--simple"],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    reports_dir = tmp_path / "reports"
    results_json = reports_dir / "multi_target_multi_out_results.json"
    model_pkl = reports_dir / "model_multi_out_multi_output.pkl"
    assert results_json.is_file()
    assert model_pkl.is_file()

    saved = joblib.load(model_pkl)
    preds = saved["model"].predict(df[["x1", "x2", "x3"]].iloc[:5])
    assert np.asarray(preds).shape == (5, 2)


@pytest.mark.tier3
@pytest.mark.f3
@pytest.mark.f4
def test_t3_5_f3_f4_upload_suggest_horizon_forecast_and_artifact_download(
    isolated_webui_client, dataset_factory, run_cli, tmp_path: Path
) -> None:
    """Pairwise F3 x F4: Upload yearJFrog -> suggest-horizon -> forecast -> /api/reports & /api/artifact (+ CLI --timeseries)."""
    _, client, _ = isolated_webui_client
    jfrog_path = dataset_factory.copy_sample("jfrog", "jfrog_t3_5.csv")

    with jfrog_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, "jfrog_t3_5.csv")}, content_type="multipart/form-data")
    assert up.status_code == 200
    uid = up.get_json()["upload_id"]

    hor = client.post(
        "/api/suggest-horizon",
        json={"upload_id": uid, "datetime_col": "Date", "target_col": "Total Data Transfer (B)"},
    )
    assert hor.status_code == 200
    horizon = hor.get_json()["horizon"] or 3

    fc = client.post(
        "/api/forecast",
        json={
            "upload_id": uid,
            "date_col": "Date",
            "value_col": "Total Data Transfer (B)",
            "agg": "sum",
            "horizon": horizon,
        },
    )
    assert fc.status_code == 200
    csv_rel = fc.get_json()["csv_path"]

    rep_list = client.get("/api/reports").get_json()["reports"]
    assert any(r["path"] == csv_rel for r in rep_list)

    dl = client.get(f"/api/artifact?path={csv_rel}")
    assert dl.status_code == 200
    text = dl.data.decode("utf-8")
    assert "(GB)" in text

    ts_cli_path = dataset_factory.write_csv(dataset_factory.make_timeseries_df(n_periods=45), "ts_cli_t3_5.csv")
    ts_dir = tmp_path / "ts_cli_out"
    ts_res = run_cli(
        [
            "--timeseries",
            "--file",
            str(ts_cli_path),
            "--target",
            "metric",
            "--datetime-col",
            "Date",
            "--horizon",
            "3",
        ],
        cwd=ts_dir,
        timeout=150.0,
    )
    assert ts_res.returncode == 0, f"STDOUT:\n{ts_res.stdout}\nSTDERR:\n{ts_res.stderr}"
    assert (ts_dir / "reports" / "ts_benchmark.html").is_file()


@pytest.mark.tier3
@pytest.mark.f4
@pytest.mark.f2
def test_t3_6_f4_f2_webui_run_parses_cli_progress_and_deletes_artifacts(webui_client, dataset_factory) -> None:
    """Pairwise F4 x F2: WebUI /api/run invokes run_acdso.py, parses [ACDSO_PROGRESS], and deletes reports."""
    _, client = webui_client
    df = dataset_factory.make_regression_df(n_rows=70)
    csv_path = dataset_factory.write_csv(df, "e2e_t3_6.csv")

    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, csv_path.name)}, content_type="multipart/form-data")
    uid = up.get_json()["upload_id"]

    run_resp = client.post(
        "/api/run",
        json={"upload_id": uid, "predict": ["y"], "simple": True, "skip_benchmark": True},
    )
    assert run_resp.status_code == 200
    job_id = run_resp.get_json()["job_id"]

    final, lines, progress_snapshots = poll_job_until_done(
        lambda cur: (
            (r := client.get(f"/api/job/{job_id}?cursor={cur}")).status_code,
            r.get_json(),
        ),
        timeout=150.0,
    )
    assert final["status"] == "finished", f"Job failed: {final}\nLines:\n" + "\n".join(lines[-30:])
    assert final["returncode"] == 0
    assert isinstance(final.get("progress"), dict)
    assert len(progress_snapshots) >= 1
    assert final["progress"]["status"] == "done"
    assert final["progress"]["percent"] == 100
    assert any(line.startswith("[phase ") for line in lines)

    created_paths = [a["path"] for a in final["artifacts"]]
    assert len(created_paths) >= 1
    for rel in created_paths:
        del_resp = client.delete(f"/api/reports?path={rel}")
        assert del_resp.status_code == 200
