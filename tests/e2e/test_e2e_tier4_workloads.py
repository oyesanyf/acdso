"""Tier 4 E2E Tests: End-to-End Real-Dataset Application Workloads (W1-W6)."""

from __future__ import annotations

import importlib.util
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from .conftest import parse_progress_events, poll_job_until_done


class _TableBalanceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.open_tables = 0
        self.close_tables = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "table":
            self.open_tables += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "table":
            self.close_tables += 1


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f2
def test_w1_cli_quick_autonomous_predict_tips(dataset_factory, run_cli, tmp_path: Path) -> None:
    """W1: CLI quick autonomous + --predict on real tips CSV produces progress, predictions, JSON, and valid HTML."""
    csv_path = dataset_factory.copy_sample("tips", "w1_tips.csv")
    res = run_cli(
        ["--file", str(csv_path), "--predict", "tip", "--simple", "--skip-benchmark", "--force"],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "\x1b[" not in res.stdout, "Unexpected ANSI escape codes in piped stdout"

    events = parse_progress_events(res.stdout)
    assert len(events) >= 1
    phases = [int(ev["phase"]) for ev in events if isinstance(ev.get("phase"), int)]
    assert phases == sorted(phases), f"Progress phases out of order: {phases}"

    reports_dir = tmp_path / "reports"
    pred_csv = reports_dir / "predictions_w1_tips_tip_full_dataset.csv"
    assert pred_csv.is_file()
    pred_df = pd.read_csv(pred_csv)
    assert len(pred_df) == 244 and "tip_predicted" in pred_df.columns

    combined_files = list(reports_dir.glob("autonomous_w1_tips*_combined.json"))
    html_files = list(reports_dir.glob("autonomous_w1_tips*_report.html"))
    assert combined_files and html_files

    html_text = html_files[0].read_text(encoding="utf-8", errors="replace")
    parser = _TableBalanceParser()
    parser.feed(html_text)
    assert parser.open_tables == parser.close_tables
    file_hrefs = re.findall(r'href="(file:///[^"]+)"', html_text)
    assert file_hrefs, "Expected clickable file:/// links in HTML report"
    assert all("\\" not in href for href in file_hrefs), f"Raw backslash inside file URI: {file_hrefs}"
    data_paths = re.findall(r'data-path="([^"]+)"', html_text)
    assert data_paths, "Expected data-path attributes on copy buttons"
    assert not re.search(r"writeText\(['\"][A-Za-z]:\\", html_text), "Unescaped Windows backslash inside JS writeText()"


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f2
def test_w2_cli_classification_auto_exclude_scratch_domains_json(dataset_factory, run_cli, tmp_path: Path) -> None:
    """W2: CLI classification + auto-exclude on scratch_domains (with record_id) converted to JSON."""
    raw_df = pd.read_csv(dataset_factory.copy_sample("scratch_domains", "sd_w2.csv"))
    raw_df["record_id"] = np.arange(1, len(raw_df) + 1)
    json_path = dataset_factory.write_json(raw_df, "w2_domains.json")

    res = run_cli(
        [
            "--file",
            str(json_path),
            "--predict",
            "ai_provider_label",
            "--task",
            "classification",
            "--simple",
            "--skip-benchmark",
        ],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    reports_dir = tmp_path / "reports"
    pred_csv = reports_dir / "predictions_w2_domains_ai_provider_label_full_dataset.csv"
    assert pred_csv.is_file()
    pred_df = pd.read_csv(pred_csv)
    assert len(pred_df) == 200
    assert "ai_provider_label_predicted" in pred_df.columns
    assert set(pred_df["ai_provider_label_predicted"]).issubset({"known_ai", "unknown"})
    assert "ai_provider_label_probability" in pred_df.columns

    saved = joblib.load(reports_dir / "model_w2_domains_ai_provider_label.pkl")
    assert "record_id" in (saved.get("exclude") or [])
    high_card = (saved.get("categorical_encoding") or {}).get("high_card_freqs") or {}
    assert "domain" in high_card


@pytest.mark.tier4
@pytest.mark.f3
@pytest.mark.f4
@pytest.mark.f1
def test_w3_live_waitress_http_workbench_full_lifecycle_tips(live_webui_server, dataset_factory) -> None:
    """W3: Full Waitress HTTP lifecycle on real tips CSV (upload -> suggest -> validate -> run -> summary -> delete)."""
    srv = live_webui_server

    st_root, _, html_bytes = srv.get("/")
    assert st_root == 200 and b"ACDSO" in html_bytes

    st_h, health = srv.get_json("/api/health")
    assert st_h == 200 and health["ok"] is True

    tips_path = dataset_factory.copy_sample("tips", "e2e_w3_tips.csv")
    st_up, up = srv.upload_file(tips_path)
    assert st_up == 200 and up["rows"] == 244
    uid = up["upload_id"]

    st_list, uploads_payload = srv.get_json("/api/uploads")
    assert st_list == 200
    entries = uploads_payload.get("uploads") if isinstance(uploads_payload, dict) else uploads_payload
    assert any(e.get("upload_id") == uid or e.get("id") == uid for e in entries)

    st_sug, sug = srv.post_json("/api/suggest-target", {"upload_id": uid})
    assert st_sug == 200 and len(sug["suggestions"]) >= 1

    st_val, val = srv.post_json("/api/validate-columns", {"upload_id": uid, "predict": ["tip"]})
    assert st_val == 200 and val["ok"] is True

    st_run, run_data = srv.post_json(
        "/api/run",
        {"upload_id": uid, "predict": ["tip"], "simple": True, "skip_benchmark": True},
    )
    assert st_run == 200
    job_id = run_data["job_id"]

    final, _, _ = poll_job_until_done(
        lambda cur: srv.get_json(f"/api/job/{job_id}?cursor={cur}"),
        timeout=150.0,
    )
    assert final["status"] == "finished" and final["returncode"] == 0
    assert isinstance(final.get("progress"), dict)

    st_rep, rep_list = srv.get_json("/api/reports?limit=50")
    assert st_rep == 200
    artifact_paths = [a["path"] for a in final["artifacts"]]
    assert any(p.endswith("_combined.json") for p in artifact_paths)

    combined_rel = next(p for p in artifact_paths if p.endswith("_combined.json"))
    st_sum, summary = srv.get_json(f"/api/report-summary?path={combined_rel}")
    assert st_sum == 200 and isinstance(summary, dict)

    st_dl, _, dl_bytes = srv.get(f"/api/artifact?path={combined_rel}")
    assert st_dl == 200 and len(dl_bytes) > 10

    for rel in artifact_paths:
        st_del, _ = srv.delete_json(f"/api/reports?path={rel}")
        assert st_del == 200


@pytest.mark.tier4
@pytest.mark.f3
@pytest.mark.f4
def test_w4_live_webui_timeseries_forecast_and_horizon_yearjfrog(live_webui_server, dataset_factory) -> None:
    """W4: Waitress WebUI time-series horizon suggestion, column guard, and filtered forecast on yearJFrog."""
    srv = live_webui_server
    jfrog_path = dataset_factory.copy_sample("jfrog", "e2e_w4_jfrog.csv")

    st_up, up = srv.upload_file(jfrog_path)
    assert st_up == 200 and up["rows"] == 4135
    uid = up["upload_id"]

    st_hor, hor = srv.post_json(
        "/api/suggest-horizon",
        {"upload_id": uid, "datetime_col": "Date", "target_col": "Total Data Transfer (B)"},
    )
    assert st_hor == 200
    assert isinstance(hor["horizon"], int) and hor["horizon"] >= 1

    st_blk, blk = srv.post_json("/api/validate-columns", {"upload_id": uid, "predict": ["User"]})
    assert st_blk == 200 and blk["ok"] is False

    st_ok, ok_val = srv.post_json(
        "/api/validate-columns",
        {"upload_id": uid, "predict": ["Total Data Transfer (B)"]},
    )
    assert st_ok == 200 and ok_val["ok"] is True

    st_fc, fc = srv.post_json(
        "/api/forecast",
        {
            "upload_id": uid,
            "date_col": "Date",
            "value_col": "Total Data Transfer (B)",
            "agg": "sum",
            "horizon": 3,
            "filters": {"Action": ["download"]},
        },
    )
    assert st_fc == 200
    assert len(fc["history"]) == 13
    assert len(fc["forecast"]) == 3

    st_dl, _, csv_bytes = srv.get(f"/api/artifact?path={fc['csv_path']}")
    assert st_dl == 200 and b"(GB)" in csv_bytes


@pytest.mark.tier4
@pytest.mark.f1
def test_w5_automl_risk_leakage_pareto_and_pickle_workflow(dataset_factory, tmp_path: Path) -> None:
    """W5: End-to-end AutoML Risk, Leakage, Multicollinearity, 100% refit, Pareto & pickle workflow."""
    from sklearn.datasets import load_diabetes

    from acdso import ACDSOClassifier, ACDSORegressor, analyze_risk, compute_risk_penalty

    X_leak, y_leak = dataset_factory.make_leaky_df(n_rows=180, seed=42)
    clf_bal = ACDSOClassifier(scorer="accuracy", cv=3, max_candidates=3, preference="balanced", random_state=42, verbose=0)
    clf_safe = ACDSOClassifier(scorer="accuracy", cv=3, max_candidates=3, preference="safe", random_state=42, verbose=0)
    clf_bal.fit(X_leak, y_leak)
    clf_safe.fit(X_leak, y_leak)

    for clf in (clf_bal, clf_safe):
        assert len(clf.risk_report_["leakage_proxy_features"]) >= 1 or clf.risk_report_["leakage_proxy_score_max"] >= 0.95
        assert len(clf.pareto_front_) >= 1
        assert clf.best_estimator_ is not None
        assert clf.audit_trail_["selected"]["refitted_full"] is True
        json.dumps(clf.audit_trail_)

    rr_leak = analyze_risk(X_leak, y_leak, task="classification", cv=3, random_state=42)
    pen_safe = compute_risk_penalty(rr_leak, cv_score_std=0.01, preference="safe")
    pen_bal = compute_risk_penalty(rr_leak, cv_score_std=0.01, preference="balanced")
    assert pen_safe >= pen_bal
    assert clf_safe.audit_trail_["selected"]["risk_penalty"] >= clf_bal.audit_trail_["selected"]["risk_penalty"]

    X_mc, y_mc = dataset_factory.make_multicollinear_df(n_rows=180, seed=42)
    reg_mc = ACDSORegressor(scorer="r2", cv=3, max_candidates=3, random_state=42, verbose=0)
    reg_mc.fit(X_mc, y_mc)
    assert reg_mc.risk_report_["vif_max"] >= 10.0
    assert reg_mc.audit_trail_["selected"]["refitted_full"] is True
    assert float(reg_mc.score(X_mc, y_mc)) > 0.7
    json.dumps(reg_mc.audit_trail_)

    diabetes = load_diabetes(as_frame=True)
    X_diab_train, X_diab_test = diabetes.data.iloc[:350], diabetes.data.iloc[350:]
    y_diab_train, y_diab_test = diabetes.target.iloc[:350], diabetes.target.iloc[350:]
    reg_diab = ACDSORegressor(cv=3, max_candidates=2, random_state=42, verbose=0)
    reg_diab.fit(X_diab_train, y_diab_train)
    assert reg_diab.audit_trail_["selected"]["refitted_full"] is True
    assert len(reg_diab.pareto_front_) >= 1
    json.dumps(reg_diab.audit_trail_)

    pkl_path = tmp_path / "diabetes_reg.pkl"
    joblib.dump(reg_diab, pkl_path)
    loaded = joblib.load(pkl_path)
    orig_preds = reg_diab.predict(X_diab_test)
    loaded_preds = loaded.predict(X_diab_test)
    assert np.allclose(loaded_preds, orig_preds)
    assert float(loaded.score(X_diab_test, y_diab_test)) > 0.0


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f2
@pytest.mark.f3
def test_w6_multi_format_and_multi_target_workload(
    isolated_webui_client, dataset_factory, run_cli, tmp_path: Path
) -> None:
    """W6: Multi-format (.pkl/.json + guarded .xlsx/.parquet) and multi-target CLI + WebUI NL-plan workload."""
    _, client, _ = isolated_webui_client
    df_multi = dataset_factory.make_multi_target_df(n_rows=90)
    pkl_path = dataset_factory.write_pickle(df_multi, "w6_multi.pkl")

    res = run_cli(
        ["--file", str(pkl_path), "--target", "t1,t2", "--multi-output", "--simple"],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert (tmp_path / "reports" / "multi_target_w6_multi_results.json").is_file()
    assert (tmp_path / "reports" / "model_w6_multi_multi_output.pkl").is_file()

    # WebUI JSON upload + NL plan
    tips_df = pd.read_csv(dataset_factory.copy_sample("tips", "w6_tips_src.csv"))
    tips_json = dataset_factory.write_json(tips_df, "w6_tips.json")
    with tips_json.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, "w6_tips.json")}, content_type="multipart/form-data")
    assert up.status_code == 200
    uid = up.get_json()["upload_id"]

    nl = client.post("/api/nl-plan", json={"upload_id": uid, "prompt": "predict tip excluding size"})
    assert nl.status_code == 200
    assert "tip" in nl.get_json()["predict"]

    # Optional Excel / Parquet roundtrip when engine is installed, or clean CLI exit 1 when absent
    if importlib.util.find_spec("openpyxl") is not None:
        xlsx_path = dataset_factory.write_xlsx(df_multi, "w6_opt.xlsx")
        r_xlsx = run_cli(["--file", str(xlsx_path), "--target", "t1", "--simple"], cwd=tmp_path, timeout=150.0)
        assert r_xlsx.returncode == 0
    else:
        fake_xlsx = tmp_path / "w6_missing_engine.xlsx"
        fake_xlsx.write_bytes(b"PK\x03\x04dummy")
        r_xlsx = run_cli(["--file", str(fake_xlsx), "--target", "t1", "--simple"], cwd=tmp_path, timeout=45.0)
        assert r_xlsx.returncode == 1
