"""Tier 1 E2E Tests: Core Feature Happy-Path & Contract Verification (24 tests: 6 per feature F1-F4)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris

from .conftest import (
    PYTHON_EXE,
    REPO_ROOT,
    RUNNER,
    SRC_ROOT,
    START_PY,
    find_free_port,
    http_json,
    http_request,
    parse_progress_events,
    poll_job_until_done,
)


class _TagCounterParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.open_tables = 0
        self.close_tables = 0
        self.has_title = False
        self.scripts: list[str] = []
        self._in_script = False
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        t = tag.lower()
        if t == "table":
            self.open_tables += 1
        elif t == "title":
            self.has_title = True
        elif t == "script":
            self._in_script = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "table":
            self.close_tables += 1
        elif t == "script" and self._in_script:
            self.scripts.append("".join(self._buf))
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._buf.append(data)


# ===========================================================================
# F1: Core AutoML Fit / Predict / Risk / Pareto (6 tests)
# ===========================================================================

@pytest.mark.tier1
@pytest.mark.f1
def test_f1_t1_public_exports_and_lazy_submodules() -> None:
    """All acdso.__all__ symbols are importable and lazy submodules defer heavy imports until accessed."""
    probe = (
        "import sys; import acdso; "
        "eager = [m for m in ('acdso.eda', 'acdso.advisors', 'acdso.feature_importance', "
        "'acdso.auto_preprocessing', 'acdso.core') if m in sys.modules]; "
        "print(','.join(eager))"
    )
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(SRC_ROOT)})
    proc = subprocess.run(
        [PYTHON_EXE, "-B", "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=30.0,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"Eagerly imported submodules on `import acdso`: {proc.stdout.strip()}"

    import acdso

    assert isinstance(acdso.__all__, list)
    assert len(acdso.__all__) >= 59
    for sym in acdso.__all__:
        assert hasattr(acdso, sym), f"Missing exported symbol: {sym}"
        assert getattr(acdso, sym) is not None

    for submod in ("integrations", "timeseries", "decision"):
        mod = getattr(acdso, submod)
        assert mod is not None


@pytest.mark.tier1
@pytest.mark.f1
def test_f1_t1_classifier_binary_and_multiclass_lifecycle(dataset_factory) -> None:
    """ACDSOClassifier fits/predicts/scores binary string targets and multiclass iris data."""
    from acdso import ACDSOClassifier

    df_bin = dataset_factory.make_classification_df(n_rows=100, n_classes=2, string_target=True)
    X_bin = df_bin.drop(columns=["target"])
    y_bin = df_bin["target"]

    clf_bin = ACDSOClassifier(scorer="f1", cv=3, max_candidates=2, random_state=42, verbose=0)
    ret = clf_bin.fit(X_bin.iloc[:80], y_bin.iloc[:80])
    assert ret is clf_bin
    assert clf_bin.best_estimator_ is not None
    assert isinstance(clf_bin.risk_report_, dict)
    assert isinstance(clf_bin.pareto_front_, list) and len(clf_bin.pareto_front_) >= 1
    assert isinstance(clf_bin.audit_trail_, dict)
    json.dumps(clf_bin.audit_trail_)

    preds = clf_bin.predict(X_bin.iloc[80:])
    assert len(preds) == 20
    assert set(preds).issubset({"yes", "no"})
    proba = clf_bin.predict_proba(X_bin.iloc[80:])
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert 0.0 <= float(clf_bin.score(X_bin.iloc[80:], y_bin.iloc[80:])) <= 1.0

    iris = load_iris()
    X_mc = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_mc = iris.target
    clf_mc = ACDSOClassifier(scorer="accuracy", cv=3, max_candidates=2, random_state=42, verbose=0)
    clf_mc.fit(X_mc.iloc[:120], y_mc[:120])
    mc_proba = clf_mc.predict_proba(X_mc.iloc[120:])
    assert mc_proba.shape == (30, 3)
    assert np.allclose(mc_proba.sum(axis=1), 1.0, atol=1e-5)
    assert float(clf_mc.score(X_mc.iloc[120:], y_mc[120:])) >= 0.7


@pytest.mark.tier1
@pytest.mark.f1
def test_f1_t1_regressor_fit_predict_score(dataset_factory) -> None:
    """ACDSORegressor fits, predicts continuous values, and populates regression risk_report_."""
    from acdso import ACDSORegressor

    df = dataset_factory.make_regression_df(n_rows=120)
    X = df.drop(columns=["y"])
    y = df["y"]

    reg = ACDSORegressor(scorer="r2", cv=3, max_candidates=2, random_state=42, verbose=0)
    reg.fit(X.iloc[:96], y.iloc[:96])
    preds = reg.predict(X.iloc[96:])
    assert preds.shape == (24,)
    assert np.issubdtype(preds.dtype, np.number)
    assert float(reg.score(X.iloc[96:], y.iloc[96:])) > 0.5
    assert reg.risk_report_["task"] == "regression"
    assert reg.risk_report_["class_imbalance_max_ratio"] is None
    assert len(reg.pareto_front_) >= 1
    for key in ("risk_report", "constraints", "candidates_evaluated", "pareto_front", "selected"):
        assert key in reg.audit_trail_
    json.dumps(reg.audit_trail_)


@pytest.mark.tier1
@pytest.mark.f1
def test_f1_t1_best_estimator_100pct_refit_and_clone(dataset_factory) -> None:
    """best_estimator_ is refitted on 100% of training rows and estimator supports sklearn clone."""
    from acdso import ACDSOClassifier

    df = dataset_factory.make_classification_df(n_rows=30, n_classes=2, seed=7)
    X = df.drop(columns=["target"])
    y = df["target"]

    clf = ACDSOClassifier(scorer="accuracy", cv=3, max_candidates=2, random_state=42, verbose=0)
    cloned = clone(clf)
    assert cloned is not clf
    assert cloned.get_params()["max_candidates"] == 2

    clf.fit(X, y)
    assert clf.best_estimator_ is not None
    assert len(clf.predict(X)) == len(X)
    assert clf.audit_trail_["selected"]["refitted_full"] is True

    # Verify StandardScaler inside best_estimator_ observed all 100% (30) rows, not an 80% split
    seen_counts: list[int] = []
    if hasattr(clf.best_estimator_, "named_steps"):
        for step in clf.best_estimator_.named_steps.values():
            if hasattr(step, "n_samples_seen_"):
                seen_counts.append(int(np.max(step.n_samples_seen_)))
            if hasattr(step, "transformers_"):
                for _, tr, _ in step.transformers_:
                    if hasattr(tr, "named_steps"):
                        for inner in tr.named_steps.values():
                            if hasattr(inner, "n_samples_seen_"):
                                seen_counts.append(int(np.max(inner.n_samples_seen_)))
    assert seen_counts, "Expected fitted scaler in selected pipeline to record n_samples_seen_"
    assert max(seen_counts) == len(X), f"Expected 100% refit on {len(X)} rows, got {seen_counts}"


@pytest.mark.tier1
@pytest.mark.f1
def test_f1_t1_analyze_risk_detects_proxy_leakage_and_vif(dataset_factory) -> None:
    """analyze_risk() detects injected proxy leakage, high multicollinearity (VIF >= 10), and linearity."""
    from acdso import analyze_risk, compute_risk_penalty

    X_leak, y_leak = dataset_factory.make_leaky_df(n_rows=200, seed=42)
    rr_leak = analyze_risk(X_leak, y_leak, task="classification", cv=3, random_state=42)
    rr_dict = rr_leak.to_dict()
    assert rr_dict["task"] == "classification"
    assert len(rr_dict["leakage_proxy_features"]) >= 1 or rr_dict["leakage_proxy_score_max"] >= 0.95

    X_mc, y_mc = dataset_factory.make_multicollinear_df(n_rows=200, seed=42)
    rr_mc = analyze_risk(X_mc, y_mc, task="regression", cv=3, random_state=42)
    assert rr_mc.vif_max >= 10.0
    assert len(rr_mc.vif_features) >= 1
    assert isinstance(rr_mc.is_linear, bool) and rr_mc.is_linear is True
    assert 0.7 <= float(rr_mc.linearity_score) <= 1.0

    pen_safe = compute_risk_penalty(rr_leak, cv_score_std=0.05, preference="safe")
    pen_bal = compute_risk_penalty(rr_leak, cv_score_std=0.05, preference="balanced")
    assert pen_safe >= pen_bal


@pytest.mark.tier1
@pytest.mark.f1
def test_f1_t1_pareto_front_knee_selection_and_constraints() -> None:
    """pareto_front excludes dominated candidates, knee_point_selection picks balanced point, constraints filter."""
    from acdso.pareto import CandidateResult, filter_by_constraints, knee_point_selection, pareto_front

    c_fast_low = CandidateResult(
        name="fast_low",
        pipeline=None,
        metric=0.78,
        train_time_s=0.1,
        peak_mem_mb=10.0,
        latency_ms_per_row=0.05,
        risk_penalty=0.05,
        objectives=(0.78, 0.095, 2.398, 0.049, 0.05, 0.02),
    )
    c_knee = CandidateResult(
        name="balanced_knee",
        pipeline=None,
        metric=0.91,
        train_time_s=0.5,
        peak_mem_mb=25.0,
        latency_ms_per_row=0.15,
        risk_penalty=0.04,
        objectives=(0.91, 0.405, 3.258, 0.140, 0.04, 0.01),
    )
    c_slow_hi = CandidateResult(
        name="slow_high",
        pipeline=None,
        metric=0.92,
        train_time_s=20.0,
        peak_mem_mb=200.0,
        latency_ms_per_row=5.0,
        risk_penalty=0.08,
        objectives=(0.92, 3.045, 5.303, 1.792, 0.08, 0.03),
    )
    c_dominated = CandidateResult(
        name="dominated",
        pipeline=None,
        metric=0.75,
        train_time_s=25.0,
        peak_mem_mb=250.0,
        latency_ms_per_row=6.0,
        risk_penalty=0.20,
        objectives=(0.75, 3.258, 5.525, 1.946, 0.20, 0.10),
    )

    front = pareto_front([c_fast_low, c_knee, c_slow_hi, c_dominated])
    names = {c.name for c in front}
    assert "dominated" not in names
    assert {"fast_low", "balanced_knee", "slow_high"}.issubset(names)

    chosen = knee_point_selection(front)
    assert chosen in front

    passing, pruned = filter_by_constraints(
        [c_fast_low, c_knee, c_slow_hi], max_train_time_s=2.0, max_peak_mem_mb=100.0
    )
    assert {c.name for c in passing} == {"fast_low", "balanced_knee"}
    assert len(pruned) == 1 and pruned[0]["name"] == "slow_high"


# ===========================================================================
# F2: CLI File Loading, Flags & Exit Codes (6 tests)
# ===========================================================================

@pytest.mark.tier1
@pytest.mark.f2
def test_f2_t1_help_fast_and_complete(run_cli) -> None:
    """run_acdso.py --help exits 0 under 500ms and documents key CLI flags."""
    # Warm OS page cache once if desired? Contract measures cold/normal CLI --help < 500ms
    t0 = time.perf_counter()
    res = run_cli(["--help"], timeout=30.0)
    elapsed_s = time.perf_counter() - t0
    assert res.returncode == 0, res.stderr
    for flag in ("--file", "--target", "--predict", "--simple", "--skip-benchmark", "--no-auto-exclude", "--horizon", "--budget"):
        assert flag in res.stdout, f"Missing {flag} in --help output"
    assert elapsed_s < 0.5, f"--help took {elapsed_s:.3f}s (contract < 0.500s)"


@pytest.mark.tier1
@pytest.mark.f2
def test_f2_t1_csv_simple_mode_and_progress_events(dataset_factory, run_cli, tmp_path: Path) -> None:
    """CLI --file --target --simple exits 0, emits [ACDSO_PROGRESS], skips TS/benchmark, writes reports."""
    csv_path = dataset_factory.copy_sample("tips", "tips_simple.csv")
    res = run_cli(["--file", str(csv_path), "--target", "tip", "--simple"], cwd=tmp_path, timeout=150.0)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "\x1b[" not in res.stdout, "Unexpected ANSI escape codes in piped stdout"

    events = parse_progress_events(res.stdout)
    assert len(events) >= 1, "Expected [ACDSO_PROGRESS] JSON events on stdout"
    for ev in events:
        assert {"phase", "total", "name", "status", "detail"}.issubset(ev.keys())
        assert ev["status"] in {"running", "done", "error", "skipped"}

    reports_dir = tmp_path / "reports"
    combined_candidates = list(reports_dir.glob("autonomous_tips_simple*_combined.json"))
    html_candidates = list(reports_dir.glob("autonomous_tips_simple*_report.html"))
    assert combined_candidates, f"Combined JSON missing in {list(reports_dir.iterdir())}"
    assert html_candidates, f"HTML report missing in {list(reports_dir.iterdir())}"
    assert (reports_dir / "audit_tips_simple.json").is_file()
    assert (reports_dir / "model_tips_simple_tip.pkl").is_file() or (reports_dir / "model_tips_simple.pkl").is_file()

    combined = json.loads(combined_candidates[0].read_text(encoding="utf-8"))
    phases = combined.get("phase_results") or {}
    all_phases = phases.get("All Phases") or []
    phase_map = {item.get("name"): str(item.get("status") or "") for item in all_phases if isinstance(item, dict)}
    assert "Time Series" in phase_map and "SKIPPED" in phase_map["Time Series"], f"Unexpected TS phase: {phase_map}"
    assert "Benchmark (Custom File)" in phase_map and "SKIPPED" in phase_map["Benchmark (Custom File)"], (
        f"Unexpected Benchmark phase: {phase_map}"
    )


@pytest.mark.tier1
@pytest.mark.f2
def test_f2_t1_predict_single_target_csv_and_html_integrity(dataset_factory, run_cli, tmp_path: Path) -> None:
    """CLI --predict writes full-dataset CSV, updates target combined JSON, and produces balanced HTML."""
    csv_path = dataset_factory.copy_sample("tips", "tips_pred.csv")
    res = run_cli(
        ["--file", str(csv_path), "--predict", "tip", "--simple", "--skip-benchmark"],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    pred_csv = tmp_path / "reports" / "predictions_tips_pred_tip_full_dataset.csv"
    assert pred_csv.is_file(), f"Missing predictions CSV: {list((tmp_path / 'reports').iterdir())}"
    pred_df = pd.read_csv(pred_csv)
    assert len(pred_df) == 244
    assert "tip_predicted" in pred_df.columns

    combined_path = tmp_path / "reports" / "autonomous_tips_pred_tip_combined.json"
    if not combined_path.is_file():
        combined_path = tmp_path / "reports" / "autonomous_tips_pred_combined.json"
    assert combined_path.is_file()
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    assert "predictions" in combined
    assert combined["predictions"]["total_predictions"] == 244

    html_files = list((tmp_path / "reports").glob("autonomous_tips_pred*_report.html"))
    assert html_files
    html_text = html_files[0].read_text(encoding="utf-8", errors="replace")
    parser = _TagCounterParser()
    parser.feed(html_text)
    assert parser.open_tables == parser.close_tables, (
        f"Unbalanced <table> tags: open={parser.open_tables}, close={parser.close_tables}"
    )


@pytest.mark.tier1
@pytest.mark.f2
def test_f2_t1_non_csv_json_and_pickle_loading(dataset_factory, run_cli, tmp_path: Path) -> None:
    """load_user_file() loads .json and .pkl via CLI --file with os.path.splitext base naming."""
    df = dataset_factory.make_regression_df(n_rows=80)
    json_path = dataset_factory.write_json(df, "reg_data.json")
    pkl_path = dataset_factory.write_pickle(df, "reg_pkl.pkl")

    for data_path, stem in ((json_path, "reg_data"), (pkl_path, "reg_pkl")):
        res = run_cli(
            ["--file", str(data_path), "--target", "y", "--simple", "--skip-benchmark"],
            cwd=tmp_path,
            timeout=150.0,
        )
        assert res.returncode == 0, f"Failed for {data_path.name}:\n{res.stdout}\n{res.stderr}"
        reports_dir = tmp_path / "reports"
        assert (reports_dir / f"audit_{stem}.json").is_file()
        assert not list(reports_dir.glob(f"*{data_path.suffix}_*")), "Extension leaked into report stem"


@pytest.mark.tier1
@pytest.mark.f2
def test_f2_t1_multi_target_predict(dataset_factory, run_cli, tmp_path: Path) -> None:
    """CLI --predict t1,t2 --simple generates per-target CSVs and multi_target combined JSON/HTML."""
    df = dataset_factory.make_multi_target_df(n_rows=90)
    csv_path = dataset_factory.write_csv(df, "multi_in.csv")

    res = run_cli(
        ["--file", str(csv_path), "--predict", "t1,t2", "--simple", "--skip-benchmark"],
        cwd=tmp_path,
        timeout=180.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    reports_dir = tmp_path / "reports"
    assert (reports_dir / "predictions_multi_in_t1_full_dataset.csv").is_file()
    assert (reports_dir / "predictions_multi_in_t2_full_dataset.csv").is_file()
    combined_json = reports_dir / "multi_target_multi_in_combined.json"
    combined_html = reports_dir / "multi_target_multi_in_combined_report.html"
    assert combined_json.is_file()
    assert combined_html.is_file()
    data = json.loads(combined_json.read_text(encoding="utf-8"))
    assert set(data["predictions"].keys()) == {"t1", "t2"}


@pytest.mark.tier1
@pytest.mark.f2
def test_f2_t1_force_cleans_prior_artifacts(dataset_factory, run_cli, tmp_path: Path) -> None:
    """CLI --force removes stale reports for the dataset stem before writing fresh artifacts."""
    csv_path = dataset_factory.copy_sample("tips", "tips_force.csv")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stale_model = reports_dir / "model_tips_force_stale.pkl"
    stale_json = reports_dir / "autonomous_tips_force_stale_combined.json"
    stale_model.write_bytes(b"stale")
    stale_json.write_text("{}", encoding="utf-8")

    res = run_cli(
        ["--file", str(csv_path), "--target", "tip", "--quick", "--force"],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert not stale_model.exists(), "Stale model was not cleaned by --force"
    assert not stale_json.exists(), "Stale JSON was not cleaned by --force"
    assert list(reports_dir.glob("autonomous_tips_force*_combined.json"))


# ===========================================================================
# F3: WebUI Startup, Upload Persistence & Suggestions (6 tests)
# ===========================================================================

@pytest.mark.tier1
@pytest.mark.f3
def test_f3_t1_webui_fast_startup_and_root_html(isolated_webui_client) -> None:
    """import acdso.webui + create_app() < 500ms without eager pandas/sklearn, and GET / renders clean HTML."""
    probe = (
        "import sys, time; "
        "t0 = time.perf_counter(); "
        "import acdso.webui; "
        "app = acdso.webui.create_app(); "
        "dt = time.perf_counter() - t0; "
        "heavy = [m for m in ('pandas', 'numpy', 'sklearn', 'scipy') if m in sys.modules]; "
        "print('{:.4f}|{}'.format(dt, ','.join(heavy)))"
    )
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(SRC_ROOT)})
    proc = subprocess.run(
        [PYTHON_EXE, "-B", "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=30.0,
    )
    assert proc.returncode == 0, proc.stderr
    dt_str, heavy_str = proc.stdout.strip().splitlines()[-1].split("|", 1)
    assert float(dt_str) < 0.5, f"create_app startup took {dt_str}s (contract < 0.500s)"
    assert heavy_str == "", f"Heavy libraries eagerly imported at startup: {heavy_str}"

    _, client, _ = isolated_webui_client
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "{{" not in html and "{%" not in html, "Unrendered Jinja template syntax in GET /"
    assert "FLAG_META" in html
    assert "/api/upload" in html and "/api/run" in html and "/api/reports" in html
    parser = _TagCounterParser()
    parser.feed(html)
    assert parser.has_title


@pytest.mark.tier1
@pytest.mark.f3
def test_f3_t1_start_py_subprocess_serves_health_and_root() -> None:
    """start.py --no-browser launches Waitress and serves GET /api/health and GET /."""
    port = find_free_port()
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": str(SRC_ROOT),
        }
    )
    proc = subprocess.Popen(
        [PYTHON_EXE, "-B", str(START_PY), "--no-browser", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 12.0
        health_payload = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=2.0)
                raise AssertionError(f"start.py exited early ({proc.returncode}): {out!r} {err!r}")
            try:
                st, data = http_json(f"{base_url}/api/health", timeout=1.5)
                if st == 200:
                    health_payload = data
                    break
            except Exception:
                time.sleep(0.2)
        assert health_payload is not None, "start.py did not respond on /api/health within 12s"
        assert health_payload["ok"] is True
        assert health_payload["runner_present"] is True
        assert "active_job" in health_payload

        st_root, _, body = http_request(f"{base_url}/", timeout=5.0)
        assert st_root == 200
        assert b"ACDSO" in body
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


@pytest.mark.tier1
@pytest.mark.f3
def test_f3_t1_upload_and_persistence_across_restart(isolated_webui_dirs, dataset_factory) -> None:
    """POST /api/upload persists upload metadata so a fresh create_app() lists it via GET /api/uploads."""
    from acdso.webui import create_app

    app1 = create_app()
    c1 = app1.test_client()
    tips_path = dataset_factory.copy_sample("tips", "tips_persist.csv")

    with tips_path.open("rb") as fh:
        resp = c1.post("/api/upload", data={"file": (fh, "tips_persist.csv")}, content_type="multipart/form-data")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    up_data = resp.get_json()
    upload_id = up_data["upload_id"]
    assert up_data["rows"] == 244
    assert "tip" in up_data["columns"]
    assert isinstance(up_data["preview"], list) and 1 <= len(up_data["preview"]) <= 10
    assert "dtypes" in up_data and "predictable_columns" in up_data

    # Simulate server restart with a brand new app instance reading the same isolated UPLOAD_DIR
    app2 = create_app()
    c2 = app2.test_client()
    list_resp = c2.get("/api/uploads")
    assert list_resp.status_code == 200, list_resp.get_data(as_text=True)
    list_data = list_resp.get_json()
    entries = list_data.get("uploads") if isinstance(list_data, dict) else list_data
    assert isinstance(entries, list)
    assert any(item.get("upload_id") == upload_id or item.get("id") == upload_id for item in entries)

    detail_resp = c2.get(f"/api/uploads/{upload_id}")
    assert detail_resp.status_code == 200, detail_resp.get_data(as_text=True)
    detail_data = detail_resp.get_json()
    assert detail_data["upload_id"] == upload_id
    assert detail_data["rows"] == 244
    assert "tip" in detail_data["columns"]
    assert isinstance(detail_data.get("preview"), list) and len(detail_data["preview"]) >= 1

    sug = c2.post("/api/suggest-target", json={"upload_id": upload_id})
    assert sug.status_code == 200


@pytest.mark.tier1
@pytest.mark.f3
def test_f3_t1_suggest_target_fast_and_warning_free(isolated_webui_client, dataset_factory) -> None:
    """POST /api/suggest-target ranks outcomes under 500ms without pd.to_datetime warnings."""
    import warnings

    _, client, _ = isolated_webui_client
    tips_path = dataset_factory.copy_sample("tips", "tips_sug.csv")
    with tips_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, "tips_sug.csv")}, content_type="multipart/form-data")
    upload_id = up.get_json()["upload_id"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        resp = client.post("/api/suggest-target", json={"upload_id": upload_id})
        elapsed = time.perf_counter() - t0

    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["suggestions"]) >= 1
    assert data["source"] in {"rules", "llm"}
    assert "tip" in data["candidates"]
    dt_warnings = [w for w in caught if "Could not infer format" in str(w.message)]
    assert dt_warnings == [], f"Unexpected pd.to_datetime warnings: {[str(w.message) for w in dt_warnings]}"
    assert elapsed < 0.5, f"/api/suggest-target took {elapsed:.3f}s (contract < 0.500s)"


@pytest.mark.tier1
@pytest.mark.f3
def test_f3_t1_suggest_horizon_periodic_series(isolated_webui_client, dataset_factory) -> None:
    """POST /api/suggest-horizon returns positive horizon and grain on yearJFrog monthly data."""
    _, client, _ = isolated_webui_client
    jfrog_path = dataset_factory.copy_sample("jfrog", "jfrog_hor.csv")
    with jfrog_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, "jfrog_hor.csv")}, content_type="multipart/form-data")
    assert up.status_code == 200
    upload_id = up.get_json()["upload_id"]

    resp = client.post(
        "/api/suggest-horizon",
        json={"upload_id": upload_id, "datetime_col": "Date", "target_col": "Total Data Transfer (B)"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data["horizon"], int) and data["horizon"] >= 1
    assert data["grain"] in {"D", "W", "M", "Q", "Y"}
    assert "data" in (data.get("votes") or {})


@pytest.mark.tier1
@pytest.mark.f3
def test_f3_t1_validate_columns_nl_plan_and_forecast(isolated_webui_client, dataset_factory) -> None:
    """POST /api/validate-columns, /api/nl-plan, and /api/forecast work end-to-end."""
    _, client, (_, rep_dir, _) = isolated_webui_client

    tips_path = dataset_factory.copy_sample("tips", "tips_nl.csv")
    with tips_path.open("rb") as fh:
        up_tips = client.post("/api/upload", data={"file": (fh, "tips_nl.csv")}, content_type="multipart/form-data")
    tips_id = up_tips.get_json()["upload_id"]

    val = client.post("/api/validate-columns", json={"upload_id": tips_id, "predict": ["tip"], "exclude": ["size"]})
    assert val.status_code == 200
    val_data = val.get_json()
    assert val_data["ok"] is True
    assert not any(i["severity"] == "block" for i in val_data["issues"])

    nl = client.post("/api/nl-plan", json={"upload_id": tips_id, "prompt": "predict tip excluding size"})
    assert nl.status_code == 200
    nl_data = nl.get_json()
    assert "tip" in nl_data["predict"]
    assert nl_data["method"] in {"rule-based", "llm"}

    ts_df = dataset_factory.make_timeseries_df(n_periods=18)
    ts_path = dataset_factory.write_csv(ts_df, "ts_fc.csv")
    with ts_path.open("rb") as fh:
        up_ts = client.post("/api/upload", data={"file": (fh, "ts_fc.csv")}, content_type="multipart/form-data")
    ts_id = up_ts.get_json()["upload_id"]

    fc = client.post(
        "/api/forecast",
        json={"upload_id": ts_id, "date_col": "Date", "value_col": "metric", "agg": "sum", "horizon": 4},
    )
    assert fc.status_code == 200, fc.get_data(as_text=True)
    fc_data = fc.get_json()
    assert len(fc_data["forecast"]) == 4
    assert fc_data["chosen_method"]
    assert (rep_dir / fc_data["csv_path"]).is_file()


# ===========================================================================
# F4: WebUI Job Execution, Progress & Report Inspection (6 tests)
# ===========================================================================

@pytest.mark.tier1
@pytest.mark.f4
def test_f4_t1_run_job_polls_structured_progress_and_finishes(webui_client, dataset_factory) -> None:
    """POST /api/run launches run_acdso.py and GET /api/job/<id> exposes lines, cursor, and progress."""
    _, client = webui_client
    df = dataset_factory.make_regression_df(n_rows=70)
    csv_path = dataset_factory.write_csv(df, "e2e_run_poll.csv")

    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, csv_path.name)}, content_type="multipart/form-data")
    assert up.status_code == 200
    upload_id = up.get_json()["upload_id"]

    run_resp = client.post(
        "/api/run",
        json={"upload_id": upload_id, "predict": ["y"], "simple": True, "skip_benchmark": True},
    )
    assert run_resp.status_code == 200, run_resp.get_data(as_text=True)
    run_data = run_resp.get_json()
    job_id = run_data["job_id"]
    assert "--file" in run_data["command"] and "--predict y" in run_data["command"]

    final, lines, progress_list = poll_job_until_done(
        lambda cur: (
            (r := client.get(f"/api/job/{job_id}?cursor={cur}")).status_code,
            r.get_json(),
        ),
        timeout=150.0,
    )
    assert final["status"] == "finished", f"Job failed: {final}\nLines:\n" + "\n".join(lines[-30:])
    assert final["returncode"] == 0
    assert (REPO_ROOT / final["log_file"]).is_file()
    assert isinstance(final.get("progress"), dict), f"Expected M3 progress object in /api/job: {final}"
    assert {"phase", "total", "name", "status"}.issubset(final["progress"].keys())
    assert len(final["artifacts"]) >= 1


@pytest.mark.tier1
@pytest.mark.f4
def test_f4_t1_jobs_list_and_health_active_job(webui_client, dataset_factory) -> None:
    """GET /api/jobs lists recent jobs and GET /api/health clears active_job upon completion."""
    _, client = webui_client
    df = dataset_factory.make_classification_df(n_rows=70)
    csv_path = dataset_factory.write_csv(df, "e2e_jobs_list.csv")

    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, csv_path.name)}, content_type="multipart/form-data")
    upload_id = up.get_json()["upload_id"]

    run_resp = client.post(
        "/api/run",
        json={"upload_id": upload_id, "target": ["target"], "simple": True, "skip_benchmark": True},
    )
    assert run_resp.status_code == 200
    job_id = run_resp.get_json()["job_id"]

    jobs_resp = client.get("/api/jobs")
    assert jobs_resp.status_code == 200
    jobs = jobs_resp.get_json()["jobs"]
    assert any(j["job_id"] == job_id for j in jobs)

    final, _, _ = poll_job_until_done(
        lambda cur: (
            (r := client.get(f"/api/job/{job_id}?cursor={cur}")).status_code,
            r.get_json(),
        ),
        timeout=150.0,
    )
    assert final["status"] == "finished"

    health = client.get("/api/health").get_json()
    assert health["active_job"] is None


@pytest.mark.tier1
@pytest.mark.f4
def test_f4_t1_stop_running_job_via_contract_route(webui_client, dataset_factory) -> None:
    """POST /api/stop/<id> cancels an active subprocess job and returns updated status."""
    _, client = webui_client
    df = dataset_factory.make_classification_df(n_rows=250)
    csv_path = dataset_factory.write_csv(df, "e2e_stop_job.csv")

    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, csv_path.name)}, content_type="multipart/form-data")
    upload_id = up.get_json()["upload_id"]

    # Launch full autonomous mode (without --simple) so subprocess stays running long enough to stop
    run_resp = client.post("/api/run", json={"upload_id": upload_id, "target": ["target"]})
    assert run_resp.status_code == 200
    job_id = run_resp.get_json()["job_id"]

    stop_resp = client.post(f"/api/stop/{job_id}")
    assert stop_resp.status_code == 200, (
        f"Contracted POST /api/stop/<id> returned {stop_resp.status_code}: {stop_resp.get_data(as_text=True)}"
    )
    stop_data = stop_resp.get_json()
    assert stop_data["status"] == "cancelled"

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        h = client.get("/api/health").get_json()
        if h["active_job"] is None:
            break
        time.sleep(0.15)
    assert client.get("/api/health").get_json()["active_job"] is None


@pytest.mark.tier1
@pytest.mark.f4
def test_f4_t1_reports_list_and_artifact_download_inline(isolated_webui_client) -> None:
    """GET /api/reports excludes logs and GET /api/artifact serves exact bytes inline and as attachment."""
    _, client, (_, rep_dir, log_dir) = isolated_webui_client
    sample_json = rep_dir / "sample_report.json"
    sample_json.write_text('{"k": 123}', encoding="utf-8")
    (log_dir / "run_ignore.log").write_text("log line", encoding="utf-8")

    rep_resp = client.get("/api/reports?limit=50")
    assert rep_resp.status_code == 200
    rep_data = rep_resp.get_json()
    paths = [r["path"] for r in rep_data["reports"]]
    assert "sample_report.json" in paths
    assert all(not p.endswith(".log") for p in paths)

    att = client.get("/api/artifact?path=sample_report.json")
    assert att.status_code == 200
    assert att.data == b'{"k": 123}'
    assert "attachment" in att.headers.get("Content-Disposition", "").lower()

    inl = client.get("/api/artifact?path=sample_report.json&inline=1")
    assert inl.status_code == 200
    assert inl.data == b'{"k": 123}'


@pytest.mark.tier1
@pytest.mark.f4
def test_f4_t1_report_summary_combined_audit_and_predictions(isolated_webui_client) -> None:
    """GET /api/report-summary inspects *_combined.json, audit_*.json, and predictions_*.csv."""
    _, client, (_, rep_dir, _) = isolated_webui_client

    combined_payload = {
        "file": "tips.csv",
        "target": "tip",
        "task": "regression",
        "eda_report": {"n_rows": 244, "n_cols": 7, "quality_score": 96.5},
        "audit_trail": {
            "risk_report": {"task": "regression", "vif_max": 1.4, "notes": ["Low risk"]},
            "candidates_evaluated": [{"name": "Ridge", "metric": 0.45}],
            "pareto_front": [{"name": "Ridge", "metric": 0.45, "train_time_s": 0.05}],
            "selected": {"name": "Ridge", "metric": 0.45},
        },
        "predictions": {"total_predictions": 244},
    }
    (rep_dir / "autonomous_tips_tip_combined.json").write_text(json.dumps(combined_payload), encoding="utf-8")
    (rep_dir / "audit_tips.json").write_text(json.dumps(combined_payload["audit_trail"]), encoding="utf-8")
    (rep_dir / "predictions_tips_tip_full_dataset.csv").write_text(
        "total_bill,tip,tip_predicted\n16.99,1.01,2.50\n10.34,1.66,1.80\n",
        encoding="utf-8",
    )

    r_comb = client.get("/api/report-summary?path=autonomous_tips_tip_combined.json")
    assert r_comb.status_code == 200, r_comb.get_data(as_text=True)
    b_comb = r_comb.get_json()
    assert b_comb["kind"] == "combined"
    assert b_comb["meta"]["target"] == "tip"
    assert b_comb["eda_report"]["quality_score"] == 96.5
    assert len(b_comb["pareto_front"]) == 1 and b_comb["pareto_front"][0]["name"] == "Ridge"
    assert b_comb["risk_report"]["vif_max"] == 1.4
    assert b_comb["selected_model"]["name"] == "Ridge"
    assert b_comb["predictions"]["total_predictions"] == 244

    r_aud = client.get("/api/report-summary?path=audit_tips.json")
    assert r_aud.status_code == 200, r_aud.get_data(as_text=True)
    b_aud = r_aud.get_json()
    assert b_aud["kind"] == "audit"
    assert len(b_aud["pareto_front"]) == 1
    assert b_aud["risk_report"]["notes"] == ["Low risk"]

    r_csv = client.get("/api/report-summary?path=predictions_tips_tip_full_dataset.csv")
    assert r_csv.status_code == 200, r_csv.get_data(as_text=True)
    b_csv = r_csv.get_json()
    assert b_csv["kind"] == "csv"
    assert b_csv["total_rows"] == 2
    assert "tip_predicted" in b_csv["columns"]
    assert len(b_csv["rows"]) == 2


@pytest.mark.tier1
@pytest.mark.f4
def test_f4_t1_delete_single_report_and_delete_all(isolated_webui_client) -> None:
    """DELETE /api/reports removes single or all report files while preserving webui_logs/."""
    _, client, (_, rep_dir, log_dir) = isolated_webui_client
    f1 = rep_dir / "a_report.json"
    f2 = rep_dir / "b_report.csv"
    keep_log = log_dir / "run_keep.log"
    f1.write_text("{}", encoding="utf-8")
    f2.write_text("x\n1\n", encoding="utf-8")
    keep_log.write_text("keep", encoding="utf-8")

    r_one = client.delete("/api/reports?path=a_report.json")
    assert r_one.status_code == 200
    assert r_one.get_json()["deleted"] == ["a_report.json"]
    assert not f1.exists() and f2.exists()

    r_all = client.delete("/api/reports?all=1")
    assert r_all.status_code == 200
    assert "b_report.csv" in r_all.get_json()["deleted"]
    assert not f2.exists()
    assert keep_log.is_file()
