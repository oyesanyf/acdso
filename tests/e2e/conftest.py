"""Shared fixtures, HTTP/CLI runners, dataset builders, and artifact isolation for ACDSO E2E tests."""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
REPORTS_DIR = REPO_ROOT / "reports"
UPLOADS_DIR = REPO_ROOT / "uploads"
RUNNER = REPO_ROOT / "run_acdso.py"
START_PY = REPO_ROOT / "start.py"
PYTHON_EXE = sys.executable

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SAMPLE_DATASETS: dict[str, Path] = {
    "tips": UPLOADS_DIR / "tips_0cccb3b1499e.csv",
    "scratch_domains": UPLOADS_DIR / "scratch_domains_92b347bb484a.csv",
    "jfrog": UPLOADS_DIR / "yearJFrog-data-transfer-20250821-20260814_146d8b6117aa.csv",
}


def pytest_configure(config: pytest.Config) -> None:
    """Register custom tier and feature markers without mutating pyproject.toml."""
    markers = [
        "tier1: Core feature happy-path E2E tests",
        "tier2: Boundary, negative, and error-handling E2E tests",
        "tier3: Cross-feature pairwise integration E2E tests",
        "tier4: Real-dataset end-to-end workload tests",
        "tier5: Tier 5 adversarial & regression tests",
        "f1: Feature 1 - Core AutoML Python API",
        "f2: Feature 2 - CLI File Loading, Flags & Exit Codes",
        "f3: Feature 3 - WebUI Startup, Upload Persistence & Suggestions",
        "f4: Feature 4 - WebUI Job Execution, Progress & Report Inspection",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


# ---------------------------------------------------------------------------
# Artifact snapshot & Windows-safe cleanup helpers
# ---------------------------------------------------------------------------

def _snapshot_dir(root: Path) -> tuple[bool, set[Path], set[Path]]:
    if not root.exists():
        return False, set(), set()
    files: set[Path] = set()
    dirs: set[Path] = set()
    for p in root.rglob("*"):
        try:
            rel = p.relative_to(root)
            if p.is_file():
                files.add(rel)
            elif p.is_dir():
                dirs.add(rel)
        except OSError:
            pass
    return True, files, dirs


def _snapshot_root_artifacts(repo_root: Path) -> tuple[set[str], set[str]]:
    json_files: set[str] = set()
    catboost_dirs: set[str] = set()
    for p in repo_root.iterdir():
        if p.is_file() and p.suffix.lower() == ".json":
            json_files.add(p.name)
        elif p.is_dir() and p.name.startswith("catboost_"):
            catboost_dirs.add(p.name)
    return json_files, catboost_dirs


def _snapshot_upload_json(uploads_dir: Path) -> dict[Path, bytes]:
    snapshots: dict[Path, bytes] = {}
    if not uploads_dir.exists():
        return snapshots
    for p in uploads_dir.rglob("*.json"):
        try:
            if p.is_file() and p.stat().st_size <= 2 * 1024 * 1024:
                snapshots[p] = p.read_bytes()
        except OSError:
            pass
    return snapshots


def _unlink_with_retry(path: Path, retries: int = 8, delay: float = 0.2) -> None:
    for attempt in range(retries):
        try:
            if path.exists():
                path.unlink()
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
        except OSError:
            return


def _cleanup_new_entries(
    root: Path,
    existed: bool,
    pre_files: set[Path],
    pre_dirs: set[Path],
) -> None:
    if not root.exists():
        return
    new_files: list[Path] = []
    new_dirs: list[Path] = []
    for p in root.rglob("*"):
        try:
            rel = p.relative_to(root)
        except OSError:
            continue
        if p.is_file() and rel not in pre_files:
            new_files.append(p)
        elif p.is_dir() and rel not in pre_dirs:
            new_dirs.append(p)

    for f in new_files:
        _unlink_with_retry(f)

    for d in sorted(new_dirs, key=lambda item: len(item.parts), reverse=True):
        for attempt in range(5):
            try:
                if d.exists() and not any(d.iterdir()):
                    d.rmdir()
                break
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.15)
            except OSError:
                break

    if not existed and root.exists():
        try:
            if not any(root.iterdir()):
                root.rmdir()
        except OSError:
            pass


def _cleanup_root_artifacts(repo_root: Path, pre_jsons: set[str], pre_catboost: set[str]) -> None:
    for p in repo_root.iterdir():
        if p.is_file() and p.suffix.lower() == ".json" and p.name not in pre_jsons:
            _unlink_with_retry(p)
        elif p.is_dir() and p.name.startswith("catboost_") and p.name not in pre_catboost:
            shutil.rmtree(p, ignore_errors=True)


def _restore_upload_json(snapshots: dict[Path, bytes]) -> None:
    for p, content in snapshots.items():
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        except OSError:
            pass


@pytest.fixture(scope="session", autouse=True)
def session_artifact_guard():
    """Ensure the repository stays clean across the entire E2E test session."""
    rep_state = _snapshot_dir(REPORTS_DIR)
    upl_state = _snapshot_dir(UPLOADS_DIR)
    root_jsons, root_cat = _snapshot_root_artifacts(REPO_ROOT)
    upl_json_bytes = _snapshot_upload_json(UPLOADS_DIR)
    try:
        yield
    finally:
        _cleanup_new_entries(REPORTS_DIR, *rep_state)
        _cleanup_new_entries(UPLOADS_DIR, *upl_state)
        _cleanup_root_artifacts(REPO_ROOT, root_jsons, root_cat)
        _restore_upload_json(upl_json_bytes)


@pytest.fixture(scope="function", autouse=True)
def clean_artifacts():
    """Snapshot and clean up any artifacts created by an individual E2E test."""
    rep_state = _snapshot_dir(REPORTS_DIR)
    upl_state = _snapshot_dir(UPLOADS_DIR)
    root_jsons, root_cat = _snapshot_root_artifacts(REPO_ROOT)
    upl_json_bytes = _snapshot_upload_json(UPLOADS_DIR)
    try:
        yield
    finally:
        _cleanup_new_entries(REPORTS_DIR, *rep_state)
        _cleanup_new_entries(UPLOADS_DIR, *upl_state)
        _cleanup_root_artifacts(REPO_ROOT, root_jsons, root_cat)
        _restore_upload_json(upl_json_bytes)


# ---------------------------------------------------------------------------
# HTTP / Multipart / Waitress / Subprocess helpers
# ---------------------------------------------------------------------------

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_request(
    url: str,
    method: str = "GET",
    json_body: Any = None,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, str], bytes]:
    req_headers = dict(headers or {})
    body = data
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as err:
        return int(err.code), {k.lower(): v for k, v in err.headers.items()}, err.read()


def http_json(
    url: str,
    method: str = "GET",
    json_body: Any = None,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    status, _, raw = http_request(
        url, method=method, json_body=json_body, data=data, headers=headers, timeout=timeout
    )
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return status, {}
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, {"_raw": text}


def build_multipart_body(
    field_name: str,
    filename: str,
    content: bytes,
    content_type: str | None = None,
) -> tuple[bytes, str]:
    boundary = f"----ACDSOE2EBoundary{uuid.uuid4().hex}"
    ctype = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {ctype}\r\n\r\n".encode("utf-8"),
        content,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def stop_job_via_client(client: Any, job_id: str) -> tuple[int, dict[str, Any]]:
    """Stop a job via contracted /api/stop/<id>, falling back to legacy /api/job/<id>/stop."""
    resp = client.post(f"/api/stop/{job_id}")
    if resp.status_code == 404:
        resp = client.post(f"/api/job/{job_id}/stop")
    return resp.status_code, (resp.get_json(silent=True) or {})


def stop_active_jobs_via_client(client: Any, timeout: float = 10.0) -> None:
    resp = client.get("/api/jobs")
    if resp.status_code != 200:
        return
    jobs = (resp.get_json(silent=True) or {}).get("jobs") or []
    running_ids = [j["job_id"] for j in jobs if j.get("status") == "running" and j.get("job_id")]
    for jid in running_ids:
        stop_job_via_client(client, jid)
    deadline = time.monotonic() + timeout
    for jid in running_ids:
        while time.monotonic() < deadline:
            r = client.get(f"/api/job/{jid}")
            if r.status_code != 200:
                break
            data = r.get_json(silent=True) or {}
            if data.get("status") != "running":
                break
            time.sleep(0.15)


def stop_active_jobs_via_http(base_url: str, timeout: float = 10.0) -> None:
    status, data = http_json(f"{base_url}/api/jobs", timeout=5.0)
    if status != 200:
        return
    running_ids = [j["job_id"] for j in (data.get("jobs") or []) if j.get("status") == "running" and j.get("job_id")]
    for jid in running_ids:
        code, _ = http_json(f"{base_url}/api/stop/{jid}", method="POST", timeout=5.0)
        if code == 404:
            http_json(f"{base_url}/api/job/{jid}/stop", method="POST", timeout=5.0)
    deadline = time.monotonic() + timeout
    for jid in running_ids:
        while time.monotonic() < deadline:
            st, payload = http_json(f"{base_url}/api/job/{jid}", timeout=5.0)
            if st != 200 or payload.get("status") != "running":
                break
            time.sleep(0.15)


@dataclass
class LiveServerHandle:
    base_url: str
    app: Any
    port: int

    def get(self, path: str, timeout: float = 30.0) -> tuple[int, dict[str, str], bytes]:
        return http_request(f"{self.base_url}{path}", method="GET", timeout=timeout)

    def get_json(self, path: str, timeout: float = 30.0) -> tuple[int, dict[str, Any]]:
        return http_json(f"{self.base_url}{path}", method="GET", timeout=timeout)

    def post_json(self, path: str, payload: Any = None, timeout: float = 30.0) -> tuple[int, dict[str, Any]]:
        return http_json(f"{self.base_url}{path}", method="POST", json_body=payload, timeout=timeout)

    def delete_json(self, path: str, timeout: float = 30.0) -> tuple[int, dict[str, Any]]:
        return http_json(f"{self.base_url}{path}", method="DELETE", timeout=timeout)

    def upload_file(
        self,
        file_path: Path | None = None,
        *,
        filename: str | None = None,
        content: bytes | None = None,
        field_name: str = "file",
        timeout: float = 30.0,
    ) -> tuple[int, dict[str, Any]]:
        if file_path is not None:
            raw = file_path.read_bytes()
            fname = filename or file_path.name
        else:
            raw = content if content is not None else b""
            fname = filename or "upload.csv"
        body, ctype = build_multipart_body(field_name, fname, raw)
        return http_json(
            f"{self.base_url}/api/upload",
            method="POST",
            data=body,
            headers={"Content-Type": ctype},
            timeout=timeout,
        )


def _patch_webui_dirs(monkeypatch: pytest.MonkeyPatch, base: Path) -> tuple[Path, Path, Path]:
    import acdso.webui.app as webui_app

    up = base / "uploads"
    rep = base / "reports"
    log = rep / "webui_logs"
    up.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(webui_app, "UPLOAD_DIR", up)
    monkeypatch.setattr(webui_app, "REPORTS_DIR", rep)
    monkeypatch.setattr(webui_app, "LOG_DIR", log)
    for attr in ("UPLOAD_INDEX_PATH", "UPLOAD_META_PATH", "UPLOADS_META_FILE", "INDEX_FILE"):
        if hasattr(webui_app, attr):
            monkeypatch.setattr(webui_app, attr, up / "uploads_index.json")
    return up, rep, log


@pytest.fixture
def isolated_webui_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    """Isolate WebUI UPLOAD_DIR, REPORTS_DIR, and LOG_DIR into tmp_path."""
    return _patch_webui_dirs(monkeypatch, tmp_path)


@pytest.fixture
def isolated_webui_client(isolated_webui_dirs):
    """Flask test client with UPLOAD_DIR/REPORTS_DIR/LOG_DIR isolated to tmp_path."""
    from acdso.webui import create_app

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    try:
        yield app, client, isolated_webui_dirs
    finally:
        stop_active_jobs_via_client(client)


@pytest.fixture
def webui_client():
    """Flask test client wired to real REPO_ROOT dirs (required when /api/run spawns run_acdso.py)."""
    from acdso.webui import create_app

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    try:
        yield app, client
    finally:
        stop_active_jobs_via_client(client)


@pytest.fixture
def live_webui_server():
    """Real ephemeral-port Waitress WSGI server running create_app()."""
    from waitress import create_server
    from acdso.webui import create_app

    app = create_app()
    srv = create_server(app, host="127.0.0.1", port=0, threads=8)
    port = int(srv.effective_port)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    handle = LiveServerHandle(base_url=base_url, app=app, port=port)

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            st, payload = handle.get_json("/api/health", timeout=2.0)
            if st == 200 and payload.get("ok"):
                break
        except Exception:
            time.sleep(0.05)

    try:
        yield handle
    finally:
        stop_active_jobs_via_http(base_url, timeout=10.0)
        srv.close()
        thread.join(timeout=5.0)


@pytest.fixture
def run_cli(tmp_path: Path) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Run run_acdso.py in an isolated subprocess (default cwd=tmp_path)."""

    def _run(
        args: list[str],
        cwd: Path | None = None,
        timeout: float = 150.0,
        env_extra: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": str(SRC_ROOT),
            }
        )
        if env_extra:
            env.update(env_extra)
        work_dir = cwd or tmp_path
        work_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.run(
            [PYTHON_EXE, "-B", str(RUNNER), *args],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )

    return _run


def parse_progress_events(stdout: str) -> list[dict[str, Any]]:
    """Parse structured `[ACDSO_PROGRESS] {...}` lines emitted by run_acdso.py."""
    events: list[dict[str, Any]] = []
    prefix = "[ACDSO_PROGRESS]"
    for line in stdout.splitlines():
        idx = line.find(prefix)
        if idx == -1:
            continue
        raw = line[idx + len(prefix) :].strip()
        events.append(json.loads(raw))
    return events


def poll_job_until_done(
    get_job_fn: Callable[[int], tuple[int, dict[str, Any]]],
    timeout: float = 150.0,
    interval: float = 0.25,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """Poll /api/job/<id>?cursor=... until status != 'running'."""
    cursor = 0
    all_lines: list[str] = []
    progress_snapshots: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status_code, payload = get_job_fn(cursor)
        assert status_code == 200, f"Unexpected /api/job status {status_code}: {payload}"
        last_payload = payload
        lines = payload.get("lines") or []
        all_lines.extend(lines)
        next_cursor = payload.get("cursor")
        if isinstance(next_cursor, int):
            cursor = next_cursor
        elif isinstance(payload.get("next_offset"), int):
            cursor = int(payload["next_offset"])
        if isinstance(payload.get("progress"), dict):
            progress_snapshots.append(payload["progress"])
        if payload.get("status") != "running":
            return last_payload, all_lines, progress_snapshots
        time.sleep(interval)
    raise AssertionError(f"Job did not finish within {timeout}s. Last payload: {last_payload}")


# ---------------------------------------------------------------------------
# Deterministic Dataset Factory
# ---------------------------------------------------------------------------

class DatasetFactory:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path

    def unique_name(self, stem: str, ext: str = ".csv") -> str:
        ext = ext if ext.startswith(".") else f".{ext}"
        return f"e2e_{uuid.uuid4().hex[:8]}_{stem}{ext}"

    def copy_sample(self, key: str, dest_name: str | None = None) -> Path:
        src = SAMPLE_DATASETS[key]
        assert src.is_file(), f"Sample dataset missing: {src}"
        dst = self.tmp_path / (dest_name or self.unique_name(key, ".csv"))
        shutil.copy2(src, dst)
        return dst

    def make_classification_df(
        self,
        n_rows: int = 120,
        n_classes: int = 2,
        seed: int = 42,
        include_id: bool = False,
        include_nan_target: bool = False,
        string_target: bool = False,
    ) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        x1 = rng.randn(n_rows)
        x2 = rng.randn(n_rows)
        x3 = rng.randn(n_rows)
        cat = rng.choice(["alpha", "beta", "gamma"], size=n_rows)
        score = 1.8 * x1 - 1.2 * x2 + 0.3 * rng.randn(n_rows)
        if n_classes == 2:
            y_int = (score > 0).astype(int)
            target: Any = np.where(y_int == 1, "yes", "no") if string_target else y_int
        else:
            q = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
            y_int = np.digitize(score, q)
            target = np.array([f"cls_{v}" for v in y_int]) if string_target else y_int

        data: dict[str, Any] = {}
        if include_id:
            data["record_id"] = np.arange(1, n_rows + 1)
        data.update({"num_a": x1, "num_b": x2, "num_c": x3, "segment": cat, "target": target})
        df = pd.DataFrame(data)
        if include_nan_target and n_rows >= 10:
            df.loc[df.index[:3], "target"] = np.nan
        return df

    def make_regression_df(self, n_rows: int = 120, seed: int = 42, include_id: bool = False) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        x1 = rng.randn(n_rows)
        x2 = rng.randn(n_rows)
        x3 = rng.uniform(0, 10, size=n_rows)
        cat = rng.choice(["north", "south", "east"], size=n_rows)
        cat_bonus = np.where(cat == "north", 2.0, np.where(cat == "south", -1.5, 0.5))
        y = 3.5 * x1 - 2.0 * x2 + 0.5 * x3 + cat_bonus + 0.1 * rng.randn(n_rows)
        data: dict[str, Any] = {}
        if include_id:
            data["row_id"] = np.arange(1000, 1000 + n_rows)
        data.update({"f1": x1, "f2": x2, "f3": x3, "region": cat, "y": y})
        return pd.DataFrame(data)

    def make_leaky_df(self, n_rows: int = 200, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.RandomState(seed)
        X = pd.DataFrame({f"feat_{i}": rng.randn(n_rows) for i in range(6)})
        y = (0.8 * X["feat_0"] - 0.6 * X["feat_1"] + 0.3 * rng.randn(n_rows) > 0).astype(int).to_numpy()
        X["leaky_feature"] = y.astype(float) + 0.01 * rng.randn(n_rows)
        return X, y

    def make_multicollinear_df(self, n_rows: int = 200, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.RandomState(seed)
        x1 = rng.randn(n_rows)
        x2 = x1 + 0.005 * rng.randn(n_rows)
        x3 = 2.0 * x1 - x2 + 0.005 * rng.randn(n_rows)
        x4 = rng.randn(n_rows)
        y = 2.5 * x1 + 0.8 * x4 + 0.1 * rng.randn(n_rows)
        return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4}), y

    def make_timeseries_df(self, n_periods: int = 18, freq: str = "MS", seed: int = 42) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        dates = pd.date_range("2024-01-01", periods=n_periods, freq=freq)
        trend = np.linspace(100.0, 220.0, n_periods)
        seasonal = 20.0 * np.sin(np.arange(n_periods) * (2 * np.pi / 6))
        values = trend + seasonal + rng.normal(0, 2.5, size=n_periods)
        return pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "metric": np.round(values, 2),
                "category": ["core"] * n_periods,
            }
        )

    def make_multi_target_df(self, n_rows: int = 120, seed: int = 42) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        x1 = rng.randn(n_rows)
        x2 = rng.randn(n_rows)
        x3 = rng.randn(n_rows)
        t1 = 2.0 * x1 + 1.5 * x2 + 0.1 * rng.randn(n_rows)
        t2 = -1.5 * x1 + 3.0 * x3 + 0.1 * rng.randn(n_rows)
        return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "t1": t1, "t2": t2})

    def write_csv(self, df: pd.DataFrame, filename: str | None = None) -> Path:
        p = self.tmp_path / (filename or self.unique_name("data", ".csv"))
        df.to_csv(p, index=False)
        return p

    def write_json(self, df: pd.DataFrame, filename: str | None = None) -> Path:
        p = self.tmp_path / (filename or self.unique_name("data", ".json"))
        df.to_json(p, orient="records")
        return p

    def write_pickle(self, df: pd.DataFrame, filename: str | None = None) -> Path:
        p = self.tmp_path / (filename or self.unique_name("data", ".pkl"))
        df.to_pickle(p)
        return p

    def write_xlsx(self, df: pd.DataFrame, filename: str | None = None) -> Path:
        pytest.importorskip("openpyxl")
        p = self.tmp_path / (filename or self.unique_name("data", ".xlsx"))
        df.to_excel(p, index=False)
        return p

    def write_parquet(self, df: pd.DataFrame, filename: str | None = None) -> Path:
        pytest.importorskip("pyarrow")
        p = self.tmp_path / (filename or self.unique_name("data", ".parquet"))
        df.to_parquet(p, index=False)
        return p


@pytest.fixture
def dataset_factory(tmp_path: Path) -> DatasetFactory:
    return DatasetFactory(tmp_path)
