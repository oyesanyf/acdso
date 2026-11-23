"""Flask application backing the ACDSO web GUI.

Design notes
------------
The pipeline itself lives in ``run_acdso.py`` (~11k lines, argparse driven).
Rather than re-import and re-wire its internals, this GUI builds the exact
same command line and runs it as a subprocess, streaming stdout back to the
browser.  That keeps the GUI honest: every flag exposed here means the same
thing it means on the command line, and there is no second code path to keep
in sync.

One run at a time.  ACDSO writes artifacts to ``reports/`` under
deterministic, dataset-derived names (``reports/predictions_<base>_<target>
_full_dataset.csv`` and friends), so two concurrent runs on the same file
would silently clobber each other's output.  The manager refuses to start a
second job while one is live.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from werkzeug.utils import secure_filename

# src/acdso/webui/app.py -> src/acdso/webui -> src/acdso -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "run_acdso.py"
UPLOAD_DIR = REPO_ROOT / "uploads"
REPORTS_DIR = REPO_ROOT / "reports"
LOG_DIR = REPORTS_DIR / "webui_logs"

#: Formats ``run_acdso.py`` documents as loadable.
ALLOWED_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".feather", ".pkl"}

MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_LOG_LINES = 40_000
MAX_PERSISTED_UPLOADS = 50

_UPLOAD_INDEX_LOCK = threading.Lock()
_REPORT_CACHE_LOCK = threading.Lock()
_REPORT_CACHE: dict[str, Any] = {"sig": None, "items": []}

_UPLOAD_FILE_RE = re.compile(r"^(?P<stem>.+)_(?P<id>[0-9a-f]{12})$", re.IGNORECASE)
_PROGRESS_RE = re.compile(r"^\[ACDSO_PROGRESS\]\s*(\{.*\})\s*$")


# ---------------------------------------------------------------------------
# Flag specification -- single source of truth for the form and for argv
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Flag:
    """One CLI flag, rendered into the form and back out into argv."""

    name: str  # form field name
    cli: str  # flag as passed to run_acdso.py
    kind: str  # bool | text | int | float | choice | columns
    label: str
    help: str = ""
    default: Any = None
    choices: tuple[str, ...] = ()
    group: str = "General"
    dead: bool = False
    tier: str = "advanced"  # core | mode | advanced | dead
    depends_on: str = ""


FLAGS: tuple[Flag, ...] = (
    # --- What to predict -------------------------------------------------
    Flag(
        "predict", "--predict", "columns", "Predict target(s)",
        "Train fresh model(s) and write predictions for the whole dataset. "
        "This is the main path: pick one or more columns.",
        group="Target", tier="core",
    ),
    Flag(
        "target", "--target", "columns", "Train-only target(s)",
        "Train without the prediction step. Leave empty when using "
        "'Predict target(s)' above -- setting both is redundant.",
        group="Target", tier="core",
    ),
    Flag(
        "multi_output", "--multi-output", "bool", "Multi-output model",
        "One model predicting all targets at once instead of one model per target.",
        default=False, group="Target", tier="advanced",
    ),
    Flag(
        "task", "--task", "choice", "Task type",
        "Leave on 'auto' to let ACDSO infer classification vs regression.",
        default="auto", choices=("auto", "classification", "regression"),
        group="Target", tier="core",
    ),
    # --- Columns ---------------------------------------------------------
    Flag(
        "exclude", "--exclude", "columns", "Exclude columns",
        "Columns to drop before training.",
        group="Columns", tier="core",
    ),
    Flag(
        "no_auto_exclude", "--no-auto-exclude", "bool", "Disable auto-exclusion",
        "Turn off automatic leakage / ID / constant-column detection.",
        default=False, group="Columns", tier="advanced",
    ),
    # --- Pipeline phases -------------------------------------------------
    Flag(
        "autonomous", "--autonomous", "bool", "Autonomous pipeline",
        "Full pipeline: EDA -> Risk -> AutoML -> Time Series. Always on -- "
        "run_acdso.py uses this as the default/fallback mode whenever no "
        "mode-switching flag is checked.",
        default=True, group="Pipeline", dead=True, tier="dead",
    ),
    Flag(
        "simple", "--simple", "bool", "Quick mode",
        "EDA + AutoML only. Skips Time Series and benchmarks.",
        default=False, group="Pipeline", tier="core",
    ),
    Flag(
        "benchmark", "--benchmark", "bool", "Run benchmarks",
        "Compare against FLAML, TPOT, H2O, AutoGluon. Slow -- 10-30 min per target.",
        default=False, group="Pipeline", tier="core",
    ),
    Flag(
        "skip_benchmark", "--skip-benchmark", "bool", "Skip benchmark phase",
        "Skip benchmarking inside autonomous mode. Recommended for multi-target runs.",
        default=True, group="Pipeline", tier="advanced",
    ),
    Flag(
        "timeseries", "--timeseries", "bool", "Time series mode",
        "SWITCHES modes rather than adding to the pipeline: runs ONLY the "
        "Time Series demo instead of the full EDA/AutoML/TimeSeries pipeline.",
        default=False, group="Pipeline", tier="mode",
    ),
    Flag(
        "decision", "--decision", "bool", "Decision intelligence",
        "SWITCHES modes: runs ONLY causal analysis, uplift modelling and "
        "action optimization on --file (or synthetic data if no file).",
        default=False, group="Pipeline", tier="mode",
    ),
    Flag(
        "policy", "--policy", "bool", "Policy learning (RLlib)",
        "SWITCHES modes: runs ONLY the reinforcement-learning policy demo "
        "on synthetic data, instead of the full pipeline.",
        default=False, group="Pipeline", tier="mode",
    ),
    Flag(
        "v3", "--v3", "bool", "v3 innovations",
        "SWITCHES modes: runs ONLY the stress-testing / successive-halving "
        "/ risk-calibration demo on built-in data, instead of the full pipeline.",
        default=False, group="Pipeline", tier="mode",
    ),
    Flag(
        "all", "--all", "bool", "Run all built-in demos",
        "SWITCHES modes: ignores any uploaded file and runs all 25 bundled "
        "demo datasets instead.",
        default=False, group="Pipeline", tier="mode",
    ),
    # --- Time series -----------------------------------------------------
    Flag(
        "datetime_col", "--datetime-col", "columns", "Datetime column",
        "Timestamp column for time-series work.",
        group="Time series", tier="core",
    ),
    Flag(
        "horizon", "--horizon", "int", "Forecast horizon",
        "Steps to forecast ahead.",
        default=7, group="Time series", tier="advanced",
    ),
    # --- Decision intelligence -------------------------------------------
    Flag(
        "treatment", "--treatment", "columns", "Treatment column",
        "The intervention column, e.g. 'offered_incentive'. Only used when "
        "'Decision intelligence' is active.",
        group="Decision", tier="advanced", depends_on="decision",
    ),
    Flag(
        "budget", "--budget", "float", "Budget constraint",
        "Budget ceiling for action optimization. Only used when "
        "'Decision intelligence' is active.",
        group="Decision", tier="advanced", depends_on="decision",
    ),
    # --- Model selection / runtime ---------------------------------------
    Flag(
        "best_score", "--best-score", "bool", "Select by best CV score",
        "Pick the highest-scoring model instead of the multi-objective "
        "knee-point trade-off (speed / memory / risk).",
        default=False, group="Runtime", tier="advanced",
    ),
    Flag(
        "time_budget", "--time-budget", "int", "Time budget (seconds)",
        "Per-system budget for no-file benchmark comparison (built-in datasets).",
        default=21600, group="Runtime", tier="advanced", depends_on="benchmark_no_file",
    ),
    Flag(
        "force", "--force", "bool", "Fresh run (--force)",
        "Delete existing models and reports for this dataset before training.",
        default=False, group="Runtime", tier="advanced",
    ),
    Flag(
        "datasets", "--datasets", "text", "Benchmark datasets",
        "Comma-separated built-in dataset names, or 'all' (no-file benchmark).",
        default=None, group="Runtime", tier="advanced", depends_on="benchmark_no_file",
    ),
    Flag(
        "output", "--output", "text", "Benchmark output JSON",
        "Optional path for no-file benchmark results JSON.",
        group="Runtime", tier="advanced", depends_on="benchmark_no_file",
    ),
)

FLAGS_BY_NAME = {f.name: f for f in FLAGS}


# ---------------------------------------------------------------------------
# Uploads & Disk Persistence
# ---------------------------------------------------------------------------


@dataclass
class Upload:
    id: str
    path: Path
    original_name: str
    columns: list[str]
    rows: int | None
    preview: list[dict[str, Any]]
    size_bytes: int = 0
    uploaded_at: float = 0.0
    dtypes: dict[str, str] = field(default_factory=dict)
    column_stats: list[dict[str, Any]] = field(default_factory=list)
    predictable_columns: list[str] = field(default_factory=list)
    excluded_columns: dict[str, dict[str, str]] = field(default_factory=dict)
    needs_inspect: bool = False
    suggest_target_cache: dict[str, Any] | None = None
    horizon_cache: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def to_summary_dict(self, include_preview: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "upload_id": self.id,
            "filename": self.original_name,
            "stored_as": self.path.name,
            "size_bytes": self.size_bytes,
            "uploaded_at": self.uploaded_at,
            "columns": self.columns,
            "rows": self.rows,
            "dtypes": self.dtypes,
            "column_stats": self.column_stats,
            "predictable_columns": self.predictable_columns,
            "excluded_columns": self.excluded_columns,
            "needs_inspect": self.needs_inspect,
        }
        if include_preview:
            payload["preview"] = self.preview
        return payload


def _read_frame(path: Path, nrows: int | None = 50, usecols: list[str] | None = None):
    """Load an uploaded file with optional row limit and column projection."""
    import pandas as pd

    def limit(frame):
        if usecols:
            keep = [c for c in usecols if c in frame.columns]
            if keep:
                frame = frame[keep]
        return frame if nrows is None else frame.head(nrows)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        last_err: Exception | None = None
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(path, nrows=nrows, usecols=usecols, encoding=enc)
            except UnicodeDecodeError as exc:
                last_err = exc
        if last_err is not None:
            raise last_err
        return pd.read_csv(path, nrows=nrows, usecols=usecols)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows, usecols=usecols)
    if suffix == ".parquet":
        if nrows is not None:
            try:
                import pyarrow.parquet as pq

                pf = pq.ParquetFile(path)
                first = next(pf.iter_batches(batch_size=max(1, nrows), columns=usecols), None)
                if first is None:
                    return pd.DataFrame(columns=usecols or [])
                return first.to_pandas().head(nrows)
            except Exception:
                pass
        return limit(pd.read_parquet(path, columns=usecols))
    if suffix == ".json":
        return limit(pd.read_json(path))
    if suffix == ".feather":
        return limit(pd.read_feather(path, columns=usecols))
    if suffix == ".pkl":
        return limit(pd.read_pickle(path))
    raise ValueError(f"Unsupported file type: {suffix}")


def _sanitize_json_value(val: Any) -> Any:
    """Recursively convert values into RFC 8259-compliant JSON primitives (non-finite -> None)."""
    if val is None or isinstance(val, str):
        return val
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return val if math.isfinite(val) else None
    if isinstance(val, dict):
        return {str(k): _sanitize_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_json_value(v) for v in val]
    if type(val).__module__.startswith("numpy") and hasattr(val, "item"):
        try:
            return _sanitize_json_value(val.item())
        except Exception:
            pass
    try:
        if val != val:  # catches float NaN / pandas NA-like scalars without eager imports
            return None
    except Exception:
        pass
    return val


def _json_safe_preview(frame, limit: int = 10) -> list[dict[str, Any]]:
    """First rows as JSON-serialisable dicts, NaN and +/-Inf collapsed to null."""
    head = frame.head(limit)
    records = head.astype(object).where(head.notna(), None).to_dict("records")
    return _sanitize_json_value(records)


def _row_count(path: Path) -> int | None:
    """Fast row count for CSV (1 MiB binary chunk scan) and Parquet metadata."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            return None
    if suffix != ".csv":
        return None
    try:
        newlines = 0
        last_byte = b""
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                newlines += chunk.count(b"\n")
                last_byte = chunk[-1:]
        if not last_byte:
            return 0
        total_lines = newlines + (0 if last_byte == b"\n" else 1)
        return max(total_lines - 1, 0)
    except OSError:
        return None


def _compute_column_stats(
    sample,
    pred_cols: list[str],
    excluded_reasons: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Compute per-column stats and role hints from the already-loaded sample."""
    import pandas as pd

    from .column_validator import _is_datetime_like

    pred_set = set(pred_cols)
    n_rows = len(sample)
    stats: list[dict[str, Any]] = []
    for col in sample.columns:
        name = str(col)
        s = sample[col]
        null_count = int(s.isna().sum())
        null_pct = round((null_count / n_rows) * 100.0, 1) if n_rows > 0 else 0.0
        nunique = int(s.nunique(dropna=True))
        excl = excluded_reasons.get(name)
        reason_code = excl["reason"] if excl else None
        if reason_code == "datetime" or _is_datetime_like(s):
            role_hint = "datetime"
        elif reason_code in ("all_null", "constant"):
            role_hint = "constant"
        elif reason_code in ("id_like", "unique_non_float"):
            role_hint = "id"
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            role_hint = "numeric"
        else:
            role_hint = "categorical"
        stats.append(
            {
                "name": name,
                "dtype": str(s.dtype),
                "nunique": nunique,
                "null_count": null_count,
                "null_pct": null_pct,
                "role_hint": role_hint,
                "predictable": name in pred_set,
                "exclude_reason": excl["detail"] if excl else None,
            }
        )
    return stats


def _inspect_upload_record(record: Upload) -> Upload:
    """Populate columns, stats, preview, and row count for an Upload record."""
    from .column_validator import analyze_predictable_columns

    sample = _read_frame(record.path, nrows=5000)
    if sample.columns.has_duplicates:
        raise ValueError("Duplicate column names are not supported in uploaded datasets.")
    suffix = record.path.suffix.lower()
    if suffix in {".csv", ".xlsx", ".xls"} and len(sample) < 5000:
        rows: int | None = int(len(sample))
    else:
        rows = _row_count(record.path)
        if rows is None:
            rows = int(len(sample)) if len(sample) < 5000 else None

    pred_cols, excluded = analyze_predictable_columns(sample)
    col_stats = _compute_column_stats(sample, pred_cols, excluded)

    record.columns = [str(c) for c in sample.columns]
    record.rows = rows
    record.preview = _json_safe_preview(sample, limit=10)
    record.dtypes = {str(c): str(t) for c, t in sample.dtypes.items()}
    record.predictable_columns = pred_cols
    record.excluded_columns = excluded
    record.column_stats = col_stats
    record.needs_inspect = False
    try:
        record.size_bytes = int(record.path.stat().st_size)
    except OSError:
        pass
    return record


def _load_upload_index(upload_dir: Path) -> dict[str, Upload]:
    """Load persisted uploads from ``.uploads_index.json`` without eager pandas reads."""
    records: dict[str, Upload] = {}
    if not upload_dir.is_dir():
        return records

    index_path = upload_dir / ".uploads_index.json"
    if index_path.is_file():
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else raw.get("uploads", [])
            for item in items:
                if not isinstance(item, dict):
                    continue
                uid = str(item.get("upload_id") or "").strip()
                stored_as = str(item.get("stored_as") or "").strip()
                if not uid or not stored_as:
                    continue
                path = upload_dir / stored_as
                if not path.is_file():
                    continue
                records[uid] = Upload(
                    id=uid,
                    path=path,
                    original_name=str(item.get("filename") or stored_as),
                    columns=[str(c) for c in (item.get("columns") or [])],
                    rows=item.get("rows"),
                    preview=list(item.get("preview") or []),
                    size_bytes=int(item.get("size_bytes") or path.stat().st_size),
                    uploaded_at=float(item.get("uploaded_at") or path.stat().st_mtime),
                    dtypes=dict(item.get("dtypes") or {}),
                    column_stats=list(item.get("column_stats") or []),
                    predictable_columns=list(item.get("predictable_columns") or []),
                    excluded_columns=dict(item.get("excluded_columns") or {}),
                    needs_inspect=bool(item.get("needs_inspect", not item.get("columns"))),
                )
        except Exception:
            pass

    # Discover existing upload files not yet in the index without parsing huge files.
    known_names = {u.path.name for u in records.values()}
    try:
        with os.scandir(upload_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                name = entry.name
                if name.startswith(".") or name in known_names:
                    continue
                p = Path(entry.path)
                if p.suffix.lower() not in ALLOWED_SUFFIXES:
                    continue
                m = _UPLOAD_FILE_RE.match(p.stem)
                if not m:
                    continue
                uid = m.group("id").lower()
                if uid in records:
                    continue
                try:
                    st = entry.stat()
                except OSError:
                    continue
                records[uid] = Upload(
                    id=uid,
                    path=p,
                    original_name=f"{m.group('stem')}{p.suffix}",
                    columns=[],
                    rows=None,
                    preview=[],
                    size_bytes=int(st.st_size),
                    uploaded_at=float(st.st_mtime),
                    needs_inspect=True,
                )
    except OSError:
        pass

    return records


def _save_upload_index(upload_dir: Path, uploads: dict[str, Upload] | list[Upload]) -> None:
    """Persist up to ``MAX_PERSISTED_UPLOADS`` upload metadata records atomically."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    index_path = upload_dir / ".uploads_index.json"
    tmp_path = upload_dir / f".uploads_index.{uuid.uuid4().hex[:8]}.tmp"
    with _UPLOAD_INDEX_LOCK:
        values = list(uploads.values()) if isinstance(uploads, dict) else list(uploads)
        ordered = sorted(values, key=lambda u: u.uploaded_at, reverse=True)[:MAX_PERSISTED_UPLOADS]
        payload = [u.to_summary_dict(include_preview=True) for u in ordered]
        try:
            tmp_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
            os.replace(tmp_path, index_path)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def _parse_nonneg_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    if isinstance(val, bool):
        raise ValueError("boolean is not a valid phase integer")
    if isinstance(val, int):
        return max(0, val)
    if isinstance(val, float):
        if not math.isfinite(val):
            raise ValueError("non-finite float is not a valid phase integer")
        return max(0, int(val))
    if isinstance(val, str):
        text = val.strip()
        if not text:
            return default
        try:
            return max(0, int(text))
        except ValueError:
            num = float(text)
            if not math.isfinite(num):
                raise ValueError("non-finite float is not a valid phase integer")
            return max(0, int(num))
    raise TypeError(f"unsupported numeric type: {type(val).__name__}")


def _parse_progress_line(line: str) -> dict[str, Any] | None:
    m = _PROGRESS_RE.match(line.strip())
    if not m:
        return None
    try:
        raw = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    try:
        phase = _parse_nonneg_int(raw.get("phase", 0), default=0)
        total = _parse_nonneg_int(raw.get("total", 0), default=0)
    except (TypeError, ValueError, OverflowError):
        return None
    name = str(raw.get("name") or "").strip()
    status = str(raw.get("status") or "running").strip().lower()
    if status not in ("running", "done", "error", "skipped"):
        status = "running"
    detail = str(raw.get("detail") or "").strip()
    percent = min(100, max(0, round((phase / total) * 100))) if total > 0 else 0
    return {
        "phase": phase,
        "total": total,
        "name": name,
        "status": status,
        "percent": percent,
        "detail": detail,
    }


def _format_progress_line(prog: dict[str, Any]) -> str:
    prefix = f"[phase {prog['phase']}/{prog['total']}]" if prog["total"] > 0 else "[phase]"
    name = prog["name"] or "Pipeline"
    status = prog["status"]
    detail = f" - {prog['detail']}" if prog["detail"] else ""
    return f"{prefix} {name} ({status}){detail}"


@dataclass
class Job:
    id: str
    argv: list[str]
    started_at: float
    log_path: Path
    lines: list[str] = field(default_factory=list)
    status: str = "running"  # queued | running | finished | failed | cancelled
    returncode: int | None = None
    dropped_lines: int = 0
    proc: subprocess.Popen | None = None
    progress: dict[str, Any] | None = None
    queued_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cond: threading.Condition = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cond = threading.Condition(self._lock)

    @property
    def command(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)

    def notify(self) -> None:
        with self._lock:
            self._cond.notify_all()

    def append(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)
            if len(self.lines) > MAX_LOG_LINES:
                overflow = len(self.lines) - MAX_LOG_LINES
                del self.lines[:overflow]
                self.dropped_lines += overflow
            self._cond.notify_all()

    def set_progress(self, progress: dict[str, Any]) -> None:
        with self._lock:
            self.progress = dict(progress)
            self._cond.notify_all()

    def get_progress(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self.progress) if self.progress is not None else None

    def slice_from(self, cursor: int) -> tuple[list[str], int]:
        """Log lines after ``cursor``, in absolute line numbering."""
        with self._lock:
            base = self.dropped_lines
            start = max(cursor - base, 0)
            return list(self.lines[start:]), base + len(self.lines)


class JobManager:
    """Runs one ACDSO job at a time with an opt-in FIFO queue and SSE notifications."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._active: str | None = None
        self._queue: list[Job] = []

    @property
    def active_job(self) -> Job | None:
        with self._lock:
            return self._jobs.get(self._active) if self._active else None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda j: j.queued_at if j.queued_at is not None else j.started_at,
                reverse=True,
            )
        return jobs[:limit]

    def list_queue(self) -> list[dict[str, Any]]:
        with self._lock:
            items = []
            for pos, j in enumerate(self._queue, start=1):
                items.append(
                    {
                        "job_id": j.id,
                        "status": j.status,
                        "queue_position": pos,
                        "queued_at": j.queued_at,
                        "command": j.command,
                    }
                )
            return items

    def queue_position(self, job_id: str) -> int:
        with self._lock:
            for pos, j in enumerate(self._queue, start=1):
                if j.id == job_id:
                    return pos
            return 0

    def cancel_queued(self, job_id: str) -> bool:
        target: Job | None = None
        with self._lock:
            for idx, j in enumerate(self._queue):
                if j.id == job_id:
                    target = self._queue.pop(idx)
                    break
        if target is None:
            return False
        with target._lock:
            target.status = "cancelled"
            target.returncode = -1
            target._cond.notify_all()
        target.append("[webui] cancelled before execution (removed from queue)")
        return True

    def start(self, argv: list[str]) -> Job:
        with self._lock:
            active = self._jobs.get(self._active) if self._active else None
            if active is not None and active.status == "running":
                raise RuntimeError(
                    "A run is already in progress. Stop it, wait for it to finish, "
                    "or pass queue=true to enqueue this run."
                )
            job = self._create_job_locked(argv, status="running")
            self._active = job.id

        self._launch_job(job)
        return job

    def enqueue(self, argv: list[str]) -> tuple[Job, int]:
        """Start immediately if idle; otherwise append to FIFO queue and return (job, position)."""
        with self._lock:
            active = self._jobs.get(self._active) if self._active else None
            if active is None or active.status != "running":
                job = self._create_job_locked(argv, status="running")
                self._active = job.id
                should_launch = True
                pos = 0
            else:
                job = self._create_job_locked(argv, status="queued")
                self._queue.append(job)
                should_launch = False
                pos = len(self._queue)

        if should_launch:
            self._launch_job(job)
            return job, 0

        job.append(f"[webui] queued at position #{pos}; waiting for active run to finish...")
        return job, pos

    def _create_job_locked(self, argv: list[str], status: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        now = time.time()
        job = Job(
            id=job_id,
            argv=argv,
            started_at=now,
            log_path=LOG_DIR / f"run_{job_id}.log",
            status=status,
            queued_at=now if status == "queued" else None,
        )
        self._jobs[job_id] = job
        return job

    def _launch_job(self, job: Job) -> None:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0

        with self._lock:
            with job._lock:
                if job.status == "cancelled":
                    if self._active == job.id:
                        self._active = None
                    job._cond.notify_all()
                    return
                job.status = "running"
                job.started_at = time.time()
                job._cond.notify_all()

        try:
            proc = subprocess.Popen(
                job.argv,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
        except Exception:
            with self._lock:
                with job._lock:
                    job.status = "failed"
                    job.returncode = -1
                    job._cond.notify_all()
                if self._active == job.id:
                    self._active = None
            self._promote_next()
            raise

        cancelled_early = False
        with job._lock:
            if job.status == "cancelled":
                cancelled_early = True
            job.proc = proc
            job._cond.notify_all()

        if cancelled_early:
            self._terminate_proc(proc)
            with self._lock:
                if self._active == job.id:
                    self._active = None
            self._promote_next()
            return

        job.append(f"$ {job.command}")
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()

    def _promote_next(self) -> None:
        next_job: Job | None = None
        with self._lock:
            active = self._jobs.get(self._active) if self._active else None
            if active is not None and active.status == "running":
                return
            while self._queue:
                cand = self._queue.pop(0)
                with cand._lock:
                    if cand.status == "queued":
                        cand.status = "running"
                        cand.started_at = time.time()
                        cand._cond.notify_all()
                        next_job = cand
                        self._active = cand.id
                        break
            if next_job is None:
                self._active = None

        if next_job is not None:
            try:
                self._launch_job(next_job)
            except Exception as exc:  # noqa: BLE001
                next_job.append(f"[webui] failed to launch queued job: {exc}")

    @staticmethod
    def _terminate_proc(proc: subprocess.Popen[str]) -> None:
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except Exception:
                pass
        else:
            try:
                proc.terminate()
            except OSError:
                pass
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=3)
        except Exception:
            pass

    def _pump(self, job: Job) -> None:
        """Drain the child's stdout into the job log until it exits."""
        proc = job.proc
        assert proc is not None and proc.stdout is not None
        try:
            with job.log_path.open("w", encoding="utf-8", errors="replace") as log:
                log.write(f"$ {job.command}\n")
                for raw in proc.stdout:
                    line = raw.rstrip("\r\n")
                    try:
                        prog = _parse_progress_line(line)
                    except Exception:  # noqa: BLE001 - never kill pump on malformed marker
                        prog = None
                    if prog is not None:
                        job.set_progress(prog)
                        display_line = _format_progress_line(prog)
                    else:
                        display_line = line
                    job.append(display_line)
                    log.write(display_line + "\n")
                    log.flush()
        finally:
            rc = proc.wait()
            with job._lock:
                job.returncode = rc
                if job.status != "cancelled":
                    job.status = "finished" if rc == 0 else "failed"
                if job.progress is not None:
                    updated = dict(job.progress)
                    if job.status == "finished":
                        updated["status"] = "done"
                        updated["percent"] = 100
                    elif job.status == "failed":
                        updated["status"] = "error"
                    job.progress = updated
                job._cond.notify_all()
            _invalidate_reports_cache()
            job.append(
                f"[webui] process exited with code {rc} "
                f"after {time.time() - job.started_at:.1f}s"
            )
            self._promote_next()

    def stop(self, job: Job) -> bool:
        with self._lock:
            with job._lock:
                if job.status == "queued":
                    try:
                        self._queue.remove(job)
                    except ValueError:
                        pass
                    job.status = "cancelled"
                    job.returncode = -1
                    job._cond.notify_all()
                    return True
                if job.status != "running":
                    return False
                proc = job.proc
                job.status = "cancelled"
                if job.progress is not None:
                    updated = dict(job.progress)
                    updated["status"] = "error"
                    updated["detail"] = "Cancelled by user"
                    job.progress = updated
                job._cond.notify_all()
                if proc is None:
                    if self._active == job.id:
                        self._active = None
                    promote = True
                else:
                    promote = False
                    if proc.poll() is not None:
                        return False

        if promote:
            self._promote_next()
            return True
        self._terminate_proc(proc)
        return True


# ---------------------------------------------------------------------------
# argv construction
# ---------------------------------------------------------------------------


def _as_list(value: Any, field_name: str = "field") -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            if v is None:
                continue
            if isinstance(v, (dict, list, tuple, set)):
                raise ValueError(f"{field_name} entries must be column names")
            text = str(v).strip()
            if text:
                out.append(text)
        return out
    raise ValueError(f"{field_name} must be a list of column names or comma-separated string")


def _json_dict_payload() -> tuple[dict[str, Any] | None, tuple[Any, int] | None]:
    """Parse request body as a JSON object (dict), returning a 400 JSON error otherwise."""
    raw_body = request.get_data(cache=True)
    if not raw_body or not raw_body.strip():
        return {}, None
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:  # noqa: BLE001
        return None, (jsonify(error="Request body must be valid JSON."), 400)
    if not isinstance(data, dict):
        return None, (jsonify(error="Request body must be a JSON object."), 400)
    return data, None


def build_argv(payload: dict[str, Any], data_file: Path | None) -> list[str]:
    """Translate a form payload into a ``run_acdso.py`` command line."""
    argv = [sys.executable, "-u", str(RUNNER)]

    if data_file is not None:
        argv += ["--file", str(data_file)]

    for flag in FLAGS:
        value = payload.get(flag.name)

        if flag.kind == "bool":
            if bool(value):
                argv.append(flag.cli)
            continue

        if flag.kind == "columns":
            joined = ",".join(_as_list(value, flag.name))
            if joined:
                argv += [flag.cli, joined]
            continue

        if value in (None, "", []):
            continue

        if flag.kind == "int":
            if isinstance(value, bool):
                raise ValueError(f"{flag.label}: expected an integer, got boolean")
            try:
                num = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{flag.label}: expected an integer, got {value!r}") from exc
            if not math.isfinite(num) or int(num) != num:
                raise ValueError(f"{flag.label}: expected an integer, got {value!r}")
            argv += [flag.cli, str(int(num))]
        elif flag.kind == "float":
            if isinstance(value, bool):
                raise ValueError(f"{flag.label}: expected a number, got boolean")
            num = float(value)
            if not math.isfinite(num):
                raise ValueError(f"{flag.label}: must be a finite number")
            argv += [flag.cli, str(num)]
        elif flag.kind == "choice":
            text = str(value)
            if text not in flag.choices:
                raise ValueError(f"{flag.label}: {text!r} is not one of {flag.choices}")
            if text != flag.default:
                argv += [flag.cli, text]
        else:
            text = str(value)
            if flag.name == "output":
                if "\x00" in text:
                    raise ValueError("Output path contains invalid NUL byte")
                base = REPORTS_DIR.resolve()
                cand = Path(text)
                resolved = cand.resolve() if cand.is_absolute() else (base / cand).resolve()
                if resolved == base or base not in resolved.parents or "webui_logs" in resolved.relative_to(base).parts:
                    raise ValueError("Output path must stay inside the reports/ directory")
                text = str(resolved)
            argv += [flag.cli, text]

    return argv


# ---------------------------------------------------------------------------
# Artifacts (scandir + snapshot-keyed cache)
# ---------------------------------------------------------------------------


_RUN_KEY_RE = re.compile(r"(?:^|_)([0-9a-f]{12}|[0-9a-f]{6})(?:_|\.)")


def _run_key(name: str) -> str | None:
    """Pull the run-hash token out of a report filename, if it has one."""
    m = _RUN_KEY_RE.search(name)
    return m.group(1) if m else None


def _invalidate_reports_cache() -> None:
    with _REPORT_CACHE_LOCK:
        _REPORT_CACHE["sig"] = None
        _REPORT_CACHE["items"] = []


def _all_report_files() -> list[dict[str, Any]]:
    """Every file under ``reports/`` (excluding webui job logs), newest first."""
    if not REPORTS_DIR.is_dir():
        return []

    base_resolved = REPORTS_DIR.resolve()
    log_dir_resolved = LOG_DIR.resolve()

    raw_entries: list[tuple[str, str, int, int, float]] = []
    stack: list[Path] = [REPORTS_DIR]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    name = entry.name
                    if entry.is_dir(follow_symlinks=False):
                        if name == "webui_logs":
                            continue
                        try:
                            if Path(entry.path).resolve() == log_dir_resolved:
                                continue
                        except OSError:
                            continue
                        stack.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if name.lower().endswith(".log"):
                        continue
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    p = Path(entry.path)
                    try:
                        rel = p.resolve().relative_to(base_resolved).as_posix()
                    except Exception:
                        rel = p.relative_to(REPORTS_DIR).as_posix()
                    raw_entries.append((rel, name, int(st.st_size), int(st.st_mtime_ns), float(st.st_mtime)))
        except OSError:
            continue

    sig = tuple(sorted((r[0], r[2], r[3]) for r in raw_entries))
    with _REPORT_CACHE_LOCK:
        if _REPORT_CACHE["sig"] == sig:
            return [dict(item) for item in _REPORT_CACHE["items"]]

    found: list[dict[str, Any]] = []
    for rel, name, size, _mtime_ns, mtime in raw_entries:
        found.append(
            {
                "path": rel,
                "name": name,
                "size": size,
                "modified": mtime,
                "kind": Path(name).suffix.lstrip(".").lower(),
                "run_key": _run_key(name),
            }
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for f in found:
        key = f["run_key"] or f["path"]
        groups.setdefault(key, []).append(f)
    for members in groups.values():
        start = min(m["modified"] for m in members)
        end = max(m["modified"] for m in members)
        for m in members:
            m["run_start"] = start
            m["run_end"] = end
            m["run_seconds"] = round(end - start, 1)

    found.sort(key=lambda a: a["modified"], reverse=True)
    with _REPORT_CACHE_LOCK:
        _REPORT_CACHE["sig"] = sig
        _REPORT_CACHE["items"] = [dict(item) for item in found]
    return found


def artifacts_since(started_at: float) -> list[dict[str, Any]]:
    """Files under ``reports/`` touched by a run, newest first."""
    cutoff = started_at - 2  # tolerate filesystem timestamp granularity
    return [f for f in _all_report_files() if f["modified"] >= cutoff]


def _write_forecast_csv(result: dict[str, Any], upload_name: str) -> str:
    """Write history+forecast to a uniquely time-stamped CSV in ``reports/``."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    base = Path(secure_filename(upload_name)).stem or "data"
    safe_col = re.sub(r"[^A-Za-z0-9]+", "_", result["value_col"]).strip("_") or "value"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"forecast_{base}_{safe_col}_{stamp}_{uuid.uuid4().hex[:6]}.csv"
    path = REPORTS_DIR / name

    is_bytes = bool(re.search(r"\(B\)|bytes?\b", result["value_col"], re.IGNORECASE))
    header = ["period", result["value_col"]]
    if is_bytes:
        header.append(re.sub(r"\(B\)", "(GB)", result["value_col"], flags=re.IGNORECASE))
    header.append("type")

    def row_values(row: dict[str, Any], kind: str) -> list[Any]:
        values = [row["period"], row["value"]]
        if is_bytes:
            values.append(round(row["value"] / 1e9, 3))
        values.append(kind)
        return values

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in result["history"]:
            writer.writerow(row_values(row, "actual"))
        for row in result["forecast"]:
            writer.writerow(row_values(row, "forecast"))

    _invalidate_reports_cache()
    return name


def _resolve_report_target(rel: str) -> tuple[Path | None, tuple[str, int] | None]:
    """Resolve ``rel`` strictly inside ``REPORTS_DIR`` (excluding ``webui_logs``)."""
    rel = (rel or "").strip()
    if not rel:
        return None, ("Missing path", 400)
    if "\x00" in rel:
        return None, ("Invalid path", 400)

    base = REPORTS_DIR.resolve()
    target = (base / rel).resolve()
    if target == base:
        return None, ("Access to reports root is forbidden", 403)
    if base not in target.parents:
        return None, ("Path outside the reports directory", 403)

    log_base = LOG_DIR.resolve()
    if target == log_base or log_base in target.parents or "webui_logs" in target.relative_to(base).parts:
        return None, ("Access to webui_logs via report API is forbidden", 403)

    if not target.is_file():
        return None, ("No such artifact", 404)
    return target, None


def _strip_heavy_report_blobs(obj: Any) -> Any:
    """Recursively strip large inline ``eda_html`` / ``console_output`` fields."""
    if not isinstance(obj, dict):
        return obj
    cleaned = {}
    for k, v in obj.items():
        if k in ("eda_html", "console_output"):
            continue
        if k == "target_reports" and isinstance(v, dict):
            cleaned[k] = {tk: _strip_heavy_report_blobs(tv) for tk, tv in v.items()}
        else:
            cleaned[k] = v
    return cleaned


def _summarize_json_report(rel: str, data: dict[str, Any]) -> dict[str, Any]:
    cleaned = _sanitize_json_value(_strip_heavy_report_blobs(data))
    if "target_reports" in cleaned or (
        "models" in cleaned and isinstance(cleaned.get("models"), dict) and "phase_results" not in cleaned
    ):
        kind = "multi_target"
    elif "candidates_evaluated" in cleaned and "phase_results" not in cleaned:
        kind = "audit"
    else:
        kind = "combined"

    audit = cleaned.get("audit_trail") if isinstance(cleaned.get("audit_trail"), dict) else {}
    if kind == "audit":
        audit = cleaned

    candidates = audit.get("candidates_evaluated") or cleaned.get("candidates_evaluated") or []
    pareto_front = audit.get("pareto_front") or cleaned.get("pareto_front") or []
    pruned_or_failed = audit.get("pruned_or_failed") or cleaned.get("pruned_or_failed") or []

    selected_model = cleaned.get("selected_model")
    if selected_model is None and isinstance(audit.get("selected"), dict):
        selected_model = audit.get("selected")

    phase_results = cleaned.get("phase_results")
    phases: list[dict[str, Any]] = []
    if isinstance(phase_results, dict):
        for group_items in phase_results.values():
            if isinstance(group_items, list):
                phases.extend(item for item in group_items if isinstance(item, dict))
    elif isinstance(phase_results, list):
        phases = [item for item in phase_results if isinstance(item, dict)]

    summary = cleaned.get("summary") or {}
    dataset_info = cleaned.get("dataset_info") or {}
    risk_report = cleaned.get("risk_report") or audit.get("risk_report")
    eda_report = cleaned.get("eda_report")
    target_reports_out: dict[str, Any] | None = None

    if isinstance(cleaned.get("target_reports"), dict):
        target_reports_out = {}
        for tname, trep in cleaned["target_reports"].items():
            if isinstance(trep, dict):
                target_reports_out[str(tname)] = _summarize_json_report(f"{rel}::{tname}", trep)
        if target_reports_out:
            primary = next(
                (tr for tr in target_reports_out.values() if tr.get("selected_model") is not None),
                next(iter(target_reports_out.values())),
            )
            if selected_model is None:
                selected_model = primary.get("selected_model")
            if not candidates:
                candidates = primary.get("candidates_evaluated") or []
            if not pareto_front:
                pareto_front = primary.get("pareto_front") or []
            if not pruned_or_failed:
                pruned_or_failed = primary.get("pruned_or_failed") or []
            if risk_report is None:
                risk_report = primary.get("risk_report")
            if eda_report is None:
                eda_report = primary.get("eda_report")
            if not dataset_info:
                dataset_info = primary.get("dataset_info") or {}
            if not summary:
                summary = primary.get("summary") or {}
    elif kind == "multi_target" and isinstance(cleaned.get("models"), dict):
        models = cleaned["models"]
        target_reports_out = {}
        if isinstance(models.get("multi_output"), dict):
            mo = models["multi_output"]
            mo_type = mo.get("type", "MultiOutput")
            fit_t = mo.get("fit_time")
            mo_targets = mo.get("targets") or cleaned.get("targets") or []
            first_metric = None
            for tname in mo_targets:
                m_dict = mo.get(tname) if isinstance(mo.get(tname), dict) else {}
                sc = m_dict.get("r2") if "r2" in m_dict else m_dict.get("f1", m_dict.get("accuracy"))
                if first_metric is None and sc is not None:
                    first_metric = sc
                target_reports_out[str(tname)] = {
                    "selected_model": {"name": mo_type, "metrics": {"metric": sc, "train_time_s": fit_t}},
                    "meta": {"target": tname, "task": cleaned.get("task")},
                }
                candidates.append({"name": f"{mo_type} ({tname})", "metric": sc, "train_time_s": fit_t})
            if selected_model is None:
                selected_model = {"name": mo_type, "metrics": {"metric": first_metric, "train_time_s": fit_t}}
        else:
            for tname, minfo in models.items():
                if not isinstance(minfo, dict):
                    continue
                mname = minfo.get("selected_model") or minfo.get("name") or "Unknown"
                sc = minfo.get("score")
                ft = minfo.get("fit_time")
                sel_obj = {"name": mname, "metrics": {"metric": sc, "train_time_s": ft}}
                target_reports_out[str(tname)] = {
                    "selected_model": sel_obj,
                    "meta": {"target": tname, "task": minfo.get("task")},
                }
                candidates.append({"name": f"{mname} ({tname})", "metric": sc, "train_time_s": ft})
                if selected_model is None:
                    selected_model = sel_obj

    meta = {
        "timestamp": cleaned.get("timestamp"),
        "mode": cleaned.get("mode") or kind,
        "file": cleaned.get("file"),
        "target": cleaned.get("target") or cleaned.get("targets"),
        "task": cleaned.get("task"),
        "datetime_col": cleaned.get("datetime_col"),
        "html_report_path": cleaned.get("html_report_path"),
    }

    feature_importance = (
        cleaned.get("feature_importance")
        or audit.get("feature_importance")
        or (cleaned.get("additional_info") or {}).get("feature_importance")
        if isinstance(cleaned.get("additional_info"), dict)
        else (cleaned.get("feature_importance") or audit.get("feature_importance"))
    )

    return _sanitize_json_value({
        "path": rel,
        "kind": kind,
        "meta": meta,
        "summary": summary,
        "dataset_info": dataset_info,
        "phase_results": phase_results,
        "phases": phases,
        "selected_model": selected_model,
        "test_performance": cleaned.get("test_performance"),
        "audit_trail": audit if audit else None,
        "candidates_evaluated": candidates,
        "pareto_front": pareto_front,
        "pruned_or_failed": pruned_or_failed,
        "risk_report": risk_report,
        "eda_report": eda_report,
        "feature_importance": feature_importance,
        "predictions": cleaned.get("predictions"),
        "timeseries": cleaned.get("timeseries"),
        "benchmark_results": cleaned.get("benchmark_results"),
        "target_reports": target_reports_out,
    })


def _summarize_csv_report(rel: str, target: Path, limit: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    total_rows = 0
    pred_cols: list[str] = []
    pred_samples: dict[str, list[str]] = {}

    with target.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return {
                "path": rel,
                "kind": "csv",
                "columns": [],
                "rows": [],
                "total_rows": 0,
                "prediction_stats": {},
            }
        columns = [str(c) for c in header]
        pred_cols = [c for c in columns if c.endswith("_predicted") or c in ("type",)]
        for pc in pred_cols:
            pred_samples[pc] = []

        for raw_row in reader:
            total_rows += 1
            if len(rows) < limit:
                row_dict = {
                    columns[i]: (raw_row[i] if i < len(raw_row) else "")
                    for i in range(len(columns))
                }
                rows.append(row_dict)
            if total_rows <= 2000:
                for pc in pred_cols:
                    idx = columns.index(pc)
                    val = raw_row[idx] if idx < len(raw_row) else ""
                    if val != "":
                        pred_samples[pc].append(val)

    prediction_stats: dict[str, Any] = {}
    for pc, vals in pred_samples.items():
        if not vals:
            continue
        counts: dict[str, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        prediction_stats[pc] = {
            "sampled": len(vals),
            "distinct_sampled": len(counts),
            "top_values": [{"value": k, "count": cnt} for k, cnt in top],
        }

    return {
        "path": rel,
        "kind": "csv",
        "columns": columns,
        "rows": rows,
        "total_rows": total_rows,
        "prediction_stats": prediction_stats,
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.json.sort_keys = False

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manager = JobManager()
    uploads_lock = threading.Lock()
    uploads: dict[str, Upload] = _load_upload_index(UPLOAD_DIR)

    def _get_upload(raw_id: Any) -> Upload | None:
        if not isinstance(raw_id, str) or not raw_id:
            return None
        with uploads_lock:
            return uploads.get(raw_id)

    def _snapshot_uploads() -> list[Upload]:
        with uploads_lock:
            return list(uploads.values())

    def _ensure_inspected(record: Upload) -> Upload:
        if record.needs_inspect or not record.columns:
            _inspect_upload_record(record)
            _save_upload_index(UPLOAD_DIR, _snapshot_uploads())
        return record

    @app.get("/")
    def index():
        groups: dict[str, list[Flag]] = {}
        for flag in FLAGS:
            groups.setdefault(flag.group, []).append(flag)
        flag_meta = [
            {
                "name": f.name,
                "cli": f.cli,
                "kind": f.kind,
                "default": f.default,
                "label": f.label,
                "dead": f.dead,
                "tier": f.tier,
                "depends_on": f.depends_on,
            }
            for f in FLAGS
        ]
        return render_template(
            "index.html",
            groups=groups,
            allowed=sorted(ALLOWED_SUFFIXES),
            repo_root=str(REPO_ROOT),
            flag_meta=json.dumps(flag_meta),
        )

    @app.get("/api/uploads")
    def list_uploads():
        ordered = sorted(_snapshot_uploads(), key=lambda u: u.uploaded_at, reverse=True)
        existing = [u for u in ordered if u.path.is_file()]
        return jsonify(
            uploads=[u.to_summary_dict(include_preview=False) for u in existing[:MAX_PERSISTED_UPLOADS]]
        )

    @app.get("/api/uploads/<upload_id>")
    def get_upload(upload_id: str):
        record = _get_upload(upload_id)
        if record is None or not record.path.is_file():
            return jsonify(error="Upload not found"), 404
        try:
            _ensure_inspected(record)
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"Could not read the file: {exc}"), 400
        return jsonify(record.to_summary_dict(include_preview=True))

    @app.post("/api/upload")
    def upload():
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify(error="No file part in the request"), 400

        raw_path = Path(file.filename)
        suffix = raw_path.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            return (
                jsonify(
                    error=f"Unsupported file type {suffix or '(none)'}. "
                    f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}"
                ),
                400,
            )

        stem = Path(secure_filename(raw_path.stem)).name or "upload"
        upload_id = uuid.uuid4().hex[:12]
        dest = UPLOAD_DIR / f"{stem}_{upload_id}{suffix}"
        file.save(dest)

        record = Upload(
            id=upload_id,
            path=dest,
            original_name=file.filename,
            columns=[],
            rows=None,
            preview=[],
            size_bytes=int(dest.stat().st_size) if dest.is_file() else 0,
            uploaded_at=time.time(),
            needs_inspect=True,
        )
        try:
            _inspect_upload_record(record)
        except Exception as exc:  # noqa: BLE001 - surface any parse failure verbatim
            dest.unlink(missing_ok=True)
            return jsonify(error=f"Could not read the file: {exc}"), 400

        with uploads_lock:
            uploads[upload_id] = record
            snapshot = list(uploads.values())
        _save_upload_index(UPLOAD_DIR, snapshot)

        return jsonify(record.to_summary_dict(include_preview=True))

    @app.post("/api/forecast")
    def forecast():
        """Aggregate an upload to period totals and project it forward."""
        from .forecast import SeriesSpec, VALID_AGGS, VALID_GRAINS, run_forecast

        payload, err_resp = _json_dict_payload()
        if err_resp is not None:
            return err_resp
        assert payload is not None

        record = _get_upload(payload.get("upload_id"))
        if record is None or not record.path.is_file():
            return jsonify(error="Upload a data file first"), 400

        date_col = payload.get("date_col")
        value_col = payload.get("value_col")
        if not isinstance(date_col, str) or not isinstance(value_col, str) or not date_col or not value_col:
            return jsonify(error="Pick both a period column and a value column"), 400

        agg = str(payload.get("agg") or "sum")
        if agg not in VALID_AGGS:
            return jsonify(error=f"Unsupported aggregation: {agg}"), 400

        grain = str(payload.get("grain") or "auto")
        if grain not in VALID_GRAINS:
            return jsonify(error=f"Unsupported grain: {grain}"), 400

        raw_h = payload.get("horizon")
        if raw_h is None or raw_h == "":
            horizon = 3
        else:
            if isinstance(raw_h, bool):
                return jsonify(error="Horizon must be a whole number between 1 and 36"), 400
            try:
                horizon_f = float(raw_h)
                if not math.isfinite(horizon_f) or int(horizon_f) != horizon_f:
                    raise ValueError("non-integer")
                horizon = int(horizon_f)
            except (TypeError, ValueError):
                return jsonify(error="Horizon must be a whole number between 1 and 36"), 400
            if horizon < 1 or horizon > 36:
                return jsonify(error="Horizon must be between 1 and 36"), 400

        raw_filters = payload.get("filters")
        if raw_filters is None:
            filters: dict[str, list[str]] = {}
        elif not isinstance(raw_filters, dict):
            return jsonify(error="filters must be a JSON object mapping column names to values"), 400
        else:
            filters = {}
            for k, v in raw_filters.items():
                if not isinstance(k, str) or not k.strip():
                    return jsonify(error="Filter column names must be non-empty strings"), 400
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    if isinstance(v, float) and not math.isfinite(v):
                        return jsonify(error=f"Filter for {k!r} contains a non-finite number"), 400
                    filters[k] = [str(v)]
                elif isinstance(v, (list, tuple)):
                    items: list[str] = []
                    for item in v:
                        if not isinstance(item, (str, int, float, bool)):
                            return jsonify(error=f"Filter values for {k!r} must be scalar values"), 400
                        if isinstance(item, float) and not math.isfinite(item):
                            return jsonify(error=f"Filter for {k!r} contains a non-finite number"), 400
                        items.append(str(item))
                    filters[k] = items
                else:
                    return jsonify(error=f"Filter for {k!r} must be a scalar or list of values"), 400

        spec = SeriesSpec(
            date_col=date_col,
            value_col=value_col,
            agg=agg,
            grain=grain,
            group_filters=filters,
            drop_first=bool(payload.get("drop_first")),
            drop_last=bool(payload.get("drop_last")),
        )

        try:
            cols_needed = list(dict.fromkeys([spec.date_col, spec.value_col, *filters.keys()]))
            try:
                frame = _read_frame(record.path, nrows=None, usecols=cols_needed)
            except Exception:
                frame = _read_frame(record.path, nrows=None)
            result = run_forecast(frame, spec, horizon)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"Forecast failed: {exc}"), 500

        result["csv_path"] = _write_forecast_csv(result, record.original_name)
        return jsonify(result)

    @app.post("/api/suggest-target")
    def suggest_target_route():
        """Rank predictable columns as plausible targets (top 3 by default)."""
        from .column_validator import analyze_predictable_columns, suggest_target_columns

        payload, err_resp = _json_dict_payload()
        if err_resp is not None:
            return err_resp
        assert payload is not None

        record = _get_upload(payload.get("upload_id"))
        if record is None or not record.path.is_file():
            return jsonify(error="Upload a data file first"), 400

        if record.suggest_target_cache is not None and not payload.get("refresh"):
            return jsonify(**record.suggest_target_cache, cached=True)

        try:
            _ensure_inspected(record)
            sample = _read_frame(record.path, nrows=1000)
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"Could not read the file: {exc}"), 400

        candidates = list(record.predictable_columns)
        excluded = dict(record.excluded_columns)
        if not candidates and not excluded:
            candidates, excluded = analyze_predictable_columns(sample)
        result = suggest_target_columns(sample, candidates, top_n=3)
        full_res = {**result, "candidates": candidates, "excluded_columns": excluded}
        record.suggest_target_cache = full_res
        return jsonify(**full_res, cached=False)

    @app.post("/api/suggest-horizon")
    def suggest_horizon_route():
        """Recommend --horizon once a datetime column and a target are picked."""
        from .column_validator import suggest_horizon

        payload, err_resp = _json_dict_payload()
        if err_resp is not None:
            return err_resp
        assert payload is not None

        record = _get_upload(payload.get("upload_id"))
        if record is None or not record.path.is_file():
            return jsonify(error="Upload a data file first"), 400

        raw_dt = payload.get("datetime_col")
        raw_tc = payload.get("target_col")
        if (raw_dt is not None and not isinstance(raw_dt, str)) or (
            raw_tc is not None and not isinstance(raw_tc, str)
        ):
            return jsonify(error="datetime_col and target_col must be strings"), 400

        datetime_col = (raw_dt or "").strip()
        target_col = (raw_tc or "").strip()
        if not datetime_col or not target_col:
            return jsonify(error="Pick a datetime column and a target column first"), 400

        try:
            _ensure_inspected(record)
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"Could not read the file: {exc}"), 400

        if record.columns:
            missing = [c for c in (datetime_col, target_col) if c not in record.columns]
            if missing:
                return jsonify(error=f"Unknown column(s): {', '.join(missing)}"), 400

        cache_key = (datetime_col, target_col)
        if cache_key in record.horizon_cache:
            return jsonify(**record.horizon_cache[cache_key], cached=True)

        try:
            sample = _read_frame(record.path, nrows=20_000, usecols=[datetime_col, target_col])
            result = suggest_horizon(sample, datetime_col, target_col)
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"Could not suggest horizon: {exc}"), 400

        record.horizon_cache[cache_key] = result
        return jsonify(**result, cached=False)

    def _column_roles_from_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
        return {
            "predict": _as_list(payload.get("predict"), "predict"),
            "target": _as_list(payload.get("target"), "target"),
            "exclude": _as_list(payload.get("exclude"), "exclude"),
            "datetime_col": _as_list(payload.get("datetime_col"), "datetime_col"),
            "treatment": _as_list(payload.get("treatment"), "treatment"),
        }

    @app.post("/api/validate-columns")
    def validate_columns_route():
        """Advisory + blocking validation of selected column roles."""
        from .column_validator import validate_columns

        payload, err_resp = _json_dict_payload()
        if err_resp is not None:
            return err_resp
        assert payload is not None

        record = _get_upload(payload.get("upload_id"))
        if record is None or not record.path.is_file():
            return jsonify(error="Upload a data file first"), 400

        try:
            roles = _column_roles_from_payload(payload)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if not any(roles.values()):
            return jsonify(ok=True, source="rules", issues=[])

        try:
            frame = _read_frame(record.path, nrows=2000)
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"Could not read the file: {exc}"), 400

        result = validate_columns(frame, roles)
        return jsonify(result.to_dict())

    @app.post("/api/nl-plan")
    def nl_plan_route():
        """Turn a free-text request into a column-role plan."""
        from .nl_plan import build_plan

        payload, err_resp = _json_dict_payload()
        if err_resp is not None:
            return err_resp
        assert payload is not None

        record = _get_upload(payload.get("upload_id"))
        if record is None or not record.path.is_file():
            return jsonify(error="Upload a data file first"), 400

        raw_prompt = payload.get("prompt")
        if raw_prompt is not None and not isinstance(raw_prompt, str):
            return jsonify(error="prompt must be a string"), 400
        prompt = (raw_prompt or "").strip()
        if not prompt:
            return jsonify(error="Type a request first"), 400

        try:
            frame = _read_frame(record.path, nrows=1000)
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"Could not read the file: {exc}"), 400

        try:
            plan = build_plan(prompt, frame)
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=f"Could not build plan: {exc}"), 500
        return jsonify(plan)

    @app.post("/api/run")
    def run():
        from .column_validator import _rule_based_validate

        payload, err_resp = _json_dict_payload()
        if err_resp is not None:
            return err_resp
        assert payload is not None

        raw_upload_id = payload.get("upload_id")
        if raw_upload_id is not None and not isinstance(raw_upload_id, str):
            return jsonify(error="upload_id must be a string"), 400
        upload_id = raw_upload_id

        run_all_demos = bool(payload.get("all"))
        standalone_no_file = bool(
            payload.get("v3")
            or payload.get("policy")
            or payload.get("timeseries")
            or payload.get("decision")
            or (payload.get("benchmark") and not upload_id)
        )

        # Numeric parameter bounds validation
        for int_field, label, min_val in (
            ("horizon", "Forecast horizon", 1),
            ("time_budget", "Time budget", 1),
        ):
            raw_val = payload.get(int_field)
            if raw_val not in (None, ""):
                if isinstance(raw_val, bool):
                    return jsonify(error=f"{label} must be an integer >= {min_val}"), 400
                try:
                    num = float(raw_val)
                    if not math.isfinite(num) or int(num) != num or int(num) < min_val:
                        raise ValueError("invalid")
                except (TypeError, ValueError):
                    return jsonify(error=f"{label} must be an integer >= {min_val}"), 400

        raw_budget = payload.get("budget")
        if raw_budget not in (None, ""):
            if isinstance(raw_budget, bool):
                return jsonify(error="Budget constraint must be a finite number >= 0"), 400
            try:
                b_val = float(raw_budget)
                if not math.isfinite(b_val) or b_val < 0:
                    raise ValueError("invalid")
            except (TypeError, ValueError):
                return jsonify(error="Budget constraint must be a finite number >= 0"), 400

        try:
            roles = _column_roles_from_payload(payload)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        data_file: Path | None = None
        record = None
        if upload_id:
            record = _get_upload(upload_id)
            if record is None or not record.path.is_file():
                return jsonify(error="Unknown upload -- re-upload the file"), 400
            data_file = record.path
        elif not run_all_demos and not standalone_no_file:
            return (
                jsonify(
                    error="Upload a data file, or select a built-in demo mode to run without one."
                ),
                400,
            )

        has_target = bool(roles["predict"] or roles["target"])
        if data_file is not None and not run_all_demos and not standalone_no_file and not has_target:
            return jsonify(error="Pick at least one target column to predict."), 400

        if record is not None:
            try:
                _ensure_inspected(record)
            except Exception as exc:  # noqa: BLE001
                return jsonify(error=f"Could not read the uploaded file: {exc}"), 400

            if any(roles.values()):
                try:
                    frame = _read_frame(record.path, nrows=2000)
                except Exception as exc:  # noqa: BLE001
                    return jsonify(error=f"Could not read the uploaded file: {exc}"), 400

                check = _rule_based_validate(frame, roles)
                if not check.ok:
                    blocking = [i for i in check.issues if i.severity == "block"]
                    return (
                        jsonify(
                            error="Column selection conflict: " + " ".join(i.message for i in blocking),
                            column_issues=check.to_dict(),
                        ),
                        400,
                    )

        try:
            argv = build_argv(payload, data_file)
        except (ValueError, TypeError) as exc:
            return jsonify(error=f"Invalid option: {exc}"), 400

        want_queue = bool(payload.get("queue"))
        if want_queue:
            job, qpos = manager.enqueue(argv)
            _invalidate_reports_cache()
            return (
                jsonify(
                    job_id=job.id,
                    command=job.command,
                    status=job.status,
                    queue_position=qpos,
                ),
                202 if qpos > 0 else 200,
            )

        try:
            job = manager.start(argv)
        except RuntimeError as exc:
            return jsonify(error=str(exc), queueable=True), 409

        _invalidate_reports_cache()
        return jsonify(job_id=job.id, command=job.command, status=job.status, queue_position=0)

    @app.get("/api/job/<job_id>")
    def job_status(job_id: str):
        job = manager.get(job_id)
        if job is None:
            return jsonify(error="Unknown job"), 404
        cursor = request.args.get("cursor", type=int, default=0)
        lines, next_cursor = job.slice_from(cursor)
        with job._lock:
            status = job.status
            returncode = job.returncode
            progress = dict(job.progress) if job.progress is not None else None
        try:
            log_file = str(job.log_path.relative_to(REPO_ROOT))
        except ValueError:
            log_file = str(job.log_path)
        return jsonify(
            job_id=job.id,
            status=status,
            returncode=returncode,
            command=job.command,
            elapsed=round(time.time() - job.started_at, 1),
            lines=lines,
            cursor=next_cursor,
            progress=progress,
            queue_position=manager.queue_position(job.id),
            log_file=log_file,
            artifacts=artifacts_since(job.started_at),
        )

    @app.get("/api/job/<job_id>/stream")
    def job_stream(job_id: str):
        """Server-Sent Events stream for real-time job logs, progress, and completion."""
        job = manager.get(job_id)
        if job is None:
            return jsonify(error="Unknown job"), 404

        init_cursor = max(0, request.args.get("cursor", type=int, default=0))
        max_seconds = request.args.get("max_seconds", type=float, default=3600.0)
        max_seconds = max(1.0, min(float(max_seconds), 86400.0))

        def _sse(event: str, payload: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(_sanitize_json_value(payload), ensure_ascii=False)}\n\n"

        @stream_with_context
        def _generate():
            cursor = init_cursor
            last_prog_key: str | None = None
            last_status: str | None = None
            started = time.monotonic()

            try:
                log_file = str(job.log_path.relative_to(REPO_ROOT))
            except ValueError:
                log_file = str(job.log_path)

            while True:
                lines, next_cursor = job.slice_from(cursor)
                with job._lock:
                    status = job.status
                    returncode = job.returncode
                    progress = dict(job.progress) if job.progress is not None else None
                    started_at = job.started_at

                if lines:
                    cursor = next_cursor
                    yield _sse("log", {"lines": lines, "cursor": cursor})

                prog_key = json.dumps(progress, sort_keys=True) if progress is not None else None
                if prog_key != last_prog_key and progress is not None:
                    last_prog_key = prog_key
                    yield _sse("progress", {"progress": progress, "status": status})

                if status != last_status:
                    last_status = status
                    yield _sse(
                        "status",
                        {
                            "job_id": job.id,
                            "status": status,
                            "returncode": returncode,
                            "command": job.command,
                            "elapsed": round(time.time() - started_at, 1),
                            "queue_position": manager.queue_position(job.id),
                            "log_file": log_file,
                        },
                    )

                if status in ("finished", "failed", "cancelled"):
                    # Flush any final lines appended right at exit
                    tail_lines, tail_cursor = job.slice_from(cursor)
                    if tail_lines:
                        cursor = tail_cursor
                        yield _sse("log", {"lines": tail_lines, "cursor": cursor})
                    yield _sse(
                        "done",
                        {
                            "job_id": job.id,
                            "status": status,
                            "returncode": returncode,
                            "elapsed": round(time.time() - started_at, 1),
                            "cursor": cursor,
                            "progress": progress,
                            "log_file": log_file,
                            "artifacts": artifacts_since(started_at),
                        },
                    )
                    break

                if time.monotonic() - started >= max_seconds:
                    yield _sse("timeout", {"cursor": cursor, "status": status})
                    break

                with job._lock:
                    has_new = (job.dropped_lines + len(job.lines)) > cursor
                    if not has_new and job.status in ("queued", "running"):
                        job._cond.wait(timeout=1.5)
                if (job.dropped_lines + len(job.lines)) == cursor:
                    yield ": keepalive\n\n"

        # Omit hop-by-hop headers (e.g. Connection) per PEP 3333 so Waitress/WSGI servers stream cleanly
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return Response(_generate(), mimetype="text/event-stream", headers=headers)

    @app.get("/api/queue")
    def queue_list():
        active = manager.active_job
        active_info = None
        if active is not None and active.status == "running":
            with active._lock:
                active_info = {
                    "job_id": active.id,
                    "status": active.status,
                    "command": active.command,
                    "elapsed": round(time.time() - active.started_at, 1),
                    "progress": dict(active.progress) if active.progress is not None else None,
                }
        return jsonify(active_job=active_info, queue=manager.list_queue())

    @app.delete("/api/queue/<job_id>")
    def queue_cancel(job_id: str):
        removed = manager.cancel_queued(job_id)
        if not removed:
            return jsonify(error="Job is not currently queued"), 404
        return jsonify(cancelled=True, job_id=job_id, queue=manager.list_queue())

    @app.post("/api/compare-reports")
    def compare_reports():
        """Compare 2-4 JSON experiment reports side-by-side with baseline deltas."""
        payload, err_resp = _json_dict_payload()
        if err_resp is not None:
            return err_resp
        assert payload is not None

        raw_paths = payload.get("paths")
        if not isinstance(raw_paths, list) or not (2 <= len(raw_paths) <= 4):
            return jsonify(error="Provide between 2 and 4 JSON report paths to compare."), 400

        summaries: list[dict[str, Any]] = []
        for raw_p in raw_paths:
            if not isinstance(raw_p, str) or not raw_p.strip():
                return jsonify(error="Each report path must be a non-empty string."), 400
            rel = raw_p.strip()
            target, err = _resolve_report_target(rel)
            if err is not None:
                return jsonify(error=err[0]), err[1]
            assert target is not None
            if target.suffix.lower() != ".json":
                return jsonify(error=f"Compare supports JSON reports only (got {rel})"), 400
            try:
                data = json.loads(target.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:  # noqa: BLE001
                return jsonify(error=f"Malformed JSON report '{rel}': {exc}"), 400
            if not isinstance(data, dict):
                return jsonify(error=f"Unsupported JSON report '{rel}'"), 400
            summaries.append(_summarize_json_report(rel, data))

        def _extract_kpis(s: dict[str, Any]) -> dict[str, Any]:
            sel = s.get("selected_model") or {}
            metrics = sel.get("metrics") or {}
            meta = s.get("meta") or {}
            rr = s.get("risk_report") or {}
            ds = s.get("dataset_info") or {}
            score = metrics.get("metric") if metrics.get("metric") is not None else sel.get("metric")
            train_t = metrics.get("train_time_s")
            mem = metrics.get("peak_mem_mb")
            risk = metrics.get("risk_penalty")
            lat = metrics.get("latency_ms_per_row")
            return {
                "path": s.get("path"),
                "target": meta.get("target"),
                "task": meta.get("task") or rr.get("task"),
                "selected_model": sel.get("name") or sel.get("model_name") or "None",
                "metric": score,
                "train_time_s": train_t,
                "peak_mem_mb": mem,
                "latency_ms_per_row": lat,
                "risk_penalty": risk,
                "test_performance": s.get("test_performance"),
                "rows": ds.get("rows"),
                "columns": ds.get("columns"),
                "candidates_count": len(s.get("candidates_evaluated") or []),
                "pareto_count": len(s.get("pareto_front") or []),
            }

        items = [_extract_kpis(s) for s in summaries]
        base = items[0]
        for idx, item in enumerate(items):
            if idx == 0:
                item["deltas"] = {"metric": 0.0, "train_time_s": 0.0, "peak_mem_mb": 0.0, "risk_penalty": 0.0}
                continue
            deltas: dict[str, float | None] = {}
            for key in ("metric", "train_time_s", "peak_mem_mb", "risk_penalty"):
                a, b = base.get(key), item.get(key)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    deltas[key] = round(float(b) - float(a), 6)
                else:
                    deltas[key] = None
            item["deltas"] = deltas

        return jsonify(_sanitize_json_value({"baseline": base["path"], "reports": items}))

    @app.post("/api/job/<job_id>/stop")
    @app.post("/api/stop/<job_id>")
    def job_stop(job_id: str):
        job = manager.get(job_id)
        if job is None:
            return jsonify(error="Unknown job"), 404
        stopped = manager.stop(job)
        with job._lock:
            status = job.status
        return jsonify(stopped=stopped, status=status)

    @app.get("/api/jobs")
    def job_list():
        now = time.time()
        items = []
        for j in manager.recent():
            with j._lock:
                items.append(
                    {
                        "job_id": j.id,
                        "status": j.status,
                        "returncode": j.returncode,
                        "started_at": j.started_at,
                        "elapsed": round(now - j.started_at, 1),
                        "command": j.command,
                        "progress": dict(j.progress) if j.progress is not None else None,
                    }
                )
        return jsonify(jobs=items)

    @app.get("/api/reports")
    def reports_list():
        """All files ever written to reports/ -- past runs, not just the live job."""
        limit = request.args.get("limit", type=int, default=300)
        limit = max(1, min(limit, 2000))
        all_reports = _all_report_files()
        return jsonify(
            reports=all_reports[:limit],
            total=len(all_reports),
            shown=min(limit, len(all_reports)),
        )

    @app.get("/api/report-summary")
    def report_summary():
        """Safely parse a JSON or CSV report inside ``reports/`` for in-app inspection."""
        rel = request.args.get("path", "")
        target, err = _resolve_report_target(rel)
        if err is not None:
            return jsonify(error=err[0]), err[1]
        assert target is not None

        limit = request.args.get("limit", type=int, default=50)
        limit = max(1, min(limit, 200))
        suffix = target.suffix.lower()
        if suffix == ".json":
            try:
                data = json.loads(target.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:  # noqa: BLE001
                return jsonify(error=f"Malformed JSON report: {exc}"), 400
            if not isinstance(data, dict):
                return jsonify(error="Unsupported JSON report structure (expected object)"), 400
            return jsonify(_summarize_json_report(rel, data))
        if suffix == ".csv":
            try:
                return jsonify(_sanitize_json_value(_summarize_csv_report(rel, target, limit)))
            except Exception as exc:  # noqa: BLE001
                return jsonify(error=f"Could not read CSV report: {exc}"), 400
        return jsonify(error=f"Report summary supports .json and .csv files, not {suffix or '(none)'}"), 400

    @app.get("/api/artifact")
    def artifact():
        rel = request.args.get("path", "")
        target, err = _resolve_report_target(rel)
        if err is not None:
            return jsonify(error=err[0]), err[1]
        assert target is not None
        inline = request.args.get("inline") == "1"
        return send_file(target, as_attachment=not inline)

    @app.delete("/api/reports")
    def reports_delete():
        """Delete one report file (?path=...) or every report file (?all=1)."""
        if manager.active_job is not None and manager.active_job.status == "running":
            return jsonify(error="Cannot delete while a job is running"), 409

        if request.args.get("all") == "1":
            deleted, errors = [], []
            for entry in _all_report_files():
                target = REPORTS_DIR / entry["path"]
                try:
                    target.unlink()
                    deleted.append(entry["path"])
                except OSError as exc:
                    errors.append({"path": entry["path"], "error": str(exc)})
            _invalidate_reports_cache()
            return jsonify(deleted=deleted, errors=errors)

        rel = request.args.get("path", "")
        target, err = _resolve_report_target(rel)
        if err is not None:
            return jsonify(error=err[0]), err[1]
        assert target is not None
        try:
            target.unlink()
        except OSError as exc:
            return jsonify(error=str(exc)), 500
        _invalidate_reports_cache()
        return jsonify(deleted=[rel])

    @app.get("/api/health")
    def health():
        active = manager.active_job
        return jsonify(
            ok=True,
            repo_root=str(REPO_ROOT),
            runner=str(RUNNER),
            runner_present=RUNNER.is_file(),
            active_job=active.id if active and active.status == "running" else None,
            queue_length=len(manager.list_queue()),
        )

    return app


_DEFAULT_APP: Flask | None = None
_DEFAULT_APP_LOCK = threading.Lock()


def __getattr__(name: str):
    global _DEFAULT_APP
    if name == "app":
        with _DEFAULT_APP_LOCK:
            if _DEFAULT_APP is None:
                _DEFAULT_APP = create_app()
            return _DEFAULT_APP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
