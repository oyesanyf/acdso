"""ACDSO Interactive Terminal UI (Textual dashboard + zero-dependency fallback)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "run_acdso.py"
REPORTS_DIR = REPO_ROOT / "reports"

TASK_CHOICES: tuple[str, ...] = (
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
TARGETLESS_TASKS: frozenset[str] = frozenset({"clustering", "anomaly", "survival", "recommender"})
CV_STRATEGIES: tuple[str, ...] = (
    "auto",
    "kfold",
    "stratified",
    "group_kfold",
    "stratified_group_kfold",
    "temporal",
)
NLP_METHODS: tuple[str, ...] = ("auto", "tfidf", "ngram", "lda", "nmf", "combined")
CLUSTER_ALGOS: tuple[str, ...] = ("auto", "kmeans", "hdbscan", "gmm")
DIM_REDUCE_MODES: tuple[str, ...] = ("none", "pca", "umap")
CLUSTER_METRICS: tuple[str, ...] = ("silhouette", "davies_bouldin")
ANOMALY_ALGOS: tuple[str, ...] = ("auto", "isolation_forest", "lof", "ocsvm")
SCALE_MODES: tuple[str, ...] = ("auto", "in_memory", "chunked", "subsample")


def has_textual() -> bool:
    """Return True when the optional ``textual`` package is installed."""
    return importlib.util.find_spec("textual") is not None


def parse_progress_marker(line: str) -> dict[str, Any] | None:
    """Parse a single-line ``[ACDSO_PROGRESS] {...}`` event emitted by run_acdso.py."""
    prefix = "[ACDSO_PROGRESS]"
    if not line.startswith(prefix):
        return None
    raw = line[len(prefix):].strip()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        raw_phase = data.get("phase")
        raw_total = data.get("total")
        if isinstance(raw_phase, bool) or isinstance(raw_total, bool):
            return None
        if isinstance(raw_phase, float) and not math.isfinite(raw_phase):
            return None
        if isinstance(raw_total, float) and not math.isfinite(raw_total):
            return None
        phase = max(0, int(raw_phase if raw_phase is not None else 0))
        total = max(1, int(raw_total if raw_total is not None else 1))
    except (ValueError, TypeError, OverflowError):
        return None
    status = str(data.get("status") or "running")
    pct = 100 if (phase >= total and status == "done") else min(99, int(round((phase / total) * 100)))
    return {
        "phase": phase,
        "total": total,
        "name": str(data.get("name") or "Pipeline"),
        "status": status,
        "percent": pct,
        "detail": str(data.get("detail") or ""),
    }


def profile_dataset(file_path: str | Path, max_rows: int = 2000) -> dict[str, Any]:
    """Summarize a dataset using bounded schema inspection & chunked sampling."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    from acdso.cli.io import inspect_file_schema, iter_file_chunks, load_user_file
    from acdso.webui.column_validator import analyze_predictable_columns

    eff_max = max(1, int(max_rows))
    schema = inspect_file_schema(str(p), sample_rows=eff_max, count_rows=True)
    sample = None
    for chunk in iter_file_chunks(str(p), chunk_size=eff_max):
        sample = chunk.head(eff_max)
        break
    if sample is None:
        sample = load_user_file(str(p)).head(eff_max)

    total_rows = int(schema["n_rows"]) if schema.get("n_rows") is not None else int(len(sample))
    predictable, excluded = analyze_predictable_columns(sample)
    pred_set = set(predictable)

    try:
        from acdso.nlp import detect_text_columns
        text_cols = [str(c) for c in detect_text_columns(sample)]
    except Exception:
        text_cols = []

    cols: list[dict[str, Any]] = []
    for c in sample.columns:
        s = sample[c]
        name = str(c)
        null_pct = round(float(s.isna().mean() * 100.0), 1) if len(s) else 0.0
        cols.append(
            {
                "name": name,
                "dtype": str(s.dtype),
                "nunique": int(s.nunique(dropna=True)),
                "null_pct": null_pct,
                "predictable": name in pred_set,
                "exclude_reason": excluded.get(name, ""),
            }
        )

    return {
        "path": str(p.resolve()),
        "name": p.name,
        "rows": total_rows,
        "cols": int(len(sample.columns)),
        "columns": cols,
        "predictable": predictable,
        "text_cols": text_cols,
    }


def suggest_targets(file_path: str | Path, top_n: int = 3) -> list[dict[str, Any]]:
    """Return ranked target suggestions using bounded sampling and shared column validator."""
    from acdso.cli.io import iter_file_chunks, load_user_file
    from acdso.webui.column_validator import analyze_predictable_columns, suggest_target_columns

    df = None
    for chunk in iter_file_chunks(str(file_path), chunk_size=1000):
        df = chunk.head(1000)
        break
    if df is None:
        df = load_user_file(str(file_path)).head(1000)
    candidates, _ = analyze_predictable_columns(df)
    res = suggest_target_columns(df, candidates, top_n=top_n)
    return list(res.get("suggestions") or [])


def _normalize_csv_arg(val: str | list[str] | tuple[str, ...] | None) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return ",".join(str(x).strip() for x in val if str(x).strip())
    return str(val).strip()


def _validate_before_run(
    file_path: str,
    target: str = "",
    task: str = "auto",
    *,
    group_col: str = "",
    cv_strategy: str = "auto",
    time_col: str = "",
    event_col: str = "",
    query_col: str = "",
    user_col: str = "",
    item_col: str = "",
    n_clusters: int | None = None,
    contamination: float | str | None = None,
    chunk_size: int | None = None,
) -> tuple[bool, str]:
    """Validate dataset, task, and required R1-R6 columns before launching CLI."""
    if not str(file_path or "").strip():
        return False, "Set a dataset file and target column before running."

    task_norm = str(task or "auto").strip().lower()
    if task_norm not in TASK_CHOICES:
        return False, f"Invalid task '{task}'. Expected one of {', '.join(TASK_CHOICES)}."

    cv_norm = str(cv_strategy or "auto").strip().lower()
    if cv_strategy and cv_norm not in CV_STRATEGIES:
        return False, f"Invalid CV strategy '{cv_strategy}'."
    if cv_norm in ("group_kfold", "stratified_group_kfold") and not str(group_col or "").strip():
        return False, f"CV strategy '{cv_norm}' requires --group-col."

    if n_clusters is not None:
        if isinstance(n_clusters, bool) or (isinstance(n_clusters, float) and not math.isfinite(n_clusters)):
            return False, f"n_clusters must be >= 2 (got {n_clusters!r})."
        try:
            nc_val = int(n_clusters)
        except (TypeError, ValueError, OverflowError):
            return False, f"n_clusters must be an integer >= 2 (got {n_clusters!r})."
        if nc_val < 2:
            return False, f"n_clusters must be >= 2 (got {n_clusters!r})."

    if contamination is not None and str(contamination).strip().lower() not in ("", "auto"):
        try:
            cval = float(contamination)
            if not math.isfinite(cval) or cval <= 0.0 or cval > 0.5:
                return False, f"contamination must be 'auto' or in (0.0, 0.5] (got {contamination!r})."
        except (TypeError, ValueError):
            return False, f"contamination must be 'auto' or a float in (0.0, 0.5] (got {contamination!r})."

    if chunk_size is not None:
        if isinstance(chunk_size, bool) or (isinstance(chunk_size, float) and not math.isfinite(chunk_size)):
            return False, f"chunk_size must be >= 1 (got {chunk_size!r})."
        try:
            cs_val = int(chunk_size)
        except (TypeError, ValueError, OverflowError):
            return False, f"chunk_size must be an integer >= 1 (got {chunk_size!r})."
        if cs_val < 1:
            return False, f"chunk_size must be >= 1 (got {chunk_size!r})."

    clean_target = str(target or "").strip()
    if task_norm in ("clustering", "anomaly"):
        return True, ""
    if task_norm == "survival":
        tc = str(time_col or "").strip()
        ec = str(event_col or "").strip()
        target_parts = [c.strip() for c in clean_target.split(",") if c.strip()]
        has_pair = bool(tc and ec)
        has_two_target = len(target_parts) == 2
        if not (has_pair or has_two_target):
            return False, "Survival task requires --time-col and --event-col (or a 2-column target)."
        if has_pair and tc == ec:
            return False, "Survival time_col and event_col must be distinct columns."
        if has_two_target and target_parts[0] == target_parts[1]:
            return False, "Survival time and event target columns must be distinct."
        return True, ""
    if task_norm == "recommender":
        uc = str(user_col or "").strip()
        ic = str(item_col or "").strip()
        if not (uc and ic):
            return False, "Recommender task requires --user-col and --item-col."
        if uc == ic:
            return False, "Recommender user_col and item_col must be distinct columns."
        return True, ""
    if task_norm == "ranking":
        if not clean_target:
            return False, "Set a target relevance column before running ranking."
        if not (str(query_col or "").strip() or str(group_col or "").strip()):
            return False, "Ranking task requires --query-col (or --group-col)."
        return True, ""

    if not clean_target:
        return False, "Set both a dataset file and a target column before running."
    return True, ""


def build_cli_command(
    file_path: str,
    target: str = "",
    predict_mode: bool = True,
    simple: bool = True,
    skip_benchmark: bool = True,
    best_score: bool = False,
    force: bool = False,
    *,
    task: str = "auto",
    group_col: str | None = None,
    cv_strategy: str | None = None,
    text_cols: str | list[str] | tuple[str, ...] | None = None,
    nlp_method: str | None = None,
    cluster_algo: str | None = None,
    n_clusters: int | None = None,
    dim_reduce: str | None = None,
    cluster_metric: str | None = None,
    anomaly_algo: str | None = None,
    contamination: float | str | None = None,
    event_col: str | None = None,
    time_col: str | None = None,
    query_col: str | None = None,
    user_col: str | None = None,
    item_col: str | None = None,
    rating_col: str | None = None,
    chunk_size: int | None = None,
    scale_mode: str | None = None,
    exclude: str | list[str] | tuple[str, ...] | None = None,
    datetime_col: str | None = None,
    horizon: int | None = None,
) -> list[str]:
    """Construct exact ``run_acdso.py`` argv matching CLI and WebUI contracts."""
    argv = [sys.executable, "-u", str(RUNNER), "--file", str(file_path)]
    task_norm = str(task or "auto").strip().lower()
    if task_norm and task_norm != "auto":
        argv.extend(["--task", task_norm])

    clean_target = str(target or "").strip()
    if clean_target and task_norm not in ("clustering", "anomaly", "recommender"):
        argv.extend(["--predict" if predict_mode else "--target", clean_target])

    if simple:
        argv.append("--simple")
    if skip_benchmark:
        argv.append("--skip-benchmark")
    if best_score:
        argv.append("--best-score")
    if force:
        argv.append("--force")

    str_flags: tuple[tuple[str, str | None], ...] = (
        ("--group-col", group_col),
        ("--cv-strategy", cv_strategy),
        ("--nlp-method", nlp_method),
        ("--cluster-algo", cluster_algo),
        ("--dim-reduce", dim_reduce),
        ("--cluster-metric", cluster_metric),
        ("--anomaly-algo", anomaly_algo),
        ("--event-col", event_col),
        ("--time-col", time_col),
        ("--query-col", query_col),
        ("--user-col", user_col),
        ("--item-col", item_col),
        ("--rating-col", rating_col),
        ("--scale-mode", scale_mode),
        ("--datetime-col", datetime_col),
    )
    for flag, raw_val in str_flags:
        if raw_val is not None and str(raw_val).strip():
            argv.extend([flag, str(raw_val).strip()])

    tcols_str = _normalize_csv_arg(text_cols)
    if tcols_str:
        argv.extend(["--text-cols", tcols_str])
    excl_str = _normalize_csv_arg(exclude)
    if excl_str:
        argv.extend(["--exclude", excl_str])

    if n_clusters is not None and not isinstance(n_clusters, bool):
        argv.extend(["--n-clusters", str(int(n_clusters))])
    if contamination is not None and not isinstance(contamination, bool) and str(contamination).strip():
        argv.extend(["--contamination", str(contamination).strip()])
    if chunk_size is not None and not isinstance(chunk_size, bool):
        argv.extend(["--chunk-size", str(int(chunk_size))])
    if horizon is not None and not isinstance(horizon, bool):
        argv.extend(["--horizon", str(int(horizon))])

    return argv


def load_latest_combined_summary(file_path: str | Path, target: str = "") -> dict[str, Any] | None:
    """Locate and parse the combined JSON report produced for (file_path, target)."""
    base = Path(file_path).stem
    raw_target = str(target or "").strip()
    clean_target = raw_target.split(",")[0].strip()
    candidates: list[Path] = []
    if "," in raw_target:
        candidates.append(REPORTS_DIR / f"autonomous_{base}_multilabel_combined.json")
    if clean_target:
        candidates.append(REPORTS_DIR / f"autonomous_{base}_{clean_target}_combined.json")
    candidates.append(REPORTS_DIR / f"autonomous_{base}_combined.json")
    if REPORTS_DIR.is_dir():
        globbed = sorted(
            REPORTS_DIR.glob(f"autonomous_{base}*_combined.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for gp in globbed:
            if gp not in candidates:
                candidates.append(gp)
    candidates.append(REPORTS_DIR / f"audit_{base}.json")

    chosen = next((c for c in candidates if c.is_file()), None)
    if chosen is None:
        return None
    try:
        data = json.loads(chosen.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    audit = data.get("audit_trail") if isinstance(data.get("audit_trail"), dict) else data
    sel = data.get("selected_model") or audit.get("selected") or audit.get("selected_candidate") or {}
    cands = (
        audit.get("candidates_evaluated")
        or audit.get("candidates")
        or data.get("candidates_evaluated")
        or data.get("candidates")
        or []
    )
    pareto = audit.get("pareto_front") or data.get("pareto_front") or []
    pareto_names = {
        (p if isinstance(p, str) else (p.get("model_name") or p.get("name") or ""))
        for p in pareto
    }
    return {
        "report_path": str(chosen),
        "task": data.get("task"),
        "metrics": data.get("metrics") or {},
        "selected_model": sel,
        "test_performance": data.get("test_performance") or {},
        "candidates": cands,
        "pareto_names": sorted(n for n in pareto_names if n),
        "clustering": data.get("clustering"),
        "anomaly": data.get("anomaly"),
        "cv": data.get("cv"),
        "nlp": data.get("nlp"),
        "specialized": data.get("specialized"),
        "scale": data.get("scale"),
        "summary": data.get("summary") or {},
    }


def _render_ascii_bar(pct: int, width: int = 24) -> str:
    filled = int(round((max(0, min(100, pct)) / 100.0) * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {pct:3d}%"


class FallbackTerminalUI:
    """Zero-dependency interactive terminal UI for ACDSO."""

    def __init__(
        self,
        initial_file: str | None = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self.file_path: str = initial_file or ""
        self.profile: dict[str, Any] | None = None
        self.target: str = ""
        self.task: str = "auto"
        self.predict_mode: bool = True
        self.simple: bool = True
        self.skip_benchmark: bool = True
        self.best_score: bool = False
        self.group_col: str = ""
        self.cv_strategy: str = "auto"
        self.text_cols: str = ""
        self.nlp_method: str = "auto"
        self.cluster_algo: str = "auto"
        self.n_clusters: int | None = None
        self.dim_reduce: str = "none"
        self.cluster_metric: str = "silhouette"
        self.anomaly_algo: str = "auto"
        self.contamination: str = "auto"
        self.event_col: str = ""
        self.time_col: str = ""
        self.query_col: str = ""
        self.user_col: str = ""
        self.item_col: str = ""
        self.rating_col: str = ""
        self.chunk_size: int | None = None
        self.scale_mode: str = "auto"
        self.last_summary: dict[str, Any] | None = None
        self._input = input_fn
        self._out = output_fn

    def _validate_before_run(self) -> tuple[bool, str]:
        return _validate_before_run(
            self.file_path,
            self.target,
            self.task,
            group_col=self.group_col,
            cv_strategy=self.cv_strategy,
            time_col=self.time_col,
            event_col=self.event_col,
            query_col=self.query_col,
            user_col=self.user_col,
            item_col=self.item_col,
            n_clusters=self.n_clusters,
            contamination=self.contamination,
            chunk_size=self.chunk_size,
        )

    def _load_file(self, path_str: str) -> None:
        self.profile = profile_dataset(path_str)
        self.file_path = self.profile["path"]
        self._out(f"Loaded {self.profile['name']} ({self.profile['rows']} rows x {self.profile['cols']} cols)")
        for col in self.profile["columns"]:
            status = "predictable" if col["predictable"] else f"excluded ({col['exclude_reason']})"
            self._out(f"  - {col['name']:<20} {col['dtype']:<10} uniq={col['nunique']:<5} null={col['null_pct']}%  [{status}]")
        if self.profile.get("text_cols"):
            self._out(f"  Detected text columns: {', '.join(self.profile['text_cols'])}")

    def _suggest_target(self) -> None:
        if not self.file_path:
            self._out("Load a dataset first.")
            return
        sugs = suggest_targets(self.file_path)
        if not sugs:
            self._out("No target suggestions available.")
            return
        self._out("Suggested targets:")
        for idx, s in enumerate(sugs, start=1):
            self._out(f"  {idx}. {s['column']} — {s.get('reason', '')}")
        self.target = sugs[0]["column"]
        self._out(f"Active target set to: {self.target}")

    def _configure_r1_r6(self) -> None:
        raw_task = self._input(f"Task ({'/'.join(TASK_CHOICES)}) [{self.task}]: ").strip().lower()
        if raw_task:
            if raw_task in TASK_CHOICES:
                self.task = raw_task
            else:
                self._out(f"Unknown task '{raw_task}', keeping {self.task}.")
        t_norm = self.task
        grp = self._input(f"Group column [{self.group_col or 'none'}]: ").strip()
        if grp:
            self.group_col = "" if grp.lower() == "none" else grp
        cvs = self._input(f"CV strategy ({'/'.join(CV_STRATEGIES)}) [{self.cv_strategy}]: ").strip().lower()
        if cvs in CV_STRATEGIES:
            self.cv_strategy = cvs
        tcols = self._input(f"Text columns (comma-sep) [{self.text_cols or 'auto'}]: ").strip()
        if tcols:
            self.text_cols = "" if tcols.lower() in ("auto", "none") else tcols
        if t_norm == "clustering":
            ca = self._input(f"Cluster algo ({'/'.join(CLUSTER_ALGOS)}) [{self.cluster_algo}]: ").strip().lower()
            if ca in CLUSTER_ALGOS:
                self.cluster_algo = ca
            nc = self._input(f"n_clusters [{self.n_clusters or 'auto'}]: ").strip()
            if nc:
                if nc.lower() == "auto":
                    self.n_clusters = None
                else:
                    try:
                        self.n_clusters = int(nc)
                    except ValueError:
                        self._out("Invalid integer for n_clusters.")
        elif t_norm == "anomaly":
            aa = self._input(f"Anomaly algo ({'/'.join(ANOMALY_ALGOS)}) [{self.anomaly_algo}]: ").strip().lower()
            if aa in ANOMALY_ALGOS:
                self.anomaly_algo = aa
            ct = self._input(f"Contamination [{self.contamination}]: ").strip()
            if ct:
                self.contamination = ct
        elif t_norm == "survival":
            tc = self._input(f"Time column [{self.time_col or 'none'}]: ").strip()
            if tc:
                self.time_col = "" if tc.lower() == "none" else tc
            ec = self._input(f"Event column [{self.event_col or 'none'}]: ").strip()
            if ec:
                self.event_col = "" if ec.lower() == "none" else ec
        elif t_norm == "ranking":
            qc = self._input(f"Query column [{self.query_col or 'none'}]: ").strip()
            if qc:
                self.query_col = "" if qc.lower() == "none" else qc
        elif t_norm == "recommender":
            uc = self._input(f"User column [{self.user_col or 'none'}]: ").strip()
            if uc:
                self.user_col = "" if uc.lower() == "none" else uc
            ic = self._input(f"Item column [{self.item_col or 'none'}]: ").strip()
            if ic:
                self.item_col = "" if ic.lower() == "none" else ic
            rc = self._input(f"Rating column [{self.rating_col or 'none'}]: ").strip()
            if rc:
                self.rating_col = "" if rc.lower() == "none" else rc
        sm = self._input(f"Scale mode ({'/'.join(SCALE_MODES)}) [{self.scale_mode}]: ").strip().lower()
        if sm in SCALE_MODES:
            self.scale_mode = sm

    def _run_pipeline(self) -> int:
        ok, err_msg = self._validate_before_run()
        if not ok:
            self._out(err_msg)
            return 1

        argv = build_cli_command(
            self.file_path,
            self.target,
            predict_mode=self.predict_mode,
            simple=self.simple,
            skip_benchmark=self.skip_benchmark,
            best_score=self.best_score,
            task=self.task,
            group_col=self.group_col or None,
            cv_strategy=self.cv_strategy if self.cv_strategy != "auto" else None,
            text_cols=self.text_cols or None,
            nlp_method=self.nlp_method if self.nlp_method != "auto" else None,
            cluster_algo=self.cluster_algo if self.cluster_algo != "auto" else None,
            n_clusters=self.n_clusters,
            dim_reduce=self.dim_reduce if self.dim_reduce != "none" else None,
            cluster_metric=self.cluster_metric if self.cluster_metric != "silhouette" else None,
            anomaly_algo=self.anomaly_algo if self.anomaly_algo != "auto" else None,
            contamination=self.contamination if self.contamination != "auto" else None,
            event_col=self.event_col or None,
            time_col=self.time_col or None,
            query_col=self.query_col or None,
            user_col=self.user_col or None,
            item_col=self.item_col or None,
            rating_col=self.rating_col or None,
            chunk_size=self.chunk_size,
            scale_mode=self.scale_mode if self.scale_mode != "auto" else None,
        )
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        self._out("Running: " + " ".join(argv))
        proc = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            prog = parse_progress_marker(line)
            if prog is not None:
                bar = _render_ascii_bar(prog["percent"])
                self._out(f"  {bar} {prog['name']} ({prog['status']})")
        rc = proc.wait()
        self._out(f"Pipeline exited with code {rc}")
        if rc == 0:
            task_norm = str(self.task or "auto").strip().lower()
            lookup_target = (
                ""
                if (
                    task_norm in ("clustering", "anomaly", "recommender")
                    or task_norm == "survival"
                )
                else self.target
            )
            self.last_summary = load_latest_combined_summary(self.file_path, lookup_target)
            self._show_results()
        return rc

    def _show_results(self) -> None:
        if not self.last_summary:
            self._out("No completed run summary loaded yet.")
            return
        task_name = self.last_summary.get("task") or self.task
        if task_name:
            self._out(f"Task: {task_name}")
        sel = self.last_summary.get("selected_model") or {}
        metrics = sel.get("metrics") or {}
        task_metrics = self.last_summary.get("metrics") or {}
        if sel or task_metrics:
            score_val = metrics.get("metric", sel.get("metric"))
            if score_val is None:
                for mk in ("silhouette_score", "f1_micro", "ndcg_at_k", "c_index", "rmse", "accuracy", "r2", "n_anomalies"):
                    if mk in metrics:
                        score_val = metrics[mk]
                        break
                    if isinstance(task_metrics, dict) and mk in task_metrics:
                        score_val = task_metrics[mk]
                        break
            sel_name = sel.get("name") or (f"{task_name} pipeline" if task_name else "Unknown")
            score_txt = (
                f"{float(score_val):.4f}"
                if isinstance(score_val, (int, float)) and not isinstance(score_val, bool) and math.isfinite(float(score_val))
                else "n/a"
            )
            self._out(f"Selected Model: {sel_name} | CV Score: {score_txt}")

        task_metrics = self.last_summary.get("metrics") or {}
        if isinstance(task_metrics, dict) and task_metrics:
            summary_pairs = ", ".join(f"{k}={v}" for k, v in list(task_metrics.items())[:6])
            self._out(f"Metrics: {summary_pairs}")

        for block_key, label in (
            ("clustering", "Clustering"),
            ("anomaly", "Anomaly"),
            ("cv", "CV"),
            ("nlp", "NLP"),
            ("specialized", "Specialized"),
            ("scale", "Scale"),
        ):
            block = self.last_summary.get(block_key)
            if isinstance(block, dict) and block:
                pairs = ", ".join(f"{k}={v}" for k, v in list(block.items())[:6] if not isinstance(v, (dict, list)))
                if pairs:
                    self._out(f"{label}: {pairs}")

        cands = self.last_summary.get("candidates") or []
        pareto_set = set(self.last_summary.get("pareto_names") or [])
        if cands:
            self._out(f"  {'Model':<26} {'Score':>8} {'Train(s)':>9} {'Mem(MB)':>8} {'Risk':>7}  Flags")
            for c in cands[:12]:
                if not isinstance(c, dict):
                    continue
                name = str(c.get("model_name") or c.get("name") or c.get("algorithm") or c.get("strategy") or c.get("method") or "")
                flags = "PARETO" if name in pareto_set else ""
                score = c.get("metric")
                if score is None:
                    for sk in ("silhouette_score", "cv_ndcg_at_k", "cv_c_index", "cv_objective", "objective", "rmse", "score"):
                        if c.get(sk) is not None:
                            score = c.get(sk)
                            break
                sc_s = f"{float(score):.4f}" if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)) else "n/a"
                tt = c.get("train_time_s")
                tt_s = f"{float(tt):.2f}" if isinstance(tt, (int, float)) and not isinstance(tt, bool) and math.isfinite(float(tt)) else "n/a"
                mm = c.get("peak_mem_mb")
                mm_s = f"{float(mm):.1f}" if isinstance(mm, (int, float)) and not isinstance(mm, bool) and math.isfinite(float(mm)) else "n/a"
                rk = c.get("risk_penalty")
                rk_s = f"{float(rk):.3f}" if isinstance(rk, (int, float)) and not isinstance(rk, bool) and math.isfinite(float(rk)) else "n/a"
                self._out(f"  {name:<26} {sc_s:>8} {tt_s:>9} {mm_s:>8} {rk_s:>7}  {flags}")

    def run(self) -> int:
        self._out("=" * 68)
        self._out("  ACDSO Interactive Terminal UI (Fallback Mode)")
        self._out("=" * 68)
        if self.file_path:
            try:
                self._load_file(self.file_path)
            except Exception as exc:
                self._out(f"[WARN] Could not pre-load {self.file_path}: {exc}")

        while True:
            self._out(
                f"\nDataset: {self.file_path or '(none)'} | Task: {self.task} | Target: {self.target or '(none)'} | "
                f"Mode: {'--predict' if self.predict_mode else '--target'} | --simple={self.simple}"
            )
            self._out(
                "[1] Load dataset  [2] Suggest target  [3] Set target  [4] Toggle --simple  "
                "[5] Run  [6] Results  [7] Task/R1-R6 opts  [8] Clear target  [q] Quit"
            )
            try:
                choice = self._input("acdso> ").strip().lower()
            except EOFError:
                return 0
            if choice in ("q", "quit", "exit"):
                return 0
            if choice == "1":
                path = self._input("File path: ").strip().strip('"')
                if path:
                    try:
                        self._load_file(path)
                    except Exception as exc:
                        self._out(f"[ERROR] {exc}")
            elif choice == "2":
                try:
                    self._suggest_target()
                except Exception as exc:
                    self._out(f"[ERROR] {exc}")
            elif choice == "3":
                t = self._input("Target column(s): ").strip()
                if t:
                    self.target = t
            elif choice == "4":
                self.simple = not self.simple
                self._out(f"Quick mode (--simple) = {self.simple}")
            elif choice == "5":
                self._run_pipeline()
            elif choice == "6":
                self._show_results()
            elif choice == "7":
                self._configure_r1_r6()
            elif choice == "8":
                self.target = ""
                self._out("Target cleared.")
            else:
                self._out("Unknown option. Enter 1-8 or q.")


def create_textual_app(initial_file: str | None = None):
    """Build and return the full-screen Textual ``ACDSOTuiApp`` instance."""
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import (
        Button,
        Checkbox,
        DataTable,
        Footer,
        Header,
        Input,
        ProgressBar,
        RichLog,
        Select,
        Static,
        TabbedContent,
        TabPane,
    )

    class ACDSOTuiApp(App):
        TITLE = "ACDSO — Interactive AutoML TUI"
        SUB_TITLE = "Risk-Aware Multi-Objective AutoML"
        CSS = """
        Screen { layout: vertical; }
        #body { height: 1fr; }
        #left-pane { width: 42%; border: round $primary; padding: 1; }
        #right-pane { width: 58%; layout: vertical; }
        #config-pane { height: auto; border: round $accent; padding: 0 1; }
        #config-tabs { height: auto; }
        #output-pane { height: 1fr; border: round $secondary; padding: 1; }
        .row { height: auto; margin-bottom: 0; }
        .row Input { width: 1fr; margin-right: 1; }
        .row Select { width: 1fr; margin-right: 1; }
        .row Button { margin-right: 1; }
        #cols-table { height: 1fr; }
        #pareto-table { height: 1fr; }
        #log-view { height: 1fr; }
        #kpi-banner { height: auto; margin-bottom: 1; color: $success; }
        """
        BINDINGS = [
            Binding("r", "run_pipeline", "Run"),
            Binding("x", "stop_pipeline", "Stop"),
            Binding("s", "suggest_target", "Suggest"),
            Binding("q", "quit", "Quit"),
        ]

        def __init__(self, start_file: str | None = None) -> None:
            super().__init__()
            self.initial_file = start_file or ""
            self.current_profile: dict[str, Any] | None = None
            self._proc: asyncio.subprocess.Process | None = None

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="body"):
                with Vertical(id="left-pane"):
                    yield Static("1. Dataset & Columns", classes="section-title")
                    with Horizontal(classes="row"):
                        yield Input(
                            value=self.initial_file,
                            placeholder="Path to CSV / XLSX / Parquet / JSON...",
                            id="file-input",
                        )
                        yield Button("Load", id="load-btn", variant="primary")
                    yield Static("No dataset loaded.", id="dataset-meta")
                    yield DataTable(id="cols-table", cursor_type="row")

                with Vertical(id="right-pane"):
                    with Vertical(id="config-pane"):
                        yield Static("2. Task, Target & Options", classes="section-title")
                        with TabbedContent(id="config-tabs"):
                            with TabPane("Task & Target", id="tab-cfg-main"):
                                with Horizontal(classes="row"):
                                    yield Select(
                                        [(t, t) for t in TASK_CHOICES],
                                        value="auto",
                                        allow_blank=False,
                                        id="task-select",
                                    )
                                    yield Input(placeholder="Target column(s)...", id="target-input")
                                    yield Button("Suggest (s)", id="suggest-btn")
                                with Horizontal(classes="row"):
                                    yield Checkbox("Predict (--predict)", value=True, id="chk-predict")
                                    yield Checkbox("Quick (--simple)", value=True, id="chk-simple")
                                    yield Checkbox("Skip Bench", value=True, id="chk-skip-bench")
                                    yield Checkbox("Best Score", value=False, id="chk-best-score")
                            with TabPane("CV / NLP / Scale", id="tab-cfg-cv"):
                                with Horizontal(classes="row"):
                                    yield Input(placeholder="Group col (--group-col)", id="group-col-input")
                                    yield Select(
                                        [(c, c) for c in CV_STRATEGIES],
                                        value="auto",
                                        allow_blank=False,
                                        id="cv-strategy-select",
                                    )
                                with Horizontal(classes="row"):
                                    yield Input(placeholder="Text cols (comma-sep)", id="text-cols-input")
                                    yield Select(
                                        [(m, m) for m in NLP_METHODS],
                                        value="auto",
                                        allow_blank=False,
                                        id="nlp-method-select",
                                    )
                                with Horizontal(classes="row"):
                                    yield Select(
                                        [(s, s) for s in SCALE_MODES],
                                        value="auto",
                                        allow_blank=False,
                                        id="scale-mode-select",
                                    )
                                    yield Input(placeholder="Chunk size (50000)", id="chunk-size-input")
                            with TabPane("Cluster / Anomaly / Spec", id="tab-cfg-spec"):
                                with Horizontal(classes="row"):
                                    yield Select(
                                        [(a, a) for a in CLUSTER_ALGOS],
                                        value="auto",
                                        allow_blank=False,
                                        id="cluster-algo-select",
                                    )
                                    yield Input(placeholder="n_clusters (>=2)", id="n-clusters-input")
                                    yield Select(
                                        [(d, d) for d in DIM_REDUCE_MODES],
                                        value="none",
                                        allow_blank=False,
                                        id="dim-reduce-select",
                                    )
                                    yield Select(
                                        [(m, m) for m in CLUSTER_METRICS],
                                        value="silhouette",
                                        allow_blank=False,
                                        id="cluster-metric-select",
                                    )
                                with Horizontal(classes="row"):
                                    yield Select(
                                        [(a, a) for a in ANOMALY_ALGOS],
                                        value="auto",
                                        allow_blank=False,
                                        id="anomaly-algo-select",
                                    )
                                    yield Input(placeholder="Contamination (auto or <=0.5)", id="contamination-input")
                                    yield Input(placeholder="Time col", id="time-col-input")
                                    yield Input(placeholder="Event col", id="event-col-input")
                                with Horizontal(classes="row"):
                                    yield Input(placeholder="Query col", id="query-col-input")
                                    yield Input(placeholder="User col", id="user-col-input")
                                    yield Input(placeholder="Item col", id="item-col-input")
                                    yield Input(placeholder="Rating col", id="rating-col-input")

                        with Horizontal(classes="row"):
                            yield Button("Run Pipeline (r)", id="run-btn", variant="success")
                            yield Button("Stop (x)", id="stop-btn", variant="error", disabled=True)
                            yield Static("Idle", id="run-status")

                    with Vertical(id="output-pane"):
                        with TabbedContent():
                            with TabPane("Live Progress & Log", id="tab-log"):
                                yield ProgressBar(total=100, show_eta=False, id="prog-bar")
                                yield Static("Waiting for run...", id="prog-label")
                                yield RichLog(id="log-view", highlight=False, markup=False)
                            with TabPane("Pareto & Results", id="tab-results"):
                                yield Static("Run a pipeline to inspect KPIs and Pareto models.", id="kpi-banner")
                                yield DataTable(id="pareto-table", cursor_type="row")
            yield Footer()

        def on_mount(self) -> None:
            cols_table = self.query_one("#cols-table", DataTable)
            cols_table.add_columns("Column", "Dtype", "Unique", "Null%", "Status")
            pareto_table = self.query_one("#pareto-table", DataTable)
            pareto_table.add_columns("Model", "CV Score", "Train(s)", "Mem(MB)", "Risk", "Flags")
            if self.initial_file:
                self._load_dataset(self.initial_file)

        def _load_dataset(self, path_str: str) -> None:
            meta = self.query_one("#dataset-meta", Static)
            table = self.query_one("#cols-table", DataTable)
            table.clear()
            try:
                prof = profile_dataset(path_str)
            except Exception as exc:
                meta.update(f"[Error] {exc}")
                return
            self.current_profile = prof
            meta.update(f"{prof['name']} — {prof['rows']:,} rows x {prof['cols']} cols ({len(prof['predictable'])} predictable)")
            for col in prof["columns"]:
                st = "ok" if col["predictable"] else (col["exclude_reason"] or "excluded")
                table.add_row(col["name"], col["dtype"], str(col["nunique"]), f"{col['null_pct']}%", st, key=col["name"])

        def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
            if event.data_table.id == "cols-table" and event.row_key.value:
                self.query_one("#target-input", Input).value = str(event.row_key.value)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            bid = event.button.id
            if bid == "load-btn":
                self._load_dataset(self.query_one("#file-input", Input).value.strip().strip('"'))
            elif bid == "suggest-btn":
                self.action_suggest_target()
            elif bid == "run-btn":
                self.action_run_pipeline()
            elif bid == "stop-btn":
                self.action_stop_pipeline()

        def action_suggest_target(self) -> None:
            path_str = self.query_one("#file-input", Input).value.strip().strip('"')
            status = self.query_one("#run-status", Static)
            if not path_str:
                status.update("Load a dataset first.")
                return
            try:
                sugs = suggest_targets(path_str)
                if sugs:
                    top = sugs[0]["column"]
                    self.query_one("#target-input", Input).value = top
                    status.update(f"Suggested: {', '.join(s['column'] for s in sugs)}")
                else:
                    status.update("No suggestions found.")
            except Exception as exc:
                status.update(f"Suggest error: {exc}")

        def _read_int_input(self, widget_id: str) -> int | None:
            raw = self.query_one(widget_id, Input).value.strip()
            if not raw:
                return None
            return int(raw)

        def action_run_pipeline(self) -> None:
            if self._proc is not None and self._proc.returncode is None:
                return
            path_str = self.query_one("#file-input", Input).value.strip().strip('"')
            target = self.query_one("#target-input", Input).value.strip()
            task_val = str(self.query_one("#task-select", Select).value or "auto")
            status = self.query_one("#run-status", Static)

            group_col = self.query_one("#group-col-input", Input).value.strip()
            cv_strategy = str(self.query_one("#cv-strategy-select", Select).value or "auto")
            text_cols = self.query_one("#text-cols-input", Input).value.strip()
            nlp_method = str(self.query_one("#nlp-method-select", Select).value or "auto")
            scale_mode = str(self.query_one("#scale-mode-select", Select).value or "auto")
            cluster_algo = str(self.query_one("#cluster-algo-select", Select).value or "auto")
            dim_reduce = str(self.query_one("#dim-reduce-select", Select).value or "none")
            cluster_metric = str(self.query_one("#cluster-metric-select", Select).value or "silhouette")
            anomaly_algo = str(self.query_one("#anomaly-algo-select", Select).value or "auto")
            contamination = self.query_one("#contamination-input", Input).value.strip()
            time_col = self.query_one("#time-col-input", Input).value.strip()
            event_col = self.query_one("#event-col-input", Input).value.strip()
            query_col = self.query_one("#query-col-input", Input).value.strip()
            user_col = self.query_one("#user-col-input", Input).value.strip()
            item_col = self.query_one("#item-col-input", Input).value.strip()
            rating_col = self.query_one("#rating-col-input", Input).value.strip()

            try:
                n_clusters = self._read_int_input("#n-clusters-input")
            except ValueError:
                status.update("Invalid n_clusters integer.")
                return
            try:
                chunk_size = self._read_int_input("#chunk-size-input")
            except ValueError:
                status.update("Invalid chunk_size integer.")
                return

            ok, err_msg = _validate_before_run(
                path_str,
                target,
                task_val,
                group_col=group_col,
                cv_strategy=cv_strategy,
                time_col=time_col,
                event_col=event_col,
                query_col=query_col,
                user_col=user_col,
                item_col=item_col,
                n_clusters=n_clusters,
                contamination=contamination or None,
                chunk_size=chunk_size,
            )
            if not ok:
                status.update(err_msg)
                return

            argv = build_cli_command(
                path_str,
                target,
                predict_mode=self.query_one("#chk-predict", Checkbox).value,
                simple=self.query_one("#chk-simple", Checkbox).value,
                skip_benchmark=self.query_one("#chk-skip-bench", Checkbox).value,
                best_score=self.query_one("#chk-best-score", Checkbox).value,
                task=task_val,
                group_col=group_col or None,
                cv_strategy=cv_strategy if cv_strategy != "auto" else None,
                text_cols=text_cols or None,
                nlp_method=nlp_method if nlp_method != "auto" else None,
                cluster_algo=cluster_algo if cluster_algo != "auto" else None,
                n_clusters=n_clusters,
                dim_reduce=dim_reduce if dim_reduce != "none" else None,
                cluster_metric=cluster_metric if cluster_metric != "silhouette" else None,
                anomaly_algo=anomaly_algo if anomaly_algo != "auto" else None,
                contamination=contamination or None,
                time_col=time_col or None,
                event_col=event_col or None,
                query_col=query_col or None,
                user_col=user_col or None,
                item_col=item_col or None,
                rating_col=rating_col or None,
                chunk_size=chunk_size,
                scale_mode=scale_mode if scale_mode != "auto" else None,
            )
            task_norm = str(task_val or "auto").strip().lower()
            lookup_target = "" if task_norm in ("clustering", "anomaly", "recommender", "survival") else target
            self.run_worker(self._run_worker(argv, path_str, lookup_target), exclusive=True)

        async def _run_worker(self, argv: list[str], path_str: str, target: str) -> None:
            run_btn = self.query_one("#run-btn", Button)
            stop_btn = self.query_one("#stop-btn", Button)
            status = self.query_one("#run-status", Static)
            bar = self.query_one("#prog-bar", ProgressBar)
            plbl = self.query_one("#prog-label", Static)
            log_view = self.query_one("#log-view", RichLog)

            run_btn.disabled = True
            stop_btn.disabled = False
            log_view.clear()
            bar.update(progress=0)
            status.update("Running...")
            plbl.update("Launching pipeline...")

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONPATH"] = str(REPO_ROOT / "src")

            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(REPO_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            assert self._proc.stdout is not None
            while True:
                raw = await self._proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                prog = parse_progress_marker(line)
                if prog is not None:
                    bar.update(progress=prog["percent"])
                    plbl.update(f"[{prog['phase']}/{prog['total']}] {prog['name']} ({prog['status']})")
                else:
                    log_view.write(line)

            rc = await self._proc.wait()
            run_btn.disabled = False
            stop_btn.disabled = True
            status.update(f"Finished (exit {rc})" if rc == 0 else f"Failed (exit {rc})")
            if rc == 0:
                bar.update(progress=100)
                eff_target = target
                if "--task" in argv:
                    idx = argv.index("--task")
                    if idx + 1 < len(argv) and str(argv[idx + 1]).strip().lower() in (
                        "clustering",
                        "anomaly",
                        "recommender",
                        "survival",
                    ):
                        eff_target = ""
                self._populate_results(path_str, eff_target)

        def _populate_results(self, path_str: str, target: str = "") -> None:
            summary = load_latest_combined_summary(path_str, target)
            banner = self.query_one("#kpi-banner", Static)
            table = self.query_one("#pareto-table", DataTable)
            table.clear()
            if not summary:
                banner.update("Run completed (no combined JSON report found).")
                return
            sel = summary.get("selected_model") or {}
            metrics = sel.get("metrics") or {}
            task_metrics = summary.get("metrics") or {}
            task_label = summary.get("task") or ""
            sc = metrics.get("metric", sel.get("metric"))
            if sc is None:
                for mk in ("silhouette_score", "f1_micro", "ndcg_at_k", "c_index", "rmse", "accuracy", "r2", "n_anomalies"):
                    if mk in metrics:
                        sc = metrics[mk]
                        break
                    if isinstance(task_metrics, dict) and mk in task_metrics:
                        sc = task_metrics[mk]
                        break
            sc_txt = f"{float(sc):.4f}" if isinstance(sc, (int, float)) and not isinstance(sc, bool) and math.isfinite(float(sc)) else "n/a"
            sel_name = str(sel.get("name") or (f"{task_label} pipeline" if task_label else "Unknown"))
            task_prefix = f"Task: {task_label} | " if task_label else ""
            banner.update(f"{task_prefix}Selected: {sel_name} | CV Score: {sc_txt} | Report: {Path(summary['report_path']).name}")
            pareto_set = set(summary.get("pareto_names") or [])
            cands = summary.get("candidates") or []
            for c in cands:
                if not isinstance(c, dict):
                    continue
                name = str(c.get("model_name") or c.get("name") or c.get("algorithm") or c.get("strategy") or c.get("method") or "")
                flags = " ".join(f for f in ("SELECTED" if name == sel_name else "", "PARETO" if name in pareto_set else "") if f)
                m_val = c.get("metric")
                if m_val is None:
                    for sk in ("silhouette_score", "cv_ndcg_at_k", "cv_c_index", "cv_objective", "objective", "rmse", "score"):
                        if c.get(sk) is not None:
                            m_val = c.get(sk)
                            break
                tt = c.get("train_time_s")
                mm = c.get("peak_mem_mb")
                rk = c.get("risk_penalty")
                table.add_row(
                    name,
                    f"{float(m_val):.4f}" if isinstance(m_val, (int, float)) and not isinstance(m_val, bool) and math.isfinite(float(m_val)) else "n/a",
                    f"{float(tt):.2f}" if isinstance(tt, (int, float)) and not isinstance(tt, bool) and math.isfinite(float(tt)) else "n/a",
                    f"{float(mm):.1f}" if isinstance(mm, (int, float)) and not isinstance(mm, bool) and math.isfinite(float(mm)) else "n/a",
                    f"{float(rk):.3f}" if isinstance(rk, (int, float)) and not isinstance(rk, bool) and math.isfinite(float(rk)) else "n/a",
                    flags,
                )
            if not cands:
                for block_key, label in (
                    ("clustering", "Clustering"),
                    ("anomaly", "Anomaly"),
                    ("specialized", "Specialized"),
                    ("scale", "Scale"),
                ):
                    blk = summary.get(block_key)
                    if isinstance(blk, dict) and blk:
                        table.add_row(label, sc_txt, "n/a", "n/a", "n/a", str(blk.get("best_algorithm") or blk.get("mode") or blk.get("best_method") or ""))

        def action_stop_pipeline(self) -> None:
            if self._proc is not None and self._proc.returncode is None:
                try:
                    self._proc.terminate()
                except ProcessLookupError:
                    pass
                self.query_one("#run-status", Static).update("Stopped by user")

    return ACDSOTuiApp(start_file=initial_file)


def run_textual_tui(initial_file: str | None = None) -> int:
    """Run the full-screen Textual TUI app."""
    app = create_textual_app(initial_file=initial_file)
    app.run()
    return 0
