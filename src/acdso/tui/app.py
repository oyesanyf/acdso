"""ACDSO Interactive Terminal UI (Textual dashboard + zero-dependency fallback)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "run_acdso.py"
REPORTS_DIR = REPO_ROOT / "reports"


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
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    phase = max(0, int(data.get("phase") or 0))
    total = max(1, int(data.get("total") or 1))
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
    """Load and summarize a dataset for TUI browsing (lazy pandas import)."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    from acdso.cli.io import load_user_file
    from acdso.webui.column_validator import analyze_predictable_columns

    df = load_user_file(str(p))
    total_rows = int(len(df))
    sample = df.head(max_rows)
    predictable, excluded = analyze_predictable_columns(sample)
    pred_set = set(predictable)

    cols: list[dict[str, Any]] = []
    for c in df.columns:
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
        "cols": int(len(df.columns)),
        "columns": cols,
        "predictable": predictable,
    }


def suggest_targets(file_path: str | Path, top_n: int = 3) -> list[dict[str, Any]]:
    """Return ranked target suggestions using ACDSO's shared column validator."""
    from acdso.cli.io import load_user_file
    from acdso.webui.column_validator import analyze_predictable_columns, suggest_target_columns

    df = load_user_file(str(file_path)).head(1000)
    candidates, _ = analyze_predictable_columns(df)
    res = suggest_target_columns(df, candidates, top_n=top_n)
    return list(res.get("suggestions") or [])


def build_cli_command(
    file_path: str,
    target: str,
    predict_mode: bool = True,
    simple: bool = True,
    skip_benchmark: bool = True,
    best_score: bool = False,
    force: bool = False,
) -> list[str]:
    """Construct exact ``run_acdso.py`` argv matching CLI and WebUI contracts."""
    argv = [sys.executable, "-u", str(RUNNER), "--file", str(file_path)]
    if target.strip():
        argv.extend(["--predict" if predict_mode else "--target", target.strip()])
    if simple:
        argv.append("--simple")
    if skip_benchmark:
        argv.append("--skip-benchmark")
    if best_score:
        argv.append("--best-score")
    if force:
        argv.append("--force")
    return argv


def load_latest_combined_summary(file_path: str | Path, target: str) -> dict[str, Any] | None:
    """Locate and parse the combined JSON report produced for (file_path, target)."""
    base = Path(file_path).stem
    clean_target = target.split(",")[0].strip()
    candidates = [
        REPORTS_DIR / f"autonomous_{base}_{clean_target}_combined.json",
        REPORTS_DIR / f"autonomous_{base}_combined.json",
        REPORTS_DIR / f"audit_{base}.json",
    ]
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
    sel = data.get("selected_model") or audit.get("selected") or {}
    cands = audit.get("candidates_evaluated") or data.get("candidates_evaluated") or []
    pareto = audit.get("pareto_front") or data.get("pareto_front") or []
    pareto_names = {
        (p if isinstance(p, str) else (p.get("model_name") or p.get("name") or ""))
        for p in pareto
    }
    return {
        "report_path": str(chosen),
        "selected_model": sel,
        "test_performance": data.get("test_performance") or {},
        "candidates": cands,
        "pareto_names": sorted(n for n in pareto_names if n),
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
        self.predict_mode: bool = True
        self.simple: bool = True
        self.skip_benchmark: bool = True
        self.best_score: bool = False
        self.last_summary: dict[str, Any] | None = None
        self._input = input_fn
        self._out = output_fn

    def _load_file(self, path_str: str) -> None:
        self.profile = profile_dataset(path_str)
        self.file_path = self.profile["path"]
        self._out(f"Loaded {self.profile['name']} ({self.profile['rows']} rows x {self.profile['cols']} cols)")
        for col in self.profile["columns"]:
            status = "predictable" if col["predictable"] else f"excluded ({col['exclude_reason']})"
            self._out(f"  - {col['name']:<20} {col['dtype']:<10} uniq={col['nunique']:<5} null={col['null_pct']}%  [{status}]")

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

    def _run_pipeline(self) -> int:
        if not self.file_path or not self.target:
            self._out("Set both a dataset file and a target column before running.")
            return 1
        argv = build_cli_command(
            self.file_path,
            self.target,
            predict_mode=self.predict_mode,
            simple=self.simple,
            skip_benchmark=self.skip_benchmark,
            best_score=self.best_score,
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
            self.last_summary = load_latest_combined_summary(self.file_path, self.target)
            self._show_results()
        return rc

    def _show_results(self) -> None:
        if not self.last_summary:
            self._out("No completed run summary loaded yet.")
            return
        sel = self.last_summary.get("selected_model") or {}
        metrics = sel.get("metrics") or {}
        self._out(f"Selected Model: {sel.get('name', 'Unknown')} | CV Score: {metrics.get('metric', sel.get('metric', 'n/a'))}")
        cands = self.last_summary.get("candidates") or []
        pareto_set = set(self.last_summary.get("pareto_names") or [])
        if cands:
            self._out(f"  {'Model':<26} {'Score':>8} {'Train(s)':>9} {'Mem(MB)':>8} {'Risk':>7}  Flags")
            for c in cands[:12]:
                name = str(c.get("model_name") or c.get("name") or "")
                flags = "PARETO" if name in pareto_set else ""
                score = c.get("metric")
                sc_s = f"{float(score):.4f}" if isinstance(score, (int, float)) else "n/a"
                tt = c.get("train_time_s")
                tt_s = f"{float(tt):.2f}" if isinstance(tt, (int, float)) else "n/a"
                mm = c.get("peak_mem_mb")
                mm_s = f"{float(mm):.1f}" if isinstance(mm, (int, float)) else "n/a"
                rk = c.get("risk_penalty")
                rk_s = f"{float(rk):.3f}" if isinstance(rk, (int, float)) else "n/a"
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
                f"\nDataset: {self.file_path or '(none)'} | Target: {self.target or '(none)'} | "
                f"Mode: {'--predict' if self.predict_mode else '--target'} | --simple={self.simple}"
            )
            self._out("[1] Load dataset  [2] Suggest target  [3] Set target  [4] Toggle --simple  [5] Run  [6] Results  [q] Quit")
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
            else:
                self._out("Unknown option. Enter 1-6 or q.")


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
        #config-pane { height: auto; border: round $accent; padding: 1; }
        #output-pane { height: 1fr; border: round $secondary; padding: 1; }
        .row { height: auto; margin-bottom: 1; }
        .row Input { width: 1fr; margin-right: 1; }
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
                        yield Static("2. Target & Options", classes="section-title")
                        with Horizontal(classes="row"):
                            yield Input(placeholder="Target column(s)...", id="target-input")
                            yield Button("Suggest (s)", id="suggest-btn")
                        with Horizontal(classes="row"):
                            yield Checkbox("Predict (--predict)", value=True, id="chk-predict")
                            yield Checkbox("Quick (--simple)", value=True, id="chk-simple")
                            yield Checkbox("Skip Bench", value=True, id="chk-skip-bench")
                            yield Checkbox("Best Score", value=False, id="chk-best-score")
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

        def action_run_pipeline(self) -> None:
            if self._proc is not None and self._proc.returncode is None:
                return
            path_str = self.query_one("#file-input", Input).value.strip().strip('"')
            target = self.query_one("#target-input", Input).value.strip()
            status = self.query_one("#run-status", Static)
            if not path_str or not target:
                status.update("Specify both file and target first.")
                return
            argv = build_cli_command(
                path_str,
                target,
                predict_mode=self.query_one("#chk-predict", Checkbox).value,
                simple=self.query_one("#chk-simple", Checkbox).value,
                skip_benchmark=self.query_one("#chk-skip-bench", Checkbox).value,
                best_score=self.query_one("#chk-best-score", Checkbox).value,
            )
            self.run_worker(self._run_worker(argv, path_str, target), exclusive=True)

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
                self._populate_results(path_str, target)

        def _populate_results(self, path_str: str, target: str) -> None:
            summary = load_latest_combined_summary(path_str, target)
            banner = self.query_one("#kpi-banner", Static)
            table = self.query_one("#pareto-table", DataTable)
            table.clear()
            if not summary:
                banner.update("Run completed (no combined JSON report found).")
                return
            sel = summary.get("selected_model") or {}
            metrics = sel.get("metrics") or {}
            sc = metrics.get("metric", sel.get("metric"))
            sc_txt = f"{float(sc):.4f}" if isinstance(sc, (int, float)) else "n/a"
            banner.update(f"Selected: {sel.get('name', 'Unknown')} | CV Score: {sc_txt} | Report: {Path(summary['report_path']).name}")
            pareto_set = set(summary.get("pareto_names") or [])
            sel_name = str(sel.get("name") or "")
            for c in summary.get("candidates") or []:
                name = str(c.get("model_name") or c.get("name") or "")
                flags = " ".join(f for f in ("SELECTED" if name == sel_name else "", "PARETO" if name in pareto_set else "") if f)
                m_val = c.get("metric")
                tt = c.get("train_time_s")
                mm = c.get("peak_mem_mb")
                rk = c.get("risk_penalty")
                table.add_row(
                    name,
                    f"{float(m_val):.4f}" if isinstance(m_val, (int, float)) else "n/a",
                    f"{float(tt):.2f}" if isinstance(tt, (int, float)) else "n/a",
                    f"{float(mm):.1f}" if isinstance(mm, (int, float)) else "n/a",
                    f"{float(rk):.3f}" if isinstance(rk, (int, float)) else "n/a",
                    flags,
                )

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
