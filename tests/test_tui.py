"""Tests for ACDSO Interactive Terminal UI (Textual dashboard + zero-dependency fallback)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_acdso.py"


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a realistic tabular CSV with mixed numeric/categorical + ID columns."""
    p = tmp_path / "orders_sample.csv"
    rows = ["order_id,total_bill,tip,sex,smoker,day,size"]
    days = ["Thur", "Fri", "Sat", "Sun"]
    for i in range(1, 37):
        bill = round(10.0 + (i * 1.35) % 35.0, 2)
        tip = round(1.25 + 0.14 * bill + (0.4 if i % 3 == 0 else -0.2), 2)
        sex = "Female" if i % 2 == 0 else "Male"
        smoker = "Yes" if i % 4 == 0 else "No"
        day = days[i % len(days)]
        size = 2 + (i % 4)
        rows.append(f"ORD-{i:04d},{bill},{tip},{sex},{smoker},{day},{size}")
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


class TestTUIHelpers:
    def test_profile_and_suggest_targets_on_real_csv(self, sample_csv: Path):
        from acdso.tui.app import profile_dataset, suggest_targets

        prof = profile_dataset(sample_csv)
        assert prof["rows"] == 36
        assert prof["cols"] == 7
        col_map = {c["name"]: c for c in prof["columns"]}
        assert "tip" in col_map
        assert col_map["tip"]["predictable"] is True
        # Unique order_id should be flagged non-predictable by column_validator
        assert col_map["order_id"]["predictable"] is False

        sugs = suggest_targets(sample_csv, top_n=3)
        assert len(sugs) >= 1
        assert all("column" in s for s in sugs)
        assert all(s["column"] != "order_id" for s in sugs)

    def test_profile_missing_file_raises(self, tmp_path: Path):
        from acdso.tui.app import profile_dataset

        with pytest.raises(FileNotFoundError):
            profile_dataset(tmp_path / "does_not_exist.csv")

    def test_parse_progress_marker_valid_and_invalid(self):
        from acdso.tui.app import parse_progress_marker

        assert parse_progress_marker("plain stdout log line") is None
        assert parse_progress_marker("[ACDSO_PROGRESS] not-json") is None

        ev = parse_progress_marker(
            '[ACDSO_PROGRESS] {"phase": 3, "total": 6, "name": "AutoML Training", "status": "running", "detail": "fitting"}'
        )
        assert ev is not None
        assert ev["phase"] == 3
        assert ev["total"] == 6
        assert ev["percent"] == 50
        assert ev["name"] == "AutoML Training"

        done_ev = parse_progress_marker(
            '[ACDSO_PROGRESS] {"phase": 6, "total": 6, "name": "Complete", "status": "done"}'
        )
        assert done_ev is not None
        assert done_ev["percent"] == 100

    def test_build_cli_command_flags(self, sample_csv: Path):
        from acdso.tui.app import build_cli_command

        cmd = build_cli_command(
            str(sample_csv),
            "tip",
            predict_mode=True,
            simple=True,
            skip_benchmark=True,
            best_score=True,
            force=True,
        )
        assert "--file" in cmd
        assert "--predict" in cmd
        assert "tip" in cmd
        assert "--simple" in cmd
        assert "--skip-benchmark" in cmd
        assert "--best-score" in cmd
        assert "--force" in cmd

        cmd_target = build_cli_command(str(sample_csv), "smoker", predict_mode=False, simple=False, skip_benchmark=False)
        assert "--target" in cmd_target
        assert "--predict" not in cmd_target
        assert "--simple" not in cmd_target
        assert "--skip-benchmark" not in cmd_target

    def test_load_latest_combined_summary(self, tmp_path: Path, monkeypatch):
        import acdso.tui.app as tui_app

        monkeypatch.setattr(tui_app, "REPORTS_DIR", tmp_path)
        rep = tmp_path / "autonomous_orders_sample_tip_combined.json"
        rep.write_text(
            json.dumps(
                {
                    "selected_model": {"name": "StackingEnsemble", "metrics": {"metric": 0.87}},
                    "audit_trail": {
                        "candidates_evaluated": [
                            {"model_name": "Ridge", "metric": 0.82, "train_time_s": 0.08, "peak_mem_mb": 1.1, "risk_penalty": 0.12},
                            {"model_name": "StackingEnsemble", "metric": 0.87, "train_time_s": 0.45, "peak_mem_mb": 3.2, "risk_penalty": 0.10},
                        ],
                        "pareto_front": [{"model_name": "Ridge"}, {"model_name": "StackingEnsemble"}],
                    },
                }
            ),
            encoding="utf-8",
        )
        summary = tui_app.load_latest_combined_summary(tmp_path / "orders_sample.csv", "tip")
        assert summary is not None
        assert summary["selected_model"]["name"] == "StackingEnsemble"
        assert len(summary["candidates"]) == 2
        assert "StackingEnsemble" in summary["pareto_names"]


class TestFallbackTerminalUI:
    def test_interactive_menu_flow_and_results_display(self, sample_csv: Path, tmp_path: Path, monkeypatch):
        import acdso.tui.app as tui_app

        monkeypatch.setattr(tui_app, "REPORTS_DIR", tmp_path)
        (tmp_path / "autonomous_orders_sample_tip_combined.json").write_text(
            json.dumps(
                {
                    "selected_model": {"name": "Ridge", "metrics": {"metric": 0.91}},
                    "audit_trail": {
                        "candidates_evaluated": [
                            {"model_name": "Ridge", "metric": 0.91, "train_time_s": 0.12, "peak_mem_mb": 1.4, "risk_penalty": 0.08},
                        ],
                        "pareto_front": [{"model_name": "Ridge"}],
                    },
                }
            ),
            encoding="utf-8",
        )

        # Scripted menu responses: load file -> suggest target -> set target -> toggle simple -> show results -> quit
        scripted = iter(["1", str(sample_csv), "2", "3", "tip", "4", "6", "q"])
        lines: list[str] = []

        ui = tui_app.FallbackTerminalUI(
            input_fn=lambda _prompt: next(scripted),
            output_fn=lines.append,
        )
        rc = ui.run()
        assert rc == 0
        assert ui.profile is not None
        assert ui.profile["rows"] == 36
        assert ui.target == "tip"
        assert ui.simple is False
        joined = "\n".join(lines)
        assert "Loaded orders_sample.csv (36 rows x 7 cols)" in joined
        assert "Suggested targets:" in joined

        # Load summary and render table directly
        ui.last_summary = tui_app.load_latest_combined_summary(sample_csv, "tip")
        ui._show_results()
        assert any("Selected Model: Ridge" in ln for ln in lines)
        assert any("PARETO" in ln for ln in lines)


class TestTextualTUI:
    def test_textual_dashboard_mounts_loads_suggests_and_selects_row(self, sample_csv: Path, tmp_path: Path, monkeypatch):
        import acdso.tui.app as tui_app

        if not tui_app.has_textual():
            pytest.skip("textual optional dependency not installed")

        from textual.widgets import DataTable, Input, Static

        monkeypatch.setattr(tui_app, "REPORTS_DIR", tmp_path)
        (tmp_path / "autonomous_orders_sample_tip_combined.json").write_text(
            json.dumps(
                {
                    "selected_model": {"name": "StackingEnsemble", "metrics": {"metric": 0.93}},
                    "audit_trail": {
                        "candidates_evaluated": [
                            {"model_name": "Ridge", "metric": 0.88, "train_time_s": 0.09, "peak_mem_mb": 1.2, "risk_penalty": 0.11},
                            {"model_name": "StackingEnsemble", "metric": 0.93, "train_time_s": 0.51, "peak_mem_mb": 3.6, "risk_penalty": 0.09},
                        ],
                        "pareto_front": [{"model_name": "Ridge"}, {"model_name": "StackingEnsemble"}],
                    },
                }
            ),
            encoding="utf-8",
        )

        async def _drive() -> None:
            app = tui_app.create_textual_app(initial_file=str(sample_csv))
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause()

                # Pre-loaded dataset populates 7 columns in #cols-table
                cols_table = app.query_one("#cols-table", DataTable)
                assert cols_table.row_count == 7

                meta_txt = str(app.query_one("#dataset-meta", Static).render())
                assert "orders_sample.csv" in meta_txt

                # Press 's' binding (or click #suggest-btn) to populate #target-input
                await pilot.click("#suggest-btn")
                await pilot.pause()
                suggested = app.query_one("#target-input", Input).value
                assert suggested != ""
                assert suggested != "order_id"

                # Populate Pareto results table from a real combined JSON report
                app._populate_results(str(sample_csv), "tip")
                await pilot.pause()
                pareto_table = app.query_one("#pareto-table", DataTable)
                assert pareto_table.row_count == 2
                banner_txt = str(app.query_one("#kpi-banner", Static).render())
                assert "StackingEnsemble" in banner_txt

                # Clean quit via 'q'
                await pilot.press("q")

        asyncio.run(_drive())


class TestTUIEntryPoints:
    def test_acdso_tui_help_exits_zero(self):
        from acdso.tui import main

        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_run_acdso_help_lists_tui_and_stays_fast(self):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-B", str(RUNNER), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(REPO_ROOT),
            check=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert proc.returncode == 0
        assert "--tui" in proc.stdout
        assert elapsed_ms < 2500.0, f"run_acdso.py --help regressed to {elapsed_ms:.0f} ms"
