"""Contract tests for ACDSO CLI/TUI, file loading, flags, progress, and HTML reports."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
import warnings
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_acdso.py"
UPLOADS_DIR = REPO_ROOT / "uploads"
TIPS_CSV = UPLOADS_DIR / "tips_0cccb3b1499e.csv"
SCRATCH_CSV = UPLOADS_DIR / "scratch_domains_92b347bb484a.csv"
JFROG_CSV = UPLOADS_DIR / "yearJFrog-data-transfer-20250821-20260814_146d8b6117aa.csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_acdso  # noqa: E402


def _run_cli(args: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    src_path = str(REPO_ROOT / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_pp}" if existing_pp else src_path
    return subprocess.run(
        [sys.executable, "-B", str(RUNNER), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _create_minimal_xlsx(path: Path, headers: list[str], rows: list[list[object]]) -> Path:
    """Create a valid minimal .xlsx workbook using stdlib zipfile + XML only."""
    shared_strings: list[str] = []
    ss_index: dict[str, int] = {}

    def _intern(s: str) -> int:
        if s not in ss_index:
            ss_index[s] = len(shared_strings)
            shared_strings.append(s)
        return ss_index[s]

    def _col_letter(idx: int) -> str:
        out = []
        n = idx + 1
        while n > 0:
            n, rem = divmod(n - 1, 26)
            out.append(chr(ord("A") + rem))
        return "".join(reversed(out))

    def _xml_esc(val: str) -> str:
        return (
            val.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    all_rows = [headers] + rows
    sheet_rows_xml: list[str] = []
    for r_idx, row in enumerate(all_rows, 1):
        cells_xml: list[str] = []
        for c_idx, val in enumerate(row):
            ref = f"{_col_letter(c_idx)}{r_idx}"
            if val is None:
                continue
            if isinstance(val, bool):
                cells_xml.append(f'<c r="{ref}" t="b"><v>{1 if val else 0}</v></c>')
            elif isinstance(val, (int, float)):
                cells_xml.append(f'<c r="{ref}"><v>{val}</v></c>')
            else:
                s_idx = _intern(str(val))
                cells_xml.append(f'<c r="{ref}" t="s"><v>{s_idx}</v></c>')
        sheet_rows_xml.append(f'<row r="{r_idx}">{"".join(cells_xml)}</row>')

    ss_items = "".join(f"<si><t>{_xml_esc(s)}</t></si>" for s in shared_strings)
    shared_strings_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">'
        f"{ss_items}</sst>"
    )
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows_xml)}</sheetData></worksheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
        'Target="sharedStrings.xml"/>'
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        "</Types>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/sharedStrings.xml", shared_strings_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return path


class _TagAuditor(HTMLParser):
    VOID = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, tuple[int, int]]] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in self.VOID:
            self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID:
            return
        if not self.stack:
            self.errors.append(f"Unexpected </{tag}> at {self.getpos()}")
            return
        if self.stack[-1][0] == tag:
            self.stack.pop()
        else:
            self.errors.append(
                f"Mismatched </{tag}> at {self.getpos()}, expected </{self.stack[-1][0]}>"
            )
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]
                    break


# ---------------------------------------------------------------------------
# 1. Startup Latency
# ---------------------------------------------------------------------------


def test_cli_help_fast_startup(tmp_path: Path) -> None:
    # Warm filesystem/OS cache once, then assert < 500 ms cold-process CLI startup
    _run_cli(["--help"], cwd=tmp_path)
    t0 = time.perf_counter()
    res = _run_cli(["--help"], cwd=tmp_path)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert res.returncode == 0
    assert "--simple" in res.stdout
    assert "--no-auto-exclude" in res.stdout
    assert elapsed_ms < 500.0, f"--help took {elapsed_ms:.1f} ms (>= 500 ms)"


# ---------------------------------------------------------------------------
# 2. Non-Zero Exit Codes & Input / NaN / Empty Validation
# ---------------------------------------------------------------------------


def test_nonzero_exit_missing_file(tmp_path: Path) -> None:
    res = _run_cli(["--file", "nonexistent_12345.csv", "--predict", "tip", "--simple"], cwd=tmp_path)
    assert res.returncode != 0


def test_nonzero_exit_missing_target_column(tmp_path: Path) -> None:
    res = _run_cli(["--file", str(TIPS_CSV), "--predict", "non_existent_col", "--simple"], cwd=tmp_path)
    assert res.returncode != 0
    assert "not found" in (res.stdout + res.stderr).lower()


def test_nonzero_exit_invalid_bounds(tmp_path: Path) -> None:
    res_h = _run_cli(["--file", str(TIPS_CSV), "--predict", "tip", "--horizon", "0", "--simple"], cwd=tmp_path)
    assert res_h.returncode != 0
    res_b = _run_cli(["--decision", "--budget", "-5"], cwd=tmp_path)
    assert res_b.returncode != 0
    res_tb = _run_cli(["--benchmark", "--time-budget", "0"], cwd=tmp_path)
    assert res_tb.returncode != 0


def test_nonzero_exit_empty_and_zero_byte_csv(tmp_path: Path) -> None:
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("a,b\n", encoding="utf-8")
    res1 = _run_cli(["--file", str(header_only), "--predict", "b", "--simple"], cwd=tmp_path)
    assert res1.returncode != 0

    zero_byte = tmp_path / "empty.csv"
    zero_byte.write_text("", encoding="utf-8")
    res2 = _run_cli(["--file", str(zero_byte), "--predict", "b", "--simple"], cwd=tmp_path)
    assert res2.returncode != 0


def test_nonzero_exit_predict_without_file(tmp_path: Path) -> None:
    res = _run_cli(["--predict", "tip"], cwd=tmp_path)
    assert res.returncode != 0


def test_nonzero_exit_all_nan_target_and_partial_nan_drops_cleanly(tmp_path: Path) -> None:
    df = pd.read_csv(TIPS_CSV).head(30).copy()
    df_all_nan = df.copy()
    df_all_nan["tip"] = np.nan
    all_nan_path = tmp_path / "all_nan.csv"
    df_all_nan.to_csv(all_nan_path, index=False)

    res_fail = _run_cli(["--file", str(all_nan_path), "--target", "tip", "--simple"], cwd=tmp_path)
    assert res_fail.returncode != 0

    df_partial = df.copy()
    df_partial.loc[0:2, "tip"] = np.nan
    df_partial.loc[3, "tip"] = np.inf
    partial_path = tmp_path / "partial_nan.csv"
    df_partial.to_csv(partial_path, index=False)

    res_ok = _run_cli(["--file", str(partial_path), "--target", "tip", "--simple"], cwd=tmp_path)
    assert res_ok.returncode == 0
    assert "[WARN] Dropped 4 row(s)" in res_ok.stdout


def test_nonzero_exit_decision_and_timeseries_invalid_cols(tmp_path: Path) -> None:
    res_dec = _run_cli(["--decision", "--file", str(TIPS_CSV), "--target", "missing_col"], cwd=tmp_path)
    assert res_dec.returncode != 0
    res_ts = _run_cli(["--timeseries", "--file", str(TIPS_CSV), "--target", "tip"], cwd=tmp_path)
    assert res_ts.returncode != 0
    res_mt = _run_cli(["--file", str(TIPS_CSV), "--target", "tip,missing_col"], cwd=tmp_path)
    assert res_mt.returncode != 0


def test_skip_benchmark_and_predict_with_comma_target(tmp_path: Path) -> None:
    # --autonomous --file ... (default) and --skip-benchmark skip benchmark phase
    df_sample = pd.read_csv(TIPS_CSV).head(35)
    sample_csv = tmp_path / "tips_sub.csv"
    df_sample.to_csv(sample_csv, index=False)

    res = _run_cli(
        ["--file", str(sample_csv), "--target", "tip", "--skip-benchmark"],
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    combined = json.loads((tmp_path / "reports" / "autonomous_tips_sub_tip_combined.json").read_text(encoding="utf-8"))
    phases = {p["name"]: p["status"] for p in combined["phase_results"]["All Phases"]}
    assert "SKIPPED" in phases["Benchmark (Custom File)"]

    # Passing --target a,b alongside --predict tip should NOT silently skip --predict
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    res2 = _run_cli(
        ["--file", str(sample_csv), "--target", "tip,size", "--predict", "tip", "--simple"],
        cwd=ws2,
    )
    assert res2.returncode == 0, res2.stdout + "\n" + res2.stderr
    assert (ws2 / "reports" / "predictions_tips_sub_tip_full_dataset.csv").is_file()


# ---------------------------------------------------------------------------
# 3. Non-CSV Loading, Stdlib .xlsx Fallback, Base Names & --predict JSON Alignment
# ---------------------------------------------------------------------------


def test_load_user_file_encodings_and_errors(tmp_path: Path) -> None:
    cp_csv = tmp_path / "cp1252_sample.csv"
    cp_csv.write_bytes("name,val\nCaf\xe9,10\nNa\xefve,20\n".encode("cp1252"))
    df_cp = run_acdso.load_user_file(str(cp_csv))
    assert len(df_cp) == 2
    assert df_cp["name"].tolist() == ["Café", "Naïve"]

    bom_csv = tmp_path / "bom_sample.csv"
    bom_csv.write_bytes("\ufeffcol_a,col_b\n1,2\n3,4\n".encode("utf-8"))
    df_bom = run_acdso.load_user_file(str(bom_csv))
    assert list(df_bom.columns) == ["col_a", "col_b"]

    with pytest.raises(FileNotFoundError):
        run_acdso.load_user_file(str(tmp_path / "missing.csv"))

    bad_json = tmp_path / "scalar.json"
    bad_json.write_text("12345", encoding="utf-8")
    with pytest.raises(ValueError):
        run_acdso.load_user_file(str(bad_json))


def test_non_csv_xlsx_json_pkl_cli_predict_roundtrip(tmp_path: Path) -> None:
    df_sample = pd.read_csv(TIPS_CSV).head(40)

    # 1) .xlsx via stdlib zipfile (works with or without openpyxl)
    xlsx_path = _create_minimal_xlsx(
        tmp_path / "tips_sample.xlsx",
        list(df_sample.columns),
        df_sample.values.tolist(),
    )
    loaded_xlsx = run_acdso.load_user_file(str(xlsx_path))
    assert loaded_xlsx.shape == df_sample.shape
    assert pd.api.types.is_numeric_dtype(loaded_xlsx["tip"])

    res_xlsx = _run_cli(["--file", str(xlsx_path), "--predict", "tip", "--simple"], cwd=tmp_path)
    assert res_xlsx.returncode == 0, res_xlsx.stdout + "\n" + res_xlsx.stderr

    reports_dir = tmp_path / "reports"
    assert (reports_dir / "model_tips_sample_tip.pkl").is_file()
    assert (reports_dir / "autonomous_tips_sample_tip_combined.json").is_file()
    assert (reports_dir / "autonomous_tips_sample_tip_report.html").is_file()
    assert (reports_dir / "predictions_tips_sample_tip_full_dataset.csv").is_file()
    # Ensure extension wasn't leaked into artifact names
    assert not any(".xlsx" in p.name for p in reports_dir.iterdir())

    combined_xlsx = json.loads((reports_dir / "autonomous_tips_sample_tip_combined.json").read_text(encoding="utf-8"))
    assert combined_xlsx["predictions"]["total_predictions"] == len(df_sample)

    # 2) .json and .pkl round-trip
    for ext, writer in [
        (".json", lambda p: df_sample.to_json(p, orient="records")),
        (".pkl", lambda p: df_sample.to_pickle(p)),
    ]:
        stem = f"tips_{ext.lstrip('.')}"
        fpath = tmp_path / f"{stem}{ext}"
        writer(fpath)
        res = _run_cli(["--file", str(fpath), "--predict", "tip", "--simple"], cwd=tmp_path)
        assert res.returncode == 0, res.stdout + "\n" + res.stderr
        cjson_path = reports_dir / f"autonomous_{stem}_tip_combined.json"
        assert cjson_path.is_file()
        assert (reports_dir / f"autonomous_{stem}_tip_report.html").is_file()
        assert (reports_dir / f"predictions_{stem}_tip_full_dataset.csv").is_file()
        cjson = json.loads(cjson_path.read_text(encoding="utf-8"))
        assert cjson["predictions"]["total_predictions"] == len(df_sample)


# ---------------------------------------------------------------------------
# 4. Flag Wiring (--simple, --no-auto-exclude), Datetime Guard & Categorical Encoding
# ---------------------------------------------------------------------------


def test_simple_mode_skips_timeseries_and_benchmark_with_date_col(tmp_path: Path) -> None:
    # Build a slice of yearJFrog that DOES contain a real Date column
    df_jf = pd.read_csv(JFROG_CSV).head(40)
    jf_slice = tmp_path / "jfrog_slice.csv"
    df_jf.to_csv(jf_slice, index=False)

    res = _run_cli(
        ["--file", str(jf_slice), "--target", "Total Data Transfer (B)", "--simple"],
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout + "\n" + res.stderr

    combined_path = tmp_path / "reports" / "autonomous_jfrog_slice_Total Data Transfer (B)_combined.json"
    assert combined_path.is_file()
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    phases = {p["name"]: p["status"] for p in combined["phase_results"]["All Phases"]}
    assert "SKIPPED" in phases["Time Series"]
    assert "SKIPPED" in phases["Benchmark (Custom File)"]
    assert combined["datetime_col"] == "Date"


def test_numeric_not_detected_as_datetime_and_jfrog_date_is(tmp_path: Path) -> None:
    df_tips = pd.read_csv(TIPS_CSV)
    df_scratch = pd.read_csv(SCRATCH_CSV)
    df_jf = pd.read_csv(JFROG_CSV).head(100)
    df_num_str = pd.DataFrame({"yr": ["2025", "2026", "2027", "2028"], "val": [1, 2, 3, 4]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert run_acdso._detect_datetime_column(df_tips) is None
        assert run_acdso._detect_datetime_column(df_scratch) is None
        assert run_acdso._detect_datetime_column(df_jf) == "Date"
        assert run_acdso._detect_datetime_column(df_num_str) is None

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not user_warnings, f"Unexpected UserWarnings during datetime detection: {user_warnings}"

    # Verify CLI run on tips records datetime_col is None
    res = _run_cli(["--file", str(TIPS_CSV), "--target", "tip", "--simple"], cwd=tmp_path)
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    combined = json.loads(
        (tmp_path / "reports" / f"autonomous_{TIPS_CSV.stem}_tip_combined.json").read_text(encoding="utf-8")
    )
    assert combined["datetime_col"] is None


def test_no_auto_exclude_respected(tmp_path: Path) -> None:
    df = pd.read_csv(TIPS_CSV).head(60).copy()
    df["row_id"] = np.arange(len(df))
    df["tip_leak_copy"] = df["tip"] * 2.0 + 0.001
    csv_path = tmp_path / "tips_with_id_leak.csv"
    df.to_csv(csv_path, index=False)

    # Default run (auto_exclude=True)
    ws_default = tmp_path / "ws_default"
    ws_default.mkdir()
    res_def = _run_cli(["--file", str(csv_path), "--target", "tip", "--simple"], cwd=ws_default)
    assert res_def.returncode == 0, res_def.stdout + "\n" + res_def.stderr
    model_def = joblib.load(ws_default / "reports" / "model_tips_with_id_leak_tip.pkl")
    assert "row_id" not in model_def["feature_columns"]
    assert "tip_leak_copy" not in model_def["feature_columns"]

    # --no-auto-exclude run
    ws_no_excl = tmp_path / "ws_no_excl"
    ws_no_excl.mkdir()
    res_no = _run_cli(
        ["--file", str(csv_path), "--target", "tip", "--simple", "--no-auto-exclude"],
        cwd=ws_no_excl,
    )
    assert res_no.returncode == 0, res_no.stdout + "\n" + res_no.stderr
    model_no = joblib.load(ws_no_excl / "reports" / "model_tips_with_id_leak_tip.pkl")
    assert "row_id" in model_no["feature_columns"]
    assert "tip_leak_copy" in model_no["feature_columns"]


def test_high_cardinality_categorical_train_and_predict_roundtrip(tmp_path: Path) -> None:
    # Subset of yearJFrog keeping >100 unique User values so frequency encoding activates
    df_jf = pd.read_csv(JFROG_CSV)
    # Take first row per unique User (168 uniques) to keep runtime fast while > 100 uniques
    df_hc = df_jf.drop_duplicates(subset=["User"]).head(130).copy()
    hc_csv = tmp_path / "jfrog_high_card.csv"
    df_hc.to_csv(hc_csv, index=False)

    res = _run_cli(
        ["--file", str(hc_csv), "--predict", "Total Data Transfer (B)", "--simple", "--no-auto-exclude"],
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    model_bundle = joblib.load(tmp_path / "reports" / "model_jfrog_high_card_Total Data Transfer (B).pkl")
    enc_meta = model_bundle.get("categorical_encoding", {})
    assert "User" in enc_meta.get("high_card_freqs", {})
    assert "User" in model_bundle["feature_columns"]
    preds_csv = tmp_path / "reports" / "predictions_jfrog_high_card_Total Data Transfer (B)_full_dataset.csv"
    assert preds_csv.is_file()
    assert len(pd.read_csv(preds_csv)) == len(df_hc)


# ---------------------------------------------------------------------------
# 5. Structured Progress & HTML Report Reliability
# ---------------------------------------------------------------------------


def test_progress_markers_monotonic_and_no_ansi_when_piped(tmp_path: Path) -> None:
    res = _run_cli(["--file", str(TIPS_CSV), "--predict", "tip", "--simple"], cwd=tmp_path)
    assert res.returncode == 0, res.stdout + "\n" + res.stderr

    # Zero ANSI escapes when stdout is piped
    assert "\x1b[" not in res.stdout

    events: list[dict[str, object]] = []
    for line in res.stdout.splitlines():
        if line.startswith("[ACDSO_PROGRESS] "):
            payload = json.loads(line[len("[ACDSO_PROGRESS] "):])
            assert set(["phase", "total", "name", "status", "detail"]).issubset(payload.keys())
            assert payload["status"] in {"running", "done", "error"}
            assert 1 <= int(payload["phase"]) <= int(payload["total"])
            events.append(payload)

    assert len(events) >= 10
    phases = [int(e["phase"]) for e in events]
    assert phases == sorted(phases), f"Non-monotonic progress phases: {phases}"
    assert events[-1]["phase"] == events[-1]["total"] == 10
    assert events[-1]["status"] == "done"


def test_html_report_valid_structure_escaping_and_js_links(tmp_path: Path) -> None:
    report = {
        "timestamp": "2026-11-22T12:00:00",
        "mode": "autonomous_custom_file",
        "configuration": {"scorer": "f1", "input_file": r"C:\Users\alice\O'Reilly\new\test.csv"},
        "selected_model": {
            "name": "RF <best> & 'fast'",
            "type": "RandomForest",
            "pipeline_steps": ["prep<1>", "clf"],
            "metrics": {"metric": 0.9123},
            "code_info": {"code": "if x < 5 and y > 1:\n    print('ok')", "execution_summary": "a < b"},
        },
        "phase_results": {
            "All Phases": [{"name": "AutoML<1>", "status": "SUCCESS", "model_name": "RF <best>", "time": 1.2}]
        },
        "audit_trail": {
            "selected": {"name": "RF <best> & 'fast'"},
            "candidates_evaluated": [{"name": "RF <best> & 'fast'", "metric": 0.9123}],
            "pareto_front": [{"name": "RF <best> & 'fast'", "metric": 0.9123}],
        },
        "timeseries": {
            "horizon": 7,
            "datetime_col": "dt",
            "target": "y<1>",
            "best_model": "Naive<1>",
            "leaderboard": [{"Model": "Naive<1>", "Score": 1.0, "Fit Time (s)": 0.1, "Family": "Base"}],
        },
        "predictions": {
            "target_column": "y<1>",
            "total_predictions": 5,
            "task": "regression",
            "output_file": r"C:\tmp\new\O'Reilly\pred.csv",
            "statistics": {"mean": 1.0},
            "sample_values": [{"row": 0, "predicted": 1.0, "actual": 1.0}],
        },
        "console_output": 'if x < 5: print("<a>")',
        "summary": {"total_phases": 1},
        "html_report_path": str(tmp_path / "rep.html"),
        "model_path": str(tmp_path / "m'1.pkl"),
        "audit_path": str(tmp_path / "a.json"),
    }

    html_out = tmp_path / "combined.html"
    run_acdso.generate_html_report_from_combined(report, str(html_out))
    html_text = html_out.read_text(encoding="utf-8")

    aud = _TagAuditor()
    aud.feed(html_text)
    assert aud.errors == [], f"HTML tag errors: {aud.errors}"
    assert aud.stack == [], f"Unclosed HTML tags: {aud.stack}"
    assert html_text.count("<table") == html_text.count("</table>")
    assert "RF &lt;best&gt;" in html_text
    assert 'data-path="C:\\tmp\\new\\O&#x27;Reilly\\pred.csv"' in html_text or "this.dataset.path" in html_text
    assert "navigator.clipboard.writeText(this.dataset.path)" in html_text
    assert not re.search(r"writeText\(['\"]C:\\", html_text)

    # Multi-target HTML report validation
    mt_results = {
        "targets": ["t'1", "t<2>"],
        "target_reports": {"t'1": report, "t<2>": report},
        "predictions": {
            "t'1": {"output_file": r"C:\tmp\a'b.csv", "count": 5, "statistics": {"mean": 1.0}},
            "t<2>": {"output_file": r"C:\tmp\c\d.csv", "count": 5, "statistics": {"distribution": {"0": 5}}},
        },
    }
    mt_html = run_acdso._generate_multi_target_html_report(mt_results, "data<1>")
    aud2 = _TagAuditor()
    aud2.feed(mt_html)
    assert aud2.errors == [], f"Multi-target HTML tag errors: {aud2.errors}"
    assert aud2.stack == [], f"Multi-target unclosed HTML tags: {aud2.stack}"
    assert 'id="tab-panel-0"' in mt_html
    assert 'data-tab="tab-panel-0"' in mt_html
    assert "showTab(this.dataset.tab, this)" in mt_html

    # Autonomous & Benchmark HTML footer timestamp escaping
    orig_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        auto_path = run_acdso._generate_autonomous_report(
            {
                "timestamp": "2026-11-22T12:00:00 <script>alert(1)</script>",
                "total_time": 1.0,
                "llm_available": False,
                "phases": {},
                "summary": {"total": 1, "success": 1, "failed": 0, "success_rate": 100.0},
            }
        )
        auto_html = Path(auto_path).read_text(encoding="utf-8")
        assert "<script>alert(1)</script>" not in auto_html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in auto_html
    finally:
        os.chdir(orig_cwd)

    class _FakeSummary:
        timestamp = "<script>alert(2)</script>"
        datasets_tested = ["ds<1>"]
        systems_tested = ["ACDSO"]
        time_budget_s = 5
        results = []
        winner_by_metric = {"score": "ACDSO"}
        leakage_tests = {}

    bench_out = tmp_path / "bench.html"
    run_acdso._generate_html_report(_FakeSummary(), str(bench_out))
    bench_html = bench_out.read_text(encoding="utf-8")
    assert "<script>alert(2)</script>" not in bench_html
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in bench_html


# ---------------------------------------------------------------------------
# 6. Lazy ML Import Safety, Categorical Alignment & Strict JSON / Proba Naming
# ---------------------------------------------------------------------------


def test_lazy_ml_imports_and_direct_demo_calls_no_nameerror(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    src_path = str(REPO_ROOT / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_pp}" if existing_pp else src_path

    # 1) Importing run_acdso must NOT eagerly load heavy ML libraries
    probe_import = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
                "import run_acdso; "
                "loaded = [m for m in ('numpy','pandas','scipy','sklearn') if m in sys.modules]; "
                "assert not loaded, f'Eagerly loaded: {loaded}'"
            ),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert probe_import.returncode == 0, probe_import.stdout + "\n" + probe_import.stderr

    # 2) AST audit: every function referencing _LAZY_ML_SYMBOLS invokes _ensure_ml_deps()
    src_text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(src_text)
    lazy_syms = set(run_acdso._LAZY_ML_SYMBOLS)
    missing: list[tuple[int, str, list[str]]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ("_ensure_ml_deps", "__getattr__"):
                continue
            assigned: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                    assigned.add(sub.id)
                elif isinstance(sub, ast.arg):
                    assigned.add(sub.arg)
                elif isinstance(sub, ast.ExceptHandler) and sub.name:
                    assigned.add(sub.name)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        assigned.add(alias.asname or alias.name.split(".")[0])
            used_global = {
                sub.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Name)
                and isinstance(sub.ctx, ast.Load)
                and sub.id in lazy_syms
                and sub.id not in assigned
            }
            calls_ensure = any(
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "_ensure_ml_deps"
                for sub in ast.walk(node)
            )
            if used_global and not calls_ensure:
                missing.append((node.lineno, node.name, sorted(used_global)))
    assert missing == [], f"Functions missing _ensure_ml_deps(): {missing}"

    # 3) Direct function calls right after `import run_acdso` (without main()) raise no NameError
    for fn_name in ("demo_risk_report", "demo_auto_timeseries"):
        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); import run_acdso; run_acdso.{fn_name}()",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        assert proc.returncode == 0, f"{fn_name} failed:\n{proc.stdout}\n{proc.stderr}"
        assert "NameError" not in proc.stderr


def test_encode_categoricals_safe_subset_and_legacy_alignment() -> None:
    X_train = pd.DataFrame({"city": ["A", "B", "C", "A"], "num": [1, 2, 3, 4]})
    X_enc, meta = run_acdso._encode_categoricals_safe(X_train, ["city"], return_meta=True)
    feature_cols = list(X_enc.columns)
    assert feature_cols == ["num", "city_B", "city_C"]
    assert X_enc.iloc[1].to_dict() == {"num": 2, "city_B": 1, "city_C": 0}

    # Both encoding_meta and legacy feature_columns fallback must align single-row subsets
    for kwargs in (
        {"encoding_meta": meta, "feature_columns": feature_cols},
        {"encoding_meta": None, "feature_columns": feature_cols},
    ):
        pred_b = run_acdso._encode_categoricals_safe(
            pd.DataFrame({"city": ["B"], "num": [2]}), ["city"], **kwargs
        ).reindex(columns=feature_cols, fill_value=0)
        assert pred_b.iloc[0].to_dict() == {"num": 2, "city_B": 1, "city_C": 0}

        pred_a = run_acdso._encode_categoricals_safe(
            pd.DataFrame({"city": ["A"], "num": [1]}), ["city"], **kwargs
        ).reindex(columns=feature_cols, fill_value=0)
        assert pred_a.iloc[0].to_dict() == {"num": 1, "city_B": 0, "city_C": 0}

        pred_bc = run_acdso._encode_categoricals_safe(
            pd.DataFrame({"city": ["B", "C"], "num": [2, 3]}), ["city"], **kwargs
        ).reindex(columns=feature_cols, fill_value=0)
        assert pred_bc.to_dict(orient="records") == [
            {"num": 2, "city_B": 1, "city_C": 0},
            {"num": 3, "city_B": 0, "city_C": 1},
        ]


def test_predict_partial_nan_strict_rfc8259_json_and_proba_col_naming(tmp_path: Path) -> None:
    # 1) Partial NaN/inf target in --predict must produce strict RFC 8259 JSON (no bare NaN/Infinity)
    df = pd.read_csv(TIPS_CSV).head(40).copy()
    df.loc[0:2, "tip"] = np.nan
    df.loc[3, "tip"] = np.inf
    nan_csv = tmp_path / "tips_partial_nan.csv"
    df.to_csv(nan_csv, index=False)

    ws_nan = tmp_path / "ws_nan"
    ws_nan.mkdir()
    res_nan = _run_cli(["--file", str(nan_csv), "--predict", "tip", "--simple"], cwd=ws_nan)
    assert res_nan.returncode == 0, res_nan.stdout + "\n" + res_nan.stderr

    combined_path = ws_nan / "reports" / "autonomous_tips_partial_nan_tip_combined.json"
    assert combined_path.is_file()

    def _reject_nonfinite(constant: str) -> None:
        raise ValueError(f"Non-RFC8259 JSON constant: {constant}")

    combined = json.loads(combined_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    samples = combined["predictions"]["sample_values"]
    assert samples[0]["actual"] is None
    assert samples[3]["actual"] is None
    assert isinstance(samples[4]["actual"], float)

    # 2) Classification --target != --predict aligns probability column prefix with trained target
    ws_clf = tmp_path / "ws_clf"
    ws_clf.mkdir()
    clf_csv = tmp_path / "tips_clf.csv"
    pd.read_csv(TIPS_CSV).iloc[::2].to_csv(clf_csv, index=False)
    res_clf = _run_cli(
        ["--file", str(clf_csv), "--target", "smoker", "--predict", "sex", "--simple"],
        cwd=ws_clf,
    )
    assert res_clf.returncode == 0, res_clf.stdout + "\n" + res_clf.stderr
    pred_df = pd.read_csv(ws_clf / "reports" / "predictions_tips_clf_smoker_full_dataset.csv")
    assert "smoker_predicted" in pred_df.columns
    assert "smoker_probability" in pred_df.columns
    assert "sex_probability" not in pred_df.columns
