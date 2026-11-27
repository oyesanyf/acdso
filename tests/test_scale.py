"""Unit and integration tests for Milestone M5: out-of-core scale, bounded I/O,
core/EDA memory guards & NLP text wiring, lazy public exports, and packaging.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
import tracemalloc
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"


def _write_minimal_xlsx(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    """Create a valid minimal .xlsx workbook using stdlib zipfile (no openpyxl required)."""
    def _col_letter(idx: int) -> str:
        res = ""
        n = idx + 1
        while n > 0:
            n, rem = divmod(n - 1, 26)
            res = chr(ord("A") + rem) + res
        return res

    all_rows = [headers] + rows
    row_xml_parts = []
    for r_i, row in enumerate(all_rows, start=1):
        cells = []
        for c_i, val in enumerate(row):
            ref = f"{_col_letter(c_i)}{r_i}"
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                cells.append(f'<c r="{ref}"><v>{val}</v></c>')
            else:
                esc = str(val).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{esc}</t></is></c>')
        row_xml_parts.append(f'<row r="{r_i}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml_parts)}</sheetData>'
        "</worksheet>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


class TestBoundedIOAndReservoirSampling:
    """Tests for src/acdso/cli/io.py and src/acdso/scale.py streaming file utilities."""

    def test_cli_io_cold_import_defers_heavy_modules(self):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(SRC_DIR)
        code = (
            "import sys, time; "
            "t0 = time.perf_counter(); "
            "import acdso.cli.io; "
            "dt = time.perf_counter() - t0; "
            "heavy = [m for m in ('pandas', 'numpy', 'scipy', 'sklearn', 'pyarrow') if m in sys.modules]; "
            "assert not heavy, f'Eager heavy imports in acdso.cli.io: {heavy}'; "
            "assert dt < 0.5, f'Import too slow: {dt:.3f}s'"
        )
        proc = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    def test_iter_file_chunks_csv_tsv_jsonl_json_xlsx_pickle(self, tmp_path: Path):
        from acdso.cli.io import iter_file_chunks

        df = pd.DataFrame(
            {
                "a": np.arange(55, dtype=float),
                "b": [f"cat_{i % 3}" for i in range(55)],
                "y": [i % 2 for i in range(55)],
            }
        )

        csv_p = tmp_path / "data.csv"
        df.to_csv(csv_p, index=False)
        chunks_csv = list(iter_file_chunks(csv_p, chunk_size=20))
        assert [len(c) for c in chunks_csv] == [20, 20, 15]
        pd.testing.assert_frame_equal(pd.concat(chunks_csv, ignore_index=True), df)

        # usecols projection
        proj = pd.concat(list(iter_file_chunks(csv_p, chunk_size=25, usecols=["a", "y"])), ignore_index=True)
        assert list(proj.columns) == ["a", "y"]
        assert len(proj) == 55

        # TSV
        tsv_p = tmp_path / "data.tsv"
        df.to_csv(tsv_p, sep="\t", index=False)
        assert sum(len(c) for c in iter_file_chunks(tsv_p, chunk_size=18)) == 55

        # JSONL
        jsonl_p = tmp_path / "data.jsonl"
        df.to_json(jsonl_p, orient="records", lines=True)
        assert [len(c) for c in iter_file_chunks(jsonl_p, chunk_size=30)] == [30, 25]

        # JSON records
        json_p = tmp_path / "data.json"
        df.to_json(json_p, orient="records")
        assert [len(c) for c in iter_file_chunks(json_p, chunk_size=30)] == [30, 25]

        # Pickle
        pkl_p = tmp_path / "data.pkl"
        df.to_pickle(pkl_p)
        assert [len(c) for c in iter_file_chunks(pkl_p, chunk_size=20)] == [20, 20, 15]

        # XLSX stdlib fallback
        xlsx_p = tmp_path / "data.xlsx"
        _write_minimal_xlsx(xlsx_p, ["a", "b", "y"], df.values.tolist())
        xlsx_chunks = list(iter_file_chunks(xlsx_p, chunk_size=20))
        assert [len(c) for c in xlsx_chunks] == [20, 20, 15]

    def test_iter_file_chunks_cp1252_encoding_and_error_edges(self, tmp_path: Path):
        from acdso.cli.io import iter_file_chunks

        cp_path = tmp_path / "cp1252.csv"
        cp_path.write_bytes("city,val\nSão Paulo,1\nMünchen,2\n".encode("cp1252"))
        chunks = list(iter_file_chunks(cp_path, chunk_size=1))
        assert len(chunks) == 2
        assert chunks[0]["city"].iloc[0] == "São Paulo"

        header_only = tmp_path / "header_only.csv"
        header_only.write_text("x1,x2,target\n", encoding="utf-8")
        assert list(iter_file_chunks(header_only, chunk_size=10)) == []

        empty_zero = tmp_path / "empty.csv"
        empty_zero.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            list(iter_file_chunks(empty_zero, chunk_size=10))

        with pytest.raises(FileNotFoundError):
            iter_file_chunks(tmp_path / "does_not_exist.csv", chunk_size=10)

        with pytest.raises(ValueError):
            iter_file_chunks(cp_path, chunk_size=0)

        with pytest.raises(ValueError):
            iter_file_chunks(cp_path, chunk_size=True)

    def test_reservoir_sample_file_determinism_edges_and_uniformity(self, tmp_path: Path):
        from acdso.cli.io import sample_file_bounded
        from acdso.scale import reservoir_sample_file

        n_total = 1200
        df = pd.DataFrame({"row_id": np.arange(n_total), "val": np.linspace(0.0, 1.0, n_total)})
        csv_p = tmp_path / "stream.csv"
        df.to_csv(csv_p, index=False)

        s1, tot1 = reservoir_sample_file(csv_p, max_rows=120, chunk_size=150, random_state=42)
        s2, tot2 = reservoir_sample_file(csv_p, max_rows=120, chunk_size=150, random_state=42)
        assert tot1 == n_total and tot2 == n_total
        assert len(s1) == 120
        pd.testing.assert_frame_equal(s1, s2)

        # Verify Algorithm R reaches tail chunks as well as head chunks
        assert int(s1["row_id"].max()) > 900
        assert int(s1["row_id"].min()) < 300

        # max_rows > total preserves full dataset
        small_p = tmp_path / "small.csv"
        df.iloc[:40].to_csv(small_p, index=False)
        s_full, tot_full = reservoir_sample_file(small_p, max_rows=500, chunk_size=15, random_state=7)
        assert tot_full == 40
        assert len(s_full) == 40
        assert list(s_full.columns) == ["row_id", "val"]

        # Header-only preserves column headers and total == 0
        hdr_p = tmp_path / "hdr.csv"
        hdr_p.write_text("col_a,col_b\n", encoding="utf-8")
        s_hdr, tot_hdr = reservoir_sample_file(hdr_p, max_rows=50, chunk_size=10)
        assert tot_hdr == 0
        assert len(s_hdr) == 0
        assert list(s_hdr.columns) == ["col_a", "col_b"]

        df_only = sample_file_bounded(csv_p, max_rows=50, chunk_size=100, return_total=False)
        assert isinstance(df_only, pd.DataFrame) and len(df_only) == 50

    def test_inspect_file_schema_and_backwards_compat(self, tmp_path: Path):
        from acdso.cli.io import _encode_categoricals_safe, inspect_file_schema, load_user_file

        df = pd.DataFrame(
            {
                "num1": [1.0, 2.5, 3.0, 4.5],
                "cat1": ["a", "b", "a", "c"],
                "target": [0, 1, 0, 1],
            }
        )
        csv_p = tmp_path / "schema.csv"
        df.to_csv(csv_p, index=False)

        schema = inspect_file_schema(csv_p, sample_rows=10, count_rows=True, chunk_size=2)
        assert schema["n_rows"] == 4
        assert schema["n_columns"] == 3
        assert schema["columns"] == ["num1", "cat1", "target"]
        assert "num1" in schema["numeric_columns"]
        assert "cat1" in schema["categorical_columns"]
        json.dumps(schema, allow_nan=False)

        loaded = load_user_file(str(csv_p))
        assert len(loaded) == 4
        enc, meta = _encode_categoricals_safe(loaded, return_meta=True)
        assert "low_card_cols" in meta and len(enc) == 4


class TestACDSOOutOfCoreTrainer:
    """Tests for ACDSOOutOfCoreTrainer across partial_fit and subsample modes."""

    def test_partial_fit_classification_with_text_and_categoricals(self, tmp_path: Path):
        from acdso import ACDSOOutOfCoreTrainer

        rng = np.random.RandomState(42)
        n = 900
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        y = (2.0 * x1 - 1.5 * x2 > 0).astype(int)
        region = ["north" if i % 2 == 0 else "south" for i in range(n)]
        notes = [
            "strong upward revenue momentum positive" if v == 1 else "sharp negative contraction decline"
            for v in y
        ]
        df = pd.DataFrame({"x1": x1, "x2": x2, "region": region, "notes": notes, "target": y})
        train_p = tmp_path / "clf_train.csv"
        df.iloc[:700].to_csv(train_p, index=False)

        trainer = ACDSOOutOfCoreTrainer(
            task="classification",
            chunk_size=150,
            n_hash_features=128,
            random_state=42,
        )
        trainer.fit_from_file(train_p, target_col="target", text_cols=["notes"])
        preds = trainer.predict(df.iloc[700:].drop(columns=["target"]))
        assert preds.shape == (200,)
        assert accuracy_score(df.iloc[700:]["target"], preds) > 0.80
        summary = trainer.get_scale_summary()
        assert summary["mode"] == "partial_fit"
        assert summary["n_rows_seen"] == 700
        json.dumps(summary, allow_nan=False)

    def test_partial_fit_regression_clustering_and_dim_reduction(self, tmp_path: Path):
        from acdso import ACDSOOutOfCoreTrainer

        # Regression
        X_reg, y_reg = make_regression(n_samples=600, n_features=6, noise=2.0, random_state=42)
        df_reg = pd.DataFrame(X_reg, columns=[f"f{i}" for i in range(6)])
        df_reg["y"] = y_reg
        reg_p = tmp_path / "reg.csv"
        df_reg.iloc[:480].to_csv(reg_p, index=False)

        reg = ACDSOOutOfCoreTrainer(task="regression", chunk_size=120, random_state=42)
        reg.fit_from_file(reg_p, target_col="y")
        reg_preds = reg.predict(df_reg.iloc[480:].drop(columns=["y"]))
        assert reg_preds.shape == (120,)
        assert r2_score(df_reg.iloc[480:]["y"], reg_preds) > 0.80

        # Clustering + streaming predict_to_file
        X_blobs, _ = make_blobs(n_samples=450, centers=3, n_features=4, cluster_std=0.5, random_state=42)
        df_cl = pd.DataFrame(X_blobs, columns=["c0", "c1", "c2", "c3"])
        cl_in = tmp_path / "cl_in.csv"
        cl_out = tmp_path / "cl_out.csv"
        df_cl.to_csv(cl_in, index=False)

        cl = ACDSOOutOfCoreTrainer(task="clustering", chunk_size=100, n_clusters=3, random_state=42)
        cl.fit_from_file(cl_in)
        out_path = cl.predict_to_file(cl_in, cl_out, chunk_size=100)
        assert out_path.is_file()
        out_df = pd.read_csv(out_path)
        assert len(out_df) == 450
        assert "cluster" in out_df.columns
        assert set(out_df["cluster"].unique()).issubset({0, 1, 2})
        assert cl.score(df_cl) > 0.4

        # Dim reduction (pca alias)
        pca_tr = ACDSOOutOfCoreTrainer(task="pca", chunk_size=100, n_components=2, random_state=42)
        pca_tr.fit_from_file(cl_in)
        proj = pca_tr.transform(df_cl)
        assert proj.shape == (450, 2)
        assert np.all(np.isfinite(proj))
        eval_res = pca_tr.evaluate(df_cl)
        assert eval_res["task"] == "dim_reduction"
        json.dumps(eval_res, allow_nan=False)

    def test_subsample_mode_and_algorithm_alias(self, tmp_path: Path):
        from acdso import ACDSOOutOfCoreTrainer

        X, y = make_regression(n_samples=500, n_features=5, noise=1.0, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        df["y"] = y
        p = tmp_path / "sub_reg.csv"
        df.iloc[:400].to_csv(p, index=False)

        tr_mode = ACDSOOutOfCoreTrainer(
            task="regression", mode="subsample", chunk_size=100, max_in_memory_rows=180, random_state=42
        )
        tr_mode.fit_from_file(p, target_col="y")
        preds = tr_mode.predict(df.iloc[400:].drop(columns=["y"]))
        assert r2_score(df.iloc[400:]["y"], preds) > 0.80
        assert tr_mode.get_scale_summary()["subsampled"] is True

        tr_alias = ACDSOOutOfCoreTrainer(
            task="regression", algorithm="subsample", chunk_size=100, max_in_memory_rows=180, random_state=42
        )
        tr_alias.fit_from_file(p, target_col="y")
        assert tr_alias.mode_ == "subsample"

    def test_edge_cases_chunk_one_nan_inf_single_pass_generator_and_errors(self, tmp_path: Path):
        from acdso import ACDSOOutOfCoreTrainer

        # chunk_size=1 + single-pass generator + unseen class mid-stream + NaN/inf
        rows = []
        for i in range(80):
            cls = 0 if i < 30 else (1 if i < 60 else 2)
            x1 = float(cls) * 3.0 + (0.1 if i % 2 == 0 else -0.1)
            x2 = np.nan if i % 10 == 0 else float(cls) * -2.0
            rows.append({"x1": x1, "x2": x2, "target": cls})
        df = pd.DataFrame(rows)
        df.loc[5, "x1"] = np.inf

        def _gen():
            for i in range(len(df)):
                yield df.iloc[i : i + 1]

        tr = ACDSOOutOfCoreTrainer(task="classification", chunk_size=1, random_state=42)
        tr.fit_chunks(_gen(), target_col="target")
        preds = tr.predict(df.drop(columns=["target"]))
        assert len(preds) == 80
        assert set(tr.classes_.tolist()) == {0, 1, 2}

        # Error paths
        unfitted = ACDSOOutOfCoreTrainer(task="classification")
        with pytest.raises(RuntimeError):
            unfitted.predict(df.drop(columns=["target"]))
        with pytest.raises(RuntimeError):
            unfitted.predict_to_file(tmp_path / "missing.csv", tmp_path / "out.csv")
        with pytest.raises(RuntimeError):
            unfitted.get_scale_summary()

        with pytest.raises(FileNotFoundError):
            unfitted.fit_from_file(tmp_path / "missing.csv", target_col="target")

        hdr_only = tmp_path / "hdr_only.csv"
        hdr_only.write_text("x1,x2,target\n", encoding="utf-8")
        with pytest.raises(ValueError):
            unfitted.fit_from_file(hdr_only, target_col="target")

        with pytest.raises(ValueError):
            ACDSOOutOfCoreTrainer(task="invalid_task").fit(df.drop(columns=["target"]), df["target"])
        with pytest.raises(ValueError):
            ACDSOOutOfCoreTrainer(task="classification", chunk_size=0).fit(
                df.drop(columns=["target"]), df["target"]
            )
        with pytest.raises(ValueError):
            tr.transform(df.drop(columns=["target"]))

    def test_25k_row_streaming_bounded_memory_tracemalloc(self, tmp_path: Path):
        from acdso import ACDSOOutOfCoreTrainer
        from acdso.scale import reservoir_sample_file

        rng = np.random.RandomState(42)
        n = 25_000
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        x3 = rng.randn(n)
        y = (2.0 * x1 - 1.8 * x2 + 0.7 * x3 > 0).astype(int)
        train_df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})
        train_p = tmp_path / "big_25k.csv"
        train_df.to_csv(train_p, index=False)

        test_df = train_df.iloc[:1000].drop(columns=["target"])
        test_p = tmp_path / "big_test.csv"
        test_df.to_csv(test_p, index=False)
        out_p = tmp_path / "big_preds.csv"

        tracemalloc.start()
        try:
            sample_df, total_rows = reservoir_sample_file(train_p, max_rows=1500, chunk_size=1500, random_state=42)
            assert total_rows == n and len(sample_df) == 1500

            tr = ACDSOOutOfCoreTrainer(
                task="classification", chunk_size=1500, max_in_memory_rows=2000, random_state=42
            )
            tr.fit_from_file(train_p, target_col="target")
            tr.predict_to_file(test_p, out_p, chunk_size=500)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak_bytes < 80 * 1024 * 1024
        preds_df = pd.read_csv(out_p)
        assert len(preds_df) == 1000
        assert "target_predicted" in preds_df.columns
        assert accuracy_score(y[:1000], preds_df["target_predicted"]) >= 0.80

    def test_dim_reduction_chunk_size_one_processes_all_rows(self):
        from acdso.scale import ACDSOOutOfCoreTrainer

        rng = np.random.RandomState(42)
        X = np.vstack(
            [
                np.column_stack(
                    [rng.randn(64) * 10.0, rng.randn(64) * 1e-3, rng.randn(64) * 1e-3, rng.randn(64) * 1e-3]
                ),
                np.column_stack(
                    [rng.randn(300) * 1e-3, rng.randn(300) * 100.0, rng.randn(300) * 50.0, rng.randn(300) * 1e-3]
                ),
            ]
        )
        df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])

        tr_1 = ACDSOOutOfCoreTrainer(task="dim_reduction", chunk_size=1, n_components=2, random_state=42)
        tr_1.fit_chunks(df.iloc[i : i + 1] for i in range(len(df)))
        assert tr_1.estimator_.n_samples_seen_ == 364

        tr_50 = ACDSOOutOfCoreTrainer(task="dim_reduction", chunk_size=50, n_components=2, random_state=42)
        tr_50.fit_chunks(df.iloc[i : i + 50] for i in range(0, len(df), 50))
        assert tr_50.estimator_.n_samples_seen_ == 364

        # Both capture the post-warmup f1/f2 dominant subspace (little weight on f3)
        proj_1 = tr_1.transform(df.iloc[-10:])
        assert proj_1.shape == (10, 2)
        assert np.all(np.isfinite(proj_1))
        assert np.allclose(tr_1.estimator_.explained_variance_ratio_, tr_50.estimator_.explained_variance_ratio_, atol=0.1)

        # Boundary: 65 rows (64 warmup + 1 post-warmup) drops 0 rows
        tr_65 = ACDSOOutOfCoreTrainer(task="dim_reduction", chunk_size=1, n_components=2, random_state=42)
        tr_65.fit_chunks(df.iloc[i : i + 1] for i in range(65))
        assert tr_65.estimator_.n_samples_seen_ == 65

    def test_predict_to_file_bool_chunk_size_and_header_only_dim_reduction(self, tmp_path: Path):
        from acdso.scale import ACDSOOutOfCoreTrainer

        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(80, 4), columns=["f0", "f1", "f2", "f3"])
        in_csv = tmp_path / "in.csv"
        df.to_csv(in_csv, index=False)
        hdr_csv = tmp_path / "hdr_only.csv"
        hdr_csv.write_text("f0,f1,f2,f3\n", encoding="utf-8")

        for mode in ("partial_fit", "subsample"):
            tr = ACDSOOutOfCoreTrainer(task="dim_reduction", mode=mode, chunk_size=20, n_components=2, random_state=42)
            tr.fit_from_file(in_csv)

            with pytest.raises(ValueError):
                tr.predict_to_file(in_csv, tmp_path / f"bad_true_{mode}.csv", chunk_size=True)
            with pytest.raises(ValueError):
                tr.predict_to_file(in_csv, tmp_path / f"bad_false_{mode}.csv", chunk_size=False)

            out_hdr = tmp_path / f"out_hdr_{mode}.csv"
            tr.predict_to_file(hdr_csv, out_hdr)
            out_df = pd.read_csv(out_hdr)
            assert len(out_df) == 0
            assert list(out_df.columns) == ["f0", "f1", "f2", "f3", "component_0", "component_1"]


class TestCoreAndEDAScaleGuardsAndNLPWiring:
    """Tests for M5 additions in src/acdso/core.py and src/acdso/eda.py."""

    def test_core_classifier_and_regressor_wire_nlp_text_branch(self):
        from acdso import ACDSOClassifier, ACDSORegressor

        rng = np.random.RandomState(42)
        n = 120
        num1 = rng.randn(n)
        y_cls = (rng.rand(n) > 0.5).astype(int)
        # Disjoint vocabulary in train vs test phrasing still generalizes via TF-IDF tokens
        pos_phrases = [
            "customer reported outstanding reliability fast onboarding experience",
            "system delivered exceptional performance zero downtime observed",
            "verified positive outcome strong user satisfaction scores",
        ]
        neg_phrases = [
            "critical failure detected repeated outage severe degradation",
            "broken workflow frequent crashes terrible user frustration",
            "unacceptable latency major regression failing health checks",
        ]
        notes = [pos_phrases[i % 3] if v == 1 else neg_phrases[i % 3] for i, v in enumerate(y_cls)]
        df = pd.DataFrame({"num1": num1, "notes": notes})

        clf = ACDSOClassifier(cv=3, max_candidates=3, enable_stacking=False, random_state=42)
        clf.fit(df, y_cls)
        pre = clf.best_estimator_.named_steps["pre"]
        assert "text" in pre.named_transformers_
        assert clf.audit_trail_["text_cols"] == ["notes"]
        assert clf.audit_trail_["text_method"] == "tfidf"

        # Unseen phrasing combining learned positive/negative tokens
        X_unseen = pd.DataFrame(
            {
                "num1": [0.0, 0.0],
                "notes": [
                    "outstanding performance strong satisfaction reliability",
                    "critical failure severe crashes outage",
                ],
            }
        )
        unseen_preds = clf.predict(X_unseen)
        assert list(unseen_preds) == [1, 0]

        # Explicit text_cols + text_method on ACDSORegressor
        y_reg = np.where(y_cls == 1, 10.0 + 0.5 * num1, -10.0 + 0.5 * num1)
        reg = ACDSORegressor(
            cv=3,
            max_candidates=3,
            enable_stacking=False,
            text_cols=["notes"],
            text_method="ngram",
            random_state=42,
        )
        reg.fit(df, y_reg)
        assert "text" in reg.best_estimator_.named_steps["pre"].named_transformers_
        assert reg.audit_trail_["text_method"] == "ngram"

    def test_core_max_in_memory_rows_scale_guard_and_sklearn_compat(self):
        from acdso import ACDSOClassifier, ACDSORegressor

        X, y = make_classification(n_samples=600, n_features=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        groups = np.repeat(np.arange(30), 20)

        clf = ACDSOClassifier(
            cv=3, max_candidates=2, enable_stacking=False, max_in_memory_rows=180, random_state=42
        )
        assert clf.get_params()["max_in_memory_rows"] == 180
        cloned = clone(clf)
        cloned.set_params(max_in_memory_rows=200)
        assert cloned.get_params()["max_in_memory_rows"] == 200

        clf.fit(df, y, groups=groups)
        guard = clf.audit_trail_["scale_guard"]
        assert guard["subsampled"] is True
        assert guard["original_rows"] == 600
        assert guard["analyzed_rows"] <= 180
        json.dumps(clf.audit_trail_, allow_nan=False)

        # Temporal guard preserves chronological tail
        Xr, yr = make_regression(n_samples=500, n_features=4, random_state=42)
        reg = ACDSORegressor(
            cv=3,
            max_candidates=2,
            enable_stacking=False,
            cv_strategy="temporal",
            max_in_memory_rows=150,
            random_state=42,
        )
        reg.fit(Xr, yr)
        assert reg.audit_trail_["scale_guard"]["subsampled"] is True
        assert reg.audit_trail_["scale_guard"]["analyzed_rows"] == 150

    def test_eda_max_in_memory_rows_and_lazy_anomaly_hook(self):
        from acdso import ACDSOAutoEDA

        rng = np.random.RandomState(42)
        df = pd.DataFrame({"a": rng.randn(800), "b": rng.randn(800) * 2.0})
        df_orig = df.copy()

        eda = ACDSOAutoEDA(max_in_memory_rows=200, random_state=42)
        report = eda.analyze(df, dataset_name="bounded_eda")
        assert report.subsampled is True
        assert report.original_n_rows == 800
        assert report.n_rows == 200
        assert report.sampling_metadata["subsampled"] is True
        assert eda.audit_trail_["subsampled"] is True
        json.dumps(report.to_dict(), allow_nan=False)
        pd.testing.assert_frame_equal(df, df_orig)

        det = eda.fit_anomaly_detector(df, algorithm="isolation_forest", random_state=42)
        assert det is eda.anomaly_detector_
        assert eda.audit_trail_["anomaly_detector_scale_guard"]["subsampled"] is True
        assert eda.audit_trail_["anomaly_detector_scale_guard"]["analyzed_rows"] == 200

        # Duplicate columns handled without full-size copy mutation
        dup_df = pd.DataFrame([[1.0, 2.0]] * 500, columns=["dup", "dup"])
        rep_dup = eda.analyze(dup_df)
        assert rep_dup.subsampled is True and rep_dup.n_rows == 200

    def test_eda_audit_trail_preserves_anomaly_detector_scale_guard(self):
        from acdso import ACDSOAutoEDA

        rng = np.random.RandomState(42)
        df = pd.DataFrame({"a": rng.randn(400), "b": rng.randn(400)})
        eda = ACDSOAutoEDA(max_in_memory_rows=120, random_state=42)
        eda.fit_anomaly_detector(df, algorithm="isolation_forest", random_state=42)
        assert "anomaly_detector_scale_guard" in eda.audit_trail_

        eda.analyze(df, dataset_name="after_anom")
        assert "anomaly_detector_scale_guard" in eda.audit_trail_
        assert eda.audit_trail_["anomaly_detector_scale_guard"]["analyzed_rows"] == 120
        assert eda.audit_trail_["dataset_name"] == "after_anom"
        assert eda.audit_trail_["subsampled"] is True
        json.dumps(eda.audit_trail_, allow_nan=False)

    def test_core_stratified_rare_class_floor_under_max_in_memory_rows(self):
        from acdso import ACDSOClassifier

        rng = np.random.RandomState(42)
        n = 1000
        y = np.zeros(n, dtype=int)
        y[-12:] = 1
        X = pd.DataFrame(
            {
                "f0": rng.randn(n) + y * 3.0,
                "f1": rng.randn(n) - y * 2.0,
                "f2": rng.randn(n),
            }
        )
        clf = ACDSOClassifier(
            cv=3,
            max_candidates=2,
            enable_stacking=False,
            cv_strategy="stratified",
            max_in_memory_rows=100,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.cv_strategy_ == "stratified"
        assert clf.audit_trail_["scale_guard"]["subsampled"] is True
        assert clf.audit_trail_["scale_guard"]["analyzed_rows"] == 100
        preds = clf.predict(X)
        assert preds.shape == (n,)

    def test_core_text_method_case_normalization(self):
        from acdso import ACDSOClassifier, ACDSORegressor

        rng = np.random.RandomState(42)
        n = 60
        y = np.array([i % 2 for i in range(n)])
        df = pd.DataFrame(
            {
                "num1": rng.randn(n),
                "notes": [
                    "positive reliable fast workflow success" if v == 1 else "critical failure broken outage crash"
                    for v in y
                ],
            }
        )
        clf = ACDSOClassifier(
            cv=3, max_candidates=2, enable_stacking=False, text_method="TFIDF", random_state=42
        )
        clf.fit(df, y)
        assert "text" in clf.best_estimator_.named_steps["pre"].named_transformers_
        assert clf.audit_trail_["text_method"] == "tfidf"
        assert clf.get_params()["text_method"] == "TFIDF"

        reg = ACDSORegressor(
            cv=3, max_candidates=2, enable_stacking=False, text_method="  NgRaM ", random_state=42
        )
        reg.fit(df, y.astype(float) * 10.0)
        assert "text" in reg.best_estimator_.named_steps["pre"].named_transformers_
        assert reg.audit_trail_["text_method"] == "ngram"

    def test_core_temporal_grouped_classification_retains_class_coverage(self):
        from acdso import ACDSOClassifier

        rng = np.random.RandomState(42)
        groups = np.repeat(np.arange(40), 10)
        y = np.where(groups < 25, 0, 1)
        X = pd.DataFrame(
            {
                "f0": rng.randn(400) * 0.3 + y * 3.0,
                "f1": rng.randn(400) * 0.3 - y * 2.5,
            }
        )
        clf = ACDSOClassifier(
            cv=3,
            max_candidates=2,
            enable_stacking=False,
            cv_strategy="temporal",
            max_in_memory_rows=150,
            random_state=42,
        )
        clf.fit(X, y, groups=groups)
        assert clf.audit_trail_["selected"]["name"] != "Baseline_Fallback"
        assert clf.audit_trail_["scale_guard"]["subsampled"] is True
        assert clf.audit_trail_["scale_guard"]["analyzed_rows"] <= 150
        preds = clf.predict(X)
        assert set(preds.tolist()) == {0, 1}


class TestPackagingAndLazyExports:
    """Tests for pyproject.toml dependencies/extras and lazy src/acdso/__init__.py exports."""

    def test_pyproject_toml_dependencies_and_extras(self):
        with open(ROOT_DIR / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)

        deps = data["project"]["dependencies"]
        assert any(d.startswith("scipy") for d in deps)
        extras = data["project"]["optional-dependencies"]
        for key in ("scale", "umap", "clustering", "survival", "ranking", "recommender", "io", "all"):
            assert key in extras

        pytest_cfg = data["tool"]["pytest"]["ini_options"]
        assert pytest_cfg["testpaths"] == ["tests"]
        assert pytest_cfg["python_files"] == ["test_*.py"]

    def test_top_level_cold_import_and_all_lazy_exports(self):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(SRC_DIR)
        code = (
            "import sys, time; "
            "t0 = time.perf_counter(); "
            "import acdso; "
            "dt = time.perf_counter() - t0; "
            "heavy = [m for m in ('numpy', 'pandas', 'scipy', 'sklearn', 'acdso.cv', 'acdso.nlp', "
            "'acdso.clustering', 'acdso.anomaly', 'acdso.multilabel', 'acdso.ranking', 'acdso.survival', "
            "'acdso.recommender', 'acdso.scale') if m in sys.modules]; "
            "assert not heavy, f'Eager imports: {heavy}'; "
            "assert dt < 1.0, f'Cold import took {dt:.3f}s'; "
            "expected = {'ACDSOOutOfCoreTrainer', 'iter_file_chunks', 'reservoir_sample_file', 'scale', "
            "'ACDSOClusterer', 'ACDSOAnomalyDetector', 'build_cv_splitter', 'TextFeaturizer', "
            "'ACDSOMultiLabelClassifier', 'ACDSORanker', 'ACDSOSurvivalAnalyzer', 'ACDSORecommender'}; "
            "missing = expected - set(acdso.__all__); "
            "assert not missing, f'Missing from __all__: {missing}'"
        )
        proc = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
