"""Multi-format dataset loading, validation, datetime detection, and safe categorical encoding."""

from __future__ import annotations

import json
import os
import re
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd


def _validate_dataframe_nonempty(df: Any, filepath: str = "") -> None:
    """Raise ValueError if df is None or has 0 rows / 0 columns."""
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) == 0 or len(df.columns) == 0:
        label = f" ({filepath})" if filepath else ""
        raise ValueError(f"Dataset{label} is empty (0 rows or 0 columns).")


def _sanitize_target_rows(
    df: "pd.DataFrame",
    target_cols: List[str],
    min_rows: int = 2,
) -> "pd.DataFrame":
    """Drop rows containing NaN/inf in target_cols and enforce min_rows."""
    import numpy as np
    import pandas as pd

    _validate_dataframe_nonempty(df)
    valid_mask = pd.Series(True, index=df.index)
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"Target column '{col}' not found in dataset.")
        s = df[col]
        col_mask = s.notna()
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            vals = s.to_numpy(dtype=float, na_value=np.nan)
            col_mask = col_mask & np.isfinite(vals)
        valid_mask = valid_mask & col_mask

    n_dropped = int(len(df) - int(valid_mask.sum()))
    if n_dropped > 0:
        print(
            f"  [WARN] Dropped {n_dropped} row(s) containing NaN/inf in target {target_cols}; "
            f"{int(valid_mask.sum())} row(s) remaining."
        )
    df_clean = df.loc[valid_mask].copy()
    if len(df_clean) < min_rows:
        raise ValueError(
            f"Insufficient valid rows ({len(df_clean)}) after dropping NaN/inf target values "
            f"(minimum required: {min_rows})."
        )
    return df_clean


def _is_datetime_series(s: "pd.Series") -> bool:
    """Return True only if Series is genuine datetime or non-numeric datetime strings."""
    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return False
    sample = s.dropna().head(50)
    if sample.empty:
        return False
    sample_str = sample.astype(str).str.strip()
    sample_str = sample_str[sample_str != ""]
    if sample_str.empty:
        return False
    if pd.to_numeric(sample_str, errors="coerce").notna().mean() > 0.8:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample_str, errors="coerce")
    return bool(parsed.notna().mean() > 0.8)


def _detect_datetime_column(df: "pd.DataFrame") -> Optional[str]:
    """Detect a genuine datetime column without misclassifying numeric/bool columns."""
    import pandas as pd

    if df is None or df.empty:
        return None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    dt_kw_re = re.compile(
        r"(?<![a-z])(?:date|time|timestamp|ts|datetime|created|updated|week|period|dt)(?![a-z])"
    )
    for col in df.columns:
        if dt_kw_re.search(str(col).lower()) and _is_datetime_series(df[col]):
            return col
    for col in df.columns:
        if _is_datetime_series(df[col]):
            return col
    return None


def _read_csv_with_fallback(filepath: str) -> "pd.DataFrame":
    """Read CSV with encoding fallbacks and clear EmptyDataError reporting."""
    import pandas as pd

    last_decode_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(filepath, encoding=enc)
        except UnicodeDecodeError as e:
            last_decode_err = e
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"Dataset file is empty (no columns/rows): {filepath}") from e
    raise ValueError(f"Unable to decode CSV file '{filepath}': {last_decode_err}")


def _read_xlsx_stdlib(filepath: str) -> "pd.DataFrame":
    """Pure-stdlib .xlsx reader fallback when openpyxl is not installed."""
    import xml.etree.ElementTree as ET
    import zipfile

    import pandas as pd

    def _col_to_idx(ref: str) -> int:
        letters = "".join(ch for ch in ref if ch.isalpha()).upper()
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        return max(idx - 1, 0)

    def _strip_ns(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    def _coerce_num(val_text: str) -> Any:
        try:
            if "." not in val_text and "e" not in val_text.lower():
                return int(val_text)
            return float(val_text)
        except ValueError:
            return val_text

    with zipfile.ZipFile(filepath, "r") as zf:
        names = set(zf.namelist())
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in names:
            ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_root:
                if _strip_ns(si.tag) == "si":
                    parts = [elem.text or "" for elem in si.iter() if _strip_ns(elem.tag) == "t"]
                    shared_strings.append("".join(parts))

        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in names:
            candidates = sorted(
                n for n in names if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
            )
            if not candidates:
                raise ValueError(f"No worksheet found in .xlsx file: {filepath}")
            sheet_name = candidates[0]

        ws_root = ET.fromstring(zf.read(sheet_name))
        rows_dict: List[Dict[int, Any]] = []
        max_col = 0
        for elem in ws_root.iter():
            if _strip_ns(elem.tag) == "row":
                row_map: Dict[int, Any] = {}
                next_idx = 0
                for cell in elem:
                    if _strip_ns(cell.tag) != "c":
                        continue
                    ref = cell.attrib.get("r", "")
                    col_idx = _col_to_idx(ref) if ref else next_idx
                    next_idx = col_idx + 1
                    cell_type = cell.attrib.get("t", "")
                    val_node = None
                    is_node = None
                    for child in cell:
                        cname = _strip_ns(child.tag)
                        if cname == "v":
                            val_node = child
                        elif cname == "is":
                            is_node = child
                    val: Any = None
                    if cell_type == "s" and val_node is not None and val_node.text is not None:
                        s_idx = int(val_node.text)
                        val = shared_strings[s_idx] if 0 <= s_idx < len(shared_strings) else ""
                    elif cell_type == "inlineStr" and is_node is not None:
                        val = "".join(t.text or "" for t in is_node.iter() if _strip_ns(t.tag) == "t")
                    elif cell_type == "b" and val_node is not None and val_node.text is not None:
                        val = val_node.text.strip() in ("1", "true", "TRUE")
                    elif cell_type in ("str", "e") and val_node is not None:
                        val = val_node.text or ""
                    elif val_node is not None and val_node.text is not None:
                        val = _coerce_num(val_node.text.strip())
                    row_map[col_idx] = val
                    if col_idx + 1 > max_col:
                        max_col = col_idx + 1
                if row_map:
                    rows_dict.append(row_map)

    if not rows_dict or max_col == 0:
        return pd.DataFrame()

    raw_headers = [rows_dict[0].get(i) for i in range(max_col)]
    headers: List[str] = []
    seen: Dict[str, int] = {}
    for i, h in enumerate(raw_headers):
        col_name = str(h).strip() if h is not None and str(h).strip() else f"Unnamed: {i}"
        if col_name in seen:
            seen[col_name] += 1
            col_name = f"{col_name}.{seen[col_name]}"
        else:
            seen[col_name] = 0
        headers.append(col_name)

    data_rows = [[r.get(i, None) for i in range(max_col)] for r in rows_dict[1:]]
    return pd.DataFrame(data_rows, columns=headers)


def load_user_file(filepath: str) -> "pd.DataFrame":
    """Load a tabular data file (CSV, Excel, Parquet, JSON, Feather, Pickle)."""
    import pandas as pd

    if not filepath or not os.path.isfile(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        return _read_csv_with_fallback(filepath)
    if ext == ".xlsx":
        try:
            return pd.read_excel(filepath)
        except ImportError:
            return _read_xlsx_stdlib(filepath)
    if ext == ".xls":
        try:
            return pd.read_excel(filepath)
        except ImportError as e:
            raise ImportError(
                f"Cannot read '.xls' file '{filepath}': missing optional dependency ({e}). "
                "Install 'xlrd' or convert the file to .xlsx or .csv."
            ) from e
    if ext == ".parquet":
        try:
            return pd.read_parquet(filepath)
        except ImportError as e:
            raise ImportError(
                f"Cannot read '.parquet' file '{filepath}': missing optional dependency ({e}). "
                "Install 'pyarrow' or 'fastparquet'."
            ) from e
    if ext == ".json":
        try:
            obj = pd.read_json(filepath)
        except ValueError as e:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, (list, dict)):
                    obj = pd.DataFrame(raw)
                else:
                    raise ValueError(f"JSON file '{filepath}' is not tabular.") from e
            except Exception as inner_e:
                raise ValueError(f"JSON file '{filepath}' could not be parsed as a table: {inner_e}") from inner_e
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"JSON file '{filepath}' did not produce a tabular DataFrame.")
        return obj
    if ext == ".feather":
        try:
            return pd.read_feather(filepath)
        except ImportError as e:
            raise ImportError(
                f"Cannot read '.feather' file '{filepath}': missing optional dependency ({e}). "
                "Install 'pyarrow'."
            ) from e
    if ext in (".pkl", ".pickle"):
        obj = pd.read_pickle(filepath)
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, (list, dict)):
            try:
                return pd.DataFrame(obj)
            except Exception as e:
                raise ValueError(f"Pickle file '{filepath}' is not tabular: {e}") from e
        raise ValueError(f"Pickle file '{filepath}' does not contain a pandas DataFrame (got {type(obj).__name__}).")
    return _read_csv_with_fallback(filepath)


def _encode_categoricals_safe(
    X,
    cat_cols=None,
    max_cardinality=100,
    encoding_meta: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
    feature_columns: Optional[List[str]] = None,
):
    """One-hot encode low-cardinality categoricals; frequency-encode the rest."""
    import pandas as pd

    X = X.copy()
    if cat_cols is None:
        cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if encoding_meta is not None:
        high_card_freqs = encoding_meta.get("high_card_freqs", {}) or {}
        low_card_cols = encoding_meta.get("low_card_cols", []) or []
        for c, freq_map in high_card_freqs.items():
            if c in X.columns:
                X[c] = X[c].map(freq_map).fillna(0.0).astype("float64")
        low_to_encode = [c for c in low_card_cols if c in X.columns]
        extra_cats = [
            c
            for c in cat_cols
            if c in X.columns and c not in high_card_freqs and c not in low_card_cols
        ]
        low_to_encode.extend(extra_cats)
        if low_to_encode:
            X = pd.get_dummies(X, columns=low_to_encode, drop_first=False, dtype=int)
        return (X, encoding_meta) if return_meta else X

    if feature_columns is not None:
        high_card_legacy = [c for c in cat_cols if c in feature_columns and c in X.columns]
        low_card_legacy = [c for c in cat_cols if c not in feature_columns and c in X.columns]
        high_card_freqs = {}
        for c in high_card_legacy:
            freq_map = X[c].value_counts(dropna=False).to_dict()
            high_card_freqs[c] = freq_map
            X[c] = X[c].map(freq_map).fillna(0.0).astype("float64")
        if low_card_legacy:
            X = pd.get_dummies(X, columns=low_card_legacy, drop_first=False, dtype=int)
        meta = {"low_card_cols": low_card_legacy, "high_card_freqs": high_card_freqs}
        return (X, meta) if return_meta else X

    low_card = [c for c in cat_cols if X[c].nunique(dropna=False) <= max_cardinality]
    high_card = [c for c in cat_cols if c not in low_card]
    high_card_freqs: Dict[str, Dict[Any, Any]] = {}

    if high_card:
        print(
            f"    [WARN] {len(high_card)} high-cardinality column(s) frequency-encoded "
            f"instead of one-hot (> {max_cardinality} uniques): {high_card}"
        )
        for c in high_card:
            freq_map = X[c].value_counts(dropna=False).to_dict()
            high_card_freqs[c] = freq_map
            X[c] = X[c].map(freq_map).fillna(0.0).astype("float64")

    if low_card:
        X = pd.get_dummies(X, columns=low_card, drop_first=True, dtype=int)

    meta = {"low_card_cols": low_card, "high_card_freqs": high_card_freqs}
    return (X, meta) if return_meta else X
