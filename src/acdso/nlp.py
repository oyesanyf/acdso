"""
ACDSO - Classical NLP & Free-Text Featurization
===============================================

Content-based free-text column detection and scikit-learn compatible
featurization (TF-IDF, n-gram counts, LDA, NMF, and combined modes).
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

_URL_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_BOOL_STRINGS = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"})
_VALID_METHODS = frozenset({"tfidf", "ngram", "lda", "nmf", "combined"})


def _is_missing_scalar(val: Any) -> bool:
    if val is None or val is pd.NA or val is pd.NaT:
        return True
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return True
    try:
        res = pd.isna(val)
        return bool(res) if isinstance(res, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def detect_text_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[Iterable[Any]] = None,
    min_avg_len: float = 24.0,
    min_avg_words: float = 3.0,
) -> List[Any]:
    """Identify free-text columns using content morphology rather than column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to inspect.
    exclude_cols : iterable, optional
        Column names to skip unconditionally.
    min_avg_len : float, default=24.0
        Minimum average non-empty character length required.
    min_avg_words : float, default=3.0
        Minimum average whitespace-delimited token count required.

    Returns
    -------
    list
        Column keys detected as unstructured free text, in DataFrame order.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    exclude = set(exclude_cols) if exclude_cols is not None else set()
    detected: List[Any] = []

    for idx, col in enumerate(df.columns):
        if col in exclude:
            continue

        s = df.iloc[:, idx]
        dtype = s.dtype

        if (
            pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_datetime64_any_dtype(dtype)
            or pd.api.types.is_timedelta64_dtype(dtype)
        ):
            continue

        if not (
            dtype == object
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            continue

        non_null = s.dropna()
        if len(non_null) == 0:
            continue

        if dtype == object:
            head_raw = non_null.head(100)
            if all(isinstance(v, (bool, np.bool_)) for v in head_raw):
                continue
            if all(
                isinstance(v, (int, float, np.number))
                and not isinstance(v, (bool, np.bool_))
                for v in head_raw
            ):
                continue

        sample = non_null.head(500).astype(str).str.strip()
        sample = sample[sample != ""]
        if len(sample) == 0:
            continue

        # Reject boolean-like tokens
        if float(sample.str.lower().isin(_BOOL_STRINGS).mean()) >= 0.90:
            continue

        # Reject numeric strings (e.g., "123.45", "00123", "1,234.5")
        num_cand = sample.str.replace(",", "", regex=False)
        if float(pd.to_numeric(num_cand, errors="coerce").notna().mean()) >= 0.85:
            continue

        # Reject datetime strings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt_ratio = float(
                pd.to_datetime(sample.head(100), errors="coerce", format="mixed")
                .notna()
                .mean()
            )
        if dt_ratio >= 0.80:
            continue

        # Reject pure URLs, emails, and UUIDs
        head_100 = sample.head(100)
        id_hits = (
            head_100.str.match(_URL_RE)
            | head_100.str.match(_EMAIL_RE)
            | head_100.str.match(_UUID_RE)
        )
        if float(id_hits.mean()) >= 0.80:
            continue

        # Reject low-alphabetic content (e.g., space-separated numeric codes or hex blocks)
        total_non_space = 0
        total_alpha = 0
        for val in head_100:
            for ch in val:
                if not ch.isspace():
                    total_non_space += 1
                    if ch.isalpha():
                        total_alpha += 1
        if (total_alpha / max(1, total_non_space)) < 0.30:
            continue

        avg_len = float(sample.str.len().mean())
        avg_words = float(sample.str.split().str.len().mean())
        if avg_len < float(min_avg_len) or avg_words < float(min_avg_words):
            continue

        detected.append(col)

    return detected


class TextFeaturizer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible classical NLP text featurizer.

    Supports TF-IDF, n-gram counts, Latent Dirichlet Allocation (LDA),
    Non-negative Matrix Factorization (NMF), and combined representations
    over single or multiple text columns.
    """

    def __init__(
        self,
        method: str = "tfidf",
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: Optional[int] = 500,
        n_topics: int = 10,
        min_df: Union[int, float] = 1,
        stop_words: Optional[Union[str, List[str]]] = "english",
        random_state: Optional[int] = 42,
    ) -> None:
        self.method = method
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.n_topics = n_topics
        self.min_df = min_df
        self.stop_words = stop_words
        self.random_state = random_state

    def _coerce_input(self, X: Any) -> Tuple[List[List[str]], List[str]]:
        if isinstance(X, pd.DataFrame):
            col_names = [str(c) for c in X.columns]
            cols_docs: List[List[str]] = []
            for i in range(X.shape[1]):
                s = X.iloc[:, i]
                vals = s.astype(object).where(s.notna(), "").astype(str).tolist()
                cols_docs.append(vals)
            return cols_docs, col_names

        if isinstance(X, pd.Series):
            col_name = str(X.name) if X.name is not None else "text_0"
            vals = X.astype(object).where(X.notna(), "").astype(str).tolist()
            return [vals], [col_name]

        if isinstance(X, np.ndarray):
            arr = X
        elif isinstance(X, (list, tuple)):
            if len(X) == 0:
                return [[]], ["text_0"]
            first = X[0]
            if isinstance(first, (list, tuple, np.ndarray, pd.Series)):
                arr = np.asarray(X, dtype=object)
            else:
                vals = ["" if _is_missing_scalar(v) else str(v) for v in X]
                return [vals], ["text_0"]
        else:
            arr = np.asarray(X, dtype=object)

        if arr.ndim == 0:
            raise ValueError("Expected 1D or 2D text input, got 0D scalar.")
        if arr.ndim == 1:
            vals = ["" if _is_missing_scalar(v) else str(v) for v in arr.tolist()]
            return [vals], ["text_0"]
        if arr.ndim == 2:
            cols_docs = []
            col_names = []
            for j in range(arr.shape[1]):
                vals = ["" if _is_missing_scalar(v) else str(v) for v in arr[:, j].tolist()]
                cols_docs.append(vals)
                col_names.append(f"text_{j}")
            return cols_docs, col_names

        raise ValueError(f"Expected 1D or 2D text input, got {arr.ndim}D array.")

    def _fit_vectorizer(
        self,
        vec_cls: Any,
        docs: List[str],
        ngram_range: Tuple[int, int],
        max_features: Optional[int],
    ) -> Tuple[Optional[Any], Optional[Any], bool]:
        n_docs = len(docs)
        eff_min_df: Union[int, float] = (
            1 if (isinstance(self.min_df, int) and self.min_df > n_docs) else self.min_df
        )
        vec = vec_cls(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=eff_min_df,
            stop_words=self.stop_words,
        )
        try:
            mat = vec.fit_transform(docs)
            if mat.shape[1] == 0:
                return None, None, True
            return vec, mat, False
        except ValueError:
            if eff_min_df != 1:
                vec = vec_cls(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    min_df=1,
                    stop_words=self.stop_words,
                )
                try:
                    mat = vec.fit_transform(docs)
                    if mat.shape[1] == 0:
                        return None, None, True
                    return vec, mat, False
                except ValueError:
                    return None, None, True
            return None, None, True

    def _fit_single_column(
        self,
        docs: List[str],
        ngram_range: Tuple[int, int],
        max_features: Optional[int],
        n_topics: int,
    ) -> Dict[str, Any]:
        method = self.method

        if method == "tfidf":
            vec, mat, is_empty = self._fit_vectorizer(
                TfidfVectorizer, docs, ngram_range, max_features
            )
            if is_empty or vec is None or mat is None:
                return {
                    "is_empty_vocab": True,
                    "width": 1,
                    "feature_names": ["tfidf_empty_fallback"],
                }
            feat_names = [f"tfidf_{t}" for t in vec.get_feature_names_out()]
            return {
                "is_empty_vocab": False,
                "tfidf": vec,
                "width": len(feat_names),
                "feature_names": feat_names,
            }

        if method == "ngram":
            vec, mat, is_empty = self._fit_vectorizer(
                CountVectorizer, docs, ngram_range, max_features
            )
            if is_empty or vec is None or mat is None:
                return {
                    "is_empty_vocab": True,
                    "width": 1,
                    "feature_names": ["ngram_empty_fallback"],
                }
            feat_names = [f"ngram_{t}" for t in vec.get_feature_names_out()]
            return {
                "is_empty_vocab": False,
                "ngram": vec,
                "width": len(feat_names),
                "feature_names": feat_names,
            }

        if method == "lda":
            vec, mat, is_empty = self._fit_vectorizer(
                CountVectorizer, docs, ngram_range, max_features
            )
            feat_names = [f"lda_topic_{i}" for i in range(n_topics)]
            if is_empty or vec is None or mat is None:
                return {
                    "is_empty_vocab": True,
                    "width": n_topics,
                    "feature_names": feat_names,
                }
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=self.random_state,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lda.fit(mat)
            return {
                "is_empty_vocab": False,
                "ngram": vec,
                "lda": lda,
                "width": n_topics,
                "feature_names": feat_names,
            }

        if method == "nmf":
            vec, mat, is_empty = self._fit_vectorizer(
                TfidfVectorizer, docs, ngram_range, max_features
            )
            feat_names = [f"nmf_topic_{i}" for i in range(n_topics)]
            if is_empty or vec is None or mat is None:
                return {
                    "is_empty_vocab": True,
                    "width": n_topics,
                    "feature_names": feat_names,
                }
            n_docs_v, n_vocab = mat.shape
            nmf_init = "nndsvda" if n_topics <= min(n_docs_v, n_vocab) else "random"
            nmf = NMF(
                n_components=n_topics,
                init=nmf_init,
                random_state=self.random_state,
                max_iter=300,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nmf.fit(mat)
            return {
                "is_empty_vocab": False,
                "tfidf": vec,
                "nmf": nmf,
                "width": n_topics,
                "feature_names": feat_names,
            }

        # method == "combined"
        tfidf_vec, mat_tfidf, empty_tfidf = self._fit_vectorizer(
            TfidfVectorizer, docs, ngram_range, max_features
        )
        count_vec, mat_counts, empty_counts = self._fit_vectorizer(
            CountVectorizer, docs, ngram_range, max_features
        )
        lda_names = [f"lda_topic_{i}" for i in range(n_topics)]
        nmf_names = [f"nmf_topic_{i}" for i in range(n_topics)]
        if (
            empty_tfidf
            or empty_counts
            or tfidf_vec is None
            or count_vec is None
            or mat_tfidf is None
            or mat_counts is None
        ):
            feat_names = ["tfidf_empty_fallback"] + lda_names + nmf_names
            return {
                "is_empty_vocab": True,
                "width": len(feat_names),
                "feature_names": feat_names,
            }

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=self.random_state,
        )
        n_docs_v, n_vocab = mat_tfidf.shape
        nmf_init = "nndsvda" if n_topics <= min(n_docs_v, n_vocab) else "random"
        nmf = NMF(
            n_components=n_topics,
            init=nmf_init,
            random_state=self.random_state,
            max_iter=300,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lda.fit(mat_counts)
            nmf.fit(mat_tfidf)

        tfidf_names = [f"tfidf_{t}" for t in tfidf_vec.get_feature_names_out()]
        feat_names = tfidf_names + lda_names + nmf_names
        return {
            "is_empty_vocab": False,
            "tfidf": tfidf_vec,
            "ngram": count_vec,
            "lda": lda,
            "nmf": nmf,
            "width": len(feat_names),
            "feature_names": feat_names,
        }

    def _build_feature_names_out(self, col_labels: List[str]) -> np.ndarray:
        names: List[str] = []
        multi_col = len(self._col_specs) > 1
        for j, spec in enumerate(self._col_specs):
            prefix = f"{col_labels[j]}__" if multi_col else ""
            for base_name in spec["feature_names"]:
                names.append(f"{prefix}{base_name}")
        return np.asarray(names, dtype=object)

    def fit(self, X: Any, y: Any = None) -> "TextFeaturizer":
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method {self.method!r}. Expected one of {sorted(_VALID_METHODS)}."
            )
        n_topics = int(self.n_topics)
        if n_topics < 1:
            raise ValueError(f"n_topics must be >= 1, got {self.n_topics!r}.")

        ngram_range = tuple(int(v) for v in self.ngram_range)
        if len(ngram_range) != 2 or ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
            raise ValueError(f"Invalid ngram_range {self.ngram_range!r}.")

        max_features = None if self.max_features is None else int(self.max_features)

        cols_docs, col_names = self._coerce_input(X)
        if len(cols_docs) == 0 or len(cols_docs[0]) == 0:
            raise ValueError("Cannot fit TextFeaturizer on empty input (0 samples or 0 columns).")

        self.n_features_in_ = len(cols_docs)
        self._col_names = col_names
        if isinstance(X, pd.DataFrame) and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        elif hasattr(self, "feature_names_in_"):
            delattr(self, "feature_names_in_")

        self._col_specs = [
            self._fit_single_column(docs, ngram_range, max_features, n_topics)
            for docs in cols_docs
        ]
        self.feature_names_out_ = self._build_feature_names_out(col_names)
        return self

    def transform(self, X: Any) -> Union[np.ndarray, sparse.csr_matrix]:
        check_is_fitted(self, ["_col_specs", "n_features_in_", "feature_names_out_"])
        cols_docs, _ = self._coerce_input(X)
        if len(cols_docs) != self.n_features_in_:
            raise ValueError(
                f"X has {len(cols_docs)} column(s), expected {self.n_features_in_}."
            )

        is_sparse_method = self.method in {"tfidf", "ngram", "combined"}
        n_rows = len(cols_docs[0])
        total_width = len(self.feature_names_out_)

        if n_rows == 0:
            if is_sparse_method:
                return sparse.csr_matrix((0, total_width), dtype=np.float64)
            return np.empty((0, total_width), dtype=np.float64)

        blocks: List[Any] = []
        for docs, spec in zip(cols_docs, self._col_specs):
            width = spec["width"]
            if spec["is_empty_vocab"]:
                if is_sparse_method:
                    blocks.append(sparse.csr_matrix((n_rows, width), dtype=np.float64))
                else:
                    blocks.append(np.zeros((n_rows, width), dtype=np.float64))
                continue

            if self.method == "tfidf":
                blocks.append(spec["tfidf"].transform(docs).astype(np.float64))
            elif self.method == "ngram":
                blocks.append(spec["ngram"].transform(docs).astype(np.float64))
            elif self.method == "lda":
                counts = spec["ngram"].transform(docs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = spec["lda"].transform(counts)
                blocks.append(
                    np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                )
            elif self.method == "nmf":
                tfidf = spec["tfidf"].transform(docs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = spec["nmf"].transform(tfidf)
                blocks.append(
                    np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                )
            else:  # combined
                mat_tfidf = spec["tfidf"].transform(docs).astype(np.float64)
                counts = spec["ngram"].transform(docs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mat_lda = np.nan_to_num(
                        spec["lda"].transform(counts), nan=0.0, posinf=0.0, neginf=0.0
                    ).astype(np.float64)
                    mat_nmf = np.nan_to_num(
                        spec["nmf"].transform(mat_tfidf), nan=0.0, posinf=0.0, neginf=0.0
                    ).astype(np.float64)
                blocks.append(
                    sparse.hstack(
                        [mat_tfidf, sparse.csr_matrix(mat_lda), sparse.csr_matrix(mat_nmf)],
                        format="csr",
                    )
                )

        if is_sparse_method:
            return blocks[0] if len(blocks) == 1 else sparse.hstack(blocks, format="csr")
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, ["_col_specs", "n_features_in_", "feature_names_out_"])
        if input_features is None:
            return self.feature_names_out_.copy()
        labels = [str(c) for c in input_features]
        if len(labels) != self.n_features_in_:
            raise ValueError(
                f"input_features has {len(labels)} feature(s), expected {self.n_features_in_}."
            )
        return self._build_feature_names_out(labels)
