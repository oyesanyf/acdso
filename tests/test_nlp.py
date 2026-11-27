"""Tests for Milestone M2: Classical NLP & Free-Text Featurization (R4)."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from acdso import ACDSOClassifier, ACDSORegressor
from acdso.auto_preprocessing import (
    AutoPreprocessor,
    PreprocessingPlan,
    create_eda_preprocessor,
)
from acdso.eda import ACDSOAutoEDA
from acdso.nlp import TextFeaturizer, detect_text_columns
from acdso.utils import build_preprocessor, split_columns


def _make_sample_corpus(n: int = 36) -> list[str]:
    templates = [
        "Customer reported intermittent timeout errors when uploading large financial spreadsheets to the portal.",
        "Excellent build quality and very responsive customer support resolved our onboarding questions quickly.",
        "Shipment arrived two days late with damaged outer packaging and missing warranty documentation.",
        "Machine learning pipeline improved quarterly churn prediction accuracy across regional cohorts.",
        "Application crashed during export of multi-year audit logs on Windows workstations.",
        "Smooth installation process with clear API documentation and reliable daily batch processing.",
    ]
    return [templates[i % len(templates)] for i in range(n)]


# ---------------------------------------------------------------------------
# A. detect_text_columns tests
# ---------------------------------------------------------------------------


class TestDetectTextColumns:
    def test_separates_free_text_from_non_text_columns(self):
        n = 24
        df = pd.DataFrame(
            {
                "num_float": np.linspace(0.0, 10.0, n),
                "num_int": np.arange(n),
                "flag_bool": [i % 2 == 0 for i in range(n)],
                "obj_bool": pd.Series([True, False] * (n // 2), dtype=object),
                "bool_str": ["yes" if i % 2 == 0 else "no" for i in range(n)],
                "low_cat": ["red", "green", "blue"] * (n // 3),
                "zip_code": [f"Z_{i:04d}" for i in range(n)],
                "order_id": [f"ORD-{1000 + i}" for i in range(n)],
                "txt": pd.Series(["a", "b", "c"] * (n // 3), dtype="string"),
                "numeric_str": [f"{100 + i}.25" for i in range(n)],
                "date_str": [f"2024-01-{(i % 28) + 1:02d}" for i in range(n)],
                "url_col": [f"https://example.com/resource/item/{i}/details" for i in range(n)],
                "email_col": [f"user{i}@organization-domain.example.com" for i in range(n)],
                "uuid_col": [f"123e4567-e89b-12d3-a456-42661417{i:04d}" for i in range(n)],
                "digit_tokens": ["0012 3456 7890 1234 5678" for _ in range(n)],
                "review_obj": _make_sample_corpus(n),
                "notes_str": pd.Series(_make_sample_corpus(n), dtype="string"),
            }
        )

        detected = detect_text_columns(df)
        assert detected == ["review_obj", "notes_str"]

    def test_exclude_cols_and_threshold_overrides(self):
        df = pd.DataFrame(
            {
                "review": _make_sample_corpus(12),
                "summary": _make_sample_corpus(12),
                "short_phrase": ["red car", "blue sky", "fast train", "green hill"] * 3,
            }
        )

        assert detect_text_columns(df, exclude_cols=["review"]) == ["summary"]
        assert "short_phrase" not in detect_text_columns(df)
        detected_low = detect_text_columns(df, min_avg_len=6.0, min_avg_words=2.0)
        assert detected_low == ["review", "summary", "short_phrase"]

    def test_missing_values_all_nan_and_empty_inputs(self):
        corpus = _make_sample_corpus(15)
        mixed_nulls = [
            corpus[i] if i % 3 != 0 else (None if i % 2 == 0 else np.nan)
            for i in range(15)
        ]
        df = pd.DataFrame(
            {
                "partial_text": mixed_nulls,
                "all_nan": [np.nan] * 15,
                "blank_strings": ["   ", ""] * 7 + [""],
                "cat_dtype_text": pd.Series(corpus, dtype="category"),
            }
        )
        detected = detect_text_columns(df)
        assert detected == ["partial_text", "cat_dtype_text"]

        assert detect_text_columns(df.iloc[0:0]) == []
        assert detect_text_columns(None) == []


# ---------------------------------------------------------------------------
# B. TextFeaturizer unit & edge-case tests
# ---------------------------------------------------------------------------


class TestTextFeaturizer:
    @pytest.mark.parametrize("method", ["tfidf", "ngram", "lda", "nmf", "combined"])
    def test_all_five_methods_produce_valid_outputs_and_names(self, method):
        corpus = _make_sample_corpus(30)
        tf = TextFeaturizer(
            method=method,
            ngram_range=(1, 2),
            max_features=40,
            n_topics=4,
            random_state=42,
        )
        out = tf.fit_transform(pd.DataFrame({"text": corpus}))
        arr = out.toarray() if sparse.issparse(out) else np.asarray(out)

        assert arr.shape[0] == len(corpus)
        assert arr.shape[1] > 0
        assert np.all(np.isfinite(arr))

        if method in {"lda", "nmf"}:
            assert not sparse.issparse(out)
            assert arr.shape == (len(corpus), 4)
            assert np.all(arr >= 0.0)
        else:
            assert sparse.issparse(out)

        names = tf.get_feature_names_out()
        assert isinstance(names, np.ndarray)
        assert len(names) == arr.shape[1]
        if method == "combined":
            assert any(str(n).startswith("tfidf_") for n in names)
            assert any(str(n).startswith("lda_topic_") for n in names)
            assert any(str(n).startswith("nmf_topic_") for n in names)

    def test_invalid_params_raise_value_error(self):
        with pytest.raises(ValueError, match="Unknown method"):
            TextFeaturizer(method="word2vec").fit(["hello world example"])
        with pytest.raises(ValueError, match="n_topics"):
            TextFeaturizer(method="lda", n_topics=0).fit(["hello world example"])
        with pytest.raises(ValueError, match="ngram_range"):
            TextFeaturizer(ngram_range=(2, 1)).fit(["hello world example"])
        with pytest.raises(ValueError, match="empty input"):
            TextFeaturizer().fit([])

    def test_input_container_polymorphism_single_and_multi_column(self):
        corpus_a = _make_sample_corpus(12)
        corpus_b = [f"secondary diagnostics log entry {i} completed" for i in range(12)]

        # 1D Series, 1D list, 1D ndarray
        for container in [
            pd.Series(corpus_a, name="notes"),
            corpus_a,
            np.asarray(corpus_a, dtype=object),
        ]:
            tf = TextFeaturizer(method="tfidf", max_features=20)
            out = tf.fit_transform(container)
            assert out.shape[0] == 12
            assert out.shape[1] == len(tf.get_feature_names_out())

        # 2D DataFrame, 2D ndarray, 2D list
        df_multi = pd.DataFrame({"col_a": corpus_a, "col_b": corpus_b})
        tf_df = TextFeaturizer(method="nmf", n_topics=3, max_features=30, random_state=42)
        out_df = tf_df.fit_transform(df_multi)
        assert out_df.shape == (12, 6)
        names = tf_df.get_feature_names_out()
        assert list(names) == [
            "col_a__nmf_topic_0",
            "col_a__nmf_topic_1",
            "col_a__nmf_topic_2",
            "col_b__nmf_topic_0",
            "col_b__nmf_topic_1",
            "col_b__nmf_topic_2",
        ]

        arr_2d = df_multi.to_numpy(dtype=object)
        out_arr = TextFeaturizer(method="tfidf", max_features=20).fit_transform(arr_2d)
        assert out_arr.shape[0] == 12

        list_2d = [[a, b] for a, b in zip(corpus_a, corpus_b)]
        out_list_2d = TextFeaturizer(method="lda", n_topics=2, random_state=42).fit_transform(list_2d)
        assert out_list_2d.shape == (12, 4)

    def test_get_feature_names_out_not_fitted_and_input_features(self):
        tf = TextFeaturizer(method="tfidf", max_features=25)
        with pytest.raises(NotFittedError):
            tf.get_feature_names_out()

        df_multi = pd.DataFrame({"t1": _make_sample_corpus(10), "t2": _make_sample_corpus(10)})
        tf.fit(df_multi)
        custom_names = tf.get_feature_names_out(["alpha", "beta"])
        assert any(str(n).startswith("alpha__") for n in custom_names)
        assert any(str(n).startswith("beta__") for n in custom_names)
        with pytest.raises(ValueError):
            tf.get_feature_names_out(["only_one"])

    def test_clone_pickle_and_determinism(self):
        corpus = _make_sample_corpus(18)
        tf = TextFeaturizer(
            method="combined",
            ngram_range=(1, 2),
            max_features=30,
            n_topics=4,
            min_df=1,
            stop_words="english",
            random_state=42,
        )
        tf_cloned = clone(tf)
        assert tf_cloned.get_params() == tf.get_params()

        out1 = tf.fit_transform(corpus).toarray()
        out2 = tf_cloned.fit_transform(corpus).toarray()
        assert np.allclose(out1, out2)

        restored = pickle.loads(pickle.dumps(tf))
        out_pickled = restored.transform(corpus[:5]).toarray()
        assert np.allclose(out1[:5], out_pickled)

    @pytest.mark.parametrize("method", ["tfidf", "ngram", "lda", "nmf", "combined"])
    def test_adversarial_empty_stopwords_nan_tiny_corpus_and_high_min_df(self, method):
        # Empty / all-stopword / punctuation-only / missing values / numeric cells
        bad_docs = pd.DataFrame(
            {"text": ["the and or", "", None, np.nan, "!!!", 42]}
        )
        tf_bad = TextFeaturizer(method=method, n_topics=4, min_df=10, random_state=42)
        out_bad = tf_bad.fit_transform(bad_docs)
        arr_bad = out_bad.toarray() if sparse.issparse(out_bad) else np.asarray(out_bad)
        assert arr_bad.shape[0] == len(bad_docs)
        assert arr_bad.shape[1] >= 1
        assert np.all(np.isfinite(arr_bad))
        assert len(tf_bad.get_feature_names_out()) == arr_bad.shape[1]

        # Transform unseen mixed missing + numeric values
        out_unseen = tf_bad.transform(pd.DataFrame({"text": [None, np.nan, 99.5]}))
        arr_unseen = out_unseen.toarray() if sparse.issparse(out_unseen) else np.asarray(out_unseen)
        assert arr_unseen.shape == (3, arr_bad.shape[1])
        assert np.all(np.isfinite(arr_unseen))

        # Tiny corpus where n_topics > min(n_docs, n_vocab)
        tiny_docs = ["alpha beta", "gamma delta", "alpha gamma"]
        tf_tiny = TextFeaturizer(method=method, n_topics=10, min_df=5, random_state=42)
        out_tiny = tf_tiny.fit_transform(tiny_docs)
        arr_tiny = out_tiny.toarray() if sparse.issparse(out_tiny) else np.asarray(out_tiny)
        assert arr_tiny.shape[0] == 3
        assert np.all(np.isfinite(arr_tiny))
        assert len(tf_tiny.get_feature_names_out()) == arr_tiny.shape[1]

        # 0-row transform preserves column width
        out_zero = tf_tiny.transform([])
        assert out_zero.shape == (0, arr_tiny.shape[1])


# ---------------------------------------------------------------------------
# C. utils.split_columns & utils.build_preprocessor integration tests
# ---------------------------------------------------------------------------


class TestUtilsTextIntegration:
    def test_split_columns_backwards_compat_and_detect_text(self):
        n = 18
        df = pd.DataFrame(
            {
                "num": np.arange(n, dtype=float),
                "txt": pd.Series(["a", "b", "c"] * 6, dtype="string"),
                "zip_code": [f"Z_{i:02d}" for i in range(n)],
                "comments": _make_sample_corpus(n),
            }
        )

        # Default 2-tuple contract unchanged
        num_cols, cat_cols = split_columns(df)
        assert num_cols == ["num"]
        assert cat_cols == ["txt", "zip_code", "comments"]

        # Opt-in 3-tuple removes free-text from cat_cols while keeping short/ID strings
        num_3, cat_3, text_3 = split_columns(df, detect_text=True)
        assert num_3 == ["num"]
        assert cat_3 == ["txt", "zip_code"]
        assert text_3 == ["comments"]

    def test_build_preprocessor_compat_and_text_branch(self):
        n = 20
        df = pd.DataFrame(
            {
                "num": [float(i) if i % 5 != 0 else np.nan for i in range(n)],
                "cat": ["x", "y", None, "z"] * 5,
                "comments": [
                    c if i % 6 != 0 else np.nan
                    for i, c in enumerate(_make_sample_corpus(n))
                ],
            }
        )

        # Backwards-compatible 2-arg call
        num_2, cat_2 = split_columns(df[["num", "cat"]])
        pre_compat = build_preprocessor(num_2, cat_2)
        out_compat = clone(pre_compat).fit_transform(df[["num", "cat"]])
        assert out_compat.shape[0] == n

        # With text_cols=None / []
        pre_empty_text = build_preprocessor(num_2, cat_2, text_cols=[])
        assert all(name != "text" for name, _, _ in pre_empty_text.transformers)
        out_empty = pre_empty_text.fit_transform(df[["num", "cat"]])
        assert "text" not in pre_empty_text.named_transformers_
        assert out_empty.shape[0] == n
        assert len(pre_empty_text.get_feature_names_out()) == out_empty.shape[1]

        # With active text_cols
        num_3, cat_3, text_3 = split_columns(df, detect_text=True)
        pre_text = build_preprocessor(
            num_3,
            cat_3,
            sparse_threshold=0.0,
            text_cols=text_3,
            text_method="combined",
            ngram_range=(1, 2),
            n_topics=3,
        )
        pre_cloned = clone(pre_text)
        out_text = pre_cloned.fit_transform(df)
        assert "text" in pre_cloned.named_transformers_
        assert isinstance(out_text, np.ndarray)
        assert np.all(np.isfinite(out_text))
        assert out_text.shape[0] == n
        assert len(pre_cloned.get_feature_names_out()) == out_text.shape[1]


# ---------------------------------------------------------------------------
# D. AutoPreprocessor & PreprocessingPlan integration tests
# ---------------------------------------------------------------------------


class TestAutoPreprocessorTextIntegration:
    def test_preprocessing_plan_backwards_compat(self):
        plan = PreprocessingPlan(
            actions=[],
            numeric_columns=["a"],
            categorical_columns=["b"],
            datetime_columns=[],
            drop_columns=[],
            target_encode_columns=[],
            skewed_columns=[],
            outlier_columns=[],
            missing_columns=[],
            correlated_groups={},
        )
        assert plan.text_columns == []
        assert plan.to_dict()["text_columns"] == []

    def test_auto_preprocessor_clone_and_mixed_dataframe(self):
        rng = np.random.RandomState(42)
        n = 48
        corpus = _make_sample_corpus(n)
        df = pd.DataFrame(
            {
                "num": rng.randn(n),
                "skew": np.exp(rng.randn(n)),
                "low_cat": rng.choice(["A", "B", "C"], size=n),
                "zip_code": [f"Z_{i % 18}" for i in range(n)],
                "review_text": [
                    corpus[i] if i % 8 != 0 else np.nan for i in range(n)
                ],
            }
        )
        y = (df["num"] > 0).astype(int).to_numpy()

        ap_template = AutoPreprocessor(
            handle_text=True,
            text_method="lda",
            text_ngram_range=(1, 3),
            text_n_topics=4,
            text_max_features=40,
            high_cardinality_threshold=10,
            verbose=0,
        )
        ap_cloned = clone(ap_template)
        assert ap_cloned.get_params() == ap_template.get_params()

        for method in ["tfidf", "ngram", "lda", "nmf", "combined"]:
            ap = AutoPreprocessor(
                handle_text=True,
                text_method=method,
                text_n_topics=3,
                text_max_features=25,
                high_cardinality_threshold=10,
                verbose=0,
            )
            out = ap.fit_transform(df, y)
            plan = ap.get_plan()
            assert "review_text" in plan.text_columns
            assert "review_text" not in plan.categorical_columns
            assert "review_text" not in plan.target_encode_columns
            assert "zip_code" in plan.target_encode_columns
            assert "text_columns" in plan.to_dict()
            assert isinstance(out, np.ndarray)
            assert np.issubdtype(out.dtype, np.number)
            assert np.all(np.isfinite(out))
            assert out.shape[0] == n
            assert len(ap.get_feature_names_out()) == out.shape[1]

    def test_text_only_dataframe_returns_dense_array(self):
        df = pd.DataFrame({"notes": _make_sample_corpus(18)})
        ap = AutoPreprocessor(handle_text=True, text_method="tfidf", verbose=0)
        out = ap.fit_transform(df)
        assert isinstance(out, np.ndarray)
        assert not sparse.issparse(out)
        assert out.shape[0] == 18
        assert out.shape[1] > 0
        assert np.all(np.isfinite(out))
        assert len(ap.get_feature_names_out()) == out.shape[1]

    def test_handle_text_false_and_high_missing_or_constant_text_drop(self):
        n = 24
        corpus = _make_sample_corpus(n)
        df = pd.DataFrame(
            {
                "num": np.arange(n, dtype=float),
                "review": corpus,
                "mostly_nan_text": [corpus[i] if i < 3 else np.nan for i in range(n)],
                "constant_text": ["This exact diagnostic sentence repeats identically every row."] * n,
            }
        )
        y = np.array([i % 2 for i in range(n)])

        ap_off = AutoPreprocessor(handle_text=False, high_cardinality_threshold=10, verbose=0)
        ap_off.fit(df[["num", "review"]], y)
        plan_off = ap_off.get_plan()
        assert plan_off.text_columns == []
        assert "review" in plan_off.categorical_columns

        ap_on = AutoPreprocessor(handle_text=True, missing_threshold=0.5, verbose=0)
        out_on = ap_on.fit_transform(df, y)
        plan_on = ap_on.get_plan()
        assert "mostly_nan_text" in plan_on.drop_columns
        assert "constant_text" in plan_on.drop_columns
        assert "mostly_nan_text" not in plan_on.text_columns
        assert "constant_text" not in plan_on.text_columns
        assert plan_on.text_columns == ["review"]
        assert isinstance(out_on, np.ndarray)
        assert np.all(np.isfinite(out_on))

    def test_explicit_text_cols_from_eda_and_create_eda_preprocessor(self):
        rng = np.random.RandomState(42)
        n = 30
        df = pd.DataFrame(
            {
                "num": rng.randn(n),
                "short_label": ["urgent billing fix", "normal status ping", "slow login page"] * 10,
                "notes": _make_sample_corpus(n),
                "target": [i % 2 for i in range(n)],
            }
        )
        X = df.drop(columns=["target"])
        y = df["target"]

        eda = ACDSOAutoEDA(verbose=0)
        eda.analyze(X)
        ap_eda = AutoPreprocessor.from_eda(
            eda,
            handle_text=True,
            text_cols=["short_label", "notes"],
            text_method="nmf",
            text_n_topics=3,
            verbose=0,
        )
        out_eda = ap_eda.fit_transform(X, y)
        assert ap_eda.get_plan().text_columns == ["short_label", "notes"]
        assert isinstance(out_eda, np.ndarray)
        assert np.all(np.isfinite(out_eda))

        ap_conv, report = create_eda_preprocessor(
            df,
            target_column="target",
            verbose=0,
            text_method="combined",
            text_n_topics=3,
            text_max_features=20,
        )
        assert report is not None
        out_conv = ap_conv.fit_transform(X, y)
        assert ap_conv.get_plan().text_columns == ["notes"]
        assert isinstance(out_conv, np.ndarray)
        assert np.all(np.isfinite(out_conv))


# ---------------------------------------------------------------------------
# E. End-to-End ACDSOClassifier / ACDSORegressor & Pipeline smoke tests
# ---------------------------------------------------------------------------


class TestEndToEndNLPSmoke:
    def test_acdso_classifier_and_regressor_with_free_text_and_pickle(self):
        rng = np.random.RandomState(42)
        n = 60
        pos_texts = [
            "Excellent fast delivery amazing product reliable analytics dashboard",
            "Outstanding support team resolved ticket immediately great experience",
        ]
        neg_texts = [
            "Broken item terrible customer service crashed repeatedly slow timeout",
            "Defective unit delayed shipping awful experience missing documentation",
        ]
        y = np.array([1 if i < n // 2 else 0 for i in range(n)])
        notes = [
            pos_texts[i % len(pos_texts)] if y[i] == 1 else neg_texts[i % len(neg_texts)]
            for i in range(n)
        ]
        X = pd.DataFrame(
            {
                "num_a": rng.randn(n) + (1.5 * y),
                "num_b": rng.uniform(0, 10, size=n),
                "region": rng.choice(["north", "south", "west"], size=n),
                "notes": notes,
            }
        )
        y_reg = 2.0 * X["num_a"].to_numpy() + 3.0 * y + 0.1 * rng.randn(n)

        clf = ACDSOClassifier(cv=3, max_candidates=2, verbose=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        probs = clf.predict_proba(X)
        assert preds.shape == (n,)
        assert probs.shape == (n, 2)
        clf_restored = pickle.loads(pickle.dumps(clf))
        assert np.array_equal(clf_restored.predict(X), preds)

        reg = ACDSORegressor(cv=3, max_candidates=2, verbose=0)
        reg.fit(X, y_reg)
        reg_preds = reg.predict(X)
        assert reg_preds.shape == (n,)
        assert np.all(np.isfinite(reg_preds))

    def test_explicit_build_preprocessor_pipeline_predicts_with_nans(self):
        n = 36
        rng = np.random.RandomState(42)
        y = np.array([i % 2 for i in range(n)])
        X = pd.DataFrame(
            {
                "num": [float(x) for x in (rng.randn(n) + y)],
                "cat": ["a", "b", "c"] * 12,
                "notes": [
                    "Positive reliable workflow fast processing" if label == 1
                    else "Negative broken timeout error crash"
                    for label in y
                ],
            }
        )

        num_cols, cat_cols, text_cols = split_columns(X, detect_text=True)
        pre = build_preprocessor(
            num_cols,
            cat_cols,
            sparse_threshold=0.0,
            text_cols=text_cols,
            text_method="nmf",
            n_topics=3,
        )
        pipe = clone(
            Pipeline(
                [
                    ("pre", pre),
                    ("clf", LogisticRegression(max_iter=200, random_state=42)),
                ]
            )
        )
        pipe.fit(X, y)

        X_test = pd.DataFrame(
            {
                "num": [np.nan, 1.5, -1.0],
                "cat": [None, "a", "unseen_cat"],
                "notes": [np.nan, "Positive reliable fast", "Negative broken crash"],
            }
        )
        preds = pipe.predict(X_test)
        assert preds.shape == (3,)
