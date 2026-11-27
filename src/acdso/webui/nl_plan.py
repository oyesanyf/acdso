"""Turn a free-text request into a run_acdso.py column-role plan.

Mirrors the OpenAI-provider pattern run_acdso.py already uses for target/
datetime auto-detection (see ``_llm_detect_target_col``), and falls back to
keyword matching against the actual column names when no LLM is configured
-- the same "LLM on top, rules underneath" shape as column_validator.py.

This never starts a run. It only proposes column roles for the webui to
show back to the user, who applies them to the existing form and presses
Run themselves -- training can take hours and --force deletes prior
models/reports, so nothing here should fire off a job unattended.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

_TASK_CHOICES = (
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
_TARGETLESS_TASKS = {"clustering", "anomaly", "survival", "recommender"}
_WORD_RE = re.compile(r"[a-z0-9]+")
_DATE_NAME_RE = re.compile(
    r"(?<![a-z])(?:date|time|timestamp|ts|datetime|period)(?![a-z])",
    re.IGNORECASE,
)
_EXCLUDE_CLAUSE_RE = re.compile(
    r"\b(?:excluding|exclude|drop|ignoring|ignore)\s+([a-z0-9_,\s]+?)(?=(?:\s+|,\s*)(?:from|using|with|group(?:ed)?\s+by|per\s+(?:group|patient|entity|device|subject|user|customer|hospital|session|site|store|item)[a-z0-9_]*|per\s+[a-z0-9_]+_id|by|for|and\s+(?:predict|cluster|train|group))\b|,\s*(?:predict|predicting|cluster|clustering|train|training|group|grouped|classify|regress|rank|recommend|detect|run|build|find|forecast|estimate)\b|[.;]|$)",
    re.IGNORECASE,
)


def _col_in_text(col_lower: str, text_lower: str) -> bool:
    """Match a column name at identifier/word boundaries in lowercased text."""
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(col_lower)}(?![a-z0-9_])", text_lower))


def _coerce_col_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [p.strip() for p in raw.split(",") if p.strip()]
    if isinstance(raw, list):
        return [str(v).strip() for v in raw if str(v).strip()]
    return []


def build_plan(prompt: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Return {predict, task, exclude, datetime_col, explanation, method, warnings}."""
    columns = [str(c) for c in frame.columns]
    plan = _llm_plan(prompt, frame, columns)
    if plan is None:
        plan = _rule_based_plan(prompt, frame, columns)
    plan["warnings"] = _target_warnings(frame, plan["predict"])
    return plan


def _target_warnings(frame: pd.DataFrame, predict: list[str]) -> list[str]:
    """Flag a predict column that looks like a categorical identifier/dimension
    rather than a genuine outcome to score.
    """
    import pandas as pd

    warnings: list[str] = []
    n_rows = len(frame)
    if n_rows == 0:
        return warnings
    for col in predict:
        if col not in frame.columns:
            continue
        s = frame[col]
        is_cat = (
            pd.api.types.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(s)
        )
        if not is_cat:
            continue
        nunique = int(s.nunique(dropna=True))
        ratio = nunique / n_rows
        if nunique > 50 or ratio > 0.3:
            warnings.append(
                f"'{col}' has {nunique} distinct values ({ratio:.0%} of rows) -- this looks more "
                "like a categorical identifier or dimension (e.g. an app/service name) than a "
                "genuine outcome to score. Double-check this is really what you want to predict."
            )
    return warnings


def _extract_json(text: str) -> str:
    """Pull the {...} block out of a response that may carry extra prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _llm_plan(prompt: str, frame: pd.DataFrame, columns: list[str]) -> dict[str, Any] | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from acdso.advisors import LLMConfig, OpenAIProvider

        provider = OpenAIProvider()
        col_info = "\n".join(
            f"  - {c}: {frame[c].dtype}, nunique={frame[c].nunique()}" for c in columns
        )
        system_prompt = (
            "You translate a plain-English ML request into a JSON column-role "
            "plan for an AutoML tool that trains one model per `predict` column, "
            "using every OTHER column in the dataset as a feature by default.\n\n"
            "Role rules:\n"
            "- `predict`: ONLY the outcome/label column(s) the model should learn "
            "to output -- e.g. a column already named like a label, class, "
            "category, score, or outcome. NEVER put a column here just because "
            "the request mentions it as a signal to use; a signal is an input, "
            "not the thing being predicted.\n"
            "- Everything not in `predict` or `exclude` is automatically used as "
            "a feature -- do not list feature/signal columns anywhere, just leave "
            "them alone. If the request names a signal (e.g. 'request volume', "
            "'user spread', 'DNS pattern'), the matching column must stay OUT of "
            "both `predict` and `exclude`.\n"
            "- `exclude`: ONLY pure identifiers, columns that duplicate the "
            "target, or columns genuinely irrelevant to the request. Keep this "
            "as short as possible -- it is not a dumping ground.\n"
            "- If no column looks like an existing label/outcome, pick the "
            "single column whose name is closest to one; do not guess by "
            "putting several unrelated columns in `predict`.\n"
            "- Default `exclude` to an empty list. Only add a column to it if "
            "you have a specific reason (raw identifier, exact duplicate of the "
            "target) -- a column simply not being the target is NOT a reason to "
            "exclude it.\n\n"
            "Worked example -- do not deviate from this pattern:\n"
            "Request: \"score domains by similarity to known providers and "
            "request volume\"\n"
            "Columns: domain, request_volume, label\n"
            "Correct: {\"predict\": [\"label\"], \"exclude\": [], "
            "\"task\": \"classification\", \"datetime_col\": [], "
            "\"explanation\": \"...\"}\n"
            "Wrong: putting request_volume or domain in exclude or predict -- "
            "they are features, so they are left out of both lists entirely.\n\n"
            "Only use column names that appear in the list given. Respond with "
            "JSON only, no prose."
        )
        user_prompt = (
            f"Request: {prompt}\n\n"
            f"Available columns:\n{col_info}\n\n"
            "Respond with exactly this JSON shape:\n"
            '{"predict": ["<column>", ...], '
            '"task": "auto|classification|regression|clustering|anomaly|multilabel|ranking|survival|recommender", '
            '"exclude": ["<column>", ...], "datetime_col": ["<column>"] or [], '
            '"group_col": ["<column>"] or [], "text_cols": ["<column>", ...], '
            '"time_col": ["<column>"] or [], "event_col": ["<column>"] or [], '
            '"query_col": ["<column>"] or [], "user_col": ["<column>"] or [], '
            '"item_col": ["<column>"] or [], "rating_col": ["<column>"] or [], '
            '"explanation": "<one sentence>"}'
        )
        config = LLMConfig(max_tokens=400, temperature=0.1, timeout_s=8.0, max_retries=1)
        response = provider.complete(system_prompt, user_prompt, config)
        data = json.loads(_extract_json(response))
        if not isinstance(data, dict):
            return None
    except Exception:
        return None

    task = data.get("task") if data.get("task") in _TASK_CHOICES else "auto"
    predict = [c for c in _coerce_col_list(data.get("predict")) if c in columns]
    if not predict and task not in _TARGETLESS_TASKS:
        return None
    if task in ("clustering", "anomaly", "recommender"):
        predict = []
    exclude = [c for c in _coerce_col_list(data.get("exclude")) if c in columns and c not in predict]
    taken = set(predict) | set(exclude)
    datetime_col = [c for c in _coerce_col_list(data.get("datetime_col")) if c in columns and c not in taken][:1]
    out: dict[str, Any] = {
        "predict": predict,
        "task": task,
        "exclude": exclude,
        "datetime_col": datetime_col,
        "explanation": str(data.get("explanation", "")).strip(),
        "method": "llm",
    }
    for role in ("group_col", "time_col", "event_col", "query_col", "user_col", "item_col", "rating_col"):
        vals = [c for c in _coerce_col_list(data.get(role)) if c in columns and c not in taken][:1]
        if vals:
            out[role] = vals
    text_vals = [c for c in _coerce_col_list(data.get("text_cols")) if c in columns and c not in taken]
    if text_vals:
        out["text_cols"] = text_vals
    return out


def _match_col_by_keywords(columns: list[str], patterns: tuple[str, ...], exclude_set: set[str]) -> list[str]:
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in columns:
            if c not in exclude_set and rx.search(c):
                return [c]
    return []


def _rule_based_plan(prompt: str, frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """No LLM configured -- infer task, column roles, and exclusions from prompt & schema."""
    import pandas as pd

    from .column_validator import _is_datetime_like, predictable_columns, suggest_target_columns

    prompt_lower = prompt.lower()
    col_by_lower = {c.lower(): c for c in columns}

    # 1. Extract explicit exclusions ("excluding col_a, col_b")
    exclude: list[str] = []
    for m in _EXCLUDE_CLAUSE_RE.finditer(prompt):
        clause = m.group(1)
        per_split = re.split(r"(?:\s+|,\s*)per\s+", clause, maxsplit=1, flags=re.IGNORECASE)
        if len(per_split) == 2:
            after_first = re.split(r"[,\s]+", per_split[1].strip())[0].lower()
            if after_first in col_by_lower:
                clause = per_split[0]
        tokens = [t.strip() for t in re.split(r"[,\s]+", clause) if t.strip()]
        for tok in tokens:
            if tok.lower() in ("and", "or", "the", "a", "an", "column", "columns", "col"):
                continue
            matched = col_by_lower.get(tok.lower())
            if matched and matched not in exclude:
                exclude.append(matched)
    exclude_set = set(exclude)

    intent_prompt_lower = _EXCLUDE_CLAUSE_RE.sub(" ", prompt_lower)
    intent_words = set(_WORD_RE.findall(intent_prompt_lower))
    candidates = [c for c in predictable_columns(frame) if c not in exclude_set]

    # 2. Detect R1-R6 task intent
    task = "auto"
    cluster_algo = None
    dim_reduce = None
    anomaly_algo = None
    nlp_method = None
    cv_strategy = None
    scale_mode = None

    has_explicit_cluster_cue = any(
        k in intent_prompt_lower
        for k in (
            "cluster",
            "clustering",
            "kmeans",
            "k-means",
            "hdbscan",
            "gmm",
            "unsupervised",
            "without a target",
            "without target",
            "no target",
        )
    )
    has_segment_cue = any(w in intent_words for w in ("segment", "segments", "segmentation"))
    has_supervised_cue = any(
        w in intent_words
        for w in (
            "predict",
            "predicting",
            "classify",
            "classification",
            "classifier",
            "regress",
            "regression",
            "regressor",
            "forecast",
            "estimate",
        )
    )
    has_explicit_target = any(
        (
            c.lower() in intent_words
            or _col_in_text(c.lower(), intent_prompt_lower)
            or bool(intent_words & set(_WORD_RE.findall(c.lower())))
        )
        for c in candidates
    )

    if has_explicit_cluster_cue or (has_segment_cue and not (has_supervised_cue and has_explicit_target)):
        task = "clustering"
        if "kmeans" in intent_prompt_lower or "k-means" in intent_prompt_lower:
            cluster_algo = "kmeans"
        elif "hdbscan" in intent_prompt_lower:
            cluster_algo = "hdbscan"
        elif "gmm" in intent_prompt_lower or "gaussian mixture" in intent_prompt_lower:
            cluster_algo = "gmm"
        if "umap" in intent_prompt_lower:
            dim_reduce = "umap"
        elif "pca" in intent_prompt_lower:
            dim_reduce = "pca"
    elif any(k in intent_prompt_lower for k in ("anomaly", "anomalies", "outlier", "outliers", "isolation forest", "ocsvm", "one-class svm")) or (
        "lof" in intent_words
    ):
        task = "anomaly"
        if "isolation" in intent_prompt_lower:
            anomaly_algo = "isolation_forest"
        elif "lof" in intent_words or "local outlier" in intent_prompt_lower:
            anomaly_algo = "lof"
        elif "ocsvm" in intent_words or "one-class" in intent_prompt_lower:
            anomaly_algo = "ocsvm"
    elif any(k in intent_prompt_lower for k in ("survival", "censored", "time-to-event", "time to event", "cox", "hazard")):
        task = "survival"
    elif any(k in intent_prompt_lower for k in ("recommend", "recommender", "recommendation", "collaborative filtering")):
        task = "recommender"
    elif any(k in intent_prompt_lower for k in ("learning to rank", "learning-to-rank", "ndcg")) or any(
        w in intent_words for w in ("rank", "ranking")
    ):
        task = "ranking"
    elif any(k in intent_prompt_lower for k in ("multilabel", "multi-label", "multi label", "multiple labels")):
        task = "multilabel"
    elif "classification" in intent_words or "classify" in intent_words or "classifier" in intent_words:
        task = "classification"
    elif "regression" in intent_words or "regress" in intent_words or "regressor" in intent_words:
        task = "regression"

    # Cross-cutting CV / NLP / Scale hints
    if "stratified_group_kfold" in intent_prompt_lower or ("stratified" in intent_words and "group" in intent_words):
        cv_strategy = "stratified_group_kfold"
    elif "group_kfold" in intent_prompt_lower or "grouped" in intent_words or "group by" in intent_prompt_lower:
        cv_strategy = "group_kfold"
    elif "temporal" in intent_words:
        cv_strategy = "temporal"
    elif "kfold" in intent_words:
        cv_strategy = "kfold"

    if "tfidf" in intent_words or "tf-idf" in intent_prompt_lower:
        nlp_method = "tfidf"
    elif "ngram" in intent_words or "n-gram" in intent_prompt_lower:
        nlp_method = "ngram"
    elif "lda" in intent_words:
        nlp_method = "lda"
    elif "nmf" in intent_words:
        nlp_method = "nmf"

    if any(k in intent_prompt_lower for k in ("out-of-core", "out of core", "chunked", "streaming")):
        scale_mode = "chunked"
    elif "subsample" in intent_words or "subsampling" in intent_words:
        scale_mode = "subsample"

    # 3. Resolve specialized & cross-cutting role columns BEFORE scoring predict
    _GENERIC_COL_TOKENS = {"id", "col", "column"}

    def word_overlap(col: str) -> int:
        c_low = col.lower()
        if c_low in intent_words or _col_in_text(c_low, intent_prompt_lower):
            return 10 + len(_WORD_RE.findall(c_low))
        col_tokens = set(_WORD_RE.findall(c_low)) - _GENERIC_COL_TOKENS
        return len(intent_words & col_tokens)

    def _pick_role_col(
        strong_clause_rx: re.Pattern[str] | None,
        weak_by_rx: re.Pattern[str] | None,
        keyword_patterns: tuple[str, ...],
        taken_cols: set[str],
        allow_weak_any: bool = False,
    ) -> list[str]:
        if strong_clause_rx is not None:
            for m in strong_clause_rx.finditer(intent_prompt_lower):
                matched = col_by_lower.get(m.group(1).strip().lower())
                if matched and matched not in taken_cols:
                    return [matched]

        kw_matches: list[tuple[int, int, int, str]] = []
        any_kw_col = False
        for idx, c in enumerate(columns):
            if c in taken_cols:
                continue
            for pat_idx, pat in enumerate(keyword_patterns):
                if re.search(pat, c, re.IGNORECASE):
                    any_kw_col = True
                    ov = word_overlap(c)
                    if ov > 0:
                        kw_matches.append((ov, -pat_idx, -idx, c))
                    break
        if kw_matches:
            kw_matches.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            return [kw_matches[0][3]]

        if weak_by_rx is not None and (allow_weak_any or not any_kw_col):
            for m in weak_by_rx.finditer(intent_prompt_lower):
                matched = col_by_lower.get(m.group(1).strip().lower())
                if matched and matched not in taken_cols:
                    return [matched]

        return _match_col_by_keywords(columns, keyword_patterns, taken_cols)

    predict: list[str] = []
    extra_roles: dict[str, Any] = {}
    taken: set[str] = set(exclude_set)

    if task == "survival":
        time_col = _pick_role_col(
            None,
            None,
            (r"^(?:time|duration|tenure|futime|survival_time)$", r"time|duration|tenure|days|months|years"),
            taken,
        )
        taken.update(time_col)
        event_col = _pick_role_col(
            None,
            None,
            (r"^(?:event|status|observed|censored|death|churn)$", r"event|status|observed|death|churn|fail"),
            taken,
        )
        taken.update(event_col)
        if time_col:
            extra_roles["time_col"] = time_col
        if event_col:
            extra_roles["event_col"] = event_col
    elif task == "recommender":
        user_col = _pick_role_col(None, None, (r"^(?:user|user_id|customer_id|uid)$", r"user|customer|account"), taken)
        taken.update(user_col)
        item_col = _pick_role_col(None, None, (r"^(?:item|item_id|product_id|movie_id|iid)$", r"item|product|movie|sku"), taken)
        taken.update(item_col)
        rating_col = _pick_role_col(None, None, (r"^(?:rating|score|stars)$", r"rating|score|stars|interaction"), taken)
        taken.update(rating_col)
        if user_col:
            extra_roles["user_col"] = user_col
        if item_col:
            extra_roles["item_col"] = item_col
        if rating_col:
            extra_roles["rating_col"] = rating_col
    elif task == "ranking":
        q_col = _pick_role_col(
            re.compile(r"\bquery(?:_col)?(?:\s*[:=]\s*|\s+)([a-z0-9_]+)\b", re.IGNORECASE),
            re.compile(r"\b(?:grouped?\s+)?by\s+([a-z0-9_]+)\b", re.IGNORECASE),
            (r"^(?:query|query_id|qid)$", r"query|qid|search"),
            taken,
        )
        if q_col:
            extra_roles["query_col"] = q_col
            taken.update(q_col)

    # Group column hint (resolved BEFORE predict so group IDs cannot hijack predict)
    _group_kw_pat = (r"group|patient|entity|user|customer|device|subject|hospital",)
    _has_per_domain_col = any(
        (col in col_by_lower and col_by_lower[col] not in exclude_set)
        for col in (
            m.group(1).strip().lower()
            for m in re.finditer(
                r"\bper\s+((?:session|site|store|item)[a-z0-9_]*)\b",
                intent_prompt_lower,
            )
        )
    )
    has_group_cue = (
        cv_strategy in ("group_kfold", "stratified_group_kfold")
        or any(w in intent_words for w in ("group", "groups", "grouped", "grouping", "entity", "entities"))
        or bool(
            re.search(
                r"\b(?:group(?:ed)?\s+by|per\s+(?:group|patient|entity|device|subject|user|customer|hospital)[a-z0-9_]*|by\s+(?:group|patient|entity|device|subject|user|customer|hospital)[a-z0-9_]*)\b",
                intent_prompt_lower,
            )
        )
        or _has_per_domain_col
    )
    if has_group_cue:
        explicit_grp_clause_rx = re.compile(
            r"\b(?:(?:group(?:ed)?\s+by|per)\s+|group_col(?:\s*[:=]\s*|\s+))([a-z0-9_]+)\b",
            re.IGNORECASE,
        )
        has_unclaimed_explicit_group = any(
            (c in col_by_lower and col_by_lower[c] not in taken)
            for c in (m.group(1).strip().lower() for m in explicit_grp_clause_rx.finditer(intent_prompt_lower))
        )
        skip_dup_ranking_group = (
            task == "ranking"
            and bool(extra_roles.get("query_col"))
            and cv_strategy not in ("stratified_group_kfold",)
            and not has_unclaimed_explicit_group
            and not any(
                word_overlap(c) > 0
                for c in columns
                if c not in taken and re.search(_group_kw_pat[0], c, re.IGNORECASE)
            )
        )
        if not skip_dup_ranking_group:
            grp = _pick_role_col(
                explicit_grp_clause_rx,
                re.compile(r"\bby\s+([a-z0-9_]+)\b", re.IGNORECASE),
                _group_kw_pat,
                taken,
                allow_weak_any=(cv_strategy in ("group_kfold", "stratified_group_kfold")),
            )
            if grp:
                extra_roles["group_col"] = grp
                taken.update(grp)

    # Text columns hint (resolved BEFORE predict so text columns cannot hijack predict)
    if nlp_method is not None or any(w in intent_words for w in ("text", "nlp", "review", "comment", "description", "topic")):
        try:
            from acdso.nlp import detect_text_columns

            tcols = [c for c in detect_text_columns(frame, exclude_cols=list(taken)) if c not in taken]
            if tcols:
                extra_roles["text_cols"] = tcols
                taken.update(tcols)
        except Exception:
            pass

    dt_candidates = [
        c
        for c in columns
        if c not in taken
        and not pd.api.types.is_numeric_dtype(frame[c])
        and _is_datetime_like(frame[c])
    ]
    # Prefer columns whose name also signals date/time among genuinely datetime-like columns.
    dt_candidates.sort(key=lambda c: (0 if _DATE_NAME_RE.search(c) else 1))
    datetime_col = dt_candidates[:1] if task not in ("clustering", "anomaly", "recommender") else []
    taken.update(datetime_col)

    # 4. Resolve predict target(s) excluding all claimed role columns
    if task in ("clustering", "anomaly"):
        predict = []
        explanation = f"Configured unsupervised {task} task without a target column."
    elif task == "survival":
        predict = []
        explanation = "Configured survival analysis task."
    elif task == "recommender":
        predict = []
        explanation = "Configured collaborative-filtering recommender task."
    else:
        predict_candidates = [c for c in candidates if c not in taken]
        scored = sorted(((c, word_overlap(c)) for c in predict_candidates), key=lambda item: item[1], reverse=True)
        mentioned = [c for c, score in scored if score > 0]

        if mentioned:
            predict = mentioned[:3] if task == "multilabel" else mentioned[:1]
            explanation = "Matched column name(s) mentioned in the request: " + ", ".join(predict)
        elif predict_candidates:
            ranked = suggest_target_columns(frame, predict_candidates, top_n=1)
            suggestions = ranked.get("suggestions") or []
            predict = [suggestions[0]["column"]] if suggestions else predict_candidates[:1]
            explanation = (
                "No column names recognized in the request -- falling back to the "
                "top-ranked predictable column."
            )
        else:
            explanation = "No predictable target columns found."

    plan: dict[str, Any] = {
        "predict": predict,
        "task": task,
        "exclude": exclude,
        "datetime_col": datetime_col,
        "explanation": explanation,
        "method": "rule-based",
    }
    plan.update(extra_roles)
    if cluster_algo:
        plan["cluster_algo"] = cluster_algo
    if dim_reduce:
        plan["dim_reduce"] = dim_reduce
    if anomaly_algo:
        plan["anomaly_algo"] = anomaly_algo
    if nlp_method:
        plan["nlp_method"] = nlp_method
    if cv_strategy:
        plan["cv_strategy"] = cv_strategy
    if scale_mode:
        plan["scale_mode"] = scale_mode
    return plan
