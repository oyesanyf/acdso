"""LLM-assisted sanity check for the GUI's column-role selections.

Picking the wrong column for a role (an ID column as the forecast target,
a numeric column as the datetime column) doesn't surface as an error until
deep into a multi-minute training run. This gives an early, advisory check
before the run starts: does each selected column actually look like it fits
the role it's been given?

Mirrors LLMColumnAnalyzer's pattern in acdso.advisors: try the LLM, fall
back to plain rules if no provider is configured or the call fails. The
rule-based pass alone already catches the unambiguous case -- a column
already claimed by one role can't also serve another -- and always runs;
the LLM pass adds softer judgment on top when a provider is available.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

ROLE_LABELS = {
    "predict": "prediction target",
    "target": "training target",
    "exclude": "excluded feature",
    "datetime_col": "datetime/period column",
    "treatment": "treatment column",
}

_DATE_TOKEN_RE = re.compile(
    r"^(?:"
    r"\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?(?:[T\s]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s,-]+\d{1,4}(?:[\s,]+\d{2,4})?"
    r"|\d{1,2}[\s-]+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*(?:[\s,-]+\d{2,4})?"
    r")$",
    re.IGNORECASE,
)

_OLLAMA_PROBE_CACHE: dict[str, Any] = {"checked_at": 0.0, "host": "", "ok": False}


@dataclass
class ColumnIssue:
    column: str
    role: str
    severity: str  # "block" | "warn"
    message: str


@dataclass
class ValidationResult:
    ok: bool
    issues: list = field(default_factory=list)
    source: str = "rules"  # "rules" | "llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source": self.source,
            "issues": [
                {"column": i.column, "role": i.role, "severity": i.severity, "message": i.message}
                for i in self.issues
            ],
        }


def _profile_columns(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    profile = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        profile[col] = {
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique()),
            "n_rows": int(len(s)),
            "null_pct": round(float(s.isnull().mean() * 100), 1),
            "sample": [str(v) for v in s.dropna().head(3).tolist()],
        }
    return profile


def _is_datetime_like(s: pd.Series) -> bool:
    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        # pd.to_datetime happily parses large floats/ints as Unix
        # timestamps, which misclassifies ordinary numeric measurement
        # columns (byte counts, prices) as date-like.
        return False
    sample = s.dropna().head(20)
    if sample.empty:
        return False
    strs = sample.astype(str).str.strip()
    if float(strs.str.match(_DATE_TOKEN_RE, na=False).mean()) < 0.5:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)
            try:
                parsed = pd.to_datetime(strs, format="mixed", errors="coerce")
            except TypeError:
                parsed = pd.to_datetime(strs, errors="coerce")
        if not pd.api.types.is_datetime64_any_dtype(parsed):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    parsed = pd.to_datetime(strs, format="mixed", utc=True, errors="coerce")
                except TypeError:
                    parsed = pd.to_datetime(strs, utc=True, errors="coerce")
        return len(parsed) > 0 and float(parsed.notna().mean()) > 0.8
    except Exception:
        return False


ID_LIKE_WORDS = ("_id", "id_", "uuid", "guid", "index", "rowid")


def _is_cat_or_text_dtype(s: pd.Series) -> bool:
    import pandas as pd

    return bool(
        pd.api.types.is_object_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(s)
    )


def analyze_predictable_columns(df: pd.DataFrame) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Return ``(predictable_columns, excluded_reasons)``.

    Every skipped column receives a structured ``{"reason": code, "detail": msg}``
    explanation so the UI and API can explain why columns are filtered out.
    """
    import pandas as pd

    n = len(df)
    keep: list[str] = []
    excluded: dict[str, dict[str, str]] = {}
    for col in df.columns:
        col_name = str(col)
        s = df[col]
        if _is_datetime_like(s):
            excluded[col_name] = {
                "reason": "datetime",
                "detail": "Looks like a date/time or period column",
            }
            continue
        nunique = int(s.nunique(dropna=True))
        if n == 0 or nunique == 0:
            excluded[col_name] = {
                "reason": "all_null",
                "detail": "All values are null",
            }
            continue
        if nunique <= 1:
            excluded[col_name] = {
                "reason": "constant",
                "detail": "Constant value (only 1 distinct non-null value)",
            }
            continue
        fully_unique = nunique >= n
        col_lower = col_name.lower()
        is_id_named = any(w in col_lower for w in ID_LIKE_WORDS)
        if is_id_named and nunique > 0.9 * n:
            excluded[col_name] = {
                "reason": "id_like",
                "detail": "Identifier-like column name with >90% unique values",
            }
            continue
        if fully_unique and not pd.api.types.is_float_dtype(s):
            excluded[col_name] = {
                "reason": "unique_non_float",
                "detail": "Every row is unique (non-float identifier/key)",
            }
            continue
        if _is_cat_or_text_dtype(s) and n > 0 and (nunique > 50 or nunique / n > 0.3):
            excluded[col_name] = {
                "reason": "high_cardinality_categorical",
                "detail": (
                    f"High-cardinality categorical ({nunique} distinct values); "
                    "blocked as a classification target"
                ),
            }
            continue
        keep.append(col_name)
    return keep, excluded


def predictable_columns(df: pd.DataFrame) -> list[str]:
    """Columns reasonable to offer as a Predict/Target choice at all."""
    keep, _ = analyze_predictable_columns(df)
    return keep


def _rule_based_validate(df: pd.DataFrame, roles: dict[str, list]) -> ValidationResult:
    import pandas as pd

    issues: list[ColumnIssue] = []

    # A column can't serve two roles at once -- unambiguous, always blocks.
    seen: dict[str, str] = {}
    for role, cols in roles.items():
        for col in cols:
            if col in seen and seen[col] != role:
                issues.append(ColumnIssue(
                    column=col, role=role, severity="block",
                    message=f"'{col}' is already selected as {ROLE_LABELS.get(seen[col], seen[col])}; "
                            f"it can't also be the {ROLE_LABELS.get(role, role)}.",
                ))
            else:
                seen[col] = role

    # Predict vs train-only target mismatch: run_acdso.py trains on --target and
    # prints a mismatch warning instead of predicting --predict when they differ.
    predict_cols = list(roles.get("predict", []))
    target_cols = list(roles.get("target", []))
    if predict_cols and target_cols and set(predict_cols) != set(target_cols):
        issues.append(ColumnIssue(
            column=", ".join(predict_cols),
            role="predict",
            severity="block",
            message=(
                f"Predict target ({', '.join(predict_cols)}) differs from train-only target "
                f"({', '.join(target_cols)}). Leave 'Train-only target(s)' empty when using "
                "'Predict target(s)', or select the same column only in 'Predict target(s)'."
            ),
        ))

    # Unknown columns in any active role.
    df_cols = {str(c) for c in df.columns}
    if df_cols:
        for role in ("predict", "target", "datetime_col", "exclude", "treatment"):
            for col in roles.get(role, []):
                if col not in df_cols:
                    issues.append(ColumnIssue(
                        column=col,
                        role=role,
                        severity="block",
                        message=f"Column '{col}' was not found in the uploaded dataset.",
                    ))

        # Check whether exclude + target + datetime_col leave zero feature columns.
        if predict_cols or target_cols:
            non_feature = (
                set(predict_cols)
                | set(target_cols)
                | set(roles.get("exclude", []))
                | set(roles.get("datetime_col", []))
            )
            remaining = [c for c in df_cols if c not in non_feature]
            if not remaining:
                issues.append(ColumnIssue(
                    column=", ".join(roles.get("exclude", []) or predict_cols or target_cols),
                    role="exclude",
                    severity="block",
                    message=(
                        "No feature columns remain after removing the selected target, "
                        "datetime, and excluded columns."
                    ),
                ))

    for col in roles.get("datetime_col", []):
        if col in df.columns and not _is_datetime_like(df[col]):
            issues.append(ColumnIssue(
                column=col, role="datetime_col", severity="warn",
                message=f"'{col}' doesn't look like a date/time column "
                        f"(dtype {df[col].dtype}) -- double check this is the period column.",
            ))

    for role in ("predict", "target"):
        for col in roles.get(role, []):
            if col not in df.columns:
                continue
            s = df[col]
            col_lower = col.lower()
            n = len(s)
            nunique = int(s.nunique(dropna=True))
            if any(w in col_lower for w in ID_LIKE_WORDS) and n > 0 and nunique > 0.9 * n:
                issues.append(ColumnIssue(
                    column=col, role=role, severity="warn",
                    message=f"'{col}' looks like an identifier column (mostly unique values) "
                            f"-- unusual choice for a {ROLE_LABELS[role]}.",
                ))
            elif (
                _is_cat_or_text_dtype(s)
                and n > 0
                and (nunique > 50 or nunique / n > 0.3)
            ):
                issues.append(ColumnIssue(
                    column=col, role=role, severity="block",
                    message=f"'{col}' has {nunique} distinct values ({nunique / n:.0%} of rows) -- "
                            f"this looks like a categorical identifier or dimension (e.g. an "
                            f"app/service name), not a genuine {ROLE_LABELS[role]}.",
                ))
            elif _is_datetime_like(s) and not pd.api.types.is_numeric_dtype(s):
                issues.append(ColumnIssue(
                    column=col, role=role, severity="warn",
                    message=f"'{col}' looks like a date/time column, not a value to predict.",
                ))

    ok = not any(i.severity == "block" for i in issues)
    return ValidationResult(ok=ok, issues=issues, source="rules")


VALIDATION_PROMPT = """You are sanity-checking column-role selections for a data pipeline, \
before it runs. You are NOT picking columns yourself -- only judging the roles a user already chose.

COLUMN PROFILES:
{profiles}

SELECTED ROLES:
{roles}

Role meanings:
- predict / target: the column being predicted or trained on. Should be a real value \
of interest (a measurement, an outcome) -- not an ID, not a date/time column, not a column \
whose name suggests it duplicates another selected column.
- exclude: a feature dropped from training. Any column is a reasonable choice here.
- datetime_col: the period/date column for time-series aggregation. Should genuinely be a \
date, timestamp, or period label -- not a numeric measurement.

Flag ONLY clear mismatches -- a column used for a role it plainly doesn't fit. Do not flag \
reasonable choices you're merely unsure about, and do not suggest alternative columns.

Respond in JSON:
{{
    "issues": [
        {{"column": "...", "role": "predict|target|exclude|datetime_col", \
"severity": "block|warn", "message": "one sentence, plain language"}}
    ]
}}
Use "block" only when the column is unmistakably the wrong type for the role (e.g. a raw ID \
as a forecast target). Use "warn" for anything softer. Empty "issues" list if everything looks fine.
"""


def _llm_validate(df: pd.DataFrame, roles: dict[str, list], provider) -> ValidationResult:
    from acdso.advisors import LLMConfig

    all_cols = sorted({c for cols in roles.values() for c in cols})
    profile = _profile_columns(df, all_cols)
    profile_lines = [
        f"- {col}: dtype={p['dtype']}, unique={p['n_unique']}/{p['n_rows']}, "
        f"null%={p['null_pct']}, sample={p['sample']}"
        for col, p in profile.items()
    ]
    role_lines = [f"- {role}: {cols}" for role, cols in roles.items() if cols]

    prompt = VALIDATION_PROMPT.format(
        profiles="\n".join(profile_lines) or "(none selected)",
        roles="\n".join(role_lines) or "(none selected)",
    )

    config = LLMConfig(max_tokens=500, temperature=0.0, timeout_s=6.0, max_retries=1)
    response = provider.complete(
        system_prompt="You are a careful data pipeline reviewer. Be conservative -- "
                      "only flag genuine mismatches.",
        user_prompt=prompt,
        config=config,
    )

    text = response if isinstance(response, str) else str(response)
    start, end = text.find("{"), text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON in LLM response")
    data = json.loads(text[start:end])

    issues = [
        ColumnIssue(
            column=i.get("column", ""),
            role=i.get("role", ""),
            severity="warn",
            message=i.get("message", ""),
        )
        for i in data.get("issues", [])
    ]
    return ValidationResult(ok=True, issues=issues, source="llm")


def _probe_ollama_fast(base_url: str, timeout_s: float = 0.4) -> bool:
    """Quick 400ms HTTP probe with 30s TTL cache so offline Ollama never stalls."""
    now = time.monotonic()
    if (
        _OLLAMA_PROBE_CACHE["host"] == base_url
        and (now - _OLLAMA_PROBE_CACHE["checked_at"]) < 30.0
    ):
        return bool(_OLLAMA_PROBE_CACHE["ok"])
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    ok = False
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - local Ollama HTTP check
            ok = 200 <= resp.status < 300
    except Exception:
        ok = False
    _OLLAMA_PROBE_CACHE["host"] = base_url
    _OLLAMA_PROBE_CACHE["checked_at"] = now
    _OLLAMA_PROBE_CACHE["ok"] = ok
    return ok


def _get_provider():
    """Reuse ACDSO's own provider selection: same env vars, same fallback order."""
    if os.environ.get("OPENAI_API_KEY"):
        from acdso.advisors import OpenAIProvider

        p = OpenAIProvider(model="gpt-4o-mini")
        if p.is_available():
            return p
    if os.environ.get("ANTHROPIC_API_KEY"):
        from acdso.advisors import AnthropicProvider

        p = AnthropicProvider(model="claude-3-haiku-20240307")
        if p.is_available():
            return p

    base_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    if not _probe_ollama_fast(base_url, timeout_s=0.4):
        return None
    try:
        from acdso.advisors import OllamaProvider

        p = OllamaProvider(model="llama3.2", base_url=base_url)
        if p.is_available():
            return p
    except Exception:
        pass
    return None


def validate_columns(df: pd.DataFrame, roles: dict[str, list]) -> ValidationResult:
    """Validate column-role selections; combines rule-based + LLM checks."""
    result = _rule_based_validate(df, roles)
    if not result.ok:
        return result

    provider = _get_provider()
    if provider is None:
        return result

    try:
        llm_result = _llm_validate(df, roles, provider)
    except Exception:
        return result

    combined_issues = result.issues + llm_result.issues
    ok = not any(i.severity == "block" for i in combined_issues)
    return ValidationResult(ok=ok, issues=combined_issues, source="llm")


# ---------------------------------------------------------------------------
# Target suggestion -- which predictable column looks like the outcome
# ---------------------------------------------------------------------------

_OUTCOME_WORDS = (
    "total", "amount", "price", "revenue", "cost", "sales", "value", "score",
    "transfer", "count", "rate", "target", "label", "outcome", "result",
    "spend", "volume", "churn", "conversion", "profit", "quantity", "qty",
    "duration", "response", "tip",
)


def _rule_based_scores(df: pd.DataFrame, candidates: list[str]) -> dict[str, float]:
    """Score candidates by dtype + name -- one independent 'vote' on target-ness."""
    import pandas as pd

    n = max(len(df), 1)
    scores = {}
    for col in candidates:
        s = df[col]
        score = 0.0
        if pd.api.types.is_float_dtype(s):
            score += 2
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            score += 1
        if any(w in col.lower() for w in _OUTCOME_WORDS):
            score += 2
        score += min(s.nunique() / n, 1.0)
        scores[col] = score
    return scores


def _learnability_scores(
    df: pd.DataFrame,
    candidates: list[str],
    max_rows: int = 1000,
    max_candidates: int = 25,
    rule_scores: dict[str, float] | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Fast pure-NumPy 3-fold Ridge + ANOVA eta^2 + balanced-accuracy scorer.

    Avoids O(K) sklearn RandomForest fits and repeated pd.get_dummies copies,
    completing in < 20 ms on typical datasets and < 100 ms on 100-col tables.
    """
    import numpy as np
    import pandas as pd

    scores: dict[str, dict[str, Any] | None] = {c: None for c in candidates}
    if not candidates or len(df) < 20:
        return scores

    work = df.sample(max_rows, random_state=42) if len(df) > max_rows else df
    work = work.reset_index(drop=True)
    n = len(work)

    if len(candidates) > max_candidates:
        r_scores = rule_scores if rule_scores is not None else _rule_based_scores(work, candidates)
        probe_candidates = sorted(
            candidates,
            key=lambda c: r_scores.get(c, 0.0),
            reverse=True,
        )[:max_candidates]
    else:
        probe_candidates = list(candidates)

    if len(probe_candidates) < 2:
        return scores

    blocks: list[np.ndarray] = []
    col_slices: dict[str, tuple[int, int]] = {}
    binned_codes: dict[str, np.ndarray] = {}
    cursor = 0

    for col in probe_candidates:
        s = work[col]
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if is_num:
            vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(vals)
            if int(valid.sum()) >= 20:
                med = float(np.median(vals[valid]))
                filled = np.where(valid, vals, med)
                std = float(np.std(filled))
                if std > 1e-12:
                    z = ((filled - float(np.mean(filled))) / std).reshape(-1, 1)
                    blocks.append(z)
                    col_slices[col] = (cursor, cursor + 1)
                    cursor += 1
                else:
                    col_slices[col] = (cursor, cursor)
                uniq = np.unique(filled)
                if len(uniq) <= 8:
                    _, codes = np.unique(filled, return_inverse=True)
                else:
                    edges = np.unique(np.quantile(filled, np.linspace(0.0, 1.0, 9)))
                    codes = np.digitize(filled, edges[1:-1]) if len(edges) > 2 else np.zeros(n, dtype=int)
                binned_codes[col] = codes.astype(int, copy=False)
            else:
                col_slices[col] = (cursor, cursor)
                binned_codes[col] = np.zeros(n, dtype=int)
        else:
            str_s = s.astype(object).where(s.notna(), "__NA__").astype(str)
            top_levels = str_s.value_counts().index[:12].tolist()
            if top_levels:
                arr = str_s.to_numpy()
                blk = np.column_stack([(arr == lvl).astype(float) for lvl in top_levels])
                blocks.append(blk)
                col_slices[col] = (cursor, cursor + blk.shape[1])
                cursor += blk.shape[1]
            else:
                col_slices[col] = (cursor, cursor)
            codes, _ = pd.factorize(str_s, sort=False)
            binned_codes[col] = codes.astype(int, copy=False)

    if not blocks or cursor == 0:
        return scores

    F = np.hstack(blocks)
    rng = np.random.default_rng(42)
    deadline = time.perf_counter() + 0.35

    for idx, target_col in enumerate(probe_candidates):
        if idx >= 3 and time.perf_counter() > deadline:
            break
        s0, s1 = col_slices[target_col]
        if s0 == 0 and s1 == F.shape[1]:
            continue
        if s0 == s1:
            X_all = F
        elif s0 == 0:
            X_all = F[:, s1:]
        elif s1 == F.shape[1]:
            X_all = F[:, :s0]
        else:
            X_all = np.hstack((F[:, :s0], F[:, s1:]))
        if X_all.shape[1] == 0:
            continue

        s_target = work[target_col]
        is_num_target = pd.api.types.is_numeric_dtype(s_target) and not pd.api.types.is_bool_dtype(s_target)
        if is_num_target:
            raw_y = pd.to_numeric(s_target, errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(raw_y)
        else:
            mask = s_target.notna().to_numpy(dtype=bool)

        m = int(mask.sum())
        if m < 20:
            continue

        X = X_all[mask]
        folds = rng.permutation(m) % 3
        is_regression = bool(is_num_target and int(s_target.nunique(dropna=True)) > 20)

        try:
            if is_regression:
                y = raw_y[mask]
                if float(np.var(y)) <= 1e-12:
                    scores[target_col] = {"metric": "R2", "score": 0.0}
                    continue
                fold_r2s: list[float] = []
                for f in range(3):
                    te = folds == f
                    tr = ~te
                    if int(tr.sum()) < 5 or int(te.sum()) < 2:
                        continue
                    Xtr, Xte = X[tr], X[te]
                    ytr, yte = y[tr], y[te]
                    x_mean = Xtr.mean(axis=0)
                    y_mean = float(ytr.mean())
                    Xtr_c = Xtr - x_mean
                    ytr_c = ytr - y_mean
                    alpha = max(1.0, 0.05 * len(Xtr))
                    XtX = Xtr_c.T @ Xtr_c
                    XtX[np.diag_indices_from(XtX)] += alpha
                    w = np.linalg.solve(XtX, Xtr_c.T @ ytr_c)
                    preds = y_mean + (Xte - x_mean) @ w
                    ss_res = float(np.sum((yte - preds) ** 2))
                    ss_tot = float(np.sum((yte - float(yte.mean())) ** 2))
                    if ss_tot > 1e-12:
                        fold_r2s.append(1.0 - ss_res / ss_tot)
                cv_r2 = float(np.mean(fold_r2s)) if fold_r2s else 0.0

                y_centered = y - float(y.mean())
                ss_total = float(np.dot(y_centered, y_centered))
                max_adj_eta2 = 0.0
                if ss_total > 1e-12 and m > 2:
                    for other in probe_candidates:
                        if other == target_col:
                            continue
                        g = binned_codes[other][mask]
                        _, g_inv, g_cnt = np.unique(g, return_inverse=True, return_counts=True)
                        n_groups = len(g_cnt)
                        if n_groups < 2 or n_groups >= m:
                            continue
                        g_sums = np.bincount(g_inv, weights=y_centered, minlength=n_groups)
                        ss_between = float(np.sum((g_sums ** 2) / g_cnt))
                        eta2 = ss_between / ss_total
                        adj_eta2 = eta2 - (n_groups - 1) / float(m - 1)
                        if adj_eta2 > max_adj_eta2:
                            max_adj_eta2 = adj_eta2

                final_r2 = float(np.clip(max(cv_r2, 0.85 * max_adj_eta2), -1.0, 1.0))
                scores[target_col] = {"metric": "R2", "score": final_r2}
            else:
                codes_y, _ = pd.factorize(s_target.iloc[np.flatnonzero(mask)], sort=False)
                n_classes = int(codes_y.max()) + 1 if len(codes_y) else 0
                if n_classes < 2 or n_classes > 60:
                    continue
                counts = np.bincount(codes_y, minlength=n_classes)
                maj_frac = float(counts.max() / m)
                Y_oh = np.eye(n_classes, dtype=float)[codes_y]
                oof_pred = np.empty(m, dtype=int)
                covered = np.zeros(m, dtype=bool)
                for f in range(3):
                    te = folds == f
                    tr = ~te
                    if int(tr.sum()) < 5 or int(te.sum()) < 1:
                        continue
                    Xtr, Xte = X[tr], X[te]
                    Ytr = Y_oh[tr]
                    x_mean = Xtr.mean(axis=0)
                    y_mean = Ytr.mean(axis=0)
                    Xtr_c = Xtr - x_mean
                    Ytr_c = Ytr - y_mean
                    alpha = max(1.0, 0.05 * len(Xtr))
                    XtX = Xtr_c.T @ Xtr_c
                    XtX[np.diag_indices_from(XtX)] += alpha
                    W = np.linalg.solve(XtX, Xtr_c.T @ Ytr_c)
                    logits = y_mean + (Xte - x_mean) @ W
                    oof_pred[te] = np.argmax(logits, axis=1)
                    covered[te] = True
                if not covered.any():
                    continue
                y_eval = codes_y[covered]
                p_eval = oof_pred[covered]
                raw_acc = float((p_eval == y_eval).mean())
                eval_counts = np.bincount(y_eval, minlength=n_classes)
                recalls = [
                    float((p_eval[y_eval == c_idx] == c_idx).mean())
                    for c_idx in range(n_classes)
                    if eval_counts[c_idx] >= 2
                ]
                bal_acc = float(np.mean(recalls)) if recalls else raw_acc
                skill = max(0.0, (raw_acc - maj_frac) / max(1.0 - maj_frac, 1e-6))
                score = (
                    (0.6 * bal_acc + 0.4 * skill)
                    if maj_frac > 0.85
                    else (0.5 * raw_acc + 0.5 * bal_acc)
                )
                scores[target_col] = {
                    "metric": "accuracy",
                    "score": float(np.clip(score, 0.0, 1.0)),
                }
        except Exception:
            scores[target_col] = None

    return scores


SUGGEST_PROMPT = """A user is about to train a model on this dataset and needs to pick a \
column (or a few) as the prediction target -- the outcome or measurement they actually care \
about. More than one column can be a reasonable target; rank the best candidates.

Two independent, mechanical methods have already voted on these candidates. Neither \
understands what the columns *mean* -- weigh their votes as evidence, not as the final answer, \
and use your own judgment of the column names/samples to decide where they're right and where \
they're misleading (e.g. a column that happens to be easy to predict from the others isn't \
automatically a good target if it's clearly a dimension/category column, not an outcome).

CANDIDATE COLUMNS AND VOTES:
{profiles}

- naming/dtype vote: a rank from a heuristic that favors numeric columns and outcome-sounding \
names (total, amount, price, revenue, etc). Rank 1 = most target-like by this heuristic alone.
- learnability vote: how well a quick model predicts this column from the OTHER candidate \
columns (R2 for continuous columns, accuracy for categorical). Higher = more learnable from \
the rest of the data. "n/a" means the probe couldn't run (too little data, etc) -- treat as no \
vote either way, not a negative signal.

Rank up to {top_n} columns by how plausibly each represents a genuine outcome of interest, \
using both the votes above AND the column names/samples. Best first.

Respond in JSON:
{{"suggestions": [
    {{"column": "<one of the candidate names, exactly>", "reason": "one sentence, referencing the votes where they influenced your ranking"}}
]}}

Only include columns that are genuinely plausible targets -- pad with less-likely candidates \
only if you have fewer than {top_n} good ones, and say so honestly in the reason rather than \
inventing a strong justification for a weak candidate.
"""


def _format_votes(
    candidates: list[str],
    profile: dict[str, dict[str, Any]],
    rule_scores: dict[str, float],
    learn_scores: dict[str, dict[str, Any] | None],
) -> str:
    rule_rank = {
        col: i + 1
        for i, col in enumerate(sorted(candidates, key=lambda c: rule_scores.get(c, 0), reverse=True))
    }
    lines = []
    for col in candidates:
        p = profile.get(col, {})
        learn = learn_scores.get(col)
        learn_text = f"{learn['metric']}={learn['score']:.2f}" if learn else "n/a"
        lines.append(
            f"- {col}: dtype={p.get('dtype')}, unique={p.get('n_unique')}/{p.get('n_rows')}, "
            f"sample={p.get('sample')} | naming/dtype vote: rank {rule_rank.get(col)} of "
            f"{len(candidates)} | learnability vote: {learn_text}"
        )
    return "\n".join(lines)


def _llm_suggest(
    df: pd.DataFrame,
    candidates: list[str],
    provider,
    top_n: int,
    rule_scores: dict[str, float],
    learn_scores: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    from acdso.advisors import LLMConfig

    profile = _profile_columns(df, candidates)
    prompt = SUGGEST_PROMPT.format(
        profiles=_format_votes(candidates, profile, rule_scores, learn_scores),
        top_n=top_n,
    )

    config = LLMConfig(max_tokens=500, temperature=0.0, timeout_s=6.0, max_retries=1)
    response = provider.complete(
        system_prompt="You are a data analyst deciding a prediction target by weighing "
                      "mechanical votes against your own judgment of what the columns mean. "
                      "Only ever choose from the given candidate list.",
        user_prompt=prompt,
        config=config,
    )

    text = response if isinstance(response, str) else str(response)
    start, end = text.find("{"), text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON in LLM response")
    data = json.loads(text[start:end])

    suggestions = []
    seen = set()
    for item in data.get("suggestions", []):
        column = item.get("column")
        if column not in candidates or column in seen:
            continue
        seen.add(column)
        suggestions.append({
            "column": column,
            "reason": item.get("reason", ""),
            "votes": {"rule_based": rule_scores.get(column), "learnability": learn_scores.get(column)},
        })
    if not suggestions:
        raise ValueError("LLM returned no valid candidate columns")
    return {"suggestions": suggestions[:top_n], "source": "llm"}


def _combined_vote_ranking(
    candidates: list[str],
    rule_scores: dict[str, float],
    learn_scores: dict[str, dict[str, Any] | None],
    top_n: int,
) -> dict[str, Any]:
    """No LLM available: combine the two mechanical votes deterministically."""
    max_rule = max(rule_scores.values(), default=1.0) or 1.0
    learn_values = [v["score"] for v in learn_scores.values() if v]
    lo, hi = (min(learn_values), max(learn_values)) if learn_values else (0.0, 1.0)
    spread = (hi - lo) or 1.0

    def combined(col: str) -> float:
        rule_norm = rule_scores.get(col, 0) / max_rule
        learn = learn_scores.get(col)
        learn_norm = (learn["score"] - lo) / spread if learn else 0.5
        return 0.5 * rule_norm + 0.5 * learn_norm

    ranked = sorted(candidates, key=combined, reverse=True)[:top_n]
    suggestions = []
    for col in ranked:
        learn = learn_scores.get(col)
        learn_text = f"{learn['metric']}={learn['score']:.2f}" if learn else "not testable"
        suggestions.append({
            "column": col,
            "reason": f"Combined vote: naming/dtype heuristic and learnability ({learn_text}) "
                      f"both weighed in -- '{col}' ranked highest of the two combined.",
            "votes": {"rule_based": rule_scores.get(col), "learnability": learn},
        })
    return {"suggestions": suggestions, "source": "rules"}


def suggest_target_columns(
    df: pd.DataFrame, candidates: list[str] | None = None, top_n: int = 3
) -> dict[str, Any]:
    """Rank up to ``top_n`` predictable columns as plausible targets."""
    if candidates is None:
        candidates = predictable_columns(df)
    if not candidates:
        return {"suggestions": [], "source": "rules"}

    rule_scores = _rule_based_scores(df, candidates)
    learn_scores = _learnability_scores(df, candidates, rule_scores=rule_scores)

    provider = _get_provider()
    if provider is not None:
        try:
            return _llm_suggest(df, candidates, provider, top_n, rule_scores, learn_scores)
        except Exception:
            pass

    return _combined_vote_ranking(candidates, rule_scores, learn_scores, top_n)


# ---------------------------------------------------------------------------
# Horizon suggestion -- same two-vote-plus-LLM pattern as target suggestion
# ---------------------------------------------------------------------------

_GRAIN_HORIZON_DEFAULTS = {"D": 7, "W": 4, "M": 3, "Q": 4, "Y": 1}


def _numpy_detect_seasonality(series: pd.Series, max_period: int = 365) -> tuple[int | None, float]:
    """Pure-NumPy ACF seasonality detector (used when statsmodels is absent)."""
    import numpy as np

    arr = series.to_numpy(dtype=float)
    clean = arr[np.isfinite(arr)]
    n = len(clean)
    if n < 4:
        return None, 0.0
    max_lag = min(max_period, n // 2)
    if max_lag < 2:
        return None, 0.0
    centered = clean - float(clean.mean())
    denom = float(np.dot(centered, centered))
    if denom <= 1e-12:
        return None, 0.0
    full_corr = np.correlate(centered, centered, mode="full")
    acf_vals = full_corr[n - 1 : n + max_lag] / denom
    peaks = [
        (i, float(acf_vals[i]))
        for i in range(2, len(acf_vals) - 1)
        if acf_vals[i] > acf_vals[i - 1] and acf_vals[i] > acf_vals[i + 1] and acf_vals[i] > 0.1
    ]
    if not peaks:
        return None, 0.0
    best_period, strength = max(peaks, key=lambda x: x[1])
    for p in (7, 12, 24, 30, 52, 365):
        if abs(best_period - p) <= 2:
            best_period = p
            break
    return int(best_period), float(min(max(strength, 0.0), 1.0))


def _data_vote_horizon(series: pd.Series, grain: str) -> dict[str, Any]:
    """The 'data vote': one seasonal cycle if autocorrelation detects one,
    else a grain-typical default capped at roughly a third of observed history.
    """
    n = len(series)
    cap = max(1, n // 3)
    try:
        if importlib.util.find_spec("statsmodels") is not None:
            from acdso.timeseries.risk import detect_seasonality

            period, strength = detect_seasonality(series)
        else:
            period, strength = _numpy_detect_seasonality(series)
    except Exception:
        period, strength = None, 0.0

    if period and strength > 0.3:
        horizon = period
    else:
        horizon = _GRAIN_HORIZON_DEFAULTS.get(grain, 4)
    return {
        "horizon": max(1, min(horizon, cap)),
        "seasonality_period": period,
        "seasonality_strength": round(float(strength), 3),
        "series_length": n,
        "cap": cap,
    }


HORIZON_PROMPT = """A user is forecasting the column '{target_col}', aggregated to {grain} \
periods ({n} periods of history available after aggregation).

DATA VOTE (from autocorrelation-based seasonality detection on the actual series):
{data_vote_text}

Recommend a practical forecast horizon (a number of future periods) for this column. Weigh \
the data vote above against what a sensible planning horizon looks like for a column named \
'{target_col}' at {grain} grain -- e.g. a short lookout for an operational daily metric, a few \
quarters for something used in budgeting. The data vote is evidence, not the final answer; \
use your judgment of what this column represents to adjust it if the name suggests a \
different practical horizon.

Respond in JSON: {{"horizon": <positive integer>, "reason": "one sentence, plain language"}}

The horizon must not exceed {cap} periods (about a third of the available history) -- forecasts \
further out than that are unreliable regardless of what seems practical.
"""


def _llm_suggest_horizon(
    target_col: str, grain: str, data_vote: dict[str, Any], provider
) -> dict[str, Any]:
    from acdso.advisors import LLMConfig

    if data_vote["seasonality_period"]:
        data_vote_text = (
            f"seasonality period {data_vote['seasonality_period']} detected "
            f"(strength {data_vote['seasonality_strength']:.2f}); data-driven horizon: "
            f"{data_vote['horizon']} periods"
        )
    else:
        data_vote_text = (
            f"no strong seasonality detected in {data_vote['series_length']} periods; "
            f"data-driven horizon: {data_vote['horizon']} periods ({grain}-grain default)"
        )

    prompt = HORIZON_PROMPT.format(
        target_col=target_col, grain=grain, n=data_vote["series_length"],
        data_vote_text=data_vote_text, cap=data_vote["cap"],
    )
    config = LLMConfig(max_tokens=150, temperature=0.0, timeout_s=6.0, max_retries=1)
    response = provider.complete(
        system_prompt="You are a forecasting analyst picking a practical horizon, weighing "
                      "statistical evidence against your own judgment of the column's meaning.",
        user_prompt=prompt,
        config=config,
    )

    text = response if isinstance(response, str) else str(response)
    start, end = text.find("{"), text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON in LLM response")
    data = json.loads(text[start:end])

    horizon = int(data.get("horizon", 0))
    if horizon <= 0:
        raise ValueError(f"LLM returned a non-positive horizon: {horizon!r}")
    horizon = max(1, min(horizon, data_vote["cap"]))

    return {
        "horizon": horizon,
        "reason": data.get("reason", ""),
        "source": "llm",
        "grain": grain,
        "votes": {"data": data_vote},
    }


def suggest_horizon(df: pd.DataFrame, datetime_col: str, target_col: str) -> dict[str, Any]:
    """Recommend a forecast horizon for --horizon."""
    import numpy as np
    from .forecast import SeriesSpec, aggregate_series

    try:
        series, meta = aggregate_series(
            df, SeriesSpec(date_col=datetime_col, value_col=target_col, agg="sum")
        )
    except Exception as exc:
        return {"horizon": None, "reason": f"Could not aggregate the series: {exc}", "source": "rules"}

    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 4:
        return {
            "horizon": None,
            "reason": f"Only {len(series)} period(s) after aggregation -- too few to suggest a horizon.",
            "source": "rules",
            "grain": meta.get("grain"),
        }

    grain = meta["grain"]
    data_vote = _data_vote_horizon(series, grain)

    provider = _get_provider()
    if provider is not None:
        try:
            return _llm_suggest_horizon(target_col, grain, data_vote, provider)
        except Exception:
            pass

    reason = (
        f"Seasonality period {data_vote['seasonality_period']} detected "
        f"(strength {data_vote['seasonality_strength']:.2f}) at {grain} grain."
        if data_vote["seasonality_period"]
        else f"No strong seasonality detected at {grain} grain; using a typical default for "
             f"this grain."
    )
    return {
        "horizon": data_vote["horizon"],
        "reason": reason,
        "source": "rules",
        "grain": grain,
        "votes": {"data": data_vote},
    }
