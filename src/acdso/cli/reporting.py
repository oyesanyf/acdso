"""HTML/JSON report generation, escaping, and sanitization for ACDSO CLI."""

from __future__ import annotations

from datetime import datetime
import html as _html
import json
import math
import os
import re
from typing import Any, Dict, List, Optional


def _load_metrics_deps():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    return np, pd, accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error


def _esc(val: Any) -> str:
    """HTML-escape arbitrary values safely including quotes."""
    return _html.escape("" if val is None else str(val), quote=True)


def _fmt_num(val: Any, fmt: str = ".4f", default: str = "N/A") -> str:
    """Safely format finite numeric scalars (never matching bool or NaN/Inf)."""
    if val is None or isinstance(val, bool):
        return _esc(val) if isinstance(val, bool) else default
    if isinstance(val, (int, float)):
        fval = float(val)
        return format(fval, fmt) if math.isfinite(fval) else default
    if hasattr(val, "item") and not isinstance(val, (str, bytes, bytearray)):
        try:
            item_val = val.item()
            if isinstance(item_val, bool):
                return _esc(item_val)
            if isinstance(item_val, (int, float)):
                fval = float(item_val)
                return format(fval, fmt) if math.isfinite(fval) else default
        except Exception:
            pass
    return _esc(val)


def _fmt_int(val: Any, default: str = "N/A") -> str:
    """Safely format integer counts with thousands separators."""
    if val is None or isinstance(val, bool):
        return _esc(val) if isinstance(val, bool) else default
    if hasattr(val, "item") and not isinstance(val, (str, bytes, bytearray)):
        try:
            val = val.item()
        except Exception:
            pass
    if isinstance(val, bool):
        return _esc(val)
    if isinstance(val, int):
        return f"{val:,}"
    if isinstance(val, float):
        return f"{int(val):,}" if math.isfinite(val) else default
    return _esc(val)


def _json_safe_scalar(val: Any) -> Any:
    """Convert scalar values to RFC 8259-compliant JSON primitives (NaN/inf -> None)."""
    import numpy as np
    import pandas as pd
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        return fval if math.isfinite(fval) else None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return str(val)


def _sanitize_json_payload(obj: Any) -> Any:
    """Recursively sanitize nested payloads for strict RFC 8259 JSON (allow_nan=False)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_payload(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_sanitize_json_payload(v) for v in sorted(obj, key=str)]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return _sanitize_json_payload(obj.tolist())
        except Exception:
            pass
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    return _json_safe_scalar(obj)


def _sanitize_embedded_html(raw_html: str) -> str:
    """Sanitize and sandbox embedded EDA/TS HTML fragments to prevent stored XSS."""
    import re

    content = str(raw_html or "")
    if "<body" in content.lower():
        m_body = re.search(r"<body[^>]*>(.*)</body\s*>", content, flags=re.DOTALL | re.IGNORECASE)
        if m_body:
            content = m_body.group(1)
    # Strip <script> tags including malformed closing tags (e.g. </script >) and event handlers
    content = re.sub(r"<script\b[^>]*>.*?</script\s*>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"\son[a-z]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", "", content, flags=re.IGNORECASE)
    content = re.sub(r"javascript\s*:", "blocked:", content, flags=re.IGNORECASE)
    return (
        f'<iframe sandbox="" srcdoc="{_esc(content)}" '
        'style="width:100%; height:600px; border:none; background:#fff;"></iframe>'
    )



def _infer_estimator_task(estimator: Any, explicit_task: Optional[str] = None) -> Optional[str]:
    """Infer canonical ACDSO task name from an estimator without eager ML imports."""
    if explicit_task and str(explicit_task).strip() and str(explicit_task).strip().lower() != "auto":
        return str(explicit_task).strip().lower()
    if estimator is None:
        return None
    cls_name = type(estimator).__name__
    if cls_name == "ACDSOClusterer":
        return "clustering"
    if cls_name == "ACDSOAnomalyDetector":
        return "anomaly"
    if cls_name == "ACDSOMultiLabelClassifier":
        return "multilabel"
    if cls_name == "ACDSORanker":
        return "ranking"
    if cls_name == "ACDSOSurvivalAnalyzer":
        return "survival"
    if cls_name == "ACDSORecommender":
        return "recommender"
    if cls_name == "ACDSOOutOfCoreTrainer":
        t = getattr(estimator, "task_", None) or getattr(estimator, "task", None)
        return str(t).lower() if t else "classification"
    if cls_name == "ACDSOClassifier":
        return "classification"
    if cls_name == "ACDSORegressor":
        return "regression"
    if hasattr(estimator, "predict_proba"):
        return "classification"
    if hasattr(estimator, "predict"):
        return "regression"
    return None


def populate_task_summary_blocks(
    report: Dict[str, Any],
    *,
    task: Optional[str] = None,
    estimator: Any = None,
    metrics: Optional[Dict[str, Any]] = None,
    clustering_info: Optional[Dict[str, Any]] = None,
    anomaly_info: Optional[Dict[str, Any]] = None,
    cv_info: Optional[Dict[str, Any]] = None,
    nlp_info: Optional[Dict[str, Any]] = None,
    specialized_info: Optional[Dict[str, Any]] = None,
    scale_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Populate canonical R1-R6 task and summary blocks on a combined report dict."""
    resolved_task = _infer_estimator_task(estimator, explicit_task=task or report.get("task"))
    if resolved_task:
        report["task"] = resolved_task

    audit = getattr(estimator, "audit_trail_", None) if estimator is not None else None
    if isinstance(audit, dict) and "audit_trail" not in report:
        report["audit_trail"] = audit
    audit_dict = report.get("audit_trail") if isinstance(report.get("audit_trail"), dict) else {}

    cls_name = type(estimator).__name__ if estimator is not None else ""

    # Clustering (R1)
    if clustering_info:
        report["clustering"] = dict(clustering_info)
    elif cls_name == "ACDSOClusterer" and hasattr(estimator, "get_cluster_summary"):
        try:
            report["clustering"] = dict(estimator.get_cluster_summary())
        except Exception:
            pass

    # Anomaly (R2)
    if anomaly_info:
        report["anomaly"] = dict(anomaly_info)
    elif cls_name == "ACDSOAnomalyDetector":
        anom_block = {
            "best_algorithm": getattr(estimator, "best_algorithm_", None) or audit_dict.get("best_algorithm"),
            "algorithm_requested": audit_dict.get("algorithm_requested", getattr(estimator, "algorithm", None)),
            "contamination": audit_dict.get("contamination", getattr(estimator, "contamination", None)),
            "n_samples": audit_dict.get("n_samples"),
            "offset": getattr(estimator, "offset_", audit_dict.get("offset")),
            "threshold": getattr(estimator, "threshold_", audit_dict.get("threshold")),
            "fit_is_sparse": audit_dict.get("fit_is_sparse"),
            "text_cols": audit_dict.get("text_cols") or getattr(estimator, "text_cols_", None),
        }
        report["anomaly"] = {k: v for k, v in anom_block.items() if v is not None}

    # CV / Grouping (R3)
    if cv_info:
        report["cv"] = dict(cv_info)
    else:
        cv_strat = audit_dict.get("cv_strategy") or getattr(estimator, "cv_strategy_", None)
        if cv_strat or audit_dict.get("groups_provided"):
            report["cv"] = {
                k: v
                for k, v in {
                    "cv_strategy": cv_strat,
                    "groups_provided": audit_dict.get("groups_provided"),
                    "n_unique_groups": audit_dict.get("n_groups", audit_dict.get("n_unique_groups")),
                }.items()
                if v is not None
            }

    # NLP / Free-text (R4)
    if nlp_info:
        report["nlp"] = dict(nlp_info)
    else:
        tcols = audit_dict.get("text_cols") or getattr(estimator, "text_cols_", None)
        tmeth = audit_dict.get("text_method") or getattr(estimator, "text_method", None)
        if tcols:
            report["nlp"] = {
                "text_cols": list(tcols),
                "n_text_cols": len(tcols),
                "nlp_method": tmeth or "tfidf",
            }

    # Scale (R6)
    if scale_info:
        report["scale"] = dict(scale_info)
    elif cls_name == "ACDSOOutOfCoreTrainer" and hasattr(estimator, "get_scale_summary"):
        try:
            report["scale"] = dict(estimator.get_scale_summary())
        except Exception:
            pass
    elif isinstance(audit_dict.get("scale_guard"), dict):
        report["scale"] = dict(audit_dict["scale_guard"])

    # Specialized Supervised (R5)
    if specialized_info:
        report["specialized"] = dict(specialized_info)
    elif cls_name in ("ACDSOMultiLabelClassifier", "ACDSORanker", "ACDSOSurvivalAnalyzer", "ACDSORecommender"):
        spec: Dict[str, Any] = {"task": resolved_task}
        for attr, key in (
            ("best_strategy_", "best_strategy"),
            ("best_algorithm_", "best_algorithm"),
            ("best_method_", "best_method"),
            ("backend_used_", "backend_used"),
            ("n_labels_", "n_labels"),
            ("n_users_", "n_users"),
            ("n_items_", "n_items"),
            ("user_col_", "user_col"),
            ("item_col_", "item_col"),
            ("rating_col_", "rating_col"),
            ("is_implicit_", "is_implicit"),
            ("k", "k"),
        ):
            val = getattr(estimator, attr, None)
            if val is not None:
                spec[key] = val
        report["specialized"] = spec

    # Metrics summary block
    merged_metrics: Dict[str, Any] = {}
    if isinstance(report.get("metrics"), dict):
        merged_metrics.update(report["metrics"])
    if isinstance(report.get("test_performance"), dict):
        for k, v in report["test_performance"].items():
            if k != "error" and v is not None:
                merged_metrics[k] = v
    if isinstance(report.get("clustering"), dict):
        for k in ("silhouette_score", "davies_bouldin_score", "n_clusters", "noise_count", "noise_ratio"):
            if report["clustering"].get(k) is not None:
                merged_metrics.setdefault(k, report["clustering"][k])
    if isinstance(report.get("anomaly"), dict):
        for k in ("n_anomalies", "anomaly_rate", "offset", "threshold"):
            if report["anomaly"].get(k) is not None:
                merged_metrics.setdefault(k, report["anomaly"][k])
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if v is not None:
                merged_metrics[k] = v
    if merged_metrics:
        report["metrics"] = merged_metrics

    return report


def generate_combined_report(
    estimator: Any,
    console_output: str,
    test_predictions: Optional[Any] = None,
    test_targets: Optional[Any] = None,
    additional_info: Optional[Dict[str, Any]] = None,
    *,
    task: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    cv_info: Optional[Dict[str, Any]] = None,
    nlp_info: Optional[Dict[str, Any]] = None,
    specialized_info: Optional[Dict[str, Any]] = None,
    scale_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive combined report that includes everything shown in console
    plus all structured data from ACDSO.
    """
    np, pd, accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error = _load_metrics_deps()
    resolved_task = _infer_estimator_task(estimator, explicit_task=task)
    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "acdso_version": "3.0",
        "console_output": console_output,
    }
    if resolved_task:
        report["task"] = resolved_task

    # Add estimator configuration
    if hasattr(estimator, 'scorer'):
        report["configuration"] = {
            "scorer": estimator.scorer,
            "cv": getattr(estimator, 'cv', None),
            "preference": getattr(estimator, 'preference', None),
            "max_candidates": getattr(estimator, 'max_candidates', None),
            "random_state": getattr(estimator, 'random_state', None),
            "time_budget_s": getattr(estimator, 'time_budget_s', None),
            "max_train_time_s": getattr(estimator, 'max_train_time_s', None),
            "max_peak_mem_mb": getattr(estimator, 'max_peak_mem_mb', None),
            "max_latency_ms_per_row": getattr(estimator, 'max_latency_ms_per_row', None),
            "enable_feature_selection": getattr(estimator, 'enable_feature_selection', None),
        }

    # Add risk report
    if hasattr(estimator, 'risk_report_') and estimator.risk_report_ is not None:
        report["risk_report"] = estimator.risk_report_

    # Add audit trail (contains all evaluation details)
    audit_dict = getattr(estimator, 'audit_trail_', None)
    if isinstance(audit_dict, dict):
        report["audit_trail"] = audit_dict
    else:
        audit_dict = {}

    # Add Pareto front
    if hasattr(estimator, 'pareto_front_') and estimator.pareto_front_ is not None:
        report["pareto_front"] = estimator.pareto_front_

    # Add selected model information
    if hasattr(estimator, 'best_estimator_') and estimator.best_estimator_ is not None:
        selected_info = audit_dict.get('selected') or audit_dict.get('selected_candidate') or {}
        if not isinstance(selected_info, dict):
            selected_info = {}
        sel_name = (
            selected_info.get('name')
            or selected_info.get('algorithm')
            or selected_info.get('strategy')
            or selected_info.get('method')
            or getattr(estimator, 'best_algorithm_', None)
            or getattr(estimator, 'best_strategy_', None)
            or getattr(estimator, 'best_method_', None)
            or type(estimator.best_estimator_).__name__
        )
        report["selected_model"] = {
            "name": sel_name,
            "type": type(estimator.best_estimator_).__name__,
            "pipeline_steps": [step[0] for step in estimator.best_estimator_.steps] if hasattr(estimator.best_estimator_, 'steps') else None,
        }
        if selected_info:
            sel_metrics: Dict[str, Any] = {}
            for mkey in (
                "metric",
                "silhouette_score",
                "davies_bouldin_score",
                "objective",
                "cv_objective",
                "cv_ndcg_at_k",
                "cv_c_index",
                "rmse",
                "mae",
                "score",
                "train_time_s",
                "peak_mem_mb",
                "latency_ms_per_row",
                "risk_penalty",
                "stress_worst_drop",
            ):
                if mkey in selected_info and selected_info.get(mkey) is not None:
                    sel_metrics[mkey] = selected_info.get(mkey)
            if sel_metrics:
                report["selected_model"]["metrics"] = sel_metrics
    elif estimator is not None and type(estimator).__name__ in (
        "ACDSOClusterer",
        "ACDSOAnomalyDetector",
        "ACDSOSurvivalAnalyzer",
        "ACDSORecommender",
        "ACDSOOutOfCoreTrainer",
    ):
        sel_name = (
            getattr(estimator, "best_algorithm_", None)
            or getattr(estimator, "best_method_", None)
            or getattr(estimator, "algorithm_", None)
            or getattr(estimator, "backend_used_", None)
            or type(estimator).__name__
        )
        report["selected_model"] = {
            "name": str(sel_name),
            "type": type(estimator).__name__,
        }
        if isinstance(metrics, dict) and metrics:
            report["selected_model"]["metrics"] = dict(metrics)

    # Add test set performance if provided for standard supervised classification/regression
    if test_predictions is not None and test_targets is not None and resolved_task in (None, "classification", "regression"):
        try:
            if resolved_task == "classification" or (resolved_task is None and hasattr(estimator, 'predict_proba')):
                test_acc = accuracy_score(test_targets, test_predictions)
                test_f1 = f1_score(test_targets, test_predictions, average='weighted')
                report["test_performance"] = {
                    "accuracy": float(test_acc),
                    "f1_weighted": float(test_f1),
                }
            else:
                test_r2 = r2_score(test_targets, test_predictions)
                test_mse = mean_squared_error(test_targets, test_predictions)
                test_mae = mean_absolute_error(test_targets, test_predictions)
                report["test_performance"] = {
                    "r2": float(test_r2),
                    "mse": float(test_mse),
                    "mae": float(test_mae),
                    "rmse": float(np.sqrt(test_mse)),
                }
        except Exception as e:
            report["test_performance"] = {"error": str(e)}

    if additional_info:
        report["additional_info"] = additional_info

    populate_task_summary_blocks(
        report,
        task=resolved_task,
        estimator=estimator,
        metrics=metrics,
        cv_info=cv_info,
        nlp_info=nlp_info,
        specialized_info=specialized_info,
        scale_info=scale_info,
    )
    return report


def save_combined_report(report: Dict[str, Any], output_path: str = "acdso_combined_report.json") -> str:
    """
    Save the combined report to a JSON file.
    
    Parameters
    ----------
    report : dict
        Combined report dictionary.
    output_path : str
        Path to save the JSON file.
    
    Returns
    -------
    str
        Path to saved file.
    """
    sanitized = _sanitize_json_payload(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False, allow_nan=False)
    return output_path


def _file_uri(path: str) -> str:
    """Convert an absolute filesystem path to a clickable file:// URI.

    Handles both Windows ('C:\\a\\b c.pkl') and POSIX ('/a/b c.pkl') paths,
    percent-encoding spaces/parentheses so browsers open them reliably.
    """
    from urllib.parse import quote
    p = path.replace('\\', '/')
    if not p.startswith('/'):
        p = '/' + p  # Windows drive letter: C:/... -> /C:/...
    return 'file://' + quote(p, safe='/:')


def _file_link_cell(path: str) -> str:
    """Table cell for a generated-file path: click to open/download, or copy."""
    raw_path = "" if path is None else str(path)
    uri = _esc(_file_uri(raw_path))
    esc_path = _esc(raw_path)
    return (
        f'<a href="{uri}" target="_blank" rel="noopener" '
        f'style="color:#93c5fd; text-decoration:none; background:#2d3748; '
        f'padding:4px 8px; border-radius:4px; font-family:Consolas,monospace; '
        f'font-size:0.85em; word-break:break-all;">{esc_path}</a> '
        f'<button type="button" data-path="{esc_path}" '
        f'onclick="navigator.clipboard.writeText(this.dataset.path).then(()=>alert(\'Path copied!\'))" '
        f'style="background:#334155; color:#94a3b8; border:none; border-radius:4px; '
        f'padding:3px 8px; cursor:pointer; font-size:0.75em; vertical-align:middle;">copy</button>'
    )


def _render_kv_table_card(title: str, subtitle: str, mapping: Dict[str, Any]) -> str:
    """Render an XSS-escaped HTML card containing a balanced key-value table."""
    if not isinstance(mapping, dict) or not mapping:
        return ""
    rows: List[str] = []
    sub_tables: List[str] = []
    for key, val in mapping.items():
        if val is None:
            continue
        if isinstance(val, dict) and val:
            sub_rows = [
                f"                <tr><th>{_esc(sk)}</th><td>{_fmt_num(sv) if (isinstance(sv, (int, float)) and not isinstance(sv, bool)) or (hasattr(sv, 'item') and not isinstance(sv, (str, bytes, bytearray))) else _esc(sv)}</td></tr>\n"
                for sk, sv in val.items()
                if sv is not None
            ]
            if sub_rows:
                sub_tables.append(
                    f"            <h3>{_esc(str(key).replace('_', ' ').title())}</h3>\n"
                    f"            <table>\n{''.join(sub_rows)}            </table>\n"
                )
            continue
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                continue
            if all(not isinstance(item, (dict, list, tuple)) for item in val):
                val_html = _esc(", ".join(str(item) for item in val))
            else:
                val_html = _esc(json.dumps(_sanitize_json_payload(val)))
        elif isinstance(val, bool):
            val_html = _esc(val)
        elif isinstance(val, int):
            val_html = _fmt_int(val)
        elif hasattr(val, "item") and not isinstance(val, (str, bytes, bytearray)):
            try:
                unwrapped = val.item()
            except Exception:
                unwrapped = val
            if isinstance(unwrapped, bool):
                val_html = _esc(unwrapped)
            elif isinstance(unwrapped, int):
                val_html = _fmt_int(unwrapped)
            else:
                val_html = _fmt_num(unwrapped)
        elif isinstance(val, float):
            val_html = _fmt_num(val)
        else:
            val_html = _esc(val)
        rows.append(f"                <tr><th>{_esc(key)}</th><td>{val_html}</td></tr>\n")

    if not rows and not sub_tables:
        return ""
    out = f"""
        <div class="card">
            <h2>{_esc(title)}</h2>
            <p class="info-text">{_esc(subtitle)}</p>
"""
    if rows:
        out += "            <table>\n" + "".join(rows) + "            </table>\n"
    if sub_tables:
        out += "".join(sub_tables)
    out += "        </div>\n"
    return out


def generate_html_report_from_combined(report: Dict[str, Any], output_path: str = None) -> str:
    """
    Generate a comprehensive HTML report from combined report data.
    
    Works for all modes: autonomous, benchmark, single demos, etc.
    Includes model names, metrics, console output, and all structured data.
    
    Parameters
    ----------
    report : dict
        Combined report dictionary.
    output_path : str, optional
        Path to save HTML file. Auto-generated if not provided.
    
    Returns
    -------
    str
        Path to saved HTML file.
    """
    import os
    from pathlib import Path
    
    # Determine output path
    if output_path is None:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        timestamp = report.get('timestamp', datetime.now().isoformat()).replace(':', '-').replace('T', '_')[:19]
        mode = report.get('mode', 'demo')
        output_path = reports_dir / f"acdso_{mode}_report_{timestamp}.html"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    mode = report.get('mode', 'demo')
    task_label = report.get('task')
    timestamp = report.get('timestamp', 'N/A')
    acdso_version = report.get('acdso_version', '3.0')
    task_subtitle = f'\n            <div class="subtitle">Task: {_esc(task_label)}</div>' if task_label else ''
    
    # Build HTML content
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%232563eb'/%3E%3Ctext x='32' y='45' font-family='Arial,sans-serif' font-size='32' font-weight='700' fill='%23ffffff' text-anchor='middle'%3EA%3C/text%3E%3C/svg%3E">
    <title>ACDSO {_esc(acdso_version)} Report - {_esc(str(mode).title())}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-light: #3b82f6;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --info: #0891b2;
            --bg: #0f172a;
            --bg-card: #1e293b;
            --bg-highlight: #334155;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, var(--bg) 0%, #1a1a2e 100%);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
            min-height: 100vh;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(139, 92, 246, 0.1));
            border-radius: 16px;
            border: 1px solid var(--border);
        }}
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }}
        h2 {{ color: var(--primary-light); margin: 1.5rem 0 1rem 0; font-size: 1.5rem; }}
        h3 {{ color: var(--primary-light); margin: 1rem 0 0.5rem 0; font-size: 1.2rem; }}
        .subtitle {{ color: var(--text-muted); margin-bottom: 0.5rem; }}
        
        /* Cards */
        .card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: rgba(37, 99, 235, 0.15);
            font-weight: 600;
            color: var(--primary-light);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        tr:hover {{ background: rgba(255, 255, 255, 0.03); }}
        .model-name {{ color: var(--success); font-weight: 600; }}
        .success {{ color: var(--success); }}
        .error {{ color: var(--danger); }}
        .selected-model {{ background: rgba(22, 163, 74, 0.15); border-left: 3px solid var(--success); }}
        .selected-model:hover {{ background: rgba(22, 163, 74, 0.25); }}
        .models-table {{ font-size: 0.9rem; }}
        .models-table th {{ position: sticky; top: 0; z-index: 10; }}
        .info-text {{ color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1rem; font-style: italic; }}
        
        /* Code/Console Output */
        .console-output {{
            background: #000;
            color: #0f0;
            padding: 1rem;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
            white-space: pre-wrap;
        }}
        
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .badge-success {{ background: rgba(22, 163, 74, 0.2); color: var(--success); }}
        .badge-danger {{ background: rgba(220, 38, 38, 0.2); color: var(--danger); }}
        .badge-info {{ background: rgba(8, 145, 178, 0.2); color: var(--info); }}
        
        /* Summary Grid */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .summary-item {{
            background: rgba(0,0,0,0.2);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-value {{
            font-size: 2rem;
            font-weight: 600;
            color: var(--primary-light);
        }}
        .summary-label {{
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ACDSO {_esc(acdso_version)} Report</h1>
            <div class="subtitle">Mode: {_esc(str(mode).title())}</div>{task_subtitle}
            <div class="subtitle">Generated: {_esc(timestamp)}</div>
        </div>
"""

    # Add configuration section
    if isinstance(report.get('configuration'), dict) and report['configuration']:
        config = report['configuration']
        html += """
        <div class="card">
            <h2>Configuration</h2>
            <table>
"""
        for key, value in config.items():
            if value is not None:
                html += f"                <tr><th>{_esc(key)}</th><td>{_esc(value)}</td></tr>\n"
        html += "            </table>\n        </div>\n"

    # Add selected model information
    if isinstance(report.get('selected_model'), dict) and report['selected_model']:
        model = report['selected_model']
        html += f"""
        <div class="card">
            <h2>Selected Model</h2>
            <table>
                <tr><th>Model Name</th><td class="model-name">{_esc(model.get('name', 'N/A'))}</td></tr>
                <tr><th>Type</th><td>{_esc(model.get('type', 'N/A'))}</td></tr>
"""
        if 'pipeline_steps' in model and model['pipeline_steps']:
            steps_html = ' → '.join(_esc(s) for s in model['pipeline_steps'])
            html += f"                <tr><th>Pipeline Steps</th><td>{steps_html}</td></tr>\n"
        if isinstance(model.get('metrics'), dict) and model['metrics']:
            metrics = model['metrics']
            html += "                <tr><th colspan='2'><h3>Metrics</h3></th></tr>\n"
            for key, value in metrics.items():
                if value is not None:
                    html += f"                <tr><th>{_esc(key)}</th><td>{_fmt_num(value)}</td></tr>\n"
        html += "            </table>\n"
        # Display actual execution code
        code_info = model.get('code_info', {})
        if code_info and isinstance(code_info, dict):
            code = code_info.get('code', '')
            execution_summary = code_info.get('execution_summary', '')
            if code:
                html += f"""
            <h3>Actual Execution Code</h3>
            <p class="info-text"><strong>Execution Summary:</strong> {_esc(execution_summary)}</p>
            <pre style="background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px; overflow-x: auto; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px;"><code>{_esc(code)}</code></pre>
"""
        html += "        </div>\n"

    # Add R1-R6 task summary cards (metrics, clustering, anomaly, cv, nlp, specialized, scale)
    for block_key, block_title, block_desc in (
        ("metrics", "Task Metrics", "Primary evaluation metrics for the executed task."),
        ("clustering", "Clustering Summary", "Unsupervised segmentation, cluster counts, and internal validation scores."),
        ("anomaly", "Anomaly Detection Summary", "Trainable anomaly estimator configuration, thresholds, and detection rate."),
        ("cv", "Cross-Validation & Grouping", "Entity-grouped or temporal cross-validation configuration and split validation."),
        ("nlp", "NLP & Free-Text Featurization", "Classical text column detection and featurization settings."),
        ("specialized", "Specialized Formulation Details", "Configuration and evaluation metrics for specialized supervised tasks."),
        ("scale", "Out-of-Core & Scale Summary", "Chunked streaming / subsampling execution metrics and resource budget."),
    ):
        block_val = report.get(block_key)
        if isinstance(block_val, dict) and block_val:
            html += _render_kv_table_card(block_title, block_desc, block_val)

    # Add phase results with model names (for autonomous mode)
    if isinstance(report.get('phase_results'), dict) and report['phase_results']:
        html += """
        <div class="card">
            <h2>Phase Results with Selected Models</h2>
"""
        for phase_name, phase_data in report['phase_results'].items():
            if isinstance(phase_data, list) and phase_data:
                html += f"            <h3>{_esc(phase_name)}</h3>\n            <table>\n"
                html += "                <tr><th>Demo Name</th><th>Status</th><th>Selected Model</th><th>Time (s)</th></tr>\n"
                for item in phase_data:
                    if isinstance(item, dict):
                        name = item.get('name', 'Unknown')
                        status = item.get('status', 'Unknown')
                        model_name = item.get('model_name', 'N/A')
                        time_val = item.get('time', 0) or 0
                        status_class = 'success' if 'SUCCESS' in str(status) else 'error'
                        time_str = _fmt_num(time_val, ".2f")
                        html += f"                <tr><td>{_esc(name)}</td><td class='{status_class}'>{_esc(status)}</td><td class='model-name'>{_esc(model_name)}</td><td>{time_str}</td></tr>\n"
                html += "            </table>\n"
        html += "        </div>\n"

    # Add audit trail with all candidates evaluated (ACDSO)
    if isinstance(report.get('audit_trail'), dict) and report['audit_trail']:
        audit = report['audit_trail']
        candidates = audit.get('candidates_evaluated') or audit.get('candidates') or []
        if isinstance(candidates, list) and candidates:
            has_supervised_cols = any(
                isinstance(c, dict) and ('metric' in c or 'train_time_s' in c)
                for c in candidates
            )
            selected_raw = audit.get('selected') or audit.get('selected_candidate') or {}
            selected = selected_raw if isinstance(selected_raw, dict) else {"name": str(selected_raw)}
            sel_name = (
                selected.get('name')
                or selected.get('algorithm')
                or selected.get('strategy')
                or selected.get('method')
            )
            if has_supervised_cols:
                html += """
        <div class="card">
            <h2>All Models Evaluated by ACDSO</h2>
            <p class="info-text">Complete list of all models tested during optimization with their scores and metrics.</p>
            <table class="models-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Model Name</th>
                        <th>CV Score</th>
                        <th>CV Std</th>
                        <th>Train Time (s)</th>
                        <th>Memory (MB)</th>
                        <th>Latency (ms/row)</th>
                        <th>Risk Penalty</th>
                        <th>Stress Drop</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
                for i, candidate in enumerate(candidates, 1):
                    if not isinstance(candidate, dict):
                        continue
                    name = (
                        candidate.get('name')
                        or candidate.get('algorithm')
                        or candidate.get('strategy')
                        or candidate.get('method')
                        or 'Unknown'
                    )
                    metric = candidate.get('metric', candidate.get('score'))
                    cv_std = candidate.get('cv_score_std', 0) or 0
                    train_time = candidate.get('train_time_s', 0) or 0
                    memory = candidate.get('peak_mem_mb', 0) or 0
                    latency = candidate.get('latency_ms_per_row', 0) or 0
                    risk_penalty = candidate.get('risk_penalty', 0) or 0
                    stress_drop = candidate.get('stress_worst_drop', None)

                    is_selected = sel_name is not None and sel_name == name
                    row_class = 'selected-model' if is_selected else ''
                    status = '✓ Selected' if is_selected else 'Evaluated'

                    stress_str = _fmt_num(stress_drop, ".4f") if stress_drop is not None else "N/A"
                    metric_str = _fmt_num(metric, ".4f")
                    cv_std_str = _fmt_num(cv_std, ".4f")
                    train_time_str = _fmt_num(train_time, ".2f")
                    memory_str = _fmt_num(memory, ".1f")
                    latency_str = _fmt_num(latency, ".3f")
                    risk_str = _fmt_num(risk_penalty, ".4f")

                    html += f"""                    <tr class="{row_class}">
                        <td>{i}</td>
                        <td class="model-name">{_esc(name)}</td>
                        <td>{metric_str}</td>
                        <td>{cv_std_str}</td>
                        <td>{train_time_str}</td>
                        <td>{memory_str}</td>
                        <td>{latency_str}</td>
                        <td>{risk_str}</td>
                        <td>{stress_str}</td>
                        <td>{status}</td>
                    </tr>
"""
                html += """                </tbody>
            </table>
        </div>
"""
            else:
                html += """
        <div class="card">
            <h2>Candidates Evaluated</h2>
            <p class="info-text">Candidate configurations evaluated during model selection.</p>
            <table class="models-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Candidate</th>
                        <th>Primary Score</th>
                        <th>Details</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
                for i, candidate in enumerate(candidates, 1):
                    if not isinstance(candidate, dict):
                        continue
                    cname = (
                        candidate.get('name')
                        or candidate.get('algorithm')
                        or candidate.get('strategy')
                        or candidate.get('method')
                        or f'Candidate {i}'
                    )
                    score_val = None
                    for skey in (
                        'silhouette_score',
                        'cv_ndcg_at_k',
                        'cv_c_index',
                        'cv_objective',
                        'objective',
                        'rmse',
                        'mae',
                        'score',
                        'metric',
                    ):
                        if candidate.get(skey) is not None:
                            score_val = candidate.get(skey)
                            break
                    details = ", ".join(
                        f"{k}={_fmt_num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v}"
                        for k, v in candidate.items()
                        if k not in ('name', 'algorithm', 'strategy', 'method') and v is not None
                    )
                    is_selected = bool(candidate.get('selected')) or (sel_name is not None and sel_name == cname)
                    row_class = 'selected-model' if is_selected else ''
                    status = '✓ Selected' if is_selected else 'Evaluated'
                    html += f"""                    <tr class="{row_class}">
                        <td>{i}</td>
                        <td class="model-name">{_esc(cname)}</td>
                        <td>{_fmt_num(score_val)}</td>
                        <td>{_esc(details or 'N/A')}</td>
                        <td>{status}</td>
                    </tr>
"""
                html += """                </tbody>
            </table>
        </div>
"""

    # Add Pareto front models
    if isinstance(report.get('audit_trail'), dict) and report['audit_trail']:
        audit = report['audit_trail']
        pareto_front = audit.get('pareto_front') or []
        if isinstance(pareto_front, list) and pareto_front:
            html += """
        <div class="card">
            <h2>Pareto Front Models (Optimal Trade-offs)</h2>
            <p class="info-text">Models on the Pareto front representing optimal trade-offs between objectives.</p>
            <table class="models-table">
                <thead>
                    <tr>
                        <th>Model Name</th>
                        <th>Score</th>
                        <th>Train Time (s)</th>
                        <th>Memory (MB)</th>
                        <th>Latency (ms/row)</th>
                        <th>Risk Penalty</th>
                    </tr>
                </thead>
                <tbody>
"""
            for model in pareto_front:
                if not isinstance(model, dict):
                    continue
                name = model.get('name', model.get('model_name', 'Unknown'))
                metric = model.get('metric', 0) or 0
                train_time = model.get('train_time_s', 0) or 0
                memory = model.get('peak_mem_mb', 0) or 0
                latency = model.get('latency_ms_per_row', 0) or 0
                risk_penalty = model.get('risk_penalty', 0) or 0

                metric_str = _fmt_num(metric, ".4f")
                train_time_str = _fmt_num(train_time, ".2f")
                memory_str = _fmt_num(memory, ".1f")
                latency_str = _fmt_num(latency, ".3f")
                risk_str = _fmt_num(risk_penalty, ".4f")
                html += f"""                    <tr>
                        <td class="model-name">{_esc(name)}</td>
                        <td>{metric_str}</td>
                        <td>{train_time_str}</td>
                        <td>{memory_str}</td>
                        <td>{latency_str}</td>
                        <td>{risk_str}</td>
                    </tr>
"""
            html += """                </tbody>
            </table>
        </div>
"""
    
    # Add benchmark results with model names and code examples
    if 'benchmark_results' in report:
        html += """
        <div class="card">
            <h2>Benchmark Results - Selected Models & Code Examples</h2>
            <p class="info-text">Final selected model for each AutoML system in the benchmark with code examples.</p>
            <table>
                <tr><th>System</th><th>Dataset</th><th>Score</th><th>F1</th><th>Time (s)</th><th>Latency (ms)</th><th>Selected Model</th></tr>
"""
        for result in report['benchmark_results']:
            system = result.get('system', 'N/A')
            dataset = result.get('dataset', 'N/A')
            score = result.get('test_score', 0) or 0
            f1 = result.get('test_f1', None)
            time_val = result.get('fit_time_s', 0) or 0
            latency = result.get('latency_ms_per_row', 0) or 0
            model_name = result.get('model_name', 'N/A')
            score_str = _fmt_num(score, ".4f")
            f1_str = _fmt_num(f1, ".4f") if f1 is not None else "N/A"
            time_str = _fmt_num(time_val, ".2f")
            latency_str = _fmt_num(latency, ".3f")
            html += f"                <tr><td>{_esc(system)}</td><td>{_esc(dataset)}</td><td>{score_str}</td><td>{f1_str}</td><td>{time_str}</td><td>{latency_str}</td><td class='model-name'>{_esc(model_name)}</td></tr>\n"
        html += "            </table>\n"
        
        # Add actual execution code for each framework
        html += """
            <h3>Actual Execution Code for Each Framework</h3>
            <p class="info-text">The following code blocks show the actual code that was executed during benchmarking, with real file paths, parameters, and results.</p>
"""
        for result in report['benchmark_results']:
            system = result.get('system', 'N/A')
            model_name = result.get('model_name', 'N/A')
            code_info = result.get('code_info', {})
            if code_info and isinstance(code_info, dict):
                code = code_info.get('code', '')
                execution_summary = code_info.get('execution_summary', '')
                if code:
                    html += f"""
            <div style="margin-bottom: 20px; border: 1px solid var(--border); border-radius: 8px; padding: 15px; background: var(--card-bg);">
                <h4 style="margin-top: 0; color: var(--primary);">{_esc(system)} - {_esc(model_name)}</h4>
                <p style="color: var(--success); font-weight: bold; margin-bottom: 10px;">✓ {_esc(execution_summary)}</p>
                <pre style="background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px; overflow-x: auto; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; margin: 0;"><code>{_esc(code)}</code></pre>
            </div>
"""
        html += "        </div>\n"
    
    # Add estimators with model names
    if 'estimators' in report and report['estimators']:
        html += """
        <div class="card">
            <h2>Estimators and Selected Models</h2>
            <table>
                <tr><th>#</th><th>Type</th><th>Selected Model Name</th></tr>
"""
        for i, est in enumerate(report['estimators'], 1):
            est_type = est.get('type', 'Unknown')
            model_name = est.get('selected_model_name', 'N/A')
            html += f"                <tr><td>{i}</td><td>{_esc(est_type)}</td><td class='model-name'>{_esc(model_name)}</td></tr>\n"
        html += "            </table>\n        </div>\n"
    
    # Add EDA HTML section if available
    if 'eda_html' in report and report['eda_html']:
        html += """
        <div class="card">
            <h2>Exploratory Data Analysis (EDA)</h2>
            <p class="info-text">Complete EDA report with data quality analysis, statistics, and visualizations.</p>
            <div style="border: 1px solid #ddd; border-radius: 4px; padding: 10px; max-height: 800px; overflow-y: auto;">
"""
        html += _sanitize_embedded_html(report['eda_html'])
        html += """
            </div>
        </div>
"""
    
    # Add Time Series HTML section if available
    if 'timeseries_html' in report and report['timeseries_html']:
        html += """
        <div class="card">
            <h2>Time Series Forecasting Results</h2>
            <p class="info-text">Time series analysis, forecasting models, and benchmark comparisons.</p>
            <div style="border: 1px solid #ddd; border-radius: 4px; padding: 10px; max-height: 800px; overflow-y: auto;">
"""
        html += _sanitize_embedded_html(report['timeseries_html'])
        html += """
            </div>
        </div>
"""
    
    # Add Time Series summary if available
    if 'timeseries' in report:
        ts = report['timeseries']
        html += """
        <div class="card">
            <h2>Time Series Summary</h2>
            <table>
                <tr><th>Forecast Horizon</th><td>{}</td></tr>
                <tr><th>Datetime Column</th><td>{}</td></tr>
                <tr><th>Target Column</th><td>{}</td></tr>
""".format(
            _esc(ts.get('horizon', 'N/A')),
            _esc(ts.get('datetime_col', 'N/A')),
            _esc(ts.get('target', 'N/A'))
        )
        if 'best_model' in ts:
            html += f"                <tr><th>Best Model</th><td class='model-name'>{_esc(ts.get('best_model', 'N/A'))}</td></tr>\n"
        html += "            </table>\n"
        
        # Display actual execution code for time series
        code_info = ts.get('code_info', {})
        if code_info and isinstance(code_info, dict):
            code = code_info.get('code', '')
            execution_summary = code_info.get('execution_summary', '')
            if code:
                html += f"""
            <h3>Actual Execution Code</h3>
            <p class="info-text" style="color: var(--success); font-weight: bold;">✓ {_esc(execution_summary)}</p>
            <pre style="background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px; overflow-x: auto; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px;"><code>{_esc(code)}</code></pre>
"""
        if 'leaderboard' in ts and ts['leaderboard']:
            html += "            <h3>Model Leaderboard</h3>\n            <table>\n"
            html += "                <tr><th>Rank</th><th>Model</th><th>Score (RMSE)</th><th>Fit Time</th><th>Family</th></tr>\n"
            for i, row in enumerate(ts['leaderboard'][:10], 1):
                # Handle both dict formats: AutoTS uses 'Model', 'Score' and older format uses 'model', 'rmse'
                model_name = row.get('Model', row.get('model', row.get('name', 'Unknown')))
                score_val = row.get('Score', row.get('rmse', row.get('metric', 'N/A')))
                fit_time = row.get('Fit Time (s)', row.get('fit_time', 'N/A'))
                family = row.get('Family', row.get('family', 'N/A'))
                # Format numeric values, keep strings as-is
                score_str = _fmt_num(score_val, ".4f")
                time_str = f"{_fmt_num(fit_time, '.2f')}s" if (isinstance(fit_time, (int, float)) and not isinstance(fit_time, bool) and math.isfinite(float(fit_time))) else _esc(fit_time)
                html += f"                <tr><td>{i}</td><td class='model-name'>{_esc(model_name)}</td><td>{score_str}</td><td>{time_str}</td><td>{_esc(family)}</td></tr>\n"
            html += "            </table>\n"
        html += "        </div>\n"
    
    # Add Time Series Benchmark Code section if available
    if 'timeseries_benchmark_code' in report and report['timeseries_benchmark_code']:
        html += """
        <div class="card">
            <h2>Time Series Benchmark - Actual Execution Code</h2>
            <p class="info-text">The following code blocks show the actual code executed for each time series forecasting system during benchmarking.</p>
"""
        for ts_code_info in report['timeseries_benchmark_code']:
            if ts_code_info and isinstance(ts_code_info, dict):
                system = ts_code_info.get('system', 'Unknown')
                code = ts_code_info.get('code', '')
                execution_summary = ts_code_info.get('execution_summary', '')
                if code:
                    html += f"""
            <div style="margin-bottom: 20px; border: 1px solid var(--border); border-radius: 8px; padding: 15px; background: var(--card-bg);">
                <h4 style="margin-top: 0; color: var(--primary);">{_esc(system)}</h4>
                <p style="color: var(--success); font-weight: bold; margin-bottom: 10px;">✓ {_esc(execution_summary)}</p>
                <pre style="background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px; overflow-x: auto; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; margin: 0;"><code>{_esc(code)}</code></pre>
            </div>
"""
        html += "        </div>\n"
    
    # Add predictions section if available
    if isinstance(report.get('predictions'), dict) and report['predictions']:
        pred = report['predictions']
        html += """
        <div class="card">
            <h2>Predictions on Full Dataset</h2>
            <p class="info-text">Predictions made on the complete dataset using the best trained model.</p>
            <table>
                <tr><th>Target Column</th><td class="model-name">{}</td></tr>
                <tr><th>Total Predictions</th><td>{}</td></tr>
                <tr><th>Task Type</th><td>{}</td></tr>
                <tr><th>Output File</th><td><code>{}</code></td></tr>
            </table>
""".format(
            _esc(pred.get('target_column', 'N/A')),
            _esc(pred.get('total_predictions', 0)),
            _esc(pred.get('task', 'N/A')),
            _esc(pred.get('output_file', 'N/A'))
        )
        
        # Add statistics
        if isinstance(pred.get('statistics'), dict) and pred['statistics']:
            stats = pred['statistics']
            html += "            <h3>Prediction Statistics</h3>\n            <table>\n"
            if 'mean' in stats and stats['mean'] is not None:
                html += f"                <tr><th>Mean</th><td>{_fmt_num(stats['mean'])}</td></tr>\n"
            if 'std' in stats and stats['std'] is not None:
                html += f"                <tr><th>Std Dev</th><td>{_fmt_num(stats['std'])}</td></tr>\n"
            if 'min' in stats and stats['min'] is not None:
                html += f"                <tr><th>Min</th><td>{_fmt_num(stats['min'])}</td></tr>\n"
            if 'max' in stats and stats['max'] is not None:
                html += f"                <tr><th>Max</th><td>{_fmt_num(stats['max'])}</td></tr>\n"
            if 'distribution' in stats and stats['distribution']:
                html += "                <tr><th>Distribution</th><td>\n"
                for class_val, count in stats['distribution'].items():
                    html += f"                    Class {_esc(class_val)}: {_esc(count)}<br>\n"
                html += "                </td></tr>\n"
            html += "            </table>\n"
        
        # Add sample of actual predicted values
        if 'sample_values' in pred and pred['sample_values']:
            samples = pred['sample_values']
            html += """
            <h3>Sample Predicted Values</h3>
            <p class="info-text">First 20 rows and last 5 rows of predictions (total: {} predictions)</p>
            <table>
                <tr><th>Row</th><th>Predicted Value</th><th>Actual Value</th></tr>
""".format(_esc(pred.get('total_predictions', len(samples))))
            for sample in samples:
                row_idx = sample.get('row', 'N/A')
                pred_val = sample.get('predicted', 'N/A')
                actual_val = sample.get('actual', 'N/A')
                # Format values
                pred_str = _fmt_num(pred_val, ".4f")
                actual_str = _fmt_num(actual_val, ".4f") if actual_val is not None else "N/A"
                html += f"                <tr><td>{_esc(row_idx)}</td><td>{pred_str}</td><td>{actual_str}</td></tr>\n"
            html += "            </table>\n"
        
        html += "        </div>\n"
    
    # Add console output
    if 'console_output' in report and report['console_output']:
        console_lines = report['console_output'].split('\n')
        html += f"""
        <div class="card">
            <h2>Console Output ({len(console_lines)} lines)</h2>
            <div class="console-output">{_esc(report['console_output'][:50000])}</div>
        </div>
"""
    
    # Add summary
    if isinstance(report.get('summary'), dict) and report['summary']:
        summary = report['summary']
        html += """
        <div class="card">
            <h2>Summary</h2>
            <div class="summary-grid">
"""
        for key, value in summary.items():
            # Format numeric values properly
            if isinstance(value, float):
                if not math.isfinite(value):
                    formatted_value = "N/A"
                elif 'time' in key.lower():
                    formatted_value = f"{value:.1f}s"
                elif 'rate' in key.lower() or 'percent' in key.lower():
                    formatted_value = f"{value:.1f}%"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int) and not isinstance(value, bool):
                formatted_value = str(value)
            else:
                formatted_value = _esc(value)
            
            html += f"                <div class='summary-item'><div class='summary-value'>{formatted_value}</div><div class='summary-label'>{_esc(key.replace('_', ' ').title())}</div></div>\n"
        html += "            </div>\n        </div>\n"
    
    # Add Files Generated section - comprehensive list of all output files
    output_files = []
    
    # Collect all file paths from report
    if report.get('html_report_path'):
        output_files.append(("📄 Combined HTML Report", os.path.abspath(str(report.get('html_report_path', '')))))
    
    if report.get('model_path'):
        output_files.append(("🤖 Trained Model (.pkl)", os.path.abspath(str(report.get('model_path', '')))))
    
    if report.get('audit_path'):
        output_files.append(("📋 Audit Trail (.json)", os.path.abspath(str(report.get('audit_path', '')))))
    
    if isinstance(report.get('predictions'), dict) and report['predictions'].get('output_file'):
        output_files.append(("📊 Predictions CSV", os.path.abspath(str(report['predictions'].get('output_file', '')))))
    
    if report.get('eda_report_path'):
        output_files.append(("📈 EDA Report (embedded)", str(report.get('eda_report_path', 'Embedded in this report'))))
    
    if report.get('timeseries_report_path'):
        output_files.append(("📉 Time Series Report (embedded)", str(report.get('timeseries_report_path', 'Embedded in this report'))))
    
    if report.get('benchmark_report_path'):
        output_files.append(("⚡ Benchmark Report (embedded)", str(report.get('benchmark_report_path', 'Embedded in this report'))))
    
    # Include any additional files from configuration
    config = report.get('configuration') if isinstance(report.get('configuration'), dict) else {}
    if config.get('input_file'):
        output_files.append(("📁 Input Dataset", str(config.get('input_file', ''))))
    
    if output_files:
        html += """
        <div class="card">
            <h2>📁 Files Generated</h2>
            <p class="info-text">All files generated during this analysis session. Click a path to open/download it, or use the copy button.</p>
            <table>
                <tr><th>File Type</th><th>Path</th></tr>
"""
        for file_type, file_path in output_files:
            if file_path and file_path != 'None':
                # Create copyable path with data-path
                html += f"""                <tr>
                    <td>{_esc(file_type)}</td>
                    <td>{_file_link_cell(file_path)}</td>
                </tr>
"""
        html += "            </table>\n        </div>\n"
    
    # Close HTML
    html += """
    </div>
</body>
</html>
"""
    
    # Save file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return str(output_path)


def _generate_multi_target_html_report(results: Dict[str, Any], base_name: str) -> str:
    """
    Generate a comprehensive HTML report for multi-target training + predictions.
    
    This combines all individual target reports into a single comprehensive report
    that includes EDA, Time Series, AutoML, Benchmarks, and Predictions for all targets.
    """
    import os
    from datetime import datetime
    from urllib.parse import quote
    
    targets = results.get("targets", [])
    target_reports = results.get("target_reports", {})
    predictions = results.get("predictions", {})

    total_predictions = 0
    for p in predictions.values():
        if isinstance(p, dict):
            c = p.get("count", 0)
            if hasattr(c, "item") and not isinstance(c, (str, bytes, bytearray)):
                try:
                    c = c.item()
                except Exception:
                    pass
            if isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(float(c)):
                total_predictions += int(c)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%232563eb'/%3E%3Ctext x='32' y='45' font-family='Arial,sans-serif' font-size='32' font-weight='700' fill='%23ffffff' text-anchor='middle'%3EA%3C/text%3E%3C/svg%3E">
    <title>ACDSO Multi-Target Report - {_esc(base_name)}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-light: #3b82f6;
            --secondary: #8b5cf6;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --info: #0891b2;
            --bg: #0f172a;
            --bg-card: #1e293b;
            --bg-highlight: #334155;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --border: #334155;
            --dark: #0f172a;
            --light: #f7fafc;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, var(--bg) 0%, #1a1a2e 100%);
            min-height: 100vh;
            padding: 2rem;
            color: var(--text);
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(139, 92, 246, 0.1));
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            text-align: center;
            border: 1px solid var(--border);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }}
        
        .header .subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}
        
        .targets-overview {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .target-card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            border-left: 4px solid var(--primary);
            border: 1px solid var(--border);
        }}
        
        .target-card h3 {{
            color: var(--primary-light);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .target-card .stat {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
            color: var(--text);
        }}
        
        .target-card .stat:last-child {{
            border-bottom: none;
        }}
        
        .section {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            border: 1px solid var(--border);
        }}
        
        .section h2 {{
            color: var(--primary-light);
            border-bottom: 3px solid var(--primary);
            padding-bottom: 0.75rem;
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
        }}
        
        .tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }}
        
        .tab {{
            padding: 0.75rem 1.5rem;
            background: var(--light);
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            color: var(--dark);
            transition: all 0.3s;
        }}
        
        .tab:hover, .tab.active {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: var(--light);
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        
        .badge-success {{ background: #c6f6d5; color: #276749; }}
        .badge-warning {{ background: #feebc8; color: #c05621; }}
        .badge-primary {{ background: #e9d8fd; color: #553c9a; }}
        
        .predictions-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }}
        
        .prediction-box {{
            background: var(--light);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border);
        }}
        
        .prediction-box h4 {{
            color: var(--primary);
            margin-bottom: 1rem;
        }}
        
        .embedded-report {{
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            margin-top: 1rem;
        }}
        
        .embedded-report iframe {{
            width: 100%;
            height: 600px;
            border: none;
        }}
        
        .file-link {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--light);
            border-radius: 8px;
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
            margin: 0.25rem;
        }}
        
        .file-link:hover {{
            background: var(--primary);
            color: white;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .stat-box {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 1.25rem;
            border-radius: 12px;
            text-align: center;
        }}
        
        .stat-box .value {{
            font-size: 2rem;
            font-weight: 700;
        }}
        
        .stat-box .label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        
        .code-block {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9rem;
            margin: 1rem 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 ACDSO Multi-Target Report</h1>
            <p class="subtitle">{_esc(base_name)} | {len(targets)} Targets | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary-stats">
            <div class="stat-box">
                <div class="value">{len(targets)}</div>
                <div class="label">Targets</div>
            </div>
            <div class="stat-box">
                <div class="value">{len(predictions)}</div>
                <div class="label">Predictions Made</div>
            </div>
            <div class="stat-box">
                <div class="value">{_fmt_int(total_predictions)}</div>
                <div class="label">Total Predictions</div>
            </div>
            <div class="stat-box">
                <div class="value">{len(target_reports)}</div>
                <div class="label">Full Reports</div>
            </div>
        </div>
        
        <div class="targets-overview">
"""
    
    # Add target cards
    for target in targets:
        pred = predictions.get(target, {})
        stats = pred.get("statistics", {})
        
        html += f"""
            <div class="target-card">
                <h3>🎯 {_esc(target)}</h3>
"""
        if "mean" in stats:
            html += f"""
                <div class="stat"><span>Mean:</span><span>{_fmt_num(stats.get('mean'))}</span></div>
                <div class="stat"><span>Std:</span><span>{_fmt_num(stats.get('std'))}</span></div>
                <div class="stat"><span>Min:</span><span>{_fmt_num(stats.get('min'))}</span></div>
                <div class="stat"><span>Max:</span><span>{_fmt_num(stats.get('max'))}</span></div>
"""
        elif "distribution" in stats:
            dist = stats.get("distribution", {})
            for k, v in list(dist.items())[:5]:
                html += f"""
                <div class="stat"><span>Class {_esc(k)}:</span><span>{_esc(v)}</span></div>
"""
        pred_count = pred.get('count', 0)
        pred_count_str = _fmt_int(pred_count)
        html += f"""
                <div class="stat"><span>Predictions:</span><span>{pred_count_str}</span></div>
                <div class="stat"><span>Output:</span><span style="font-size:0.8em;">{_esc(os.path.basename(pred.get('output_file', 'N/A')))}</span></div>
            </div>
"""
    
    html += """
        </div>
        
        <div class="section">
            <h2>📊 Detailed Target Reports</h2>
            <div class="tabs">
"""
    
    # Add tabs for each target
    for i, target in enumerate(targets):
        active = "active" if i == 0 else ""
        html += f"""                <button type="button" class="tab {active}" data-tab="tab-panel-{i}" onclick="showTab(this.dataset.tab, this)">{_esc(target)}</button>
"""
    
    html += """            </div>
"""
    
    # Add tab content for each target
    for i, target in enumerate(targets):
        active = "active" if i == 0 else ""
        report = target_reports.get(target, {}) or {}
        
        html += f"""
            <div id="tab-panel-{i}" class="tab-content {active}">
                <h3>🎯 Target: {_esc(target)}</h3>
"""
        
        # Selected Model (from the actual combined report structure)
        selected_model = report.get("selected_model") or {}
        if isinstance(selected_model, dict) and selected_model:
            model_name = selected_model.get('name', 'N/A')
            metrics = selected_model.get('metrics') or {}
            html += f"""
                <h4>🤖 Selected Model</h4>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Model Name</td><td><span class="badge badge-primary">{_esc(model_name)}</span></td></tr>
"""
            for key, value in metrics.items():
                if value is not None:
                    html += f"                    <tr><td>{_esc(key)}</td><td>{_fmt_num(value)}</td></tr>\n"
            html += """                </table>
"""
        
        # All Models Evaluated (from audit_trail.candidates_evaluated)
        audit_trail = report.get("audit_trail") or {}
        candidates = audit_trail.get("candidates_evaluated", []) if isinstance(audit_trail, dict) else []
        if candidates:
            html += """
                <h4>📋 All Models Evaluated</h4>
                <table>
                    <tr><th>#</th><th>Model</th><th>CV Score</th><th>CV Std</th><th>Time (s)</th><th>Memory (MB)</th><th>Latency (ms/row)</th><th>Risk</th></tr>
"""
            for idx, c in enumerate(candidates, 1):
                name = c.get('name', 'N/A')
                metric = c.get('metric', 0) or 0
                cv_std = c.get('cv_score_std', 0) or 0
                train_time = c.get('train_time_s', 0) or 0
                memory = c.get('peak_mem_mb', 0) or 0
                latency = c.get('latency_ms_per_row', 0) or 0
                risk = c.get('risk_penalty', 0) or 0
                
                # Format values safely
                metric_str = _fmt_num(metric, ".4f")
                cv_std_str = _fmt_num(cv_std, ".4f")
                train_time_str = _fmt_num(train_time, ".2f")
                memory_str = _fmt_num(memory, ".1f")
                latency_str = _fmt_num(latency, ".3f")
                risk_str = _fmt_num(risk, ".4f")
                
                html += f"""
                    <tr>
                        <td>{idx}</td>
                        <td>{_esc(name)}</td>
                        <td>{metric_str}</td>
                        <td>{cv_std_str}</td>
                        <td>{train_time_str}</td>
                        <td>{memory_str}</td>
                        <td>{latency_str}</td>
                        <td>{risk_str}</td>
                    </tr>
"""
            html += """                </table>
"""
        
        # Pareto Front Models
        pareto_front = audit_trail.get("pareto_front", []) if isinstance(audit_trail, dict) else []
        if pareto_front:
            html += """
                <h4>⚖️ Pareto Front Models (Optimal Trade-offs)</h4>
                <table>
                    <tr><th>Model</th><th>Score</th><th>Time (s)</th><th>Memory (MB)</th><th>Latency (ms/row)</th></tr>
"""
            for p in pareto_front:
                name = p.get('name', 'N/A')
                metric = p.get('metric', 0) or 0
                train_time = p.get('train_time_s', 0) or 0
                memory = p.get('peak_mem_mb', 0) or 0
                latency = p.get('latency_ms_per_row', 0) or 0
                
                metric_str = _fmt_num(metric, ".4f")
                train_time_str = _fmt_num(train_time, ".2f")
                memory_str = _fmt_num(memory, ".1f")
                latency_str = _fmt_num(latency, ".3f")
                
                html += f"""
                    <tr>
                        <td>{_esc(name)}</td>
                        <td>{metric_str}</td>
                        <td>{train_time_str}</td>
                        <td>{memory_str}</td>
                        <td>{latency_str}</td>
                    </tr>
"""
            html += """                </table>
"""
        
        # Time Series results
        ts = report.get("timeseries", {})
        ts_best = report.get("timeseries_best", {})
        if ts or ts_best:
            best_model = ts_best.get('best_model', ts.get('best_model', 'N/A'))
            rmse = ts_best.get('rmse', ts.get('cv_score', 'N/A'))
            html += f"""
                <h4>📈 Time Series Results</h4>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Best Model</td><td><span class="badge badge-success">{_esc(best_model)}</span></td></tr>
                    <tr><td>RMSE</td><td>{_fmt_num(rmse, ".4f")}</td></tr>
                </table>
"""
        
        # Time Series Leaderboard
        ts_leaderboard = report.get("timeseries_leaderboard", [])
        if ts_leaderboard:
            html += """
                <h4>📊 Time Series Leaderboard</h4>
                <table>
                    <tr><th>System</th><th>RMSE</th><th>MAE</th><th>Time (s)</th></tr>
"""
            for row in ts_leaderboard:
                if isinstance(row, dict):
                    html += f"""
                    <tr>
                        <td>{_esc(row.get('System', row.get('system', 'N/A')))}</td>
                        <td>{_fmt_num(row.get('RMSE', row.get('rmse', 'N/A')), ".4f")}</td>
                        <td>{_fmt_num(row.get('MAE', row.get('mae', 'N/A')), ".4f")}</td>
                        <td>{_fmt_num(row.get('Time(s)', row.get('time', 'N/A')), ".2f")}</td>
                    </tr>
"""
            html += """                </table>
"""
        
        # Benchmark results
        bench_results = report.get("benchmark_results", [])
        if bench_results:
            html += """
                <h4>🏆 Benchmark Comparison</h4>
                <table>
                    <tr><th>System</th><th>Score</th><th>Time (s)</th><th>Latency (ms)</th><th>Model</th></tr>
"""
            for res in bench_results:
                system = res.get('system', 'N/A')
                raw_score = res.get('test_score')
                score = 0 if raw_score is None else raw_score
                raw_time = res.get('fit_time_s')
                time_val = 0 if raw_time is None else raw_time
                raw_lat = res.get('latency_ms_per_row')
                latency = 0 if raw_lat is None else raw_lat
                model_name = res.get('model_name', 'N/A')
                
                score_str = _fmt_num(score, ".4f")
                time_str = _fmt_num(time_val, ".2f")
                latency_str = _fmt_num(latency, ".3f")
                
                html += f"""
                    <tr>
                        <td>{_esc(system)}</td>
                        <td>{score_str}</td>
                        <td>{time_str}</td>
                        <td>{latency_str}</td>
                        <td>{_esc(model_name)}</td>
                    </tr>
"""
            html += """                </table>
"""
        
        # EDA summary
        eda_report = report.get("eda_report", {})
        if eda_report:
            quality_score = eda_report.get('quality_score', 'N/A')
            n_rows = eda_report.get('n_rows', 'N/A')
            n_cols = eda_report.get('n_cols', 'N/A')
            n_rows_str = _fmt_int(n_rows)
            n_cols_str = _fmt_int(n_cols)
            quality_str = _fmt_num(quality_score, "g")
            html += f"""
                <h4>🔍 EDA Summary</h4>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Quality Score</td><td><span class="badge badge-success">{quality_str}/100</span></td></tr>
                    <tr><td>Rows</td><td>{n_rows_str}</td></tr>
                    <tr><td>Columns</td><td>{n_cols_str}</td></tr>
                </table>
"""
        
        # Embedded EDA HTML (if available)
        eda_html = report.get("eda_html", "")
        if eda_html:
            html += f"""
                <h4>📊 Full EDA Report</h4>
                <div style="border: 1px solid var(--border); border-radius: 8px; padding: 10px; max-height: 600px; overflow-y: auto; background: white;">
                    {_sanitize_embedded_html(eda_html)}
                </div>
"""
        
        # Phase Results
        phase_results = report.get("phase_results", {})
        if phase_results:
            html += """
                <h4>📋 Phase Results</h4>
                <table>
                    <tr><th>Phase</th><th>Status</th><th>Model</th><th>Time (s)</th></tr>
"""
            for phase_name, items in phase_results.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            name = item.get('name', 'N/A')
                            status = item.get('status', 'N/A')
                            model = item.get('model_name', 'N/A') or 'N/A'
                            raw_time = item.get('time')
                            time_val = 0 if raw_time is None else raw_time
                            status_class = 'badge-success' if 'SUCCESS' in str(status) else 'badge-warning'
                            time_str = _fmt_num(time_val, ".2f")
                            status_str = str(status)
                            status_disp = status_str[:50] if len(status_str) > 50 else status_str
                            html += f"""
                    <tr>
                        <td>{_esc(name)}</td>
                        <td><span class="badge {status_class}">{_esc(status_disp)}</span></td>
                        <td>{_esc(model)}</td>
                        <td>{time_str}</td>
                    </tr>
"""
            html += """                </table>
"""
        
        html += """
            </div>
"""
    
    html += """
        </div>
        
        <div class="section">
            <h2>📁 Predictions Files</h2>
            <div class="predictions-grid">
"""
    
    for target in targets:
        pred = predictions.get(target, {})
        pred_file = pred.get('output_file')
        pred_href = _esc(_file_uri(pred_file)) if pred_file else '#'
        pred_count = pred.get('count', 0)
        pred_count_str = _fmt_int(pred_count)
        html += f"""
                <div class="prediction-box">
                    <h4>🎯 {_esc(target)}</h4>
                    <p>Predictions: <strong>{pred_count_str}</strong></p>
                    <a href="{pred_href}" class="file-link">📥 Download CSV</a>
                </div>
"""
    
    html += """
            </div>
        </div>
        
        <div class="section">
            <h2>📜 Individual Report Links</h2>
            <p>Each target has its own detailed autonomous report with full EDA, Time Series, AutoML, and Benchmark analysis:</p>
            <div style="margin-top: 1rem;">
"""
    
    for target in targets:
        ind_href = _esc(quote(f"autonomous_{base_name}_{target}_report.html", safe="/:._-"))
        html += f"""
                <a href="{ind_href}" class="file-link">📊 {_esc(target)} Report</a>
"""
    
    html += """
            </div>
        </div>
        
        <div class="section">
            <h2>📁 All Files Generated</h2>
            <p style="color: var(--text-muted); margin-bottom: 1rem;">Click a path to open/download it, or use the copy button.</p>
            <table>
                <tr><th>Type</th><th>Target</th><th>File Path</th></tr>
"""
    
    # Add model files
    for target in targets:
        report = target_reports.get(target, {})
        model_path = report.get("model_path", f"reports/model_{base_name}_{target}.pkl")
        abs_path = os.path.abspath(model_path) if model_path else "N/A"
        html += f"""                <tr>
                    <td>🤖 Model</td>
                    <td>{_esc(target)}</td>
                    <td>{_file_link_cell(abs_path)}</td>
                </tr>
"""

    # Add prediction CSV files
    for target in targets:
        pred = predictions.get(target, {})
        output_file = pred.get('output_file', f"reports/predictions_{base_name}_{target}_full_dataset.csv")
        abs_path = os.path.abspath(output_file) if output_file else "N/A"
        html += f"""                <tr>
                    <td>📊 Predictions CSV</td>
                    <td>{_esc(target)}</td>
                    <td>{_file_link_cell(abs_path)}</td>
                </tr>
"""

    # Add individual report links
    for target in targets:
        report_path = f"reports/autonomous_{base_name}_{target}_report.html"
        abs_path = os.path.abspath(report_path)
        html += f"""                <tr>
                    <td>📄 Individual Report</td>
                    <td>{_esc(target)}</td>
                    <td>{_file_link_cell(abs_path)}</td>
                </tr>
"""

    # Add combined report
    combined_html_path = f"reports/multi_target_{base_name}_combined_report.html"
    abs_combined = os.path.abspath(combined_html_path)
    combined_json_path = f"reports/multi_target_{base_name}_combined.json"
    abs_json = os.path.abspath(combined_json_path)
    
    html += f"""                <tr>
                    <td>📄 Combined HTML Report</td>
                    <td>All Targets</td>
                    <td>{_file_link_cell(abs_combined)}</td>
                </tr>
                <tr>
                    <td>📋 Combined JSON Data</td>
                    <td>All Targets</td>
                    <td>{_file_link_cell(abs_json)}</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        function showTab(panelId, btn) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            
            // Show selected tab
            const panel = document.getElementById(panelId);
            if (panel) panel.classList.add('active');
            if (btn) btn.classList.add('active');
        }}
    </script>
</body>
</html>
"""
    
    return html


