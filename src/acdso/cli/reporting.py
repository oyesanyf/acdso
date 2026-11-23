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
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_payload(v) for v in obj]
    if isinstance(obj, str) or obj is None:
        return obj
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



def generate_combined_report(
    estimator: Any,
    console_output: str,
    test_predictions: Optional[Any] = None,
    test_targets: Optional[Any] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive combined report that includes everything shown in console
    plus all structured data from ACDSO.
    
    Parameters
    ----------
    estimator : ACDSOClassifier or ACDSORegressor
        Fitted ACDSO estimator.
    console_output : str
        Captured console output text.
    test_predictions : array-like, optional
        Test set predictions for additional metrics.
    test_targets : array-like, optional
        Test set true targets for additional metrics.
    additional_info : dict, optional
        Additional information to include in the report.
    
    Returns
    -------
    dict
        Combined report with all information.
    """
    np, pd, accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error = _load_metrics_deps()
    report = {
        "timestamp": datetime.now().isoformat(),
        "acdso_version": "3.0",
        "console_output": console_output,
    }
    
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
    if hasattr(estimator, 'audit_trail_') and estimator.audit_trail_ is not None:
        report["audit_trail"] = estimator.audit_trail_
    
    # Add Pareto front
    if hasattr(estimator, 'pareto_front_') and estimator.pareto_front_ is not None:
        report["pareto_front"] = estimator.pareto_front_
    
    # Add selected model information
    if hasattr(estimator, 'best_estimator_') and estimator.best_estimator_ is not None:
        report["selected_model"] = {
            "name": estimator.audit_trail_.get('selected', {}).get('name', 'Unknown') if hasattr(estimator, 'audit_trail_') and estimator.audit_trail_ else 'Unknown',
            "type": type(estimator.best_estimator_).__name__,
            "pipeline_steps": [step[0] for step in estimator.best_estimator_.steps] if hasattr(estimator.best_estimator_, 'steps') else None,
        }
        
        # Add detailed metrics from audit trail
        if hasattr(estimator, 'audit_trail_') and estimator.audit_trail_:
            selected_info = estimator.audit_trail_.get('selected', {})
            report["selected_model"]["metrics"] = {
                "metric": selected_info.get('metric'),
                "train_time_s": selected_info.get('train_time_s'),
                "peak_mem_mb": selected_info.get('peak_mem_mb'),
                "latency_ms_per_row": selected_info.get('latency_ms_per_row'),
                "risk_penalty": selected_info.get('risk_penalty'),
                "stress_worst_drop": selected_info.get('stress_worst_drop'),
            }
    
    # Add test set performance if provided
    if test_predictions is not None and test_targets is not None:
        try:
            if hasattr(estimator, 'predict_proba'):
                # Classification
                test_acc = accuracy_score(test_targets, test_predictions)
                test_f1 = f1_score(test_targets, test_predictions, average='weighted')
                report["test_performance"] = {
                    "accuracy": float(test_acc),
                    "f1_weighted": float(test_f1),
                }
            else:
                # Regression
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
    
    # Add additional information if provided
    if additional_info:
        report["additional_info"] = additional_info
    
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
    timestamp = report.get('timestamp', 'N/A')
    acdso_version = report.get('acdso_version', '3.0')
    
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
            <div class="subtitle">Mode: {_esc(str(mode).title())}</div>
            <div class="subtitle">Generated: {_esc(timestamp)}</div>
        </div>
"""
    
    # Add configuration section
    if 'configuration' in report:
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
    if 'selected_model' in report:
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
        if 'metrics' in model:
            metrics = model['metrics']
            html += "                <tr><th colspan='2'><h3>Metrics</h3></th></tr>\n"
            for key, value in metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        html += f"                <tr><th>{_esc(key)}</th><td>{value:.4f}</td></tr>\n"
                    else:
                        html += f"                <tr><th>{_esc(key)}</th><td>{_esc(value)}</td></tr>\n"
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
    
    # Add phase results with model names (for autonomous mode)
    if 'phase_results' in report:
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
                        time_str = f"{time_val:.2f}" if isinstance(time_val, (int, float)) else _esc(time_val)
                        html += f"                <tr><td>{_esc(name)}</td><td class='{status_class}'>{_esc(status)}</td><td class='model-name'>{_esc(model_name)}</td><td>{time_str}</td></tr>\n"
                html += "            </table>\n"
        html += "        </div>\n"
    
    # Add audit trail with all candidates evaluated (ACDSO)
    if 'audit_trail' in report and report['audit_trail']:
        audit = report['audit_trail']
        candidates = audit.get('candidates_evaluated', [])
        if candidates:
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
                name = candidate.get('name', 'Unknown')
                metric = candidate.get('metric', 0) or 0
                cv_std = candidate.get('cv_score_std', 0) or 0
                train_time = candidate.get('train_time_s', 0) or 0
                memory = candidate.get('peak_mem_mb', 0) or 0
                latency = candidate.get('latency_ms_per_row', 0) or 0
                risk_penalty = candidate.get('risk_penalty', 0) or 0
                stress_drop = candidate.get('stress_worst_drop', None)
                
                # Check if this is the selected model
                selected = audit.get('selected', {})
                is_selected = selected.get('name') == name
                row_class = 'selected-model' if is_selected else ''
                status = '✓ Selected' if is_selected else 'Evaluated'
                
                stress_str = f"{stress_drop:.4f}" if isinstance(stress_drop, (int, float)) else (_esc(stress_drop) if stress_drop is not None else "N/A")
                
                # Format values safely
                metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else _esc(metric)
                cv_std_str = f"{cv_std:.4f}" if isinstance(cv_std, (int, float)) else _esc(cv_std)
                train_time_str = f"{train_time:.2f}" if isinstance(train_time, (int, float)) else _esc(train_time)
                
                memory_str = f"{memory:.1f}" if isinstance(memory, (int, float)) else _esc(memory)
                latency_str = f"{latency:.3f}" if isinstance(latency, (int, float)) else _esc(latency)
                risk_str = f"{risk_penalty:.4f}" if isinstance(risk_penalty, (int, float)) else _esc(risk_penalty)
                
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
    
    # Add Pareto front models
    if 'audit_trail' in report and report['audit_trail']:
        audit = report['audit_trail']
        pareto_front = audit.get('pareto_front', [])
        if pareto_front:
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
                name = model.get('name', 'Unknown')
                metric = model.get('metric', 0) or 0
                train_time = model.get('train_time_s', 0) or 0
                memory = model.get('peak_mem_mb', 0) or 0
                latency = model.get('latency_ms_per_row', 0) or 0
                risk_penalty = model.get('risk_penalty', 0) or 0
                
                metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else _esc(metric)
                train_time_str = f"{train_time:.2f}" if isinstance(train_time, (int, float)) else _esc(train_time)
                memory_str = f"{memory:.1f}" if isinstance(memory, (int, float)) else _esc(memory)
                latency_str = f"{latency:.3f}" if isinstance(latency, (int, float)) else _esc(latency)
                risk_str = f"{risk_penalty:.4f}" if isinstance(risk_penalty, (int, float)) else _esc(risk_penalty)
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
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else _esc(score)
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else (_esc(f1) if f1 is not None else "N/A")
            time_str = f"{time_val:.2f}" if isinstance(time_val, (int, float)) else _esc(time_val)
            latency_str = f"{latency:.3f}" if isinstance(latency, (int, float)) else _esc(latency)
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
                score_str = f"{score_val:.4f}" if isinstance(score_val, (int, float)) else _esc(score_val)
                time_str = f"{fit_time:.2f}s" if isinstance(fit_time, (int, float)) else _esc(fit_time)
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
    if 'predictions' in report:
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
        if 'statistics' in pred:
            stats = pred['statistics']
            html += "            <h3>Prediction Statistics</h3>\n            <table>\n"
            if 'mean' in stats and stats['mean'] is not None:
                html += f"                <tr><th>Mean</th><td>{stats['mean']:.4f}</td></tr>\n"
            if 'std' in stats and stats['std'] is not None:
                html += f"                <tr><th>Std Dev</th><td>{stats['std']:.4f}</td></tr>\n"
            if 'min' in stats and stats['min'] is not None:
                html += f"                <tr><th>Min</th><td>{stats['min']:.4f}</td></tr>\n"
            if 'max' in stats and stats['max'] is not None:
                html += f"                <tr><th>Max</th><td>{stats['max']:.4f}</td></tr>\n"
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
                pred_str = f"{pred_val:.4f}" if isinstance(pred_val, (int, float)) else _esc(pred_val)
                actual_str = f"{actual_val:.4f}" if isinstance(actual_val, (int, float)) and actual_val is not None else (_esc(actual_val) if actual_val is not None else "N/A")
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
    if 'summary' in report:
        summary = report['summary']
        html += """
        <div class="card">
            <h2>Summary</h2>
            <div class="summary-grid">
"""
        for key, value in summary.items():
            # Format numeric values properly
            if isinstance(value, float):
                if 'time' in key.lower():
                    formatted_value = f"{value:.1f}s"
                elif 'rate' in key.lower() or 'percent' in key.lower():
                    formatted_value = f"{value:.1f}%"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = str(value)
            else:
                formatted_value = _esc(value)
            
            html += f"                <div class='summary-item'><div class='summary-value'>{formatted_value}</div><div class='summary-label'>{_esc(key.replace('_', ' ').title())}</div></div>\n"
        html += "            </div>\n        </div>\n"
    
    # Add Files Generated section - comprehensive list of all output files
    output_files = []
    
    # Collect all file paths from report
    if 'html_report_path' in report:
        output_files.append(("📄 Combined HTML Report", os.path.abspath(str(report.get('html_report_path', '')))))
    
    if 'model_path' in report:
        output_files.append(("🤖 Trained Model (.pkl)", os.path.abspath(str(report.get('model_path', '')))))
    
    if 'audit_path' in report:
        output_files.append(("📋 Audit Trail (.json)", os.path.abspath(str(report.get('audit_path', '')))))
    
    if 'predictions' in report and report['predictions'].get('output_file'):
        output_files.append(("📊 Predictions CSV", os.path.abspath(str(report['predictions'].get('output_file', '')))))
    
    if 'eda_report_path' in report:
        output_files.append(("📈 EDA Report (embedded)", str(report.get('eda_report_path', 'Embedded in this report'))))
    
    if 'timeseries_report_path' in report:
        output_files.append(("📉 Time Series Report (embedded)", str(report.get('timeseries_report_path', 'Embedded in this report'))))
    
    if 'benchmark_report_path' in report:
        output_files.append(("⚡ Benchmark Report (embedded)", str(report.get('benchmark_report_path', 'Embedded in this report'))))
    
    # Include any additional files from configuration
    config = report.get('configuration', {})
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
                <div class="value">{sum(p.get('count', 0) for p in predictions.values()):,}</div>
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
                <div class="stat"><span>Mean:</span><span>{stats.get('mean', 0):.4f}</span></div>
                <div class="stat"><span>Std:</span><span>{stats.get('std', 0):.4f}</span></div>
                <div class="stat"><span>Min:</span><span>{stats.get('min', 0):.4f}</span></div>
                <div class="stat"><span>Max:</span><span>{stats.get('max', 0):.4f}</span></div>
"""
        elif "distribution" in stats:
            dist = stats.get("distribution", {})
            for k, v in list(dist.items())[:5]:
                html += f"""
                <div class="stat"><span>Class {_esc(k)}:</span><span>{_esc(v)}</span></div>
"""
        pred_count = pred.get('count', 0)
        pred_count_str = f"{pred_count:,}" if isinstance(pred_count, (int, float)) else _esc(pred_count)
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
        report = target_reports.get(target, {})
        
        html += f"""
            <div id="tab-panel-{i}" class="tab-content {active}">
                <h3>🎯 Target: {_esc(target)}</h3>
"""
        
        # Selected Model (from the actual combined report structure)
        selected_model = report.get("selected_model", {})
        if selected_model:
            model_name = selected_model.get('name', 'N/A')
            metrics = selected_model.get('metrics', {})
            html += f"""
                <h4>🤖 Selected Model</h4>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Model Name</td><td><span class="badge badge-primary">{_esc(model_name)}</span></td></tr>
"""
            for key, value in metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        html += f"                    <tr><td>{_esc(key)}</td><td>{value:.4f}</td></tr>\n"
                    else:
                        html += f"                    <tr><td>{_esc(key)}</td><td>{_esc(value)}</td></tr>\n"
            html += """                </table>
"""
        
        # All Models Evaluated (from audit_trail.candidates_evaluated)
        audit_trail = report.get("audit_trail", {})
        candidates = audit_trail.get("candidates_evaluated", [])
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
                metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else _esc(metric)
                cv_std_str = f"{cv_std:.4f}" if isinstance(cv_std, (int, float)) else _esc(cv_std)
                train_time_str = f"{train_time:.2f}" if isinstance(train_time, (int, float)) else _esc(train_time)
                memory_str = f"{memory:.1f}" if isinstance(memory, (int, float)) else _esc(memory)
                latency_str = f"{latency:.3f}" if isinstance(latency, (int, float)) else _esc(latency)
                risk_str = f"{risk:.4f}" if isinstance(risk, (int, float)) else _esc(risk)
                
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
        pareto_front = audit_trail.get("pareto_front", [])
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
                
                metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else _esc(metric)
                train_time_str = f"{train_time:.2f}" if isinstance(train_time, (int, float)) else _esc(train_time)
                memory_str = f"{memory:.1f}" if isinstance(memory, (int, float)) else _esc(memory)
                latency_str = f"{latency:.3f}" if isinstance(latency, (int, float)) else _esc(latency)
                
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
                    <tr><td>RMSE</td><td>{_esc(rmse)}</td></tr>
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
                        <td>{_esc(row.get('RMSE', row.get('rmse', 'N/A')))}</td>
                        <td>{_esc(row.get('MAE', row.get('mae', 'N/A')))}</td>
                        <td>{_esc(row.get('Time(s)', row.get('time', 'N/A')))}</td>
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
                score = res.get('test_score', 0) or 0
                time_val = res.get('fit_time_s', 0) or 0
                latency = res.get('latency_ms_per_row', 0) or 0
                model_name = res.get('model_name', 'N/A')
                
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else _esc(score)
                time_str = f"{time_val:.2f}" if isinstance(time_val, (int, float)) else _esc(time_val)
                latency_str = f"{latency:.3f}" if isinstance(latency, (int, float)) else _esc(latency)
                
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
            n_rows_str = f"{n_rows:,}" if isinstance(n_rows, (int, float)) else _esc(n_rows)
            html += f"""
                <h4>🔍 EDA Summary</h4>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Quality Score</td><td><span class="badge badge-success">{_esc(quality_score)}/100</span></td></tr>
                    <tr><td>Rows</td><td>{n_rows_str}</td></tr>
                    <tr><td>Columns</td><td>{_esc(n_cols)}</td></tr>
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
                            time_val = item.get('time', 0) or 0
                            status_class = 'badge-success' if 'SUCCESS' in str(status) else 'badge-warning'
                            time_str = f"{time_val:.2f}" if isinstance(time_val, (int, float)) else _esc(time_val)
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
        pred_count_str = f"{pred_count:,}" if isinstance(pred_count, (int, float)) else _esc(pred_count)
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


