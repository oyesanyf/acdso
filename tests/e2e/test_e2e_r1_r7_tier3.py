"""Tier 3 E2E Tests: R1-R7 Cross-Feature Pairwise Integration Scenarios (12 tests)."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from .conftest import parse_progress_events, poll_job_until_done


class _TableBalanceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.open_tables = 0
        self.close_tables = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "table":
            self.open_tables += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "table":
            self.close_tables += 1


def _reject_nonfinite(val: str) -> Any:
    raise ValueError(f"Non-RFC-8259 JSON constant encountered: {val}")


def _loads_strict_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw, parse_constant=_reject_nonfinite)
    assert isinstance(data, dict), f"Expected JSON object in {path}, got {type(data)}"
    return data


@pytest.mark.tier3
@pytest.mark.f1
def test_r1_r4_t3_nlp_text_columns_into_clustering() -> None:
    """Pairwise R1 x R4: Mixed numeric + free-text descriptions clustered via ACDSOClusterer (TF-IDF + PCA)."""
    from sklearn.metrics import adjusted_rand_score

    from acdso import ACDSOClusterer

    rng = np.random.RandomState(42)
    topics = [
        (
            "laptop",
            0,
            1400.0,
            12.0,
            [
                "ultrabook laptop solid state drive fast processor long battery life",
                "portable notebook computer bright display keyboard aluminum chassis",
                "workstation laptop graphics memory high resolution screen fast charging",
            ],
        ),
        (
            "footwear",
            1,
            85.0,
            0.4,
            [
                "running sneakers cushioned sole breathable mesh lightweight athletic shoe",
                "trail hiking boots waterproof grip durable outsole ankle support",
                "casual walking shoes soft foam insole comfortable everyday footwear",
            ],
        ),
        (
            "kitchen",
            2,
            260.0,
            4.5,
            [
                "stainless steel espresso coffee maker grinder steam wand kitchen appliance",
                "programmable countertop convection toaster oven timer baking tray",
                "high speed blender glass pitcher smoothie kitchen mixer blades",
            ],
        ),
    ]

    rows: list[dict[str, Any]] = []
    true_labels: list[int] = []
    for _, topic_id, base_price, base_weight, templates in topics:
        for i in range(30):
            rows.append(
                {
                    "price": float(base_price + rng.normal(0.0, base_price * 0.08)),
                    "weight_kg": float(max(0.1, base_weight + rng.normal(0.0, base_weight * 0.1))),
                    "description": f"{templates[i % len(templates)]} SKU batch {i}",
                }
            )
            true_labels.append(topic_id)

    df = pd.DataFrame(rows)
    clusterer = ACDSOClusterer(
        algorithm="kmeans",
        n_clusters=3,
        dim_reduction="pca",
        n_components=4,
        metric="silhouette",
        text_cols=["description"],
        text_method="tfidf",
        random_state=42,
    )
    labels = clusterer.fit_predict(df)

    assert len(labels) == len(df)
    assert clusterer.n_clusters_ == 3
    assert clusterer.silhouette_score_ > 0.15
    assert clusterer.davies_bouldin_score_ > 0.0
    ari = adjusted_rand_score(true_labels, labels)
    assert ari >= 0.80, f"Expected distinct topic separation (ARI >= 0.80), got {ari:.3f}"

    reduced = clusterer.transform(df.iloc[:6])
    assert reduced.shape[0] == 6 and reduced.shape[1] >= 2

    unseen = pd.DataFrame(
        [
            {"price": 1380.0, "weight_kg": 1.4, "description": "fast laptop processor bright display solid state drive"},
            {"price": 80.0, "weight_kg": 0.42, "description": "breathable running sneakers cushioned sole lightweight shoe"},
            {"price": 255.0, "weight_kg": 4.4, "description": "stainless steel espresso coffee maker kitchen appliance"},
        ]
    )
    unseen_preds = clusterer.predict(unseen)
    assert len(unseen_preds) == 3
    assert len(set(int(x) for x in unseen_preds)) == 3
    json.dumps(clusterer.audit_trail_)


@pytest.mark.tier3
@pytest.mark.f1
def test_r2_r4_t3_nlp_text_columns_into_anomaly_detector() -> None:
    """Pairwise R2 x R4: Normal vs anomalous log texts + numeric metrics scored by ACDSOAnomalyDetector."""
    from acdso import ACDSOAnomalyDetector

    rng = np.random.RandomState(17)
    normal_phrases = [
        "request completed successfully status ok user session verified",
        "health check passed database connection pool nominal latency",
        "scheduled backup finished cleanly checksum verified no warnings",
        "cache refresh completed items updated response time normal",
    ]
    anomaly_phrases = [
        "CRITICAL kernel panic segmentation fault unauthorized root privilege escalation",
        "FATAL memory corruption buffer overflow exploit payload blocked emergency halt",
        "ALERT brute force credential stuffing exfiltration dump detected firewall breach",
    ]

    train_rows = [
        {
            "latency_ms": float(45.0 + rng.normal(0.0, 6.0)),
            "error_count": int(rng.choice([0, 0, 0, 1])),
            "text": normal_phrases[i % len(normal_phrases)],
        }
        for i in range(90)
    ]
    train_df = pd.DataFrame(train_rows)

    detector = ACDSOAnomalyDetector(
        algorithm="isolation_forest",
        contamination=0.05,
        text_cols=["text"],
        random_state=42,
    )
    detector.fit(train_df)

    unseen_normal = pd.DataFrame(
        [
            {
                "latency_ms": float(46.0 + rng.normal(0.0, 5.0)),
                "error_count": 0,
                "text": normal_phrases[i % len(normal_phrases)],
            }
            for i in range(15)
        ]
    )
    unseen_anom = pd.DataFrame(
        [
            {
                "latency_ms": float(4800.0 + i * 250.0),
                "error_count": 85 + i * 5,
                "text": anomaly_phrases[i % len(anomaly_phrases)],
            }
            for i in range(10)
        ]
    )
    unseen = pd.concat([unseen_normal, unseen_anom], ignore_index=True)

    preds = detector.predict(unseen)
    is_anom = detector.predict_is_anomaly(unseen)
    anom_scores = detector.anomaly_scores(unseen)
    decision_scores = detector.decision_function(unseen)

    assert preds.shape == (25,)
    assert set(np.unique(preds)).issubset({-1, 1})
    assert is_anom.dtype == bool
    assert is_anom[15:].mean() >= 0.80, f"Expected >=80% anomalous rows flagged, got {is_anom[15:].mean():.2f}"
    assert anom_scores[15:].mean() > anom_scores[:15].mean()
    assert decision_scores[:15].mean() > decision_scores[15:].mean()
    json.dumps(detector.audit_trail_)


@pytest.mark.tier3
@pytest.mark.f1
def test_r3_r4_t3_grouped_cv_with_nlp_classifier() -> None:
    """Pairwise R3 x R4: ACDSOClassifier.fit(X, y, groups=...) on mixed numeric + categorical + free-text DataFrame."""
    from acdso import ACDSOClassifier
    from acdso.cv import build_cv_splitter, validate_no_group_leakage

    rng = np.random.RandomState(42)
    n_groups = 12
    rows_per_group = 10
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    n_rows = len(groups)

    # Alternate label cleanly at group level as well as row signal so StratifiedGroupKFold works reliably
    group_bias = np.array([1 if g % 2 == 0 else -1 for g in groups])
    num_signal = group_bias * 1.4 + rng.normal(0.0, 0.4, size=n_rows)
    num_aux = rng.normal(0.0, 1.0, size=n_rows)
    region = np.where(group_bias > 0, rng.choice(["east", "north"], n_rows), rng.choice(["west", "south"], n_rows))
    notes = [
        (
            "urgent escalation hardware failure warranty replacement approved immediately"
            if (group_bias[i] > 0 and num_signal[i] > 0)
            else "routine scheduled maintenance inspection completed without any defects"
        )
        for i in range(n_rows)
    ]
    y = (num_signal > 0).astype(int)
    X = pd.DataFrame(
        {
            "signal_score": num_signal,
            "aux_metric": num_aux,
            "region": region,
            "notes": notes,
        }
    )

    splitter = build_cv_splitter(cv=3, y=y, groups=groups, task="classification", cv_strategy="auto", random_state=42)
    splits = list(splitter.split(X, y, groups=groups))
    validate_no_group_leakage(splits, groups)

    clf = ACDSOClassifier(scorer="accuracy", cv=3, max_candidates=3, random_state=42, verbose=0)
    clf.fit(X, y, groups=groups)

    assert clf.best_estimator_ is not None
    assert clf.score(X, y) >= 0.75
    preds = clf.predict(X.iloc[:12])
    assert len(preds) == 12
    json.dumps(clf.audit_trail_)


@pytest.mark.tier3
@pytest.mark.f1
def test_r3_r5a_t3_grouped_cv_multilabel_classifier() -> None:
    """Pairwise R3 x R5a: ACDSOMultiLabelClassifier.fit(X, y, groups=entity_ids) with zero group leakage."""
    from acdso import ACDSOMultiLabelClassifier
    from acdso.cv import build_cv_splitter, validate_no_group_leakage

    rng = np.random.RandomState(21)
    n_entities = 15
    rows_per_entity = 10
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    n_rows = len(entity_ids)

    x1 = rng.randn(n_rows)
    x2 = rng.randn(n_rows)
    x3 = rng.randn(n_rows)
    x4 = rng.randn(n_rows)
    X = pd.DataFrame({"f1": x1, "f2": x2, "f3": x3, "f4": x4})

    y = np.column_stack(
        [
            (x1 + 0.3 * x2 > 0.0).astype(int),
            (x2 - 0.5 * x3 > 0.1).astype(int),
            (x1 + x4 > 0.2).astype(int),
        ]
    )

    train_mask = entity_ids < 11
    test_mask = ~train_mask
    assert set(entity_ids[train_mask]).isdisjoint(set(entity_ids[test_mask]))

    splitter = build_cv_splitter(
        cv=3, y=y[train_mask], groups=entity_ids[train_mask], task="multilabel", cv_strategy="group_kfold"
    )
    validate_no_group_leakage(list(splitter.split(X[train_mask], y[train_mask], groups=entity_ids[train_mask])), entity_ids[train_mask])

    clf = ACDSOMultiLabelClassifier(strategy="auto", scorer="f1_micro", cv=3, random_state=42)
    clf.fit(X[train_mask], y[train_mask], groups=entity_ids[train_mask], cv_strategy="group_kfold")

    preds = clf.predict(X[test_mask])
    proba = clf.predict_proba(X[test_mask])
    score = clf.score(X[test_mask], y[test_mask])
    metrics = clf.evaluate(X[test_mask], y[test_mask])

    assert preds.shape == y[test_mask].shape
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.asarray(proba).shape[0] == int(test_mask.sum())
    assert score >= 0.70, f"Expected solid multi-label score >= 0.70, got {score:.3f}"
    assert isinstance(metrics, dict) and len(metrics) >= 1


@pytest.mark.tier3
@pytest.mark.f1
def test_r3_r5b_t3_grouped_cv_learning_to_rank() -> None:
    """Pairwise R3 x R5b: ACDSORanker.fit/score with query_ids enforces query-group isolation & high NDCG@k."""
    from acdso import ACDSORanker, ndcg_at_k
    from acdso.cv import build_cv_splitter, validate_no_group_leakage

    rng = np.random.RandomState(99)
    n_queries = 18
    docs_per_query = 10
    query_ids = np.repeat(np.arange(n_queries), docs_per_query)
    n_rows = len(query_ids)

    bm25 = rng.uniform(0.0, 1.0, size=n_rows)
    ctr = rng.uniform(0.0, 1.0, size=n_rows)
    title_match = rng.uniform(0.0, 1.0, size=n_rows)
    noise = rng.normal(0.0, 0.05, size=n_rows)

    latent = 2.5 * bm25 + 1.8 * ctr + 1.2 * title_match + noise
    relevance = np.zeros(n_rows, dtype=int)
    for q in range(n_queries):
        idx = np.where(query_ids == q)[0]
        ranks = np.argsort(np.argsort(latent[idx]))
        relevance[idx] = np.where(ranks >= 8, 3, np.where(ranks >= 5, 2, np.where(ranks >= 2, 1, 0)))

    X = pd.DataFrame({"bm25": bm25, "ctr": ctr, "title_match": title_match, "noise": noise})
    train_mask = query_ids < 13
    test_mask = ~train_mask

    splitter = build_cv_splitter(
        cv=3, y=relevance[train_mask], groups=query_ids[train_mask], task="ranking", cv_strategy="group_kfold"
    )
    validate_no_group_leakage(
        list(splitter.split(X[train_mask], relevance[train_mask], groups=query_ids[train_mask])),
        query_ids[train_mask],
    )

    ranker = ACDSORanker(algorithm="auto", k=5, cv=3, random_state=42)
    ranker.fit(X[train_mask], relevance[train_mask], groups=query_ids[train_mask])

    preds = ranker.predict(X[test_mask])
    ndcg_score = ranker.score(X[test_mask], relevance[test_mask], groups=query_ids[test_mask])
    eval_dict = ranker.evaluate(X[test_mask], relevance[test_mask], groups=query_ids[test_mask], k=5)
    direct_ndcg = ndcg_at_k(relevance[test_mask], preds, groups=query_ids[test_mask], k=5)
    inverted_ndcg = ndcg_at_k(relevance[test_mask], -preds, groups=query_ids[test_mask], k=5)

    assert np.isclose(ndcg_score, direct_ndcg, atol=1e-6)
    assert 0.75 <= ndcg_score <= 1.0, f"Expected NDCG@5 >= 0.75, got {ndcg_score:.3f}"
    assert ndcg_score > inverted_ndcg + 0.2
    assert isinstance(eval_dict, dict) and len(eval_dict) >= 1


@pytest.mark.tier3
@pytest.mark.f1
def test_r3_r5c_t3_grouped_cv_survival_analyzer() -> None:
    """Pairwise R3 x R5c: ACDSOSurvivalAnalyzer with hospital_ids grouping & censored time-to-event C-index."""
    from acdso import ACDSOSurvivalAnalyzer, concordance_index
    from acdso.cv import build_cv_splitter, validate_no_group_leakage

    rng = np.random.RandomState(123)
    n_hospitals = 10
    patients_per_hosp = 20
    hospital_ids = np.repeat(np.arange(n_hospitals), patients_per_hosp)
    n_rows = len(hospital_ids)

    age_z = rng.normal(0.0, 1.0, size=n_rows)
    biomarker = rng.normal(0.0, 1.0, size=n_rows)
    severity = rng.normal(0.0, 1.0, size=n_rows)
    eta = 1.1 * severity + 0.8 * biomarker + 0.5 * age_z

    true_t = rng.exponential(scale=10.0, size=n_rows) * np.exp(-eta)
    censor_t = rng.uniform(4.0, 18.0, size=n_rows)
    obs_time = np.maximum(0.05, np.minimum(true_t, censor_t))
    event = (true_t <= censor_t).astype(int)

    X = pd.DataFrame({"age_z": age_z, "biomarker": biomarker, "severity": severity})
    train_mask = hospital_ids < 7
    test_mask = ~train_mask

    splitter = build_cv_splitter(
        cv=3, y=event[train_mask], groups=hospital_ids[train_mask], task="survival", cv_strategy="group_kfold"
    )
    validate_no_group_leakage(
        list(splitter.split(X[train_mask], event[train_mask], groups=hospital_ids[train_mask])),
        hospital_ids[train_mask],
    )

    surv = ACDSOSurvivalAnalyzer(l2_penalty=1e-2, backend="auto", cv=3, random_state=42)
    surv.fit(X[train_mask], time=obs_time[train_mask], event=event[train_mask], groups=hospital_ids[train_mask])

    risk_scores = surv.predict_risk(X[test_mask])
    c_index = surv.score(X[test_mask], time=obs_time[test_mask], event=event[test_mask])
    direct_c = concordance_index(obs_time[test_mask], risk_scores, event[test_mask])
    eval_dict = surv.evaluate(X[test_mask], time=obs_time[test_mask], event=event[test_mask])

    assert np.isclose(c_index, direct_c, atol=1e-6)
    assert c_index > 0.65, f"Expected C-index > 0.65 on held-out hospitals, got {c_index:.3f}"
    assert isinstance(eval_dict, dict) and len(eval_dict) >= 1


@pytest.mark.tier3
@pytest.mark.f2
@pytest.mark.f4
def test_r3_r7_t3_cli_grouped_cv_and_report_summary(
    isolated_webui_client, dataset_factory, run_cli, tmp_path: Path
) -> None:
    """Pairwise R3 x R7: CLI --group-col/--cv-strategy produces grouped CV report inspected via /api/report-summary."""
    _, client, (_, rep_dir, _) = isolated_webui_client
    rng = np.random.RandomState(42)
    n_patients = 15
    rows_per_patient = 8
    patient_id = np.repeat([f"PAT-{i:03d}" for i in range(n_patients)], rows_per_patient)
    n_rows = len(patient_id)
    patient_risk = np.repeat([1.2 if i % 2 == 0 else -1.2 for i in range(n_patients)], rows_per_patient)
    x1 = patient_risk + rng.normal(0.0, 0.35, size=n_rows)
    x2 = rng.normal(0.0, 1.0, size=n_rows)
    target = (x1 > 0).astype(int)

    df = pd.DataFrame({"patient_id": patient_id, "biomarker": x1, "age_norm": x2, "target": target})
    csv_path = dataset_factory.write_csv(df, "grouped_cv_cli.csv")

    res = run_cli(
        [
            "--file",
            str(csv_path),
            "--task",
            "classification",
            "--predict",
            "target",
            "--group-col",
            "patient_id",
            "--cv-strategy",
            "stratified_group_kfold",
            "--simple",
            "--skip-benchmark",
        ],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "\x1b[" not in res.stdout

    events = parse_progress_events(res.stdout)
    assert len(events) >= 1

    pred_csv = rep_dir / "predictions_grouped_cv_cli_target_full_dataset.csv"
    assert pred_csv.is_file()
    assert len(pd.read_csv(pred_csv)) == n_rows

    combined_files = list(rep_dir.glob("autonomous_grouped_cv_cli*_combined.json"))
    assert combined_files
    raw_combined = _loads_strict_json(combined_files[0])
    assert raw_combined.get("task") == "classification"

    rel_name = combined_files[0].relative_to(rep_dir).as_posix()
    sum_resp = client.get(f"/api/report-summary?path={rel_name}")
    assert sum_resp.status_code == 200, sum_resp.get_data(as_text=True)
    summary = sum_resp.get_json()
    assert summary["kind"] == "combined"
    assert summary["meta"]["task"] == "classification"
    assert isinstance(summary.get("risk_report"), dict)
    summary_text = json.dumps(summary).lower()
    assert "patient_id" in summary_text or "stratified_group_kfold" in summary_text or "cv" in summary


@pytest.mark.tier3
@pytest.mark.f2
def test_r4_r7_t3_cli_nlp_text_branch_to_html_json_report(dataset_factory, run_cli, tmp_path: Path) -> None:
    """Pairwise R4 x R7: CLI --text-cols/--nlp-method produces [ACDSO_PROGRESS], strict JSON, balanced HTML, predictions CSV."""
    rng = np.random.RandomState(55)
    pos_texts = [
        "excellent build quality outstanding battery life fast delivery highly recommend",
        "amazing product exceeded expectations great screen bright colors love it",
        "reliable performance smooth operation sturdy design five stars purchase",
    ]
    neg_texts = [
        "terrible defect stopped working immediately poor customer service total waste",
        "broken on arrival cheap plastic fragile material disappointed refund requested",
        "overheats constantly slow laggy software crashes every single day awful",
    ]
    n_rows = 100
    is_pos = np.array([1 if i % 2 == 0 else 0 for i in range(n_rows)])
    df = pd.DataFrame(
        {
            "price": np.where(is_pos == 1, rng.uniform(50, 120, n_rows), rng.uniform(10, 45, n_rows)),
            "category": rng.choice(["electronics", "accessories", "audio"], size=n_rows),
            "review_text": [pos_texts[i % 3] if is_pos[i] else neg_texts[i % 3] for i in range(n_rows)],
            "sentiment": is_pos,
        }
    )
    csv_path = dataset_factory.write_csv(df, "nlp_cli_r4_r7.csv")

    res = run_cli(
        [
            "--file",
            str(csv_path),
            "--predict",
            "sentiment",
            "--text-cols",
            "review_text",
            "--nlp-method",
            "tfidf",
            "--simple",
            "--skip-benchmark",
        ],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "\x1b[" not in res.stdout

    events = parse_progress_events(res.stdout)
    assert len(events) >= 1
    phases = [int(ev["phase"]) for ev in events if isinstance(ev.get("phase"), int)]
    assert phases == sorted(phases)

    reports_dir = tmp_path / "reports"
    combined_files = list(reports_dir.glob("autonomous_nlp_cli_r4_r7*_combined.json"))
    html_files = list(reports_dir.glob("autonomous_nlp_cli_r4_r7*_report.html"))
    pred_csv = reports_dir / "predictions_nlp_cli_r4_r7_sentiment_full_dataset.csv"

    assert combined_files and html_files and pred_csv.is_file()
    combined = _loads_strict_json(combined_files[0])
    assert combined.get("selected_model") is not None

    html_text = html_files[0].read_text(encoding="utf-8", errors="replace")
    parser = _TableBalanceParser()
    parser.feed(html_text)
    assert parser.open_tables >= 1 and parser.open_tables == parser.close_tables

    pred_df = pd.read_csv(pred_csv)
    assert len(pred_df) == n_rows
    assert "sentiment_predicted" in pred_df.columns


@pytest.mark.tier3
@pytest.mark.f3
@pytest.mark.f4
def test_r1_r7_t3_webui_upload_clustering_run_and_report_summary(webui_client, dataset_factory) -> None:
    """Pairwise R1 x R7: WebUI upload -> /api/run (task=clustering, empty target) -> poll -> /api/report-summary."""
    _, client = webui_client
    rng = np.random.RandomState(42)
    c1 = rng.normal([0.0, 0.0], 0.4, size=(30, 2))
    c2 = rng.normal([6.0, 0.0], 0.4, size=(30, 2))
    c3 = rng.normal([3.0, 6.0], 0.4, size=(30, 2))
    pts = np.vstack([c1, c2, c3])
    df = pd.DataFrame({"feat_x": pts[:, 0], "feat_y": pts[:, 1], "noise": rng.normal(0.0, 0.1, size=90)})
    csv_path = dataset_factory.write_csv(df, "e2e_cluster_r1_r7.csv")

    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, csv_path.name)}, content_type="multipart/form-data")
    assert up.status_code == 200, up.get_data(as_text=True)
    uid = up.get_json()["upload_id"]

    created_paths: list[str] = []
    try:
        run_resp = client.post(
            "/api/run",
            json={
                "upload_id": uid,
                "task": "clustering",
                "predict": [],
                "target": [],
                "cluster_algo": "kmeans",
                "n_clusters": 3,
                "simple": True,
                "skip_benchmark": True,
            },
        )
        assert run_resp.status_code == 200, run_resp.get_data(as_text=True)
        job_id = run_resp.get_json()["job_id"]

        final, lines, _ = poll_job_until_done(
            lambda cur: ((r := client.get(f"/api/job/{job_id}?cursor={cur}")).status_code, r.get_json()),
            timeout=150.0,
        )
        assert final["status"] == "finished", f"Clustering job failed: {final}\n" + "\n".join(lines[-30:])
        assert final["returncode"] == 0

        created_paths = [a["path"] for a in final["artifacts"]]
        combined_rel = next((p for p in created_paths if p.endswith("_combined.json")), None)
        assert combined_rel is not None, f"Missing combined JSON in {created_paths}"

        sum_resp = client.get(f"/api/report-summary?path={combined_rel}")
        assert sum_resp.status_code == 200, sum_resp.get_data(as_text=True)
        summary = sum_resp.get_json()
        assert summary["meta"]["task"] == "clustering"
        assert isinstance(summary.get("clustering"), dict) and len(summary["clustering"]) >= 1
    finally:
        for rel in created_paths:
            client.delete(f"/api/reports?path={rel}")


@pytest.mark.tier3
@pytest.mark.f3
@pytest.mark.f4
def test_r2_r7_t3_webui_upload_anomaly_run_and_report_summary(webui_client, dataset_factory) -> None:
    """Pairwise R2 x R7: WebUI upload -> /api/run (task=anomaly, empty target) -> poll -> /api/report-summary."""
    _, client = webui_client
    rng = np.random.RandomState(42)
    normal = rng.normal(0.0, 0.5, size=(85, 3))
    outliers = np.array([[8.5, -7.0, 9.0], [-9.0, 8.0, -8.5], [10.0, 10.0, 10.0], [-10.0, -9.0, 8.0], [9.5, -8.5, -9.5]])
    pts = np.vstack([normal, outliers])
    df = pd.DataFrame({"cpu_load": pts[:, 0], "mem_delta": pts[:, 1], "net_spikes": pts[:, 2]})
    csv_path = dataset_factory.write_csv(df, "e2e_anomaly_r2_r7.csv")

    with csv_path.open("rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, csv_path.name)}, content_type="multipart/form-data")
    assert up.status_code == 200, up.get_data(as_text=True)
    uid = up.get_json()["upload_id"]

    created_paths: list[str] = []
    try:
        run_resp = client.post(
            "/api/run",
            json={
                "upload_id": uid,
                "task": "anomaly",
                "predict": [],
                "target": [],
                "anomaly_algo": "isolation_forest",
                "simple": True,
                "skip_benchmark": True,
            },
        )
        assert run_resp.status_code == 200, run_resp.get_data(as_text=True)
        job_id = run_resp.get_json()["job_id"]

        final, lines, _ = poll_job_until_done(
            lambda cur: ((r := client.get(f"/api/job/{job_id}?cursor={cur}")).status_code, r.get_json()),
            timeout=150.0,
        )
        assert final["status"] == "finished", f"Anomaly job failed: {final}\n" + "\n".join(lines[-30:])
        assert final["returncode"] == 0

        created_paths = [a["path"] for a in final["artifacts"]]
        combined_rel = next((p for p in created_paths if p.endswith("_combined.json")), None)
        assert combined_rel is not None, f"Missing combined JSON in {created_paths}"

        sum_resp = client.get(f"/api/report-summary?path={combined_rel}")
        assert sum_resp.status_code == 200, sum_resp.get_data(as_text=True)
        summary = sum_resp.get_json()
        assert summary["meta"]["task"] == "anomaly"
        assert isinstance(summary.get("anomaly"), dict) and len(summary["anomaly"]) >= 1
    finally:
        for rel in created_paths:
            client.delete(f"/api/reports?path={rel}")


@pytest.mark.tier3
@pytest.mark.f2
def test_r6_r7_t3_chunked_out_of_core_cli_and_streaming_predict(dataset_factory, run_cli, tmp_path: Path) -> None:
    """Pairwise R6 x R7: Multi-chunk CSV processed by CLI --scale-mode chunked --chunk-size 500."""
    rng = np.random.RandomState(42)
    n_rows = 2200
    x1 = rng.randn(n_rows)
    x2 = rng.randn(n_rows)
    x3 = rng.randn(n_rows)
    target = (2.0 * x1 - 1.5 * x2 + 0.2 * rng.randn(n_rows) > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target})
    csv_path = dataset_factory.write_csv(df, "chunked_scale_r6_r7.csv")

    res = run_cli(
        [
            "--file",
            str(csv_path),
            "--task",
            "classification",
            "--predict",
            "target",
            "--scale-mode",
            "chunked",
            "--chunk-size",
            "500",
            "--simple",
            "--skip-benchmark",
        ],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    reports_dir = tmp_path / "reports"
    pred_csv = reports_dir / "predictions_chunked_scale_r6_r7_target_full_dataset.csv"
    assert pred_csv.is_file()
    pred_df = pd.read_csv(pred_csv)
    assert len(pred_df) == n_rows
    assert "target_predicted" in pred_df.columns

    combined_files = list(reports_dir.glob("autonomous_chunked_scale_r6_r7*_combined.json"))
    assert combined_files
    combined = _loads_strict_json(combined_files[0])
    assert isinstance(combined.get("scale"), dict) and len(combined["scale"]) >= 1


@pytest.mark.tier3
@pytest.mark.f2
def test_r5d_r7_t3_tui_build_cli_command_and_cli_execution_recommender(
    dataset_factory, run_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pairwise R5d x R7: TUI build_cli_command(task='recommender') -> run_cli -> load_latest_combined_summary."""
    import acdso.tui.app as tui_app

    rng = np.random.RandomState(42)
    users = [f"U{u:02d}" for u in range(15)]
    items = [f"I{i:02d}" for i in range(12)]
    records: list[dict[str, Any]] = []
    for u_idx, u in enumerate(users):
        for i_idx, it in enumerate(items):
            if (u_idx + i_idx) % 2 == 0 or rng.rand() < 0.35:
                base = 4.5 if (u_idx % 3 == i_idx % 3) else 2.0
                rating = float(np.clip(base + rng.normal(0.0, 0.3), 1.0, 5.0))
                records.append({"user_id": u, "item_id": it, "rating": round(rating, 2)})

    df = pd.DataFrame(records)
    csv_path = dataset_factory.write_csv(df, "tui_rec_r5d_r7.csv")

    cmd = tui_app.build_cli_command(
        str(csv_path),
        "",
        task="recommender",
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        simple=True,
        skip_benchmark=True,
    )
    assert "--task" in cmd and "recommender" in cmd
    assert "--user-col" in cmd and "user_id" in cmd
    assert "--item-col" in cmd and "item_id" in cmd
    assert "--rating-col" in cmd and "rating" in cmd
    assert "--predict" not in cmd and "--target" not in cmd

    file_flag_idx = cmd.index("--file")
    cli_args = cmd[file_flag_idx:]
    res = run_cli(cli_args, cwd=tmp_path, timeout=150.0)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    monkeypatch.setattr(tui_app, "REPORTS_DIR", tmp_path / "reports")
    summary = tui_app.load_latest_combined_summary(csv_path, "")
    assert summary is not None, f"load_latest_combined_summary returned None in {tmp_path / 'reports'}"
    assert summary.get("report_path") and Path(summary["report_path"]).is_file()
    raw_report = _loads_strict_json(Path(summary["report_path"]))
    assert raw_report.get("task") == "recommender"
