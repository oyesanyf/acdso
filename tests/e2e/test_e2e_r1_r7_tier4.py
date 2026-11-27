"""Tier 4 E2E Tests: R1-R7 Real-World Application Workloads (7 tests)."""

from __future__ import annotations

import asyncio
import inspect
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
    assert isinstance(data, dict), f"Expected JSON object in {path}"
    return data


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f2
def test_w1_r1_r4_r7_t4_customer_segmentation_workload(dataset_factory, run_cli, tmp_path: Path) -> None:
    """W1 (R1, R4, R7): Realistic customer behavior + support notes -> ACDSOClusterer + CLI clustering report."""
    from sklearn.metrics import adjusted_rand_score

    from acdso import ACDSOClusterer

    rng = np.random.RandomState(42)
    personas = [
        (
            0,
            18000.0,
            42.0,
            2,
            "enterprise",
            "enterprise dedicated account sla renewal expansion multi seat deployment",
        ),
        (
            1,
            450.0,
            14.0,
            1,
            "starter",
            "seasonal discount coupon promo starter plan self service checkout",
        ),
        (
            2,
            120.0,
            3.0,
            9,
            "trial",
            "cancel subscription billing dispute slow response frustrated churn warning",
        ),
    ]

    records: list[dict[str, Any]] = []
    true_seg: list[int] = []
    for seg_id, spend_mu, tenure_mu, ticket_mu, tier, note in personas:
        for i in range(45):
            records.append(
                {
                    "annual_spend": float(max(20.0, spend_mu + rng.normal(0.0, spend_mu * 0.12))),
                    "tenure_months": float(max(1.0, tenure_mu + rng.normal(0.0, 2.0))),
                    "support_tickets": int(max(0, ticket_mu + rng.randint(-1, 2))),
                    "plan_tier": tier,
                    "support_notes": f"{note} ticket ref {i}",
                }
            )
            true_seg.append(seg_id)

    df = pd.DataFrame(records)
    clusterer = ACDSOClusterer(
        algorithm="auto",
        k_range=(2, 5),
        dim_reduction="pca",
        n_components=4,
        metric="silhouette",
        text_cols=["support_notes"],
        text_method="tfidf",
        random_state=42,
    )
    labels = clusterer.fit_predict(df)

    assert clusterer.n_clusters_ >= 2
    assert clusterer.silhouette_score_ > 0.15
    assert clusterer.davies_bouldin_score_ > 0.0
    assert adjusted_rand_score(true_seg, labels) >= 0.70

    summary = clusterer.get_cluster_summary()
    assert isinstance(summary, dict) and len(summary) >= 1
    assert len(clusterer.predict(df.iloc[:6])) == 6
    json.dumps(clusterer.audit_trail_)

    csv_path = dataset_factory.write_csv(df, "w1_customers.csv")
    res = run_cli(
        [
            "--file",
            str(csv_path),
            "--task",
            "clustering",
            "--text-cols",
            "support_notes",
            "--nlp-method",
            "tfidf",
            "--dim-reduce",
            "pca",
            "--simple",
            "--skip-benchmark",
        ],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert len(parse_progress_events(res.stdout)) >= 1

    reports_dir = tmp_path / "reports"
    combined_files = list(reports_dir.glob("autonomous_w1_customers*_combined.json"))
    html_files = list(reports_dir.glob("autonomous_w1_customers*_report.html"))
    assert combined_files and html_files

    combined = _loads_strict_json(combined_files[0])
    assert combined.get("task") == "clustering"
    assert isinstance(combined.get("clustering"), dict) and len(combined["clustering"]) >= 1

    parser = _TableBalanceParser()
    parser.feed(html_files[0].read_text(encoding="utf-8", errors="replace"))
    assert parser.open_tables >= 1 and parser.open_tables == parser.close_tables


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f2
@pytest.mark.f4
def test_w2_r2_r4_r7_t4_transaction_review_anomaly_workload(
    isolated_webui_client, dataset_factory, run_cli, tmp_path: Path
) -> None:
    """W2 (R2, R4, R7): E-commerce transaction + review fraud anomaly pipeline via AutoEDA hook, CLI, and /api/report-summary."""
    from acdso import ACDSOAutoEDA

    _, client, (_, rep_dir, _) = isolated_webui_client
    rng = np.random.RandomState(42)

    normal_reviews = [
        "verified purchase arrived on time packaging intact works as expected",
        "good everyday item matches description shipping was prompt no issues",
        "happy customer reasonable price solid quality smooth checkout experience",
    ]
    fraud_reviews = [
        "URGENT WIRE GIFT CARD CRYPTO REFUND LINK SCAMBOT PHISHING PAYLOAD",
        "FREE BITCOIN CLICK NOW STOLEN CARD VELOCITY BOT SPAM INJECTION",
    ]

    train_df = pd.DataFrame(
        [
            {
                "order_amount": float(max(5.0, 65.0 + rng.normal(0.0, 18.0))),
                "velocity_1h": int(rng.choice([1, 1, 1, 2])),
                "chargeback_risk": float(max(0.0, 0.04 + rng.normal(0.0, 0.015))),
                "review_text": normal_reviews[i % len(normal_reviews)],
            }
            for i in range(100)
        ]
    )

    eda = ACDSOAutoEDA()
    eda.analyze(train_df, dataset_name="transactions_train")
    detector = eda.fit_anomaly_detector(
        train_df,
        algorithm="auto",
        contamination=0.05,
        text_cols=["review_text"],
        random_state=42,
    )
    assert getattr(eda, "anomaly_detector_", None) is detector

    holdout_norm = pd.DataFrame(
        [
            {
                "order_amount": float(62.0 + rng.normal(0.0, 12.0)),
                "velocity_1h": 1,
                "chargeback_risk": 0.04,
                "review_text": normal_reviews[i % len(normal_reviews)],
            }
            for i in range(15)
        ]
    )
    holdout_fraud = pd.DataFrame(
        [
            {
                "order_amount": float(3800.0 + i * 400.0),
                "velocity_1h": 35 + i * 4,
                "chargeback_risk": 0.96,
                "review_text": fraud_reviews[i % len(fraud_reviews)],
            }
            for i in range(10)
        ]
    )
    holdout = pd.concat([holdout_norm, holdout_fraud], ignore_index=True)
    is_anom = detector.predict_is_anomaly(holdout)
    anom_scores = detector.anomaly_scores(holdout)

    assert is_anom[15:].mean() >= 0.80
    assert anom_scores[15:].mean() > anom_scores[:15].mean()

    cli_df = pd.concat([train_df, holdout_fraud], ignore_index=True)
    csv_path = dataset_factory.write_csv(cli_df, "w2_tx_anomaly.csv")
    res = run_cli(
        [
            "--file",
            str(csv_path),
            "--task",
            "anomaly",
            "--text-cols",
            "review_text",
            "--anomaly-algo",
            "isolation_forest",
            "--simple",
            "--skip-benchmark",
        ],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    combined_files = list(rep_dir.glob("autonomous_w2_tx_anomaly*_combined.json"))
    assert combined_files
    rel_name = combined_files[0].relative_to(rep_dir).as_posix()
    sum_resp = client.get(f"/api/report-summary?path={rel_name}")
    assert sum_resp.status_code == 200, sum_resp.get_data(as_text=True)
    summary = sum_resp.get_json()
    assert summary["meta"]["task"] == "anomaly"
    assert isinstance(summary.get("anomaly"), dict) and len(summary["anomaly"]) >= 1


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f2
def test_w3_r3_r5c_r7_t4_clinical_survival_and_grouped_outcome_workload(
    dataset_factory, run_cli, tmp_path: Path
) -> None:
    """W3 (R3, R5c, R7): Multi-hospital clinical cohort -> grouped ACDSOClassifier + ACDSOSurvivalAnalyzer + CLI survival."""
    from acdso import ACDSOClassifier, ACDSOSurvivalAnalyzer, concordance_index
    from acdso.cv import build_cv_splitter, validate_no_group_leakage

    rng = np.random.RandomState(2025)
    n_hospitals = 10
    patients_per_hosp = 20
    hospital_id = np.repeat([f"HOSP_{h:02d}" for h in range(n_hospitals)], patients_per_hosp)
    patient_id = np.array([f"P_{i:04d}" for i in range(n_hospitals * patients_per_hosp)])
    n_rows = len(patient_id)

    age_z = rng.normal(0.0, 1.0, size=n_rows)
    creatinine = rng.normal(0.0, 1.0, size=n_rows)
    severity = rng.normal(0.0, 1.0, size=n_rows)
    eta = 1.2 * severity + 0.85 * creatinine + 0.55 * age_z

    true_time = 12.0 * np.exp(-eta) * rng.exponential(1.0, size=n_rows)
    censor_time = rng.uniform(5.0, 20.0, size=n_rows)
    obs_time = np.round(np.maximum(0.1, np.minimum(true_time, censor_time)), 3)
    event = (true_time <= censor_time).astype(int)
    complication = (eta + rng.normal(0.0, 0.35, size=n_rows) > 0.0).astype(int)

    X = pd.DataFrame({"age_z": age_z, "creatinine": creatinine, "severity": severity})
    train_mask = np.isin(hospital_id, [f"HOSP_{h:02d}" for h in range(7)])
    test_mask = ~train_mask

    splitter = build_cv_splitter(
        cv=3, y=complication[train_mask], groups=hospital_id[train_mask], task="classification", cv_strategy="auto"
    )
    validate_no_group_leakage(
        list(splitter.split(X[train_mask], complication[train_mask], groups=hospital_id[train_mask])),
        hospital_id[train_mask],
    )

    clf = ACDSOClassifier(scorer="accuracy", cv=3, max_candidates=3, random_state=42, verbose=0)
    clf.fit(X[train_mask], complication[train_mask], groups=hospital_id[train_mask])
    assert clf.score(X[test_mask], complication[test_mask]) >= 0.72

    surv = ACDSOSurvivalAnalyzer(l2_penalty=1e-2, backend="auto", cv=3, random_state=42)
    surv.fit(X[train_mask], time=obs_time[train_mask], event=event[train_mask], groups=hospital_id[train_mask])

    risk_test = surv.predict_risk(X[test_mask])
    cidx = surv.score(X[test_mask], time=obs_time[test_mask], event=event[test_mask])
    assert np.isclose(cidx, concordance_index(obs_time[test_mask], risk_test, event[test_mask]), atol=1e-6)
    assert cidx > 0.65, f"Expected survival C-index > 0.65, got {cidx:.3f}"

    sf = surv.predict_survival_function(X[test_mask].iloc[:4])
    sf_arr = np.asarray(sf)
    assert sf_arr.shape[0] == 4 and sf_arr.shape[1] >= 2
    assert np.all((sf_arr >= -1e-6) & (sf_arr <= 1.0 + 1e-6))
    assert np.all(np.diff(sf_arr, axis=1) <= 1e-6)

    cohort_df = X.copy()
    cohort_df["hospital_id"] = hospital_id
    cohort_df["time"] = obs_time
    cohort_df["event"] = event
    csv_path = dataset_factory.write_csv(cohort_df, "w3_clinical_cohort.csv")

    res = run_cli(
        [
            "--file",
            str(csv_path),
            "--task",
            "survival",
            "--time-col",
            "time",
            "--event-col",
            "event",
            "--group-col",
            "hospital_id",
            "--simple",
            "--skip-benchmark",
        ],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert len(parse_progress_events(res.stdout)) >= 1

    reports_dir = tmp_path / "reports"
    combined_files = list(reports_dir.glob("autonomous_w3_clinical_cohort*_combined.json"))
    html_files = list(reports_dir.glob("autonomous_w3_clinical_cohort*_report.html"))
    assert combined_files and html_files
    combined = _loads_strict_json(combined_files[0])
    assert combined.get("task") == "survival"

    parser = _TableBalanceParser()
    parser.feed(html_files[0].read_text(encoding="utf-8", errors="replace"))
    assert parser.open_tables >= 1 and parser.open_tables == parser.close_tables


@pytest.mark.tier4
@pytest.mark.f1
def test_w4_r3_r4_r5a_r5b_t4_search_relevance_and_product_tagging_workload() -> None:
    """W4 (R3, R4, R5a, R5b): Search relevance ACDSORanker + brand-grouped ACDSOMultiLabelClassifier with NLP text."""
    from acdso import ACDSOMultiLabelClassifier, ACDSORanker, ndcg_at_k

    rng = np.random.RandomState(77)

    # Part A: Query-grouped search ranking with text descriptions
    n_queries = 20
    docs_per_q = 10
    query_ids = np.repeat(np.arange(n_queries), docs_per_q)
    n_docs = len(query_ids)

    title_bm25 = rng.uniform(0.0, 1.0, size=n_docs)
    clicks = rng.uniform(0.0, 1.0, size=n_docs)
    freshness = rng.uniform(0.0, 1.0, size=n_docs)
    latent = 2.6 * title_bm25 + 1.9 * clicks + 0.9 * freshness + rng.normal(0.0, 0.08, size=n_docs)

    relevance = np.zeros(n_docs, dtype=int)
    for q in range(n_queries):
        idx = np.where(query_ids == q)[0]
        order = np.argsort(np.argsort(latent[idx]))
        relevance[idx] = np.where(order >= 8, 3, np.where(order >= 5, 2, np.where(order >= 2, 1, 0)))

    desc_list = [
        (
            "exact match official verified bestseller top rated item"
            if relevance[i] >= 2
            else "unrelated generic accessory placeholder catalog entry"
        )
        for i in range(n_docs)
    ]
    X_rank = pd.DataFrame(
        {
            "title_bm25": title_bm25,
            "clicks": clicks,
            "freshness": freshness,
            "product_desc": desc_list,
        }
    )

    q_tr = query_ids < 14
    q_te = ~q_tr
    ranker = ACDSORanker(algorithm="auto", k=5, cv=3, text_cols=["product_desc"], random_state=42)
    ranker.fit(X_rank[q_tr], relevance[q_tr], groups=query_ids[q_tr])
    rank_ndcg = ranker.score(X_rank[q_te], relevance[q_te], groups=query_ids[q_te])
    shuffled = rng.permutation(len(relevance[q_te])).astype(float)
    baseline_ndcg = ndcg_at_k(relevance[q_te], shuffled, groups=query_ids[q_te], k=5)
    assert rank_ndcg > 0.75 and rank_ndcg > baseline_ndcg + 0.15

    # Part B: Brand-grouped multi-label product tagging with text
    n_brands = 15
    prods_per_brand = 10
    brand_ids = np.repeat(np.arange(n_brands), prods_per_brand)
    n_prods = len(brand_ids)

    weight = rng.normal(0.0, 1.0, size=n_prods)
    price = rng.normal(0.0, 1.0, size=n_prods)
    wattage = rng.normal(0.0, 1.0, size=n_prods)
    tag_elec = (wattage > 0.0).astype(int)
    tag_port = (weight < 0.0).astype(int)
    tag_prem = (price > 0.1).astype(int)
    Y_tags = np.column_stack([tag_elec, tag_port, tag_prem])

    prod_texts = [
        " ".join(
            [
                "electric digital smart device" if tag_elec[i] else "manual mechanical analog craft",
                "compact portable travel pocket" if tag_port[i] else "heavy duty stationary desktop",
                "luxury titanium flagship pro" if tag_prem[i] else "budget economy value basic",
            ]
        )
        for i in range(n_prods)
    ]
    X_tag = pd.DataFrame({"weight": weight, "price": price, "wattage": wattage, "product_desc": prod_texts})

    b_tr = brand_ids < 11
    b_te = ~b_tr
    ml_clf = ACDSOMultiLabelClassifier(
        strategy="auto", scorer="f1_micro", cv=3, text_cols=["product_desc"], random_state=42
    )
    ml_clf.fit(X_tag[b_tr], Y_tags[b_tr], groups=brand_ids[b_tr], cv_strategy="group_kfold")
    ml_score = ml_clf.score(X_tag[b_te], Y_tags[b_te])
    assert ml_score >= 0.75, f"Expected multi-label F1 >= 0.75 on held-out brands, got {ml_score:.3f}"


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f3
@pytest.mark.f4
def test_w5_r5d_r7_t4_ecommerce_recommender_live_webui_workload(live_webui_server, dataset_factory) -> None:
    """W5 (R5d, R7): ACDSORecommender top-N + cold-start + full Waitress HTTP lifecycle."""
    from acdso import ACDSORecommender

    rng = np.random.RandomState(42)
    users = [f"user_{u:02d}" for u in range(24)]
    items = [f"item_{i:02d}" for i in range(18)]

    rows: list[dict[str, Any]] = []
    for u_idx, u in enumerate(users):
        u_pref = u_idx % 3
        for i_idx, it in enumerate(items):
            if (u_idx + i_idx) % 2 == 0 or rng.rand() < 0.35:
                i_group = i_idx % 3
                rating = 4.6 if u_pref == i_group else 1.8
                rating = float(np.clip(rating + rng.normal(0.0, 0.25), 1.0, 5.0))
                rows.append({"user_id": u, "item_id": it, "rating": round(rating, 2)})

    full_df = pd.DataFrame(rows)
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx].reset_index(drop=True)
    test_df = full_df.iloc[split_idx:].reset_index(drop=True)

    rec = ACDSORecommender(method="auto", n_factors=6, n_neighbors=8, random_state=42)
    rec.fit(train_df, user_col="user_id", item_col="item_id", rating_col="rating")

    target_user = "user_00"
    seen_items = set(train_df.loc[train_df["user_id"] == target_user, "item_id"])
    top_recs = rec.recommend(target_user, n=5, exclude_seen=True)
    assert len(top_recs) == 5
    assert all(item not in seen_items for item, _ in top_recs)

    cold_pred = rec.predict(["unseen_user_999"], ["unseen_item_999"])
    assert cold_pred.shape == (1,) and np.isfinite(cold_pred[0])

    eval_metrics = rec.evaluate(test_df, k=5, user_col="user_id", item_col="item_id", rating_col="rating")
    assert isinstance(eval_metrics, dict) and len(eval_metrics) >= 1
    if "rmse" in eval_metrics:
        assert float(eval_metrics["rmse"]) < 1.6

    # Full Waitress HTTP lifecycle for recommender task
    srv = live_webui_server
    st_root, _, html_bytes = srv.get("/")
    assert st_root == 200 and b"ACDSO" in html_bytes

    st_h, health = srv.get_json("/api/health")
    assert st_h == 200 and health.get("ok") is True

    rec_csv = dataset_factory.write_csv(full_df, "e2e_w5_recommender.csv")
    st_up, up = srv.upload_file(rec_csv)
    assert st_up == 200 and up["rows"] == len(full_df)
    uid = up["upload_id"]

    st_val, val = srv.post_json(
        "/api/validate-columns",
        {"upload_id": uid, "task": "recommender", "user_col": "user_id", "item_col": "item_id", "rating_col": "rating"},
    )
    assert st_val == 200 and val["ok"] is True

    created_paths: list[str] = []
    try:
        st_run, run_data = srv.post_json(
            "/api/run",
            {
                "upload_id": uid,
                "task": "recommender",
                "predict": [],
                "target": [],
                "user_col": "user_id",
                "item_col": "item_id",
                "rating_col": "rating",
                "simple": True,
                "skip_benchmark": True,
            },
        )
        assert st_run == 200, run_data
        job_id = run_data["job_id"]

        final, lines, _ = poll_job_until_done(
            lambda cur: srv.get_json(f"/api/job/{job_id}?cursor={cur}"),
            timeout=150.0,
        )
        assert final["status"] == "finished", f"Recommender job failed: {final}\n" + "\n".join(lines[-30:])
        assert final["returncode"] == 0

        created_paths = [a["path"] for a in final["artifacts"]]
        st_rep, _ = srv.get_json("/api/reports?limit=50")
        assert st_rep == 200

        combined_rel = next(p for p in created_paths if p.endswith("_combined.json"))
        st_sum, summary = srv.get_json(f"/api/report-summary?path={combined_rel}")
        assert st_sum == 200 and summary["meta"]["task"] == "recommender"

        st_dl, _, dl_bytes = srv.get(f"/api/artifact?path={combined_rel}")
        assert st_dl == 200 and len(dl_bytes) > 10

        st_del, _ = srv.delete_json(f"/api/reports?path={combined_rel}")
        assert st_del == 200
        created_paths = [p for p in created_paths if p != combined_rel]
    finally:
        for rel in created_paths:
            srv.delete_json(f"/api/reports?path={rel}")


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f2
def test_w6_r6_r7_t4_large_scale_out_of_core_streaming_workload(run_cli, tmp_path: Path) -> None:
    """W6 (R6, R7): 25k-row multi-chunk CSV -> ACDSOOutOfCoreTrainer partial_fit + subsample + predict_to_file + CLI."""
    from sklearn.metrics import accuracy_score

    from acdso.scale import ACDSOOutOfCoreTrainer, reservoir_sample_file

    rng = np.random.RandomState(42)
    n_train = 25000
    x1 = rng.randn(n_train)
    x2 = rng.randn(n_train)
    x3 = rng.randn(n_train)
    x4 = rng.randn(n_train)
    y_train = (2.2 * x1 - 1.7 * x2 + 0.8 * x3 + 0.25 * rng.randn(n_train) > 0).astype(int)
    train_df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "target": y_train})

    train_csv = tmp_path / "w6_large_train.csv"
    train_df.to_csv(train_csv, index=False)

    n_test = 3000
    tx1 = rng.randn(n_test)
    tx2 = rng.randn(n_test)
    tx3 = rng.randn(n_test)
    tx4 = rng.randn(n_test)
    y_test = (2.2 * tx1 - 1.7 * tx2 + 0.8 * tx3 + 0.25 * rng.randn(n_test) > 0).astype(int)
    test_df = pd.DataFrame({"x1": tx1, "x2": tx2, "x3": tx3, "x4": tx4})
    test_csv = tmp_path / "w6_large_test.csv"
    test_df.to_csv(test_csv, index=False)

    sample_df, total_seen = reservoir_sample_file(train_csv, max_rows=2000, chunk_size=1500, random_state=42)
    assert len(sample_df) == 2000
    assert total_seen == n_train

    trainer = ACDSOOutOfCoreTrainer(
        task="classification",
        chunk_size=1500,
        max_in_memory_rows=2000,
        random_state=42,
    )
    trainer.fit_from_file(train_csv, target_col="target")

    pred_out = tmp_path / "w6_stream_preds.csv"
    trainer.predict_to_file(test_csv, pred_out, chunk_size=1500)
    assert pred_out.is_file()
    pred_df = pd.read_csv(pred_out)
    assert len(pred_df) == n_test

    mem_preds = trainer.predict(test_df)
    assert len(mem_preds) == n_test
    assert accuracy_score(y_test, mem_preds) >= 0.78

    # Bounded subsample fallback mode
    init_params = inspect.signature(ACDSOOutOfCoreTrainer.__init__).parameters
    fit_params = inspect.signature(ACDSOOutOfCoreTrainer.fit_from_file).parameters
    if "mode" in init_params:
        sub_trainer = ACDSOOutOfCoreTrainer(
            task="classification", chunk_size=1500, max_in_memory_rows=2000, mode="subsample", random_state=42
        )
        sub_trainer.fit_from_file(train_csv, target_col="target")
    elif "mode" in fit_params:
        sub_trainer = ACDSOOutOfCoreTrainer(
            task="classification", chunk_size=1500, max_in_memory_rows=2000, random_state=42
        )
        sub_trainer.fit_from_file(train_csv, target_col="target", mode="subsample")
    else:
        sub_trainer = ACDSOOutOfCoreTrainer(
            task="classification", algorithm="subsample", chunk_size=1500, max_in_memory_rows=2000, random_state=42
        )
        sub_trainer.fit_from_file(train_csv, target_col="target")
    sub_preds = sub_trainer.predict(test_df)
    assert len(sub_preds) == n_test and accuracy_score(y_test, sub_preds) >= 0.75

    # CLI --scale-mode subsample verification
    res = run_cli(
        [
            "--file",
            str(train_csv),
            "--task",
            "classification",
            "--predict",
            "target",
            "--scale-mode",
            "subsample",
            "--chunk-size",
            "1500",
            "--simple",
            "--skip-benchmark",
        ],
        cwd=tmp_path,
        timeout=150.0,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    combined_files = list((tmp_path / "reports").glob("autonomous_w6_large_train*_combined.json"))
    assert combined_files
    combined = _loads_strict_json(combined_files[0])
    assert isinstance(combined.get("scale"), dict)


@pytest.mark.tier4
@pytest.mark.f3
@pytest.mark.f4
def test_w7_r7_t4_cross_interface_tui_and_webui_workbench_workload(
    live_webui_server, dataset_factory, run_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W7 (R7): Cross-interface FallbackTerminalUI + Textual ACDSOTuiApp + Waitress /api/nl-plan & /api/compare-reports."""
    import acdso.tui.app as tui_app

    rng = np.random.RandomState(42)
    df1 = pd.DataFrame(
        {
            "patient_id": np.repeat([f"P{i:02d}" for i in range(12)], 8),
            "f1": rng.randn(96),
            "f2": rng.randn(96),
            "target": (rng.randn(96) > 0).astype(int),
        }
    )
    # Make target strongly predictable from f1
    df1["target"] = (df1["f1"] - 0.5 * df1["f2"] > 0).astype(int)
    df2 = df1.copy()
    df2["f3"] = df2["f1"] * 0.5 + rng.normal(0.0, 0.2, size=len(df2))

    csv1 = dataset_factory.write_csv(df1, "w7_cohort_a.csv")
    csv2 = dataset_factory.write_csv(df2, "w7_cohort_b.csv")

    # Part A: Produce real combined JSON in tmp_path/reports and exercise FallbackTerminalUI, build_cli_command & ACDSOTuiApp
    cmd_cls = tui_app.build_cli_command(str(csv1), "target", task="classification", simple=True)
    assert "--file" in cmd_cls and str(csv1) in cmd_cls
    assert "--task" in cmd_cls and cmd_cls[cmd_cls.index("--task") + 1] == "classification"
    assert "--predict" in cmd_cls or "--target" in cmd_cls

    cmd_clu = tui_app.build_cli_command(str(csv2), "", task="clustering", simple=True)
    assert "--file" in cmd_clu and str(csv2) in cmd_clu
    assert "--task" in cmd_clu and cmd_clu[cmd_clu.index("--task") + 1] == "clustering"
    assert "--target" not in cmd_clu and "--predict" not in cmd_clu

    tui_cwd = tmp_path / "tui_run"
    cli_res = run_cli(
        ["--file", str(csv1), "--task", "classification", "--predict", "target", "--simple", "--skip-benchmark"],
        cwd=tui_cwd,
        timeout=150.0,
    )
    assert cli_res.returncode == 0, f"STDOUT:\n{cli_res.stdout}\nSTDERR:\n{cli_res.stderr}"

    monkeypatch.setattr(tui_app, "REPORTS_DIR", tui_cwd / "reports")
    scripted = iter(["1", str(csv1), "2", "3", "target", "6", "q"])
    tui_lines: list[str] = []
    ui = tui_app.FallbackTerminalUI(input_fn=lambda _p: next(scripted), output_fn=tui_lines.append)
    assert ui.run() == 0
    assert ui.profile is not None and ui.profile["rows"] == 96
    assert ui.target == "target"
    ui.last_summary = tui_app.load_latest_combined_summary(csv1, "target")
    assert ui.last_summary is not None
    ui._show_results()
    assert any("Selected Model:" in ln for ln in tui_lines)

    if tui_app.has_textual():
        from textual.widgets import DataTable, Input, Static

        async def _drive_textual() -> None:
            app = tui_app.create_textual_app(initial_file=str(csv1))
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause()
                cols_table = app.query_one("#cols-table", DataTable)
                assert cols_table.row_count == 4
                meta_txt = str(app.query_one("#dataset-meta", Static).render())
                assert "w7_cohort_a.csv" in meta_txt
                await pilot.click("#suggest-btn")
                await pilot.pause()
                assert app.query_one("#target-input", Input).value != ""
                app._populate_results(str(csv1), "target")
                await pilot.pause()
                banner_txt = str(app.query_one("#kpi-banner", Static).render())
                assert banner_txt.strip() != ""
                await pilot.press("q")

        asyncio.run(_drive_textual())

    # Part B: Waitress WebUI workbench: upload -> nl-plan -> validate-columns -> 2 runs (R3 grouped cls + R1 clustering) -> compare-reports -> delete
    srv = live_webui_server
    st1, up1 = srv.upload_file(csv1)
    st2, up2 = srv.upload_file(csv2)
    assert st1 == 200 and st2 == 200
    uid1, uid2 = up1["upload_id"], up2["upload_id"]

    st_nl, nl_plan = srv.post_json("/api/nl-plan", {"upload_id": uid1, "prompt": "predict target excluding patient_id"})
    assert st_nl == 200
    assert "target" in (nl_plan.get("predict") or [])

    st_val_ok, val_ok = srv.post_json(
        "/api/validate-columns",
        {"upload_id": uid1, "predict": ["target"], "group_col": ["patient_id"]},
    )
    assert st_val_ok == 200 and val_ok["ok"] is True

    st_val_bad, val_bad = srv.post_json(
        "/api/validate-columns",
        {"upload_id": uid1, "predict": ["target"], "group_col": ["target"]},
    )
    assert st_val_bad == 200 and val_bad["ok"] is False

    all_created: list[str] = []
    try:
        # Run 1 (R3): Grouped classification with group_col=patient_id
        r1_st, r1_data = srv.post_json(
            "/api/run",
            {
                "upload_id": uid1,
                "task": "classification",
                "predict": ["target"],
                "group_col": "patient_id",
                "simple": True,
                "skip_benchmark": True,
            },
        )
        assert r1_st == 200, r1_data
        fin1, lines1, _ = poll_job_until_done(lambda cur: srv.get_json(f"/api/job/{r1_data['job_id']}?cursor={cur}"), timeout=150.0)
        assert fin1["status"] == "finished" and fin1["returncode"] == 0, "\n".join(lines1[-20:])
        paths1 = [a["path"] for a in fin1["artifacts"]]
        all_created.extend(paths1)
        comb1 = next(p for p in paths1 if p.endswith("_combined.json"))

        st_sum1, sum1 = srv.get_json(f"/api/report-summary?path={comb1}")
        assert st_sum1 == 200 and sum1["meta"]["task"] == "classification"

        # Run 2 (R1): Unsupervised clustering with empty target
        r2_st, r2_data = srv.post_json(
            "/api/run",
            {
                "upload_id": uid2,
                "task": "clustering",
                "predict": [],
                "target": [],
                "cluster_algo": "kmeans",
                "n_clusters": 3,
                "simple": True,
                "skip_benchmark": True,
            },
        )
        assert r2_st == 200, r2_data
        fin2, lines2, _ = poll_job_until_done(lambda cur: srv.get_json(f"/api/job/{r2_data['job_id']}?cursor={cur}"), timeout=150.0)
        assert fin2["status"] == "finished" and fin2["returncode"] == 0, "\n".join(lines2[-20:])
        paths2 = [a["path"] for a in fin2["artifacts"]]
        all_created.extend(paths2)
        comb2 = next(p for p in paths2 if p.endswith("_combined.json"))

        st_sum2, sum2 = srv.get_json(f"/api/report-summary?path={comb2}")
        assert st_sum2 == 200 and sum2["meta"]["task"] == "clustering"
        assert isinstance(sum2.get("clustering"), dict) and len(sum2["clustering"]) >= 1

        st_cmp, cmp_data = srv.post_json("/api/compare-reports", {"paths": [comb1, comb2]})
        assert st_cmp == 200, cmp_data
        assert cmp_data["baseline"] == comb1
        assert len(cmp_data["reports"]) == 2
        assert cmp_data["reports"][0]["task"] == "classification"
        assert cmp_data["reports"][1]["task"] == "clustering"
        assert isinstance(cmp_data["reports"][1].get("deltas"), dict)
    finally:
        for rel in all_created:
            srv.delete_json(f"/api/reports?path={rel}")
