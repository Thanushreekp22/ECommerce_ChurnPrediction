import io

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.feature_mapper import canonical_schema
from backend.trainer import train_model


client = TestClient(app)


def _unlabeled_csv():
    frame = pd.DataFrame({
        "user_id": ["U001", "U002", "U003"],
        "purchase_count": [2, 4, 1],
        "lifetime_value": [20.0, 80.0, 10.0],
        "last_order": ["2026-08-01", "2026-08-10", "2026-08-12"],
    })
    content = io.BytesIO()
    frame.to_csv(content, index=False)
    content.seek(0)
    return content


def test_unlabeled_prediction_rejects_labeled_input():
    frame = pd.DataFrame({
        "user_id": ["U001", "U002", "U003"],
        "purchase_count": [2, 4, 1],
        "lifetime_value": [20.0, 80.0, 10.0],
        "last_order": ["2026-08-01", "2026-08-10", "2026-08-12"],
        "Churn": [0, 1, 0],
    })
    labeled_file = io.BytesIO()
    frame.to_csv(labeled_file, index=False)
    labeled_file.seek(0)

    response = client.post(
        "/predict/unlabeled",
        files={"file": ("labeled.csv", labeled_file, "text/csv")},
    )

    assert response.status_code == 422
    assert "labeled analysis flow" in response.json()["detail"]


def test_unlabeled_prediction_never_trains_without_compatible_model():
    response = client.post(
        "/predict/unlabeled",
        files={"file": ("customers.csv", _unlabeled_csv(), "text/csv")},
    )

    assert response.status_code == 422
    assert "No compatible churn model" in response.json()["detail"]


def test_unlabeled_prediction_scores_two_column_variants_without_target(tmp_path, monkeypatch):
    rows = 40
    labeled = pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(rows)],
        "purchase_count": [1, 2, 4, 8] * (rows // 4),
        "lifetime_value": [20.0, 50.0, 120.0, 300.0] * (rows // 4),
        "last_order": pd.date_range("2026-01-01", periods=rows, freq="D"),
        "Churn": [0, 1] * (rows // 2),
    })
    canonical, schema = canonical_schema(labeled, "phase-3-test")
    model_dir = tmp_path / schema["dataset_hash"]
    train_model(canonical, schema, model_dir)
    monkeypatch.setattr("backend.main.ARTIFACTS_BASE", tmp_path)

    variants = [
        labeled.drop(columns="Churn"),
        labeled.drop(columns="Churn").rename(columns={
            "user_id": "client_id",
            "purchase_count": "number_of_orders",
            "lifetime_value": "amount_spent",
            "last_order": "recent_transaction_date",
        }),
    ]
    for variant in variants:
        content = io.BytesIO()
        variant.to_csv(content, index=False)
        content.seek(0)
        response = client.post(
            "/predict/unlabeled",
            files={"file": ("customers.csv", content, "text/csv")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["dataset_type"] == "unlabeled"
        assert len(body["customers"]) == rows
        assert body["customers"][0]["top_shap_features"]