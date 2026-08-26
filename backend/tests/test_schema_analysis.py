import io

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_schema_analysis_returns_mapping_and_dataset_type():
    frame = pd.DataFrame({
        "user_id": ["U001", "U002"],
        "purchase_count": [2, 4],
        "lifetime_value": [20.0, 80.0],
        "last_order_date": ["2026-08-01", "2026-08-10"],
        "Churn": [0, 1],
    })
    content = io.BytesIO()
    frame.to_csv(content, index=False)
    content.seek(0)

    response = client.post(
        "/schema/analyze",
        files={"file": ("customers.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_type"] == "labeled"
    assert body["target_column"] == "Churn"
    assert body["compatibility_score"] == 1.0


def test_schema_analysis_rejects_empty_upload():
    response = client.post(
        "/schema/analyze",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unable to read dataset."


def test_unlabeled_canonical_dataset_never_starts_training_without_model():
    frame = pd.DataFrame({
        "user_id": ["U001", "U002", "U003"],
        "purchase_count": [2, 4, 1],
        "lifetime_value": [20.0, 80.0, 10.0],
        "last_order_date": ["2026-08-01", "2026-08-10", "2026-08-12"],
    })
    content = io.BytesIO()
    frame.to_csv(content, index=False)
    content.seek(0)

    response = client.post(
        "/predict/batch-analyze",
        files={"file": ("unlabeled-canonical.csv", content, "text/csv")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] in {"No compatible model.", "Not enough usable features."}
