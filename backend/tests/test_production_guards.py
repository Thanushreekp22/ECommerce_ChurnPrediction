import io

import pandas as pd
from fastapi.testclient import TestClient

from backend.feature_mapper import canonical_schema
from backend.main import app
from backend.model_compatibility import find_compatible_model
from utils.schema_inferrer import infer_schema


client = TestClient(app)


def _labeled_frame(churn):
    return pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "orders": [1, 2, 3, 4, 5, 6],
        "revenue": [10, 20, 30, 40, 50, 60],
        "Churn": churn,
    })


def test_labeled_data_with_same_contract_gets_a_distinct_model_id():
    first, first_schema = canonical_schema(_labeled_frame([0, 1, 0, 1, 0, 1]), "first")
    second, second_schema = canonical_schema(_labeled_frame([1, 0, 1, 0, 1, 0]), "second")

    assert first_schema["dataset_hash"] != second_schema["dataset_hash"]
    assert list(first.columns) == list(second.columns)


def test_legacy_labeled_data_with_same_columns_gets_a_distinct_model_id():
    first = infer_schema(_labeled_frame([0, 1, 0, 1, 0, 1]), "first")
    second = infer_schema(_labeled_frame([1, 0, 1, 0, 1, 0]), "second")

    assert first["dataset_hash"] != second["dataset_hash"]


def test_low_quality_canonical_model_is_not_selectable(tmp_path):
    model_dir = tmp_path / "low-quality"
    model_dir.mkdir()
    (model_dir / "churn_pipeline.joblib").touch()
    (model_dir / "schema.json").write_text('{"canonical": true, "feature_cols": ["total_orders", "total_spend"]}')
    (model_dir / "feature_spec.json").write_text('{"features": ["total_orders", "total_spend"]}')
    (model_dir / "model_metadata.json").write_text('{"domain": "ecommerce", "auc_cv": 0.49}')

    assert find_compatible_model(tmp_path, ["total_orders", "total_spend"]) is None


def test_model_switch_is_disabled_without_admin_key():
    response = client.post("/model/switch/not-a-model")

    assert response.status_code == 403


def test_upload_limit_is_enforced(monkeypatch):
    monkeypatch.setattr("backend.main.MAX_UPLOAD_BYTES", 1)

    response = client.post(
        "/schema/analyze",
        files={"file": ("customers.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
    )

    assert response.status_code == 413
