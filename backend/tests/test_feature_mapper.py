import pandas as pd
import numpy as np
from pathlib import Path

from backend.feature_mapper import canonicalize_dataset, map_dataset
from backend.feature_mapper import canonical_schema
from backend.trainer import load_artifacts, train_model


def test_dataset_aliases_map_to_canonical_features():
    frame = pd.DataFrame({
        "customer_id": ["C001", "C002"],
        "orders": [2, 4],
        "revenue": [20.0, 80.0],
        "last_purchase": ["2026-08-01", "2026-08-10"],
    })

    result = map_dataset(frame)
    mapped = canonicalize_dataset(frame)

    assert result["missing_features"] == []
    assert {item["canonical"] for item in result["mappings"]} == {
        "customer_id", "total_orders", "total_spend", "last_purchase_date"
    }
    assert "recency" in mapped.columns


def test_alternate_aliases_have_same_canonical_representation():
    frame = pd.DataFrame({
        "user_id": ["U001", "U002"],
        "purchase_count": [2, 4],
        "lifetime_value": [20.0, 80.0],
        "last_order_date": ["2026-08-01", "2026-08-10"],
    })

    result = map_dataset(frame)

    assert result["missing_features"] == []
    assert result["compatibility_score"] == 1.0
    assert result["dataset_type"] == "unlabeled"


def test_client_and_recent_transaction_aliases_are_supported():
    frame = pd.DataFrame({
        "client": ["A", "B"],
        "number_of_orders": [1, 3],
        "amount_spent": [10.0, 30.0],
        "recent_transaction": ["2026-08-01", "2026-08-10"],
    })

    result = map_dataset(frame)

    assert result["missing_features"] == []


def test_unrelated_dataset_is_not_declared_compatible():
    frame = pd.DataFrame({"product": ["a", "b"], "category": ["x", "y"], "color": ["red", "blue"]})

    result = map_dataset(frame)

    assert result["missing_features"] == ["customer_id", "total_orders", "total_spend"]
    assert result["compatibility_score"] == 0.0
    assert result["mappings"] == []


def test_type_validation_rejects_non_numeric_order_alias():
    frame = pd.DataFrame({
        "customer_id": ["C001", "C002"],
        "orders": ["many", "few"],
        "revenue": [20.0, 80.0],
    })

    result = map_dataset(frame)

    assert "total_orders" in result["missing_features"]
    assert any("incompatible" in warning for warning in result["warnings"])


def test_canonical_model_contract_trains_and_reloads_in_isolated_artifacts(tmp_path):
    rows = 40
    frame = pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(rows)],
        "purchase_count": np.tile([1, 2, 4, 8], rows // 4),
        "lifetime_value": np.tile([20.0, 50.0, 120.0, 300.0], rows // 4),
        "last_order_date": pd.date_range("2026-01-01", periods=rows, freq="D"),
        "Churn": np.tile([0, 1], rows // 2),
    })
    canonical, schema = canonical_schema(frame, "isolated-test")

    result = train_model(canonical, schema, Path(tmp_path))
    artifacts = load_artifacts(Path(tmp_path))

    assert result["dataset_hash"] == schema["dataset_hash"]
    assert artifacts["schema"]["canonical"] is True
    assert len(artifacts["pipeline"].predict_proba(canonical[schema["feature_cols"]])) == rows
