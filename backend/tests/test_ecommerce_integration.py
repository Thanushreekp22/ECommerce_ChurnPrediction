import io

import pandas as pd
from fastapi.testclient import TestClient

from backend.integrations.feature_builder import build_customer_features
from backend.integrations.normalizer import normalize_transactions
from backend.integrations.schema import analyze_transaction_schema
from backend.integrations.storage import import_transactions
from backend.main import app


def _transactions():
    return pd.DataFrame({
        "Order ID": ["O1", "O2", "O3", "O2", "O4"],
        "Customer": ["C1", "C1", "C2", "C1", "C3"],
        "Purchase Date": ["2026-01-01", "2026-01-11", "2026-01-21", "2026-01-11", "bad-date"],
        "Total": [100, 200, -50, 200, "invalid"],
        "Qty": [1, 2, 1, 2, 1],
        "Product": ["Phone", "Case", "Refund", "Case", "Broken"],
    })


def test_transaction_schema_maps_aliases_and_reports_duplicates():
    analysis = analyze_transaction_schema(_transactions())

    assert analysis["missing_required"] == []
    assert analysis["mapping"]["customer_id"]["source"] == "Customer"
    assert analysis["mapping"]["amount"]["source"] == "Total"
    assert any("duplicate order IDs" in warning for warning in analysis["warnings"])


def test_normalizer_classifies_refunds_and_rejects_invalid_rows():
    transactions, result = normalize_transactions(_transactions(), analyze_transaction_schema(_transactions()))

    assert len(transactions) == 4
    assert any(transaction.status == "refund" for transaction in transactions)
    assert result["invalid_dates"] == 1


def test_storage_upserts_orders_and_builds_customer_features(tmp_path):
    frame = _transactions().iloc[:3]
    transactions, validation = normalize_transactions(frame, analyze_transaction_schema(frame))
    database = tmp_path / "integration.sqlite3"

    first = import_transactions(database, "orders.csv", len(frame), transactions, validation)
    second = import_transactions(database, "orders.csv", len(frame), transactions, validation)
    features = build_customer_features(database)

    c1 = features.loc[features.customer_id == "C1"].iloc[0]
    assert first["orders_imported"] == 3
    assert second["orders_imported"] == 0
    assert second["duplicates"] == 3
    assert c1.total_orders == 2
    assert c1.total_spend == 300
    assert c1.average_order_value == 150
    assert c1.frequency == 2
    assert c1.monetary == 300


def test_integration_api_preview_import_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.main.INTEGRATION_DB_PATH", tmp_path / "integration.sqlite3")
    client = TestClient(app)
    content = io.BytesIO()
    _transactions().iloc[:3].to_csv(content, index=False)
    content.seek(0)

    preview = client.post("/integrations/ecommerce/analyze", files={"file": ("orders.csv", content, "text/csv")})
    assert preview.status_code == 200
    content.seek(0)
    imported = client.post("/integrations/ecommerce/import", files={"file": ("orders.csv", content, "text/csv")})
    assert imported.status_code == 200
    assert imported.json()["orders_imported"] == 3
    summary = client.get("/integrations/ecommerce/summary")
    assert summary.status_code == 200
    assert summary.json()["customers"] == 2
