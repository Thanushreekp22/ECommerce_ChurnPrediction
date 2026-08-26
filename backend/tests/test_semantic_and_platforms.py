from pathlib import Path

import pandas as pd

from backend.integrations.platforms.mock import MockStoreAdapter
from backend.integrations.platform_sync import sync_platform, platform_status
from backend.semantic_mapper import semantic_column_map


def test_semantic_mapper_recognises_unseen_aliases():
    assert semantic_column_map(["customer_email"])["customer_email"]["canonical"] == "customer_id"
    assert semantic_column_map(["member_joined"])["member_joined"]["canonical"] == "signup_date"
    assert semantic_column_map(["total_revenue"])["total_revenue"]["canonical"] == "total_spend"
    assert semantic_column_map(["last_activity_date"])["last_activity_date"]["canonical"] == "last_purchase_date"


def test_semantic_mapper_rejects_unrelated_columns():
    for column in ["product", "category", "color", "ship_address", "billing_email", "sku"]:
        assert column not in semantic_column_map([column])


def test_semantic_mapper_handles_empty_input():
    assert semantic_column_map([]) == {}


def test_mock_store_produces_canonical_transactions():
    adapter = MockStoreAdapter(order_count=5, seed=3)
    pulled = adapter.sync()
    assert pulled["raw_orders"] == 5
    assert len(pulled["transactions"]) == 5
    for transaction in pulled["transactions"]:
        assert transaction.order_id
        assert transaction.customer_id.endswith("@example.com")
        assert transaction.order_date
        assert transaction.amount > 0


def test_platform_sync_mock_imports_into_db(tmp_path):
    db = tmp_path / "integration.sqlite3"
    result = sync_platform("mock", db, adapter=MockStoreAdapter(order_count=4, seed=1))
    assert result["status"] == "completed"
    assert result["orders_imported"] == 4
    assert result["source"] == "platform:mock"


def test_platform_status_reports_mock_configured():
    status = platform_status(["shopify", "woocommerce", "mock"])
    assert status["mock"]["configured"] is True