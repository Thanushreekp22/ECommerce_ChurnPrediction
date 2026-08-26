"""
backend/integrations/platform_sync.py

Orchestrates pulling orders from a connected e-commerce platform (Shopify,
WooCommerce or the offline mock store) and importing them through the same
normalisation / storage pipeline used by CSV uploads, so platform purchases
flow into customer features and churn scoring automatically.
"""

from pathlib import Path

from backend.integrations.platforms import (
    ShopifyAdapter,
    WooCommerceAdapter,
    MockStoreAdapter,
)
from backend.integrations.models import CanonicalTransaction
from backend.integrations.storage import import_transactions, latest_platform_cursor


def platform_adapter(platform: str, since: str | None = None):
    platform = (platform or "").strip().lower()
    if platform == "shopify":
        from backend.integrations.platforms.shopify import build_shopify_adapter
        return build_shopify_adapter(since=since)
    if platform == "woocommerce":
        from backend.integrations.platforms.woocommerce import build_woocommerce_adapter
        return build_woocommerce_adapter(since=since)
    if platform in ("mock", "demo", "mock-store"):
        from backend.integrations.platforms.mock import build_mock_adapter
        return build_mock_adapter()
    raise ValueError(f"Unsupported platform '{platform}'. Use 'shopify', 'woocommerce' or 'mock'.")


def sync_platform(
    platform: str,
    db_path: Path,
    validator=None,
    adapter=None,
) -> dict:
    """
    Pull orders from a platform, convert them to canonical transactions and
    import them into the integration database.

    Returns an import result dict (see storage.import_transactions) enriched
    with the raw order counts.
    """
    source_name = f"platform:{platform}"
    try:
        cursor = latest_platform_cursor(db_path) if adapter is None else None
        adapter = adapter or platform_adapter(platform, since=cursor)
        pulled = adapter.sync()
        transactions: list[CanonicalTransaction] = pulled.get("transactions", [])

        # Light validation for the mock path (real CSV normalizer already does
        # this, but platform adapters produce clean canonical rows).
        invalid = 0
        for t in transactions:
            if not t.order_id or not t.customer_id or not t.order_date:
                invalid += 1
        validation = {
            "rows_valid": len(transactions) - invalid,
            "rows_failed": invalid,
            "invalid_dates": 0,
            "invalid_amounts": 0,
            "missing_identity": invalid,
            "negative_amounts": 0,
        }
        return import_transactions(
            db_path=db_path,
            source_name=source_name,
            received=len(transactions),
            transactions=transactions,
            validation=validation,
        )
    except Exception as exc:
        return {
            "status": "error",
            "source": source_name,
            "error": str(exc),
        }


def platform_status(platforms: list[str]) -> dict:
    """Report which platforms are configured / available."""
    from backend.integrations.platforms.shopify import shopify_config
    from backend.integrations.platforms.woocommerce import woocommerce_config

    status = {}
    for platform in platforms:
        if platform == "shopify":
            cfg = shopify_config()
            status[platform] = {"configured": bool(cfg["shop"] and cfg["token"])}
        elif platform == "woocommerce":
            cfg = woocommerce_config()
            status[platform] = {"configured": bool(cfg["url"] and cfg["key"] and cfg["secret"])}
        elif platform in ("mock", "demo", "mock-store"):
            status[platform] = {"configured": True}
        else:
            status[platform] = {"configured": False}
    return status