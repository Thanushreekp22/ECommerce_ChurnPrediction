"""
backend/integrations/platforms/woocommerce.py

WooCommerce REST API order puller (basic auth).

Uses only environment-driven configuration:
    CHURNIQ_WOO_URL       e.g. https://store.example.com/wp-json/wc/v3
    CHURNIQ_WOO_KEY       consumer key
    CHURNIQ_WOO_SECRET    consumer secret
    CHURNIQ_WOO_SINCE     ISO-8601 cursor; omit for all orders
"""

import os
from datetime import datetime, timezone
from typing import Any

import requests

from backend.integrations.models import CanonicalTransaction


def woocommerce_config() -> dict:
    return {
        "url": os.getenv("CHURNIQ_WOO_URL", "").strip(),
        "key": os.getenv("CHURNIQ_WOO_KEY", "").strip(),
        "secret": os.getenv("CHURNIQ_WOO_SECRET", "").strip(),
        "since": os.getenv("CHURNIQ_WOO_SINCE", "").strip(),
    }


class WooCommerceAdapter:
    """Minimal WooCommerce REST client (iterate /orders)."""

    name = "woocommerce"

    def __init__(self, url: str, key: str, secret: str, since: str = ""):
        if not url or not key or not secret:
            raise ValueError("WooCommerce integration requires CHURNIQ_WOO_URL, CHURNIQ_WOO_KEY and CHURNIQ_WOO_SECRET.")
        self.base = url.rstrip("/")
        self.auth = (key, secret)
        self.since = since

    def fetch_orders(self) -> list[dict]:
        url = f"{self.base}/orders"
        params: dict[str, Any] = {"per_page": 100, "page": 1, "orderby": "modified", "order": "desc"}
        if self.since:
            params["modified_after"] = self.since  # incremental sync cursor
        all_orders: list[dict] = []
        for page in range(1, 51):
            params["page"] = page
            response = requests.get(url, auth=self.auth, params=params, timeout=30)
            if response.status_code == 404 and page > 1:
                break
            if response.status_code != 200:
                raise RuntimeError(f"WooCommerce API returned {response.status_code}: {response.text[:300]}")
            orders = response.json()
            all_orders.extend(orders)
            if len(orders) < params["per_page"]:
                break
        return all_orders

    def to_transactions(self, orders: list[dict]) -> list[CanonicalTransaction]:
        transactions: list[CanonicalTransaction] = []
        for order in orders:
            order_id = str(order.get("id") or order.get("number") or "").strip()
            billing = order.get("billing") or {}
            customer_id = str(billing.get("email") or billing.get("user_id") or order.get("customer_id") or "") or order_id
            if not order_id or not customer_id:
                continue
            date = order.get("date_created") or order.get("date_created_gmt") or datetime.now(timezone.utc).isoformat()
            status = str(order.get("status") or "completed").lower()
            status = "refund" if status == "refunded" else ("cancelled" if status in ("cancelled", "failed", "trash") else "completed")
            try:
                amount = float(order.get("total") or 0.0)
            except (TypeError, ValueError):
                amount = 0.0
            for item in order.get("line_items", []) or [{}]:
                transactions.append(CanonicalTransaction(
                    order_id=order_id,
                    customer_id=customer_id,
                    order_date=date,
                    amount=amount if len(order.get("line_items", [])) <= 1 else float(item.get("total") or amount),
                    quantity=float(item.get("quantity") or 1),
                    product_id=str(item.get("product_id") or ""),
                    product_name=str(item.get("name") or "Unknown"),
                    category=str(item.get("category") or ""),
                    currency=str(order.get("currency") or "USD").upper() or "USD",
                    source_updated_at=str(order.get("date_modified") or date or ""),
                    status=status,
                ))
        return transactions

    def sync(self) -> dict:
        orders = self.fetch_orders()
        transactions = self.to_transactions(orders)
        return {"raw_orders": len(orders), "transactions": transactions}


def build_woocommerce_adapter(since: str | None = None) -> WooCommerceAdapter:
    cfg = woocommerce_config()
    return WooCommerceAdapter(url=cfg["url"], key=cfg["key"], secret=cfg["secret"], since=since if since is not None else cfg["since"])