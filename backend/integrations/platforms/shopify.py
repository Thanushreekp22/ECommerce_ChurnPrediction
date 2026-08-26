"""
backend/integrations/platforms/shopify.py

Shopify REST Admin API order puller.

Uses only environment-driven configuration (never hardcoded secrets):
    CHURNIQ_SHOPIFY_SHOP      e.g. my-store.myshopify.com
    CHURNIQ_SHOPIFY_TOKEN     Admin API access token
    CHURNIQ_SHOPIFY_SINCE     ISO-8601 cursor; omit for all orders

Converts each Shopify order into the canonical transaction dataclass so the
existing importer/feature-builder can consume it.
"""

import os
from datetime import datetime, timezone
from typing import Any

import requests

from backend.integrations.models import CanonicalTransaction


def shopify_config() -> dict:
    return {
        "shop": os.getenv("CHURNIQ_SHOPIFY_SHOP", "").strip(),
        "token": os.getenv("CHURNIQ_SHOPIFY_TOKEN", "").strip(),
        "since": os.getenv("CHURNIQ_SHOPIFY_SINCE", "").strip(),
    }


class ShopifyAdapter:
    """Minimal Shopify Admin API client (iterate Orders)."""

    name = "shopify"

    def __init__(self, shop: str, token: str, since: str = ""):
        if not shop or not token:
            raise ValueError("Shopify integration requires CHURNIQ_SHOPIFY_SHOP and CHURNIQ_SHOPIFY_TOKEN.")
        self.shop = shop.rstrip("/")
        if not self.shop.startswith("https://"):
            self.shop = "https://" + self.shop
        self.token = token
        self.since = since

    # ─── API helpers ──────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "X-Shopify-Access-Token": self.token,
            "Content-Type": "application/json",
        }

    def fetch_orders(self) -> list[dict]:
        """Fetch orders (newest first) with paginated/cursor handling."""
        url = f"{self.shop}/admin/api/2024-01/orders.json"
        params = {"status": "any", "limit": 250, "order": "updated_at ASC"}
        if self.since:
            params["updated_at_min"] = self.since  # incremental: only what changed since last sync

        all_orders: list[dict] = []
        page = 0
        while url and page < 50:
            response = requests.get(url, headers=self._headers(), params=params, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Shopify API returned {response.status_code}: {response.text[:300]}"
                )
            payload = response.json()
            orders = payload.get("orders", [])
            all_orders.extend(orders)
            link = response.headers.get("Link", "")
            next_url = self._extract_next(link)
            if not next_url or not orders:
                break
            url = next_url
            params = None
            page += 1
        return all_orders

    @staticmethod
    def _extract_next(link_header: str) -> str | None:
        for part in link_header.split(","):
            if 'rel="next"' in part:
                return part[part.index("<") + 1 : part.index(">")]
        return None

    # ─── Conversion ───────────────────────────────────────────────────────────

    def to_transactions(self, orders: list[dict]) -> list[CanonicalTransaction]:
        transactions: list[CanonicalTransaction] = []
        for order in orders:
            order_id = str(order.get("id") or order.get("name") or "").strip()
            customer = order.get("customer") or {}
            email = customer.get("email") or order.get("email") or ""
            customer_id = email or str(customer.get("id") or "") or order_id
            if not order_id or not customer_id:
                continue
            created = order.get("created_at")
            date = created.replace("Z", "+00:00") if created else datetime.now(timezone.utc).isoformat()
            financial = order.get("financial_status", "").lower()
            status = "refund" if financial == "refunded" else ("completed" if financial in ("paid", "authorized", "") else "cancelled")
            try:
                amount = float(order.get("total_price") or 0.0)
            except (TypeError, ValueError):
                amount = 0.0
            line_items = order.get("line_items", [])
            for item in line_items or [{}]:
                transactions.append(CanonicalTransaction(
                    order_id=order_id,
                    customer_id=customer_id,
                    order_date=date,
                    amount=amount if len(line_items) <= 1 else float(item.get("price") or amount),
                    quantity=float(item.get("quantity") or 1),
                    product_id=str(item.get("product_id") or ""),
                    product_name=str(item.get("title") or "Unknown"),
                    source_updated_at=str(order.get("updated_at") or created or ""),
                    category=str(item.get("product_type") or ""),
                    currency=str(order.get("currency") or "USD").upper() or "USD",
                    status=status,
                ))
        return transactions

    def sync(self) -> dict:
        orders = self.fetch_orders()
        transactions = self.to_transactions(orders)
        return {"raw_orders": len(orders), "transactions": transactions}


def build_shopify_adapter(since: str | None = None) -> ShopifyAdapter:
    cfg = shopify_config()
    return ShopifyAdapter(shop=cfg["shop"], token=cfg["token"], since=since if since is not None else cfg["since"])