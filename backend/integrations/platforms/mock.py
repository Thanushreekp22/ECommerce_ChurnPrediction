"""
backend/integrations/platforms/mock.py

Offline demo store adapter so the whole platform-integration flow can be
demonstrated end-to-end without any live credentials.  Generates a small
realistic order stream every time it runs (deterministic for the demo date).
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.integrations.models import CanonicalTransaction

CUSTOMERS = [
    ("alice@example.com", "+15550000001"),
    ("bob@example.com",   "+15550000002"),
    ("carol@example.com", "+15550000003"),
    ("dave@example.com",  "+15550000004"),
    ("erin@example.com",  "+15550000005"),
    ("frank@example.com", "+15550000006"),
]
PHONE_BY_EMAIL = dict(CUSTOMERS)

PRODUCTS = [
    ("P1", "Wireless Headphones", "Electronics", 89.0),
    ("P2", "USB-C Cable", "Electronics", 12.5),
    ("P3", "Laptop Sleeve", "Accessories", 24.0),
    ("P4", "Desk Lamp", "Home", 39.0),
    ("P5", "Coffee Mug", "Kitchen", 15.0),
    ("P6", "Phone Case", "Accessories", 19.0),
]


class MockStoreAdapter:
    """Deterministic mock store that produces a handful of fresh orders."""

    name = "mock"

    def __init__(self, order_count: int = 12, seed: int = 7):
        self.order_count = order_count
        self.seed = seed

    def fetch_orders(self) -> list[dict]:
        rng = random.Random(self.seed)
        today = datetime.now(timezone.utc)
        orders = []
        for i in range(self.order_count):
            email = CUSTOMERS[rng.randrange(len(CUSTOMERS))][0]
            created = today - timedelta(days=rng.randint(0, 45), hours=rng.randint(0, 23))
            lines = []
            for _ in range(rng.randint(1, 3)):
                product_id, name, category, price = rng.choice(PRODUCTS)
                qty = rng.randint(1, 3)
                lines.append({
                    "product_id": product_id,
                    "name": name,
                    "category": category,
                    "price": price,
                    "quantity": qty,
                })
            total = round(sum(float(line["price"]) * int(line["quantity"]) for line in lines), 2)
            orders.append({
                "id": f"MOCK-{1000 + i}",
                "email": email,
                "phone": PHONE_BY_EMAIL[email],
                "created_at": created.isoformat(),
                "currency": "USD",
                "total_price": total,
                "financial_status": "paid",
                "line_items": lines,
            })
        return orders

    def to_transactions(self, orders: list[dict]) -> list[CanonicalTransaction]:
        transactions: list[CanonicalTransaction] = []
        for order in orders:
            order_id = str(order["id"])
            customer_id = order["email"]
            transactions.append(CanonicalTransaction(
                order_id=order_id,
                customer_id=customer_id,
                order_date=order["created_at"],
                amount=float(order["total_price"]),
                quantity=1,
                product_id=str(order["line_items"][0]["product_id"]),
                product_name=order["line_items"][0]["name"],
                category=order["line_items"][0]["category"],
                currency="USD",
                status="completed",
                source_updated_at=order["created_at"],
                phone=order.get("phone"),
            ))
        return transactions

    def sync(self) -> dict:
        orders = self.fetch_orders()
        transactions = self.to_transactions(orders)
        return {"raw_orders": len(orders), "transactions": transactions}


def build_mock_adapter() -> MockStoreAdapter:
    return MockStoreAdapter()