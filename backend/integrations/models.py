from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CanonicalTransaction:
    order_id: str
    customer_id: str
    order_date: str
    amount: float
    quantity: float = 1.0
    product_id: str | None = None
    product_name: str | None = None
    category: str | None = None
    currency: str | None = None
    status: str = "completed"
    source_updated_at: str | None = None  # platform's own last-modified time; drives incremental syncs
    phone: str | None = None  # customer phone (used for WhatsApp outreach)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
