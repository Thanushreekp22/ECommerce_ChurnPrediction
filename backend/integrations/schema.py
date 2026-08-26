import re
from typing import Any

import pandas as pd


TRANSACTION_ALIASES = {
    "order_id": ["order_id", "orderid", "transaction_id", "transactionid", "purchase_id", "id"],
    "customer_id": ["customer_id", "customerid", "customer", "user_id", "userid", "client_id", "buyer_id"],
    "order_date": ["order_date", "orderdate", "purchase_date", "transaction_date", "created_at", "purchase_datetime"],
    "amount": ["amount", "order_amount", "order_value", "total", "total_amount", "purchase_amount", "revenue"],
    "quantity": ["quantity", "qty", "items", "item_count", "units"],
    "product_id": ["product_id", "productid", "sku", "item_id"],
    "product_name": ["product_name", "product", "item_name", "title"],
    "category": ["category", "product_category", "collection"],
    "currency": ["currency", "currency_code"],
    "status": ["status", "order_status", "payment_status"],
}
REQUIRED_FIELDS = ("order_id", "customer_id", "order_date", "amount")


def normalize_column_name(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())).strip("_")


def _is_numeric(series: pd.Series) -> bool:
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").notna().mean() >= 0.8


def _is_date(series: pd.Series) -> bool:
    return pd.to_datetime(series, errors="coerce", format="mixed").notna().mean() >= 0.8


def analyze_transaction_schema(df: pd.DataFrame) -> dict[str, Any]:
    alias_map = {normalize_column_name(alias): field for field, aliases in TRANSACTION_ALIASES.items() for alias in aliases}
    mapping: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    used_fields: set[str] = set()
    for column in df.columns:
        field = alias_map.get(normalize_column_name(column))
        if not field or field in used_fields:
            continue
        if field in {"amount", "quantity"} and not _is_numeric(df[column]):
            warnings.append(f"'{column}' was not mapped to '{field}' because values are not predominantly numeric.")
            continue
        if field == "order_date" and not _is_date(df[column]):
            warnings.append(f"'{column}' was not mapped to 'order_date' because values are not predominantly dates.")
            continue
        mapping[field] = {"source": column, "confidence": 0.99 if field in REQUIRED_FIELDS else 0.95}
        used_fields.add(field)
    missing = [field for field in REQUIRED_FIELDS if field not in mapping]
    if "order_id" in mapping:
        duplicate_ids = int(df[mapping["order_id"]["source"]].duplicated(keep=False).sum())
        if duplicate_ids:
            warnings.append(f"{duplicate_ids} rows have duplicate order IDs; imports will upsert by order ID.")
    return {
        "integration_type": "ecommerce_transactions",
        "rows": len(df),
        "columns": len(df.columns),
        "mapping": mapping,
        "missing_required": missing,
        "warnings": warnings,
        "ready_to_import": not missing,
    }
