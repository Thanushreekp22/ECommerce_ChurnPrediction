from typing import Any

import pandas as pd

from backend.integrations.models import CanonicalTransaction


def _number(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).replace(",", "")
    text = "".join(char for char in text if char.isdigit() or char in ".-")
    try:
        return float(text)
    except ValueError:
        return None


def normalize_transactions(df: pd.DataFrame, analysis: dict) -> tuple[list[CanonicalTransaction], dict]:
    mapping = analysis["mapping"]
    if analysis["missing_required"]:
        raise ValueError(f"Missing required transaction fields: {', '.join(analysis['missing_required'])}")
    transactions: list[CanonicalTransaction] = []
    invalid_dates = invalid_amounts = missing_identity = negative_amounts = 0
    optional = lambda field, row: row[mapping[field]["source"]] if field in mapping else None
    for _, row in df.iterrows():
        order_id, customer_id = str(optional("order_id", row)).strip(), str(optional("customer_id", row)).strip()
        date = pd.to_datetime(optional("order_date", row), errors="coerce")
        amount = _number(optional("amount", row))
        if not order_id or order_id.lower() == "nan" or not customer_id or customer_id.lower() == "nan":
            missing_identity += 1
            continue
        if pd.isna(date):
            invalid_dates += 1
            continue
        if amount is None:
            invalid_amounts += 1
            continue
        if amount < 0:
            negative_amounts += 1
        quantity = _number(optional("quantity", row)) or 1.0
        status = str(optional("status", row) or "completed").strip().lower()
        if amount < 0 and status == "completed":
            status = "refund"
        transactions.append(CanonicalTransaction(
            order_id=order_id, customer_id=customer_id, order_date=date.isoformat(), amount=amount,
            quantity=quantity, product_id=_text(optional("product_id", row)),
            product_name=_text(optional("product_name", row)), category=_text(optional("category", row)),
            currency=_text(optional("currency", row)), status=status,
        ))
    return transactions, {
        "rows_valid": len(transactions), "rows_failed": invalid_dates + invalid_amounts + missing_identity,
        "invalid_dates": invalid_dates, "invalid_amounts": invalid_amounts,
        "missing_identity": missing_identity, "negative_amounts": negative_amounts,
    }


def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
