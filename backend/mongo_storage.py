"""MongoDB persistence for server-side retention strategy state."""

from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient

from utils.config import MONGO_DB_NAME, MONGO_URI


def _collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    collection = client[MONGO_DB_NAME]["applied_strategies"]
    collection.create_index(
        [("customer_id", ASCENDING), ("strategy_key", ASCENDING)],
        unique=True,
    )
    collection.create_index([("applied_at", DESCENDING)])
    return client, collection


def record_strategy_applied(
    customer_ids: list[str],
    strategy_key: str,
    action: str | None,
    applied: bool = True,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    client, collection = _collection()
    try:
        changed = 0
        for customer_id in customer_ids:
            if not customer_id:
                continue
            query = {"customer_id": customer_id, "strategy_key": strategy_key}
            if applied:
                collection.update_one(
                    query,
                    {"$set": {"action": action, "applied_at": now}},
                    upsert=True,
                )
            else:
                collection.delete_one(query)
            changed += 1
        return {
            "applied": applied,
            "strategy_key": strategy_key,
            "customers_changed": changed,
        }
    finally:
        client.close()


def applied_for_customer(customer_id: str) -> list[dict[str, Any]]:
    client, collection = _collection()
    try:
        return [
            {key: value for key, value in row.items() if key != "_id"}
            for row in collection.find(
                {"customer_id": customer_id},
                {"_id": 0},
            ).sort("applied_at", DESCENDING)
        ]
    finally:
        client.close()


def applied_records(limit: int = 20000) -> list[dict[str, Any]]:
    client, collection = _collection()
    try:
        return [
            {key: value for key, value in row.items() if key != "_id"}
            for row in collection.find({}, {"_id": 0}).sort("applied_at", DESCENDING).limit(limit)
        ]
    finally:
        client.close()


def applied_summary() -> dict[str, Any]:
    client, collection = _collection()
    try:
        rows = list(collection.aggregate([
            {
                "$group": {
                    "_id": "$strategy_key",
                    "customers": {"$sum": 1},
                    "last_applied": {"$max": "$applied_at"},
                }
            },
            {"$sort": {"customers": -1}},
        ]))
        by_strategy = {
            row["_id"]: {
                "customers": row["customers"],
                "last_applied": row["last_applied"],
            }
            for row in rows
        }
        unique_customers = len(collection.distinct("customer_id"))
        return {
            "total_applied_records": sum(row["customers"] for row in rows),
            "unique_customers": unique_customers,
            "by_strategy": by_strategy,
        }
    finally:
        client.close()