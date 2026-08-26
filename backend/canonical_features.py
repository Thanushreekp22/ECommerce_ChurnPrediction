CANONICAL_FEATURES = {
    "customer_id": {
        "type": "identifier",
        "required": True,
        "aliases": [
            "customer_id", "customerid", "customer", "client", "client_id",
            "clientid", "user_id", "userid", "user", "cust_ref"
        ],
    },
    "total_orders": {
        "type": "numeric",
        "required": True,
        "aliases": [
            "total_orders", "orders", "order_count", "orders_count",
            "purchase_count", "number_of_orders", "total_purchases", "ordercount"
        ],
    },
    "total_spend": {
        "type": "numeric",
        "required": True,
        "aliases": [
            "total_spend", "total_revenue", "revenue", "lifetime_value",
            "customer_lifetime_value", "ltv", "amount_spent", "spend",
            "avg_order_value", "cashbackamount", "cashback_amount",
            "customer_cashback", "monthly_spend"
        ],
    },
    "last_purchase_date": {
        "type": "date",
        "required": False,
        "aliases": [
            "last_purchase_date", "last_purchase", "last_order_date",
            "last_order", "recent_purchase", "recent_transaction",
            "recent_transaction_date", "last_transaction_date"
        ],
    },
    "signup_date": {
        "type": "date",
        "required": False,
        "aliases": [
            "signup_date", "registration_date", "registered_date",
            "join_date", "created_at"
        ],
    },
    "age": {
        "type": "numeric",
        "required": False,
        "aliases": ["age", "customer_age"],
    },
}


CANONICAL_MODEL_FEATURE_ORDER = [
    "total_orders",
    "total_spend",
    "average_order_value",
    "recency",
    "tenure_days",
    "age",
]
