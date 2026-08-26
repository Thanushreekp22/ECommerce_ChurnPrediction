import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.integrations.models import CanonicalTransaction


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    _create_schema(connection)
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
    CREATE TABLE IF NOT EXISTS integration_sources (
      source_id INTEGER PRIMARY KEY, source_type TEXT NOT NULL, source_name TEXT NOT NULL UNIQUE,
      status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_sync_at TEXT
    );
    CREATE TABLE IF NOT EXISTS customers (
      customer_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS orders (
      order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, order_date TEXT NOT NULL, amount REAL NOT NULL,
      quantity REAL NOT NULL, product_id TEXT, product_name TEXT, category TEXT, currency TEXT, status TEXT NOT NULL,
      source_id INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
      FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
      FOREIGN KEY(source_id) REFERENCES integration_sources(source_id)
    );
    CREATE TABLE IF NOT EXISTS applied_strategies (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      customer_id TEXT NOT NULL,
      strategy_key TEXT NOT NULL,
      action TEXT,
      applied_at TEXT NOT NULL,
      UNIQUE(customer_id, strategy_key)
    );
    CREATE TABLE IF NOT EXISTS sync_runs (
      sync_id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL, started_at TEXT NOT NULL, completed_at TEXT,
      rows_received INTEGER NOT NULL, rows_inserted INTEGER NOT NULL DEFAULT 0, rows_updated INTEGER NOT NULL DEFAULT 0,
      rows_failed INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, error_message TEXT,
      FOREIGN KEY(source_id) REFERENCES integration_sources(source_id)
    );
    """)
    # Earlier development builds created a smaller integration.sqlite3. Upgrade
    # those databases in-place before creating indexes or querying new fields.
    _ensure_columns(connection, "integration_sources", {
        "source_type": "TEXT", "source_name": "TEXT", "status": "TEXT",
        "created_at": "TEXT", "updated_at": "TEXT", "last_sync_at": "TEXT",
    })
    _ensure_columns(connection, "customers", {
        "customer_id": "TEXT", "created_at": "TEXT", "updated_at": "TEXT", "phone": "TEXT",
    })
    _ensure_columns(connection, "orders", {
        "order_id": "TEXT", "customer_id": "TEXT", "order_date": "TEXT", "amount": "REAL",
        "quantity": "REAL", "product_id": "TEXT", "product_name": "TEXT", "category": "TEXT",
        "currency": "TEXT", "status": "TEXT", "source_id": "INTEGER", "created_at": "TEXT", "updated_at": "TEXT", "source_updated_at": "TEXT",
    })
    _ensure_columns(connection, "sync_runs", {
        "source_id": "INTEGER", "started_at": "TEXT", "completed_at": "TEXT", "rows_received": "INTEGER",
        "rows_inserted": "INTEGER", "rows_updated": "INTEGER", "rows_failed": "INTEGER", "status": "TEXT", "error_message": "TEXT",
    })
    # Legacy databases may not have declared order_id as unique. Retain the
    # newest physical row per ID before adding the constraint required for
    # idempotent CSV imports.
    connection.execute("DELETE FROM orders WHERE order_id IS NOT NULL AND rowid NOT IN (SELECT MAX(rowid) FROM orders WHERE order_id IS NOT NULL GROUP BY order_id)")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_order_id_unique ON orders(order_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer_date ON orders(customer_id, order_date)")
    connection.commit()


def _ensure_columns(connection: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    for name, sql_type in columns.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


def import_transactions(db_path: Path, source_name: str, received: int, transactions: list[CanonicalTransaction], validation: dict) -> dict:
    """Bulk upsert orders/customers. Set-based so large platform syncs stay fast."""
    with connect(db_path) as connection:
        now = _now()
        connection.execute("INSERT INTO integration_sources(source_type, source_name, status, created_at, updated_at, last_sync_at) VALUES ('csv', ?, 'syncing', ?, ?, ?) ON CONFLICT(source_name) DO UPDATE SET status='syncing', updated_at=excluded.updated_at", (source_name, now, now, now))
        source_id = connection.execute("SELECT source_id FROM integration_sources WHERE source_name=?", (source_name,)).fetchone()[0]
        run_id = connection.execute("INSERT INTO sync_runs(source_id, started_at, rows_received, rows_failed, status) VALUES (?, ?, ?, ?, 'running')", (source_id, now, received, validation["rows_failed"])).lastrowid

        existing_orders = {row[0] for row in connection.execute("SELECT order_id FROM orders")}
        inserted = updated = duplicates = 0
        insert_rows, update_rows = [], []
        touched_customers = []
        seen_this_run = set()
        for t in transactions:
            if t.order_id in existing_orders or t.order_id in seen_this_run:
                updated += 1; duplicates += 1
                update_rows.append((t.customer_id, t.order_date, t.amount, t.quantity, t.product_id, t.product_name,
                                    t.category, t.currency, t.status, source_id, now, t.source_updated_at, t.order_id))
            else:
                existing_orders.add(t.order_id); seen_this_run.add(t.order_id)
                inserted += 1
                insert_rows.append((t.order_id, t.customer_id, t.order_date, t.amount, t.quantity, t.product_id,
                                    t.product_name, t.category, t.currency, t.status, source_id, now, now, t.source_updated_at))
            touched_customers.append(t.customer_id)

        # customers: bulk upsert timestamps + phone (for WhatsApp outreach)
        distinct = list(dict.fromkeys(touched_customers))
        phone_of = {t.customer_id: t.phone for t in transactions if t.phone}
        if distinct:
            ph = ",".join("?" for _ in distinct)
            known = {r[0] for r in connection.execute(f"SELECT customer_id FROM customers WHERE customer_id IN ({ph})", distinct)}
            newc = [(cid, now, now, phone_of.get(cid)) for cid in distinct if cid not in known]
            if newc:
                connection.executemany("INSERT INTO customers(customer_id, created_at, updated_at, phone) VALUES (?, ?, ?, ?)", newc)
            customers_created = len(newc); customers_updated = len(distinct) - len(newc)
            for cid in known:
                pnum = phone_of.get(cid)
                if pnum:
                    connection.execute("UPDATE customers SET phone=?, updated_at=? WHERE customer_id=?", (pnum, now, cid))
                else:
                    connection.execute("UPDATE customers SET updated_at=? WHERE customer_id=?", (now, cid))
        else:
            customers_created = customers_updated = 0

        if insert_rows:
            connection.executemany(
                "INSERT INTO orders(order_id, customer_id, order_date, amount, quantity, product_id, product_name, category, currency, status, source_id, created_at, updated_at, source_updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                insert_rows)
        if update_rows:
            connection.executemany(
                "UPDATE orders SET customer_id=?, order_date=?, amount=?, quantity=?, product_id=?, product_name=?, category=?, currency=?, status=?, source_id=?, updated_at=?, source_updated_at=? WHERE order_id=?",
                update_rows)

        completed = _now()
        connection.execute("UPDATE sync_runs SET completed_at=?, rows_inserted=?, rows_updated=?, status='completed' WHERE sync_id=?", (completed, inserted, updated, run_id))
        connection.execute("UPDATE integration_sources SET status='healthy', updated_at=?, last_sync_at=? WHERE source_id=?", (completed, completed, source_id))
        connection.commit()
    return {"status": "completed", "source": source_name, "rows_received": received, "orders_imported": inserted, "orders_updated": updated, "customers_created": customers_created, "customers_updated": customers_updated, "duplicates": duplicates, "errors": validation["rows_failed"], "validation": validation}


def customer_features(db_path: Path) -> pd.DataFrame:
    with connect(db_path) as connection:
        rows = connection.execute("""SELECT customer_id, COUNT(*) AS total_orders, SUM(amount) AS total_spend,
          AVG(amount) AS average_order_value, MIN(order_date) AS signup_date, MAX(order_date) AS last_purchase_date
          FROM orders WHERE status != 'refund' GROUP BY customer_id ORDER BY customer_id""").fetchall()
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        return frame
    latest = pd.to_datetime(frame["last_purchase_date"], errors="coerce").max()
    frame["recency"] = (latest - pd.to_datetime(frame["last_purchase_date"], errors="coerce")).dt.days
    frame["tenure_days"] = (latest - pd.to_datetime(frame["signup_date"], errors="coerce")).dt.days
    frame["frequency"] = frame["total_orders"]
    frame["monetary"] = frame["total_spend"]
    return frame


def integration_summary(db_path: Path) -> dict:
    with connect(db_path) as connection:
        orders, revenue, last_order = connection.execute("SELECT COUNT(*), COALESCE(SUM(amount), 0), MAX(order_date) FROM orders WHERE status != 'refund'").fetchone()
        customers = connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        sources = [dict(row) for row in connection.execute("SELECT source_name, status, last_sync_at FROM integration_sources ORDER BY updated_at DESC").fetchall()]
        history = [dict(row) for row in connection.execute("SELECT s.source_name, r.* FROM sync_runs r JOIN integration_sources s ON s.source_id=r.source_id ORDER BY r.sync_id DESC LIMIT 20").fetchall()]
    return {"orders": orders, "customers": customers, "revenue": revenue, "last_transaction": last_order, "sources": sources, "history": history}


# ─── Applied retention strategies (server-side truth) ─────────────────────────

def record_strategy_applied(db_path: Path, customer_ids: list[str], strategy_key: str, action: str | None, applied: bool = True) -> dict:
    """Apply/unapply a strategy for many customers in one transaction."""
    now = _now()
    with connect(db_path) as connection:
        changed = 0
        for cid in customer_ids:
            if not cid:
                continue
            if applied:
                connection.execute(
                    "INSERT INTO applied_strategies(customer_id, strategy_key, action, applied_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(customer_id, strategy_key) DO UPDATE SET action=excluded.action, applied_at=excluded.applied_at",
                    (cid, strategy_key, action, now))
            else:
                connection.execute("DELETE FROM applied_strategies WHERE customer_id=? AND strategy_key=?", (cid, strategy_key))
            changed += 1
        connection.commit()
    return {"applied": applied, "strategy_key": strategy_key, "customers_changed": changed}


def applied_for_customer(db_path: Path, customer_id: str) -> list[dict]:
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT customer_id, strategy_key, action, applied_at FROM applied_strategies WHERE customer_id=? ORDER BY applied_at DESC",
            (customer_id,)).fetchall()
    return [dict(r) for r in rows]


def applied_records(db_path: Path, limit: int = 20000) -> list[dict]:
    """Full audit trail of applied strategies (newest first)."""
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT customer_id, strategy_key, action, applied_at FROM applied_strategies ORDER BY applied_at DESC LIMIT ?",
            (limit,)).fetchall()
    return [dict(r) for r in rows]


def applied_summary(db_path: Path) -> dict:
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT strategy_key, COUNT(*) AS customers, MAX(applied_at) AS last_applied FROM applied_strategies GROUP BY strategy_key ORDER BY customers DESC"
        ).fetchall()
        total = connection.execute("SELECT COUNT(DISTINCT customer_id) FROM applied_strategies").fetchone()[0]
    by_strategy = {r["strategy_key"]: {"customers": r["customers"], "last_applied": r["last_applied"]} for r in rows}
    return {"total_applied_records": sum(r["customers"] for r in rows), "unique_customers": total, "by_strategy": by_strategy}


def latest_platform_cursor(db_path: Path) -> str | None:
    """Newest platform-side modification time already synced — used as the next incremental filter."""
    with connect(db_path) as connection:
        row = connection.execute("SELECT MAX(source_updated_at) FROM orders WHERE source_updated_at IS NOT NULL").fetchone()
    return row[0] if row and row[0] else None
