# ChurnIQ

ChurnIQ trains and scores e-commerce churn models, supports compatible
unlabeled datasets through canonical feature mapping, and presents results in a
browser dashboard.

## Local run

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500/upload.html`.

## E-commerce transaction integration

Open `http://127.0.0.1:5500/ecommerce.html` to preview and import raw order
CSV/XLSX files. Required fields are order ID, customer ID, order date, and
amount; common aliases such as `Order ID`, `Customer`, `Purchase Date`, and
`Total` are detected automatically. Imports are normalized and upserted into
`artifacts/integration.sqlite3`, so re-importing the same order IDs updates
rather than duplicates data.

Transaction-derived customer features are `total_orders`, `total_spend`,
`average_order_value`, `recency`, and `tenure_days` (with frequency and
monetary aliases). The integration prediction endpoint only scores these
customers when a quality-approved canonical model has a compatible feature
contract; it never fills unrelated model inputs with fake values.

## Verification

```powershell
.\venv\Scripts\python.exe -m pytest -q
Get-ChildItem frontend\js -Filter *.js | ForEach-Object { node --check $_.FullName }
```

## Runtime configuration

Configure production values in the deployment environment, never in source
control:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHURNIQ_CORS_ORIGINS` | local frontend origins | Comma-separated allowed browser origins. |
| `CHURNIQ_MAX_UPLOAD_BYTES` | `26214400` | Maximum CSV/XLSX upload size. |
| `CHURNIQ_MAX_TRAINING_WORKERS` | `1` | Concurrent training jobs per process. |
| `CHURNIQ_MODEL_MIN_AUC` | `0.50` | Minimum CV AUC for an unlabeled-compatible model. |
| `CHURNIQ_ADMIN_API_KEY` | unset | Required for `POST /model/switch/{model_id}` via `X-Admin-Api-Key`. |

For a multi-instance deployment, replace the in-memory training-job store with
a shared queue/database and store artifacts in durable shared storage.


## AI semantic feature mapping

When an uploaded column name is not in the static alias table, ChurnIQ now
applies a local TF-IDF + fuzzy-semantic matcher (`backend/semantic_mapper.py`)
to recognise semantically equivalent names. For example `customer_email` maps
to `customer_id`, `member_joined` to `signup_date`, and `total_revenue` to
`total_spend`  --  even though none are literal aliases. Unrelated columns
(e.g. `product`, `category`, `color`) are left unmapped and never force-mapped.
Each semantic mapping is reported with a confidence score in the schema
analysis response.

## E-commerce platform integration

Beyond CSV/XLSX order imports, the project can pull orders directly from a
connected e-commerce store:

| Platform | Configuration (environment) |
| --- | --- |
| Shopify | `CHURNIQ_SHOPIFY_SHOP`, `CHURNIQ_SHOPIFY_TOKEN` |
| WooCommerce | `CHURNIQ_WOO_URL`, `CHURNIQ_WOO_KEY`, `CHURNIQ_WOO_SECRET` |
| Retention email (SMTP) | unset | Optional. Set `CHURNIQ_SMTP_HOST`, `CHURNIQ_SMTP_PORT`, `CHURNIQ_SMTP_USER`, `CHURNIQ_SMTP_PASS` (and optionally `CHURNIQ_SMTP_FROM`) to send real emails when applying retention strategies. Without them the app runs in **demo mode**: messages are logged to `artifacts/notifications_outbox.json` and shown on the E-commerce page. |
| WhatsApp | none needed | Applying a strategy can generate a **wa.me deep link** with the message pre-filled (uses the customer phone stored during platform sync). Opening it composes the WhatsApp message — no paid API required. |
| Demo mock store | always available (no credentials) |

```
POST /integrations/platform/sync/{platform}     # pull + import orders
POST /integrations/platform/predict/{platform} # sync, then score customers
GET  /integrations/platform/status             # configuration status
```

Purchases pulled from a platform flow into the same SQLite integration store
and customer-feature builder, so they are scored and visualised exactly like
CSV imports.

## Personalised retention recommendations

Every scored customer now includes per-customer `recommendations` driven by
their top SHAP churn drivers (`backend/recommender.py` rules). The Customers
detail view and the Retention Actions page surface these individualised
retention strategies rather than only generic campaign ideas.
