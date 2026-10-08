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

## Deployment

The included `render.yaml` deploys the FastAPI backend to Render. After the
backend is deployed, copy its public URL into `frontend/config.js`:

```js
window.CHURNIQ_API_URL = 'https://your-render-service.onrender.com';
```

Deploy the `frontend` directory as a static site on Netlify. Set
`CHURNIQ_CORS_ORIGINS` on Render to the final Netlify URL, then redeploy the
backend. Keep MongoDB, SMTP, and admin credentials in Render environment
variables only; never commit them to `.env` or source control.

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
| `CHURNIQ_MONGO_URI` | `mongodb://127.0.0.1:27017` | MongoDB connection string for retention strategy state. |
| `CHURNIQ_MONGO_DB` | `churniq` | MongoDB database name. |

Training-job status remains in memory; use a shared queue for multi-instance
training deployments and durable shared storage for model artifacts.


## AI semantic feature mapping

When an uploaded column name is not in the static alias table, ChurnIQ now
applies a local TF-IDF + fuzzy-semantic matcher (`backend/semantic_mapper.py`)
to recognise semantically equivalent names. For example `customer_email` maps
to `customer_id`, `member_joined` to `signup_date`, and `total_revenue` to
`total_spend`  --  even though none are literal aliases. Unrelated columns
(e.g. `product`, `category`, `color`) are left unmapped and never force-mapped.
Each semantic mapping is reported with a confidence score in the schema
analysis response.

## Personalised retention recommendations

Every scored customer now includes per-customer `recommendations` driven by
their top SHAP churn drivers (`backend/recommender.py` rules). The Customers
detail view and the Retention Actions page surface these individualised
retention strategies rather than only generic campaign ideas.
