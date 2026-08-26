"""
backend/main.py

FastAPI application for ChurnIQ — e-commerce customer churn prediction.

Endpoints:
  GET  /health
  GET  /model/info
  GET  /model/list
  POST /model/switch/{model_id}
  GET  /features/importance
  POST /predict
  POST /predict/batch-analyze
  GET  /train/status/{job_id}
  GET  /train/result/{job_id}
"""

import io
import re
import uuid
import threading
import secrets
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path

from utils.config import (
    get_active_model_dir, set_active_model,
    list_available_models, ARTIFACTS_BASE, MAX_UPLOAD_BYTES,
    MAX_TRAINING_WORKERS, MODEL_MIN_AUC, ADMIN_API_KEY, CORS_ORIGINS,
    INTEGRATION_DB_PATH,
)
from utils.schema_inferrer import infer_schema, schema_matches_existing
from backend.predictor import predict_batch, predict_batch_with_artifacts, reload_artifacts, get_metadata, get_feature_importance
from backend.trainer import train_model, load_artifacts
from backend.feature_mapper import map_dataset, canonical_schema, canonicalize_dataset
from backend.model_compatibility import find_compatible_model
from backend.integrations.csv_adapter import analyze_csv_dataframe, normalize_csv_dataframe
from backend.integrations.feature_builder import build_customer_features
from backend.integrations.storage import import_transactions, integration_summary
from backend.integrations.platform_sync import sync_platform, platform_status
from backend.integrations.storage import record_strategy_applied, applied_for_customer, applied_summary, applied_records
from pydantic import BaseModel


# ─── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "ChurnIQ API",
    description = "E-commerce customer churn prediction — adaptive to any e-commerce dataset",
    version     = "3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = CORS_ORIGINS,
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── Job store ─────────────────────────────────────────────────────────────────
# In-memory dict tracking background training jobs.
# Key:   job_id (human-readable + random suffix)
# Value: {label, status, step, pct, result, error}

_jobs: dict = {}
_jobs_lock = threading.Lock()
_training_executor = ThreadPoolExecutor(
    max_workers=MAX_TRAINING_WORKERS,
    thread_name_prefix="churniq-training",
)


def _make_job_id(filename: str) -> str:
    """Create a readable job_id like 'e_commerce_dataset_a3f9c2'."""
    base = re.sub(r'[^a-z0-9]+', '_', (filename or "dataset").lower().rsplit('.', 1)[0])
    base = base.strip('_')[:30] or "dataset"
    short_random = uuid.uuid4().hex[:6]
    return f"{base}_{short_random}"


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _run_training_job(job_id: str, df: pd.DataFrame, schema: dict, hash_dir: Path):
    """
    Runs in a background thread. Trains the model, updates _jobs progress,
    then scores the dataframe and stores the final dashboard payload.
    """
    try:
        def progress_callback(step: str, pct: int):
            _update_job(job_id, step=step, pct=pct, status="training")

        result = train_model(
            df            = df,
            schema        = schema,
            artifacts_dir = hash_dir,
            progress_callback = progress_callback,
        )

        if result["auc_cv"] is None or result["auc_cv"] < MODEL_MIN_AUC:
            raise ValueError(
                f"Training completed, but CV AUC {result['auc_cv']} is below the "
                f"minimum production threshold of {MODEL_MIN_AUC:.2f}."
            )

        _update_job(job_id, step="Scoring customers", pct=98, status="training")

        # A request-scoped training result must never change the active model
        # used by other requests.
        payload = predict_batch_with_artifacts(df, hash_dir)
        payload["trained_on_upload"] = True
        payload["training_result"]   = {
            "auc_cv":           result["auc_cv"],
            "duration_seconds": result["duration_seconds"],
            "n_features":       result["n_features"],
            "dataset_hash":     result["dataset_hash"],
        }

        _update_job(job_id, status="complete", step="Done", pct=100, result=payload, error=None)

    except Exception as e:
        _update_job(job_id, status="error", step="Failed", pct=0, error=str(e))


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        meta = get_metadata()
        return {
            "status":       "ok",
            "active_model": meta.get("dataset_name", "unknown"),
            "auc_cv":       meta.get("auc_cv"),
        }
    except Exception as e:
        return {"status": "ok", "active_model": "none", "note": str(e)}


# ─── Model info ────────────────────────────────────────────────────────────────

@app.get("/model/info")
def model_info():
    return get_metadata()


@app.get("/model/list")
def model_list():
    return {"models": list_available_models()}


@app.post("/model/switch/{model_id}")
def model_switch(model_id: str, x_admin_api_key: str | None = Header(default=None)):
    if not ADMIN_API_KEY or not x_admin_api_key or not secrets.compare_digest(x_admin_api_key, ADMIN_API_KEY):
        raise HTTPException(status_code=403, detail="Model switching requires a valid admin API key.")
    model_dir = ARTIFACTS_BASE / model_id
    if model_dir.name != model_id or not model_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    try:
        load_artifacts(model_dir)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Model '{model_id}' is incomplete or invalid: {exc}")
    set_active_model(model_id)
    reload_artifacts()
    return {"status": "switched", "active_model": model_id}


# ─── Feature importance ────────────────────────────────────────────────────────

@app.get("/features/importance")
def features_importance():
    return {"feature_importance": get_feature_importance()}


# ─── Single predict ────────────────────────────────────────────────────────────

@app.post("/predict")
def predict(customer_data: dict):
    try:
        from backend.predictor import predict_single
        return predict_single(customer_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Batch analyze ─────────────────────────────────────────────────────────────

@app.post("/schema/analyze")
async def analyze_schema(file: UploadFile = File(...)):
    content = await _read_upload(file)
    if not content:
        raise HTTPException(status_code=400, detail="Unable to read dataset.")

    try:
        df = _read_file(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read dataset: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")

    analysis = map_dataset(df)
    if analysis["dataset_type"] == "unlabeled" and not analysis["missing_features"]:
        _, schema = canonical_schema(df, dataset_name=file.filename)
        model = find_compatible_model(ARTIFACTS_BASE, schema["feature_cols"])
        if model:
            analysis["model"] = {
                "model_id": model["model_id"],
                "model_name": model["model_name"],
                "version": model["version"],
                "auc_cv": model["auc_cv"],
                "compatibility": model["compatibility"],
            }
    return analysis


@app.post("/predict/unlabeled")
async def predict_unlabeled(file: UploadFile = File(...)):
    content = await _read_upload(file)
    if not content:
        raise HTTPException(status_code=400, detail="Unable to read dataset.")

    try:
        df = _read_file(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read dataset: {e}")
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")

    analysis = map_dataset(df)
    if analysis["dataset_type"] == "labeled":
        raise HTTPException(status_code=422, detail="This dataset contains a churn target. Use the labeled analysis flow.")
    if analysis["missing_features"]:
        raise HTTPException(
            status_code=422,
            detail=f"No compatible churn model was found. Missing: {', '.join(analysis['missing_features'])}.",
        )

    canonical_df, schema = canonical_schema(df, dataset_name=file.filename)
    model = find_compatible_model(ARTIFACTS_BASE, schema["feature_cols"])
    if model is None:
        raise HTTPException(status_code=422, detail="No compatible churn model was found.")

    try:
        payload = predict_batch_with_artifacts(canonical_df, model["model_dir"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    payload["dataset_type"] = "unlabeled"
    payload["source_file"] = file.filename
    payload["model"] = {
        "model_id": model["model_id"],
        "model_name": model["model_name"],
        "version": model["version"],
        "auc_cv": model["auc_cv"],
        "compatibility": model["compatibility"],
    }
    payload["compatibility_score"] = model["compatibility"]["score"]
    payload["predicted_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return JSONResponse(content=payload)


# Transaction integration endpoints are deliberately separate from the dataset
# upload flow. They import normalized orders into SQLite, then reuse the same
# canonical model-compatibility and prediction services.
@app.post("/integrations/ecommerce/analyze")
async def analyze_ecommerce_import(file: UploadFile = File(...)):
    content = await _read_upload(file)
    if not content:
        raise HTTPException(status_code=400, detail="Unable to read transaction file.")
    try:
        df = _read_file(content, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read transaction file: {exc}")
    if df.empty:
        raise HTTPException(status_code=400, detail="Transaction file is empty.")
    return analyze_csv_dataframe(df)


@app.post("/integrations/ecommerce/import")
async def import_ecommerce_transactions(file: UploadFile = File(...)):
    content = await _read_upload(file)
    if not content:
        raise HTTPException(status_code=400, detail="Unable to read transaction file.")
    try:
        df = _read_file(content, file.filename)
        analysis = analyze_csv_dataframe(df)
        if analysis["missing_required"]:
            raise HTTPException(
                status_code=422,
                detail=f"Transaction import is missing required fields: {', '.join(analysis['missing_required'])}.",
            )
        transactions, validation = normalize_csv_dataframe(df, analysis)
        if not transactions:
            raise HTTPException(status_code=422, detail="No valid transactions were found to import.")
        return import_transactions(INTEGRATION_DB_PATH, file.filename or "transactions", len(df), transactions, validation)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Transaction import failed: {exc}")


@app.get("/integrations/ecommerce/summary")
def ecommerce_integration_summary():
    return integration_summary(INTEGRATION_DB_PATH)


@app.get("/integrations/ecommerce/features")
def ecommerce_features():
    customers = build_customer_features(INTEGRATION_DB_PATH)
    return {
        "customers": len(customers),
        "features": [
            "total_orders", "total_spend", "average_order_value", "recency",
            "tenure_days", "frequency", "monetary",
        ] if not customers.empty else [],
    }


@app.get("/integrations/ecommerce/customers")
def ecommerce_customers():
    customers = build_customer_features(INTEGRATION_DB_PATH)
    return {"customers": customers.where(pd.notna(customers), None).to_dict(orient="records")}


@app.post("/integrations/ecommerce/predict")
def predict_ecommerce_customers():
    customers = build_customer_features(INTEGRATION_DB_PATH)
    if customers.empty:
        raise HTTPException(status_code=422, detail="No imported transactions are available for prediction.")
    model = find_compatible_model(ARTIFACTS_BASE, customers.columns.tolist())
    if model is None:
        raise HTTPException(
            status_code=422,
            detail="No compatible churn model found for transaction-derived customer features. Train a canonical transaction churn model first.",
        )
    try:
        payload = predict_batch_with_artifacts(customers, model["model_dir"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}")
    payload.update({
        "dataset_type": "ecommerce_transactions",
        "source_file": "integration.sqlite3",
        "compatibility_score": model["compatibility"]["score"],
        "predicted_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": {key: model[key] for key in ("model_id", "model_name", "version", "auc_cv")},
    })
    return JSONResponse(content=payload)

# ─── E-commerce platform integrations (Shopify / WooCommerce / mock store) ─────

@app.get("/integrations/platform/status")
def platform_integration_status():
    summary = integration_summary(INTEGRATION_DB_PATH)
    last_sync = {
        str(src.get("source_name")): src.get("last_sync_at")
        for src in summary.get("sources", [])
        if str(src.get("source_name", "")).startswith("platform:")
    }
    return {
        "platforms": platform_status(["shopify", "woocommerce", "mock"]),
        "orders": summary.get("orders"),
        "customers": summary.get("customers"),
        "last_sync": last_sync,
    }


@app.post("/integrations/platform/sync/{platform}")
def platform_sync_endpoint(platform: str):
    try:
        result = sync_platform(platform, INTEGRATION_DB_PATH)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("error", "Platform sync failed."))
    return result


@app.post("/integrations/platform/predict/{platform}")
def platform_predict_endpoint(platform: str):
    # Sync a platform then score the resulting customers (mock/demo friendly).
    result = sync_platform(platform, INTEGRATION_DB_PATH)
    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("error", "Platform sync failed."))
    customers = build_customer_features(INTEGRATION_DB_PATH)
    if customers.empty:
        raise HTTPException(status_code=422, detail="No imported customers are available for prediction.")
    model = find_compatible_model(ARTIFACTS_BASE, customers.columns.tolist())
    if model is None:
        raise HTTPException(
            status_code=422,
            detail="No compatible churn model found for transaction-derived customer features. Train a canonical transaction churn model first.",
        )
    try:
        payload = predict_batch_with_artifacts(customers, model["model_dir"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}")
    payload.update({
        "dataset_type": "ecommerce_transactions",
        "source_platform": platform,
        "compatibility_score": model["compatibility"]["score"],
        "predicted_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": {key: model[key] for key in ("model_id", "model_name", "version", "auc_cv")},
    })
    return JSONResponse(content=payload)

class StrategyApplyRequest(BaseModel):
    customer_ids: list[str]
    strategy_key: str
    action: str | None = None
    applied: bool = True


@app.post("/strategies/apply")
def strategies_apply(req: StrategyApplyRequest):
    """Apply (or un-apply) one retention strategy for many customers at once."""
    ids = [c for c in (req.customer_ids or []) if c]
    if not ids:
        raise HTTPException(status_code=422, detail="customer_ids must not be empty.")
    if not req.strategy_key:
        raise HTTPException(status_code=422, detail="strategy_key is required.")
    try:
        return record_strategy_applied(INTEGRATION_DB_PATH, ids, req.strategy_key, req.action, req.applied)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save strategy state: {exc}")


@app.get("/strategies/applied")
def strategies_applied(customer_id: str | None = None, strategy_key: str | None = None):
    if customer_id:
        return {
            "customer_id": customer_id,
            "applied": applied_for_customer(INTEGRATION_DB_PATH, customer_id),
        }
    if strategy_key:
        summary = applied_summary(INTEGRATION_DB_PATH)
        entry = summary["by_strategy"].get(strategy_key, {"customers": 0, "last_applied": None})
        return {"strategy_key": strategy_key, **entry}
    return {**applied_summary(INTEGRATION_DB_PATH), "records": applied_records(INTEGRATION_DB_PATH)}


class StrategyNotifyRequest(BaseModel):
    customer_id: str
    strategy_key: str
    action: str | None = None
    channel: str = "email"   # email | whatsapp | both
    phone: str | None = None


@app.post("/strategies/notify")
def strategies_notify(req: StrategyNotifyRequest):
    """Best-effort outreach: email / WhatsApp the retention strategy to a customer."""
    from backend.notifications import notify_retention
    if not req.customer_id:
        raise HTTPException(status_code=422, detail="customer_id is required.")
    if req.channel not in ("email", "whatsapp", "both"):
        raise HTTPException(status_code=422, detail="channel must be email, whatsapp or both.")
    try:
        return notify_retention(
            INTEGRATION_DB_PATH,
            req.customer_id,
            req.strategy_key or "general_retention",
            req.action,
            channel=req.channel,
            phone=req.phone,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Notification failed: {exc}")


@app.get("/notifications/outbox")
def notifications_outbox():
    """Last sent outreach messages (demo mode / SMTP)."""
    from utils.config import NOTIFICATIONS_OUTBOX
    if NOTIFICATIONS_OUTBOX.exists():
        try:
            import json
            return {"messages": json.loads(NOTIFICATIONS_OUTBOX.read_text(encoding="utf-8"))}
        except Exception:
            return {"messages": []}
    return {"messages": []}


@app.get("/strategies/applied/summary")
def strategies_applied_summary():
    return applied_summary(INTEGRATION_DB_PATH)

@app.post("/predict/batch-analyze")
async def batch_analyze(file: UploadFile = File(...)):
    """
    Upload any CSV/XLSX of e-commerce customer data.

    - If columns match the active model's schema exactly → score instantly,
      return full payload directly (status 200, no job).
    - If columns match a previously trained model's schema → reuse that model,
      score instantly, return full payload directly.
    - Otherwise → start a background training job, return {job_id} immediately
      with status 202. Frontend polls /train/status/{job_id}.
    """
    content = await _read_upload(file)

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = _read_file(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty or unreadable.")

    # ── Check schema match against active model ───────────────────────────────
    active_dir    = get_active_model_dir()
    active_schema = active_dir / "schema.json"

    uploaded_schema = infer_schema(df, dataset_name=file.filename)
    if (
        uploaded_schema["target_col"] is None
        and active_schema.exists()
        and schema_matches_existing(df, active_schema)
    ):
        try:
            payload = predict_batch(df)
            payload["trained_on_upload"] = False
            return JSONResponse(content=payload)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Canonical datasets use a stable feature contract. Labeled uploads train
    # content-addressed models; unlabeled uploads select a quality-approved
    # compatible model from the registry.
    mapping = map_dataset(df)
    if not mapping["missing_features"]:
        canonical_df, canonical = canonical_schema(df, dataset_name=file.filename)
        if len(canonical["feature_cols"]) < 3:
            raise HTTPException(status_code=422, detail="Not enough usable features.")

        if canonical["target_col"] is None:
            model = find_compatible_model(ARTIFACTS_BASE, canonical["feature_cols"])
            if model is None:
                raise HTTPException(status_code=422, detail="No compatible model.")
            try:
                payload = predict_batch_with_artifacts(canonical_df, model["model_dir"])
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}")
            payload.update({
                "trained_on_upload": False,
                "reused_model": True,
                "canonical": True,
                "dataset_type": "unlabeled",
                "source_file": file.filename,
                "compatibility_score": model["compatibility"]["score"],
            })
            return JSONResponse(content=payload)

        canonical_dir = ARTIFACTS_BASE / canonical["dataset_hash"]
        has_model = (canonical_dir / "churn_pipeline.joblib").exists()

        if has_model:
            try:
                payload = predict_batch_with_artifacts(canonical_df, canonical_dir)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
            payload["trained_on_upload"] = False
            payload["reused_model"] = True
            payload["canonical"] = True
            return JSONResponse(content=payload)

        job_id = _make_job_id(file.filename)
        with _jobs_lock:
            _jobs[job_id] = {
                "label": file.filename, "status": "training", "step": "Queued",
                "pct": 0, "result": None, "error": None,
            }
        _training_executor.submit(_run_training_job, job_id, canonical_df.copy(), canonical, canonical_dir)
        return JSONResponse(
            content={"job_id": job_id, "status": "training", "label": file.filename, "canonical": True},
            status_code=202,
        )

    # ── New dataset → infer schema ─────────────────────────────────────────────
    inferred = uploaded_schema

    if inferred["target_col"] is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "No churn column detected. "
                "Ensure your dataset has a binary column named e.g. "
                "'Churn', 'Churned', 'Exited', 'Attrition'."
            )
        )

    if len(inferred["feature_cols"]) < 3:
        raise HTTPException(
            status_code=422,
            detail="Not enough usable features. Check your dataset."
        )

    # ── Reuse existing model if same schema hash ───────────────────────────────
    hash_dir = ARTIFACTS_BASE / inferred["dataset_hash"]
    if hash_dir.exists() and (hash_dir / "churn_pipeline.joblib").exists():
        payload = predict_batch_with_artifacts(df, hash_dir)
        payload["trained_on_upload"] = False
        payload["reused_model"]      = True
        return JSONResponse(content=payload, status_code=200)

    # ── New dataset, no existing model → start background training job ────────
    job_id = _make_job_id(file.filename)

    with _jobs_lock:
        _jobs[job_id] = {
            "label":  file.filename,
            "status": "training",
            "step":   "Queued",
            "pct":    0,
            "result": None,
            "error":  None,
        }

    _training_executor.submit(_run_training_job, job_id, df.copy(), inferred, hash_dir)

    return JSONResponse(
        content={"job_id": job_id, "status": "training", "label": file.filename},
        status_code=202,
    )


# ─── Training status & result ──────────────────────────────────────────────────

@app.get("/train/status/{job_id}")
def train_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "job_id": job_id,
        "label":  job["label"],
        "status": job["status"],
        "step":   job["step"],
        "pct":    job["pct"],
        "error":  job.get("error"),
    }


@app.get("/train/result/{job_id}")
def train_result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Training failed."))

    if job["status"] != "complete":
        raise HTTPException(status_code=425, detail="Training not yet complete.")

    return JSONResponse(content=job["result"])


# ─── File reader ────────────────────────────────────────────────────────────────

async def _read_upload(file: UploadFile) -> bytes:
    """Read one upload while enforcing the API memory budget."""
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    return content


def _read_file(content: bytes, filename: str) -> pd.DataFrame:
    """Read CSV or XLSX. For XLSX, always picks the sheet with the most rows."""
    fname = (filename or "").lower()

    if fname.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))

    if fname.endswith(".xlsx") or fname.endswith(".xls"):
        xl = pd.ExcelFile(io.BytesIO(content))

        best_sheet = xl.sheet_names[0]
        best_rows  = 0
        for sheet in xl.sheet_names:
            try:
                df_try = pd.read_excel(io.BytesIO(content), sheet_name=sheet, nrows=10000)
                if len(df_try) > best_rows:
                    best_rows  = len(df_try)
                    best_sheet = sheet
            except Exception:
                continue

        return pd.read_excel(io.BytesIO(content), sheet_name=best_sheet)

    raise ValueError(f"Unsupported file type: {filename}. Upload CSV or XLSX.")
