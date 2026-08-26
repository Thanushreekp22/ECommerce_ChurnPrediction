"""
utils/config.py

Single source of truth for paths and model store configuration.
No longer hardcodes feature names — those come from schema.json per model.
"""

import os
from pathlib import Path

# ─── Project root ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
ARTIFACTS_BASE = BASE_DIR / "artifacts"
DATA_DIR       = BASE_DIR / "data"

def _load_dotenv(path: Path) -> None:
    """
    Tiny dependency-free .env loader.

    Reads KEY=VALUE lines from <project>/.env (if present) into os.environ,
    without overriding variables that are already set in the environment.
    Lets you keep all credentials in one file instead of exporting them
    manually before every run.
    """
    try:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass  # never block startup over optional config


_load_dotenv(BASE_DIR / ".env")
INTEGRATION_DB_PATH = ARTIFACTS_BASE / "integration.sqlite3"

# ─── Default model (existing E-Commerce model) ────────────────────────────────
DEFAULT_MODEL_DIR  = ARTIFACTS_BASE / "b139e4227328"
DEFAULT_MODEL_HASH = "b139e4227328"

# ─── Model store ──────────────────────────────────────────────────────────────
# Each trained model lives in ARTIFACTS_BASE / {dataset_hash}/
# The active model is tracked in ARTIFACTS_BASE / active_model.txt

ACTIVE_MODEL_FILE = ARTIFACTS_BASE / "active_model.txt"

# Runtime configuration is deliberately environment-driven so the same code can
# run locally and behind a reverse proxy without source edits.
MAX_UPLOAD_BYTES = int(os.getenv("CHURNIQ_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_TRAINING_WORKERS = int(os.getenv("CHURNIQ_MAX_TRAINING_WORKERS", "1"))
MODEL_MIN_AUC = float(os.getenv("CHURNIQ_MODEL_MIN_AUC", "0.50"))
ADMIN_API_KEY = os.getenv("CHURNIQ_ADMIN_API_KEY")
CORS_ORIGINS = [
    origin.strip() for origin in os.getenv(
        "CHURNIQ_CORS_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500"
    ).split(",") if origin.strip()
]

# ─── Retention notifications (email / WhatsApp) ───────────────────────────────
# Real email is sent over SMTP when these are set; otherwise the app runs in
# "demo mode" and records outgoing messages to NOTIFICATIONS_OUTBOX so the
# whole flow can be demonstrated with zero credentials.
SMTP_HOST = os.getenv("CHURNIQ_SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("CHURNIQ_SMTP_PORT", "587"))
SMTP_USER = os.getenv("CHURNIQ_SMTP_USER", "").strip()
SMTP_PASS = os.getenv("CHURNIQ_SMTP_PASS", "").strip()
SMTP_FROM = os.getenv("CHURNIQ_SMTP_FROM", SMTP_USER).strip()
SMTP_USE_TLS = os.getenv("CHURNIQ_SMTP_TLS", "1").strip() not in ("0", "false", "no", "")

NOTIFICATIONS_OUTBOX = BASE_DIR / "artifacts" / "notifications_outbox.json"

def smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER)


def get_active_model_dir() -> Path:
    """
    Returns the directory of the currently active model.
    Falls back to DEFAULT_MODEL_DIR if no active model is set.
    """
    if ACTIVE_MODEL_FILE.exists():
        model_id = ACTIVE_MODEL_FILE.read_text().strip()
        candidate = ARTIFACTS_BASE / model_id
        if candidate.exists():
            return candidate
    return DEFAULT_MODEL_DIR


def set_active_model(model_id: str):
    """
    Set the active model by writing its ID to active_model.txt.
    model_id is either 'ecommerce_default' or a dataset hash string.
    """
    ARTIFACTS_BASE.mkdir(parents=True, exist_ok=True)
    ACTIVE_MODEL_FILE.write_text(model_id.strip())


def get_model_dir(model_id: str) -> Path:
    """Return the artifact directory for a given model_id."""
    return ARTIFACTS_BASE / model_id


def list_available_models() -> list:
    """
    List all available trained models in the artifacts directory.
    Returns list of dicts with id, name, hash, auc_cv.
    """
    import json
    models = []
    for d in sorted(ARTIFACTS_BASE.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "model_metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            models.append({
                "id":           d.name,
                "dataset_name": meta.get("dataset_name", d.name),
                "dataset_hash": meta.get("dataset_hash", d.name),
                "auc_cv":       meta.get("auc_cv"),
                "n_rows":       meta.get("n_rows"),
                "churn_rate":   meta.get("churn_rate"),
            })
        except Exception:
            continue
    return models
