import json
from pathlib import Path
from typing import Any

from backend.canonical_features import CANONICAL_FEATURES
from utils.config import MODEL_MIN_AUC


AUTOMATIC_THRESHOLD = 0.85


def calculate_model_compatibility(
    mapped_features: list[str] | set[str],
    required_features: list[str] | set[str],
    optional_features: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    available = set(mapped_features)
    required = set(required_features)
    optional = set(optional_features or [])
    matched_required = required & available
    matched_optional = optional & available
    missing_required = sorted(required - available)
    missing_optional = sorted(optional - available)
    score = len(matched_required) / len(required) if required else 0.0
    return {
        "score": round(score, 4),
        "compatible": bool(required) and not missing_required and score >= AUTOMATIC_THRESHOLD,
        "matched_features": sorted(matched_required | matched_optional),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }


def list_model_registry(artifacts_base: Path) -> list[dict[str, Any]]:
    models = []
    for directory in sorted(artifacts_base.iterdir()):
        if not directory.is_dir() or not (directory / "churn_pipeline.joblib").exists():
            continue
        schema_path = directory / "schema.json"
        feature_spec_path = directory / "feature_spec.json"
        metadata_path = directory / "model_metadata.json"
        try:
            schema = json.loads(schema_path.read_text()) if schema_path.exists() else {}
            feature_spec = json.loads(feature_spec_path.read_text()) if feature_spec_path.exists() else {}
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            continue
        if not schema.get("canonical"):
            continue
        features = feature_spec.get("features", schema.get("feature_cols", schema.get("numeric_cols", [])))
        core_required = {
            name for name, config in CANONICAL_FEATURES.items()
            if config.get("required") and name != "customer_id"
        }
        required_features = [feature for feature in features if feature in core_required]
        optional_features = [feature for feature in features if feature not in required_features]
        auc_cv = metadata.get("auc_cv")
        is_usable = isinstance(auc_cv, (int, float)) and auc_cv >= MODEL_MIN_AUC
        models.append({
            "model_id": directory.name,
            "model_dir": directory,
            "required_features": required_features,
            "optional_features": optional_features,
            "domain": metadata.get("domain", "ecommerce"),
            "model_name": metadata.get("model_name", metadata.get("dataset_name", directory.name)),
            "version": metadata.get("version", "1.0"),
            "auc_cv": auc_cv,
            "is_usable": is_usable,
            "schema": schema,
        })
    return models


def find_compatible_model(
    artifacts_base: Path,
    mapped_features: list[str] | set[str],
    domain: str = "ecommerce",
) -> dict[str, Any] | None:
    candidates = []
    for model in list_model_registry(artifacts_base):
        if model["domain"] != domain or not model["is_usable"]:
            continue
        compatibility = calculate_model_compatibility(
            mapped_features,
            model["required_features"],
            model["optional_features"],
        )
        model["compatibility"] = compatibility
        if compatibility["compatible"]:
            candidates.append(model)
    if not candidates:
        return None
    return max(candidates, key=lambda model: (model["compatibility"]["score"], model["auc_cv"] or 0))
