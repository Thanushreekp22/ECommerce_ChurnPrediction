import re
import hashlib
import json
from typing import Any

import pandas as pd

from backend.canonical_features import CANONICAL_FEATURES, CANONICAL_MODEL_FEATURE_ORDER
from backend.feature_engineering import derive_features
from backend.semantic_mapper import semantic_column_map


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def build_alias_map() -> dict[str, str]:
    mapping = {}
    for canonical, config in CANONICAL_FEATURES.items():
        for alias in config["aliases"]:
            mapping[normalize_column_name(alias)] = canonical
    return mapping


def is_numeric_series(series: pd.Series) -> bool:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().mean() >= 0.8


def is_date_series(series: pd.Series) -> bool:
    converted = pd.to_datetime(series, errors="coerce", format="mixed")
    return converted.notna().mean() >= 0.8


def uniqueness_ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.nunique(dropna=True) / len(series))


def exact_match_columns(columns) -> dict[str, str]:
    alias_map = build_alias_map()
    return {
        column: alias_map[normalize_column_name(column)]
        for column in columns
        if normalize_column_name(column) in alias_map
    }


def _valid_type(series: pd.Series, canonical: str) -> bool:
    expected = CANONICAL_FEATURES[canonical]["type"]
    if expected == "numeric":
        return is_numeric_series(series)
    if expected == "date":
        return is_date_series(series)
    return uniqueness_ratio(series) >= 0.5


def _target_column(df: pd.DataFrame) -> str | None:
    target_words = ("churn", "churned", "churn_flag", "is_churn", "attrition", "exited", "cancelled", "canceled")
    for column in df.columns:
        normalized = normalize_column_name(column)
        values = df[column].dropna().nunique()
        if values >= 2 and values <= 5 and any(word in normalized for word in target_words):
            return column
    return None


def map_dataset(df: pd.DataFrame) -> dict[str, Any]:
    matches = exact_match_columns(df.columns)
    mappings = []
    used_canonical = set()
    warnings = []

    for source, canonical in matches.items():
        if canonical in used_canonical:
            warnings.append(f"Multiple columns map to '{canonical}'; only the first was used.")
            continue
        if not _valid_type(df[source], canonical):
            warnings.append(f"'{source}' was not mapped to '{canonical}' because its data type is incompatible.")
            continue

        confidence = 0.99 if canonical == "customer_id" else 0.98
        if canonical == "customer_id" and uniqueness_ratio(df[source]) < 0.8:
            confidence = 0.82
            warnings.append(f"'{source}' has low uniqueness for a customer identifier.")
        mappings.append({
            "source": source,
            "canonical": canonical,
            "method": "alias",
            "confidence": confidence,
        })
        used_canonical.add(canonical)

    # ── Semantic fallback: map any remaining canonical feature names not
    # caught by the static alias table, using the local TF-IDF + fuzzy
    # column matcher (the "AI" canonical-mapping upgrade). This lets us
    # recognise unseen but semantically equivalent columns (e.g.
    # 'customer_email' -> customer_id, 'member_joined' -> signup_date).
    alias_matched_sources = {item["source"] for item in mappings}
    semantic = semantic_column_map([col for col in df.columns if col not in alias_matched_sources])
    for source, info in semantic.items():
        canonical = info["canonical"]
        if canonical in used_canonical:
            continue
        if not _valid_type(df[source], canonical):
            warnings.append(f"'{source}' was not mapped semantically to '{canonical}' because its data type is incompatible.")
            continue
        mappings.append({
            "source": source,
            "canonical": canonical,
            "method": "semantic",
            "confidence": info["confidence"],
            "reason": info["reason"],
        })
        used_canonical.add(canonical)
        warnings.append(f"'{source}' was mapped to '{canonical}' via semantic similarity (confidence {info['confidence']}).")

    mapped_features = {item["canonical"] for item in mappings}
    missing_features = [
        name for name, config in CANONICAL_FEATURES.items()
        if config["required"] and name not in mapped_features
    ]
    derived_df = derive_features(_canonical_dataframe(df, mappings))
    derived_features = [
        name for name in ("average_order_value", "recency", "tenure_days")
        if name in derived_df.columns and name not in df.columns
    ]
    mapped_sources = {item["source"] for item in mappings}
    unmapped_columns = [column for column in df.columns if column not in mapped_sources]
    required_count = sum(config["required"] for config in CANONICAL_FEATURES.values())
    compatibility_score = round((required_count - len(missing_features)) / required_count, 2)

    return {
        "source_columns": list(df.columns),
        "mappings": mappings,
        "missing_features": missing_features,
        "derived_features": derived_features,
        "unmapped_columns": unmapped_columns,
        "compatibility_score": compatibility_score,
        "target_column": _target_column(df),
        "dataset_type": "labeled" if _target_column(df) else "unlabeled",
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "warnings": warnings,
    }


def _canonical_dataframe(df: pd.DataFrame, mappings: list[dict]) -> pd.DataFrame:
    rename = {item["source"]: item["canonical"] for item in mappings}
    return df.rename(columns=rename)


def canonicalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    result = map_dataset(df)
    return derive_features(_canonical_dataframe(df, result["mappings"]))


def canonical_model_id(feature_columns: list[str]) -> str:
    contract = json.dumps(sorted(feature_columns), separators=(",", ":"))
    return "canonical_" + hashlib.sha256(contract.encode("utf-8")).hexdigest()[:12]


def canonical_training_model_id(df: pd.DataFrame, feature_columns: list[str], target_col: str | None) -> str:
    """Create a stable ID for one labeled training dataset.

    The feature contract alone is intentionally not enough: two historical
    datasets with the same columns can contain different labels and must not
    silently share a trained model.
    """
    contract = json.dumps(sorted(feature_columns), separators=(",", ":"))
    relevant = list(feature_columns) + ([target_col] if target_col in df.columns else [])
    content = pd.util.hash_pandas_object(df[relevant], index=True).values.tobytes()
    digest = hashlib.sha256(contract.encode("utf-8") + content).hexdigest()[:12]
    return f"canonical_{digest}"


def canonical_schema(df: pd.DataFrame, dataset_name: str = "canonical dataset") -> tuple[pd.DataFrame, dict]:
    analysis = map_dataset(df)
    canonical_df = canonicalize_dataset(df)
    target = analysis["target_column"]
    if target and target not in canonical_df.columns:
        target = None

    feature_columns = [
        feature for feature in CANONICAL_MODEL_FEATURE_ORDER
        if feature in canonical_df.columns and is_numeric_series(canonical_df[feature])
    ]
    schema = {
        "dataset_name": dataset_name,
        "dataset_hash": canonical_model_id(feature_columns),
        "canonical": True,
        "n_rows": len(canonical_df),
        "n_cols": len(canonical_df.columns),
        "target_col": target,
        "id_cols": ["customer_id"] if "customer_id" in canonical_df.columns else [],
        "numeric_cols": feature_columns,
        "categorical_cols": [],
        "datetime_cols": [column for column in ("last_purchase_date", "signup_date") if column in canonical_df],
        "text_cols": [],
        "drop_cols": [column for column in ("customer_id", "last_purchase_date", "signup_date") if column in canonical_df],
        "feature_cols": feature_columns,
        "churn_rate": None,
        "class_balance": "unknown",
        "warnings": [],
    }
    if target:
        from utils.schema_inferrer import normalise_target
        labels = normalise_target(canonical_df[target])
        if labels is not None:
            schema["churn_rate"] = round(float(labels.mean()) * 100, 2)
            schema["dataset_hash"] = canonical_training_model_id(canonical_df, feature_columns, target)
    return canonical_df, schema
