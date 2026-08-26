# """
# backend/predictor.py

# Loads artifacts from the active model directory and runs predictions.
# Works with any model trained by backend/trainer.py — not hardcoded
# to the E-Commerce dataset.
# """

# import json
# import numpy as np
# import pandas as pd
# import shap

# from pathlib import Path
# from typing import Optional

# from utils.config import get_active_model_dir
# from backend.trainer import load_artifacts
from backend.recommender import generate_recommendations


# # ─── Global artifact cache ─────────────────────────────────────────────────────
# # Loaded once at startup (or on model switch), reused for all requests.

# _cache: dict = {}
# _cache_model_dir: Optional[Path] = None


# def _load_if_needed():
#     """Load artifacts from the active model dir, only if not already cached."""
#     global _cache, _cache_model_dir

#     active_dir = get_active_model_dir()
#     if _cache_model_dir == active_dir and _cache:
#         return  # already loaded and up to date

#     _cache = load_artifacts(active_dir)
#     _cache_model_dir = active_dir


# def reload_artifacts():
#     """Force reload artifacts — called after training a new model."""
#     global _cache, _cache_model_dir
#     _cache = {}
#     _cache_model_dir = None
#     _load_if_needed()


# def get_schema() -> dict:
#     _load_if_needed()
#     return _cache.get("schema", {})


# def get_feature_importance() -> dict:
#     _load_if_needed()
#     return _cache.get("feature_importance", {})


# def get_metadata() -> dict:
#     _load_if_needed()
#     return _cache.get("metadata", {})


# # ─── Single prediction ─────────────────────────────────────────────────────────

# def predict_single(customer_data: dict) -> dict:
#     """
#     Predict churn probability for a single customer.

#     Args:
#         customer_data: dict of {column_name: value} matching the trained schema

#     Returns:
#         dict with churn_probability, risk_level, top_shap_features
#     """
#     _load_if_needed()

#     schema        = _cache["schema"]
#     pipeline      = _cache["pipeline"]
#     explainer     = _cache["explainer"]
#     feature_order = _cache["feature_order"]

#     feature_cols = schema["numeric_cols"] + schema["categorical_cols"]

#     # Build a single-row dataframe with only feature columns
#     row = {col: customer_data.get(col, np.nan) for col in feature_cols}
#     df_input = pd.DataFrame([row])

#     # Predict
#     prob = float(pipeline.predict_proba(df_input)[0, 1])
#     risk = _risk_level(prob)

#     # SHAP for this row
#     shap_features = []
#     if explainer is not None:
#         try:
#             X_transformed = pipeline.named_steps["preprocessor"].transform(df_input)
#             shap_vals     = explainer.shap_values(X_transformed)[0]

#             # Top 3 features by absolute SHAP
#             top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
#             for idx in top_idx:
#                 fname = feature_order[idx] if idx < len(feature_order) else f"feature_{idx}"
#                 shap_features.append({
#                     "feature":    fname,
#                     "shap_value": round(float(shap_vals[idx]), 4),
#                     "direction":  "increases" if shap_vals[idx] > 0 else "decreases",
#                 })
#         except Exception:
#             pass

#     return {
#         "churn_probability": round(prob, 4),
#         "risk_level":        risk,
#         "top_shap_features": shap_features,
#     }


# # ─── Batch prediction ──────────────────────────────────────────────────────────

# def predict_batch(df: pd.DataFrame) -> dict:
#     """
#     Predict churn for all rows in a dataframe.
#     Uses tiered SHAP: full SHAP for high-risk (prob >= 0.45),
#     global importance approximation for stable customers.

#     Returns the full analytics payload consumed by the frontend.
#     """
#     _load_if_needed()

#     schema        = _cache["schema"]
#     pipeline      = _cache["pipeline"]
#     explainer     = _cache["explainer"]
#     feature_order = _cache["feature_order"]
#     global_fi     = _cache["feature_importance"]

#     feature_cols = schema["numeric_cols"] + schema["categorical_cols"]
#     target_col   = schema.get("target_col", "Churn")
#     id_col       = _detect_id_col(df, schema)

#     # Drop target if present in upload
#     if target_col in df.columns:
#         df = df.drop(columns=[target_col])

#     # Assign customer IDs
#     if id_col and id_col in df.columns:
#         customer_ids = df[id_col].astype(str).tolist()
#     else:
#         customer_ids = [str(i + 1) for i in range(len(df))]

#     # Build feature matrix — only known feature columns
#     available_features = [c for c in feature_cols if c in df.columns]
#     X = df[available_features].copy()

#     # Add missing feature columns as NaN (pipeline imputer handles them)
#     for col in feature_cols:
#         if col not in X.columns:
#             X[col] = np.nan
#     X = X[feature_cols]  # enforce column order

#     # Predict all at once
#     probs      = pipeline.predict_proba(X)[:, 1]
#     risk_flags = [_risk_level(p) for p in probs]

#     # Transform for SHAP
#     X_transformed = pipeline.named_steps["preprocessor"].transform(X)

#     # Tiered SHAP
#     high_risk_mask = probs >= 0.45
#     shap_results   = _tiered_shap(
#         X_transformed  = X_transformed,
#         probs          = probs,
#         high_risk_mask = high_risk_mask,
#         explainer      = explainer,
#         feature_order  = feature_order,
#         global_fi      = global_fi,
#     )

#     # Build customer records
#     customers = []
#     for i, (cid, prob, risk) in enumerate(zip(customer_ids, probs, risk_flags)):
#         record = {
#             "customer_id":       cid,
#             "churn_probability": round(float(prob), 4),
#             "risk_level":        risk,
#             "churn_timeline":    _churn_timeline(prob),
#             "top_shap_features": shap_results[i],
#         }
#         # Attach raw feature values for the detail modal
#         for col in feature_cols:
#             record[col] = _safe_val(df, i, col)

#         customers.append(record)

#     # Build analytics payload
#     payload = _build_analytics_payload(
#         customers    = customers,
#         probs        = probs,
#         schema       = schema,
#         df           = df,
#         feature_cols = feature_cols,
#         global_fi    = global_fi,
#     )

#     return payload


# # ─── Tiered SHAP ───────────────────────────────────────────────────────────────

# def _tiered_shap(
#     X_transformed, probs, high_risk_mask,
#     explainer, feature_order, global_fi,
#     max_exact=500
# ) -> list:
#     """
#     Compute SHAP values efficiently:
#     - Exact SHAP for high-risk customers (prob >= 0.45), capped at max_exact rows
#     - Global importance approximation for stable customers
#     """
#     n = len(probs)
#     results = [[] for _ in range(n)]

#     if explainer is not None:
#         high_risk_idx = np.where(high_risk_mask)[0]

#         # Cap at max_exact to keep batch time under 30s
#         if len(high_risk_idx) > max_exact:
#             high_risk_idx = high_risk_idx[:max_exact]

#         if len(high_risk_idx) > 0:
#             shap_vals = explainer.shap_values(X_transformed[high_risk_idx])
#             for j, i in enumerate(high_risk_idx):
#                 row_shap = shap_vals[j]
#                 top_idx  = np.argsort(np.abs(row_shap))[::-1][:3]
#                 results[i] = [
#                     {
#                         "feature":    feature_order[k] if k < len(feature_order) else f"feature_{k}",
#                         "shap_value": round(float(row_shap[k]), 4),
#                         "direction":  "increases" if row_shap[k] > 0 else "decreases",
#                     }
#                     for k in top_idx
#                 ]

#     # Approximate SHAP for remaining rows using global importance
#     top_global = sorted(global_fi.items(), key=lambda x: x[1], reverse=True)[:3]
#     approx_shap = [
#         {"feature": f, "shap_value": round(v, 4), "direction": "increases"}
#         for f, v in top_global
#     ]
#     for i in range(n):
#         if not results[i]:
#             results[i] = approx_shap

#     return results


# # ─── Analytics Payload ─────────────────────────────────────────────────────────

# def _build_analytics_payload(
#     customers, probs, schema, df, feature_cols, global_fi
# ) -> dict:
#     total        = len(customers)
#     churn_count  = sum(1 for c in customers if c["churn_probability"] >= 0.5)
#     high_risk    = sum(1 for c in customers if c["risk_level"] == "High")
#     medium_risk  = sum(1 for c in customers if c["risk_level"] == "Medium")
#     low_risk     = sum(1 for c in customers if c["risk_level"] == "Low")

#     timeline_counts = {"1_month": 0, "3_months": 0, "6_months": 0, "stable": 0}
#     for c in customers:
#         tl = c["churn_timeline"].replace(" ", "_").lower()
#         if tl in timeline_counts:
#             timeline_counts[tl] += 1

#     # Segment distributions for categorical columns
#     cat_cols    = schema.get("categorical_cols", [])
#     segments    = _build_segments(customers, df, cat_cols, feature_cols)

#     # Top churn drivers from global feature importance
#     top_drivers = [
#         {"feature": f, "importance": round(v, 4)}
#         for f, v in sorted(global_fi.items(), key=lambda x: x[1], reverse=True)[:10]
#     ]

#     # Probability histogram
#     hist_counts, hist_edges = np.histogram(probs, bins=20, range=(0, 1))
#     histogram = [
#         {"bin_start": round(float(hist_edges[i]), 2),
#          "bin_end":   round(float(hist_edges[i+1]), 2),
#          "count":     int(hist_counts[i])}
#         for i in range(len(hist_counts))
#     ]

#     return {
#         "summary": {
#             "total_customers":    total,
#             "predicted_churn":    churn_count,
#             "high_risk":          high_risk,
#             "medium_risk":        medium_risk,
#             "low_risk":           low_risk,
#             "churn_rate_pct":     round(churn_count / total * 100, 2) if total else 0,
#         },
#         "timeline":       timeline_counts,
#         "segments":       segments,
#         "top_drivers":    top_drivers,
#         "histogram":      histogram,
#         "customers":      customers,
#         "schema": {
#             "dataset_name":    schema.get("dataset_name", "Unknown"),
#             "dataset_hash":    schema.get("dataset_hash", ""),
#             "numeric_cols":    schema.get("numeric_cols", []),
#             "categorical_cols": schema.get("categorical_cols", []),
#             "feature_cols":    feature_cols,
#         }
#     }


# # ─── Segment Builder ───────────────────────────────────────────────────────────

# def _build_segments(customers, df, cat_cols, feature_cols) -> dict:
#     """Build churn rate by category for each categorical column."""
#     segments = {}
#     for col in cat_cols:
#         if col not in df.columns:
#             continue
#         col_seg = {}
#         for i, c in enumerate(customers):
#             val      = str(_safe_val(df, i, col))
#             prob     = c["churn_probability"]
#             if val not in col_seg:
#                 col_seg[val] = {"churned": 0, "retained": 0, "total": 0}
#             col_seg[val]["total"] += 1
#             if prob >= 0.5:
#                 col_seg[val]["churned"] += 1
#             else:
#                 col_seg[val]["retained"] += 1
#         segments[col] = col_seg
#     return segments


# # ─── Helpers ──────────────────────────────────────────────────────────────────

# def _risk_level(prob: float) -> str:
#     if prob >= 0.7:
#         return "High"
#     elif prob >= 0.4:
#         return "Medium"
#     else:
#         return "Low"


# def _churn_timeline(prob: float) -> str:
#     if prob >= 0.7:
#         return "1 Month"
#     elif prob >= 0.5:
#         return "3 Months"
#     elif prob >= 0.3:
#         return "6 Months"
#     else:
#         return "Stable"


# def _detect_id_col(df: pd.DataFrame, schema: dict) -> Optional[str]:
#     """Find the customer ID column in the uploaded dataframe."""
#     id_cols = schema.get("id_cols", [])
#     for col in id_cols:
#         if col in df.columns:
#             return col
#     # Fallback: look for common ID column names
#     for col in df.columns:
#         if col.lower() in ["customerid", "customer_id", "id", "userid", "user_id"]:
#             return col
#     return None


# def _safe_val(df: pd.DataFrame, i: int, col: str):
#     """Safely get value from dataframe row, return None if missing."""
#     try:
#         val = df.iloc[i][col]
#         if pd.isna(val):
#             return None
#         if hasattr(val, 'item'):
#             return val.item()
#         return val
#     except Exception:
#         return None

"""
backend/predictor.py

Loads artifacts from the active model directory and runs predictions.

Works with any model trained by backend/trainer.py.

IMPORTANT:
The same feature engineering used during training is applied during
prediction so that RFM/engineered features remain consistent.
"""

import numpy as np
import pandas as pd
import shap
import threading

from pathlib import Path
from typing import Optional

from utils.feature_engineering import apply_feature_engineering
from utils.config import get_active_model_dir
from backend.trainer import load_artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Global artifact cache
# ─────────────────────────────────────────────────────────────────────────────

_cache: dict = {}
_cache_model_dir: Optional[Path] = None
_artifact_prediction_lock = threading.RLock()


def _load_if_needed():
    """
    Load artifacts from the active model directory.

    Artifacts are cached and reused until the active model changes.
    """

    global _cache
    global _cache_model_dir

    active_dir = get_active_model_dir()

    if (
        _cache_model_dir == active_dir
        and _cache
    ):
        return

    _cache = load_artifacts(
        active_dir
    )

    _cache_model_dir = active_dir


def reload_artifacts():
    """
    Force reload artifacts.

    Called after training a new model.
    """

    global _cache, _cache_model_dir
    with _artifact_prediction_lock:
        _cache = {}
        _cache_model_dir = None
        _load_if_needed()


def predict_batch_with_artifacts(df: pd.DataFrame, artifacts_dir: Path) -> dict:
    """Predict with a selected artifact without changing the active model."""
    global _cache, _cache_model_dir
    with _artifact_prediction_lock:
        previous_cache = _cache
        previous_model_dir = _cache_model_dir
        try:
            _cache = load_artifacts(Path(artifacts_dir))
            _cache_model_dir = Path(artifacts_dir)
            return _predict_batch(df)
        finally:
            _cache = previous_cache
            _cache_model_dir = previous_model_dir


def get_schema() -> dict:
    with _artifact_prediction_lock:
        _load_if_needed()
        return _cache.get("schema", {})


def get_feature_importance() -> dict:
    with _artifact_prediction_lock:
        _load_if_needed()
        return _cache.get("feature_importance", {})


def get_metadata() -> dict:
    with _artifact_prediction_lock:
        _load_if_needed()
        return _cache.get("metadata", {})


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering helper
# ─────────────────────────────────────────────────────────────────────────────

def _apply_feature_engineering(
    df: pd.DataFrame,
    schema: dict
) -> pd.DataFrame:
    """
    Apply the exact same feature engineering used during training.

    This is especially important for RFM features.

    Example:

        DaySinceLastOrder
        OrderCount
        CashbackAmount

            ↓

        RFM_Recency_Score
        RFM_Frequency_Score
        RFM_Composite_Score

    The function returns a new dataframe and does not modify
    the original dataframe.
    """

    try:

        from utils.feature_engineering import (
            apply_feature_engineering
        )

        return apply_feature_engineering(
            df.copy(),
            schema
        )

    except Exception as e:

        raise RuntimeError(
            "Feature engineering failed during prediction: "
            f"{str(e)}"
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Single prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(
    customer_data: dict
) -> dict:
    """
    Predict churn probability for a single customer.

    Args:
        customer_data:
            Dictionary containing customer feature values.

    Returns:
        Dictionary containing:
            - churn_probability
            - risk_level
            - churn_timeline
            - top_shap_features
    """

    _load_if_needed()

    schema = _cache["schema"]

    pipeline = _cache[
        "pipeline"
    ]

    explainer = _cache[
        "explainer"
    ]

    feature_order = _cache[
        "feature_order"
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # Get model feature columns
    # ─────────────────────────────────────────────────────────────────────────

    feature_cols = (
        schema.get(
            "numeric_cols",
            []
        )
        +
        schema.get(
            "categorical_cols",
            []
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Build raw one-row dataframe
    #
    # Do NOT require engineered RFM columns from the user.
    # They will be generated automatically below.
    # ─────────────────────────────────────────────────────────────────────────

    raw_columns = list(
        customer_data.keys()
    )

    row = {
        col: customer_data.get(
            col,
            np.nan
        )
        for col in raw_columns
    }

    df_input = pd.DataFrame(
        [row]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Apply same feature engineering used during training
    # ─────────────────────────────────────────────────────────────────────────

    df_input = _apply_feature_engineering(
        df_input,
        schema
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Add missing model features
    # ─────────────────────────────────────────────────────────────────────────

    for col in feature_cols:

        if col not in df_input.columns:

            df_input[col] = np.nan

    # Keep exact feature order expected by pipeline.
    df_input = df_input[
        feature_cols
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────────────────────────────────

    prob = float(
        pipeline
        .predict_proba(
            df_input
        )[0, 1]
    )

    risk = _risk_level(
        prob
    )

    timeline = _churn_timeline(
        prob
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SHAP explanation
    # ─────────────────────────────────────────────────────────────────────────

    shap_features = []

    if explainer is not None:

        try:

            X_transformed = (
                pipeline
                .named_steps[
                    "preprocessor"
                ]
                .transform(
                    df_input
                )
            )

            shap_vals = (
                explainer
                .shap_values(
                    X_transformed
                )
            )

            # Some SHAP versions return a list.
            if isinstance(
                shap_vals,
                list
            ):
                shap_vals = shap_vals[0]

            shap_vals = np.asarray(
                shap_vals
            )

            if shap_vals.ndim > 1:
                shap_vals = shap_vals[0]

            # Top 3 features by absolute SHAP.
            top_idx = (
                np.argsort(
                    np.abs(
                        shap_vals
                    )
                )[::-1][:3]
            )

            for idx in top_idx:

                if idx >= len(
                    shap_vals
                ):
                    continue

                fname = (
                    feature_order[idx]
                    if idx < len(
                        feature_order
                    )
                    else f"feature_{idx}"
                )

                shap_value = float(
                    shap_vals[idx]
                )

                shap_features.append(
                    {
                        "feature": fname,
                        "shap_value": round(
                            shap_value,
                            4
                        ),
                        "direction": (
                            "increases"
                            if shap_value > 0
                            else "decreases"
                        ),
                    }
                )

        except Exception:
            # Prediction should still work even if SHAP explanation fails.
            shap_features = []

    # ─────────────────────────────────────────────────────────────────────────
    # Return result
    # ─────────────────────────────────────────────────────────────────────────

    # Personalised retention recommendations for this single customer.
    recommendations = generate_recommendations(
        top_shap_features=shap_features,
        customer_data=customer_data,
        churn_probability=prob,
        n=3,
    )

    return {
        "churn_probability": round(
            prob,
            4
        ),
        "risk_level": risk,
        "churn_timeline": timeline,
        "top_shap_features": shap_features,
        "recommendations": recommendations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(
    df: pd.DataFrame
) -> dict:
    """
    Predict churn for all rows in a dataframe.

    The same feature engineering used during training is applied first.

    Uses tiered SHAP:

        High-risk customers:
            Exact SHAP explanation.

        Stable customers:
            Global feature-importance approximation.

    Returns the complete analytics payload consumed by the frontend.
    """

    # Selected-artifact prediction temporarily swaps the cache.  Serialising
    # the complete prediction prevents another request from observing that
    # temporary cache state.
    with _artifact_prediction_lock:
        _load_if_needed()
        return _predict_batch(df)


def _predict_batch(df: pd.DataFrame) -> dict:
    """Implementation for :func:`predict_batch`; caller holds the cache lock."""
    if not _cache:
        raise RuntimeError(
            "No model artifacts are loaded. Call predict_batch() or "
            "predict_batch_with_artifacts() with a valid model directory."
        )

    schema = _cache[
        "schema"
    ]

    pipeline = _cache[
        "pipeline"
    ]

    explainer = _cache[
        "explainer"
    ]

    feature_order = _cache[
        "feature_order"
    ]

    global_fi = _cache[
        "feature_importance"
    ]

    # Work on a copy.
    df = df.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Identify ID before feature engineering
    # ─────────────────────────────────────────────────────────────────────────

    id_col = _detect_id_col(
        df,
        schema
    )

    # Save customer IDs.
    if (
        id_col
        and id_col in df.columns
    ):

        customer_ids = (
            df[id_col]
            .astype(str)
            .tolist()
        )

    else:

        customer_ids = [
            str(i + 1)
            for i in range(
                len(df)
            )
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Remove target if present
    # ─────────────────────────────────────────────────────────────────────────

    target_col = schema.get(
        "target_col"
    )

    if (
        target_col
        and target_col in df.columns
    ):

        df = df.drop(
            columns=[
                target_col
            ]
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Apply same feature engineering as training
    # ─────────────────────────────────────────────────────────────────────────

    df = _apply_feature_engineering(
        df,
        schema
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Model feature columns
    # ─────────────────────────────────────────────────────────────────────────

    feature_cols = (
        schema.get(
            "numeric_cols",
            []
        )
        +
        schema.get(
            "categorical_cols",
            []
        )
    )

    # Remove duplicate feature names while preserving order.
    feature_cols = list(
        dict.fromkeys(
            feature_cols
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Build feature matrix
    # ─────────────────────────────────────────────────────────────────────────

    available_features = [
        col
        for col in feature_cols
        if col in df.columns
    ]

    X = df[
        available_features
    ].copy()

    # Missing columns are allowed.
    # The training pipeline's imputers will handle them.
    for col in feature_cols:

        if col not in X.columns:

            X[col] = np.nan

    # Exact order expected by the trained pipeline.
    X = X[
        feature_cols
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # Predict all rows
    # ─────────────────────────────────────────────────────────────────────────

    probs = (
        pipeline
        .predict_proba(
            X
        )[:, 1]
    )

    risk_flags = [
        _risk_level(
            p
        )
        for p in probs
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # Transform features for SHAP
    # ─────────────────────────────────────────────────────────────────────────

    X_transformed = (
        pipeline
        .named_steps[
            "preprocessor"
        ]
        .transform(
            X
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Tiered SHAP
    # ─────────────────────────────────────────────────────────────────────────

    high_risk_mask = (
        probs >= 0.45
    )

    shap_results = _tiered_shap(
        X_transformed=X_transformed,
        probs=probs,
        high_risk_mask=high_risk_mask,
        explainer=explainer,
        feature_order=feature_order,
        global_fi=global_fi,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Build customer records
    # ─────────────────────────────────────────────────────────────────────────

    customers = []

    for i, (
        cid,
        prob,
        risk
    ) in enumerate(
        zip(
            customer_ids,
            probs,
            risk_flags
        )
    ):

        record = {
            "customer_id": cid,

            "churn_probability": round(
                float(prob),
                4
            ),

            "risk_level": risk,

            "churn_timeline":
                _churn_timeline(
                    prob
                ),

            "top_shap_features":
                shap_results[i],
        }


        # Personalised retention recommendations driven by this
        # customer's top SHAP features (from the recommender rules).
        customer_data = {
            col: _safe_val(df, i, col)
            for col in feature_cols
        }
        record["recommendations"] = generate_recommendations(
            top_shap_features=shap_results[i],
            customer_data=customer_data,
            churn_probability=prob,
            n=3,
        )
        # Attach feature values.
        #
        # This includes engineered RFM features as well.
        for col in feature_cols:

            record[col] = _safe_val(
                df,
                i,
                col
            )

        customers.append(
            record
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Analytics payload
    # ─────────────────────────────────────────────────────────────────────────

    payload = (
        _build_analytics_payload(
            customers=customers,
            probs=probs,
            schema=schema,
            df=df,
            feature_cols=feature_cols,
            global_fi=global_fi,
        )
    )

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Tiered SHAP
# ─────────────────────────────────────────────────────────────────────────────

def _tiered_shap(
    X_transformed,
    probs,
    high_risk_mask,
    explainer,
    feature_order,
    global_fi,
    max_exact=500
) -> list:
    """
    Compute customer-specific SHAP explanations for every row.

    This intentionally ignores the old global-importance fallback.
    Each customer gets SHAP values based on their own transformed feature set.
    """

    n = len(probs)
    results = [[] for _ in range(n)]

    if explainer is None:
        return results

    try:
        shap_vals = explainer.shap_values(X_transformed)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_vals = np.asarray(shap_vals)

        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(1, -1)

        if shap_vals.shape[0] != n:
            raise ValueError(
                "SHAP row count does not match customer count: "
                f"{shap_vals.shape[0]} != {n}"
            )

        if shap_vals.shape[1] != len(feature_order):
            raise ValueError(
                "SHAP feature count does not match feature_order: "
                f"{shap_vals.shape[1]} != {len(feature_order)}"
            )

        for i in range(n):
            row_shap = shap_vals[i]
            top_idx = np.argsort(np.abs(row_shap))[::-1][:3]

            customer_features = []
            for k in top_idx:
                shap_value = float(row_shap[k])
                customer_features.append(
                    {
                        "feature": (
                            feature_order[k]
                            if k < len(feature_order)
                            else f"feature_{k}"
                        ),
                        "shap_value": round(shap_value, 4),
                        "direction": (
                            "increases" if shap_value > 0 else "decreases"
                        ),
                    }
                )

            results[i] = customer_features

    except Exception as e:
        print(f"SHAP calculation failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analytics Payload
# ─────────────────────────────────────────────────────────────────────────────

def _build_analytics_payload(
    customers,
    probs,
    schema,
    df,
    feature_cols,
    global_fi
) -> dict:

    total = len(
        customers
    )

    churn_count = sum(
        1
        for customer
        in customers
        if customer[
            "churn_probability"
        ] >= 0.5
    )

    high_risk = sum(
        1
        for customer
        in customers
        if customer[
            "risk_level"
        ] == "High"
    )

    medium_risk = sum(
        1
        for customer
        in customers
        if customer[
            "risk_level"
        ] == "Medium"
    )

    low_risk = sum(
        1
        for customer
        in customers
        if customer[
            "risk_level"
        ] == "Low"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Timeline distribution
    # ─────────────────────────────────────────────────────────────────────────

    timeline_counts = {
        "1_month": 0,
        "3_months": 0,
        "6_months": 0,
        "stable": 0,
    }

    for customer in customers:

        timeline = (
            customer[
                "churn_timeline"
            ]
            .replace(
                " ",
                "_"
            )
            .lower()
        )

        if timeline in timeline_counts:

            timeline_counts[
                timeline
            ] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Segments
    # ─────────────────────────────────────────────────────────────────────────

    cat_cols = schema.get(
        "categorical_cols",
        []
    )

    segments = _build_segments(
        customers,
        df,
        cat_cols,
        feature_cols
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Top churn drivers
    # ─────────────────────────────────────────────────────────────────────────

    top_drivers = [
        {
            "feature": feature,
            "importance": round(
                float(importance),
                4
            ),
        }

        for feature, importance
        in sorted(
            global_fi.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # Probability histogram
    # ─────────────────────────────────────────────────────────────────────────

    hist_counts, hist_edges = np.histogram(
        probs,
        bins=20,
        range=(0, 1)
    )

    histogram = [
        {
            "bin_start": round(
                float(
                    hist_edges[i]
                ),
                2
            ),

            "bin_end": round(
                float(
                    hist_edges[i + 1]
                ),
                2
            ),

            "count": int(
                hist_counts[i]
            ),
        }

        for i in range(
            len(hist_counts)
        )
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # Payload
    # ─────────────────────────────────────────────────────────────────────────

    return {

        "summary": {

            "total_customers":
                total,

            "predicted_churn":
                churn_count,

            "high_risk":
                high_risk,

            "medium_risk":
                medium_risk,

            "low_risk":
                low_risk,

            "churn_rate_pct": (
                round(
                    churn_count
                    / total
                    * 100,
                    2
                )
                if total
                else 0
            ),
        },

        "timeline":
            timeline_counts,

        "segments":
            segments,

        "top_drivers":
            top_drivers,

        "histogram":
            histogram,

        "customers":
            customers,

        "schema": {

            "dataset_name":
                schema.get(
                    "dataset_name",
                    "Unknown"
                ),

            "dataset_hash":
                schema.get(
                    "dataset_hash",
                    ""
                ),

            "numeric_cols":
                schema.get(
                    "numeric_cols",
                    []
                ),

            "categorical_cols":
                schema.get(
                    "categorical_cols",
                    []
                ),

            "feature_cols":
                feature_cols,

            "engineered_cols":
                schema.get(
                    "engineered_cols",
                    []
                ),

            "rfm_engineered_cols":
                schema.get(
                    "rfm_engineered_cols",
                    []
                ),

            "rfm":
                schema.get(
                    "rfm",
                    {}
                ),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Segment Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_segments(
    customers,
    df,
    cat_cols,
    feature_cols
) -> dict:
    """
    Build predicted churn distribution by categorical feature.
    """

    segments = {}

    for col in cat_cols:

        if col not in df.columns:
            continue

        col_seg = {}

        for i, customer in enumerate(
            customers
        ):

            val = str(
                _safe_val(
                    df,
                    i,
                    col
                )
            )

            prob = customer[
                "churn_probability"
            ]

            if val not in col_seg:

                col_seg[val] = {
                    "churned": 0,
                    "retained": 0,
                    "total": 0,
                }

            col_seg[val][
                "total"
            ] += 1

            if prob >= 0.5:

                col_seg[val][
                    "churned"
                ] += 1

            else:

                col_seg[val][
                    "retained"
                ] += 1

        segments[col] = col_seg

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Risk helpers
# ─────────────────────────────────────────────────────────────────────────────

def _risk_level(
    prob: float
) -> str:

    if prob >= 0.7:
        return "High"

    elif prob >= 0.4:
        return "Medium"

    else:
        return "Low"


def _churn_timeline(
    prob: float
) -> str:

    if prob >= 0.7:
        return "1 Month"

    elif prob >= 0.5:
        return "3 Months"

    elif prob >= 0.3:
        return "6 Months"

    else:
        return "Stable"


# ─────────────────────────────────────────────────────────────────────────────
# ID detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_id_col(
    df: pd.DataFrame,
    schema: dict
) -> Optional[str]:
    """
    Find the customer ID column in the uploaded dataframe.
    """

    id_cols = schema.get(
        "id_cols",
        []
    )

    for col in id_cols:

        if col in df.columns:
            return col

    # Generic fallback.
    for col in df.columns:

        normalised = (
            str(col)
            .strip()
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        if normalised in [
            "customerid",
            "customer_id",
            "id",
            "userid",
            "user_id",
        ]:

            return col

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Safe dataframe value
# ─────────────────────────────────────────────────────────────────────────────

def _safe_val(
    df: pd.DataFrame,
    i: int,
    col: str
):
    """
    Safely retrieve a dataframe value.

    Returns None when:
        - column does not exist
        - row does not exist
        - value is NaN
    """

    try:

        if col not in df.columns:
            return None

        val = df.iloc[i][col]

        if pd.isna(val):
            return None

        if hasattr(
            val,
            "item"
        ):

            return val.item()

        return val

    except Exception:

        return None
