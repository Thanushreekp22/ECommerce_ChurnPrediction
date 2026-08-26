"""
utils/feature_engineering.py

Shared feature engineering for ChurnIQ.

This module creates RFM-derived features from the RFM metadata
detected by schema_inferrer.py.

Important:
- It is dataset-agnostic.
- It never assumes fixed ecommerce column names.
- It never invents a missing RFM component.
- The same function is used during training and prediction.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def apply_feature_engineering(
    df: pd.DataFrame,
    schema: dict
) -> pd.DataFrame:
    """
    Apply all schema-driven feature engineering.

    Currently includes:
        - RFM-derived features when RFM components are available.

    The function does NOT modify the original dataframe in-place.

    Args:
        df:
            Raw customer dataframe.

        schema:
            Schema returned by utils.schema_inferrer.infer_schema().

    Returns:
        A new dataframe containing the original columns plus
        any safely generated engineered features.
    """

    result = df.copy()

    rfm = schema.get("rfm", {})

    if not rfm.get("available", False):
        return result

    result = _add_rfm_features(result, rfm)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RFM feature generation
# ─────────────────────────────────────────────────────────────────────────────

def _add_rfm_features(
    df: pd.DataFrame,
    rfm: dict
) -> pd.DataFrame:
    """
    Add normalized RFM component scores.

    Generated columns can include:

        RFM_Recency_Score
        RFM_Frequency_Score
        RFM_Monetary_Score
        RFM_Composite_Score

    Only components that actually exist are generated.

    Recency:
        Lower raw recency is generally better because fewer days
        since the last purchase means the customer is more recent.

    Frequency:
        Higher frequency is better.

    Monetary:
        Higher monetary value is better.
    """

    result = df.copy()

    recency_col = rfm.get("recency_col")
    frequency_col = rfm.get("frequency_col")
    monetary_col = rfm.get("monetary_col")

    generated_scores = []

    # ─────────────────────────────────────────────────────────────────────────
    # Recency
    # ─────────────────────────────────────────────────────────────────────────

    if (
        recency_col
        and recency_col in result.columns
    ):
        recency_values = _to_numeric(
            result[recency_col]
        )

        # If recency comes from a date column, convert it to
        # "days since most recent date in this dataset".
        if rfm.get("recency_source") == "date":
            recency_values = _date_to_recency_days(
                result[recency_col]
            )

        if recency_values.notna().any():

            # Smaller number of days = better recency.
            score = _percentile_score(
                recency_values,
                lower_is_better=True
            )

            result["RFM_Recency_Score"] = score

            generated_scores.append(
                result["RFM_Recency_Score"]
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Frequency
    # ─────────────────────────────────────────────────────────────────────────

    if (
        frequency_col
        and frequency_col in result.columns
    ):
        frequency_values = _to_numeric(
            result[frequency_col]
        )

        if frequency_values.notna().any():

            # Higher frequency = better.
            score = _percentile_score(
                frequency_values,
                lower_is_better=False
            )

            result["RFM_Frequency_Score"] = score

            generated_scores.append(
                result["RFM_Frequency_Score"]
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Monetary
    # ─────────────────────────────────────────────────────────────────────────

    if (
        monetary_col
        and monetary_col in result.columns
    ):
        monetary_values = _to_numeric(
            result[monetary_col]
        )

        if monetary_values.notna().any():

            # Higher monetary value = better.
            score = _percentile_score(
                monetary_values,
                lower_is_better=False
            )

            result["RFM_Monetary_Score"] = score

            generated_scores.append(
                result["RFM_Monetary_Score"]
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Composite RFM score
    # ─────────────────────────────────────────────────────────────────────────

    if generated_scores:

        score_frame = pd.concat(
            generated_scores,
            axis=1
        )

        # Average only the components that actually exist.
        result["RFM_Composite_Score"] = (
            score_frame.mean(
                axis=1,
                skipna=True
            )
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Numeric conversion
# ─────────────────────────────────────────────────────────────────────────────

def _to_numeric(
    series: pd.Series
) -> pd.Series:
    """
    Safely convert a series to numeric.

    Handles common monetary formatting such as:
        "$1,250"
        "₹1,250"
        "1,250.50"
    """

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce"
        )

    cleaned = (
        series
        .astype(str)
        .str.replace(
            r"[₹$€£,\s]",
            "",
            regex=True
        )
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Percentile scoring
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_score(
    series: pd.Series,
    lower_is_better: bool = False
) -> pd.Series:
    """
    Convert values into a 0–100 percentile score.

    Higher score = better customer value.

    For:
        Recency → lower raw value is better.
        Frequency → higher raw value is better.
        Monetary → higher raw value is better.

    Missing values remain NaN.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid = numeric.notna()

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    if valid.sum() == 0:
        return result

    values = numeric[valid]

    # Rank-based percentile avoids problems with very different
    # scales between datasets.
    ranks = values.rank(
        method="average",
        pct=True
    )

    if lower_is_better:
        scores = (1.0 - ranks) * 100.0
    else:
        scores = ranks * 100.0

    # A single unique value should not produce all-zero scores.
    # If every customer has the same value, the feature contains
    # no discriminating information, so use a neutral score.
    if values.nunique() <= 1:
        scores = pd.Series(
            50.0,
            index=values.index
        )

    result.loc[valid] = scores

    return result.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# Date → Recency
# ─────────────────────────────────────────────────────────────────────────────

def _date_to_recency_days(
    series: pd.Series
) -> pd.Series:
    """
    Convert a purchase/order date into days since the latest date
    present in the dataset.

    Example:

        2026-08-01 → 0 days
        2026-07-25 → 7 days

    This uses the latest observed date as the reference point,
    making the transformation reproducible during training/prediction.
    """

    dates = pd.to_datetime(
        series,
        errors="coerce"
    )

    if dates.notna().sum() == 0:
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float
        )

    reference_date = dates.max()

    return (
        reference_date - dates
    ).dt.total_seconds() / 86400.0