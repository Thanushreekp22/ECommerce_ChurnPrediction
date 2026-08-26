# """
# utils/schema_inferrer.py

# Tier 1: Automatic schema inference for any uploaded dataset.
# Detects column types, target column, ID columns, and builds
# a schema registry JSON — without any hardcoding.
# """

# import json
# import hashlib
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from typing import Optional


# # ─── Constants ────────────────────────────────────────────────────────────────

# CHURN_KEYWORDS = [
#     'churn', 'churned', 'attrition', 'attrit', 'left', 'cancelled',
#     'canceled', 'exited', 'exit', 'inactive', 'lapsed', 'dropout'
# ]

# ID_KEYWORDS = [
#     'id', 'uuid', 'guid', 'key', 'code', 'number', 'num', 'no',
#     'identifier', 'ref', 'reference', 'index'
# ]

# # A numeric column is categorical only if BOTH conditions are true:
# #   - unique count <= this value
# #   - value range <= this value
# # Anything beyond these → always numeric
# MAX_CATEGORICAL_UNIQUE = 5
# MAX_CATEGORICAL_RANGE  = 4


# # ─── Main Inference Function ───────────────────────────────────────────────────

# def infer_schema(df: pd.DataFrame, dataset_name: str = "unknown") -> dict:
#     schema = {
#         "dataset_name": dataset_name,
#         "dataset_hash": _hash_columns(df.columns.tolist()),
#         "n_rows": len(df),
#         "n_cols": len(df.columns),
#         "target_col": None,
#         "id_cols": [],
#         "numeric_cols": [],
#         "categorical_cols": [],
#         "datetime_cols": [],
#         "text_cols": [],
#         "drop_cols": [],
#         "feature_cols": [],
#         "churn_rate": None,
#         "class_balance": "unknown",
#         "warnings": []
#     }

#     for col in df.columns:
#         role = _detect_column_role(df, col)
#         if role == "target":
#             schema["target_col"] = col
#         elif role == "id":
#             schema["id_cols"].append(col)
#         elif role == "numeric":
#             schema["numeric_cols"].append(col)
#         elif role == "categorical":
#             schema["categorical_cols"].append(col)
#         elif role == "datetime":
#             schema["datetime_cols"].append(col)
#         elif role == "text":
#             schema["text_cols"].append(col)

#     # Fallback target detection by value pattern
#     if schema["target_col"] is None:
#         schema["target_col"] = _detect_target_by_values(df, schema)
#         if schema["target_col"]:
#             for lst in ["numeric_cols", "categorical_cols"]:
#                 if schema["target_col"] in schema[lst]:
#                     schema[lst].remove(schema["target_col"])

#     # Churn rate and class balance
#     if schema["target_col"]:
#         target_series = _normalise_target(df[schema["target_col"]])
#         if target_series is not None:
#             schema["churn_rate"] = round(float(target_series.mean()) * 100, 2)
#             schema["class_balance"] = _classify_balance(target_series.mean())

#     schema["drop_cols"] = list(set(
#         schema["id_cols"] +
#         schema["datetime_cols"] +
#         schema["text_cols"]
#     ))

#     schema["feature_cols"] = schema["numeric_cols"] + schema["categorical_cols"]
#     schema["warnings"] = _generate_warnings(df, schema)

#     return schema


# # ─── Column Role Detection ─────────────────────────────────────────────────────

# def _detect_column_role(df: pd.DataFrame, col: str) -> str:
#     col_lower = col.lower().replace(" ", "_").replace("-", "_")
#     series    = df[col]
#     n_rows    = len(df)
#     n_unique  = int(series.nunique())
#     dtype     = series.dtype

#     # ── 1. Target column by name ──────────────────────────────────────────────
#     if any(kw in col_lower for kw in CHURN_KEYWORDS) and n_unique <= 5:
#         return "target"

#     # ── 2. ID column ──────────────────────────────────────────────────────────
#     is_name_id = any(
#         col_lower == kw or
#         col_lower.endswith(f"_{kw}") or
#         col_lower.startswith(f"{kw}_")
#         for kw in ID_KEYWORDS
#     )
#     if series.nunique() == n_rows:
#         return "id"
#     if is_name_id and pd.api.types.is_integer_dtype(dtype) and n_unique / n_rows > 0.8:
#         return "id"

#     # ── 3. Datetime ───────────────────────────────────────────────────────────
#     if pd.api.types.is_datetime64_any_dtype(dtype):
#         return "datetime"
#     if dtype == object:
#         sample = series.dropna().head(20)
#         try:
#             parsed = pd.to_datetime(sample, format='mixed', errors='coerce')
#             if parsed.notna().mean() > 0.8:
#                 return "datetime"
#         except Exception:
#             pass

#     # ── 4. Free text ──────────────────────────────────────────────────────────
#     if dtype == object:
#         avg_len = series.dropna().astype(str).str.len().mean()
#         if avg_len > 50 or n_unique / max(n_rows, 1) > 0.5:
#             return "text"

#     # ── 5. Numeric vs categorical ─────────────────────────────────────────────
#     if pd.api.types.is_numeric_dtype(dtype):
#         clean = series.dropna()

#         # 5a. Strict binary flag {0,1} → categorical
#         unique_vals = set(clean.unique())
#         if unique_vals <= {0, 1, 0.0, 1.0}:
#             return "categorical"

#         # 5b. Compute range
#         val_range = float(clean.max()) - float(clean.min())

#         # 5c. Low unique count AND small range → categorical
#         #     Examples that pass: CityTier (3 unique, range 2)
#         #     Examples that fail: SatisfactionScore (5 unique, range 4 → range > MAX)
#         #                         HourSpendOnApp (6 unique, range 5 → unique > MAX)
#         #                         NumberOfDeviceRegistered (6 unique → unique > MAX)
#         if n_unique <= MAX_CATEGORICAL_UNIQUE and val_range <= MAX_CATEGORICAL_RANGE:
#             return "categorical"

#         # 5d. Everything else → numeric
#         return "numeric"

#     # ── 6. Object → categorical ───────────────────────────────────────────────
#     if dtype == object:
#         return "categorical"

#     return "numeric"


# # ─── Target Detection Fallback ─────────────────────────────────────────────────

# def _detect_target_by_values(df: pd.DataFrame, schema: dict) -> Optional[str]:
#     BINARY_VALUE_SETS = [
#         {0, 1}, {0.0, 1.0},
#         {'True', 'False'}, {'true', 'false'},
#         {'Yes', 'No'}, {'yes', 'no'},
#         {'Y', 'N'}, {'1', '0'}
#     ]
#     candidates = []
#     for col in df.columns:
#         if col in schema.get("id_cols", []):
#             continue
#         unique_vals = set(df[col].dropna().unique())
#         if unique_vals in BINARY_VALUE_SETS or len(unique_vals) == 2:
#             candidates.append(col)

#     if not candidates:
#         return None

#     for col in candidates:
#         if any(kw in col.lower() for kw in CHURN_KEYWORDS):
#             return col

#     return candidates[-1]


# # ─── Target Normalisation ──────────────────────────────────────────────────────

# def normalise_target(series: pd.Series) -> Optional[pd.Series]:
#     return _normalise_target(series)


# def _normalise_target(series: pd.Series) -> Optional[pd.Series]:
#     s = series.dropna()
#     if s.empty:
#         return None

#     if pd.api.types.is_numeric_dtype(s):
#         unique = set(s.unique())
#         if unique <= {0, 1, 0.0, 1.0}:
#             return series.fillna(0).astype(int)

#     if s.dtype == bool:
#         return series.astype(int)

#     s_str = s.astype(str).str.strip().str.lower()
#     unique_lower = set(s_str.unique())
#     mapping = None

#     if unique_lower <= {'true', 'false'}:
#         mapping = {'true': 1, 'false': 0}
#     elif unique_lower <= {'yes', 'no'}:
#         mapping = {'yes': 1, 'no': 0}
#     elif unique_lower <= {'y', 'n'}:
#         mapping = {'y': 1, 'n': 0}
#     elif unique_lower <= {'1', '0'}:
#         mapping = {'1': 1, '0': 0}
#     elif len(unique_lower) == 2:
#         counts   = s_str.value_counts()
#         minority = counts.idxmin()
#         majority = counts.idxmax()
#         mapping  = {minority: 1, majority: 0}

#     if mapping:
#         return series.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)

#     return None


# # ─── Class Balance ─────────────────────────────────────────────────────────────

# def _classify_balance(churn_rate: float) -> str:
#     if 0.30 <= churn_rate <= 0.70:
#         return "balanced"
#     elif 0.10 <= churn_rate < 0.30 or 0.70 < churn_rate <= 0.90:
#         return "imbalanced"
#     else:
#         return "severe"


# # ─── Warnings ─────────────────────────────────────────────────────────────────

# def _generate_warnings(df: pd.DataFrame, schema: dict) -> list:
#     warnings = []

#     if schema["target_col"] is None:
#         warnings.append(
#             "No churn target column detected. "
#             "Ensure your dataset has a binary churn column."
#         )

#     if schema["n_rows"] < 500:
#         warnings.append(
#             f"Only {schema['n_rows']} rows detected. "
#             f"Model accuracy may be low — recommend at least 1,000 rows."
#         )

#     if len(schema["feature_cols"]) < 3:
#         warnings.append(
#             "Fewer than 3 feature columns detected. "
#             "Schema inference may have been too aggressive."
#         )

#     if schema["class_balance"] == "severe":
#         rate = schema.get("churn_rate", 0)
#         warnings.append(
#             f"Severe class imbalance ({rate}% churn). "
#             f"Model will apply class weights automatically."
#         )

#     for col in schema["feature_cols"]:
#         null_pct = df[col].isna().mean() * 100
#         if null_pct > 40:
#             warnings.append(
#                 f"'{col}' has {null_pct:.0f}% missing values — "
#                 f"consider dropping it manually."
#             )

#     return warnings


# # ─── Schema Match Check ────────────────────────────────────────────────────────

# def schema_matches_existing(df: pd.DataFrame, existing_schema_path: Path) -> bool:
#     if not existing_schema_path.exists():
#         return False

#     with open(existing_schema_path, "r") as f:
#         saved = json.load(f)

#     saved_cols = set(
#         saved.get("feature_cols", []) +
#         [saved.get("target_col", "")] +
#         saved.get("id_cols", [])
#     )
#     saved_cols.discard("")

#     return set(df.columns.tolist()) == saved_cols


# # ─── Utilities ─────────────────────────────────────────────────────────────────

# def _hash_columns(columns: list) -> str:
#     col_string = ",".join(sorted(columns))
#     return hashlib.md5(col_string.encode()).hexdigest()[:12]


# def save_schema(schema: dict, artifacts_dir: Path) -> Path:
#     artifacts_dir.mkdir(parents=True, exist_ok=True)
#     schema_path = artifacts_dir / "schema.json"
#     with open(schema_path, "w") as f:
#         json.dump(schema, f, indent=2)
#     return schema_path


# def load_schema(artifacts_dir: Path) -> Optional[dict]:
#     schema_path = artifacts_dir / "schema.json"
#     if not schema_path.exists():
#         return None
#     with open(schema_path, "r") as f:
#         return json.load(f)

"""
utils/schema_inferrer.py

Automatic schema inference for uploaded customer/churn datasets.

Detects:
- Target column
- ID columns
- Numeric columns
- Categorical columns
- Datetime columns
- Text columns
- RFM candidates
- Data-quality warnings

The logic is schema-agnostic:
it does not depend on one particular dataset or fixed column names.
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CHURN_KEYWORDS = [
    "churn",
    "churned",
    "is_churn",
    "churn_status",
    "churn_flag",
    "attrition",
    "attrit",
    "left",
    "cancelled",
    "canceled",
    "exited",
    "exit",
    "inactive",
    "lapsed",
    "dropout",
]

ID_KEYWORDS = [
    "id",
    "uuid",
    "guid",
    "key",
    "identifier",
    "ref",
    "reference",
]

TARGET_HINTS = [
    "target",
    "label",
    "outcome",
    "flag",
    "status",
]

MAX_CATEGORICAL_UNIQUE = 5
MAX_CATEGORICAL_RANGE = 4


# ─────────────────────────────────────────────────────────────────────────────
# RFM keyword groups
# ─────────────────────────────────────────────────────────────────────────────

RFM_KEYWORDS = {
    "recency": [
        "recency",
        "days_since_last_order",
        "day_since_last_order",
        "days_since_last_purchase",
        "day_since_last_purchase",
        "days_since_purchase",
        "days_inactive",
        "inactive_days",
        "last_purchase_days",
        "last_order_days",
        "last_purchase_date",
        "last_order_date",
        "recent_purchase",
        "purchase_recency",
    ],

    "frequency": [
        "frequency",
        "purchase_frequency",
        "order_frequency",
        "purchase_count",
        "order_count",
        "orders",
        "total_orders",
        "number_of_orders",
        "number_of_purchases",
        "purchase_number",
        "transaction_count",
        "transactions",
        "total_transactions",
    ],

    "monetary": [
        "monetary",
        "monetary_value",
        "total_spend",
        "total_spent",
        "amount_spent",
        "amount_spend",
        "total_amount",
        "total_revenue",
        "revenue",
        "sales",
        "sales_amount",
        "purchase_value",
        "purchase_amount",
        "order_value",
        "total_order_value",
        "customer_lifetime_value",
        "lifetime_value",
        "clv",
        "customer_value",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Main schema inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_schema(
    df: pd.DataFrame,
    dataset_name: str = "unknown"
) -> dict:

    schema = {
        "dataset_name": dataset_name,
        "dataset_hash": _hash_columns(df.columns.tolist()),
        "n_rows": len(df),
        "n_cols": len(df.columns),

        "target_col": None,

        "id_cols": [],
        "numeric_cols": [],
        "categorical_cols": [],
        "datetime_cols": [],
        "text_cols": [],

        "drop_cols": [],
        "feature_cols": [],

        "churn_rate": None,
        "class_balance": "unknown",

        "rfm": {
            "available": False,
            "completeness": "none",
            "recency_col": None,
            "frequency_col": None,
            "monetary_col": None,
            "recency_source": None,
            "frequency_source": None,
            "monetary_source": None,
            "warnings": [],
        },

        "warnings": [],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Detect column roles
    # ─────────────────────────────────────────────────────────────────────────

    target_candidates = []

    for col in df.columns:

        role = _detect_column_role(df, col)

        if role == "target":
            target_candidates.append(col)

        elif role == "id":
            schema["id_cols"].append(col)

        elif role == "numeric":
            schema["numeric_cols"].append(col)

        elif role == "categorical":
            schema["categorical_cols"].append(col)

        elif role == "datetime":
            schema["datetime_cols"].append(col)

        elif role == "text":
            schema["text_cols"].append(col)

    # ─────────────────────────────────────────────────────────────────────────
    # Choose target column
    # ─────────────────────────────────────────────────────────────────────────

    if target_candidates:
        schema["target_col"] = _choose_best_target(
            df,
            target_candidates
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Fallback target detection
    # ─────────────────────────────────────────────────────────────────────────

    if schema["target_col"] is None:

        fallback_target = _detect_target_by_values(
            df,
            schema
        )

        schema["target_col"] = fallback_target

    # Remove target from feature groups.
    if schema["target_col"]:

        for lst_name in [
            "numeric_cols",
            "categorical_cols",
            "datetime_cols",
            "text_cols",
        ]:

            if schema["target_col"] in schema[lst_name]:
                schema[lst_name].remove(
                    schema["target_col"]
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Target statistics
    # ─────────────────────────────────────────────────────────────────────────

    if schema["target_col"]:

        target_series = _normalise_target(
            df[schema["target_col"]]
        )

        if target_series is not None:

            churn_rate = float(
                target_series.mean()
            )

            schema["churn_rate"] = round(
                churn_rate * 100,
                2
            )

            schema["class_balance"] = _classify_balance(
                churn_rate
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Drop columns
    # ─────────────────────────────────────────────────────────────────────────

    schema["drop_cols"] = list(
        dict.fromkeys(
            schema["id_cols"]
            + schema["datetime_cols"]
            + schema["text_cols"]
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Feature columns
    # ─────────────────────────────────────────────────────────────────────────

    schema["feature_cols"] = (
        schema["numeric_cols"]
        + schema["categorical_cols"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # RFM detection
    # ─────────────────────────────────────────────────────────────────────────

    schema["rfm"] = _detect_rfm(
        df,
        schema
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Warnings
    # ─────────────────────────────────────────────────────────────────────────

    schema["warnings"] = _generate_warnings(
        df,
        schema
    )

    # A labeled upload represents training data, not merely a feature
    # contract.  Include its content in the artifact ID so a new historical
    # dataset with identical columns cannot silently reuse an old model.
    if schema["target_col"]:
        relevant = [
            col for col in schema["feature_cols"] + [schema["target_col"]]
            if col in df.columns
        ]
        content_hash = pd.util.hash_pandas_object(df[relevant], index=True).values.tobytes()
        schema["dataset_hash"] = hashlib.sha256(
            schema["dataset_hash"].encode("utf-8") + content_hash
        ).hexdigest()[:12]

    return schema


# ─────────────────────────────────────────────────────────────────────────────
# Column role detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_column_role(
    df: pd.DataFrame,
    col: str
) -> str:

    col_lower = _normalise_column_name(col)

    series = df[col]

    n_rows = len(df)

    n_unique = int(
        series.nunique(dropna=True)
    )

    dtype = series.dtype

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Target detection by column name
    # ─────────────────────────────────────────────────────────────────────────

    if (
        any(
            keyword in col_lower
            for keyword in CHURN_KEYWORDS
        )
        and 2 <= n_unique <= 5
    ):
        return "target"

    # ─────────────────────────────────────────────────────────────────────────
    # 2. ID detection
    # ─────────────────────────────────────────────────────────────────────────

    is_name_id = _looks_like_id_name(
        col_lower
    )

    uniqueness_ratio = (
        n_unique / max(n_rows, 1)
    )

    # Strong ID signal:
    # identifier-like name + high uniqueness
    if (
        is_name_id
        and uniqueness_ratio > 0.8
    ):
        return "id"

    # Numeric sequential identifiers
    if (
        pd.api.types.is_numeric_dtype(dtype)
        and uniqueness_ratio > 0.95
    ):

        clean = series.dropna()

        if len(clean) >= 3:

            values = np.sort(
                clean.unique()
            )

            diffs = np.diff(values)

            if (
                len(diffs) > 0
                and np.all(diffs == 1)
            ):
                return "id"

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Datetime detection
    # ─────────────────────────────────────────────────────────────────────────

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    # IMPORTANT:
    # pandas may represent Excel string columns as:
    #   object
    #   string
    #   str
    #
    # Therefore do NOT use only `dtype == object`.
    is_string_like = (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    )

    if is_string_like:

        sample = (
            series
            .dropna()
            .head(20)
        )

        if not sample.empty:

            try:

                parsed = pd.to_datetime(
                    sample,
                    format="mixed",
                    errors="coerce"
                )

                # Only classify as datetime when most values
                # genuinely look like dates.
                if (
                    parsed.notna().mean()
                    > 0.8
                ):
                    return "datetime"

            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # 4. String / free-text detection
    # ─────────────────────────────────────────────────────────────────────────

    if is_string_like:

        non_null = (
            series
            .dropna()
            .astype(str)
        )

        if not non_null.empty:

            avg_len = (
                non_null
                .str.len()
                .mean()
            )

            text_uniqueness = (
                n_unique
                / max(n_rows, 1)
            )

            # Long text or mostly-unique strings
            # are treated as free text.
            if (
                avg_len > 50
                or text_uniqueness > 0.7
            ):
                return "text"

        # Short string columns such as:
        # Gender
        # PaymentMode
        # Device
        # MaritalStatus
        # OrderCategory
        #
        # are categorical.
        return "categorical"

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Numeric vs categorical
    # ─────────────────────────────────────────────────────────────────────────

    if pd.api.types.is_numeric_dtype(dtype):

        clean = series.dropna()

        if clean.empty:
            return "numeric"

        unique_vals = set(
            clean.unique()
        )

        # Binary 0/1 columns are normally flags/categories.
        if unique_vals <= {
            0,
            1,
            0.0,
            1.0
        }:
            return "categorical"

        val_range = (
            float(clean.max())
            - float(clean.min())
        )

        # Low cardinality + small range
        # usually means ordinal/category.
        if (
            n_unique <= MAX_CATEGORICAL_UNIQUE
            and val_range <= MAX_CATEGORICAL_RANGE
        ):
            return "categorical"

        return "numeric"

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Fallback
    # ─────────────────────────────────────────────────────────────────────────

    return "categorical"

# ─────────────────────────────────────────────────────────────────────────────
# ID helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_column_name(
    col: str
) -> str:

    name = str(col).strip()

    # CamelCase / PascalCase → snake_case
    # DaySinceLastOrder → day_since_last_order
    # OrderCount → order_count
    name = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        name
    )

    # Handle acronyms.
    name = re.sub(
        r"([A-Z]+)([A-Z][a-z])",
        r"\1_\2",
        name
    )

    name = name.lower()

    # Spaces, hyphens and other separators → underscore.
    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    )

    # Remove duplicate underscores.
    name = re.sub(
        r"_+",
        "_",
        name
    )

    return name.strip("_")


def _looks_like_id_name(
    col_lower: str
) -> bool:

    tokens = set(
        col_lower.split("_")
    )

    if col_lower in {
        "id",
        "uuid",
        "guid",
        "identifier",
    }:
        return True

    for keyword in ID_KEYWORDS:

        if (
            keyword in tokens
            or col_lower.endswith(
                "_" + keyword
            )
            or col_lower.startswith(
                keyword + "_"
            )
        ):
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Target selection
# ─────────────────────────────────────────────────────────────────────────────

def _choose_best_target(
    df: pd.DataFrame,
    candidates: list
) -> Optional[str]:

    if not candidates:
        return None

    def score(col):

        name = _normalise_column_name(
            col
        )

        score_value = 0

        if "churn" in name:
            score_value += 100

        if "attrition" in name:
            score_value += 90

        if "exited" in name:
            score_value += 80

        if "cancel" in name:
            score_value += 70

        if "inactive" in name:
            score_value += 60

        unique_count = (
            df[col]
            .dropna()
            .nunique()
        )

        if unique_count == 2:
            score_value += 20

        elif unique_count <= 5:
            score_value += 5

        return score_value

    return max(
        candidates,
        key=score
    )


# ─────────────────────────────────────────────────────────────────────────────
# Target detection fallback
# ─────────────────────────────────────────────────────────────────────────────

def _detect_target_by_values(
    df: pd.DataFrame,
    schema: dict
) -> Optional[str]:

    candidates = []

    for col in df.columns:

        if col in schema.get(
            "id_cols",
            []
        ):
            continue

        unique_values = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )

        unique_set = set(
            unique_values
        )

        if len(unique_set) != 2:
            continue

        name = _normalise_column_name(
            col
        )

        if (
            any(
                keyword in name
                for keyword in CHURN_KEYWORDS
            )
        ):
            candidates.append(
                (col, 100)
            )
            continue

        if (
            any(
                hint in name.split("_")
                for hint in TARGET_HINTS
            )
        ):
            candidates.append(
                (col, 50)
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return candidates[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# RFM detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_rfm(
    df: pd.DataFrame,
    schema: dict
) -> dict:

    rfm = {
        "available": False,
        "completeness": "none",

        "recency_col": None,
        "frequency_col": None,
        "monetary_col": None,

        "recency_source": None,
        "frequency_source": None,
        "monetary_source": None,

        "warnings": [],
    }

    excluded = set(
        schema.get("id_cols", [])
        + [schema.get("target_col")]
    )

    excluded.discard(None)

    # Find best candidate for each component.
    recency_col, recency_source = _find_rfm_candidate(
        df,
        "recency",
        excluded
    )

    frequency_col, frequency_source = _find_rfm_candidate(
        df,
        "frequency",
        excluded
    )

    monetary_col, monetary_source = _find_rfm_candidate(
        df,
        "monetary",
        excluded
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Recency can also come from a date column.
    # ─────────────────────────────────────────────────────────────────────────

    if recency_col is None:

        date_candidates = []

        for col in schema.get(
            "datetime_cols",
            []
        ):

            name = _normalise_column_name(
                col
            )

            score = 0

            if (
                "purchase" in name
                and "date" in name
            ):
                score += 50

            if (
                "order" in name
                and "date" in name
            ):
                score += 50

            if "last" in name:
                score += 30

            if "recent" in name:
                score += 20

            if score > 0:
                date_candidates.append(
                    (col, score)
                )

        if date_candidates:

            date_candidates.sort(
                key=lambda x: x[1],
                reverse=True
            )

            recency_col = date_candidates[0][0]
            recency_source = "date"

    rfm["recency_col"] = recency_col
    rfm["frequency_col"] = frequency_col
    rfm["monetary_col"] = monetary_col

    rfm["recency_source"] = recency_source
    rfm["frequency_source"] = frequency_source
    rfm["monetary_source"] = monetary_source

    available_count = sum(
        value is not None
        for value in [
            recency_col,
            frequency_col,
            monetary_col,
        ]
    )

    if available_count == 3:

        rfm["available"] = True
        rfm["completeness"] = "full"

    elif available_count == 2:

        rfm["available"] = True
        rfm["completeness"] = "partial"

        rfm["warnings"].append(
            "Only two RFM components were detected. "
            "RFM analysis will be partial."
        )

    elif available_count == 1:

        rfm["available"] = True
        rfm["completeness"] = "partial"

        rfm["warnings"].append(
            "Only one RFM component was detected. "
            "RFM analysis will be limited."
        )

    else:

        rfm["available"] = False
        rfm["completeness"] = "none"

    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# RFM candidate finder
# ─────────────────────────────────────────────────────────────────────────────

def _find_rfm_candidate(
    df: pd.DataFrame,
    component: str,
    excluded: set
):
    """
    Find the strongest candidate column for one RFM component.

    Returns:
        (column_name, source)
        or
        (None, None)
    """

    keyword_list = RFM_KEYWORDS[component]

    candidates = []

    for col in df.columns:

        # Never use target or ID columns.
        if col in excluded:
            continue

        name = _normalise_column_name(col)

        series = df[col]

        # Frequency and Monetary must be numeric.
        # Recency can be numeric or date-based.
        if component != "recency":

            if not pd.api.types.is_numeric_dtype(
                series
            ):
                continue

        score = 0

        # ─────────────────────────────────────────────────────────────────────
        # 1. Exact / strong keyword matching
        # ─────────────────────────────────────────────────────────────────────

        for keyword in keyword_list:

            keyword_normalised = _normalise_column_name(
                keyword
            )

            # Exact match.
            if name == keyword_normalised:
                score += 100
                break

            # Strong partial match.
            if keyword_normalised in name:
                score += 60

        # ─────────────────────────────────────────────────────────────────────
        # 2. Token-based semantic signals
        # ─────────────────────────────────────────────────────────────────────

        tokens = set(
            name.split("_")
        )

        # ─────────────────────────────────────────────────────────────────────
        # RECENCY
        # ─────────────────────────────────────────────────────────────────────

        if component == "recency":

            if (
                "last" in tokens
                and (
                    "order" in tokens
                    or "purchase" in tokens
                )
            ):
                score += 40

            if (
                "days" in tokens
                and "last" in tokens
            ):
                score += 30

            if (
                "day" in tokens
                and "last" in tokens
            ):
                score += 30

            if (
                "recent" in tokens
                and (
                    "purchase" in tokens
                    or "order" in tokens
                )
            ):
                score += 30

        # ─────────────────────────────────────────────────────────────────────
        # FREQUENCY
        # ─────────────────────────────────────────────────────────────────────

        elif component == "frequency":

            if (
                "order" in tokens
                and (
                    "count" in tokens
                    or "number" in tokens
                )
            ):
                score += 40

            if (
                "purchase" in tokens
                and "count" in tokens
            ):
                score += 40

            if (
                "transaction" in tokens
                and (
                    "count" in tokens
                    or "number" in tokens
                )
            ):
                score += 40

            if "frequency" in tokens:
                score += 30

        # ─────────────────────────────────────────────────────────────────────
        # MONETARY
        # ─────────────────────────────────────────────────────────────────────

        elif component == "monetary":

            # Total spending / revenue.
            if (
                "total" in tokens
                and (
                    "spend" in tokens
                    or "spent" in tokens
                    or "amount" in tokens
                    or "revenue" in tokens
                    or "sales" in tokens
                )
            ):
                score += 40

            # Purchase amount / value.
            if (
                "purchase" in tokens
                and (
                    "amount" in tokens
                    or "value" in tokens
                )
            ):
                score += 40

            # Order value.
            if (
                "order" in tokens
                and "value" in tokens
            ):
                score += 40

            # Customer lifetime value.
            if (
                (
                    "lifetime" in tokens
                    and "value" in tokens
                )
                or
                (
                    "customer" in tokens
                    and "value" in tokens
                )
            ):
                score += 40

            # Explicit monetary terms.
            if (
                "revenue" in tokens
                or "sales" in tokens
                or "monetary" in tokens
            ):
                score += 30

            # IMPORTANT:
            # Prevent time-related columns from becoming monetary.
            #
            # Examples:
            # HourSpendOnApp
            # TimeSpentOnWebsite
            # HoursSpentOnline
            # DurationSpent
            if (
                "hour" in tokens
                or "hours" in tokens
                or "time" in tokens
                or "duration" in tokens
                or "minute" in tokens
                or "minutes" in tokens
            ):
                score = 0

        # ─────────────────────────────────────────────────────────────────────
        # Store valid candidate
        # ─────────────────────────────────────────────────────────────────────

        if score > 0:

            candidates.append(
                (col, score)
            )

    # No candidate.
    if not candidates:
        return None, None

    # Highest score first.
    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_col, best_score = candidates[0]

    # Ignore weak matches.
    if best_score < 50:
        return None, None

    return best_col, "direct"


# ─────────────────────────────────────────────────────────────────────────────
# Target normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_target(
    series: pd.Series
) -> Optional[pd.Series]:

    return _normalise_target(series)


def _normalise_target(
    series: pd.Series
) -> Optional[pd.Series]:

    s = series.dropna()

    if s.empty:
        return None

    # Numeric 0/1.
    if pd.api.types.is_numeric_dtype(s):

        unique = set(
            s.unique()
        )

        if unique <= {
            0,
            1,
            0.0,
            1.0
        }:
            return (
                series
                .fillna(0)
                .astype(int)
            )

    # Boolean.
    if pd.api.types.is_bool_dtype(s):
        return series.astype(int)

    s_str = (
        s.astype(str)
        .str.strip()
        .str.lower()
    )

    unique_lower = set(
        s_str.unique()
    )

    mapping = None

    if unique_lower <= {
        "true",
        "false"
    }:
        mapping = {
            "true": 1,
            "false": 0,
        }

    elif unique_lower <= {
        "yes",
        "no"
    }:
        mapping = {
            "yes": 1,
            "no": 0,
        }

    elif unique_lower <= {
        "y",
        "n"
    }:
        mapping = {
            "y": 1,
            "n": 0,
        }

    elif unique_lower <= {
        "1",
        "0"
    }:
        mapping = {
            "1": 1,
            "0": 0,
        }

    elif len(unique_lower) == 2:

        # Generic binary target.
        # Minority class is treated as churn.
        counts = s_str.value_counts()

        minority = counts.idxmin()
        majority = counts.idxmax()

        mapping = {
            minority: 1,
            majority: 0,
        }

    if mapping:

        return (
            series
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
            .fillna(0)
            .astype(int)
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Class balance
# ─────────────────────────────────────────────────────────────────────────────

def _classify_balance(
    churn_rate: float
) -> str:

    if 0.30 <= churn_rate <= 0.70:
        return "balanced"

    elif (
        0.10 <= churn_rate < 0.30
        or
        0.70 < churn_rate <= 0.90
    ):
        return "imbalanced"

    else:
        return "severe"


# ─────────────────────────────────────────────────────────────────────────────
# Warnings
# ─────────────────────────────────────────────────────────────────────────────

def _generate_warnings(
    df: pd.DataFrame,
    schema: dict
) -> list:

    warnings = []

    # Target warning.
    if schema["target_col"] is None:

        warnings.append(
            "No churn target column detected. "
            "Ensure the dataset contains a binary churn/attrition target."
        )

    # Dataset size.
    if schema["n_rows"] < 500:

        warnings.append(
            f"Only {schema['n_rows']} rows detected. "
            "Model predictions may be unreliable."
        )

    # Feature count.
    if len(schema["feature_cols"]) == 0:

        warnings.append(
            "No usable feature columns were detected."
        )

    elif len(schema["feature_cols"]) < 3:

        warnings.append(
            f"Only {len(schema['feature_cols'])} usable feature columns "
            "were detected. Model performance may be limited."
        )

    # Class balance.
    if schema["class_balance"] == "severe":

        rate = schema.get(
            "churn_rate",
            0
        )

        warnings.append(
            f"Severe class imbalance ({rate}% churn). "
            "Class weighting or resampling may be required."
        )

    # Missing values.
    for col in schema["feature_cols"]:

        null_pct = (
            df[col]
            .isna()
            .mean()
            * 100
        )

        if null_pct > 40:

            warnings.append(
                f"'{col}' has {null_pct:.0f}% missing values."
            )

    # RFM warnings.
    rfm = schema.get(
        "rfm",
        {}
    )

    for warning in rfm.get(
        "warnings",
        []
    ):

        warnings.append(
            f"RFM: {warning}"
        )

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Schema match check
# ─────────────────────────────────────────────────────────────────────────────

def schema_matches_existing(
    df: pd.DataFrame,
    existing_schema_path: Path
) -> bool:

    if not existing_schema_path.exists():
        return False

    with open(
        existing_schema_path,
        "r"
    ) as f:

        saved = json.load(f)

    saved_cols = set(
        saved.get(
            "feature_cols",
            []
        )
        +
        [
            saved.get(
                "target_col",
                ""
            )
        ]
        +
        saved.get(
            "id_cols",
            []
        )
    )

    saved_cols.discard("")

    return (
        set(df.columns.tolist())
        == saved_cols
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _hash_columns(
    columns: list
) -> str:

    col_string = ",".join(
        sorted(
            str(col)
            for col in columns
        )
    )

    return hashlib.md5(
        col_string.encode()
    ).hexdigest()[:12]


def save_schema(
    schema: dict,
    artifacts_dir: Path
) -> Path:

    artifacts_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    schema_path = (
        artifacts_dir
        / "schema.json"
    )

    with open(
        schema_path,
        "w"
    ) as f:

        json.dump(
            schema,
            f,
            indent=2,
            default=str
        )

    return schema_path


def load_schema(
    artifacts_dir: Path
) -> Optional[dict]:

    schema_path = (
        artifacts_dir
        / "schema.json"
    )

    if not schema_path.exists():
        return None

    with open(
        schema_path,
        "r"
    ) as f:

        return json.load(f)
