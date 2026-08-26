import numpy as np
import pandas as pd


def derive_average_order_value(df: pd.DataFrame) -> pd.DataFrame:
    if "total_spend" not in df.columns or "total_orders" not in df.columns:
        return df

    orders = pd.to_numeric(df["total_orders"], errors="coerce")
    spend = pd.to_numeric(df["total_spend"], errors="coerce")
    df["average_order_value"] = np.where(orders > 0, spend / orders, 0)
    return df


def derive_recency(df: pd.DataFrame) -> pd.DataFrame:
    if "last_purchase_date" not in df.columns:
        return df

    dates = pd.to_datetime(df["last_purchase_date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    df["recency"] = (today - dates).dt.days
    return df


def derive_tenure(df: pd.DataFrame) -> pd.DataFrame:
    if "signup_date" not in df.columns:
        return df

    dates = pd.to_datetime(df["signup_date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    df["tenure_days"] = (today - dates).dt.days
    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    derive_average_order_value(result)
    derive_recency(result)
    derive_tenure(result)
    return result
