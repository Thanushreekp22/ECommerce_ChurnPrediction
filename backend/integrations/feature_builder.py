from pathlib import Path

import pandas as pd

from backend.integrations.storage import customer_features


def build_customer_features(db_path: Path) -> pd.DataFrame:
    """Return transaction-derived customer features without filling invented fields."""
    return customer_features(db_path)
