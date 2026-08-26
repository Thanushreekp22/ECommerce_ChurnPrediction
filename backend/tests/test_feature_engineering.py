import pandas as pd

from backend.feature_engineering import derive_features


def test_derived_features_are_safe_and_correct():
    frame = pd.DataFrame({
        "total_orders": [2, 0],
        "total_spend": [20.0, 100.0],
        "last_purchase_date": ["2026-08-01", None],
        "signup_date": ["2026-01-01", "2026-02-01"],
    })

    result = derive_features(frame)

    assert result["average_order_value"].tolist() == [10.0, 0.0]
    assert result["recency"].iloc[0] >= 0
    assert result["recency"].iloc[1] != result["recency"].iloc[1]
    assert result["tenure_days"].iloc[0] >= 0
    assert "average_order_value" not in frame.columns


def test_missing_inputs_do_not_create_derived_features():
    result = derive_features(pd.DataFrame({"total_spend": [10.0]}))

    assert "average_order_value" not in result.columns
    assert "recency" not in result.columns
    assert "tenure_days" not in result.columns
