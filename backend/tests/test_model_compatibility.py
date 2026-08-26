from backend.model_compatibility import calculate_model_compatibility


def test_required_features_control_compatibility():
    result = calculate_model_compatibility(
        ["recency", "frequency"],
        ["recency", "frequency", "monetary"],
    )

    assert result["score"] == 0.6667
    assert result["compatible"] is False
    assert result["missing_required"] == ["monetary"]


def test_optional_features_can_be_missing():
    result = calculate_model_compatibility(
        ["recency", "frequency", "monetary"],
        ["recency", "frequency", "monetary"],
        ["tenure_days"],
    )

    assert result["compatible"] is True
    assert result["missing_optional"] == ["tenure_days"]


def test_model_registry_uses_canonical_artifacts_only(tmp_path):
    (tmp_path / "legacy").mkdir()
    (tmp_path / "legacy" / "churn_pipeline.joblib").touch()
    assert __import__("backend.model_compatibility", fromlist=["list_model_registry"]).list_model_registry(tmp_path) == []