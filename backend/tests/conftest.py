import pytest


@pytest.fixture(autouse=True)
def isolated_artifact_store(tmp_path, monkeypatch):
    """Keep API tests independent of developer and production artifacts."""
    import backend.main as main
    import backend.predictor as predictor

    monkeypatch.setattr(main, "ARTIFACTS_BASE", tmp_path)
    predictor._cache = {}
    predictor._cache_model_dir = None
    yield
    predictor._cache = {}
    predictor._cache_model_dir = None
