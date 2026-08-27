import pytest

from gridshift.config import Settings
from gridshift.pipeline import run_demo_pipeline


@pytest.mark.slow
def test_demo_pipeline_smoke(tmp_path) -> None:
    result = run_demo_pipeline(Settings(data_dir=tmp_path), days=120, n_splits=2)

    assert result["forecast_hours"] == 24
    assert (tmp_path / "gridshift.duckdb").exists()
    assert (tmp_path / "artifacts" / "forecast_next_day.parquet").exists()

