from pathlib import Path

import pandas as pd
import pytest

from gridshift import pipeline
from gridshift.config import Settings
from gridshift.pipeline import append_next_day_horizon, run_demo_pipeline


@pytest.mark.slow
def test_demo_pipeline_smoke(tmp_path) -> None:
    result = run_demo_pipeline(Settings(data_dir=tmp_path), days=120, n_splits=2)

    assert result["forecast_hours"] == 24
    assert (tmp_path / "gridshift.duckdb").exists()
    assert (tmp_path / "artifacts" / "forecast_next_day.parquet").exists()


def test_append_next_day_horizon_uses_week_lag() -> None:
    timestamps = pd.date_range("2025-01-01", periods=8 * 24, freq="h", tz="UTC")
    energy = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "price_eur_mwh": range(len(timestamps)),
            "load_mwh": [50_000.0] * len(timestamps),
            "load_forecast_mwh": [51_000.0] * len(timestamps),
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "temperature_c": range(len(timestamps)),
            "weather_is_forecast": False,
        }
    )

    extended_energy, extended_weather = append_next_day_horizon(energy, weather, horizon=24)
    future_energy = extended_energy.tail(24)
    future_weather = extended_weather.tail(24)

    assert future_energy["price_eur_mwh"].isna().all()
    assert future_energy["load_mwh"].isna().all()
    assert future_energy["load_forecast_mwh"].isna().all()
    assert future_energy["is_forecast_horizon"].all()
    assert future_weather["weather_is_forecast"].all()
    assert future_weather.iloc[0]["temperature_c"] == weather.iloc[24]["temperature_c"]

    unchanged_energy, unchanged_weather = append_next_day_horizon(
        extended_energy, extended_weather, horizon=24
    )
    assert len(unchanged_energy) == len(extended_energy)
    assert len(unchanged_weather) == len(extended_weather)


def test_live_pipeline_survives_forecast_outage(monkeypatch, tmp_path) -> None:
    timestamps = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    energy = pd.DataFrame({"timestamp_utc": timestamps, "price_eur_mwh": 30.0})
    weather = pd.DataFrame({"timestamp_utc": timestamps, "temperature_c": 5.0})
    writes: list[Path] = []
    expected = {"selected_model": "linear"}

    monkeypatch.setattr(pipeline, "ingest_smard", lambda *args, **kwargs: energy)
    monkeypatch.setattr(pipeline, "ingest_dwd_observations", lambda *args, **kwargs: weather)
    monkeypatch.setattr(
        pipeline, "ingest_dwd_forecast", lambda settings: (_ for _ in ()).throw(OSError())
    )
    monkeypatch.setattr(
        pipeline, "append_next_day_horizon", lambda energy_frame, weather_frame: (energy, weather)
    )
    monkeypatch.setattr(
        pipeline, "atomic_write_parquet", lambda frame, path: writes.append(Path(path))
    )
    monkeypatch.setattr(pipeline, "_model_and_simulate", lambda *args, **kwargs: expected)

    result = pipeline.run_live_pipeline(
        Settings(data_dir=tmp_path), "2025-01-01", "2025-01-02", n_splits=2
    )

    assert result == expected
    assert len(writes) == 2


def test_run_dbt_build_invokes_project(monkeypatch, tmp_path) -> None:
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "dbt")
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda command, cwd, check: calls.append((command, cwd)),
    )
    monkeypatch.setattr(pipeline.Path, "exists", lambda self: False)

    pipeline.run_dbt_build(Settings(data_dir=tmp_path / "runtime"), project_root=tmp_path)

    command, cwd = calls[0]
    assert command[:2] == ["dbt", "build"]
    assert str(tmp_path / "dbt") in command
    assert cwd == tmp_path.resolve()
