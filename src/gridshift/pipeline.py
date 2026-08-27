"""End-to-end orchestration used by the CLI and CI smoke test."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gridshift.config import Settings
from gridshift.demo import generate_demo_data
from gridshift.features import build_feature_frame
from gridshift.ingest.dwd import ingest_dwd_forecast, ingest_dwd_observations
from gridshift.ingest.smard import ingest_smard
from gridshift.models import run_modeling
from gridshift.simulation import run_default_simulations
from gridshift.storage import atomic_write_parquet, read_hourly
from gridshift.warehouse import materialize_warehouse


def run_demo_pipeline(settings: Settings, days: int = 730, n_splits: int = 4) -> dict[str, object]:
    energy, weather = generate_demo_data(settings, days=days)
    return _model_and_simulate(settings, energy, weather, n_splits)


def run_live_pipeline(
    settings: Settings,
    start: str,
    end: str,
    n_splits: int = 4,
) -> dict[str, object]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if len(end) == 10:
        end_ts += pd.Timedelta(hours=23)
    energy = ingest_smard(settings, start_ts, end_ts)
    weather = ingest_dwd_observations(settings, start_ts, end_ts)
    try:
        ingest_dwd_forecast(settings)
        weather = read_hourly(settings.weather_path)
    except Exception:
        # Historical/weather-lag fallback remains usable and is transparent in
        # the resulting feature values; source network outages should not erase
        # a successful market-data backfill.
        pass
    energy, weather = append_next_day_horizon(energy, weather)
    atomic_write_parquet(energy, settings.energy_path)
    atomic_write_parquet(weather, settings.weather_path)
    return _model_and_simulate(settings, energy, weather, n_splits)


def append_next_day_horizon(
    energy: pd.DataFrame, weather: pd.DataFrame, horizon: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure unlabeled forecast rows exist, using week-lag fallbacks if needed."""
    energy = energy.copy().sort_values("timestamp_utc").reset_index(drop=True)
    weather = weather.copy().sort_values("timestamp_utc").reset_index(drop=True)
    energy["timestamp_utc"] = pd.to_datetime(energy["timestamp_utc"], utc=True)
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)
    if energy["price_eur_mwh"].isna().sum() >= horizon:
        return energy, weather

    last_labeled = energy.loc[energy["price_eur_mwh"].notna(), "timestamp_utc"].max()
    future_timestamps = pd.date_range(
        last_labeled + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )
    future = pd.DataFrame({"timestamp_utc": future_timestamps})
    existing = energy.set_index("timestamp_utc")
    for column in energy.columns.drop("timestamp_utc"):
        future[column] = [
            existing.at[timestamp, column] if timestamp in existing.index else np.nan
            for timestamp in future_timestamps
        ]
    actual_columns = [
        column
        for column in energy.columns
        if column.endswith("_mwh")
        and "forecast" not in column
        or column in {"price_eur_mwh", "renewable_share", "carbon_intensity_gco2_kwh"}
    ]
    future[actual_columns] = np.nan
    future["is_forecast_horizon"] = True
    energy = pd.concat(
        [energy.loc[~energy["timestamp_utc"].isin(future_timestamps)], future], ignore_index=True
    ).sort_values("timestamp_utc")

    weather_index = weather.set_index("timestamp_utc")
    weather_future = pd.DataFrame({"timestamp_utc": future_timestamps})
    for column in weather.columns.drop("timestamp_utc"):
        values = []
        for timestamp in future_timestamps:
            if timestamp in weather_index.index:
                values.append(weather_index.at[timestamp, column])
            elif timestamp - pd.Timedelta(days=7) in weather_index.index:
                values.append(weather_index.at[timestamp - pd.Timedelta(days=7), column])
            else:
                values.append(np.nan)
        weather_future[column] = values
    weather_future["weather_is_forecast"] = True
    weather = pd.concat(
        [weather.loc[~weather["timestamp_utc"].isin(future_timestamps)], weather_future],
        ignore_index=True,
    ).sort_values("timestamp_utc")
    return energy.reset_index(drop=True), weather.reset_index(drop=True)


def run_dbt_build(settings: Settings, project_root: Path | None = None) -> None:
    environment_executable = Path(sys.executable).with_name(
        "dbt.exe" if sys.platform == "win32" else "dbt"
    )
    executable = (
        str(environment_executable) if environment_executable.exists() else shutil.which("dbt")
    )
    if not executable:
        raise RuntimeError('dbt is not installed. Run: pip install -e ".[dbt]"')
    root = (project_root or Path.cwd()).resolve()
    variables = "{data_dir: '" + settings.data_dir.resolve().as_posix().replace("'", "''") + "'}"
    subprocess.run(
        [
            executable,
            "build",
            "--project-dir",
            str(root / "dbt"),
            "--profiles-dir",
            str(root / "dbt"),
            "--vars",
            variables,
        ],
        cwd=root,
        check=True,
    )


def _model_and_simulate(
    settings: Settings,
    energy: pd.DataFrame,
    weather: pd.DataFrame,
    n_splits: int,
) -> dict[str, object]:
    features = build_feature_frame(energy, weather)
    atomic_write_parquet(features, settings.silver_dir / "model_features.parquet")
    metrics, _, forecast = run_modeling(features, settings, n_splits=n_splits)
    _, flex_summary = run_default_simulations(forecast, settings)
    database_path = materialize_warehouse(settings)
    return {
        "selected_model": metrics.iloc[0]["model"],
        "best_mae_eur_mwh": round(float(metrics.iloc[0]["mae_eur_mwh"]), 3),
        "forecast_hours": len(forecast),
        "negative_hours": int(forecast["negative_price_predicted"].sum()),
        "max_cost_savings_pct": round(float(flex_summary["cost_savings_pct"].max()), 3),
        "database": str(database_path),
    }
