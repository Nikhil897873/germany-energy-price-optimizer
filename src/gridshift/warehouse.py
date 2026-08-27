"""Lightweight DuckDB materialization for the app and local exploration."""

from __future__ import annotations

from pathlib import Path

import duckdb

from gridshift.config import Settings


def materialize_warehouse(settings: Settings) -> Path:
    """Build core DuckDB marts; dbt builds the governed equivalents in ``main``."""
    database_path = settings.data_dir / "gridshift.duckdb"
    energy = _sql_path(settings.energy_path)
    weather = _sql_path(settings.weather_path)
    forecast = _sql_path(settings.artifacts_dir / "forecast_next_day.parquet")
    schedules = _sql_path(settings.artifacts_dir / "flex_schedules.parquet")
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            f"""
            create or replace table energy_hourly as
            select * from read_parquet('{energy}');

            create or replace table weather_hourly as
            select * from read_parquet('{weather}');

            create or replace table market_weather_hourly as
            select
                e.*,
                w.temperature_c,
                w.wind_speed_ms,
                w.cloud_cover_pct,
                w.global_radiation_kj_m2,
                w.station_count
            from energy_hourly e
            left join weather_hourly w using (timestamp_utc);

            create or replace table forecast_next_day as
            select * from read_parquet('{forecast}');

            create or replace table flex_schedules as
            select * from read_parquet('{schedules}');
            """
        )
    return database_path


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")

