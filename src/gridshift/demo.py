"""Realistic synthetic data for a zero-credential portfolio demo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridshift.config import Settings
from gridshift.ingest.smard import add_energy_aggregates
from gridshift.storage import atomic_write_json, atomic_write_parquet


def generate_demo_data(
    settings: Settings,
    days: int = 730,
    as_of: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate hourly German-market-like history plus a 24-hour forecast horizon.

    Demo values are synthetic and are deliberately marked as such. They preserve
    daily/weekly seasonality, weather/renewable relationships, price spikes, and
    negative-price regimes so every product path can be reviewed offline.
    """
    if days < 120:
        raise ValueError("At least 120 days are required for seasonal backtesting")
    settings.ensure_directories()
    rng = np.random.default_rng(settings.random_seed)
    anchor = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC").floor("D")
    anchor = anchor.tz_localize("UTC") if anchor.tzinfo is None else anchor.tz_convert("UTC")
    history_start = anchor - pd.Timedelta(days=days)
    timestamps = pd.date_range(
        history_start, anchor + pd.Timedelta(days=1), freq="h", inclusive="left"
    )
    n = len(timestamps)
    local = timestamps.tz_convert("Europe/Berlin")
    hour = local.hour.to_numpy()
    day_of_year = local.dayofyear.to_numpy()
    weekday = local.dayofweek.to_numpy()
    history_mask = timestamps < anchor

    annual = np.sin(2 * np.pi * (day_of_year - 172) / 365.25)
    daily_solar = np.maximum(0, np.sin(np.pi * (hour - 6) / 13))
    cloud = np.clip(55 + 25 * _ar_noise(rng, n, 0.92) - 10 * annual, 0, 100)
    temperature = (
        10 + 10 * annual + 3 * np.sin(2 * np.pi * (hour - 14) / 24) + rng.normal(0, 1.7, n)
    )
    wind_speed = np.clip(5.5 + 2.6 * _ar_noise(rng, n, 0.96) - 0.7 * annual, 0.2, 18)
    radiation = np.clip(2400 * daily_solar * (1 - 0.0075 * cloud) * (1 + 0.35 * annual), 0, None)

    workday = (weekday < 5).astype(float)
    morning_peak = np.exp(-(((hour - 9) / 3.1) ** 2))
    evening_peak = np.exp(-(((hour - 19) / 3.5) ** 2))
    heating = np.maximum(0, 13 - temperature)
    load = (
        43_000
        + 8_000 * workday
        + 7_500 * morning_peak
        + 8_500 * evening_peak
        + 650 * heating
        + rng.normal(0, 1_150, n)
    )

    solar = np.clip(radiation / 2400 * 54_000 + rng.normal(0, 650, n), 0, None)
    wind_onshore = np.clip(1_700 * wind_speed**1.45 + rng.normal(0, 1_200, n), 900, 55_000)
    wind_offshore = np.clip(440 * wind_speed**1.55 + rng.normal(0, 450, n), 200, 12_500)
    biomass = np.clip(4_700 + rng.normal(0, 220, n), 3_700, 5_700)
    hydro = np.clip(2_900 - 300 * annual + rng.normal(0, 240, n), 1_500, 4_500)
    other_renewable = np.clip(850 + rng.normal(0, 80, n), 500, 1_200)
    renewables = solar + wind_onshore + wind_offshore + biomass + hydro + other_renewable
    residual = load - renewables

    lignite = np.clip(7_500 + 0.20 * residual + rng.normal(0, 500, n), 2_500, 18_000)
    hard_coal = np.clip(1_800 + 0.12 * residual + rng.normal(0, 650, n), 200, 11_000)
    gas = np.clip(
        1_600 + 0.22 * residual + 4_500 * evening_peak + rng.normal(0, 700, n), 300, 18_000
    )
    nuclear = np.clip(900 + rng.normal(0, 70, n), 500, 1_200)
    other_conventional = np.clip(2_600 + rng.normal(0, 250, n), 1_500, 4_000)

    scarcity = np.maximum(0, residual - 32_000) / 700
    oversupply = np.maximum(0, renewables - load * 0.88) / 500
    price = (
        37
        + 0.00105 * residual
        + 8 * evening_peak
        + 4 * morning_peak
        + scarcity
        - 0.8 * oversupply
        + 5 * np.maximum(0, -annual)
        + rng.standard_t(df=5, size=n) * 5
    )
    # Sparse scarcity shocks make MAE/RMSE differences commercially meaningful.
    shocks = (rng.random(n) < 0.003) * rng.uniform(45, 180, n)
    price = np.clip(price + shocks, -180, 420)

    load_forecast = load + rng.normal(0, 850, n)
    solar_forecast = np.clip(solar + rng.normal(0, 1_250, n), 0, None)
    wind_onshore_forecast = np.clip(wind_onshore + rng.normal(0, 1_900, n), 0, None)
    wind_offshore_forecast = np.clip(wind_offshore + rng.normal(0, 600, n), 0, None)

    energy = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "price_eur_mwh": price,
            "load_mwh": load,
            "load_forecast_mwh": load_forecast,
            "solar_mwh": solar,
            "wind_onshore_mwh": wind_onshore,
            "wind_offshore_mwh": wind_offshore,
            "biomass_mwh": biomass,
            "hydro_mwh": hydro,
            "other_renewable_mwh": other_renewable,
            "lignite_mwh": lignite,
            "hard_coal_mwh": hard_coal,
            "natural_gas_mwh": gas,
            "nuclear_mwh": nuclear,
            "other_conventional_mwh": other_conventional,
            "solar_forecast_mwh": solar_forecast,
            "wind_onshore_forecast_mwh": wind_onshore_forecast,
            "wind_offshore_forecast_mwh": wind_offshore_forecast,
            "is_forecast_horizon": ~history_mask,
            "data_mode": "synthetic_demo",
        }
    )
    energy = add_energy_aggregates(energy)
    future_actual_columns = [
        "price_eur_mwh",
        "load_mwh",
        "solar_mwh",
        "wind_onshore_mwh",
        "wind_offshore_mwh",
        "biomass_mwh",
        "hydro_mwh",
        "other_renewable_mwh",
        "lignite_mwh",
        "hard_coal_mwh",
        "natural_gas_mwh",
        "nuclear_mwh",
        "other_conventional_mwh",
        "renewable_mwh",
        "renewable_share",
        "carbon_intensity_gco2_kwh",
    ]
    energy.loc[~history_mask, future_actual_columns] = np.nan

    weather = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "temperature_c": temperature,
            "wind_speed_ms": wind_speed,
            "cloud_cover_pct": cloud,
            "global_radiation_kj_m2": radiation,
            "station_count": 5,
            "weather_is_forecast": ~history_mask,
            "data_mode": "synthetic_demo",
        }
    )
    atomic_write_parquet(energy, settings.energy_path)
    atomic_write_parquet(weather, settings.weather_path)
    atomic_write_json(
        {
            "data_mode": "synthetic_demo",
            "warning": "Synthetic demo data; do not use for operational decisions.",
            "seed": settings.random_seed,
            "history_days": days,
            "history_start_utc": history_start,
            "forecast_start_utc": anchor,
            "forecast_hours": 24,
        },
        settings.bronze_dir / "demo_metadata.json",
    )
    return energy, weather


def _ar_noise(rng: np.random.Generator, size: int, persistence: float) -> np.ndarray:
    innovations = rng.normal(0, 0.25, size)
    values = np.zeros(size)
    for index in range(1, size):
        values[index] = persistence * values[index - 1] + innovations[index]
    scale = values.std() or 1.0
    return values / scale

