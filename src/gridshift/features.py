"""Leakage-aware feature engineering for day-ahead forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "price_eur_mwh"
FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday_proxy",
    "price_lag_24",
    "price_lag_48",
    "price_lag_168",
    "price_roll_mean_24",
    "price_roll_std_168",
    "load_forecast_mwh",
    "renewable_forecast_mwh",
    "renewable_forecast_share",
    "temperature_c",
    "wind_speed_ms",
    "cloud_cover_pct",
    "global_radiation_kj_m2",
]


def build_feature_frame(energy: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Build known-at-forecast-time features on a complete UTC hourly grid."""
    energy = energy.copy()
    weather = weather.copy()
    energy["timestamp_utc"] = pd.to_datetime(energy["timestamp_utc"], utc=True)
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)
    frame = energy.merge(weather, on="timestamp_utc", how="left", suffixes=("", "_weather"))
    frame = frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")

    local = frame["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    hour = local.dt.hour + local.dt.minute / 60
    week_hour = local.dt.dayofweek * 24 + hour
    month = local.dt.month
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["week_sin"] = np.sin(2 * np.pi * week_hour / 168)
    frame["week_cos"] = np.cos(2 * np.pi * week_hour / 168)
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)
    frame["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    # A deterministic proxy covers the largest fixed-date German holidays while
    # avoiding a country-specific runtime dependency.
    month_day = local.dt.strftime("%m-%d")
    frame["is_holiday_proxy"] = month_day.isin(
        {"01-01", "05-01", "10-03", "12-24", "12-25", "12-26", "12-31"}
    ).astype(int)

    frame["price_lag_24"] = frame[TARGET].shift(24)
    frame["price_lag_48"] = frame[TARGET].shift(48)
    frame["price_lag_168"] = frame[TARGET].shift(168)
    # The 24-hour offset ensures every member of a next-day batch is computed
    # only from prices already known when the forecast is issued.
    issued_history = frame[TARGET].shift(24)
    frame["price_roll_mean_24"] = issued_history.rolling(24, min_periods=18).mean()
    frame["price_roll_std_168"] = issued_history.rolling(168, min_periods=72).std()

    actual_renewable_columns = [
        column
        for column in (
            "solar_mwh",
            "wind_onshore_mwh",
            "wind_offshore_mwh",
            "biomass_mwh",
            "hydro_mwh",
            "other_renewable_mwh",
        )
        if column in frame
    ]
    forecast_renewable_columns = [
        column
        for column in (
            "solar_forecast_mwh",
            "wind_onshore_forecast_mwh",
            "wind_offshore_forecast_mwh",
        )
        if column in frame
    ]
    actual_renewable = (
        frame[actual_renewable_columns].sum(axis=1, min_count=1)
        if actual_renewable_columns
        else pd.Series(np.nan, index=frame.index)
    )
    forecast_renewable = (
        frame[forecast_renewable_columns].sum(axis=1, min_count=1)
        if forecast_renewable_columns
        else pd.Series(np.nan, index=frame.index)
    )
    frame["renewable_forecast_mwh"] = forecast_renewable.fillna(actual_renewable.shift(168))
    load_fallback = frame.get("load_mwh", pd.Series(np.nan, index=frame.index)).shift(168)
    if "load_forecast_mwh" not in frame:
        frame["load_forecast_mwh"] = load_fallback
    else:
        frame["load_forecast_mwh"] = frame["load_forecast_mwh"].fillna(load_fallback)
    frame["renewable_forecast_share"] = (
        frame["renewable_forecast_mwh"] / frame["load_forecast_mwh"]
    ).clip(0, 1.3)
    frame["forecast_carbon_intensity_gco2_kwh"] = (
        590 - 520 * frame["renewable_forecast_share"]
    ).clip(35, 850)

    weather_columns = [
        "temperature_c",
        "wind_speed_ms",
        "cloud_cover_pct",
        "global_radiation_kj_m2",
    ]
    for column in weather_columns:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = frame[column].interpolate(limit=6).ffill().bfill()
    return frame.reset_index(drop=True)


def training_rows(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Return labeled rows after the longest lag is available."""
    required = [TARGET, "price_lag_168", "load_forecast_mwh"]
    return feature_frame.dropna(subset=required).reset_index(drop=True)


def forecast_rows(feature_frame: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Select unlabeled future rows and validate a complete next-day horizon."""
    rows = feature_frame.loc[feature_frame[TARGET].isna()].head(horizon).copy()
    if len(rows) < horizon:
        raise ValueError(
            f"Expected {horizon} unlabeled future hours, found {len(rows)}. "
            "Ingest next-day SMARD/DWD forecasts or run demo generation first."
        )
    if rows[FEATURE_COLUMNS].isna().all(axis=1).any():
        raise ValueError("At least one forecast row has no usable model features")
    return rows

