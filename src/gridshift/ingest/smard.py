"""Incremental ingestion for the public SMARD chart-data interface."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd
import requests

from gridshift.config import SMARD_SERIES, Settings, SmardSeries
from gridshift.storage import atomic_write_json, atomic_write_parquet


class SmardError(RuntimeError):
    """Raised when SMARD returns an invalid or unusable response."""


def parse_series_payload(payload: dict[str, Any], column: str) -> pd.DataFrame:
    """Normalize one SMARD ``series`` response into a UTC time series."""
    rows = payload.get("series")
    if not isinstance(rows, list):
        raise SmardError("SMARD response does not contain a 'series' list")
    normalized: list[tuple[Any, Any]] = []
    for row in rows:
        if isinstance(row, list | tuple) and len(row) >= 2:
            normalized.append((row[0], row[1]))
    frame = pd.DataFrame(normalized, columns=["timestamp_ms", column])
    if frame.empty:
        return pd.DataFrame(columns=["timestamp_utc", column])
    frame["timestamp_utc"] = pd.to_datetime(frame.pop("timestamp_ms"), unit="ms", utc=True)
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates("timestamp_utc", keep="last")


class SmardClient:
    """Resilient client for SMARD's timestamp-windowed JSON responses."""

    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "gridshift-de/0.1 (+open-source research)"})

    def _get_json(self, url: str, attempts: int = 3) -> dict[str, Any]:
        error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.get(url, timeout=self.settings.http_timeout_seconds)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise SmardError(f"Expected an object from {url}")
                return payload
            except (requests.RequestException, json.JSONDecodeError, SmardError) as exc:
                error = exc
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
        raise SmardError(f"Failed to download {url}: {error}") from error

    def available_windows(self, series: SmardSeries, resolution: str = "hour") -> list[int]:
        url = (
            f"{self.settings.smard_base_url}/{series.filter_id}/{series.region}"
            f"/index_{resolution}.json"
        )
        payload = self._get_json(url)
        timestamps = payload.get("timestamps", [])
        return sorted(int(value) for value in timestamps)

    def fetch_window(
        self, series: SmardSeries, timestamp_ms: int, resolution: str = "hour"
    ) -> pd.DataFrame:
        url = (
            f"{self.settings.smard_base_url}/{series.filter_id}/{series.region}/"
            f"{series.filter_id}_{series.region}_{resolution}_{timestamp_ms}.json"
        )
        payload = self._get_json(url)
        raw_path = (
            self.settings.bronze_dir / "smard" / str(series.filter_id) / f"{timestamp_ms}.json"
        )
        atomic_write_json(payload, raw_path)
        return parse_series_payload(payload, series.column)

    def fetch_series(
        self,
        series: SmardSeries,
        start: pd.Timestamp,
        end: pd.Timestamp,
        resolution: str = "hour",
    ) -> pd.DataFrame:
        start = _as_utc(start)
        end = _as_utc(end)
        windows = self.available_windows(series, resolution)
        # A chart-data window covers roughly a week. Include the preceding index
        # timestamp so a range beginning inside a window is never truncated.
        candidates = [
            value
            for value in windows
            if pd.Timestamp(value, unit="ms", tz="UTC") <= end
            and pd.Timestamp(value, unit="ms", tz="UTC") >= start - pd.Timedelta(days=8)
        ]
        frames = [self.fetch_window(series, value, resolution) for value in candidates]
        if not frames:
            return pd.DataFrame(columns=["timestamp_utc", series.column])
        frame = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp_utc", keep="last")
        return frame.loc[frame["timestamp_utc"].between(start, end)].sort_values("timestamp_utc")


def ingest_smard(
    settings: Settings,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    selected_columns: Iterable[str] | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Download configured SMARD series and materialize an hourly wide table.

    Forecast series are not present for every historical market-area combination.
    In non-strict mode unavailable optional series are logged into ingestion metadata
    and the remaining data are still written.
    """
    settings.ensure_directories()
    start_ts, end_ts = _as_utc(start), _as_utc(end)
    wanted = set(selected_columns or ())
    series_list = [item for item in SMARD_SERIES if not wanted or item.column in wanted]
    client = SmardClient(settings)
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for series in series_list:
        try:
            frame = client.fetch_series(series, start_ts, end_ts)
            if not frame.empty:
                frames.append(frame)
        except SmardError as exc:
            if strict:
                raise
            failures[series.column] = str(exc)
    if not frames:
        raise SmardError("No SMARD series could be downloaded")
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="timestamp_utc", how="outer", validate="one_to_one")
    for series in SMARD_SERIES:
        if series.column not in result:
            result[series.column] = float("nan")
    result["is_forecast_horizon"] = False
    result["data_mode"] = "live"
    result = add_energy_aggregates(result.sort_values("timestamp_utc").reset_index(drop=True))
    atomic_write_parquet(result, settings.energy_path)
    atomic_write_json(
        {
            "source": "Bundesnetzagentur | SMARD.de",
            "license": "CC BY 4.0",
            "retrieved_at_utc": pd.Timestamp.now(tz="UTC"),
            "range": {"start": start_ts, "end": end_ts},
            "rows": len(result),
            "failed_optional_series": failures,
        },
        settings.bronze_dir / "smard" / "ingestion_metadata.json",
    )
    return result


def add_energy_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add renewable totals, shares, and a transparent carbon-intensity proxy."""
    result = frame.copy()
    renewable = [
        "solar_mwh",
        "wind_onshore_mwh",
        "wind_offshore_mwh",
        "biomass_mwh",
        "hydro_mwh",
        "other_renewable_mwh",
    ]
    present_renewable = [column for column in renewable if column in result]
    if present_renewable:
        result["renewable_mwh"] = result[present_renewable].sum(axis=1, min_count=1)
        denominator = result.get("load_mwh", result[present_renewable].sum(axis=1))
        result["renewable_share"] = (result["renewable_mwh"] / denominator).clip(0, 1.2)

    emission_factors = {
        "lignite_mwh": 1100.0,
        "hard_coal_mwh": 900.0,
        "natural_gas_mwh": 400.0,
        "other_conventional_mwh": 650.0,
        "nuclear_mwh": 12.0,
        "biomass_mwh": 230.0,
        "hydro_mwh": 24.0,
        "solar_mwh": 45.0,
        "wind_onshore_mwh": 11.0,
        "wind_offshore_mwh": 12.0,
        "other_renewable_mwh": 40.0,
    }
    present_generation = [column for column in emission_factors if column in result]
    if present_generation:
        emissions = sum(
            result[column].fillna(0) * emission_factors[column] for column in present_generation
        )
        generation = result[present_generation].sum(axis=1, min_count=1)
        result["carbon_intensity_gco2_kwh"] = (emissions / generation).clip(0, 1200)
    return result


def _as_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")

