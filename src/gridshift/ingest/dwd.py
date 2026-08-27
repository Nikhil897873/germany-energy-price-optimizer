"""DWD CDC observations and MOSMIX-L forecast ingestion."""

from __future__ import annotations

import io
import re
import time
import zipfile
from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests

from gridshift.config import Settings
from gridshift.storage import atomic_write_parquet

CDC_PRODUCTS = {
    "air_temperature": {"prefix": "TU", "field": "TT_TU", "output": "temperature_c"},
    "wind": {"prefix": "FF", "field": "F", "output": "wind_speed_ms"},
}

# A geographically spread default sample. CDC and MOSMIX use different station
# identifiers, so both sets are intentionally explicit and configurable.
DEFAULT_CDC_STATIONS = ("00433", "01420", "01975", "03379", "04931")
DEFAULT_MOSMIX_STATIONS = ("10147", "10382", "10410", "10637", "10868")


class DwdError(RuntimeError):
    """Raised when no usable DWD data can be obtained."""


def parse_cdc_zip(content: bytes, product: str, station_id: str) -> pd.DataFrame:
    """Parse a DWD CDC product zip without extracting it to disk."""
    metadata = CDC_PRODUCTS[product]
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = [name for name in archive.namelist() if Path(name).name.startswith("produkt_")]
        if not members:
            raise DwdError("CDC archive contains no produkt_ file")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, sep=";", encoding="latin-1", dtype=str)
    frame.columns = [column.strip() for column in frame.columns]
    if "MESS_DATUM" not in frame or metadata["field"] not in frame:
        raise DwdError(f"Unexpected columns in {product} archive")
    result = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                frame["MESS_DATUM"].str.strip(), format="%Y%m%d%H", utc=True, errors="coerce"
            ),
            "station_id": station_id,
            "variable": metadata["output"],
            "value": pd.to_numeric(frame[metadata["field"]], errors="coerce"),
        }
    )
    result.loc[result["value"] <= -999, "value"] = pd.NA
    return result.dropna(subset=["timestamp_utc"])


def parse_mosmix_kmz(content: bytes, station_id: str) -> pd.DataFrame:
    """Parse selected hourly variables from a station-level MOSMIX-L KMZ."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        kml_names = [name for name in archive.namelist() if name.lower().endswith(".kml")]
        if not kml_names:
            raise DwdError("MOSMIX archive contains no KML file")
        root = ElementTree.fromstring(archive.read(kml_names[0]))

    timestamps = [
        pd.Timestamp(element.text)
        for element in root.iter()
        if _local_name(element.tag) == "TimeStep" and element.text
    ]
    values_by_element: dict[str, list[float | None]] = {}
    for element in root.iter():
        if _local_name(element.tag) != "Forecast":
            continue
        name = next(
            (value for key, value in element.attrib.items() if _local_name(key) == "elementName"),
            None,
        )
        value_node = next(
            (child for child in element.iter() if _local_name(child.tag) == "value"), None
        )
        if name and value_node is not None and value_node.text:
            values_by_element[name] = [_number_or_none(token) for token in value_node.text.split()]

    result = pd.DataFrame({"timestamp_utc": pd.to_datetime(timestamps, utc=True)})
    mapping = {
        "TTT": "temperature_c",
        "FF": "wind_speed_ms",
        "N": "cloud_cover_pct",
        "Rad1h": "global_radiation_kj_m2",
    }
    for source, target in mapping.items():
        values = values_by_element.get(source)
        if values and len(values) == len(result):
            result[target] = values
    if "temperature_c" in result:
        result["temperature_c"] = result["temperature_c"] - 273.15
    result["station_id"] = station_id
    return result


class DwdClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "gridshift-de/0.1 (+open-source research)"})

    def _get(self, url: str, attempts: int = 3) -> bytes:
        error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.get(url, timeout=self.settings.http_timeout_seconds)
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                error = exc
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
        raise DwdError(f"Failed to download {url}: {error}") from error

    def _archive_links(self, product: str, period: str, station_id: str) -> list[str]:
        metadata = CDC_PRODUCTS[product]
        base = f"{self.settings.dwd_cdc_base_url}/{product}/{period}/"
        listing = self._get(base).decode("utf-8", errors="ignore")
        prefix = re.escape(str(metadata["prefix"]))
        station = re.escape(station_id)
        names = re.findall(rf'href="(stundenwerte_{prefix}_{station}_[^"]+\.zip)"', listing)
        return [base + name for name in sorted(set(names))]

    def fetch_cdc_station(
        self, product: str, station_id: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for period in ("historical", "recent"):
            for url in self._archive_links(product, period, station_id):
                content = self._get(url)
                target = self.settings.bronze_dir / "dwd" / "cdc" / product / Path(url).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                frames.append(parse_cdc_zip(content, product, station_id))
        if not frames:
            return pd.DataFrame(columns=["timestamp_utc", "station_id", "variable", "value"])
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(
            ["timestamp_utc", "station_id", "variable"], keep="last"
        )
        return combined.loc[combined["timestamp_utc"].between(start, end)]

    def fetch_mosmix_station(self, station_id: str) -> pd.DataFrame:
        filename = f"MOSMIX_L_LATEST_{station_id}.kmz"
        url = f"{self.settings.dwd_mosmix_base_url}/{station_id}/kml/{filename}"
        content = self._get(url)
        target = self.settings.bronze_dir / "dwd" / "mosmix" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return parse_mosmix_kmz(content, station_id)


def ingest_dwd_observations(
    settings: Settings,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    stations: Iterable[str] = DEFAULT_CDC_STATIONS,
) -> pd.DataFrame:
    """Download CDC observations and aggregate stations to a national proxy."""
    settings.ensure_directories()
    start_ts, end_ts = _as_utc(start), _as_utc(end)
    client = DwdClient(settings)
    frames: list[pd.DataFrame] = []
    for product in CDC_PRODUCTS:
        for station_id in stations:
            try:
                frame = client.fetch_cdc_station(product, station_id, start_ts, end_ts)
            except DwdError:
                continue
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise DwdError("No CDC station observations could be downloaded")
    long = pd.concat(frames, ignore_index=True)
    national = (
        long.groupby(["timestamp_utc", "variable"], as_index=False)["value"]
        .mean()
        .pivot(index="timestamp_utc", columns="variable", values="value")
        .reset_index()
        .rename_axis(columns=None)
    )
    counts = long.groupby("timestamp_utc")["station_id"].nunique().rename("station_count")
    national = national.merge(counts, on="timestamp_utc", how="left")
    for column in ("cloud_cover_pct", "global_radiation_kj_m2"):
        if column not in national:
            national[column] = float("nan")
    national["weather_is_forecast"] = False
    national["data_mode"] = "live"
    atomic_write_parquet(national, settings.weather_path)
    return national


def ingest_dwd_forecast(
    settings: Settings, stations: Iterable[str] = DEFAULT_MOSMIX_STATIONS
) -> pd.DataFrame:
    """Download MOSMIX-L station forecasts and append/merge them into weather silver."""
    settings.ensure_directories()
    client = DwdClient(settings)
    frames: list[pd.DataFrame] = []
    for station_id in stations:
        try:
            frames.append(client.fetch_mosmix_station(station_id))
        except DwdError:
            continue
    if not frames:
        raise DwdError("No MOSMIX station forecasts could be downloaded")
    combined = pd.concat(frames, ignore_index=True)
    numeric = [
        column for column in combined.columns if column not in {"timestamp_utc", "station_id"}
    ]
    forecast = combined.groupby("timestamp_utc", as_index=False)[numeric].mean()
    forecast["station_count"] = combined.groupby("timestamp_utc")["station_id"].nunique().values
    forecast["weather_is_forecast"] = True
    forecast["data_mode"] = "live"

    if settings.weather_path.exists():
        observed = pd.read_parquet(settings.weather_path)
        observed["timestamp_utc"] = pd.to_datetime(observed["timestamp_utc"], utc=True)
        observed["weather_is_forecast"] = observed.get("weather_is_forecast", False)
        cutoff = forecast["timestamp_utc"].min()
        result = pd.concat(
            [observed.loc[observed["timestamp_utc"] < cutoff], forecast], ignore_index=True
        )
    else:
        result = forecast
    result = result.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    atomic_write_parquet(result, settings.weather_path)
    return forecast


def _number_or_none(value: str) -> float | None:
    if value in {"-", "--", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _as_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
