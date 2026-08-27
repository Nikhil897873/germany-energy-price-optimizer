"""Application settings and source-series metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class SmardSeries:
    filter_id: int
    column: str
    region: str = "DE"
    unit: str = "MWh"


# SMARD chart-data filters. Forecast filters are the public module IDs without
# their UI category prefixes (for example, UI module 2000125 -> filter 125).
SMARD_SERIES: tuple[SmardSeries, ...] = (
    SmardSeries(4169, "price_eur_mwh", "DE-LU", "EUR/MWh"),
    SmardSeries(410, "load_mwh"),
    SmardSeries(411, "load_forecast_mwh"),
    SmardSeries(4068, "solar_mwh"),
    SmardSeries(4067, "wind_onshore_mwh"),
    SmardSeries(1225, "wind_offshore_mwh"),
    SmardSeries(4066, "biomass_mwh"),
    SmardSeries(1226, "hydro_mwh"),
    SmardSeries(1228, "other_renewable_mwh"),
    SmardSeries(1223, "lignite_mwh"),
    SmardSeries(4069, "hard_coal_mwh"),
    SmardSeries(4071, "natural_gas_mwh"),
    SmardSeries(1224, "nuclear_mwh"),
    SmardSeries(1227, "other_conventional_mwh"),
    SmardSeries(125, "solar_forecast_mwh"),
    SmardSeries(123, "wind_onshore_forecast_mwh"),
    SmardSeries(3791, "wind_offshore_forecast_mwh"),
)


class Settings(BaseSettings):
    """Environment-driven runtime settings."""

    model_config = SettingsConfigDict(env_prefix="GRIDSHIFT_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    smard_base_url: str = "https://www.smard.de/app/chart_data"
    dwd_cdc_base_url: str = (
        "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
    )
    dwd_mosmix_base_url: str = (
        "https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_L/single_stations"
    )
    http_timeout_seconds: int = 60
    random_seed: int = 42

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def energy_path(self) -> Path:
        return self.silver_dir / "energy_hourly.parquet"

    @property
    def weather_path(self) -> Path:
        return self.silver_dir / "weather_hourly.parquet"

    def ensure_directories(self) -> None:
        for path in (self.bronze_dir, self.silver_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)
