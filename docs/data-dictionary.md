# Data dictionary

## `energy_hourly.parquet`

| Column | Unit | Description |
|---|---:|---|
| `timestamp_utc` | UTC | Unique start of hourly interval |
| `price_eur_mwh` | EUR/MWh | DE-LU day-ahead wholesale price |
| `load_mwh` | MWh | Realized German total grid load |
| `load_forecast_mwh` | MWh | Day-ahead load forecast |
| `solar_mwh` | MWh | Realized photovoltaic generation |
| `wind_onshore_mwh` | MWh | Realized onshore wind generation |
| `wind_offshore_mwh` | MWh | Realized offshore wind generation |
| `renewable_mwh` | MWh | Sum of available renewable categories |
| `renewable_share` | ratio | Renewable generation divided by load, capped only for quality protection |
| `*_forecast_mwh` | MWh | Published solar/onshore/offshore forecast fields |
| `carbon_intensity_gco2_kwh` | gCO2e/kWh | Generation-weighted lifecycle emissions proxy |
| `is_forecast_horizon` | boolean | True for unlabeled next-day rows |
| `data_mode` | text | `live` or `synthetic_demo` provenance marker |

Fuel-specific generation fields follow the same `{fuel}_mwh` convention.

## `weather_hourly.parquet`

| Column | Unit | Description |
|---|---:|---|
| `timestamp_utc` | UTC | Unique hourly timestamp |
| `temperature_c` | °C | Mean selected-station air temperature |
| `wind_speed_ms` | m/s | Mean selected-station wind speed |
| `cloud_cover_pct` | % | MOSMIX cloud cover when available |
| `global_radiation_kj_m2` | kJ/m² | MOSMIX hourly global radiation when available |
| `station_count` | count | Contributing stations before national aggregation |
| `weather_is_forecast` | boolean | Forecast rather than observation marker |

## `forecast_next_day.parquet`

| Column | Unit | Description |
|---|---:|---|
| `predicted_price_eur_mwh` | EUR/MWh | Selected-model point forecast |
| `lower_90_eur_mwh` | EUR/MWh | Lower empirical 90% interval bound |
| `upper_90_eur_mwh` | EUR/MWh | Upper empirical 90% interval bound |
| `negative_price_probability` | probability | Normal residual approximation of price below zero |
| `negative_price_predicted` | boolean | Point forecast below zero |
| `renewable_forecast_share` | ratio | Variable renewable forecast divided by load forecast |
| `forecast_carbon_intensity_gco2_kwh` | gCO2e/kWh | Renewable-share-based next-day carbon proxy |

## Flexibility artifacts

`flex_schedules.parquet` contains every hourly scenario, including baseline/optimized MWh, costs, emissions, load shift, and `CONSUME`/`REDUCE`/`HOLD` signal. `flex_summary.csv` contains scenario totals and percentage savings.

