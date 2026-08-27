with source as (
    select *
    from read_parquet('{{ var("data_dir") }}/silver/energy_hourly.parquet')
)

select
    cast(timestamp_utc as timestamptz) as timestamp_utc,
    cast(price_eur_mwh as double) as price_eur_mwh,
    cast(load_mwh as double) as load_mwh,
    cast(load_forecast_mwh as double) as load_forecast_mwh,
    cast(solar_mwh as double) as solar_mwh,
    cast(wind_onshore_mwh as double) as wind_onshore_mwh,
    cast(wind_offshore_mwh as double) as wind_offshore_mwh,
    cast(renewable_mwh as double) as renewable_mwh,
    cast(renewable_share as double) as renewable_share,
    cast(carbon_intensity_gco2_kwh as double) as carbon_intensity_gco2_kwh,
    cast(solar_forecast_mwh as double) as solar_forecast_mwh,
    cast(wind_onshore_forecast_mwh as double) as wind_onshore_forecast_mwh,
    cast(wind_offshore_forecast_mwh as double) as wind_offshore_forecast_mwh,
    coalesce(cast(is_forecast_horizon as boolean), false) as is_forecast_horizon,
    coalesce(cast(data_mode as varchar), 'live') as data_mode
from source

