with source as (
    select *
    from read_parquet('{{ var("data_dir") }}/silver/weather_hourly.parquet')
)

select
    cast(timestamp_utc as timestamptz) as timestamp_utc,
    cast(temperature_c as double) as temperature_c,
    cast(wind_speed_ms as double) as wind_speed_ms,
    cast(cloud_cover_pct as double) as cloud_cover_pct,
    cast(global_radiation_kj_m2 as double) as global_radiation_kj_m2,
    cast(station_count as integer) as station_count,
    coalesce(cast(weather_is_forecast as boolean), false) as weather_is_forecast
from source


