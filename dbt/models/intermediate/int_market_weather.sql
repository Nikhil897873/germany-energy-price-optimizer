select
    energy.*,
    weather.temperature_c,
    weather.wind_speed_ms,
    weather.cloud_cover_pct,
    weather.global_radiation_kj_m2,
    weather.station_count,
    weather.weather_is_forecast,
    timezone('Europe/Berlin', energy.timestamp_utc) as timestamp_de
from {{ ref('stg_energy_hourly') }} as energy
left join {{ ref('stg_weather_hourly') }} as weather
    using (timestamp_utc)

