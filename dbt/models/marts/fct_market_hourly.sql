select
    timestamp_utc,
    timestamp_de,
    price_eur_mwh,
    load_mwh,
    load_forecast_mwh,
    solar_mwh,
    wind_onshore_mwh,
    wind_offshore_mwh,
    renewable_mwh,
    renewable_share,
    carbon_intensity_gco2_kwh,
    solar_forecast_mwh,
    wind_onshore_forecast_mwh,
    wind_offshore_forecast_mwh,
    (
        coalesce(solar_forecast_mwh, 0)
        + coalesce(wind_onshore_forecast_mwh, 0)
        + coalesce(wind_offshore_forecast_mwh, 0)
    ) as variable_renewable_forecast_mwh,
    is_forecast_horizon,
    weather_is_forecast,
    temperature_c,
    wind_speed_ms,
    cloud_cover_pct,
    global_radiation_kj_m2,
    data_mode
from {{ ref('int_market_weather') }}


