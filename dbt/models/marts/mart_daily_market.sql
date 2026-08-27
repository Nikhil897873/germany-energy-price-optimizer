select
    cast(timestamp_de as date) as market_date,
    avg(price_eur_mwh) as average_price_eur_mwh,
    min(price_eur_mwh) as minimum_price_eur_mwh,
    max(price_eur_mwh) as maximum_price_eur_mwh,
    count(*) filter (where price_eur_mwh < 0) as negative_price_hours,
    sum(load_mwh) as consumption_mwh,
    sum(renewable_mwh) as renewable_generation_mwh,
    avg(renewable_share) as average_renewable_share,
    avg(carbon_intensity_gco2_kwh) as average_carbon_intensity_gco2_kwh,
    max(data_mode) as data_mode
from {{ ref('fct_market_hourly') }}
where not is_forecast_horizon
group by 1

