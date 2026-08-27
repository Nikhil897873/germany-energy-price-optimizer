# Price forecast model card

## Intended use

Rank the next 24 German market hours for flexible electricity consumption and identify likely negative-price periods. The output supports planning and scenario analysis for factories, charging fleets, and storage operators.

## Not intended for

Autonomous market bidding, regulated financial reporting, settlement, imbalance management, or safety-critical control. Forecast prices exclude organization-specific taxes, levies, network tariffs, contracts, and transaction costs unless added downstream.

## Candidate models

1. Seasonal naive: price from the same hour one week earlier.
2. Linear: standardized Ridge regression for a transparent multivariate baseline.
3. LightGBM: nonlinear boosted trees. If the optional LightGBM package is absent, the code uses scikit-learn histogram gradient boosting while retaining the comparison slot; production and CI install LightGBM.

The winner minimizes mean absolute error across expanding-window folds. The evaluation also reports RMSE, sMAPE, negative-event precision/recall/F1, and empirical 90% interval coverage.

## Inputs

Calendar cycles; 24/48/168-hour price lags; price rolling statistics delayed by 24 hours; load and wind/solar forecasts; and DWD temperature, wind, cloud, and radiation. Forecast inputs can fall back to seven-day lags when a published series is unavailable. That fallback should be monitored because it reduces responsiveness to regime changes.

## Key risks

- Extreme prices are rare, policy-sensitive, and harder to estimate than ordinary hours.
- Generation-weighted carbon intensity is not marginal carbon intensity and omits cross-border flows.
- Equal-weight weather stations do not reflect the geographical distribution of wind/PV capacity.
- SMARD revisions can change historical values after first ingestion.
- Structural market changes, fuel prices, outages, and transmission constraints are not explicitly modeled.
- Residual intervals can under-cover during regime shifts.

## Monitoring recommendations

Track data freshness/completeness, feature distributions, hourly MAE and bias, negative-event precision/recall, interval coverage, savings realized versus simulated, and forecast/model age. Retrain on a schedule and after material drift; retain the seasonal baseline as an operational fallback.


