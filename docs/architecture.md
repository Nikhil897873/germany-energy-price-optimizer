# Architecture decisions

## Design goals

GridShift DE separates reproducible analytics from volatile external interfaces. Raw responses remain inspectable, canonical hourly contracts are engine-independent Parquet, dbt owns business-facing transformations, and the application consumes versionable artifacts rather than fitting models during page loads.

## Time and grain

The canonical key is `timestamp_utc` at one-hour grain. Europe/Berlin time is derived for human display and calendar features. This choice preserves both distinct 02:00 hours when clocks move backward and naturally represents the missing hour in spring.

SMARD values are energy per interval (MWh), except price (EUR/MWh) and derived ratios/intensities. DWD station measurements are averaged to a deliberately simple national proxy. A production implementation could use capacity-weighted weather regions.

## Source ingestion

The SMARD client first requests the available window timestamps, then downloads only windows overlapping the requested interval. Responses are retried, retained under the filter ID, normalized, and outer-joined. Optional forecast series can fail independently; the metadata file lists every omission.

The DWD client reads directory listings, downloads matching historical/recent station archives, and parses product files directly from ZIP bytes. MOSMIX-L station KMZ files provide forward weather. National fields are equal-weight station means with a station-count quality field.

## Model boundary

Feature construction and training are separate from source ingestion. A 24-hour feature offset prevents later hours in one next-day batch from accidentally depending on earlier target hours in that same batch. Walk-forward folds use an additional 24-hour train/test embargo.

The best model is selected by MAE, then fitted to all labeled rows. Prediction intervals use held-out residual quantiles during backtesting and full-history residual quantiles for the final portfolio artifact. For high-stakes operation, replace this with a dedicated rolling calibration window and formal coverage monitoring.

## Serving

Parquet artifacts make local review simple. DuckDB supplies low-ops analytical serving. dbt supplies lineage, contracts, documentation, and tests. Streamlit reads already-materialized outputs and recomputes only the small scheduling scenario controlled by the user.

## Production evolution

Recommended next steps are object storage for bronze data, an orchestrator with retries/SLA monitoring, a managed analytical database, model/feature registries, scheduled rolling calibration, data-drift checks, and a constraint solver representing the actual process or battery state of charge.

