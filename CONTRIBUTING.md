# Contributing

1. Open an issue for material source-schema or modeling changes.
2. Create a focused branch and install `pip install -e ".[all,dev]"`.
3. Add tests for ingestion contracts, time-boundary behavior, and any decision logic.
4. Run `ruff check .`, `ruff format --check .`, `pytest`, and `gridshift dbt-build`.
5. Explain data lineage, target-leakage implications, and model/decision impact in the pull request.

Do not commit downloaded source data, fitted models, credentials, or `.env`. Keep timestamps in UTC at storage boundaries and add local timezone only as a derived field.


