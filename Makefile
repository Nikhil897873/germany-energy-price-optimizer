.PHONY: install demo live dbt dashboard test lint clean

install:
	python -m pip install -e ".[all,dev]"

demo:
	gridshift run --demo --days 730

live:
	gridshift run --start 2023-01-01 --end 2025-12-31

dbt:
	gridshift dbt-build

dashboard:
	streamlit run dashboard/app.py

test:
	pytest --cov=gridshift --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

clean:
	python scripts/clean.py

