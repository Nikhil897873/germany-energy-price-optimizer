"""Command-line interface for ingestion, training, and serving preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from gridshift.config import Settings
from gridshift.demo import generate_demo_data
from gridshift.features import build_feature_frame
from gridshift.ingest.dwd import ingest_dwd_forecast, ingest_dwd_observations
from gridshift.ingest.smard import ingest_smard
from gridshift.models import run_modeling
from gridshift.pipeline import run_dbt_build, run_demo_pipeline, run_live_pipeline
from gridshift.simulation import run_default_simulations
from gridshift.storage import atomic_write_parquet, read_hourly

app = typer.Typer(
    name="gridshift",
    no_args_is_help=True,
    help="German day-ahead price forecasts and flexible-load scheduling.",
)


def _settings(data_dir: Path) -> Settings:
    return Settings(data_dir=data_dir)


@app.command("generate-demo")
def generate_demo(
    days: Annotated[int, typer.Option(min=120, help="History length in days.")] = 730,
    data_dir: Annotated[Path, typer.Option(help="Runtime data directory.")] = Path("data"),
) -> None:
    energy, weather = generate_demo_data(_settings(data_dir), days=days)
    typer.echo(f"Wrote {len(energy):,} energy rows and {len(weather):,} weather rows.")


@app.command("ingest-smard")
def smard_command(
    start: Annotated[str, typer.Option(help="Inclusive start date (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option(help="Inclusive end date (YYYY-MM-DD).")],
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
    strict: Annotated[
        bool, typer.Option(help="Fail if any optional series is unavailable.")
    ] = False,
) -> None:
    frame = ingest_smard(
        _settings(data_dir), start, pd.Timestamp(end) + pd.Timedelta(hours=23), strict=strict
    )
    typer.echo(f"Wrote {len(frame):,} SMARD hourly rows.")


@app.command("ingest-dwd")
def dwd_command(
    start: Annotated[str, typer.Option(help="Inclusive start date (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option(help="Inclusive end date (YYYY-MM-DD).")],
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
    include_forecast: Annotated[bool, typer.Option(help="Append latest MOSMIX-L forecast.")] = True,
) -> None:
    settings = _settings(data_dir)
    frame = ingest_dwd_observations(settings, start, pd.Timestamp(end) + pd.Timedelta(hours=23))
    if include_forecast:
        ingest_dwd_forecast(settings)
    typer.echo(f"Wrote {len(frame):,} DWD observed hourly rows.")


@app.command("train")
def train_command(
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
    splits: Annotated[int, typer.Option(min=2, max=8)] = 4,
) -> None:
    settings = _settings(data_dir)
    energy, weather = read_hourly(settings.energy_path), read_hourly(settings.weather_path)
    features = build_feature_frame(energy, weather)
    atomic_write_parquet(features, settings.silver_dir / "model_features.parquet")
    metrics, _, forecast = run_modeling(features, settings, n_splits=splits)
    typer.echo(metrics.to_string(index=False))
    typer.echo(f"\nProduced {len(forecast)} next-day forecast rows.")


@app.command("simulate")
def simulate_command(
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    settings = _settings(data_dir)
    forecast = read_hourly(settings.artifacts_dir / "forecast_next_day.parquet")
    _, summaries = run_default_simulations(forecast, settings)
    typer.echo(summaries.to_string(index=False))


@app.command("dbt-build")
def dbt_command(
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    run_dbt_build(_settings(data_dir))


@app.command("run")
def run_command(
    demo: Annotated[bool, typer.Option(help="Use deterministic synthetic data.")] = False,
    start: Annotated[str, typer.Option(help="Live-mode start date.")] = "2023-01-01",
    end: Annotated[str | None, typer.Option(help="Live-mode end date (defaults to today).")] = None,
    days: Annotated[int, typer.Option(min=120, help="Demo history length.")] = 730,
    splits: Annotated[int, typer.Option(min=2, max=8)] = 4,
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    settings = _settings(data_dir)
    if demo:
        result = run_demo_pipeline(settings, days=days, n_splits=splits)
    else:
        end = end or str(pd.Timestamp.now(tz="Europe/Berlin").date())
        result = run_live_pipeline(settings, start=start, end=end, n_splits=splits)
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()

