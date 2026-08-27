import pandas as pd

from gridshift import cli


def test_ingestion_commands_report_rows(monkeypatch, tmp_path, capsys) -> None:
    energy = pd.DataFrame({"timestamp_utc": pd.date_range("2025-01-01", periods=2, tz="UTC")})
    weather = energy.iloc[:1].copy()
    forecast_calls: list[object] = []

    monkeypatch.setattr(cli, "generate_demo_data", lambda settings, days: (energy, weather))
    monkeypatch.setattr(cli, "ingest_smard", lambda *args, **kwargs: energy)
    monkeypatch.setattr(cli, "ingest_dwd_observations", lambda *args, **kwargs: weather)
    monkeypatch.setattr(
        cli, "ingest_dwd_forecast", lambda settings: forecast_calls.append(settings)
    )

    cli.generate_demo(days=120, data_dir=tmp_path)
    cli.smard_command("2025-01-01", "2025-01-02", tmp_path, strict=True)
    cli.dwd_command("2025-01-01", "2025-01-02", tmp_path, include_forecast=True)

    output = capsys.readouterr().out
    assert "2 energy rows" in output
    assert "2 SMARD hourly rows" in output
    assert "1 DWD observed hourly rows" in output
    assert len(forecast_calls) == 1


def test_train_and_simulate_commands(monkeypatch, tmp_path, capsys) -> None:
    source = pd.DataFrame({"timestamp_utc": pd.date_range("2025-01-01", periods=2, tz="UTC")})
    features = source.assign(price_eur_mwh=[1.0, 2.0])
    metrics = pd.DataFrame({"model": ["linear"], "mae_eur_mwh": [3.25]})
    forecast = pd.DataFrame({"predicted_price_eur_mwh": [1.0] * 24})
    summaries = pd.DataFrame({"objective": ["balanced"], "cost_savings_pct": [12.0]})
    reads = iter([source, source, forecast])
    writes: list[object] = []

    monkeypatch.setattr(cli, "read_hourly", lambda path: next(reads))
    monkeypatch.setattr(cli, "build_feature_frame", lambda energy, weather: features)
    monkeypatch.setattr(cli, "atomic_write_parquet", lambda frame, path: writes.append(path))
    monkeypatch.setattr(cli, "run_modeling", lambda *args, **kwargs: (metrics, None, forecast))
    monkeypatch.setattr(
        cli, "run_default_simulations", lambda forecast_frame, settings: (None, summaries)
    )

    cli.train_command(data_dir=tmp_path, splits=2)
    cli.simulate_command(data_dir=tmp_path)

    output = capsys.readouterr().out
    assert "Produced 24 next-day forecast rows" in output
    assert "balanced" in output
    assert writes


def test_dbt_and_run_commands(monkeypatch, tmp_path, capsys) -> None:
    calls: list[str] = []
    demo_result = {"selected_model": "lightgbm", "forecast_hours": 24}
    live_result = {"selected_model": "linear", "forecast_hours": 24}

    monkeypatch.setattr(cli, "run_dbt_build", lambda settings: calls.append("dbt"))
    monkeypatch.setattr(cli, "run_demo_pipeline", lambda *args, **kwargs: demo_result)
    monkeypatch.setattr(cli, "run_live_pipeline", lambda *args, **kwargs: live_result)

    cli.dbt_command(data_dir=tmp_path)
    cli.run_command(
        demo=True,
        start="2025-01-01",
        end=None,
        days=120,
        splits=2,
        data_dir=tmp_path,
    )
    cli.run_command(
        demo=False,
        start="2025-01-01",
        end="2025-01-31",
        days=120,
        splits=2,
        data_dir=tmp_path,
    )

    output = capsys.readouterr().out
    assert calls == ["dbt"]
    assert '"selected_model": "lightgbm"' in output
    assert '"selected_model": "linear"' in output
