import numpy as np
import pandas as pd
import pytest

from gridshift.simulation import simulate_flexible_load


@pytest.fixture
def forecast() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC"),
            "predicted_price_eur_mwh": np.linspace(-20, 120, 24),
            "forecast_carbon_intensity_gco2_kwh": np.linspace(80, 700, 24),
        }
    )


def test_cost_dispatch_conserves_energy_and_reduces_cost(forecast) -> None:
    schedule, summary = simulate_flexible_load(
        forecast, flexible_share=0.2, objective="cost", baseline_mwh=10
    )

    assert schedule["optimized_consumption_mwh"].sum() == pytest.approx(240)
    assert summary["optimized_cost_eur"] <= summary["baseline_cost_eur"]
    assert set(schedule["dispatch_signal"]) >= {"CONSUME", "REDUCE"}


def test_emissions_dispatch_reduces_emissions(forecast) -> None:
    _, summary = simulate_flexible_load(
        forecast, flexible_share=0.1, objective="emissions", baseline_mwh=5
    )
    assert summary["emissions_savings_kg"] > 0


def test_invalid_flexibility_is_rejected(forecast) -> None:
    with pytest.raises(ValueError, match="flexible_share"):
        simulate_flexible_load(forecast, flexible_share=0.9)

