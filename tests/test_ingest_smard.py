import pandas as pd
import pytest

from gridshift.ingest.smard import SmardError, add_energy_aggregates, parse_series_payload


def test_parse_smard_series_normalizes_utc_and_values() -> None:
    payload = {"series": [[1_700_000_000_000, 42.5], [1_700_003_600_000, None]]}
    frame = parse_series_payload(payload, "price_eur_mwh")

    assert list(frame.columns) == ["price_eur_mwh", "timestamp_utc"]
    assert str(frame["timestamp_utc"].dt.tz) == "UTC"
    assert frame.iloc[0]["price_eur_mwh"] == pytest.approx(42.5)
    assert pd.isna(frame.iloc[1]["price_eur_mwh"])


def test_parse_smard_rejects_invalid_payload() -> None:
    with pytest.raises(SmardError, match="series"):
        parse_series_payload({"not_series": []}, "value")


def test_energy_aggregates_are_bounded_and_interpretable() -> None:
    frame = pd.DataFrame(
        {
            "load_mwh": [100.0],
            "solar_mwh": [30.0],
            "wind_onshore_mwh": [20.0],
            "natural_gas_mwh": [50.0],
        }
    )
    result = add_energy_aggregates(frame)

    assert result.loc[0, "renewable_mwh"] == pytest.approx(50.0)
    assert result.loc[0, "renewable_share"] == pytest.approx(0.5)
    assert 0 < result.loc[0, "carbon_intensity_gco2_kwh"] < 500

