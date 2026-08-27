import pandas as pd

from gridshift.config import Settings
from gridshift.demo import generate_demo_data
from gridshift.features import FEATURE_COLUMNS, build_feature_frame, forecast_rows, training_rows


def test_feature_frame_has_train_and_complete_future_rows(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, random_seed=7)
    energy, weather = generate_demo_data(settings, days=120, as_of="2025-01-01")

    features = build_feature_frame(energy, weather)
    train = training_rows(features)
    future = forecast_rows(features)

    assert train["timestamp_utc"].max() < future["timestamp_utc"].min()
    assert len(future) == 24
    assert not future[FEATURE_COLUMNS].isna().all(axis=1).any()
    assert future["price_eur_mwh"].isna().all()


def test_berlin_dst_hours_keep_unique_utc_keys(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, random_seed=8)
    energy, weather = generate_demo_data(settings, days=220, as_of="2025-04-01")
    features = build_feature_frame(energy, weather)

    assert features["timestamp_utc"].is_unique
    assert pd.api.types.is_datetime64_any_dtype(features["timestamp_utc"])
