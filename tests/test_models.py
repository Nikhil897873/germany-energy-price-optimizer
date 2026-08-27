from gridshift.config import Settings
from gridshift.demo import generate_demo_data
from gridshift.features import build_feature_frame
from gridshift.models import walk_forward_backtest


def test_walk_forward_outputs_all_comparators(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, random_seed=11)
    energy, weather = generate_demo_data(settings, days=130, as_of="2025-01-01")
    features = build_feature_frame(energy, weather)

    metrics, predictions = walk_forward_backtest(features, seed=11, n_splits=2, test_hours=72)

    assert set(metrics["model"]) == {"seasonal_naive", "linear", "lightgbm"}
    assert predictions["actual_price_eur_mwh"].notna().all()
    assert (metrics["mae_eur_mwh"] >= 0).all()
    assert (metrics["interval_coverage_90"].between(0, 1)).all()
