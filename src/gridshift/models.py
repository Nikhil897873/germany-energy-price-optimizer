"""Walk-forward model comparison and persisted next-day forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridshift.config import Settings
from gridshift.features import FEATURE_COLUMNS, TARGET, forecast_rows, training_rows
from gridshift.storage import atomic_write_json, atomic_write_parquet


@dataclass
class FittedForecast:
    model_name: str
    estimator: RegressorMixin | None
    feature_columns: list[str]
    residual_lower: float
    residual_upper: float
    residual_std: float
    trained_through_utc: str


def _make_estimator(name: str, seed: int) -> RegressorMixin | None:
    if name == "seasonal_naive":
        return None
    if name == "linear":
        return TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=12.0)),
                ]
            ),
            transformer=StandardScaler(),
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=350,
                learning_rate=0.035,
                num_leaves=24,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.5,
                random_state=seed,
                verbosity=-1,
                n_jobs=-1,
            )
        except ImportError:
            return HistGradientBoostingRegressor(
                learning_rate=0.06,
                max_iter=250,
                max_leaf_nodes=24,
                l2_regularization=1.5,
                random_state=seed,
            )
    raise ValueError(f"Unknown model: {name}")


def walk_forward_backtest(
    feature_frame: pd.DataFrame,
    seed: int = 42,
    n_splits: int = 4,
    test_hours: int = 24 * 14,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare all models on expanding windows with a 24-hour embargo."""
    data = training_rows(feature_frame)
    minimum_train = max(24 * 90, len(data) - n_splits * test_hours - 24)
    if len(data) < minimum_train + test_hours + 24:
        raise ValueError("Not enough history for walk-forward backtesting")

    names = ("seasonal_naive", "linear", "lightgbm")
    predictions: list[pd.DataFrame] = []
    for fold in range(n_splits):
        test_start = len(data) - (n_splits - fold) * test_hours
        test_end = test_start + test_hours
        train_end = test_start - 24
        if train_end < minimum_train:
            continue
        train = data.iloc[:train_end]
        test = data.iloc[test_start:test_end]
        for name in names:
            estimator = _make_estimator(name, seed + fold)
            y_hat = _fit_predict(estimator, train, test)
            calibration_actual, calibration_pred = _calibration_predictions(
                name, estimator, train, seed + fold
            )
            residuals = calibration_actual - calibration_pred
            lower_offset, upper_offset = np.quantile(residuals, [0.05, 0.95])
            fold_frame = pd.DataFrame(
                {
                    "timestamp_utc": test["timestamp_utc"].to_numpy(),
                    "actual_price_eur_mwh": test[TARGET].to_numpy(),
                    "predicted_price_eur_mwh": y_hat,
                    "lower_90_eur_mwh": y_hat + lower_offset,
                    "upper_90_eur_mwh": y_hat + upper_offset,
                    "model": name,
                    "fold": fold + 1,
                }
            )
            predictions.append(fold_frame)
    all_predictions = pd.concat(predictions, ignore_index=True)
    metrics = summarize_metrics(all_predictions)
    return metrics, all_predictions


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for model, group in predictions.groupby("model", sort=False):
        actual = group["actual_price_eur_mwh"].to_numpy()
        predicted = group["predicted_price_eur_mwh"].to_numpy()
        actual_negative = actual < 0
        predicted_negative = predicted < 0
        denominator = np.abs(actual) + np.abs(predicted)
        smape = np.mean(np.divide(2 * np.abs(actual - predicted), denominator + 1e-6)) * 100
        coverage = np.mean(
            (actual >= group["lower_90_eur_mwh"].to_numpy())
            & (actual <= group["upper_90_eur_mwh"].to_numpy())
        )
        rows.append(
            {
                "model": model,
                "mae_eur_mwh": mean_absolute_error(actual, predicted),
                "rmse_eur_mwh": math.sqrt(mean_squared_error(actual, predicted)),
                "smape_pct": smape,
                "negative_precision": precision_score(
                    actual_negative, predicted_negative, zero_division=0
                ),
                "negative_recall": recall_score(
                    actual_negative, predicted_negative, zero_division=0
                ),
                "negative_f1": f1_score(actual_negative, predicted_negative, zero_division=0),
                "interval_coverage_90": coverage,
                "observations": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_eur_mwh").reset_index(drop=True)


def train_best_model(
    feature_frame: pd.DataFrame, metrics: pd.DataFrame, settings: Settings
) -> FittedForecast:
    data = training_rows(feature_frame)
    winner = str(metrics.iloc[0]["model"])
    estimator = _make_estimator(winner, settings.random_seed)
    if estimator is not None:
        estimator.fit(data[FEATURE_COLUMNS], data[TARGET])
        fitted_values = estimator.predict(data[FEATURE_COLUMNS])
    else:
        fitted_values = data["price_lag_168"].to_numpy()
    residuals = data[TARGET].to_numpy() - fitted_values
    lower, upper = np.quantile(residuals, [0.05, 0.95])
    fitted = FittedForecast(
        model_name=winner,
        estimator=estimator,
        feature_columns=list(FEATURE_COLUMNS),
        residual_lower=float(lower),
        residual_upper=float(upper),
        residual_std=float(np.std(residuals)),
        trained_through_utc=str(data["timestamp_utc"].max()),
    )
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, settings.artifacts_dir / "price_model.joblib")
    return fitted


def predict_next_day(
    feature_frame: pd.DataFrame,
    fitted: FittedForecast,
    settings: Settings,
    horizon: int = 24,
) -> pd.DataFrame:
    future = forecast_rows(feature_frame, horizon)
    if fitted.estimator is None:
        point = future["price_lag_168"].to_numpy()
    else:
        point = fitted.estimator.predict(future[fitted.feature_columns])
    std = max(fitted.residual_std, 1e-6)
    negative_probability = np.array([_normal_cdf((0 - value) / std) for value in point])
    forecast = pd.DataFrame(
        {
            "timestamp_utc": future["timestamp_utc"].to_numpy(),
            "predicted_price_eur_mwh": point,
            "lower_90_eur_mwh": point + fitted.residual_lower,
            "upper_90_eur_mwh": point + fitted.residual_upper,
            "negative_price_probability": negative_probability,
            "negative_price_predicted": point < 0,
            "load_forecast_mwh": future["load_forecast_mwh"].to_numpy(),
            "renewable_forecast_mwh": future["renewable_forecast_mwh"].to_numpy(),
            "renewable_forecast_share": future["renewable_forecast_share"].to_numpy(),
            "forecast_carbon_intensity_gco2_kwh": future[
                "forecast_carbon_intensity_gco2_kwh"
            ].to_numpy(),
            "model": fitted.model_name,
            "issued_at_utc": pd.Timestamp.now(tz="UTC"),
        }
    )
    atomic_write_parquet(forecast, settings.artifacts_dir / "forecast_next_day.parquet")
    return forecast


def run_modeling(
    feature_frame: pd.DataFrame, settings: Settings, n_splits: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, backtest_predictions = walk_forward_backtest(
        feature_frame, settings.random_seed, n_splits=n_splits
    )
    fitted = train_best_model(feature_frame, metrics, settings)
    forecast = predict_next_day(feature_frame, fitted, settings)
    metrics.to_csv(settings.artifacts_dir / "model_metrics.csv", index=False)
    atomic_write_parquet(
        backtest_predictions, settings.artifacts_dir / "backtest_predictions.parquet"
    )
    atomic_write_json(
        {
            "selected_model": fitted.model_name,
            "trained_through_utc": fitted.trained_through_utc,
            "feature_count": len(fitted.feature_columns),
            "forecast_start_utc": forecast["timestamp_utc"].min(),
            "forecast_end_utc": forecast["timestamp_utc"].max(),
        },
        settings.artifacts_dir / "model_run.json",
    )
    return metrics, backtest_predictions, forecast


def load_fitted_model(path: Path) -> FittedForecast:
    return joblib.load(path)


def _fit_predict(
    estimator: RegressorMixin | None, train: pd.DataFrame, test: pd.DataFrame
) -> np.ndarray:
    if estimator is None:
        return test["price_lag_168"].to_numpy()
    estimator.fit(train[FEATURE_COLUMNS], train[TARGET])
    return np.asarray(estimator.predict(test[FEATURE_COLUMNS]))


def _calibration_predictions(
    name: str,
    fitted_estimator: RegressorMixin | None,
    train: pd.DataFrame,
    seed: int,
    calibration_hours: int = 24 * 28,
) -> tuple[np.ndarray, np.ndarray]:
    size = min(calibration_hours, max(24, len(train) // 5))
    calibration = train.iloc[-size:]
    proper_train = train.iloc[:-size]
    if name == "seasonal_naive":
        return calibration[TARGET].to_numpy(), calibration["price_lag_168"].to_numpy()
    estimator = _make_estimator(name, seed)
    if estimator is None:
        raise AssertionError("Non-seasonal estimator unexpectedly missing")
    estimator.fit(proper_train[FEATURE_COLUMNS], proper_train[TARGET])
    return calibration[TARGET].to_numpy(), np.asarray(
        estimator.predict(calibration[FEATURE_COLUMNS])
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))

