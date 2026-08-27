"""Flexible-load scheduling and business-impact simulation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from gridshift.config import Settings
from gridshift.storage import atomic_write_json, atomic_write_parquet

Objective = Literal["cost", "emissions", "balanced"]


def simulate_flexible_load(
    forecast: pd.DataFrame,
    flexible_share: float = 0.15,
    objective: Objective = "balanced",
    baseline_mwh: float | np.ndarray = 10.0,
    max_load_multiplier: float = 1.6,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Reallocate flexible energy to the best hours while conserving total MWh.

    The optimization is a deterministic merit-order allocation. It retains an
    inflexible base in every hour and caps optimized hourly demand relative to
    the original baseline, making its assumptions auditable for business users.
    """
    if not 0 <= flexible_share <= 0.5:
        raise ValueError("flexible_share must be between 0 and 0.5")
    if max_load_multiplier < 1:
        raise ValueError("max_load_multiplier must be at least 1")
    if objective not in {"cost", "emissions", "balanced"}:
        raise ValueError(f"Unsupported objective: {objective}")

    result = forecast.copy().reset_index(drop=True)
    n = len(result)
    baseline = np.broadcast_to(np.asarray(baseline_mwh, dtype=float), (n,)).copy()
    if np.any(baseline < 0):
        raise ValueError("baseline_mwh cannot be negative")
    price = result["predicted_price_eur_mwh"].to_numpy(dtype=float)
    carbon = result["forecast_carbon_intensity_gco2_kwh"].to_numpy(dtype=float)
    merit = _merit_score(price, carbon, objective)

    inflexible = baseline * (1 - flexible_share)
    available_capacity = np.maximum(0, baseline * max_load_multiplier - inflexible)
    flexible_pool = float(np.sum(baseline * flexible_share))
    allocation = np.zeros(n)
    for index in np.argsort(merit, kind="stable"):
        placed = min(available_capacity[index], flexible_pool)
        allocation[index] = placed
        flexible_pool -= placed
        if flexible_pool <= 1e-9:
            break
    if flexible_pool > 1e-6:
        raise ValueError("Hourly cap is too restrictive to place all flexible consumption")

    optimized = inflexible + allocation
    result["baseline_consumption_mwh"] = baseline
    result["optimized_consumption_mwh"] = optimized
    result["load_shift_mwh"] = optimized - baseline
    result["baseline_cost_eur"] = baseline * price
    result["optimized_cost_eur"] = optimized * price
    result["baseline_emissions_kg"] = baseline * carbon
    result["optimized_emissions_kg"] = optimized * carbon
    result["dispatch_signal"] = np.select(
        [result["load_shift_mwh"] > 1e-6, result["load_shift_mwh"] < -1e-6],
        ["CONSUME", "REDUCE"],
        default="HOLD",
    )
    result["objective_score"] = merit

    baseline_cost = float(result["baseline_cost_eur"].sum())
    optimized_cost = float(result["optimized_cost_eur"].sum())
    baseline_emissions = float(result["baseline_emissions_kg"].sum())
    optimized_emissions = float(result["optimized_emissions_kg"].sum())
    summary: dict[str, float | str] = {
        "objective": objective,
        "flexible_share": flexible_share,
        "energy_mwh": float(baseline.sum()),
        "baseline_cost_eur": baseline_cost,
        "optimized_cost_eur": optimized_cost,
        "cost_savings_eur": baseline_cost - optimized_cost,
        "cost_savings_pct": _safe_percent(baseline_cost - optimized_cost, baseline_cost),
        "baseline_emissions_kg": baseline_emissions,
        "optimized_emissions_kg": optimized_emissions,
        "emissions_savings_kg": baseline_emissions - optimized_emissions,
        "emissions_savings_pct": _safe_percent(
            baseline_emissions - optimized_emissions, baseline_emissions
        ),
    }
    return result, summary


def run_default_simulations(
    forecast: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedules: list[pd.DataFrame] = []
    summaries: list[dict[str, float | str]] = []
    for objective in ("cost", "emissions", "balanced"):
        for flexible_share in (0.10, 0.15, 0.20):
            schedule, summary = simulate_flexible_load(
                forecast,
                flexible_share=flexible_share,
                objective=objective,
            )
            schedule["objective"] = objective
            schedule["flexible_share"] = flexible_share
            schedules.append(schedule)
            summaries.append(summary)
    all_schedules = pd.concat(schedules, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    atomic_write_parquet(all_schedules, settings.artifacts_dir / "flex_schedules.parquet")
    summary_frame.to_csv(settings.artifacts_dir / "flex_summary.csv", index=False)
    default_summary = next(
        item
        for item in summaries
        if item["objective"] == "balanced" and item["flexible_share"] == 0.15
    )
    atomic_write_json(default_summary, settings.artifacts_dir / "flex_default_summary.json")
    return all_schedules, summary_frame


def _merit_score(price: np.ndarray, carbon: np.ndarray, objective: Objective) -> np.ndarray:
    if objective == "cost":
        return price
    if objective == "emissions":
        return carbon
    return 0.55 * _robust_scale(price) + 0.45 * _robust_scale(carbon)


def _robust_scale(values: np.ndarray) -> np.ndarray:
    low, high = np.quantile(values, [0.05, 0.95])
    width = high - low
    if width <= 1e-9:
        return np.zeros_like(values)
    return np.clip((values - low) / width, 0, 1)


def _safe_percent(numerator: float, denominator: float) -> float:
    return 100 * numerator / denominator if abs(denominator) > 1e-9 else 0.0

