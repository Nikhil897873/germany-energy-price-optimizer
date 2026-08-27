"""Streamlit decision dashboard for GridShift DE."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gridshift.simulation import simulate_flexible_load

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data" / "artifacts"
SILVER = ROOT / "data" / "silver"

st.set_page_config(
    page_title="GridShift DE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {
        background:linear-gradient(145deg, #ffffff 0%, #f1f7fb 100%);
        border:1px solid #d5e2eb;
        box-shadow:0 4px 14px rgba(15, 23, 42, .07);
        padding:16px 18px;
        border-radius:12px;
        min-height:112px;
      }
      [data-testid="stMetricLabel"],
      [data-testid="stMetricLabel"] p {
        color:#526275 !important;
        font-weight:600;
      }
      [data-testid="stMetricLabel"] p {
        white-space:normal !important;
        overflow:visible !important;
        text-overflow:clip !important;
        line-height:1.2;
      }
      [data-testid="stMetricValue"],
      [data-testid="stMetricValue"] div {
        color:#0f172a !important;
      }
      [data-testid="stMetricValue"] {
        width:100%;
        overflow:visible !important;
      }
      [data-testid="stMetricValue"] div {
        font-size:clamp(1.4rem, 1.8vw, 2rem) !important;
        line-height:1.15;
        white-space:nowrap !important;
        overflow:visible !important;
        text-overflow:clip !important;
      }
      .signal-consume {color:#55d6a7; font-weight:700;}
      .signal-reduce {color:#ff7b72; font-weight:700;}
      .small-note {color:#8c9bab; font-size:.84rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    forecast = pd.read_parquet(ARTIFACTS / "forecast_next_day.parquet")
    metrics = pd.read_csv(ARTIFACTS / "model_metrics.csv")
    energy = pd.read_parquet(SILVER / "energy_hourly.parquet")
    metadata_path = ROOT / "data" / "bronze" / "demo_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    for frame in (forecast, energy):
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return forecast, metrics, energy, metadata


required = [
    ARTIFACTS / "forecast_next_day.parquet",
    ARTIFACTS / "model_metrics.csv",
    SILVER / "energy_hourly.parquet",
]
if not all(path.exists() for path in required):
    st.title("⚡ GridShift DE")
    st.info("Generate the decision artifacts first: `gridshift run --demo --days 730`")
    st.stop()

forecast, metrics, energy, metadata = load_data()
forecast["timestamp_de"] = forecast["timestamp_utc"].dt.tz_convert("Europe/Berlin")

with st.sidebar:
    st.subheader("Operating scenario")
    flexible_share = st.slider("Flexible consumption", 10, 20, 15, 1) / 100
    objective_label = st.radio(
        "Optimization objective",
        ["Balanced", "Lowest cost", "Lowest emissions"],
        horizontal=False,
    )
    objective = {
        "Balanced": "balanced",
        "Lowest cost": "cost",
        "Lowest emissions": "emissions",
    }[objective_label]
    hourly_baseline = st.number_input(
        "Baseline consumption (MWh/hour)", min_value=0.1, value=10.0, step=1.0
    )
    max_multiplier = st.slider("Maximum hourly load", 1.1, 2.5, 1.6, 0.1)
    st.divider()
    st.caption("All decisions are shown in Europe/Berlin time. Modeling remains in UTC.")

schedule, summary = simulate_flexible_load(
    forecast,
    flexible_share=flexible_share,
    objective=objective,
    baseline_mwh=hourly_baseline,
    max_load_multiplier=max_multiplier,
)
schedule["timestamp_de"] = pd.to_datetime(schedule["timestamp_utc"], utc=True).dt.tz_convert(
    "Europe/Berlin"
)

title_col, badge_col = st.columns([4, 1])
with title_col:
    st.title("GridShift DE")
    st.caption("Next-day electricity price intelligence for flexible industrial demand")
with badge_col:
    if metadata.get("data_mode") == "synthetic_demo":
        st.warning("DEMO DATA", icon="🧪")
    else:
        st.success("LIVE SOURCES", icon="●")

kpi_columns = st.columns(5)
kpi_columns[0].metric("Average forecast", f"€{forecast['predicted_price_eur_mwh'].mean():,.1f}/MWh")
kpi_columns[1].metric("Negative-price hours", int(forecast["negative_price_predicted"].sum()))
kpi_columns[2].metric(
    "Cost saved",
    f"€{summary['cost_savings_eur']:,.0f}",
    f"{summary['cost_savings_pct']:.1f}%",
)
kpi_columns[3].metric(
    "Emissions avoided",
    f"{summary['emissions_savings_kg']:,.0f} kg",
    f"{summary['emissions_savings_pct']:.1f}%",
)
kpi_columns[4].metric(
    "Renewable share",
    f"{forecast['renewable_forecast_share'].mean():.0%}",
)

tab_forecast, tab_dispatch, tab_models, tab_data = st.tabs(
    ["Price forecast", "Load schedule", "Model performance", "Data quality"]
)

with tab_forecast:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=forecast["timestamp_de"],
            y=forecast["upper_90_eur_mwh"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["timestamp_de"],
            y=forecast["lower_90_eur_mwh"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(61, 184, 255, .16)",
            name="90% interval",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["timestamp_de"],
            y=forecast["predicted_price_eur_mwh"],
            mode="lines+markers",
            line={"color": "#50b7f5", "width": 3},
            marker={"size": 6},
            name="Price forecast",
            customdata=forecast[["negative_price_probability"]],
            hovertemplate="%{x|%a %H:%M}<br>€%{y:.1f}/MWh<br>P(negative): %{customdata[0]:.0%}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_dash="dot", line_color="#ff7b72")
    figure.update_layout(
        height=440,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        yaxis_title="EUR/MWh",
        xaxis_title=None,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
    )
    st.plotly_chart(figure, width="stretch")

    renewable = go.Figure()
    renewable.add_trace(
        go.Bar(
            x=forecast["timestamp_de"],
            y=forecast["renewable_forecast_share"] * 100,
            marker_color="#55d6a7",
            name="Renewable share",
        )
    )
    renewable.update_layout(
        height=260,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis_title="% of forecast load",
        xaxis_title=None,
        showlegend=False,
    )
    st.plotly_chart(renewable, width="stretch")

with tab_dispatch:
    dispatch_figure = go.Figure()
    dispatch_figure.add_trace(
        go.Bar(
            x=schedule["timestamp_de"],
            y=schedule["baseline_consumption_mwh"],
            name="Baseline",
            marker_color="#5d6b7b",
            opacity=0.55,
        )
    )
    dispatch_figure.add_trace(
        go.Scatter(
            x=schedule["timestamp_de"],
            y=schedule["optimized_consumption_mwh"],
            name="Optimized",
            mode="lines+markers",
            line={"color": "#55d6a7", "width": 3},
        )
    )
    dispatch_figure.update_layout(
        height=390,
        barmode="overlay",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        yaxis_title="Consumption (MWh)",
        xaxis_title=None,
        legend={"orientation": "h", "y": 1.08},
    )
    st.plotly_chart(dispatch_figure, width="stretch")
    display = schedule[
        [
            "timestamp_de",
            "dispatch_signal",
            "predicted_price_eur_mwh",
            "renewable_forecast_share",
            "baseline_consumption_mwh",
            "optimized_consumption_mwh",
        ]
    ].copy()
    display.columns = ["Hour", "Action", "Price", "Renewables", "Baseline MWh", "Optimized MWh"]
    st.dataframe(
        display.style.format(
            {
                "Hour": lambda value: value.strftime("%a %H:%M"),
                "Price": "€{:.1f}",
                "Renewables": "{:.0%}",
                "Baseline MWh": "{:.1f}",
                "Optimized MWh": "{:.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

with tab_models:
    best = metrics.iloc[0]["model"]
    st.subheader(f"Selected model: {best}")
    model_chart = go.Figure(
        go.Bar(
            x=metrics["model"],
            y=metrics["mae_eur_mwh"],
            marker_color=["#55d6a7" if model == best else "#4c6178" for model in metrics["model"]],
            text=metrics["mae_eur_mwh"].map(lambda value: f"{value:.2f}"),
            textposition="outside",
        )
    )
    model_chart.update_layout(
        height=330,
        yaxis_title="Walk-forward MAE (EUR/MWh)",
        xaxis_title=None,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    st.plotly_chart(model_chart, width="stretch")
    st.dataframe(metrics.style.format(precision=3), width="stretch", hide_index=True)
    st.caption(
        "Every test fold occurs strictly after its training window, separated by a 24-hour embargo."
    )

with tab_data:
    observed = energy.loc[energy["price_eur_mwh"].notna()]
    quality_columns = st.columns(4)
    quality_columns[0].metric("Historical hours", f"{len(observed):,}")
    quality_columns[1].metric("Start", observed["timestamp_utc"].min().strftime("%Y-%m-%d"))
    quality_columns[2].metric(
        "Latest observation", observed["timestamp_utc"].max().strftime("%Y-%m-%d %H:%M")
    )
    quality_columns[3].metric(
        "Price completeness", f"{observed['price_eur_mwh'].notna().mean():.1%}"
    )
    st.markdown(
        "SMARD market and generation values are stored at source granularity in the bronze layer. "
        "DWD station values are averaged into a national weather proxy. Carbon intensity is an "
        "explicit generation-weighted proxy and does not account for imported electricity."
    )
