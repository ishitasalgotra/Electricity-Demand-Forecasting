from pathlib import Path
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing import (
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    chronological_split,
    inspect_data_quality,
    load_electricity_data,
    validate_time_series,
)


DATA_PATH = Path("data/india_electricity_clean.csv")
BEST_MODEL_PATH = Path("models/demand-only_lstm_24h.keras")
RESULTS_DIR = Path("results")
LOOKBACK = 24


st.set_page_config(
    page_title="Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 760;
        margin-bottom: 0.1rem;
        color: #102033;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #5b677a;
        margin-bottom: 1.4rem;
    }
    .metric-card {
        border: 1px solid #e4e8ef;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        box-shadow: 0 1px 4px rgba(16, 32, 51, 0.05);
        min-height: 96px;
    }
    .metric-label {
        color: #667085;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.02rem;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: #102033;
        font-size: 1.35rem;
        font-weight: 720;
        line-height: 1.2;
    }
    .forecast-box {
        border: 1px solid #d9e4f2;
        border-left: 5px solid #2f80ed;
        border-radius: 8px;
        padding: 1.2rem;
        background: #f8fbff;
        margin: 0.5rem 0 1rem;
    }
    .small-note {
        color: #667085;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = load_electricity_data(DATA_PATH)
    validate_time_series(df)
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    metrics_path = RESULTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing required metrics file: {metrics_path}")
    return pd.read_csv(metrics_path)


@st.cache_data(show_spinner=False)
def load_best_predictions() -> pd.DataFrame:
    predictions_path = RESULTS_DIR / "best_lstm_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing required predictions file: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    predictions[TIMESTAMP_COLUMN] = pd.to_datetime(predictions[TIMESTAMP_COLUMN])
    return predictions.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_best_model() -> tf.keras.Model:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model file: {BEST_MODEL_PATH}")
    return tf.keras.models.load_model(BEST_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def build_training_scalers(df: pd.DataFrame) -> tuple[MinMaxScaler, MinMaxScaler]:
    train_df, _, _ = chronological_split(df)

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[[TARGET_COLUMN]])

    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[[TARGET_COLUMN]].to_numpy())

    return feature_scaler, target_scaler


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_mw(value: float) -> str:
    return f"{value:,.0f} MW"


def get_valid_timestamps(df: pd.DataFrame) -> pd.Series:
    return df[TIMESTAMP_COLUMN].iloc[LOOKBACK:].reset_index(drop=True)


def predict_for_timestamp(
    df: pd.DataFrame,
    selected_timestamp: pd.Timestamp,
    model: tf.keras.Model,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
) -> tuple[float, float, pd.DataFrame]:
    matching_indices = df.index[df[TIMESTAMP_COLUMN] == selected_timestamp].tolist()
    if not matching_indices:
        raise ValueError("Selected timestamp was not found in the dataset.")

    target_idx = matching_indices[0]
    if target_idx < LOOKBACK:
        raise ValueError("This timestamp does not have 24 previous hourly records.")

    previous_window = df.iloc[target_idx - LOOKBACK : target_idx].copy()
    if len(previous_window) != LOOKBACK:
        raise ValueError("Could not build a complete 24-hour input window.")

    scaled_window = feature_scaler.transform(previous_window[[TARGET_COLUMN]]).astype(np.float32)
    model_input = scaled_window.reshape(1, LOOKBACK, 1)
    scaled_prediction = model.predict(model_input, verbose=0).reshape(-1)
    prediction = target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).reshape(-1)[0]
    actual = float(df.loc[target_idx, TARGET_COLUMN])
    return float(prediction), actual, previous_window


def plot_previous_window(previous_window: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(previous_window[TIMESTAMP_COLUMN], previous_window[TARGET_COLUMN], marker="o", linewidth=1.8)
    ax.set_title("Previous 24 Hours of Electricity Demand")
    ax.set_xlabel("Time")
    ax.set_ylabel("Demand (MW)")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)


def plot_actual_vs_predicted(predictions: pd.DataFrame) -> None:
    recent = predictions.tail(48)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(recent[TIMESTAMP_COLUMN], recent["Actual"], label="Actual", linewidth=1.8)
    ax.plot(recent[TIMESTAMP_COLUMN], recent["Predicted"], label="Predicted", linewidth=1.8)
    ax.set_title("Actual vs Predicted Demand")
    ax.set_xlabel("Time")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)


def plot_mape_comparison(metrics: pd.DataFrame) -> None:
    labels = metrics["Model"] + " (" + metrics["Lookback"] + ")"
    colors = [
        "#2f80ed" if row["Model"] == "Demand-only LSTM" and row["Lookback"] == "24h" else "#9aa7b5"
        for _, row in metrics.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, metrics["MAPE"], color=colors)
    ax.set_title("MAPE Comparison Across Models")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    st.pyplot(fig, clear_figure=True)


def show_existing_image(image_name: str, caption: str) -> None:
    image_path = RESULTS_DIR / image_name
    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing plot: {image_path}")


try:
    df = load_dataset()
    metrics_df = load_metrics()
    best_predictions_df = load_best_predictions()
except Exception as exc:
    st.error(f"Could not load required project files: {exc}")
    st.stop()


st.markdown('<div class="hero-title">⚡ Short-Term Electricity Demand Forecasting</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">LSTM-Based Forecasting with Renewable Energy Features</div>',
    unsafe_allow_html=True,
)

top_cols = st.columns(4)
with top_cols[0]:
    metric_card("Dataset", "India Electricity Data")
with top_cols[1]:
    metric_card("Best Model", "LSTM — 24 Hour Lookback")
with top_cols[2]:
    metric_card("MAPE", "1.74%")
with top_cols[3]:
    metric_card("RMSE", "4830 MW")


valid_timestamps = get_valid_timestamps(df)
valid_dates = sorted(valid_timestamps.dt.date.unique())

st.sidebar.header("Forecast Settings")
selected_date = st.sidebar.date_input(
    "Date",
    value=valid_dates[-1],
    min_value=valid_dates[0],
    max_value=valid_dates[-1],
)

date_timestamps = valid_timestamps[valid_timestamps.dt.date == selected_date]
if date_timestamps.empty:
    st.sidebar.error("No predictable timestamps are available for this date.")
    st.stop()

time_options = [ts.time() for ts in date_timestamps]
selected_time = st.sidebar.selectbox("Hour / Time", options=time_options, index=len(time_options) - 1)
selected_timestamp = pd.Timestamp.combine(selected_date, selected_time)
predict_clicked = st.sidebar.button("Predict", type="primary", use_container_width=True)

st.sidebar.caption("Timestamps with fewer than 24 previous observations are excluded.")


st.subheader("Next-Hour Electricity Demand Forecast")
if predict_clicked:
    try:
        model = load_best_model()
        feature_scaler, target_scaler = build_training_scalers(df)
        predicted_demand, actual_demand, previous_window = predict_for_timestamp(
            df,
            selected_timestamp,
            model,
            feature_scaler,
            target_scaler,
        )
        absolute_error = abs(actual_demand - predicted_demand)
        percentage_error = (absolute_error / actual_demand) * 100 if actual_demand != 0 else np.nan

        st.markdown('<div class="forecast-box">', unsafe_allow_html=True)
        result_cols = st.columns(4)
        result_cols[0].metric("Predicted Demand", format_mw(predicted_demand))
        result_cols[1].metric("Actual Demand", format_mw(actual_demand))
        result_cols[2].metric("Absolute Error", format_mw(absolute_error))
        result_cols[3].metric("Percentage Error", f"{percentage_error:.2f}%")
        st.markdown(
            '<div class="small-note">Model: Demand-only LSTM &nbsp;&nbsp;|&nbsp;&nbsp; Lookback: 24 hours</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        plot_previous_window(previous_window)
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
else:
    st.info("Choose a date and hour from the sidebar, then click Predict.")


st.divider()
st.subheader("Actual vs Predicted Demand")
plot_actual_vs_predicted(best_predictions_df)


st.divider()
st.subheader("Model Performance Comparison")
display_metrics = metrics_df.copy()
display_metrics[["MAE", "RMSE", "MAPE"]] = display_metrics[["MAE", "RMSE", "MAPE"]].round(2)
st.dataframe(
    display_metrics,
    use_container_width=True,
    hide_index=True,
)
st.success("Best model: Demand-only LSTM — 24h")
plot_mape_comparison(metrics_df)


st.divider()
st.subheader("Renewable Energy Feature Experiment")
renewable_cols = st.columns(2)
demand_24 = metrics_df[(metrics_df["Model"] == "Demand-only LSTM") & (metrics_df["Lookback"] == "24h")]
renewable_24 = metrics_df[(metrics_df["Model"] == "Demand+Wind+Solar LSTM") & (metrics_df["Lookback"] == "24h")]
with renewable_cols[0]:
    value = demand_24["MAPE"].iloc[0] if not demand_24.empty else np.nan
    st.metric("Demand-only LSTM, 24h", f"{value:.2f}%")
with renewable_cols[1]:
    value = renewable_24["MAPE"].iloc[0] if not renewable_24.empty else np.nan
    st.metric("Demand + Wind + Solar, 24h", f"{value:.2f}%")
st.write(
    "We tested whether adding wind and solar generation data improves short-term demand forecasting. "
    "In this dataset and model configuration, the demand-only LSTM performed better than the models "
    "using wind and solar. This is an experimental result, not a claim that renewable features are "
    "always harmful."
)


st.divider()
with st.expander("Dataset & Exploratory Analysis", expanded=False):
    quality = inspect_data_quality(df)
    info_cols = st.columns(3)
    info_cols[0].metric("Number of records", f"{len(df):,}")
    info_cols[1].metric("Start date", str(quality["start"]))
    info_cols[2].metric("End date", str(quality["end"]))

    info_cols = st.columns(3)
    info_cols[0].metric("Number of features", str(len(df.columns) - 1))
    info_cols[1].metric("Missing values", str(sum(quality["missing_values"].values())))
    info_cols[2].metric("Duplicate timestamps", str(quality["duplicate_timestamps"]))

    show_existing_image("eda_demand_over_time.png", "Electricity demand over time")
    show_existing_image("eda_demand_two_weeks.png", "Representative two-week demand pattern")
    show_existing_image("eda_demand_distribution.png", "Demand distribution")
    show_existing_image("eda_correlations.png", "Demand, wind, and solar correlations")
    show_existing_image("eda_demand_wind_solar_example.png", "Demand, wind, and solar example")


st.divider()
st.subheader("Key Findings")
st.markdown(
    """
    1. The 24-hour demand-only LSTM achieved the best performance.
    2. The 24-hour lookback performed better than 12-hour and 48-hour lookbacks.
    3. The LSTM substantially outperformed the previous-day baseline.
    4. Adding wind and solar did not improve performance for this dataset/configuration.
    5. Historical electricity demand showed strong short-term predictive value.
    """
)
