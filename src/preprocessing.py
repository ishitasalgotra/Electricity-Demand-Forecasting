from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


TARGET_COLUMN = "Demand (MW)"
RENEWABLE_COLUMNS = ["Wind (MW)", "Solar (MW)"]
LEAKAGE_COLUMN = "Total Generation (MW)"
TIMESTAMP_COLUMN = "Timestamp"


def load_electricity_data(csv_path: str | Path) -> pd.DataFrame:
    """Load, parse, sort, and validate the electricity demand dataset."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    expected_columns = [
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        "Wind (MW)",
        "Solar (MW)",
        LEAKAGE_COLUMN,
    ]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")
    if df[TIMESTAMP_COLUMN].isna().any():
        bad_count = int(df[TIMESTAMP_COLUMN].isna().sum())
        raise ValueError(f"Found {bad_count} rows with unparseable timestamps.")

    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return df


def inspect_data_quality(df: pd.DataFrame) -> dict:
    """Return useful quality checks without changing the data."""
    intervals = df[TIMESTAMP_COLUMN].diff().dropna()
    hourly = intervals == pd.Timedelta(hours=1)
    return {
        "shape": df.shape,
        "start": df[TIMESTAMP_COLUMN].min(),
        "end": df[TIMESTAMP_COLUMN].max(),
        "duplicate_timestamps": int(df[TIMESTAMP_COLUMN].duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),
        "is_chronological": bool(df[TIMESTAMP_COLUMN].is_monotonic_increasing),
        "non_hourly_intervals": int((~hourly).sum()),
    }


def validate_time_series(df: pd.DataFrame) -> None:
    """Stop early if time-series assumptions are violated."""
    quality = inspect_data_quality(df)
    if quality["duplicate_timestamps"] > 0:
        raise ValueError("Duplicate timestamps found. Resolve them before training.")
    if sum(quality["missing_values"].values()) > 0:
        raise ValueError("Missing values found. Resolve them before training.")
    if not quality["is_chronological"]:
        raise ValueError("Timestamps are not chronological.")
    if quality["non_hourly_intervals"] > 0:
        raise ValueError("Timestamps are not consistently hourly.")


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by time order into train, validation, and test sets."""
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("Train + validation ratios must leave room for test data.")

    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def fit_feature_scaler(train_df: pd.DataFrame, feature_columns: list[str]) -> MinMaxScaler:
    """Fit a scaler on training features only to avoid data leakage."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_columns])
    return scaler


def transform_features(df: pd.DataFrame, feature_columns: list[str], scaler: MinMaxScaler) -> np.ndarray:
    """Scale feature columns using an already-fitted scaler."""
    return scaler.transform(df[feature_columns]).astype(np.float32)


def create_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    lookback: int,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows that predict the next hour's demand.

    The input at index i uses rows [i-lookback, i) and predicts row i.
    That keeps future target information out of the feature window.
    """
    if lookback < 1:
        raise ValueError("Lookback must be at least 1.")
    if len(df) <= lookback:
        raise ValueError("Not enough rows to create sequences for this lookback.")

    scaled_features = transform_features(df, feature_columns, feature_scaler)
    target_values = df[[TARGET_COLUMN]].to_numpy(dtype=np.float32)
    scaled_target = target_scaler.transform(target_values).astype(np.float32)

    x_values = []
    y_values = []
    timestamps = []
    actual_values = []

    for end_idx in range(lookback, len(df)):
        start_idx = end_idx - lookback
        x_values.append(scaled_features[start_idx:end_idx])
        y_values.append(scaled_target[end_idx, 0])
        timestamps.append(df[TIMESTAMP_COLUMN].iloc[end_idx])
        actual_values.append(target_values[end_idx, 0])

    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(y_values, dtype=np.float32),
        np.asarray(timestamps),
        np.asarray(actual_values, dtype=np.float32),
    )


def inverse_scale_demand(values: np.ndarray, target_scaler: MinMaxScaler) -> np.ndarray:
    """Convert scaled demand predictions back to MW."""
    values_2d = np.asarray(values).reshape(-1, 1)
    return target_scaler.inverse_transform(values_2d).reshape(-1)
