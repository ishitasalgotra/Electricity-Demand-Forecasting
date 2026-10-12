from pathlib import Path

import numpy as np
import pandas as pd


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and MAPE for demand forecasts."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same length.")

    errors = actual - predicted
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    nonzero_mask = actual != 0
    if not np.any(nonzero_mask):
        mape = np.nan
    else:
        mape = np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100

    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def save_metrics(metrics: list[dict], output_path: str | Path) -> pd.DataFrame:
    """Save metrics in the requested comparison-table format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df
