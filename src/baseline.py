import pandas as pd

from src.evaluation import regression_metrics
from src.preprocessing import TARGET_COLUMN, TIMESTAMP_COLUMN


def previous_day_baseline(test_df: pd.DataFrame, lag_hours: int = 24) -> tuple[pd.DataFrame, dict[str, float]]:
    """Predict test demand with the value from 24 hours earlier.

    The first 24 rows of the test set are skipped because they do not have
    a previous-day value inside the same test period.
    """
    if len(test_df) <= lag_hours:
        raise ValueError("Test set is too small for a previous-day baseline.")

    predictions = test_df[TARGET_COLUMN].shift(lag_hours)
    result = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: test_df[TIMESTAMP_COLUMN],
            "Actual": test_df[TARGET_COLUMN],
            "Predicted": predictions,
        }
    ).dropna().reset_index(drop=True)

    metrics = regression_metrics(result["Actual"].to_numpy(), result["Predicted"].to_numpy())
    return result, metrics
