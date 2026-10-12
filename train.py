from pathlib import Path
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.baseline import previous_day_baseline
from src.eda import run_eda
from src.evaluation import regression_metrics, save_metrics
from src.lstm_model import build_lstm_model, save_model, set_random_seeds, train_lstm_model
from src.preprocessing import (
    RENEWABLE_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    chronological_split,
    create_sequences,
    fit_feature_scaler,
    inverse_scale_demand,
    load_electricity_data,
    validate_time_series,
)


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def plot_actual_vs_predicted(best_predictions: pd.DataFrame, results_dir: Path) -> None:
    sample = best_predictions.head(24 * 14)
    plt.figure(figsize=(12, 4))
    plt.plot(sample[TIMESTAMP_COLUMN], sample["Actual"], label="Actual", linewidth=1.2)
    plt.plot(sample[TIMESTAMP_COLUMN], sample["Predicted"], label="Best LSTM", linewidth=1.2)
    plt.title("Actual vs Predicted Demand: Best LSTM")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "actual_vs_predicted_best_lstm.png", dpi=150)
    plt.close()


def plot_baseline_vs_best(baseline_predictions: pd.DataFrame, best_predictions: pd.DataFrame, results_dir: Path) -> None:
    merged = baseline_predictions.merge(
        best_predictions,
        on=TIMESTAMP_COLUMN,
        suffixes=("_baseline", "_lstm"),
    )
    sample = merged.head(24 * 14)
    plt.figure(figsize=(12, 4))
    plt.plot(sample[TIMESTAMP_COLUMN], sample["Actual_baseline"], label="Actual", linewidth=1.2)
    plt.plot(sample[TIMESTAMP_COLUMN], sample["Predicted_baseline"], label="Previous-day baseline", linewidth=1.0)
    plt.plot(sample[TIMESTAMP_COLUMN], sample["Predicted_lstm"], label="Best LSTM", linewidth=1.0)
    plt.title("Baseline vs Best LSTM Forecast")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "baseline_vs_best_lstm.png", dpi=150)
    plt.close()


def plot_lookback_comparison(metrics_df: pd.DataFrame, results_dir: Path) -> None:
    lstm_df = metrics_df[metrics_df["Lookback"] != "24h lag"].copy()
    lstm_df["LookbackHours"] = lstm_df["Lookback"].str.replace("h", "", regex=False).astype(int)

    plt.figure(figsize=(8, 4))
    for model_name, group in lstm_df.groupby("Model"):
        group = group.sort_values("LookbackHours")
        plt.plot(group["LookbackHours"], group["RMSE"], marker="o", label=model_name)
    plt.title("RMSE Across Lookback Windows")
    plt.xlabel("Lookback window (hours)")
    plt.ylabel("RMSE")
    plt.xticks([12, 24, 48])
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "lookback_rmse_comparison.png", dpi=150)
    plt.close()


def train_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    lookback: int,
    model_label: str,
    args: argparse.Namespace,
    models_dir: Path,
) -> tuple[dict, pd.DataFrame]:
    print(f"\nTraining {model_label} with {lookback}h lookback")
    print(f"Features: {feature_columns}")

    feature_scaler = fit_feature_scaler(train_df, feature_columns)
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[[TARGET_COLUMN]].to_numpy())

    x_train, y_train, _, _ = create_sequences(train_df, feature_columns, lookback, feature_scaler, target_scaler)
    x_val, y_val, _, _ = create_sequences(val_df, feature_columns, lookback, feature_scaler, target_scaler)
    x_test, _, test_timestamps, y_test_actual = create_sequences(
        test_df,
        feature_columns,
        lookback,
        feature_scaler,
        target_scaler,
    )

    model = build_lstm_model(
        input_shape=(x_train.shape[1], x_train.shape[2]),
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
    )
    train_lstm_model(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    scaled_predictions = model.predict(x_test, batch_size=args.batch_size, verbose=0).reshape(-1)
    predictions = inverse_scale_demand(scaled_predictions, target_scaler)
    metrics = regression_metrics(y_test_actual, predictions)

    safe_model_name = model_label.lower().replace("+", "plus").replace(" ", "_")
    model_path = models_dir / f"{safe_model_name}_{lookback}h.keras"
    save_model(model, model_path)

    prediction_df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: test_timestamps,
            "Actual": y_test_actual,
            "Predicted": predictions,
            "Model": model_label,
            "Lookback": f"{lookback}h",
        }
    )
    metric_row = {
        "Model": model_label,
        "Lookback": f"{lookback}h",
        **metrics,
    }
    print(
        f"{model_label} {lookback}h -> "
        f"MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%"
    )
    return metric_row, prediction_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train electricity demand forecasting baselines and LSTMs.")
    parser.add_argument("--data-path", default="data/india_electricity_clean.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lstm-units", type=int, default=32)
    parser.add_argument("--dense-units", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-eda", action="store_true")
    args = parser.parse_args()

    set_random_seeds(args.seed)

    results_dir = Path("results")
    models_dir = Path("models")
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    if not args.skip_eda:
        print("Running EDA and saving plots...")
        run_eda(args.data_path, results_dir)

    df = load_electricity_data(args.data_path)
    validate_time_series(df)
    train_df, val_df, test_df = chronological_split(df)

    print("\nChronological split:")
    print(f"Train: {len(train_df)} rows ({train_df[TIMESTAMP_COLUMN].min()} to {train_df[TIMESTAMP_COLUMN].max()})")
    print(f"Validation: {len(val_df)} rows ({val_df[TIMESTAMP_COLUMN].min()} to {val_df[TIMESTAMP_COLUMN].max()})")
    print(f"Test: {len(test_df)} rows ({test_df[TIMESTAMP_COLUMN].min()} to {test_df[TIMESTAMP_COLUMN].max()})")

    metrics_rows = []
    prediction_outputs = []

    baseline_predictions, baseline_metrics = previous_day_baseline(test_df)
    metrics_rows.append({"Model": "Previous-day baseline", "Lookback": "24h lag", **baseline_metrics})
    baseline_predictions.to_csv(results_dir / "baseline_predictions.csv", index=False)
    print(
        "\nPrevious-day baseline -> "
        f"MAE: {baseline_metrics['MAE']:.2f}, "
        f"RMSE: {baseline_metrics['RMSE']:.2f}, "
        f"MAPE: {baseline_metrics['MAPE']:.2f}%"
    )

    experiments = [
        ("Demand-only LSTM", [TARGET_COLUMN]),
        ("Demand+Wind+Solar LSTM", [TARGET_COLUMN, *RENEWABLE_COLUMNS]),
    ]
    for model_label, feature_columns in experiments:
        for lookback in [12, 24, 48]:
            metric_row, prediction_df = train_experiment(
                train_df,
                val_df,
                test_df,
                feature_columns,
                lookback,
                model_label,
                args,
                models_dir,
            )
            metrics_rows.append(metric_row)
            prediction_outputs.append(prediction_df)

    metrics_df = save_metrics(metrics_rows, results_dir / "metrics.csv")
    all_predictions = pd.concat(prediction_outputs, ignore_index=True)
    all_predictions.to_csv(results_dir / "lstm_predictions.csv", index=False)

    lstm_metrics = metrics_df[metrics_df["Model"] != "Previous-day baseline"]
    best_idx = lstm_metrics["RMSE"].idxmin()
    best_row = metrics_df.loc[best_idx]
    best_predictions = all_predictions[
        (all_predictions["Model"] == best_row["Model"]) & (all_predictions["Lookback"] == best_row["Lookback"])
    ].copy()
    best_predictions.to_csv(results_dir / "best_lstm_predictions.csv", index=False)

    plot_actual_vs_predicted(best_predictions, results_dir)
    plot_baseline_vs_best(baseline_predictions, best_predictions, results_dir)
    plot_lookback_comparison(metrics_df, results_dir)

    print("\nFinal comparison table:")
    print(metrics_df.to_string(index=False, formatters={col: "{:.2f}".format for col in ["MAE", "RMSE", "MAPE"]}))
    print(f"\nBest LSTM by RMSE: {best_row['Model']} with {best_row['Lookback']} lookback")
    print("Saved metrics, predictions, plots, and models.")


if __name__ == "__main__":
    main()
