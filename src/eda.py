from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing import (
    RENEWABLE_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    inspect_data_quality,
    load_electricity_data,
)


def run_eda(csv_path: str | Path = "data/india_electricity_clean.csv", results_dir: str | Path = "results") -> None:
    """Create readable exploratory plots and a short text summary."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_electricity_data(csv_path)
    quality = inspect_data_quality(df)

    summary_path = results_dir / "eda_summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("Electricity Demand Dataset Summary\n")
        file.write("==================================\n")
        file.write(f"Shape: {quality['shape']}\n")
        file.write(f"Time range: {quality['start']} to {quality['end']}\n")
        file.write(f"Duplicate timestamps: {quality['duplicate_timestamps']}\n")
        file.write(f"Non-hourly intervals: {quality['non_hourly_intervals']}\n\n")
        file.write("Missing values:\n")
        for column, missing_count in quality["missing_values"].items():
            file.write(f"- {column}: {missing_count}\n")
        file.write("\nBasic statistics:\n")
        file.write(df[[TARGET_COLUMN, *RENEWABLE_COLUMNS]].describe().to_string())
        file.write("\n\nCorrelations:\n")
        file.write(df[[TARGET_COLUMN, *RENEWABLE_COLUMNS]].corr().to_string())

    plt.figure(figsize=(12, 4))
    plt.plot(df[TIMESTAMP_COLUMN], df[TARGET_COLUMN], linewidth=0.8)
    plt.title("Electricity Demand Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand (MW)")
    plt.tight_layout()
    plt.savefig(results_dir / "eda_demand_over_time.png", dpi=150)
    plt.close()

    short_period = df.iloc[: 24 * 14]
    plt.figure(figsize=(12, 4))
    plt.plot(short_period[TIMESTAMP_COLUMN], short_period[TARGET_COLUMN], linewidth=1.2)
    plt.title("Electricity Demand: First Two Weeks")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand (MW)")
    plt.tight_layout()
    plt.savefig(results_dir / "eda_demand_two_weeks.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(df[TARGET_COLUMN], bins=40, color="#4c78a8", edgecolor="white")
    plt.title("Demand Distribution")
    plt.xlabel("Demand (MW)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(results_dir / "eda_demand_distribution.png", dpi=150)
    plt.close()

    corr = df[[TARGET_COLUMN, *RENEWABLE_COLUMNS]].corr()
    plt.figure(figsize=(5, 4))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=30, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    plt.title("Feature Correlations")
    plt.tight_layout()
    plt.savefig(results_dir / "eda_correlations.png", dpi=150)
    plt.close()

    example = df.iloc[: 24 * 7]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(example[TIMESTAMP_COLUMN], example[TARGET_COLUMN], color="#4c78a8")
    axes[0].set_ylabel("Demand")
    axes[1].plot(example[TIMESTAMP_COLUMN], example["Wind (MW)"], color="#59a14f")
    axes[1].set_ylabel("Wind")
    axes[2].plot(example[TIMESTAMP_COLUMN], example["Solar (MW)"], color="#f28e2b")
    axes[2].set_ylabel("Solar")
    axes[2].set_xlabel("Timestamp")
    fig.suptitle("Demand, Wind, and Solar Example: First Week")
    fig.tight_layout()
    fig.savefig(results_dir / "eda_demand_wind_solar_example.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_eda()
