# Short-Term Electricity Demand Forecasting Using LSTM with Renewable Energy Features

This project forecasts next-hour electricity demand for India using a simple previous-day baseline and small CPU-friendly LSTM models. It compares whether adding renewable energy features, specifically wind and solar generation, improves short-term demand forecasting.

## Dataset

The project uses only:

```text
data/india_electricity_clean.csv
```

Required columns:

- `Timestamp`
- `Demand (MW)`
- `Wind (MW)`
- `Solar (MW)`
- `Total Generation (MW)`

`Total Generation (MW)` is intentionally not used as an input feature because it may be too closely related to demand and could introduce leakage.

## Problem Statement

Predict the next-hour `Demand (MW)` from historical electricity demand and, in the multivariate model, historical wind and solar generation.

## Methodology

The pipeline:

1. Loads and parses timestamps.
2. Sorts data chronologically.
3. Checks duplicate timestamps, missing values, and hourly spacing.
4. Splits data chronologically into 70% train, 15% validation, and 15% test.
5. Fits scalers only on the training split.
6. Builds sliding-window sequences for 12, 24, and 48 hour lookbacks.
7. Trains small LSTM models with early stopping.
8. Evaluates all models on the held-out test set.

The time series is never randomly shuffled.

## Models

### Previous-Day Baseline

The baseline predicts demand at time `t` using actual demand from `t-24 hours`.

### Demand-Only LSTM

Uses historical `Demand (MW)` only.

Lookbacks:

- 12 hours
- 24 hours
- 48 hours

### Demand + Wind + Solar LSTM

Uses historical:

- `Demand (MW)`
- `Wind (MW)`
- `Solar (MW)`

Lookbacks:

- 12 hours
- 24 hours
- 48 hours

## Model Architecture

The LSTM is intentionally small for normal laptops:

```text
Input
LSTM(32)
Dropout(0.2)
Dense(16, ReLU)
Dense(1)
```

Training uses Adam optimizer, MSE loss, batch size 32, and early stopping.

## Evaluation Metrics

Models are evaluated with:

- MAE
- RMSE
- MAPE

Final metrics are saved to:

```text
results/metrics.csv
```

## How to Run

From the project root:

```powershell
.\venv\Scripts\python.exe train.py
```

To run fewer epochs while testing:

```powershell
.\venv\Scripts\python.exe train.py --epochs 5
```

To regenerate only EDA plots:

```powershell
.\venv\Scripts\python.exe -m src.eda
```

## Outputs

The pipeline saves:

- `results/metrics.csv`
- `results/baseline_predictions.csv`
- `results/lstm_predictions.csv`
- `results/best_lstm_predictions.csv`
- EDA plots under `results/`
- Forecast comparison plots under `results/`
- Trained models under `models/`

## Results

Measured results from the completed run are saved in `results/metrics.csv`.

| Model | Lookback | MAE | RMSE | MAPE |
| --- | --- | ---: | ---: | ---: |
| Previous-day baseline | 24h lag | 6647.46 | 9049.92 | 3.33 |
| Demand-only LSTM | 12h | 6339.60 | 7940.99 | 3.02 |
| Demand-only LSTM | 24h | 3614.24 | 4830.07 | 1.74 |
| Demand-only LSTM | 48h | 7189.34 | 9036.32 | 3.41 |
| Demand+Wind+Solar LSTM | 12h | 8933.66 | 9999.11 | 4.32 |
| Demand+Wind+Solar LSTM | 24h | 9500.13 | 11201.48 | 4.59 |
| Demand+Wind+Solar LSTM | 48h | 10377.17 | 11662.86 | 4.99 |

In this run, the demand-only LSTM with a 24-hour lookback performed best by RMSE. Adding wind and solar did not improve test performance for this model setup.

## Limitations

- The project uses a compact LSTM rather than extensive hyperparameter tuning.
- Results depend on the available cleaned dataset and its date range.
- Weather, calendar, holiday, and regional demand drivers are not included.
- The baseline is simple and may be strong when daily seasonality dominates.

## Future Improvements

- Add calendar features such as hour of day, day of week, and holidays.
- Test additional baselines such as weekly lag and moving averages.
- Compare with tree-based models or statistical forecasting methods.
- Tune lookback windows and LSTM sizes more systematically.
- Add rolling-origin validation for more robust time-series evaluation.
