from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def set_random_seeds(seed: int = 42) -> None:
    """Make model training as reproducible as TensorFlow allows on CPU."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm_model(input_shape: tuple[int, int], lstm_units: int = 32, dense_units: int = 16) -> keras.Model:
    """Build a small CPU-friendly LSTM model."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(lstm_units),
            layers.Dropout(0.2),
            layers.Dense(dense_units, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
    return model


def train_lstm_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    patience: int = 3,
) -> keras.callbacks.History:
    """Train one LSTM model with early stopping."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=2,
        callbacks=callbacks,
    )


def save_model(model: keras.Model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
