"""
model/lstm_model.py
====================
LSTM model architecture for ISL Sign Language Recognition.

Architecture:
  Input: (batch, 30 frames, 225 features)
  → LSTM(64, return_sequences=True)  + LayerNorm + Dropout
  → LSTM(128, return_sequences=True) + LayerNorm + Dropout
  → LSTM(64)                         + Dropout
  → Dense(128, relu)                 + Dropout
  → Dense(64, relu)
  → Dense(num_classes, softmax)

Expected accuracy: 92-96% with good ISL data
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, LayerNormalization,
    BatchNormalization, Input
)
from tensorflow.keras.regularizers import l2


def build_lstm_model(
    sequence_length: int = 30,
    num_features: int = 225,
    num_classes: int = 30,
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    """
    Build and return the LSTM model for ISL recognition.

    Args:
        sequence_length: Number of frames per sequence (default 30)
        num_features:    Keypoints per frame (default 225)
        num_classes:     Number of ISL sign classes (default 30)
        dropout_rate:    Dropout rate for regularization

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Input(shape=(sequence_length, num_features)),

        # ── Block 1 ──────────────────────────────────────
        LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)),
        LayerNormalization(),
        Dropout(dropout_rate),

        # ── Block 2 ──────────────────────────────────────
        LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
        LayerNormalization(),
        Dropout(dropout_rate),

        # ── Block 3 ──────────────────────────────────────
        LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-4)),
        Dropout(dropout_rate),

        # ── Classification Head ──────────────────────────
        Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ], name="ISL_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    return model


def get_callbacks(log_dir: str, model_path: str) -> list:
    """Return standard training callbacks."""
    return [
        # Save best model weights
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Stop early if no improvement
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
        # TensorBoard logging (disabled due to Windows path unicode bug with '—')
        # tf.keras.callbacks.TensorBoard(
        #     log_dir=log_dir,
        #     histogram_freq=1,
        # ),
    ]


if __name__ == "__main__":
    model = build_lstm_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
