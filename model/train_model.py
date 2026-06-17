"""
model/train_model.py
=====================
Loads preprocessed .npy sequences, trains the LSTM model, and saves weights.

Usage:
    python model/train_model.py
    python model/train_model.py --epochs 300 --batch_size 32

Output:
    model/isl_lstm_model.h5      — best model weights
    logs/training/               — TensorBoard logs
    dataset/label_map.json       — sign → index mapping
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ISL_SIGNS, DATASET_DIR, SEQUENCE_LENGTH, NUM_FEATURES, MODEL_DIR, MODEL_PATH, LOGS_DIR, BASE_DIR
from model.lstm_model import build_lstm_model, get_callbacks


def parse_args():
    p = argparse.ArgumentParser(description="ISL LSTM Trainer")
    p.add_argument("--epochs",      type=int,   default=200,  help="Max training epochs")
    p.add_argument("--batch_size",  type=int,   default=32,   help="Batch size")
    p.add_argument("--val_split",   type=float, default=0.15, help="Validation split ratio")
    p.add_argument("--test_split",  type=float, default=0.10, help="Test split ratio")
    p.add_argument("--min_seqs",    type=int,   default=10,   help="Min sequences required per sign")
    return p.parse_args()


def load_dataset(min_sequences: int = 10) -> tuple:
    """
    Load all .npy sequences from DATASET_DIR.
    Returns (X, y_labels, valid_signs) where X shape is (N, 30, 225).
    """
    sequences, labels, valid_signs = [], [], []
    skipped_signs = []

    print(f"\nLoading dataset from: {DATASET_DIR}\n")

    for sign in ISL_SIGNS:
        sign_dir = os.path.join(DATASET_DIR, sign)

        if not os.path.exists(sign_dir):
            skipped_signs.append(f"{sign} (directory missing)")
            continue

        npy_files = sorted(Path(sign_dir).glob("*.npy"))
        sign_seqs = []

        for npy_file in npy_files:
            try:
                seq = np.load(str(npy_file))
                if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                    sign_seqs.append(seq)
                else:
                    pass  # skip malformed
            except Exception:
                pass

        if len(sign_seqs) < min_sequences:
            skipped_signs.append(f"{sign} ({len(sign_seqs)} seqs, need {min_sequences}+)")
            continue

        sequences.extend(sign_seqs)
        labels.extend([sign] * len(sign_seqs))
        valid_signs.append(sign)
        print(f"  [OK] {sign:<20} {len(sign_seqs):>4} sequences")

    if skipped_signs:
        print(f"\n  [SKIPPED] Skipped signs:")
        for s in skipped_signs:
            print(f"     - {s}")

    if not sequences:
        raise ValueError("[ERROR] No valid sequences found! Run preprocess_dataset.py first.")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels)
    return X, y, valid_signs


def build_label_map(valid_signs: list) -> dict:
    """Create and save sign → index mapping."""
    label_map = {sign: i for i, sign in enumerate(sorted(valid_signs))}
    label_map_path = os.path.join(BASE_DIR, "dataset", "label_map.json")
    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n  [SAVED] Label map saved: {label_map_path}")
    return label_map


def plot_training_history(history, output_dir: str) -> None:
    """Save training curves as PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f23")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Acc",  color="#00e876", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    color="#00c8ff", linewidth=2, linestyle="--")
    axes[0].set_title("Model Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white")
    axes[0].grid(alpha=0.2)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", color="#ff6b6b", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#ffa500", linewidth=2, linestyle="--")
    axes[1].set_title("Model Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [CHART] Training curves saved: {out_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("  ISL LSTM Model Training")
    print("=" * 60)

    # ── GPU Setup ─────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n  [GPU] GPU detected: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("  Memory growth enabled.")
    else:
        print("\n  [CPU] No GPU detected, using CPU.")

    # ── Load Data ─────────────────────────────────────────────
    X, y_labels, valid_signs = load_dataset(min_sequences=args.min_seqs)

    print(f"\n  Dataset shape: {X.shape}")
    print(f"  Classes: {len(valid_signs)}")

    # ── Label Encoding ────────────────────────────────────────
    label_map = build_label_map(valid_signs)
    le = LabelEncoder()
    le.classes_ = np.array(sorted(valid_signs))
    y_encoded = np.array([label_map[sign] for sign in y_labels])
    y_cat = to_categorical(y_encoded, num_classes=len(valid_signs))

    # ── Train / Val / Test Split ──────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_cat, test_size=args.val_split + args.test_split, random_state=42, stratify=y_encoded
    )
    val_fraction = args.val_split / (args.val_split + args.test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_fraction, random_state=42
    )

    print(f"\n  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # ── Build Model ───────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    model = build_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        num_features=NUM_FEATURES,
        num_classes=len(valid_signs),
    )
    model.summary()

    # ── Train ─────────────────────────────────────────────────
    log_dir = os.path.join(LOGS_DIR, f"run_{int(time.time())}")
    callbacks = get_callbacks(log_dir, MODEL_PATH)

    print(f"\n  [START] Training started  (max {args.epochs} epochs, batch {args.batch_size})")
    print(f"  TensorBoard: tensorboard --logdir {LOGS_DIR}\n")

    t_start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t_start

    # ── Evaluate on Test Set ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION ON TEST SET")
    print("=" * 60)
    test_loss, test_acc, test_top3 = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test Top-3 Acc: {test_top3*100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Training Time : {elapsed/60:.1f} min")

    # ── Save model + metadata ─────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n  [SAVED] Model saved: {MODEL_PATH}")

    meta = {
        "signs": sorted(valid_signs),
        "num_classes": len(valid_signs),
        "sequence_length": SEQUENCE_LENGTH,
        "num_features": NUM_FEATURES,
        "test_accuracy": round(test_acc, 4),
        "test_top3_accuracy": round(test_top3, 4),
        "training_epochs": len(history.history["accuracy"]),
        "training_time_min": round(elapsed / 60, 2),
    }
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [META] Metadata saved: {meta_path}")

    # ── Plot curves ───────────────────────────────────────────
    plot_training_history(history, MODEL_DIR)

    print("\n[DONE] Training complete!")
    print("Next step: python model/evaluate_model.py")
    print("Then:       python app/realtime_detection.py")


if __name__ == "__main__":
    main()
