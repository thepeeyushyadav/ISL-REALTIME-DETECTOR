"""
model/evaluate_model.py
========================
Comprehensive model evaluation:
  - Classification report (precision, recall, F1 per sign)
  - Confusion matrix heatmap
  - Per-class accuracy bar chart
  - Worst-performing signs analysis

Usage:
    python model/evaluate_model.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ISL_SIGNS, DATASET_DIR, SEQUENCE_LENGTH, NUM_FEATURES, MODEL_DIR, MODEL_PATH, BASE_DIR


def load_test_data(label_map: dict, min_seqs: int = 5) -> tuple:
    """Reload dataset and return test split (same seed as training)."""
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder

    sequences, labels = [], []
    valid_signs = list(label_map.keys())

    for sign in valid_signs:
        sign_dir = os.path.join(DATASET_DIR, sign)
        if not os.path.exists(sign_dir):
            continue
        npy_files = sorted(Path(sign_dir).glob("*.npy"))
        for npy_file in npy_files:
            try:
                seq = np.load(str(npy_file))
                if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                    sequences.append(seq)
                    labels.append(sign)
            except Exception:
                pass

    X = np.array(sequences, dtype=np.float32)
    y_str = np.array(labels)
    y_enc = np.array([label_map[s] for s in y_str])
    y_cat = to_categorical(y_enc, num_classes=len(valid_signs))

    # Reproduce same split as training
    _, X_temp, _, y_temp_cat, _, y_temp_enc = train_test_split(
        X, y_cat, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )
    _, X_test, _, y_test_cat, _, y_test_enc = train_test_split(
        X_temp, y_temp_cat, y_temp_enc, test_size=0.40, random_state=42
    )

    return X_test, y_test_cat, y_test_enc


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: str) -> None:
    """Save a large annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(max(12, len(class_names)), max(10, len(class_names) - 2)))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#1a1a2e")

    # Normalize
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="#333",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", color="white", fontsize=11)
    ax.set_ylabel("True Label",      color="white", fontsize=11)
    ax.set_title("ISL — Confusion Matrix (Normalized)", color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white", labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Confusion matrix saved: {out_path}")


def plot_per_class_accuracy(class_names: list, per_class_acc: list, out_path: str) -> None:
    """Save per-class accuracy bar chart."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#1a1a2e")

    x = np.arange(len(class_names))
    colors = ["#00e876" if a >= 0.9 else ("#ffa500" if a >= 0.7 else "#dc3545") for a in per_class_acc]
    bars = ax.bar(x, [a * 100 for a in per_class_acc], color=colors, alpha=0.85, width=0.7)

    ax.axhline(90, color="#00c8ff", linestyle="--", linewidth=1.5, label="90% threshold")
    ax.axhline(70, color="orange",  linestyle="--", linewidth=1.0, label="70% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9, color="white")
    ax.set_ylabel("Accuracy (%)", color="white")
    ax.set_title("Per-Class Accuracy", color="white", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="white")
    ax.grid(axis="y", alpha=0.15)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc*100:.0f}%", ha="center", va="bottom", fontsize=7, color="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Per-class accuracy chart saved: {out_path}")


def main():
    print("=" * 60)
    print("  ISL Model Evaluation")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Run: python model/train_model.py first")
        sys.exit(1)

    print(f"\n  Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # ── Load label map ────────────────────────────────────────
    label_map_path = os.path.join(BASE_DIR, "dataset", "label_map.json")
    with open(label_map_path) as f:
        label_map = json.load(f)

    idx_to_sign = {v: k for k, v in label_map.items()}
    class_names = [idx_to_sign[i] for i in range(len(label_map))]

    # ── Load test data ────────────────────────────────────────
    print("  Loading test data...")
    X_test, y_test_cat, y_test_enc = load_test_data(label_map)
    print(f"  Test samples: {len(X_test)}")

    # ── Predictions ───────────────────────────────────────────
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # ── Metrics ───────────────────────────────────────────────
    overall_acc = accuracy_score(y_test_enc, y_pred)
    top3_acc    = top_k_accuracy_score(y_test_enc, y_pred_prob, k=3)

    print(f"\n  Overall Accuracy : {overall_acc*100:.2f}%")
    print(f"  Top-3 Accuracy   : {top3_acc*100:.2f}%")

    # ── Classification Report ─────────────────────────────────
    report = classification_report(y_test_enc, y_pred, target_names=class_names, digits=3)
    print(f"\n{report}")

    report_path = os.path.join(MODEL_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall Accuracy : {overall_acc*100:.2f}%\n")
        f.write(f"Top-3 Accuracy   : {top3_acc*100:.2f}%\n\n")
        f.write(report)
    print(f"  📄 Report saved: {report_path}")

    # ── Confusion Matrix ──────────────────────────────────────
    cm = confusion_matrix(y_test_enc, y_pred)
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    # ── Per-class Accuracy ────────────────────────────────────
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    acc_path = os.path.join(MODEL_DIR, "per_class_accuracy.png")
    plot_per_class_accuracy(class_names, per_class_acc.tolist(), acc_path)

    # ── Worst signs ───────────────────────────────────────────
    worst = sorted(zip(class_names, per_class_acc.tolist()), key=lambda x: x[1])[:5]
    print("\n  ⚠  5 Worst-Performing Signs:")
    for sign, acc in worst:
        print(f"     {sign:<20} {acc*100:.1f}%")

    print("\n✅ Evaluation complete!")
    print(f"   Results in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
