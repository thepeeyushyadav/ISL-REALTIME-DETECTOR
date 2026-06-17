"""
data_collection/augment_data.py
================================
Data augmentation for ISL sequences to improve model accuracy.

Techniques applied to each .npy sequence:
  1. Gaussian noise       - adds small random noise
  2. Time scaling         - speeds up / slows down sequence
  3. Spatial scaling      - scales x,y,z coordinates
  4. Coordinate jitter    - slight landmark position shifts

This multiplies dataset size by ~5x without new recordings.

Usage:
    python data_collection/augment_data.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ISL_SIGNS, DATASET_DIR, SEQUENCE_LENGTH, NUM_FEATURES

AUGMENT_FACTOR = 5   # Each sequence -> 5 augmented copies
NOISE_STD      = 0.01
SCALE_RANGE    = (0.85, 1.15)


def add_gaussian_noise(seq: np.ndarray) -> np.ndarray:
    """Add small Gaussian noise to all coordinates."""
    noise = np.random.normal(0, NOISE_STD, seq.shape).astype(np.float32)
    return np.clip(seq + noise, 0.0, 1.0)


def time_warp(seq: np.ndarray) -> np.ndarray:
    """Resample sequence at slightly different speed."""
    T = seq.shape[0]
    factor = np.random.uniform(0.85, 1.15)
    new_len = int(T * factor)
    if new_len < 2:
        return seq
    indices = np.linspace(0, T - 1, new_len)
    warped = np.array([
        seq[int(i)] * (1 - i % 1) + seq[min(int(i) + 1, T - 1)] * (i % 1)
        for i in indices
    ], dtype=np.float32)
    # Resize back to SEQUENCE_LENGTH
    out_indices = np.linspace(0, len(warped) - 1, SEQUENCE_LENGTH, dtype=int)
    return warped[out_indices]


def spatial_scale(seq: np.ndarray) -> np.ndarray:
    """Scale all coordinates by a random factor."""
    scale = np.random.uniform(*SCALE_RANGE)
    return np.clip(seq * scale, 0.0, 1.0)


def coordinate_jitter(seq: np.ndarray) -> np.ndarray:
    """Apply per-frame random jitter to landmark positions."""
    jitter = np.random.uniform(-0.005, 0.005, seq.shape).astype(np.float32)
    return np.clip(seq + jitter, 0.0, 1.0)


AUGMENTATIONS = [add_gaussian_noise, time_warp, spatial_scale, coordinate_jitter]


def augment_sequence(seq: np.ndarray) -> list:
    """Return AUGMENT_FACTOR augmented versions of seq."""
    augmented = []
    for _ in range(AUGMENT_FACTOR):
        aug = seq.copy()
        # Apply 1-3 random augmentations
        n_augs = np.random.randint(1, len(AUGMENTATIONS))
        chosen = np.random.choice(AUGMENTATIONS, size=n_augs, replace=False)
        for fn in chosen:
            aug = fn(aug)
        if aug.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
            augmented.append(aug)
    return augmented


def main():
    print("=" * 60)
    print("  ISL Data Augmentation")
    print(f"  Multiplying each sequence x{AUGMENT_FACTOR}")
    print("=" * 60)

    total_original = 0
    total_augmented = 0

    for sign in tqdm(ISL_SIGNS, desc="Augmenting signs"):
        sign_dir = os.path.join(DATASET_DIR, sign)
        if not os.path.isdir(sign_dir):
            continue

        # Only augment original files (no aug_ prefix)
        original_files = [f for f in sorted(Path(sign_dir).glob("*.npy"))
                          if not f.stem.startswith("aug_")]

        sign_aug_count = 0
        aug_idx = 0

        for orig_file in original_files:
            try:
                seq = np.load(str(orig_file))
                if seq.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                    continue
                total_original += 1

                augmented = augment_sequence(seq)
                for aug_seq in augmented:
                    out_path = os.path.join(sign_dir, f"aug_{aug_idx}.npy")
                    np.save(out_path, aug_seq)
                    aug_idx += 1
                    sign_aug_count += 1

            except Exception:
                pass

        total_augmented += sign_aug_count
        tqdm.write(f"  {sign:<20} {len(original_files)} orig -> +{sign_aug_count} aug")

    print("\n" + "=" * 60)
    print(f"  Original sequences : {total_original}")
    print(f"  Augmented added    : {total_augmented}")
    print(f"  Total now          : {total_original + total_augmented}")
    print("=" * 60)
    print("\nNext: python model/train_model.py --epochs 200")


if __name__ == "__main__":
    main()
