"""
data_collection/preprocess_dataset.py
======================================
Converts downloaded "Mediapipe Landmarks on INCLUDE" dataset
(daskoushik/landmarks-include) into standardized .npy sequences.

Dataset format (Parquet per video):
  Columns: type, landmark_index, x, y, z, video_id, frame
  Types:   face(468), pose(33), left_hand(21), right_hand(21)

We extract ONLY:
  - pose:       33 landmarks x 3 = 99 features
  - left_hand:  21 landmarks x 3 = 63 features
  - right_hand: 21 landmarks x 3 = 63 features
  Total = 225 features per frame  (matching LSTM input)

Output:
  dataset/MP_Data/<sign_name>/<seq_index>.npy   shape: (30, 225)

Usage:
    python data_collection/preprocess_dataset.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ISL_SIGNS, DATASET_DIR, SEQUENCE_LENGTH, NUM_FEATURES, BASE_DIR

RAW_DIR       = os.path.join(BASE_DIR, "dataset", "raw_download", "landmark_files_all")
MAX_SEQUENCES = 100
MIN_SEQUENCES = 10

# Landmark counts and order (must produce 225 features total)
USED_TYPES = ["pose", "left_hand", "right_hand"]
LANDMARK_COUNTS = {"pose": 33, "left_hand": 21, "right_hand": 21}


def parquet_to_sequence(parquet_path: str) -> np.ndarray | None:
    """
    Convert one .parquet file (one video) into a (SEQUENCE_LENGTH, 225) array.
    Returns None if the file is invalid.
    """
    try:
        df = pd.read_parquet(parquet_path)
        
        # Replace NaN values with 0.0 (happens when landmarks are not detected)
        df[["x", "y", "z"]] = df[["x", "y", "z"]].fillna(0.0)

        # Validate columns
        required = {"type", "landmark_index", "x", "y", "z", "frame"}
        if not required.issubset(set(df.columns)):
            return None

        # Only keep pose, hands (skip face — saves space, not needed for ISL)
        df = df[df["type"].isin(USED_TYPES)].copy()
        frames = sorted(df["frame"].unique())
        if not frames:
            return None

        frame_vectors = []
        for frame_idx in frames:
            frame_df = df[df["frame"] == frame_idx]
            frame_kp = []

            for lm_type in USED_TYPES:
                n = LANDMARK_COUNTS[lm_type]
                type_df = (
                    frame_df[frame_df["type"] == lm_type]
                    .sort_values("landmark_index")
                )

                if len(type_df) >= n:
                    coords = type_df[["x", "y", "z"]].values[:n].flatten()
                else:
                    # Pad with zeros for missing landmarks
                    coords = np.zeros(n * 3, dtype=np.float32)
                    if len(type_df) > 0:
                        available = type_df[["x", "y", "z"]].values.flatten()
                        coords[: len(available)] = available

                frame_kp.append(coords)

            frame_vectors.append(np.concatenate(frame_kp))  # (225,)

        sequence = np.array(frame_vectors, dtype=np.float32)  # (T, 225)
        T = len(sequence)

        if T == 0:
            return None
        elif T == SEQUENCE_LENGTH:
            return sequence
        elif T > SEQUENCE_LENGTH:
            indices = np.linspace(0, T - 1, SEQUENCE_LENGTH, dtype=int)
            return sequence[indices]
        else:
            # Pad by repeating last frame
            pad = np.tile(sequence[-1:], (SEQUENCE_LENGTH - T, 1))
            return np.vstack([sequence, pad]).astype(np.float32)

    except Exception:
        return None


def process_sign(sign_name: str) -> int:
    """Process all parquet files for one sign. Returns count saved."""
    sign_folder = os.path.join(RAW_DIR, sign_name)
    if not os.path.isdir(sign_folder):
        return 0

    out_dir = os.path.join(DATASET_DIR, sign_name)
    os.makedirs(out_dir, exist_ok=True)

    parquet_files = sorted(Path(sign_folder).glob("*.parquet"))[:MAX_SEQUENCES]
    saved = 0

    for i, pq_file in enumerate(parquet_files):
        seq = parquet_to_sequence(str(pq_file))
        if seq is not None and seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
            np.save(os.path.join(out_dir, f"{i}.npy"), seq)
            saved += 1

    return saved


def main():
    print("=" * 60)
    print("  ISL Dataset Preprocessing")
    print("  Parquet -> .npy sequences (30 frames, 225 features)")
    print("=" * 60)

    if not os.path.isdir(RAW_DIR):
        print(f"\nERROR: Dataset not found at:\n  {RAW_DIR}")
        print("\nExtract archive.zip to:")
        print(f"  {os.path.join(BASE_DIR, 'dataset', 'raw_download')}")
        sys.exit(1)

    os.makedirs(DATASET_DIR, exist_ok=True)

    print(f"\nSource : {RAW_DIR}")
    print(f"Output : {DATASET_DIR}")
    print(f"Signs  : {len(ISL_SIGNS)}\n")

    stats = {}
    total = 0

    for sign in tqdm(ISL_SIGNS, desc="Processing signs"):
        count = process_sign(sign)
        stats[sign] = count
        total += count

        ok = "OK  " if count >= MIN_SEQUENCES else ("LOW " if count > 0 else "FAIL")
        tqdm.write(f"  [{ok}] {sign:<20} -> {count} sequences")

    # Summary
    ready = [s for s, c in stats.items() if c >= MIN_SEQUENCES]
    missing = [s for s, c in stats.items() if c == 0]

    print("\n" + "=" * 60)
    print(f"  Total sequences  : {total}")
    print(f"  Signs ready      : {len(ready)}/{len(ISL_SIGNS)}")
    if missing:
        print(f"  Signs not found  : {missing}")

    # Save stats
    stats_path = os.path.join(BASE_DIR, "dataset", "preprocessing_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"total": total, "ready": ready, "stats": stats}, f, indent=2)

    print(f"\n  Stats saved: {stats_path}")

    if len(ready) >= 5:
        print("\nNext step: python model/train_model.py")
    else:
        print("\nWARNING: Too few signs ready. Check dataset structure.")


if __name__ == "__main__":
    main()
