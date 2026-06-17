"""
data_collection/download_dataset.py
=====================================
Downloads and prepares the ISL dataset from Kaggle.

Uses:  kaggle.com/datasets/sagnikghosh/mediapipe-landmarks-on-include
       (Pre-extracted MediaPipe landmarks from the INCLUDE ISL dataset)
       263 ISL signs, 4287 videos worth of landmarks

We filter to our 30 target signs and convert to .npy sequences.

Usage:
    python data_collection/download_dataset.py

Prerequisites:
    1. Install Kaggle API:  pip install kaggle
    2. Get API key from:   https://www.kaggle.com/settings → API → Create New Token
    3. Place kaggle.json in: C:/Users/<you>/.kaggle/kaggle.json
       OR set env vars: KAGGLE_USERNAME and KAGGLE_KEY
"""

import os
import sys
import json
import shutil
import zipfile
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ISL_SIGNS, DATASET_DIR, BASE_DIR

KAGGLE_DATASET = "sagnikghosh/mediapipe-landmarks-on-include"
DOWNLOAD_DIR   = os.path.join(BASE_DIR, "dataset", "raw_download")


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials exist."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_ok = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    if kaggle_json.exists() or env_ok:
        return True
    print("\n" + "="*60)
    print("  [WARNING] Kaggle API credentials not found!")
    print("="*60)
    print("""
Steps to set up Kaggle API:
  1. Go to: https://www.kaggle.com/settings
  2. Click 'API' section → 'Create New Token'
  3. Download kaggle.json
  4. Place it at: C:\\Users\\<YourName>\\.kaggle\\kaggle.json

Then re-run this script.
""")
    return False


def download_dataset() -> str:
    """Download the Kaggle dataset and return the extraction path."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print(f"\n[DOWNLOAD] Downloading dataset: {KAGGLE_DATASET}")
    print(f"   Saving to: {DOWNLOAD_DIR}")
    print("   This may take a few minutes...\n")

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", DOWNLOAD_DIR, "--unzip"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print("[OK] Dataset downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Kaggle download failed:\n{e.stderr}")
        print("\nAlternative: Download manually from:")
        print(f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print(f"  Extract to: {DOWNLOAD_DIR}")
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] 'kaggle' command not found. Run: pip install kaggle")
        sys.exit(1)

    return DOWNLOAD_DIR


def explore_dataset_structure(download_path: str) -> dict:
    """
    Explore downloaded dataset and return structure mapping.
    Returns: {sign_name: [list of .npy or data files]}
    """
    print("\n[INFO] Exploring dataset structure...")
    structure = {}

    for root, dirs, files in os.walk(download_path):
        rel_path = os.path.relpath(root, download_path)
        if rel_path == ".":
            continue
        parts = rel_path.replace("\\", "/").split("/")
        folder_name = parts[0].lower().strip()

        npy_files = [f for f in files if f.endswith(".npy")]
        csv_files = [f for f in files if f.endswith(".csv")]
        json_files = [f for f in files if f.endswith(".json")]

        if npy_files or csv_files or json_files:
            structure[folder_name] = {
                "path": root,
                "npy": npy_files,
                "csv": csv_files,
                "json": json_files,
            }

    print(f"   Found {len(structure)} sign categories in dataset")
    return structure


def find_matching_signs(structure: dict) -> dict:
    """Match our target ISL_SIGNS to dataset folders."""
    matched = {}
    unmatched = []

    for sign in ISL_SIGNS:
        sign_lower = sign.lower()
        if sign_lower in structure:
            matched[sign] = structure[sign_lower]
        else:
            # Try partial match
            found = False
            for key in structure:
                if sign_lower in key or key in sign_lower:
                    matched[sign] = structure[key]
                    print(f"   [MATCH] Partial match: '{sign}' -> '{key}'")
                    found = True
                    break
            if not found:
                unmatched.append(sign)

    print(f"\n[OK] Matched: {len(matched)}/{len(ISL_SIGNS)} signs")
    if unmatched:
        print(f"[WARNING] Unmatched signs: {unmatched}")
        print("   These signs may need custom data collection.")

    return matched


def main():
    print("=" * 60)
    print("  ISL Dataset Download & Preparation")
    print("=" * 60)

    if not check_kaggle_credentials():
        sys.exit(1)

    # Download
    download_path = download_dataset()

    # Explore structure
    structure = explore_dataset_structure(download_path)

    if not structure:
        print("\n[WARNING] No data files found in download.")
        print(f"   Please check the contents of: {download_path}")
        sys.exit(1)

    # Match signs
    matched = find_matching_signs(structure)

    print(f"\n[OK] Dataset ready at: {download_path}")
    print("\nNext step: Run preprocess_dataset.py to convert to .npy sequences")
    print("  python data_collection/preprocess_dataset.py")

    # Save structure summary
    summary_path = os.path.join(BASE_DIR, "dataset", "dataset_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        # Serialize structure (paths only)
        serializable = {k: {"path": v["path"]} for k, v in structure.items()}
        json.dump({"matched_signs": list(matched.keys()), "structure": serializable}, f, indent=2)
    print(f"   Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
