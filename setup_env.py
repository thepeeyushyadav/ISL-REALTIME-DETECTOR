"""
setup_env.py
=============
One-click environment setup script.
Creates a virtual environment and installs all dependencies.

Usage:
    python setup_env.py
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"
REQ_FILE = BASE_DIR / "requirements.txt"


def run(cmd: list, **kwargs):
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"  ❌ Command failed: {' '.join(cmd)}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  ISL Sign Language — Environment Setup")
    print("=" * 60)
    print(f"  Python: {sys.version}")
    print(f"  Base:   {BASE_DIR}\n")

    # Create venv
    if not VENV_DIR.exists():
        print("📦 Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print("✅ Virtual environment already exists.")

    # Determine pip path
    if os.name == "nt":  # Windows
        pip  = str(VENV_DIR / "Scripts" / "pip.exe")
        python = str(VENV_DIR / "Scripts" / "python.exe")
    else:
        pip  = str(VENV_DIR / "bin" / "pip")
        python = str(VENV_DIR / "bin" / "python")

    # Upgrade pip
    print("\n📦 Upgrading pip...")
    run([python, "-m", "pip", "install", "--upgrade", "pip"])

    # Install requirements
    print("\n📦 Installing requirements...")
    run([pip, "install", "-r", str(REQ_FILE)])

    # Verify key imports
    print("\n🔍 Verifying installations...")
    checks = [
        ("tensorflow",  "import tensorflow as tf; print(f'  TF: {tf.__version__}')"),
        ("mediapipe",   "import mediapipe as mp; print(f'  MediaPipe: {mp.__version__}')"),
        ("cv2",         "import cv2; print(f'  OpenCV: {cv2.__version__}')"),
        ("numpy",       "import numpy as np; print(f'  NumPy: {np.__version__}')"),
        ("sklearn",     "import sklearn; print(f'  scikit-learn: {sklearn.__version__}')"),
    ]

    all_ok = True
    for pkg, check_code in checks:
        try:
            result = subprocess.run([python, "-c", check_code], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"  ⚠  {pkg}: {result.stderr.strip()}")
                all_ok = False
        except Exception as e:
            print(f"  ❌ {pkg}: {e}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ Setup complete!\n")

        if os.name == "nt":
            activate = r".\venv\Scripts\activate"
        else:
            activate = "source ./venv/bin/activate"

        print(f"  Activate environment: {activate}")
        print("\n  Next steps:")
        print("  1. Activate venv (command above)")
        print("  2. python data_collection/download_dataset.py")
        print("  3. python data_collection/preprocess_dataset.py")
        print("  4. python data_collection/verify_data.py")
        print("  5. python model/train_model.py")
        print("  6. python model/evaluate_model.py")
        print("  7. python app/realtime_detection.py")
    else:
        print("  ⚠  Some packages may have issues. Check above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
