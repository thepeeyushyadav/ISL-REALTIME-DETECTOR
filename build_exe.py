import os
import site
import subprocess
from pathlib import Path

def get_mediapipe_path():
    for site_pkg in site.getsitepackages():
        mp_path = os.path.join(site_pkg, "mediapipe")
        if os.path.isdir(mp_path):
            return mp_path
    
    # Try user site packages
    mp_path = os.path.join(site.getusersitepackages(), "mediapipe")
    if os.path.isdir(mp_path):
        return mp_path
        
    raise Exception("MediaPipe not found in site-packages!")

def build():
    print("Finding MediaPipe installation...")
    mp_path = get_mediapipe_path()
    print(f"Found MediaPipe at: {mp_path}")

    # Build command
    cmd = [
        "pyinstaller",
        "--name", "ISL-Detector",
        "--onedir",
        "--noconfirm",
        "--windowed", # Don't show console window if possible (but we might need it for print statements, let's keep console for now)
        "--add-data", f"model/isl_lstm_model.h5;model/",
        "--add-data", f"{mp_path};mediapipe/",
        "main.py"
    ]
    
    # Remove --windowed so users can see logs if it crashes
    cmd.remove("--windowed")

    print("\nRunning PyInstaller...")
    subprocess.run(cmd, check=True)
    print("\nBuild Complete! Executable is in the 'dist/ISL-Detector' folder.")

if __name__ == "__main__":
    build()
