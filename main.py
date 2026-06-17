import os
import sys

# Crucial for PyInstaller multiprocessing (used by some mediapipe/tf internals)
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()

import cv2
from app.realtime_detection import main

if __name__ == "__main__":
    print("Starting ISL Real-Time Detector...")
    try:
        main()
    except Exception as e:
        print(f"Application crashed: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Exiting...")
        sys.exit(0)
