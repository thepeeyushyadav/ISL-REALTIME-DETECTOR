"""
ISL Sign Language to Text
=========================
MediaPipe + LSTM based Indian Sign Language Recognition System

Supports:
- Real-time webcam detection
- Video file processing
- 30-class ISL word recognition
"""

# ─── ISL Sign Classes ─────────────────────────────────────────────────────────
# 30 signs selected from the INCLUDE dataset (daskoushik/landmarks-include)
# Folder names must match EXACTLY (case-sensitive) as they appear in the dataset
ISL_SIGNS = [
    "Thumbs Up",
    "Thumbs Down",
    "Open Palm",
    "Closed Hand",
    "Victory",
    "OK",
    "I Love You",
    "Call Me",
    "Right",
    "Left",
]

NUM_SIGNS = len(ISL_SIGNS)
SEQUENCE_LENGTH = 30   # frames per sequence
NUM_FEATURES = 225     # 21*2 hands*3 + 33 pose*3 = 126 + 99

# ─── Paths ────────────────────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "isl_lstm_model.h5")
LOGS_DIR = os.path.join(BASE_DIR, "logs", "training")
