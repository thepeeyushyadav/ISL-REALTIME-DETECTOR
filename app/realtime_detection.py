"""
app/realtime_detection.py
==========================
Real-time ISL Sign Language detection from webcam.

Features:
  - Live MediaPipe hand + pose landmark overlay
  - LSTM inference on 30-frame sliding window
  - Sentence builder (accumulates detected signs)
  - Top-3 predictions with confidence bars
  - Sign history panel
  - FPS display

Controls:
  C — Clear sentence
  S — Save screenshot
  Q — Quit

Usage:
    python app/realtime_detection.py
    python app/realtime_detection.py --camera 1 --threshold 0.75
"""

import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
from collections import deque
from pathlib import Path
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SEQUENCE_LENGTH, NUM_FEATURES, MODEL_PATH, BASE_DIR
from utils.mediapipe_utils import (
    get_holistic_model, mediapipe_detection,
    extract_keypoints, draw_styled_landmarks, hands_visible,
)
from utils.visualization import (
    draw_top_bar, draw_sentence_panel, draw_top3_predictions,
    draw_sign_history, draw_no_hand_warning,
)


def parse_args():
    p = argparse.ArgumentParser(description="ISL Real-Time Detector")
    p.add_argument("--camera",    type=int,   default=0,    help="Camera index")
    p.add_argument("--threshold", type=float, default=0.85, help="Min confidence to accept prediction")
    p.add_argument("--cooldown",  type=float, default=1.5,  help="Seconds between same-sign repeats")
    p.add_argument("--width",     type=int,   default=1280, help="Camera width")
    p.add_argument("--height",    type=int,   default=720,  help="Camera height")
    return p.parse_args()


def load_model_and_labels() -> tuple:
    """Load LSTM model and label map. Returns (model, idx_to_sign)."""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("   Train first: python model/train_model.py")
        sys.exit(1)

    print(f"  Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Reconstruct label map identically to how train_model.py built it
    from config import ISL_SIGNS
    idx_to_sign = {i: sign for i, sign in enumerate(sorted(ISL_SIGNS))}

    print(f"  Loaded {len(idx_to_sign)} sign classes")
    return model, idx_to_sign


def predict_sign(
    model,
    sequence: np.ndarray,
    idx_to_sign: dict,
    threshold: float,
) -> tuple:
    """
    Run LSTM prediction on a (30, 225) sequence.
    Returns (top_sign, confidence, top3_list)
    where top3_list = [(sign, confidence), ...]
    """
    seq_input = np.expand_dims(sequence, axis=0)           # (1, 30, 225)
    probs = model.predict(seq_input, verbose=0)[0]         # (num_classes,)

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(idx_to_sign[i], float(probs[i])) for i in top3_idx]

    best_idx  = top3_idx[0]
    best_conf = float(probs[best_idx])
    best_sign = idx_to_sign[best_idx] if best_conf >= threshold else None

    return best_sign, best_conf, top3


def should_add_to_sentence(
    sign: str,
    sentence: list,
    last_sign: str,
    last_sign_time: float,
    cooldown: float,
) -> bool:
    """Decide whether to append a detected sign to the sentence."""
    now = time.time()
    if sign is None:
        return False
    # Same sign cooldown
    if sign == last_sign and (now - last_sign_time) < cooldown:
        return False
    # Don't repeat if last word is same
    if sentence and sentence[-1] == sign:
        return False
    return True


def main():
    args = parse_args()

    print("=" * 60)
    print("  ISL Real-Time Sign Language Detector")
    print("=" * 60)

    model, idx_to_sign = load_model_and_labels()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        sys.exit(1)

    print("\n  [OK] Camera opened. Starting detection...")
    print("  Controls: C=Clear sentence | S=Screenshot | Q=Quit\n")

    # ── State ──────────────────────────────────────────────────
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)   # rolling 30-frame buffer
    sentence       = []                                # accumulated words
    sign_history   = []                                # all detected signs
    current_sign   = ""
    current_conf   = 0.0
    top3           = []
    last_sign      = ""
    last_sign_time = 0.0
    fps            = 0.0
    frame_count    = 0
    t_fps          = time.time()
    screenshot_dir = os.path.join(BASE_DIR, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    with get_holistic_model(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame read error.")
                break

            frame = cv2.flip(frame, 1)

            # ── MediaPipe Processing ──────────────────────────
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # ── Keypoint Extraction & Buffer ─────────────────
            hands_present = hands_visible(results)

            if hands_present:
                keypoints = extract_keypoints(results)
                sequence_buffer.append(keypoints)
            else:
                # No hands in frame — clear buffer to avoid stale predictions
                sequence_buffer.clear()
                current_sign = ""
                current_conf = 0.0
                top3 = []

            # ── Prediction (only when hands present & buffer full) ──
            if hands_present and len(sequence_buffer) == SEQUENCE_LENGTH:
                seq = np.array(sequence_buffer)  # (30, 225)
                sign, conf, top3 = predict_sign(model, seq, idx_to_sign, args.threshold)

                if sign is not None:
                    current_sign = sign
                    current_conf = conf

                    if should_add_to_sentence(sign, sentence, last_sign, last_sign_time, args.cooldown):
                        sentence.append(sign)
                        sign_history.append(sign)
                        last_sign = sign
                        last_sign_time = time.time()
                else:
                    # Show best guess even if below threshold (greyed out)
                    current_sign = top3[0][0] if top3 else ""
                    current_conf = top3[0][1] if top3 else 0.0

            # ── FPS Calculation ───────────────────────────────
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - t_fps)
                t_fps = time.time()

            # ── Draw HUD ──────────────────────────────────────
            draw_top_bar(image, current_sign, current_conf, fps)
            draw_sentence_panel(image, sentence)
            if top3:
                draw_top3_predictions(image, top3)
            draw_sign_history(image, sign_history)
            if not hands_present:
                draw_no_hand_warning(image)

            cv2.imshow("ISL Sign Language Detector", image)

            # ── Key Controls ──────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[QUIT] Exiting.")
                break
            elif key == ord("c"):
                sentence.clear()
                print("  [CLEAR] Sentence cleared.")
            elif key == ord("s"):
                ts = int(time.time())
                ss_path = os.path.join(screenshot_dir, f"isl_{ts}.png")
                cv2.imwrite(ss_path, image)
                print(f"  [SCREENSHOT] Saved: {ss_path}")

    cap.release()
    cv2.destroyAllWindows()

    if sentence:
        print(f"\n[RESULT] Final Sentence: {' '.join(sentence)}")
    print("[DONE] Detection session ended.")


if __name__ == "__main__":
    main()
