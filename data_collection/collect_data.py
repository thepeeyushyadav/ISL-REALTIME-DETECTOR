"""
data_collection/collect_data.py
================================
Webcam-based data collector for custom ISL signs.
Use this when a sign is missing from the downloaded dataset,
or to add more sequences for better accuracy.

Controls:
  SPACE — start recording a sequence
  S     — skip current sign
  Q     — quit

Usage:
    python data_collection/collect_data.py
    python data_collection/collect_data.py --sign hello --sequences 50
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ISL_SIGNS, DATASET_DIR, SEQUENCE_LENGTH, NUM_FEATURES
from utils.mediapipe_utils import get_holistic_model, mediapipe_detection, extract_keypoints, draw_styled_landmarks
from utils.visualization import draw_countdown, draw_recording_indicator


def parse_args():
    parser = argparse.ArgumentParser(description="ISL Data Collector")
    parser.add_argument("--sign", type=str, default=None,
                        help="Specific sign to record (default: all missing signs)")
    parser.add_argument("--sequences", type=int, default=50,
                        help="Number of sequences to record per sign (default: 50)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    return parser.parse_args()


def count_existing_sequences(sign: str) -> int:
    sign_dir = os.path.join(DATASET_DIR, sign)
    if not os.path.exists(sign_dir):
        return 0
    return len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])


def get_next_sequence_index(sign: str) -> int:
    sign_dir = os.path.join(DATASET_DIR, sign)
    if not os.path.exists(sign_dir):
        return 0
    existing = [int(f.replace(".npy", "")) for f in os.listdir(sign_dir) if f.endswith(".npy")]
    return max(existing) + 1 if existing else 0


def draw_ui(frame, sign: str, sign_idx: int, total_signs: int,
            seq_done: int, seq_total: int, status: str, countdown: int = -1):
    """Draw data collection UI overlay."""
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (15, 15, 35), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.line(frame, (0, 100), (w, 100), (60, 60, 100), 2)

    # Sign name (large)
    cv2.putText(frame, f"Sign: {sign.upper()}", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 120), 3)

    # Progress
    cv2.putText(frame, f"Sign {sign_idx+1}/{total_signs}", (w - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 140), 1)
    cv2.putText(frame, f"Seq: {seq_done}/{seq_total}", (w - 160, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 140), 1)

    # Status bar at bottom
    status_colors = {
        "READY": (0, 200, 255),
        "RECORDING": (0, 0, 220),
        "SAVED": (0, 230, 120),
    }
    color = status_colors.get(status, (200, 200, 200))
    cv2.rectangle(frame, (0, h - 60), (w, h), (15, 15, 35), -1)
    cv2.putText(frame, f"Status: {status}", (15, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, "SPACE=Record  S=Skip  Q=Quit", (w - 320, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 140), 1)

    # Countdown
    if countdown > 0:
        cv2.putText(frame, str(countdown), (w // 2 - 20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 200, 255), 5)


def record_sequence(cap, model, sign: str, seq_index: int) -> bool:
    """
    Record one 30-frame sequence for the given sign.
    Returns True on success, False on cancel.
    """
    sign_dir = os.path.join(DATASET_DIR, sign)
    os.makedirs(sign_dir, exist_ok=True)

    keypoints_sequence = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            return False

        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, model)
        draw_styled_landmarks(image, results)

        # Progress bar for recording
        h, w = image.shape[:2]
        prog = frame_num / SEQUENCE_LENGTH
        cv2.rectangle(image, (10, h - 90), (w - 10, h - 75), (40, 40, 60), -1)
        cv2.rectangle(image, (10, h - 90), (10 + int((w - 20) * prog), h - 75), (0, 0, 220), -1)
        cv2.putText(image, f"RECORDING... {frame_num}/{SEQUENCE_LENGTH}", (15, h - 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

        # Blink indicator
        if (frame_num // 5) % 2 == 0:
            cv2.circle(image, (w - 30, 115), 10, (0, 0, 220), -1)

        cv2.imshow("ISL Data Collector", image)

        keypoints = extract_keypoints(results)
        keypoints_sequence.append(keypoints)
        frame_num += 1

        if frame_num >= SEQUENCE_LENGTH:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False

    # Save
    npy_path = os.path.join(sign_dir, f"{seq_index}.npy")
    np.save(npy_path, np.array(keypoints_sequence))
    return True


def main():
    args = parse_args()

    # Which signs to record
    if args.sign:
        signs_to_record = [args.sign] if args.sign in ISL_SIGNS else []
        if not signs_to_record:
            print(f"❌ Sign '{args.sign}' not in ISL_SIGNS list.")
            print(f"   Available: {ISL_SIGNS}")
            sys.exit(1)
    else:
        # Record all signs that are below target count
        signs_to_record = [
            s for s in ISL_SIGNS
            if count_existing_sequences(s) < args.sequences
        ]

    if not signs_to_record:
        print("✅ All signs already have enough sequences!")
        sys.exit(0)

    print("=" * 60)
    print("  ISL Webcam Data Collector")
    print("=" * 60)
    print(f"  Signs to record: {len(signs_to_record)}")
    print(f"  Sequences per sign: {args.sequences}")
    print(f"  Frames per sequence: {SEQUENCE_LENGTH}")
    print("\n  Controls: SPACE=Record | S=Skip | Q=Quit")
    print("=" * 60)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with get_holistic_model() as model:
        for sign_idx, sign in enumerate(signs_to_record):
            existing = count_existing_sequences(sign)
            need = max(0, args.sequences - existing)

            print(f"\n📌 Sign: {sign.upper()} — need {need} more sequences (have {existing})")

            seq_done = 0
            skip_sign = False

            while seq_done < need and not skip_sign:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, model)
                draw_styled_landmarks(image, results)

                draw_ui(image, sign, sign_idx, len(signs_to_record),
                        existing + seq_done, args.sequences, "READY")
                cv2.imshow("ISL Data Collector", image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n⏹ Quit by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                elif key == ord("s"):
                    print(f"  ⏭ Skipping '{sign}'")
                    skip_sign = True
                elif key == ord(" "):
                    # Countdown 3..2..1
                    for countdown in [3, 2, 1]:
                        t_end = time.time() + 1.0
                        while time.time() < t_end:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame = cv2.flip(frame, 1)
                            image, results = mediapipe_detection(frame, model)
                            draw_styled_landmarks(image, results)
                            draw_ui(image, sign, sign_idx, len(signs_to_record),
                                    existing + seq_done, args.sequences, "GET READY", countdown)
                            cv2.imshow("ISL Data Collector", image)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break

                    # Record
                    seq_index = get_next_sequence_index(sign)
                    success = record_sequence(cap, model, sign, seq_index)
                    if success:
                        seq_done += 1
                        print(f"  ✅ Sequence {existing + seq_done}/{args.sequences} saved")

                        # Show "SAVED" flash
                        for _ in range(15):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame = cv2.flip(frame, 1)
                            image, results = mediapipe_detection(frame, model)
                            draw_ui(image, sign, sign_idx, len(signs_to_record),
                                    existing + seq_done, args.sequences, "SAVED")
                            cv2.imshow("ISL Data Collector", image)
                            cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Data collection complete!")
    print("Next step: python model/train_model.py")


if __name__ == "__main__":
    main()
