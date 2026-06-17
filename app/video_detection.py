"""
app/video_detection.py
=======================
Process a pre-recorded video file and output:
  - Annotated video with sign overlays
  - Text transcript (.txt)
  - Per-frame prediction log (.csv)

Usage:
    python app/video_detection.py --input path/to/video.mp4
    python app/video_detection.py --input demo.mp4 --output output/annotated.mp4
    python app/video_detection.py --input demo.mp4 --threshold 0.65
"""

import os
import sys
import cv2
import json
import time
import argparse
import csv
import numpy as np
from collections import deque
from pathlib import Path
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SEQUENCE_LENGTH, NUM_FEATURES, MODEL_PATH, BASE_DIR
from utils.mediapipe_utils import (
    get_holistic_model, mediapipe_detection,
    extract_keypoints, draw_styled_landmarks,
)
from utils.visualization import draw_top_bar, draw_sentence_panel, draw_top3_predictions


def parse_args():
    p = argparse.ArgumentParser(description="ISL Video File Processor")
    p.add_argument("--input",     required=True,  type=str,   help="Input video path")
    p.add_argument("--output",    default=None,   type=str,   help="Output annotated video path")
    p.add_argument("--threshold", default=0.65,   type=float, help="Confidence threshold")
    p.add_argument("--cooldown",  default=1.0,    type=float, help="Min seconds between same-sign adds")
    p.add_argument("--no_save",   action="store_true",        help="Don't save output video (preview only)")
    return p.parse_args()


def load_model_and_labels() -> tuple:
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        sys.exit(1)

    label_map_path = os.path.join(BASE_DIR, "dataset", "label_map.json")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(label_map_path) as f:
        label_map = json.load(f)
    idx_to_sign = {v: k for k, v in label_map.items()}
    return model, idx_to_sign


def setup_video_writer(cap, output_path: str):
    """Create a VideoWriter matching input video properties."""
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h))


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input video not found: {args.input}")
        sys.exit(1)

    # ── Output paths ──────────────────────────────────────────
    input_stem = Path(args.input).stem
    output_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    if args.output is None:
        args.output = os.path.join(output_dir, f"{input_stem}_annotated.mp4")

    transcript_path = os.path.join(output_dir, f"{input_stem}_transcript.txt")
    log_path        = os.path.join(output_dir, f"{input_stem}_predictions.csv")

    print("=" * 60)
    print("  ISL Video Processor")
    print("=" * 60)
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output}")
    print(f"  Threshold: {args.threshold}")

    model, idx_to_sign = load_model_and_labels()

    cap = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None if args.no_save else setup_video_writer(cap, args.output)

    # ── State ──────────────────────────────────────────────────
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    sentence        = []
    sign_events     = []   # (frame_no, timestamp, sign, confidence)
    last_sign       = ""
    last_sign_time  = 0.0
    frame_no        = 0
    current_sign    = ""
    current_conf    = 0.0
    top3            = []

    print(f"\n  Processing {total_frames} frames @ {video_fps:.1f} FPS...\n")
    t_start = time.time()

    with get_holistic_model() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence_buffer.append(keypoints)

            timestamp = frame_no / video_fps

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                seq = np.array(sequence_buffer)
                seq_input = np.expand_dims(seq, 0)
                probs = model.predict(seq_input, verbose=0)[0]

                top3_idx = np.argsort(probs)[::-1][:3]
                top3 = [(idx_to_sign[i], float(probs[i])) for i in top3_idx]

                best_idx  = top3_idx[0]
                best_conf = float(probs[best_idx])
                best_sign = idx_to_sign[best_idx]

                current_sign = best_sign
                current_conf = best_conf

                if best_conf >= args.threshold:
                    add_it = True
                    if best_sign == last_sign and (timestamp - last_sign_time) < args.cooldown:
                        add_it = False
                    if sentence and sentence[-1] == best_sign:
                        add_it = False

                    if add_it:
                        sentence.append(best_sign)
                        sign_events.append((frame_no, round(timestamp, 2), best_sign, round(best_conf, 4)))
                        last_sign = best_sign
                        last_sign_time = timestamp

            # ── HUD ───────────────────────────────────────────
            fps_proc = frame_no / max(time.time() - t_start, 1e-5)
            draw_top_bar(image, current_sign, current_conf, fps_proc)
            draw_sentence_panel(image, sentence)
            if top3:
                draw_top3_predictions(image, top3)

            # Progress bar at bottom
            h, w = image.shape[:2]
            prog = frame_no / max(total_frames, 1)
            cv2.rectangle(image, (0, h - 6), (w, h), (30, 30, 50), -1)
            cv2.rectangle(image, (0, h - 6), (int(w * prog), h), (0, 200, 255), -1)

            if writer:
                writer.write(image)

            cv2.imshow("ISL Video Processor", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  ⏹ Stopped early by user.")
                break

            frame_no += 1

            if frame_no % 100 == 0:
                elapsed = time.time() - t_start
                pct = frame_no / max(total_frames, 1) * 100
                eta = (total_frames - frame_no) / max(fps_proc, 1)
                print(f"  [{pct:5.1f}%] Frame {frame_no}/{total_frames} | ETA: {eta:.0f}s")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Save transcript ───────────────────────────────────────
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("ISL Sign Language — Video Transcript\n")
        f.write("=" * 40 + "\n\n")
        f.write("SENTENCE:\n")
        f.write(" ".join(sentence) + "\n\n")
        f.write("SIGN EVENTS (frame | time | sign | confidence):\n")
        for ev in sign_events:
            f.write(f"  Frame {ev[0]:>5} | {ev[1]:>7.2f}s | {ev[2]:<20} | {ev[3]:.3f}\n")

    # ── Save prediction CSV log ───────────────────────────────
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame", "timestamp_s", "sign", "confidence"])
        writer_csv.writerows(sign_events)

    # ── Final Report ──────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("  ✅ Processing Complete")
    print("=" * 60)
    print(f"  Sentence  : {' '.join(sentence) if sentence else '(none detected)'}")
    print(f"  Signs     : {len(sign_events)}")
    print(f"  Time taken: {elapsed_total:.1f}s for {frame_no} frames")
    if not args.no_save:
        print(f"\n  📹 Annotated video : {args.output}")
    print(f"  📝 Transcript      : {transcript_path}")
    print(f"  📄 Prediction log  : {log_path}")


if __name__ == "__main__":
    main()
