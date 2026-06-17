"""
utils/visualization.py
======================
OpenCV-based visualization utilities for the ISL recognition system.
Renders prediction overlays, confidence bars, sentence panel, etc.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

# ─── Color Palette ────────────────────────────────────────────────────────────
COLOR_BG_DARK   = (15, 15, 30)
COLOR_PRIMARY   = (0, 200, 255)      # Cyan
COLOR_SUCCESS   = (0, 230, 120)      # Green
COLOR_WARNING   = (0, 165, 255)      # Orange
COLOR_DANGER    = (60, 60, 220)      # Red
COLOR_WHITE     = (255, 255, 255)
COLOR_GRAY      = (120, 120, 140)
COLOR_PANEL_BG  = (25, 25, 50)
COLOR_PANEL_BOR = (60, 60, 100)


def draw_top_bar(
    frame: np.ndarray,
    current_sign: str,
    confidence: float,
    fps: float,
) -> None:
    """
    Draw top HUD bar with current sign, confidence, and FPS.
    """
    h, w = frame.shape[:2]

    # Semi-transparent top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Border line
    cv2.line(frame, (0, 90), (w, 90), COLOR_PANEL_BOR, 2)

    # Title
    cv2.putText(frame, "ISL Sign Language Detector", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PRIMARY, 2)

    # FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GRAY, 1)

    # Current sign (large)
    sign_display = current_sign.upper() if current_sign else "---"
    cv2.putText(frame, sign_display, (15, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLOR_SUCCESS, 3)

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = w - 220, 45, 200, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_PANEL_BOR, -1)
    fill_w = int(bar_w * confidence)
    bar_color = COLOR_SUCCESS if confidence > 0.8 else (COLOR_WARNING if confidence > 0.5 else COLOR_DANGER)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"{confidence*100:.0f}%", (bar_x + bar_w + 8, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    cv2.putText(frame, "Confidence", (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)


def draw_sentence_panel(
    frame: np.ndarray,
    sentence: List[str],
    max_words: int = 8,
) -> None:
    """
    Draw sentence accumulation panel at the bottom.
    """
    h, w = frame.shape[:2]
    panel_h = 70

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.line(frame, (0, h - panel_h), (w, h - panel_h), COLOR_PANEL_BOR, 2)

    cv2.putText(frame, "Sentence:", (15, h - panel_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)

    words = sentence[-max_words:]
    sentence_text = " ".join(words)
    cv2.putText(frame, sentence_text, (15, h - panel_h + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_WHITE, 2)


def draw_top3_predictions(
    frame: np.ndarray,
    predictions: List[Tuple[str, float]],
) -> None:
    """
    Draw top-3 sign predictions with confidence bars on the right side.
    predictions: [(sign_name, confidence), ...]
    """
    h, w = frame.shape[:2]
    panel_w = 200
    start_x = w - panel_w - 10
    start_y = 100

    cv2.putText(frame, "Top Predictions:", (start_x, start_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1)

    for i, (sign, conf) in enumerate(predictions[:3]):
        y = start_y + 25 + i * 40
        # Label
        cv2.putText(frame, sign, (start_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
        # Bar
        bar_y = y + 5
        bar_w = panel_w - 10
        cv2.rectangle(frame, (start_x, bar_y), (start_x + bar_w, bar_y + 12), COLOR_PANEL_BOR, -1)
        fill = int(bar_w * conf)
        colors = [COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER]
        cv2.rectangle(frame, (start_x, bar_y), (start_x + fill, bar_y + 12), colors[i], -1)
        cv2.putText(frame, f"{conf*100:.0f}%", (start_x + bar_w + 4, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)


def draw_sign_history(
    frame: np.ndarray,
    history: List[str],
    max_show: int = 5,
) -> None:
    """
    Draw recent sign history on the left side.
    """
    h, w = frame.shape[:2]
    start_y = 110

    cv2.putText(frame, "History:", (15, start_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1)

    recent = history[-max_show:]
    for i, sign in enumerate(reversed(recent)):
        alpha = 1.0 - (i * 0.18)
        color = tuple(int(c * alpha) for c in COLOR_WHITE)
        cv2.putText(frame, f"  {sign}", (15, start_y + 20 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_recording_indicator(frame: np.ndarray, recording: bool, frame_count: int, total_frames: int) -> None:
    """Show a recording indicator / progress bar while collecting a sequence."""
    h, w = frame.shape[:2]
    if recording:
        # Blinking red circle
        if (frame_count // 5) % 2 == 0:
            cv2.circle(frame, (w - 40, 110), 12, (0, 0, 220), -1)
        cv2.putText(frame, "REC", (w - 80, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)

        # Progress bar
        prog = frame_count / max(total_frames, 1)
        bar_x = 10
        bar_y = h - 80
        bar_w = w - 20
        bar_h = 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_PANEL_BOR, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * prog), bar_y + bar_h), COLOR_DANGER, -1)


def draw_countdown(frame: np.ndarray, seconds: int, label: str = "") -> None:
    """Overlay a large countdown number on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, str(seconds), (w // 2 - 30, h // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5, COLOR_PRIMARY, 6)
    if label:
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(frame, label, ((w - text_size[0]) // 2, h // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)


def draw_no_hand_warning(frame: np.ndarray) -> None:
    """Show a warning when no hands are detected."""
    h, w = frame.shape[:2]
    cv2.putText(frame, "⚠ No hands detected", (15, h - 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WARNING, 2)
