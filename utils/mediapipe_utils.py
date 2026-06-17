"""
utils/mediapipe_utils.py
========================
MediaPipe Holistic keypoint extraction utilities.

Extracts 225 features per frame:
  - Left hand:  21 landmarks × 3 (x, y, z) = 63
  - Right hand: 21 landmarks × 3 (x, y, z) = 63
  - Pose:       33 landmarks × 3 (x, y, z) = 99
  Total = 225 features
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    # ─── MediaPipe Setup ──────────────────────────────────────────────────────
    mp_holistic       = mp.solutions.holistic
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    _MEDIAPIPE_OK = True
except ImportError:
    _MEDIAPIPE_OK = False
    mp = None
    mp_holistic = mp_drawing = mp_drawing_styles = None
    print("[WARNING] mediapipe not installed. Run: pip install mediapipe")


def get_holistic_model(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Return a configured MediaPipe Holistic model."""
    if not _MEDIAPIPE_OK:
        raise ImportError("mediapipe is not installed. Run: pip install mediapipe")
    return mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def mediapipe_detection(frame: np.ndarray, model) -> tuple:
    """
    Run MediaPipe Holistic on a BGR frame.

    Returns:
        (annotated_frame, results) where results contains landmarks.
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results) -> np.ndarray:
    """
    Extract and flatten all keypoints from MediaPipe results.

    Returns:
        np.ndarray of shape (225,)
        - pose:        99 values  (33 landmarks × 3)
        - left_hand:   63 values  (21 landmarks × 3)
        - right_hand:  63 values  (21 landmarks × 3)
    """
    pose = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 3)
    )
    left_hand = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    right_hand = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, left_hand, right_hand])  # (225,)


def draw_styled_landmarks(image: np.ndarray, results) -> None:
    """
    Draw styled MediaPipe landmarks on the image (in-place).
    - Pose connections (blue)
    - Left hand connections (yellow)
    - Right hand connections (green)
    """
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    # Left Hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    # Right Hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def hands_visible(results) -> bool:
    """Returns True if at least one hand is detected."""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
