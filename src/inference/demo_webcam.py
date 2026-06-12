"""
Demo nhận diện hành động bóng rổ qua webcam (OpenCV window).

Chạy:
    python -m src.inference.demo_webcam

Nhấn ESC để thoát.
"""

import os
from collections import deque

import cv2
import mediapipe as mp

from src.utils.path_utils import get_base_dir
from src.utils.inference_utils import (
    load_model,
    pose_to_vector,
    predict_action,
    draw_skeleton,
)

BASE_DIR = get_base_dir(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

CONFIDENCE_THRESHOLD = 0.7
HISTORY_SIZE = 10

# Màu hiển thị theo từng class (BGR)
COLORS = {
    "dribbling": (255, 0, 0),
    "shooting": (0, 255, 0),
    "defense": (0, 0, 255),
    "Unknown": (0, 255, 255),
    "No Person": (200, 200, 200),
}

mp_pose = mp.solutions.pose


def main():
    model, scaler = load_model(MODEL_PATH, SCALER_PATH)

    pose = mp_pose.Pose()
    history = deque(maxlen=HISTORY_SIZE)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # width
    cap.set(4, 720)   # height

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        label = "No Person"
        confidence = 0

        if result.pose_landmarks:
            vec = pose_to_vector(result.pose_landmarks.landmark)
            label, confidence = predict_action(
                vec, model, scaler, threshold=CONFIDENCE_THRESHOLD
            )

            history.append(label)
            label = max(set(history), key=history.count)

            draw_skeleton(frame, result.pose_landmarks)

        # --- UI ---
        color = COLORS.get(label, (255, 255, 255))

        cv2.rectangle(frame, (0, 0), (400, 100), (0, 0, 0), -1)

        cv2.putText(
            frame, f"Action: {label}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, color, 2
        )

        cv2.putText(
            frame, f"Confidence: {confidence:.2f}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )

        cv2.imshow("Basketball AI Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
