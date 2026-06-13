"""
Demo nhận diện hành động bóng rổ v2 (position + velocity features)
qua webcam.

Chạy:
    python -m src.inference.demo_webcam_v2

Nhấn ESC để thoát.
"""

import os
from collections import deque

import cv2
import mediapipe as mp

from src.utils.path_utils import get_base_dir
from src.utils.inference_utils import (
    load_model,
    pose_to_vector_with_velocity,
    predict_action,
    draw_skeleton,
)

BASE_DIR = get_base_dir(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model_v2.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_v2.pkl")

CONFIDENCE_THRESHOLD = 0.7
HISTORY_SIZE = 10

COLORS = {
    "dribbling": (255, 0, 0),
    "shooting": (0, 255, 0),
    "defense": (0, 0, 255),
    "idle": (255, 255, 0),
    "Unknown": (0, 255, 255),
    "No Person": (200, 200, 200),
}

mp_pose = mp.solutions.pose


def main():
    model, scaler = load_model(MODEL_PATH, SCALER_PATH)

    pose = mp_pose.Pose()
    history = deque(maxlen=HISTORY_SIZE)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    prev_vec = None  # position vector của frame trước (132 chiều)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        label = "No Person"
        confidence = 0

        if result.pose_landmarks:
            full_vec, current_vec = pose_to_vector_with_velocity(
                result.pose_landmarks.landmark, prev_vec
            )
            prev_vec = current_vec

            label, confidence = predict_action(
                full_vec, model, scaler, threshold=CONFIDENCE_THRESHOLD
            )

            history.append(label)
            label = max(set(history), key=history.count)

            draw_skeleton(frame, result.pose_landmarks)
        else:
            prev_vec = None  # mất pose -> reset velocity

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

        cv2.imshow("Basketball AI Demo v2", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
