"""
Các hàm dùng chung cho inference (demo_webcam.py và app.py):
- load model + scaler đã train
- chuyển landmarks pose thành vector đặc trưng
- dự đoán hành động (action) từ vector
- vẽ skeleton lên frame
"""

import pickle
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def load_model(model_path, scaler_path):
    """Load model SVM và scaler đã được train sẵn."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def pose_to_vector(landmarks):
    """Chuyển 33 landmarks pose (x, y, z, visibility) thành vector 1D."""
    vec = []
    for lm in landmarks:
        vec.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(vec)


def predict_action(vec, model, scaler, threshold=0.7):
    """
    Dự đoán hành động từ vector đặc trưng.

    Trả về:
        label (str): tên hành động hoặc "Unknown" nếu confidence thấp
        confidence (float): độ tin cậy của dự đoán
    """
    X = scaler.transform(vec.reshape(1, -1))

    probs = model.predict_proba(X)[0]
    idx = np.argmax(probs)
    confidence = probs[idx]

    if confidence < threshold:
        label = "Unknown"
    else:
        label = model.classes_[idx]

    return label, confidence


def draw_skeleton(frame, pose_landmarks):
    """Vẽ skeleton (landmarks + connections) lên frame."""
    mp_drawing.draw_landmarks(
        frame,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    return frame
