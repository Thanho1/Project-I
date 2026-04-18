import cv2
import os
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

video_path = "shoot2.mp4"
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

saved_count = 0
MAX_FRAMES = 1200   # 🎯 mục tiêu

def get_pose_vector(result):
    if not result.pose_landmarks:
        return None

    vec = []
    for lm in result.pose_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec)

def pose_diff(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

THRESHOLD = 0.2  # chỉnh nếu cần

cap = cv2.VideoCapture(video_path)
prev_vec = None

print("Đang xử lý:", video_path)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret or saved_count >= MAX_FRAMES:
        break

    # 🔥 giảm mật độ frame (video dài → skip nhiều hơn)
    if frame_count % 5 != 0:
        frame_count += 1
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    vec = get_pose_vector(result)

    if vec is not None:
        if prev_vec is None:
            save = True
        else:
            diff = pose_diff(vec, prev_vec)
            save = diff > THRESHOLD

        if save:
            file_name = f"{output_folder}/shoot2_{saved_count}.jpg"
            cv2.imwrite(file_name, frame)
            saved_count += 1
            prev_vec = vec

    frame_count += 1

cap.release()

print("Tổng ảnh:", saved_count)