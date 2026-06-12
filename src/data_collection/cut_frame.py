"""
Script test nhanh: cắt frame từ MỘT video duy nhất (dùng để thử nghiệm
trước khi chạy auto_dataset.py cho toàn bộ dataset).

Output: data/dataset_test/<video_name>_<index>.jpg
"""

import os
import cv2
import mediapipe as mp
import numpy as np

from src.utils.path_utils import get_base_dir

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

BASE_DIR = get_base_dir(__file__)

# Tên video cần test, đặt trong data/raw_videos/
VIDEO_NAME = "shoot2.mp4"

VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw_videos", VIDEO_NAME)
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "dataset_test")

MAX_FRAMES = 1200   # số ảnh tối đa muốn lấy ra
FRAME_SKIP = 5      # video dài -> bỏ qua nhiều frame hơn
THRESHOLD = 0.2     # ngưỡng thay đổi pose để giữ lại frame

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def get_pose_vector(result):
    """Trích vector (x, y, z) của 33 landmarks từ kết quả pose."""
    if not result.pose_landmarks:
        return None

    vec = []
    for lm in result.pose_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])

    return np.array(vec)


def pose_diff(vec1, vec2):
    """Khoảng cách Euclidean giữa 2 vector pose."""
    return np.linalg.norm(vec1 - vec2)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    prev_vec = None
    saved_count = 0
    frame_count = 0

    print("Đang xử lý:", VIDEO_PATH)

    while True:
        ret, frame = cap.read()
        if not ret or saved_count >= MAX_FRAMES:
            break

        # Giảm mật độ frame xử lý (video dài -> skip nhiều hơn)
        if frame_count % FRAME_SKIP != 0:
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
                video_basename = os.path.splitext(VIDEO_NAME)[0]
                file_name = os.path.join(
                    OUTPUT_FOLDER, f"{video_basename}_{saved_count}.jpg"
                )
                cv2.imwrite(file_name, frame)
                saved_count += 1
                prev_vec = vec

        frame_count += 1

    cap.release()
    print("Tổng ảnh:", saved_count)


if __name__ == "__main__":
    main()
