"""
Tự động cắt frame từ video gốc trong data/raw_videos/ để tạo dataset.

Với mỗi video, lấy đều ~TARGET_FRAMES frame, chỉ giữ lại frame có
sự thay đổi pose đủ lớn so với frame trước (tránh frame trùng/thừa)
và có độ visibility của landmark đầu tiên (mũi) > 0.7.

Output: data/dataset/<label>/<video_name>_<index>.jpg
"""

import os
import cv2
import mediapipe as mp
import numpy as np

from src.utils.path_utils import get_base_dir

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

BASE_DIR = get_base_dir(__file__)

INPUT_ROOT = os.path.join(BASE_DIR, "data", "raw_videos")
OUTPUT_ROOT = os.path.join(BASE_DIR, "data", "dataset")

TARGET_FRAMES = 120  # số frame mục tiêu lấy ra cho mỗi video
THRESHOLD = 0.35      # ngưỡng thay đổi pose để giữ lại frame


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


def process_video(video_path, output_folder, video_name):
    """Xử lý 1 video: cắt frame và lưu vào output_folder."""
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // TARGET_FRAMES)

    saved_count = 0
    frame_count = 0
    prev_vec = None

    while True:
        ret, frame = cap.read()

        if not ret or saved_count >= TARGET_FRAMES:
            break

        if frame_count % step != 0:
            frame_count += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        vec = get_pose_vector(result)

        if (
            vec is not None
            and result.pose_landmarks.landmark[0].visibility > 0.7
        ):
            if prev_vec is None:
                save = True
            else:
                diff = pose_diff(vec, prev_vec)
                save = diff > THRESHOLD

            if save:
                filename = f"{video_name}_{saved_count}.jpg"
                filepath = os.path.join(output_folder, filename)
                cv2.imwrite(filepath, frame)

                saved_count += 1
                prev_vec = vec

        frame_count += 1

    cap.release()
    return saved_count


def main():
    for label in os.listdir(INPUT_ROOT):
        input_folder = os.path.join(INPUT_ROOT, label)
        output_folder = os.path.join(OUTPUT_ROOT, label)

        if not os.path.isdir(input_folder):
            continue

        os.makedirs(output_folder, exist_ok=True)

        print(f"\n===== CLASS: {label} =====")

        for video_name in os.listdir(input_folder):
            video_path = os.path.join(input_folder, video_name)

            print("Đang xử lý:", video_path)

            saved_count = process_video(video_path, output_folder, video_name)

            print(f"-> {video_name}: {saved_count} ảnh")

    print("\nDONE toàn bộ dataset!")


if __name__ == "__main__":
    main()
