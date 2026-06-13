"""
Trích xuất pose keypoints + velocity (chênh lệch pose giữa 2 ảnh liên
tiếp cùng video) từ các ảnh trong data/dataset/<label>/.

Mỗi ảnh được đặt tên dạng: <video_name>_<index>.jpg
(do auto_dataset.py sinh ra). Các ảnh cùng <video_name> được sort theo
<index> để xác định thứ tự thời gian trong video, từ đó tính velocity
= pose hiện tại - pose trước đó (cùng video). Ảnh đầu tiên của mỗi
video có velocity = 0.

Output: data/data_v2.csv với mỗi dòng gồm:
    [position: x0,y0,z0,v0 ... x32,y32,z32,v32]   (132 giá trị)
    [velocity: dx0,dy0,dz0,dv0 ... dx32,dy32,dz32,dv32]  (132 giá trị)
    label
=> tổng 265 cột (264 feature + 1 label)
"""

import os
import re
import csv
import cv2
import mediapipe as mp
import numpy as np

from src.utils.path_utils import get_base_dir

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

BASE_DIR = get_base_dir(__file__)

DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "data_v2.csv")

LABELS = ["dribbling", "shooting", "defense", "idle"]

# Tên file dạng: <video_name>_<index>.jpg -> tách video_name và index
FILENAME_PATTERN = re.compile(r"^(.*)_(\d+)\.jpg$")


def build_header():
    header = []
    for i in range(33):
        header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
    for i in range(33):
        header += [f"dx{i}", f"dy{i}", f"dz{i}", f"dv{i}"]
    header.append("label")
    return header


def parse_filename(filename):
    """Tách (video_name, index) từ tên file 'video_name_123.jpg'."""
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None, None
    video_name, index = match.groups()
    return video_name, int(index)


def get_pose_vector(result):
    """Vector pose 132 chiều: x,y,z,visibility cho 33 landmarks."""
    if not result.pose_landmarks:
        return None

    vec = []
    for lm in result.pose_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(vec)


def main():
    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_header())

        total = 0
        skipped = 0

        for label in LABELS:
            folder = os.path.join(DATASET_PATH, label)

            if not os.path.isdir(folder):
                print(f"Bỏ qua (không tìm thấy folder): {folder}")
                continue

            print("Đang xử lý:", label)

            # Group các file theo video_name, sort theo index
            groups = {}
            for filename in os.listdir(folder):
                video_name, index = parse_filename(filename)
                if video_name is None:
                    continue
                groups.setdefault(video_name, []).append((index, filename))

            for video_name, items in groups.items():
                items.sort(key=lambda x: x[0])  # sort theo index

                prev_vec = None

                for index, filename in items:
                    path = os.path.join(folder, filename)

                    image = cv2.imread(path)
                    if image is None:
                        skipped += 1
                        continue

                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)

                    vec = get_pose_vector(result)

                    if vec is None:
                        skipped += 1
                        continue

                    if prev_vec is None:
                        velocity = np.zeros_like(vec)
                    else:
                        velocity = vec - prev_vec

                    row = list(vec) + list(velocity) + [label]
                    writer.writerow(row)

                    prev_vec = vec
                    total += 1

    print("Done!")
    print("Số sample:", total)
    print("Bị bỏ qua:", skipped)


if __name__ == "__main__":
    main()
