"""
Trích xuất pose keypoints từ các ảnh trong data/dataset/<label>/
và lưu thành bảng dữ liệu data/data.csv để dùng cho training.

Mỗi dòng trong data.csv gồm:
    x0, y0, z0, v0, x1, y1, z1, v1, ..., x32, y32, z32, v32, label
(33 landmarks x 4 giá trị + 1 cột label)
"""

import os
import csv
import cv2
import mediapipe as mp

from src.utils.path_utils import get_base_dir

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

BASE_DIR = get_base_dir(__file__)

DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "data.csv")

LABELS = ["dribbling", "shooting", "defense", "idle"]


def build_header():
    header = []
    for i in range(33):
        header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
    header.append("label")
    return header


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

            for file in os.listdir(folder):
                path = os.path.join(folder, file)

                image = cv2.imread(path)
                if image is None:
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    row = []
                    for lm in result.pose_landmarks.landmark:
                        row += [lm.x, lm.y, lm.z, lm.visibility]

                    row.append(label)
                    writer.writerow(row)
                    total += 1
                else:
                    skipped += 1

    print("Done!")
    print("Số sample:", total)
    print("Bị bỏ qua:", skipped)


if __name__ == "__main__":
    main()
