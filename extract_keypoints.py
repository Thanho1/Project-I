import cv2
import os
import mediapipe as mp
import csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

dataset_path = "dataset"
output_file = "data.csv"

labels = ["dribbling", "shooting", "defense"]

with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)

    # 👉 tạo header
    header = []
    for i in range(33):
        header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
    header.append("label")
    writer.writerow(header)

    total = 0
    skipped = 0

    for label in labels:
        folder = os.path.join(dataset_path, label)
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