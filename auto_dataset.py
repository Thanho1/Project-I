import cv2
import os
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

input_root = r"D:\Prj 1\Project I\videos"
output_root = r"D:\Prj 1\Project I\dataset"

TARGET_FRAMES = 120
THRESHOLD = 0.35


def get_pose_vector(result):

    if not result.pose_landmarks:
        return None

    vec = []

    for lm in result.pose_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])

    return np.array(vec)


def pose_diff(vec1, vec2):

    return np.linalg.norm(vec1 - vec2)


for label in os.listdir(input_root):

    input_folder = os.path.join(input_root, label)
    output_folder = os.path.join(output_root, label)

    os.makedirs(output_folder, exist_ok=True)

    print(f"\n===== CLASS: {label} =====")

    for video_name in os.listdir(input_folder):

        video_path = os.path.join(input_folder, video_name)

        print("Đang xử lý:", video_path)

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

        print(f"→ {video_name}: {saved_count} ảnh")

print("\nDONE toàn bộ dataset!")