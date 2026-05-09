import cv2
import mediapipe as mp
import numpy as np
import pickle
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque
import time
from datetime import datetime
import os

# tạo folder lưu video
os.makedirs("records", exist_ok=True)

# load model
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# smoothing
history = deque(maxlen=15)

# màu
colors = {
    "dribbling": (255, 0, 0),
    "shooting": (0, 255, 0),
    "defense": (0, 0, 255),
    "idle": (255, 255, 0),
    "Unknown": (200, 200, 200),
    "No Person": (150, 150, 150)
}

# UI
window = tk.Tk()
window.title("Basketball AI Pro")
window.geometry("1100x700")
window.configure(bg="#121212")

main_frame = tk.Frame(window, bg="#121212")
main_frame.pack(fill="both", expand=True)

sidebar = tk.Frame(main_frame, bg="#1f1f1f", width=200)
sidebar.pack(side="left", fill="y")

title = tk.Label(sidebar, text="AI APP",
                 font=("Arial", 16, "bold"),
                 fg="white", bg="#1f1f1f")
title.pack(pady=20)

status_label = tk.Label(sidebar, text="Status: OFF",
                        fg="red", bg="#1f1f1f",
                        font=("Arial", 12))
status_label.pack(pady=10)

video_frame = tk.Frame(main_frame, bg="black")
video_frame.pack(side="right", fill="both", expand=True)

video_label = tk.Label(video_frame)
video_label.pack(fill="both", expand=True)

info_frame = tk.Frame(video_frame, bg="#121212")
info_frame.pack(fill="x")

action_label = tk.Label(info_frame, text="Action: None",
                        font=("Arial", 16, "bold"),
                        fg="white", bg="#121212")
action_label.pack(side="left", padx=20)

conf_label = tk.Label(info_frame, text="Confidence: 0",
                      font=("Arial", 12),
                      fg="gray", bg="#121212")
conf_label.pack(side="left", padx=20)

fps_label = tk.Label(info_frame, text="FPS: 0",
                     font=("Arial", 12),
                     fg="gray", bg="#121212")
fps_label.pack(side="right", padx=20)

running = False
cap = None
prev_time = 0

# record
recording = False
video_writer = None
current_filename = None

def start():
    global running, cap
    if not running:
        cap = cv2.VideoCapture(0)
        cap.set(3, 960)
        cap.set(4, 720)
        running = True
        status_label.config(text="Status: ON", fg="green")
        update_frame()

def stop():
    global running, cap
    running = False
    if cap:
        cap.release()
    status_label.config(text="Status: OFF", fg="red")

def toggle_record():
    global recording, video_writer, current_filename

    if not recording:
        recording = True
        video_writer = None

        current_filename = datetime.now().strftime("records/record_%Y%m%d_%H%M%S.avi")
        print("Recording:", current_filename)

        record_btn.config(text="STOP REC", bg="#ff4444")
    else:
        recording = False
        if video_writer:
            video_writer.release()
            video_writer = None

        print("Saved:", current_filename)
        record_btn.config(text="RECORD", bg="#007bff")

def create_button(text, cmd, color):
    return tk.Button(sidebar, text=text,
                     command=cmd,
                     font=("Arial", 12),
                     bg=color, fg="white",
                     activebackground=color,
                     width=15, height=2,
                     bd=0)

create_button("START", start, "#28a745").pack(pady=10)
create_button("STOP", stop, "#dc3545").pack(pady=10)
record_btn = create_button("RECORD", toggle_record, "#007bff")
record_btn.pack(pady=10)

def update_frame():
    global running, prev_time, video_writer

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    label = "No Person"
    confidence = 0

    if result.pose_landmarks:
        vec = []
        for lm in result.pose_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z, lm.visibility])

        X = np.array(vec).reshape(1, -1)
        X = scaler.transform(X)

        probs = model.predict_proba(X)[0]
        idx = np.argmax(probs)
        confidence = probs[idx]

        if confidence < 0.75:
            label = "Unknown"
        else:
            label = model.classes_[idx]

        history.append(label)
        label = max(set(history), key=history.count)

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # scale chuẩn
    frame_h, frame_w = frame.shape[:2]
    target_w = video_frame.winfo_width()
    target_h = video_frame.winfo_height()

    if target_w > 0 and target_h > 0:
        scale = min(target_w / frame_w, target_h / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        frame = canvas

    # overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (450, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    color = colors.get(label, (255, 255, 255))

    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, f"Conf: {confidence:.2f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    action_label.config(text=f"Action: {label}")
    conf_label.config(text=f"Confidence: {confidence:.2f}")
    fps_label.config(text=f"FPS: {int(fps)}")

    # record (không ghi đè nữa)
    if recording:
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(
                current_filename,
                fourcc,
                20.0,
                (frame.shape[1], frame.shape[0])
            )
        video_writer.write(frame)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10, update_frame)

window.mainloop()