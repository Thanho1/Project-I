import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque

#  load model 
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

#  mediapipe 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

#  smoothing 
history = deque(maxlen=10)

#  màu theo class   
colors = {
    "dribbling": (255, 0, 0),
    "shooting": (0, 255, 0),
    "defense": (0, 0, 255),
    "Unknown": (0, 255, 255),
    "No Person": (200, 200, 200)
}

#  mở camera    
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

        # threshold 
        if confidence < 0.7:
            label = "Unknown"
        else:
            label = model.classes_[idx]

        history.append(label)

        #  smoothing 
        label = max(set(history), key=history.count)

        #  vẽ skeleton
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    else:
        label = "No Person"
        confidence = 0

    #  UI 
    color = colors.get(label, (255, 255, 255))

    # nền đen
    cv2.rectangle(frame, (0, 0), (400, 100), (0, 0, 0), -1)

    # label
    cv2.putText(frame, f"Action: {label}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2)

    # confidence
    cv2.putText(frame, f"Confidence: {confidence:.2f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

    cv2.imshow("Basketball AI Demo", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()