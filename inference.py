import cv2
import numpy as np
import mediapipe as mp
from keras import models

# Load trained model
model = models.load_model("posture_model.h5")
labels = np.load("labels.npy")

pose = mp.solutions.pose.Pose()
drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    lst = []
    ret, frm = cap.read()
    if not ret:
        break

    frm = cv2.flip(frm, 1)
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        landmarks = res.pose_landmarks.landmark

        base_x = landmarks[24].x
        base_y = landmarks[24].y

        for lm in landmarks:
            lst.append(lm.x - base_x)
            lst.append(lm.y - base_y)

        lst = np.array(lst).reshape(1, -1)
        pred = labels[np.argmax(model.predict(lst))]

        color = (0, 0, 255) if pred.lower() == "slouch" else (0, 255, 0)
        cv2.putText(frm, pred.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    drawing.draw_landmarks(frm, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    cv2.imshow("Posture Detection", frm)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
