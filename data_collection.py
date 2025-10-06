import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

name = input("Enter the posture name (e.g. upright / slouch): ")

pose = mp.solutions.pose.Pose()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

print("\n[INFO] Press 'ESC' to stop or wait till 100 samples are captured.\n")

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

        # Take x, y coordinates of main posture joints relative to mid-hip
        base_x = landmarks[24].x  # right hip
        base_y = landmarks[24].y

        for lm in landmarks:
            lst.append(lm.x - base_x)
            lst.append(lm.y - base_y)

        X.append(lst)
        data_size += 1

    drawing.draw_landmarks(frm, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    cv2.putText(frm, f"Samples: {data_size}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Data Collection - Posture", frm)

    if cv2.waitKey(1) == 27 or data_size >= 200:
        break

cap.release()
cv2.destroyAllWindows()

np.save(f"{name}.npy", np.array(X))
print(f"\nData saved as {name}.npy with shape {np.array(X).shape}")
