import cv2
import mediapipe as mp
import numpy as np
import PoseModule as om
from tensorflow.keras.models import load_model
import os
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
model = load_model('models/pushups_model.h5')

cap = cv2.VideoCapture('AI-Trainer-Test-Videos/1.mp4')
detector = om.poseDetector()
count = 0
dir = 0
frame_count = 0
is_correct = True
prob = 0
form_history = [True]
pTime = 0

while True:
    success, img = cap.read()
    if not success: break

    h_orig, w_orig, _ = img.shape
    #img = cv2.resize(img, (int(w_orig * 0.7), int(h_orig * 0.7)))
    h, w, c = img.shape
    #img = cv2.flip(img, 1)
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        for id in [11, 12, 23, 24]:
            cx, cy = lmList[id][1], lmList[id][2]
            cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 8, (0, 255, 0), 2)

        important_landmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        x_vals = [lm.x * w for lm in detector.results.pose_landmarks.landmark]
        y_vals = [lm.y * h for lm in detector.results.pose_landmarks.landmark]

        base_x, base_y = x_vals[11], y_vals[11]
        scale = max(max(x_vals) - min(x_vals), max(y_vals) - min(y_vals))
        if scale == 0: scale = 1

        current_frame_data = []
        for i in important_landmarks:
            lm = detector.results.pose_landmarks.landmark[i]
            current_frame_data.extend([(lm.x * w - base_x) / scale, (lm.y * h - base_y) / scale])

        angle = detector.findAngle(img, 12, 14, 16, False)
        per = np.interp(angle, (80, 160), (100, 0))
        bar = np.interp(per, (0, 100), (h - 50, h - 150))

        frame_count += 1
        if frame_count % 3 == 0:
            prediction = model.predict(np.array([current_frame_data]), verbose=0)
            prob = prediction[0][0]
            is_correct = prob > 0.6
            if per > 30: form_history.append(is_correct)
            if len(form_history) > 3: form_history.pop(0)

        final_predict = (sum(form_history) / len(form_history)) > 0.4 if len(form_history) > 0 else True

        if per >= 80:
            if dir == 0:
                dir = 1
                if final_predict: count += 0.5
        elif per <= 10:
            if dir == 1:
                dir = 0
                if final_predict: count += 0.5
                form_history = []

        cv2.rectangle(img, (20, h - 150), (50, h - 50), (255, 255, 255), 3)
        cv2.rectangle(img, (20, int(bar)), (50, h - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (10, h - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.rectangle(img, (-3, -3), (73, 73), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(img, (0, 0), (70, 70), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (20, 52), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        if not final_predict:
            cv2.putText(img, "FIX FORM!", (int(w / 2) - 150, h - 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)

        status_text = "Correct Form" if final_predict else "Wrong Form"
        status_color = (0, 255, 0) if final_predict else (0, 0, 255)


        cv2.putText(img, f'Status: {status_text}', (int(w / 2) - 200, 40), cv2.FONT_HERSHEY_PLAIN, 2, status_color,
                    3)
        cv2.putText(img, f'Confidence: {int(prob * 100)}%', (int(w / 2) - 120, 80), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    status_color, 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (w - 100, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

    cv2.imshow("Smart Push-up Trainer", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()