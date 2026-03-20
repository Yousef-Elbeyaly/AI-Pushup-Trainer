import cv2
import mediapipe as mp
import pandas as pd
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def process_folder(folder_path, label):
    output_data = []

    if not os.path.exists(folder_path):
        print("Folder not found")
        return []
    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)
        cap = cv2.VideoCapture(video_path)

        print(f"Processing video {video_name}")

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = []

                h, w, c = img.shape
                important_landmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

                x_vals = [lm.x * w for lm in results.pose_landmarks.landmark]
                y_vals = [lm.y * h for lm in results.pose_landmarks.landmark]

                base_x = x_vals[11]
                base_y = y_vals[11]
                scale = max(max(x_vals) - min(x_vals), max(y_vals) - min(y_vals))                # base_x = results.pose_landmarks.landmark[0].x
                # base_y = results.pose_landmarks.landmark[0].y
                if scale == 0 : scale = 1
                for i in important_landmarks:
                    lm = results.pose_landmarks.landmark[i]
                    norm_x = (lm.x * w - base_x) / scale
                    norm_y = (lm.y * h - base_y) / scale
                    landmarks.extend([norm_x, norm_y])

                landmarks.append(label)
                output_data.append(landmarks)

        cap.release()
    return output_data


data_correct = process_folder('AI-Trainer-Train-Videos/Correct sequence', 1)
data_wrong = process_folder('AI-Trainer-Train-Videos/Wrong sequence', 0)

full_data = data_correct + data_wrong
df = pd.DataFrame(full_data)
df.to_csv('pushups_dataset.csv', index = False)

print(f"Total frames : {len(df)}")