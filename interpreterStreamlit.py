#to run this app on localost, on terminal run : streamlit run "path_to_interpreterStreamlit.py_file"

import streamlit as st 
import mediapipe as mp
import cv2
import numpy as np
import pickle
import pandas as pd

label_encoder_path = "C:/Users/KIIT/Kodessa/Untitled Folder/label_encoder_final.pickle"
model_path = "C:/Users/KIIT/Kodessa/Untitled Folder/signLangModel_final.pickle"

try:
    with open(label_encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)
except Exception as e:
    st.error(f"Error loading label encoder: {e}")
    label_encoder = None

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading trained model: {e}")
    model = None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

st.title('Real-time Sign Language Interpreter')

frame_placeholder = st.empty()

start_button = st.button("Interpret")

cap = None

if start_button:
    frame_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    stop_button = st.button("Stop Interpreting")
    def process_frame(frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                x_lmrk = [landmark.x for landmark in landmarks]
                y_lmrk = [landmark.y for landmark in landmarks]
                features = []
                for landmark in landmarks:
                    features.append(landmark.x - min(x_lmrk))
                    features.append(landmark.y - min(y_lmrk))

                features = np.array(features).reshape(1, -1)

                prediction = model.predict(features)
                predicted_class = label_encoder.inverse_transform(prediction)[0]

                img_h, img_w, _ = frame.shape
                x_min = int(min(x_lmrk) * img_w)
                y_min = int(min(y_lmrk) * img_h)
                x_max = int(max(x_lmrk) * img_w)
                y_max = int(max(y_lmrk) * img_h)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f'Predicted: {predicted_class}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return frame

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture video")
                break

            frame = process_frame(frame)
            frame_placeholder.image(frame, channels='BGR')

            # Check if the Stop button is pressed after displaying each frame
            if stop_button:
                break

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
