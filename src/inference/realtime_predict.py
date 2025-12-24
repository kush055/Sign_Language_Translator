import cv2
import time
import pyttsx3
import numpy as np
import joblib
import os
import sys

# FIX IMPORT PATH — So we can run this script inside "src"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.hand_detector import HandDetector
from utils.gesture_model import GestureModel

#  PATH SETUP — Match Project Structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(BASE_DIR, "models")

#  Load Model and Encoder
print(" Loading model and label encoder...")

model_path = os.path.join(MODEL_DIR, "sign_model.pkl")
encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError(" Model or label encoder not found. Please train your model first.")

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

#  Initialize Hand Detector and Gesture Model
detector = HandDetector(max_num_hands=2)
gesture_model = GestureModel(model=model, label_encoder=label_encoder)

#  Text-to-Speech Setup
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

#  Variables
sentence = ""
last_prediction = ""
last_prediction_time = 0
PREDICTION_COOLDOWN = 2.0  # seconds
prev_frame_time = 0
new_frame_time = 0

#  Initialize Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError(" Could not access webcam. Check your camera permissions.")

print(" Webcam started — press 'q' to quit, 'c' to clear sentence, 's' to speak.")

#  Main Loop
while True:
    success, frame = cap.read()
    if not success or frame is None:
        print(" Frame capture failed — check camera.")
        time.sleep(0.5)
        continue

    # Flip the frame for mirror view
    frame = cv2.flip(frame, 1)

    #  FPS Calculation
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
    prev_frame_time = new_frame_time

    #  Detect Hands and Extract Features
    try:
        frame = detector.find_hands(frame, draw=True)
    except Exception as e:
        print(f" Hand detection error: {e}")
        continue

    if frame is None or not isinstance(frame, np.ndarray):
        print(" Invalid frame returned from hand detector.")
        continue

    try:
        features = detector.extract_features(frame)
    except Exception as e:
        print(f" Feature extraction error: {e}")
        continue

    #  Predict Gesture
    if features is not None and len(features) > 0:
        current_time = time.time()
        if current_time - last_prediction_time >= PREDICTION_COOLDOWN:
            try:
                pred = gesture_model.predict(features.reshape(1, -1))
                if pred and pred != last_prediction:
                    sentence += pred + " "
                    last_prediction = pred
                    last_prediction_time = current_time
                    print(f" Predicted: {pred}")
            except Exception as e:
                print(f" Prediction error: {e}")

    #  Display Frame
    try:
        cv2.putText(frame, f"Prediction: {last_prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {sentence}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(" Sign Language Translator", frame)
    except cv2.error as e:
        print(f"Display error: {e}")
        continue

    #  Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
        last_prediction = ""
        print(" Sentence cleared.")
    elif key == ord('s'):
        text = sentence.strip()
        if text:
            engine.say(text)
            engine.runAndWait()
            print(f" Speaking: {text}")

#Cleanup
cap.release()
cv2.destroyAllWindows()
print(" Exiting...")
