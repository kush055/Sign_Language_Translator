import mediapipe as mp
import cv2
import numpy as np

#  Hand Detector Class using Mediapipe
class HandDetector:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.6):
        #  Initialize Mediapipe Hands
        self.max_num_hands = max_num_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    #Detect Hands and Draw Landmarks
    
    def find_hands(self, frame, draw=True):
        if frame is None or not isinstance(frame, np.ndarray):
            print(" Invalid frame input to find_hands()")
            return frame  # Always return the original frame

        # Convert to RGB for Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        # Draw landmarks if hands are detected
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        # Always return the processed frame
        return frame

    #  Extract Hand Landmark Features
    def extract_features(self, frame):
        if self.results is None or not self.results.multi_hand_landmarks:
            return None

        features = []
        for hand_lms in self.results.multi_hand_landmarks:
            for lm in hand_lms.landmark:
                features.extend([lm.x, lm.y, lm.z])

        # Pad feature vector to fixed size (126 values for 21 landmarks × 3 coords × 2 hands)
        while len(features) < 126:
            features.append(0.0)

        return np.array(features[:126])
