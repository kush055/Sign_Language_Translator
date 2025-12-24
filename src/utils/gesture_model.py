import numpy as np
import joblib

class GestureModel:
    def __init__(self, model=None, label_encoder=None, model_path=None, encoder_path=None):
        """
        Initialize GestureModel with either:
        - Preloaded model & label_encoder (objects), OR
        - Paths to load them from disk
        """
        if model is not None and label_encoder is not None:
            self.model = model
            self.label_encoder = label_encoder
        elif model_path and encoder_path:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
        else:
            raise ValueError(" Provide either model objects or file paths.")

    def extract_features(self, hands):
        """Extract landmarks or embeddings from hand detections."""
        if not hands:
            return None
        hand = hands[0]  # You can adjust if you want both hands
        landmarks = np.array(hand["lmList"]).flatten()
        return landmarks.reshape(1, -1)

    def predict(self, features):
        """Predict the gesture from extracted features."""
        try:
            prediction = self.model.predict(features)
            return self.label_encoder.inverse_transform(prediction)[0]
        except Exception as e:
            print(f" Prediction error: {e}")
            return None
