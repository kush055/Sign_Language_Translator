import os
import cv2
import numpy as np
import pandas as pd
from src.utils.hand_detector import HandDetector

SAVE_DIR = os.path.join("data", "raw")
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    label = input("Enter label for gesture (e.g., hello): ").strip()
    num_samples = int(input("How many samples to capture? (e.g., 300): "))

    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2)
    samples = []
    count = 0

    print(f"Collecting data for '{label}' ... press 'q' to quit.")

    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        landmarks = detector.extract_features(img)

        if landmarks is not None:
            samples.append(np.append(landmarks, label))
            count += 1
            cv2.putText(img, f"Samples: {count}/{num_samples}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Data Collection", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()

    if samples:
        df = pd.DataFrame(samples)
        df.to_csv(os.path.join(SAVE_DIR, f"{label}.csv"), index=False, header=False)
        print(f"Saved {len(samples)} samples for '{label}'.")
    else:
        print("No samples captured.")

if __name__ == "__main__":
    main()
