import cv2
import os
import csv
import sys


# Auto Path Fix (ensures imports always work)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(BASE_DIR)

from src.utils.hand_detector import HandDetector

#  Path Configuration
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
IMG_DIR = os.path.join(BASE_DIR, "data", "images")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

#  Initialize Hand Detector
detector = HandDetector(max_num_hands=2)
samples_per_class = 300  # You can adjust this as needed

#  Gesture Recording Function
def record_gesture(label):
    print(f"\n Recording gesture: {label}")
    print("Press 's' to start recording, 'q' to quit early.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: Cannot access webcam.")
        return

    started, recorded = False, 0
    collected = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame)
        features = detector.extract_features(frame)

        # Display info
        cv2.putText(frame, f"Label: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {recorded}/{samples_per_class}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow("Recording", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            started = True
            print(" Recording started...")
        elif key == ord('q'):
            print(" Recording stopped early.")
            break

        if started and features is not None:
            collected.append(features.tolist() + [label])
            recorded += 1

            # Save snapshots
            if recorded % 50 == 0:
                img_path = os.path.join(IMG_DIR, f"{label}_{recorded}.jpg")
                cv2.imwrite(img_path, frame)
                print(f" Saved snapshot: {img_path}")

            if recorded >= samples_per_class:
                print(f" {samples_per_class} samples collected for {label}")
                break

    cap.release()
    cv2.destroyAllWindows()

    # Save to CSV
    csv_path = os.path.join(DATA_DIR, f"{label}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(collected)
    print(f" Saved {len(collected)} samples to {csv_path}")

#  Main Interactive Loop
if __name__ == "__main__":
    while True:
        label = input("\nEnter gesture label (A–Z, 0–9, or word): ").strip().upper()
        if not label:
            continue
        record_gesture(label)
        cont = input("Add another gesture? (y/n): ").strip().lower()
        if cont != 'y':
            print(" Exiting gesture recording.")
            break
