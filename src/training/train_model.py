import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#  Correct Path Setup Based on Your Structure
# This gets the absolute path of your project root (SLT)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Data directory where your CSV files are stored
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

# Models directory where .pkl files will be saved
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


#  Data Loading Function
def load_data():
    files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {os.path.abspath(DATA_PATH)}")

    print(f" Found {len(files)} CSV file(s) in {DATA_PATH}")
    dfs = [pd.read_csv(f, header=None) for f in files]
    data = pd.concat(dfs, ignore_index=True)

    X = data.iloc[:, :-1].astype(np.float32)
    y = data.iloc[:, -1].astype(str)
    print(f" Data loaded successfully | Samples: {len(data)}, Features: {X.shape[1]}")
    return X, y


#  Training and Saving Model
if __name__ == "__main__":
    print(" Loading data...")
    X, y = load_data()

    print(" Encoding labels...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(" Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    print(" Training model...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=25, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f" Model trained successfully | Accuracy: {acc*100:.2f}%")

    # Save model and label encoder
    joblib.dump(clf, os.path.join(MODEL_DIR, "sign_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print("Model & label encoder saved in /models/")
