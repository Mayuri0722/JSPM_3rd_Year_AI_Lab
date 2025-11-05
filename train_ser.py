import os, glob
import numpy as np
import librosa, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# --- Config ---
DATA_DIR = "data/ravdess"     # Path to RAVDESS folder
MODEL_PATH = "models/ser_svm.joblib"
SR = 22050                    # Sample rate
DUR = 3.0                     # Seconds per audio clip
N_MFCC = 40

# Emotion code mapping
EMO_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Extract features from audio
def extract_features(y, sr):
    target_len = int(SR * DUR)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1)

# Get emotion from filename
def parse_emotion_from_filename(path):
    base = os.path.basename(path).split(".")[0]
    parts = base.split("-")
    if len(parts) >= 3:
        code = parts[2]
        return EMO_MAP.get(code, None)
    return None

# Load data & extract features
def load_data():
    X, y = [], []
    wavs = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
    for p in wavs:
        label = parse_emotion_from_filename(p)
        if not label:
            continue
        audio, sr = librosa.load(p, sr=SR, mono=True)
        feats = extract_features(audio, sr)
        X.append(feats)
        y.append(label)
    return np.array(X), np.array(y)

# Train & save model
def main():
    print("Loading & featurizing audio...")
    X, y = load_data()
    print(f"Samples: {len(X)}, Features per sample: {X.shape[1]}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=5, gamma="scale", probability=True, random_state=42)),
    ])

    print("Training SVM...")
    clf.fit(Xtr, ytr)

    preds = clf.predict(Xte)
    acc = accuracy_score(yte, preds)
    print(f"Test Accuracy: {acc:.3f}")
    print(classification_report(yte, preds))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"pipeline": clf, "sr": SR, "dur": DUR, "n_mfcc": N_MFCC}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
