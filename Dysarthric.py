# =========================================================
# DYSARTHRIA DETECTION - FINAL STABLE IMPLEMENTATION
# TORGO + Personal / Indian Normal Voices
# =========================================================

import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Model
import kagglehub

# =========================================================
# GLOBAL PATH
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# STEP 1: DOWNLOAD TORGO DATASET
# =========================================================
print("Downloading TORGO dataset...")
path = kagglehub.dataset_download("iamhungundji/dysarthria-detection")
torgo_dir = os.path.join(path, "torgo_data")

data = pd.read_csv(os.path.join(torgo_dir, "data.csv"))
data["filename"] = data["filename"].apply(lambda x: os.path.join(path, x))

print("TORGO samples:", len(data))

# =========================================================
# STEP 2: FEATURE EXTRACTION (FIXED)
# =========================================================
def extract_features(audio_path, max_len=200):
    y, sr = librosa.load(audio_path, sr=16000)

    # Normalize loudness (do NOT trim silence aggressively)
    y = librosa.util.normalize(y)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Prosodic features
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # Stack features → 62 total
    features = np.vstack([mfcc, delta, delta2, zcr, rms])  # (62, T)

    # Pad or trim time axis
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])))
    else:
        features = features[:, :max_len]

    # Normalize per feature (IMPORTANT)
    features = (features - features.mean(axis=1, keepdims=True)) / \
               (features.std(axis=1, keepdims=True) + 1e-6)

    return features.T  # (200, 62)

# =========================================================
# FEATURE GROUP INDICES (FOR EXPLAINABILITY)
# =========================================================
FEATURE_GROUPS = {
    "mfcc": slice(0, 20),
    "delta": slice(20, 40),
    "delta2": slice(40, 60),
    "zcr": slice(60, 61),
    "rms": slice(61, 62)
}


# =========================================================
# STEP 3: LOAD TORGO FEATURES
# =========================================================
X, y = [], []

for _, row in tqdm(data.iterrows(), total=len(data)):
    try:
        feat = extract_features(row["filename"])
        X.append(feat)
        y.append(1 if row["is_dysarthria"] == "dysarthria" else 0)
    except Exception as e:
        print("Skipped:", row["filename"], e)

X = np.array(X)
y = np.array(y)

print("TORGO feature shape:", X.shape)

# =========================================================
# STEP 4: LOAD EXTRA NORMAL VOICES
# =========================================================
NORMAL_FOLDER = os.path.join(BASE_DIR, "Norma_voice")

def load_extra_normal(folder):
    Xn, yn = [], []
    for f in os.listdir(folder):
        if f.lower().endswith(".wav"):
            try:
                feat = extract_features(os.path.join(folder, f))
                Xn.append(feat)
                yn.append(0)
            except Exception as e:
                print("Skipped normal:", f, e)
    return np.array(Xn), np.array(yn)

if os.path.exists(NORMAL_FOLDER):
    Xn, yn = load_extra_normal(NORMAL_FOLDER)
    X = np.concatenate([X, Xn])
    y = np.concatenate([y, yn])
    print("Added normal samples:", len(Xn))
else:
    print("⚠ Norma_voice folder not found")

# =========================================================
# STEP 5: TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================================================
# STEP 6: CLASS WEIGHTS
# =========================================================
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}

# =========================================================
# STEP 7: CNN MODEL (INPUT = 62 FEATURES)
# =========================================================
inputs = layers.Input(shape=(200, 62))

x = layers.Conv1D(64, 5, activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2)(x)

x = layers.Conv1D(128, 3, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2)(x)

x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC()]
)

model.summary()

# =========================================================
# STEP 8: TRAIN MODEL
# =========================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    )
]

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=8,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save(os.path.join(BASE_DIR, "dysarthria_model.keras"))
# =========================================================
# COMPUTE NORMAL FEATURE BASELINE (FOR XAI)
# =========================================================
normal_features = X_train[y_train == 0]
NORMAL_BASELINE = {
    "mean": np.mean(normal_features, axis=(0, 1)),
    "std": np.std(normal_features, axis=(0, 1)) + 1e-6
}
np.savez(
    os.path.join(BASE_DIR, "normal_baseline.npz"),
    mean=NORMAL_BASELINE["mean"],
    std=NORMAL_BASELINE["std"]
)

print("✅ normal_baseline.npz saved")



# =========================================================
# STEP 9: PREDICTION FUNCTION
# =========================================================
# =========================================================
# STEP 9: PREDICTION + EXPLAINABLE AI
# =========================================================
def predict_audio(audio_path):
    feat = extract_features(audio_path)
    feat_batch = np.expand_dims(feat, axis=0)

    prob = model.predict(feat_batch)[0][0]
    label = "DYSARTHRIC" if prob > 0.5 else "NORMAL"

    explanation, reasons = explain_prediction(feat, prob)

    print("\nFile:", audio_path)
    print("Probability:", round(prob, 3))
    print("Prediction:", label)
    print("Feature Analysis:", explanation)
    print("Reasons:", reasons)

    return {
        "prediction": label,
        "confidence": round(prob * 100, 2),
        "feature_analysis": explanation,
        "reasons": reasons
    }

# =========================================================
# EXPLAINABLE AI - FEATURE CONTRIBUTION
# =========================================================
# =========================================================
# EXPLAINABLE AI - CORRECTED VERSION
# =========================================================
def explain_prediction(features, prob):
    explanation = {}
    reasons = []

    for group, idx in FEATURE_GROUPS.items():
        values = features[:, idx]

        deviation = np.mean(
            np.abs((values - NORMAL_BASELINE["mean"][idx]) /
                   NORMAL_BASELINE["std"][idx])
        )

        if deviation > 1.2:
            explanation[group] = "High deviation"
        elif deviation > 0.7:
            explanation[group] = "Moderate deviation"
        else:
            explanation[group] = "Normal"

    # 🔑 IMPORTANT: Conditional reasoning
    if prob > 0.6:
        if explanation["mfcc"] != "Normal":
            reasons.append("Articulation clarity affected")

        if explanation["zcr"] != "Normal":
            reasons.append("Irregular speech pauses")

        if explanation["rms"] != "Normal":
            reasons.append("Abnormal loudness control")

    if not reasons:
        reasons.append("Speech patterns within healthy range")

    return explanation, reasons



# =========================================================
# STEP 10: TEST FILES
# =========================================================
# =========================================================
# STEP 10: TEST FILES (UPDATED PATHS)
# =========================================================

predict_audio(os.path.join(BASE_DIR, "DYSARTHIC.wav.wav"))
predict_audio(os.path.join(BASE_DIR, "NORMAL.wav.wav"))
predict_audio(os.path.join(BASE_DIR, "Norma_voice", "Mine.wav.wav"))

for f in os.listdir(os.path.join(BASE_DIR, "Norma_voice")):
    if f.endswith(".wav"):
        predict_audio(os.path.join(BASE_DIR, "Norma_voice", f))