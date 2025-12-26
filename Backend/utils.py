import librosa
import numpy as np

def extract_features(audio_path, max_len=200):
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # ❌ Very short or empty audio (common in tap-to-record)
        if y is None or len(y) < sr:
            raise ValueError("Audio too short")

        # Normalize
        y = librosa.util.normalize(y)

        # ===== Feature extraction =====
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        # Stack → (62, T)
        features = np.vstack([mfcc, delta, delta2, zcr, rms])

        # ===== Pad / Truncate =====
        if features.shape[1] < max_len:
            pad_width = max_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)))
        else:
            features = features[:, :max_len]

        # ===== Normalize per feature =====
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True) + 1e-6
        features = (features - mean) / std

        # Final shape → (200, 62)
        return features.T.astype(np.float32)

    except Exception as e:
        print("❌ Feature extraction failed:", e)
        return None
