import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set paths
AUDIO_DIR = "ESC-50-master/audio"
META_PATH = "ESC-50-master/meta/esc50.csv"
SPEC_OUTPUT_DIR = "spectrograms"

# Create spectrogram folder
os.makedirs(SPEC_OUTPUT_DIR, exist_ok=True)

# Load metadata
meta = pd.read_csv(META_PATH)

tabular_features = []
labels = []

for idx, row in meta.iterrows():
    filename = row['filename']
    label = row['category']
    class_dir = os.path.join(SPEC_OUTPUT_DIR, label)
    os.makedirs(class_dir, exist_ok=True)
    
    filepath = os.path.join(AUDIO_DIR, filename)
    try:
        # Load and pad/trim audio
        y, sr = librosa.load(filepath, sr=22050)
        y, _ = librosa.effects.trim(y)
        max_len = sr * 3
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        else:
            y = y[:max_len]

        # --- Spectrogram for CNN ---
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())

        spec_filename = os.path.join(class_dir, filename.replace(".wav", ".png"))
        plt.imsave(spec_filename, log_mel, cmap='gray')

        # --- Tabular features for traditional ML models ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1)
        ])
        tabular_features.append(features)
        labels.append(label)

        print(f"[{idx+1}/2000] Processed: {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save tabular features
df = pd.DataFrame(tabular_features)
df['label'] = labels
df.to_csv("esc50_features.csv", index=False)
print("✅ Done! Saved tabular features to esc50_features.csv")
