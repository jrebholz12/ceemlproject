import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# --- SETTINGS ---
WAV_FILE = "cough.wav"  # Replace with your custom .wav
MODEL_PATH = "cnn_model_v2.keras"
CLASS_INDEX_PATH = "class_indices_v2.txt"
SPEC_IMG = "custom_spec.png"

# --- STEP 1: Convert wav to spectrogram ---
y, sr = librosa.load(WAV_FILE, sr=22050)
y = librosa.util.fix_length(y, size=sr*3)  # 3 seconds

mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
log_mel = librosa.power_to_db(mel)
log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())

plt.imsave(SPEC_IMG, log_mel, cmap='gray')

# --- STEP 2: Load and prepare image ---
img = Image.open(SPEC_IMG).convert("L").resize((128, 128))
img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 128, 128, 1)

# --- STEP 3: Load model and class mapping ---
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = eval(f.read())
labels = list(class_indices.keys())

# --- STEP 4: Predict Top 3 ---
predictions = model.predict(img_array)[0]  # Get raw array of predictions
top_3_indices = predictions.argsort()[-3:][::-1]  # Indices of top 3 predictions, descending

print("🔊 Top 3 Predictions:")
for idx in top_3_indices:
    label = labels[idx]
    confidence = predictions[idx] * 100
    print(f" - {label}: {confidence:.2f}%")
