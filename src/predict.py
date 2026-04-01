import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

model_path = os.path.join(PROJECT_ROOT, "models", "digit_model.h5")
data_folder = os.path.join(PROJECT_ROOT, "data")

# =========================
# Load Model
# =========================
model = load_model(model_path)

# =========================
# Loop through all images
# =========================
for file in os.listdir(data_folder):

    if file.endswith(".png") or file.endswith(".jpg"):

        img_path = os.path.join(data_folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Skipping {file} (not readable)")
            continue

        # Preprocessing
        img = 255 - img
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

        img = cv2.resize(img, (20, 20))
        img = np.pad(img, ((4,4),(4,4)), mode='constant')

        img = img / 255.0
        img = img.reshape(1, 28, 28)

        # Prediction
        prediction = model.predict(img)
        digit = np.argmax(prediction)

        print(f"📁 {file} → Predicted: {digit}")
