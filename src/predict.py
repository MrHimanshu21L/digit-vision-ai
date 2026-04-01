import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# =========================
# Path Setup (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

model_path = os.path.join(PROJECT_ROOT, "models", "digit_model.h5")
image_path = os.path.join(PROJECT_ROOT, "data", "digit.png")

# =========================
# Load Model
# =========================
model = load_model(model_path)

# =========================
# Load Image
# =========================
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ Error: Image not found at", image_path)
    exit()

# =========================
# Preprocessing (MNIST-style)
# =========================

# Invert colors
img = 255 - img

# Threshold (remove noise)
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

# Find bounding box
coords = cv2.findNonZero(img)
x, y, w, h = cv2.boundingRect(coords)

# Crop digit
img = img[y:y+h, x:x+w]

# Resize to 20x20
img = cv2.resize(img, (20, 20))

# Pad to 28x28
img = np.pad(img, ((4,4),(4,4)), mode='constant')

# Normalize
img = img / 255.0

# Reshape for model
img = img.reshape(1, 28, 28)

# =========================
# Prediction
# =========================
prediction = model.predict(img)
digit = np.argmax(prediction)

# =========================
# Output
# =========================
print("✅ Predicted digit:", digit)
print("📊 Probabilities:", prediction)