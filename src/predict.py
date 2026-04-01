import os
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_image

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

print("✅ Model loaded successfully\n")

# =========================
# Loop through all images
# =========================
for file in os.listdir(data_folder):

    if file.lower().endswith((".png", ".jpg", ".jpeg")):

        img_path = os.path.join(data_folder, file)

        try:
            # Preprocess using utils
            img = preprocess_image(img_path)

            # Prediction
            prediction = model.predict(img, verbose=0)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)

            print(f"📁 {file} → Predicted: {digit} (Confidence: {confidence:.4f})")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
