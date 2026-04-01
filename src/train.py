import tensorflow as tf
from tensorflow.keras import layers, models
import os

# =========================
# Path Fix (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # DigitAI/

models_dir = os.path.join(PROJECT_ROOT, "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "digit_model.h5")

# =========================
# Load dataset
# =========================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# =========================
# Build CNN
# =========================
model = models.Sequential([
    layers.Reshape((28,28,1), input_shape=(28,28)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# Train
# =========================
model.fit(x_train, y_train, epochs=5)

# =========================
# Save Model
# =========================
model.save(model_path)

print("✅ Model saved at:", model_path)