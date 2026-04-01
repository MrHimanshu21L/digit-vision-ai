import cv2
import numpy as np

def preprocess_image(img_path):
    """
    Load and preprocess image to MNIST format
    """

    # Load grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    # Invert colors (MNIST style)
    img = 255 - img

    # Threshold (Otsu)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find bounding box
    coords = cv2.findNonZero(img)
    if coords is None:
        raise ValueError("No digit found in image")

    x, y, w, h = cv2.boundingRect(coords)

    # Crop digit
    img = img[y:y+h, x:x+w]

    # Resize to 20x20
    img = cv2.resize(img, (20, 20))

    # Pad to 28x28
    img = np.pad(img, ((4,4),(4,4)), mode='constant')

    # ===== Centering (IMPORTANT) =====
    coords = np.column_stack(np.where(img > 0))
    if len(coords) > 0:
        center = coords.mean(axis=0)
        shift_x = int(14 - center[1])
        shift_y = int(14 - center[0])

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (28, 28))

    # Normalize
    img = img / 255.0

    # Reshape
    img = img.reshape(1, 28, 28)

    return img