# 🚀 DigitVision AI

A deep learning-based handwritten digit recognition system built using a Convolutional Neural Network (CNN).
The project supports both standard MNIST-style inputs and custom handwritten images through a robust preprocessing pipeline.

---

## 📌 Overview

DigitVision AI is designed to recognize handwritten digits (0–9) from images.
It uses a CNN trained on the MNIST dataset and includes preprocessing steps to handle real-world input variations such as scaling, noise, and misalignment.

---

## 🧠 Key Features

* 🔢 Handwritten digit classification (0–9)
* 🧠 CNN-based deep learning model
* 🖼 Custom image prediction support
* ⚙️ Advanced preprocessing:

  * Grayscale conversion
  * Noise reduction (thresholding)
  * Bounding box cropping
  * Resizing and padding (28×28)
* 📊 Probability output for predictions

---

## 🏗 Project Structure

```
DigitAI/
│
├── data/
│   └── digit.png              # Input image for prediction
│
├── models/
│   └── digit_model.h5         # Trained CNN model
│
├── src/
│   ├── train.py               # Model training script
│   └── predict.py             # Prediction script
│
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/MrHimanshu21L/digit-vision-ai.git
cd digit-vision-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1️⃣ Train the Model

```bash
python src/train.py
```

This will train the CNN on MNIST and save the model in the `models/` directory.

---

### 2️⃣ Predict a Digit

Place your image inside:

```
data/digit.png
```

Then run:

```bash
python src/predict.py
```

---

## 📊 Example Output

```
Predicted digit: 3
Probabilities: [[0.00001 ... 0.9997 ...]]
```

---

## 🧪 Model Architecture

* Input: 28×28 grayscale image
* Layers:

  * Conv2D (32 filters)
  * MaxPooling
  * Conv2D (64 filters)
  * MaxPooling
  * Dense (128)
  * Output (Softmax, 10 classes)

---

## 🧠 Technical Insights

* Model trained on MNIST dataset
* Uses normalization for stable training
* Preprocessing aligns custom images with MNIST distribution
* Addresses domain shift between real-world inputs and training data

---

## 🚀 Future Improvements

* GUI for drawing digits (mouse-based input)
* Web deployment (Flask / Streamlit)
* Fine-tuning on custom handwriting dataset
* Real-time digit recognition via webcam

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork this repo and submit pull requests.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Developed by Himanshu Kumar Gupta

---

## ⭐ If you found this useful

Give this repo a star ⭐ to support the project!
