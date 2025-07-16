# 🗑️ Real-Time Waste Classification using MobileNetV2

Classify waste as **Organic** or **Recyclable** in real-time using a webcam feed. This project leverages a lightweight deep learning model (MobileNetV2) and PyTorch to enable fast, accurate image classification of waste for better environmental impact.

---

## 🚀 Project Overview

This project aims to automate waste sorting using deep learning. A MobileNetV2 model is trained to distinguish between **organic** and **recyclable** waste from image data. Once trained, the model is deployed via webcam for real-time inference.

---

## 🧠 Model Architecture

- **Base Model**: `MobileNetV2` (pre-trained on ImageNet)
- **Fine-tuned**: Only classifier layers
- **Output**: Binary classification — `0: Organic`, `1: Recyclable`

---

## 📂 Dataset Structure

Organize your dataset as follows:

```
Data/
├── train/
│   ├── Organic/
│   └── Recyclable/
└── test/
    ├── Organic/
    └── Recyclable/
```

Each folder contains images of the corresponding waste category.

---

## 🧼 Image Preprocessing

Images are resized and normalized to match MobileNetV2's requirements:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```

---

## 🏋️ Training

- Model trained using PyTorch with the **Adam optimizer**
- Loss function: `CrossEntropyLoss`
- Model checkpoint saved at: `../Model/mobilenet_waste_classifier.pth`

Training code is located in `Training.ipynb`.

---

## 📷 Real-Time Inference

Using `cv2.VideoCapture`, the trained model is deployed to classify images captured from the webcam in real-time.

Steps:
1. Load trained MobileNetV2 model
2. Capture frame from webcam
3. Preprocess and classify
4. Display classification label on screen

Code available in `ModelTest.ipynb`.

---

## 🧾 Required Libraries

Make sure you have the following Python packages installed:

```bash
pip install torch torchvision opencv-python matplotlib numpy
```

---

## 💡 Key Features

- ⚡ Fast and lightweight model
- 🔁 Real-time classification via webcam
- 🧠 Transfer learning for quick adaptation
- ✅ 224x224 image input size for compatibility

---

## 📌 How to Run

1. Clone this repository.
2. Place your dataset in the `Data/` folder.
3. Run `Training.ipynb` to train the model.
4. Run `ModelTest.ipynb` to launch real-time webcam classification.

---

## 🖼️ Demo

_A happy monkey classifying banana peels and plastic bottles!_ 🍌🧃

---

## 📚 Future Enhancements

- Multi-class waste classification
- Dataset expansion with more categories
- Deployment via mobile app or Raspberry Pi
