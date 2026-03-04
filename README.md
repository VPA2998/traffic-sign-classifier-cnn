# 🚦 Traffic Sign Classifier (CNN + CLIP)

**Deep learning-based traffic sign recognition using a custom CNN architecture with CLIP-generated textual explanations.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Accuracy-93.4%25-orange)

---

## 📋 Overview

This project implements a **complete traffic sign classification pipeline** on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset:

- **Model**: Custom CNN architecture (Conv → ReLU → Pool → FC)
- **Training**: 15 epochs with Adam optimizer, achieving **93.4% test accuracy**
- **Data Augmentation**: Contrast, Gaussian noise, JPEG compression, motion blur, pixelation, spatter
- **Inference**: Real-time prediction with confidence scores
- **Explainability**: CLIP (Contrastive Language-Image Pre-training) for human-readable descriptions
- **Deployment**: Interactive Gradio web app

---

## 🎯 Key Features

### CNN Architecture
- **Input**: 32x32 RGB images
- **Feature Extractor**:
  - Conv2d(3→32, 3x3) + ReLU + MaxPool(2x2)
  - Conv2d(32→64, 3x3) + ReLU + MaxPool(2x2)
- **Classifier**:
  - Flatten → Linear(4096→128) + ReLU + Dropout(0.5)
  - Linear(128→43) (43 GTSRB classes)

### Data Augmentation
Robustness tested against **7 corruption types**:
- Contrast adjustment
- Gaussian noise
- Impulse noise
- JPEG compression
- Motion blur
- Pixelation
- Spatter (rain/mud)

### CLIP Integration
- **Model**: OpenAI CLIP ViT-B/32
- **Purpose**: Generates human-readable descriptions for predicted signs
- **Usage**: Post-hoc explanation only (not for classification)

---

## 📁 Project Structure

```text
traffic-sign-classifier-cnn/
├── notebooks/
│ └── 01_traffic_sign_classification_gtsrb.ipynb # Full pipeline
├── src/ # Modular Python packages
│ ├── init.py
│ ├── model.py # CNN architecture definition
│ ├── train.py # Training loop
│ ├── evaluate.py # Evaluation metrics
│ └── inference.py # Prediction + CLIP explanation
├── models/ # Saved model weights (.pth)
├── outputs/ # Generated plots, confusion matrices
│ └── demo/ # Demo images for README
├── data/ # Sample test images (optional)
├── docs/ # Architecture diagrams
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/VPA2998/traffic-sign-classifier-cnn.git
cd traffic-sign-classifier-cnn
python3 -m venv .venv
source .venv/bin/activate  # WSL/Linux
pip install -r requirements.txt
```
### 2. Run the Notebook
```bash
jupyter notebook
```

Open [`notebooks/01_traffic_sign_classification_gtsrb.ipynb`](notebooks/01_traffic_sign_classification_gtsrb.ipynb) and run all cells.

**What happens:**

1. ✅ Loads GTSRB dataset from Hugging Face

2. ✅ Applies preprocessing (resize to 32x32, normalize)

3. ✅ Trains CNN for 15 epochs (~5-10 min on GPU)

4. ✅ Evaluates on test set (expect ~93% accuracy)

5. ✅ Runs inference example with CLIP explanation

6. ✅ Launches Gradio demo for interactive testing

### 📊 Results

**Performance Metrics**

| Split    | Samples | Accuracy |
| -------- | ------- | -------- |
| Training | 10,000  | 92.1%    |
| Test     | 1,000   | 93.4%    |

**Training Progress**

| Epoch | Loss | Test Accuracy |
| ----- | ---- | ------------- |
| 1     | 2.62 | 51.9%         |
| 5     | 0.41 | 87.6%         |
| 10    | 0.19 | 92.4%         |
| 15    | 0.13 | 93.4%         |

**Example Predictions**

| Image | Prediction           | Confidence | CLIP Description                                   |
| ----- | -------------------- | ---------- | -------------------------------------------------- |
| 🚦    | Turn right mandatory | 98.2%      | "A white arrow turning right on a blue background" |
| 🛑    | STOP                 | 99.5%      | "A red octagon with white STOP text"               |
| ⚠️    | General danger       | 97.8%      | "A red triangle with black exclamation mark"       |
---
### 🎬 Interactive Demo

Launch the Gradio app to test the model with your own images:
```python
# In notebook, run the last cell:
gr.Interface(fn=predict_gradio, inputs=gr.Image(type="pil"), outputs="text")
```

**Features:**

- Upload any traffic sign image

- See CNN prediction + confidence score

- Get CLIP-generated textual description

- Test robustness with augmented images

