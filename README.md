# Braille-to-Text OCR Pipeline: Heuristic & Deep Learning Approaches

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange.svg)

## 📌 Project Overview
This project implements a fully functioning Optical Character Recognition (OCR) system for translating Braille images into English text. It was developed to explore the evolution of problem-solving in Artificial Intelligence, contrasting a pure mathematical/heuristic approach with a modern deep learning architecture.

The project is split into two distinct versions:
1. **Version 1:** A pure Computer Vision heuristic model using mathematical spacing and grid clustering.
2. **Version 2:** A Hybrid Machine Learning pipeline featuring a Convolutional Neural Network (CNN) trained on an auto-generated dataset, complete with a graphical user interface (GUI).

## ✨ Features
* **Scale & Skew Invariant:** Mathematically deskews and dynamically calculates grid sizes, making it immune to global column drift.
* **Heuristic Auto-Labeling:** Uses spatial algorithms to automatically crop and label thousands of training images from a single source image.
* **Robust Data Augmentation:** The CNN is trained with translation and zoom variance to read both sharp digital dots and blurry ink dots.
* **Interactive GUI:** Built with Tkinter, allowing users to upload images and see the neural network decode them in real-time.

## 🏗️ Architecture

### Version 1: Spatial Heuristics (OpenCV)
The V1 pipeline relies on strict spatial geometry. It uses OpenCV to threshold the image and extract connected components. Instead of a rigid global grid, it utilizes **Nearest-Neighbor Distance (NND)** arrays to group dots into horizontal text lines and vertical character blocks. It then maps the coordinates to a strict 2x3 matrix and queries a hardcoded Braille dictionary.

### Version 2: Hybrid CNN Pipeline (TensorFlow/Keras)
The V2 pipeline removes the hardcoded dictionary and replaces it with deep learning:
1. **Dataset Generation:** V1 spatial logic is repurposed to dynamically crop and sort Braille characters into labeled directories (`A-Z`).
2. **Model Training:** A lightweight CNN is trained on the generated dataset.
3. **Inference:** The spatial logic isolates the character bounding boxes, applying a **Fixed Anchor** mathematical crop to preserve the 2x3 aspect ratio. The boxes are resized to 32x32 pixels and batched into the CNN for instantaneous classification.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/braille-ocr.git](https://github.com/yourusername/braille-ocr.git)
   cd braille-ocr
   ```

2. **Install required dependencies:**
   ```bash
   pip install opencv-python numpy tensorflow pillow
   ```

## 💻 Usage Guide

### Running Version 1 (Heuristic)
Simply run the V1 script. It will process `braille.png` and output the text directly to the console.
```bash
python v1_heuristic.py
```

### Running Version 2 (Machine Learning)
**Step 1: Generate the Dataset**
```bash
python generate_dataset.py
```
*This extracts characters from the source image and populates the `dataset/` folder.*

**Step 2: Train the Neural Network**
```bash
python train_model.py
```
*This trains the CNN and saves the weights as `braille_cnn.keras`.*

**Step 3: Launch the GUI**
```bash
python app.py
```
*Opens the Tkinter interface. Load any Braille image (e.g., `test_braille.png`) and click "Decode with AI".*

## 🧠 Key Learnings & Bug Fixes
* **Domain Shift:** Resolved CNN overfitting by injecting `RandomZoom` and `RandomTranslation` layers, allowing the model to generalize across different dot thicknesses.
* **Aspect Ratio Distortion:** Fixed bounding box stretch by implementing an anchor-based crop, ensuring narrow letters (like 'A') maintain their correct spatial matrix when passed to the AI.
* **Blob Merging:** Calibrated array slicing logic to strictly measure intra-character dot gaps, preventing adjacent letters from merging during dynamic cropping.

## 👨‍💻 Author
**Muhammad Jibran** | *BS Software Engineering*
