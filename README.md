# 🤟 Real-Time Indian Sign Language (ISL) Detector

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-lightgrey.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.0%25-success.svg)

A high-performance, real-time Indian Sign Language (ISL) detection system powered by **Google MediaPipe** and **Deep Learning (LSTM)**. This project detects static and dynamic hand gestures through a webcam with **99.00% accuracy** and translates them into text instantly.

## ✨ Features & Highlights
- **100% Custom Built Dataset:** The most unique aspect of this project is that the dataset was **created entirely from scratch** by the author. Instead of relying on generic pre-existing datasets (like Kaggle INCLUDE), thousands of frames were manually recorded, curated, and augmented to ensure zero domain-gap and maximum real-world reliability.
- **10 Practical ISL Gestures:** Trained specifically on 10 hand-picked, highly useful Indian Sign Language gestures.
- **Lightning Fast:** Uses MediaPipe to extract holistic keypoints (126 hand coordinates + 99 pose coordinates) resulting in incredibly fast inference times.
- **Deep Sequence Learning:** Uses a deep Long Short-Term Memory (LSTM) neural network architecture to process sequences of frames.
- **Data Augmentation:** Implements custom spatial scaling, temporal warping, and Gaussian noise augmentation for extreme robustness.

---

## 🎯 Supported Signs

The model currently recognizes the following 10 practical signs seamlessly:
1. 👍 **Thumbs Up** (Yes / Good)
2. 👎 **Thumbs Down** (No / Bad)
3. ✋ **Open Palm** (Stop / Wait)
4. ✊ **Closed Hand** (Solid / Hold)
5. ✌️ **Victory** (Win / Peace)
6. 👌 **OK** (Perfect / Understood)
7. 🤟 **Awesome** (Super / Rock On)
8. 🤙 **Call Me**
9. 👉 **Right**
10. 👈 **Left**

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ISL-Realtime-Detector.git
cd ISL-Realtime-Detector
```

### 2. Install Dependencies
Make sure you have Python 3.9+ installed. Run the following command to install required libraries:
```bash
pip install -r requirements.txt
```

### 3. Run the Real-Time App
Run the detection script. This will open your webcam and start predicting gestures in real-time.
```bash
python app/realtime_detection.py
```
*(Press `Q` to quit the application.)*

---

## 🧠 Model Architecture & Training

The model was trained for **135 epochs** (with early stopping) and achieved a flawless validation test accuracy of **99.00%**.

- **Feature Extraction:** MediaPipe Holistic is used to extract `(30, 225)` shape sequences.
- **Network:** 
  - `LSTM(64) -> Dropout(0.2) -> LSTM(128) -> Dropout(0.2) -> LSTM(64)`
  - `Dense(64) -> Dense(32) -> Softmax(10)`
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy

*(If you wish to train your own custom signs, you can use the `data_collection/collect_data.py` and `model/train_model.py` scripts provided in the source code.)*

---

## 👨‍💻 Deployment & Future Scope
The next step for this project is to package it into a standalone **`.exe` desktop application** using PyInstaller, so that anyone can download and run the software without writing a single line of code or installing Python.

Feel free to fork this project and add more complex dynamic sentences to the dataset!
