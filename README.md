# Emotion and Gesture Recognition

A real-time Python-based system for recognizing human **facial emotions** and **hand gestures** using a webcam feed. The system uses OpenCV for face detection, MediaPipe for hand detection, and a deep learning model for emotion classification.

---

## 📁 Project Structure

```plaintext
emotion_gesture_recognition/
│
├── data/
|     └──fer2013.csv                     # sample images/videos
├── models/                             # place your trained models here
│     └── emotion_classifier.pkl
|     └── haarcascade_fontalface_default.xml
├── src/
│   ├── emotion_recognizer_detector.py
│   ├── gesture_recognizer_detector.py
│   ├── __init__.py
│
|── main.py
├── requirements.txt
|
└── README.md
````

---

## 🚀 Features

* Real-time webcam-based emotion recognition
* Real-time hand gesture detection
* well-structured code
* Simple to extend for more classes or advanced gestures
* Cross-platform (Windows, Linux, Mac)

---

## 🔧 Requirements

* Python 3.11+
* OpenCV
* MediaPipe
* PyTorch
* numpy
* pandas 
* pickle
* scikit-learn 

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python main.py
```

**Press q to exit the window.**

---

## 🧩 Models

* The emotion detection model can be trained on **FER-2013** or similar datasets


* The gesture detector uses MediaPipe Hands and does not require a custom model by default.

---

## 📝 Notes

* MediaPipe sometimes tries to import TensorFlow via its “tasks” module.
  → In this project, only `mediapipe.solutions.hands` is used to avoid TensorFlow dependency issues.

* If you see errors related to `pywrap_tensorflow`, remove MediaPipe “tasks” imports and stick to `mediapipe.solutions` only.

## 📌 License

This project is for educational and research use.

---

**Happy Coding!** 🚀


