# EmotiSense AI: Real-Time Emotion, Gesture & Posture Recognition

A production-ready, real-time Python system for recognizing human **facial emotions**, **hand gestures**, and **body posture** using a webcam feed. The system uses OpenCV for video capture, MediaPipe for landmark detection, and a trained Scikit-learn model for emotion classification.

---

## 🚀 Key Features

* **Real-time Webcam Processing:** Seamless tracking of facial, hand, and body movements.
* **Emotion Recognition:** Uses MediaPipe Face Mesh and a pre-trained SVM model for multi-class emotion classification (e.g., Happy, Sad, Angry).
* **Gesture Recognition:** Uses an efficient rule-based logic to classify hand gestures (e.g., Peace, Fist). Includes a rolling history buffer to eliminate jitter and false positives.
* **Posture Detection:** Tracks shoulder and nose alignment via MediaPipe Pose to warn users if they are slouching.
* **Automated Activity Logging:** Includes a lightweight SQLite database (`events.db`) that intelligently records changes in your emotion, gesture, and posture over time.
* **Production-Ready Architecture:** 
  * Robust `try...except` handling prevents single-frame processing errors from crashing the application.
  * Comprehensive `logging` instead of basic print statements for server-level monitoring.
  * Bulletproof resource cleanup and proper window closing mechanics.

---

## 📁 Project Structure

```plaintext
emotion_gesture_recognition/
│
├── data/                               # Dataset & Database directory
│     └── events.db                     # SQLite database logging activity
├── models/                             # Directory for trained models
│     └── emotion_classifier.pkl        # Pre-trained Scikit-learn SVM model
├── src/                                # Core logic modules
│   ├── emotion_recognizer_sklearn.py   # Emotion detection
│   ├── gesture_recognizer.py           # Gesture classification
│   ├── posture_recognizer.py           # Posture detection
│   ├── db_logger.py                    # SQLite database manager
│   └── __init__.py
│
├── main.py                             # Main execution script and stream handler
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 🔧 Requirements

* **Python 3.11+**
* OpenCV (`opencv-python`, `opencv-contrib-python`)
* MediaPipe (`mediapipe`)
* Numpy, Pandas, Scikit-learn (`scikit-learn`)

Install all dependencies via pip:
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Simply execute the main script:

```bash
python main.py
```
> **Controls:** Press `q` or `ESC`, or simply click the `X` button on the webcam window to gracefully exit the application.

---

## 🧠 How It Works Under the Hood

1. **Camera Feed:** `main.py` opens a video stream using `cv2.VideoCapture`.
2. **Smoothing Algorithm:** All detectors (Emotion, Gesture, Posture) use a sliding window (history buffer). They store the predictions of the last N frames and calculate the statistical mode, guaranteeing stable, flicker-free output.
3. **Emotion Pipeline:** `emotion_recognizer_sklearn.py` extracts facial landmarks and feeds normalized distances into the loaded `emotion_classifier.pkl` SVM model.
4. **Gesture Pipeline:** `gesture_recognizer.py` mathematically evaluates if fingers are extended by comparing fingertip y-coordinates to their corresponding knuckles.
5. **Posture Pipeline:** `posture_recognizer.py` measures the vertical distance between the user's nose and shoulders. If the nose drops too close to the shoulder line, it triggers a slouching warning.
6. **Smart Logging:** `db_logger.py` monitors the combined state of the user. To prevent database bloat, it only inserts a new row into `events.db` when the user's state changes.

---

## 📌 License

This project is intended for educational, research, and under the [MIT LICENSE](https://github.com/BipronathSaha12/EmotiSense-AI/blob/main/LICENSE).

---

