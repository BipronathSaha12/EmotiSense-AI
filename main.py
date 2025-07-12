# main.py
import cv2
from src.emotion_recognizer_sklearn import EmotionRecognizerSKL
from src.gesture_recognizer import GestureRecognizer # This remains the same

def main():
    # --- Paths to Models ---
    EMOTION_MODEL_PATH = 'models/emotion_classifier.pkl'

    # --- Initialize Detectors ---
    emotion_recognizer = EmotionRecognizerSKL(model_path=EMOTION_MODEL_PATH)
    gesture_recognizer = GestureRecognizer()

    # --- Initialize Webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # --- Perform Detections ---
        # 1. Detect emotions using the scikit-learn model
        frame = emotion_recognizer.detect_emotions(frame)
        
        # 2. Recognize gestures
        frame, gesture = gesture_recognizer.recognize_gesture(frame)

        # --- Display the resulting frame ---
        cv2.putText(frame, "Engine: Scikit-learn", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Lightweight Emotion & Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    emotion_recognizer.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()