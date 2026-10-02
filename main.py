# main.py
import cv2
import logging
import os
from src.emotion_recognizer_sklearn import EmotionRecognizerSKL
from src.gesture_recognizer import GestureRecognizer
from src.posture_recognizer import PostureRecognizer
from src.db_logger import DatabaseLogger

# Configure logging for production-level monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- Paths to Models ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_classifier.pkl')

    # --- Initialize Database Logger ---
    db = DatabaseLogger(db_path=os.path.join(BASE_DIR, 'data', 'events.db'))

    # --- Initialize Detectors ---
    try:
        emotion_recognizer = EmotionRecognizerSKL(model_path=EMOTION_MODEL_PATH)
        logging.info("Emotion recognizer initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize emotion recognizer: {e}")
        return

    try:
        gesture_recognizer = GestureRecognizer()
        logging.info("Gesture recognizer initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize gesture recognizer: {e}")
        return
        
    try:
        posture_recognizer = PostureRecognizer()
        logging.info("Posture recognizer initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize posture recognizer: {e}")
        return

    # --- Initialize Webcam ---
    CAMERA_INDEX = 0
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logging.error(f"Error: Could not open webcam at index {CAMERA_INDEX}.")
        return

    WINDOW_NAME = 'Lightweight Emotion & Gesture Recognition'
    # Create window explicitly so we can track its properties
    cv2.namedWindow(WINDOW_NAME)
    
    logging.info("Starting video stream. Press 'q', 'ESC', or click 'X' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame from webcam. Exiting stream.")
                break

            frame = cv2.flip(frame, 1)

            # Default states
            emotion = "No Face Detected"
            gesture = "No Gesture"
            posture = "No Person Detected"

            # --- Perform Detections ---
            try:
                # 1. Detect emotions using the scikit-learn model
                frame, emotion = emotion_recognizer.detect_emotions(frame)
            except Exception as e:
                logging.error(f"Error processing emotion detection for frame: {e}")
                
            try:
                # 2. Recognize gestures
                frame, gesture = gesture_recognizer.recognize_gesture(frame)
            except Exception as e:
                logging.error(f"Error processing gesture detection for frame: {e}")

            try:
                # 3. Recognize posture
                frame, posture = posture_recognizer.recognize_posture(frame)
            except Exception as e:
                logging.error(f"Error processing posture detection for frame: {e}")

            # --- Database Logging ---
            db.log_if_changed(emotion, gesture, posture)

            # --- Display the resulting frame ---
            cv2.putText(frame, "Engine: Scikit-learn + MediaPipe", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)

            # --- Exit Conditions ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 27 is ESC key
                logging.info("Exit key pressed. Terminating...")
                break
                
            # Check if window was closed by the user (clicking 'X')
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                logging.info("Window closed by user. Terminating...")
                break
                
    except KeyboardInterrupt:
        logging.info("Application interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        logging.critical(f"An unexpected error occurred during execution: {e}")
    finally:
        # --- Cleanup ---
        logging.info("Cleaning up resources...")
        try:
            emotion_recognizer.close()
            if hasattr(gesture_recognizer, 'close'):
                gesture_recognizer.close()
            if hasattr(posture_recognizer, 'close'):
                posture_recognizer.close()
            db.close()
        except Exception as e:
            logging.error(f"Error during recognizer cleanup: {e}")
            
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Shutdown complete.")

if __name__ == '__main__':
    main()