import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from statistics import mode, StatisticsError

class GestureRecognizer:
    def __init__(self, history_len=15):
        """
        Initializes the GestureRecognizer using MediaPipe Hands.
        history_len: Number of frames to keep in history for smoothing (prevents random jumping).
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        
        # History buffer for robust production-level smoothing
        self.gesture_history = deque(maxlen=history_len)

    def recognize_gesture(self, frame):
        raw_gesture = "No Gesture"
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = [lm for lm in hand_landmarks.landmark]
                raw_gesture = self._classify_gesture(landmarks)
                break # Only process first hand for the label

        # Smooth out the predictions by taking the mode of recent history
        self.gesture_history.append(raw_gesture)

        try:
            final_gesture = mode(self.gesture_history)
        except StatisticsError:
            final_gesture = "No Gesture"

        cv2.putText(frame, f"Gesture: {final_gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return frame, final_gesture

    def _classify_gesture(self, landmarks):
        fingers_up = []
        
        # Thumb
        if landmarks[self.tip_ids[0]].x < landmarks[self.tip_ids[0] - 1].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

        # Other 4 Fingers
        for i in range(1, 5):
            if landmarks[self.tip_ids[i]].y < landmarks[self.tip_ids[i] - 2].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

        total_fingers = sum(fingers_up)

        if total_fingers == 5:
            return "Open Hand"
        elif total_fingers == 0:
            return "Fist"
        elif total_fingers == 1 and fingers_up[0] == 1:
            return "Thumbs Up"
        elif total_fingers == 1 and fingers_up[1] == 1:
            return "Pointing"
        elif total_fingers == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:
            return "Peace"
        elif total_fingers == 3 and fingers_up[1] == 1 and fingers_up[2] == 1 and fingers_up[3] == 1:
            return "Three Fingers"
            
        return "Unknown Gesture"

    def close(self):
        self.hands.close()