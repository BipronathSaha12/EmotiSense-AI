import cv2
import mediapipe as mp
from collections import deque
from statistics import mode, StatisticsError

class PostureRecognizer:
    def __init__(self, history_len=15):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.posture_history = deque(maxlen=history_len)

    def recognize_posture(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        raw_posture = "No Person Detected"

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            landmarks = results.pose_landmarks.landmark
            
            # Simple heuristic for posture based on shoulders and nose
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            
            # Y goes down in image coordinates
            shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            
            # If nose is too close to shoulders vertically, person is likely slouching/leaning
            if nose.y > shoulder_avg_y - (shoulder_width * 0.4):
                raw_posture = "Slouching (Poor Posture)"
            else:
                raw_posture = "Upright (Good Posture)"

        self.posture_history.append(raw_posture)
        
        try:
            final_posture = mode(self.posture_history)
        except StatisticsError:
            final_posture = "No Person Detected"

        cv2.putText(frame, f"Posture: {final_posture}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        return frame, final_posture
        
    def close(self):
        self.pose.close()
