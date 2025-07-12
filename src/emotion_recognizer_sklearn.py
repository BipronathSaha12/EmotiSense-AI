import cv2
import numpy as np
import mediapipe as mp
import pickle

class EmotionRecognizerSKL:
    def __init__(self, model_path):
        """
        Initializes the recognizer with a pre-trained scikit-learn model.
        """
        # Load the trained SVM model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    def _extract_features(self, landmarks):
        """Extracts the same features used during training."""
        anchor_point = landmarks[1]
        features = []
        for i in range(len(landmarks)):
            if i == 1: continue
            dist = np.sqrt((landmarks[i].x - anchor_point.x)**2 + 
                           (landmarks[i].y - anchor_point.y)**2 +
                           (landmarks[i].z - anchor_point.z)**2)
            features.append(dist)
        
        face_width = np.sqrt((landmarks[234].x - landmarks[454].x)**2 + 
                             (landmarks[234].y - landmarks[454].y)**2)
        
        if face_width < 1e-6: return None
        
        normalized_features = np.array(features) / face_width
        return normalized_features

    def detect_emotions(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)

                # Extract features for prediction
                landmarks_list = [lm for lm in face_landmarks.landmark]
                features = self._extract_features(landmarks_list)
                
                if features is not None:
                    features = features.reshape(1, -1) # Reshape for prediction
                    prediction = self.model.predict(features)
                    emotion = self.emotion_labels[prediction[0]]
                    
                    # Get bounding box to display text
                    h, w, _ = frame.shape
                    x_min = int(min([lm.x for lm in landmarks_list]) * w)
                    y_min = int(min([lm.y for lm in landmarks_list]) * h)
                    
                    cv2.putText(frame, emotion, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        return frame

    def close(self):
        self.face_mesh.close()