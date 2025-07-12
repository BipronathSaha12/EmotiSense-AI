import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def extract_features(image):
    """Extracts normalized facial landmark distances as features."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # --- Feature Engineering ---
    # Anchor point (e.g., nose tip)
    anchor_point = landmarks[1]
    
    features = []
    for i in range(len(landmarks)):
        if i == 1: continue
        # Calculate Euclidean distance
        dist = np.sqrt((landmarks[i].x - anchor_point.x)**2 + 
                       (landmarks[i].y - anchor_point.y)**2 +
                       (landmarks[i].z - anchor_point.z)**2)
        features.append(dist)
    
    # Normalize features by the width of the face (distance between landmarks on cheeks)
    # This makes features scale-invariant
    face_width = np.sqrt((landmarks[234].x - landmarks[454].x)**2 + 
                         (landmarks[234].y - landmarks[454].y)**2)
    
    if face_width < 1e-6: # Avoid division by zero
        return None
        
    normalized_features = np.array(features) / face_width
    return normalized_features

print("Loading dataset...")
data = pd.read_csv('data/fer2013.csv')
print("Dataset loaded.")

# Emotion mapping from the dataset
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

X = []
y = []

print("Extracting features from dataset... This may take a while.")
# Use tqdm for a progress bar
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    pixels = np.array(row['pixels'].split(' '), 'uint8')
    image = pixels.reshape((48, 48))
    
    features = extract_features(image)
    if features is not None and features.shape[0] == 467: # Ensure correct number of features
        X.append(features)
        y.append(row['emotion'])

print(f"Feature extraction complete. Total samples: {len(X)}")

X = np.array(X)
y = np.array(y)

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the SVM classifier...")
# Using SVC with RBF kernel, a powerful and common choice
# Class weights are balanced to handle imbalanced datasets like FER2013
model = SVC(kernel='rbf', probability=True, C=1.0, class_weight='balanced')
model.fit(X_train, y_train)
print("Training complete.")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("Saving the model...")
with open('models/emotion_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved successfully as models/emotion_classifier.pkl")

face_mesh.close()