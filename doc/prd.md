# Product Requirements Document (PRD): EmotiSense AI

## 1. Product Overview
**Product Name:** EmotiSense AI  
**Version:** 1.0 (Production Release)  
**Product Vision:** To provide an unobtrusive, lightweight, and highly reliable real-time tracking system for human emotions, hand gestures, and physical posture using standard webcam hardware.

## 2. Target Audience
* **Ergonomic Users:** Individuals wanting to monitor their posture over long working hours.
* **Researchers & Educators:** Academics analyzing engagement levels through facial expressions and physical states.
* **Hobbyists & Developers:** Engineers looking for a production-ready boilerplate for Human-Computer Interaction (HCI).

## 3. Core Features & Requirements

### 3.1. Emotion Recognition
* **Requirement:** Must identify core human emotions (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral).
* **Implementation:** MediaPipe Face Mesh for landmark extraction; Scikit-learn Support Vector Machine (SVM) for classification.
* **Constraint:** Must process quickly without requiring heavy GPU-bound deep learning models.

### 3.2. Hand Gesture Recognition
* **Requirement:** Accurately classify basic gestures (Fist, Open Hand, Peace, Pointing, Thumbs Up, Three Fingers).
* **Implementation:** Rule-based coordinate geometry using MediaPipe Hands (comparing fingertips to PIP joints).

### 3.3. Ergonomic Posture Tracking
* **Requirement:** Detect whether the user is sitting upright or slouching.
* **Implementation:** MediaPipe Pose tracking the vertical distance ratio between the nose and the shoulder baseline.

### 3.4. State Stabilization (Smoothing)
* **Requirement:** The UI and logs must not flicker or produce random false positives due to lighting glitches.
* **Implementation:** A rolling history buffer (Sliding Window of 15 frames) that calculates the statistical mode for all three pipelines.

### 3.5. Local Activity Logging
* **Requirement:** Maintain a persistent, lightweight record of user states without bloating storage.
* **Implementation:** An SQLite database (`events.db`) that logs exact timestamps *only* when the user's state (emotion + gesture + posture combined) changes.

## 4. Non-Functional Requirements
* **Performance:** Must maintain a smooth frame rate (minimum 24 FPS) on standard CPU architecture.
* **Resilience:** The application must never crash due to a single dropped frame, obscured face, or missing landmark.
* **Resource Management:** Must gracefully release webcam hardware and window instances upon user exit (via ESC or 'X' button) or system interrupt.

## 5. Future Roadmap (v2.0+)
* Cloud syncing for the SQLite database.
* Desktop notifications/alerts when the user has been slouching for > 10 minutes.
* Multi-person tracking capabilities.
