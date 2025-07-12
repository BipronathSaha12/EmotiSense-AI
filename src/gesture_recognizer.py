import cv2
import mediapipe as mp
import numpy as np

class GestureRecognizer:
    def __init__(self):
        """
        Initializes the GestureRecognizer using MediaPipe Hands.
        This class is responsible for detecting hands and classifying simple gestures.
        """
        # Initialize MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,                # Allow detection of up to two hands
            min_detection_confidence=0.7,   # Higher confidence for initial detection
            min_tracking_confidence=0.5     # Lower confidence for tracking after detection
        )
        
        # Initialize MediaPipe drawing utilities to draw landmarks
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Landmark IDs for the tips of the fingers
        # Order: Thumb, Index, Middle, Ring, Pinky
        self.tip_ids = [4, 8, 12, 16, 20]

    def recognize_gesture(self, frame):
        """
        Processes a single video frame to detect hands and recognize gestures.

        Args:
            frame (numpy.ndarray): The input video frame from OpenCV (in BGR format).

        Returns:
            tuple: A tuple containing:
                - frame (numpy.ndarray): The frame with hand landmarks and gesture labels drawn on it.
                - gesture (str): The name of the detected gesture (e.g., "Peace", "Fist").
        """
        # Default gesture if no hand is confidently detected
        final_gesture = "No Gesture"
        
        # MediaPipe works with RGB images, but OpenCV provides BGR. Convert the color space.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find hand landmarks
        self.results = self.hands.process(rgb_frame)

        # Check if any hands were detected
        if self.results.multi_hand_landmarks:
            # Loop through each detected hand
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw the hand landmarks and connections on the original frame
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extract landmark coordinates into a more accessible list
                landmarks = [lm for lm in hand_landmarks.landmark]
                
                # Classify the gesture for the current hand
                gesture = self._classify_gesture(landmarks)
                
                # For simplicity, we'll display the gesture of the first detected hand
                # In a multi-hand scenario, you could label each hand separately
                if final_gesture == "No Gesture":
                    final_gesture = gesture

        # Display the detected gesture on the frame
        cv2.putText(frame, final_gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        return frame, final_gesture

    def _classify_gesture(self, landmarks):
        """
        A simple rule-based classifier for hand gestures using landmark positions.

        Args:
            landmarks (list): A list of 21 landmark objects for a single hand.

        Returns:
            str: The name of the classified gesture.
        """
        # A list to store which fingers are extended (1 for up, 0 for down)
        fingers_up = []
        
        # --- Thumb ---
        # We check if the thumb tip's x-coordinate is to the left (for a right hand)
        # or right (for a left hand) of its preceding joint. This accounts for hand rotation.
        # A simpler, more robust check is to compare the thumb tip to the PIP (Proximal Interphalangeal) joint.
        if landmarks[self.tip_ids[0]].x < landmarks[self.tip_ids[0] - 1].x:
            fingers_up.append(1) # Thumb is out
        else:
            fingers_up.append(0) # Thumb is in

        # --- Other 4 Fingers ---
        # For the other four fingers, we check if the fingertip's y-coordinate is above
        # the y-coordinate of the joint two landmarks below it (the PIP joint).
        # A lower y-value means it's higher up in the frame.
        for i in range(1, 5):
            if landmarks[self.tip_ids[i]].y < landmarks[self.tip_ids[i] - 2].y:
                fingers_up.append(1) # Finger is up
            else:
                fingers_up.append(0) # Finger is down

        # Calculate the total number of fingers that are up
        total_fingers = sum(fingers_up)

        # --- Gesture Classification Rules ---
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
        """Releases the MediaPipe hands resources."""
        self.hands.close()