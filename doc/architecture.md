# System Architecture: EmotiSense AI

## 1. High-Level Architecture Overview
EmotiSense AI follows a modular, monolithic architecture designed for local execution. It uses a central orchestrator (`main.py`) that acts as a video frame router, passing image data through three independent recognition pipelines and aggregating the results for display and logging.

## 2. Component Breakdown

### 2.1. The Orchestrator (`main.py`)
* **Role:** Initializes the camera feed, manages the infinite `while` loop, and handles graceful shutdowns.
* **Exception Handling:** Wraps every pipeline call in a `try...except` block to guarantee the stream survives pipeline-specific crashes.
* **Exit Logic:** Monitors `cv2.waitKey` and `cv2.getWindowProperty` to catch UI exit events safely.

### 2.2. Recognition Pipelines (The `src/` Directory)
Each recognizer follows a standardized interface, accepting a raw frame and returning the modified frame alongside a text-based classification.

* **`EmotionRecognizerSKL`**
  * **Engine:** MediaPipe Face Mesh + Scikit-learn SVM.
  * **Logic:** Normalizes 3D spatial distances of facial landmarks relative to a core anchor point before predicting.
* **`GestureRecognizer`**
  * **Engine:** MediaPipe Hands.
  * **Logic:** Purely mathematical. Evaluates finger extension by comparing the Y-axis of the fingertip to the Y-axis of the knuckle joint.
* **`PostureRecognizer`**
  * **Engine:** MediaPipe Pose.
  * **Logic:** Evaluates slouching by measuring if the nose drops below a calculated shoulder-width threshold threshold relative to the shoulder Y-axis average.

### 2.3. Data Persistence (`db_logger.py`)
* **Role:** Local state storage using SQLite3.
* **Optimization Strategy:** Implements state-change debouncing. It holds the `last_logged_state` in memory and only executes an SQL `INSERT` when the new state tuple differs from the previous one. This prevents thousands of redundant rows per minute.

## 3. Data Flow Diagram

```text
[Webcam (OpenCV)] 
       │
       ▼ (Raw Frame)
[main.py Orchestrator]
       │
       ├──► [EmotionRecognizer] ──(Feature Extraction)──► [SVM Model] ──┐
       │                                                                │
       ├──► [GestureRecognizer] ──(Rule-Based Math)─────────────────────┼──► (Combined State)
       │                                                                │          │
       └──► [PostureRecognizer] ──(Nose/Shoulder Math)──────────────────┘          │
                                                                                   ▼
                                                                        [DatabaseLogger]
                                                                         (Logs on Change)
                                                                                   │
                                                                                   ▼
                                                                           [events.db]
```

## 4. Resource Cleanup Flow
When an exit signal is intercepted, `main.py` triggers a `finally` block:
1. Recognizers close their internal MediaPipe instances to free memory.
2. The Database Logger closes its SQLite connection to prevent lock errors.
3. OpenCV releases the hardware camera (`cap.release()`).
4. OpenCV destroys all active UI windows.
