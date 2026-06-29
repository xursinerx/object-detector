# ABOUTME: Combines YOLOv8 person detection with MediaPipe hand gesture recognition.
# ABOUTME: Detects if a person is present and what hand gesture they're making.

import cv2
from ultralytics import YOLO
import mediapipe as mp
import time
from gestures import fingers_up, gestures

model = YOLO("yolov8n.pt")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands( max_num_hands=2, min_detection_confidence=0.3)

cap = cv2.VideoCapture(0)
success, frame = cap.read()
while success:
    start = time.time()
    # Person detection
    results = model(frame, conf=0.6, classes=[0])
    person_count = len(results[0].boxes)
    annotated = results[0].plot()

    # Gesture detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)

    gesture_text = "No hands"
    if hand_results.multi_hand_landmarks:
        detected_gestures = []
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            position = tuple(fingers_up(hand_landmarks))
            detected_gestures.append(gestures.get(position, "Unknown"))
        gesture_text = " | ".join(detected_gestures)
    end = time.time()
    fps_text = f"FPS: {1 / (end - start):.1f}"

    cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, f"People: {person_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(annotated, f"Gesture: {gesture_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Detector", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    success, frame = cap.read()

cap.release()
