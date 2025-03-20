import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define arithmetic operation state
operation = None

# Function to detect arithmetic operators
def detect_operator(landmarks):
    tips = [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
    count = 0

    # Thumb detection
    if landmarks[4].x < landmarks[3].x:
        count += 1

    # Count extended fingers
    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1

    # Assign operation based on finger count
    if count == 0:
        return '+'
    elif count == 1:
        return '-'
    elif count == 2:
        return '*'
    elif count == 3:
        return '/'
    elif count == 4:
        return '('
    elif count == 5:
        return ')'
    else:
        return None

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            operation = detect_operator(hand_landmarks.landmark)

    # Display detected operator
    cv2.putText(frame, f'Operator: {operation if operation else "None"}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Finger Gesture Operator Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
