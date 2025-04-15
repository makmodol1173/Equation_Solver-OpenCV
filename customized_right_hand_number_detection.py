import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Define finger tip landmarks
FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# Open webcam
cap = cv2.VideoCapture(0)

def count_fingers_0_to_9(hand_landmarks):
    """Detect hand gesture numbers from 0 to 9 using finger combinations."""
    fingers = []

    # Thumb
    thumb_up = hand_landmarks.landmark[FINGER_TIPS[0]].x > hand_landmarks.landmark[FINGER_TIPS[0] - 1].x
    fingers.append(1 if thumb_up else 0)

    # Other fingers
    for tip in FINGER_TIPS[1:]:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)

    f = fingers

    if f == [0, 0, 0, 0, 0]: return 0
    if f == [0, 1, 0, 0, 0]: return 1
    if f == [0, 1, 1, 0, 0]: return 2
    if f == [0, 1, 1, 1, 0]: return 3
    if f == [0, 1, 1, 1, 1]: return 4
    if f == [1, 1, 1, 1, 1]: return 5
    if f == [1, 0, 0, 0, 1]: return 6
    if f == [1, 1, 0, 0, 1]: return 7
    if f == [1, 1, 1, 0, 1]: return 8
    if f == [0, 1, 0, 0, 1]: return 9

    return -1  # Unknown gesture

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    number = -1

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_info in enumerate(results.multi_handedness):
            label = hand_info.classification[0].label  # 'Left' or 'Right'

            if label == 'Right':
                hand_landmarks = results.multi_hand_landmarks[idx]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                number = count_fingers_0_to_9(hand_landmarks)
                break  # Only use first detected right hand

    if number != -1:
        cv2.putText(frame, f'Number (Right Hand): {number}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, 'Show Right Hand to Detect Number', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Right Hand Number Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
