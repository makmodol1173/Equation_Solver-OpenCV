import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define finger tip landmarks
FINGER_TIPS = [4, 8, 12, 16, 20]

# Open webcam
cap = cv2.VideoCapture(0)

expression = ""
last_update_time = time.time()
update_interval = 3  # 3 seconds interval for counting new gestures
result = ""  # Store result separately
hand_removed = False  # Flag to detect hand removal
mode = "numbers"  # Mode to switch between numbers and operators


def count_fingers(hand_landmarks):
    """Counts the number of extended fingers including 6-9 gestures."""
    fingers = []

    # Thumb: Compare with lower joint (different than other fingers)
    thumb_up = hand_landmarks.landmark[FINGER_TIPS[0]].x < hand_landmarks.landmark[FINGER_TIPS[0] - 1].x
    fingers.append(1 if thumb_up else 0)

    # Other four fingers: If tip is higher than the middle joint
    for tip in FINGER_TIPS[1:]:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)

    count = sum(fingers)

    # Adjust for numbers 6-9
    if count == 2 and fingers[0] and fingers[4]:
        return 6  # Thumb and pinky up
    elif count == 3 and fingers[0] and fingers[4] and fingers[1]:
        return 7  # Thumb, pinky, and index up
    elif count == 4 and fingers[0] and fingers[4] and fingers[1] and fingers[2]:
        return 8  # Thumb, pinky, index, and middle up
    elif count == 2 and fingers[1] and fingers[4]:
        return 9  # Index and pinky up

    return count  # Return regular count for 0-4


def detect_operation(hand_landmarks):
    """Detects arithmetic operation based on hand gesture."""
    fingers = [
        hand_landmarks.landmark[FINGER_TIPS[i]].y < hand_landmarks.landmark[FINGER_TIPS[i] - 2].y
        for i in range(1, 5)
    ]
    thumb_up = hand_landmarks.landmark[FINGER_TIPS[0]].x < hand_landmarks.landmark[FINGER_TIPS[0] - 1].x

    if fingers[0] and not any(fingers[1:]) and not thumb_up:
        return '+'
    elif fingers[3] and not any(fingers[:3]) and not thumb_up:
        return '-'
    elif thumb_up and not any(fingers):
        return '*'
    elif thumb_up and fingers[0]:
        return '/'
    elif fingers[0] and fingers[1]:
        return '('
    elif thumb_up and fingers[0] and fingers[1]:
        return ')'
    return ''


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()
    if results.multi_hand_landmarks:
        hand_removed = False  # Reset flag when hand is detected
        hand_landmarks = results.multi_hand_landmarks[0]

        if current_time - last_update_time > update_interval:
            if mode == "numbers":
                detected_number = count_fingers(hand_landmarks)
                if detected_number is not None:
                    expression += str(detected_number)
                    mode = "operators"  # Switch to operator mode after number input
            else:
                result = "Error"

    # Display the expression and result
    cv2.putText(frame, f'Expression: {expression}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Result: {result}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
