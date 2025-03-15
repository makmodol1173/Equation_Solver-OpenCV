import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define arithmetic operation state
current_number = None
operation = None
result = 0


# Function to count extended fingers
def count_fingers(landmarks):
    tips = [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
    count = 0

    # Thumb (different detection method)
    if landmarks[4].x < landmarks[3].x:
        count += 1

    # Other fingers
    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1

    return count


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

            finger_count = count_fingers(hand_landmarks.landmark)

            # Operation logic
            if finger_count == 1:
                current_number = 1
            elif finger_count == 2:
                current_number = 2
            elif finger_count == 3:
                current_number = 3
            elif finger_count == 4:
                current_number = 4
            elif finger_count == 5:
                current_number = 5
            elif finger_count == 0 and current_number is not None:  # Gesture for operation
                if operation == '+':
                    result += current_number
                elif operation == '-':
                    result -= current_number
                elif operation == '*':
                    result *= current_number
                elif operation == '/' and current_number != 0:
                    result /= current_number
                current_number = None  # Reset input after calculation

            # Assign operation based on number of fingers
            if finger_count == 2:
                operation = '+'
            elif finger_count == 3:
                operation = '-'
            elif finger_count == 4:
                operation = '*'
            elif finger_count == 5:
                operation = '/'

    # Display results
    cv2.putText(frame, f'Operation: {operation if operation else "None"}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Result: {result}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Finger Gesture Calculator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()