import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to count extended fingers
def count_fingers(landmarks):
    tips = [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
    count = 0

    # Thumb detection
    if landmarks[4].x < landmarks[3].x:
        count += 1

    # Other fingers
    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1

    return count  # Returns count of extended fingers

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

first_number = None
second_number = None
operation = None
result = None
state = "first_number"  # Current step in the process
last_time = time.time()  # Timer to avoid flickering inputs

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    detected_number = None
    detected_operation = None

    if results.multi_hand_landmarks:
        hand_list = []

        # Extract hand landmarks and sort based on x-coordinates (left to right)
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            avg_x = sum(x_coords) / len(x_coords)
            hand_list.append((avg_x, hand_landmarks))

        # Sort hands left to right
        hand_list.sort(key=lambda x: x[0])

        if len(hand_list) >= 1:
            detected_number = count_fingers(hand_list[-1][1].landmark)  # Rightmost hand

        if len(hand_list) >= 2:
            op_count = count_fingers(hand_list[0][1].landmark)  # Leftmost hand for operation
            if op_count == 1:
                detected_operation = '+'
            elif op_count == 2:
                detected_operation = '-'
            elif op_count == 3:
                detected_operation = '*'
            elif op_count == 4:
                detected_operation = '/'

    current_time = time.time()
    if current_time - last_time > 1.5:  # Adding a delay to prevent rapid input changes
        last_time = current_time  # Reset timer

        # Detect first number
        if state == "first_number" and detected_number is not None:
            first_number = detected_number
            state = "operator"

        # Detect operation
        elif state == "operator" and detected_operation is not None:
            operation = detected_operation
            state = "second_number"

        # Detect second number
        elif state == "second_number" and detected_number is not None:
            second_number = detected_number

            # Perform calculation
            if operation == '+':
                result = first_number + second_number
            elif operation == '-':
                result = first_number - second_number
            elif operation == '*':
                result = first_number * second_number
            elif operation == '/' and second_number != 0:
                result = first_number / second_number
            else:
                result = "Error"

            # Reset for next operation
            state = "first_number"
            first_number = None
            second_number = None
            operation = None

    # Display extracted numbers and results
    cv2.putText(frame, f'First: {first_number if first_number is not None else "?"}', (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Operator: {operation if operation else "?"}', (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f'Second: {second_number if second_number is not None else "?"}', (50, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Result: {result if result is not None else "?"}', (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Finger Gesture Calculator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
