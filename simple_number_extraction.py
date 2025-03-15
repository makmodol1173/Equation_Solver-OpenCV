import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


# Function to count extended fingers, including special cases for 6-9
def count_fingers(landmarks):
    tips = [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
    count = 0
    extended_fingers = []

    # Thumb detection
    if landmarks[4].x < landmarks[3].x:
        count += 1
        extended_fingers.append(4)

    # Other fingers
    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1
            extended_fingers.append(tip)

    # Special handling for 6-9
    if count == 2 and 4 in extended_fingers and 20 in extended_fingers:  # Thumb + Pinky
        return 6
    elif count == 3 and 4 in extended_fingers and 8 in extended_fingers and 20 in extended_fingers:  # Thumb + Index + Pinky
        return 7
    elif count == 3 and 4 in extended_fingers and 12 in extended_fingers and 20 in extended_fingers:  # Thumb + Middle + Pinky
        return 8
    elif count == 3 and 4 in extended_fingers and 16 in extended_fingers and 20 in extended_fingers:  # Thumb + Ring + Pinky
        return 9

    return count  # Default return for numbers 0-5


# OpenCV Video Capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    tens = 0
    ones = 0

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
            ones = count_fingers(hand_list[-1][1].landmark)  # Rightmost hand (ones place)

        if len(hand_list) == 2:
            tens = count_fingers(hand_list[0][1].landmark)  # Leftmost hand (tens place)

    # Calculate final number
    number = (tens * 10) + ones

    # Display extracted number
    cv2.putText(frame, f'Number: {number}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Finger Gesture Number Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()