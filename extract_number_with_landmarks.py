import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Define finger tip landmarks
FINGER_TIPS = [4, 8, 12, 16, 20]

# Open webcam
cap = cv2.VideoCapture(0)


def count_fingers(hand_landmarks):
    """Counts the number of extended fingers including 6-9 gestures."""
    fingers = []

    # Thumb: Compare with lower joint (different than other fingers)
    thumb_up = hand_landmarks.landmark[FINGER_TIPS[0]].x > hand_landmarks.landmark[FINGER_TIPS[0] - 1].x
    fingers.append(1 if thumb_up else 0)

    # Other four fingers: If tip is higher than the middle joint
    for tip in FINGER_TIPS[1:]:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)

    count = sum(fingers)

    # Ensure 5 is correctly detected before checking 6-9
    if count == 5:
        return 5  # All fingers up

    # Adjust for numbers 6-9
    if count == 2 and fingers[0] and fingers[4]:
        return 6  # Thumb and pinky up
    elif count == 3 and fingers[0] and fingers[4] and fingers[1]:
        return 7  # Thumb, pinky, and index up
    elif count == 4 and fingers[0] and fingers[4] and fingers[1] and fingers[2]:
        return 8  # Thumb, pinky, index, and middle up
    elif count == 2 and fingers[1] and fingers[4]:
        return 9  # index, and pinky up

    return count  # Return regular count for 0-4


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    numbers = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers
            numbers.append(count_fingers(hand_landmarks))

    # Assign tens and units place
    if len(numbers) == 2:
        number = numbers[0] * 10 + numbers[1]  # First hand for tens, second hand for units
    elif len(numbers) == 1:
        number = numbers[0]
    else:
        number = 0

    # Display the number on the screen
    cv2.putText(frame, f'Number: {number}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()